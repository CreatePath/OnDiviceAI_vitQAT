"""CPU numerical/export checks; these do not establish native TensorRT speed."""
import copy
from types import SimpleNamespace

import pytest
import torch

from vitqat.quantization import QATLinear, Quantizer
from vitqat.tensorrt_int8 import FrozenQDQ, audit_attention_qdq, convert_tensorrt_int8, export_onnx


def test_w2_values_preserved_and_source_untouched():
    layer = QATLinear(8, 4)
    layer.input_quant = Quantizer(8)
    layer.weight_quant = Quantizer(2, weight=True)
    layer.output_quant = Quantizer(8)
    layer(torch.randn(3, 8) + 2)
    layer.eval()
    expected = layer.weight_quant(layer.weight).detach()
    before = copy.deepcopy(layer.state_dict())
    converted = convert_tensorrt_int8(layer)
    torch.testing.assert_close(converted.weight_quant(converted.weight), expected, rtol=0, atol=0)
    assert all(torch.unique(row).numel() <= 4 for row in converted.weight)
    assert converted.input_quant.zero.item() == 0
    fq = layer.input_quant.fake_quant
    expected_scale = fq.scale * torch.maximum(fq.zero_point, 255 - fq.zero_point) / 127
    torch.testing.assert_close(converted.input_quant.scale, expected_scale)
    assert all(torch.equal(value, layer.state_dict()[key]) for key, value in before.items())


def test_reject_uncalibrated_or_wrong_bits():
    layer = QATLinear(8, 4)
    layer.input_quant = Quantizer(8)
    layer.weight_quant = Quantizer(2, weight=True)
    layer.output_quant = Quantizer(8)
    with pytest.raises(ValueError, match="calibrated"):
        convert_tensorrt_int8(layer)
    layer(torch.randn(3, 8))
    layer.weight_quant.bits = 4
    with pytest.raises(ValueError, match="calibrated"):
        convert_tensorrt_int8(layer)


def test_native_result_summary():
    from scripts.benchmark_tensorrt_int8 import summarize_results

    logits = torch.tensor([[0.25, 0.75]])
    result = summarize_results([1, 4, 3, 2], logits, logits)
    assert result["gpu_compute_ms_median"] == 2.5
    assert result["gpu_compute_ms_p95"] == 4
    assert result["synthetic_native_vs_reference_max_abs_error"] == 0
    with pytest.raises(ValueError, match="timings"):
        summarize_results([], logits, logits)
    with pytest.raises(ValueError, match="logits"):
        summarize_results([1], logits.flatten(), logits)


def test_native_requires_cuda(monkeypatch):
    from scripts.benchmark_tensorrt_int8 import require_tensorrt

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        require_tensorrt("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native TensorRT requires CUDA PyTorch")
@pytest.mark.parametrize("quantized", [False, True])
def test_native_python_execution(tmp_path, quantized):
    pytest.importorskip("tensorrt")
    from scripts.benchmark_tensorrt_int8 import benchmark_engine, require_tensorrt

    torch.manual_seed(42)
    x = torch.randn(16, 32)
    if quantized:
        layer = QATLinear(32, 32)
        layer.input_quant, layer.weight_quant, layer.output_quant = Quantizer(8), Quantizer(2, weight=True), Quantizer(8)
        with torch.no_grad():
            layer(x)
        layer = convert_tensorrt_int8(layer.eval())
        tolerance = float(layer.output_quant.scale) * 2
    else:
        layer = torch.nn.Linear(32, 32).eval()
        tolerance = 1e-4
    with torch.no_grad():
        reference = layer(x)
    export_onnx(layer, x, tmp_path / "native.onnx")
    options = SimpleNamespace(device="cuda", warmup_ms=0, duration=0, iterations=2)
    result = benchmark_engine(require_tensorrt("cuda"), tmp_path, "native", x, reference, options)
    assert result["samples"] == 2
    assert result["synthetic_native_vs_reference_max_abs_error"] <= tolerance
    assert (tmp_path / "native.engine").stat().st_size > 0
    assert (tmp_path / "native.layers.json").is_file()


@pytest.mark.parametrize("cropr,llf,scope", [
    (False, False, "all"), (True, False, "all"), (True, True, "all"),
    (True, False, "backbone"), (True, True, "backbone"),
])
def test_full_export_and_onnx_runtime(tmp_path, cropr, llf, scope):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from vitqat.cli import parser
    from vitqat.models import build_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    args = parser().parse_args(["--mode", "qat", "--activation-bits", "8", "--image-size", "64", "--output", "unused"])
    args.use_cropr, args.cropr_llf, args.qat_scope, args.cropr_pruning_rate = cropr, llf, scope, 1
    source = build_model(args)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        source(x)  # Populate all observers, including auxiliary training branches.
    source.eval()
    converted = convert_tensorrt_int8(source)
    assert not any(isinstance(module, Quantizer) for module in converted.modules())
    for name, module in source.named_modules():
        if isinstance(module, Quantizer) and name.endswith(("weight_quant", "query_quant")):
            parent_name, _, role = name.rpartition(".")
            original, target = source.get_submodule(parent_name), converted.get_submodule(parent_name)
            parameter = "weight" if role == "weight_quant" else "queries"
            with torch.no_grad():
                expected = module(getattr(original, parameter))
            torch.testing.assert_close(getattr(target, role)(getattr(target, parameter)), expected, rtol=0, atol=0)
    assert isinstance(converted.blocks[0].attn.prob_quant, FrozenQDQ)
    path = tmp_path / "model.onnx"
    graph = export_onnx(converted, x, path)
    rows = audit_attention_qdq(graph)
    expected_count = 24 + (len(source.cropr) if cropr and scope == "all" else 0)
    assert sum(row["both_inputs_qdq"] for row in rows) == expected_count, rows
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
    result = torch.from_numpy(session.run(None, {"images": x.numpy()})[0])
    with torch.no_grad():
        expected = converted(x)
    # Small GEMM/LayerNorm differences can cross quantization boundaries and
    # accumulate across 12 blocks. This is a drift bound, not bitwise equivalence.
    torch.testing.assert_close(result, expected, atol=float(converted.head.output_quant.scale) * 8, rtol=0)
    assert (result - expected).norm() / expected.norm().clamp_min(1e-6) < 0.05
    if not cropr:
        from scripts.benchmark_tensorrt_int8 import main

        checkpoint = tmp_path / "checkpoint.pt"
        torch.save({"model": source.state_dict(), "config": vars(args)}, checkpoint)
        destination = tmp_path / "cli-export"
        report = main(["--checkpoint", str(checkpoint), "--output", str(destination), "--export-only"])
        assert report["engine_execution"] == "not_run"
        assert len(report["attention_onnx"]) == 24
        assert (destination / "fp32_master.onnx").is_file()
        assert (destination / "images.bin").stat().st_size == x.numel() * 4
