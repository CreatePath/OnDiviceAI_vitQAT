"""Adapter checks use an explicit test double; native checks require Linux CUDA."""
import importlib.util
import platform
from types import SimpleNamespace

import pytest
import torch

from vitqat import w2a8_bitblas as ops
from vitqat.cuda_w2a8 import A8Quantizer, W2A8Linear, convert_cuda_w2a8
from vitqat.quantization import QATLinear, Quantizer


class ReferenceMatmul:
    """Test double only: emulate the library boundary with exact integer arithmetic."""
    def __init__(self, config, **kwargs):
        self.config = config
        assert config.A_dtype == "int8"
        assert config.W_dtype in ("uint2", "int8")
        assert config.accum_dtype == config.out_dtype == "int32"
        assert not config.propagate_a and not config.propagate_b and not config.fast_decoding
        assert not config.with_scaling and not config.with_zeros and not config.with_bias

    def transform_weight(self, codes):
        assert self.config.W_dtype == "uint2"
        values = codes.reshape(codes.shape[0], -1, 4)
        return (values[..., 0] | (values[..., 1] << 2) | (values[..., 2] << 4) | (values[..., 3] << 6)).view(torch.int8)

    def __call__(self, a, w, output=None):
        assert a.dtype == w.dtype == torch.int8
        assert a.shape == (self.config.M, self.config.K)
        if self.config.W_dtype == "uint2":
            shifts = torch.arange(4, device=w.device) * 2
            w = ((w.to(torch.uint8).int()[..., None] >> shifts) & 3).flatten(-2)
        assert w.shape == (self.config.N, self.config.K)
        result = (a.cpu().int() @ w.cpu().int().t()).to(a.device)
        if output is not None:
            output.copy_(result)
            return output
        return result


@pytest.fixture(params=["adapter", "native"])
def bitblas_backend(request, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required, including for adapter checks")
    torch.set_num_threads(2)
    ops._operator.cache_clear()
    if request.param == "adapter":
        monkeypatch.setattr(ops, "require_bitblas", lambda: SimpleNamespace(
            MatmulConfig=SimpleNamespace, Matmul=ReferenceMatmul,
            auto_detect_nvidia_target=lambda: "test-double"))
    elif platform.system() != "Linux" or importlib.util.find_spec("bitblas") is None:
        pytest.skip("Native BitBLAS requires Linux CUDA and requirements-bitblas.txt")
    yield request.param
    ops._operator.cache_clear()


@pytest.mark.parametrize("m,k,n", [(1, 19, 37), (17, 192, 101), (197, 768, 192)])
def test_bitblas_linear_exact_accumulation_and_state(bitblas_backend, m, k, n):
    torch.manual_seed(19)
    layer = QATLinear(k, n).cuda()
    layer.input_quant, layer.weight_quant, layer.output_quant = Quantizer(8).cuda(), Quantizer(2, weight=True).cuda(), Quantizer(8).cuda()
    x = torch.randn(m, k, device="cuda") * 2 + 1
    layer(x)
    layer.eval()
    converted = W2A8Linear(layer)
    x = x * 4
    iq, wq = layer.input_quant.fake_quant, layer.weight_quant.fake_quant
    a = (x * iq.scale.reciprocal()).round().add(iq.zero_point).clamp(0, 255).int() - iq.zero_point
    w = (layer.weight * wq.scale.reciprocal()[:, None]).round().add(wq.zero_point[:, None]).clamp(0, 3).int() - wq.zero_point[:, None]
    expected = a.cpu().int() @ w.cpu().int().t()
    actual = ops.linear(x, converted, raw=True)
    assert torch.equal(actual.cpu(), expected)
    with torch.inference_mode():
        torch.testing.assert_close(converted(x), layer(x), atol=float(layer.output_quant.fake_quant.scale) * 1.01, rtol=0)
    restored = W2A8Linear(layer)
    restored.load_state_dict(converted.state_dict())
    assert torch.equal(restored(x), converted(x))


@pytest.mark.parametrize("za,zb", [(0, 255), (128, 128), (221, 17)])
def test_bitblas_attention_exact_and_transposed(bitblas_backend, za, zb):
    torch.manual_seed(7)
    a = torch.randint(-128, 128, (2, 3, 5, 19), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (2, 3, 7, 19), device="cuda", dtype=torch.int8).transpose(-2, -1)
    aq = SimpleNamespace(scale=torch.tensor([0.125], device="cuda"), zero=torch.tensor([za], device="cuda"))
    bq = SimpleNamespace(scale=torch.tensor([0.25], device="cuda"), zero=torch.tensor([zb], device="cuda"))
    expected = (a.cpu().int() + 128 - za) @ (b.cpu().int() + 128 - zb)
    assert torch.equal(ops.activation_matmul(a, b, aq, bq, raw=True).cpu(), expected)
    torch.testing.assert_close(ops.activation_matmul(a, b, aq, bq, factor=0.5).cpu(), expected.float() * 0.125 * 0.25 * 0.5)


@pytest.mark.parametrize("cropr,llf,scope", [
    (False, False, "all"), (True, False, "all"), (True, True, "all"),
    (True, False, "backbone"), (True, True, "backbone"),
])
def test_bitblas_full_model(bitblas_backend, tmp_path, monkeypatch, cropr, llf, scope):
    import json
    from vitqat.cli import parser
    from vitqat.models import build_model
    from vitqat.quantization import QATConv2d, QATAttention, QATCrossAttention
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    torch.manual_seed(42)
    args = parser().parse_args(["--mode", "qat", "--activation-bits", "8", "--image-size", "64", "--output", str(tmp_path)])
    args.use_cropr, args.qat_scope, args.cropr_pruning_rate = cropr, scope, 1
    args.cropr_llf = llf
    source = build_model(args).cuda()
    x = torch.randn(1, 3, 64, 64, device="cuda")
    with torch.no_grad():
        source(x)
    source.eval()
    converted = convert_cuda_w2a8(source, "cuda")
    assert not any(isinstance(m, (Quantizer, FakeQuantizeBase)) for m in converted.modules())
    sites = [(name, module) for name, module in source.named_modules() if isinstance(module, Quantizer)]
    for name, quant in sites:
        parent_name, _, role = name.rpartition(".")
        original, target = source.get_submodule(parent_name), converted.get_submodule(parent_name)
        if isinstance(original, (QATLinear, QATConv2d)):
            prefix = role.removesuffix("_quant")
            scale, zero = getattr(target, prefix + "_scale"), getattr(target, prefix + "_zero")
        elif isinstance(original, (QATAttention, QATCrossAttention)):
            if role == "query_quant":
                with torch.inference_mode():
                    query = original.q_quant(original.q(quant(original.queries)))
                query = query.reshape(1, original.num_queries, -1, original.head_dim).transpose(1, 2)
                restored = (target.query_codes.int() + 128 - target.q_quant.zero).float() * target.q_quant.scale
                assert torch.equal(query, restored), name
                continue
            scale, zero = getattr(target, role).scale, getattr(target, role).zero
        else:
            pytest.fail(f"Unaccounted QAT site: {name}")
        assert torch.equal(scale, quant.fake_quant.scale), name
        assert torch.equal(zero, quant.fake_quant.zero_point), name
    calls = dict(linear_conv=0, attention=0)
    original_linear, original_attention = ops.linear, ops.activation_matmul

    def linear(*a, **kw):
        calls["linear_conv"] += 1
        return original_linear(*a, **kw)

    def attention(*a, **kw):
        calls["attention"] += 1
        assert a[0].dtype == a[1].dtype == torch.int8
        return original_attention(*a, **kw)

    monkeypatch.setattr(ops, "linear", linear)
    monkeypatch.setattr(ops, "activation_matmul", attention)
    with torch.inference_mode():
        expected, actual = source(x), converted(x)
    assert torch.isfinite(actual).all()
    assert (expected - actual).abs().max() < 0.1
    pruners = (10 if llf else 11) if cropr and scope == "all" else 0
    assert len(sites) == 198 + 14 * pruners
    assert calls == dict(linear_conv=50, attention=24 + pruners)
    print(json.dumps(dict(validation=bitblas_backend, cropr=cropr, llf=llf, scope=scope,
                          qat_sites=len(sites), remaining_fake_quant=0, **calls)))


def test_bitblas_platform_error_is_explicit(monkeypatch):
    monkeypatch.setattr(ops.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="No fallback"):
        ops.require_bitblas()


def test_training_evaluation_export_and_benchmark(bitblas_backend, tmp_path, monkeypatch):
    from torch.utils.data import TensorDataset
    from vitqat.cli import parser
    from vitqat.models import build_model
    from vitqat.runner import run
    from vitqat.train import measure_model, train_epoch
    from scripts.benchmark_w2a8 import main as benchmark
    if bitblas_backend == "adapter":
        monkeypatch.setattr("vitqat.runner.version", lambda package: "test-double")
        monkeypatch.setattr("scripts.benchmark_w2a8.version", lambda package: "test-double")
    args = parser().parse_args([
        "--mode", "qat", "--activation-bits", "8", "--device", "cuda",
        "--cuda-w2a8-inference", "--image-size", "32", "--epochs", "1",
        "--warmup-epochs", "0", "--workers", "0", "--batch-size", "2",
        "--latency-warmup", "0", "--latency-repeats", "1", "--output", str(tmp_path / "run")])

    def data_factory(config):
        data = TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        return data, data, data, {"sha256": "synthetic-test"}

    result = run(args, build_model=build_model, measure_model=measure_model,
                 train_epoch_fn=train_epoch, data_factory=data_factory)
    assert result["cuda_w2a8_latency_ms"] > 0
    assert result["cuda_w2a8_conv_layers"] == 1
    assert result["cuda_w2a8_attention_layers"] == 12
    assert result["cuda_w2a8_remaining_fake_quantizers"] == 0
    assert result["cuda_w2a8_state_bytes"] > result["cuda_w2a8_packed_weight_bytes"] > 0
    assert (tmp_path / "run" / "cuda_w2a8_state.pt").is_file()
    results = benchmark(["--checkpoint", str(tmp_path / "run" / "best.pt"),
                         "--output", str(tmp_path / "benchmark.json"), "--warmup", "0", "--repeats", "1"])
    assert results["packed_w2a8"]["host_ms_median"] > 0
    assert results["fp32_master"]["cuda_event_ms_median"] > 0
