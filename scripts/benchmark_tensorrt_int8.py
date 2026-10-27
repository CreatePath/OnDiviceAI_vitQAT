"""Export W2-valued INT8 ONNX and benchmark with the TensorRT Python API."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

import torch

from vitqat.cli import parser as training_parser
from vitqat.models import build_model
from vitqat.tensorrt_int8 import audit_attention_qdq, convert_tensorrt_int8, export_onnx
from vitqat.training import seed_all


def summarize_results(times, logits, reference):
    times = sorted(times)
    if not times or not all(math.isfinite(value) and value > 0 for value in times):
        raise ValueError("Missing or invalid native GPU timings")
    if logits.shape != reference.shape or not torch.isfinite(logits).all():
        raise ValueError("Invalid native logits shape or nonfinite values")
    return dict(gpu_compute_ms_median=statistics.median(times),
                gpu_compute_ms_p95=times[math.ceil(len(times) * 0.95) - 1], samples=len(times),
                synthetic_native_vs_reference_max_abs_error=(logits - reference).abs().max().item(),
                synthetic_native_vs_reference_top1_agreement=(logits.argmax(-1) == reference.argmax(-1)).float().mean().item())


def require_tensorrt(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU execution requires CUDA-enabled PyTorch; use --export-only for CPU export")
    try:
        import tensorrt as trt
    except (ImportError, OSError) as error:
        raise RuntimeError("Install TensorRT with: python -m pip install -r requirements-tensorrt.txt") from error
    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This runner requires TensorRT 10.x; install requirements-tensorrt.txt")
    return trt


@torch.inference_mode()
def benchmark_engine(trt, output, name, x, reference, options):
    """Build once, bind PyTorch CUDA buffers, then time serialized forwards."""
    with torch.cuda.device(options.device):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)  # TensorRT 10 always uses explicit batch.
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(output / f"{name}.onnx")):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"{name}: TensorRT ONNX parsing failed:\n{errors}")
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"{name}: TensorRT engine build failed; see TensorRT messages above")
        (output / f"{name}.engine").write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError(f"{name}: TensorRT engine deserialization failed")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"{name}: cannot create execution context")
        buffers = {"images": x.to(options.device).contiguous(),
                   "logits": torch.empty(reference.shape, dtype=torch.float32, device=options.device)}
        if engine.num_io_tensors != 2:
            raise RuntimeError(f"{name}: expected images/logits I/O only")
        for key, tensor in buffers.items():
            expected_mode = trt.TensorIOMode.INPUT if key == "images" else trt.TensorIOMode.OUTPUT
            if (engine.get_tensor_dtype(key) != trt.float32
                    or engine.get_tensor_mode(key) != expected_mode
                    or engine.get_tensor_location(key) != trt.TensorLocation.DEVICE
                    or engine.get_tensor_format(key) != trt.TensorFormat.LINEAR
                    or tuple(engine.get_tensor_shape(key)) != tuple(tensor.shape)):
                raise RuntimeError(f"{name}: unexpected fixed FP32 I/O contract for {key}")
            if not context.set_tensor_address(key, tensor.data_ptr()):
                raise RuntimeError(f"{name}: could not bind {key}")
        inspector = engine.create_engine_inspector()
        inspector.execution_context = context
        (output / f"{name}.layers.json").write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding="utf-8")
        stream = torch.cuda.current_stream()

        def launch():
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"{name}: TensorRT execution failed")

        launch()  # First execution and input transfer are outside measurements.
        stream.synchronize()
        deadline = time.perf_counter() + options.warmup_ms / 1000
        while time.perf_counter() < deadline:
            launch()
            stream.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        deadline = time.perf_counter() + options.duration
        while len(times) < options.iterations or time.perf_counter() < deadline:
            start.record(stream)
            launch()
            end.record(stream)
            end.synchronize()
            times.append(start.elapsed_time(end))
        logits = buffers["logits"].cpu()
        (output / f"{name}.times.json").write_text(
            json.dumps([{"computeMs": value} for value in times]), encoding="utf-8")
        (output / f"{name}.outputs.json").write_text(json.dumps([
            {"name": "logits", "shape": list(logits.shape), "values": logits.flatten().tolist()}]), encoding="utf-8")
        return summarize_results(times, logits, reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Directory for ONNX, engines, logs and reports")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--export-only", action="store_true", help="CPU export; no TensorRT required")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--duration", type=int, default=10, help="Minimum benchmark seconds per engine")
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args(argv)
    if options.batch_size < 1 or options.warmup_ms < 0 or options.duration < 1 or options.iterations < 1:
        parser.error("batch-size/duration/iterations must be positive; warmup-ms must be nonnegative")
    trt = None if options.export_only else require_tensorrt(options.device)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    config = training_parser().parse_args(["--mode", "qat", "--output", "unused"])
    vars(config).update(checkpoint["config"])
    if config.mode != "qat" or (config.weight_bits, config.activation_bits) != (2, 8):
        parser.error("A calibrated W2A8 QAT checkpoint is required")
    seed_all(config.seed)
    source = build_model(config)
    source.load_state_dict(checkpoint["model"], strict=True)
    source.eval()
    converted = convert_tensorrt_int8(source)
    fp_config = copy.copy(config)
    fp_config.mode = "fp32"
    fp32 = build_model(fp_config)
    fp32.load_state_dict({key: checkpoint["model"][key] for key in fp32.state_dict()}, strict=True)
    fp32.eval()
    x = torch.randn(options.batch_size, 3, config.image_size, config.image_size)
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected, actual = source(x), converted(x)
        references = {"fp32_master": fp32(x), "w2_int8": actual}
    report = dict(checkpoint=str(options.checkpoint), torch_version=str(torch.__version__),
                  batch_size=options.batch_size, image_size=config.image_size,
                  weight_values="original W2 grid exported with INT8 Q/DQ; kernel precision unverified",
                  activations="symmetric INT8 derived from trained affine range; no new calibration",
                  qat_scope=config.qat_scope,
                  synthetic_logit_max_abs_error=(expected - actual).abs().max().item(),
                  synthetic_top1_agreement=(expected.argmax(-1) == actual.argmax(-1)).float().mean().item(),
                  engine_execution="not_run", kernel_precision_verified=False,
                  note="Synthetic checks are not Food-101 accuracy. Q/DQ coverage is not proof of INT8 kernels.")
    for name, net in (("fp32_master", fp32), ("w2_int8", converted)):
        graph = export_onnx(net, x, output / f"{name}.onnx")
        if name == "w2_int8":
            rows = audit_attention_qdq(graph)
            expected_count = 2 * len(source.blocks)
            if config.use_cropr and config.qat_scope == "all":
                expected_count += len(source.cropr)
            quantized = [row for row in rows if row["both_inputs_qdq"]]
            if len(quantized) != expected_count:
                raise RuntimeError(f"Expected {expected_count} Q/DQ attention BMMs, found {len(quantized)}: {rows}")
            report["attention_onnx"] = rows
    # Use identical actual input bytes for both native engines.
    x.numpy().tofile(output / "images.bin")
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not options.export_only:
        report.update(backend="tensorrt_python", backend_version=trt.__version__,
                      device=torch.cuda.get_device_name(options.device), cuda_version=torch.version.cuda,
                      tf32=False, amp=False, cuda_graph=False,
                      timing="CUDA events, synchronized each forward; excludes build and transfers",
                      warmup_ms=options.warmup_ms, duration=options.duration, iterations=options.iterations)
        for name in ("fp32_master", "w2_int8"):
            report["engine_execution"] = f"building_or_running_{name}"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Building/benchmarking {name} with TensorRT Python API", flush=True)
            try:
                report[name] = benchmark_engine(trt, output, name, x, references[name], options)
            except Exception as error:
                report["engine_execution"] = f"failed_{name}"
                report["error"] = str(error)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                raise
        report["gpu_compute_speedup_vs_trt_fp32"] = (
            report["fp32_master"]["gpu_compute_ms_median"] / report["w2_int8"]["gpu_compute_ms_median"])
        report["engine_execution"] = "completed; inspect layer reports to confirm INT8 attention kernels"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
