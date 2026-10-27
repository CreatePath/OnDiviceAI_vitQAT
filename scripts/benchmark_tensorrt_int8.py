"""Export W2-valued INT8 ONNX and compare TensorRT engines using trtexec."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

import torch

from vitqat.cli import parser as training_parser
from vitqat.models import build_model
from vitqat.tensorrt_int8 import audit_attention_qdq, convert_tensorrt_int8, export_onnx
from vitqat.training import seed_all


def read_native_results(output, name, reference):
    timings = json.loads((output / f"{name}.times.json").read_text(encoding="utf-8"))
    times = sorted(row["computeMs"] for row in timings)
    if not times or not all(math.isfinite(value) and value > 0 for value in times):
        raise ValueError(f"{name}: missing or invalid native GPU timings")
    outputs = json.loads((output / f"{name}.outputs.json").read_text(encoding="utf-8"))
    logits = torch.tensor(next(item["values"] for item in outputs if item["name"] == "logits"))
    logits = logits.reshape(reference.shape)
    if not torch.isfinite(logits).all():
        raise ValueError(f"{name}: nonfinite native logits")
    return dict(gpu_compute_ms_median=statistics.median(times),
                gpu_compute_ms_p95=times[math.ceil(len(times) * 0.95) - 1], samples=len(times),
                synthetic_native_vs_reference_max_abs_error=(logits - reference).abs().max().item(),
                synthetic_native_vs_reference_top1_agreement=(logits.argmax(-1) == reference.argmax(-1)).float().mean().item())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Directory for ONNX, engines, logs and reports")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--export-only", action="store_true", help="CPU export; no TensorRT required")
    parser.add_argument("--trtexec", default="trtexec", help="Path to standard TensorRT 10.x trtexec")
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--duration", type=int, default=10, help="Minimum benchmark seconds per engine")
    parser.add_argument("--iterations", type=int, default=100)
    options = parser.parse_args(argv)
    if options.batch_size < 1 or options.warmup_ms < 0 or options.duration < 1 or options.iterations < 1:
        parser.error("batch-size/duration/iterations must be positive; warmup-ms must be nonnegative")
    executable = shutil.which(options.trtexec)
    if not options.export_only and executable is None:
        parser.error("trtexec not found; install standard TensorRT or use --export-only")
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
                  engine_execution="not_run", kernel_precision_verified=False, commands=[],
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
        for name in ("fp32_master", "w2_int8"):
            command = [executable, f"--onnx={output / (name + '.onnx')}",
                       f"--saveEngine={output / (name + '.engine')}",
                       "--noTF32", "--noDataTransfers", "--useSpinWait",
                       "--profilingVerbosity=detailed", "--dumpLayerInfo",
                       f"--exportLayerInfo={output / (name + '.layers.json')}",
                       f"--exportTimes={output / (name + '.times.json')}",
                       f"--exportOutput={output / (name + '.outputs.json')}",
                       f"--loadInputs='images':{output / 'images.bin'}",
                       f"--warmUp={options.warmup_ms}", f"--duration={options.duration}",
                       f"--iterations={options.iterations}"]
            report["commands"].append(command)
            report["engine_execution"] = f"building_or_running_{name}"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Building/benchmarking {name}; see {output / (name + '.log')}", flush=True)
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                try:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError:
                    report["engine_execution"] = f"failed_{name}; see {name}.log"
                    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    raise
            report[name] = read_native_results(output, name, references[name])
        report["gpu_compute_speedup_vs_trt_fp32"] = (
            report["fp32_master"]["gpu_compute_ms_median"] / report["w2_int8"]["gpu_compute_ms_median"])
        report["engine_execution"] = "completed; inspect layer reports to confirm INT8 attention kernels"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
