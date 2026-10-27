"""Checkpoint-based GPU comparison; run with python -m scripts.benchmark_w2a8."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time
from importlib.metadata import version

import torch

from vitqat.cli import parser as training_parser
from vitqat.cuda_w2a8 import convert_cuda_w2a8, require_cuda_w2a8
from vitqat.models import build_model
from vitqat.training import seed_all


@torch.inference_mode()
def benchmark(model, x, warmup, repeats):
    model.eval()
    model(x)  # JIT compilation is never measured.
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize(x.device)
    host_times, gpu_times = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize(x.device)
        host_times.append((time.perf_counter() - start) * 1000)
    # A separate pass avoids including CUDA event creation in the host measurements.
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(x.device):
        for _ in range(repeats):
            start_event.record()
            model(x)
            end_event.record()
            end_event.synchronize()
            gpu_times.append(start_event.elapsed_time(end_event))
    return dict(host_ms_median=statistics.median(host_times), host_ms_mean=statistics.mean(host_times),
                host_ms_p95=sorted(host_times)[math.ceil(0.95 * repeats) - 1],
                cuda_event_ms_median=statistics.median(gpu_times))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    options = parser.parse_args(argv)
    if options.batch_size < 1 or options.warmup < 0 or options.repeats < 1:
        parser.error("batch-size/repeats must be positive and warmup nonnegative")
    require_cuda_w2a8(options.device)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    config = training_parser().parse_args(["--mode", "qat", "--output", "unused"])
    vars(config).update(checkpoint["config"])
    if config.mode != "qat" or (config.weight_bits, config.activation_bits) != (2, 8):
        parser.error("A W2A8 QAT checkpoint is required")
    seed_all(config.seed)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(options.device).eval()
    packed = convert_cuda_w2a8(model, options.device)
    fp_config = copy.copy(config)
    fp_config.mode = "fp32"
    fp32 = build_model(fp_config)
    # Same master parameters without QAT modules: latency baseline, not a separately trained FP32 model.
    fp32.load_state_dict({k: checkpoint["model"][k] for k in fp32.state_dict()}, strict=True)
    fp32.to(options.device).eval()
    x = torch.randn(options.batch_size, 3, config.image_size, config.image_size, device=options.device)
    with torch.inference_mode():
        reference, actual = model(x), packed(x)
    results = {name: benchmark(net, x, options.warmup, options.repeats)
               for name, net in (("fp32_master", fp32), ("fake_quant_w2a8", model), ("packed_w2a8", packed))}
    results.update(checkpoint=str(options.checkpoint), device=torch.cuda.get_device_name(options.device),
                   compute_capability=list(torch.cuda.get_device_capability(options.device)),
                   torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                   backend="bitblas",
                   backend_version=version("bitblas"),
                   batch_size=options.batch_size,
                   image_size=config.image_size, warmup=options.warmup, repeats=options.repeats,
                   tf32=False, amp=False, cuda_graph=False,
                   synthetic_logit_max_abs_error=(reference - actual).abs().max().item(),
                   synthetic_top1_agreement=(reference.argmax(-1) == actual.argmax(-1)).float().mean().item(),
                   note="Synthetic input; not Food-101 accuracy. Timings include model forward, not data loading/transfers. CUDA events can include host launch gaps.")
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
