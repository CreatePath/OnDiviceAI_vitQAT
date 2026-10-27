"""Training lifecycle independent of model construction and metric implementation."""
import csv
import copy
from importlib.metadata import version
import json
import math
import platform
import statistics
import time
from pathlib import Path

import timm
import torch
import torchvision

from .augment import AUGMENT_DEFAULTS, validate_augment_args
from .data import loader, make_data
from .training import evaluate, read_checkpoint, save_checkpoint, seed_all, train_epoch


def run(args, *, build_model, measure_model, train_epoch_fn=train_epoch,
        evaluate_fn=evaluate, data_factory=make_data):
    """Allow model, loss/training, evaluation and metrics to evolve independently."""
    args = copy.copy(args)
    if args.resume and args.eval_checkpoint:
        raise ValueError("Choose resume or eval-checkpoint")
    checkpoint = None
    if args.eval_checkpoint:
        checkpoint = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=True)
        # Preserve runtime choices; restore the saved experiment before building data/model.
        runtime = {"eval_checkpoint", "resume", "output", "data_root", "download", "device",
                   "workers", "threads", "batch_size", "latency_warmup", "latency_repeats",
                   "cuda_w2a8_inference", "experiment"}
        for key, value in checkpoint["config"].items():
            if key in vars(args) and key not in runtime:
                setattr(args, key, value)
        if args.experiment is None:
            args.experiment = checkpoint["config"].get("experiment")
        # Historical checkpoints predate configurable augmentation.
        for key, value in AUGMENT_DEFAULTS.items():
            if key not in checkpoint["config"]:
                setattr(args, key, value)
    if args.mode not in ("fp32", "qat"):
        raise ValueError("--mode is required for training; evaluation restores it from the checkpoint")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size or workers")
    if not args.eval_checkpoint:
        if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
            raise ValueError("Require epochs >= 1 and 0 <= warmup_epochs < epochs")
        if args.accum_steps < 1:
            raise ValueError("Invalid accumulation steps")
        if not 0 <= args.min_lr <= args.lr or args.lr <= 0 or args.clip_grad <= 0:
            raise ValueError("Require 0 <= min_lr <= lr, lr > 0, clip_grad > 0")
        if not 0 <= args.label_smoothing < 1 or not 0 <= args.drop_path < 1 or args.weight_decay < 0:
            raise ValueError("Invalid label smoothing, drop path or weight decay")
        validate_augment_args(args)
    if args.latency_repeats < 1 or args.latency_warmup < 0:
        raise ValueError("Require latency_repeats >= 1 and latency_warmup >= 0")
    if getattr(args, "cuda_w2a8_inference", False):
        if args.mode != "qat" or (args.weight_bits, args.activation_bits) != (2, 8):
            raise ValueError("--cuda-w2a8-inference requires --mode qat --weight-bits 2 --activation-bits 8")
        from .cuda_w2a8 import require_cuda_w2a8
        require_cuda_w2a8(args.device)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and (Path(args.resume).resolve() != (output / "last.pt").resolve() or not (output / "best.pt").exists()):
        raise ValueError("Resume must use this output directory's last.pt and retain best.pt")
    if not args.resume and not args.eval_checkpoint and any(output.iterdir()):
        raise ValueError("Choose an empty output directory or explicitly resume")
    seed_all(args.seed)
    train_data, val_data, test_data, manifest = data_factory(args)
    model = build_model(args).to(args.device)
    optimizer = None
    if not args.eval_checkpoint:
        decay, no_decay = [], []
        excluded = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
        for name, parameter in model.named_parameters():
            (no_decay if parameter.ndim == 1 or name in excluded else decay).append(parameter)
        optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=args.weight_decay),
                                       dict(params=no_decay, weight_decay=0.0)], lr=args.lr)
    start, best = 0, -1.0
    if args.resume:
        checkpoint = read_checkpoint(args.resume, model, args, manifest["sha256"], optimizer)
        start, best = checkpoint["epoch"] + 1, checkpoint["best_acc"]
    if args.eval_checkpoint:
        if checkpoint["split_hash"] != manifest["sha256"]:
            raise ValueError("Checkpoint data split does not match")
        model.load_state_dict(checkpoint["model"], strict=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
    if not args.eval_checkpoint:
        for epoch in range(start, args.epochs):
            train_loader = loader(train_data, args, training=True, epoch=epoch)
            stats = train_epoch_fn(model, train_loader, optimizer, args, epoch,
                                math.ceil(len(train_loader) / args.accum_steps))
            val = evaluate_fn(model, loader(val_data, args), args.device)
            improved = val["top1_acc"] > best
            best = max(best, val["top1_acc"])
            record = dict(epoch=epoch, **stats, val_loss=val["loss"], val_top1_acc=val["top1_acc"])
            print(json.dumps(record), flush=True)
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            save_checkpoint(output / "last.pt", model, optimizer, epoch, best, args, manifest["sha256"])
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, epoch, best, args, manifest["sha256"])
        selected = output / "best.pt"
        checkpoint = read_checkpoint(selected, model, args, manifest["sha256"])
    else:
        selected = Path(args.eval_checkpoint)
    test = evaluate_fn(model, loader(test_data, args), args.device)
    metrics = measure_model(model, args)
    if getattr(args, "cuda_w2a8_inference", False):
        from .cuda_w2a8 import A8Attention, A8CrossAttention, W2A8Linear, W2A8Conv2d, convert_cuda_w2a8
        cuda_model = convert_cuda_w2a8(model, args.device)
        cuda_test = evaluate_fn(cuda_model, loader(test_data, args), args.device)
        x = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)
        with torch.inference_mode():
            cuda_model(x)  # Compile the batch-1 shapes outside timing, even with warmup=0.
            for _ in range(args.latency_warmup):
                cuda_model(x)
            torch.cuda.synchronize(args.device)
            timings = []
            for _ in range(args.latency_repeats):
                start_time = time.perf_counter()
                cuda_model(x)
                torch.cuda.synchronize(args.device)
                timings.append((time.perf_counter() - start_time) * 1000)
        torch.save(cuda_model.state_dict(), output / "cuda_w2a8_state.pt")
        packed_layers = [m for m in cuda_model.modules() if isinstance(m, W2A8Linear)]
        metrics.update(cuda_w2a8_top1_acc=cuda_test["top1_acc"], cuda_w2a8_test_loss=cuda_test["loss"],
                       cuda_w2a8_latency_ms=statistics.median(timings),
                       cuda_w2a8_latency_ms_mean=statistics.mean(timings),
                       cuda_w2a8_latency_ms_p95=sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
                       cuda_w2a8_backend="bitblas",
                       cuda_w2a8_backend_version=version("bitblas"),
                       cuda_w2a8_device=args.device,
                       cuda_w2a8_compute_capability=list(torch.cuda.get_device_capability(args.device)),
                       cuda_w2a8_packed_weight_bytes=sum(m.packed_weight.numel() for m in packed_layers),
                       cuda_w2a8_converted_layers=len(packed_layers),
                       cuda_w2a8_conv_layers=sum(isinstance(m, W2A8Conv2d) for m in packed_layers),
                       cuda_w2a8_attention_layers=sum(isinstance(m, A8Attention) for m in cuda_model.modules()),
                       cuda_w2a8_cropr_scorers=sum(isinstance(m, A8CrossAttention) for m in cuda_model.modules()),
                       cuda_w2a8_remaining_fake_quantizers=0,  # Conversion rejects any remaining fake-quant module.
                       cuda_w2a8_state_bytes=(output / "cuda_w2a8_state.pt").stat().st_size,
                       cuda_w2a8_scope="QAT Linear/Conv W2A8, QAT attention/Cropr A8xA8; softmax/norm/residuals FP32; non-QAT Cropr remains FP32")
    torch.save(model.state_dict(), output / "model_state.pt")
    metrics.update(experiment=args.experiment,
                   top1_acc=test["top1_acc"], test_loss=test["loss"], test_samples=test["samples"],
                   selected_epoch=checkpoint["epoch"], best_val_top1_acc=checkpoint["best_acc"],
                   checkpoint=str(selected), split_hash=manifest["sha256"],
                   storage_model_state_bytes=(output / "model_state.pt").stat().st_size,
                   seed=args.seed, device=args.device,
                   device_name=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else platform.processor(),
                   torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                   timm_version=timm.__version__, python_version=platform.python_version(),
                   cpu_threads=torch.get_num_threads(), image_size=args.image_size,
                   pretrained=False, amp=False, tf32=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
