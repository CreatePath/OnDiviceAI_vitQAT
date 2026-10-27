import math
import statistics
import time

import torch
from torch import nn
from vitqat.cropr_adapter import Cropr, CrossAttention

from vitqat.quantization import QATConv2d, QATLinear, Quantizer
from .pruning import AttentionRouter


@torch.inference_mode()
def measure_vit(model, image_size, device, warmup=10, repeats=100, *, sample=None):
    """Count actual inference shapes, including Cropr pruning and LLF restoration."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 are required")
    model.eval()
    x = torch.zeros(1, 3, image_size, image_size, device=device) if sample is None else sample
    if x.shape != (1, 3, image_size, image_size) or x.device != torch.device(device):
        # cuda and cuda:0 refer to the same current device.
        expected_device = torch.empty(0, device=device).device
        if x.shape != (1, 3, image_size, image_size) or x.device != expected_device:
            raise ValueError("Latency sample must have shape [1,3,image_size,image_size] on the evaluation device")
    flops = 0
    tokens_per_block, tokens_after_pruning = [], []

    def count_attention(module, inputs):
        nonlocal flops
        n = inputs[0].shape[1]
        tokens_per_block.append(n)
        flops += 4 * n**2 * model.embed_dim

    def count_pruner(module, inputs, output):
        nonlocal flops
        # Cropr scoring adds dot products; CLS attention reuses the existing attention matrix.
        if isinstance(module, Cropr):
            flops += 2 * module.num_queries * inputs[0].shape[1] * module.embed_dim
        tokens_after_pruning.append(output[0].shape[1])

    def count(module, inputs, output):
        nonlocal flops
        if isinstance(module, nn.Linear):
            flops += 2 * output.numel() * module.in_features
        elif isinstance(module, nn.Conv2d):
            flops += 2 * output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    hooks += [block.register_forward_pre_hook(count_attention) for block in model.blocks]
    hooks += [m.register_forward_hook(count_pruner) for m in model.modules() if isinstance(m, (Cropr, AttentionRouter))]
    try:
        model(x)
    finally:
        for handle in hooks:
            handle.remove()
    tokens = tokens_per_block[0]
    patch_tokens = tokens - model.num_prefix_tokens
    minimum_tokens = min(tokens_after_pruning, default=tokens)
    params = sum(p.numel() for p in model.parameters())
    auxiliary_params = 0
    for module in model.modules():
        if isinstance(module, Cropr) and hasattr(module, "head"):
            auxiliary_params += sum(p.numel() for p in module.head.parameters())
        if isinstance(module, CrossAttention):
            for name in ("v", "proj", "mlp_norm", "mlp"):
                branch = getattr(module, name, None)
                if branch is not None:
                    auxiliary_params += sum(p.numel() for p in branch.parameters())
    estimated_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_weights = 0
    quantized_tensors = []
    for module in model.modules():
        if isinstance(module, (QATLinear, QATConv2d)):
            quantized_tensors.append((module.weight, module.weight_quant.bits))
        if isinstance(module, CrossAttention) and isinstance(getattr(module, "query_quant", None), Quantizer):
            quantized_tensors.append((module.queries, module.query_quant.bits))
    for tensor, bits in quantized_tensors:
        n = tensor.numel()
        quantized_weights += n
        estimated_bytes -= n * tensor.element_size()
        estimated_bytes += math.ceil(n * bits / 8)
        estimated_bytes += tensor.shape[0] * 8  # float32 scale + int32 zero point/channel
    # Activation quantization metadata for a hypothetical packed model (no observers).
    for module in model.modules():
        if isinstance(module, Quantizer) and module.fake_quant.ch_axis == -1:
            estimated_bytes += 8

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model(x)
    sync()
    timings = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        model(x)
        sync()
        timings.append((time.perf_counter() - start) * 1000)
    ordered = sorted(timings)
    return dict(
        params=params, inference_params=params - auxiliary_params,
        quantized_weight_params=quantized_weights, gflops=flops / 1e9,
        router_params=sum(p.numel() for m in model.modules() if isinstance(m, AttentionRouter) for p in m.parameters()),
        flops_convention="2 FLOPs/MAC; inference Conv/Linear, ViT QK/AV, Cropr QK when present; CLS score reduction excluded; excludes bias, norm, softmax, sorting, quantization",
        storage_estimated_packed_bytes=estimated_bytes,
        storage_estimate_note="All parameter payload including training-only auxiliary branches + quantization scales/zero points; excludes serialization overhead; not an exported integer model",
        token_pruning_ratio=(tokens - minimum_tokens) / patch_tokens,
        final_token_pruning_ratio=(tokens - tokens_per_block[-1]) / patch_tokens,
        pruning_ratio_definition="Maximum fraction of patch tokens removed before LLF; excludes CLS",
        tokens_per_block=tokens_per_block, tokens_after_pruning=tokens_after_pruning,
        latency_ms_mean=statistics.mean(timings), latency_ms_median=statistics.median(timings),
        latency_ms_p95=ordered[math.ceil(0.95 * len(ordered)) - 1],
        latency_backend="fake_quant_fp32" if quantized_weights else "fp32",
        integer_latency_ms=None, latency_batch_size=1, latency_warmup=warmup, latency_repeats=repeats,
        latency_scope="model forward only; input already on device; no data loading or transfer",
    )


def measure_dense_vit(model, image_size, device, warmup=10, repeats=100):
    """Compatibility entry point for existing baseline callers."""
    if getattr(model, "use_cropr", False):
        raise ValueError("Use measure_vit for a model with Cropr enabled")
    return measure_vit(model, image_size, device, warmup, repeats)
