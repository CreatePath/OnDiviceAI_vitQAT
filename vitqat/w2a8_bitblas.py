"""BitBLAS W2A8 and A8 attention; no project-owned GPU kernels.

BitBLAS computes integer dot products. PyTorch applies the original QAT affine
corrections and scales. Compilation is cached per shape/device, before timing.
"""
from functools import lru_cache
import platform

import torch
from torch.nn import functional as F


def require_bitblas():
    if platform.system() != "Linux":
        raise RuntimeError("BitBLAS requires Linux CUDA; see docs/cuda-w2a8.md. No fallback was used.")
    try:
        import bitblas
    except ImportError as error:
        raise RuntimeError("Install requirements-bitblas.txt in the Linux CUDA environment") from error
    return bitblas


def _ceil(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@lru_cache(maxsize=128)
def _operator(m, n, k, weight_dtype, device_index):
    bitblas = require_bitblas()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up all BitBLAS shapes before CUDA Graph capture")
    with torch.cuda.device(device_index):
        config = bitblas.MatmulConfig(
            M=m, N=n, K=k, A_dtype="int8", W_dtype=weight_dtype,
            accum_dtype="int32", out_dtype="int32", layout="nt",
            with_bias=False, with_scaling=False, with_zeros=False,
            fast_decoding=False, propagate_a=False, propagate_b=False)
        return bitblas.Matmul(config, target=bitblas.auto_detect_nvidia_target(), enable_tuning=False)


def prepare_weight(codes):
    """Use BitBLAS's own UINT2 packing once, outside inference."""
    n, k = codes.shape
    np, kp = _ceil(n, 16), _ceil(k, 32)
    with torch.cuda.device(codes.device):
        op = _operator(16, np, kp, "uint2", codes.device.index)
        return op.transform_weight(F.pad(codes, (0, kp - k, 0, np - n))).contiguous()


def quantize_activation(x, scale, zero):
    if x.device.type != "cuda" or x.dtype != torch.float32 or x.device != scale.device:
        raise ValueError("A8 quantization requires FP32 input and qparams on the same CUDA device")
    scaled = torch.minimum(torch.maximum(x * scale.reciprocal(), -zero), 255 - zero)
    return (scaled.round() + zero - 128).to(torch.int8)


def linear(x, layer, *, raw=False):
    a = quantize_activation(x, layer.input_scale, layer.input_zero)
    m, k = a.shape
    n = layer.out_features
    if not m:
        return torch.empty((0, n), dtype=torch.int32 if raw else torch.float32, device=x.device)
    mp, np, kp = _ceil(m, 16), _ceil(n, 16), _ceil(k, 32)
    with torch.cuda.device(x.device):
        op = _operator(mp, np, kp, "uint2", x.device.index)
        acc = op(F.pad(a, (0, kp - k, 0, mp - m)), layer.packed_weight)[:m, :n]
    # Raw weights are unsigned QAT codes, not centered/requantized INT2 values.
    acc = acc - a.sum(-1, dtype=torch.int32)[:, None] * layer.weight_zero
    acc = acc + (128 - layer.input_zero) * layer.weight_sum
    if raw:
        return acc
    value = (acc.float() * layer.input_scale) * layer.weight_scale + layer.bias
    scaled = value * layer.output_scale.reciprocal()
    scaled = torch.minimum(torch.maximum(scaled, -layer.output_zero), 255 - layer.output_zero)
    return scaled.round() * layer.output_scale


def activation_matmul(a, b, aq, bq, *, factor=1.0, raw=False):
    if a.dtype != torch.int8 or b.dtype != torch.int8 or a.device.type != "cuda" or a.device != b.device:
        raise ValueError("A8 matmul requires INT8 operands on the same CUDA device")
    if a.ndim < 3 or a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
        raise ValueError("A8 matmul requires matching batch and reduction dimensions")
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    if k > (2**31 - 1) // 65536:
        raise ValueError("A8 matmul reduction can overflow INT32")
    shape = (*a.shape[:-2], m, n)
    if not a.numel() or not b.numel():
        return torch.zeros(shape, device=a.device, dtype=torch.int32 if raw else torch.float32)
    mp, np, kp = _ceil(m, 16), _ceil(n, 16), _ceil(k, 32)
    aa = F.pad(a.reshape(-1, m, k), (0, kp - k, 0, mp - m)).contiguous()
    bb = F.pad(b.transpose(-2, -1).reshape(-1, n, k), (0, kp - k, 0, np - n)).contiguous()
    output = torch.empty((aa.shape[0], mp, np), device=a.device, dtype=torch.int32)
    with torch.cuda.device(a.device):
        op = _operator(mp, np, kp, "int8", a.device.index)
        # ponytail: BitBLAS Matmul is 2D; loop over independent heads/batches.
        # Replace with a library batched API if launch cost dominates measurements.
        for index in range(aa.shape[0]):
            op(aa[index], bb[index], output=output[index])
    acc = output[:, :m, :n].reshape(shape)
    da, db = 128 - aq.zero, 128 - bq.zero
    acc = acc + da * b.sum(-2, dtype=torch.int32).unsqueeze(-2)
    acc = acc + db * a.sum(-1, dtype=torch.int32).unsqueeze(-1) + k * da * db
    if raw:
        return acc
    return ((acc.float() * aq.scale) * bq.scale) * factor
