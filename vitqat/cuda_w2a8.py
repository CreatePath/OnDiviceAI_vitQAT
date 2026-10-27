"""BitBLAS W2A8 CUDA conversion preserving frozen QAT parameters."""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from . import w2a8_bitblas as ops
from .quantization import QATAttention, QATCrossAttention, QATConv2d, QATLinear, Quantizer


def require_cuda_w2a8(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("W2A8 inference requires an NVIDIA CUDA device")
    if torch.cuda.get_device_capability(device) < (8, 0):
        raise ValueError("W2A8 requires SM80 or newer; RTX 4090 is SM89")
    ops.require_bitblas()
    return ops


def _checked_quantizer(quantizer, bits):
    fq = quantizer.fake_quant
    if quantizer.bits != bits or not bool(fq.fake_quant_enabled.item()):
        raise ValueError(f"Packed W2A8 requires enabled {bits}-bit quantization")
    minimum = fq.activation_post_process.min_val
    if minimum.numel() == 0 or not torch.isfinite(minimum).all():
        raise ValueError("Uncalibrated QAT observer")
    if not torch.isfinite(fq.scale).all() or not (fq.scale > 0).all():
        raise ValueError("QAT scales must be finite and positive")
    return fq


class W2A8Linear(nn.Module):
    """Frozen BitBLAS layer; conversion and execution require CUDA."""
    def __init__(self, layer):
        super().__init__()
        wq = _checked_quantizer(layer.weight_quant, 2)
        iq = _checked_quantizer(layer.input_quant, 8)
        oq = _checked_quantizer(layer.output_quant, 8)
        weight = layer.weight.detach().float().flatten(1)
        self.out_features, self.in_features = weight.shape
        if self.in_features > (2**31 - 1) // (255 * 3):
            raise ValueError("Reduction dimension can overflow INT32 accumulation")
        scale, zero = wq.scale.detach().float(), wq.zero_point.detach().to(torch.int32)
        # FakeQuantize multiplies by the reciprocal before rounding.
        codes = (torch.round(weight * scale.reciprocal()[:, None]) + zero[:, None]).clamp(0, 3).to(torch.uint8)
        self.register_buffer("packed_weight", ops.prepare_weight(codes))
        self.register_buffer("weight_scale", scale.clone())
        self.register_buffer("weight_zero", zero.clone())
        self.register_buffer("weight_sum", (codes.int() - zero[:, None]).sum(1).to(torch.int32))
        self.register_buffer("bias", torch.zeros_like(scale) if layer.bias is None else layer.bias.detach().float().clone())
        for name, fq in (("input", iq), ("output", oq)):
            self.register_buffer(f"{name}_scale", fq.scale.detach().float().clone())
            self.register_buffer(f"{name}_zero", fq.zero_point.detach().to(torch.int32).clone())
        self.eval()

    def forward(self, x):
        if self.training:
            raise RuntimeError("W2A8 layers are inference-only; call eval()")
        if x.device != self.packed_weight.device or x.device.type != "cuda" or x.dtype != torch.float32:
            raise ValueError("W2A8 input must be FP32 on the same CUDA device as the weights")
        if x.shape[-1] != self.in_features:
            raise ValueError("W2A8 input feature dimension does not match the layer")
        linear = ops.linear
        shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features).contiguous()
        return linear(x, self).reshape(*shape, self.out_features)


class W2A8Conv2d(W2A8Linear):
    """Unfold + packed GEMM for ViT patch projection (groups=1, zero padding)."""
    def __init__(self, layer):
        if layer.groups != 1 or layer.padding_mode != "zeros" or isinstance(layer.padding, str):
            raise ValueError("W2A8 Conv2d supports groups=1 and numeric zero padding")
        super().__init__(layer)
        self.kernel_size, self.stride = layer.kernel_size, layer.stride
        self.padding, self.dilation = layer.padding, layer.dilation

    def forward(self, x):
        height, width = [
            (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            for size, pad, dilation, kernel, stride in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)
        ]
        patches = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride).transpose(1, 2)
        output = super().forward(patches)
        return output.transpose(1, 2).reshape(x.shape[0], self.out_features, height, width).contiguous()


class A8Quantizer(nn.Module):
    """Frozen QAT parameters; forward produces INT8 codes, not fake-quant FP32."""
    def __init__(self, source):
        super().__init__()
        fq = _checked_quantizer(source, 8)
        self.register_buffer("scale", fq.scale.detach().float().clone())
        self.register_buffer("zero", fq.zero_point.detach().to(torch.int32).clone())
        self.eval()

    def forward(self, x):
        if self.training:
            raise RuntimeError("A8 quantizers are inference-only; call eval()")
        quantize_activation = ops.quantize_activation
        return quantize_activation(x, self.scale, self.zero)


class A8Attention(nn.Module):
    def __init__(self, source):
        super().__init__()
        for name in ("qkv", "q_norm", "k_norm", "attn_drop", "proj", "proj_drop"):
            setattr(self, name, getattr(source, name))
        self.num_heads, self.head_dim, self.scale = source.num_heads, source.head_dim, source.scale
        for name in ("q_quant", "k_quant", "v_quant", "prob_quant"):
            setattr(self, name, A8Quantizer(getattr(source, name)))
        self.eval()

    def forward(self, x):
        activation_matmul = ops.activation_matmul
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k, v = self.q_quant(self.q_norm(q)), self.k_quant(self.k_norm(k)), self.v_quant(v)
        logits = activation_matmul(q, k.transpose(-2, -1), self.q_quant, self.k_quant, factor=self.scale)
        prob = self.prob_quant(self.attn_drop(logits.softmax(dim=-1)))
        x = activation_matmul(prob, v, self.prob_quant, self.v_quant).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class A8CrossAttention(nn.Module):
    """Cropr attention with a constant-folded W2 query and A8 integer matmuls."""
    def __init__(self, source):
        super().__init__()
        self.num_queries, self.head_dim, self.scale = source.num_queries, source.head_dim, source.scale
        for name in ("k", "attn_norm", "v", "proj", "mlp_norm", "mlp"):
            if hasattr(source, name):
                setattr(self, name, getattr(source, name))
        for name in ("q_quant", "k_quant", "v_quant", "prob_quant"):
            if hasattr(source, name):
                setattr(self, name, A8Quantizer(getattr(source, name)))
        # The learnable query and its optional projection are constant in inference.
        # Apply its trained W2 grid once, then cache the final A8 Q codes.
        fq = _checked_quantizer(source.query_quant, 2)
        scale, zero = fq.scale.reshape(-1, 1, 1), fq.zero_point.reshape(-1, 1, 1)
        with torch.no_grad():
            codes = (torch.round(source.queries * scale.reciprocal()) + zero).clamp(0, 3)
            query = (codes - zero) * scale
            q = self.q_quant(source.q(query)).reshape(1, self.num_queries, -1, self.head_dim).transpose(1, 2)
        self.register_buffer("query_codes", q.contiguous())
        self.eval()

    def _query_key(self, x):
        b, n = x.shape[:2]
        q = self.query_codes.expand(b, -1, -1, -1)
        k = self.k_quant(self.k(x)).reshape(b, n, -1, self.head_dim).transpose(1, 2)
        return q, k

    def forward_scorer(self, x):
        activation_matmul = ops.activation_matmul
        q, k = self._query_key(self.attn_norm(x))
        return activation_matmul(q, k.transpose(-2, -1), self.q_quant, self.k_quant).sum((1, 2))

    def forward(self, x):
        # Retain the auxiliary aggregation API, although normal inference only calls the scorer.
        activation_matmul = ops.activation_matmul
        b, n, d = x.shape
        x = self.attn_norm(x)
        q, k = self._query_key(x)
        v = self.v_quant(self.v(x)).reshape(b, n, -1, self.head_dim).transpose(1, 2)
        attn = activation_matmul(q, k.transpose(-2, -1), self.q_quant, self.k_quant)
        prob = self.prob_quant((attn * self.scale).softmax(dim=-1))
        x = activation_matmul(prob, v, self.prob_quant, self.v_quant).transpose(1, 2).reshape(b, self.num_queries, d).squeeze(1)
        x = self.proj(x)
        if hasattr(self, "mlp"):
            x = x + self.mlp(self.mlp_norm(x))
        return x, attn.sum((1, 2))


def convert_cuda_w2a8(model, device):
    """Convert every QAT site on a copy; floating-point nonlinear operations remain."""
    require_cuda_w2a8(device)
    model = copy.deepcopy(model).to(device).eval()
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (QATLinear, QATConv2d)):
            converted = W2A8Conv2d(module) if isinstance(module, QATConv2d) else W2A8Linear(module)
            if not name:
                return converted
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, converted)
            count += 1
    if not count:
        raise ValueError("No QAT Linear/Conv2d layers found")
    for name, module in list(model.named_modules()):
        if isinstance(module, (QATAttention, QATCrossAttention)):
            converted = A8Attention(module) if isinstance(module, QATAttention) else A8CrossAttention(module)
            if not name:
                model = converted
            else:
                parent_name, _, child = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child, converted)
    remaining = [name for name, module in model.named_modules() if isinstance(module, (Quantizer, FakeQuantizeBase))]
    if remaining:
        raise ValueError(f"Unconverted fake quantization sites: {remaining}")
    return model
