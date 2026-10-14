import timm
from timm.models.vision_transformer import VisionTransformer
from .cropr_adapter import adapt_model

from .quantization import prepare_qat

def build_scratch_vit(args):
    if args.image_size <= 0:
        raise ValueError("image_size must be positive")
    model = timm.create_model(args.model, pretrained=False, num_classes=101,
                              img_size=args.image_size, global_pool="token", drop_path_rate=args.drop_path)
    if not isinstance(model, VisionTransformer):
        raise ValueError("--model must name a timm VisionTransformer architecture")
    if any(args.image_size % size for size in model.patch_embed.patch_size):
        raise ValueError("image_size must be divisible by the model patch size")
    # Use the same explicit attention computation for FP32 and QAT latency comparisons.
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


def build_model(args):
    model = build_scratch_vit(args)
    model = adapt_model(model, args)
    if args.mode == "qat":
        prepare_qat(model, args.weight_bits, args.activation_bits, scope=args.qat_scope)
    return model
