"""Inference-only Cropr routing; no random queries or backbone retraining.

Paper sections 3.1--3.5: learned query dot product, Top-K, first R+1,
and last-layer fusion. Keep the upstream classification sequence order.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from vitqat.cropr_adapter import Cropr


@dataclass(frozen=True)
class PruningConfig:
    pruning_rate: int = 16
    llf: bool = True
    first_stage_extra: int = 1

    def schedule(self, depth, patch_tokens):
        if (type(self.pruning_rate) is not int or self.pruning_rate < 1
                or self.llf is not True or self.first_stage_extra != 1):
            raise ValueError("Require positive integer R, LLF on and first-stage extra=1")
        if depth < 3:
            raise ValueError("LLF requires at least three transformer blocks")
        rates = (self.pruning_rate + 1,) + (self.pruning_rate,) * (depth - 3)
        if sum(rates) >= patch_tokens:
            raise ValueError("Pruning schedule must retain at least one patch token")
        return rates

    def describe(self, depth, patch_tokens):
        rates = self.schedule(depth, patch_tokens)
        return dict(asdict(self), schedule_id=f"cropr-r{self.pruning_rate}-first{rates[0]}-llf",
                    after_blocks=list(range(1, len(rates) + 1)), remove_counts=list(rates),
                    scorer="learned-query-dot-product", num_queries=1, num_heads=1,
                    q_proj=False, k_proj=False, v_proj=False, pre_attn_norm=False,
                    router_precision="fp32", selection="argsort-descending-stable-false",
                    fusion_order="keep,removed_stage1,...,removed_stageN")


class QueryRouter(nn.Module):
    """Router only: a supplied learned query, no aggregator or auxiliary head."""

    def __init__(self, query, pruning_rate):
        super().__init__()
        if (not isinstance(query, torch.Tensor) or query.ndim != 3
                or query.shape[:2] != (1, 1) or not query.is_floating_point()
                or not torch.isfinite(query).all()):
            raise ValueError("Query must be a finite floating tensor with shape [1,1,D]")
        self.query = nn.Parameter(query.detach().to(dtype=torch.float32).clone(), requires_grad=False)
        self.pruning_rate = pruning_rate
        self.embed_dim = query.shape[-1]
        self.num_queries = 1

    def scores(self, x):
        return (self.query.expand(x.shape[0], -1, -1) @ x.float().transpose(-2, -1)).squeeze(1)

    def forward(self, x):
        if self.training:
            raise RuntimeError("QueryRouter is inference-only; call eval()")
        if x.shape[1] <= self.pruning_rate + 1:
            raise ValueError("Router must retain CLS and at least one patch")
        scores = self.scores(x)
        scores[:, 0] = torch.inf
        # Directly reuse upstream selector and its score-sorted sequence order.
        return Cropr.prune(self, x, scores)


class InferencePrunedViT(nn.Module):
    """Wrap an already-loaded dense CroprViT without changing backbone state.

    The wrapper shares backbone parameters; it never constructs a new backbone.
    Trace IDs are optional diagnostics and are not calculated during timing.
    """

    @staticmethod
    def _validate_queries(backbone, queries, rates):
        expected = {str(i) for i in range(1, len(rates) + 1)}
        if not isinstance(queries, dict) or set(queries) != expected:
            raise ValueError(f"Query block mapping must be exactly {sorted(expected, key=int)}")
        if any(not isinstance(q, torch.Tensor) or q.shape != (1, 1, backbone.embed_dim)
               or not q.is_floating_point() or not torch.isfinite(q).all()
               for q in queries.values()):
            raise ValueError("Query dimension/finite floating values must match backbone embed_dim")

    def __init__(self, backbone, queries, config=None):
        super().__init__()
        if getattr(backbone, "use_cropr", False) or backbone.num_prefix_tokens != 1:
            raise ValueError("Require a dense backbone with exactly one CLS token")
        for method in ("forward_embed", "forward_head"):
            if not callable(getattr(backbone, method, None)):
                raise ValueError("Backbone must use the repository Cropr adapter forward contract")
        if any(p.dtype != torch.float32 for p in backbone.parameters()):
            raise ValueError("Inference pruning currently supports FP32/fake-quant FP32 backbones")
        self.config = config or PruningConfig()
        rates = self.config.schedule(len(backbone.blocks), backbone.patch_embed.num_patches)
        self._validate_queries(backbone, queries, rates)
        self.backbone = backbone
        self.routers = nn.ModuleList([QueryRouter(queries[str(i + 1)], rate)
                                      for i, rate in enumerate(rates)])
        self.routers.to(next(backbone.parameters()).device)
        self.enabled = True
        self.capture_trace = False
        self.last_trace = None
        self.eval()

    @property
    def blocks(self):
        return self.backbone.blocks

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    @property
    def num_prefix_tokens(self):
        return self.backbone.num_prefix_tokens

    def forward(self, images):
        self.last_trace = None
        if self.training or self.backbone.training:
            raise RuntimeError("InferencePrunedViT cannot train a backbone or query")
        if not self.enabled:
            return self.backbone(images)
        x = self.backbone.forward_embed(images)
        removed = []
        if self.capture_trace:
            ids = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
            removed_ids, counts = [], []
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                x = torch.cat([x] + removed, dim=1)
                if self.capture_trace:
                    ids = torch.cat([ids] + removed_ids, dim=1)
            if self.capture_trace:
                counts.append(x.shape[1])
            x = block(x)
            if i < len(self.routers):
                router = self.routers[i]
                if self.capture_trace:
                    scores = router.scores(x)
                    scores[:, 0] = torch.inf
                    order = torch.argsort(scores, dim=1, descending=True, stable=False)
                    reordered_ids = ids.gather(1, order)
                    keep = x.shape[1] - router.pruning_rate
                    ids, dropped_ids = reordered_ids[:, :keep], reordered_ids[:, keep:]
                    removed_ids.append(dropped_ids)
                x, dropped = router(x)
                removed.append(dropped)
        if self.capture_trace:
            self.last_trace = dict(tokens_per_block=counts, restored_ids=ids.detach().cpu(),
                                   removed_ids=[v.detach().cpu() for v in removed_ids])
        return self.backbone.forward_head(self.backbone.norm(x))
