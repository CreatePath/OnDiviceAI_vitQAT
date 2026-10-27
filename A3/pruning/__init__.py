"""Reusable Cropr API. No datasets, CLI parsing or checkpoint I/O on import."""
from .inference import InferencePrunedViT, PruningConfig, AttentionRouter
from .model import PrunedOutput, attach_pruning

__all__ = ["attach_pruning", "PrunedOutput", "PruningConfig",
           "InferencePrunedViT", "AttentionRouter"]
