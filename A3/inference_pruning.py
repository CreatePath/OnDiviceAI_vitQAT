"""Compatibility imports; new integrations should import A3.pruning."""
from .pruning import InferencePrunedViT, PruningConfig, AttentionRouter

__all__ = ["InferencePrunedViT", "PruningConfig", "AttentionRouter"]
