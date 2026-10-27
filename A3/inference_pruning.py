"""Compatibility imports; new integrations should import A3.pruning."""
from .pruning import InferencePrunedViT, PruningConfig, QueryRouter

__all__ = ["InferencePrunedViT", "PruningConfig", "QueryRouter"]
