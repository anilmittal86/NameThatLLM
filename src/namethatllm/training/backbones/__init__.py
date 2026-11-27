"""
Backbone Adapter Registry Exports. Importing this module registers `qwen`, `deberta`, and `bge`.
"""

from __future__ import annotations

from .base import (
    _BACKBONE_REGISTRY,
    BaseBackboneAdapter,
    PrecisionSpec,
    TokenizationSpec,
    register_backbone,
    resolve_backbone_adapter,
)
from .bge import BgeAdapter, package_encoder_ridge_checkpoint
from .deberta import DebertaAdapter, run_deberta_specialist_suite
from .qwen import QwenAdapter, _disable_final_norm, _stabilize_fp16_backbone

__all__ = [
    "BaseBackboneAdapter",
    "BgeAdapter",
    "DebertaAdapter",
    "PrecisionSpec",
    "QwenAdapter",
    "TokenizationSpec",
    "_BACKBONE_REGISTRY",
    "_disable_final_norm",
    "_stabilize_fp16_backbone",
    "package_encoder_ridge_checkpoint",
    "register_backbone",
    "resolve_backbone_adapter",
    "run_deberta_specialist_suite",
]
