"""
Name That LLM - Authorship Attribution Engine
==============================================
Ensemble package for LLM passage attribution.
"""

__version__ = "0.42.0"

from .config import LABELS, K, LAB2I, FAMILIES, align_cols, log_softmax, softmax
from .lexical_engine import run_lexical_engine
from .neural_extractor import (
    forward_and_pool_batch,
    resolve_head_tensors,
    merge_lora,
    in_place_copy,
    compute_centered_logits,
    UniversalLoRATrunkLoader
)
from .wpca_gaussian import (
    fit_shared_gaussian_lowram,
    transform_lowram,
    ExactAuthorPosterior,
    split_oversized_clusters_dist,
    scores_exact_dp
)
from .spectral_bank import fit_joint_spectral_bank
from .postprocess import (
    blend_main_and_complementary,
    apply_claude_specialist_gate,
    apply_curly_quote_shield,
    sinkhorn_balancing
)

__all__ = [
    "__version__",
    "LABELS", "K", "LAB2I", "FAMILIES", "align_cols", "log_softmax", "softmax",
    "run_lexical_engine",
    "forward_and_pool_batch", "resolve_head_tensors", "merge_lora", "in_place_copy",
    "compute_centered_logits", "UniversalLoRATrunkLoader",
    "fit_shared_gaussian_lowram", "transform_lowram", "ExactAuthorPosterior",
    "split_oversized_clusters_dist", "scores_exact_dp",
    "fit_joint_spectral_bank",
    "blend_main_and_complementary", "apply_claude_specialist_gate",
    "apply_curly_quote_shield", "sinkhorn_balancing"
]
