"""
Name That LLM - Authorship Attribution Engine
==============================================
Ensemble package for LLM passage attribution.
"""

__version__ = "0.42.0"

from .config import LABELS, K, LAB2I, FAMILIES, align_cols, log_softmax, softmax
from .lexical_engine import (
    run_lexical_engine,
    extract_handwriting_features,
    extract_handwriting_and_orthographic_features,
    fit_handwriting_model,
    shape_text,
    function_text,
    FUNCTION_WORDS,
    STOP_WORDS
)
from .neural_extractor import (
    forward_and_pool_batch,
    resolve_head_tensors,
    merge_lora,
    in_place_copy,
    compute_centered_logits,
    UniversalLoRATrunkLoader,
    read_checkpoint_meta,
    extract_deberta_checkpoint
)
from .wpca_gaussian import (
    fit_shared_gaussian_lowram,
    transform_lowram,
    ExactAuthorPosterior,
    split_oversized_clusters_dist,
    scores_exact_dp
)
from .spectral_bank import fit_joint_spectral_bank, make_spectral_blocks
from .postprocess import (
    blend_main_and_complementary,
    apply_claude_specialist_gate,
    build_orthographic_penalty_matrix,
    apply_unified_handwriting_shield,
    apply_curly_quote_shield,
    sinkhorn_balancing,
    generate_submission
)
from .pipeline import run_submission_pipeline

__all__ = [
    "__version__",
    "LABELS", "K", "LAB2I", "FAMILIES", "align_cols", "log_softmax", "softmax",
    "run_lexical_engine", "extract_handwriting_features", "extract_handwriting_and_orthographic_features",
    "fit_handwriting_model", "shape_text", "function_text", "FUNCTION_WORDS", "STOP_WORDS",
    "forward_and_pool_batch", "resolve_head_tensors", "merge_lora", "in_place_copy",
    "compute_centered_logits", "UniversalLoRATrunkLoader", "read_checkpoint_meta", "extract_deberta_checkpoint",
    "fit_shared_gaussian_lowram", "transform_lowram", "ExactAuthorPosterior",
    "split_oversized_clusters_dist", "scores_exact_dp",
    "fit_joint_spectral_bank", "make_spectral_blocks",
    "blend_main_and_complementary", "apply_claude_specialist_gate",
    "build_orthographic_penalty_matrix", "apply_unified_handwriting_shield",
    "apply_curly_quote_shield", "sinkhorn_balancing", "generate_submission",
    "run_submission_pipeline"
]
