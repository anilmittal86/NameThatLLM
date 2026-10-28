"""
Postprocessing, Fusion & Calibration Engine
===========================================
Pure two-tier convex mixture (85% Main + 15% Complementary),
mass-preserving Claude Specialist Gate, gentle curly quote shield,
and official 300-step exact multiplicative Sinkhorn mass balancing.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from .config import LABELS, K, FAMILIES, softmax

def blend_main_and_complementary(
    LOGIT_MAIN: np.ndarray,
    COMP_LIST: List[Tuple[float, np.ndarray]],
    main_ratio: float = 0.85,
    comp_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blends Main Stack and Complementary Stack in probability space:
    P_mix = main_ratio * R_main + comp_ratio * P_comp
    """
    R_main = softmax(LOGIT_MAIN, axis=1)
    if len(COMP_LIST) > 0:
        comp_weight_total = sum(w for w, _ in COMP_LIST)
        P_comp = sum(w * P_c for w, P_c in COMP_LIST) / max(1e-8, comp_weight_total)
        P_mix = main_ratio * R_main + comp_ratio * P_comp
    else:
        P_mix = R_main
    return P_mix, R_main

def apply_claude_specialist_gate(
    P_mix: np.ndarray,
    R_main: np.ndarray,
    claude_spec_logits: Optional[np.ndarray],
    claude_classes: Optional[list],
    weight: float = 0.15
) -> np.ndarray:
    """
    Refines passages predicted in Claude family using specialist head logits.
    Mass-preserving within the Claude family (preserves exact marginal family mass).
    """
    if claude_spec_logits is None:
        return P_mix

    claude_cols = FAMILIES["claude"]
    P_claude_head = softmax(claude_spec_logits, axis=-1)
    if claude_classes is not None:
        src_c = [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in claude_classes]
        c_idx = [src_c.index(LABELS[c]) for c in claude_cols]
        P_claude_head = P_claude_head[:, c_idx]

    pred_main = R_main.argmax(axis=1)
    claude_mask = np.isin(pred_main, claude_cols)
    if claude_mask.any():
        for i in np.where(claude_mask)[0]:
            tot_mass = P_mix[i, claude_cols].sum()
            blended = (1.0 - weight) * P_mix[i, claude_cols] + weight * (tot_mass * P_claude_head[i])
            P_mix[i, claude_cols] = blended
    return P_mix

def apply_curly_quote_shield(
    P_mix: np.ndarray,
    test_texts: List[str],
    multiplier: float = 0.36787944117144233  # np.exp(-1.0)
) -> np.ndarray:
    """
    Gentle zero-occurrence guard on Claude and GLM curly quote absence.
    Multiplies target columns by multiplier and re-normalizes rows to simplex.
    """
    claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
    for i, txt in enumerate(test_texts):
        txt_s = str(txt)
        if any(q in txt_s for q in ['\u2018', '\u2019', '\u201c', '\u201d']):
            P_mix[i, claude_and_glm_idx] *= multiplier

    # Re-normalize rows to simplex
    P_mix /= P_mix.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P_mix

def sinkhorn_balancing(P: np.ndarray, iters: int = 300) -> np.ndarray:
    """Official 300-step exact multiplicative Sinkhorn column mass balancing."""
    if len(P) < 50:
        return P
    P = np.clip(P.astype(np.float64), 1e-12, 1.0)
    target_col = 1.0 / P.shape[1]
    for _ in range(iters):
        P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
        P /= (P.sum(axis=0, keepdims=True) / (target_col * P.shape[0])).clip(min=1e-12)
    P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P

def generate_submission(
    P_final: np.ndarray,
    test_ids: pd.Series,
    working_dir: Path
) -> Tuple[pd.DataFrame, Path, Path]:
    """Generates submission.parquet and submission.csv with exact schema."""
    preds = [LABELS[idx] for idx in P_final.argmax(axis=1)]
    df_sub = pd.DataFrame({"id": test_ids, "model": preds})
    sub_parquet_path = working_dir / "submission.parquet"
    sub_csv_path = working_dir / "submission.csv"

    df_sub.to_parquet(sub_parquet_path, index=False)
    df_sub.to_csv(sub_csv_path, index=False)
    return df_sub, sub_parquet_path, sub_csv_path
