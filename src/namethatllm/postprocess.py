"""
Postprocessing, Fusion & Calibration Engine
===========================================
Pure two-tier convex mixture (85% Main + 15% Complementary),
mass-preserving Claude Specialist Gate, gentle curly quote shield,
and official 300-step exact multiplicative Sinkhorn mass balancing.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from .config import LABELS, K, LAB2I, FAMILIES, softmax
from .lexical_engine import extract_handwriting_and_orthographic_features

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
    claude_classes: Optional[list] = None,
    labels: list = LABELS,
    claude_cols: list = FAMILIES["claude"],
    gate_weight: float = 0.15,
    weight: Optional[float] = None,
) -> np.ndarray:
    """
    Refines passages predicted in Claude family using specialist head logits.
    Mass-preserving within the Claude family (preserves exact marginal family mass).
    Vectorized NumPy implementation.
    """
    if claude_spec_logits is None:
        return P_mix
    if weight is not None:
        gate_weight = weight

    P_out = P_mix.copy()
    P_claude_head = softmax(claude_spec_logits, axis=-1)
    if claude_classes is not None:
        src_c = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in claude_classes]
        c_idx = [src_c.index(labels[c]) for c in claude_cols]
        P_claude_head = P_claude_head[:, c_idx]

    claude_mask = np.isin(R_main.argmax(axis=1), claude_cols)
    if claude_mask.any():
        rows = np.flatnonzero(claude_mask)
        sub_mix = P_out[np.ix_(rows, claude_cols)]
        tot_mass = sub_mix.sum(axis=1, keepdims=True)
        blended = (1.0 - gate_weight) * sub_mix + gate_weight * (tot_mass * P_claude_head[rows])
        P_out[np.ix_(rows, claude_cols)] = blended
    return P_out

def build_orthographic_penalty_matrix(
    curly_quote_log_penalty: float = -1.00,      # Rule 1: 0/1049 Claude, 0/300 GLM
    multi_em_dash_log_penalty: float = 0.0,      # Rule 2: >=2 em-dashes (v44 default: 0.0)
    straight_contr_log_penalty: float = 0.0,     # Rule 3: >=2 straight prose contractions (v44 default: 0.0)
    curly_contr_log_penalty: float = 0.0,        # Rule 4: extra curly contraction penalty (v44 default: 0.0)
) -> np.ndarray:
    """
    Returns the (4, 18) log-penalty matrix W_pen mapping the 4 discrete orthographic
    rule indicators directly to the 18 canonical model classes.
    """
    W_pen = np.zeros((4, K), dtype=np.float64)
    claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
    gemini_idx = FAMILIES["gemini"]
    sol_idx = LAB2I["gpt-5.6-sol"]

    # Rule 1: Any curly quote -> penalize Claude (5) and GLM (2)
    W_pen[0, claude_and_glm_idx] = float(curly_quote_log_penalty)
    # Rule 2: >= 2 prose em-dashes -> gently penalize Gemini (4)
    W_pen[1, gemini_idx] = float(multi_em_dash_log_penalty)
    # Rule 3: >= 2 prose straight contractions (with 0 curly) -> penalize gpt-5.6-sol
    W_pen[2, sol_idx] = float(straight_contr_log_penalty)
    # Rule 4: >= 1 prose curly contraction -> reinforce penalty on Claude & GLM
    W_pen[3, claude_and_glm_idx] = float(curly_contr_log_penalty)
    return W_pen

def _extract_rules_fallback(texts: List[str]) -> np.ndarray:
    """Fallback discrete rules extractor if lexical_engine is not in namespace."""
    re_code = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
    re_str = re.compile(r"[A-Za-z]'[A-Za-z]")
    re_cur = re.compile(r"[A-Za-z]\u2019[A-Za-z]")
    rules = np.zeros((len(texts), 4), dtype=np.float32)
    for i, raw_t in enumerate(texts):
        t = str(raw_t)
        prose = re_code.sub(" ", t)
        raw_curly = t.count("\u2018") + t.count("\u2019") + t.count("\u201c") + t.count("\u201d")
        rules[i, 0] = 1.0 if raw_curly > 0 else 0.0
        rules[i, 1] = 1.0 if prose.count("\u2014") >= 2 else 0.0
        rules[i, 2] = 1.0 if (len(re_str.findall(prose)) >= 2 and (t.count("\u2018") + t.count("\u2019")) == 0) else 0.0
        rules[i, 3] = 1.0 if len(re_cur.findall(prose)) >= 1 else 0.0
    return rules

def apply_unified_handwriting_shield(
    P_mix: np.ndarray,
    test_texts: List[str],
    log_p_handwriting: Optional[np.ndarray] = None,
    rule_flags: Optional[np.ndarray] = None,
    curly_quote_multiplier: float = 0.36787944117144233,  # exp(-1.0)
    multi_em_dash_log_penalty: float = 0.0,
    straight_contr_log_penalty: float = 0.0,
    curly_contr_log_penalty: float = 0.0,
    continuous_hw_weight: float = 0.0,
) -> np.ndarray:
    """
    Single unified point of application for ALL handwriting and orthographic rules:
    1. Applies vectorized discrete log-penalties: B_disc @ W_pen
    2. Adds continuous 26-feature handwriting log-likelihoods: continuous_hw_weight * log_p_handwriting
    3. Re-normalizes rows to the probability simplex before Sinkhorn balancing.
    """
    if rule_flags is None:
        if "extract_handwriting_and_orthographic_features" in globals():
            _, rule_flags = extract_handwriting_and_orthographic_features(test_texts)
        else:
            rule_flags = _extract_rules_fallback(test_texts)

    curly_log_pen = float(np.log(max(curly_quote_multiplier, 1e-12)))
    W_pen = build_orthographic_penalty_matrix(
        curly_quote_log_penalty=curly_log_pen,
        multi_em_dash_log_penalty=multi_em_dash_log_penalty,
        straight_contr_log_penalty=straight_contr_log_penalty,
        curly_contr_log_penalty=curly_contr_log_penalty,
    )

    log_P = np.log(np.clip(P_mix.astype(np.float64), 1e-12, 1.0))
    # 1. Discrete Bayesian rule subtraction (vectorized matrix multiply)
    log_P += rule_flags.astype(np.float64) @ W_pen

    # 2. Continuous 26-feature handwriting log-likelihood contribution
    if log_p_handwriting is not None and continuous_hw_weight > 0.0:
        log_P += float(continuous_hw_weight) * log_p_handwriting.astype(np.float64)

    P_out = softmax(log_P, axis=1)
    return P_out

def apply_curly_quote_shield(
    P_mix: np.ndarray,
    test_texts: List[str],
    multiplier: float = 0.36787944117144233
) -> np.ndarray:
    """Backward-compatible wrapper invoking the unified handwriting shield."""
    return apply_unified_handwriting_shield(
        P_mix=P_mix,
        test_texts=test_texts,
        log_p_handwriting=None,
        rule_flags=None,
        curly_quote_multiplier=multiplier,
        multi_em_dash_log_penalty=0.0,
        straight_contr_log_penalty=0.0,
        curly_contr_log_penalty=0.0,
        continuous_hw_weight=0.0,
    )

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
