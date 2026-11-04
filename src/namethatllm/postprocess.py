"""
Postprocessing, Fusion & Calibration Engine
===========================================
Pure two-tier convex mixture (85% Main + 15% Complementary),
mass-preserving Claude Specialist Gate, gentle curly quote shield,
and official 300-step exact multiplicative Sinkhorn mass balancing.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

from .config import LABELS, K, LAB2I, FAMILIES, SPECIALIST_GROUPS, softmax, log_softmax
from .lexical_engine import extract_handwriting_and_orthographic_features

def apply_all_family_confidence_gates(
    P_mix: np.ndarray,
    family_gates: Optional[Dict[str, np.ndarray]],
    max_strength: float = 0.20,
    margin_thr: float = 0.45,
) -> np.ndarray:
    """
    Confidence-aware within-family and cross-family specialist reranker across
    all groups in SPECIALIST_GROUPS (4 families + 3 tie-breaker groups).
    Preserves exact probability mass while resolving low-margin confusions.
    """
    if not family_gates:
        return P_mix
    P_out = P_mix.astype(np.float64).copy()
    for fam_name, gate_lp in family_gates.items():
        cols = SPECIALIST_GROUPS.get(fam_name, FAMILIES.get(fam_name))
        if cols is None:
            continue
        if gate_lp is None or gate_lp.shape != (len(P_out), len(cols)):
            continue

        if fam_name == "frontier_and_fable":
            row_mask = np.isin(P_out.argmax(axis=1), cols)
            if not row_mask.any():
                continue
        elif fam_name == "gemini_37_38_pair":
            row_mask = np.isin(P_out.argmax(axis=1), FAMILIES.get("gemini", [5, 6, 7, 8]))
            if not row_mask.any():
                continue
        else:
            row_mask = np.ones(len(P_out), dtype=bool)

        eff_margin_thr = 0.65 if fam_name == "gemini_37_38_pair" else margin_thr
        eff_max_strength = max(float(max_strength), 0.32) if fam_name == "gemini_37_38_pair" else float(max_strength)

        sub_p = np.clip(P_out[np.ix_(row_mask, cols)], 1e-12, 1.0)
        tot_mass = sub_p.sum(axis=1, keepdims=True)
        base_rel_p = sub_p / tot_mass
        sp = np.sort(base_rel_p, axis=1)
        margin = sp[:, -1] - sp[:, -2]
        strength = np.clip((eff_margin_thr - margin) / max(eff_margin_thr, 1e-6), 0.0, 1.0)[:, None] * eff_max_strength
        base_lp = np.log(np.clip(base_rel_p, 1e-12, 1.0))
        gate_p = softmax(gate_lp[row_mask], axis=1)
        gate_lp_norm = np.log(np.clip(gate_p, 1e-12, 1.0))
        mixed_rel_p = softmax((1.0 - strength) * base_lp + strength * gate_lp_norm, axis=1)
        P_out[np.ix_(row_mask, cols)] = tot_mass * mixed_rel_p
    return P_out / P_out.sum(axis=1, keepdims=True).clip(min=1e-12)

def apply_cluster_uniqueness_repulsion(
    P_mix: np.ndarray,
    groups: Optional[np.ndarray],
    strength: float = 0.35,
    iters: int = 3,
) -> np.ndarray:
    """
    Enforces soft at-most-once author uniqueness within each detected test prompt
    cluster (2 <= size <= 18) via cavity mean-field repulsion before global Sinkhorn.
    """
    N = len(P_mix)
    if N < 18 or groups is None or len(groups) != N or strength <= 0.0:
        return P_mix
    _, inv_g = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    safe_g = inv_g + 1
    sz = np.bincount(safe_g)[safe_g]
    valid_g = np.unique(safe_g[(sz >= 2) & (sz <= 18)])
    if len(valid_g) == 0:
        return P_mix

    log_P0 = np.log(np.clip(P_mix.astype(np.float64), 1e-12, 1.0))
    q = P_mix.astype(np.float64).copy()
    for _ in range(iters):
        L = log_P0.copy()
        for g in valid_g:
            m = (safe_g == g)
            q_g = q[m]
            lo = np.log(np.clip(1.0 - q_g, 1e-4, 1.0))
            L[m] += float(strength) * (lo.sum(axis=0, keepdims=True) - lo)
        q = softmax(L, axis=1)
    return q

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

def sinkhorn_balancing(
    P: np.ndarray,
    iters: int = 300,
    sinkhorn_strength: float = 1.0,
    target_mix: float = 1.0,
    train_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sinkhorn column mass balancing.
    Supports both:
    1. Exact equal column mass balancing (sinkhorn_strength=1.0, target_mix=1.0, train_prior=None)
    2. Prior-shrunk soft balancing (sinkhorn_strength < 1.0, e.g. 0.65, target_mix=0.35, train_prior=...)
    """
    if len(P) < 18 or sinkhorn_strength <= 0.0:
        return P

    P_work = np.clip(P.astype(np.float64), 1e-12, 1.0)
    K_cols = P_work.shape[1]

    # Target column marginal
    if train_prior is not None:
        p_train = np.clip(train_prior.astype(np.float64), 1e-6, None)
        p_train /= p_train.sum()
    else:
        p_train = np.full(K_cols, 1.0 / K_cols, dtype=np.float64)

    if target_mix < 1.0:
        pred_marg = np.clip(P_work.mean(axis=0), 1e-6, None)
        pred_marg /= pred_marg.sum()
        rho = float(np.clip(target_mix, 0.0, 1.0))
        target_col = (1.0 - rho) * pred_marg + rho * p_train
        target_col = np.clip(target_col, 1e-6, None)
        target_col /= target_col.sum()
    else:
        target_col = p_train

    Q = P_work.copy()
    for _ in range(iters):
        Q /= Q.sum(axis=1, keepdims=True).clip(min=1e-12)
        Q *= (len(Q) * target_col[None, :]) / Q.sum(axis=0, keepdims=True).clip(min=1e-12)
    Q /= Q.sum(axis=1, keepdims=True).clip(min=1e-12)

    if sinkhorn_strength >= 1.0:
        return Q

    # Soft logarithmic mixture between original P and balanced Q
    log_P = (1.0 - sinkhorn_strength) * np.log(np.clip(P_work, 1e-12, None)) + sinkhorn_strength * np.log(np.clip(Q, 1e-12, None))
    return softmax(log_P, axis=1)

def apply_typobayes_postprocess(
    P_or_logP: np.ndarray,
    train_texts: List[str],
    y_train: np.ndarray,
    test_texts: List[str],
    labels: list = LABELS,
    kappa: float = 24.0,
    evidence_scale: float = 1.2,
    evidence_cap: float = 0.30,
) -> np.ndarray:
    """
    Applies the official Judith 0.664 (v38A) Hierarchical Beta-Bernoulli 37-Marker TypoBayes
    postprocessing to the calibrated log-probabilities/probabilities.
    """
    if len(train_texts) == 0 or len(test_texts) == 0 or len(y_train) == 0:
        return P_or_logP

    y_tr = np.array([LAB2I[y] if isinstance(y, str) else int(y) for y in y_train], dtype=int)
    marks = [
        ' — ', '—', '–', '’', '‘', '“', '”', '«', '»', '"', ';', ':', '(', ')', '[', ']',
        '**', '`', '…', '•', '\n', '\n\n', 'e.g.', 'i.e.', 'However,', 'Overall,',
        'In conclusion', 'Therefore,', 'Moreover,', '\t', ' - ', ' = ', '\n- ', '\n1.', '###', '```', '> '
    ]
    K_cls = len(labels)
    Xtr = np.array([[float(m in t) for m in marks] for t in train_texts], dtype=np.float32)
    Xte = np.array([[float(m in t) for m in marks] for t in test_texts], dtype=np.float32)
    freq = np.bincount(y_tr, minlength=K_cls).astype(np.float64)
    hits = np.stack([Xtr[y_tr == k].sum(axis=0) for k in range(K_cls)], axis=0).astype(np.float64)

    # Hierarchical Beta-Bernoulli smoothing: class-specific marker rates shrink toward the global rate.
    global_rate = (Xtr.sum(axis=0) + 2.0) / (len(Xtr) + 4.0)
    rate = (hits + kappa * global_rate[None, :]) / (freq[:, None] + kappa)
    rate = np.clip(rate, 0.01, 0.99)
    global_rate = np.clip(global_rate, 0.01, 0.99)

    log_rate = np.log(rate)
    log_not_rate = np.log1p(-rate)
    base_log_rate = np.log(global_rate)
    base_log_not_rate = np.log1p(-global_rate)

    # Weight markers by how consistently they separate classes, capped to avoid one marker dominating.
    signal = np.sqrt(np.mean((rate - global_rate[None, :]) ** 2, axis=0))
    med_signal = float(np.median(signal[signal > 0])) if np.any(signal > 0) else 1.0
    fw = signal / max(med_signal, 1e-6)
    fw = np.clip(fw, 0.25, 2.0)

    # Matrix form: weighted Bernoulli log likelihood ratio for each class.
    evidence = (Xte * fw[None, :]) @ log_rate.T + ((1.0 - Xte) * fw[None, :]) @ log_not_rate.T
    evidence -= (((Xte * fw[None, :]) @ base_log_rate) + (((1.0 - Xte) * fw[None, :]) @ base_log_not_rate))[:, None]
    evidence -= evidence.mean(axis=1, keepdims=True)
    evidence = np.clip(evidence_scale * evidence / max(float(fw.sum()), 1e-6), -evidence_cap, evidence_cap)

    # Input can be log-probabilities or probabilities
    if np.all(P_or_logP <= 0.0) and not np.all(P_or_logP == 0.0):
        raw_logP = P_or_logP.copy()
        probs = softmax(raw_logP, axis=1)
    else:
        probs = P_or_logP.copy()
        raw_logP = np.log(np.clip(probs, 1e-12, 1.0))

    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    confidence_gate = np.clip((0.60 - margin) / 0.60, 0.0, 1.0)
    word_count = np.array([len(re.findall(r'\b\w+\b', str(t))) for t in test_texts], dtype=np.float32)
    length_gate = np.clip(np.sqrt(np.maximum(word_count, 1.0) / 80.0), 0.40, 1.0)
    active_markers = Xte.sum(axis=1)
    coverage_gate = np.clip(np.sqrt((active_markers + 1.0) / 4.0), 0.45, 1.0)
    gate = confidence_gate * length_gate * coverage_gate

    safe_logits = raw_logP + gate[:, None] * evidence
    return softmax(safe_logits, axis=1)

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
