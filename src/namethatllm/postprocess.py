"""
Postprocessing, Fusion & Calibration Engine
===========================================
This module handles all post-ensemble refinement stages:
1. Two-tier convex mixture (85% Main Stack + 15% Complementary Stack)
2. Mass-preserving Specialist Neural Gates (Claude head, Gemini LoRA, GLM LoRA, GPT LoRA)
3. Confidence-aware Within-Family Spectral Gates (dynamic margin tie-breakers)
4. Unified Handwriting & Orthographic Shield (discrete Bayesian tokenization rules + continuous density)
5. Cavity Mean-Field Cluster Uniqueness Repulsion (soft at-most-once constraint on prompt clusters)
6. Multiplicative Sinkhorn-Knopp Column Marginal Balancing (global class prior equalization)
7. Hierarchical Beta-Bernoulli TypoBayes Postprocessor
8. Final Kaggle submission generation (submission.parquet and submission.csv)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FAMILIES, LAB2I, LABELS, SPECIALIST_GROUPS, K, softmax
from .lexical_engine import extract_handwriting_and_orthographic_features

# =============================================================================
# 1. WITHIN-FAMILY CONFIDENCE GATES (SPECTRAL / TIE-BREAKER HEURISTICS)
# =============================================================================


def apply_all_family_confidence_gates(
    P_mix: np.ndarray,
    family_gates: dict[str, np.ndarray] | None,
    max_strength: float = 0.20,
    margin_thr: float = 0.45,
) -> np.ndarray:
    """
    Confidence-aware within-family and cross-family specialist reranker.

    Concept:
    --------
    When an ensemble predicts probabilities across 18 classes, sister models within
    the same family (e.g., Gemini 3.7 vs 3.8, or GLM-5.3 vs GLM-5.3-flash) can have
    extremely close probabilities (low margin). This function uses auxiliary gate
    logits (from spectral/TF-IDF classifiers) to break ties in low-margin cases.

    Key Invariant - Mass Conservation:
    ----------------------------------
    The total probability assigned to the family is strictly preserved!
    Only the internal relative distribution among the sisters is adjusted.
    Probabilities for classes outside the family are completely untouched.

    Dynamic Confidence Gating:
    --------------------------
    - Margin = (Top-1 prob within family) - (Top-2 prob within family).
    - If margin >= margin_thr: Model is already confident; gate strength = 0 (no intervention).
    - If margin < margin_thr: Low confidence; gate strength scales up to max_strength.

    Parameters:
    -----------
    P_mix : np.ndarray of shape (N, 18)
        Current blended prediction probabilities (summing to 1.0 per row).
    family_gates : dict mapping group_name -> log_probabilities of shape (N, K_group)
        Pre-computed log-probabilities from the auxiliary family classifiers.
    max_strength : float
        Maximum interpolation weight for the gate (0.0 to 1.0). If <= 0, gates are bypassed.
    margin_thr : float
        Confidence margin threshold above which the gate does not intervene.

    Returns:
    --------
    P_out : np.ndarray of shape (N, 18)
        Calibrated probabilities with sister-class confusions resolved.
    """
    # Early exit: If gates are disabled or weight is zero, return immediately
    if not family_gates or float(max_strength) <= 0.0:
        return P_mix

    P_out = P_mix.astype(np.float64).copy()

    for fam_name, gate_lp in family_gates.items():
        # Identify the column indices belonging to this family/specialist group
        cols = SPECIALIST_GROUPS.get(fam_name, FAMILIES.get(fam_name))
        if cols is None:
            continue
        if gate_lp is None or gate_lp.shape != (len(P_out), len(cols)):
            continue

        # Determine which samples are eligible for this gate:
        # Cross-family triads only apply when top-1 prediction is in the triad.
        # Single-family gates apply across all rows where that family has mass.
        if fam_name in ("frontier_and_fable", "kimi_gpt55_qwen", "glm_kimi_triad"):
            row_mask = np.isin(P_out.argmax(axis=1), cols)
            if not row_mask.any():
                continue
        elif fam_name == "gemini_37_38_pair":
            row_mask = np.isin(P_out.argmax(axis=1), FAMILIES.get("gemini", [5, 6, 7, 8]))
            if not row_mask.any():
                continue
        else:
            row_mask = np.ones(len(P_out), dtype=bool)

        # Hard-sister groups (e.g. Gemini 3.7 vs 3.8, GLM) use wider margin threshold
        eff_margin_thr = 0.65 if fam_name in ("gemini_37_38_pair", "glm", "glm_kimi_triad") else margin_thr
        eff_max_strength = (
            max(float(max_strength), 0.35) if fam_name in ("gemini_37_38_pair", "glm", "glm_kimi_triad") else float(max_strength)
        )

        # Step 1: Extract sub-probabilities and preserve total marginal mass
        sub_p = np.clip(P_out[np.ix_(row_mask, cols)], 1e-12, 1.0)
        tot_mass = sub_p.sum(axis=1, keepdims=True)
        base_rel_p = sub_p / tot_mass  # Relative distribution within the family (sums to 1.0)

        # Step 2: Compute margin between top-1 and top-2 candidates within family
        sp = np.sort(base_rel_p, axis=1)
        margin = sp[:, -1] - sp[:, -2]

        # Step 3: Compute dynamic intervention strength (higher when margin is small)
        strength = np.clip((eff_margin_thr - margin) / max(eff_margin_thr, 1e-6), 0.0, 1.0)[:, None] * eff_max_strength

        # Step 4: Blend base probabilities with gate probabilities in log-space
        base_lp = np.log(np.clip(base_rel_p, 1e-12, 1.0))
        gate_p = softmax(gate_lp[row_mask], axis=1)
        gate_lp_norm = np.log(np.clip(gate_p, 1e-12, 1.0))
        mixed_rel_p = softmax((1.0 - strength) * base_lp + strength * gate_lp_norm, axis=1)

        # Step 5: Restore original total family mass, leaving external classes unaffected
        P_out[np.ix_(row_mask, cols)] = tot_mass * mixed_rel_p

    # Final safety re-normalization to guarantee row sums == 1.0
    return P_out / P_out.sum(axis=1, keepdims=True).clip(min=1e-12)


# =============================================================================
# 2. PROMPT-CLUSTER UNIQUENESS REPULSION (CAVITY MEAN-FIELD)
# =============================================================================


def apply_cluster_uniqueness_repulsion(
    P_mix: np.ndarray,
    groups: np.ndarray | None,
    strength: float = 0.35,
    iters: int = 3,
) -> np.ndarray:
    """
    Enforces soft at-most-once author uniqueness within test prompt clusters.

    Domain Insight:
    ---------------
    In LLM benchmarking, multiple completions are generated by presenting the
    EXACT SAME PROMPT to different LLM models. Hence, when two or more test passages
    share an identical prompt (forming a 'prompt cluster'), competition rules imply
    that each completion came from a DIFFERENT model (at most 1 completion per model).

    Mathematical Formulation (Cavity Mean-Field):
    ---------------------------------------------
    For each text i in cluster g, the 'cavity field' for class k is the probability
    that no other text j in cluster g (j != i) was generated by class k:
        log P(other != k) = sum_{j in g, j != i} log(1 - q_{j, k})

    We update the unnormalized log-posterior:
        L_{i, k} = log(P0_{i, k}) + strength * sum_{j != i} log(1 - q_{j, k})
    where P0 is the original prediction prior, and q is the current cluster belief.
    Softmax is applied over k, iterating 3 times to reach a self-consistent equilibrium.

    Parameters:
    -----------
    P_mix : np.ndarray of shape (N, 18)
        Input prediction probabilities.
    groups : np.ndarray of shape (N,)
        Cluster IDs assigned to each test sample (e.g. from TF-IDF prompt clustering).
    strength : float
        Repulsion strength factor (typically 0.35).
    iters : int
        Number of mean-field coordinate ascent iterations (typically 3).

    Returns:
    --------
    q : np.ndarray of shape (N, 18)
        Repulsed probabilities penalizing multiple identical predictions in one prompt cluster.
    """
    N = len(P_mix)
    # Require at least 18 rows and valid cluster assignments
    if N < 18 or groups is None or len(groups) != N or strength <= 0.0:
        return P_mix

    # Map cluster identifiers to dense integer labels 1, 2, ...
    _, inv_g = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    safe_g = inv_g + 1
    sz = np.bincount(safe_g)[safe_g]

    # Only operate on multi-sample clusters between 2 and 18 texts (size <= 18 since there are 18 classes)
    valid_g = np.unique(safe_g[(sz >= 2) & (sz <= 18)])
    if len(valid_g) == 0:
        return P_mix

    log_P0 = np.log(np.clip(P_mix.astype(np.float64), 1e-12, 1.0))
    q = P_mix.astype(np.float64).copy()

    # Mean-field relaxation loop
    for _ in range(iters):
        L = log_P0.copy()
        for g in valid_g:
            m = safe_g == g
            q_g = q[m]  # Shape: (cluster_size, 18)

            # lo[j, k] = log(1 - q_{j, k}): log-probability that sample j did NOT pick class k
            lo = np.log(np.clip(1.0 - q_g, 1e-4, 1.0))

            # Cavity sum: sum of lo over all samples in cluster minus sample i's own term
            # Vectorized identity: sum_{j} lo[j] - lo[i] equals sum_{j != i} lo[j]
            cavity_penalty = lo.sum(axis=0, keepdims=True) - lo
            L[m] += float(strength) * cavity_penalty

        # Re-normalize across the 18 classes for next iteration
        q = softmax(L, axis=1)

    return q


# =============================================================================
# 3. TWO-TIER CONVEX ENSEMBLE BLENDING
# =============================================================================


def blend_main_and_complementary(
    LOGIT_MAIN: np.ndarray, COMP_LIST: list[tuple[float, np.ndarray]], main_ratio: float = 0.85, comp_ratio: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blends the Main Stack and Complementary Stack in probability space.

    Architecture:
    -------------
    - Main Stack (85% weight): High-capacity deep readers (Qwen3-14B Part 2,
      in-house champion LoRA, etc.) providing top semantic representations.
    - Complementary Stack (15% weight): Fast, diverse, non-neural or shallow readers
      (Style SVC 4-view, Lexical 5-view, DeBERTa, 8B patches) providing orthographic
      and stylistic diversity that regularizes neural hallucinations.

    Formula:
    --------
        R_main = softmax(LOGIT_MAIN)
        P_comp = sum(w_i * P_i) / sum(w_i)
        P_mix  = main_ratio * R_main + comp_ratio * P_comp

    Parameters:
    -----------
    LOGIT_MAIN : np.ndarray of shape (N, 18)
        Raw logit sum from the primary deep models.
    COMP_LIST : list of tuples (weight, probability_array)
        List of complementary models and their assigned ensemble weights.
    main_ratio : float
        Convex weight for Main Stack (default 0.85).
    comp_ratio : float
        Convex weight for Complementary Stack (default 0.15).

    Returns:
    --------
    P_mix : np.ndarray of shape (N, 18)
        Blended ensemble probabilities.
    R_main : np.ndarray of shape (N, 18)
        Main stack probabilities alone (used for routing/gating decisions).
    """
    R_main = softmax(LOGIT_MAIN, axis=1)
    if len(COMP_LIST) > 0:
        comp_weight_total = sum(w for w, _ in COMP_LIST)
        P_comp = sum(w * P_c for w, P_c in COMP_LIST) / max(1e-8, comp_weight_total)
        P_mix = main_ratio * R_main + comp_ratio * P_comp
    else:
        P_mix = R_main
    return P_mix, R_main


# =============================================================================
# 4. NEURAL SPECIALIST GATES (CLAUDE HEAD & FAMILY LORAS)
# =============================================================================


def apply_claude_specialist_gate(
    P_mix: np.ndarray,
    R_main: np.ndarray,
    claude_spec_logits: np.ndarray | None,
    claude_classes: list | None = None,
    labels: list = LABELS,
    claude_cols: list = FAMILIES["claude"],
    gate_weight: float = 0.15,
    weight: float | None = None,
) -> np.ndarray:
    """
    Refines passages predicted in the Claude family using a specialist head
    while strictly conserving total Claude family probability mass.
    """
    if claude_spec_logits is None:
        return P_mix
    if weight is not None:
        gate_weight = weight
    if float(gate_weight) <= 0.0:
        return P_mix

    P_out = P_mix.copy()
    P_claude_head = softmax(claude_spec_logits, axis=-1)

    if claude_classes is not None:
        src_c = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in claude_classes]
        if all(labels[c] in src_c for c in claude_cols):
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


def apply_family_specialist_lora_gate(
    P_mix: np.ndarray,
    R_main: np.ndarray,
    spec_logits: np.ndarray | None,
    spec_classes: list | None = None,
    labels: list = LABELS,
    family_cols: list = FAMILIES["gemini"],
    gate_weight: float = 0.65,
) -> np.ndarray:
    """
    Refines intra-family predictions using fine-tuned Qwen3-14B Family-Specialist LoRAs
    while strictly preserving 100% of total marginal family probability mass.
    """
    if spec_logits is None or len(spec_logits) != len(P_mix) or float(gate_weight) <= 0.0:
        return P_mix

    P_out = P_mix.astype(np.float64).copy()
    spec_arr = np.asarray(spec_logits, dtype=np.float64)
    if spec_arr.ndim != 2 or spec_arr.shape[1] != len(family_cols):
        return P_mix

    P_spec = softmax(spec_arr, axis=-1)

    if spec_classes is not None and len(spec_classes) == len(family_cols):
        src_c = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in spec_classes]
        if all(labels[c] in src_c for c in family_cols):
            c_idx = [src_c.index(labels[c]) for c in family_cols]
            P_spec = P_spec[:, c_idx]

    fam_mass = P_out[:, family_cols].sum(axis=1)
    has_signal = np.ptp(spec_arr, axis=1) > 1e-6
    fam_mask = (np.isin(R_main.argmax(axis=1), family_cols) | (fam_mass >= 0.35)) & has_signal

    if fam_mask.any():
        rows = np.flatnonzero(fam_mask)
        sub_mix = np.clip(P_out[np.ix_(rows, family_cols)], 1e-12, 1.0)
        tot_mass = sub_mix.sum(axis=1, keepdims=True)
        base_rel = sub_mix / tot_mass

        log_rel = (1.0 - float(gate_weight)) * np.log(base_rel) + float(gate_weight) * np.log(np.clip(P_spec[rows], 1e-12, 1.0))
        blended_rel = softmax(log_rel, axis=1)
        P_out[np.ix_(rows, family_cols)] = tot_mass * blended_rel

    return P_out


# =============================================================================
# 5. ORTHOGRAPHIC RULES & HANDWRITING SHIELD
# =============================================================================


def build_orthographic_penalty_matrix(
    curly_quote_log_penalty: float = -1.00,  # Rule 1: Claude & GLM never emit curly quotes
    multi_em_dash_log_penalty: float = 0.0,  # Rule 2: Multiple prose em-dashes (rare in Gemini)
    straight_contr_log_penalty: float = 0.0,  # Rule 3: Straight contractions
    curly_contr_log_penalty: float = 0.0,  # Rule 4: Curly contractions
) -> np.ndarray:
    """
    Builds the (4, 18) log-penalty matrix W_pen mapping discrete orthographic
    rule indicators to the 18 canonical model classes.

    Empirical Foundations:
    ----------------------
    - Rule 1 (Curly Quotes): In the competition training corpus, Claude (0 / 1049)
      and GLM (0 / 300) NEVER produce typographic curly quotes (‘, ’, “, ”) in standard prose.
      They consistently output straight quotes (', "). If a passage contains curly quotes,
      the probability of it being Claude or GLM is astronomically low.
      Log penalty of -1.0 reduces their odds by a factor of e^(-1) ≈ 0.368.
    - Rule 2 (Multiple Em-Dashes): Gemini models rarely use two or more prose em-dashes (—).
    - Rule 3 (Straight Contractions): Distinctive signature for gpt-5.6-sol vs sisters.
    - Rule 4 (Curly Contractions): Reinforces rejection of Claude & GLM.
    """
    W_pen = np.zeros((4, K), dtype=np.float64)
    claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
    gemini_idx = FAMILIES["gemini"]
    sol_idx = LAB2I["gpt-5.6-sol"]

    # Rule 1: Any curly quote -> penalize Claude (5 models) and GLM (2 models)
    W_pen[0, claude_and_glm_idx] = float(curly_quote_log_penalty)
    # Rule 2: >= 2 prose em-dashes -> penalize Gemini (4 models)
    W_pen[1, gemini_idx] = float(multi_em_dash_log_penalty)
    # Rule 3: >= 2 prose straight contractions (with 0 curly) -> penalize gpt-5.6-sol
    W_pen[2, sol_idx] = float(straight_contr_log_penalty)
    # Rule 4: >= 1 prose curly contraction -> reinforce penalty on Claude & GLM
    W_pen[3, claude_and_glm_idx] = float(curly_contr_log_penalty)

    return W_pen


def _extract_rules_fallback(texts: list[str]) -> np.ndarray:
    """
    Fallback discrete rules extractor using regex when lexical_engine is unavailable.
    Detects curly quotes, em-dashes, and straight/curly prose contractions.
    """
    re_code = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
    re_str = re.compile(r"[A-Za-z]'[A-Za-z]")
    re_cur = re.compile(r"[A-Za-z]\u2019[A-Za-z]")
    rules = np.zeros((len(texts), 4), dtype=np.float32)

    for i, raw_t in enumerate(texts):
        t = str(raw_t)
        prose = re_code.sub(" ", t)  # Strip inline and block code before checking prose
        raw_curly = t.count("\u2018") + t.count("\u2019") + t.count("\u201c") + t.count("\u201d")
        rules[i, 0] = 1.0 if raw_curly > 0 else 0.0
        rules[i, 1] = 1.0 if prose.count("\u2014") >= 2 else 0.0
        rules[i, 2] = 1.0 if (len(re_str.findall(prose)) >= 2 and (t.count("\u2018") + t.count("\u2019")) == 0) else 0.0
        rules[i, 3] = 1.0 if len(re_cur.findall(prose)) >= 1 else 0.0

    return rules


def apply_unified_handwriting_shield(
    P_mix: np.ndarray,
    test_texts: list[str],
    log_p_handwriting: np.ndarray | None = None,
    rule_flags: np.ndarray | None = None,
    curly_quote_multiplier: float = 0.36787944117144233,  # exp(-1.0)
    multi_em_dash_log_penalty: float = 0.0,
    straight_contr_log_penalty: float = 0.0,
    curly_contr_log_penalty: float = 0.0,
    continuous_hw_weight: float = 0.0,
) -> np.ndarray:
    """
    Single unified point of application for ALL handwriting and orthographic rules:
    1. Computes vectorized discrete log-penalties via matrix product: rule_flags @ W_pen
    2. Adds continuous 26-feature handwriting Gaussian log-likelihoods
    3. Re-normalizes rows to the probability simplex via softmax.
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

    return softmax(log_P, axis=1)


def apply_curly_quote_shield(P_mix: np.ndarray, test_texts: list[str], multiplier: float = 0.36787944117144233) -> np.ndarray:
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


# =============================================================================
# 6. SINKHORN-KNOPP COLUMN MASS BALANCING
# =============================================================================


def sinkhorn_balancing(
    P: np.ndarray,
    iters: int = 100,
    sinkhorn_strength: float = 0.65,
    target_mix: float = 0.35,
    train_prior: np.ndarray | None = None,
    allow_pure: bool = False,
) -> np.ndarray:
    """
    Soft Multiplicative Sinkhorn-Knopp Column Marginal Balancing.
    =============================================================
    Equalizes global class marginals using Soft Sinkhorn shrinkage.
    Pure Sinkhorn (1.0 / 1.0) is disallowed by default as it destroys test performance
    by hard-forcing uniform marginals on batches with natural class variance.

    Parameters:
    -----------
    P : np.ndarray of shape (N, 18)
        Probability matrix to balance.
    iters : int
        Number of alternating row/column scaling steps (default 100).
    sinkhorn_strength : float
        Soft shrinkage interpolation between original P and balanced Q (default 0.65).
        Capped at 0.75 unless allow_pure=True.
    target_mix : float
        Convex blend between predicted marginals and prior (default 0.35).
        Capped at 0.50 unless allow_pure=True.
    train_prior : np.ndarray of shape (18,), optional
        Empirical training class frequencies (used when target_mix < 1.0).
    allow_pure : bool, optional
        Whether to allow 100% hard pure Sinkhorn (default False).

    Returns:
    --------
    Balanced probability matrix Q of shape (N, 18).
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

    eff_mix = target_mix if allow_pure else min(target_mix, 0.50)
    if eff_mix < 1.0:
        pred_marg = np.clip(P_work.mean(axis=0), 1e-6, None)
        pred_marg /= pred_marg.sum()
        rho = float(np.clip(eff_mix, 0.0, 1.0))
        target_col = (1.0 - rho) * pred_marg + rho * p_train
        target_col = np.clip(target_col, 1e-6, None)
        target_col /= target_col.sum()
    else:
        target_col = p_train

    # Alternating row and column projections
    Q = P_work.copy()
    for _ in range(iters):
        # Row normalization: sum across classes = 1.0
        Q /= Q.sum(axis=1, keepdims=True).clip(min=1e-12)
        # Column scaling: sum across samples for class k = N * target_col[k]
        Q *= (len(Q) * target_col[None, :]) / Q.sum(axis=0, keepdims=True).clip(min=1e-12)

    # Final row normalization
    Q /= Q.sum(axis=1, keepdims=True).clip(min=1e-12)

    if allow_pure and sinkhorn_strength >= 1.0:
        return Q

    # Soft logarithmic mixture between original P and balanced Q (pure Sinkhorn disallowed)
    eff_strength = float(np.clip(sinkhorn_strength, 0.0, 0.75 if not allow_pure else 1.0))
    if eff_strength <= 0.0:
        return P_work

    log_P = (1.0 - eff_strength) * np.log(np.clip(P_work, 1e-12, None)) + eff_strength * np.log(np.clip(Q, 1e-12, None))
    return softmax(log_P, axis=1)


# =============================================================================
# 7. TYPOBAYES POSTPROCESSOR (HIERARCHICAL BETA-BERNOULLI SHRINKAGE)
# =============================================================================


def apply_typobayes_postprocess(
    P_or_logP: np.ndarray,
    train_texts: list[str],
    y_train: np.ndarray,
    test_texts: list[str],
    labels: list = LABELS,
    kappa: float = 24.0,
    evidence_scale: float = 1.2,
    evidence_cap: float = 0.30,
) -> np.ndarray:
    """
    Applies Hierarchical Beta-Bernoulli 37-Marker TypoBayes postprocessing.

    Concept:
    --------
    Extracts 37 orthographic, punctuation, and structural markdown markers
    (e.g., ' — ', '`', '###', 'However,', '\n- ') across training texts and models.
    Uses Beta-Bernoulli empirical Bayes with prior strength kappa to calculate
    smoothed class-conditional emission rates, deriving log-likelihood ratio evidence.
    Applies conservative confidence gating so high-confidence predictions are not overridden.
    """
    if len(train_texts) == 0 or len(test_texts) == 0 or len(y_train) == 0:
        return P_or_logP

    y_tr = np.array([LAB2I[y] if isinstance(y, str) else int(y) for y in y_train], dtype=int)
    marks = [
        " \u2014 ",
        "\u2014",
        "\u2013",
        "\u2019",
        "\u2018",
        "\u201c",
        "\u201d",
        "\u00ab",
        "\u00bb",
        '"',
        ";",
        ":",
        "(",
        ")",
        "[",
        "]",
        "**",
        "`",
        "\u2026",
        "\u2022",
        "\n",
        "\n\n",
        "e.g.",
        "i.e.",
        "However,",
        "Overall,",
        "In conclusion",
        "Therefore,",
        "Moreover,",
        "\t",
        " - ",
        " = ",
        "\n- ",
        "\n1.",
        "###",
        "```",
        "> ",
    ]
    K_cls = len(labels)

    def _extract_marker_row(t: str) -> list[float]:
        # Mutually exclusive decoupling for overlapping substrings:
        # 1. ' — ' vs unspaced '—'
        has_spaced_em = " \u2014 " in t
        t_no_spaced = t.replace(" \u2014 ", "")
        has_unspaced_em = "\u2014" in t_no_spaced

        # 2. '\n\n', '\n- ', '\n1.' vs standalone '\n'
        has_double_nl = "\n\n" in t
        has_bullet = "\n- " in t
        has_num_list = "\n1." in t
        t_no_compound_nl = t.replace("\n\n", "").replace("\n- ", "").replace("\n1.", "")
        has_single_nl = "\n" in t_no_compound_nl

        row = []
        for m in marks:
            if m == " \u2014 ":
                row.append(float(has_spaced_em))
            elif m == "\u2014":
                row.append(float(has_unspaced_em))
            elif m == "\n\n":
                row.append(float(has_double_nl))
            elif m == "\n- ":
                row.append(float(has_bullet))
            elif m == "\n1.":
                row.append(float(has_num_list))
            elif m == "\n":
                row.append(float(has_single_nl))
            else:
                row.append(float(m in t))
        return row

    Xtr = np.array([_extract_marker_row(t) for t in train_texts], dtype=np.float32)
    Xte = np.array([_extract_marker_row(t) for t in test_texts], dtype=np.float32)
    freq = np.bincount(y_tr, minlength=K_cls).astype(np.float64)
    hits = np.stack([Xtr[y_tr == k].sum(axis=0) for k in range(K_cls)], axis=0).astype(np.float64)

    # Hierarchical Beta-Bernoulli smoothing: class rates shrink toward global rate
    global_rate = (Xtr.sum(axis=0) + 2.0) / (len(Xtr) + 4.0)
    rate = (hits + kappa * global_rate[None, :]) / (freq[:, None] + kappa)
    rate = np.clip(rate, 0.01, 0.99)
    global_rate = np.clip(global_rate, 0.01, 0.99)

    log_rate = np.log(rate)
    log_not_rate = np.log1p(-rate)
    base_log_rate = np.log(global_rate)
    base_log_not_rate = np.log1p(-global_rate)

    # Weight markers by discriminative power across classes
    signal = np.sqrt(np.mean((rate - global_rate[None, :]) ** 2, axis=0))
    med_signal = float(np.median(signal[signal > 0])) if np.any(signal > 0) else 1.0
    fw = signal / max(med_signal, 1e-6)
    fw = np.clip(fw, 0.25, 2.0)

    # Matrix form: weighted Bernoulli log-likelihood ratio for each class
    evidence = (Xte * fw[None, :]) @ log_rate.T + ((1.0 - Xte) * fw[None, :]) @ log_not_rate.T
    evidence -= (((Xte * fw[None, :]) @ base_log_rate) + (((1.0 - Xte) * fw[None, :]) @ base_log_not_rate))[:, None]
    evidence -= evidence.mean(axis=1, keepdims=True)
    evidence = np.clip(evidence_scale * evidence / max(float(fw.sum()), 1e-6), -evidence_cap, evidence_cap)

    # Convert inputs to probabilities and log-probabilities
    if np.all(P_or_logP <= 0.0) and not np.all(P_or_logP == 0.0):
        raw_logP = P_or_logP.copy()
        probs = softmax(raw_logP, axis=1)
    else:
        probs = P_or_logP.copy()
        raw_logP = np.log(np.clip(probs, 1e-12, 1.0))

    # Gating: only apply evidence when model is uncertain, text is long, and markers are present
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    confidence_gate = np.clip((0.60 - margin) / 0.60, 0.0, 1.0)
    word_count = np.array([len(re.findall(r"\b\w+\b", str(t))) for t in test_texts], dtype=np.float32)
    length_gate = np.clip(np.sqrt(np.maximum(word_count, 1.0) / 80.0), 0.40, 1.0)
    active_markers = Xte.sum(axis=1)
    coverage_gate = np.clip(np.sqrt((active_markers + 1.0) / 4.0), 0.45, 1.0)
    gate = confidence_gate * length_gate * coverage_gate

    safe_logits = raw_logP + gate[:, None] * evidence
    return softmax(safe_logits, axis=1)


# =============================================================================
# 8. FINAL SUBMISSION EXPORT
# =============================================================================


def generate_submission(P_final: np.ndarray, test_ids: pd.Series, working_dir: Path) -> tuple[pd.DataFrame, Path, Path]:
    """
    Generates submission.parquet and submission.csv with exact Kaggle competition schema:
    - id: test sample identifier string
    - model: predicted model class name string (argmax over calibrated probabilities)
    """
    preds = [LABELS[idx] for idx in P_final.argmax(axis=1)]
    df_sub = pd.DataFrame({"id": test_ids, "model": preds})
    sub_parquet_path = working_dir / "submission.parquet"
    sub_csv_path = working_dir / "submission.csv"

    df_sub.to_parquet(sub_parquet_path, index=False)
    df_sub.to_csv(sub_csv_path, index=False)
    return df_sub, sub_parquet_path, sub_csv_path
