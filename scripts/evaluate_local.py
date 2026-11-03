#!/usr/bin/env python3
"""
scripts/evaluate_local.py
=========================
Fail-Fast Local Evaluation & Ablation Harness

Provides sub-second local tuning and evaluation across two benchmarks:
1. Benchmark A (--mode split, Zero-GPU):
   - Splits train.csv strictly by prompt_id using GroupKFold.
   - Evaluates the dual-output Lexical Engine, Handwriting & Orthographic Shield,
     and Sinkhorn mass balancing on held-out prompts (~540 rows, 30 prompts x 18 models).
   - Prints stage-by-stage Balanced Accuracy and the Top-5 Most Confused Sister-Model Pairs.
2. Benchmark B (--mode cache, Zero-GPU):
   - Replays cached neural representations from local_eval_cache.npz (or --mode simulate)
     in < 0.5 seconds!
   - Runs UnseenStepwiseAuditor, Caruana greedy forward stepwise selection,
     Claude gate, Handwriting shield, and Sinkhorn mass balancing.
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.special import softmax, log_softmax
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR / "src"))

from namethatllm.config import LABELS, K, LAB2I, FAMILIES, SPECIALIST_GROUPS
from namethatllm.lexical_engine import run_lexical_engine
from namethatllm.ablation import UnseenStepwiseAuditor, _fast_sinkhorn_bacc

CANONICAL_NAME_MAP: Dict[str, str] = {
    "LinearSVC (4-View)": "svc_4view",
    "Lexical Comp (5-View)": "lexical_5v",
    "Qwen3-14B Part-2": "qwen3_14b_p2",
    "Qwen3-14B In-House": "qwen3_14b_inhouse",
    "DeBERTa-v3-Large": "deberta_p2",
    "DeBERTa-InHouse": "deberta_inhouse",
    "Qwen3-8B Claude Head": "qwen3_8b_claude",
    "Qwen2.5-14B Part-2": "qwen25_14b_p2",
    "Qwen3-8B In-Style": "instyle",
    "Qwen3-8B In-House": "qwen3_8b_inhouse",
    "qwen3_8b_v5": "qwen3_8b_v5",
    "qwen3_14b_v5": "qwen3_14b_v5",
    "wpca_dp": "wpca_dp",
    "spectral_bank": "spectral_bank",
}

from namethatllm.postprocess import (
    apply_claude_specialist_gate,
    apply_all_family_confidence_gates,
    apply_cluster_uniqueness_repulsion,
    apply_unified_handwriting_shield,
    sinkhorn_balancing,
    apply_typobayes_postprocess,
)


def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    default_cfg = WORKSPACE_DIR / "configs" / "production_ensemble.json"
    if default_cfg.exists():
        with open(default_cfg, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Identifies the most frequent misclassification pairs, highlighting sister models."""
    model_to_fam: Dict[int, str] = {}
    for fam, indices in FAMILIES.items():
        for idx in indices:
            model_to_fam[idx] = fam

    pair_counts: Dict[Tuple[int, int], int] = {}
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            p = (min(yt, yp), max(yt, yp))
            pair_counts[p] = pair_counts.get(p, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for (i, j), count in sorted_pairs:
        fam_i = model_to_fam.get(i)
        fam_j = model_to_fam.get(j)
        is_sister = (fam_i == fam_j and fam_i is not None)
        results.append({
            "model_a": labels[i],
            "model_b": labels[j],
            "count": count,
            "family": fam_i if is_sister else "Cross-Family",
            "is_sister": is_sister,
        })
    return results


def evaluate_prompt_split(
    train_path: Path,
    n_splits: int = 5,
    fast: bool = True,
    config_path: Optional[Path] = None,
    working_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Benchmark A: Zero-leakage prompt-grouped holdout evaluation.
    Evaluates lexical engine, shields, and Sinkhorn on held-out prompts.
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at: {train_path}")

    df_tr = pd.read_csv(train_path)
    cfg = load_config(config_path)
    post_cfg = cfg.get("postprocess", cfg.get("postprocessing", {}))
    t0 = time.time()

    if "prompt_id" not in df_tr.columns:
        raise ValueError(f"train.csv must contain 'prompt_id' for grouped evaluation. Found columns: {list(df_tr.columns)}")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df_tr["prompt_id"].values
    y_all = np.array([LAB2I[m] for m in df_tr["model"].values], dtype=int)

    splits = list(gkf.split(df_tr, y_all, groups=groups))
    eval_splits = splits[:1] if fast else splits

    print("\n" + "=" * 90)
    print(f" BENCHMARK A: PROMPT-GROUPED HOLDOUT EVALUATION ({len(eval_splits)} of {n_splits} folds)")
    print(f" Dataset: {len(df_tr)} rows across {df_tr['prompt_id'].nunique()} prompts & {K} author classes")
    print("=" * 90)

    fold_metrics: List[Dict[str, float]] = []
    all_val_true: List[np.ndarray] = []
    all_val_preds: List[np.ndarray] = []

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        base_tmp = Path(tmp_dir) if working_dir is None else Path(working_dir)

        for fold_idx, (tr_i, val_i) in enumerate(eval_splits):
            fold_t0 = time.time()
            df_fold_tr = df_tr.iloc[tr_i]
            df_fold_val = df_tr.iloc[val_i]

            f_dir = base_tmp / f"fold_{fold_idx}"
            f_dir.mkdir(parents=True, exist_ok=True)

            f_tr_csv = f_dir / "train_fold.csv"
            f_val_csv = f_dir / "val_fold.csv"
            df_fold_tr.to_csv(f_tr_csv, index=False)
            df_fold_val.to_csv(f_val_csv, index=False)

            val_texts = df_fold_val["text"].fillna("").astype(str).tolist()
            y_val = y_all[val_i]

            # Execute lexical engine in-process
            lex_out = run_lexical_engine(
                train_path=f_tr_csv,
                test_path=f_val_csv,
                pool_file=None,
                qwen_tok_dir="",
                working_dir=f_dir,
            )

            lp_svc = lex_out["log_p_svc_main"]
            p_lex = lex_out["p_lex_comp"]
            lp_hw = lex_out.get("log_p_handwriting")
            rule_flags = lex_out.get("rule_flags_test")

            # Stage 1: 4-View LinearSVC Raw
            preds_svc = lp_svc.argmax(axis=1)
            bacc_svc = float(balanced_accuracy_score(y_val, preds_svc))

            # Stage 2: 5-View BPE + SGD + Twin + Family Specialists Raw
            preds_lex = p_lex.argmax(axis=1)
            bacc_lex = float(balanced_accuracy_score(y_val, preds_lex))

            # Stage 3: Combined Lexical Blend (0.85 SVC + 0.15 Lexical Comp)
            p_svc = softmax(lp_svc, axis=1)
            p_blend = 0.85 * p_svc + 0.15 * p_lex
            preds_blend = p_blend.argmax(axis=1)
            bacc_blend = float(balanced_accuracy_score(y_val, preds_blend))

            # Stage 4: + Unified Handwriting & Orthographic Shield
            multiplier = post_cfg.get("curly_quote_multiplier", float(np.exp(-1.0)))
            p_shield = apply_unified_handwriting_shield(
                P_mix=p_blend.copy(),
                test_texts=val_texts,
                log_p_handwriting=lp_hw,
                rule_flags=rule_flags,
                curly_quote_multiplier=multiplier,
                multi_em_dash_log_penalty=post_cfg.get("multi_em_dash_log_penalty", 0.0),
                straight_contr_log_penalty=post_cfg.get("straight_contr_log_penalty", 0.0),
                curly_contr_log_penalty=post_cfg.get("curly_contr_log_penalty", 0.0),
                continuous_hw_weight=post_cfg.get("continuous_hw_weight", 0.0),
            )
            preds_shield = p_shield.argmax(axis=1)
            bacc_shield = float(balanced_accuracy_score(y_val, preds_shield))

            # Stage 5: + 300-Step Sinkhorn Balancing
            sink_iters = post_cfg.get("sinkhorn_iterations", 300)
            p_final = sinkhorn_balancing(p_shield.copy(), iters=sink_iters)
            preds_final = p_final.argmax(axis=1)
            bacc_final = float(balanced_accuracy_score(y_val, preds_final))

            fold_time = time.time() - fold_t0
            metrics = {
                "fold": fold_idx,
                "n_val": len(df_fold_val),
                "n_prompts_val": df_fold_val["prompt_id"].nunique(),
                "svc_raw_bacc": bacc_svc,
                "lex_raw_bacc": bacc_lex,
                "lex_blend_bacc": bacc_blend,
                "shield_bacc": bacc_shield,
                "sinkhorn_bacc": bacc_final,
                "seconds": fold_time,
            }
            fold_metrics.append(metrics)
            all_val_true.append(y_val)
            all_val_preds.append(preds_final)

            print(f" Fold {fold_idx}: {len(df_fold_val)} val texts ({df_fold_val['prompt_id'].nunique()} prompts) in {fold_time:.2f}s")
            print(f"   - Step 1: 4-View LinearSVC (Raw)                                    : BAcc = {bacc_svc:.4f}")
            print(f"   - Step 2: 5-View BPE + SGD + Twin + Family Specialists (Raw)        : BAcc = {bacc_lex:.4f}")
            print(f"   - Step 3: Combined Lexical Blend (0.85 * SVC + 0.15 * LexComp)      : BAcc = {bacc_blend:.4f} (Delta={bacc_blend - bacc_svc:+.4f})")
            print(f"   - Step 4: + Unified Handwriting & Orthographic Shield               : BAcc = {bacc_shield:.4f} (Delta={bacc_shield - bacc_blend:+.4f})")
            print(f"   - Step 5: + 300-Step Multiplicative Sinkhorn Mass Balancing         : BAcc = {bacc_final:.4f} (Delta={bacc_final - bacc_shield:+.4f})")

    y_cat_true = np.concatenate(all_val_true)
    y_cat_pred = np.concatenate(all_val_preds)
    top_confused = compute_top_confused_pairs(y_cat_true, y_cat_pred, LABELS, top_n=5)

    print("\n Top-5 Most Confused Sister-Model Pairs (Held-Out Prompts):")
    for rank, cp in enumerate(top_confused, 1):
        sister_tag = f"[{cp['family']} Sister Pair]" if cp["is_sister"] else "[Cross-Family]"
        print(f"   {rank}. {cp['model_a']:<22} <---> {cp['model_b']:<22} : {cp['count']} errors {sister_tag}")

    avg_bacc = float(np.mean([m["sinkhorn_bacc"] for m in fold_metrics]))
    total_time = time.time() - t0
    print(f"\n Summary: Mean Balanced Accuracy = {avg_bacc:.4f} across {len(fold_metrics)} fold(s) in {total_time:.2f}s")
    print("=" * 90)

    return {
        "mean_bacc": avg_bacc,
        "fold_metrics": fold_metrics,
        "top_confused": top_confused,
        "total_seconds": total_time,
    }


def generate_synthetic_cache(
    output_path: Path,
    N_unseen: int = 360,
    N_test: int = 500,
    seed: int = 42,
) -> Path:
    """Generates a realistic multi-backbone simulation cache for instant offline evaluation."""
    np.random.seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_unseen = np.tile(np.arange(K), N_unseen // K).astype(np.int32)
    groups_unseen = ((np.arange(N_unseen) // 3) + 1).astype(np.int32)

    # 10 Simulated models with realistic accuracies and characteristics
    models_spec = [
        ("Qwen3-14B Part-2", "Main", 0.50, 0.88, 3.2),
        ("Qwen2.5-14B Part-2", "Main", 0.35, 0.78, 2.4),
        ("DeBERTa-v3-Large", "Main", 0.30, 0.74, 2.2),
        ("Qwen3-8B Claude Head", "Comp", 0.16, 0.70, 2.0),
        ("Qwen3-8B In-Style", "Comp", 0.20, 0.68, 1.9),
        ("Qwen3-14B In-House", "Comp", 0.45, 0.76, 2.3),
        ("LinearSVC (4-View)", "Main", 0.85, 0.65, 1.7),
        ("Lexical Comp (5-View)", "Comp", 0.15, 0.60, 1.5),
        ("Poison-Adversarial", "Comp", 0.20, 0.05, -3.0),
    ]

    cache_dict: Dict[str, Any] = {
        "N_test": np.array([N_test], dtype=np.int32),
        "N_unseen": np.array([N_unseen], dtype=np.int32),
        "y_unseen": y_unseen,
        "groups_unseen": groups_unseen,
        "D_tfidf": np.zeros((N_test, N_test), dtype=np.float32),
        "D_hybrid": np.zeros((N_test, N_test), dtype=np.float32),
        "groups_085": np.ones(N_test, dtype=np.int32),
        "model_names": np.array([m[0] for m in models_spec], dtype=object),
        "model_tiers": np.array([m[1] for m in models_spec], dtype=object),
        "model_prior_weights": np.array([m[2] for m in models_spec], dtype=np.float64),
        "model_learned_weights": np.array([m[2] for m in models_spec], dtype=np.float64),
        "test_texts": np.array([f"Synthetic test text passage {i} for benchmarking." for i in range(N_test)], dtype=object),
    }

    for idx, (m_name, m_tier, m_prior, m_acc, m_signal) in enumerate(models_spec):
        lp_u = np.random.randn(N_unseen, K) * 0.6
        for i, y in enumerate(y_unseen):
            if m_signal > 0:
                if (i + idx) % 5 != 0:
                    lp_u[i, y] += m_signal
            else:
                lp_u[i, (y + 1) % K] += 3.5
        cache_dict[f"m_{idx}_unseen_lp"] = log_softmax(lp_u, axis=1).astype(np.float32)

        lp_te = np.random.randn(N_test, K) * 0.6
        cache_dict[f"m_{idx}_test_lp"] = log_softmax(lp_te, axis=1).astype(np.float32)

    # Extra streams (wPCA DP & Spectral Bank)
    extra_streams = [
        ("wpca_dp", 0.22, log_softmax(np.random.randn(N_test, K) * 0.5, axis=1).astype(np.float32)),
        ("spectral_bank", 0.12, log_softmax(np.random.randn(N_test, K) * 0.5, axis=1).astype(np.float32)),
    ]
    for ex_idx, (ex_name, ex_w, ex_lp) in enumerate(extra_streams):
        cache_dict[f"extra_{ex_idx}_name"] = np.array([ex_name], dtype=object)
        cache_dict[f"extra_{ex_idx}_weight"] = np.array([ex_w], dtype=np.float64)
        cache_dict[f"extra_{ex_idx}_lp"] = ex_lp
    cache_dict["num_extra"] = np.array([len(extra_streams)], dtype=np.int32)

    cache_dict["rule_flags_test"] = np.zeros((N_test, 4), dtype=np.float32)
    cache_dict["log_p_handwriting"] = log_softmax(np.random.randn(N_test, K), axis=1).astype(np.float32)

    np.savez_compressed(output_path, **cache_dict)
    return output_path



def print_npz_ablation_diagnostics(
    auditor: UnseenStepwiseAuditor,
    active_weights_map: Dict[str, float],
) -> None:
    """
    Comprehensive .npz Ablation Diagnostics:
    - Table A: Effective Vote Share & Logit Scale Audit (Nominal vs. Actual Impact).
    - Table B: Pairwise Prediction Agreement & Error-Rescue Matrix vs. 14B Anchor.
    - Table C: Leave-One-Out (LOO) & Clean-Subset Ablation (Disjoint Non-v5 Subset).
    """
    y_unseen = auditor.y_unseen
    models = auditor.models
    if len(y_unseen) < 18 or len(models) == 0:
        return

    print("\n" + "=" * 115)
    print(" TABLE A: EFFECTIVE VOTE SHARE & LOGIT SCALE AUDIT (Why Nominal Weights Mislead)")
    print("=" * 115)
    print(f"{'Model / Checkpoint':<25} | {'Nominal Wt':<10} | {'LP_std':<8} | {'Gap (1-2)':<9} | {'Effective Share':<15} | {'Learned Wt':<10} | {'Solo BAcc':<9} | {'Sink BAcc':<9}")
    print("-" * 115)

    eff_powers = []
    for m in models:
        u_lp = m["unseen_lp"]
        w = float(m.get("prior_weight", m["weight"]))
        lp_std = float(u_lp.std(axis=1).mean()) if len(u_lp) > 0 else 0.0
        eff_powers.append(w * lp_std)
    total_eff_power = sum(eff_powers)

    for idx, m in enumerate(models):
        name = m["name"]
        w_nom = float(m.get("prior_weight", m["weight"]))
        w_learned = float(active_weights_map.get(name, m["weight"]))
        u_lp = m["unseen_lp"]
        lp_std = float(u_lp.std(axis=1).mean()) if len(u_lp) > 0 else 0.0
        sorted_lp = np.sort(u_lp, axis=1) if len(u_lp) > 0 else np.zeros((0, 2))
        gap = float(np.mean(sorted_lp[:, -1] - sorted_lp[:, -2])) if len(sorted_lp) > 0 else 0.0
        eff_share = (eff_powers[idx] / max(1e-9, total_eff_power)) * 100.0
        solo_raw = float(m.get("solo_unseen_bacc", 0.0))
        solo_sink = float(m.get("solo_unseen_sinkhorn_bacc", 0.0))

        print(f"{name:<25} | {w_nom:<10.2f} | {lp_std:<8.3f} | {gap:<9.3f} | {eff_share:<14.1f}% | {w_learned:<10.2f} | {solo_raw:<9.4f} | {solo_sink:<9.4f}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print(" TABLE B: PAIRWISE PREDICTION AGREEMENT & ERROR-RESCUE MATRIX (vs Primary 14B Anchor)")
    print("=" * 115)
    anchor_idx = None
    for i, m in enumerate(models):
        if "14B Part-2" in m["name"] or "qwen3_14b_p2" in m["name"]:
            anchor_idx = i
            break
    if anchor_idx is None:
        cand_14b = [i for i, m in enumerate(models) if "14b" in m["name"].lower()]
        anchor_idx = max(cand_14b, key=lambda i: models[i]["weight"]) if cand_14b else 0

    anchor_m = models[anchor_idx]
    anchor_name = anchor_m["name"]
    anchor_preds = anchor_m["unseen_lp"].argmax(axis=1)
    anchor_wrong = (anchor_preds != y_unseen)
    n_anchor_err = int(anchor_wrong.sum())

    print(f"Primary Anchor: {anchor_name} | Total Unseen Errors: {n_anchor_err}/{len(y_unseen)} ({n_anchor_err/len(y_unseen)*100.0:.1f}%)")
    print("-" * 115)
    print(f"{'Model / Checkpoint':<25} | {'Tier':<6} | {'Solo BAcc':<9} | {'Agreement w/ Anchor':<21} | {'Errors Rescued':<16} | {'Rescue Rate':<11}")
    print("-" * 115)

    for m in models:
        name = m["name"]
        tier = m["tier"]
        u_lp = m["unseen_lp"]
        preds = u_lp.argmax(axis=1) if len(u_lp) > 0 else np.array([])
        solo_bacc = float(m.get("solo_unseen_bacc", 0.0))
        agree_pct = float(np.mean(preds == anchor_preds) * 100.0) if len(preds) > 0 else 0.0
        rescued = int((anchor_wrong & (preds == y_unseen)).sum()) if len(preds) > 0 else 0
        rescue_rate = (rescued / max(1, n_anchor_err)) * 100.0

        is_anchor_mark = " (Anchor)" if name == anchor_name else ""
        print(f"{name + is_anchor_mark:<25} | {tier:<6} | {solo_bacc:<9.4f} | {agree_pct:<20.1f}% | {rescued:>3d} / {n_anchor_err:<8d} | {rescue_rate:<10.1f}%")
    print("=" * 115)

    print("\n" + "=" * 115)
    print(" TABLE C: LEAVE-ONE-OUT (LOO) & CLEAN-SUBSET ABLATION (Excluding nlm_pool_v5 Contamination)")
    print("=" * 115)

    clean_models = [m for m in models if "v5" not in m["name"].lower()]
    print(f"Clean Disjoint Subset: {len(clean_models)} models ([{', '.join(m['name'] for m in clean_models)}])")
    print("-" * 115)

    def _eval_subset_lp(subset_lps, weights):
        if not subset_lps:
            return 0.0, 99.0
        cum_lp = np.zeros_like(subset_lps[0], dtype=np.float64)
        for lp_i, w_i in zip(subset_lps, weights):
            if w_i > 0:
                cum_lp += w_i * lp_i
        P = softmax(cum_lp, axis=1)
        P_s = P.copy()
        for _ in range(100):
            P_s = P_s / np.maximum(P_s.sum(axis=0, keepdims=True), 1e-12) / 18.0
            P_s = P_s / np.maximum(P_s.sum(axis=1, keepdims=True), 1e-12)
        bacc = float(balanced_accuracy_score(y_unseen, P_s.argmax(axis=1)))
        nll = float(-np.mean(np.log(np.clip(P_s[np.arange(len(y_unseen)), y_unseen], 1e-9, 1.0))))
        return bacc, nll

    clean_lps = [m["unseen_lp"] for m in clean_models]
    clean_ws = [float(active_weights_map.get(m["name"], m["weight"])) for m in clean_models]

    base_clean_bacc, base_clean_nll = _eval_subset_lp(clean_lps, clean_ws)
    print(f"Clean Subset Baseline: Sinkhorn BAcc = {base_clean_bacc:.4f} | NLL = {base_clean_nll:.4f}")
    print("-" * 115)
    print(f"{'Excluded Model':<25} | {'Active Wt':<10} | {'LOO BAcc':<9} | {'Delta BAcc (Drop)':<18} | {'LOO NLL':<8} | {'Delta NLL':<10} | {'Verdict':<15}")
    print("-" * 115)

    for i, m in enumerate(clean_models):
        name = m["name"]
        w = clean_ws[i]
        if w <= 0.0:
            print(f"{name:<25} | {w:<10.2f} | {'N/A':<9} | {'N/A':<18} | {'N/A':<8} | {'N/A':<10} | {'Already Pruned':<15}")
            continue

        loo_lps = [lp for j, lp in enumerate(clean_lps) if j != i]
        loo_ws = [wt for j, wt in enumerate(clean_ws) if j != i]
        loo_bacc, loo_nll = _eval_subset_lp(loo_lps, loo_ws)
        delta_bacc = base_clean_bacc - loo_bacc
        delta_nll = loo_nll - base_clean_nll

        verdict = "Essential (+)" if delta_bacc > 0.005 else "Beneficial (+)" if delta_bacc >= 0.000 else "Neutral/Hurts (-)"
        sign_bacc = f"+{delta_bacc:.4f}" if delta_bacc >= 0 else f"{delta_bacc:.4f}"
        sign_nll = f"+{delta_nll:.4f}" if delta_nll >= 0 else f"{delta_nll:.4f}"
        print(f"{name:<25} | {w:<10.2f} | {loo_bacc:<9.4f} | {sign_bacc:<18} | {loo_nll:<8.4f} | {sign_nll:<10} | {verdict:<15}")
    print("=" * 115)

    print_per_class_accuracy_table(auditor, active_weights_map)


def print_per_class_accuracy_table(
    auditor: UnseenStepwiseAuditor,
    active_weights_map: Dict[str, float],
) -> pd.DataFrame:
    """
    Table D: Per-Class Accuracy (Recall) across all 18 classes for every model in the cache
    plus the calibrated ensemble, alongside the Top Confused Sister/Cross-Family class per author.
    """
    from sklearn.metrics import confusion_matrix

    y_true = auditor.y_unseen
    models = auditor.models
    if len(y_true) < 18 or len(models) == 0:
        return pd.DataFrame()

    # Compute Clean Ensemble (excluding leaked v5 models if non-v5 models exist) and Full Ensemble
    clean_models = [m for m in models if "v5" not in m["name"].lower() and float(active_weights_map.get(m["name"], m["weight"])) > 0]
    eval_models = clean_models if len(clean_models) >= 2 else [m for m in models if float(active_weights_map.get(m["name"], m["weight"])) > 0]

    ens_lp = np.zeros((len(y_true), K), dtype=np.float64)
    for m in eval_models:
        w = float(active_weights_map.get(m["name"], m["weight"]))
        if w > 0:
            ens_lp += w * m["unseen_lp"]
    P_ens = softmax(ens_lp, axis=1)
    for _ in range(60):
        P_ens /= np.maximum(P_ens.sum(axis=0, keepdims=True), 1e-12) / float(K)
        P_ens /= np.maximum(P_ens.sum(axis=1, keepdims=True), 1e-12)
    ens_preds = P_ens.argmax(axis=1)

    # Per-model predictions after 60-step Sinkhorn
    model_preds: Dict[str, np.ndarray] = {}
    for m in models:
        P_m = softmax(m["unseen_lp"], axis=1)
        for _ in range(60):
            P_m /= np.maximum(P_m.sum(axis=0, keepdims=True), 1e-12) / float(K)
            P_m /= np.maximum(P_m.sum(axis=1, keepdims=True), 1e-12)
        model_preds[m["name"]] = P_m.argmax(axis=1)

    cm_ens = confusion_matrix(y_true, ens_preds, labels=list(range(K)))

    print("\n" + "=" * 130)
    print(" TABLE D: PER-CLASS ACCURACY (RECALL) ACROSS ALL 18 AUTHOR CLASSES & TOP CONFUSED SISTER PAIRS")
    print("=" * 130)

    # Short column headers for each model
    m_names = [m["name"] for m in models]
    short_hdrs = [n[:12] for n in m_names]
    hdr_str = " | ".join(f"{h:<12}" for h in short_hdrs)
    print(f"{'Class / Author Label':<24} | {'N':<3} | {hdr_str} | {'Clean Ens':<10} | {'Most Confused With (Ens)'}")
    print("-" * 130)

    rows_out = []
    for c_idx, label in enumerate(LABELS):
        mask_c = (y_true == c_idx)
        n_c = int(mask_c.sum())
        rec_ens = float((ens_preds[mask_c] == c_idx).mean()) if n_c > 0 else 0.0

        # Find top confused class in ensemble
        row_cm = cm_ens[c_idx].copy()
        row_cm[c_idx] = 0
        if row_cm.sum() > 0:
            top_err_idx = int(row_cm.argmax())
            top_err_cnt = int(row_cm[top_err_idx])
            confused_str = f"{LABELS[top_err_idx]} ({top_err_cnt}/{n_c})"
        else:
            confused_str = "None (100%)"

        m_rec_strs = []
        row_dict = {"class_idx": c_idx, "label": label, "n_samples": n_c, "clean_ens_bacc": rec_ens}
        for m_name in m_names:
            r_m = float((model_preds[m_name][mask_c] == c_idx).mean()) if n_c > 0 else 0.0
            row_dict[m_name] = r_m
            m_rec_strs.append(f"{r_m:<12.4f}")

        print(f"{label:<24} | {n_c:<3d} | {' | '.join(m_rec_strs)} | {rec_ens:<10.4f} | {confused_str}")
        rows_out.append(row_dict)

    print("-" * 130)
    mean_m_strs = [f"{float(np.mean([r[m_name] for r in rows_out])):<12.4f}" for m_name in m_names]
    mean_ens = float(np.mean([r["clean_ens_bacc"] for r in rows_out]))
    print(f"{'MEAN BALANCED ACCURACY':<24} | {len(y_true):<3d} | {' | '.join(mean_m_strs)} | {mean_ens:<10.4f} |")
    print("=" * 130)
    return pd.DataFrame(rows_out)



def print_stepwise_greedy_table(data: Dict[str, Any]) -> None:
    """
    Stepwise forward greedy analysis on zero-leakage models:
    Evaluates solo Sinkhorn BAcc to pick Base Model (Step 0), then iteratively adds
    1 model at a time searching a weight grid, reporting Marginal Lift and Top 3 Contenders.
    """
    if "y_unseen" not in data or len(data["y_unseen"]) < 18:
        print("[Stepwise] Cannot run stepwise analysis: unseen holdout is missing or too small.")
        return

    y = data["y_unseen"].astype(int)
    model_names = [str(x) for x in data["model_names"]]

    # Collect zero-leakage models
    candidates: Dict[str, np.ndarray] = {}
    sorted_labels = sorted(LABELS)
    perm = [sorted_labels.index(lbl) for lbl in LABELS]

    for idx, name in enumerate(model_names):
        c_name = CANONICAL_NAME_MAP.get(name, name)
        # Skip v5 models contaminated with nlm_pool_v5 holdout leakage
        if "v5" in c_name.lower():
            continue
        u_lp = data[f"m_{idx}_unseen_lp"].copy()
        # Fix label alignment for claude head if applicable
        if c_name == "qwen3_8b_claude":
            u_lp = u_lp[:, perm]
        candidates[c_name] = u_lp

    if not candidates:
        return

    def fast_sinkhorn_bacc(lp: np.ndarray) -> float:
        P = softmax(lp, axis=1)
        for _ in range(60):
            P = P / np.maximum(P.sum(axis=0, keepdims=True), 1e-12) / 18.0
            P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        return float(balanced_accuracy_score(y, P.argmax(axis=1)))

    weight_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

    solo_scores = {k: fast_sinkhorn_bacc(v) for k, v in candidates.items()}
    sorted_solo = sorted(solo_scores.items(), key=lambda x: x[1], reverse=True)
    base_model, base_bacc = sorted_solo[0]
    top3_base = ", ".join([f"{k} ({s:.4f})" for k, s in sorted_solo[:3]])

    print("\n" + "=" * 125)
    print(" ZERO-LEAKAGE STEPWISE GREEDY MODEL & WEIGHT SELECTION (With Sinkhorn Balancing)")
    print("=" * 125)
    print(f"{'Step':<16} | {'Selected Model':<20} | {'Selected Weight':<15} | {'Ensemble BAcc':<14} | {'Marginal Lift':<14} | {'Top 3 Contenders Evaluated at this Step'}")
    print("-" * 125)
    print(f"{'Base model -Step':<16} | {base_model:<20} | {'1.00 (Anchor)':<15} | {base_bacc:<14.4f} | {'Baseline':<14} | {top3_base}")

    ensemble_lp = candidates[base_model].copy()
    cur_bacc = base_bacc
    remaining = [k for k in candidates.keys() if k != base_model]

    step_num = 1
    while remaining:
        cand_evals = []
        for cand in remaining:
            cand_lp = candidates[cand]
            best_w = 0.0
            best_bacc = 0.0
            for w in weight_grid:
                trial = ensemble_lp + w * cand_lp
                bacc = fast_sinkhorn_bacc(trial)
                if bacc > best_bacc:
                    best_bacc = bacc
                    best_w = w
            cand_evals.append((cand, best_w, best_bacc, best_bacc - cur_bacc))

        cand_evals.sort(key=lambda x: x[2], reverse=True)
        best_cand, best_w, best_bacc, best_lift = cand_evals[0]
        top3_str = ", ".join([f"{c} (w={w:.2f}, {b:.4f}, {l:+.4f})" for c, w, b, l in cand_evals[:3]])

        step_label = f"Model {step_num} - Step"
        print(f"{step_label:<16} | {best_cand:<20} | {best_w:<15.2f} | {best_bacc:<14.4f} | {best_lift:<+14.4f} | {top3_str}")

        ensemble_lp = ensemble_lp + best_w * candidates[best_cand]
        cur_bacc = best_bacc
        remaining.remove(best_cand)
        step_num += 1

    print("=" * 125 + "\n")

def evaluate_from_cache(
    cache_path: Path,
    config_path: Optional[Path] = None,
    working_dir: Optional[Path] = None,
    strict_config: bool = True,
    run_stepwise: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark B: Sub-second offline replay of cached neural representations.
    Executes UnseenStepwiseAuditor, Caruana selection, post-processing, and Sinkhorn.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at: {cache_path}")

    t0 = time.time()
    raw_npz = np.load(cache_path, allow_pickle=True)
    data = {k: raw_npz[k] for k in raw_npz.files}
    raw_npz.close()

    cfg = load_config(config_path)
    post_cfg = cfg.get("postprocess", cfg.get("postprocessing", {}))
    cfg_weights = cfg.get("model_weights", {})
    lock_priors = bool(cfg.get("lock_prior_weights", True))

    N_test = int(data["N_test"][0])
    N_unseen = int(data["N_unseen"][0])
    y_unseen = data["y_unseen"].astype(int)
    groups_unseen = data["groups_unseen"].astype(int)
    model_names = [str(x) for x in data["model_names"]]
    model_tiers = [str(x) for x in data["model_tiers"]]
    model_prior_weights = data["model_prior_weights"].astype(float)

    auditor = UnseenStepwiseAuditor(y_unseen=y_unseen, groups_unseen=groups_unseen, N_test=N_test)

    active_models_summary = []
    pruned_models_summary = []
    added_names = set()

    for idx, (raw_name, tier, prior_w) in enumerate(zip(model_names, model_tiers, model_prior_weights)):
        canon_name = CANONICAL_NAME_MAP.get(raw_name, raw_name)
        u_lp = data[f"m_{idx}_unseen_lp"]
        t_lp = data[f"m_{idx}_test_lp"]

        if strict_config and cfg_weights:
            if canon_name in cfg_weights:
                mw = cfg_weights[canon_name]
                cfg_w = float(mw.get("plain_weight", 0.0) + mw.get("twin_weight", 0.0))
                if cfg_w > 0.0:
                    auditor.add_model(name=canon_name, tier=tier, weight=cfg_w, unseen_log_probs=u_lp, test_log_probs=t_lp)
                    active_models_summary.append((canon_name, cfg_w, tier))
                    added_names.add(canon_name)
                else:
                    pruned_models_summary.append((f"{raw_name} ({canon_name})", f"Weight is {cfg_w:.2f} (Specialist Gate Only / Pruned)"))
            else:
                pruned_models_summary.append((f"{raw_name} ({canon_name})", "Not in active ensemble config"))
        else:
            auditor.add_model(name=canon_name, tier=tier, weight=prior_w, unseen_log_probs=u_lp, test_log_probs=t_lp)
            active_models_summary.append((canon_name, prior_w, tier))
            added_names.add(canon_name)

    # Reconstruct extra streams
    extra_streams: List[Tuple[str, float, np.ndarray]] = []
    active_extras_summary = []
    num_extra = int(data["num_extra"][0]) if "num_extra" in data else 0
    for ex_idx in range(num_extra):
        raw_ex_name = str(data[f"extra_{ex_idx}_name"][0])
        canon_ex_name = CANONICAL_NAME_MAP.get(raw_ex_name, raw_ex_name)
        ex_w = float(data[f"extra_{ex_idx}_weight"][0])
        ex_lp = data[f"extra_{ex_idx}_lp"]

        if strict_config and cfg_weights:
            if canon_ex_name in cfg_weights:
                cfg_ex_w = float(cfg_weights[canon_ex_name].get("plain_weight", ex_w))
                if cfg_ex_w > 0.0:
                    extra_streams.append((canon_ex_name, cfg_ex_w, ex_lp))
                    active_extras_summary.append((canon_ex_name, cfg_ex_w))
                    added_names.add(canon_ex_name)
                else:
                    pruned_models_summary.append((f"{raw_ex_name} ({canon_ex_name})", f"Weight is {cfg_ex_w:.2f}"))
            else:
                pruned_models_summary.append((f"{raw_ex_name} ({canon_ex_name})", "Not in active ensemble config"))
        else:
            extra_streams.append((canon_ex_name, ex_w, ex_lp))
            active_extras_summary.append((canon_ex_name, ex_w))
            added_names.add(canon_ex_name)

    missing_models_summary = []
    if strict_config and cfg_weights:
        for k, mw in cfg_weights.items():
            tot_w = float(mw.get("plain_weight", 0.0) + mw.get("twin_weight", 0.0))
            if tot_w > 0.0 and k not in added_names:
                missing_models_summary.append((k, tot_w))

    print("\n" + "=" * 90)
    print(f" CONFIG-DRIVEN ROSTER AUDIT (strict_config={strict_config})")
    print("=" * 90)
    print(f" Active Models in Ensemble ({len(active_models_summary)}):")
    for name, w, tier in active_models_summary:
        print(f"   - {name:<22}: weight = {w:<5.2f} ({tier})")
    print(f" Active Extra Streams ({len(active_extras_summary)}):")
    for name, w in active_extras_summary:
        print(f"   - {name:<22}: weight = {w:<5.2f}")
    if pruned_models_summary:
        print(f" Pruned / Excluded Models ({len(pruned_models_summary)}):")
        for name, reason in pruned_models_summary:
            print(f"   - {name:<22}: {reason}")
    if missing_models_summary:
        print(f" Configured Models Missing in Cache ({len(missing_models_summary)}):")
        for name, w in missing_models_summary:
            print(f"   - {name:<22}: Config weight = {w:<5.2f} (To be extracted / mounted)")
    print("=" * 90)

    if run_stepwise:
        print_stepwise_greedy_table(data)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        report_dir = Path(tmp_dir) if working_dir is None else Path(working_dir)
        optimize_on_unseen = (N_unseen >= 36)

        # 1. Caruana Greedy Stepwise Selection with Smooth NLL Tie-Breaking
        P_mix, active_weights_map, df_ablation = auditor.optimize_weights_and_blend(
            working_dir=report_dir,
            extra_test_log_probs=extra_streams,
            optimize_on_unseen=optimize_on_unseen,
            lock_prior_weights=lock_priors,
        )

        for raw_name in model_names:
            c_name = CANONICAL_NAME_MAP.get(raw_name, raw_name)
            if c_name not in active_weights_map:
                active_weights_map[c_name] = 0.0
            if raw_name not in active_weights_map:
                active_weights_map[raw_name] = 0.0
        R_main = P_mix.copy()

        # 2. Claude Specialist Gate
        gate_weight = post_cfg.get("claude_specialist_weight", 0.15)
        claude_spec = data["claude_spec_logits"] if "claude_spec_logits" in data else None
        claude_classes = list(data["claude_classes"]) if "claude_classes" in data else None
        if claude_spec is not None:
            P_mix = apply_claude_specialist_gate(
                P_mix=P_mix,
                R_main=R_main,
                claude_spec_logits=claude_spec,
                claude_classes=claude_classes,
                labels=LABELS,
                claude_cols=FAMILIES["claude"],
                gate_weight=gate_weight,
            )

        # 2b. Specialist Confidence Gates (4 families + 3 targeted tie-breaker groups)
        family_gates = {}
        for fam_k in SPECIALIST_GROUPS.keys():
            if f"gate_{fam_k}_lp" in data:
                family_gates[fam_k] = data[f"gate_{fam_k}_lp"]
        if family_gates:
            P_mix = apply_all_family_confidence_gates(
                P_mix=P_mix,
                family_gates=family_gates,
                max_strength=post_cfg.get("family_gate_strength", 0.20),
            )

        # 3. Unified Handwriting & Orthographic Shield
        test_texts = list(data["test_texts"]) if "test_texts" in data else [""] * N_test
        rule_flags = data["rule_flags_test"] if "rule_flags_test" in data else None
        log_p_hw = data["log_p_handwriting"] if "log_p_handwriting" in data else None
        multiplier = post_cfg.get("curly_quote_multiplier", float(np.exp(-1.0)))

        P_mix = apply_unified_handwriting_shield(
            P_mix=P_mix,
            test_texts=test_texts,
            log_p_handwriting=log_p_hw,
            rule_flags=rule_flags,
            curly_quote_multiplier=multiplier,
            multi_em_dash_log_penalty=post_cfg.get("multi_em_dash_log_penalty", 0.0),
            straight_contr_log_penalty=post_cfg.get("straight_contr_log_penalty", 0.0),
            curly_contr_log_penalty=post_cfg.get("curly_contr_log_penalty", 0.0),
            continuous_hw_weight=post_cfg.get("continuous_hw_weight", 0.0),
        )

        # 3b. Cavity Cluster Uniqueness Repulsion
        groups_065 = data.get("groups_065", data.get("groups_085", None))
        if groups_065 is not None:
            P_mix = apply_cluster_uniqueness_repulsion(
                P_mix=P_mix,
                groups=groups_065,
                strength=post_cfg.get("cluster_repulsion_strength", 0.35),
                iters=3,
            )

        # Load train_texts and y_train if available in data or fallback to local_data/train.csv
        train_texts_list = list(data["train_texts"]) if "train_texts" in data else []
        y_train_arr = data["y_train"].astype(int) if "y_train" in data else np.array([], dtype=int)
        default_train_csv = WORKSPACE_DIR / "local_data" / "train.csv"
        if len(train_texts_list) == 0 and default_train_csv.exists():
            try:
                df_tr_local = pd.read_csv(default_train_csv)
                if "text" in df_tr_local.columns and "model" in df_tr_local.columns:
                    train_texts_list = df_tr_local["text"].fillna("").astype(str).tolist()
                    y_train_arr = np.array([LAB2I[m] for m in df_tr_local["model"].values if m in LAB2I], dtype=int)
            except Exception:
                pass

        # 4. Sinkhorn Mass Balancing (with soft prior-shrunk balancing option)
        sinkhorn_iters = post_cfg.get("sinkhorn_iterations", 100)
        sinkhorn_strength = float(post_cfg.get("sinkhorn_strength", 0.65))
        sinkhorn_target_mix = float(post_cfg.get("sinkhorn_target_mix", 0.35))
        if len(y_train_arr) > 0:
            train_prior = np.bincount(y_train_arr, minlength=18).astype(np.float64) + 1.0
        elif len(y_unseen) > 0:
            train_prior = np.bincount(y_unseen, minlength=18).astype(np.float64) + 1.0
        else:
            train_prior = None

        P_final = sinkhorn_balancing(
            P_mix,
            iters=sinkhorn_iters,
            sinkhorn_strength=sinkhorn_strength,
            target_mix=sinkhorn_target_mix,
            train_prior=train_prior
        )

        # 5. TypoBayes 37-Marker Hierarchical Beta-Bernoulli Postprocessing
        if post_cfg.get("enable_typobayes", True) and "test_texts" in data and len(train_texts_list) > 0:
            P_final = apply_typobayes_postprocess(
                P_final,
                train_texts=train_texts_list,
                y_train=y_train_arr if len(y_train_arr) == len(train_texts_list) else y_unseen,
                test_texts=test_texts,
                labels=LABELS,
                kappa=float(post_cfg.get("typobayes_kappa", 24.0)),
                evidence_scale=float(post_cfg.get("typobayes_evidence_scale", 1.2)),
                evidence_cap=float(post_cfg.get("typobayes_evidence_cap", 0.30)),
            )

        test_final_preds = P_final.argmax(axis=1) if len(P_final) > 0 else np.array([])
        df_report = auditor.print_and_save_report(report_dir, final_test_preds=test_final_preds)
        print_npz_ablation_diagnostics(auditor, active_weights_map)

    replay_time = time.time() - t0
    print(f"\n[Cache Evaluator] Replay completed in {replay_time:.4f}s across {len(model_names)} models and {num_extra} extra streams!")
    print(f"[Cache Evaluator] Active Weights Map: {active_weights_map}")
    return {
        "P_final": P_final,
        "weights_map": active_weights_map,
        "df_report": df_report,
        "replay_time": replay_time,
    }


def main():
    parser = argparse.ArgumentParser(description="NameThatLLM - Fail-Fast Local Evaluation & Ablation Harness")
    parser.add_argument("--mode", choices=["split", "cache", "simulate", "stepwise"], default="split",
                        help="Evaluation mode: 'split' (prompt-grouped holdout), 'cache' (replay local_eval_cache.npz), 'simulate' (synthetic cache), or 'stepwise' (zero-leakage stepwise analysis).")
    parser.add_argument("--strict-config", action=argparse.BooleanOptionalAction, default=True,
                        help="Strictly filter cache models against configs/production_ensemble.json (default: True).")
    parser.add_argument("--stepwise", action="store_true",
                        help="Run zero-leakage stepwise forward greedy selection analysis.")
    parser.add_argument("--train-path", type=str, default="local_data/train.csv",
                        help="Path to train.csv for --mode split.")
    parser.add_argument("--cache-file", type=str, default="local_data/local_eval_cache.npz",
                        help="Path to local_eval_cache.npz for --mode cache.")
    parser.add_argument("--config", type=str, default="configs/production_ensemble.json",
                        help="Path to ensemble JSON configuration.")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of GroupKFold splits for --mode split.")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Evaluate fold 0 only for ~4s execution.")
    parser.add_argument("--all-folds", action="store_true",
                        help="Evaluate all 5 folds.")

    args = parser.parse_args()

    if args.mode == "split":
        fast = False if args.all_folds else args.fast
        evaluate_prompt_split(
            train_path=Path(args.train_path),
            n_splits=args.folds,
            fast=fast,
            config_path=Path(args.config) if args.config else None,
        )
    elif args.mode == "simulate":
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            sim_cache = Path(tmp_dir) / "sim_eval_cache.npz"
            print(f"[Simulate] Synthesizing realistic 10-model multi-trunk cache at {sim_cache}...")
            generate_synthetic_cache(sim_cache)
            evaluate_from_cache(sim_cache, config_path=Path(args.config) if args.config else None)
    elif args.mode in ("cache", "stepwise"):
        run_stepwise = (args.mode == "stepwise" or args.stepwise)
        c_path = Path(args.cache_file)
        if not c_path.exists():
            print(f"[Warning] Cache file {c_path} not found; falling back to --mode simulate...")
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
                sim_cache = Path(tmp_dir) / "sim_eval_cache.npz"
                generate_synthetic_cache(sim_cache)
                evaluate_from_cache(
                    sim_cache,
                    config_path=Path(args.config) if args.config else None,
                    strict_config=args.strict_config,
                    run_stepwise=run_stepwise,
                )
        else:
            evaluate_from_cache(
                c_path,
                config_path=Path(args.config) if args.config else None,
                strict_config=args.strict_config,
                run_stepwise=run_stepwise,
            )


if __name__ == "__main__":
    main()
