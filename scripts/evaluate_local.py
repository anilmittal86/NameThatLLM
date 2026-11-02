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

from namethatllm.config import LABELS, K, LAB2I, FAMILIES
from namethatllm.lexical_engine import run_lexical_engine
from namethatllm.ablation import UnseenStepwiseAuditor, _fast_sinkhorn_bacc
from namethatllm.postprocess import (
    apply_claude_specialist_gate,
    apply_unified_handwriting_shield,
    sinkhorn_balancing,
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
    post_cfg = cfg.get("postprocessing", {})
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
                multi_em_dash_log_penalty=post_cfg.get("multi_em_dash_log_penalty", -0.45),
                straight_contr_log_penalty=post_cfg.get("straight_contr_log_penalty", -0.60),
                curly_contr_log_penalty=post_cfg.get("curly_contr_log_penalty", -0.40),
                continuous_hw_weight=post_cfg.get("continuous_hw_weight", 0.04),
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


def evaluate_from_cache(
    cache_path: Path,
    config_path: Optional[Path] = None,
    working_dir: Optional[Path] = None,
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
    post_cfg = cfg.get("postprocessing", {})

    N_test = int(data["N_test"][0])
    N_unseen = int(data["N_unseen"][0])
    y_unseen = data["y_unseen"].astype(int)
    groups_unseen = data["groups_unseen"].astype(int)
    model_names = [str(x) for x in data["model_names"]]
    model_tiers = [str(x) for x in data["model_tiers"]]
    model_prior_weights = data["model_prior_weights"].astype(float)

    auditor = UnseenStepwiseAuditor(y_unseen=y_unseen, groups_unseen=groups_unseen, N_test=N_test)

    for idx, (name, tier, prior_w) in enumerate(zip(model_names, model_tiers, model_prior_weights)):
        u_lp = data[f"m_{idx}_unseen_lp"]
        t_lp = data[f"m_{idx}_test_lp"]
        auditor.add_model(name=name, tier=tier, weight=prior_w, unseen_log_probs=u_lp, test_log_probs=t_lp)

    # Reconstruct extra streams
    extra_streams: List[Tuple[str, float, np.ndarray]] = []
    num_extra = int(data["num_extra"][0]) if "num_extra" in data else 0
    for ex_idx in range(num_extra):
        ex_name = str(data[f"extra_{ex_idx}_name"][0])
        ex_w = float(data[f"extra_{ex_idx}_weight"][0])
        ex_lp = data[f"extra_{ex_idx}_lp"]
        extra_streams.append((ex_name, ex_w, ex_lp))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        report_dir = Path(tmp_dir) if working_dir is None else Path(working_dir)
        optimize_on_unseen = (N_unseen >= 36)

        # 1. Caruana Greedy Stepwise Selection with Smooth NLL Tie-Breaking
        P_mix, active_weights_map, df_ablation = auditor.optimize_weights_and_blend(
            working_dir=report_dir,
            extra_test_log_probs=extra_streams,
            optimize_on_unseen=optimize_on_unseen,
        )
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
            multi_em_dash_log_penalty=post_cfg.get("multi_em_dash_log_penalty", -0.45),
            straight_contr_log_penalty=post_cfg.get("straight_contr_log_penalty", -0.60),
            curly_contr_log_penalty=post_cfg.get("curly_contr_log_penalty", -0.40),
            continuous_hw_weight=post_cfg.get("continuous_hw_weight", 0.04),
        )

        # 4. 300-Step Sinkhorn Mass Balancing
        sinkhorn_iters = post_cfg.get("sinkhorn_iterations", 300)
        P_final = sinkhorn_balancing(P_mix, iters=sinkhorn_iters)

        test_final_preds = P_final.argmax(axis=1) if len(P_final) > 0 else np.array([])
        df_report = auditor.print_and_save_report(report_dir, final_test_preds=test_final_preds)

    replay_time = time.time() - t0
    print(f"\n[Cache Evaluator] Replay completed in {replay_time:.4f}s across {len(model_names)} models and {num_extra} extra streams!")
    return {
        "P_final": P_final,
        "weights_map": active_weights_map,
        "df_report": df_report,
        "replay_time": replay_time,
    }


def main():
    parser = argparse.ArgumentParser(description="NameThatLLM - Fail-Fast Local Evaluation & Ablation Harness")
    parser.add_argument("--mode", choices=["split", "cache", "simulate"], default="split",
                        help="Evaluation mode: 'split' (prompt-grouped holdout), 'cache' (replay local_eval_cache.npz), or 'simulate' (synthetic cache).")
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
    elif args.mode == "cache":
        c_path = Path(args.cache_file)
        if not c_path.exists():
            print(f"[Warning] Cache file {c_path} not found; falling back to --mode simulate...")
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
                sim_cache = Path(tmp_dir) / "sim_eval_cache.npz"
                generate_synthetic_cache(sim_cache)
                evaluate_from_cache(sim_cache, config_path=Path(args.config) if args.config else None)
        else:
            evaluate_from_cache(c_path, config_path=Path(args.config) if args.config else None)


if __name__ == "__main__":
    main()
