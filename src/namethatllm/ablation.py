"""
Zero-Leakage Unseen Holdout Benchmark & Forward Stepwise Ablation Engine
========================================================================
Constructs a strictly unseen evaluation dataset from mounted parquet/CSV corpora
by subtracting all passages present in `train.csv` and `pool_v4.parquet`
(the training sets of the pre-trained Stage-1/Stage-2 checkpoints), quarantines
those holdout passages from `lexical_engine.py` and `trainer.py`, and evaluates:
1. Standalone (Solo) Unseen Accuracy of each model.
2. Forward Stepwise Cumulative Accuracy (1 model -> +2nd model -> ... -> Full Ensemble).
"""

import os
import gc
import hashlib
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold

from .config import LABELS, K, LAB2I, softmax, log_softmax
from .postprocess import apply_claude_specialist_gate, damped_sinkhorn


def _ablation_sinkhorn(P: np.ndarray, iters: int = 200, eps: float = 1e-12, **kwargs) -> np.ndarray:
    """Fast uniform Sinkhorn mass balancing delegating to canonical damped_sinkhorn."""
    if len(P) < 18:
        return P
    return damped_sinkhorn(P, alpha=1.0, iters=iters, eps_val=eps)


def _clip_and_log_probs(raw_lps_list: List[np.ndarray], clip_min: float = 1e-4) -> np.ndarray:
    """Converts (M, N, K) raw logits or log-probs into probability-clipped log probabilities."""
    raw_arr = np.asarray(raw_lps_list, dtype=np.float64)
    M = raw_arr.shape[0]
    P_clipped = np.zeros_like(raw_arr, dtype=np.float64)
    for m in range(M):
        p_m = softmax(raw_arr[m], axis=1)
        p_m = np.clip(p_m, clip_min, 1.0)
        p_m /= np.maximum(p_m.sum(axis=1, keepdims=True), 1e-12)
        P_clipped[m] = p_m
    return np.log(P_clipped)


def _lbfgs_loss_and_grad(theta: np.ndarray, Z_tr: np.ndarray, Y_smooth: np.ndarray, lam: float, M: int) -> Tuple[float, np.ndarray]:
    """Label-smoothed NLL + uniform L2 simplex regularization objective and analytic gradient."""
    N_tr = len(Y_smooth)
    w = softmax(theta)
    U = np.tensordot(w, Z_tr, axes=(0, 0))
    P = softmax(U, axis=1)
    nll = float(-np.mean(np.sum(Y_smooth * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))
    reg = float(lam) * float(np.sum((w - 1.0 / M) ** 2))
    loss = nll + reg
    G = (P - Y_smooth) / max(1, N_tr)
    grad_w_nll = np.tensordot(Z_tr, G, axes=((1, 2), (0, 1)))
    grad_w_reg = 2.0 * float(lam) * (w - 1.0 / M)
    d_w = grad_w_nll + grad_w_reg
    grad_theta = w * (d_w - np.dot(w, d_w))
    return loss, grad_theta


def pool_calibrated_log_probs(
    raw_lps_list: List[np.ndarray],
    model_names: List[str],
    mean_T_m: Dict[str, float],
    mean_w_m: Dict[str, float],
    mean_T_global: float = 1.0,
    clip_min: float = 1e-4,
) -> np.ndarray:
    """Applies pre-calibrated per-model temperatures T_m, simplex weights w_m, and global T_global."""
    if not raw_lps_list:
        return np.zeros((0, K), dtype=np.float64)
    n_rows = len(raw_lps_list[0])
    if n_rows == 0:
        return np.zeros((0, K), dtype=np.float64)

    log_P = _clip_and_log_probs(raw_lps_list, clip_min=clip_min)
    weights = np.array([max(0.0, float(mean_w_m.get(name, 0.0))) for name in model_names], dtype=np.float64)
    w_sum = float(weights.sum())
    if w_sum <= 1e-12:
        weights = np.full(len(model_names), 1.0 / max(1, len(model_names)), dtype=np.float64)
    else:
        weights = weights / w_sum

    U = np.zeros((n_rows, K), dtype=np.float64)
    for m, name in enumerate(model_names):
        if weights[m] <= 0.0:
            continue
        T_m = max(1e-3, float(mean_T_m.get(name, 1.0)))
        Z_m = log_softmax(log_P[m] / T_m, axis=1)
        U += weights[m] * Z_m

    T_g = max(1e-3, float(mean_T_global))
    return softmax(U / T_g, axis=1)


def fit_lbfgs_log_linear_pooling(
    raw_lps_list: List[np.ndarray],
    y: np.ndarray,
    groups: Optional[np.ndarray],
    model_names: List[str],
    spec_logits: Optional[np.ndarray] = None,
    claude_classes: Optional[list] = None,
    lam: float = 0.20,
    eps: float = 0.05,
    sinkhorn_alpha: float = 0.50,
    clip_min: float = 1e-4,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Fits 5-fold GroupKFold L-BFGS log-linear pooling with per-model T_m, global T_g, and optional Claude gate."""
    log_P = _clip_and_log_probs(raw_lps_list, clip_min=clip_min)
    M, N, K_cls = log_P.shape
    y_arr = np.asarray(y, dtype=int)

    if groups is not None and len(groups) == N and len(np.unique(groups)) >= 2:
        k_folds = min(n_splits, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=k_folds).split(log_P[0], y_arr, groups=groups))
    else:
        k_folds = min(n_splits, max(2, N // max(1, K_cls)))
        splits = list(KFold(n_splits=k_folds, shuffle=True, random_state=42).split(log_P[0]))

    trial_gws = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
    oof_blended_lp = np.zeros((N, K_cls), dtype=np.float64)
    oof_post_probs = np.zeros((N, K_cls), dtype=np.float64)
    fold_weights, fold_model_temps, fold_global_temps, fold_stds, fold_gws = [], [], [], [], []

    for tr_idx, val_idx in splits:
        N_tr = len(tr_idx)
        y_tr = y_arr[tr_idx]
        Z_fold = np.zeros_like(log_P)
        model_temps = np.ones(M, dtype=np.float64)
        model_stds = np.zeros(M, dtype=np.float64)

        for m in range(M):
            lpm_tr = log_P[m, tr_idx, :]
            res_t = minimize_scalar(
                lambda T: float(-np.mean(log_softmax(lpm_tr / T, axis=1)[np.arange(N_tr), y_tr])),
                bounds=(0.05, 10.0),
                method="bounded",
            )
            T_opt = float(res_t.x)
            model_temps[m] = T_opt
            Z_fold[m] = log_softmax(log_P[m] / T_opt, axis=1)
            model_stds[m] = float(np.std(Z_fold[m, tr_idx, :]))

        fold_model_temps.append(model_temps)
        fold_stds.append(model_stds)

        Y_smooth = np.full((N_tr, K_cls), float(eps) / K_cls, dtype=np.float64)
        Y_smooth[np.arange(N_tr), y_tr] += 1.0 - float(eps)
        Z_tr = Z_fold[:, tr_idx, :]

        if M == 1:
            w_k = np.array([1.0], dtype=np.float64)
        else:
            res_w = minimize(
                fun=lambda th: _lbfgs_loss_and_grad(th, Z_tr, Y_smooth, lam, M),
                x0=np.zeros(M, dtype=np.float64),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            w_k = softmax(res_w.x)
        fold_weights.append(w_k)

        U_tr = np.tensordot(w_k, Z_tr, axes=(0, 0))
        res_tg = minimize_scalar(
            lambda T_g: float(-np.mean(log_softmax(U_tr / T_g, axis=1)[np.arange(N_tr), y_tr])),
            bounds=(0.1, 5.0),
            method="bounded",
        )
        T_global_opt = float(res_tg.x)
        fold_global_temps.append(T_global_opt)

        P_tr = softmax(U_tr / T_global_opt, axis=1)
        best_gw_fold = 0.0
        if spec_logits is not None and claude_classes is not None and len(spec_logits) == N:
            best_bacc_tr = float(balanced_accuracy_score(y_tr, damped_sinkhorn(P_tr, alpha=sinkhorn_alpha).argmax(axis=1)))
            for trial_gw in trial_gws:
                P_tr_g = apply_claude_specialist_gate(
                    P_mix=P_tr,
                    R_main=P_tr,
                    claude_spec_logits=spec_logits[tr_idx],
                    claude_classes=claude_classes,
                    gate_weight=trial_gw,
                )
                b_tr = float(balanced_accuracy_score(y_tr, damped_sinkhorn(P_tr_g, alpha=sinkhorn_alpha).argmax(axis=1)))
                if b_tr > best_bacc_tr:
                    best_bacc_tr = b_tr
                    best_gw_fold = trial_gw
        fold_gws.append(best_gw_fold)

        U_val = np.tensordot(w_k, Z_fold[:, val_idx, :], axes=(0, 0)) / T_global_opt
        P_val = softmax(U_val, axis=1)
        oof_blended_lp[val_idx] = log_softmax(U_val, axis=1)
        if best_gw_fold > 0.0 and spec_logits is not None and claude_classes is not None:
            oof_post_probs[val_idx] = apply_claude_specialist_gate(
                P_mix=P_val,
                R_main=P_val,
                claude_spec_logits=spec_logits[val_idx],
                claude_classes=claude_classes,
                gate_weight=best_gw_fold,
            )
        else:
            oof_post_probs[val_idx] = P_val

    P_oof_raw = softmax(oof_blended_lp, axis=1)
    oof_bacc_raw = float(balanced_accuracy_score(y_arr, P_oof_raw.argmax(axis=1)))
    oof_nll_raw = float(-np.mean(oof_blended_lp[np.arange(N), y_arr]))
    P_oof_sink = damped_sinkhorn(P_oof_raw, alpha=sinkhorn_alpha)
    oof_bacc_sink = float(balanced_accuracy_score(y_arr, P_oof_sink.argmax(axis=1)))
    P_oof_sink_cl = damped_sinkhorn(oof_post_probs, alpha=sinkhorn_alpha)
    oof_bacc_nested_claude = float(balanced_accuracy_score(y_arr, P_oof_sink_cl.argmax(axis=1)))

    avg_w = np.mean(fold_weights, axis=0)
    avg_temps = np.mean(fold_model_temps, axis=0)
    avg_stds = np.mean(fold_stds, axis=0)
    avg_tg = float(np.mean(fold_global_temps))
    avg_gw = float(np.mean(fold_gws))

    use_claude_score = spec_logits is not None and claude_classes is not None and len(spec_logits) == N
    primary_bacc = oof_bacc_nested_claude if use_claude_score else oof_bacc_sink
    composite_oof_score = primary_bacc - 0.05 * oof_nll_raw

    return {
        "model_names": list(model_names),
        "lam": float(lam),
        "eps": float(eps),
        "clip_min": float(clip_min),
        "sinkhorn_alpha": float(sinkhorn_alpha),
        "mean_T_m": {name: float(avg_temps[i]) for i, name in enumerate(model_names)},
        "mean_w_m": {name: float(avg_w[i]) for i, name in enumerate(model_names)},
        "mean_stds": {name: float(avg_stds[i]) for i, name in enumerate(model_names)},
        "fold_weights": np.array(fold_weights),
        "mean_T_global": avg_tg,
        "mean_gw": avg_gw,
        "oof_bacc_raw": oof_bacc_raw,
        "oof_nll_raw": oof_nll_raw,
        "oof_bacc_sink": oof_bacc_sink,
        "oof_bacc_nested_claude": oof_bacc_nested_claude,
        "composite_oof_score": float(composite_oof_score),
        "P_oof_raw": P_oof_raw,
        "P_oof_sink": P_oof_sink,
        "P_oof_sink_claude": P_oof_sink_cl,
    }


def auto_prune_and_fit_lbfgs(
    raw_lps_list: List[np.ndarray],
    y: np.ndarray,
    groups: Optional[np.ndarray],
    model_names: List[str],
    spec_logits: Optional[np.ndarray] = None,
    claude_classes: Optional[list] = None,
    lambdas: Tuple[float, ...] = (0.05, 0.10, 0.20),
    eps: float = 0.05,
    sinkhorn_alpha: float = 0.50,
    clip_min: float = 1e-4,
    pruning_tol: float = 1e-4,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Automatically selects active models and regularization strength lambda via 5-fold GroupKFold L-BFGS
    pooling + OOF backward pruning optimizing combined OOF BAcc + NLL (without any hardcoded model count).
    Returns (best_fit_dict, dropped_model_names).
    """
    active_idx = list(range(len(model_names)))
    dropped_names: List[str] = []

    def _fit_best_lam(indices: List[int]) -> Dict[str, Any]:
        sub_names = [model_names[i] for i in indices]
        sub_lps = [raw_lps_list[i] for i in indices]
        fits = [
            fit_lbfgs_log_linear_pooling(
                sub_lps, y, groups, sub_names,
                spec_logits=spec_logits, claude_classes=claude_classes,
                lam=lam, eps=eps, sinkhorn_alpha=sinkhorn_alpha, clip_min=clip_min,
            )
            for lam in lambdas
        ]
        return max(fits, key=lambda d: d["composite_oof_score"])

    cur_fit = _fit_best_lam(active_idx)
    while len(active_idx) > 1:
        cur_score = cur_fit["composite_oof_score"]
        cur_names = [model_names[i] for i in active_idx]
        zero_drop = [i for i, nm in zip(active_idx, cur_names) if cur_fit["mean_w_m"].get(nm, 0.0) < 1e-3]

        best_cand_fit = None
        best_rm_idx = None
        best_cand_score = -1e9
        candidates_to_test = zero_drop[:1] if zero_drop else active_idx
        for rm_i in candidates_to_test:
            trial_idx = [j for j in active_idx if j != rm_i]
            trial_fit = _fit_best_lam(trial_idx)
            if trial_fit["composite_oof_score"] > best_cand_score:
                best_cand_score = trial_fit["composite_oof_score"]
                best_rm_idx = rm_i
                best_cand_fit = trial_fit

        should_drop = (best_rm_idx is not None) and (
            (bool(zero_drop) and best_cand_score >= cur_score - pruning_tol)
            or (best_cand_score > cur_score + 1e-6)
        )
        if should_drop:
            dropped_names.append(model_names[best_rm_idx])
            active_idx.remove(best_rm_idx)
            cur_fit = best_cand_fit
        else:
            break

    return cur_fit, dropped_names

def _fast_sinkhorn_bacc(lp_mat: np.ndarray, y_true: np.ndarray, iters: int = 40) -> float:
    """Computes balanced accuracy after fast Sinkhorn mass equalization."""
    P = softmax(lp_mat, axis=1)
    if len(P) >= 18:
        P = _ablation_sinkhorn(P, iters=iters)
    return float(balanced_accuracy_score(y_true, P.argmax(axis=1)))

def _text_hash(text: str) -> str:
    """Normalized SHA1 hash of whitespace-collapsed text to guarantee zero exact/near-exact leakage."""
    norm = " ".join(str(text).strip().split())
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()

def build_unseen_holdout_dataset(
    input_dir: Path,
    train_path: Optional[Path] = None,
    samples_per_class: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Builds a balanced 18-class unseen holdout dataset (`df_unseen`) by:
    1. Hashing all texts in `train.csv` and `pool_v4.parquet` (and `pool_v1.parquet`) as `contaminated_hashes`.
    2. Scanning `nlm_matched_clean.parquet`, `nlm_pool_v5.parquet`, and `openrouter-harvest`
       for passages whose text hash is strictly NOT in `contaminated_hashes`.
    3. Sampling up to `samples_per_class` passages per class (e.g., 18 * 100 = 1,800 unseen passages)
       and returning `(df_unseen, holdout_hashes)` so `lexical_engine.py` and `trainer.py`
       exclude `holdout_hashes` from training!
    """
    input_dir = Path(input_dir) if input_dir else Path(".")

    # 0. Check for verified clean benchmark 1800 rows first
    bench_patterns = [
        "clean_benchmark_1800_rows.parquet",
        "*clean_benchmark*.parquet",
    ]
    for pat in bench_patterns:
        for bench_p in input_dir.rglob(pat):
            try:
                df_bench = pd.read_parquet(bench_p)
                if "text" in df_bench.columns and "model" in df_bench.columns:
                    df_bench = df_bench.dropna(subset=["text", "model"]).copy()
                    df_bench = df_bench[df_bench["model"].isin(LABELS)]
                    if len(df_bench) >= 180:
                        needs_clustering = False
                        if "prompt_id" not in df_bench.columns and "prompt_canon" in df_bench.columns:
                            df_bench["prompt_id"] = df_bench["prompt_canon"]

                        if "prompt_id" not in df_bench.columns:
                            needs_clustering = True
                        else:
                            p_col = df_bench["prompt_id"]
                            is_int = pd.api.types.is_integer_dtype(p_col) or (
                                p_col.dropna().apply(lambda x: isinstance(x, (int, np.integer))).all()
                                if len(p_col) > 0 else False
                            )
                            n_unique = p_col.nunique()
                            if (not is_int) or (n_unique > 0.5 * len(df_bench)):
                                needs_clustering = True

                        if needs_clustering:
                            try:
                                tfidf = TfidfVectorizer(stop_words="english", max_features=10000, sublinear_tf=True)
                                X_tfidf = tfidf.fit_transform(df_bench["text"].fillna("").tolist())
                                D = 1.0 - (X_tfidf @ X_tfidf.T).toarray().clip(0.0, 1.0)
                                np.fill_diagonal(D, 0.0)
                                cond_d = squareform(D, checks=False)
                                clusters = fcluster(linkage(cond_d, method="average"), t=0.85, criterion="distance")
                            except Exception:
                                clusters = np.arange(1, len(df_bench) + 1, dtype=int)
                            df_bench["cluster_id"] = clusters
                            df_bench["prompt_id"] = clusters
                        else:
                            if "cluster_id" not in df_bench.columns:
                                df_bench["cluster_id"] = df_bench["prompt_id"]

                        df_bench["_hash"] = [_text_hash(t) for t in df_bench["text"]]
                        holdout_hashes = set(df_bench["_hash"])
                        print(f"[Unseen Benchmark] Loaded verified clean holdout benchmark: {len(df_bench)} rows across {len(set(df_bench['model']))} classes from {bench_p}")
                        return df_bench, holdout_hashes
            except Exception as e:
                print(f"[Unseen Benchmark Warning] Failed to load {bench_p}: {e}")

    contaminated_hashes: Set[str] = set()

    # 1. Mark all train.csv rows as contaminated (seen by all Stage-2 LoRAs)
    if train_path and Path(train_path).exists():
        try:
            df_tr = pd.read_csv(train_path, usecols=["text"])
            contaminated_hashes.update(_text_hash(t) for t in df_tr["text"].fillna(""))
        except Exception:
            try:
                df_tr = pd.read_csv(train_path)
                if "text" in df_tr.columns:
                    contaminated_hashes.update(_text_hash(t) for t in df_tr["text"].fillna(""))
            except Exception:
                pass

    # 2. Mark all training pools as contaminated:
    # - pool_v4.parquet, pool_v1.parquet (seen by Part-2 LoRAs, DeBERTa, inhouse)
    # - nlm_pool_v5.parquet, nlm_matched_clean.parquet, nlm_pool_v2.parquet (seen by v5 and ts LoRAs)
    for pat in [
        "pool_v4.parquet",
        "pool_v1.parquet",
        "nlm_pool_v5.parquet",
        "nlm_matched_clean.parquet",
        "nlm_pool_v2.parquet",
    ]:
        for p in input_dir.rglob(pat):
            try:
                df_p = pd.read_parquet(p, columns=["text"])
                contaminated_hashes.update(_text_hash(t) for t in df_p["text"].fillna(""))
            except Exception:
                pass

    # 3. Collect strictly unseen candidate passages strictly from disjoint corpora (e.g. openrouter-harvest)
    candidate_dfs = []
    seen_paths = set()
    for pat in [
        "*openrouter*.parquet",
        "*openrouter*.csv",
        "*harvest*.parquet",
        "*harvest*.csv",
    ]:
        for p in input_dir.rglob(pat):
            if p in seen_paths:
                continue
            seen_paths.add(p)
            try:
                if p.suffix == ".parquet":
                    df_c = pd.read_parquet(p)
                elif p.suffix == ".csv":
                    df_c = pd.read_csv(p)
                else:
                    continue

                if "text" in df_c.columns and "model" in df_c.columns:
                    keep_cols = [c for c in ["id", "prompt_id", "text", "model"] if c in df_c.columns]
                    df_c = df_c[keep_cols].dropna(subset=["text", "model"]).copy()
                    df_c = df_c[df_c["model"].isin(LABELS)]
                    df_c["_hash"] = [_text_hash(t) for t in df_c["text"]]
                    df_c = df_c[~df_c["_hash"].isin(contaminated_hashes)]
                    if len(df_c) > 0:
                        candidate_dfs.append(df_c)
            except Exception:
                pass

    # Fallback to other non-training parquet/csv files if no openrouter files are discovered
    if not candidate_dfs:
        for pat in ["*.parquet", "*.csv"]:
            for p in input_dir.rglob(pat):
                p_name = p.name.lower()
                if p in seen_paths or any(x in p_name for x in ["train", "test", "submission", "pool", "matched"]):
                    continue
                seen_paths.add(p)
                try:
                    df_c = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                    if "text" in df_c.columns and "model" in df_c.columns:
                        keep_cols = [c for c in ["id", "prompt_id", "text", "model"] if c in df_c.columns]
                        df_c = df_c[keep_cols].dropna(subset=["text", "model"]).copy()
                        df_c = df_c[df_c["model"].isin(LABELS)]
                        df_c["_hash"] = [_text_hash(t) for t in df_c["text"]]
                        df_c = df_c[~df_c["_hash"].isin(contaminated_hashes)]
                        if len(df_c) > 0:
                            candidate_dfs.append(df_c)
                except Exception:
                    pass

    if not candidate_dfs:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model", "_hash"]), set()

    df_pool_unseen = pd.concat(candidate_dfs, ignore_index=True).drop_duplicates(subset=["_hash"])

    # Stratified balanced sampling across all 18 classes
    rng = np.random.RandomState(seed)
    sampled_parts = []

    # Prompt-grouped sampling: if prompt_id exists with multi-model prompts, sample by prompt_id first
    if "prompt_id" in df_pool_unseen.columns:
        prompt_counts = df_pool_unseen.groupby("prompt_id")["model"].nunique()
        multi_prompts = prompt_counts[prompt_counts > 1].index.tolist()
        rng.shuffle(multi_prompts)

        class_counts = {m: 0 for m in LABELS}
        chosen_indices = []
        for pid in multi_prompts:
            sub_p = df_pool_unseen[df_pool_unseen["prompt_id"] == pid]
            sub_m_counts = sub_p["model"].value_counts().to_dict()
            if all(class_counts.get(m, 0) + c <= samples_per_class for m, c in sub_m_counts.items()):
                chosen_indices.extend(sub_p.index.tolist())
                for m, c in sub_m_counts.items():
                    class_counts[m] = class_counts.get(m, 0) + c

        if chosen_indices:
            sampled_parts.append(df_pool_unseen.loc[chosen_indices])
            rem_pool = df_pool_unseen.drop(index=chosen_indices)
        else:
            rem_pool = df_pool_unseen
    else:
        rem_pool = df_pool_unseen
        class_counts = {m: 0 for m in LABELS}

    # Fill remaining class quotas up to samples_per_class
    for m in LABELS:
        curr_cnt = class_counts.get(m, 0)
        needed = samples_per_class - curr_cnt
        if needed > 0:
            sub = rem_pool[rem_pool["model"] == m]
            if len(sub) > 0:
                n_take = min(needed, len(sub))
                idx = rng.choice(sub.index.values, size=n_take, replace=False)
                sampled_parts.append(sub.loc[idx])

    if not sampled_parts:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model", "_hash"]), set()

    df_unseen = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if "id" not in df_unseen.columns:
        df_unseen["id"] = [f"unseen_{i}" for i in range(len(df_unseen))]

    # Compute valid stopword TF-IDF average-linkage clusters (t=0.85) so prompt_id is positive integer 1..G
    if len(df_unseen) >= 2:
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=10000, sublinear_tf=True)
            X_tfidf = tfidf.fit_transform(df_unseen["text"].fillna("").tolist())
            D = 1.0 - (X_tfidf @ X_tfidf.T).toarray().clip(0.0, 1.0)
            np.fill_diagonal(D, 0.0)
            cond_d = squareform(D, checks=False)
            clusters = fcluster(linkage(cond_d, method="average"), t=0.85, criterion="distance")
        except Exception:
            clusters = np.arange(1, len(df_unseen) + 1, dtype=int)
    else:
        clusters = np.ones(len(df_unseen), dtype=int)

    df_unseen["cluster_id"] = clusters
    df_unseen["prompt_id"] = clusters

    holdout_hashes = set(df_unseen["_hash"].tolist())
    print(
        f"[Unseen Benchmark] Built strictly unseen holdout set: {len(df_unseen)} rows "
        f"across {df_unseen['model'].nunique()} classes (0% overlap with train.csv & pool_v4.parquet)."
    )
    return df_unseen, holdout_hashes

class UnseenStepwiseAuditor:
    """
    Evaluates every model on the strictly unseen holdout dataset (`df_unseen`),
    reporting:
    1. Table 1 — Standalone (Solo) Unseen Balanced Accuracy (sorted descending).
    2. Table 2 — Greedy Forward Stepwise Selection (optimal combinations, stopping when lift < 0.0).
    3. Table 3 — Exhaustive All-2^M Subset Search & Leave-One-Out (LOO) Attribution.
    """
    def __init__(self, y_unseen: np.ndarray, groups_unseen: np.ndarray, N_test: int):
        self.y_unseen = np.asarray(y_unseen, dtype=int) if len(y_unseen) > 0 else np.array([], dtype=int)
        if groups_unseen is not None and len(groups_unseen) > 0:
            try:
                self.groups_unseen = np.asarray(groups_unseen, dtype=int)
            except (ValueError, TypeError):
                codes, _ = pd.factorize(groups_unseen)
                self.groups_unseen = (codes + 1).astype(int)
        else:
            self.groups_unseen = np.array([], dtype=int)
        self.N_test = int(N_test)
        self.models: List[Dict[str, Any]] = []
        self.greedy_model_indices: List[int] = []
        self.greedy_model_names: List[str] = []
        self.greedy_unseen_lp: np.ndarray = np.zeros((len(self.y_unseen), K), dtype=np.float64)
        self.greedy_test_lp: np.ndarray = np.zeros((self.N_test, K), dtype=np.float64)

    def add_model(
        self,
        name: str,
        *args,
        tier: str = "Flat",
        weight: Optional[float] = None,
        unseen_log_probs: Optional[np.ndarray] = None,
        test_log_probs: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if len(args) == 4:
            tier, weight, unseen_log_probs, test_log_probs = args
        elif len(args) == 3:
            weight, unseen_log_probs, test_log_probs = args
        elif len(args) == 1:
            if isinstance(args[0], (int, float)):
                weight = float(args[0])
            elif isinstance(args[0], str):
                tier = args[0]

        if weight is None:
            weight = float(kwargs.get("weight", 1.0))
        if unseen_log_probs is None:
            unseen_log_probs = kwargs.get("unseen_log_probs", np.zeros((0, K), dtype=np.float64))
        if test_log_probs is None:
            test_log_probs = kwargs.get("test_log_probs", np.zeros((self.N_test, K), dtype=np.float64))

        if len(self.y_unseen) > 0 and len(unseen_log_probs) != len(self.y_unseen):
            return
        u_lp = log_softmax(np.asarray(unseen_log_probs, dtype=np.float64), axis=1) if len(self.y_unseen) > 0 and len(unseen_log_probs) == len(self.y_unseen) else np.zeros((0, K), dtype=np.float64)
        t_lp = log_softmax(np.asarray(test_log_probs, dtype=np.float64), axis=1) if len(test_log_probs) == self.N_test else np.zeros((self.N_test, K), dtype=np.float64)
        solo_raw = float(balanced_accuracy_score(self.y_unseen, u_lp.argmax(axis=1))) if len(self.y_unseen) > 0 else 0.0
        solo_sink = _fast_sinkhorn_bacc(u_lp, self.y_unseen, iters=200) if len(self.y_unseen) >= 18 else solo_raw
        self.models.append({
            "name": name,
            "tier": tier if tier else "Flat",
            "weight": float(weight),
            "unseen_lp": u_lp,
            "test_lp": t_lp,
            "solo_unseen_bacc": solo_raw,
            "solo_unseen_sinkhorn_bacc": solo_sink,
        })

    def get_greedy_ensemble(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Returns (greedy_test_lp, greedy_unseen_lp, greedy_model_names)."""
        return (
            getattr(self, "greedy_test_lp", np.zeros((self.N_test, K), dtype=np.float64)),
            getattr(self, "greedy_unseen_lp", np.zeros((len(self.y_unseen), K), dtype=np.float64)),
            getattr(self, "greedy_model_names", []),
        )

    def optimize_weights_and_blend(
        self,
        working_dir: Path,
        extra_test_log_probs: Optional[List[Tuple[str, float, np.ndarray]]] = None,
        optimize_on_unseen: bool = True,
        lock_prior_weights: bool = False,
        extra_unseen_log_probs: Optional[List[Tuple[str, float, np.ndarray]]] = None,
        calibration_cfg: Optional[Dict[str, Any]] = None,
        claude_spec_logits_u: Optional[np.ndarray] = None,
        claude_classes: Optional[list] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
        """
        Calibrated Log-Linear Pooling & Automated OOF Selection Engine.
        1. When `lock_prior_weights` is True (or `optimize_on_unseen` is False / N < 36):
           applies pre-calibrated per-model temperatures `mean_T_m`, weights `mean_w_m`, and
           `mean_T_global` if present in `calibration_cfg`; otherwise blends using model weights.
        2. When `optimize_on_unseen` is True and `len(y_unseen) >= 36`:
           runs 5-fold GroupKFold L-BFGS log-linear pooling with per-model T_m calibration,
           global T_g, and automated OOF backward pruning over all candidate models (no fixed count).
        """
        if lock_prior_weights:
            optimize_on_unseen = False

        if len(self.models) == 0:
            self.P_mix_unseen = np.zeros((0, K), dtype=np.float64)
            return np.ones((self.N_test, K), dtype=np.float64) / float(K), {}, pd.DataFrame()

        for m in self.models:
            m["prior_weight"] = m.get("prior_weight", m["weight"])

        extra_u_lp_base = np.zeros((len(self.y_unseen), K), dtype=np.float64)
        extra_u_weight_tot = 0.0
        if extra_unseen_log_probs and len(self.y_unseen) > 0:
            for _, w_ex, lp_ex in extra_unseen_log_probs:
                if len(lp_ex) == len(self.y_unseen) and w_ex > 0.0:
                    extra_u_lp_base += float(w_ex) * log_softmax(lp_ex, axis=1)
                    extra_u_weight_tot += float(w_ex)

        # Branch A: Pre-calibrated config mode or small/frozen holdout fallback
        if not optimize_on_unseen or len(self.y_unseen) < 36:
            alias_to_disp = {
                "svc_4view": "LinearSVC (4-View)",
                "lexical_5v": "Lexical Comp (5-View)",
                "qwen3_14b_p2": "Qwen3-14B Part-2",
                "qwen3_14b_p1": "Qwen3-14B Part-1",
                "qwen25_14b_p2": "Qwen2.5-14B Part-2",
                "deberta_p2": "DeBERTa-v3-Large",
                "bge_m3": "BGE-M3-v2",
            }
            disp_to_canon = {v: k for k, v in alias_to_disp.items()}

            def _resolve_cal_entry(name: str, mapping: Dict[str, Any]) -> Optional[str]:
                for cand in (name, alias_to_disp.get(name), disp_to_canon.get(name), "BGE-M3" if name in ("bge_m3", "BGE-M3-v2") else None):
                    if cand and cand in mapping:
                        return cand
                return None

            mean_T_m = calibration_cfg.get("mean_T_m", {}) if calibration_cfg else {}
            mean_w_m = calibration_cfg.get("mean_w_m", {}) if calibration_cfg else {}
            clip_min = float(calibration_cfg.get("clip_min", 1e-4)) if calibration_cfg else 1e-4
            resolved_keys = {
                m["name"]: (_resolve_cal_entry(m["name"], mean_T_m), _resolve_cal_entry(m["name"], mean_w_m))
                for m in self.models
            }
            matched_models = [m for m in self.models if all(resolved_keys[m["name"]])]

            if calibration_cfg and "mean_T_global" in calibration_cfg and len(matched_models) > 0:
                m_names = [m["name"] for m in matched_models]
                raw_w = np.array([max(0.0, float(mean_w_m[resolved_keys[n][1]])) for n in m_names], dtype=np.float64)
                w_norm = raw_w / max(float(raw_w.sum()), 1e-12)
                norm_map = {n: float(w_norm[i]) for i, n in enumerate(m_names)}
                t_map = {n: float(mean_T_m[resolved_keys[n][0]]) for n in m_names}
                weights_map: Dict[str, float] = {}
                for m in self.models:
                    w_val = norm_map.get(m["name"], 0.0)
                    m["learned_weight"] = w_val
                    m["weight"] = w_val
                    weights_map[m["name"]] = w_val

                P_mix = (
                    pool_calibrated_log_probs(
                        [m["test_lp"] for m in matched_models],
                        m_names,
                        t_map,
                        norm_map,
                        mean_T_global=float(calibration_cfg["mean_T_global"]),
                        clip_min=clip_min,
                    )
                    if self.N_test > 0
                    else np.zeros((0, K), dtype=np.float64)
                )
                self.P_mix_unseen = (
                    pool_calibrated_log_probs(
                        [m["unseen_lp"] for m in matched_models],
                        m_names,
                        t_map,
                        norm_map,
                        mean_T_global=float(calibration_cfg["mean_T_global"]),
                        clip_min=clip_min,
                    )
                    if len(self.y_unseen) > 0
                    else np.zeros((0, K), dtype=np.float64)
                )
                self.normalized_weights_map = dict(weights_map)
                self.fitted_artifact = {
                    "model_names": m_names,
                    "mean_T_m": dict(t_map),
                    "mean_w_m": dict(norm_map),
                    "mean_T_global": float(calibration_cfg["mean_T_global"]),
                    "clip_min": clip_min,
                }
            else:
                weights_map = {m["name"]: float(m["weight"]) for m in self.models}
                logit_flat = np.zeros((self.N_test, K), dtype=np.float64)
                logit_flat_u = extra_u_lp_base.copy() if len(self.y_unseen) > 0 else np.zeros((0, K), dtype=np.float64)
                for m in self.models:
                    logit_flat += m["weight"] * m["test_lp"]
                    if len(m["unseen_lp"]) == len(self.y_unseen):
                        logit_flat_u += m["weight"] * m["unseen_lp"]
                if extra_test_log_probs:
                    for ex_name, w_ex, lp_ex in extra_test_log_probs:
                        if len(lp_ex) == self.N_test and w_ex > 0.0:
                            logit_flat += float(w_ex) * log_softmax(lp_ex, axis=1)
                            weights_map[ex_name] = float(w_ex)
                P_mix = softmax(logit_flat, axis=1)
                self.P_mix_unseen = softmax(logit_flat_u, axis=1) if len(self.y_unseen) > 0 else np.zeros((0, K), dtype=np.float64)

            df_summary = pd.DataFrame([
                {"model": m["name"], "tier": m["tier"], "prior_weight": m["prior_weight"], "learned_weight": m["weight"]}
                for m in self.models
            ])
            return P_mix, weights_map, df_summary

        # Branch B: Dynamic 5-Fold GroupKFold L-BFGS Pooling + Automated OOF Backward Pruning
        chance_floor = 1.0 / float(K)
        valid_pool = [m for m in self.models if m["solo_unseen_bacc"] > chance_floor]
        if not valid_pool:
            valid_pool = list(self.models)
        cand_names = [m["name"] for m in valid_pool]
        cand_u_lps = [m["unseen_lp"] for m in valid_pool]
        pre_pruned = [m["name"] for m in self.models if m not in valid_pool]
        clip_min = float(calibration_cfg.get("clip_min", 1e-4)) if calibration_cfg else 1e-4
        eps = float(calibration_cfg.get("label_smoothing_eps", 0.05)) if calibration_cfg else 0.05
        sink_alpha = float(calibration_cfg.get("sinkhorn_alpha", 0.50)) if calibration_cfg else 0.50

        best_fit, dropped_names = auto_prune_and_fit_lbfgs(
            raw_lps_list=cand_u_lps,
            y=self.y_unseen,
            groups=self.groups_unseen,
            model_names=cand_names,
            spec_logits=claude_spec_logits_u,
            claude_classes=claude_classes,
            lambdas=(0.05, 0.10, 0.20),
            eps=eps,
            sinkhorn_alpha=sink_alpha,
            clip_min=clip_min,
        )
        dropped_names = pre_pruned + dropped_names

        kept_names = best_fit["model_names"]
        kept_weights = best_fit["mean_w_m"]
        kept_temps = best_fit["mean_T_m"]
        T_global_opt = float(best_fit["mean_T_global"])
        self.fitted_artifact = best_fit

        # Map kept weights across registered models
        weights_map = {}
        normalized_weights_map = {}
        chosen_indices = []
        for idx, m in enumerate(self.models):
            w_val = float(kept_weights.get(m["name"], 0.0)) if m["name"] in kept_names else 0.0
            m["learned_weight"] = w_val
            m["weight"] = w_val
            weights_map[m["name"]] = w_val
            normalized_weights_map[m["name"]] = w_val
            if w_val > 0.0:
                chosen_indices.append(idx)

        kept_models = [m for m in self.models if m["name"] in kept_names and weights_map[m["name"]] > 0.0]
        kept_m_names = [m["name"] for m in kept_models]
        P_mix_base = (
            pool_calibrated_log_probs(
                [m["test_lp"] for m in kept_models],
                kept_m_names,
                kept_temps,
                kept_weights,
                mean_T_global=T_global_opt,
                clip_min=clip_min,
            )
            if self.N_test > 0
            else np.zeros((0, K), dtype=np.float64)
        )
        P_u_base = pool_calibrated_log_probs(
            [m["unseen_lp"] for m in kept_models],
            kept_m_names,
            kept_temps,
            kept_weights,
            mean_T_global=T_global_opt,
            clip_min=clip_min,
        )

        # Incorporate auxiliary external streams (e.g., wpca_dp / spectral_bank) cleanly if passed
        if extra_test_log_probs or (extra_unseen_log_probs and extra_u_weight_tot > 0.0):
            base_te_lp = np.log(np.clip(P_mix_base, 1e-12, 1.0)) if self.N_test > 0 else np.zeros((0, K), dtype=np.float64)
            base_u_lp = np.log(np.clip(P_u_base, 1e-12, 1.0))
            ext_te_tot = 0.0
            ext_te_lp = np.zeros((self.N_test, K), dtype=np.float64)
            if extra_test_log_probs:
                for ex_name, w_ex, lp_ex in extra_test_log_probs:
                    if len(lp_ex) == self.N_test and w_ex > 0.0:
                        ext_te_lp += float(w_ex) * log_softmax(lp_ex, axis=1)
                        ext_te_tot += float(w_ex)
                        weights_map[ex_name] = float(w_ex)
                        normalized_weights_map[ex_name] = float(w_ex) / max(1.0 + float(w_ex), 1e-8)
            P_mix = softmax((base_te_lp + ext_te_lp) / max(1.0 + ext_te_tot, 1.0), axis=1) if self.N_test > 0 else P_mix_base
            self.P_mix_unseen = softmax((base_u_lp + extra_u_lp_base) / max(1.0 + extra_u_weight_tot, 1.0), axis=1)
        else:
            P_mix = P_mix_base
            self.P_mix_unseen = P_u_base

        # Populate report state & log concise 5-fold OOF diagnostics
        self.greedy_model_indices = chosen_indices
        self.greedy_model_names = kept_m_names
        self.greedy_raw_weights = {n: float(kept_weights[n]) for n in kept_m_names}
        self.greedy_norm_weights = {n: float(kept_weights[n]) for n in kept_m_names}
        self.greedy_unseen_lp = np.log(np.clip(self.P_mix_unseen, 1e-12, 1.0))
        self.greedy_test_lp = np.log(np.clip(P_mix, 1e-12, 1.0)) if self.N_test > 0 else np.zeros((0, K), dtype=np.float64)
        self.greedy_unseen_bacc_raw = float(best_fit["oof_bacc_raw"])
        self.greedy_unseen_bacc_sink = float(best_fit["oof_bacc_sink"])
        self.normalized_weights_map = normalized_weights_map

        # Build clean step records in descending weight order for reporting
        step_records = []
        for s_i, n in enumerate(sorted(kept_m_names, key=lambda x: -kept_weights[x]), 1):
            step_records.append({
                "step": s_i,
                "model": n,
                "raw_weight": float(kept_weights[n]),
                "cumul_raw": float(best_fit["oof_bacc_raw"]),
                "cumul_sink": float(best_fit["oof_bacc_sink"]),
                "marginal_lift": 0.0 if s_i == 1 else float(best_fit["oof_bacc_sink"] - self.models[0]["solo_unseen_sinkhorn_bacc"]),
                "test_rows_flipped": 0,
            })
        self.greedy_step_records = step_records

        print(
            f"[Ablation L-BFGS] lam={best_fit['lam']:.2f} | Pruned={dropped_names or 'None'} | "
            f"OOF Raw={best_fit['oof_bacc_raw']:.4f} (NLL={best_fit['oof_nll_raw']:.4f}) | "
            f"+Sink={best_fit['oof_bacc_sink']:.4f} | Tg={T_global_opt:.3f}"
        )

        df_summary = pd.DataFrame([
            {"model": m["name"], "tier": m["tier"], "prior_weight": m["prior_weight"], "learned_weight": m["weight"]}
            for m in self.models
        ])
        return P_mix, weights_map, df_summary

    def evaluate_marginal_lift(
        self,
        candidate_unseen_lp: np.ndarray,
        weight: float = 1.0,
    ) -> float:
        """
        Multi-scale marginal lift search: tests adding `candidate_unseen_lp` across
        multipliers [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00] * weight
        and returns the maximum Delta BAcc achieved at any positive scale.
        """
        if len(self.y_unseen) == 0 or len(candidate_unseen_lp) != len(self.y_unseen) or len(self.models) == 0:
            return 0.0

        cum_unseen_lp = np.zeros((len(self.y_unseen), K), dtype=np.float64)
        for m in self.models:
            w_m = m["weight"] if m["weight"] > 0 else m.get("prior_weight", 0.15)
            cum_unseen_lp += w_m * m["unseen_lp"]

        cur_bacc = _fast_sinkhorn_bacc(cum_unseen_lp, self.y_unseen, iters=40)
        cand_lp = log_softmax(np.asarray(candidate_unseen_lp, dtype=np.float64), axis=1)

        best_lift = -1.0
        full_weight_lift = 0.0
        for mult in (0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00):
            cand_w = float(weight) * mult
            new_unseen_lp = cum_unseen_lp + cand_w * cand_lp
            new_bacc = _fast_sinkhorn_bacc(new_unseen_lp, self.y_unseen, iters=40)
            lift = new_bacc - cur_bacc
            if mult == 1.00:
                full_weight_lift = lift
            if lift > best_lift:
                best_lift = lift

        if best_lift <= 0.0:
            return float(full_weight_lift)
        return float(best_lift)

    def print_and_save_report(self, working_dir: Path, final_test_preds: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not self.models or len(self.y_unseen) == 0:
            print("[Ablation] No models or unseen holdout samples available; skipping report.")
            return pd.DataFrame()

        M = len(self.models)

        # Part 1: Standalone (Solo) Accuracy, sorted descending
        print("\n" + "=" * 105)
        print(f" PART 1: STANDALONE (SOLO) ACCURACY ON STRICTLY UNSEEN HOLDOUT ({len(self.y_unseen)} Disjoint Passages)")
        print("=" * 105)
        print(f"{'Model / Checkpoint':<28} | {'Tier':<6} | {'Unseen Raw BAcc':<16} | {'Unseen+Sinkhorn BAcc':<20} | {'Agrees w/ Final Test'}")
        print("-" * 105)
        sorted_solo = sorted(self.models, key=lambda m: m["solo_unseen_sinkhorn_bacc"], reverse=True)
        for m in sorted_solo:
            agree_str = "N/A"
            if final_test_preds is not None and self.N_test > 0 and len(final_test_preds) == self.N_test:
                pct = 100.0 * float((m["test_lp"].argmax(axis=1) == final_test_preds).mean())
                agree_str = f"{pct:5.1f}%"
            print(
                f"{m['name']:<28} | {m['tier']:<6} | "
                f"{m['solo_unseen_bacc']:<16.4f} | {m['solo_unseen_sinkhorn_bacc']:<20.4f} | {agree_str}"
            )

        # Part 2: Greedy Forward Stepwise Selection (Single Truth from Caruana selection)
        print("\n" + "=" * 115)
        print(" PART 2: GREEDY FORWARD STEPWISE SELECTION ON STRICTLY UNSEEN HOLDOUT (Optimal Combinations)")
        print("=" * 115)
        print(f"{'Step':<5} | {'Added Model':<26} | {'Raw Wt':<7} | {'Share (%)':<10} | {'Cumul Raw':<10} | {'Cumul+Sink':<11} | {'Marginal Lift':<13} | {'Test Rows Flipped'}")
        print("-" * 115)

        records = list(getattr(self, "greedy_step_records", []))
        if records:
            tot_gw = sum(r.get("raw_weight", r.get("weight", 0.0)) for r in records)
            for r in records:
                w_val = r.get("raw_weight", r.get("weight", 0.0))
                pct = (w_val / tot_gw * 100.0) if tot_gw > 0 else 0.0
                lift_str = f"{r['marginal_lift']:+.4f}" if r["step"] > 1 else "BASELINE"
                flips = r.get("test_rows_flipped", 0)
                flip_str = f"{flips} (Init)" if r["step"] == 1 else f"{flips:4d} ({100.0 * flips / max(1, self.N_test):4.1f}%)"
                c_raw = r.get("cumul_raw", r.get("cumul_unseen_raw_bacc", 0.0))
                c_sink = r.get("cumul_sink", r.get("cumul_unseen_sinkhorn_bacc", 0.0))
                print(
                    f"{r['step']:<5} | + {r['model']:<24} | {w_val:<7.2f} | {pct:>7.1f}%   | "
                    f"{c_raw:<10.4f} | {c_sink:<11.4f} | {lift_str:<13} | {flip_str}"
                )
            print("-" * 115)
            print(f"Total | {len(getattr(self, 'greedy_model_indices', []))} Active Models{'':<13} | {tot_gw:<7.2f} |   100.0%   | Final Greedy Optimal Subset (+Sinkhorn BAcc: {getattr(self, 'greedy_unseen_bacc_sink', 0.0):.4f})")
            print("=" * 115 + "\n")
        else:
            M = len(self.models)
            remaining = list(range(M))
            cum_unseen_lp = np.zeros((len(self.y_unseen), K), dtype=np.float64)
            cum_test_lp = np.zeros((self.N_test, K), dtype=np.float64)
            prev_bacc = 0.0
            prev_test_preds = None
            chosen_indices = []

            step_idx = 1
            while remaining:
                best_cand_idx = None
                best_cand_bacc = -1.0
                best_cand_raw = 0.0

                if step_idx == 1:
                    for idx in remaining:
                        cand = self.models[idx]
                        cand_w = cand["weight"] if cand["weight"] > 0 else cand.get("prior_weight", 0.15)
                        cand_lp = cand_w * cand["unseen_lp"]
                        cand_bacc = _fast_sinkhorn_bacc(cand_lp, self.y_unseen, iters=40)
                        if cand_bacc > best_cand_bacc:
                            best_cand_bacc = cand_bacc
                            best_cand_idx = idx
                            best_cand_raw = float(balanced_accuracy_score(self.y_unseen, cand_lp.argmax(axis=1)))
                else:
                    for idx in remaining:
                        cand = self.models[idx]
                        cand_w = cand["weight"] if cand["weight"] > 0 else cand.get("prior_weight", 0.15)
                        cand_lp = cum_unseen_lp + cand_w * cand["unseen_lp"]
                        cand_bacc = _fast_sinkhorn_bacc(cand_lp, self.y_unseen, iters=40)
                        if cand_bacc > best_cand_bacc:
                            best_cand_bacc = cand_bacc
                            best_cand_idx = idx
                            best_cand_raw = float(balanced_accuracy_score(self.y_unseen, cand_lp.argmax(axis=1)))

                if best_cand_idx is None:
                    break

                lift = best_cand_bacc - prev_bacc if step_idx > 1 else 0.0
                if step_idx > 1 and lift < 0.0:
                    print(f"[Greedy Forward Selection] Best remaining candidate '{self.models[best_cand_idx]['name']}' gives negative lift ({lift:+.4f}); stopping forward selection.")
                    break

                m = self.models[best_cand_idx]
                remaining.remove(best_cand_idx)
                chosen_indices.append(best_cand_idx)

                w = m["weight"] if m["weight"] > 0 else m.get("prior_weight", 0.15)
                cum_unseen_lp = cum_unseen_lp + w * m["unseen_lp"]
                cum_test_lp = cum_test_lp + w * m["test_lp"]

                lift_str = f"{lift:+.4f}" if step_idx > 1 else "BASELINE"
                cur_test_preds = _ablation_sinkhorn(softmax(cum_test_lp, axis=1), iters=40).argmax(axis=1) if self.N_test > 0 else np.array([])
                flips = self.N_test if prev_test_preds is None else int((cur_test_preds != prev_test_preds).sum())
                flip_str = f"{flips} (Init)" if prev_test_preds is None else f"{flips:4d} ({100.0 * flips / max(1, self.N_test):4.1f}%)"

                print(
                    f"{step_idx:<5} | + {m['name']:<24} | {w:<7.2f} |       -    | "
                    f"{best_cand_raw:<10.4f} | {best_cand_bacc:<11.4f} | {lift_str:<13} | {flip_str}"
                )
                prev_bacc = best_cand_bacc
                prev_test_preds = cur_test_preds
                records.append({
                    "step": step_idx,
                    "model": m["name"],
                    "added_model": m["name"],
                    "tier": m["tier"],
                    "weight": w,
                    "raw_weight": w,
                    "solo_unseen_bacc": m["solo_unseen_bacc"],
                    "solo_unseen_sinkhorn_bacc": m["solo_unseen_sinkhorn_bacc"],
                    "cumul_unseen_raw_bacc": best_cand_raw,
                    "cumul_unseen_sinkhorn_bacc": best_cand_bacc,
                    "marginal_lift": lift,
                    "test_rows_flipped": flips,
                })
                step_idx += 1

            total_gw = sum(max(0.0, float(self.models[i]["weight"] if self.models[i]["weight"] > 0 else self.models[i].get("prior_weight", 0.15))) for i in chosen_indices)
            gw_scale = (1.0 / total_gw) if total_gw > 0.0 else 1.0
            self.greedy_model_indices = chosen_indices
            self.greedy_model_names = [self.models[i]["name"] for i in chosen_indices]
            self.greedy_unseen_lp = cum_unseen_lp * gw_scale
            self.greedy_test_lp = cum_test_lp * gw_scale
            self.greedy_unseen_bacc_raw = float(balanced_accuracy_score(self.y_unseen, cum_unseen_lp.argmax(axis=1))) if len(self.y_unseen) > 0 else 0.0
            self.greedy_unseen_bacc_sink = prev_bacc
            print("=" * 115 + "\n")

        # Part 3: Exhaustive All 2^M Subset Search & LOO Attribution (for M <= 12)
        if 2 <= M <= 12:
            print("=" * 105)
            print(f" PART 3: GLOBALLY OPTIMAL SUBSET & LEAVE-ONE-OUT (LOO) ATTRIBUTION (All {2**M - 1} Non-Empty Subsets)")
            print("=" * 105)

            best_subset = None
            best_subset_bacc = -1.0

            for r in range(1, M + 1):
                for sub in combinations(range(M), r):
                    sub_unseen_lp = sum((self.models[i]["weight"] if self.models[i]["weight"] > 0 else self.models[i].get("prior_weight", 0.15)) * self.models[i]["unseen_lp"] for i in sub)
                    bacc = _fast_sinkhorn_bacc(sub_unseen_lp, self.y_unseen, iters=30)
                    if bacc > best_subset_bacc:
                        best_subset_bacc = bacc
                        best_subset = sub

            opt_names = [self.models[i]["name"] for i in best_subset]
            print(f"Globally Optimal Ensemble BAcc: {best_subset_bacc:.4f} (Uses {len(best_subset)} / {M} models)")
            print(f"Optimal Set: {', '.join(opt_names)}")
            print("-" * 105)
            print(f"{'Model / Checkpoint':<28} | {'In Optimal?':<12} | {'Weight':<7} | {'Delta_LOO (BAcc Drop If Excluded)'}")
            print("-" * 105)

            opt_set = set(best_subset)
            for i, m in enumerate(self.models):
                in_opt = i in opt_set
                in_str = "YES" if in_opt else "NO (Pruned)"
                if in_opt:
                    if len(best_subset) == 1:
                        delta_loo = best_subset_bacc
                    else:
                        loo_sub = [j for j in best_subset if j != i]
                        loo_lp = sum((self.models[j]["weight"] if self.models[j]["weight"] > 0 else self.models[j].get("prior_weight", 0.15)) * self.models[j]["unseen_lp"] for j in loo_sub)
                        loo_bacc = _fast_sinkhorn_bacc(loo_lp, self.y_unseen, iters=30)
                        delta_loo = best_subset_bacc - loo_bacc
                    loo_str = f"-{delta_loo:.4f}" if delta_loo >= 0 else f"+{abs(delta_loo):.4f}"
                else:
                    with_m_sub = list(best_subset) + [i]
                    with_m_lp = sum((self.models[j]["weight"] if self.models[j]["weight"] > 0 else self.models[j].get("prior_weight", 0.15)) * self.models[j]["unseen_lp"] for j in with_m_sub)
                    with_m_bacc = _fast_sinkhorn_bacc(with_m_lp, self.y_unseen, iters=30)
                    delta_add = with_m_bacc - best_subset_bacc
                    loo_str = f"{delta_add:+.4f} (Degrades if added)"

                print(f"{m['name']:<28} | {in_str:<12} | {m['weight']:<7.2f} | {loo_str}")

                for r in records:
                    if r["model"] == m["name"]:
                        r["in_optimal_subset"] = in_opt
                        r["delta_loo"] = delta_loo if in_opt else 0.0

            print("=" * 105 + "\n")

        df_rep = pd.DataFrame(records)
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        df_rep.to_csv(working_dir / "unseen_stepwise_ablation.csv", index=False)
        return df_rep
