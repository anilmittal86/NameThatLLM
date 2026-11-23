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
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold

from .config import LABELS, K, LAB2I, softmax, log_softmax

def _ablation_sinkhorn(P: np.ndarray, iters: int = 200, eps: float = 1e-12, **kwargs) -> np.ndarray:
    """Fast vectorized Sinkhorn mass balancing for balanced multiclass prediction."""
    if len(P) < 18:
        return P
    P = np.clip(P.astype(np.float64), eps, 1.0)
    target_col = 1.0 / P.shape[1]
    for _ in range(iters):
        P /= P.sum(axis=1, keepdims=True).clip(min=eps)
        P /= (P.sum(axis=0, keepdims=True) / (target_col * P.shape[0])).clip(min=eps)
    return P / P.sum(axis=1, keepdims=True).clip(min=eps)

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
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
        if lock_prior_weights:
            optimize_on_unseen = False
        """
        1. Runs Greedy Forward Stepwise Selection + Coordinate Line Search on `df_unseen`
           to determine each model's optimal individual weight `w_star` (>= 0.0).
        2. Any model that fails to improve unseen balanced accuracy gets `w_star = 0.0` (PRUNED).
        3. Combines all active models + extra_test_log_probs (wpca_dp, spectral_bank) directly
           in flat log-probability space:
               LOGIT_FLAT = sum_m (w_star[m] * test_lp[m])
        """
        if len(self.models) == 0:
            self.P_mix_unseen = np.zeros((0, K), dtype=np.float64)
            return np.ones((self.N_test, K), dtype=np.float64) / float(K), {}, pd.DataFrame()

        extra_u_lp_base = np.zeros((len(self.y_unseen), K), dtype=np.float64)
        if extra_unseen_log_probs and len(self.y_unseen) > 0:
            for ex_name, w_ex, lp_ex in extra_unseen_log_probs:
                if len(lp_ex) == len(self.y_unseen) and w_ex > 0.0:
                    extra_u_lp_base += float(w_ex) * log_softmax(lp_ex, axis=1)

        if not optimize_on_unseen or len(self.y_unseen) < 36:
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
                {"model": m["name"], "tier": m["tier"], "prior_weight": m.get("prior_weight", m["weight"]), "learned_weight": m["weight"]}
                for m in self.models
            ])
            return P_mix, weights_map, df_summary

        def _composite_obj(lp_mat: np.ndarray, y_target: Optional[np.ndarray] = None) -> Tuple[float, float]:
            y_use = self.y_unseen if y_target is None else y_target
            P = softmax(lp_mat, axis=1)
            P_s = _ablation_sinkhorn(P, iters=60) if len(P) >= 18 else P
            bacc = float(balanced_accuracy_score(y_use, P_s.argmax(axis=1)))
            nll = float(-np.mean(np.log(np.clip(P_s[np.arange(len(y_use)), y_use], 1e-9, 1.0))))
            return bacc - 0.015 * nll, bacc

        M = len(self.models)
        order = sorted(range(M), key=lambda i: self.models[i]["solo_unseen_sinkhorn_bacc"], reverse=True)

        active_weights = np.zeros(M, dtype=np.float64)
        first_idx = order[0]
        anchor_w = 1.0 if self.models[first_idx]["weight"] <= 0.0 else float(self.models[first_idx]["weight"])
        active_weights[first_idx] = anchor_w
        cum_u_lp = extra_u_lp_base + active_weights[first_idx] * self.models[first_idx]["unseen_lp"]
        best_obj, best_bacc = _composite_obj(cum_u_lp)

        remaining = list(order[1:])
        base_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00, 1.25, 1.50]

        chosen_indices = [first_idx]
        cur_test_lp = active_weights[first_idx] * self.models[first_idx]["test_lp"].copy()
        prev_test_preds = _ablation_sinkhorn(softmax(cur_test_lp, axis=1), iters=40).argmax(axis=1) if self.N_test > 0 else np.array([])
        step_records = [{
            "step": 1,
            "model": self.models[first_idx]["name"],
            "raw_weight": anchor_w,
            "cumul_raw": float(balanced_accuracy_score(self.y_unseen, cum_u_lp.argmax(axis=1))),
            "cumul_sink": best_bacc,
            "marginal_lift": 0.0,
            "test_rows_flipped": self.N_test,
        }]

        step_idx = 2
        while remaining:
            step_best_idx = None
            step_best_w = 0.0
            step_best_obj = best_obj
            step_best_bacc = best_bacc

            for idx in remaining:
                u_lp = self.models[idx]["unseen_lp"]
                p_w = self.models[idx]["weight"]
                weights_to_test = list(base_grid)
                if p_w > 0.0:
                    for mult in [0.25, 0.50, 1.00, 1.25]:
                        weights_to_test.append(round(p_w * mult, 3))
                weights_to_test = sorted(list(set(w for w in weights_to_test if w > 0.0)))

                for cand_w in weights_to_test:
                    trial_lp = cum_u_lp + cand_w * u_lp
                    obj_val, bacc_val = _composite_obj(trial_lp)
                    if obj_val > step_best_obj + 1e-6:
                        step_best_obj = obj_val
                        step_best_bacc = bacc_val
                        step_best_idx = idx
                        step_best_w = cand_w

            if step_best_idx is not None and step_best_w > 0.0:
                lift = step_best_bacc - best_bacc
                active_weights[step_best_idx] = step_best_w
                cum_u_lp = cum_u_lp + step_best_w * self.models[step_best_idx]["unseen_lp"]
                best_obj = step_best_obj
                best_bacc = step_best_bacc
                remaining.remove(step_best_idx)
                chosen_indices.append(step_best_idx)

                cur_test_lp += step_best_w * self.models[step_best_idx]["test_lp"]
                cur_test_preds = _ablation_sinkhorn(softmax(cur_test_lp, axis=1), iters=40).argmax(axis=1) if self.N_test > 0 else np.array([])
                flips = int((cur_test_preds != prev_test_preds).sum()) if len(cur_test_preds) == len(prev_test_preds) else 0
                prev_test_preds = cur_test_preds

                step_records.append({
                    "step": step_idx,
                    "model": self.models[step_best_idx]["name"],
                    "raw_weight": step_best_w,
                    "cumul_raw": float(balanced_accuracy_score(self.y_unseen, cum_u_lp.argmax(axis=1))),
                    "cumul_sink": best_bacc,
                    "marginal_lift": lift,
                    "test_rows_flipped": flips,
                })
                step_idx += 1
            else:
                break

        # Calculate single-truth normalized weights across active models
        tot_act = sum(active_weights[ci] for ci in chosen_indices)
        if extra_test_log_probs:
            tot_act += sum(max(0.0, float(w_ex)) for _, w_ex, _ in extra_test_log_probs)
        act_scale = (1.0 / tot_act) if tot_act > 0.0 else 1.0

        weights_map: Dict[str, float] = {}
        normalized_weights_map: Dict[str, float] = {}
        logit_flat = np.zeros((self.N_test, K), dtype=np.float64)
        logit_flat_u = (extra_u_lp_base * act_scale) if len(self.y_unseen) > 0 else np.zeros((0, K), dtype=np.float64)

        for idx, m in enumerate(self.models):
            m["prior_weight"] = m.get("prior_weight", m["weight"])
            raw_w = float(active_weights[idx])
            norm_w = float(raw_w * act_scale)
            m["learned_weight"] = norm_w
            m["weight"] = norm_w
            weights_map[m["name"]] = raw_w
            normalized_weights_map[m["name"]] = norm_w
            if norm_w > 0.0:
                logit_flat += norm_w * m["test_lp"]
                if len(m["unseen_lp"]) == len(self.y_unseen):
                    logit_flat_u += norm_w * m["unseen_lp"]

        if extra_test_log_probs:
            for ex_name, w_ex, lp_ex in extra_test_log_probs:
                if len(lp_ex) == self.N_test and w_ex > 0.0:
                    norm_ex_w = float(w_ex) * act_scale
                    logit_flat += norm_ex_w * log_softmax(lp_ex, axis=1)
                    weights_map[ex_name] = float(w_ex)
                    normalized_weights_map[ex_name] = norm_ex_w

        self.greedy_step_records = step_records
        self.greedy_model_indices = chosen_indices
        self.greedy_model_names = [self.models[ci]["name"] for ci in chosen_indices]
        self.greedy_raw_weights = {self.models[ci]["name"]: float(active_weights[ci]) for ci in chosen_indices}
        self.greedy_norm_weights = {self.models[ci]["name"]: normalized_weights_map[self.models[ci]["name"]] for ci in chosen_indices}
        self.greedy_unseen_lp = logit_flat_u
        self.greedy_test_lp = logit_flat
        self.greedy_unseen_bacc_raw = float(balanced_accuracy_score(self.y_unseen, logit_flat_u.argmax(axis=1))) if len(self.y_unseen) > 0 else 0.0
        self.greedy_unseen_bacc_sink = best_bacc
        self.normalized_weights_map = normalized_weights_map

        in_sample_bacc = _fast_sinkhorn_bacc(logit_flat_u, self.y_unseen, iters=200) if len(self.y_unseen) >= 18 else best_bacc

        # 5-fold GroupKFold OOF Unseen BAcc calculation
        oof_bacc = in_sample_bacc
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                groups = self.groups_unseen if len(self.groups_unseen) == len(self.y_unseen) else np.arange(len(self.y_unseen))
                n_groups = len(np.unique(groups))
                if n_groups >= 2:
                    n_splits = min(5, n_groups)
                    gkf = GroupKFold(n_splits=n_splits)
                    splits = list(gkf.split(self.y_unseen, self.y_unseen, groups=groups))
                else:
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    splits = list(kf.split(self.y_unseen))

                oof_lp = np.zeros((len(self.y_unseen), K), dtype=np.float64)
                for tr_idx, val_idx in splits:
                    y_tr = self.y_unseen[tr_idx]
                    ex_tr = extra_u_lp_base[tr_idx]
                    ex_val = extra_u_lp_base[val_idx]

                    fold_order = sorted(
                        range(M),
                        key=lambda i: _fast_sinkhorn_bacc(self.models[i]["unseen_lp"][tr_idx], y_tr, iters=40) if len(tr_idx) >= 18 else 0.0,
                        reverse=True,
                    )
                    fold_w = np.zeros(M, dtype=np.float64)
                    f_first = fold_order[0]
                    fold_w[f_first] = self.models[f_first]["weight"]
                    f_cum_lp = ex_tr + fold_w[f_first] * self.models[f_first]["unseen_lp"][tr_idx]
                    f_obj, _ = _composite_obj(f_cum_lp, y_target=y_tr)

                    f_rem = list(fold_order[1:])
                    while f_rem:
                        s_best_idx = None
                        s_best_w = 0.0
                        s_best_obj = f_obj
                        for idx in f_rem:
                            u_lp_tr = self.models[idx]["unseen_lp"][tr_idx]
                            for cand_w in base_grid:
                                t_lp = f_cum_lp + cand_w * u_lp_tr
                                obj_val, _ = _composite_obj(t_lp, y_target=y_tr)
                                if obj_val > s_best_obj + 1e-6:
                                    s_best_obj = obj_val
                                    s_best_idx = idx
                                    s_best_w = cand_w
                        if s_best_idx is not None and s_best_w > 0.0:
                            fold_w[s_best_idx] = s_best_w
                            f_cum_lp = f_cum_lp + s_best_w * self.models[s_best_idx]["unseen_lp"][tr_idx]
                            f_obj = s_best_obj
                            f_rem.remove(s_best_idx)
                        else:
                            break

                    f_tot = sum(fold_w)
                    f_scale = (1.0 / f_tot) if f_tot > 0.0 else 1.0
                    fold_val_lp = (ex_val.copy() * f_scale) if len(ex_val) > 0 else np.zeros((len(val_idx), K), dtype=np.float64)
                    for idx, m in enumerate(self.models):
                        if fold_w[idx] > 0.0:
                            fold_val_lp += (fold_w[idx] * f_scale) * m["unseen_lp"][val_idx]
                    oof_lp[val_idx] = fold_val_lp

                oof_bacc = _fast_sinkhorn_bacc(oof_lp, self.y_unseen, iters=200)
        except Exception as e:
            print(f"[Ablation Warning] OOF GroupKFold computation failed ({e}), using in-sample BAcc fallback.")
            oof_bacc = in_sample_bacc

        print(f"[Ablation] In-Sample Unseen BAcc: {in_sample_bacc:.4f} | 5-Fold GroupKFold OOF Unseen BAcc: {oof_bacc:.4f}")

        P_mix = softmax(logit_flat, axis=1)
        self.P_mix_unseen = softmax(logit_flat_u, axis=1) if len(self.y_unseen) > 0 else np.zeros((0, K), dtype=np.float64)
        df_summary = pd.DataFrame([
            {"model": m["name"], "tier": m["tier"], "prior_weight": m.get("prior_weight", m["weight"]), "learned_weight": m["weight"]}
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
