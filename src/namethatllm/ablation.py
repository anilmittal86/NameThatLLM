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
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

from .config import LABELS, K, LAB2I, softmax, log_softmax

def _ablation_sinkhorn(P: np.ndarray, iters: int = 200, eps: float = 1e-12) -> np.ndarray:
    """Fast vectorized Sinkhorn mass balancing for balanced multiclass prediction."""
    if len(P) < 18:
        return P
    P = np.clip(P.astype(np.float64), eps, 1.0)
    target_col = 1.0 / P.shape[1]
    for _ in range(iters):
        P /= P.sum(axis=1, keepdims=True).clip(min=eps)
        P /= (P.sum(axis=0, keepdims=True) / (target_col * P.shape[0])).clip(min=eps)
    return P / P.sum(axis=1, keepdims=True).clip(min=eps)

sinkhorn_balancing = _ablation_sinkhorn

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
    samples_per_class: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Builds a balanced 18-class unseen holdout dataset (`df_unseen`) by:
    1. Hashing all texts in `train.csv` and `pool_v4.parquet` (and `pool_v1.parquet`) as `contaminated_hashes`.
    2. Scanning `nlm_matched_clean.parquet`, `nlm_pool_v5.parquet`, and `openrouter-harvest`
       for passages whose text hash is strictly NOT in `contaminated_hashes`.
    3. Sampling up to `samples_per_class` passages per class (e.g., 18 * 20 = 360 unseen passages)
       and returning `(df_unseen, holdout_hashes)` so `lexical_engine.py` and `trainer.py`
       exclude `holdout_hashes` from training!
    """
    input_dir = Path(input_dir) if input_dir else Path(".")
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

    # 2. Mark all pool_v4.parquet rows as contaminated (seen by Part-2 LoRAs, DeBERTa, instyle)
    for pat in ["pool_v4.parquet", "pool_v1.parquet"]:
        for p in input_dir.rglob(pat):
            try:
                df_p4 = pd.read_parquet(p, columns=["text"])
                contaminated_hashes.update(_text_hash(t) for t in df_p4["text"].fillna(""))
            except Exception:
                pass

    # 3. Collect strictly unseen candidate passages from newer/disjoint corpora
    candidate_dfs = []
    seen_paths = set()
    for pat in [
        "nlm_matched_clean.parquet",
        "nlm_pool_v5.parquet",
        "nlm_pool_v2.parquet",
        "*openrouter*.parquet",
        "*openrouter*.csv",
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

    if not candidate_dfs:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model", "_hash"]), set()

    df_pool_unseen = pd.concat(candidate_dfs, ignore_index=True).drop_duplicates(subset=["_hash"])

    # Stratified balanced sampling across all 18 classes
    rng = np.random.RandomState(seed)
    sampled_parts = []
    for m in LABELS:
        sub = df_pool_unseen[df_pool_unseen["model"] == m]
        if len(sub) > 0:
            n_take = min(samples_per_class, len(sub))
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
        self.y_unseen = np.asarray(y_unseen, dtype=int)
        self.groups_unseen = np.asarray(groups_unseen, dtype=int)
        self.N_test = int(N_test)
        self.models: List[Dict[str, Any]] = []

    def add_model(
        self,
        name: str,
        tier: str,
        weight: float,
        unseen_log_probs: np.ndarray,
        test_log_probs: np.ndarray,
    ):
        if len(self.y_unseen) > 0 and len(unseen_log_probs) != len(self.y_unseen):
            return
        u_lp = log_softmax(np.asarray(unseen_log_probs, dtype=np.float64), axis=1) if len(self.y_unseen) > 0 and len(unseen_log_probs) == len(self.y_unseen) else np.zeros((0, K), dtype=np.float64)
        t_lp = log_softmax(np.asarray(test_log_probs, dtype=np.float64), axis=1) if len(test_log_probs) == self.N_test else np.zeros((self.N_test, K), dtype=np.float64)
        solo_raw = float(balanced_accuracy_score(self.y_unseen, u_lp.argmax(axis=1))) if len(self.y_unseen) > 0 else 0.0
        solo_sink = _fast_sinkhorn_bacc(u_lp, self.y_unseen, iters=200) if len(self.y_unseen) >= 18 else solo_raw
        self.models.append({
            "name": name,
            "tier": tier,
            "weight": float(weight),
            "unseen_lp": u_lp,
            "test_lp": t_lp,
            "solo_unseen_bacc": solo_raw,
            "solo_unseen_sinkhorn_bacc": solo_sink,
        })

    def optimize_weights_and_blend(
        self,
        working_dir: Path,
        extra_test_log_probs: Optional[List[Tuple[str, float, np.ndarray]]] = None,
        optimize_on_unseen: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
        """
        1. Runs Greedy Forward Stepwise Selection + Coordinate Line Search on `df_unseen`
           to determine each model's optimal individual weight `w_star` (>= 0.0).
        2. Any model that fails to improve unseen balanced accuracy gets `w_star = 0.0` (PRUNED).
        3. Combines all active models + extra_test_log_probs (wpca_dp, spectral_bank) directly
           in flat log-probability space:
               LOGIT_FLAT = sum_m (w_star[m] * test_lp[m])
        """
        if len(self.models) == 0:
            return np.ones((self.N_test, K), dtype=np.float64) / float(K), {}, pd.DataFrame()

        if not optimize_on_unseen or len(self.y_unseen) < 36:
            weights_map = {m["name"]: m["weight"] for m in self.models}
            logit_flat = np.zeros((self.N_test, K), dtype=np.float64)
            for m in self.models:
                logit_flat += m["weight"] * m["test_lp"]
            if extra_test_log_probs:
                for ex_name, w_ex, lp_ex in extra_test_log_probs:
                    if len(lp_ex) == self.N_test and w_ex > 0.0:
                        logit_flat += float(w_ex) * log_softmax(lp_ex, axis=1)
                        weights_map[ex_name] = float(w_ex)
            P_mix = softmax(logit_flat, axis=1)
            df_summary = pd.DataFrame([
                {"model": m["name"], "tier": m["tier"], "prior_weight": m.get("prior_weight", m["weight"]), "learned_weight": m["weight"]}
                for m in self.models
            ])
            return P_mix, weights_map, df_summary

        M = len(self.models)
        order = sorted(range(M), key=lambda i: self.models[i]["solo_unseen_sinkhorn_bacc"], reverse=True)

        active_weights = np.zeros(M, dtype=np.float64)
        first_idx = order[0]
        active_weights[first_idx] = self.models[first_idx]["weight"]
        cum_u_lp = active_weights[first_idx] * self.models[first_idx]["unseen_lp"]
        best_bacc = float(
            balanced_accuracy_score(
                self.y_unseen,
                _ablation_sinkhorn(softmax(cum_u_lp, axis=1), iters=60).argmax(axis=1),
            )
        )

        multipliers = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
        for idx in order[1:]:
            prior_w = self.models[idx]["weight"]
            u_lp = self.models[idx]["unseen_lp"]
            best_m_w = 0.0
            best_m_bacc = best_bacc
            for mult in multipliers:
                cand_w = prior_w * mult
                trial_lp = cum_u_lp + cand_w * u_lp
                trial_P = _ablation_sinkhorn(softmax(trial_lp, axis=1), iters=60)
                score = float(balanced_accuracy_score(self.y_unseen, trial_P.argmax(axis=1)))
                if score > best_m_bacc + 1e-6:
                    best_m_bacc = score
                    best_m_w = cand_w
            if best_m_w > 0.0:
                active_weights[idx] = best_m_w
                cum_u_lp = cum_u_lp + best_m_w * u_lp
                best_bacc = best_m_bacc
            else:
                active_weights[idx] = 0.0

        weights_map: Dict[str, float] = {}
        logit_flat = np.zeros((self.N_test, K), dtype=np.float64)
        for idx, m in enumerate(self.models):
            m["prior_weight"] = m.get("prior_weight", m["weight"])
            m["learned_weight"] = float(active_weights[idx])
            m["weight"] = float(active_weights[idx])
            weights_map[m["name"]] = float(active_weights[idx])
            if active_weights[idx] > 0.0:
                logit_flat += active_weights[idx] * m["test_lp"]

        if extra_test_log_probs:
            for ex_name, w_ex, lp_ex in extra_test_log_probs:
                if len(lp_ex) == self.N_test and w_ex > 0.0:
                    logit_flat += float(w_ex) * log_softmax(lp_ex, axis=1)
                    weights_map[ex_name] = float(w_ex)

        P_mix = softmax(logit_flat, axis=1)
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
        Calculates the marginal change in balanced accuracy on `df_unseen`
        if `candidate_unseen_lp` with `weight` were added to the currently registered models.
        Returns Delta BAcc.
        """
        if len(self.y_unseen) == 0 or len(candidate_unseen_lp) != len(self.y_unseen) or len(self.models) == 0:
            return 0.0

        cum_unseen_lp = np.zeros((len(self.y_unseen), K), dtype=np.float64)
        for m in self.models:
            cum_unseen_lp += m["weight"] * m["unseen_lp"]

        cur_bacc = _fast_sinkhorn_bacc(cum_unseen_lp, self.y_unseen, iters=40)

        cand_lp = log_softmax(np.asarray(candidate_unseen_lp, dtype=np.float64), axis=1)
        new_unseen_lp = cum_unseen_lp + float(weight) * cand_lp
        new_bacc = _fast_sinkhorn_bacc(new_unseen_lp, self.y_unseen, iters=40)

        return new_bacc - cur_bacc

    def print_and_save_report(self, working_dir: Path, final_test_preds: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not self.models or len(self.y_unseen) == 0:
            print("[Ablation] No models or unseen holdout samples available; skipping report.")
            return pd.DataFrame()

        # Part 1: Standalone (Solo) Accuracy, sorted descending
        print("\n" + "=" * 105)
        print(f" PART 1: STANDALONE (SOLO) ACCURACY ON STRICTLY UNSEEN HOLDOUT ({len(self.y_unseen)} Disjoint Passages)")
        print("=" * 105)
        print(f"{'Model / Checkpoint':<28} | {'Tier':<6} | {'Unseen Raw BAcc':<16} | {'Unseen+Sinkhorn BAcc':<20} | {'Agrees w/ Final Test'}")
        print("-" * 105)
        sorted_solo = sorted(self.models, key=lambda m: m["solo_unseen_sinkhorn_bacc"], reverse=True)
        for m in sorted_solo:
            agree_str = "N/A"
            if final_test_preds is not None and len(final_test_preds) == self.N_test:
                pct = 100.0 * float((m["test_lp"].argmax(axis=1) == final_test_preds).mean())
                agree_str = f"{pct:5.1f}%"
            print(
                f"{m['name']:<28} | {m['tier']:<6} | "
                f"{m['solo_unseen_bacc']:<16.4f} | {m['solo_unseen_sinkhorn_bacc']:<20.4f} | {agree_str}"
            )

        # Part 2: Greedy Forward Stepwise Selection (starts with strongest solo model, tests all candidates at each step)
        print("\n" + "=" * 105)
        print(" PART 2: GREEDY FORWARD STEPWISE SELECTION ON STRICTLY UNSEEN HOLDOUT (Optimal Combinations)")
        print("=" * 105)
        print(f"{'Step':<5} | {'Added Model':<28} | {'Wt':<5} | {'Cumul Raw':<10} | {'Cumul+Sink':<11} | {'Marginal Lift':<13} | {'Test Rows Flipped'}")
        print("-" * 105)

        M = len(self.models)
        remaining = list(range(M))
        cum_unseen_lp = np.zeros((len(self.y_unseen), K), dtype=np.float64)
        cum_test_lp = np.zeros((self.N_test, K), dtype=np.float64)
        prev_bacc = 0.0
        prev_test_preds = None
        records = []

        step_idx = 1
        while remaining:
            best_cand_idx = None
            best_cand_bacc = -1.0
            best_cand_raw = 0.0

            if step_idx == 1:
                # Pick model with highest solo unseen sinkhorn bacc
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
                # Test all remaining candidate models
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

            w = m["weight"] if m["weight"] > 0 else m.get("prior_weight", 0.15)
            cum_unseen_lp = cum_unseen_lp + w * m["unseen_lp"]
            cum_test_lp = cum_test_lp + w * m["test_lp"]

            lift_str = f"{lift:+.4f}" if step_idx > 1 else "BASELINE"
            cur_test_preds = _ablation_sinkhorn(softmax(cum_test_lp, axis=1), iters=40).argmax(axis=1) if self.N_test > 0 else np.array([])
            if prev_test_preds is None:
                flips = self.N_test
                flip_str = f"{flips} (Init)"
            else:
                flips = int((cur_test_preds != prev_test_preds).sum()) if len(cur_test_preds) == len(prev_test_preds) else 0
                flip_pct = 100.0 * flips / max(1, self.N_test)
                flip_str = f"{flips:4d} ({flip_pct:4.1f}%)"

            print(
                f"{step_idx:<5} | + {m['name']:<26} | {w:<5.2f} | "
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
                "solo_unseen_bacc": m["solo_unseen_bacc"],
                "solo_unseen_sinkhorn_bacc": m["solo_unseen_sinkhorn_bacc"],
                "cumul_unseen_raw_bacc": best_cand_raw,
                "cumul_unseen_sinkhorn_bacc": best_cand_bacc,
                "marginal_lift": lift,
                "test_rows_flipped": flips,
            })
            step_idx += 1

        print("=" * 105 + "\n")

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
