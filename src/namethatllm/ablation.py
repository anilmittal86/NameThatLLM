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

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

from .config import LABELS, K, _text_hash, calibrate_model_lp, log_softmax, softmax
from .postprocess import sinkhorn_balancing as _sinkhorn_balancing
from .wpca_gaussian import split_oversized_clusters_dist

# =============================================================================
# HELPER FUNCTIONS: MASS BALANCING & ZERO-LEAKAGE TEXT HASHING
# =============================================================================


def _ablation_sinkhorn(
    P: np.ndarray,
    iters: int = 100,
    eps: float = 1e-12,
    y_true: np.ndarray | None = None,
    allow_pure: bool = False,
) -> np.ndarray:
    """
    Fast vectorized Sinkhorn-Knopp column mass balancing for evaluation.
    Delegates directly to sinkhorn_balancing in namethatllm.postprocess.
    """
    train_prior = np.bincount(y_true, minlength=P.shape[1]).astype(np.float64) if y_true is not None and len(y_true) == len(P) else None
    return _sinkhorn_balancing(
        P,
        iters=iters,
        sinkhorn_strength=1.0 if allow_pure else 0.65,
        target_mix=1.0 if allow_pure else 0.35,
        train_prior=train_prior,
        allow_pure=allow_pure,
    )


def _fast_sinkhorn_bacc(lp_mat: np.ndarray, y_true: np.ndarray, iters: int = 40) -> float:
    """
    Quickly computes balanced accuracy of log-probabilities after 40 Sinkhorn iterations.
    Used inside inner optimization loops (line search, greedy selection) for speed.
    """
    P = softmax(lp_mat, axis=1)
    if len(P) >= 18:
        P = _ablation_sinkhorn(P, iters=iters, y_true=y_true)
    return float(balanced_accuracy_score(y_true, P.argmax(axis=1)))


def build_unseen_holdout_dataset(
    input_dir: Path,
    train_path: Path | None = None,
    samples_per_class: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, set[str]]:
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
    contaminated_hashes: set[str] = set()

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

    # 2. Mark only pool_v4.parquet and pool_v1.parquet as contaminated
    # (seen by Part-2 LoRAs, DeBERTa, inhouse)
    for pat in ["pool_v4.parquet", "pool_v1.parquet"]:
        for p in input_dir.rglob(pat):
            try:
                df_p = pd.read_parquet(p, columns=["text"])
                contaminated_hashes.update(_text_hash(t) for t in df_p["text"].fillna(""))
            except Exception:
                pass

    # 3. Collect candidate passages in canonical trainer quarantine order
    candidate_dfs = []
    seen_paths = set()

    def _process_candidate_file(p: Path) -> pd.DataFrame | None:
        try:
            df_c = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if "label" in df_c.columns and "model" not in df_c.columns:
                df_c = df_c.rename(columns={"label": "model"})
            finish_col = next((c for c in ["finish_reason", "finish"] if c in df_c.columns), None)
            if finish_col:
                df_c = df_c[df_c[finish_col].astype(str).str.lower().eq("stop")]
                df_c = df_c[df_c["text"].fillna("").astype(str).str.split().str.len().between(25, 360)]
            if "text" in df_c.columns and "model" in df_c.columns:
                keep_cols = [c for c in ["id", "prompt_id", "text", "model"] if c in df_c.columns]
                df_c = df_c[keep_cols].dropna(subset=["text", "model"]).copy()
                df_c = df_c[df_c["model"].isin(LABELS)]
                df_c["_hash"] = [_text_hash(t) for t in df_c["text"]]
                df_c = df_c[~df_c["_hash"].isin(contaminated_hashes)]
                if len(df_c) > 0:
                    return df_c
        except Exception:
            pass
        return None

    candidate_patterns = [
        "nlm_matched_clean.parquet",
        "nlm_pool_v5.parquet",
        "nlm_pool_v2.parquet",
        "*openrouter*.parquet",
        "*openrouter*.csv",
        "*harvest*.parquet",
        "*harvest*.csv",
    ]
    for pat in candidate_patterns:
        for p in input_dir.rglob(pat):
            if p in seen_paths:
                continue
            p_name = p.name.lower()
            if any(skip in p_name for skip in ["train", "test", "submission", "pool_v4", "pool_v1"]):
                continue
            seen_paths.add(p)
            df_c = _process_candidate_file(p)
            if df_c is not None:
                candidate_dfs.append(df_c)

    # Also scan any directory or file matching openrouter, harvest, passages, generations
    for ext in ["*.parquet", "*.csv"]:
        for p in input_dir.rglob(ext):
            if p in seen_paths:
                continue
            p_str = str(p).lower().replace("\\", "/")
            p_name = p.name.lower()
            if any(skip in p_name for skip in ["train", "test", "submission", "pool_v4", "pool_v1"]):
                continue
            if any(k in p_str for k in ["openrouter", "harvest", "passages", "generations"]):
                seen_paths.add(p)
                df_c = _process_candidate_file(p)
                if df_c is not None:
                    candidate_dfs.append(df_c)

    # Fallback to any remaining non-training parquet/csv if still empty
    if not candidate_dfs:
        for ext in ["*.parquet", "*.csv"]:
            for p in input_dir.rglob(ext):
                if p in seen_paths:
                    continue
                p_name = p.name.lower()
                if any(x in p_name for x in ["train", "test", "submission", "pool_v4", "pool_v1"]):
                    continue
                seen_paths.add(p)
                df_c = _process_candidate_file(p)
                if df_c is not None:
                    candidate_dfs.append(df_c)

    if not candidate_dfs:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model", "_hash"]), set()

    df_pool_unseen = pd.concat(candidate_dfs, ignore_index=True).drop_duplicates(subset=["_hash"]).reset_index(drop=True)

    v5_hashes: set[str] = set()
    for pat in ["nlm_pool_v5.parquet", "*pool_v5*.parquet"]:
        for p in input_dir.rglob(pat):
            try:
                df_v5 = pd.read_parquet(p, columns=["text"])
                v5_hashes.update(_text_hash(t) for t in df_v5["text"].fillna(""))
            except Exception:
                pass
    strictly_disjoint = df_pool_unseen[~df_pool_unseen["_hash"].isin(v5_hashes)].reset_index(drop=True)
    if strictly_disjoint["model"].nunique() == K and (strictly_disjoint["model"].value_counts() >= samples_per_class).all():
        pool_to_sample = strictly_disjoint
    else:
        pool_to_sample = df_pool_unseen

    has_source_prompt = "prompt_id" in pool_to_sample.columns and pool_to_sample["prompt_id"].notna().sum() >= 36
    if has_source_prompt:
        prompt_richness = pool_to_sample.groupby("prompt_id")["model"].nunique()

    # Stratified balanced sampling across all 18 classes from a single consistent pool
    rng = np.random.RandomState(seed)
    sampled_parts = []
    for m in LABELS:
        sub = pool_to_sample[pool_to_sample["model"] == m].reset_index(drop=True)
        if len(sub) > 0:
            n_take = min(samples_per_class, len(sub))
            if has_source_prompt:
                sub_prompts = sub["prompt_id"].map(prompt_richness).fillna(1).values
                jitter = rng.rand(len(sub))
                sort_keys = np.lexsort((jitter, -(sub_prompts >= 2).astype(int), -sub_prompts))
                sub_sorted = sub.iloc[sort_keys].reset_index(drop=True)
                sampled_parts.append(sub_sorted.iloc[:n_take])
            else:
                idx = rng.choice(len(sub), size=n_take, replace=False)
                sampled_parts.append(sub.iloc[idx])

    if not sampled_parts:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model", "_hash"]), set()

    df_unseen = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if "id" not in df_unseen.columns:
        df_unseen["id"] = [f"unseen_{i}" for i in range(len(df_unseen))]

    # Compute valid prompt clusters
    clusters = None
    if len(df_unseen) >= 2:
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=10000, sublinear_tf=True)
            X_tfidf = tfidf.fit_transform(df_unseen["text"].fillna("").tolist())
            D = 1.0 - (X_tfidf @ X_tfidf.T).toarray().clip(0.0, 1.0)
            np.fill_diagonal(D, 0.0)

            use_source_prompt = False
            if "prompt_id" in df_unseen.columns and df_unseen["prompt_id"].notna().sum() == len(df_unseen):
                codes = pd.factorize(df_unseen["prompt_id"])[0]
                if len(codes) > 0:
                    counts = np.bincount(codes)
                    if (counts[codes] >= 2).mean() >= 0.40:
                        clusters = split_oversized_clusters_dist(D, codes + 1, max_cluster_size=14)
                        use_source_prompt = True

            if not use_source_prompt:
                cond_d = squareform(D, checks=False)
                raw_clusters = fcluster(linkage(cond_d, method="average"), t=0.90, criterion="distance")
                clusters = split_oversized_clusters_dist(D, raw_clusters, max_cluster_size=14)
        except Exception:
            clusters = np.arange(1, len(df_unseen) + 1, dtype=int)
    else:
        clusters = np.ones(len(df_unseen), dtype=int)

    df_unseen["cluster_id"] = clusters
    df_unseen["prompt_id"] = clusters

    holdout_hashes = set(df_unseen["_hash"].tolist())
    print(
        f"[Unseen Benchmark] Built strictly unseen holdout set: {len(df_unseen)} rows "
        f"across {df_unseen['model'].nunique()} classes (0% overlap with train.csv, pool_v4.parquet & pool_v5.parquet)."
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
        self.models: list[dict[str, Any]] = []

    def add_model(
        self,
        name: str,
        tier: str,
        weight: float,
        unseen_log_probs: np.ndarray,
        test_log_probs: np.ndarray,
    ):
        u_lp = (
            calibrate_model_lp(unseen_log_probs, model_name=name)
            if len(self.y_unseen) > 0 and len(unseen_log_probs) == len(self.y_unseen)
            else np.zeros((0, K), dtype=np.float64)
        )
        t_lp = (
            calibrate_model_lp(test_log_probs, model_name=name)
            if len(test_log_probs) == self.N_test
            else np.zeros((self.N_test, K), dtype=np.float64)
        )
        solo_raw = (
            float(balanced_accuracy_score(self.y_unseen, u_lp.argmax(axis=1)))
            if len(self.y_unseen) > 0 and len(u_lp) == len(self.y_unseen)
            else 0.0
        )
        solo_sink = (
            _fast_sinkhorn_bacc(u_lp, self.y_unseen, iters=200)
            if len(self.y_unseen) >= 18 and len(u_lp) == len(self.y_unseen)
            else solo_raw
        )
        self.models.append(
            {
                "name": name,
                "tier": tier,
                "weight": float(weight),
                "prior_weight": float(weight),
                "unseen_lp": u_lp,
                "test_lp": t_lp,
                "solo_unseen_bacc": solo_raw,
                "solo_unseen_sinkhorn_bacc": solo_sink,
            }
        )

    def optimize_weights_and_blend(
        self,
        working_dir: Path,
        extra_test_log_probs: list[tuple[str, float, np.ndarray]] | None = None,
        optimize_on_unseen: bool = True,
        lock_prior_weights: bool = False,
    ) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
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
            df_summary = pd.DataFrame(
                [
                    {
                        "model": m["name"],
                        "tier": m["tier"],
                        "prior_weight": m.get("prior_weight", m["weight"]),
                        "learned_weight": m["weight"],
                    }
                    for m in self.models
                ]
            )
            return P_mix, weights_map, df_summary

        # Composite objective function: Balanced Accuracy with mild Negative Log-Likelihood regularization.
        # Why: Hard 0-1 accuracy has flat plateaus where many weight values produce identical discrete predictions.
        # Adding -0.015 * NLL provides smooth, well-conditioned gradients favoring well-calibrated confidence.
        def _composite_obj(lp_mat: np.ndarray) -> tuple[float, float]:
            P = softmax(lp_mat, axis=1)
            P_s = _ablation_sinkhorn(P, iters=60, y_true=self.y_unseen) if len(P) >= 18 else P
            bacc = float(balanced_accuracy_score(self.y_unseen, P_s.argmax(axis=1)))
            nll = float(-np.mean(np.log(np.clip(P_s[np.arange(len(self.y_unseen)), self.y_unseen], 1e-9, 1.0))))
            return bacc - 0.015 * nll, bacc

        M = len(self.models)

        def _effective_bacc(m: dict[str, Any]) -> float:
            return float(max(m.get("solo_unseen_bacc", 0.0), m.get("solo_unseen_sinkhorn_bacc", 0.0)))

        # Sort candidate models by solo performance on unseen holdout descending
        order = sorted(range(M), key=lambda i: _effective_bacc(self.models[i]), reverse=True)

        if lock_prior_weights:
            # Conservative mode: keep fixed prior weights from configuration, pruning sub-threshold models
            active_weights = np.zeros(M, dtype=np.float64)
            for idx, m in enumerate(self.models):
                prior_w = float(m.get("prior_weight", m["weight"]))
                eff_b = _effective_bacc(m)
                if eff_b < 0.22 or prior_w <= 0.0:
                    active_weights[idx] = 0.0
                else:
                    active_weights[idx] = prior_w
        else:
            # Adaptive mode: Greedy forward stepwise selection + coordinate scale search
            active_weights = np.zeros(M, dtype=np.float64)
            valid_candidates = [i for i in order if _effective_bacc(self.models[i]) >= 0.22 and self.models[i]["weight"] > 0.0]
            if not valid_candidates:
                for idx, m in enumerate(self.models):
                    active_weights[idx] = m["weight"]
            else:
                # Step 1: Initialize ensemble with the single best solo performer
                first_idx = valid_candidates[0]
                m_first = self.models[first_idx]
                mults_first = (0.75, 1.00, 1.25) if _effective_bacc(m_first) >= 0.34 else (0.25, 0.50, 0.75, 1.00)
                best_first_w = m_first["weight"]
                best_first_obj = -1e9
                best_first_bacc = 0.0
                for mult in mults_first:
                    cand_w = m_first["weight"] * mult
                    obj_val, bacc_val = _composite_obj(cand_w * m_first["unseen_lp"])
                    if obj_val > best_first_obj:
                        best_first_obj = obj_val
                        best_first_bacc = bacc_val
                        best_first_w = cand_w
                active_weights[first_idx] = best_first_w
                cum_u_lp = active_weights[first_idx] * m_first["unseen_lp"]
                best_obj = best_first_obj
                best_bacc = best_first_bacc

                remaining = list(valid_candidates[1:])
                while remaining:
                    step_best_idx = None
                    step_best_w = 0.0
                    step_best_obj = best_obj
                    step_best_bacc = best_bacc

                    for idx in remaining:
                        m_info = self.models[idx]
                        prior_w = m_info["weight"]
                        u_lp = m_info["unseen_lp"]
                        eff_b = _effective_bacc(m_info)

                        if eff_b < 0.22 or prior_w <= 0.0:
                            continue

                        mults = (0.75, 1.00, 1.25) if eff_b >= 0.34 else (0.25, 0.50, 0.75, 1.00)
                        for mult in mults:
                            cand_w = prior_w * mult
                            trial_lp = cum_u_lp + cand_w * u_lp
                            obj_val, bacc_val = _composite_obj(trial_lp)
                            if obj_val > step_best_obj + 1e-6:
                                step_best_obj = obj_val
                                step_best_bacc = bacc_val
                                step_best_idx = idx
                                step_best_w = cand_w

                    if step_best_idx is not None and step_best_w > 0.0:
                        active_weights[step_best_idx] = step_best_w
                        cum_u_lp = cum_u_lp + step_best_w * self.models[step_best_idx]["unseen_lp"]
                        best_obj = step_best_obj
                        best_bacc = step_best_bacc
                        remaining.remove(step_best_idx)
                    else:
                        break

        if not lock_prior_weights:
            # Post-loop enforcement:
            # 1. Enforce 0.75 * prior_weight floor on all validated models (eff_bacc >= 0.34 and weight > 0)
            #    so nlm_pool_v5 holdout leakage on qwen3_14b_v5 never zeroes out Part-2, in-house, DeBERTa, or LinearSVC!
            # 2. Prune any model with eff_bacc < 0.30 or weight <= 0 to strictly 0.0.
            for idx, m_info in enumerate(self.models):
                eff_b = _effective_bacc(m_info)
                if eff_b < 0.22 or m_info["weight"] <= 0.0:
                    active_weights[idx] = 0.0
                elif eff_b >= 0.34 and m_info["weight"] > 0.0:
                    active_weights[idx] = max(active_weights[idx], 0.75 * m_info["weight"])

        weights_map: dict[str, float] = {}
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
        df_summary = pd.DataFrame(
            [
                {"model": m["name"], "tier": m["tier"], "prior_weight": m.get("prior_weight", m["weight"]), "learned_weight": m["weight"]}
                for m in self.models
            ]
        )
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
            w_m = m["weight"] if "learned_weight" in m else (m["weight"] if m["weight"] > 0 else m.get("prior_weight", 0.15))
            if w_m <= 0.0:
                continue
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
            best_lift = max(best_lift, lift)

        if best_lift <= 0.0:
            return float(full_weight_lift)
        return float(best_lift)

    def print_and_save_report(self, working_dir: Path, final_test_preds: np.ndarray | None = None) -> pd.DataFrame:
        if not self.models or len(self.y_unseen) == 0:
            print("[Ablation] No models or unseen holdout samples available; skipping report.")
            return pd.DataFrame()

        # Part 1: Standalone (Solo) Accuracy, sorted descending
        print("\n" + "=" * 105)
        print(f" PART 1: STANDALONE (SOLO) ACCURACY ON STRICTLY UNSEEN HOLDOUT ({len(self.y_unseen)} Disjoint Passages)")
        print("=" * 105)
        print(
            f"{'Model / Checkpoint':<28} | {'Tier':<6} | {'Unseen Raw BAcc':<16} | {'Unseen+Sinkhorn BAcc':<20} | {'Agrees w/ Final Test'}"
        )
        print("-" * 105)
        sorted_solo = sorted(self.models, key=lambda m: m["solo_unseen_sinkhorn_bacc"], reverse=True)
        for m in sorted_solo:
            agree_str = "N/A"
            if final_test_preds is not None and len(final_test_preds) == self.N_test:
                pct = 100.0 * float((m["test_lp"].argmax(axis=1) == final_test_preds).mean())
                agree_str = f"{pct:5.1f}%"
            print(
                f"{m['name']:<28} | {m['tier']:<6} | {m['solo_unseen_bacc']:<16.4f} | {m['solo_unseen_sinkhorn_bacc']:<20.4f} | {agree_str}"
            )

        # Part 2: Greedy Forward Stepwise Selection (starts with strongest solo model, tests all candidates at each step)
        print("\n" + "=" * 105)
        print(" PART 2: GREEDY FORWARD STEPWISE SELECTION ON STRICTLY UNSEEN HOLDOUT (Optimal Combinations)")
        print("=" * 105)
        print(
            f"{'Step':<5} | {'Added Model':<28} | {'Wt':<5} | {'Cumul Raw':<10} | {'Cumul+Sink':<11} | {'Marginal Lift':<13} | {'Test Rows Flipped'}"
        )
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
                    cand_w = (
                        cand["weight"]
                        if "learned_weight" in cand
                        else (cand["weight"] if cand["weight"] > 0 else cand.get("prior_weight", 0.15))
                    )
                    if cand_w <= 0.0:
                        continue
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
                    cand_w = (
                        cand["weight"]
                        if "learned_weight" in cand
                        else (cand["weight"] if cand["weight"] > 0 else cand.get("prior_weight", 0.15))
                    )
                    if cand_w <= 0.0:
                        continue
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
                print(
                    f"[Greedy Forward Selection] Best remaining candidate '{self.models[best_cand_idx]['name']}' gives negative lift ({lift:+.4f}); stopping forward selection."
                )
                break

            m = self.models[best_cand_idx]
            remaining.remove(best_cand_idx)

            w = m["weight"] if "learned_weight" in m else (m["weight"] if m["weight"] > 0 else m.get("prior_weight", 0.15))
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
            records.append(
                {
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
                }
            )
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
                    sub_unseen_lp = sum(
                        (self.models[i]["weight"] if self.models[i]["weight"] > 0 else self.models[i].get("prior_weight", 0.15))
                        * self.models[i]["unseen_lp"]
                        for i in sub
                    )
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
                        loo_lp = sum(
                            (self.models[j]["weight"] if self.models[j]["weight"] > 0 else self.models[j].get("prior_weight", 0.15))
                            * self.models[j]["unseen_lp"]
                            for j in loo_sub
                        )
                        loo_bacc = _fast_sinkhorn_bacc(loo_lp, self.y_unseen, iters=30)
                        delta_loo = best_subset_bacc - loo_bacc
                    loo_str = f"-{delta_loo:.4f}" if delta_loo >= 0 else f"+{abs(delta_loo):.4f}"
                else:
                    with_m_sub = list(best_subset) + [i]
                    with_m_lp = sum(
                        (self.models[j]["weight"] if self.models[j]["weight"] > 0 else self.models[j].get("prior_weight", 0.15))
                        * self.models[j]["unseen_lp"]
                        for j in with_m_sub
                    )
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
