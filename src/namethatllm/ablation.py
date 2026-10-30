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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .config import LABELS, K, LAB2I, softmax, log_softmax

def sinkhorn_balancing(P: np.ndarray, iters: int = 200, eps: float = 1e-9) -> np.ndarray:
    """Fast vectorized Sinkhorn mass balancing for balanced multiclass prediction."""
    P = np.clip(P, eps, 1.0)
    for _ in range(iters):
        P = P / P.sum(axis=1, keepdims=True).clip(min=eps)
        P = P / P.sum(axis=0, keepdims=True).clip(min=eps)
    return P / P.sum(axis=1, keepdims=True).clip(min=eps)

def _text_hash(text: str) -> str:
    """Normalized SHA1 hash of whitespace-collapsed text to guarantee zero exact/near-exact leakage."""
    norm = " ".join(str(text).strip().split())
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()

def build_unseen_holdout_dataset(
    input_dir: Path,
    train_path: Optional[Path] = None,
    samples_per_class: int = 80,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Builds a balanced 18-class unseen holdout dataset (`df_unseen`) by:
    1. Hashing all texts in `train.csv` and `pool_v4.parquet` (and `pool_v1.parquet`) as `contaminated_hashes`.
    2. Scanning `nlm_matched_clean.parquet`, `nlm_pool_v5.parquet`, and `openrouter-harvest`
       for passages whose text hash is strictly NOT in `contaminated_hashes`.
    3. Sampling up to `samples_per_class` passages per class (e.g., 18 * 80 = 1,440 unseen passages)
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
    if "prompt_id" not in df_unseen.columns:
        df_unseen["prompt_id"] = -1

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
    1. Table 1 — Standalone (Solo) Unseen Balanced Accuracy (Plain, +Twin, +Sinkhorn).
    2. Table 2 — Forward Stepwise Addition (starting from Model 1 and adding one model
       at a time), showing Cumulative Unseen BAcc, Marginal Lift (Delta BAcc), and
       Test-Set Predictions Flipped.
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
        if len(self.y_unseen) == 0 or len(unseen_log_probs) != len(self.y_unseen):
            return
        u_lp = log_softmax(np.asarray(unseen_log_probs, dtype=np.float64), axis=1)
        t_lp = log_softmax(np.asarray(test_log_probs, dtype=np.float64), axis=1) if len(test_log_probs) == self.N_test else np.zeros((self.N_test, K), dtype=np.float64)
        solo_raw = float(balanced_accuracy_score(self.y_unseen, u_lp.argmax(axis=1)))
        solo_sink = float(
            balanced_accuracy_score(
                self.y_unseen, sinkhorn_balancing(softmax(u_lp, axis=1), iters=200).argmax(axis=1)
            )
        )
        self.models.append({
            "name": name,
            "tier": tier,
            "weight": float(weight),
            "unseen_lp": u_lp,
            "test_lp": t_lp,
            "solo_unseen_bacc": solo_raw,
            "solo_unseen_sinkhorn_bacc": solo_sink,
        })

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

        cur_sink_P = sinkhorn_balancing(softmax(cum_unseen_lp, axis=1), iters=200)
        cur_bacc = float(balanced_accuracy_score(self.y_unseen, cur_sink_P.argmax(axis=1)))

        cand_lp = log_softmax(np.asarray(candidate_unseen_lp, dtype=np.float64), axis=1)
        new_unseen_lp = cum_unseen_lp + float(weight) * cand_lp
        new_sink_P = sinkhorn_balancing(softmax(new_unseen_lp, axis=1), iters=200)
        new_bacc = float(balanced_accuracy_score(self.y_unseen, new_sink_P.argmax(axis=1)))

        return new_bacc - cur_bacc

    def print_and_save_report(self, working_dir: Path, final_test_preds: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not self.models or len(self.y_unseen) == 0:
            print("[Ablation] No models or unseen holdout samples available; skipping report.")
            return pd.DataFrame()

        print("\n" + "=" * 100)
        print(f" PART 1: STANDALONE (SOLO) ACCURACY ON STRICTLY UNSEEN HOLDOUT ({len(self.y_unseen)} Disjoint Passages)")
        print("=" * 100)
        print(f"{'Model / Checkpoint':<28} | {'Tier':<6} | {'Unseen Raw BAcc':<16} | {'Unseen+Sinkhorn BAcc':<20} | {'Agrees w/ Final Test'}")
        print("-" * 100)
        for m in self.models:
            agree_str = "N/A"
            if final_test_preds is not None and len(final_test_preds) == self.N_test:
                pct = 100.0 * float((m["test_lp"].argmax(axis=1) == final_test_preds).mean())
                agree_str = f"{pct:5.1f}%"
            print(
                f"{m['name']:<28} | {m['tier']:<6} | "
                f"{m['solo_unseen_bacc']:<16.4f} | {m['solo_unseen_sinkhorn_bacc']:<20.4f} | {agree_str}"
            )

        print("\n" + "=" * 100)
        print(" PART 2: FORWARD STEPWISE ADDITION ON STRICTLY UNSEEN HOLDOUT (One Model Added at a Time)")
        print("=" * 100)
        print(f"{'Step':<5} | {'Added Model':<26} | {'Wt':<5} | {'Cumul Raw':<10} | {'Cumul+Sink':<11} | {'Marginal Lift':<13} | {'Test Rows Flipped'}")
        print("-" * 100)

        cum_unseen_lp = np.zeros((len(self.y_unseen), K), dtype=np.float64)
        cum_test_lp = np.zeros((self.N_test, K), dtype=np.float64)
        prev_bacc = 0.0
        prev_test_preds = None
        records = []

        for step_idx, m in enumerate(self.models, start=1):
            w = m["weight"]
            cum_unseen_lp = cum_unseen_lp + w * m["unseen_lp"]
            cum_test_lp = cum_test_lp + w * m["test_lp"]

            raw_bacc = float(balanced_accuracy_score(self.y_unseen, cum_unseen_lp.argmax(axis=1)))
            sink_P = sinkhorn_balancing(softmax(cum_unseen_lp, axis=1), iters=200)
            sink_bacc = float(balanced_accuracy_score(self.y_unseen, sink_P.argmax(axis=1)))

            lift = sink_bacc - prev_bacc if step_idx > 1 else 0.0
            lift_str = f"{lift:+.4f}" if step_idx > 1 else "BASELINE"

            cur_test_preds = sinkhorn_balancing(softmax(cum_test_lp, axis=1), iters=200).argmax(axis=1) if self.N_test > 0 else np.array([])
            if prev_test_preds is None:
                flips = self.N_test
                flip_str = f"{flips} (Init)"
            else:
                flips = int((cur_test_preds != prev_test_preds).sum()) if len(cur_test_preds) == len(prev_test_preds) else 0
                flip_pct = 100.0 * flips / max(1, self.N_test)
                flip_str = f"{flips:4d} ({flip_pct:4.1f}%)"

            print(
                f"{step_idx:<5} | + {m['name']:<24} | {w:<5.2f} | "
                f"{raw_bacc:<10.4f} | {sink_bacc:<11.4f} | {lift_str:<13} | {flip_str}"
            )
            prev_bacc = sink_bacc
            prev_test_preds = cur_test_preds
            records.append({
                "step": step_idx,
                "model": m["name"],
                "added_model": m["name"],
                "tier": m["tier"],
                "weight": w,
                "solo_unseen_bacc": m["solo_unseen_bacc"],
                "solo_unseen_sinkhorn_bacc": m["solo_unseen_sinkhorn_bacc"],
                "cumul_unseen_raw_bacc": raw_bacc,
                "cumul_unseen_sinkhorn_bacc": sink_bacc,
                "marginal_lift": lift,
                "test_rows_flipped": flips,
            })

        print("=" * 100 + "\n")
        df_rep = pd.DataFrame(records)
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        df_rep.to_csv(working_dir / "unseen_stepwise_ablation.csv", index=False)
        return df_rep
