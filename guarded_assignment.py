"""
Module 3: guarded_assignment.py (Step 4 - Guarded Prompt Clustering & Assignment)

Post-processing module using semantic prompt clustering and Guarded Hungarian matching
(scipy.optimize.linear_sum_assignment) to resolve duplicate model predictions within
multi-passage prompt clusters.

Features:
- Semantic prompt embedding with unit normalization (SentenceTransformer / HF AutoModel fallback).
- AgglomerativeClustering with distance threshold (1.0 - sim_threshold).
- Strictly guarded Hungarian assignment:
    Case 1 (|C| == 1): Singleton, unchanged top-1 argmax.
    Case 2 (2 <= |C| <= 18): Cost M = -log(clip(P, 1e-7, 1.0)), linear_sum_assignment(M) 1-to-1 matching.
    Case 3 (|C| > 18): Oversized cluster safety guardrail (fallback to unconstrained argmax).
- evaluate_assignment(): Audits full and sparse Monte Carlo prompt subsampling.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("guarded_assignment")

LABELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
NUM_CLASSES = len(LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


class PromptSemanticEmbedder:
    """Computes unit-normalized embeddings for passages using SentenceTransformer or HF AutoModel."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        self.model_id = model_id
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.st_model = None
        self._init_model()

    def _resolve_path(self, model_id: str) -> str:
        candidates = [
            Path(model_id),
            Path(f"/kaggle/input/{model_id}"),
            Path(f"/kaggle/input/models/{model_id}"),
            Path(f"/kaggle/input/{model_id.split('/')[-1]}"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return model_id

    def _init_model(self):
        resolved = self._resolve_path(self.model_id)
        # Attempt 1: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(resolved)
            logger.info(f"Loaded SentenceTransformer from: {resolved}")
            return
        except Exception as e:
            logger.info(f"SentenceTransformer not available or failed ({e}). Falling back to HF AutoModel.")

        # Attempt 2: HuggingFace AutoModel with mean pooling
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(resolved)
            self.model = AutoModel.from_pretrained(resolved)
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval()
            logger.info(f"Loaded HF AutoModel from: {resolved}")
            return
        except Exception as e2:
            logger.warning(f"HF AutoModel failed ({e2}). Will fall back to character/word n-gram embeddings if needed.")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts into unit L2-normalized embeddings."""
        N = len(texts)
        if N == 0:
            return np.empty((0, 384), dtype=np.float32)

        if self.st_model is not None:
            embs = self.st_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embs.astype(np.float32)

        if self.model is not None and self.tokenizer is not None:
            device = next(self.model.parameters()).device
            all_embs = []
            with torch.no_grad():
                for start in range(0, N, self.batch_size):
                    batch_texts = texts[start : start + self.batch_size]
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors="pt",
                    ).to(device)

                    outputs = self.model(**encoded)
                    # Mean pooling with attention mask
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    sum_embs = torch.sum(outputs[0] * mask, dim=1)
                    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
                    mean_embs = sum_embs / denom
                    # L2 normalize
                    norm_embs = mean_embs / torch.clamp(torch.norm(mean_embs, p=2, dim=-1, keepdim=True), min=1e-9)
                    all_embs.append(norm_embs.cpu().numpy())

            return np.vstack(all_embs).astype(np.float32)

        # Fallback: TF-IDF normalized vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=512)
        sparse_embs = vec.fit_transform(texts)
        dense = sparse_embs.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


class GuardedPromptAssigner:
    """Clusters passages by prompt topic and applies Hungarian matching to prevent duplicate model predictions."""

    def __init__(
        self,
        sim_threshold: float = 0.65,
        embedder: Optional[PromptSemanticEmbedder] = None,
    ):
        self.sim_threshold = sim_threshold
        self.embedder = embedder or PromptSemanticEmbedder()

    def cluster_passages(self, embeddings: np.ndarray) -> np.ndarray:
        """Clusters passages by cosine similarity distance (dist = 1 - sim)."""
        N = len(embeddings)
        if N <= 1:
            return np.zeros(N, dtype=np.int32)

        distance_threshold = max(0.01, 1.0 - self.sim_threshold)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(embeddings)
        return labels

    def assign_cluster(self, cluster_indices: List[int], probs: np.ndarray) -> List[int]:
        """Assigns labels to passages within a single cluster using Hungarian algorithm."""
        k = len(cluster_indices)

        # Case 1: Singleton (no duplicates possible)
        if k == 1:
            return [int(np.argmax(probs[cluster_indices[0]]))]

        # Case 3: Oversized cluster (> 18 passages cannot be 1-to-1 matched across 18 models)
        if k > NUM_CLASSES:
            logger.warning(f"Oversized cluster detected (size {k} > 18). Applying unconstrained argmax guardrail.")
            return [int(np.argmax(probs[idx])) for idx in cluster_indices]

        # Case 2: Verified cluster (2 <= k <= 18)
        # Cost matrix M_rk = -log(clip(P_rk, 1e-7, 1.0))
        cluster_probs = probs[cluster_indices]  # shape (k, 18)
        clipped_probs = np.clip(cluster_probs, 1e-7, 1.0)
        cost_matrix = -np.log(clipped_probs)  # shape (k, 18)

        # Solve optimal bipartite matching (each passage gets a distinct class)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Map back to passage positions
        assigned_labels = [0] * k
        for r, c in zip(row_ind, col_ind):
            assigned_labels[r] = int(c)

        return assigned_labels

    def assign(
        self,
        texts: List[str],
        probs: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs full guarded Hungarian assignment across all passages.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (assigned_class_indices, cluster_labels)
        """
        N = len(texts)
        if N == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        if embeddings is None:
            embeddings = self.embedder.encode(texts)

        cluster_labels = self.cluster_passages(embeddings)
        final_preds = np.zeros(N, dtype=np.int32)

        unique_clusters = np.unique(cluster_labels)
        for cl in unique_clusters:
            member_indices = np.where(cluster_labels == cl)[0].tolist()
            assigned = self.assign_cluster(member_indices, probs)
            for idx, pred in zip(member_indices, assigned):
                final_preds[idx] = pred

        return final_preds, cluster_labels


def evaluate_assignment(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    texts: List[str],
    sim_threshold: float = 0.65,
    embedder: Optional[PromptSemanticEmbedder] = None,
    n_monte_carlo_trials: int = 20,
) -> Dict[str, float]:
    """Audits Guarded Hungarian Assignment on both full and simulated sparse prompt distributions."""
    assigner = GuardedPromptAssigner(sim_threshold=sim_threshold, embedder=embedder)
    embs = assigner.embedder.encode(texts)

    raw_preds = np.argmax(raw_probs, axis=1)
    raw_bacc = balanced_accuracy_score(y_true, raw_preds)
    raw_acc = accuracy_score(y_true, raw_preds)

    guarded_preds, cluster_labels = assigner.assign(texts, raw_probs, embeddings=embs)
    guarded_bacc = balanced_accuracy_score(y_true, guarded_preds)
    guarded_acc = accuracy_score(y_true, guarded_preds)

    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    n_singletons = int(np.sum(counts == 1))
    n_multi = int(len(counts) - n_singletons)
    mean_cluster_size = float(np.mean(counts))

    # Monte Carlo Sparse Prompt Simulation (subsample 1 to 4 passages per cluster)
    sparse_lifts = []
    if len(texts) > 20:
        rng = np.random.RandomState(42)
        for _ in range(n_monte_carlo_trials):
            sub_indices = []
            for cl in unique_clusters:
                cl_members = np.where(cluster_labels == cl)[0]
                sample_k = min(len(cl_members), rng.randint(1, 5))
                sub_indices.extend(rng.choice(cl_members, size=sample_k, replace=False))
            sub_indices = np.array(sub_indices, dtype=np.int32)

            sub_y = y_true[sub_indices]
            sub_raw = raw_preds[sub_indices]
            sub_raw_bacc = balanced_accuracy_score(sub_y, sub_raw)

            sub_guarded, _ = assigner.assign(
                [texts[i] for i in sub_indices],
                raw_probs[sub_indices],
                embeddings=embs[sub_indices],
            )
            sub_guard_bacc = balanced_accuracy_score(sub_y, sub_guarded)
            sparse_lifts.append(sub_guard_bacc - sub_raw_bacc)

    mean_sparse_lift = float(np.mean(sparse_lifts)) if sparse_lifts else 0.0

    delta_bacc = guarded_bacc - raw_bacc
    results = {
        "sim_threshold": sim_threshold,
        "n_clusters": len(unique_clusters),
        "n_singletons": n_singletons,
        "n_multi_clusters": n_multi,
        "mean_cluster_size": mean_cluster_size,
        "raw_bacc": float(raw_bacc),
        "guarded_bacc": float(guarded_bacc),
        "delta_bacc": float(delta_bacc),
        "sparse_mean_lift": mean_sparse_lift,
    }

    logger.info(
        f"Threshold: {sim_threshold:.2f} | Clusters: {len(unique_clusters)} ({n_singletons} singletons) | "
        f"Raw BAcc: {raw_bacc:.4f} -> Guarded BAcc: {guarded_bacc:.4f} (Lift: {delta_bacc:+.4f}) | "
        f"Sparse Lift: {mean_sparse_lift:+.4f}"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Guarded Hungarian Assignment on OOF probabilities")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--oof_probs", type=str, default=None, help="Path to oof probabilities .npy")
    parser.add_argument("--sim_threshold", type=float, default=0.65, help="Semantic cosine similarity threshold")
    parser.add_argument("--tune", action="store_true", help="Audit thresholds in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    texts = df["text"].fillna("").astype(str).tolist()
    y_true = np.array([LABEL2ID[m] for m in df["model"].tolist()], dtype=np.int32)

    if args.oof_probs and Path(args.oof_probs).exists():
        raw_probs = np.load(args.oof_probs)
    else:
        # Check standard checkpoints
        cand = Path("./checkpoints/oof_stage4_probabilities.npy")
        if not cand.exists():
            cand = Path("./artifacts/oof_stage4_probabilities.npy")
        if cand.exists():
            raw_probs = np.load(cand)
        else:
            raise FileNotFoundError("Please provide valid --oof_probs path.")

    embedder = PromptSemanticEmbedder()
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80] if args.tune else [args.sim_threshold]

    best_thresh = args.sim_threshold
    best_lift = -999.0

    print("\n" + "=" * 90)
    print("GUARDED HUNGARIAN ASSIGNMENT AUDIT:")
    print("=" * 90)
    for t in thresholds:
        res = evaluate_assignment(y_true, raw_probs, texts, sim_threshold=t, embedder=embedder)
        if res["delta_bacc"] > best_lift:
            best_lift = res["delta_bacc"]
            best_thresh = t

    print(f"\nRecommended optimal similarity threshold: {best_thresh:.2f} (Lift: {best_lift:+.4f})")


if __name__ == "__main__":
    main()
