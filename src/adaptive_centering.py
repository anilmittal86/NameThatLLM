"""
Adaptive Centering and High-Precision Semantic Clustering with Strict Singleton Protection.
Implements:
1. High-precision cosine prompt clustering on E_semantic (AgglomerativeClustering with tuned distance_threshold tau).
2. Multi-scale mini-chunk training (k in [2, 3, 4]) to learn invariant authorship representations.
3. Dedicated centered classifier heads fitted on chunk-demeaned style & semantic features.
4. Strict singleton protection at test time:
   - lambda_i = 0 for singletons (|C(i)| == 1) -> 100% uncentered prediction.
   - lambda_i = lambda_max * min(1.0, (|C(i)| - 1) / 3.0) * mean_sim(i, C(i)) for |C(i)| >= 2.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from scipy.special import log_softmax, softmax
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from src.config import NUM_CLASSES, RANDOM_SEED, TARGET_MODELS


class HighPrecisionPromptClusterer(BaseEstimator):
    """Discovers latent prompt groups using cosine distance on E_semantic."""

    def __init__(
        self,
        sim_threshold: float = 0.75,
        tau: Optional[float] = None,
        max_cluster_size: int = 6,
    ):
        self.sim_threshold = sim_threshold
        self.tau = tau if tau is not None else (1.0 - sim_threshold)
        self.max_cluster_size = max_cluster_size

    def find_clusters(self, E_semantic: np.ndarray) -> np.ndarray:
        """Clusters unit-normalized semantic embeddings."""
        N = E_semantic.shape[0]
        if N <= 1:
            return np.zeros(N, dtype=np.int32)

        norms = np.linalg.norm(E_semantic, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embs = E_semantic / norms

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.tau,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(norm_embs)
        return labels

    def tune_tau(
        self,
        E_semantic_val: np.ndarray,
        val_prompt_ids: np.ndarray,
        tau_candidates: List[float] = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    ) -> float:
        """Tuning tau to maximize Cluster Purity x Coverage on validation prompts."""
        N = len(val_prompt_ids)
        best_tau = 0.35
        best_score = -1.0

        for t in tau_candidates:
            self.tau = t
            labels = self.find_clusters(E_semantic_val)
            unique_labels, counts = np.unique(labels, return_counts=True)

            non_singleton_samples = 0
            purity_sum = 0.0
            non_singleton_clusters = 0

            for lbl, count in zip(unique_labels, counts):
                if count > 1:
                    non_singleton_samples += count
                    non_singleton_clusters += 1
                    members = val_prompt_ids[labels == lbl]
                    # purity: fraction of majority prompt
                    _, p_counts = np.unique(members, return_counts=True)
                    purity = np.max(p_counts) / count
                    purity_sum += purity * count

            coverage = non_singleton_samples / max(1, N)
            mean_purity = (purity_sum / max(1, non_singleton_samples)) if non_singleton_samples > 0 else 0.0
            score = mean_purity * coverage

            if score > best_score:
                best_score = score
                best_tau = t

        self.tau = best_tau
        return best_tau


class MultiScaleCenteredHead(BaseEstimator):
    """Trains dedicated centered classification head on mini-chunk demeaned features."""

    def __init__(
        self,
        C: float = 0.1,
        chunk_sizes: Tuple[int, ...] = (2, 3, 4),
        n_repeats: int = 4,
        random_seed: int = RANDOM_SEED,
    ):
        self.C = C
        self.chunk_sizes = chunk_sizes
        self.n_repeats = n_repeats
        self.random_seed = random_seed
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=500,
            solver="lbfgs",
            random_state=self.random_seed,
        )

    def fit(
        self,
        features: np.ndarray,
        y_model_idx: np.ndarray,
        prompt_ids: np.ndarray,
    ):
        """Partitions training prompt groups into mini-chunks of sizes (2,3,4) and fits centered head."""
        rng = np.random.RandomState(self.random_seed)
        centered_X = []
        centered_y = []

        unique_prompts = np.unique(prompt_ids)

        for rep in range(self.n_repeats):
            for pid in unique_prompts:
                p_indices = np.where(prompt_ids == pid)[0]
                if len(p_indices) < 2:
                    continue

                # Shuffle indices within prompt
                p_perm = rng.permutation(p_indices)
                idx = 0
                while idx < len(p_perm):
                    k = rng.choice(self.chunk_sizes)
                    chunk_idx = p_perm[idx : idx + k]
                    idx += k
                    if len(chunk_idx) < 2:
                        continue

                    chunk_feats = features[chunk_idx]
                    chunk_mean = np.mean(chunk_feats, axis=0, keepdims=True)
                    demeaned = chunk_feats - chunk_mean

                    centered_X.append(demeaned)
                    centered_y.append(y_model_idx[chunk_idx])

        if len(centered_X) > 0:
            X_all = np.vstack(centered_X)
            y_all = np.concatenate(centered_y)
            self.clf.fit(X_all, y_all)
        else:
            self.clf.fit(features, y_model_idx)

        return self

    def predict_proba(self, centered_features: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(centered_features)


class AdaptiveChunkCenterer:
    """Adaptive centering with strict singleton protection and soft log-prob blending."""

    def __init__(self, lambda_max: float = 0.35):
        self.lambda_max = lambda_max

    def refine_cluster_probabilities(
        self,
        base_probs: np.ndarray,
        cluster_labels: np.ndarray,
        E_semantic: np.ndarray,
        features_dense: Optional[np.ndarray] = None,
        centered_head: Optional[MultiScaleCenteredHead] = None,
    ) -> np.ndarray:
        """Refines probabilities using singleton-protected adaptive centering."""
        N = len(base_probs)
        refined_logprobs = np.log(np.maximum(base_probs, 1e-9))
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        size_dict = dict(zip(unique_labels, counts))

        # Compute cosine similarity matrix on E_semantic
        norms = np.linalg.norm(E_semantic, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embs = E_semantic / norms
        sim_matrix = np.dot(norm_embs, norm_embs.T)

        for lbl, k in size_dict.items():
            if k <= 1:
                # STRICT SINGLETON PROTECTION: lambda_i = 0.0 (100% uncentered)
                continue

            indices = np.where(cluster_labels == lbl)[0]

            # Compute internal cohesion / mean similarity to cluster peers for each passage
            for idx in indices:
                peer_sims = [sim_matrix[idx, peer] for peer in indices if peer != idx]
                mean_sim = float(np.mean(peer_sims)) if peer_sims else 0.0

                # Adaptive scaling: lambda_i = lambda_max * min(1.0, (k - 1) / 3.0) * mean_sim
                lambda_i = self.lambda_max * min(1.0, (k - 1.0) / 3.0) * max(0.0, mean_sim)

                if lambda_i <= 0.001:
                    continue

                if centered_head is not None and features_dense is not None:
                    # Centered head probability
                    cluster_mean = np.mean(features_dense[indices], axis=0, keepdims=True)
                    demeaned = features_dense[idx : idx + 1] - cluster_mean
                    centered_p = centered_head.predict_proba(demeaned)[0]
                    centered_lp = np.log(np.maximum(centered_p, 1e-9))

                    refined_logprobs[idx] = (1.0 - lambda_i) * refined_logprobs[idx] + lambda_i * centered_lp
                else:
                    # Soft competition refinement when dedicated head not supplied
                    pass

        refined_probs = softmax(refined_logprobs, axis=1).astype(np.float32)
        return refined_probs


if __name__ == "__main__":
    print("Testing upgraded adaptive_centering.py...")
    N_samples = 40
    dim = 64
    np.random.seed(42)
    embs = np.random.randn(N_samples, dim).astype(np.float32)
    prompts = np.repeat(np.arange(10), 4)

    clusterer = HighPrecisionPromptClusterer(sim_threshold=0.75)
    labels = clusterer.find_clusters(embs)
    print(f"Discovered {len(np.unique(labels))} clusters.")

    tau = clusterer.tune_tau(embs, prompts)
    print(f"Tuned tau: {tau:.2f}")

    head = MultiScaleCenteredHead(C=0.1)
    head.fit(embs, np.random.randint(0, NUM_CLASSES, N_samples), prompts)

    centerer = AdaptiveChunkCenterer(lambda_max=0.35)
    dummy_p = np.random.dirichlet(np.ones(NUM_CLASSES), size=N_samples)
    refined = centerer.refine_cluster_probabilities(dummy_p, labels, embs, embs, head)
    assert refined.shape == (N_samples, NUM_CLASSES)
    assert np.allclose(refined.sum(axis=1), 1.0, atol=1e-5)
    print("adaptive_centering.py verified successfully!")
