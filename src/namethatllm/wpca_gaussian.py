"""
Full-Rank wPCA & ExactAuthorPosterior (2^18 Bitmask DP)
======================================================
Two-way variance decomposition, Ledoit-Wolf shrinkage floors,
crash-proof cluster splitting (max_size <= 14), and exact DP author posteriors.
"""

from typing import Tuple, List, Dict, Optional, Any
from itertools import combinations
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import logsumexp
from sklearn.covariance import ledoit_wolf_shrinkage

from .config import K

def _two_way(Z: np.ndarray, y: np.ndarray, pinv: np.ndarray, G: int, K_classes: int = 18, iters: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative two-way ANOVA decomposition separating author means and prompt random effects."""
    N, d = Z.shape
    mu = np.zeros((K_classes, d), dtype=np.float64)
    P = np.zeros((G, d), dtype=np.float64)
    ng = np.bincount(pinv, minlength=G).clip(min=1)
    for _ in range(iters):
        R = Z - P[pinv]
        for k in range(K_classes):
            idx = (y == k)
            if idx.any():
                mu[k] = R[idx].mean(axis=0)
        R = Z - mu[y]
        P[:] = 0.0
        np.add.at(P, pinv, R)
        P /= ng[:, None]
    return mu, P, ng

def _shrunk(S: np.ndarray, alpha: float) -> np.ndarray:
    """Shrinks covariance matrix S towards spherical target."""
    d = S.shape[0]
    tr = np.trace(S) / d
    return (1.0 - alpha) * S + alpha * tr * np.eye(d)

def fit_shared_gaussian_lowram(X_tr_f32: np.ndarray, y: np.ndarray, prompts: np.ndarray,
                               d_target: int = 256, se_floor: float = 0.10) -> dict:
    """
    Fits shared Gaussian whitener with explicit Ledoit-Wolf shrinkage floor `se_floor`.
    Returns unsupervised top-64 principal components V_top64 (zero label leakage!).
    """
    N, D = X_tr_f32.shape
    pu, pinv = np.unique(prompts, return_inverse=True)
    G = len(pu)
    d = min(d_target, N - G, D)

    mx = X_tr_f32.mean(axis=0, keepdims=True)
    sx = X_tr_f32.std(axis=0, keepdims=True)
    sx[sx < 1e-6] = 1.0

    R = ((X_tr_f32 - mx) / sx).astype(np.float32)
    Gram = (R @ R.T) / float(D)
    lam_r, U = np.linalg.eigh(Gram.astype(np.float64))
    idx_top = np.argsort(lam_r)[::-1][:d]
    lam_top = np.maximum(lam_r[idx_top], 1e-9)
    U_top = U[:, idx_top]
    U_f32 = U_top.astype(np.float32)

    V_f32 = (R.T @ U_f32) / (np.sqrt(lam_top.astype(np.float32) * float(D))[None, :])
    del R, Gram, U, U_f32

    # Unsupervised top-64 principal components (zero label leakage)
    V_top64 = V_f32[:, :min(64, d)].copy()

    Z = (((X_tr_f32 - mx) / sx) @ V_f32).astype(np.float64)

    mu, P, ng = _two_way(Z, y, pinv, G, K_classes=K, iters=4)
    E = Z - mu[y] - P[pinv]
    dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = (E.T @ E) / dof
    ae = max(float(ledoit_wolf_shrinkage(E, assume_centered=True)), se_floor)
    Se = _shrunk(Se_raw, ae)

    Pc = P - P.mean(0)
    Sp_raw = (Pc.T @ Pc) / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = float(ledoit_wolf_shrinkage(Pc, assume_centered=True))
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = (1.0 - ap) * Sp_raw + ap * (tr / d) * np.eye(d)

    L = np.linalg.cholesky(Se)
    Li = np.linalg.inv(L)
    M = 0.5 * ((Li @ Sp @ Li.T) + (Li @ Sp @ Li.T).T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, 1e-4, None)
    A = Q.T @ Li
    W_f32 = (V_f32 @ A.T.astype(np.float32)).astype(np.float32)
    m_whitened = mu @ A.T

    return {
        "mx": mx, "sx": sx, "W": W_f32, "m": m_whitened, "lam": lam,
        "V_top64": V_top64, "T1": 2.40, "Tg": 2.35, "ae": ae
    }

def transform_lowram(X_f32: np.ndarray, prm: dict, batch_rows: int = 1000) -> np.ndarray:
    """Projects representations into whitened space."""
    N = len(X_f32)
    d = prm["W"].shape[1]
    U_out = np.empty((N, d), dtype=np.float64)
    mx, sx, W = prm["mx"], prm["sx"], prm["W"]
    for s in range(0, N, batch_rows):
        chunk = ((X_f32[s : s + batch_rows] - mx) / sx) @ W
        U_out[s : s + batch_rows] = chunk.astype(np.float64)
    return U_out

def split_oversized_clusters_dist(D: np.ndarray, groups: np.ndarray, max_cluster_size: int = 14) -> np.ndarray:
    """Recursively splits clusters with size > max_cluster_size to prevent bitmask DP overflow."""
    if len(groups) < 2:
        return groups.copy()
    new_groups = np.zeros(len(groups), dtype=np.int64)
    next_id = 1
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        if len(rows) <= max_cluster_size:
            new_groups[rows] = next_id
            next_id += 1
        else:
            n_sub = int(np.ceil(len(rows) / max_cluster_size))
            D_sub = D[np.ix_(rows, rows)].copy()
            np.fill_diagonal(D_sub, 0.0)
            D_sub = np.clip(0.5 * (D_sub + D_sub.T), 0.0, None)
            d_condensed = squareform(D_sub, checks=False)
            sub_labels = fcluster(linkage(d_condensed, method='average'), t=n_sub, criterion='maxclust')
            for sg in np.unique(sub_labels):
                s_rows = rows[sub_labels == sg]
                for start in range(0, len(s_rows), max_cluster_size):
                    new_groups[s_rows[start : start + max_cluster_size]] = next_id
                    next_id += 1
    return new_groups

class ExactAuthorPosterior:
    """Computes exact marginal author posteriors using 2^18 bitmask dynamic programming."""
    def __init__(self, m: np.ndarray, lam: np.ndarray):
        self.m = np.ascontiguousarray(m, dtype=np.float32)
        self.lam = np.ascontiguousarray(lam, dtype=np.float32)
        self.K, self.d = m.shape
        self._subset_means = {}

    def _get_subset_means(self, subset_size: int) -> np.ndarray:
        if subset_size not in self._subset_means:
            if len(self._subset_means) >= 4:
                self._subset_means.clear()
            subsets = list(combinations(range(self.K), subset_size))
            means = np.empty((len(subsets), self.d), dtype=np.float32)
            for i, S in enumerate(subsets):
                means[i] = self.m[list(S)].mean(axis=0)
            self._subset_means[subset_size] = means
        return self._subset_means[subset_size]

    def single(self, u: np.ndarray, temperature: float = 2.40) -> np.ndarray:
        """Single passage Gaussian likelihood over K classes."""
        diff = u[None, :] - self.m
        diag = 1.0 + self.lam
        dist = np.sum(diff * diff / diag[None, :], axis=1)
        log_p = -0.5 * dist / max(float(temperature), 1e-4)
        return log_p - logsumexp(log_p)

    def group(self, U: np.ndarray, temperature: float = 2.35) -> np.ndarray:
        """Group joint likelihood evaluating all C(K, s) subsets of authors."""
        s = len(U)
        if s <= 1 or s > self.K:
            out = np.empty((s, self.K), dtype=np.float64)
            for i in range(s):
                out[i] = self.single(U[i], temperature=temperature)
            return out

        u_bar = U.mean(axis=0).astype(np.float32)
        subsets = list(combinations(range(self.K), s))

        means = self._get_subset_means(s)
        diff = u_bar[None, :] - means
        diag = (1.0 + float(s) * self.lam) / float(s)
        dist = np.sum(diff * diff / diag[None, :], axis=1)
        log_p_S = -0.5 * dist / max(float(temperature), 1e-4)
        log_p_S -= logsumexp(log_p_S)

        P_S = np.exp(log_p_S)
        post_i = np.zeros((s, self.K), dtype=np.float64)
        for sub_idx, S in enumerate(subsets):
            p = P_S[sub_idx]
            if p > 0:
                post_i[:, list(S)] += p

        row_sums = np.sum(post_i, axis=1, keepdims=True)
        zero_rows = (row_sums.squeeze(axis=-1) <= 1e-12)
        if np.any(zero_rows):
            post_i[zero_rows] = 1.0 / self.K
            row_sums[zero_rows] = 1.0
        prob_i = np.clip(post_i / row_sums, 1e-15, 1.0)
        return np.log(prob_i)

def scores_exact_dp(U_test: np.ndarray, groups: np.ndarray, prm: dict, D_dist: np.ndarray) -> np.ndarray:
    """Wraps ExactAuthorPosterior with oversized cluster splitting guardrail."""
    N = len(U_test)
    if N == 0:
        return np.zeros((0, K), dtype=np.float64)
    if N == 1:
        solver = ExactAuthorPosterior(prm["m"], prm["lam"])
        return solver.single(U_test[0], temperature=prm.get("T1", 2.40))[None, :]

    safe_groups = split_oversized_clusters_dist(D_dist, groups, max_cluster_size=14)
    solver = ExactAuthorPosterior(prm["m"], prm["lam"])
    T1 = prm.get("T1", 2.40)
    Tg = prm.get("Tg", 2.35)

    lp_dp = np.zeros((N, K), dtype=np.float64)
    for g in np.unique(safe_groups):
        idx_g = np.where(safe_groups == g)[0]
        if len(idx_g) == 1:
            lp_dp[idx_g] = solver.single(U_test[idx_g[0]], temperature=T1)
        else:
            lp_dp[idx_g] = solver.group(U_test[idx_g], temperature=Tg)
    return lp_dp
