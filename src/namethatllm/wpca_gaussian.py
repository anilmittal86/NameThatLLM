"""
Full-Rank wPCA & ExactAuthorPosterior (2^18 Bitmask DP)
======================================================
Two-way variance decomposition, Ledoit-Wolf shrinkage floors,
crash-proof cluster splitting (max_size <= 14), and exact forward-backward
bitmask DP author posteriors under the unique-authorship constraint.
"""

from functools import lru_cache
from typing import Tuple, List, Dict, Optional, Any
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
    Fits shared Gaussian whitener with within-prompt centering and Ledoit-Wolf shrinkage floor.
    Returns unsupervised top-64 within-prompt principal components V_top64 (zero label leakage!).
    """
    N, D = X_tr_f32.shape
    pu, pinv = np.unique(prompts, return_inverse=True)
    G = len(pu)

    mx = X_tr_f32.mean(axis=0, dtype=np.float64).astype(np.float32)
    sx = X_tr_f32.std(axis=0, dtype=np.float64).astype(np.float32)
    sx = np.where(sx > 1e-6, sx, 1.0).astype(np.float32)

    # Standardize and subtract prompt means in-place (Within-Prompt Whitened PCA!)
    R = (X_tr_f32 - mx) / sx
    Pm = np.zeros((G, D), dtype=np.float32)
    np.add.at(Pm, pinv, R)
    Pm /= np.bincount(pinv, minlength=G)[:, None].clip(min=1).astype(np.float32)
    R -= Pm[pinv]
    del Pm

    d = int(min(d_target, max(N - G - 1, 1), D))
    Gram = (R @ R.T).astype(np.float64)
    lam_r, U = np.linalg.eigh(Gram)
    idx_top = np.argsort(lam_r)[::-1][:d]
    lam_top = np.clip(lam_r[idx_top], 1e-10, None)
    U_f32 = U[:, idx_top].astype(np.float32)
    V_f32 = (R.T @ U_f32) / np.sqrt(lam_top).astype(np.float32)[None, :]
    del R, Gram, U, U_f32

    # Unsupervised within-prompt top-64 principal components (zero label leakage)
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

@lru_cache(maxsize=4)
def _bitmask_levels(k: int = 18):
    """Precomputes bitmasks partitioned by population count (number of set bits)."""
    masks = np.arange(1 << k, dtype=np.int64)
    counts = np.fromiter((int(x).bit_count() for x in masks), dtype=np.uint8, count=len(masks))
    return tuple(masks[counts == n] for n in range(k + 1))

class ExactAuthorPosterior:
    """Exact shared-prompt Gaussian posteriors under author uniqueness via 2^18 bitmask DP."""
    def __init__(self, m: np.ndarray, lam: np.ndarray):
        self.m = np.asarray(m, dtype=np.float64)
        self.lam = np.asarray(lam, dtype=np.float64)
        self.K, self.d = self.m.shape
        self.levels = _bitmask_levels(self.K)
        self._subset_means = {}

    def single(self, u: np.ndarray, temperature: float = 2.40) -> np.ndarray:
        diff = u[None, :] - self.m
        dist = np.sum((diff * diff) / (1.0 + self.lam)[None, :], axis=1)
        log_p = -0.5 * dist / max(float(temperature), 1e-4)
        return log_p - logsumexp(log_p)

    def group(self, U: np.ndarray, temperature: float = 2.35) -> np.ndarray:
        U = np.asarray(U, dtype=np.float64)
        n = len(U)
        if n <= 1 or n > self.K:
            return np.vstack([self.single(U[i], temperature=temperature) for i in range(n)])

        if n not in self._subset_means:
            if len(self._subset_means) >= 3:
                self._subset_means.clear()
            masks = self.levels[n]
            membership = ((masks[:, None] >> np.arange(self.K)) & 1).astype(np.float32)
            self._subset_means[n] = membership @ self.m.astype(np.float32)

        terminal = self.levels[n]
        gamma = self.lam / (1.0 + n * self.lam)
        residual = U.sum(axis=0, keepdims=True) - self._subset_means[n].astype(np.float64)
        correction = 0.5 * np.einsum("nd,d,nd->n", residual, gamma, residual) / max(float(temperature), 1e-4)
        correction -= correction.max()

        # Passage-specific unary log-likelihoods A[i, author]
        A = -0.5 * np.sum((U[:, None, :] - self.m[None, :, :]) ** 2, axis=2) / max(float(temperature), 1e-4)
        A -= A.max(axis=1, keepdims=True)

        forward = np.full(1 << self.K, -np.inf, dtype=np.float64)
        forward[0] = 0.0
        for count in range(1, n + 1):
            masks = self.levels[count]
            for author in range(self.K):
                bit = 1 << author
                chosen = masks[(masks & bit) != 0]
                forward[chosen] = np.logaddexp(forward[chosen], forward[chosen ^ bit] + A[count - 1, author])

        backward = np.full_like(forward, -np.inf)
        backward[terminal] = correction
        for count in range(n - 1, -1, -1):
            masks = self.levels[count]
            for author in range(self.K):
                bit = 1 << author
                chosen = masks[(masks & bit) == 0]
                backward[chosen] = np.logaddexp(backward[chosen], A[count, author] + backward[chosen | bit])

        result = np.empty((n, self.K), dtype=np.float64)
        for row in range(n):
            masks = self.levels[row]
            for author in range(self.K):
                bit = 1 << author
                chosen = masks[(masks & bit) == 0]
                result[row, author] = logsumexp(forward[chosen] + A[row, author] + backward[chosen | bit]) - backward[0]

        result -= logsumexp(result, axis=1, keepdims=True)
        return result

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
