import unittest
import numpy as np
import json
import ast
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

def align_cols(arr: np.ndarray, src_labels: list) -> np.ndarray:
    src_list = [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in src_labels]
    idx = [src_list.index(l) for l in LABELS]
    if arr.ndim == 1:
        return arr[idx]
    elif arr.ndim == 2:
        if arr.shape[0] == len(src_list) and arr.shape[1] != len(src_list):
            return arr[idx, :]
        elif arr.shape[1] == len(src_list):
            return arr[:, idx]
    return arr[..., idx]

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)

def sinkhorn(P: np.ndarray, iters: int = 300) -> np.ndarray:
    if len(P) < 50:
        return P
    P = np.clip(P.astype(np.float64), 1e-15, 1.0)
    c = np.ones(P.shape[1], dtype=np.float64)
    target = np.full(P.shape[1], len(P) / float(P.shape[1]), dtype=np.float64)
    for _ in range(iters):
        R = P * c
        R /= R.sum(axis=1, keepdims=True).clip(min=1e-15)
        c *= target / R.sum(axis=0).clip(min=1e-15)
    R = P * c
    return R / R.sum(axis=1, keepdims=True).clip(min=1e-15)

def split_oversized_clusters_dist(D: np.ndarray, groups: np.ndarray, max_cluster_size: int = 14) -> np.ndarray:
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

def compute_centered_logits(Z_test: np.ndarray, groups: np.ndarray, W_c: np.ndarray, b_c: np.ndarray, src_labels: list):
    N = len(Z_test)
    out_lp = np.zeros((N, K), dtype=np.float64)
    if N < 2 or W_c is None:
        return out_lp, np.zeros(N, dtype=bool)
    sz = np.bincount(groups)[groups]
    valid = (sz >= 2) & (sz <= 18)
    if not valid.any():
        return out_lp, valid
    Zc = Z_test.copy()
    for g in np.unique(groups[valid]):
        mask = (groups == g)
        Zc[mask] -= Zc[mask].mean(axis=0, keepdims=True)
    raw_c = Zc[valid] @ W_c.T + b_c
    out_lp[valid] = log_softmax(align_cols(raw_c, src_labels))
    return out_lp, valid


class TestVersion37Pipeline(unittest.TestCase):
    def test_sinkhorn_exact_mass_equalization(self):
        np.random.seed(42)
        N = 200
        logits = np.random.randn(N, 18)
        P = softmax(logits, axis=1)
        P_sink = sinkhorn(P, iters=300)

        # Row simplex invariant
        self.assertTrue(np.allclose(P_sink.sum(axis=1), 1.0))
        # Column marginal mass equality
        target = N / 18.0
        col_sums = P_sink.sum(axis=0)
        self.assertTrue(np.allclose(col_sums, target, atol=1e-10))

    def test_split_oversized_clusters_dist(self):
        np.random.seed(42)
        N = 100
        D = np.random.uniform(0.1, 0.9, size=(N, N))
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
        # Giant clusters of size 40 and 60
        groups = np.array([1] * 40 + [2] * 60)
        new_groups = split_oversized_clusters_dist(D, groups, max_cluster_size=14)

        max_size = np.bincount(new_groups).max()
        self.assertLessEqual(max_size, 14)
        self.assertGreater(len(np.unique(new_groups)), 2)
        self.assertEqual(len(new_groups), N)

    def test_loo_clc_and_soft_hungarian(self):
        np.random.seed(42)
        N = 15
        LOGIT = log_softmax(np.random.randn(N, 18))
        groups_tight = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9])
        LOGIT_debiased = LOGIT.copy()

        for g in np.unique(groups_tight):
            rows = np.flatnonzero(groups_tight == g)
            m = len(rows)
            if 2 <= m <= 18:
                L_grp = LOGIT[rows]
                sum_L = L_grp.sum(axis=0, keepdims=True)
                loo_mean = (sum_L - L_grp) / float(m - 1)
                L_contrasted = log_softmax(L_grp - 0.22 * loo_mean)
                row_ind, col_ind = linear_sum_assignment(-L_contrasted)
                L_contrasted[row_ind, col_ind] += 0.25
                LOGIT_debiased[rows] = log_softmax(L_contrasted)

        self.assertEqual(LOGIT_debiased.shape, (N, 18))
        self.assertFalse(np.isnan(LOGIT_debiased).any())
        self.assertTrue(np.allclose(np.exp(LOGIT_debiased).sum(axis=1), 1.0))

    def test_align_cols_shapes(self):
        # 1D bias
        b = np.arange(18, dtype=float)
        b_rev = align_cols(b, list(reversed(LABELS)))
        self.assertEqual(len(b_rev), 18)
        self.assertEqual(b_rev[0], 17.0)

        # 2D weight matrix (18, 64)
        W = np.random.randn(18, 64)
        W_aligned = align_cols(W, list(reversed(LABELS)))
        self.assertEqual(W_aligned.shape, (18, 64))
        self.assertTrue(np.allclose(W_aligned[0], W[17]))

        # 2D logits matrix (10, 18)
        L = np.random.randn(10, 18)
        L_aligned = align_cols(L, list(reversed(LABELS)))
        self.assertEqual(L_aligned.shape, (10, 18))
        self.assertTrue(np.allclose(L_aligned[:, 0], L[:, 17]))

    def test_compute_centered_logits(self):
        np.random.seed(42)
        N = 8
        D = 32
        Z = np.random.randn(N, D)
        W = np.random.randn(K, D)
        b = np.random.randn(K)
        groups = np.array([1, 1, 1, 2, 2, 3, 4, 5])
        out_lp, valid = compute_centered_logits(Z, groups, W, b, LABELS)
        self.assertEqual(valid.sum(), 5)
        self.assertEqual(out_lp.shape, (N, 18))
        self.assertTrue(np.all(out_lp[~valid] == 0.0))

    def test_notebook_ast_and_subprocess_syntax(self):
        nb_path = Path("kaggle_submission.ipynb")
        self.assertTrue(nb_path.exists())
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()

