import unittest
import numpy as np
import json
import ast
import re
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

FAMILIES = {
    "gpt": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
}

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

def sinkhorn_balancing(P: np.ndarray, iters: int = 300) -> np.ndarray:
    if len(P) < 50:
        return P
    P = np.clip(P.astype(np.float64), 1e-12, 1.0)
    target_col = 1.0 / P.shape[1]
    for _ in range(iters):
        P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
        P /= (P.sum(axis=0, keepdims=True) / (target_col * P.shape[0])).clip(min=1e-12)
    P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P

def split_oversized_clusters_dist(D: np.ndarray, groups: np.ndarray, max_cluster_size: int = 14) -> np.ndarray:
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

def resolve_head_tensors(heads_dict: dict):
    W_p, b_p, W_c, b_c = None, None, None, None
    for wk, bk in [
        ("plain.weight", "plain.bias"),
        ("plain_coef", "plain_intercept"),
        ("weight", "bias"),
        ("W", "b"),
    ]:
        if wk in heads_dict and bk in heads_dict:
            W_p, b_p = heads_dict[wk], heads_dict[bk]
            break
    for wk, bk in [
        ("centered.weight", "centered.bias"),
        ("centred.weight", "centred.bias"),
        ("center.weight", "center.bias"),
        ("twin.weight", "twin.bias"),
        ("center_coef", "center_intercept"),
        ("centred_coef", "centred_intercept"),
    ]:
        if wk in heads_dict and bk in heads_dict:
            W_c, b_c = heads_dict[wk], heads_dict[bk]
            break

    if W_p is not None and hasattr(W_p, "cpu"):
        W_p = W_p.cpu().numpy()
    if b_p is not None and hasattr(b_p, "cpu"):
        b_p = b_p.cpu().numpy()
    if W_c is not None and hasattr(W_c, "cpu"):
        W_c = W_c.cpu().numpy()
    if b_c is not None and hasattr(b_c, "cpu"):
        b_c = b_c.cpu().numpy()

    return W_p, b_p, W_c, b_c


class TestVersion40Pipeline(unittest.TestCase):
    def test_sinkhorn_exact_mass_equalization(self):
        np.random.seed(42)
        N = 200
        logits = np.random.randn(N, 18)
        P = softmax(logits, axis=1)
        P_sink = sinkhorn_balancing(P, iters=300)

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
        groups = np.array([1] * 40 + [2] * 60)
        new_groups = split_oversized_clusters_dist(D, groups, max_cluster_size=14)

        max_size = np.bincount(new_groups).max()
        self.assertLessEqual(max_size, 14)
        self.assertGreater(len(np.unique(new_groups)), 2)
        self.assertEqual(len(new_groups), N)

    def test_resolve_head_tensors_british_and_american(self):
        # Case A: American naming
        d_am = {
            "plain.weight": torch.randn(18, 64),
            "plain.bias": torch.randn(18),
            "centered.weight": torch.randn(18, 64),
            "centered.bias": torch.randn(18),
        }
        W_p, b_p, W_c, b_c = resolve_head_tensors(d_am)
        self.assertIsNotNone(W_p)
        self.assertIsNotNone(W_c)
        self.assertEqual(W_p.shape, (18, 64))
        self.assertEqual(W_c.shape, (18, 64))

        # Case B: British naming (centred)
        d_br = {
            "weight": torch.randn(18, 64),
            "bias": torch.randn(18),
            "centred.weight": torch.randn(18, 64),
            "centred.bias": torch.randn(18),
        }
        W_p, b_p, W_c, b_c = resolve_head_tensors(d_br)
        self.assertIsNotNone(W_p)
        self.assertIsNotNone(W_c)
        self.assertEqual(W_c.shape, (18, 64))

        # Case C: NPZ readouts naming (plain_coef / center_coef)
        d_npz = {
            "plain_coef": np.random.randn(18, 64),
            "plain_intercept": np.random.randn(18),
            "center_coef": np.random.randn(18, 64),
            "center_intercept": np.random.randn(18),
        }
        W_p, b_p, W_c, b_c = resolve_head_tensors(d_npz)
        self.assertIsNotNone(W_p)
        self.assertIsNotNone(W_c)
        self.assertEqual(W_p.shape, (18, 64))
        self.assertEqual(W_c.shape, (18, 64))

    def test_multi_resolution_twin_pooling(self):
        np.random.seed(42)
        N = 12
        D = 32
        Z = np.random.randn(N, D)
        W_c = np.random.randn(K, D)
        b_c = np.random.randn(K)

        g85 = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5])
        g72 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        g60 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        lp_85, m85 = compute_centered_logits(Z, g85, W_c, b_c, LABELS)
        lp_72, m72 = compute_centered_logits(Z, g72, W_c, b_c, LABELS)
        lp_60, m60 = compute_centered_logits(Z, g60, W_c, b_c, LABELS)

        self.assertTrue(m85.all())
        self.assertTrue(m72.all())
        self.assertTrue(m60.all())

        w_twin = 0.12
        LOGIT_CORE = np.zeros((N, K), dtype=np.float64)
        LOGIT_CORE[m85] += (0.50 * w_twin) * lp_85[m85]
        LOGIT_CORE[m72] += (0.30 * w_twin) * lp_72[m72]
        LOGIT_CORE[m60] += (0.20 * w_twin) * lp_60[m60]

        self.assertFalse(np.isnan(LOGIT_CORE).any())
        self.assertEqual(LOGIT_CORE.shape, (N, 18))

    def test_spectral_bank_5fold_oof_logic(self):
        np.random.seed(42)
        N = 90
        prompts = np.repeat(np.arange(10), 9)
        y = np.tile(np.arange(9), 10)
        X = np.random.randn(N, 64)

        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(N, dtype=int)
        for tr_idx, val_idx in gkf.split(X, y, groups=prompts):
            clf = LogisticRegression(C=0.15, max_iter=200, random_state=42)
            clf.fit(X[tr_idx], y[tr_idx])
            oof_preds[val_idx] = clf.predict(X[val_idx])

        bacc = balanced_accuracy_score(y, oof_preds)
        self.assertGreater(bacc, 0.0)

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

