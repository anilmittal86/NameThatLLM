"""
Unit and Integration Tests for namethatllm.wpca_gaussian
========================================================
"""

import unittest
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from namethatllm.wpca_gaussian import (
    _two_way,
    _shrunk,
    fit_shared_gaussian_lowram,
    transform_lowram,
    ExactAuthorPosterior,
    split_oversized_clusters_dist,
    scores_exact_dp
)
from namethatllm.config import K

class TestWPCAGaussian(unittest.TestCase):
    def test_two_way_decomposition(self):
        np.random.seed(42)
        N = 90
        d = 32
        G = 5
        Z = np.random.randn(N, d)
        y = np.tile(np.arange(18), 5)
        pinv = np.repeat(np.arange(G), 18)

        mu, P, ng = _two_way(Z, y, pinv, G, K_classes=18, iters=4)
        self.assertEqual(mu.shape, (18, d))
        self.assertEqual(P.shape, (G, d))
        self.assertEqual(len(ng), G)

    def test_shrunk_covariance(self):
        S = np.array([[2.0, 0.5], [0.5, 3.0]])
        S_s = _shrunk(S, 0.2)
        self.assertEqual(S_s.shape, (2, 2))
        tr_orig = np.trace(S)
        tr_shrunk = np.trace(S_s)
        self.assertAlmostEqual(tr_orig, tr_shrunk)

    def test_fit_shared_gaussian_shrinkage_floors(self):
        np.random.seed(42)
        N = 180
        D = 64
        X = np.random.randn(N, D).astype(np.float32)
        prompts = np.repeat(np.arange(10), 18)
        y = np.tile(np.arange(18), 10)

        prm1 = fit_shared_gaussian_lowram(X, y, prompts, d_target=32, se_floor=0.10)
        self.assertGreaterEqual(prm1["ae"], 0.10)
        self.assertEqual(prm1["V_top64"].shape, (D, 32))

        prm2 = fit_shared_gaussian_lowram(X, y, prompts, d_target=32, se_floor=0.35)
        self.assertGreaterEqual(prm2["ae"], 0.35)

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

    def test_exact_author_posterior_dp(self):
        np.random.seed(42)
        d = 16
        m = np.random.randn(K, d).astype(np.float32)
        lam = np.abs(np.random.randn(d)).astype(np.float32)
        solver = ExactAuthorPosterior(m, lam)

        # Single
        u = np.random.randn(d).astype(np.float32)
        lp_single = solver.single(u)
        self.assertEqual(lp_single.shape, (K,))
        self.assertAlmostEqual(np.exp(lp_single).sum(), 1.0, places=5)

        # Group
        U = np.random.randn(4, d).astype(np.float32)
        lp_group = solver.group(U)
        self.assertEqual(lp_group.shape, (4, K))
        row_sums = np.exp(lp_group).sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-4))
        # Differential assertion: distinct passages MUST have distinct marginal author posteriors
        self.assertFalse(np.allclose(lp_group[0], lp_group[1]), "Mean collapse bug detected: rows must not be identical!")
        self.assertFalse(np.allclose(lp_group[1], lp_group[2]), "Mean collapse bug detected: rows must not be identical!")

    def test_scores_exact_dp_guardrail(self):
        np.random.seed(42)
        N = 25
        d = 16
        U_test = np.random.randn(N, d)
        groups = np.ones(N, dtype=int)
        D = np.random.uniform(0.1, 0.9, size=(N, N))
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)

        prm = {
            "m": np.random.randn(K, d).astype(np.float32),
            "lam": np.abs(np.random.randn(d)).astype(np.float32),
            "T1": 2.40,
            "Tg": 2.35
        }

        # Size 25 > 18: must safely split and run without bitmask DP overflow
        lp_dp = scores_exact_dp(U_test, groups, prm, D)
        self.assertEqual(lp_dp.shape, (N, K))
        self.assertFalse(np.isnan(lp_dp).any())

if __name__ == "__main__":
    unittest.main()

