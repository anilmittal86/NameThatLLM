"""
Unit and Integration Tests for namethatllm.spectral_bank
========================================================
"""

import unittest
import numpy as np

from namethatllm.spectral_bank import fit_joint_spectral_bank, make_spectral_blocks
from namethatllm.wpca_gaussian import fit_shared_gaussian_lowram
from namethatllm.config import K, PRIOR_SCALES

class TestSpectralBank(unittest.TestCase):
    def test_make_spectral_blocks(self):
        np.random.seed(42)
        N_tr, N_te, D = 180, 18, 128
        Z_tr = np.random.randn(N_tr, D).astype(np.float32)
        Z_te = np.random.randn(N_te, D).astype(np.float32)
        prompts = np.repeat(np.arange(10), 18)
        y = np.tile(np.arange(18), 10)
        res = fit_shared_gaussian_lowram(Z_tr, y, prompts, d_target=64, se_floor=0.10)

        S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res, prior_scale=0.85)
        self.assertEqual(S_tr.shape, (N_tr, 64))
        self.assertEqual(S_te.shape, (N_te, 64))
        self.assertTrue(np.allclose(np.linalg.norm(S_tr, axis=1), 0.85, atol=1e-5))
        self.assertTrue(np.allclose(np.linalg.norm(S_te, axis=1), 0.85, atol=1e-5))

    def test_unsupervised_v_top64_zero_leakage(self):
        """Proof that unsupervised principal components are identical regardless of labels y."""
        np.random.seed(42)
        N = 180
        D = 128
        X = np.random.randn(N, D).astype(np.float32)
        prompts = np.repeat(np.arange(10), 18)
        y_true = np.tile(np.arange(18), 10)
        y_shuffled = np.random.permutation(y_true)

        prm_true = fit_shared_gaussian_lowram(X, y_true, prompts, d_target=64, se_floor=0.10)
        prm_shuffled = fit_shared_gaussian_lowram(X, y_shuffled, prompts, d_target=64, se_floor=0.10)

        # Unsupervised top-64 projection MUST NOT depend on labels!
        self.assertTrue(np.allclose(prm_true["V_top64"], prm_shuffled["V_top64"], atol=1e-6))

    def test_prior_scaling_normalization(self):
        np.random.seed(42)
        N = 50
        d = 64
        Z = np.random.randn(N, d).astype(np.float32)
        scale = PRIOR_SCALES["qwen25_14b_p2"]  # 0.85
        Z_scaled = (Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-6)) * scale

        row_norms = np.linalg.norm(Z_scaled, axis=1)
        self.assertTrue(np.allclose(row_norms, scale, atol=1e-5))

    def test_fit_joint_spectral_bank_execution(self):
        np.random.seed(42)
        N_tr = 90
        N_te = 20
        d = 64

        tr_blocks = [np.random.randn(N_tr, d).astype(np.float32) for _ in range(3)]
        te_blocks = [np.random.randn(N_te, d).astype(np.float32) for _ in range(3)]

        prompts_sub = np.repeat(np.arange(5), 18)
        y_sub = np.tile(np.arange(18), 5)

        D_tfidf = np.random.uniform(0.1, 0.9, size=(N_te, N_te))
        np.fill_diagonal(D_tfidf, 0.0)
        D_neural = D_tfidf.copy()

        P_spec, oof_bacc = fit_joint_spectral_bank(
            tr_blocks, te_blocks, y_sub, prompts_sub,
            D_tfidf=D_tfidf, D_neural_sum=D_neural, D_neural_weight_total=1.0,
            N_test=N_te, plain_C=0.04, center_C=0.03, center_blend=0.35
        )

        self.assertEqual(P_spec.shape, (N_te, 18))
        self.assertTrue(np.allclose(P_spec.sum(axis=1), 1.0))
        self.assertGreater(oof_bacc, 0.0)

if __name__ == "__main__":
    unittest.main()

