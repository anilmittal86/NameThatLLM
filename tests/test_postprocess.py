"""
Unit and Integration Tests for namethatllm.postprocess
======================================================
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from namethatllm.postprocess import (
    blend_main_and_complementary,
    apply_claude_specialist_gate,
    apply_curly_quote_shield,
    sinkhorn_balancing,
    generate_submission
)
from namethatllm.config import LABELS, K, FAMILIES, softmax

class TestPostprocess(unittest.TestCase):
    def test_two_tier_blend_simplex(self):
        np.random.seed(42)
        N = 30
        logit_main = np.random.randn(N, 18)
        p1 = softmax(np.random.randn(N, 18), axis=1)
        p2 = softmax(np.random.randn(N, 18), axis=1)
        comp_list = [(0.35, p1), (0.20, p2)]

        P_mix, R_main = blend_main_and_complementary(logit_main, comp_list, main_ratio=0.85, comp_ratio=0.15)
        self.assertTrue(np.allclose(R_main.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(P_mix.sum(axis=1), 1.0))
        self.assertTrue((P_mix >= 0.0).all())
        self.assertTrue((P_mix <= 1.0).all())

    def test_claude_specialist_gate_mass_preservation(self):
        np.random.seed(42)
        N = 20
        claude_cols = FAMILIES["claude"]
        P_mix = softmax(np.random.randn(N, 18), axis=1)
        R_main = P_mix.copy()

        # Mock Claude prediction on passage 0
        R_main[0, :] = 0.0
        R_main[0, claude_cols[0]] = 1.0

        raw_cl = np.random.randn(N, len(claude_cols))
        claude_classes = [LABELS[c] for c in claude_cols]

        orig_claude_mass = P_mix[:, claude_cols].sum(axis=1)
        P_refined = apply_claude_specialist_gate(P_mix, R_main, raw_cl, claude_classes, weight=0.15)
        new_claude_mass = P_refined[:, claude_cols].sum(axis=1)

        # Exact total mass preservation within the Claude family
        self.assertTrue(np.allclose(orig_claude_mass, new_claude_mass))
        self.assertTrue(np.allclose(P_refined.sum(axis=1), 1.0))

    def test_curly_quote_shield(self):
        N = 4
        P_mix = np.ones((N, 18)) / 18.0
        test_texts = [
            "Normal plain text without quotes.",
            "Text with \u201ccurly\u201d quotes.",
            "Single \u2018curly\u2019 quote text.",
            "Another standard text."
        ]
        P_shielded = apply_curly_quote_shield(P_mix, test_texts)

        claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
        # Rows 0 and 3 are uniform
        self.assertTrue(np.allclose(P_shielded[0], 1.0 / 18.0))
        self.assertTrue(np.allclose(P_shielded[3], 1.0 / 18.0))

        # Rows 1 and 2: Claude and GLM classes are dampened
        self.assertLess(P_shielded[1, claude_and_glm_idx[0]], P_shielded[1, 0])
        self.assertLess(P_shielded[2, claude_and_glm_idx[0]], P_shielded[2, 0])
        self.assertTrue(np.allclose(P_shielded.sum(axis=1), 1.0))

    def test_sinkhorn_exact_mass_equalization(self):
        np.random.seed(42)
        N = 180
        P = softmax(np.random.randn(N, 18), axis=1)
        P_sink = sinkhorn_balancing(P, iters=300)

        self.assertTrue(np.allclose(P_sink.sum(axis=1), 1.0))
        target_col = N / 18.0
        self.assertTrue(np.allclose(P_sink.sum(axis=0), target_col, atol=1e-10))

    def test_generate_submission_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            P_final = softmax(np.random.randn(10, 18), axis=1)
            test_ids = pd.Series([f"id_{i}" for i in range(10)])

            df_sub, p_path, c_path = generate_submission(P_final, test_ids, tmp_path)
            self.assertEqual(len(df_sub), 10)
            self.assertListEqual(list(df_sub.columns), ["id", "model"])
            self.assertTrue(p_path.exists())
            self.assertTrue(c_path.exists())

            df_read = pd.read_parquet(p_path)
            self.assertEqual(len(df_read), 10)
            self.assertTrue(df_read["model"].isin(LABELS).all())

if __name__ == "__main__":
    unittest.main()

