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
    apply_unified_handwriting_shield,
    build_orthographic_penalty_matrix,
    apply_curly_quote_shield,
    sinkhorn_balancing,
    generate_submission
)
from namethatllm.config import LABELS, K, LAB2I, FAMILIES, softmax

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
        P_mix_before = P_mix.copy()
        P_shielded = apply_curly_quote_shield(P_mix, test_texts)
        self.assertTrue(np.allclose(P_mix, P_mix_before), "apply_curly_quote_shield must not mutate input array in-place")

        claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
        # Rows 0 and 3 are uniform
        self.assertTrue(np.allclose(P_shielded[0], 1.0 / 18.0))
        self.assertTrue(np.allclose(P_shielded[3], 1.0 / 18.0))

        # Rows 1 and 2: Claude and GLM classes are dampened
        self.assertLess(P_shielded[1, claude_and_glm_idx[0]], P_shielded[1, 0])
        self.assertLess(P_shielded[2, claude_and_glm_idx[0]], P_shielded[2, 0])
        self.assertTrue(np.allclose(P_shielded.sum(axis=1), 1.0))

    def test_unified_handwriting_shield_all_rules(self):
        claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
        gemini_idx = FAMILIES["gemini"]
        sol_idx = LAB2I["gpt-5.6-sol"]

        test_texts = [
            "Text with \u201ccurly\u201d quotes.",
            "First \u2014 second \u2014 third.",
            "First \u2014 second.",
            "It's clear that don't works.",
            "```python\nx = 'hello'\ny = 'world'\n```",
        ]
        N = len(test_texts)
        P_mix = np.ones((N, 18)) / 18.0

        # Run with explicit discrete penalties to test all rules
        P_out = apply_unified_handwriting_shield(
            P_mix,
            test_texts,
            log_p_handwriting=None,
            rule_flags=None,
            multi_em_dash_log_penalty=-0.45,
            straight_contr_log_penalty=-0.60,
            curly_contr_log_penalty=-0.40,
            continuous_hw_weight=0.0
        )
        self.assertTrue(np.allclose(P_out.sum(axis=1), 1.0))

        # Check 1: Row 0 has curly quotes -> Claude and GLM dampened
        self.assertLess(P_out[0, claude_and_glm_idx[0]], P_out[0, 0])

        # Check 2: Row 1 has >= 2 prose em-dashes -> Gemini penalized
        self.assertLess(P_out[1, gemini_idx[0]], P_out[1, 0])
        # Row 2 has only 1 em-dash -> Gemini NOT penalized (uniform with class 0)
        self.assertAlmostEqual(P_out[2, gemini_idx[0]], P_out[2, 0])

        # Check 3: Row 3 has >= 2 straight contractions in prose -> gpt-5.6-sol penalized
        self.assertLess(P_out[3, sol_idx], P_out[3, 0])
        # Row 4 straight quotes are inside code block -> gpt-5.6-sol NOT penalized
        self.assertAlmostEqual(P_out[4, sol_idx], P_out[4, 0])

        # Check 4: Continuous log_p_handwriting integration and simplex preservation
        np.random.seed(42)
        mock_lp_hw = np.random.randn(N, 18)
        P_out_cont = apply_unified_handwriting_shield(
            P_mix,
            test_texts,
            log_p_handwriting=mock_lp_hw,
            continuous_hw_weight=0.04
        )
        self.assertTrue(np.allclose(P_out_cont.sum(axis=1), 1.0))
        self.assertFalse(np.allclose(P_out, P_out_cont))

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

