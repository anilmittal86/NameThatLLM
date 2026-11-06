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
    apply_family_specialist_lora_gate,
    apply_all_family_confidence_gates,
    apply_cluster_uniqueness_repulsion,
    apply_unified_handwriting_shield,
    build_orthographic_penalty_matrix,
    apply_curly_quote_shield,
    sinkhorn_balancing,
    apply_typobayes_postprocess,
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

    def test_typobayes_postprocess(self):
        train_texts = [
            "Normal sample without marks",
            "Sample with — spaced em dash",
            "Sample with “curly quotes”",
            "Sample with ‘apostrophe’",
        ]
        y_train = np.array([0, 1, 2, 3])
        test_texts = [
            "Test text with — spaced em dash",
            "Test text with “curly quotes”",
        ]
        P_in = np.ones((len(test_texts), 18)) / 18.0
        P_out = apply_typobayes_postprocess(P_in, train_texts, y_train, test_texts)
        self.assertEqual(P_out.shape, (len(test_texts), 18))
        self.assertTrue(np.allclose(P_out.sum(axis=1), 1.0))
        # P_out should shift slightly based on typographical evidence
        self.assertFalse(np.allclose(P_in, P_out))

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

    def test_apply_all_family_confidence_gates_preserves_simplex(self):
        np.random.seed(42)
        N = 25
        P_in = softmax(np.random.randn(N, 18), axis=1)
        family_gates = {
            "gpt": np.random.randn(N, len(FAMILIES["gpt"])),
            "gemini": np.random.randn(N, len(FAMILIES["gemini"])),
            "claude": np.random.randn(N, len(FAMILIES["claude"])),
            "glm": np.random.randn(N, len(FAMILIES["glm"])),
        }
        P_out = apply_all_family_confidence_gates(P_in, family_gates, max_strength=0.20)
        self.assertEqual(P_out.shape, (N, 18))
        self.assertTrue(np.allclose(P_out.sum(axis=1), 1.0, atol=1e-6))
        # Total mass for unaffected columns must be preserved
        kimi_col = [17]
        self.assertTrue(np.allclose(P_in[:, kimi_col], P_out[:, kimi_col], atol=1e-5))

    def test_apply_all_family_confidence_gates_with_specialist_groups(self):
        from namethatllm.config import SPECIALIST_GROUPS
        np.random.seed(42)
        N = 30
        P_in = softmax(np.random.randn(N, 18), axis=1)
        gates = {k: np.random.randn(N, len(v)) for k, v in SPECIALIST_GROUPS.items()}
        P_out = apply_all_family_confidence_gates(P_in, gates, max_strength=0.20)
        self.assertEqual(P_out.shape, (N, 18))
        self.assertTrue(np.allclose(P_out.sum(axis=1), 1.0, atol=1e-6))

    def test_apply_cluster_uniqueness_repulsion_preserves_simplex(self):
        np.random.seed(42)
        N = 20
        P_in = softmax(np.random.randn(N, 18), axis=1)
        # Cluster of 5 items, cluster of 3 items, and singletons
        groups = np.array([1, 1, 1, 1, 1, 2, 2, 2] + list(range(3, 15)))
        P_out = apply_cluster_uniqueness_repulsion(P_in, groups, strength=0.35, iters=3)
        self.assertEqual(P_out.shape, (N, 18))
        self.assertTrue(np.allclose(P_out.sum(axis=1), 1.0, atol=1e-6))

    def test_soft_prior_shrunk_sinkhorn_preserves_simplex(self):
        np.random.seed(42)
        N = 100
        P_in = softmax(np.random.randn(N, 18), axis=1)
        train_prior = np.ones(18)
        P_out = sinkhorn_balancing(P_in, iters=100, sinkhorn_strength=0.65, target_mix=0.35, train_prior=train_prior)
        self.assertEqual(P_out.shape, (N, 18))
        self.assertTrue(np.allclose(P_out.sum(axis=1), 1.0, atol=1e-6))

    def test_family_specialist_lora_gate_mass_preservation_and_sibling_swaps(self):
        np.random.seed(42)
        N = 25

        # Test all 3 families: GPT (5 classes), Gemini (4 classes), GLM (2 classes)
        for fam_name, default_gate_w in [("gpt", 0.55), ("gemini", 0.65), ("glm", 0.70)]:
            fam_cols = FAMILIES[fam_name]
            K_fam = len(fam_cols)
            fam_labels = [LABELS[c] for c in fam_cols]

            P_mix = softmax(np.random.randn(N, 18), axis=1)
            R_main = P_mix.copy()

            # Assign row 0 to be clearly predicted in the family
            R_main[0, :] = 0.01
            R_main[0, fam_cols[0]] = 0.80

            # Specialist logits favoring sibling index 1 over index 0
            spec_logits = np.zeros((N, K_fam), dtype=np.float32)
            spec_logits[0, 1] = 5.0
            spec_logits[0, 0] = -5.0

            orig_fam_mass = P_mix[:, fam_cols].sum(axis=1)
            orig_outer_p = P_mix.copy()

            P_refined = apply_family_specialist_lora_gate(
                P_mix=P_mix,
                R_main=R_main,
                spec_logits=spec_logits,
                spec_classes=fam_labels,
                labels=LABELS,
                family_cols=fam_cols,
                gate_weight=default_gate_w,
            )

            # 1. Simplex preservation
            self.assertTrue(np.allclose(P_refined.sum(axis=1), 1.0, atol=1e-6))
            self.assertTrue((P_refined >= 0.0).all())

            # 2. Strict family mass preservation
            new_fam_mass = P_refined[:, fam_cols].sum(axis=1)
            self.assertTrue(np.allclose(orig_fam_mass, new_fam_mass, atol=1e-6))

            # 3. Outer non-family classes completely unchanged
            outer_cols = [c for c in range(18) if c not in fam_cols]
            self.assertTrue(np.allclose(P_mix[:, outer_cols], P_refined[:, outer_cols], atol=1e-6))

            # 4. Sibling swap resolution: row 0 favored index 1 in specialist logits
            self.assertGreater(P_refined[0, fam_cols[1]], P_refined[0, fam_cols[0]])

if __name__ == "__main__":
    unittest.main()

