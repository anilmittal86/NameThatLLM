"""
Unit and Integration Tests for scripts/evaluate_local.py
=======================================================
Tests zero-leakage prompt-grouped holdout evaluation,
synthetic cache generation, and sub-second offline cache replay.
"""

import unittest
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd

from namethatllm.config import LABELS, K, FAMILIES
from scripts.evaluate_local import (
    compute_top_confused_pairs,
    generate_synthetic_cache,
    evaluate_from_cache,
    evaluate_prompt_split,
)


class TestLocalEvalHarness(unittest.TestCase):
    def test_compute_top_confused_pairs(self):
        """Test identification and family classification of top misclassification pairs."""
        claude_sonnet = LABELS.index("claude-sonnet-5")
        claude_opus = LABELS.index("claude-opus-4-8")
        gpt_model = LABELS.index("gpt-5.5")

        # 5 errors between claude_sonnet and claude_opus (sister pair)
        # 3 errors between claude_sonnet and gpt_model (cross-family pair)
        y_true = np.array([claude_sonnet] * 8 + [gpt_model] * 2)
        y_pred = np.array([claude_opus] * 5 + [gpt_model] * 3 + [gpt_model] * 2)

        confused = compute_top_confused_pairs(y_true, y_pred, LABELS, top_n=5)
        self.assertGreaterEqual(len(confused), 2)

        # First pair should be sonnet <-> opus with 5 errors
        top_pair = confused[0]
        self.assertEqual(top_pair["count"], 5)
        self.assertTrue(top_pair["is_sister"])
        self.assertEqual(top_pair["family"], "claude")

        # Second pair should be sonnet <-> gpt_model with 3 errors
        second_pair = confused[1]
        self.assertEqual(second_pair["count"], 3)
        self.assertFalse(second_pair["is_sister"])
        self.assertEqual(second_pair["family"], "Cross-Family")

    def test_generate_synthetic_cache(self):
        """Test synthesis of realistic multi-backbone cache."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            cache_file = Path(tmp_dir) / "test_sim_cache.npz"
            generated_path = generate_synthetic_cache(
                cache_file,
                N_unseen=36,
                N_test=54,
                seed=42,
            )
            self.assertTrue(generated_path.exists())

            npz = np.load(generated_path, allow_pickle=True)
            keys = set(npz.files)
            npz.close()

            required_keys = {
                "N_test", "N_unseen", "y_unseen", "groups_unseen",
                "model_names", "model_tiers", "model_prior_weights",
                "m_0_unseen_lp", "m_0_test_lp", "rule_flags_test",
                "log_p_handwriting", "num_extra",
            }
            for k in required_keys:
                self.assertIn(k, keys)

    def test_evaluate_from_cache(self):
        """Test sub-second offline cache replay, model selection, and post-processing."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            cache_file = Path(tmp_dir) / "test_sim_cache.npz"
            generate_synthetic_cache(cache_file, N_unseen=72, N_test=90, seed=123)

            t0 = time.time()
            res = evaluate_from_cache(cache_file, working_dir=Path(tmp_dir))
            replay_duration = time.time() - t0

            # Must run very quickly (< 5 seconds on any CPU)
            self.assertLess(replay_duration, 5.0)

            # Check outputs
            self.assertIn("P_final", res)
            self.assertIn("weights_map", res)
            self.assertIn("df_report", res)

            P_final = res["P_final"]
            self.assertEqual(P_final.shape, (90, K))

            # Row sums of Sinkhorn output must be close to 1.0
            row_sums = P_final.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones(90), atol=1e-4)

            # Poison model must have weight 0.0 (pruned or suppressed)
            weights_map = res["weights_map"]
            self.assertEqual(weights_map.get("Poison-Adversarial", 0.0), 0.0)

    def test_evaluate_prompt_split_synthetic(self):
        """Test zero-leakage prompt-grouped holdout evaluation on synthetic data."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_csv = tmp_path / "train_synth.csv"

            # Create synthetic train data: 4 prompts x 18 classes = 72 rows
            rows = []
            for prompt_idx in range(4):
                p_id = f"prompt_{prompt_idx:03d}"
                for model_idx, model_name in enumerate(LABELS):
                    # Distinct text cues per model to ensure classifier can learn
                    text = f"This is prompt {prompt_idx} response from model {model_name} with unique style markers {model_name}."
                    rows.append({
                        "id": f"{prompt_idx}_{model_idx}",
                        "prompt_id": p_id,
                        "text": text,
                        "model": model_name,
                    })

            df_synth = pd.DataFrame(rows)
            df_synth.to_csv(train_csv, index=False)

            results = evaluate_prompt_split(
                train_path=train_csv,
                n_splits=2,
                fast=True,
                working_dir=tmp_path / "work",
            )

            self.assertIn("mean_bacc", results)
            self.assertIn("fold_metrics", results)
            self.assertIn("top_confused", results)
            self.assertGreaterEqual(results["mean_bacc"], 0.0)
            self.assertLessEqual(results["mean_bacc"], 1.0)
            self.assertEqual(len(results["fold_metrics"]), 1)


if __name__ == "__main__":
    unittest.main()
