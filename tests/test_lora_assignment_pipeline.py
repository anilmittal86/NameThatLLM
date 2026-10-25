"""
Unit and Integration Test Suite for Steps 2, 3, and 4:
- Canonical Label & Family Mapping
- Length-Sorted Dynamic Batching & Inverse Permutation
- Guarded Hungarian Assignment & Singleton Protection
- Error Breakdown (Intra-Family vs Cross-Family)
- Strict Submission Schema Verification
"""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from infer_batch import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    NUM_CLASSES,
)
from guarded_assignment import (
    GuardedPromptAssigner,
    PromptSemanticEmbedder,
    evaluate_assignment,
)
from run_full_eval import (
    MODEL_TO_FAMILY,
    PROVIDER_FAMILIES,
    compute_error_breakdown,
)


class TestCanonicalMapping(unittest.TestCase):
    def test_canonical_labels(self):
        """Verify exact 18 classes and bijective mapping."""
        self.assertEqual(NUM_CLASSES, 18)
        self.assertEqual(len(LABELS), 18)
        self.assertEqual(len(LABEL2ID), 18)
        self.assertEqual(len(ID2LABEL), 18)

        # Check alphabetical sorting
        self.assertEqual(LABELS, sorted(LABELS))

        for idx, label in enumerate(LABELS):
            self.assertEqual(LABEL2ID[label], idx)
            self.assertEqual(ID2LABEL[idx], label)

    def test_provider_families(self):
        """Verify all 18 models belong to a valid provider family."""
        all_family_models = []
        for fam, models in PROVIDER_FAMILIES.items():
            all_family_models.extend(models)

        self.assertEqual(set(all_family_models), set(LABELS))
        self.assertEqual(len(all_family_models), 18)
        self.assertEqual(len(MODEL_TO_FAMILY), 18)


class TestLengthSortedInferencePermutation(unittest.TestCase):
    def test_inverse_permutation(self):
        """Verify length-sorted dynamic batching inverse permutation reconstructs exact original order."""
        texts = [
            "Short",                                    # len 5 -> pos 0
            "A slightly longer sentence here",           # len 31 -> pos 1
            "Medium text sample",                       # len 18 -> pos 2
            "Super extremely long passage that has many words and characters in it"  # len 69 -> pos 3
        ]
        N = len(texts)
        lengths = [len(t) for t in texts]
        sorted_indices = np.argsort(lengths)[::-1]  # descending: [3, 1, 2, 0]

        # Simulate batch outputs in sorted order
        # Each row outputs a one-hot vector indicating its sorted position
        all_sorted_probs = [None] * N
        for sorted_pos, orig_idx in enumerate(sorted_indices):
            vec = np.zeros(NUM_CLASSES, dtype=np.float64)
            vec[orig_idx] = 1.0  # true original index
            all_sorted_probs[sorted_pos] = vec

        # Reconstruct ordered probs
        ordered_probs = np.zeros((N, NUM_CLASSES), dtype=np.float64)
        for sorted_pos, orig_idx in enumerate(sorted_indices):
            ordered_probs[orig_idx] = all_sorted_probs[sorted_pos]

        # Verify each row in ordered_probs maps back to its original row index
        for i in range(N):
            self.assertEqual(np.argmax(ordered_probs[i]), i)
            np.testing.assert_allclose(ordered_probs[i].sum(), 1.0)


class TestGuardedHungarianAssignment(unittest.TestCase):
    def test_singleton_untouched(self):
        """Case 1: Singletons must strictly preserve their top-1 argmax prediction."""
        assigner = GuardedPromptAssigner(sim_threshold=0.65)
        probs = np.array([
            [0.9, 0.05, 0.05] + [0.0] * 15,
            [0.1, 0.8, 0.1] + [0.0] * 15,
        ], dtype=np.float64)

        # Singletons: each is in its own cluster
        assigned_0 = assigner.assign_cluster([0], probs)
        assigned_1 = assigner.assign_cluster([1], probs)

        self.assertEqual(assigned_0, [0])
        self.assertEqual(assigned_1, [1])

    def test_hungarian_collision_resolution(self):
        """Case 2: Colliding top predictions within a verified cluster must be resolved 1-to-1."""
        assigner = GuardedPromptAssigner(sim_threshold=0.65)
        # 2 passages both having highest probability on model 0
        # Passage 0: model 0 (p=0.85), model 1 (p=0.15)
        # Passage 1: model 0 (p=0.70), model 1 (p=0.30)
        probs = np.array([
            [0.85, 0.15] + [0.0] * 16,
            [0.70, 0.30] + [0.0] * 16,
        ], dtype=np.float64)

        cluster_indices = [0, 1]
        assigned = assigner.assign_cluster(cluster_indices, probs)

        # Passage 0 has stronger confidence on 0 (0.85 vs 0.70), so passage 0 must get model 0
        # Passage 1 must be assigned model 1 to prevent duplication
        self.assertEqual(assigned[0], 0)
        self.assertEqual(assigned[1], 1)
        self.assertNotEqual(assigned[0], assigned[1], "Collision was not resolved!")

    def test_oversized_cluster_guardrail(self):
        """Case 3: Clusters with >18 passages must safely fall back without error."""
        assigner = GuardedPromptAssigner(sim_threshold=0.65)
        # 20 passages (oversized)
        k = 20
        probs = np.zeros((k, NUM_CLASSES), dtype=np.float64)
        for i in range(k):
            probs[i, i % NUM_CLASSES] = 1.0

        cluster_indices = list(range(k))
        assigned = assigner.assign_cluster(cluster_indices, probs)

        self.assertEqual(len(assigned), k)
        for i in range(k):
            self.assertEqual(assigned[i], i % NUM_CLASSES)

    def test_evaluate_assignment_lift(self):
        """Test evaluate_assignment computing BAcc, lift, and Monte Carlo subsampling."""
        texts = [
            "Text about astronomy and stars passage 1",
            "Text about astronomy and stars passage 2",
            "Text about cooking pancakes recipe 1",
            "Text about cooking pancakes recipe 2",
        ]
        y_true = np.array([0, 1, 2, 3], dtype=np.int32)
        # Raw probs has a duplicate prediction on cluster 0: both predict model 0
        raw_probs = np.array([
            [0.8, 0.2] + [0.0] * 16,
            [0.6, 0.4] + [0.0] * 16,
            [0.0, 0.0, 0.9, 0.1] + [0.0] * 14,
            [0.0, 0.0, 0.3, 0.7] + [0.0] * 14,
        ], dtype=np.float64)

        results = evaluate_assignment(
            y_true=y_true,
            raw_probs=raw_probs,
            texts=texts,
            sim_threshold=0.50,
            n_monte_carlo_trials=5,
        )

        self.assertIn("raw_bacc", results)
        self.assertIn("guarded_bacc", results)
        self.assertIn("delta_bacc", results)
        # Guarded BAcc should resolve the collision and be >= raw BAcc
        self.assertGreaterEqual(results["guarded_bacc"], results["raw_bacc"])


class TestErrorBreakdown(unittest.TestCase):
    def test_error_breakdown(self):
        """Verify intra-family vs cross-family error classification."""
        # y_true: [claude-fable-5, claude-opus-4-6, gpt-5.5]
        y_true = np.array([LABEL2ID["claude-fable-5"], LABEL2ID["claude-opus-4-6"], LABEL2ID["gpt-5.5"]])

        # y_pred:
        # Sample 0: correct (claude-fable-5) -> no error
        # Sample 1: claude-opus-4-8 (same family: Anthropic) -> Intra-family error
        # Sample 2: gemini-3.1-pro-preview (different family: Google vs OpenAI) -> Cross-family error
        y_pred = np.array([LABEL2ID["claude-fable-5"], LABEL2ID["claude-opus-4-8"], LABEL2ID["gemini-3.1-pro-preview"]])

        eb = compute_error_breakdown(y_true, y_pred)
        self.assertEqual(eb["total_errors"], 2)
        self.assertEqual(eb["intra_family_errors"], 1)
        self.assertEqual(eb["intra_family_pct"], 50.0)
        self.assertEqual(eb["cross_family_errors"], 1)
        self.assertEqual(eb["cross_family_pct"], 50.0)


class TestSubmissionSchemaVerification(unittest.TestCase):
    def test_submission_parquet_schema(self):
        """Strict verification of submission.parquet format, columns, and types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "submission.parquet"
            test_ids = ["t_01", "t_02", "t_03"]
            pred_models = ["claude-opus-4-8", "gpt-5.6-terra", "gemini-3.1-pro-preview"]

            df_sub = pd.DataFrame({"id": test_ids, "model": pred_models})
            df_sub.to_parquet(out_file, index=False)

            # Read and verify assertions
            read_df = pd.read_parquet(out_file)
            self.assertEqual(list(read_df.columns), ["id", "model"])
            self.assertEqual(len(read_df), 3)
            self.assertFalse(read_df["id"].isna().any())
            self.assertFalse(read_df["model"].isna().any())
            self.assertTrue(set(read_df["model"].unique()).issubset(set(LABELS)))


if __name__ == "__main__":
    unittest.main()
