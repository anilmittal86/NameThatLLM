import tempfile
import unittest
from pathlib import Path
import numpy as np

from scripts.benchmark_feature_manager import (
    save_benchmark_feature_bundle,
    load_benchmark_features,
    format_bundle_dirname,
)


class TestBenchmarkFeatureManager(unittest.TestCase):
    def test_bundle_dirname_formatting(self):
        dname = format_bundle_dirname("v84", "20261118")
        self.assertEqual(dname, "v84_20261118")

        dname2 = format_bundle_dirname("84", "20261118")
        self.assertEqual(dname2, "v84_20261118")

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            N = 10
            D_qwen = 64
            D_deb = 32

            features = {
                "qwen3_14b_v5": np.random.randn(N, D_qwen).astype(np.float32),
                "deberta_p2": np.random.randn(N, D_deb).astype(np.float32),
            }
            labels = np.random.randint(0, 18, size=N)
            groups = np.array([f"prompt_{i % 3}" for i in range(N)])

            bundle_dir = save_benchmark_feature_bundle(
                features_dict=features,
                labels=labels,
                groups=groups,
                version="v84",
                date_str="20261118",
                output_root=out_root,
            )

            self.assertTrue(bundle_dir.exists())
            self.assertTrue((bundle_dir / "metadata.json").exists())
            self.assertTrue((bundle_dir / "benchmark_1800_features_all_models_v84_20261118.npz").exists())
            self.assertTrue((bundle_dir / "Z_unseen_qwen3_14b_v5_v84_20261118.npz").exists())
            self.assertTrue((bundle_dir / "Z_unseen_deberta_p2_v84_20261118.npz").exists())

            # Load master bundle
            loaded_feats, loaded_y, loaded_g, meta = load_benchmark_features(
                version="v84", date_str="20261118", features_dir=out_root
            )
            self.assertIn("Z_unseen_qwen3_14b_v5", loaded_feats)
            self.assertIn("Z_unseen_deberta_p2", loaded_feats)
            self.assertEqual(loaded_feats["Z_unseen_qwen3_14b_v5"].shape, (N, D_qwen))
            self.assertEqual(loaded_feats["Z_unseen_deberta_p2"].shape, (N, D_deb))
            self.assertTrue(np.array_equal(labels, loaded_y))
            self.assertEqual(meta["version"], "v84")

            # Load single model
            single_feat, single_y, single_g, _ = load_benchmark_features(
                version="v84", model_name="qwen3_14b_v5", features_dir=out_root
            )
            self.assertIn("qwen3_14b_v5", single_feat)
            self.assertEqual(single_feat["qwen3_14b_v5"].shape, (N, D_qwen))


if __name__ == "__main__":
    unittest.main()
