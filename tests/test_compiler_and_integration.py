"""
Compiler Parity & End-to-End Integration Test Suite
===================================================
1. Validates that scripts/compile_notebook.py is a thin file-bundler (<= 150 lines, 0 ML strings).
2. Verifies 100% AST parity between src/namethatllm/*.py and compiled kaggle_submission.ipynb.
3. Tests end-to-end pipeline execution in smoke mode on synthetic competition data.
"""

import os
import ast
import json
import unittest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

from namethatllm.config import LABELS, K
from namethatllm.pipeline import run_submission_pipeline
from scripts.compile_notebook import compile_notebook

class TestCompilerAndIntegration(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_compiler_is_thin_bundler_not_template(self):
        compiler_path = self.repo_root / "scripts" / "compile_notebook.py"
        self.assertTrue(compiler_path.exists(), "scripts/compile_notebook.py must exist")

        lines = compiler_path.read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(
            len(lines), 150,
            f"Compiler must be a thin bundler <= 150 lines, but found {len(lines)} lines"
        )

        content = compiler_path.read_text(encoding="utf-8")
        forbidden_ml_strings = ["LOGIT_MAIN", "sinkhorn", "ExactAuthorPosterior", "softmax("]
        for s in forbidden_ml_strings:
            self.assertNotIn(
                s, content,
                f"Compiler must not contain hardcoded ML string '{s}'"
            )

    def test_ast_parity_between_src_and_compiled_notebook(self):
        cfg_path = self.repo_root / "configs" / "production_ensemble.json"
        out_nb_path = self.repo_root / "kernels" / "submission" / "kaggle_submission.ipynb"

        # Recompile notebook
        compile_notebook(cfg_path, out_nb_path)
        self.assertTrue(out_nb_path.exists())

        # Collect top-level functions and classes in src/namethatllm/*.py
        pkg_dir = self.repo_root / "src" / "namethatllm"
        src_defs = set()
        for py_file in pkg_dir.glob("*.py"):
            if py_file.name in ("__init__.py", "trainer.py"):
                continue
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=py_file.name)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    src_defs.add(node.name)

        # Collect top-level functions and classes in notebook cells
        nb_data = json.loads(out_nb_path.read_text(encoding="utf-8"))
        nb_defs = set()
        for idx, cell in enumerate(nb_data.get("cells", [])):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                if src.startswith("%%writefile"):
                    src = src.split("\n", 1)[1]
                tree = ast.parse(src, filename=f"cell_{idx}")
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        nb_defs.add(node.name)

        missing_in_notebook = src_defs - nb_defs
        self.assertEqual(
            len(missing_in_notebook), 0,
            f"Functions/classes in src/ missing in compiled notebook: {missing_in_notebook}"
        )
        self.assertGreater(len(src_defs), 15, "Expected at least 15 definitions in src/namethatllm")

    def test_end_to_end_smoke_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Synthesize 54-row train.csv (3 prompts x 18 classes)
            prompts = [101, 102, 103]
            train_rows = []
            row_id = 0
            for p in prompts:
                for m in LABELS:
                    train_rows.append({
                        "id": row_id,
                        "prompt_id": p,
                        "model": m,
                        "text": f"This is an analytical essay about prompt {p} authored by {m}. Furthermore, punctuation is stylized; 'quotes' and numbers 1234 are used."
                    })
                    row_id += 1
            df_train = pd.DataFrame(train_rows)
            train_csv = tmp_path / "train.csv"
            df_train.to_csv(train_csv, index=False)

            # 2. Synthesize 36-row test.csv (2 test prompts x 18 passages)
            test_rows = []
            test_prompts = [201, 202]
            t_id = 1000
            for p in test_prompts:
                for i in range(18):
                    test_rows.append({
                        "id": t_id,
                        "prompt_id": p,
                        "text": f"Test mystery response passage {i} for prompt {p}. Includes standard words and content."
                    })
                    t_id += 1
            df_test = pd.DataFrame(test_rows)
            test_csv = tmp_path / "test.csv"
            df_test.to_csv(test_csv, index=False)

            # 3. Run pipeline in smoke mode
            os.environ["NTL_SMOKE"] = "1"
            try:
                cfg = {
                    "train_path": str(train_csv),
                    "test_path": str(test_csv),
                    "working_dir": str(tmp_path),
                    "input_dir": str(tmp_path),
                }
                df_sub, parquet_path = run_submission_pipeline(cfg)

                # 4. Assert schema and validity
                self.assertTrue(parquet_path.exists(), "submission.parquet must exist")
                self.assertEqual(len(df_sub), 36, "Submission must have exactly 36 rows")
                self.assertListEqual(list(df_sub.columns), ["id", "model"])
                self.assertFalse(df_sub["model"].isna().any(), "Submission must have no NaN models")
                self.assertTrue(df_sub["model"].isin(LABELS).all(), "All predicted models must be in LABELS")

                # Verify parquet file reads back correctly
                readback = pd.read_parquet(parquet_path)
                self.assertEqual(len(readback), 36)
                self.assertListEqual(list(readback.columns), ["id", "model"])
            finally:
                os.environ.pop("NTL_SMOKE", None)

    def test_neural_extractor_layernorm_specs(self):
        import inspect
        from namethatllm.neural_extractor import UniversalLoRATrunkLoader
        src = inspect.getsource(UniversalLoRATrunkLoader.__init__)
        self.assertIn("nn.LayerNorm", src, "Norms must be initialized with nn.LayerNorm to accept bias")
        self.assertNotIn("nn.RMSNorm", src, "nn.RMSNorm lacks bias parameters present in heads.safetensors")
        self.assertIn("nn.Identity", src, "Final trunk norm must be replaced with nn.Identity to prevent double norm")

    def test_full_neural_pipeline_branch_with_mocks(self):
        """
        Validates 100% of the neural pipeline execution flow (all 4 trunks, 6 checkpoints,
        dynamic meta parsing, multi-res twins, exact DP, joint spectral bank, Claude specialist gate,
        and Sinkhorn mass balancing) without requiring multi-gigabyte pretrained model downloads.
        """
        import torch
        from unittest.mock import patch
        from safetensors.torch import save_file
        from namethatllm.config import FAMILIES

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "input"
            working_dir = tmp_path / "working"
            input_dir.mkdir(parents=True, exist_ok=True)
            working_dir.mkdir(parents=True, exist_ok=True)

            # 1. Synthesize 180-row train.csv (10 prompts x 18 classes)
            prompts = list(range(101, 111))
            train_rows = []
            row_id = 0
            for p in prompts:
                for m in LABELS:
                    train_rows.append({
                        "id": row_id,
                        "prompt_id": p,
                        "model": m,
                        "text": f"Prompt {p} discourse sample authored by {m} with diverse features and stylistic punctuation."
                    })
                    row_id += 1
            df_train = pd.DataFrame(train_rows)
            train_csv = tmp_path / "train.csv"
            df_train.to_csv(train_csv, index=False)

            # 2. Synthesize 36-row test.csv (2 test prompts x 18 passages)
            test_rows = []
            for p in [201, 202]:
                for i in range(18):
                    test_rows.append({
                        "id": 1000 + len(test_rows),
                        "prompt_id": p,
                        "text": f"Mystery test passage {i} for prompt {p} with curly ‘quotes’ and words."
                    })
            df_test = pd.DataFrame(test_rows)
            test_csv = tmp_path / "test.csv"
            df_test.to_csv(test_csv, index=False)

            # 2B. Synthesize public pool parquet
            pd.DataFrame({"text": ["Pool text sample"] * 10}).to_parquet(input_dir / "pool_v4.parquet")

            # 3. Create mock base model directories with config.json
            base_q3_14b = input_dir / "base_qwen3_14b"
            base_q3_14b.mkdir(parents=True, exist_ok=True)
            (base_q3_14b / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40}))

            base_q25_14b = input_dir / "base_qwen25_14b"
            base_q25_14b.mkdir(parents=True, exist_ok=True)
            (base_q25_14b / "config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 48}))

            base_q3_8b = input_dir / "base_qwen3_8b"
            base_q3_8b.mkdir(parents=True, exist_ok=True)
            (base_q3_8b / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36}))

            base_deb = input_dir / "base_deberta"
            base_deb.mkdir(parents=True, exist_ok=True)
            (base_deb / "config.json").write_text(json.dumps({"model_type": "deberta", "hidden_size": 1024, "num_hidden_layers": 24}))

            # Helper to create standard 18-class head safetensors
            def make_18class_head(t_dict=None):
                d = {
                    "plain.weight": torch.randn(18, 512),
                    "plain.bias": torch.zeros(18),
                    "center.weight": torch.randn(18, 512),
                    "center.bias": torch.zeros(18),
                    "norms.0.weight": torch.ones(64),
                    "norms.0.bias": torch.zeros(64),
                }
                if t_dict:
                    d.update(t_dict)
                return d

            # 4. Create mock checkpoints
            # 4A. qwen3_14b_p2
            p2_dir = input_dir / "lora_qwen3_14b"
            p2_dir.mkdir(parents=True, exist_ok=True)
            (p2_dir / "meta.json").write_text(json.dumps({"cfg": {"layers": [2, 4, 6, 9, 13, 17, 22, 27], "keep": 27, "max_len": 447, "end_token_id": 151643}, "labels": LABELS}))
            save_file(make_18class_head(), str(p2_dir / "heads.safetensors"))

            # 4B. qwen3_14b_p1
            p1_dir = input_dir / "lora14b_v1"
            p1_dir.mkdir(parents=True, exist_ok=True)
            (p1_dir / "meta.json").write_text(json.dumps({"cfg": {"layers": [2, 4, 6, 9, 13, 17, 21, 24], "keep": 24, "max_len": 447, "end_token_id": 151643}, "labels": LABELS}))
            save_file(make_18class_head(), str(p1_dir / "heads.safetensors"))

            # 4C. qwen25_14b_p2
            q25_dir = input_dir / "lora_qwen25_14b"
            q25_dir.mkdir(parents=True, exist_ok=True)
            (q25_dir / "meta.json").write_text(json.dumps({"cfg": {"layers": [3, 5, 8, 11, 16, 21, 27, 32], "keep": 32, "max_len": 447, "end_token_id": 151643}, "labels": LABELS}))
            save_file(make_18class_head(), str(q25_dir / "heads.safetensors"))

            # 4D. instyle
            ins_dir = input_dir / "instyle"
            ins_dir.mkdir(parents=True, exist_ok=True)
            (ins_dir / "meta.json").write_text(json.dumps({"cfg": {"layers": [2, 4, 6, 8, 12, 16, 20, 24], "keep": 24, "max_len": 447, "end_token_id": 151643}, "labels": LABELS}))
            save_file(make_18class_head(), str(ins_dir / "heads.safetensors"))

            # 4E. qwen3_8b_claude (5-class specialist + 18-class readouts.npz)
            claude_dir = input_dir / "qwen3_8b_claude"
            claude_dir.mkdir(parents=True, exist_ok=True)
            claude_classes = [LABELS[i] for i in FAMILIES["claude"]]
            (claude_dir / "meta.json").write_text(json.dumps({"classes": claude_classes, "cfg": {"layers": [2, 4, 6, 8, 12, 16, 20, 24], "keep": 24, "max_len": 447, "end_token_id": 151643}}))
            save_file({"norms.0.weight": torch.ones(64), "norms.0.bias": torch.zeros(64)}, str(claude_dir / "heads.safetensors"))
            np.savez(
                str(claude_dir / "readouts.npz"),
                plain_coef=np.random.randn(18, 512).astype(np.float32),
                plain_intercept=np.zeros(18, dtype=np.float32),
                center_coef=np.random.randn(18, 512).astype(np.float32),
                center_intercept=np.zeros(18, dtype=np.float32),
                claude_coef=np.random.randn(5, 512).astype(np.float32),
                claude_intercept=np.zeros(5, dtype=np.float32),
                claude_classes=np.array(claude_classes),
            )
            save_file({"weight": torch.randn(5, 512), "bias": torch.zeros(5)}, str(claude_dir / "claude_head.safetensors"))

            # 4F. deberta_p2
            deb_dir = input_dir / "deberta_p2"
            deb_dir.mkdir(parents=True, exist_ok=True)
            (deb_dir / "meta.json").write_text(json.dumps({"cfg": {"layers": [3, 6, 9, 12, 15, 18, 21, 24], "keep": 24, "max_len": 447}, "labels": LABELS}))
            save_file({"plain.weight": torch.randn(18, 512), "plain.bias": torch.zeros(18)}, str(deb_dir / "head.safetensors"))

            # 5. Define MockTrunkLoader and Mock DeBERTa
            class MockTrunkLoader:
                def __init__(self, base_dir, keep_layers, layers, max_len=447):
                    self.layers = layers
                    self.max_len = max_len
                    self.model = torch.nn.Linear(10, 10)
                def snapshot_weights(self, adapter_path):
                    return {}
                def restore_weights(self, backup):
                    pass
                def set_active_layers(self, layers, max_len=None):
                    self.layers = layers
                    if max_len is not None:
                        self.max_len = max_len
                def load_norm_weights(self, heads_sd):
                    pass
                def extract_exact(self, texts, batch_size=8, end_id=151643):
                    rng = np.random.RandomState(42)
                    return rng.randn(len(texts), 512).astype(np.float32) * 0.1
                def free_memory(self):
                    pass

            def mock_extract_deberta(deb_d, base_deb_d, texts_test, texts_train, y_train, max_len=512, batch_size=8):
                rng = np.random.RandomState(42)
                Z_te = rng.randn(len(texts_test), 512).astype(np.float32) * 0.1
                Z_tr = rng.randn(len(texts_train), 512).astype(np.float32) * 0.1
                W_p = rng.randn(18, 512).astype(np.float32) * 0.1
                b_p = np.zeros(18, dtype=np.float32)
                W_c = rng.randn(18, 512).astype(np.float32) * 0.1
                b_c = np.zeros(18, dtype=np.float32)
                return Z_te, Z_tr, W_p, b_p, W_c, b_c, list(LABELS)

            # Ensure NTL_SMOKE is NOT set (so has_neural branch executes!)
            os.environ["NTL_SMOKE"] = "0"
            try:
                cfg = {
                    "train_path": str(train_csv),
                    "test_path": str(test_csv),
                    "working_dir": str(working_dir),
                    "input_dir": str(input_dir),
                }

                with patch("namethatllm.pipeline.UniversalLoRATrunkLoader", MockTrunkLoader), \
                     patch("namethatllm.pipeline.extract_deberta_checkpoint", mock_extract_deberta), \
                     patch("namethatllm.pipeline.merge_lora", return_value=True):
                    df_sub, parquet_path = run_submission_pipeline(cfg)

                self.assertTrue(parquet_path.exists(), "submission.parquet must exist")
                self.assertEqual(len(df_sub), 36, "Submission must have exactly 36 rows")
                self.assertListEqual(list(df_sub.columns), ["id", "model"])
                self.assertFalse(df_sub["model"].isna().any(), "Submission must have no NaN models")
                self.assertTrue(df_sub["model"].isin(LABELS).all(), "All predicted models must be in LABELS")

                # Verify parquet file reads back correctly
                readback = pd.read_parquet(parquet_path)
                self.assertEqual(len(readback), 36)
                self.assertListEqual(list(readback.columns), ["id", "model"])
            finally:
                os.environ.pop("NTL_SMOKE", None)

    def test_load_head_file_formats_and_pool_discovery(self):
        """Tests unified loading of heads.safetensors and heads.pt, and multi-pool discovery."""
        import torch
        from safetensors.torch import save_file
        from namethatllm.neural_extractor import load_head_file, resolve_head_tensors
        from namethatllm.config import resolve_all_pool_files, find_checkpoints

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # 1. Test heads.safetensors
            d1 = tmp / "ckpt_safetensors"
            d1.mkdir()
            t_w = torch.randn(18, 128)
            t_b = torch.randn(18)
            save_file({"plain.weight": t_w, "plain.bias": t_b}, str(d1 / "heads.safetensors"))
            loaded_1 = load_head_file(d1)
            W_p1, b_p1, _, _ = resolve_head_tensors(loaded_1)
            self.assertEqual(W_p1.shape, (18, 128))
            self.assertEqual(b_p1.shape, (18,))

            # 2. Test heads.pt (flat)
            d2 = tmp / "l24x_q3b8"
            d2.mkdir()
            torch.save({"plain.weight": t_w, "plain.bias": t_b, "norms.0.weight": torch.randn(128)}, str(d2 / "heads.pt"))
            loaded_2 = load_head_file(d2)
            W_p2, b_p2, _, _ = resolve_head_tensors(loaded_2)
            self.assertEqual(W_p2.shape, (18, 128))
            self.assertEqual(b_p2.shape, (18,))
            self.assertIn("norms.0.weight", loaded_2)

            # 3. Test heads.pt (nested)
            d3 = tmp / "nested_pt"
            d3.mkdir()
            torch.save({"plain": {"weight": t_w, "bias": t_b}}, str(d3 / "heads.pt"))
            loaded_3 = load_head_file(d3)
            W_p3, b_p3, _, _ = resolve_head_tensors(loaded_3)
            self.assertEqual(W_p3.shape, (18, 128))
            self.assertEqual(b_p3.shape, (18,))

            # 4. Test find_checkpoints discovers heads.pt
            ckpts = find_checkpoints(tmp)
            self.assertIn("qwen3_8b_ts", ckpts)
            self.assertEqual(ckpts["qwen3_8b_ts"], d2)

            # 5. Test multi-pool resolution
            p_v5 = tmp / "nlm_pool_v5.parquet"
            p_v4 = tmp / "pool_v4.parquet"
            pd.DataFrame({"text": ["a"], "model": ["gpt-5.5"]}).to_parquet(p_v5)
            pd.DataFrame({"text": ["b"], "model": ["gpt-5.5"]}).to_parquet(p_v4)
            pools = resolve_all_pool_files(tmp)
            self.assertEqual(len(pools), 2)
            self.assertEqual(pools[0].name, "nlm_pool_v5.parquet")
            self.assertEqual(pools[1].name, "pool_v4.parquet")

    def test_preflight_validation_raises_on_missing_assets_when_not_smoke(self):
        """Pre-flight asset validation fails closed immediately when NTL_SMOKE=0 and assets are missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            train_p = tmp / "train.csv"
            test_p = tmp / "test.csv"
            pd.DataFrame({"id": [1], "prompt_id": [1], "model": [LABELS[0]], "text": ["hello"]}).to_csv(train_p, index=False)
            pd.DataFrame({"id": [2], "prompt_id": [2], "text": ["world"]}).to_csv(test_p, index=False)

            cfg = {
                "train_path": str(train_p),
                "test_path": str(test_p),
                "working_dir": str(tmp / "working"),
                "input_dir": str(tmp / "input"),
            }
            (tmp / "input").mkdir(parents=True, exist_ok=True)
            (tmp / "working").mkdir(parents=True, exist_ok=True)

            os.environ["NTL_SMOKE"] = "0"
            try:
                with self.assertRaises(FileNotFoundError) as ctx:
                    run_submission_pipeline(cfg)
                err_str = str(ctx.exception)
                self.assertIn("PRE-FLIGHT ASSET AUDIT FAILED", err_str)
                self.assertIn("No public pool parquet", err_str)
                self.assertIn("Required base model 'qwen3_14b'", err_str)
                self.assertIn("Required checkpoint 'qwen3_14b_p2'", err_str)
            finally:
                os.environ.pop("NTL_SMOKE", None)

    def test_preflight_validation_passes_when_all_required_assets_present(self):
        """Pre-flight asset validation succeeds when all required base trunks, checkpoints, and pools exist."""
        from namethatllm.config import validate_production_assets
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            train_p = tmp / "train.csv"
            test_p = tmp / "test.csv"
            pool_p = tmp / "pool_v4.parquet"
            pd.DataFrame({"id": [1], "prompt_id": [1], "model": [LABELS[0]], "text": ["hello"]}).to_csv(train_p, index=False)
            pd.DataFrame({"id": [2], "prompt_id": [2], "text": ["world"]}).to_csv(test_p, index=False)
            pd.DataFrame({"text": ["dummy"]}).to_parquet(pool_p)

            # Create 4 base models
            b_q3_14 = tmp / "q3_14"
            b_q3_14.mkdir()
            (b_q3_14 / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40}))

            b_q25_14 = tmp / "q25_14"
            b_q25_14.mkdir()
            (b_q25_14 / "config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 48}))

            b_q3_8 = tmp / "q3_8"
            b_q3_8.mkdir()
            (b_q3_8 / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36}))

            b_deb = tmp / "deb"
            b_deb.mkdir()
            (b_deb / "config.json").write_text(json.dumps({"model_type": "deberta", "hidden_size": 1024, "num_hidden_layers": 24}))

            # Mock checkpoints dict
            ckpts = {
                "qwen3_14b_p2": tmp / "p2_14",
                "qwen3_14b_p1": tmp / "p1_14",
                "qwen25_14b_p2": tmp / "p2_q25",
                "deberta_p2": tmp / "deb_p2",
                "instyle": tmp / "instyle",
                "qwen3_8b_claude": tmp / "cl_8b",
            }
            for p in ckpts.values():
                p.mkdir()

            verified_bases = validate_production_assets(
                input_dir=tmp,
                train_path=train_p,
                test_path=test_p,
                pool_files=[pool_p],
                ckpts=ckpts,
                cfg={"required_assets": {
                    "require_pool_parquet": True,
                    "required_base_models": [
                        {"name": "qwen3_14b", "family": "qwen3", "hidden_size": 5120, "num_layers": 40},
                        {"name": "qwen25_14b", "family": "qwen2", "hidden_size": 5120, "num_layers": 48},
                        {"name": "qwen3_8b", "family": "qwen3", "hidden_size": 4096, "num_layers": 36},
                        {"name": "deberta_v3_large", "family": "deberta", "hidden_size": 1024, "num_layers": 24},
                    ],
                    "required_checkpoints": list(ckpts.keys()),
                }}
            )
            self.assertEqual(len(verified_bases), 4)
            self.assertIn("qwen3_14b", verified_bases)
            self.assertIn("qwen25_14b", verified_bases)
            self.assertIn("qwen3_8b", verified_bases)
            self.assertIn("deberta_v3_large", verified_bases)

if __name__ == "__main__":
    unittest.main()

