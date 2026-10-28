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
        cfg_path = self.repo_root / "configs" / "v42_ensemble.json"
        out_nb_path = self.repo_root / "kernels" / "submission" / "kaggle_submission.ipynb"

        # Recompile notebook
        compile_notebook(cfg_path, out_nb_path)
        self.assertTrue(out_nb_path.exists())

        # Collect top-level functions and classes in src/namethatllm/*.py
        pkg_dir = self.repo_root / "src" / "namethatllm"
        src_defs = set()
        for py_file in pkg_dir.glob("*.py"):
            if py_file.name == "__init__.py":
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

if __name__ == "__main__":
    unittest.main()
