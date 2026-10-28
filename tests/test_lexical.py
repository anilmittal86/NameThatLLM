"""
Unit and Integration Tests for namethatllm.lexical_engine
=========================================================
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from namethatllm.lexical_engine import shape_text, function_text, STOP_WORDS, run_lexical_engine
from namethatllm.config import LABELS, K

class TestLexicalEngine(unittest.TestCase):
    def test_shape_text_patterns(self):
        self.assertEqual(shape_text("Hello World 123!"), "X X 123!")
        self.assertEqual(shape_text("hello world 123!"), "x x 123!")
        self.assertEqual(shape_text("AI and ML models"), "X x X x")

    def test_function_text_stopwords(self):
        text = "This is a simple test of the function words."
        ft = function_text(text)
        tokens = ft.split()
        self.assertIn("this", tokens)
        self.assertIn("is", tokens)
        self.assertIn("a", tokens)
        self.assertIn("of", tokens)
        self.assertIn("the", tokens)
        self.assertIn("CONTENT", tokens)

        # Strict check: punctuation marks must be preserved and NEVER masked to CONTENT
        text_punct = "However, why was it done; exactly?"
        ft_punct = function_text(text_punct)
        tokens_punct = ft_punct.split()
        self.assertIn(",", tokens_punct)
        self.assertIn(";", tokens_punct)
        self.assertIn("?", tokens_punct)

    def test_stopwords_integrity(self):
        self.assertGreater(len(STOP_WORDS), 50)
        self.assertIn("the", STOP_WORDS)
        self.assertIn("because", STOP_WORDS)
        self.assertIn("which", STOP_WORDS)

    def test_run_lexical_engine_mock(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock train.csv
            train_rows = []
            for i in range(36):
                m = LABELS[i % K]
                train_rows.append({"id": i, "prompt_id": i % 4, "text": f"This is training document {i} from model {m}.", "model": m})
            df_tr = pd.DataFrame(train_rows)
            tr_file = tmp_path / "train.csv"
            df_tr.to_csv(tr_file, index=False)

            # Create mock test.csv
            test_rows = []
            for i in range(18):
                test_rows.append({"id": 100 + i, "text": f"This is mystery passage {i} for testing."})
            df_te = pd.DataFrame(test_rows)
            te_file = tmp_path / "test.csv"
            df_te.to_csv(te_file, index=False)

            res = run_lexical_engine(tr_file, te_file, pool_file=None, qwen_tok_dir="", working_dir=tmp_path)

            self.assertIn("log_p_svc_main", res)
            self.assertIn("p_lex_comp", res)
            self.assertIn("D_tfidf", res)
            self.assertIn("groups_085", res)

            self.assertEqual(res["log_p_svc_main"].shape, (18, 18))
            self.assertEqual(res["p_lex_comp"].shape, (18, 18))
            self.assertEqual(res["D_tfidf"].shape, (18, 18))
            self.assertEqual(len(res["groups_085"]), 18)

            # Probability simplex
            self.assertTrue(np.allclose(res["p_lex_comp"].sum(axis=1), 1.0))

if __name__ == "__main__":
    unittest.main()
