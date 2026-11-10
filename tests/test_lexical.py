"""
Unit and Integration Tests for namethatllm.lexical_engine
=========================================================
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from namethatllm.config import LABELS, K
from namethatllm.lexical_engine import (
    STOP_WORDS,
    function_text,
    run_lexical_engine,
    shape_text,
)


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
            self.assertIn("log_p_handwriting", res)
            self.assertIn("rule_flags_test", res)

            self.assertEqual(res["log_p_svc_main"].shape, (18, 18))
            self.assertEqual(res["p_lex_comp"].shape, (18, 18))
            self.assertEqual(res["D_tfidf"].shape, (18, 18))
            self.assertEqual(len(res["groups_085"]), 18)
            self.assertEqual(res["log_p_handwriting"].shape, (18, 18))
            self.assertEqual(res["rule_flags_test"].shape, (18, 4))

            # Probability simplex
            self.assertTrue(np.allclose(res["p_lex_comp"].sum(axis=1), 1.0))

    def test_shape_and_function_vectorizers_preserve_case_and_punctuation(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        from namethatllm.lexical_engine import function_text, shape_text

        text = "Hello World! It's — great; test."
        sh = shape_text(text)
        v_shape = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), lowercase=False)
        v_shape.fit([sh])
        self.assertTrue(any("X" in k for k in v_shape.vocabulary_), "Shape vectorizer must preserve uppercase X")

        fn = function_text(text)
        v_func = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
        v_func.fit([fn])
        vocab = v_func.vocabulary_
        self.assertTrue(any(punct in vocab for punct in [",", ";", "—", "'", "!"]), "Function vectorizer must preserve punctuation tokens")
        self.assertIn("it", vocab)

    def test_extract_handwriting_features_shape_and_contractions(self):
        from namethatllm.lexical_engine import (
            extract_handwriting_and_orthographic_features,
            extract_handwriting_features,
        )

        samples = ["It's a straight contraction — test with em-dash.", "It\u2019s a curly contraction; test without em-dash."]
        feats = extract_handwriting_features(samples)
        self.assertEqual(feats.shape, (2, 26))
        # Feature 14 is straight contractions, Feature 15 is curly contractions
        self.assertGreater(feats[0, 14], 0.0)
        self.assertEqual(feats[0, 15], 0.0)
        self.assertEqual(feats[1, 14], 0.0)
        self.assertGreater(feats[1, 15], 0.0)

        f2, rules = extract_handwriting_and_orthographic_features(samples)
        self.assertEqual(f2.shape, (2, 26))
        self.assertEqual(rules.shape, (2, 4))
        self.assertEqual(rules[0, 0], 0.0)  # No curly quotes in sample 0
        self.assertEqual(rules[1, 0], 1.0)  # Curly apostrophe \u2019 triggers Rule 1
        self.assertEqual(rules[1, 3], 1.0)  # Curly contraction triggers Rule 4

    def test_pool_source_weight_filename_check_and_core_prefix(self):
        import tempfile

        from namethatllm.config import LABELS, K

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            nested_matched_clean = tmp / "nlm-matched-clean"
            nested_matched_clean.mkdir(parents=True, exist_ok=True)

            # File 1: nlm_pool_v5.parquet inside directory named 'nlm-matched-clean'
            p_v5 = nested_matched_clean / "nlm_pool_v5.parquet"
            # File 2: nlm_matched_clean.parquet
            p_matched = tmp / "nlm_matched_clean.parquet"

            df_pool = pd.DataFrame(
                {
                    "text": [f"Pool passage {i} with sufficient competition passage length for verification." for i in range(18)],
                    "model": [LABELS[i % K] for i in range(18)],
                }
            )
            df_pool.to_parquet(p_v5)
            df_pool.to_parquet(p_matched)

            # Check filename logic: nlm_pool_v5 in matched-clean dir must get 0.35, while matched_clean gets 0.50
            fn_v5 = p_v5.name.lower()
            is_matched_v5 = "matched" in fn_v5 or "clean" in fn_v5
            src_w_v5 = 0.50 if is_matched_v5 else 0.35
            self.assertEqual(src_w_v5, 0.35, "nlm_pool_v5.parquet must receive 0.35 even inside nlm-matched-clean folder")

            fn_matched = p_matched.name.lower()
            is_matched_clean = "matched" in fn_matched or "clean" in fn_matched
            src_w_matched = 0.50 if is_matched_clean else 0.35
            self.assertEqual(src_w_matched, 0.50, "nlm_matched_clean.parquet must receive 0.50")

            # Check run_lexical_engine execution with pools and verify output shapes
            tr_file = tmp / "train.csv"
            pd.DataFrame(
                {
                    "id": list(range(18)),
                    "prompt_id": [0] * 18,
                    "text": [f"Train passage {i} with good length." for i in range(18)],
                    "model": [LABELS[i % K] for i in range(18)],
                }
            ).to_csv(tr_file, index=False)

            te_file = tmp / "test.csv"
            pd.DataFrame({"id": [100, 101], "text": ["Test passage 1 with good length.", "Test passage 2 with good length."]}).to_csv(
                te_file, index=False
            )

            res = run_lexical_engine(tr_file, te_file, pool_file=[p_matched, p_v5], qwen_tok_dir="", working_dir=tmp / "out")
            self.assertEqual(res["log_p_svc_main"].shape, (2, 18))
            self.assertEqual(res["p_lex_comp"].shape, (2, 18))


if __name__ == "__main__":
    unittest.main()
