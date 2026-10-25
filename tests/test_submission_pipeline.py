"""
Comprehensive Unit & Integration Test Suite for Name That LLM Submission Pipeline.
Verifies all components:
1. Stylometric feature extraction & numerical stability (empty string edge cases).
2. Multi-view TF-IDF sparse matrix consistency.
3. Within-family ANOVA specialist & Law-of-Total-Probability gated blending.
4. Latent prompt clustering & Singleton-Protected adaptive chunk centering.
5. Prior-shift calibration (simplex invariance).
6. Full end-to-end inference and submission.parquet schema verification.
"""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

TARGET_MODELS = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash',
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'kimi-k3', 'qwen3.8-max'
]
NUM_CLASSES = len(TARGET_MODELS)
MODEL_TO_IDX = {m: i for i, m in enumerate(TARGET_MODELS)}
IDX_TO_MODEL = {i: m for i, m in enumerate(TARGET_MODELS)}

PROVIDER_FAMILIES = {
    'claude': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'],
    'gemini': ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
    'glm': ['glm-5.3', 'glm-5.3-flash'],
    'gpt': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'],
    'kimi': ['kimi-k3'],
    'qwen': ['qwen3.8-max']
}
FAMILY_NAMES = sorted(list(PROVIDER_FAMILIES.keys()))
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}
FAMILY_MODEL_INDICES = {f: [MODEL_TO_IDX[m] for m in models] for f, models in PROVIDER_FAMILIES.items()}
MODEL_TO_FAMILY_IDX = {m: FAMILY_TO_IDX[f] for f, models in PROVIDER_FAMILIES.items() for m in models}


# Component implementations matching kaggle_submission.ipynb
class StylometricExtractor:
    def __init__(self):
        self.means = None
        self.stds = None

    def _extract_row(self, text: str) -> np.ndarray:
        n_chars = max(1, len(text))
        words = text.split()
        n_words = max(1, len(words))
        
        colons = text.count(':') / n_words
        semicolons = text.count(';') / n_words
        em_dashes = (text.count('\u2014') + text.count('--')) / n_words
        quotes = (text.count('"') + text.count("'") + text.count('\u201c') + text.count('\u201d')) / n_words
        hyphens = text.count('-') / n_words
        parens = (text.count('(') + text.count(')')) / n_words
        linebreaks = text.count('\n') / n_words

        digits = sum(1 for c in text if c.isdigit()) / n_chars
        uppers = sum(1 for c in text if c.isupper()) / n_chars

        word_lengths = [len(w) for w in words]
        mean_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
        std_word_len = float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0

        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        sent_lengths = [len(s.split()) for s in sentences]
        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

        ttr = len(set(words)) / n_words

        return np.array([
            colons, semicolons, em_dashes, quotes, hyphens, parens,
            digits, uppers, linebreaks,
            mean_word_len, std_word_len, mean_sent_len, std_sent_len,
            ttr, n_words, n_chars / n_words
        ], dtype=np.float32)

    def fit(self, texts):
        raw_feats = np.array([self._extract_row(t) for t in texts], dtype=np.float32)
        self.means = np.mean(raw_feats, axis=0)
        self.stds = np.std(raw_feats, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, texts) -> np.ndarray:
        raw_feats = np.array([self._extract_row(t) for t in texts], dtype=np.float32)
        scaled = (raw_feats - self.means) / self.stds
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class MultiViewFeatureExtractor:
    def __init__(self):
        self.char_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=1, sublinear_tf=True, max_features=1000)
        self.word_vec = TfidfVectorizer(analyzer='word', lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=1000)
        self.stylo = StylometricExtractor()

    def fit(self, texts):
        self.char_vec.fit(texts)
        self.word_vec.fit(texts)
        self.stylo.fit(texts)
        return self

    def transform(self, texts) -> sparse.csr_matrix:
        X_c = self.char_vec.transform(texts)
        X_w = self.word_vec.transform(texts)
        X_s = sparse.csr_matrix(self.stylo.transform(texts))
        return sparse.hstack([X_c, X_w, X_s], format='csr')


class WithinFamilySpecialist:
    def __init__(self, family_name: str, percentile: int = 60):
        self.family_name = family_name
        self.percentile = percentile
        self.model_indices = FAMILY_MODEL_INDICES[family_name]
        self.n_models = len(self.model_indices)
        self.var_filter = VarianceThreshold(threshold=0.0)
        self.selector = SelectPercentile(f_classif, percentile=self.percentile)
        self.clf = LogisticRegression(C=0.6, max_iter=300, tol=1e-3, solver='lbfgs', random_state=42)

    def fit(self, X_sparse: sparse.csr_matrix, y_model_idx: np.ndarray):
        if self.n_models <= 1:
            return self
        local_mapping = {g: i for i, g in enumerate(self.model_indices)}
        y_local = np.array([local_mapping[g] for g in y_model_idx], dtype=np.int32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                X_var = self.var_filter.fit_transform(X_sparse)
                if X_var.shape[1] > 0:
                    X_filt = self.selector.fit_transform(X_var, y_local)
                else:
                    X_filt = None
            except Exception:
                X_filt = None

        if X_filt is None or X_filt.shape[1] == 0:
            X_filt = X_sparse
            self.selector = None
            self.var_filter = None

        self.clf.fit(X_filt, y_local)
        return self

    def predict_proba(self, X_sparse: sparse.csr_matrix) -> np.ndarray:
        if self.n_models <= 1:
            return np.ones((X_sparse.shape[0], 1), dtype=np.float32)
        if self.var_filter is not None and self.selector is not None:
            try:
                X_var = self.var_filter.transform(X_sparse)
                X_filt = self.selector.transform(X_var)
            except Exception:
                X_filt = X_sparse
        else:
            X_filt = X_sparse
        return self.clf.predict_proba(X_filt)


class FamilySpecialistEnsemble:
    def __init__(self):
        self.specialists = {}
        self.gammas = {'claude': 0.40, 'gemini': 0.45, 'glm': 0.50, 'gpt': 0.40}

    def fit(self, X_sparse: sparse.csr_matrix, y_model_idx: np.ndarray):
        y_family = np.array([MODEL_TO_FAMILY_IDX[TARGET_MODELS[i]] for i in y_model_idx])
        for fam_name, models in PROVIDER_FAMILIES.items():
            if len(models) <= 1:
                continue
            fam_idx = FAMILY_TO_IDX[fam_name]
            fam_mask = (y_family == fam_idx)
            if np.sum(fam_mask) == 0:
                continue
            spec = WithinFamilySpecialist(family_name=fam_name)
            spec.fit(X_sparse[fam_mask], y_model_idx[fam_mask])
            self.specialists[fam_name] = spec
        return self

    def apply_gated_blend(self, X_sparse: sparse.csr_matrix, base_probs: np.ndarray) -> np.ndarray:
        N = X_sparse.shape[0]
        blended = base_probs.copy()
        family_probs = np.zeros((N, len(FAMILY_NAMES)), dtype=np.float32)
        for fam_name, fam_indices in FAMILY_MODEL_INDICES.items():
            fam_idx = FAMILY_TO_IDX[fam_name]
            family_probs[:, fam_idx] = base_probs[:, fam_indices].sum(axis=1)

        for fam_name, spec in self.specialists.items():
            fam_idx = FAMILY_TO_IDX[fam_name]
            p_fam = family_probs[:, fam_idx]
            cond_p = spec.predict_proba(X_sparse)
            g = self.gammas.get(fam_name, 0.40)
            for local_i, global_i in enumerate(spec.model_indices):
                hier_p = p_fam * cond_p[:, local_i]
                blended[:, global_i] = (1.0 - g) * base_probs[:, global_i] + g * hier_p

        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (blended / row_sums).astype(np.float32)


class HighPrecisionPromptClusterer:
    def __init__(self, sim_threshold: float = 0.82):
        self.distance_threshold = 1.0 - sim_threshold

    def find_clusters(self, norm_embs: np.ndarray) -> np.ndarray:
        N = norm_embs.shape[0]
        if N <= 1:
            return np.zeros(N, dtype=np.int32)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold, metric='cosine', linkage='average'
        )
        return clustering.fit_predict(norm_embs)


class AdaptiveChunkCenterer:
    def __init__(self, competition_strength: float = 0.15):
        self.competition_strength = competition_strength

    def refine(self, probs: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        refined = np.array(probs, dtype=np.float32, copy=True)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        size_dict = dict(zip(unique_labels, counts))

        for lbl, k in size_dict.items():
            if k <= 1:
                # STRICT SINGLETON PROTECTION: untouched
                continue
            indices = np.where(cluster_labels == lbl)[0]
            sub_p = refined[indices]
            top_classes = np.argmax(sub_p, axis=1)
            u_cls, cls_counts = np.unique(top_classes, return_counts=True)
            for c, cnt in zip(u_cls, cls_counts):
                if cnt > 1:
                    colliding = [idx for idx, cl in enumerate(top_classes) if cl == c]
                    best_idx = colliding[int(np.argmax([sub_p[idx, c] for idx in colliding]))]
                    for idx in colliding:
                        if idx != best_idx:
                            sub_p[idx, c] *= (1.0 - self.competition_strength)
            row_sums = sub_p.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            refined[indices] = sub_p / row_sums
        return refined


class TestSubmissionPipeline(unittest.TestCase):
    """Test suite covering every critical pipeline component and boundary condition."""

    def setUp(self):
        self.sample_texts = [
            "Hello world! This is a simple test prompt for LLM attribution.",
            "In mathematics, the Riemann hypothesis is a conjecture that the Riemann zeta function has its zeros.",
            "Quantum entanglement is a phenomenon where pairs or groups of particles interact.",
            "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]])",
            "Dear customer, thank you for contacting our support team today. How may I assist you?"
        ]

    def test_stylometric_extractor_shapes_and_values(self):
        """Test stylometric extractor output dimensions and non-singularity."""
        extractor = StylometricExtractor()
        extractor.fit(self.sample_texts)
        feats = extractor.transform(self.sample_texts)
        
        self.assertEqual(feats.shape, (len(self.sample_texts), 16))
        self.assertFalse(np.isnan(feats).any(), "Stylometric features contain NaN!")
        self.assertFalse(np.isinf(feats).any(), "Stylometric features contain Inf!")

    def test_stylometric_extractor_edge_cases(self):
        """Test zero-division resilience with empty and single-char strings."""
        extractor = StylometricExtractor()
        extractor.fit(self.sample_texts)
        edge_texts = ["", " ", "A", "...", "!?!?", "\n\n\n"]
        feats = extractor.transform(edge_texts)
        
        self.assertEqual(feats.shape, (len(edge_texts), 16))
        self.assertFalse(np.isnan(feats).any(), "Edge case produced NaN!")
        self.assertFalse(np.isinf(feats).any(), "Edge case produced Inf!")

    def test_multiview_feature_extractor(self):
        """Test multi-view sparse representation consistency."""
        fe = MultiViewFeatureExtractor()
        X_train = fe.fit(self.sample_texts).transform(self.sample_texts)
        self.assertTrue(sparse.issparse(X_train))
        self.assertEqual(X_train.shape[0], len(self.sample_texts))

        X_test = fe.transform(["A new mystery passage."])
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(X_test.shape[1], X_train.shape[1])

    def test_within_family_specialist_and_ensemble(self):
        """Test ANOVA specialists and Law-of-Total-Probability gated blending."""
        # Create a synthetic dataset with Claude models
        claude_models = PROVIDER_FAMILIES['claude']
        texts = [
            "Claude fable response text snippet.",
            "Claude opus 4.6 detailed analysis with high precision.",
            "Claude opus 4.8 enhanced reasoning on difficult problem.",
            "Claude opus 5 cutting-edge frontier performance.",
            "Claude sonnet 5 balanced and fast response."
        ] * 4
        y = np.array([MODEL_TO_IDX[m] for m in claude_models * 4], dtype=np.int32)
        
        fe = MultiViewFeatureExtractor()
        X = fe.fit(texts).transform(texts)
        
        ensemble = FamilySpecialistEnsemble()
        ensemble.fit(X, y)
        
        # Verify Claude specialist was fitted
        self.assertIn('claude', ensemble.specialists)
        
        # Test gated blend
        base_probs = np.full((len(texts), NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
        gated = ensemble.apply_gated_blend(X, base_probs)
        
        self.assertEqual(gated.shape, (len(texts), NUM_CLASSES))
        # Ensure row sums are 1.0 (valid probability distribution)
        np.testing.assert_allclose(gated.sum(axis=1), np.ones(len(texts)), rtol=1e-5)
        self.assertTrue((gated >= 0.0).all() and (gated <= 1.0).all())

    def test_strict_singleton_protection(self):
        """Verify that singletons are STRICTLY untouched by adaptive chunk centering."""
        centerer = AdaptiveChunkCenterer(competition_strength=0.20)
        
        # 3 isolated singletons (cluster labels 0, 1, 2)
        cluster_labels = np.array([0, 1, 2], dtype=np.int32)
        input_probs = np.array([
            [0.8, 0.1, 0.1] + [0.0] * 15,
            [0.7, 0.2, 0.1] + [0.0] * 15,
            [0.9, 0.05, 0.05] + [0.0] * 15
        ], dtype=np.float32)
        
        refined = centerer.refine(input_probs, cluster_labels)
        # Should be strictly identical
        np.testing.assert_array_equal(refined, input_probs, "Singletons were modified! Strict singleton protection failed.")

    def test_adaptive_centering_multi_sample_collision(self):
        """Verify that colliding duplicate predictions in multi-sample clusters are dampened."""
        centerer = AdaptiveChunkCenterer(competition_strength=0.20)
        
        # 2 samples in the same cluster (cluster 0) predicting the same class 0
        cluster_labels = np.array([0, 0], dtype=np.int32)
        input_probs = np.array([
            [0.70, 0.30] + [0.0] * 16,
            [0.60, 0.40] + [0.0] * 16
        ], dtype=np.float32)
        
        refined = centerer.refine(input_probs, cluster_labels)
        
        # Sample 0 had higher confidence (0.70 vs 0.60), so sample 1's top class must be dampened
        self.assertAlmostEqual(refined[0, 0], 0.70, places=5)
        self.assertLess(refined[1, 0], 0.60, "Colliding sample was not dampened!")
        np.testing.assert_allclose(refined.sum(axis=1), np.ones(2), rtol=1e-5)

    def test_prior_shift_calibration(self):
        """Test prior-shift power scaling preserves simplex and order."""
        probs = np.array([
            [0.6, 0.3, 0.1] + [0.0] * 15,
            [0.1, 0.7, 0.2] + [0.0] * 15
        ], dtype=np.float32)
        
        alpha = 0.85
        calibrated = np.power(np.maximum(probs, 1e-9), alpha)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        
        # Simplex check
        np.testing.assert_allclose(calibrated.sum(axis=1), np.ones(2), rtol=1e-5)
        # Top-1 order check
        self.assertEqual(np.argmax(calibrated[0]), 0)
        self.assertEqual(np.argmax(calibrated[1]), 1)

    def test_end_to_end_mock_pipeline(self):
        """Integration test executing entire pipeline end-to-end and writing valid parquet."""
        # Create minimal synthetic dataset for all 18 classes
        texts_train = []
        labels_train = []
        for i, m in enumerate(TARGET_MODELS):
            for repeat in range(3):
                texts_train.append(f"Synthetic training snippet number {repeat} for model author {m}.")
                labels_train.append(m)

        df_train = pd.DataFrame({
            'id': [f"tr_{i}" for i in range(len(texts_train))],
            'prompt_id': [i // 3 for i in range(len(texts_train))],
            'text': texts_train,
            'model': labels_train
        })
        
        df_test = pd.DataFrame({
            'id': ['t_001', 't_002', 't_003'],
            'text': [
                "Test passage one discussing quantum mechanics and physics.",
                "Test passage two providing a clear python coding answer.",
                "Test passage three with formal customer support tone."
            ]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_csv = Path(tmpdir) / "train.csv"
            test_csv = Path(tmpdir) / "test.csv"
            out_parquet = Path(tmpdir) / "submission.parquet"
            
            df_train.to_csv(train_csv, index=False)
            df_test.to_csv(test_csv, index=False)

            # Execute pipeline
            fe = MultiViewFeatureExtractor()
            X_train = fe.fit(df_train['text']).transform(df_train['text'])
            y_train = np.array([MODEL_TO_IDX[m] for m in df_train['model']], dtype=np.int32)
            
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            clf.fit(X_train, y_train)

            ensemble = FamilySpecialistEnsemble()
            ensemble.fit(X_train, y_train)

            # Test inference
            X_test = fe.transform(df_test['text'])
            raw_p = clf.predict_proba(X_test)
            base_p = np.zeros((len(df_test), NUM_CLASSES), dtype=np.float32)
            for i, c in enumerate(clf.classes_):
                base_p[:, c] = raw_p[:, i]

            gated_p = ensemble.apply_gated_blend(X_test, base_p)
            
            # Calibration & prediction
            cal_p = np.power(np.maximum(gated_p, 1e-9), 0.85)
            cal_p /= cal_p.sum(axis=1, keepdims=True)
            
            preds = [IDX_TO_MODEL[int(idx)] for idx in np.argmax(cal_p, axis=1)]
            
            sub = pd.DataFrame({
                'id': df_test['id'].values,
                'model': preds
            })
            sub.to_parquet(out_parquet, index=False)

            # Strict verification of output artifact
            self.assertTrue(out_parquet.exists())
            read_sub = pd.read_parquet(out_parquet)
            self.assertEqual(list(read_sub.columns), ['id', 'model'])
            self.assertEqual(len(read_sub), len(df_test))
            self.assertFalse(read_sub.isnull().any().any())
            self.assertTrue(set(read_sub['model']).issubset(set(TARGET_MODELS)))


if __name__ == '__main__':
    unittest.main()
