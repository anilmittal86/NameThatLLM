"""
Linear and Hierarchical modeling pipeline for 18-class authorship attribution.
Includes:
1. Multi-view case-sensitive TF-IDF (char 2-5, char_wb 3-5, word 1-3) + stylometry stack.
2. Calibrated Flat 18-Class Classifiers (Logistic Regression, Ridge with softmax calibration).
3. Hierarchical Family Specialists (6-way Provider Classifier + Within-Family Specialists).
4. Blended probability inference with family gating.
"""

from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from src.config import (
    FAMILY_MODEL_INDICES,
    FAMILY_NAMES,
    FAMILY_TO_IDX,
    MODEL_TO_FAMILY,
    MODEL_TO_FAMILY_IDX,
    MODEL_TO_IDX,
    NUM_CLASSES,
    NUM_FAMILIES,
    PROVIDER_FAMILIES,
    RANDOM_SEED,
    TARGET_MODELS,
)
from src.features_stylo import StylometricExtractor


class MultiViewFeatureExtractor(BaseEstimator):
    """Extracts multi-view character and word TF-IDF features combined with stylometrics."""

    def __init__(
        self,
        max_char_features: int = 25000,
        max_char_wb_features: int = 15000,
        max_word_case_features: int = 20000,
        max_word_lower_features: int = 15000,
        include_stylometrics: bool = True,
    ):
        self.max_char_features = max_char_features
        self.max_char_wb_features = max_char_wb_features
        self.max_word_case_features = max_word_case_features
        self.max_word_lower_features = max_word_lower_features
        self.include_stylometrics = include_stylometrics

        self.char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=3,
            max_df=0.98,
            sublinear_tf=True,
            max_features=self.max_char_features,
        )
        self.char_wb_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=self.max_char_wb_features,
        )
        self.word_case_vec = TfidfVectorizer(
            analyzer="word",
            lowercase=False,
            ngram_range=(1, 3),
            min_df=2,
            sublinear_tf=True,
            max_features=self.max_word_case_features,
        )
        self.word_lower_vec = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=self.max_word_lower_features,
        )
        self.stylo_extractor = (
            StylometricExtractor(scale_features=True) if include_stylometrics else None
        )

    def fit(self, texts: Union[List[str], pd.Series], y=None):
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        self.char_vec.fit(texts)
        self.char_wb_vec.fit(texts)
        self.word_case_vec.fit(texts)
        self.word_lower_vec.fit(texts)
        if self.include_stylometrics and self.stylo_extractor is not None:
            self.stylo_extractor.fit(texts)
        return self

    def transform(self, texts: Union[List[str], pd.Series]) -> sparse.csr_matrix:
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        feat_char = self.char_vec.transform(texts)
        feat_char_wb = self.char_wb_vec.transform(texts)
        feat_word_case = self.word_case_vec.transform(texts)
        feat_word_lower = self.word_lower_vec.transform(texts)

        matrices = [feat_char, feat_char_wb, feat_word_case, feat_word_lower]

        if self.include_stylometrics and self.stylo_extractor is not None:
            feat_stylo = self.stylo_extractor.transform(texts)
            matrices.append(sparse.csr_matrix(feat_stylo))

        return sparse.hstack(matrices, format="csr")

    def fit_transform(self, texts: Union[List[str], pd.Series], y=None) -> sparse.csr_matrix:
        return self.fit(texts, y).transform(texts)


class CalibratedRidgeClassifier(BaseEstimator, ClassifierMixin):
    """Ridge Classifier with temperature-calibrated decision function probabilities."""

    def __init__(self, alpha: float = 1.0, temperature: float = 1.0, random_state: int = RANDOM_SEED):
        self.alpha = alpha
        self.temperature = temperature
        self.random_state = random_state
        self.model = RidgeClassifier(alpha=self.alpha, random_state=self.random_state)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def decision_function(self, X) -> np.ndarray:
        return self.model.decision_function(X)

    def predict_proba(self, X) -> np.ndarray:
        df = self.model.decision_function(X)
        if df.ndim == 1:
            # Binary case
            df = np.vstack([-df, df]).T
        # Softmax with temperature
        scaled = df / max(1e-4, self.temperature)
        return softmax(scaled, axis=1)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class HierarchicalFamilySpecialistModel(BaseEstimator):
    """Hierarchical two-stage classifier:

    Stage 1: 6-way Provider Family Classifier (claude, gemini, glm, gpt, kimi, qwen).
    Stage 2: Within-Family Specialists for multi-model families.
    Probability synthesis: P(model_i) = P(family_j) * P(model_i | family_j).
    """

    def __init__(self, C: float = 1.5, random_seed: int = RANDOM_SEED):
        self.C = C
        self.random_seed = random_seed
        self.family_classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_seed,
        )
        self.specialists: Dict[str, Union[LogisticRegression, str]] = {}

    def fit(self, X: sparse.csr_matrix, y_model_idx: np.ndarray):
        # Derive family labels
        y_family_idx = np.array([MODEL_TO_FAMILY_IDX[TARGET_MODELS[i]] for i in y_model_idx])
        self.family_classifier.fit(X, y_family_idx)

        # Train specialists for each family
        for family_name, model_names in PROVIDER_FAMILIES.items():
            if len(model_names) == 1:
                # Singleton family (kimi, qwen) -> probability is 1.0 given the family
                self.specialists[family_name] = model_names[0]
            else:
                family_idx = FAMILY_TO_IDX[family_name]
                mask = y_family_idx == family_idx
                if np.sum(mask) == 0:
                    continue
                X_fam = X[mask]
                y_fam_models = y_model_idx[mask]

                clf = LogisticRegression(
                    C=self.C,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=self.random_seed + family_idx,
                )
                clf.fit(X_fam, y_fam_models)
                self.specialists[family_name] = clf
        return self

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        N = X.shape[0]
        # P(family) of shape (N, 6)
        family_probs_raw = self.family_classifier.predict_proba(X)
        family_probs = np.zeros((N, NUM_FAMILIES), dtype=np.float32)
        for i, c in enumerate(self.family_classifier.classes_):
            family_probs[:, c] = family_probs_raw[:, i]

        final_probs = np.zeros((N, NUM_CLASSES), dtype=np.float32)

        for family_name, spec in self.specialists.items():
            fam_idx = FAMILY_TO_IDX[family_name]
            p_fam = family_probs[:, fam_idx : fam_idx + 1]  # shape (N, 1)

            if isinstance(spec, str):
                # Singleton family
                model_idx = MODEL_TO_IDX[spec]
                final_probs[:, model_idx] = p_fam.squeeze(1)
            else:
                # Multi-model family specialist
                spec_probs_raw = spec.predict_proba(X)  # shape (N, len(spec.classes_))
                for i, model_idx in enumerate(spec.classes_):
                    final_probs[:, model_idx] = p_fam[:, 0] * spec_probs_raw[:, i]

        # Normalize to ensure sum to 1.0
        row_sums = final_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return final_probs / row_sums


class BlendedAuthorshipModel(BaseEstimator):
    """Ensemble model blending Flat 18-class LogReg/Ridge with Hierarchical Family Specialists."""

    def __init__(
        self,
        C_flat: float = 1.2,
        C_hier: float = 1.5,
        hier_weight: float = 0.40,
        random_seed: int = RANDOM_SEED,
    ):
        self.C_flat = C_flat
        self.C_hier = C_hier
        self.hier_weight = hier_weight
        self.random_seed = random_seed

        self.flat_model = LogisticRegression(
            C=self.C_flat,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_seed,
        )
        self.hier_model = HierarchicalFamilySpecialistModel(
            C=self.C_hier,
            random_seed=self.random_seed,
        )

    def fit(self, X: sparse.csr_matrix, y_model_idx: np.ndarray):
        self.flat_model.fit(X, y_model_idx)
        self.hier_model.fit(X, y_model_idx)
        return self

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        flat_p_raw = self.flat_model.predict_proba(X)
        N = X.shape[0]
        flat_p = np.zeros((N, NUM_CLASSES), dtype=np.float32)
        for i, c in enumerate(self.flat_model.classes_):
            flat_p[:, c] = flat_p_raw[:, i]

        hier_p = self.hier_model.predict_proba(X)

        blended = (1.0 - self.hier_weight) * flat_p + self.hier_weight * hier_p
        # Normalize
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (blended / row_sums).astype(np.float32)

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    from src.config import TRAIN_PATH

    print("Testing Linear and Hierarchical Pipeline...")
    df = pd.read_csv(TRAIN_PATH)
    sample_df = df.head(100)
    texts = sample_df["text"]
    labels = np.array([MODEL_TO_IDX[m] for m in sample_df["model"]])

    print("Fitting MultiViewFeatureExtractor...")
    fe = MultiViewFeatureExtractor(
        max_char_features=1000,
        max_char_wb_features=500,
        max_word_case_features=500,
        max_word_lower_features=500,
        include_stylometrics=True,
    )
    X = fe.fit_transform(texts)
    print(f"Extracted sparse feature matrix shape: {X.shape}")

    print("Fitting BlendedAuthorshipModel...")
    blended = BlendedAuthorshipModel(C_flat=1.0, C_hier=1.0, hier_weight=0.4)
    blended.fit(X, labels)
    probs = blended.predict_proba(X)
    print(f"Predicted probabilities shape: {probs.shape}")
    assert probs.shape == (100, 18), "Invalid probability shape!"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1.0!"
    print("Linear models and family specialists verified successfully!")
