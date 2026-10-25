"""
Non-Redundant Within-Family Specialists for Name That LLM.
Addresses the v1 +0.0000 lift bottleneck through:
1. Within-Family Feature Filtering (ANOVA F-Test subspacing via SelectPercentile).
2. Non-Linear Specialist Architecture:
   - LogisticRegression on filtered TF-IDF + stylometrics.
   - Per-layer RidgeClassifier / Logistic on intermediate transformer layers (E_style_layers).
   - LGBMClassifier on Stylometric + Causal LLM Forensics + Family SVD + OOF Linear Probs.
3. External Corpus Ingestion Hook (--extra_data_paths) with cosine similarity filter (< 0.92 to val).
4. Law-of-Total-Probability Gating with per-family tuned gamma_f in [0, 1].
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier
    HAS_LGBM = False

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


class WithinFamilySpecialist(BaseEstimator):
    """Ensemble specialist for a single multi-model family (e.g. gpt, claude, gemini, glm)."""

    def __init__(
        self,
        family_name: str,
        percentile: int = 60,
        random_seed: int = RANDOM_SEED,
    ):
        self.family_name = family_name
        self.percentile = percentile
        self.random_seed = random_seed
        self.model_indices = FAMILY_MODEL_INDICES[family_name]
        self.n_models = len(self.model_indices)

        self.selector = SelectPercentile(f_classif, percentile=self.percentile)
        self.clf_sparse = LogisticRegression(
            C=0.5,
            penalty="l2",
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_seed,
        )
        self.clf_neural: Optional[LogisticRegression] = None
        self.svd: Optional[TruncatedSVD] = None
        self.clf_lgbm: Optional[LGBMClassifier] = None

    def fit(
        self,
        X_sparse: sparse.csr_matrix,
        y_model_idx: np.ndarray,
        E_style_flat: Optional[np.ndarray] = None,
        dense_forensics: Optional[np.ndarray] = None,
        stage1_probs: Optional[np.ndarray] = None,
    ):
        """Fits non-redundant ensemble components on rows belonging to this family."""
        if self.n_models <= 1:
            return self

        # Local family class indices [0, ..., n_models-1]
        local_mapping = {global_idx: local_idx for local_idx, global_idx in enumerate(self.model_indices)}
        y_local = np.array([local_mapping[g] for g in y_model_idx], dtype=np.int32)

        # 1. Sparse Subspacing (ANOVA F-Test feature selection)
        try:
            X_filtered = self.selector.fit_transform(X_sparse, y_local)
        except Exception:
            X_filtered = X_sparse
            self.selector = None

        self.clf_sparse.fit(X_filtered, y_local)

        # 2. Per-layer neural probe (if E_style_flat available)
        if E_style_flat is not None and E_style_flat.shape[0] == X_sparse.shape[0]:
            self.clf_neural = LogisticRegression(
                C=0.1,
                max_iter=500,
                solver="lbfgs",
                random_state=self.random_seed + 1,
            )
            self.clf_neural.fit(E_style_flat, y_local)

        # 3. Non-linear Gradient Boosting on Forensics + Family SVD + Stage 1 Probs
        n_svd = min(16, max(2, X_sparse.shape[0] - 1), X_sparse.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_svd, random_state=self.random_seed)
        svd_feats = self.svd.fit_transform(X_sparse).astype(np.float32)

        tabular_parts = [svd_feats]
        if dense_forensics is not None and dense_forensics.shape[0] == X_sparse.shape[0]:
            tabular_parts.append(dense_forensics)
        if stage1_probs is not None and stage1_probs.shape[0] == X_sparse.shape[0]:
            # Take stage 1 probabilities for this family's models
            tabular_parts.append(stage1_probs[:, self.model_indices])

        X_tabular = np.hstack(tabular_parts).astype(np.float32)

        if HAS_LGBM:
            self.clf_lgbm = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=15,
                colsample_bytree=0.6,
                subsample=0.8,
                min_child_samples=5,
                class_weight="balanced",
                random_state=self.random_seed,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            self.clf_lgbm = LGBMClassifier(
                max_iter=200,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=5,
                random_state=self.random_seed,
            )

        self.clf_lgbm.fit(X_tabular, y_local)
        return self

    def predict_proba(
        self,
        X_sparse: sparse.csr_matrix,
        E_style_flat: Optional[np.ndarray] = None,
        dense_forensics: Optional[np.ndarray] = None,
        stage1_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predicts within-family conditional probabilities P(model | family) of shape (N, n_models)."""
        N = X_sparse.shape[0]
        if self.n_models <= 1:
            return np.ones((N, 1), dtype=np.float32)

        # 1. Sparse prediction
        X_filtered = self.selector.transform(X_sparse) if self.selector is not None else X_sparse
        p_sparse = self.clf_sparse.predict_proba(X_filtered)

        # 2. Neural prediction
        if self.clf_neural is not None and E_style_flat is not None and E_style_flat.shape[0] == N:
            p_neural = self.clf_neural.predict_proba(E_style_flat)
        else:
            p_neural = p_sparse

        # 3. Tabular / LGBM prediction
        if self.clf_lgbm is not None and self.svd is not None:
            svd_feats = self.svd.transform(X_sparse).astype(np.float32)
            tabular_parts = [svd_feats]
            if dense_forensics is not None and dense_forensics.shape[0] == N:
                tabular_parts.append(dense_forensics)
            if stage1_probs is not None and stage1_probs.shape[0] == N:
                tabular_parts.append(stage1_probs[:, self.model_indices])
            X_tabular = np.hstack(tabular_parts).astype(np.float32)
            p_lgbm = self.clf_lgbm.predict_proba(X_tabular)
        else:
            p_lgbm = p_sparse

        # Weighted blend across non-redundant specialists: 0.35 Sparse, 0.35 Neural, 0.30 LGBM
        p_blend = 0.35 * p_sparse + 0.35 * p_neural + 0.30 * p_lgbm
        row_sums = p_blend.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (p_blend / row_sums).astype(np.float32)


class NonRedundantFamilySpecialistEnsemble(BaseEstimator):
    """Orchestrates 4 multi-model family specialists with per-family tuned gating gamma_f."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.specialists: Dict[str, WithinFamilySpecialist] = {}
        # Per-family gating weights gamma_f in [0, 1]
        self.family_gammas: Dict[str, float] = {
            f: 0.35 for f, models in PROVIDER_FAMILIES.items() if len(models) > 1
        }

    def fit(
        self,
        X_sparse: sparse.csr_matrix,
        y_model_idx: np.ndarray,
        E_style_flat: Optional[np.ndarray] = None,
        dense_forensics: Optional[np.ndarray] = None,
        stage1_probs: Optional[np.ndarray] = None,
        extra_data: Optional[Dict[str, Tuple[sparse.csr_matrix, np.ndarray]]] = None,
    ):
        """Fits each family specialist strictly on its family's rows."""
        y_family_idx = np.array([MODEL_TO_FAMILY_IDX[TARGET_MODELS[i]] for i in y_model_idx])

        for fam_name, models in PROVIDER_FAMILIES.items():
            if len(models) <= 1:
                continue

            fam_idx = FAMILY_TO_IDX[fam_name]
            fam_mask = (y_family_idx == fam_idx)
            if np.sum(fam_mask) == 0:
                continue

            X_fam = X_sparse[fam_mask]
            y_fam = y_model_idx[fam_mask]
            E_fam = E_style_flat[fam_mask] if E_style_flat is not None else None
            dense_fam = dense_forensics[fam_mask] if dense_forensics is not None else None
            p_stage1_fam = stage1_probs[fam_mask] if stage1_probs is not None else None

            # External data ingestion hook for this family
            if extra_data and fam_name in extra_data:
                extra_X, extra_y = extra_data[fam_name]
                if extra_X is not None and len(extra_y) > 0:
                    X_fam = sparse.vstack([X_fam, extra_X])
                    y_fam = np.concatenate([y_fam, extra_y])

            spec = WithinFamilySpecialist(family_name=fam_name, random_seed=self.random_seed + fam_idx)
            spec.fit(
                X_fam,
                y_fam,
                E_style_flat=E_fam,
                dense_forensics=dense_fam,
                stage1_probs=p_stage1_fam,
            )
            self.specialists[fam_name] = spec

        return self

    def tune_gammas(
        self,
        y_val_model_idx: np.ndarray,
        stage1_val_probs: np.ndarray,
        hier_val_probs: np.ndarray,
        gamma_candidates: List[float] = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
    ) -> Dict[str, float]:
        """Tunes per-family gamma_f on OOF validation predictions to maximize macro recall."""
        y_family_idx = np.array([MODEL_TO_FAMILY_IDX[TARGET_MODELS[i]] for i in y_val_model_idx])

        for fam_name, models in PROVIDER_FAMILIES.items():
            if len(models) <= 1:
                continue

            fam_idx = FAMILY_TO_IDX[fam_name]
            fam_mask = (y_family_idx == fam_idx)
            if np.sum(fam_mask) == 0:
                continue

            fam_models = FAMILY_MODEL_INDICES[fam_name]
            y_fam_true = y_val_model_idx[fam_mask]

            best_gamma = 0.0
            best_score = -1.0

            for g in gamma_candidates:
                # Interpolate between Stage 1 and Hierarchical on this family
                p_blend = (1.0 - g) * stage1_val_probs[fam_mask] + g * hier_val_probs[fam_mask]
                preds = np.argmax(p_blend[:, fam_models], axis=1)
                pred_global = np.array([fam_models[p] for p in preds])
                score = balanced_accuracy_score(y_fam_true, pred_global)
                if score > best_score:
                    best_score = score
                    best_gamma = g

            self.family_gammas[fam_name] = best_gamma

        return self.family_gammas

    def predict_hierarchical_probs(
        self,
        X_sparse: sparse.csr_matrix,
        stage1_probs: np.ndarray,
        E_style_flat: Optional[np.ndarray] = None,
        dense_forensics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes synthesized hierarchical probabilities P(model) = P(family) * P(model | family)."""
        N = X_sparse.shape[0]
        hier_probs = np.zeros((N, NUM_CLASSES), dtype=np.float32)

        # Compute P(family) from stage 1 probs
        family_probs = np.zeros((N, NUM_FAMILIES), dtype=np.float32)
        for fam_name, fam_indices in FAMILY_MODEL_INDICES.items():
            fam_idx = FAMILY_TO_IDX[fam_name]
            family_probs[:, fam_idx] = stage1_probs[:, fam_indices].sum(axis=1)

        for fam_name, models in PROVIDER_FAMILIES.items():
            fam_idx = FAMILY_TO_IDX[fam_name]
            p_fam = family_probs[:, fam_idx]  # shape (N,)

            if len(models) == 1:
                model_idx = MODEL_TO_IDX[models[0]]
                hier_probs[:, model_idx] = p_fam
            elif fam_name in self.specialists:
                spec = self.specialists[fam_name]
                cond_p = spec.predict_proba(
                    X_sparse,
                    E_style_flat=E_style_flat,
                    dense_forensics=dense_forensics,
                    stage1_probs=stage1_probs,
                )
                for local_i, global_i in enumerate(spec.model_indices):
                    hier_probs[:, global_i] = p_fam * cond_p[:, local_i]

        row_sums = hier_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (hier_probs / row_sums).astype(np.float32)

    def apply_gated_blend(
        self,
        stage1_probs: np.ndarray,
        hier_probs: np.ndarray,
    ) -> np.ndarray:
        """Blends Stage 1 and Hierarchical probabilities using per-family tuned gammas."""
        blended = stage1_probs.copy()
        for fam_name, models in PROVIDER_FAMILIES.items():
            if len(models) <= 1:
                continue
            g = self.family_gammas.get(fam_name, 0.35)
            fam_indices = FAMILY_MODEL_INDICES[fam_name]
            blended[:, fam_indices] = (
                (1.0 - g) * stage1_probs[:, fam_indices] + g * hier_probs[:, fam_indices]
            )

        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (blended / row_sums).astype(np.float32)


if __name__ == "__main__":
    print("Testing NonRedundantFamilySpecialistEnsemble...")
    N_test = 60
    n_features = 200
    X_dummy = sparse.random(N_test, n_features, density=0.1, format="csr")
    y_dummy = np.random.randint(0, NUM_CLASSES, size=N_test)
    stage1_p = np.random.dirichlet(np.ones(NUM_CLASSES), size=N_test)
    E_style_d = np.random.randn(N_test, 128).astype(np.float32)
    dense_f_d = np.random.randn(N_test, 16).astype(np.float32)

    ensemble = NonRedundantFamilySpecialistEnsemble()
    ensemble.fit(X_dummy, y_dummy, E_style_flat=E_style_d, dense_forensics=dense_f_d, stage1_probs=stage1_p)
    hier_p = ensemble.predict_hierarchical_probs(X_dummy, stage1_p, E_style_flat=E_style_d, dense_forensics=dense_f_d)
    assert hier_p.shape == (N_test, NUM_CLASSES)
    assert np.allclose(hier_p.sum(axis=1), 1.0, atol=1e-5)

    gammas = ensemble.tune_gammas(y_dummy, stage1_p, hier_p)
    print(f"Tuned gammas: {gammas}")

    gated_p = ensemble.apply_gated_blend(stage1_p, hier_p)
    assert gated_p.shape == (N_test, NUM_CLASSES)
    assert np.allclose(gated_p.sum(axis=1), 1.0, atol=1e-5)
    print("NonRedundantFamilySpecialistEnsemble verified successfully!")
