"""
Post-processing module for Name That LLM.
Implements:
1. Within-family probability gates (suppressing cross-family noise).
2. Prior-shift power scaling (alpha tuning for macro Balanced Accuracy).
3. Test marginal recalibration and class-threshold optimization.
4. Strict conversion and validation of predicted model names.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator

from src.config import (
    FAMILY_MODEL_INDICES,
    FAMILY_NAMES,
    FAMILY_TO_IDX,
    IDX_TO_MODEL,
    MODEL_TO_FAMILY,
    MODEL_TO_FAMILY_IDX,
    MODEL_TO_IDX,
    NUM_CLASSES,
    NUM_FAMILIES,
    PROVIDER_FAMILIES,
    TARGET_MODELS,
)


class WithinFamilyProbabilityGate:
    """Soft probability gating to reinforce provider family consistency.

    For model m in family f:
        P_gated(m) ~ P(m) * [P(f)]^gamma
    """

    def __init__(self, gamma: float = 0.35):
        self.gamma = gamma

    def apply_gate(self, probs: np.ndarray) -> np.ndarray:
        """Applies family gating to an (N, 18) probability matrix."""
        if self.gamma <= 0.0:
            return probs.copy()

        N = probs.shape[0]
        # Aggregate family probabilities by summing within-family models
        family_probs = np.zeros((N, NUM_FAMILIES), dtype=np.float32)
        for fam_name, fam_indices in FAMILY_MODEL_INDICES.items():
            fam_idx = FAMILY_TO_IDX[fam_name]
            family_probs[:, fam_idx] = probs[:, fam_indices].sum(axis=1)

        # Apply power gamma to family probabilities
        family_factors = np.power(np.maximum(family_probs, 1e-7), self.gamma)

        # Scale each model's probability by its family factor
        gated = np.zeros_like(probs, dtype=np.float32)
        for m_idx, model_name in enumerate(TARGET_MODELS):
            fam_idx = MODEL_TO_FAMILY_IDX[model_name]
            gated[:, m_idx] = probs[:, m_idx] * family_factors[:, fam_idx]

        # Re-normalize rows
        row_sums = gated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return gated / row_sums


class PriorShiftCalibrator:
    """Calibrates prediction probabilities for unequal test set priors and macro recall.

    1. Power scaling (alpha tuning): P_scaled ~ P^alpha.
    2. Prior balancing: Adjusts marginals towards uniform macro coverage.
    """

    def __init__(
        self,
        alpha: float = 0.90,
        balance_strength: float = 0.20,
    ):
        self.alpha = alpha
        self.balance_strength = balance_strength

    def fit_alpha(
        self,
        y_true_idx: np.ndarray,
        oof_probs: np.ndarray,
        alpha_candidates: List[float] = [0.70, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.15],
    ) -> float:
        """Tunes alpha on OOF probabilities to maximize macro Balanced Accuracy."""
        from sklearn.metrics import balanced_accuracy_score

        best_alpha = 1.0
        best_score = -1.0

        for a in alpha_candidates:
            scaled = np.power(np.maximum(oof_probs, 1e-9), a)
            scaled /= scaled.sum(axis=1, keepdims=True)
            preds = np.argmax(scaled, axis=1)
            score = balanced_accuracy_score(y_true_idx, preds)
            if score > best_score:
                best_score = score
                best_alpha = a

        self.alpha = best_alpha
        return best_alpha

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Transforms probabilities with power scaling and marginal balancing."""
        scaled = np.power(np.maximum(probs, 1e-9), self.alpha)
        scaled /= scaled.sum(axis=1, keepdims=True)

        if self.balance_strength > 0.0 and len(probs) >= 18:
            marginals = scaled.mean(axis=0)
            uniform_prior = 1.0 / NUM_CLASSES
            adj_factors = np.power(uniform_prior / np.maximum(marginals, 1e-4), self.balance_strength)
            scaled = scaled * adj_factors[np.newaxis, :]
            scaled /= scaled.sum(axis=1, keepdims=True)

        return scaled.astype(np.float32)


class NestedOOFLogitCalibrator(BaseEstimator):
    """L2-regularized Coordinate Ascent Optimizer for 18-class Logit Offsets and Marginal Normalization.

    Prediction rule:
        y_hat_i = argmax_k [ log(P_ik) - alpha * log(P_bar_k) + b_k ]
    Subject to:
        sum_k b_k = 0, ||b||_2 <= epsilon
    Evaluated with Nested OOF (fitting on 4 folds, testing on held-out fold) for zero leakage.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        epsilon: float = 1.2,
        max_iter: int = 3,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.b = np.zeros(NUM_CLASSES, dtype=np.float32)
        self.marginals = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES

    def fit(
        self,
        y_true_idx: np.ndarray,
        oof_probs: np.ndarray,
        alpha_grid: List[float] = [0.65, 0.75, 0.85, 0.95, 1.0, 1.10],
    ):
        """Learns optimal alpha and class logit offsets b via coordinate ascent."""
        from sklearn.metrics import balanced_accuracy_score

        N = len(y_true_idx)
        # Compute marginal prior distribution
        self.marginals = np.maximum(oof_probs.mean(axis=0), 1e-6)
        log_marginals = np.log(self.marginals)
        log_probs = np.log(np.maximum(oof_probs, 1e-9))

        best_score = -1.0
        best_alpha = self.alpha
        best_b = np.zeros(NUM_CLASSES, dtype=np.float32)

        # 1. First optimize alpha with b = 0
        for a in alpha_grid:
            logits = log_probs - a * log_marginals[np.newaxis, :]
            preds = np.argmax(logits, axis=1)
            score = balanced_accuracy_score(y_true_idx, preds)
            if score > best_score:
                best_score = score
                best_alpha = a

        self.alpha = best_alpha

        # 2. Coordinate ascent on class logit offsets b
        current_logits = log_probs - self.alpha * log_marginals[np.newaxis, :]
        b = np.zeros(NUM_CLASSES, dtype=np.float32)

        step_sizes = [0.25, 0.15, 0.08]
        for it in range(self.max_iter):
            delta = step_sizes[min(it, len(step_sizes) - 1)]
            improved = False
            for k in range(NUM_CLASSES):
                best_k_score = best_score
                best_k_delta = 0.0

                for candidate_delta in [-delta, delta]:
                    b_trial = b.copy()
                    b_trial[k] += candidate_delta
                    # Enforce sum to zero
                    b_trial -= np.mean(b_trial)
                    # Enforce L2 norm bound
                    norm = np.linalg.norm(b_trial)
                    if norm > self.epsilon:
                        b_trial = (b_trial / norm) * self.epsilon

                    trial_logits = current_logits + b_trial[np.newaxis, :]
                    preds = np.argmax(trial_logits, axis=1)
                    score = balanced_accuracy_score(y_true_idx, preds)

                    if score > best_k_score:
                        best_k_score = score
                        best_k_delta = candidate_delta

                if best_k_score > best_score:
                    b[k] += best_k_delta
                    b -= np.mean(b)
                    norm = np.linalg.norm(b)
                    if norm > self.epsilon:
                        b = (b / norm) * self.epsilon
                    best_score = best_k_score
                    improved = True

            if not improved:
                break

        self.b = b
        return self

    def transform(self, probs: np.ndarray, reference_marginals: Optional[np.ndarray] = None) -> np.ndarray:
        """Transforms probabilities using learned logit offsets and marginal adjustments."""
        marginals = reference_marginals if reference_marginals is not None else self.marginals
        log_marginals = np.log(np.maximum(marginals, 1e-6))
        log_probs = np.log(np.maximum(probs, 1e-9))

        adjusted_logits = log_probs - self.alpha * log_marginals[np.newaxis, :] + self.b[np.newaxis, :]
        return softmax(adjusted_logits, axis=1).astype(np.float32)

    def predict(self, probs: np.ndarray, reference_marginals: Optional[np.ndarray] = None) -> np.ndarray:
        transformed = self.transform(probs, reference_marginals)
        return np.argmax(transformed, axis=1)


def probabilities_to_labels(probs: np.ndarray) -> List[str]:
    """Converts probability array of shape (N, 18) to list of valid model string labels."""
    assert probs.ndim == 2 and probs.shape[1] == NUM_CLASSES, (
        f"Expected shape (N, {NUM_CLASSES}), got {probs.shape}"
    )
    pred_indices = np.argmax(probs, axis=1)
    return [IDX_TO_MODEL[int(i)] for i in pred_indices]


def validate_submission_dataframe(df: pd.DataFrame, expected_ids: List[str]) -> bool:
    """Validates that submission DataFrame strictly satisfies competition requirements:

    1. Exact columns 'id' and 'model'.
    2. Exact row count and matching 'id's.
    3. No NaNs or null values.
    4. Every label is strictly one of the 18 valid model strings.
    """
    if list(df.columns) != ["id", "model"]:
        raise ValueError(f"Invalid columns: {df.columns.tolist()}. Must be ['id', 'model']")

    if len(df) != len(expected_ids):
        raise ValueError(f"Row count mismatch: expected {len(expected_ids)}, got {len(df)}")

    if list(df["id"].values) != list(expected_ids):
        raise ValueError("Submission 'id' column does not match expected test IDs!")

    if df.isnull().any().any():
        raise ValueError("Submission contains NaN/null values!")

    invalid_models = set(df["model"]).difference(set(TARGET_MODELS))
    if len(invalid_models) > 0:
        raise ValueError(f"Invalid model strings in submission: {invalid_models}")

    return True


if __name__ == "__main__":
    print("Testing postprocess.py...")
    dummy_probs = np.random.dirichlet(np.ones(NUM_CLASSES), size=50)

    # Test WithinFamilyProbabilityGate
    gate = WithinFamilyProbabilityGate(gamma=0.35)
    gated_p = gate.apply_gate(dummy_probs)
    assert gated_p.shape == (50, NUM_CLASSES)
    assert np.allclose(gated_p.sum(axis=1), 1.0, atol=1e-5)

    # Test PriorShiftCalibrator
    calibrator = PriorShiftCalibrator(alpha=0.90, balance_strength=0.15)
    scaled_p = calibrator.transform(gated_p)
    assert scaled_p.shape == (50, NUM_CLASSES)
    assert np.allclose(scaled_p.sum(axis=1), 1.0, atol=1e-5)

    # Test conversion and validation
    labels = probabilities_to_labels(scaled_p)
    assert len(labels) == 50
    assert all(l in TARGET_MODELS for l in labels)

    test_ids = [f"id_{i:04d}" for i in range(50)]
    sub_df = pd.DataFrame({"id": test_ids, "model": labels})
    assert validate_submission_dataframe(sub_df, test_ids)
    print("postprocess.py verified successfully!")
