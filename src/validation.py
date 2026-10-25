"""
Validation module for Name That LLM.
Implements Dual CV evaluation:
1. Full GroupKFold (grouped by prompt_id, ensuring zero prompt leakage).
2. Sparse Subsampled CV (simulating the test set's sparse 1..4 passages per prompt & singletons).
Evaluates using sklearn.metrics.balanced_accuracy_score (macro-averaged recall across 18 classes).
"""

from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold

from src.config import (
    MODEL_TO_IDX,
    NUM_CLASSES,
    RANDOM_SEED,
    TARGET_MODELS,
)


def compute_balanced_accuracy(
    y_true: Union[List[Union[str, int]], np.ndarray, pd.Series],
    y_pred: Union[List[Union[str, int]], np.ndarray, pd.Series],
) -> float:
    """Computes the competition evaluation metric: balanced accuracy score."""
    # Convert string model labels to integer indices if necessary
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    if len(y_true) > 0 and isinstance(y_true[0], str):
        y_true = np.array([MODEL_TO_IDX.get(m, -1) for m in y_true])
    if len(y_pred) > 0 and isinstance(y_pred[0], str):
        y_pred = np.array([MODEL_TO_IDX.get(m, -1) for m in y_pred])

    return float(balanced_accuracy_score(y_true, y_pred))


def compute_per_class_recalls(
    y_true: Union[List[Union[str, int]], np.ndarray],
    y_pred: Union[List[Union[str, int]], np.ndarray],
) -> Dict[str, float]:
    """Computes individual recall for each of the 18 classes."""
    if len(y_true) > 0 and isinstance(y_true[0], str):
        y_true_idx = np.array([MODEL_TO_IDX[m] for m in y_true])
    else:
        y_true_idx = np.array(y_true)

    if len(y_pred) > 0 and isinstance(y_pred[0], str):
        y_pred_idx = np.array([MODEL_TO_IDX[m] for m in y_pred])
    else:
        y_pred_idx = np.array(y_pred)

    recalls = {}
    for idx, model_name in enumerate(TARGET_MODELS):
        mask = y_true_idx == idx
        if np.sum(mask) == 0:
            recalls[model_name] = 0.0
        else:
            recalls[model_name] = float(np.mean(y_pred_idx[mask] == idx))
    return recalls


class DualCVEvaluator:
    """Dual Cross-Validation Evaluator for 18-class authorship attribution.

    Provides:
    1. Full GroupKFold splits (grouped by prompt_id).
    2. Test-distribution-matched Sparse Subsampling (prompt size k in [1..4] with singletons).
    """

    def __init__(
        self,
        n_splits: int = 5,
        group_col: str = "prompt_id",
        target_col: str = "model",
        random_seed: int = RANDOM_SEED,
    ):
        self.n_splits = n_splits
        self.group_col = group_col
        self.target_col = target_col
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def split_full(
        self, df: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generates GroupKFold train/val index splits based on prompt_id."""
        groups = df[self.group_col].values
        gkf = GroupKFold(n_splits=self.n_splits)
        splits = list(gkf.split(df, groups=groups))
        # Verify zero group leakage
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_prompts = set(df.iloc[train_idx][self.group_col])
            val_prompts = set(df.iloc[val_idx][self.group_col])
            leakage = train_prompts.intersection(val_prompts)
            assert len(leakage) == 0, f"Prompt leakage in fold {fold}: {leakage}"
        return splits

    def generate_sparse_subsample(
        self,
        df_val: pd.DataFrame,
        k_min: int = 1,
        k_max: int = 4,
        singleton_prob: float = 0.40,
        unequal_prior_strength: float = 0.5,
    ) -> pd.DataFrame:
        """Subsamples validation prompts to mimic the hidden test set structure:

        - Prompt groups are small (1 to 4 passages per prompt, averaging ~2.3).
        - Many singletons (~40% of unique prompts appear only once).
        - Unequal class priors across passages.
        """
        rng = np.random.RandomState(self.random_seed + 100)
        selected_rows = []

        # Optional prior skew weights across 18 models to simulate test prior shifts
        if unequal_prior_strength > 0:
            skew_weights = rng.dirichlet(np.ones(NUM_CLASSES) * (2.0 - unequal_prior_strength))
            model_weights = {TARGET_MODELS[i]: skew_weights[i] for i in range(NUM_CLASSES)}
        else:
            model_weights = {m: 1.0 for m in TARGET_MODELS}

        grouped = df_val.groupby(self.group_col)
        for prompt_id, group in grouped:
            n_available = len(group)
            if n_available == 0:
                continue

            # Determine k for this prompt group
            if rng.uniform() < singleton_prob or n_available == 1:
                k = 1
            else:
                k = rng.randint(k_min, min(k_max, n_available) + 1)

            # Sample k passages with model weights
            if unequal_prior_strength > 0:
                p = np.array([model_weights.get(m, 1.0) for m in group[self.target_col]])
                p = p / p.sum()
                chosen_idx = rng.choice(group.index.values, size=k, replace=False, p=p)
            else:
                chosen_idx = rng.choice(group.index.values, size=k, replace=False)

            selected_rows.extend(chosen_idx)

        subsampled_df = df_val.loc[selected_rows].copy()
        return subsampled_df

    def evaluate_oof(
        self,
        df_full: pd.DataFrame,
        oof_preds: Union[np.ndarray, List[str]],
        subsample_eval: bool = True,
        calibrator = None,
    ) -> Dict[str, float]:
        """Evaluates OOF predictions on full dataset and leak-free sparse subsampled scenario."""
        y_true = df_full[self.target_col].values

        if isinstance(oof_preds, np.ndarray) and oof_preds.ndim == 2:
            # Full population marginals computed from all out-of-fold samples
            full_marginals = np.maximum(oof_preds.mean(axis=0), 1e-6)
            if calibrator is not None and hasattr(calibrator, "transform"):
                calibrated_probs = calibrator.transform(oof_preds, reference_marginals=full_marginals)
                y_pred_idx = np.argmax(calibrated_probs, axis=1)
            else:
                y_pred_idx = np.argmax(oof_preds, axis=1)
            y_pred_str = [TARGET_MODELS[i] for i in y_pred_idx]
        else:
            y_pred_str = list(oof_preds)

        full_bal_acc = compute_balanced_accuracy(y_true, y_pred_str)
        results = {"full_cv_balanced_accuracy": full_bal_acc}

        if subsample_eval:
            # Subsample prompts with k ~ Uniform(1, 4)
            sparse_df = self.generate_sparse_subsample(df_full)
            sparse_indices = sparse_df.index.values
            sparse_y_true = df_full.loc[sparse_indices, self.target_col].values

            if isinstance(oof_preds, np.ndarray) and oof_preds.ndim == 2:
                # Use full population marginals (zero leakage, no micro-batch bias)
                if calibrator is not None and hasattr(calibrator, "transform"):
                    sparse_probs = calibrator.transform(
                        oof_preds[sparse_indices], reference_marginals=full_marginals
                    )
                    sparse_pred_idx = np.argmax(sparse_probs, axis=1)
                else:
                    sparse_pred_idx = np.argmax(oof_preds[sparse_indices], axis=1)
                sparse_y_pred = [TARGET_MODELS[i] for i in sparse_pred_idx]
            else:
                sparse_y_pred = [y_pred_str[i] for i in sparse_indices]

            # Pool predictions across all validation prompts before computing balanced accuracy
            sparse_bal_acc = compute_balanced_accuracy(sparse_y_true, sparse_y_pred)
            results["sparse_subsampled_balanced_accuracy"] = sparse_bal_acc
            results["sparse_sample_count"] = len(sparse_indices)
            results["sparse_prompt_count"] = sparse_df[self.group_col].nunique()

        return results


if __name__ == "__main__":
    from src.config import TRAIN_PATH

    print("Testing Dual CV Evaluator...")
    df = pd.read_csv(TRAIN_PATH)
    evaluator = DualCVEvaluator(n_splits=5)
    splits = evaluator.split_full(df)
    print(f"Generated {len(splits)} folds. Fold 0 train: {len(splits[0][0])}, val: {len(splits[0][1])}")

    sparse_val = evaluator.generate_sparse_subsample(df)
    print(f"Sparse subsample created: {len(sparse_val)} rows from {sparse_val['prompt_id'].nunique()} prompts.")
    counts = sparse_val.groupby("prompt_id").size().value_counts().to_dict()
    print(f"Prompt group size distribution (k): {counts}")

    dummy_preds = np.random.choice(TARGET_MODELS, size=len(df))
    metrics = evaluator.evaluate_oof(df, dummy_preds)
    print(f"Dummy random metrics: {metrics}")
    print("Dual CV Evaluator verified successfully!")
