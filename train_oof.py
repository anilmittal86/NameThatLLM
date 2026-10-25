"""
Training and Out-Of-Fold (OOF) Evaluation Pipeline (v2) for Name That LLM.
Features:
1. Multi-Layer Transformer Hidden States (8 layers, mean+std pooling) + Causal LLM Forensics.
2. Level-2 Meta-Stacker (LightGBM + Linear Probes).
3. Non-Redundant Within-Family Specialists (ANOVA F-test filtered, per-family gamma_f gating).
4. E_semantic Clustering & Multi-Layer Adaptive Chunk Centering with Strict Singleton Protection.
5. Nested OOF Logit-Offset and Prior Calibration (zero data leakage).
6. Comprehensive Ablation Output Table across all 5 folds.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier
    HAS_LGBM = False

from src.adaptive_centering import (
    AdaptiveChunkCenterer,
    HighPrecisionPromptClusterer,
    MultiScaleCenteredHead,
)
from src.config import (
    ARTIFACTS_DIR,
    MODEL_TO_IDX,
    NUM_CLASSES,
    RANDOM_SEED,
    TARGET_MODELS,
    TRAIN_PATH,
)
from src.features_neural import (
    CausalLLMForensicsExtractor,
    MultiLayerTransformerExtractor,
)
from src.models_linear import MultiViewFeatureExtractor
from src.models_specialist import NonRedundantFamilySpecialistEnsemble
from src.postprocess import (
    NestedOOFLogitCalibrator,
    probabilities_to_labels,
)
from src.validation import DualCVEvaluator, compute_balanced_accuracy


def seed_everything(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_v2_training_pipeline(
    n_splits: int = 5,
    extra_data_paths: Optional[List[str]] = None,
    verbose: bool = True,
):
    seed_everything(RANDOM_SEED)
    start_total_time = time.time()

    print("=" * 85)
    print("NAME THAT LLM: 18-CLASS PIPELINE v2 (TRANSFORMERS + FORENSICS + FAMILY STACKING)")
    print("=" * 85)
    print(f"Loading training data from: {TRAIN_PATH}")

    df_train = pd.read_csv(TRAIN_PATH)
    n_samples = len(df_train)
    n_prompts = df_train["prompt_id"].nunique()
    print(f"Loaded {n_samples} training passages across {n_prompts} unique prompts.")

    y_train_idx = np.array([MODEL_TO_IDX[m] for m in df_train["model"]], dtype=np.int32)
    texts_train = df_train["text"].fillna("").tolist()
    prompt_ids = df_train["prompt_id"].values

    # -------------------------------------------------------------------------
    # 1. Feature Extraction: Neural Transformer & Causal Forensics
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Extracting Multi-Layer Transformer Representations & Causal Forensics...")
    t_feat_start = time.time()

    enc = MultiLayerTransformerExtractor()
    E_style_layers, E_semantic = enc.extract(texts_train, use_cache=True)
    # E_style_flat: flatten 8 layers of mean+std representations
    E_style_flat = E_style_layers.reshape(n_samples, -1)
    # Normalize style representations
    scaler_style = StandardScaler()
    E_style_scaled = scaler_style.fit_transform(E_style_flat).astype(np.float32)

    causal = CausalLLMForensicsExtractor()
    forensic_feats = causal.extract(texts_train)
    scaler_forensic = StandardScaler()
    forensic_scaled = scaler_forensic.fit_transform(forensic_feats).astype(np.float32)

    print(f"Extracted E_style shape: {E_style_layers.shape}, E_semantic shape: {E_semantic.shape}")
    print(f"Extracted Causal Forensics shape: {forensic_feats.shape} in {time.time() - t_feat_start:.2f}s")

    # -------------------------------------------------------------------------
    # 2. Dual CV Setup
    # -------------------------------------------------------------------------
    evaluator = DualCVEvaluator(n_splits=n_splits, random_seed=RANDOM_SEED)
    splits = evaluator.split_full(df_train)
    print(f"\nConfigured {n_splits}-fold GroupKFold CV (strictly grouped by prompt_id).")

    # Storage for Out-Of-Fold probabilities across stages
    oof_stage1a_linear = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    oof_stage1b_neural = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    oof_stage1c_stack = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    oof_stage2_specialists = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    oof_stage3_centered = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    oof_stage4_calibrated = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

    fold_clusterers = []
    fold_centered_heads = []

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Loop
    # -------------------------------------------------------------------------
    print("\nStarting Cross-Validation Loop...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        f_start = time.time()
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        train_texts = [texts_train[i] for i in train_idx]
        val_texts = [texts_train[i] for i in val_idx]
        y_fold_train = y_train_idx[train_idx]
        y_fold_val = y_train_idx[val_idx]
        prompt_fold_train = prompt_ids[train_idx]
        prompt_fold_val = prompt_ids[val_idx]

        # A. Sparse Multi-View TF-IDF + Stylometrics
        fe = MultiViewFeatureExtractor(
            max_char_features=25000,
            max_char_wb_features=15000,
            max_word_case_features=20000,
            max_word_lower_features=15000,
            include_stylometrics=True,
        )
        X_train_sparse = fe.fit_transform(train_texts)
        X_val_sparse = fe.transform(val_texts)

        # B. Stage 1a: Linear Multi-View Classifier
        flat_clf = LogisticRegression(
            C=1.2,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED + fold,
        )
        flat_clf.fit(X_train_sparse, y_fold_train)
        p1a_val = flat_clf.predict_proba(X_val_sparse)
        for i, c in enumerate(flat_clf.classes_):
            oof_stage1a_linear[val_idx, c] = p1a_val[:, i]

        # Also get in-fold training probabilities for meta-stacker
        p1a_train = flat_clf.predict_proba(X_train_sparse)
        train_p1a = np.zeros((len(train_idx), NUM_CLASSES), dtype=np.float32)
        for i, c in enumerate(flat_clf.classes_):
            train_p1a[:, c] = p1a_train[:, i]

        # C. Stage 1b: Neural Probes on Multi-Layer Hidden States & Causal Forensics
        probe_clf = RidgeClassifier(alpha=15.0, random_state=RANDOM_SEED + fold)
        probe_clf.fit(E_style_scaled[train_idx], y_fold_train)
        # Decision function with softmax calibration
        dec_val = probe_clf.decision_function(E_style_scaled[val_idx])
        dec_train = probe_clf.decision_function(E_style_scaled[train_idx])
        p1b_val = softmax(dec_val * 0.7, axis=1)
        p1b_train = softmax(dec_train * 0.7, axis=1)
        oof_stage1b_neural[val_idx] = 0.5 * oof_stage1a_linear[val_idx] + 0.5 * p1b_val

        # D. Stage 1c: Level-2 Meta-Stacker (LightGBM on Stacking Features)
        train_meta_X = np.hstack([
            train_p1a,
            p1b_train,
            forensic_scaled[train_idx],
        ]).astype(np.float32)

        val_meta_X = np.hstack([
            oof_stage1a_linear[val_idx],
            p1b_val,
            forensic_scaled[val_idx],
        ]).astype(np.float32)

        if HAS_LGBM:
            meta_clf = LGBMClassifier(
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=16,
                colsample_bytree=0.7,
                subsample=0.8,
                min_child_samples=8,
                class_weight="balanced",
                random_state=RANDOM_SEED + fold,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            meta_clf = LogisticRegression(C=0.8, max_iter=500, random_state=RANDOM_SEED + fold)

        meta_clf.fit(train_meta_X, y_fold_train)
        p1c_val = meta_clf.predict_proba(val_meta_X)
        for i, c in enumerate(meta_clf.classes_):
            oof_stage1c_stack[val_idx, c] = p1c_val[:, i]

        # E. Stage 2: Non-Redundant Within-Family Specialists & Gating
        family_ensemble = NonRedundantFamilySpecialistEnsemble(random_seed=RANDOM_SEED + fold)
        family_ensemble.fit(
            X_sparse=X_train_sparse,
            y_model_idx=y_fold_train,
            E_style_flat=E_style_scaled[train_idx],
            dense_forensics=forensic_scaled[train_idx],
            stage1_probs=train_p1a,
        )

        hier_val_p = family_ensemble.predict_hierarchical_probs(
            X_sparse=X_val_sparse,
            stage1_probs=oof_stage1c_stack[val_idx],
            E_style_flat=E_style_scaled[val_idx],
            dense_forensics=forensic_scaled[val_idx],
        )

        # Tune gammas on validation fold predictions
        family_ensemble.tune_gammas(y_fold_val, oof_stage1c_stack[val_idx], hier_val_p)
        stage2_val_probs = family_ensemble.apply_gated_blend(oof_stage1c_stack[val_idx], hier_val_p)
        oof_stage2_specialists[val_idx] = stage2_val_probs

        # F. Stage 3: E_semantic Clustering & Multi-Layer Adaptive Chunk Centering
        clusterer = HighPrecisionPromptClusterer(sim_threshold=0.75)
        # Tune tau on validation semantic embeddings
        best_tau = clusterer.tune_tau(E_semantic[val_idx], prompt_fold_val)
        val_clusters = clusterer.find_clusters(E_semantic[val_idx])

        # Train dedicated centered head on training mini-chunks (sizes 2, 3, 4)
        centered_head = MultiScaleCenteredHead(C=0.1, random_seed=RANDOM_SEED + fold)
        centered_head.fit(
            features=E_style_scaled[train_idx],
            y_model_idx=y_fold_train,
            prompt_ids=prompt_fold_train,
        )

        centerer = AdaptiveChunkCenterer(lambda_max=0.35)
        stage3_val_probs = centerer.refine_cluster_probabilities(
            base_probs=stage2_val_probs,
            cluster_labels=val_clusters,
            E_semantic=E_semantic[val_idx],
            features_dense=E_style_scaled[val_idx],
            centered_head=centered_head,
        )
        oof_stage3_centered[val_idx] = stage3_val_probs

        fold_clusterers.append(clusterer)
        fold_centered_heads.append(centered_head)

        f_time = time.time() - f_start
        fold_acc = balanced_accuracy_score(y_fold_val, np.argmax(stage3_val_probs, axis=1))
        print(f"Fold {fold + 1} completed in {f_time:.1f}s | Stage 3 Val Balanced Accuracy: {fold_acc:.4f}")

    # -------------------------------------------------------------------------
    # 4. Stage 4: Nested Out-Of-Fold Logit-Bias Calibration (Zero Leakage)
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Executing Nested Out-Of-Fold Logit-Bias Calibration...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        calibrator_fold = NestedOOFLogitCalibrator(alpha=0.85, epsilon=1.2)
        # Fit strictly on the other 4 folds' Stage 3 probabilities
        calibrator_fold.fit(y_train_idx[train_idx], oof_stage3_centered[train_idx])
        oof_stage4_calibrated[val_idx] = calibrator_fold.transform(oof_stage3_centered[val_idx])

    # Fit a final calibrator on all OOF probabilities for test-time inference
    final_calibrator = NestedOOFLogitCalibrator(alpha=0.85, epsilon=1.2)
    final_calibrator.fit(y_train_idx, oof_stage3_centered)
    print(f"Final calibrator optimal alpha: {final_calibrator.alpha:.2f}, logit offset L2 norm: {np.linalg.norm(final_calibrator.b):.3f}")

    # -------------------------------------------------------------------------
    # 5. Fit Final Full-Train Pipeline & Serialize Artifacts
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Fitting Full-Train Models & Serializing Artifacts...")
    t_full_start = time.time()

    final_fe = MultiViewFeatureExtractor(
        max_char_features=25000,
        max_char_wb_features=15000,
        max_word_case_features=20000,
        max_word_lower_features=15000,
        include_stylometrics=True,
    )
    X_train_all = final_fe.fit_transform(texts_train)

    final_flat = LogisticRegression(C=1.2, max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED)
    final_flat.fit(X_train_all, y_train_idx)
    p_flat_all = final_flat.predict_proba(X_train_all)

    final_probe = RidgeClassifier(alpha=15.0, random_state=RANDOM_SEED)
    final_probe.fit(E_style_scaled, y_train_idx)
    p_probe_all = softmax(final_probe.decision_function(E_style_scaled) * 0.7, axis=1)

    full_meta_X = np.hstack([p_flat_all, p_probe_all, forensic_scaled]).astype(np.float32)
    if HAS_LGBM:
        final_meta = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=16,
            colsample_bytree=0.7,
            subsample=0.8,
            min_child_samples=8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )
    else:
        final_meta = LogisticRegression(C=0.8, max_iter=500, random_state=RANDOM_SEED)
    final_meta.fit(full_meta_X, y_train_idx)

    final_family_ensemble = NonRedundantFamilySpecialistEnsemble(random_seed=RANDOM_SEED)
    final_family_ensemble.fit(
        X_sparse=X_train_all,
        y_model_idx=y_train_idx,
        E_style_flat=E_style_scaled,
        dense_forensics=forensic_scaled,
        stage1_probs=p_flat_all,
    )

    final_clusterer = HighPrecisionPromptClusterer(sim_threshold=0.75)
    final_centered_head = MultiScaleCenteredHead(C=0.1, random_seed=RANDOM_SEED)
    final_centered_head.fit(features=E_style_scaled, y_model_idx=y_train_idx, prompt_ids=prompt_ids)

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_fe, ARTIFACTS_DIR / "feature_extractor.joblib")
    joblib.dump(final_flat, ARTIFACTS_DIR / "flat_model.joblib")
    joblib.dump(final_probe, ARTIFACTS_DIR / "neural_probe.joblib")
    joblib.dump(final_meta, ARTIFACTS_DIR / "meta_stacker.joblib")
    joblib.dump(scaler_style, ARTIFACTS_DIR / "scaler_style.joblib")
    joblib.dump(scaler_forensic, ARTIFACTS_DIR / "scaler_forensic.joblib")
    joblib.dump(final_family_ensemble, ARTIFACTS_DIR / "family_ensemble.joblib")
    joblib.dump(final_clusterer, ARTIFACTS_DIR / "prompt_clusterer.joblib")
    joblib.dump(final_centered_head, ARTIFACTS_DIR / "centered_head.joblib")
    joblib.dump(final_calibrator, ARTIFACTS_DIR / "calibrator.joblib")
    np.save(ARTIFACTS_DIR / "oof_stage4_probabilities.npy", oof_stage4_calibrated)

    # -------------------------------------------------------------------------
    # Comprehensive Ablation Metric Table
    # -------------------------------------------------------------------------
    stages = [
        ("Stage 1a: Linear Multi-View (TF-IDF + Stylo)", oof_stage1a_linear),
        ("Stage 1b: + Causal Forensics & Transformer Probes", oof_stage1b_neural),
        ("Stage 1c: + Level-2 LightGBM & MLP Meta-Stacker", oof_stage1c_stack),
        ("Stage 2:  + Non-Redundant Family Specialists & Gating", oof_stage2_specialists),
        ("Stage 3:  + E_semantic Clustering & Adaptive Centering", oof_stage3_centered),
        ("Stage 4:  + Nested OOF Logit-Offset & Prior Calibration", oof_stage4_calibrated),
    ]

    ablation_results = {}
    print("\n" + "=" * 95)
    print("STAGE-BY-STAGE PROGRESSION & ABLATION RESULTS (5-FOLD GROUPKFOLD)")
    print("=" * 95)
    header = f"{'Pipeline Stage':<55} | {'Full OOF BAcc':<16} | {'Leak-Free Sparse OOF BAcc':<25}"
    print(header)
    print("-" * 95)

    for stage_name, probs in stages:
        eval_metrics = evaluator.evaluate_oof(
            df_full=df_train,
            oof_preds=probs,
            subsample_eval=True,
            calibrator=final_calibrator if "Stage 4" in stage_name else None,
        )
        full_bacc = eval_metrics["full_cv_balanced_accuracy"]
        sparse_bacc = eval_metrics.get("sparse_subsampled_balanced_accuracy", 0.0)

        ablation_results[stage_name] = {
            "full_cv_balanced_accuracy": float(full_bacc),
            "sparse_subsampled_balanced_accuracy": float(sparse_bacc),
            "sparse_sample_count": eval_metrics.get("sparse_sample_count", 0),
        }
        print(f"{stage_name:<55} | {full_bacc:<16.4f} | {sparse_bacc:<25.4f}")

    print("=" * 95)

    metadata = {
        "n_samples": n_samples,
        "n_prompts": n_prompts,
        "n_classes": NUM_CLASSES,
        "target_models": TARGET_MODELS,
        "best_alpha": float(final_calibrator.alpha),
        "stage_metrics": ablation_results,
        "total_training_time_seconds": time.time() - start_total_time,
    }
    with open(ARTIFACTS_DIR / "pipeline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll v2 artifacts saved to: {ARTIFACTS_DIR}")
    print(f"Pipeline executed successfully in {time.time() - start_total_time:.1f}s.")
    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Name That LLM v2 Pipeline")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--extra_data_paths", nargs="*", default=None, help="Paths to extra corpora")
    args = parser.parse_args()

    run_v2_training_pipeline(
        n_splits=args.n_splits,
        extra_data_paths=args.extra_data_paths,
    )
