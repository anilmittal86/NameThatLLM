"""
Standalone Inference and Submission Script (v2) for Name That LLM.
Generates submission.parquet adhering strictly to all Kaggle competition constraints:
- Exact columns: 'id', 'model'
- 18 valid target model strings
- No duplicate IDs, no NaNs
- Singleton-protected adaptive centering on E_semantic
- Offline-resilient and compatible with Kaggle T4 / CPU environments
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

# Add workspace directory to sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from src.adaptive_centering import (
    AdaptiveChunkCenterer,
    HighPrecisionPromptClusterer,
    MultiScaleCenteredHead,
)
from src.config import (
    ARTIFACTS_DIR,
    SAMPLE_SUB_PATH,
    SUBMISSION_PATH,
    TARGET_MODELS,
    TEST_PATH,
)
from src.features_neural import (
    CausalLLMForensicsExtractor,
    MultiLayerTransformerExtractor,
)
from src.postprocess import (
    NestedOOFLogitCalibrator,
    probabilities_to_labels,
    validate_submission_dataframe,
)


def load_v2_artifacts():
    """Loads all trained v2 pipeline artifacts from ARTIFACTS_DIR, running training if needed."""
    required_files = [
        "feature_extractor.joblib",
        "flat_model.joblib",
        "neural_probe.joblib",
        "meta_stacker.joblib",
        "scaler_style.joblib",
        "scaler_forensic.joblib",
        "family_ensemble.joblib",
        "prompt_clusterer.joblib",
        "centered_head.joblib",
        "calibrator.joblib",
    ]
    missing = [f for f in required_files if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        print(f"Notice: Missing artifacts {missing}. Running v2 training pipeline...")
        from train_oof import run_v2_training_pipeline
        run_v2_training_pipeline(n_splits=5)

    fe = joblib.load(ARTIFACTS_DIR / "feature_extractor.joblib")
    flat_model = joblib.load(ARTIFACTS_DIR / "flat_model.joblib")
    neural_probe = joblib.load(ARTIFACTS_DIR / "neural_probe.joblib")
    meta_stacker = joblib.load(ARTIFACTS_DIR / "meta_stacker.joblib")
    scaler_style = joblib.load(ARTIFACTS_DIR / "scaler_style.joblib")
    scaler_forensic = joblib.load(ARTIFACTS_DIR / "scaler_forensic.joblib")
    family_ensemble = joblib.load(ARTIFACTS_DIR / "family_ensemble.joblib")
    prompt_clusterer = joblib.load(ARTIFACTS_DIR / "prompt_clusterer.joblib")
    centered_head = joblib.load(ARTIFACTS_DIR / "centered_head.joblib")
    calibrator = joblib.load(ARTIFACTS_DIR / "calibrator.joblib")

    return (
        fe,
        flat_model,
        neural_probe,
        meta_stacker,
        scaler_style,
        scaler_forensic,
        family_ensemble,
        prompt_clusterer,
        centered_head,
        calibrator,
    )


def run_inference(test_csv_path: Path = TEST_PATH, output_path: Path = SUBMISSION_PATH):
    start_time = time.time()
    print("=" * 85)
    print("NAME THAT LLM: STANDALONE v2 INFERENCE & SUBMISSION GENERATOR")
    print("=" * 85)
    print(f"Loading test data from: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    test_ids = list(df_test["id"].values)
    test_texts = df_test["text"].fillna("").tolist()
    n_test = len(df_test)
    print(f"Loaded {n_test} test passages for attribution.")

    # 1. Load pipeline artifacts
    print("Loading v2 pipeline artifacts...")
    (
        fe,
        flat_model,
        neural_probe,
        meta_stacker,
        scaler_style,
        scaler_forensic,
        family_ensemble,
        prompt_clusterer,
        centered_head,
        calibrator,
    ) = load_v2_artifacts()

    # 2. Extract Features
    print("Extracting multi-view TF-IDF & stylometrics...")
    X_test_sparse = fe.transform(test_texts)

    print("Extracting multi-layer transformer representations & causal forensics...")
    enc = MultiLayerTransformerExtractor()
    E_style_layers, E_semantic = enc.extract(test_texts, use_cache=True)
    E_style_flat = E_style_layers.reshape(n_test, -1)
    E_style_scaled = scaler_style.transform(E_style_flat).astype(np.float32)

    causal = CausalLLMForensicsExtractor()
    forensic_feats = causal.extract(test_texts)
    forensic_scaled = scaler_forensic.transform(forensic_feats).astype(np.float32)

    # 3. Stage 1 Base Predictions & Meta-Stacking
    print("Computing Stage 1 flat & neural probe probabilities...")
    p_flat_raw = flat_model.predict_proba(X_test_sparse)
    p_flat = np.zeros((n_test, len(TARGET_MODELS)), dtype=np.float32)
    for i, c in enumerate(flat_model.classes_):
        p_flat[:, c] = p_flat_raw[:, i]

    p_probe = softmax(neural_probe.decision_function(E_style_scaled) * 0.7, axis=1)

    meta_X = np.hstack([p_flat, p_probe, forensic_scaled]).astype(np.float32)
    p_meta_raw = meta_stacker.predict_proba(meta_X)
    p_meta = np.zeros((n_test, len(TARGET_MODELS)), dtype=np.float32)
    for i, c in enumerate(meta_stacker.classes_):
        p_meta[:, c] = p_meta_raw[:, i]

    # 4. Stage 2 Non-Redundant Within-Family Specialist Gating
    print("Applying within-family specialist synthesis & per-family gating...")
    hier_probs = family_ensemble.predict_hierarchical_probs(
        X_sparse=X_test_sparse,
        stage1_probs=p_meta,
        E_style_flat=E_style_scaled,
        dense_forensics=forensic_scaled,
    )
    stage2_probs = family_ensemble.apply_gated_blend(p_meta, hier_probs)

    # 5. Stage 3 Semantic Clustering & Adaptive Centering Refinement
    if n_test > 1:
        print("Discovering latent prompt clusters on E_semantic...")
        test_clusters = prompt_clusterer.find_clusters(E_semantic)
        unique_clusters, counts = np.unique(test_clusters, return_counts=True)
        n_singletons = sum(1 for c in counts if c == 1)
        print(f"Found {len(unique_clusters)} clusters: {n_singletons} singletons, {len(unique_clusters) - n_singletons} multi-sample clusters.")

        centerer = AdaptiveChunkCenterer(lambda_max=0.35)
        stage3_probs = centerer.refine_cluster_probabilities(
            base_probs=stage2_probs,
            cluster_labels=test_clusters,
            E_semantic=E_semantic,
            features_dense=E_style_scaled,
            centered_head=centered_head,
        )
    else:
        print("Skipping clustering (n_test <= 1). Preserving raw probabilities.")
        stage3_probs = stage2_probs

    # 6. Stage 4 Nested OOF Logit-Offset & Prior Calibration
    print("Applying prior-shift logit calibration...")
    final_probs = calibrator.transform(stage3_probs)

    # 7. Convert probabilities to valid model string labels
    predicted_labels = probabilities_to_labels(final_probs)

    # 8. Create submission DataFrame
    sub_df = pd.DataFrame({
        "id": test_ids,
        "model": predicted_labels,
    })

    # 9. Strict Validation
    print("Validating submission constraints...")
    validate_submission_dataframe(sub_df, test_ids)

    # 10. Write submission.parquet
    print(f"Writing submission to: {output_path}")
    sub_df.to_parquet(output_path, index=False)

    print("\nSubmission summary:")
    print("-" * 40)
    print(f"Rows: {len(sub_df)}")
    print("Predicted class distribution in test:")
    print(sub_df["model"].value_counts())
    print("-" * 40)
    print("First 5 rows:")
    print(sub_df.head())

    # Check sample submission parity if present
    if SAMPLE_SUB_PATH.exists():
        sample_sub = pd.read_parquet(SAMPLE_SUB_PATH)
        print(f"\nSample submission comparison:")
        print(f"Sample shape: {sample_sub.shape}, Output shape: {sub_df.shape}")
        assert list(sub_df.columns) == list(sample_sub.columns), "Column mismatch with sample submission!"

    total_time = time.time() - start_time
    print(f"\nInference and submission generation completed successfully in {total_time:.2f}s!")
    return sub_df


if __name__ == "__main__":
    run_inference()
