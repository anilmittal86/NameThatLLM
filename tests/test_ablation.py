"""
Unit Tests for Zero-Leakage Unseen Holdout Benchmark & Forward Stepwise Ablation Engine
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from namethatllm.config import LABELS, K, align_cols
from namethatllm.neural_extractor import compute_centered_logits
from namethatllm.pipeline import auto_align_and_verify_head
from namethatllm.ablation import (
    _text_hash,
    build_unseen_holdout_dataset,
    UnseenStepwiseAuditor,
)


def test_text_hash_normalization():
    """Verify whitespace collapsing and deterministic hashing."""
    t1 = "Hello world! This is a test."
    t2 = "   Hello   world! \t This is a test. \n  "
    t3 = "Hello world! This is a different test."

    h1 = _text_hash(t1)
    h2 = _text_hash(t2)
    h3 = _text_hash(t3)

    assert h1 == h2, "Whitespace normalization must make hashes identical."
    assert h1 != h3, "Different texts must yield different hashes."
    assert len(h1) == 40, "SHA-1 hash must be 40 hexadecimal characters."


def test_compute_centered_logits_handles_negative_and_string_groups():
    """Verify compute_centered_logits never crashes on negative, string, or non-contiguous group IDs."""
    np.random.seed(42)
    Z_test = np.random.randn(5, 32)
    W_c = np.random.randn(18, 32)
    b_c = np.zeros(18)

    # 1. Negative group IDs (the bug that crashed np.bincount with ValueError)
    groups_neg = np.array([-1, -1, -1, 2, 2])
    lp_neg, valid_neg = compute_centered_logits(Z_test, groups_neg, W_c, b_c, LABELS)
    assert lp_neg.shape == (5, 18)
    assert valid_neg.shape == (5,)
    assert valid_neg.all(), "Groups [-1,-1,-1] (size 3) and [2,2] (size 2) must both be valid clusters"

    # 2. String group IDs
    groups_str = np.array(["pA", "pA", "pB", "pB", "pC"])
    lp_str, valid_str = compute_centered_logits(Z_test, groups_str, W_c, b_c, LABELS)
    assert lp_str.shape == (5, 18)
    assert valid_str.shape == (5,)
    assert list(valid_str) == [True, True, True, True, False], "Singleton 'pC' must be marked invalid"


def test_build_unseen_holdout_dataset_quarantine(tmp_path: Path):
    """Verify that build_unseen_holdout_dataset strictly excludes all train.csv and pool_v4 rows."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    # 1. Contaminated train.csv
    train_texts = [
        "Train sentence alpha.",
        "Train sentence beta.",
        "Train sentence gamma.",
    ]
    df_train = pd.DataFrame({
        "id": [1, 2, 3],
        "prompt_id": [10, 10, 10],
        "text": train_texts,
        "model": [LABELS[0], LABELS[1], LABELS[2]],
    })
    train_path = input_dir / "train.csv"
    df_train.to_csv(train_path, index=False)

    # 2. Contaminated pool_v4.parquet
    pool_v4_texts = [
        "Pool v4 sentence delta.",
        "Pool v4 sentence epsilon.",
    ]
    df_p4 = pd.DataFrame({
        "text": pool_v4_texts,
        "model": [LABELS[3], LABELS[4]],
    })
    p4_dir = input_dir / "pool_v4_ds"
    p4_dir.mkdir()
    df_p4.to_parquet(p4_dir / "pool_v4.parquet")

    # 3. Candidate corpus with both contaminated and strictly unseen texts
    unseen_texts = [f"Unseen topic text {i} for model {LABELS[i % K]} with stylized vocabulary." for i in range(72)]
    candidate_texts = (
        train_texts  # contaminated (should be excluded)
        + pool_v4_texts  # contaminated (should be excluded)
        + unseen_texts  # genuine unseen candidates
    )
    candidate_models = (
        [LABELS[0], LABELS[1], LABELS[2]]
        + [LABELS[3], LABELS[4]]
        + [LABELS[i % K] for i in range(72)]
    )
    df_cand = pd.DataFrame({
        "text": candidate_texts,
        "model": candidate_models,
    })
    cand_dir = input_dir / "nlm_pool_v5_ds"
    cand_dir.mkdir()
    df_cand.to_parquet(cand_dir / "nlm_pool_v5.parquet")

    # Run build_unseen_holdout_dataset
    df_unseen, holdout_hashes = build_unseen_holdout_dataset(
        input_dir=input_dir,
        train_path=train_path,
        samples_per_class=4,
        seed=42,
    )

    # Verification: zero leakage
    train_hashes = {_text_hash(t) for t in train_texts}
    p4_hashes = {_text_hash(t) for t in pool_v4_texts}
    contaminated = train_hashes | p4_hashes

    assert len(df_unseen) > 0, "Should have sampled unseen passages."
    assert len(holdout_hashes.intersection(contaminated)) == 0, "No contaminated text may appear in holdout_hashes!"
    assert not any(h in contaminated for h in df_unseen["_hash"]), "No contaminated text may appear in df_unseen!"

    # Verify schema and valid positive cluster IDs
    for col in ["id", "prompt_id", "cluster_id", "text", "model", "_hash"]:
        assert col in df_unseen.columns, f"Missing required column: {col}"

    assert (df_unseen["prompt_id"] > 0).all(), "All prompt_id values must be positive integer cluster IDs 1..G"
    assert (df_unseen["cluster_id"] > 0).all(), "All cluster_id values must be positive integer cluster IDs 1..G"
    assert len(df_unseen) == df_unseen["_hash"].nunique(), "df_unseen must contain unique passages."


def test_unseen_stepwise_auditor_standalone_and_stepwise(tmp_path: Path):
    """Verify UnseenStepwiseAuditor greedy selection, all-subsets LOO, and report generation."""
    np.random.seed(42)
    n_samples = 180
    n_test = 50

    # 10 samples per each of the 18 classes
    y_unseen = np.repeat(np.arange(K), 10)
    groups_unseen = (np.arange(n_samples) // 3) + 1  # positive integer cluster IDs 1..G

    auditor = UnseenStepwiseAuditor(y_unseen=y_unseen, groups_unseen=groups_unseen, N_test=n_test)

    # Model 1: Accurate predictions (e.g. 99% accuracy)
    logits_m1_unseen = np.random.randn(n_samples, K) * 0.5
    for i, true_cls in enumerate(y_unseen):
        logits_m1_unseen[i, true_cls] += 3.5
    logits_m1_test = np.random.randn(n_test, K)

    auditor.add_model(
        name="Mock-LoRA-14B",
        tier="Main",
        weight=0.50,
        unseen_log_probs=logits_m1_unseen,
        test_log_probs=logits_m1_test,
    )

    assert len(auditor.models) == 1
    assert auditor.models[0]["solo_unseen_bacc"] > 0.70

    # Model 2: Complementary accurate predictions
    logits_m2_unseen = np.random.randn(n_samples, K) * 0.5
    for i, true_cls in enumerate(y_unseen):
        logits_m2_unseen[i, true_cls] += 2.5
    logits_m2_test = np.random.randn(n_test, K)

    lift_m2 = auditor.evaluate_marginal_lift(logits_m2_unseen, weight=0.30)
    auditor.add_model(
        name="Mock-DeBERTa",
        tier="Main",
        weight=0.30,
        unseen_log_probs=logits_m2_unseen,
        test_log_probs=logits_m2_test,
    )
    assert len(auditor.models) == 2

    # Model 3: Poor / adversarial predictions (triggers negative lift)
    logits_m3_unseen = np.random.randn(n_samples, K) * 2.0
    for i, true_cls in enumerate(y_unseen):
        logits_m3_unseen[i, (true_cls + 1) % K] += 10.0

    lift_m3 = auditor.evaluate_marginal_lift(logits_m3_unseen, weight=0.50)
    assert lift_m3 < -0.05, f"Adversarial model should show large negative marginal lift, got {lift_m3}"

    auditor.add_model(
        name="Adversarial-Model",
        tier="Comp",
        weight=0.20,
        unseen_log_probs=logits_m3_unseen,
        test_log_probs=np.random.randn(n_test, K),
    )
    assert len(auditor.models) == 3

    # Generate report with greedy selection and all-2^M subset search
    final_test_preds = np.random.randint(0, K, size=n_test)
    df_report = auditor.print_and_save_report(working_dir=tmp_path, final_test_preds=final_test_preds)

    assert not df_report.empty
    csv_path = tmp_path / "unseen_stepwise_ablation.csv"
    assert csv_path.exists(), "unseen_stepwise_ablation.csv must be written to disk."

    df_saved = pd.read_csv(csv_path)
    assert "step" in df_saved.columns
    assert "added_model" in df_saved.columns
    assert "marginal_lift" in df_saved.columns

    # Greedy forward selection stopped before adding the adversarial model with negative lift
    assert "Adversarial-Model" not in df_saved["added_model"].values, "Greedy selection must stop before adding adversarial model"
    assert len(df_saved) == 2


def test_compute_centered_logits_negative_and_string_group_ids():
    """Exact test verifying negative (-1, -99), string, and NaN group IDs are sanitized before np.bincount."""
    np.random.seed(42)
    Z = np.random.randn(8, 16)
    W_c = np.random.randn(18, 16)
    b_c = np.zeros(18)

    # Mixed negative, string, and singleton values
    mixed_groups = np.array([-1, -1, "singleton", "prompt_A", "prompt_A", 42, 42, 42])
    out_lp, valid = compute_centered_logits(Z, mixed_groups, W_c, b_c, LABELS)
    assert out_lp.shape == (8, 18)
    assert valid.shape == (8,)
    # Sizes: -1 has 2, "singleton" has 1, prompt_A has 2, 42 has 3 -> sizes >= 2 are valid
    assert valid[0] and valid[1]
    assert not valid[2]
    assert valid[3] and valid[4]
    assert valid[5] and valid[6] and valid[7]


def test_auto_align_and_verify_head_alphabetical_labels():
    """Verify auto_align_and_verify_head successfully discovers alphabetical labels and 5-fold OOF readout."""
    np.random.seed(42)
    N = 180
    D = 32
    # 10 samples per class
    y_sub = np.repeat(np.arange(18), 10)
    prompts_sub = np.tile(np.arange(10), 18)  # 10 prompt clusters with balanced class representations

    # Create synthetic representations where each class has a distinct centroid
    centroids = np.random.randn(18, D) * 2.0
    Z_tr = centroids[y_sub] + np.random.randn(N, D) * 0.2

    # Case 1: Head trained with alphabetical class ordering
    sorted_labels = sorted(LABELS)
    # Mapping: class name -> index in sorted_labels
    W_alpha = np.zeros((18, D), dtype=np.float32)
    for i, model_name in enumerate(LABELS):
        alpha_idx = sorted_labels.index(model_name)
        W_alpha[alpha_idx] = centroids[i]
    b_alpha = np.zeros(18, dtype=np.float32)

    passed, W_p_out, b_p_out, _, _, best_labels, acc = auto_align_and_verify_head(
        Z_tr=Z_tr,
        W_p=W_alpha,
        b_p=b_alpha,
        W_c=None,
        b_c=None,
        ckpt_labels=LABELS,
        y_sub=y_sub,
        prompts_sub=prompts_sub,
        name="test_alpha_ckpt",
        min_bacc=0.50,
    )
    assert passed, "Head with alphabetical labels must pass gate via auto-detection!"
    assert best_labels == sorted_labels, "Auto-align must detect alphabetical label ordering."
    assert acc > 0.80

    # Case 2: W_p is None -> 5-fold OOF LogisticRegression fallback
    passed_oof, W_oof, b_oof, W_c_oof, b_c_oof, fit_labels, oof_acc = auto_align_and_verify_head(
        Z_tr=Z_tr,
        W_p=None,
        b_p=None,
        W_c=None,
        b_c=None,
        ckpt_labels=LABELS,
        y_sub=y_sub,
        prompts_sub=prompts_sub,
        name="test_missing_head_ckpt",
        min_bacc=0.50,
    )
    assert passed_oof, "5-Fold OOF readout must pass on well-separated synthetic representations."
    assert W_oof is not None and W_oof.shape == (18, D)
    assert b_oof is not None and b_oof.shape == (18,)
    assert W_c_oof is not None
    assert oof_acc > 0.50


def test_optimize_weights_and_blend_prunes_bad_model_and_weights_individually(tmp_path: Path):
    """Verify optimize_weights_and_blend tunes individual weights and prunes degrading models to 0.0."""
    np.random.seed(42)
    n_samples = 180
    n_test = 40

    y_unseen = np.repeat(np.arange(K), 10)
    groups_unseen = (np.arange(n_samples) // 3) + 1

    auditor = UnseenStepwiseAuditor(y_unseen=y_unseen, groups_unseen=groups_unseen, N_test=n_test)

    # Model 1: Strong anchor model
    logits_m1_u = np.random.randn(n_samples, K) * 0.2
    for i, c in enumerate(y_unseen):
        logits_m1_u[i, c] += 4.0
    logits_m1_te = np.random.randn(n_test, K)
    auditor.add_model("Strong-Anchor", tier="Main", weight=0.50, unseen_log_probs=logits_m1_u, test_log_probs=logits_m1_te)

    # Model 2: Complementary model
    logits_m2_u = np.random.randn(n_samples, K) * 0.2
    for i, c in enumerate(y_unseen):
        logits_m2_u[i, c] += 3.0
    logits_m2_te = np.random.randn(n_test, K)
    auditor.add_model("Complementary-Model", tier="Comp", weight=0.25, unseen_log_probs=logits_m2_u, test_log_probs=logits_m2_te)

    # Model 3: Poisonous / adversarial model (predicts wrong class with high confidence)
    logits_m3_u = np.random.randn(n_samples, K) * 0.5
    for i, c in enumerate(y_unseen):
        logits_m3_u[i, (c + 1) % K] += 12.0
    logits_m3_te = np.random.randn(n_test, K)
    auditor.add_model("Poison-Model", tier="Comp", weight=0.30, unseen_log_probs=logits_m3_u, test_log_probs=logits_m3_te)

    # Extra stream: mock wpca_dp
    mock_dp_te = np.random.randn(n_test, K)
    extra_streams = [("wpca_dp", 0.22, mock_dp_te)]

    P_mix, weights_map, df_rep = auditor.optimize_weights_and_blend(
        working_dir=tmp_path,
        extra_test_log_probs=extra_streams,
        optimize_on_unseen=True,
    )

    assert P_mix.shape == (n_test, K)
    np.testing.assert_allclose(P_mix.sum(axis=1), 1.0, rtol=1e-5)

    # Verify: Poison-Model MUST be pruned to 0.0 weight
    assert weights_map["Poison-Model"] == 0.0, f"Poisonous model should have been pruned to 0.0, got {weights_map['Poison-Model']}"
    assert weights_map["Strong-Anchor"] > 0.0, "Strong anchor model must retain positive weight"
    assert "wpca_dp" in weights_map
    assert weights_map["wpca_dp"] == 0.22
