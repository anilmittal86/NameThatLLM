"""
Tests for ExperimentLedger, Layer-Probing Importance Weights, and Gate Experiment Runner
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from namethatllm.config import FAMILIES, LABELS
from namethatllm.experiment_logger import ExperimentLedger, LEDGER_COLUMNS
from namethatllm.layer_probing import build_supervised_family_subspace
from scripts.run_gate_experiments import run_sweep


def test_experiment_ledger_creation_and_logging(tmp_path: Path):
    ledger = ExperimentLedger(ledger_dir=tmp_path)
    csv_file = tmp_path / "experiment_ledger.csv"
    md_file = tmp_path / "experiment_ledger.md"

    assert csv_file.exists()

    # Add EXP-00 baseline
    entry0 = {
        "exp_id": "EXP-00",
        "phase": "P1_baseline",
        "target_family": "gemini",
        "feature_set": "F0_plain_solo",
        "subspace_dim": 0,
        "head_type": "plain",
        "gate_weight": 0.0,
        "holdout_family_bacc": 0.70,
        "hard_sister_acc": 0.65,
        "holdout_18cls_raw_bacc": 0.81,
        "holdout_18cls_sinkhorn_bacc": 0.835,
        "notes": "Baseline Plain Solo",
    }
    ledger.add_entry(entry0)

    # Add EXP-01
    entry1 = {
        "exp_id": "EXP-01",
        "phase": "P2_features",
        "target_family": "gemini",
        "feature_set": "F1_fisher_per_layer",
        "subspace_dim": 24,
        "head_type": "mlp",
        "gate_weight": 0.40,
        "holdout_family_bacc": 0.75,
        "hard_sister_acc": 0.72,
        "holdout_18cls_raw_bacc": 0.82,
        "holdout_18cls_sinkhorn_bacc": 0.845,
        "notes": "Per-Layer Fisher + MLP",
    }
    res1 = ledger.add_entry(entry1)

    assert md_file.exists()
    assert abs(res1["delta_vs_baseline_bacc"] - (0.845 - 0.835)) < 1e-5

    df = ledger.get_dataframe()
    assert len(df) == 2
    assert list(df.columns) == LEDGER_COLUMNS

    best = ledger.get_best_run(target_family="gemini")
    assert best is not None
    assert best["exp_id"] == "EXP-01"

    # Test export copy
    copy_dir = tmp_path / "exported"
    ledger.export_copy(copy_dir)
    assert (copy_dir / "experiment_ledger.csv").exists()
    assert (copy_dir / "experiment_ledger.md").exists()


def test_build_supervised_family_subspace_weights_and_velocity():
    np.random.seed(42)
    torch.manual_seed(42)

    # 4 classes, 10 samples each, 4 layers of 16 dims = 64 dims total
    N = 40
    D_layer = 16
    num_layers = 4
    D_total = D_layer * num_layers
    num_classes = 4

    y_train = np.repeat(np.arange(num_classes), 10)
    Z_train = np.random.randn(N, D_total).astype(np.float32)
    for c in range(num_classes):
        Z_train[y_train == c, c * D_layer : (c + 1) * D_layer] += 3.0

    # Test 1: layer_weighting=True, include_velocity=True
    C_eff_full, mu_full, diag_full = build_supervised_family_subspace(
        Z_train=Z_train,
        y_train=y_train,
        num_family_classes=num_classes,
        num_tapped_layers=num_layers,
        include_velocity=True,
        layer_weighting=True,
        hard_sister_pair=(2, 3),
        device=torch.device("cpu"),
    )

    assert C_eff_full.shape[1] == D_total
    assert mu_full.shape == (1, D_total)
    assert "layer_scalar_weights" in diag_full
    assert len(diag_full["layer_scalar_weights"]) == num_layers
    assert abs(sum(diag_full["layer_scalar_weights"]) - 1.0) < 1e-4
    assert "per_layer_oof_bacc" in diag_full
    assert len(diag_full["per_layer_oof_bacc"]) == num_layers

    # Test 2: include_velocity=False
    C_eff_no_vel, _, diag_no_vel = build_supervised_family_subspace(
        Z_train=Z_train,
        y_train=y_train,
        num_family_classes=num_classes,
        num_tapped_layers=num_layers,
        include_velocity=False,
        layer_weighting=True,
        hard_sister_pair=(2, 3),
        device=torch.device("cpu"),
    )
    assert diag_no_vel["n_supervised_dirs"] <= diag_full["n_supervised_dirs"]


def test_run_gate_experiments_sweep(tmp_path: Path):
    np.random.seed(42)
    torch.manual_seed(42)

    # Create synthetic bundle for Gemini
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True)

    target_family = "gemini"
    fam_cols = FAMILIES[target_family]
    fam_models = [LABELS[i] for i in fam_cols]
    num_classes = len(fam_models)

    # 10 samples per class + 10 negative distractors = 50 samples
    n_per_c = 10
    models = []
    for m in fam_models:
        models.extend([m] * n_per_c)
    models.extend(["__OTHER__"] * 10)
    N_tr = len(models)

    # 64 dims (8 tapped layers x 8 dims) for lightning-fast test execution
    D = 64
    Z_tr = np.random.randn(N_tr, D).astype(np.float32) * 0.1
    # Add signal to features
    for idx, m in enumerate(models):
        if m in fam_models:
            c = fam_models.index(m)
            Z_tr[idx, c * 8 : (c + 1) * 8] += 2.0

    save_file({"features_concat": torch.from_numpy(Z_tr)}, str(bundle_dir / "features_concat.safetensors"))
    df_meta = pd.DataFrame({"model": models, "prompt_id": [f"p_{i % 5}" for i in range(N_tr)]})
    df_meta.to_parquet(bundle_dir / "metadata.parquet")

    # Benchmark holdout: 18 samples (1 per class)
    N_bench = 18
    Z_ev = np.random.randn(N_bench, D).astype(np.float32) * 0.1
    save_file({"benchmark_features_concat": torch.from_numpy(Z_ev)}, str(bundle_dir / "benchmark_features_concat.safetensors"))
    df_bench = pd.DataFrame({"model": list(LABELS), "label": list(range(18))})
    df_bench.to_parquet(bundle_dir / "benchmark_metadata.parquet")

    bench_logits = np.random.randn(N_bench, 18).astype(np.float32)
    for i in range(18):
        bench_logits[i, i] += 2.0
    save_file({"logits_plain": torch.from_numpy(bench_logits)}, str(bundle_dir / "benchmark_champion_logits.safetensors"))

    # Execute run_sweep on synthetic bundle
    out_dir = tmp_path / "export_gemini_gate_mlp"
    ledger_dir = tmp_path / "experiments"

    res = run_sweep(
        embeddings_dir=bundle_dir,
        target_family="gemini",
        output_dir=out_dir,
        ledger_dir=ledger_dir,
        device=torch.device("cpu"),
    )

    assert res["target_family"] == "gemini"
    assert res["total_experiments"] >= 7
    assert (out_dir / "heads.safetensors").exists()
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "experiment_ledger.csv").exists()
    assert (out_dir / "experiment_ledger.md").exists()

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert "hard_pair_eval_acc" in meta
    assert "default_gate_weight" in meta
    assert meta["target_family"] == "gemini"
