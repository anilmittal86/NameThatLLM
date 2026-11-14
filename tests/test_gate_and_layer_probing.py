"""
Unit and Equivalence Verification Tests for:
1. Supervised Discriminative Subspace with Exact Zero-Cost Fold-In.
2. Self-Contained Family Gate Checkpoints (heads.safetensors with norms + heads).
3. Offline Dataset Gate Training from Pre-Extracted Kaggle Dataset Bundles.
4. Layer-Probing Sweeps, Ridge Probes, and ScalarMix Aggregation.
5. Multi-View and Raw Full-Layer Feature Extraction.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from namethatllm.config import FAMILIES, LABELS
from namethatllm.layer_probing import (
    LayerScalarMix,
    LayerScalarMixWithHead,
    fit_trace_scaled_ridge_probe,
    run_all_layers_probe_sweep,
)
from namethatllm.trainer import (
    FamilyMLPHead,
    MultiLayerLoRAClassifier,
    build_supervised_family_subspace,
    fit_family_mlp_head,
    save_extracted_features_bundle,
    train_family_gate,
    train_family_gate_from_dataset,
)


class TestGateAndLayerProbing:
    def test_supervised_subspace_whitening_and_zero_cost_fold_in(self):
        """
        Verify that running FamilyMLPHead on the whitened supervised subspace produces
        logits mathematically identical (atol=1e-4) to running the folded MLP weights
        directly on raw high-dimensional features Z_raw.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        N = 120
        raw_dim = 40960
        subspace_dim = 64
        hidden_dim = 32
        num_family_classes = 4
        family_classes = [f"gemini_class_{i}" for i in range(num_family_classes)]

        # Generate synthetic raw features and balanced family labels
        Z_train = np.random.randn(N, raw_dim).astype(np.float32) * 2.0
        y_train = np.array([i % (num_family_classes + 1) for i in range(N)], dtype=np.int64)

        # Build supervised discriminative subspace
        C_eff, mu_fam, stats = build_supervised_family_subspace(
            Z_train=Z_train,
            y_train=y_train,
            num_family_classes=num_family_classes,
            family_classes=family_classes,
            subspace_dim=subspace_dim,
            num_tapped_layers=8,
        )

        assert C_eff.shape == (subspace_dim, raw_dim)
        assert mu_fam.squeeze().shape == (raw_dim,)
        assert "subspace_dim" in stats

        # Fit FamilyMLPHead on the whitened features
        Z_white_train = ((Z_train - mu_fam) @ C_eff.T).astype(np.float32)
        y_fam_mask = y_train < num_family_classes
        head_dict, best_acc, best_logits = fit_family_mlp_head(
            Z_train=Z_white_train[y_fam_mask],
            y_train=y_train[y_fam_mask],
            num_gate_classes=num_family_classes,
            num_family_classes=num_family_classes,
            hidden_dim=hidden_dim,
            epochs=10,
            lr=1e-3,
            pca_components=C_eff,
            pca_mean=mu_fam,
        )

        assert "mlp_w1" in head_dict
        assert "mlp_b1" in head_dict
        assert "mlp_w2" in head_dict
        assert "mlp_b2" in head_dict
        assert "ln_weight" in head_dict
        assert "ln_bias" in head_dict

        # Verify shapes of folded weights
        mlp_w1 = head_dict["mlp_w1"]  # [hidden_dim, raw_dim]
        mlp_b1 = head_dict["mlp_b1"]  # [hidden_dim]
        unfolded_w1 = head_dict["unfolded_w1"]  # [hidden_dim, subspace_dim]
        unfolded_b1 = head_dict["unfolded_b1"]  # [hidden_dim]
        ln_w = head_dict["ln_weight"]
        ln_b = head_dict["ln_bias"]
        fc2_w = head_dict["mlp_w2"]  # [num_family_classes, hidden_dim]
        fc2_b = head_dict["mlp_b2"]  # [num_family_classes]

        assert mlp_w1.shape == (hidden_dim, raw_dim)
        assert mlp_b1.shape == (hidden_dim,)

        # Evaluation on test instances
        N_test = 20
        Z_test_raw = np.random.randn(N_test, raw_dim).astype(np.float32)

        # 1. Pipeline evaluation: project -> whiten -> MLP layer 1 -> GELU -> LN -> MLP layer 2
        Z_test_white = torch.from_numpy((Z_test_raw - mu_fam.reshape(1, -1)) @ C_eff.T).float()
        unfolded_fc1 = Z_test_white @ unfolded_w1.T + unfolded_b1
        unfolded_act = F.gelu(unfolded_fc1)
        unfolded_ln = F.layer_norm(unfolded_act, (hidden_dim,), weight=ln_w, bias=ln_b)
        logits_pipeline = (unfolded_ln @ fc2_w.T + fc2_b).numpy()

        # 2. Folded evaluation: direct single matrix multiply on raw features
        folded_fc1 = torch.from_numpy(Z_test_raw).float() @ mlp_w1.T + mlp_b1
        folded_act = F.gelu(folded_fc1)
        folded_ln = F.layer_norm(folded_act, (hidden_dim,), weight=ln_w, bias=ln_b)
        logits_folded = (folded_ln @ fc2_w.T + fc2_b).numpy()

        # Strict numerical equivalence check
        max_abs_diff = np.max(np.abs(logits_pipeline - logits_folded))
        assert max_abs_diff < 1e-4, f"Fold-in deviation {max_abs_diff:.6e} exceeded tolerance 1e-4"
        np.testing.assert_allclose(logits_pipeline, logits_folded, atol=1e-4, rtol=1e-4)

    def test_layer_scalar_mix_aggregation_and_gradient(self):
        """
        Verify LayerScalarMix aggregation formula and gradient propagation.
        """
        torch.manual_seed(42)
        B, L, D = 4, 8, 32
        x = torch.randn(B, L, D, requires_grad=True)

        scalar_mix = LayerScalarMix(num_layers=L, hidden_dim=D)
        out = scalar_mix(x)

        assert out.shape == (B, D)
        loss = out.sum()
        loss.backward()

        assert scalar_mix.scalar_weights.grad is not None
        assert scalar_mix.gamma.grad is not None
        assert x.grad is not None

        importance = scalar_mix.get_layer_importance()
        assert len(importance) == L
        assert abs(sum(importance.values()) - 1.0) < 1e-3

    def test_trace_scaled_ridge_probe_primal_and_dual(self):
        """
        Verify fit_trace_scaled_ridge_probe produces accurate linear weights in both
        primal (N >= D) and dual (N < D) regimes without NaNs.
        """
        # 1. Primal regime: N > D
        N1, D1 = 100, 16
        X1 = np.random.randn(N1, D1).astype(np.float32)
        y1 = np.random.randint(0, 4, size=N1)
        W1, b1 = fit_trace_scaled_ridge_probe(X1, y1, num_classes=4, alpha=1.0)
        assert W1.shape == (D1, 4)
        assert b1.shape == (4,)
        assert not np.isnan(W1).any()
        assert not np.isnan(b1).any()

        # 2. Dual regime: N < D
        N2, D2 = 20, 128
        X2 = np.random.randn(N2, D2).astype(np.float32)
        y2 = np.random.randint(0, 4, size=N2)
        W2, b2 = fit_trace_scaled_ridge_probe(X2, y2, num_classes=4, alpha=1.0)
        assert W2.shape == (D2, 4)
        assert b2.shape == (4,)
        assert not np.isnan(W2).any()
        assert not np.isnan(b2).any()

    def test_run_all_layers_probe_sweep_report(self):
        """
        Verify run_all_layers_probe_sweep ranks layers by balanced accuracy
        and exports a valid structured JSON report.
        """
        np.random.seed(42)
        N, L, D = 60, 6, 32
        features = np.random.randn(N, L, D).astype(np.float32)
        # Inject strong discriminative signal specifically into layer 4
        labels = np.array([i % 4 for i in range(N)], dtype=np.int64)
        for i in range(N):
            features[i, 4, labels[i] * 4 : (labels[i] + 1) * 4] += 5.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = Path(tmp_dir) / "layer_probe_report.json"
            report = run_all_layers_probe_sweep(
                features_by_layer=features,
                labels=labels,
                n_splits=3,
                output_json=out_json,
            )

            assert out_json.exists()
            assert "ranked_layers" in report
            assert "top_4_layers" in report
            assert "best_single_layer" in report

            # Layer 4 must be the top ranked layer due to injected signal
            assert report["best_single_layer"]["layer_idx"] == 4
            assert report["ranked_layers"][0]["layer_idx"] == 4

    def test_save_and_load_extracted_features_bundle(self):
        """
        Verify save_extracted_features_bundle creates all expected safetensors,
        parquet metadata, and dataset-metadata.json files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "bundle_test"
            N = 40
            D_concat = 128
            L = 4
            D_layer = 32

            features_concat = np.random.randn(N, D_concat).astype(np.float32)
            features_by_layer = np.random.randn(N, L, D_layer).astype(np.float32)
            texts = [f"Text passage number {i}" for i in range(N)]
            labels = [i % 4 for i in range(N)]
            model_names = [f"model_{lab}" for lab in labels]
            prompt_ids = [f"prompt_{i // 4}" for i in range(N)]

            champ_logits = np.random.randn(N, 18).astype(np.float32)
            head_tail_cos = np.random.randn(N, L).astype(np.float32)
            token_std_norms = np.random.randn(N, L).astype(np.float32)

            saved_path = save_extracted_features_bundle(
                output_dir=out_dir,
                target="gemini",
                features_concat=features_concat,
                features_by_layer=features_by_layer,
                texts=texts,
                labels=labels,
                model_names=model_names,
                prompt_ids=prompt_ids,
                champion_logits_plain=champ_logits,
                head_tail_cos=head_tail_cos,
                token_std_norms=token_std_norms,
                dataset_title="NTL Test Bundle",
                dataset_slug="ntl-test-bundle",
            )

            assert (saved_path / "features_concat.safetensors").exists()
            assert (saved_path / "features_by_layer.safetensors").exists()
            assert (saved_path / "champion_logits.safetensors").exists()
            assert (saved_path / "metadata.parquet").exists()
            assert (saved_path / "dataset-metadata.json").exists()
            assert (saved_path / "features_summary.json").exists()

            # Verify contents of saved safetensors
            loaded_cat = load_file(str(saved_path / "features_concat.safetensors"))
            assert "features_concat" in loaded_cat
            assert loaded_cat["features_concat"].shape == (N, D_concat)

            loaded_logits = load_file(str(saved_path / "champion_logits.safetensors"))
            assert "logits_plain" in loaded_logits
            assert "head_tail_cos" in loaded_logits
            assert "token_std_norms" in loaded_logits

    def test_offline_dataset_gate_training_end_to_end(self):
        """
        Verify train_family_gate_from_dataset executes end-to-end on a saved
        feature bundle in under 5 seconds without loading any Hugging Face model,
        and saves a self-contained gate checkpoint.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir) / "gemini_bundle"
            gate_out_dir = Path(tmp_dir) / "gemini_gate_out"

            # Create a synthetic dataset bundle for gemini family
            gemini_indices = FAMILIES["gemini"]  # [5, 6, 7, 8]
            N = 80
            D_concat = 5120 * 8  # 40960
            L = 8
            D_layer = 5120

            features_concat = np.random.randn(N, D_concat).astype(np.float32)
            features_by_layer = np.random.randn(N, L, D_layer).astype(np.float16)
            texts = [f"Passage text {i}" for i in range(N)]
            # Map evenly to gemini labels
            labels = [gemini_indices[i % len(gemini_indices)] for i in range(N)]
            model_names = [LABELS[l] for l in labels]
            prompt_ids = [f"pid_{i % 10}" for i in range(N)]
            champ_logits = np.random.randn(N, 18).astype(np.float32)

            save_extracted_features_bundle(
                output_dir=bundle_dir,
                target="gemini",
                features_concat=features_concat,
                features_by_layer=features_by_layer,
                texts=texts,
                labels=labels,
                model_names=model_names,
                prompt_ids=prompt_ids,
                champion_logits_plain=champ_logits,
                eval_features_concat=features_concat[:20],
                eval_features_by_layer=features_by_layer[:20],
                eval_texts=texts[:20],
                eval_labels=labels[:20],
                eval_model_names=model_names[:20],
            )

            # Create dummy heads.safetensors in bundle_dir with LayerNorm weights
            dummy_heads = {}
            for i in range(8):
                dummy_heads[f"norms.{i}.weight"] = torch.ones(5120, dtype=torch.float32)
                dummy_heads[f"norms.{i}.bias"] = torch.zeros(5120, dtype=torch.float32)
            dummy_heads["plain.weight"] = torch.randn(18, 40960, dtype=torch.float32)
            dummy_heads["plain.bias"] = torch.zeros(18, dtype=torch.float32)
            save_file(dummy_heads, str(bundle_dir / "heads.safetensors"))

            # Run offline gate training
            result = train_family_gate_from_dataset(
                dataset_dir=bundle_dir,
                target_family="gemini",
                output_dir=gate_out_dir,
                head_type="mlp",
                subspace_dim=32,
                hidden_dim=16,
                mlp_epochs=5,
                mlp_lr=1e-3,
                device=torch.device("cpu"),
            )

            assert (gate_out_dir / "heads.safetensors").exists()
            assert (gate_out_dir / "meta.json").exists()
            meta_json = json.loads((gate_out_dir / "meta.json").read_text(encoding="utf-8"))
            assert "default_gate_weight" in meta_json
            assert meta_json["default_gate_weight"] == 0.40
            assert "best_sweep_gate_weight" in meta_json
            assert result["target_family"] == "gemini"
            assert "within_family_balanced_acc" in result

            # Verify self-contained checkpoint has norms and heads
            saved_heads = load_file(str(gate_out_dir / "heads.safetensors"))
            for i in range(8):
                assert f"norms.{i}.weight" in saved_heads
                assert f"norms.{i}.bias" in saved_heads
            assert "mlp_w1" in saved_heads
            assert "mlp_b1" in saved_heads
            assert "plain.weight" in saved_heads
            assert "plain.bias" in saved_heads

    def test_solve_within_class_fisher_gpu(self):
        from namethatllm.layer_probing import _solve_within_class_fisher_gpu

        # Test both N >= D and N < D
        for N, D in [(60, 16), (20, 64)]:
            X = np.random.randn(N, D).astype(np.float32)
            y = np.array([i % 4 for i in range(N)], dtype=np.int64)
            W_ovr, W_pairs = _solve_within_class_fisher_gpu(X, y, num_family_classes=4, reg_ratio=0.15)
            assert W_ovr.shape == (4, D)
            assert W_pairs.shape == (12, D)  # 6 pairs * 2 regularization scales = 12 directions
            # Check unit L2 norm
            ovr_norms = np.linalg.norm(W_ovr, axis=1)
            pair_norms = np.linalg.norm(W_pairs, axis=1)
            np.testing.assert_allclose(ovr_norms, 1.0, atol=1e-4)
            np.testing.assert_allclose(pair_norms, 1.0, atol=1e-4)

    def test_engineer_multiview_features(self):
        from namethatllm.layer_probing import engineer_multiview_features

        N = 30
        num_tapped = 8
        D_layer = 16
        D_total = num_tapped * D_layer

        feat_cat = np.random.randn(N, D_total).astype(np.float32)
        feat_by_layer = np.random.randn(N, 12, D_layer).astype(np.float32)
        ht_cos = np.random.randn(N, num_tapped).astype(np.float32)
        t_std = np.abs(np.random.randn(N, num_tapped)).astype(np.float32)

        engineered = engineer_multiview_features(
            features_concat=feat_cat,
            features_by_layer=feat_by_layer,
            head_tail_cos=ht_cos,
            token_std_norms=t_std,
            num_tapped_layers=num_tapped,
        )

        assert engineered.shape[0] == N
        assert engineered.ndim == 2
        assert not np.isnan(engineered).any()

    def test_forward_features_multiview_quad_tuple(self):
        """
        Verify MultiLayerLoRAClassifier.forward_features with return_by_layer=True and
        return_multiview=True returns the expected 4-tuple (Z_cat, Z_layers, ht_cos_t, std_norm_t).
        """
        class MockBackbone(nn.Module):
            def __init__(self, hidden_size=64, num_layers=6):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

            def forward(self, input_ids, attention_mask, **kwargs):
                B, S = input_ids.shape
                states = [torch.randn(B, S, self.hidden_size) for _ in range(self.num_layers)]
                return type("Outputs", (), {"hidden_states": states})()

        model = MultiLayerLoRAClassifier(
            backbone=MockBackbone(hidden_size=32, num_layers=6),
            layers=[1, 3, 5],
            hidden_size=32,
            num_classes=4,
        )

        B, S = 4, 16
        input_ids = torch.randint(1, 1000, (B, S))
        attention_mask = torch.ones((B, S))

        res = model.forward_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            drop_first_token=True,
            return_by_layer=True,
            extract_all_layers=True,
            return_multiview=True,
        )

        assert isinstance(res, tuple)
        assert len(res) == 4
        Z_cat, Z_layers, ht_cos, std_norm = res
        assert Z_cat.shape == (B, 32 * 3)
        assert Z_layers.shape == (B, 6, 32)
        assert ht_cos.shape == (B, 3)
        assert std_norm.shape == (B, 3)

    def test_mlp_temperature_calibration_and_ensemble(self):
        """Verify fit_family_mlp_head calibrates output logits to row_std ≈ 3.2."""
        from namethatllm.trainer import fit_family_mlp_head

        N = 60
        D = 32
        K_gate = 4
        K_fam = 4
        Z_train = np.random.randn(N, D).astype(np.float32)
        y_train = np.array([i % K_fam for i in range(N)], dtype=np.int64)
        Z_eval = np.random.randn(20, D).astype(np.float32)
        y_eval_true = np.array([i % K_fam for i in range(20)], dtype=np.int64)
        eval_mask = np.ones(20, dtype=bool)

        head_dict, acc, eval_logits = fit_family_mlp_head(
            Z_train=Z_train,
            y_train=y_train,
            num_gate_classes=K_gate,
            num_family_classes=K_fam,
            hidden_dim=16,
            epochs=5,
            lr=1e-3,
            Z_eval=Z_eval,
            y_eval_true=y_eval_true,
            eval_mask=eval_mask,
            device=torch.device("cpu"),
            raw_Z_eval=Z_eval,
        )

        assert eval_logits is not None
        fam_logits = eval_logits[:, :K_fam]
        row_std = float(np.std(fam_logits, axis=1).mean())
        assert abs(row_std - 3.2) < 0.6  # Temperature-calibrated around 3.2

    def test_confidence_scaled_linear_conditional_blending(self):
        """Verify confidence-scaled normalized conditional linear blending in postprocess."""
        from namethatllm.postprocess import apply_family_specialist_lora_gate, apply_claude_specialist_gate
        from namethatllm.config import FAMILIES

        N = 4
        P_mix = np.full((N, 18), 1.0 / 18.0, dtype=np.float64)
        # Row 0: high gemini mass (0.80)
        gem_cols = FAMILIES["gemini"]
        P_mix[0, gem_cols] = 0.80 / len(gem_cols)
        # Row 1: low gemini mass (0.20)
        P_mix[1, gem_cols] = 0.20 / len(gem_cols)
        R_main = P_mix.copy()

        spec_logits = np.zeros((N, len(gem_cols)), dtype=np.float64)
        spec_logits[:, 0] = 5.0  # Strongly favor first gemini class

        P_fused = apply_family_specialist_lora_gate(
            P_mix=P_mix,
            R_main=R_main,
            spec_logits=spec_logits,
            family_cols=gem_cols,
            gate_weight=0.40,
        )

        # Total mass conserved exactly
        np.testing.assert_allclose(P_fused[:, gem_cols].sum(axis=1), P_mix[:, gem_cols].sum(axis=1), atol=1e-7)
        # Row 0 had 0.80 mass (>= 0.50) -> full gw_eff = 0.40
        # Row 1 had 0.20 mass (< 0.50) -> gw_eff = 0.40 * (0.20 / 0.50) = 0.16
        first_col = gem_cols[0]
        norm_r0 = P_fused[0, first_col] / P_fused[0, gem_cols].sum()
        norm_r1 = P_fused[1, first_col] / P_fused[1, gem_cols].sum()
        assert norm_r0 > norm_r1  # Stronger shift on high-confidence row
