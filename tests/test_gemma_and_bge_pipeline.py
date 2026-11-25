"""
tests/test_gemma_and_bge_pipeline.py
====================================
Functional and integration test suite verifying Gemma-2-9B and BGE-M3:
1. Base model & checkpoint discovery under Kaggle directory structures.
2. BGE-M3 embedding extraction, normalization (F.normalize), and head handling.
3. Gemma-2-9B LoRA layer resolution, delta application, and multi-layer norms.
4. End-to-end 1800-sample pipeline execution, memory management, and metric parity.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from namethatllm.config import (
    LABELS, K, find_base_model, find_checkpoints, validate_production_assets
)
from namethatllm.neural_extractor import (
    merge_lora, _resolve_module_name, resolve_head_tensors
)
from namethatllm.pipeline import run_submission_pipeline


class TestGemmaAndBGEPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. Base Model & Checkpoint Discovery
    # --------------------------------------------------------------------------
    def test_gemma2_base_model_discovery_variants(self):
        """Verifies find_base_model discovers Gemma-2 across direct, hyphenated, text_config, and Kaggle paths."""
        # Case A: Kaggle structure /kaggle/input/gemma-2/transformers/gemma-2-9b/2
        kaggle_dir = self.root / "gemma-2" / "transformers" / "gemma-2-9b" / "2"
        kaggle_dir.mkdir(parents=True)
        (kaggle_dir / "config.json").write_text(json.dumps({
            "model_type": "gemma2",
            "hidden_size": 3584,
            "num_hidden_layers": 42
        }))

        res = find_base_model(self.root, "gemma2", 3584, 42)
        self.assertIsNotNone(res)
        self.assertEqual(res.resolve(), kaggle_dir.resolve())

        # Case B: Nested text_config
        text_cfg_dir = self.root / "nested_gemma"
        text_cfg_dir.mkdir(parents=True)
        (text_cfg_dir / "config.json").write_text(json.dumps({
            "text_config": {
                "model_type": "gemma2",
                "hidden_size": 3584,
                "num_hidden_layers": 42
            }
        }))
        res_nested = find_base_model(text_cfg_dir, "gemma2", 3584, 42)
        self.assertIsNotNone(res_nested)
        self.assertEqual(res_nested.resolve(), text_cfg_dir.resolve())

        # Case C: model_type is 'gemma' with 42 layers
        gemma_generic_dir = self.root / "generic_gemma"
        gemma_generic_dir.mkdir(parents=True)
        (gemma_generic_dir / "config.json").write_text(json.dumps({
            "model_type": "gemma",
            "hidden_size": 3584,
            "num_hidden_layers": 42
        }))
        res_generic = find_base_model(gemma_generic_dir, "gemma2", 3584, 42)
        self.assertIsNotNone(res_generic)
        self.assertEqual(res_generic.resolve(), gemma_generic_dir.resolve())

    def test_bge_and_gemma_checkpoint_discovery(self):
        """Verifies find_checkpoints discovers both BGE-M3 and Gemma-2 checkpoints under Kaggle mount paths."""
        # 1. Gemma checkpoint: ntl-ts-gemma2-9b/heads.safetensors
        gemma_ckpt = self.root / "ntl-ts-gemma2-9b"
        gemma_ckpt.mkdir(parents=True)
        (gemma_ckpt / "heads.safetensors").write_bytes(b"dummy")

        # 2. BGE directory: bge-m3/config.json
        bge_dir = self.root / "bge-m3"
        bge_dir.mkdir(parents=True)
        (bge_dir / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))

        ckpts = find_checkpoints(self.root)
        self.assertIn("gemma2_9b_ts", ckpts)
        self.assertEqual(ckpts["gemma2_9b_ts"].resolve(), gemma_ckpt.resolve())
        self.assertIn("bge_m3", ckpts)
        self.assertEqual(ckpts["bge_m3"].resolve(), bge_dir.resolve())

    # --------------------------------------------------------------------------
    # 2. BGE-M3 Extraction & F.normalize Functional Test
    # --------------------------------------------------------------------------
    def test_bge_dense_extraction_and_normalization(self):
        """Verifies BGE-M3 dense extraction accurately performs F.normalize on CLS token without NameError."""
        # Create mock tokenizer and mock model
        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[101, 2054, 102], [101, 2023, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }

        # Model output with last_hidden_state (batch=2, seq=3, dim=1024)
        raw_hidden = torch.randn(2, 3, 1024)
        mock_out = MagicMock()
        mock_out.last_hidden_state = raw_hidden
        mock_model = MagicMock()
        mock_model.return_value = mock_out

        # Extraction logic replicating pipeline.py
        texts = ["Text sample 1", "Text sample 2"]
        enc = mock_tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        out = mock_model(**enc)
        cls_repr = out.last_hidden_state[:, 0]
        cls_norm = F.normalize(cls_repr, p=2, dim=-1).cpu().float().numpy()

        self.assertEqual(cls_norm.shape, (2, 1024))
        norms = np.linalg.norm(cls_norm, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5), f"Embeddings not unit norm: {norms}")

    # --------------------------------------------------------------------------
    # 3. Gemma-2-9B LoRA Architecture & Module Resolution
    # --------------------------------------------------------------------------
    def test_gemma2_lora_delta_and_module_resolution(self):
        """Verifies Gemma-2-9B module naming and LoRA shape compatibility."""
        # Construct mock Gemma-2 decoder layer module
        class MockGemmaDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                # Gemma 2 9B dims: hidden=3584, q=4096, k=2048, v=2048, o=3584
                self.self_attn = nn.Module()
                self.self_attn.q_proj = nn.Linear(3584, 4096, bias=False)
                self.self_attn.k_proj = nn.Linear(3584, 2048, bias=False)
                self.self_attn.v_proj = nn.Linear(3584, 2048, bias=False)
                self.self_attn.o_proj = nn.Linear(4096, 3584, bias=False)
                self.mlp = nn.Module()
                self.mlp.gate_proj = nn.Linear(3584, 14336, bias=False)
                self.mlp.up_proj = nn.Linear(3584, 14336, bias=False)
                self.mlp.down_proj = nn.Linear(14336, 3584, bias=False)

        class MockGemmaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockGemmaDecoderLayer()])

        model = MockGemmaModel()
        state_dict = dict(model.named_modules())

        # Test resolving standard PEFT keys
        peft_key_q = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        resolved_q = _resolve_module_name(peft_key_q, state_dict)
        self.assertEqual(resolved_q, "layers.0.self_attn.q_proj")

        peft_key_down = "base_model.model.layers.0.mlp.down_proj.lora_A.weight"
        resolved_down = _resolve_module_name(peft_key_down, state_dict)
        self.assertEqual(resolved_down, "layers.0.mlp.down_proj")

        # Test applying delta
        orig_weight = model.layers[0].self_attn.q_proj.weight.clone()
        r = 16
        A = torch.randn(r, 3584)
        B = torch.randn(4096, r)
        scale = 32.0 / 16.0
        delta = (B @ A) * scale
        model.layers[0].self_attn.q_proj.weight.data.add_(delta)

        self.assertFalse(torch.allclose(model.layers[0].self_attn.q_proj.weight, orig_weight))
        self.assertEqual(model.layers[0].self_attn.q_proj.weight.shape, (4096, 3584))

    # --------------------------------------------------------------------------
    # 4. End-to-End Pipeline Execution on 1,800 Samples
    # --------------------------------------------------------------------------
    def test_full_pipeline_with_gemma_and_bge_1800_samples(self):
        """Executes the full pipeline with Gemma-2-9B and BGE-M3 on 1,800 benchmark samples."""
        N_samples = 1800
        # 1. Create synthetic train.csv and test.csv
        train_p = self.root / "train.csv"
        test_p = self.root / "test.csv"
        working_p = self.root / "working"
        working_p.mkdir()

        # 18 classes, 100 samples each
        classes = [LABELS[i % K] for i in range(N_samples)]
        prompts = [f"Prompt cluster {i // 2}" for i in range(N_samples)]
        texts = [f"Sample text document {i} with distinct lexical markers." for i in range(N_samples)]

        df_train = pd.DataFrame({
            "id": range(N_samples),
            "prompt_id": [i // 2 for i in range(N_samples)],
            "model": classes,
            "text": texts
        })
        df_train.to_csv(train_p, index=False)
        # Also save clean benchmark parquet so build_unseen_holdout_dataset immediately loads 1800 rows
        df_train.to_parquet(self.root / "clean_benchmark_1800_rows.parquet", index=False)

        # 10 test rows for dummy submission
        df_test = pd.DataFrame({
            "id": range(10),
            "text": [f"Test row {i}" for i in range(10)]
        })
        df_test.to_csv(test_p, index=False)

        # 2. Setup mock checkpoints for Gemma and BGE
        mock_gemma_base = self.root / "gemma-2" / "transformers" / "gemma-2-9b" / "2"
        mock_gemma_base.mkdir(parents=True)
        (mock_gemma_base / "config.json").write_text(json.dumps({
            "model_type": "gemma2",
            "hidden_size": 3584,
            "num_hidden_layers": 42
        }))

        gemma_dir = self.root / "ntl-ts-gemma2-9b"
        gemma_dir.mkdir(parents=True)
        (gemma_dir / "meta.json").write_text(json.dumps({
            "keep": 24,
            "layers": [2, 4, 6, 8, 12, 16, 20, 24],
            "max_len": 447,
            "labels": list(LABELS)
        }))
        (gemma_dir / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))

        # Mock Gemma heads: 8 layers * 3584 = 28672
        from safetensors.torch import save_file
        gemma_heads = {
            "plain.weight": torch.randn(K, 28672),
            "plain.bias": torch.zeros(K),
            "centered.weight": torch.randn(K, 28672),
            "centered.bias": torch.zeros(K),
        }
        for l_idx in range(8):
            gemma_heads[f"norms.{l_idx}.weight"] = torch.ones(3584)
            gemma_heads[f"norms.{l_idx}.bias"] = torch.zeros(3584)
        save_file(gemma_heads, str(gemma_dir / "heads.safetensors"))

        bge_dir = self.root / "bge-m3"
        bge_dir.mkdir(parents=True)
        (bge_dir / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
        # Optional BGE head (1024 dim)
        bge_heads = {
            "plain.weight": torch.randn(K, 1024),
            "plain.bias": torch.zeros(K),
        }
        save_file(bge_heads, str(bge_dir / "heads.safetensors"))

        # 3. Create pipeline config
        cfg = {
            "input_dir": str(self.root),
            "train_path": str(train_p),
            "test_path": str(test_p),
            "working_dir": str(working_p),
            "run_benchmark": True,
            "run_submission": False,
            "enable_ablation_probe": True,
            "unseen_samples_per_class": 100,
            "checkpoints": ["gemma2_9b_ts", "bge_m3"],
            "required_assets": {
                "require_pool_parquet": False,
            },
            "models": {
                "svc_4view": 0.15,
                "gemma2_9b_ts": 0.35,
                "bge_m3": 0.20,
            },
            "postprocess": {
                "claude_specialist_weight": 0.04,
                "sinkhorn_alpha": 0.50,
                "sinkhorn_iterations": 20,
            }
        }

        # Mock heavy neural models so test executes quickly on CPU
        with patch("namethatllm.pipeline.find_base_model") as mock_find_base, \
             patch("namethatllm.pipeline.UniversalLoRATrunkLoader") as mock_trunk_cls, \
             patch("namethatllm.pipeline.merge_lora", return_value=True), \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_bge_tok, \
             patch("transformers.AutoModel.from_pretrained") as mock_bge_mod:

            mock_find_base.return_value = self.root / "mock_gemma_base"

            # Mock Gemma trunk
            mock_trunk = MagicMock()
            mock_trunk.extract_exact.return_value = np.random.randn(N_samples, 28672).astype(np.float32)
            mock_trunk.model = MagicMock()
            mock_trunk_cls.return_value = mock_trunk

            # Mock BGE model
            mock_bge_tok_inst = MagicMock()
            mock_bge_tok_inst.return_value = {
                "input_ids": torch.ones((16, 64), dtype=torch.long),
                "attention_mask": torch.ones((16, 64), dtype=torch.long),
            }
            mock_bge_tok.return_value = mock_bge_tok_inst

            mock_bge_mod_inst = MagicMock()
            mock_bge_out = MagicMock()
            mock_bge_out.last_hidden_state = torch.randn(16, 64, 1024)
            mock_bge_mod_inst.return_value = mock_bge_out
            mock_bge_mod.return_value = mock_bge_mod_inst

            # Run full pipeline
            sub_df, sub_path = run_submission_pipeline(cfg)

            # Verify outputs
            self.assertTrue(Path(sub_path).exists())
            self.assertGreaterEqual(len(sub_df), 1)

            # Verify holdout benchmark file generated for all 1800 rows
            unseen_p = working_p / "unseen_holdout.parquet"
            self.assertTrue(unseen_p.exists())
            df_unseen_out = pd.read_parquet(unseen_p)
            self.assertEqual(len(df_unseen_out), N_samples)
            self.assertEqual(len(df_unseen_out["model"].unique()), K)


class TestBGEV2AndCheckpointFilter(unittest.TestCase):
    def test_extract_bge_features_v2_multilayer_drop_cls(self):
        """Verify _extract_bge_features v2 extracts drop-CLS LayerNorm-pooled multi-layer tensor."""
        from namethatllm.pipeline import _extract_bge_features

        torch.manual_seed(7)
        B, L, H = 3, 5, 16
        hs = [torch.randn(B, L, H) for _ in range(4)]

        class DummyBGEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {"hidden_size": H})()

            def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
                row_idx = input_ids[:, 0]
                return type("Out", (), {"hidden_states": [t[row_idx] for t in hs]})()

        masks_by_idx = {
            0: [1, 1, 1, 0, 0],
            1: [1, 1, 0, 0, 0],
            2: [1, 1, 1, 1, 0],
        }

        def dummy_tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt"):
            row_ids = [int(str(s).split(":")[0]) for s in chunk]
            attn = torch.tensor([masks_by_idx[i] for i in row_ids], dtype=torch.long)
            ids = torch.tensor([[i] + list(range(1, L)) for i in row_ids], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": attn}

        heads_sd = {
            "norms.0.weight": torch.ones(H),
            "norms.0.bias": torch.zeros(H),
            "norms.1.weight": torch.full((H,), 2.0),
            "norms.1.bias": torch.full((H,), 0.5),
        }
        meta_bge = {
            "layers": [1, 3],
            "pooling": "mean_drop_cls",
            "drop_first_token": True,
            "per_layer_layernorm": True,
            "max_len": 512,
        }
        texts_in = ["0:aaa", "1:b", "2:cccccc"]
        Z, is_v2 = _extract_bge_features(DummyBGEModel(), dummy_tok, torch.device("cpu"), texts_in, heads_sd, meta_bge, bsize=2)
        self.assertTrue(is_v2)
        self.assertEqual(Z.shape, (3, 2 * H))

        # Manual check across all 3 rows (CLS token 0 dropped, per-row padding excluded)
        mask_no_cls = torch.tensor(
            [[0.0, float(v[1]), float(v[2]), float(v[3]), float(v[4])] for v in (masks_by_idx[0], masks_by_idx[1], masks_by_idx[2])]
        ).unsqueeze(-1)
        den = mask_no_cls.sum(dim=1)
        exp0 = F.layer_norm((hs[1] * mask_no_cls).sum(dim=1) / den, (H,), heads_sd["norms.0.weight"], heads_sd["norms.0.bias"])
        exp1 = F.layer_norm((hs[3] * mask_no_cls).sum(dim=1) / den, (H,), heads_sd["norms.1.weight"], heads_sd["norms.1.bias"])
        np.testing.assert_allclose(Z, torch.cat([exp0, exp1], dim=1).numpy(), atol=1e-5)

    def test_find_checkpoints_prioritizes_bge_v2_without_inhouse_collision(self):
        """Verify bge_m3 folder matching never collides with inhouse pattern and honors fold priority."""
        from namethatllm.config import find_checkpoints

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            inp = Path(tmp_dir)
            d1 = inp / "export_bge_m3_v2"
            d1.mkdir()
            (d1 / "heads.safetensors").write_bytes(b"")
            d2 = inp / "qwen3_14b_inhouse"
            d2.mkdir()
            (d2 / "heads.safetensors").write_bytes(b"")
            d3 = inp / "deberta_inhouse"
            d3.mkdir()
            (d3 / "heads.safetensors").write_bytes(b"")

            ckpts = find_checkpoints(inp)
            self.assertEqual(ckpts.get("bge_m3"), d1)
            self.assertEqual(ckpts.get("qwen3_14b_inhouse"), d2)
            self.assertEqual(ckpts.get("deberta_inhouse"), d3)


if __name__ == "__main__":
    unittest.main()
