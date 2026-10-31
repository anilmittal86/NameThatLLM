"""
Unit and Integration Tests for namethatllm.trainer
==================================================
"""

import unittest
import math
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

from namethatllm.config import LABELS, K
from namethatllm.trainer import (
    MultiLayerLoRAClassifier,
    compute_two_stage_loss,
    PromptGroupBatchSampler,
    export_compatible_checkpoint
)
from namethatllm.neural_extractor import (
    read_checkpoint_meta,
    load_head_file,
    resolve_head_tensors
)

class MockHiddenStatesModel(nn.Module):
    """Tiny mock causal model that outputs 4 hidden states of dimension hidden_size=32."""
    def __init__(self, hidden_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(100, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        h = self.embed(input_ids)
        hs = [h]
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        return SimpleNamespace(hidden_states=hs)

class TestTrainer(unittest.TestCase):
    def test_multi_layer_lora_classifier_forward_and_two_stage_loss(self):
        torch.manual_seed(42)
        hidden_size = 32
        layers = [1, 2, 3]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=4)
        model = MultiLayerLoRAClassifier(
            backbone=mock_backbone,
            hidden_size=hidden_size,
            layers=layers,
            num_classes=K
        )

        batch_size = 6
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention_mask[0, -2:] = 0  # test variable padding

        Z, logits_p, logits_c = model(input_ids, attention_mask, drop_first_token=True)
        self.assertEqual(Z.shape, (batch_size, len(layers) * hidden_size))
        self.assertEqual(logits_p.shape, (batch_size, K))
        self.assertEqual(logits_c.shape, (batch_size, K))

        labels = torch.randint(0, K, (batch_size,))
        prompt_ids = ["p1", "p1", "p1", "p2", "p2", "p3"]

        tot_loss, l_plain, l_centered = compute_two_stage_loss(
            model, Z, labels, prompt_ids=prompt_ids, centered_weight=0.5
        )
        self.assertTrue(torch.isfinite(tot_loss))
        self.assertTrue(torch.isfinite(l_plain))
        self.assertTrue(torch.isfinite(l_centered))

        # Backward pass gradient check
        tot_loss.backward()
        self.assertIsNotNone(model.plain_head.weight.grad)
        self.assertIsNotNone(model.centered_head.weight.grad)
        self.assertGreater(model.plain_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.centered_head.weight.grad.abs().sum().item(), 0.0)

    def test_prompt_group_batch_sampler(self):
        prompt_ids = ["p1", "p1", "p2", "p2", "p2", "p3", "p4", "p4"]
        sampler = PromptGroupBatchSampler(prompt_ids, batch_size=4, shuffle=False)
        batches = list(iter(sampler))
        all_indices = [idx for b in batches for idx in b]
        self.assertEqual(len(all_indices), len(prompt_ids))
        self.assertEqual(sorted(all_indices), list(range(len(prompt_ids))))

    def test_round_trip_checkpoint_export_and_inference_load(self):
        torch.manual_seed(42)
        hidden_size = 32
        layers = [1, 2, 3]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=4)
        model = MultiLayerLoRAClassifier(
            backbone=mock_backbone,
            hidden_size=hidden_size,
            layers=layers,
            num_classes=K
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "export_test"
            N_tr = 36
            Z_train = np.random.randn(N_tr, len(layers) * hidden_size).astype(np.float32)
            y_train = np.random.randint(0, K, size=N_tr)
            prompts_train = np.array([f"p_{i % 4}" for i in range(N_tr)])

            export_compatible_checkpoint(
                model=model,
                output_dir=out_p,
                layers=layers,
                keep=3,
                max_len=447,
                end_id=151643,
                Z_train=Z_train,
                y_train=y_train,
                prompts_train=prompts_train
            )

            # 1. Assert files exist
            self.assertTrue((out_p / "heads.safetensors").exists())
            self.assertTrue((out_p / "meta.json").exists())
            self.assertTrue((out_p / "wPCA_bank.npz").exists())

            # 2. Assert meta loads accurately
            meta = read_checkpoint_meta(out_p, default_keep=10, default_layers=[])
            self.assertEqual(meta["keep"], 3)
            self.assertEqual(meta["layers"], layers)
            self.assertEqual(meta["max_len"], 447)
            self.assertEqual(meta["end_id"], 151643)
            self.assertEqual(meta["labels"], list(LABELS))

            # 3. Assert heads load with 100% numerical equality
            heads_sd = load_head_file(out_p)
            W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

            expected_W_p = model.plain_head.weight.detach().cpu().numpy()
            expected_b_p = model.plain_head.bias.detach().cpu().numpy()
            expected_W_c = model.centered_head.weight.detach().cpu().numpy()
            expected_b_c = model.centered_head.bias.detach().cpu().numpy()

            self.assertTrue(np.allclose(W_p, expected_W_p, atol=1e-7))
            self.assertTrue(np.allclose(b_p, expected_b_p, atol=1e-7))
            self.assertTrue(np.allclose(W_c, expected_W_c, atol=1e-7))
            self.assertTrue(np.allclose(b_c, expected_b_c, atol=1e-7))

    def test_collate_text_batch_preserves_token_zero(self):
        from namethatllm.trainer import collate_text_batch
        b1 = {
            "input_ids": torch.tensor([0, 42, 99], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "label": torch.tensor(1, dtype=torch.long),
            "prompt_id": 101,
        }
        b2 = {
            "input_ids": torch.tensor([15, 20, 25, 30], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "label": torch.tensor(2, dtype=torch.long),
            "prompt_id": 102,
        }
        res = collate_text_batch([b1, b2], pad_token_id=151643)
        # Token ID 0 must NOT be treated as padding
        self.assertEqual(res["input_ids"][0, 0].item(), 0)
        self.assertEqual(res["attention_mask"][0, 0].item(), 1)
        # Padding value must be 151643 and attention_mask must be 0
        self.assertEqual(res["input_ids"][0, 3].item(), 151643)
        self.assertEqual(res["attention_mask"][0, 3].item(), 0)
        # Second sequence should have mask 1 on all positions
        self.assertTrue((res["attention_mask"][1] == 1).all())

    def test_peft_wrapped_model_disables_final_norm(self):
        from namethatllm.trainer import _disable_final_norm
        class MockNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(16))
            def forward(self, x):
                return x * self.weight

        class MockBaseInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = MockNorm()

        class MockBase(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockBaseInner()

        class MockPeftWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = MockBase()

        wrapped = MockPeftWrapper()
        self.assertIsInstance(wrapped.base_model.model.norm, MockNorm)
        _disable_final_norm(wrapped)
        self.assertIsInstance(wrapped.base_model.model.norm, nn.Identity)

    def test_train_prompt_group_step_microbatched(self):
        from namethatllm.trainer import train_prompt_group_step
        torch.manual_seed(42)
        hidden_size = 32
        layers = [1, 2, 3]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=4)
        model = MultiLayerLoRAClassifier(
            backbone=mock_backbone,
            hidden_size=hidden_size,
            layers=layers,
            num_classes=K
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 100, (6, 12))
        attention_mask = torch.ones(6, 12, dtype=torch.long)
        attention_mask[0, 8:] = 0
        labels = torch.randint(0, K, (6,))

        opt.zero_grad()
        tot_loss, l_plain, l_centered = train_prompt_group_step(
            model=model,
            opt=opt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            micro_batch_size=2,
            centered_weight=0.5,
            label_smoothing=0.10
        )
        self.assertTrue(math.isfinite(tot_loss))
        self.assertTrue(math.isfinite(l_plain))
        self.assertTrue(math.isfinite(l_centered))

        # Check gradients are populated on heads
        self.assertIsNotNone(model.plain_head.weight.grad)
        self.assertIsNotNone(model.centered_head.weight.grad)
        self.assertGreater(model.plain_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.centered_head.weight.grad.abs().sum().item(), 0.0)

    def test_family_margin_cross_entropy_penalizes_sister_classes(self):
        from namethatllm.trainer import family_margin_cross_entropy, _build_sister_mask
        from namethatllm.config import FAMILIES
        mask = _build_sister_mask(torch.device("cpu"))
        self.assertEqual(mask.shape, (K, K))

        # Check GPT family sister relationships
        gpt_members = FAMILIES["gpt"]
        for m1 in gpt_members:
            for m2 in gpt_members:
                if m1 != m2:
                    self.assertEqual(mask[m1, m2].item(), 1.0)
                else:
                    self.assertEqual(mask[m1, m2].item(), 0.0)

        # Sister margin penalty check
        logits = torch.zeros(2, K)
        labels = torch.tensor([0, 5])  # 0 is GPT, 5 is Gemini
        loss_zero_margin = family_margin_cross_entropy(logits, labels, family_margin=0.0)
        loss_with_margin = family_margin_cross_entropy(logits, labels, family_margin=0.35)
        self.assertGreater(loss_with_margin.item(), loss_zero_margin.item())

    def test_bucketed_length_batch_sampler(self):
        from namethatllm.trainer import BucketedLengthBatchSampler
        lengths = [10, 500, 15, 480, 20, 510, 12, 490]
        sampler = BucketedLengthBatchSampler(lengths, batch_size=2, bucket_size_multiplier=2, shuffle=False)
        batches = list(iter(sampler))
        all_indices = [idx for b in batches for idx in b]
        self.assertEqual(sorted(all_indices), list(range(len(lengths))))

    def test_trainer_preserves_cross_entropy_heads_without_ridge_overwrite(self):
        torch.manual_seed(42)
        hidden_size = 16
        layers = [1, 2]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=3)
        model = MultiLayerLoRAClassifier(
            backbone=mock_backbone,
            hidden_size=hidden_size,
            layers=layers,
            num_classes=K
        )
        # Custom head values
        model.plain_head.weight.data.fill_(0.42)
        model.plain_head.bias.data.fill_(0.15)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "ckpt_export"
            export_compatible_checkpoint(
                model=model,
                output_dir=out_p,
                layers=layers,
                keep=2,
                max_len=128,
                end_id=151643
            )
            heads_sd = load_head_file(out_p)
            W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
            self.assertIsNotNone(W_p)
            self.assertIsNotNone(b_p)
            self.assertAlmostEqual(float(W_p[0, 0]), 0.42, places=4)
            self.assertAlmostEqual(float(b_p[0]), 0.15, places=4)

    def test_compute_supcon_loss_unique_labels_no_nan(self):
        from namethatllm.trainer import compute_supcon_loss
        torch.manual_seed(42)
        Z_c_mb = torch.randn(2, 64, requires_grad=True)
        labels_mb = torch.tensor([0, 1])
        centered_head_weight = torch.randn(K, 64, requires_grad=True)
        Z_group_centered = torch.randn(18, 64)

        loss = compute_supcon_loss(
            Z_c_mb,
            labels_mb,
            centered_head_weight,
            Z_group_centered=Z_group_centered,
            temperature=0.10
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(Z_c_mb.grad)
        self.assertIsNotNone(centered_head_weight.grad)
        self.assertGreater(Z_c_mb.grad.abs().sum().item(), 0.0)
        self.assertGreater(centered_head_weight.grad.abs().sum().item(), 0.0)

    def test_stage1_time_budget_guard_and_metadata_backbone(self):
        from namethatllm.config import _inspect_checkpoint_dim_and_keep, find_checkpoints
        torch.manual_seed(42)
        hidden_size = 16
        layers = [1, 2]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=3)
        model = MultiLayerLoRAClassifier(
            backbone=mock_backbone,
            hidden_size=hidden_size,
            layers=layers,
            num_classes=K
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Export Qwen3-14B inhouse
            q3_p = tmp_path / "export_champion_qwen3_14b"
            export_compatible_checkpoint(
                model=model,
                output_dir=q3_p,
                layers=layers,
                keep=27,
                max_len=128,
                end_id=151643,
                backbone_name="Qwen/Qwen3-14B-Base"
            )
            dim3, keep3, bb3 = _inspect_checkpoint_dim_and_keep(q3_p)
            self.assertIn("qwen3", bb3)
            self.assertEqual(keep3, 27)

            # Export Qwen2.5-14B inhouse
            q25_p = tmp_path / "export_champion_qwen25_14b"
            export_compatible_checkpoint(
                model=model,
                output_dir=q25_p,
                layers=layers,
                keep=32,
                max_len=128,
                end_id=151643,
                backbone_name="Qwen/Qwen2.5-14B"
            )
            dim25, keep25, bb25 = _inspect_checkpoint_dim_and_keep(q25_p)
            self.assertIn("qwen2.5", bb25)
            self.assertEqual(keep25, 32)

            # Checkpoint routing
            ckpts = find_checkpoints(tmp_path)
            self.assertIn("qwen3_14b_inhouse", ckpts)
            self.assertIn("qwen25_14b_inhouse", ckpts)
            self.assertEqual(ckpts["qwen3_14b_inhouse"], q3_p)
            self.assertEqual(ckpts["qwen25_14b_inhouse"], q25_p)

    def test_both_notebooks_compile_with_ast_parity(self):
        repo_root = Path(__file__).resolve().parents[1]
        build_script = repo_root / "scripts" / "build_training_notebook.py"
        self.assertFalse(build_script.exists(), "scripts/build_training_notebook.py must NOT exist")

        compile_script = repo_root / "scripts" / "compile_notebook.py"
        self.assertTrue(compile_script.exists())
        lines = compile_script.read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(lines), 150, f"compile_notebook.py is {len(lines)} lines; must be <= 150")

        # Compile both notebooks
        from scripts.compile_notebook import compile_submission_notebook, compile_training_notebook
        sub_out = compile_submission_notebook(
            repo_root / "configs" / "production_ensemble.json",
            repo_root / "kernels" / "submission" / "kaggle_submission.ipynb"
        )
        self.assertTrue(sub_out.exists())

        trn_out = compile_training_notebook(
            repo_root / "configs" / "training_lora.json",
            repo_root / "kernels" / "training" / "lora_training.ipynb"
        )
        self.assertTrue(trn_out.exists())

if __name__ == "__main__":
    unittest.main()

