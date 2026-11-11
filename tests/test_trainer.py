"""
Unit and Integration Tests for namethatllm.trainer
==================================================
"""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from namethatllm.config import LABELS, K
from namethatllm.neural_extractor import (
    load_head_file,
    read_checkpoint_meta,
    resolve_head_tensors,
)
from namethatllm.trainer import (
    MultiLayerLoRAClassifier,
    PromptGroupBatchSampler,
    compute_two_stage_loss,
    evaluate_stage2_oof_cv,
    export_compatible_checkpoint,
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
        model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K)

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

        tot_loss, l_plain, l_centered = compute_two_stage_loss(model, Z, labels, prompt_ids=prompt_ids, centered_weight=0.5)
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
        model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K)

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
                prompts_train=prompts_train,
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
        model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K)
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
            label_smoothing=0.10,
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
        from namethatllm.config import FAMILIES
        from namethatllm.trainer import _build_sister_mask, family_margin_cross_entropy

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

        # Single-family specialist margin checks:
        # Gemini specialist (4 classes): 3.7-flash (2) <-> 3.8-flash (3) gets 1.5 boost
        gem_logits = torch.zeros(2, 4)
        gem_labels = torch.tensor([2, 0])
        gem_loss = family_margin_cross_entropy(gem_logits, gem_labels, family_margin=0.35)
        self.assertGreater(gem_loss.item(), 0.0)

        # GPT specialist (5 classes): luna (1) <-> sol (2) <-> terra (3) gets 1.4 boost
        gpt_logits = torch.zeros(3, 5)
        gpt_labels = torch.tensor([1, 2, 3])
        gpt_loss = family_margin_cross_entropy(gpt_logits, gpt_labels, family_margin=0.40)
        self.assertGreater(gpt_loss.item(), 0.0)

        # GLM specialist (2 classes)
        glm_logits = torch.zeros(2, 2)
        glm_labels = torch.tensor([0, 1])
        glm_loss = family_margin_cross_entropy(glm_logits, glm_labels, family_margin=0.40)
        self.assertGreater(glm_loss.item(), 0.0)

        # Boosted 18-class sister mask checks:
        boosted_mask = _build_sister_mask(torch.device("cpu"), boosted=True)
        self.assertAlmostEqual(boosted_mask[7, 8].item(), 2.0, places=5)
        self.assertAlmostEqual(boosted_mask[8, 7].item(), 1.3, places=5)
        self.assertAlmostEqual(boosted_mask[2, 3].item(), 1.4, places=5)
        self.assertAlmostEqual(boosted_mask[15, 16].item(), 1.5, places=5)
        self.assertAlmostEqual(boosted_mask[16, 15].item(), 1.7, places=5)

    def test_family_specialists_checkpoint_export_and_inference_load(self):
        """Verify export_compatible_checkpoint and resolve_head_tensors for 5-class, 4-class, and 2-class specialists."""
        from namethatllm.config import FAMILIES, LABELS
        from namethatllm.neural_extractor import (
            load_head_file,
            resolve_head_tensors,
        )
        from namethatllm.trainer import (
            MultiLayerLoRAClassifier,
            export_compatible_checkpoint,
        )

        hidden_size = 32
        layers = [1, 2, 3]

        for fam_name in ("gpt", "gemini", "glm"):
            fam_cols = FAMILIES[fam_name]
            K_fam = len(fam_cols)
            active_labels = [LABELS[c] for c in fam_cols]

            mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=4)
            model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K_fam)

            with tempfile.TemporaryDirectory() as tmp_dir:
                out_p = Path(tmp_dir) / f"export_{fam_name}"
                export_compatible_checkpoint(
                    model=model,
                    output_dir=out_p,
                    layers=layers,
                    keep=3,
                    max_len=447,
                    end_id=151643,
                    class_labels=active_labels,
                    target_family=fam_name,
                )

                # 1. Assert files exist
                self.assertTrue((out_p / "heads.safetensors").exists())
                self.assertTrue((out_p / "meta.json").exists())
                # wPCA_bank must NOT be written for specialists
                self.assertFalse((out_p / "wPCA_bank.npz").exists())

                # 2. Check meta.json
                meta_json = json.loads((out_p / "meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta_json["target_family"], fam_name)
                self.assertEqual(meta_json["labels"], active_labels)
                self.assertEqual(meta_json["classes"], active_labels)

                # 3. Check heads.safetensors resolution
                heads_sd = load_head_file(out_p)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd, num_classes=K_fam)

                self.assertIsNotNone(W_p)
                self.assertEqual(W_p.shape, (K_fam, len(layers) * hidden_size))
                self.assertIsNotNone(b_p)
                self.assertEqual(b_p.shape, (K_fam,))
                self.assertIsNotNone(W_c)
                self.assertEqual(W_c.shape, (K_fam, len(layers) * hidden_size))
                self.assertIsNotNone(b_c)
                self.assertEqual(b_c.shape, (K_fam,))

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
        model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K)
        # Custom head values
        model.plain_head.weight.data.fill_(0.42)
        model.plain_head.bias.data.fill_(0.15)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "ckpt_export"
            export_compatible_checkpoint(model=model, output_dir=out_p, layers=layers, keep=2, max_len=128, end_id=151643)
            heads_sd = load_head_file(out_p)
            W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
            self.assertIsNotNone(W_p)
            self.assertIsNotNone(b_p)
            self.assertAlmostEqual(float(W_p[0, 0]), 0.42, places=4)
            self.assertAlmostEqual(float(b_p[0]), 0.15, places=4)

    def test_compute_supcon_loss_unique_labels_no_nan(self):
        from namethatllm.trainer import compute_supcon_loss

        torch.manual_seed(42)

        # 1. Verification of 2-arg signature on distinct-class positive cosine vectors
        z_pos = torch.tensor([[1.0, 0.0], [0.8, 0.6]], requires_grad=True)
        labels_distinct = torch.tensor([0, 1])
        loss_pos = compute_supcon_loss(z_pos, labels_distinct)
        self.assertTrue(torch.isfinite(loss_pos))
        self.assertGreater(loss_pos.item(), 0.0)

        loss_pos.backward()
        self.assertIsNotNone(z_pos.grad)
        self.assertGreater(z_pos.grad.abs().sum().item(), 0.0)

        # 2. Verification of multi-sample prompt group with prototypes
        Z_c_mb = torch.randn(2, 64, requires_grad=True)
        labels_mb = torch.tensor([0, 1])
        centered_head_weight = torch.randn(K, 64, requires_grad=True)
        Z_group_centered = torch.randn(18, 64)

        loss = compute_supcon_loss(Z_c_mb, labels_mb, temperature=0.10, prototypes=centered_head_weight, Z_group_centered=Z_group_centered)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(Z_c_mb.grad)
        self.assertIsNotNone(centered_head_weight.grad)
        self.assertGreater(Z_c_mb.grad.abs().sum().item(), 0.0)
        self.assertGreater(centered_head_weight.grad.abs().sum().item(), 0.0)

    def test_stage1_time_budget_guard_and_metadata_backbone(self):
        from safetensors.torch import save_file

        from namethatllm.config import (
            _inspect_checkpoint_dim_and_keep,
            find_checkpoints,
        )

        torch.manual_seed(42)
        hidden_size = 16
        layers = [1, 2]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=3)
        model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Export Qwen3-14B inhouse
            q3_p = tmp_path / "export_champion_qwen3_14b"
            export_compatible_checkpoint(
                model=model, output_dir=q3_p, layers=layers, keep=27, max_len=128, end_id=151643, backbone_name="Qwen/Qwen3-14B-Base"
            )
            # Create a 14B heads file containing 1D norms.0.weight (5120) and 2D plain.weight (18, 40960)
            mock_14b_heads = {
                "norms.0.weight": torch.ones(5120),
                "plain.weight": torch.randn(18, 40960),
                "plain.bias": torch.zeros(18),
                "centered.weight": torch.randn(18, 40960),
                "centered.bias": torch.zeros(18),
            }
            save_file(mock_14b_heads, str(q3_p / "heads.safetensors"))

            dim3, keep3, bb3 = _inspect_checkpoint_dim_and_keep(q3_p)
            self.assertEqual(dim3, 40960, "Must inspect 2D tensor shape 40960, NOT 1D norm tensor 5120")
            self.assertIn("qwen3", bb3)
            self.assertEqual(keep3, 27)

            # Export Qwen2.5-14B inhouse
            q25_p = tmp_path / "export_champion_qwen25_14b"
            export_compatible_checkpoint(
                model=model, output_dir=q25_p, layers=layers, keep=32, max_len=128, end_id=151643, backbone_name="Qwen/Qwen2.5-14B"
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
        from scripts.compile_notebook import (
            compile_submission_notebook,
            compile_training_notebook,
        )

        sub_out = compile_submission_notebook(
            repo_root / "configs" / "production_ensemble.json", repo_root / "kernels" / "submission" / "kaggle_submission.ipynb"
        )
        self.assertTrue(sub_out.exists())

        trn_out = compile_training_notebook(
            repo_root / "configs" / "training_lora.json", repo_root / "kernels" / "training" / "lora_training.ipynb"
        )
        self.assertTrue(trn_out.exists())

    def test_merge_lora_atomic_rejection_on_mlp_shape_mismatch(self):
        """Verify merge_lora atomically rejects mismatched adapter (e.g. Llama vs Qwen) without partial mutation."""
        from safetensors.torch import save_file

        from namethatllm.neural_extractor import merge_lora

        class DummyTrunk(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "self_attn": nn.ModuleDict(
                                    {
                                        "q_proj": nn.Linear(64, 64, bias=False),
                                    }
                                ),
                                "mlp": nn.ModuleDict(
                                    {
                                        "up_proj": nn.Linear(64, 128, bias=False),  # Qwen shape: out=128, in=64
                                    }
                                ),
                            }
                        )
                    ]
                )

        trunk = DummyTrunk()
        orig_q = trunk.layers[0]["self_attn"]["q_proj"].weight.clone()
        orig_up = trunk.layers[0]["mlp"]["up_proj"].weight.clone()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg = {"r": 4, "lora_alpha": 16}
            (tmp / "adapter_config.json").write_text(json.dumps(cfg))

            # q_proj matches (out=64, in=64), but up_proj has Llama shape (out=256, in=64)
            sd = {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 64),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 4),
                "base_model.model.layers.0.mlp.up_proj.lora_A.weight": torch.randn(4, 64),
                "base_model.model.layers.0.mlp.up_proj.lora_B.weight": torch.randn(256, 4),  # Mismatch: out=256 != 128
            }
            save_file(sd, str(tmp / "adapter_model.safetensors"))

            merged = merge_lora(trunk, tmp, sign=+1.0)
            self.assertFalse(merged, "merge_lora must return False on shape mismatch")
            # Verify 100% untouched: q_proj must NOT have been mutated even though its shape matched!
            self.assertTrue(torch.equal(trunk.layers[0]["self_attn"]["q_proj"].weight, orig_q), "q_proj must remain 100% unmodified")
            self.assertTrue(torch.equal(trunk.layers[0]["mlp"]["up_proj"].weight, orig_up), "up_proj must remain 100% unmodified")

    def test_multilayer_lora_gradient_checkpointing_backward_pass(self):
        """Verify MultiLayerLoRAClassifier forward + loss.backward() with gradient checkpointing does not raise CheckpointError."""
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.for_model(
            "qwen2", hidden_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, intermediate_size=128
        )
        base = AutoModel.from_config(cfg)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        clf = MultiLayerLoRAClassifier(backbone=base, hidden_size=64, layers=[1, 3], num_classes=18)

        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        Z, logits_p, logits_c = clf(input_ids, attention_mask)
        loss = F.cross_entropy(logits_p, labels)
        loss.backward()

        self.assertIsNotNone(clf.plain_head.weight.grad)
        self.assertGreater(clf.plain_head.weight.grad.abs().sum().item(), 0.0)

    def test_inspect_checkpoint_dim_and_keep_ignores_1d_norms(self):
        """Verify _inspect_checkpoint_dim_and_keep ignores 1D norms.0.weight and returns 2D plain.weight dimension."""
        from safetensors.torch import save_file

        from namethatllm.config import _inspect_checkpoint_dim_and_keep

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            meta = {"cfg": {"keep": 27, "backbone": "qwen3-14b"}}
            (tmp / "meta.json").write_text(json.dumps(meta))

            sd = {
                "norms.0.weight": torch.ones(5120),  # 1D tensor
                "norms.0.bias": torch.zeros(5120),
                "plain.weight": torch.randn(18, 40960),  # 2D tensor: dim=40960
                "plain.bias": torch.zeros(18),
            }
            save_file(sd, str(tmp / "heads.safetensors"))

            dim, keep, backbone = _inspect_checkpoint_dim_and_keep(tmp)
            self.assertEqual(dim, 40960, "dim must be 40960 (from 2D plain.weight), NOT 5120 (from 1D norms)")
            self.assertEqual(keep, 27)
            self.assertEqual(backbone, "qwen3-14b")

    def test_setup_training_env_alias_and_execution(self):
        """Verify _setup_training_env and _configure_training_environment alias configure environment without side-effects."""
        from namethatllm.trainer import _configure_training_environment, _setup_training_env

        self.assertIs(_setup_training_env, _configure_training_environment)
        _setup_training_env()
        self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True")

    def test_modal_trainer_launcher_thin_design(self):
        """Verify scripts/modal_trainer.py is a thin launcher delegating to namethatllm.trainer."""
        modal_script = Path(__file__).resolve().parents[1] / "scripts" / "modal_trainer.py"
        self.assertTrue(modal_script.exists())
        code = modal_script.read_text(encoding="utf-8")

        # Must NOT contain legacy duplicated model / training loops
        self.assertNotIn("def build_model_and_tokenizer", code)
        self.assertNotIn("class PassageDataset", code)
        self.assertNotIn("def collate_fn", code)

        # Must delegate to namethatllm.trainer
        self.assertIn("from namethatllm.trainer import run_training_pipeline", code)
        self.assertIn("train_cloud_worker", code)

    def test_evaluate_stage2_oof_cv_overfitting_gap(self):
        """Verify 5-stage GroupKFold CV partitions by prompt_id, quantifies overfitting gap, and exports report + predictions."""
        np.random.seed(42)
        n_prompts = 10
        samples_per_prompt = K
        n_samples = n_prompts * samples_per_prompt
        dim = 64

        y = np.tile(np.arange(K), n_prompts)
        prompts = np.repeat([f"prompt_{p}" for p in range(n_prompts)], samples_per_prompt)
        author_prototypes = np.random.randn(K, dim)
        prompt_effects = np.random.randn(n_prompts, dim)

        Z = np.zeros((n_samples, dim), dtype=np.float32)
        for i in range(n_samples):
            p_idx = int(prompts[i].split("_")[1])
            Z[i] = author_prototypes[y[i]] + 0.3 * prompt_effects[p_idx] + 0.1 * np.random.randn(dim)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir)
            report = evaluate_stage2_oof_cv(
                Z=Z,
                y=y,
                prompts=prompts,
                n_splits=5,
                alpha=5.0,
                label_smoothing=0.05,
                num_classes=K,
                class_labels=list(LABELS),
                output_dir=out_p,
            )

            self.assertEqual(report["n_splits"], 5)
            self.assertEqual(report["n_samples"], 180)
            self.assertEqual(report["n_prompts"], 10)
            self.assertEqual(len(report["folds"]), 5)
            self.assertIn("mean_train_bacc", report)
            self.assertIn("oof_sink_bacc", report)
            self.assertIn("overfit_gap", report)
            self.assertIsInstance(report["overfit_gap"], float)
            self.assertIn(report["verdict"].split()[0], {"LOW", "MODERATE", "HIGH"})

            json_file = out_p / "stage2_cv_report.json"
            self.assertTrue(json_file.exists())
            saved_report = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["n_splits"], 5)
            self.assertEqual(len(saved_report["folds"]), 5)

            npz_file = out_p / "stage2_oof_predictions.npz"
            self.assertTrue(npz_file.exists())
            with np.load(npz_file) as npz_data:
                self.assertEqual(npz_data["oof_plain_lp"].shape, (180, K))
                self.assertEqual(npz_data["oof_centered_lp"].shape, (180, K))
                self.assertEqual(npz_data["oof_blend_lp"].shape, (180, K))
                self.assertEqual(npz_data["oof_sink_probs"].shape, (180, K))
                np.testing.assert_array_equal(npz_data["y_true"], y)
                np.testing.assert_array_equal(npz_data["prompts"], prompts)

            single_prompt = np.array(["p1"] * 20)
            empty_report = evaluate_stage2_oof_cv(
                Z=Z[:20],
                y=y[:20],
                prompts=single_prompt,
                n_splits=5,
                num_classes=K,
            )
            self.assertEqual(empty_report, {})

    def test_loraplus_param_groups_asymmetric_learning_rates(self):
        """Verify LoRA+ splits lora_A and lora_B parameters with asymmetric multiplier."""
        from namethatllm.trainer import _get_loraplus_param_groups

        hidden_size = 16
        layers = [1, 2]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=3)
        # Mock lora_A and lora_B named parameters in backbone
        mock_backbone.lora_A = nn.Linear(hidden_size, 4, bias=False)
        mock_backbone.lora_B = nn.Linear(4, hidden_size, bias=False)

        model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K)

        lr_lora = 1e-4
        lr_head = 5e-4
        ratio = 2.5
        groups = _get_loraplus_param_groups(model, lr_lora=lr_lora, lr_head=lr_head, lora_plus_ratio=ratio, include_centered_head=True)

        lrs = [g["lr"] for g in groups]
        self.assertIn(lr_lora, lrs)
        self.assertIn(lr_lora * ratio, lrs)
        self.assertIn(lr_head, lrs)

    def test_family_margin_cross_entropy_hard_class_weights(self):
        """Verify family_margin_cross_entropy accepts class weights and increases penalty on hard classes."""
        from namethatllm.trainer import family_margin_cross_entropy

        # Sample 0 has high confidence for class 0 (low loss ~0.01), sample 1 has low confidence for class 3 (high loss ~2.89)
        logits = torch.zeros(2, K)
        logits[0, 0] = 5.0
        labels = torch.tensor([0, 3])  # 3 is in HARD_CLASS_INDICES
        weights = torch.ones(K)
        weights[3] = 2.0  # Upweight the high-loss hard class

        unweighted_loss = family_margin_cross_entropy(logits, labels, family_margin=0.0)
        weighted_loss = family_margin_cross_entropy(logits, labels, family_margin=0.0, weight=weights)

        self.assertTrue(torch.isfinite(weighted_loss))
        self.assertGreater(weighted_loss.item(), unweighted_loss.item())

    def test_variable_size_sub_cluster_centering_step(self):
        """Verify train_prompt_group_step executes with N >= 8 variable sub-cluster centering."""
        from namethatllm.trainer import train_prompt_group_step

        torch.manual_seed(42)
        hidden_size = 16
        layers = [1, 2]
        mock_backbone = MockHiddenStatesModel(hidden_size=hidden_size, num_layers=3)
        model = MultiLayerLoRAClassifier(backbone=mock_backbone, hidden_size=hidden_size, layers=layers, num_classes=K)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        N = 10  # >= 8 to trigger variable sub-cluster logic
        seq_len = 12
        input_ids = torch.randint(0, 100, (N, seq_len))
        attention_mask = torch.ones((N, seq_len), dtype=torch.long)
        labels = torch.arange(N) % K

        tot_loss, p_loss, c_loss = train_prompt_group_step(
            model=model,
            opt=opt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            micro_batch_size=4,
            centered_weight=0.60,
            family_margin=0.35,
            supcon_weight=0.20,
            compute_dtype=torch.float32,
        )
        self.assertTrue(math.isfinite(tot_loss))
        self.assertTrue(math.isfinite(p_loss))
        self.assertTrue(math.isfinite(c_loss))

    def test_specialist_hard_class_weights_mapping(self):
        """Verify hard class weights in specialist training map correctly by label name (not local index)."""
        from namethatllm.config import HARD_CLASS_INDICES, LAB2I

        hard_set = set(HARD_CLASS_INDICES)
        sub_lab2i_glm = {"glm-5.3": 0, "glm-5.3-flash": 1}
        num_classes = len(sub_lab2i_glm)
        stage1_hard_weight = 1.40

        stage1_class_weights = torch.ones(num_classes, dtype=torch.float32)
        for lbl, local_idx in sub_lab2i_glm.items():
            if LAB2I.get(lbl) in hard_set:
                stage1_class_weights[local_idx] = stage1_hard_weight

        # Both glm-5.3 (15) and glm-5.3-flash (16) are in HARD_CLASS_INDICES
        self.assertAlmostEqual(stage1_class_weights[0].item(), 1.40, places=4)
        self.assertAlmostEqual(stage1_class_weights[1].item(), 1.40, places=4)

        # Non-hard specialist class test (e.g., gpt-5.6-astra is 0 (not hard), gpt-5.6-luna is 1 (hard))
        sub_lab2i_gpt = {"gpt-5.6-astra": 0, "gpt-5.6-luna": 1}
        weights_gpt = torch.ones(2, dtype=torch.float32)
        for lbl, local_idx in sub_lab2i_gpt.items():
            if LAB2I.get(lbl) in hard_set:
                weights_gpt[local_idx] = stage1_hard_weight

        self.assertAlmostEqual(weights_gpt[0].item(), 1.00, places=4)
        self.assertAlmostEqual(weights_gpt[1].item(), 1.40, places=4)

    def test_blend_head_scaled_and_centered_bias(self):
        """Verify _blend_head scales and centers the ridge bias to prevent class prior distortion."""
        from namethatllm.trainer import _blend_head

        head = torch.nn.Linear(4, 2)
        head.weight.data = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        head.bias.data = torch.tensor([0.5, -0.5])

        # W_r has norm 10x smaller than head.weight
        W_r = np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
        b_r = np.array([0.05, -0.05], dtype=np.float32)

        _blend_head(head, W_r, b_r, w_neural=0.5, w_ridge=0.5)

        # Scale is 10.0. b_r centered & scaled: (0.05 - 0)*10 + 0 = 0.5. Blended bias = 0.5*0.5 + 0.5*0.5 = 0.5
        self.assertAlmostEqual(head.bias.data[0].item(), 0.50, places=4)
        self.assertAlmostEqual(head.bias.data[1].item(), -0.50, places=4)
        self.assertAlmostEqual(head.weight.data[0, 0].item(), 1.00, places=4)

    def test_resolve_all_pool_files_includes_pool_120k(self):
        """Verify resolve_all_pool_files discovers pool_120k.parquet."""
        from namethatllm.config import resolve_all_pool_files

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            p120 = tmp_p / "pool_120k.parquet"
            p120.touch()

            discovered = resolve_all_pool_files(tmp_p)
            self.assertIn(p120, discovered)

    def test_compute_grouped_inference_logits_and_trace_scaled_ridge(self):
        """Verify trace-scaled fit_ridge_heads and compute_grouped_inference_logits."""
        from namethatllm.trainer import fit_ridge_heads, compute_grouped_inference_logits

        np.random.seed(42)
        N, D = 36, 64
        Z = np.random.randn(N, D).astype(np.float32) * 5.0
        y = np.random.randint(0, 18, size=N)
        prompts = np.repeat(np.arange(12), 3)

        W_p, b_p, W_c, b_c = fit_ridge_heads(Z, y, prompts=prompts, alpha=10.0, num_classes=18)
        self.assertEqual(W_p.shape, (18, D))
        self.assertEqual(W_c.shape, (18, D))

        # Test grouped inference with prompt groups
        res_grouped = compute_grouped_inference_logits(
            Z, W_p, b_p, W_c, b_c, groups=prompts, blend_centered=0.65
        )
        self.assertIn("logits_plain", res_grouped)
        self.assertIn("logits_centered", res_grouped)
        self.assertIn("logits_blend", res_grouped)
        self.assertIn("probs_sinkhorn", res_grouped)
        self.assertEqual(res_grouped["probs_sinkhorn"].shape, (N, 18))
        self.assertTrue(np.allclose(res_grouped["probs_sinkhorn"].sum(axis=1), 1.0, atol=1e-5))

        # Test grouped inference with None groups (global batch centering fallback)
        res_none = compute_grouped_inference_logits(
            Z, W_p, b_p, W_c, b_c, groups=None, blend_centered=0.65
        )
        self.assertEqual(res_none["probs_sinkhorn"].shape, (N, 18))


if __name__ == "__main__":
    unittest.main()
