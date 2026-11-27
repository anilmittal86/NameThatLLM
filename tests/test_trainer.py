"""
Unit and Integration Tests for namethatllm.trainer
==================================================
"""

import unittest
import math
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        loss = compute_supcon_loss(
            Z_c_mb,
            labels_mb,
            temperature=0.10,
            prototypes=centered_head_weight,
            Z_group_centered=Z_group_centered
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
        from safetensors.torch import save_file
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
            # Create a 14B heads file containing 1D norms.0.weight (5120) and 2D plain.weight (18, 40960)
            mock_14b_heads = {
                "norms.0.weight": torch.ones(5120),
                "plain.weight": torch.randn(18, 40960),
                "plain.bias": torch.zeros(18),
                "centered.weight": torch.randn(18, 40960),
                "centered.bias": torch.zeros(18)
            }
            save_file(mock_14b_heads, str(q3_p / "heads.safetensors"))

            dim3, keep3, bb3 = _inspect_checkpoint_dim_and_keep(q3_p)
            self.assertEqual(dim3, 40960, "Must inspect 2D tensor shape 40960, NOT 1D norm tensor 5120")
            self.assertIn("qwen3", bb3)
            self.assertEqual(keep3, 27)

            # Export V2 Qwen3-14B inhouse with an internal stage1_best backup subfolder
            q3_v2_p = tmp_path / "export_inhouse_qwen3_14b_v2"
            export_compatible_checkpoint(
                model=model,
                output_dir=q3_v2_p,
                layers=layers,
                keep=27,
                max_len=384,
                end_id=151643,
                backbone_name="Qwen/Qwen3-14B-Base"
            )
            save_file(mock_14b_heads, str(q3_v2_p / "heads.safetensors"))
            export_compatible_checkpoint(
                model=model,
                output_dir=q3_v2_p / "stage1_best",
                layers=layers,
                keep=27,
                max_len=384,
                end_id=151643,
                backbone_name="Qwen/Qwen3-14B-Base"
            )
            save_file(mock_14b_heads, str(q3_v2_p / "stage1_best" / "heads.safetensors"))

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

            # Checkpoint routing (both v1 and v2 stay distinct, and stage1_best is skipped)
            ckpts = find_checkpoints(tmp_path)
            self.assertIn("qwen3_14b_inhouse", ckpts)
            self.assertIn("qwen3_14b_inhouse_v2", ckpts)
            self.assertIn("qwen25_14b_inhouse", ckpts)
            self.assertEqual(ckpts["qwen3_14b_inhouse"], q3_p)
            self.assertEqual(ckpts["qwen3_14b_inhouse_v2"], q3_v2_p)
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

        # Execute all module cells of compiled training notebook in an isolated namespace
        nb_data = json.loads(trn_out.read_text(encoding="utf-8"))
        ns = {"__name__": "__nb_verify__"}
        for cell in nb_data["cells"][:-1]:
            exec("".join(cell["source"]), ns)
        for req_sym in (
            "run_training_pipeline",
            "apply_prior_and_sinkhorn",
            "sinkhorn_balancing",
            "compute_grouped_inference_logits",
            "evaluate_stage2_oof_cv",
            "run_all_layers_probe_sweep",
            "ExperimentLedger",
        ):
            self.assertIn(req_sym, ns, f"Missing symbol {req_sym} in compiled lora_training.ipynb")

        res_grp = ns["compute_grouped_inference_logits"](
            np.random.randn(4, 16).astype(np.float32),
            np.random.randn(K, 16).astype(np.float32),
            np.zeros(K, dtype=np.float32),
            np.random.randn(K, 16).astype(np.float32),
            np.zeros(K, dtype=np.float32),
        )
        self.assertEqual(res_grp["probs_sinkhorn"].shape, (4, K))

    def test_merge_lora_atomic_rejection_on_mlp_shape_mismatch(self):
        """Verify merge_lora atomically rejects mismatched adapter (e.g. Llama vs Qwen) without partial mutation."""
        from namethatllm.neural_extractor import merge_lora
        from safetensors.torch import save_file

        class DummyTrunk(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        "self_attn": nn.ModuleDict({
                            "q_proj": nn.Linear(64, 64, bias=False),
                        }),
                        "mlp": nn.ModuleDict({
                            "up_proj": nn.Linear(64, 128, bias=False),  # Qwen shape: out=128, in=64
                        })
                    })
                ])

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
            "qwen2",
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128
        )
        base = AutoModel.from_config(cfg)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        clf = MultiLayerLoRAClassifier(
            backbone=base,
            hidden_size=64,
            layers=[1, 3],
            num_classes=18
        )

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
        from namethatllm.config import _inspect_checkpoint_dim_and_keep
        from safetensors.torch import save_file

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

    def test_encoder_stage1_and_trace_ridge_solvers_in_trainer(self):
        """Verify unified encoder Stage-1 helpers and single trace-scaled Ridge eigensolver in trainer.py."""
        from namethatllm.trainer import (
            build_targeted_epoch_indices,
            compute_detailed_metrics,
            gpu_trace_ridge_eval,
            solve_trace_ridge_wb,
        )

        # 1. Quota-balanced sliding epoch indexing
        labels = np.repeat(np.arange(K, dtype=np.int64), 4)
        quotas = {m: 2 for m in LABELS}
        quotas[LABELS[0]] = 6  # exercise oversampling (> n_avail)
        idx_ep0 = build_targeted_epoch_indices(labels, quotas, epoch=0, seed=42)
        idx_ep1 = build_targeted_epoch_indices(labels, quotas, epoch=1, seed=42)
        self.assertEqual(len(idx_ep0), 2 * (K - 1) + 6)
        self.assertFalse(np.array_equal(idx_ep0, idx_ep1))

        # 2. Dual (N < D) and Primal (N >= D) trace-scaled Ridge solver consistency
        rng = np.random.default_rng(7)
        for n_tr, dim in ((36, 64), (80, 16)):
            y_tr = np.tile(np.arange(K, dtype=np.int64), (n_tr // K) + 1)[:n_tr]
            Z_tr = rng.normal(0.0, 0.3, size=(n_tr, dim)).astype(np.float32)
            Z_tr[np.arange(n_tr), y_tr % dim] += 2.5
            wb_map = solve_trace_ridge_wb(Z_tr, y_tr, alphas=[0.25], device=torch.device("cpu"))
            W, b = wb_map[0.25]
            self.assertEqual(W.shape, (K, dim))
            self.assertEqual(b.shape, (K,))
            lg_map = gpu_trace_ridge_eval(Z_tr, y_tr, Z_tr, alphas=[0.25], device=torch.device("cpu"))
            np.testing.assert_allclose(lg_map[0.25], Z_tr @ W.T + b, atol=1e-5)

        # 3. Detailed metrics breakdown
        y_eval = np.repeat(np.arange(K, dtype=np.int64), 3)
        logits_eval = np.zeros((len(y_eval), K), dtype=np.float32)
        logits_eval[np.arange(len(y_eval)), y_eval] = 5.0
        det = compute_detailed_metrics(logits_eval, y_eval)
        self.assertAlmostEqual(det["bacc"], 1.0, places=5)
        self.assertAlmostEqual(det["gem_within_bacc"], 1.0, places=5)
        self.assertEqual(det["zero_recall_classes"], [])

    def test_v2_bugfixes_sister_mask_weights_pool_selection_and_pipeline_e2e(self):
        """Verifies boosted sister/sink mask, match_weight clipping, V2-only pool filter, oversampling, and milestone evals."""
        import pandas as pd
        from namethatllm.config import LAB2I
        from namethatllm.trainer import (
            _build_class_weight_tensor,
            _build_sister_mask,
            _normalize_match_weights,
            _select_stage1_pool_files,
            run_training_pipeline,
        )

        # 1. Upgraded boosted sister & em-dash sink penalties
        M = _build_sister_mask(torch.device("cpu"), boosted=True)
        self.assertGreaterEqual(M[LAB2I["kimi-k3"], LAB2I["glm-5.3-flash"]].item(), 1.8)
        self.assertGreaterEqual(M[LAB2I["glm-5.3-flash"], LAB2I["claude-fable-5"]].item(), 1.6)
        self.assertGreaterEqual(M[LAB2I["gemini-3.8-flash"], LAB2I["gemini-3.7-flash"]].item(), 1.8)
        self.assertGreaterEqual(M[LAB2I["gpt-5.6-sol"], LAB2I["gpt-5.6-luna"]].item(), 1.8)

        # 2. Extreme match_weight clipping + mean=1.0 normalization
        w = _normalize_match_weights(pd.Series([0.02, 1.0, 50.0, None]), weight_clip=(0.5, 2.5))
        self.assertGreaterEqual(float(w.min()), 0.3)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=4)
        self.assertLess(float(w.max()), 3.0)

        # 3. V2 exclusive selection when both v1 and v2 exist
        selected = _select_stage1_pool_files(
            [Path("/in/clean_master_pool_v2.parquet"), Path("/in/clean_master_pool.parquet"), Path("/in/pool_v4.parquet")],
            use_v2_only=True,
        )
        self.assertEqual([p.name for p in selected], ["clean_master_pool_v2.parquet"])

        # 4. Per-class weight mapping (no blind glm-5.3-flash inflation)
        cw = _build_class_weight_tensor(
            list(LABELS),
            dict(LAB2I),
            {"class_weights": {"kimi-k3": 1.20, "glm-5.3-flash": 0.95}},
            default_hard_weight=1.0,
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(float(cw[LAB2I["kimi-k3"]]), 1.20, places=5)
        self.assertAlmostEqual(float(cw[LAB2I["glm-5.3-flash"]]), 0.95, places=5)

        # 5. End-to-end 18-class Stage1+Stage2 pipeline check (V2-only, oversampling, milestones)
        class _SimpleTok:
            pad_token_id = 0
            eos_token_id = 2
            pad_token = "<pad>"
            eos_token = "</s>"

            def __call__(self, text, add_special_tokens=False, truncation=False, **kwargs):
                return {"input_ids": [ord(c) % 80 + 5 for c in str(text)[:24]]}

            def __len__(self):
                return 100

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            in_d = tmp / "input"
            out_d = tmp / "out_ckpt"
            in_d.mkdir(parents=True, exist_ok=True)

            # Put both v1 and v2 pools in input_dir; v1 must be excluded when use_v2_only=True
            pd.DataFrame([{"text": "Old noisy v1 pool row that must be excluded " * 2, "model": "kimi-k3"}]).to_parquet(
                in_d / "clean_master_pool.parquet", index=False
            )
            v2_rows = []
            for m_idx, m_name in enumerate(LABELS):
                n_m = 2 if m_name == "kimi-k3" else 4
                for j in range(n_m):
                    mw = 50.0 if j == 0 else 0.02
                    v2_rows.append(
                        {"text": f"Clean V2 training passage number {j} written by model {m_name} {m_idx} " * 2, "model": m_name, "match_weight": mw}
                    )
            pd.DataFrame(v2_rows).to_parquet(in_d / "clean_master_pool_v2.parquet", index=False)

            bench_rows = [{"text": f"Holdout benchmark validation sample {m_idx} written by {m_name} " * 2, "model": m_name} for m_idx, m_name in enumerate(LABELS)]
            pd.DataFrame(bench_rows).to_parquet(in_d / "clean_benchmark_1800_rows.parquet", index=False)

            train_rows = []
            for p_i in range(2):
                for m_idx, m_name in enumerate(LABELS):
                    train_rows.append({"text": f"Gold train prompt {p_i} response from {m_name} ({m_idx}) " * 2, "model": m_name, "prompt_id": f"p_{p_i}"})
            pd.DataFrame(train_rows).to_csv(in_d / "train.csv", index=False)

            quotas = {m: 2 for m in LABELS}
            quotas["kimi-k3"] = 5  # oversample beyond 2 available rows
            mock_bb = MockHiddenStatesModel(hidden_size=32, num_layers=4)
            mock_bb.config = SimpleNamespace(hidden_size=32, num_hidden_layers=4)
            cfg = {
                "backbone": "mock-qwen",
                "fresh_init": True,
                "input_dir": str(in_d),
                "output_dir": str(out_d),
                "keep_layers": 4,
                "layers": [1, 2, 3],
                "max_len": 32,
                "end_id": 2,
                "lora_r": 4,
                "lora_alpha": 8,
                "family_margin": 0.42,
                "supcon_weight": 0.1,
                "smoke": True,
                "mock_tokenizer": _SimpleTok(),
                "mock_backbone": mock_bb,
                "stage1": {
                    "enabled": True,
                    "use_v2_only": True,
                    "use_match_weight": True,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "epochs": 1,
                    "eval_steps": 2,
                    "eval_subset_size": 18,
                    "save_steps": 5,
                    "class_quotas": quotas,
                    "class_weights": {"kimi-k3": 1.2, "glm-5.3-flash": 0.95},
                },
                "stage2": {
                    "enabled": True,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 1,
                    "epochs": 1,
                    "n_splits": 2,
                    "swa": True,
                },
            }
            res_dir = run_training_pipeline(cfg)
            self.assertEqual(res_dir, out_d)
            audit = json.loads((out_d / "data_audit_report.json").read_text(encoding="utf-8"))
            self.assertEqual(list(audit["source_files"].keys()), ["clean_master_pool_v2.parquet"])
            self.assertTrue((out_d / "milestone_history.json").exists())
            self.assertTrue((out_d / "benchmark_1800_eval.json").exists())
            history = json.loads((out_d / "milestone_history.json").read_text(encoding="utf-8"))
            tags = [h["tag"] for h in history]
            self.assertIn("S1_End_Full", tags)
            self.assertIn("S2_Final_Plain", tags)
            meta = json.loads((out_d / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("bench_1800_plain_bacc", meta)
            self.assertIn("bench_1800_plain_sink_bacc", meta)
            bench_eval = json.loads((out_d / "benchmark_1800_eval.json").read_text(encoding="utf-8"))
            self.assertIn("plain_sinkhorn", bench_eval)
            self.assertIn("blend_global_approx_sinkhorn", bench_eval)

    def test_stage2_bugfixes_centered_ridge_dropout_supcon_and_sister_balance(self):
        """Verifies Stage-2 fixes: centered Ridge intercept, eval Pass-1, self-masked SupCon, balanced sister mask."""
        from namethatllm.config import LAB2I
        from namethatllm.trainer import (
            _build_sister_mask,
            compute_supcon_loss,
            fit_ridge_heads,
            train_prompt_group_step,
        )

        # 1. Centered fit_ridge_heads must learn non-trivial unpenalized intercept even when ||mu_Z||^2 >> variance
        rng = np.random.default_rng(7)
        N, D = 90, 64
        y_arr = np.tile(np.arange(K, dtype=np.int64), N // K)
        mu = rng.normal(0.0, 20.0, size=(1, D)).astype(np.float32)
        Z = mu + rng.normal(0.0, 0.25, size=(N, D)).astype(np.float32)
        for i in range(N):
            Z[i, y_arr[i] % D] += 0.4
        W_p, b_p, W_c, b_c = fit_ridge_heads(Z, y_arr, prompts=None, alpha=10.0, label_smoothing=0.10, num_classes=K)
        self.assertGreater(float(np.abs(b_p).max()), 0.05, "Intercept must not be crushed into W @ mu by uncentered Ridge")
        self.assertTrue(np.all(np.isfinite(W_p)) and np.all(np.isfinite(b_p)))
        self.assertTrue(np.all(np.isfinite(W_c)) and np.all(np.isfinite(b_c)))

        # 2. _build_sister_mask(boosted=True) protects claude-opus-4-6 cleanly without huge asymmetry
        M = _build_sister_mask(torch.device("cpu"), boosted=True)
        opus46 = LAB2I["claude-opus-4-6"]
        self.assertGreaterEqual(float(M[opus46, LAB2I["claude-opus-4-8"]].item()), 1.5)
        self.assertGreaterEqual(float(M[LAB2I["kimi-k3"], opus46].item()), 1.4)
        diff_asym = torch.abs(M.sum(dim=0) - M.sum(dim=1))
        self.assertLessEqual(float(diff_asym.max().item()), 0.8)

        # 3. compute_supcon_loss excludes current micro-batch rows via group_slice without NaN
        z_mb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
        z_grp = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.7]])
        loss_sc = compute_supcon_loss(z_mb, torch.tensor([0, 1]), temperature=0.10, Z_group_centered=z_grp, group_slice=(0, 2))
        self.assertTrue(torch.isfinite(loss_sc))
        loss_sc.backward()
        self.assertIsNotNone(z_mb.grad)

        # 4. train_prompt_group_step runs Pass 1 cleanly in eval mode even when backbone has dropout
        class _DropoutBackbone(MockHiddenStatesModel):
            def __init__(self, hidden_size=16, num_layers=3):
                super().__init__(hidden_size=hidden_size, num_layers=num_layers)
                self.drop = nn.Dropout(0.5)

            def forward(self, input_ids, attention_mask, **kwargs):
                out = super().forward(input_ids, attention_mask, **kwargs)
                return SimpleNamespace(hidden_states=[self.drop(h) for h in out.hidden_states])

        bb = _DropoutBackbone(hidden_size=16, num_layers=3)
        clf = MultiLayerLoRAClassifier(bb, hidden_size=16, layers=[1, 2], num_classes=K)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
        opt.zero_grad()
        tot_l, _, _ = train_prompt_group_step(
            clf,
            opt,
            torch.randint(0, 50, (8, 10)),
            torch.ones(8, 10, dtype=torch.long),
            torch.arange(8) % K,
            micro_batch_size=2,
            supcon_weight=0.10,
            compute_dtype=torch.float32,
        )
        self.assertTrue(math.isfinite(tot_l))
        self.assertTrue(clf.training, "train_prompt_group_step must restore model.train() for Pass 2 and after return")

        # 5. sample_weight in family_margin_cross_entropy normalizes by effective weight sum
        from namethatllm.trainer import family_margin_cross_entropy

        lg_sw = torch.tensor([[2.0] + [0.0] * (K - 1), [0.0, 0.0] + [0.0] * (K - 2)])
        lb_sw = torch.tensor([0, 1])
        uniform_sw = family_margin_cross_entropy(lg_sw, lb_sw, family_margin=0.0, sample_weight=torch.ones(2))
        plain_no_sw = family_margin_cross_entropy(lg_sw, lb_sw, family_margin=0.0, sample_weight=None)
        self.assertAlmostEqual(float(uniform_sw.item()), float(plain_no_sw.item()), places=6)
        weighted_loss = family_margin_cross_entropy(lg_sw, lb_sw, family_margin=0.0, sample_weight=torch.tensor([0.5, 2.5]))
        # Sample 1 has higher loss than Sample 0, so weighting sample 1 by 2.5 increases the normalized average loss
        self.assertGreater(float(weighted_loss.item()), float(plain_no_sw.item()))

    def test_v2_manual_fp16_loss_scaler_and_grad_guard(self):
        """Part 4.1: _ManualFP16LossScaler scales, unscales before isfinite, backs off on inf/nan, and grows cleanly."""
        from namethatllm.trainer import _ManualFP16LossScaler, _step_with_grad_guard

        p = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
        m = nn.Module()
        m.p = p
        opt = torch.optim.SGD([p], lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
        scaler = _ManualFP16LossScaler(enabled=True, scale=256.0, backoff_factor=0.5, growth_factor=2.0, growth_interval=3)

        loss = (m.p ** 2).sum()
        scaled = scaler.scale_loss(loss)
        self.assertAlmostEqual(float(scaled.item()), float(loss.item()) * 256.0, places=4)
        scaled.backward()
        # grad = [2*256, -4*256] -> after unscale + clip_grad_norm(1.0) should modify weights cleanly
        ok = _step_with_grad_guard(m, opt, sched, manual_scaler=scaler, max_grad_norm=100.0)
        self.assertTrue(ok)
        # Expected unscaled grad is [2.0, -4.0], lr=0.1 -> p = [0.8, -1.6]
        self.assertTrue(torch.allclose(m.p.data, torch.tensor([0.8, -1.6]), atol=1e-5))
        self.assertEqual(scaler._growth_tracker, 1)

        # Two more clean steps reach growth_interval=3 -> doubles scale 256 -> 512
        for _ in range(2):
            (m.p ** 2).sum().mul(scaler.scale).backward()
            self.assertTrue(_step_with_grad_guard(m, opt, sched, manual_scaler=scaler, max_grad_norm=100.0))
        self.assertEqual(scaler.scale, 512.0)
        self.assertEqual(scaler._growth_tracker, 0)

        # Inject non-finite gradient: weights must remain exact, scale halves 512 -> 256
        before_weights = m.p.data.clone()
        m.p.grad = torch.tensor([float("inf"), 1.0])
        ok_inf = _step_with_grad_guard(m, opt, sched, manual_scaler=scaler, max_grad_norm=100.0)
        self.assertFalse(ok_inf)
        self.assertTrue(torch.equal(m.p.data, before_weights))
        self.assertEqual(scaler.scale, 256.0)
        self.assertEqual(scaler._growth_tracker, 0)

        # Inject NaN gradient too -> halves again 256 -> 128 without modifying weights
        m.p.grad = torch.tensor([float("nan"), -1.0])
        ok_nan = _step_with_grad_guard(m, opt, sched, manual_scaler=scaler, max_grad_norm=100.0)
        self.assertFalse(ok_nan)
        self.assertTrue(torch.equal(m.p.data, before_weights))
        self.assertEqual(scaler.scale, 128.0)

        # Disabled scaler leaves loss/grads untouched
        dis_scaler = _ManualFP16LossScaler(enabled=False, scale=256.0)
        self.assertEqual(float(dis_scaler.scale_loss(torch.tensor(3.5)).item()), 3.5)

    def test_v2_normalize_match_weights_per_class(self):
        """Part 4.2: _normalize_match_weights(..., groups=models) enforces mean=1.0 +- 1e-5 per class."""
        import pandas as pd
        from namethatllm.trainer import _normalize_match_weights

        models = pd.Series(["qwen3.8-max"] * 6 + ["claude-opus-4-6"] * 6 + ["kimi-k3"] * 4)
        raw_w = pd.Series([2.0, 2.5, 1.5, 2.0, 2.2, 1.8] + [0.6, 1.4, 1.0, 0.8, 1.2, 1.0] + [0.2, 3.5, 0.5, 1.5])
        normed = _normalize_match_weights(raw_w, weight_clip=(0.5, 2.5), groups=models)
        for cls_name in ("qwen3.8-max", "claude-opus-4-6", "kimi-k3"):
            cls_mean = float(normed[models == cls_name].mean())
            self.assertAlmostEqual(cls_mean, 1.0, delta=1e-5)

    def test_v2_family_margin_ce_uniform_sample_weight_equivalence(self):
        """Part 4.3: family_margin_cross_entropy with sample_weight=ones matches None within 1e-6 under non-uniform weight."""
        from namethatllm.trainer import family_margin_cross_entropy

        rng = torch.Generator().manual_seed(123)
        logits = torch.randn((8, K), generator=rng)
        labels = torch.tensor([0, 2, 5, 7, 10, 14, 15, 17], dtype=torch.long)
        class_w = torch.linspace(0.7, 1.5, K)
        loss_none = family_margin_cross_entropy(
            logits, labels, family_margin=0.42, label_smoothing=0.05, weight=class_w, boost_hard_sisters=True, sample_weight=None
        )
        loss_ones = family_margin_cross_entropy(
            logits, labels, family_margin=0.42, label_smoothing=0.05, weight=class_w, boost_hard_sisters=True, sample_weight=torch.ones(8)
        )
        self.assertLess(abs(float(loss_none.item()) - float(loss_ones.item())), 1e-6)

    def test_v2_supcon_negative_cosine_zero_repulsion(self):
        """Part 4.4: Negative cosines (< -0.05) in Z_group_centered contribute 0.0 repulsion when pos_mask.sum() == 0."""
        from namethatllm.trainer import compute_supcon_loss

        z_mb = torch.tensor([[1.0, 0.3, 0.0], [-1.0, 0.3, 0.0]], dtype=torch.float32)
        z_grp = torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 0.1], [0.0, -1.0, -0.1]], dtype=torch.float32)
        loss_neg = compute_supcon_loss(z_mb, torch.tensor([0, 1]), temperature=0.10, Z_group_centered=z_grp)
        self.assertAlmostEqual(float(loss_neg.item()), 0.0, delta=1e-6)

        z_pos_repel = torch.tensor([[1.0, 0.3, 0.0], [-1.0, 0.3, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        loss_pos = compute_supcon_loss(
            z_mb, torch.tensor([0, 1]), temperature=0.10, Z_group_centered=z_pos_repel, group_slice=(0, 2)
        )
        self.assertGreater(float(loss_pos.item()), 0.0)

    def test_v2_train_prompt_group_step_singleton_group(self):
        """Part 4.5: train_prompt_group_step handles N == 1 singleton prompt group with finite grads and zero NaNs."""
        from namethatllm.trainer import train_prompt_group_step

        bb = MockHiddenStatesModel(hidden_size=16, num_layers=3)
        clf = MultiLayerLoRAClassifier(bb, hidden_size=16, layers=[1, 2], num_classes=K)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
        opt.zero_grad()
        tot_l, l_p, l_c = train_prompt_group_step(
            clf,
            opt,
            torch.randint(0, 50, (1, 10)),
            torch.ones(1, 10, dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
            micro_batch_size=2,
            centered_weight=0.50,
            family_margin=0.42,
            supcon_weight=0.12,
            compute_dtype=torch.float32,
            boost_hard_sisters=True,
        )
        self.assertTrue(math.isfinite(tot_l) and math.isfinite(l_p))
        self.assertEqual(l_c, 0.0)
        grads = [p.grad for p in clf.parameters() if p.requires_grad and p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_v2_bucketed_length_sampler_shuffles_identical_lengths(self):
        """Part 4.6: BucketedLengthBatchSampler jitters identical-length sequences so duplicates are not strictly paired."""
        from namethatllm.trainer import BucketedLengthBatchSampler

        lengths = [100] * 40
        sampler = BucketedLengthBatchSampler(lengths, batch_size=2, shuffle=True, seed=42)
        sampler.set_epoch(0)
        batches = list(sampler)
        ordered_pairs = [[2 * i, 2 * i + 1] for i in range(20)]
        self.assertEqual(len(batches), 20)
        # First-half (0..19) and second-half (20..39) identical lengths should mix into the same batches
        mixed_batches = sum(1 for b in batches if (b[0] < 20) != (b[1] < 20))
        self.assertGreater(mixed_batches, 0)
        self.assertNotEqual(sorted(batches), ordered_pairs)

    def test_v2_training_lora_config_36k_contract(self):
        """Part 4.7: training_lora.json is valid JSON with all 18 LABELS and 36,000 exact quota total."""
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "training_lora.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["version"], "3.1.0")
        self.assertEqual(cfg["backbone_family"], "qwen")
        s1 = cfg["stage1"]
        quotas = s1["class_quotas"]
        self.assertEqual(set(quotas.keys()), set(LABELS))
        self.assertEqual(len(quotas), 18)
        self.assertEqual(sum(quotas.values()), 36000)
        self.assertEqual(s1["max_samples"], 36000)
        self.assertAlmostEqual(float(s1["max_hours"]), 7.20, places=4)
        self.assertEqual(s1["save_steps"], 250)
        self.assertEqual(s1["eval_steps"], 375)
        self.assertEqual(s1["eval_batch_size"], 4)
        self.assertEqual(cfg["stage2"]["class_weights"], {})

    def test_v2_run_all_layers_probe_sweep_ast_callsites(self):
        """Part 4.8: AST/signature check confirms every run_all_layers_probe_sweep call in trainer.py matches layer_probing.py."""
        import ast
        import inspect
        from namethatllm.layer_probing import run_all_layers_probe_sweep

        repo_root = Path(__file__).resolve().parent.parent
        trainer_src = (repo_root / "src" / "namethatllm" / "training" / "gates.py").read_text(encoding="utf-8")
        tree = ast.parse(trainer_src)

        sig = inspect.signature(run_all_layers_probe_sweep)
        explicit_params = [
            name
            for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        # Canonical params in layer_probing.py (excluding legacy aliases if any, all explicit params accepted)
        canonical_params = {"features_by_layer", "y", "num_family_classes", "hard_sister_pair", "prompts", "alpha", "n_splits", "output_dir"}

        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                fn_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if fn_name == "run_all_layers_probe_sweep":
                    calls.append(node)

        self.assertEqual(len(calls), 3)
        for call_node in calls:
            kw_names = [kw.arg for kw in call_node.keywords if kw.arg is not None]
            for kw_name in kw_names:
                self.assertIn(kw_name, explicit_params)
                self.assertIn(kw_name, canonical_params, f"Legacy/unknown kwarg {kw_name} used at call site")
            self.assertIn("y", kw_names)
            self.assertIn("features_by_layer", kw_names)

    def test_v2_full_smoke_stage1_stage2_export_and_resume(self):
        """Part 4.9: End-to-end smoke Stage 1 -> Stage 2 -> export -> resume_from_checkpoint reload."""
        import tempfile
        import pandas as pd
        from namethatllm.trainer import run_training_pipeline

        class _SimpleTok:
            pad_token_id = 0
            cls_token_id = 1
            sep_token_id = 2
            eos_token_id = 2
            pad_token = "<pad>"
            eos_token = "</s>"

            def __call__(self, text, **kwargs):
                if isinstance(text, list):
                    return {"input_ids": [[ord(c) % 80 + 10 for c in str(t)[:24]] for t in text]}
                return {"input_ids": [ord(c) % 80 + 10 for c in str(text)[:24]]}

            def __len__(self):
                return 256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "input"
            inp.mkdir()

            train_rows = []
            for p_idx in range(2):
                for m_idx, m_name in enumerate(LABELS):
                    train_rows.append(
                        {
                            "prompt_id": f"p_{p_idx}",
                            "model": m_name,
                            "text": f"Gold resume prompt {p_idx} written distinctively by {m_name} model sample identifier text {m_idx}.",
                        }
                    )
            pd.DataFrame(train_rows).to_csv(inp / "train.csv", index=False)

            bench_rows = [
                {
                    "prompt_id": "bench_0",
                    "model": m_name,
                    "text": f"Holdout bench clean example distinctly for {m_name} architecture index {m_idx} verification.",
                }
                for m_idx, m_name in enumerate(LABELS)
            ]
            pd.DataFrame(bench_rows).to_parquet(inp / "clean_benchmark_1800_rows.parquet", index=False)

            v2_rows = []
            for m_idx, m_name in enumerate(LABELS):
                for r_i in range(2):
                    v2_rows.append(
                        {
                            "text": f"Clean V2 pool synthetic sample {r_i} from {m_name} index {m_idx} sufficiently long text payload.",
                            "model": m_name,
                            "match_weight": 0.8 + 0.4 * r_i,
                            "priority": 1,
                        }
                    )
            pd.DataFrame(v2_rows).to_parquet(inp / "clean_master_pool_v2.parquet", index=False)

            out_stage12 = root / "out_initial"
            quotas = {m: 2 for m in LABELS}
            cfg1 = {
                "pipeline_mode": "full_lora",
                "backbone": "mock-qwen",
                "fresh_init": True,
                "resume_from_checkpoint": None,
                "input_dir": str(inp),
                "output_dir": str(out_stage12),
                "layers": [1, 2],
                "keep_layers": 2,
                "max_len": 32,
                "end_id": 2,
                "lora_r": 4,
                "lora_alpha": 8,
                "family_margin": 0.42,
                "supcon_weight": 0.1,
                "smoke": True,
                "mock_tokenizer": _SimpleTok(),
                "mock_backbone": MockHiddenStatesModel(hidden_size=32, num_layers=3),
                "stage1": {
                    "enabled": True,
                    "use_v2_only": True,
                    "use_match_weight": True,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "epochs": 1,
                    "eval_steps": 2,
                    "eval_subset_size": 18,
                    "save_steps": 4,
                    "class_quotas": quotas,
                },
                "stage2": {
                    "enabled": True,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 1,
                    "epochs": 1,
                    "n_splits": 2,
                    "swa": True,
                },
            }
            res1 = run_training_pipeline(cfg1)
            self.assertEqual(res1, out_stage12)
            self.assertTrue((out_stage12 / "adapter" / "adapter_model.safetensors").exists())
            self.assertTrue((out_stage12 / "heads.safetensors").exists())
            self.assertTrue((out_stage12 / "meta.json").exists())

            # Reload exported checkpoint with resume_from_checkpoint and run Stage 2 only
            out_resumed = root / "out_resumed"
            cfg2 = {
                "pipeline_mode": "full_lora",
                "backbone": "mock-qwen",
                "fresh_init": False,
                "resume_from_checkpoint": str(out_stage12),
                "input_dir": str(inp),
                "output_dir": str(out_resumed),
                "layers": [1, 2],
                "keep_layers": 2,
                "max_len": 32,
                "end_id": 2,
                "lora_r": 4,
                "lora_alpha": 8,
                "family_margin": 0.42,
                "supcon_weight": 0.1,
                "smoke": True,
                "mock_tokenizer": _SimpleTok(),
                "mock_backbone": MockHiddenStatesModel(hidden_size=32, num_layers=3),
                "stage1": {"enabled": False},
                "stage2": {
                    "enabled": True,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 1,
                    "epochs": 1,
                    "n_splits": 2,
                    "swa": False,
                    "warmstart_centered_from_plain": False,
                },
            }
            res2 = run_training_pipeline(cfg2)
            self.assertEqual(res2, out_resumed)
            self.assertTrue((out_resumed / "adapter" / "adapter_model.safetensors").exists())
            self.assertTrue((out_resumed / "heads.safetensors").exists())
            self.assertTrue((out_resumed / "meta.json").exists())

            # Fix 4: Fail fast when heads.safetensors cannot be loaded compatibly on resume
            from safetensors.torch import save_file as _save_st

            bad_ckpt = root / "bad_ckpt"
            bad_ckpt.mkdir(parents=True)
            (bad_ckpt / "adapter").mkdir()
            import shutil

            shutil.copy(out_stage12 / "adapter" / "adapter_model.safetensors", bad_ckpt / "adapter" / "adapter_model.safetensors")
            shutil.copy(out_stage12 / "adapter" / "adapter_config.json", bad_ckpt / "adapter" / "adapter_config.json")
            _save_st({"plain.weight": torch.zeros((2, 4), dtype=torch.float32)}, str(bad_ckpt / "heads.safetensors"))
            cfg_bad = dict(cfg2, resume_from_checkpoint=str(bad_ckpt), output_dir=str(root / "out_bad"))
            with self.assertRaises(RuntimeError):
                run_training_pipeline(cfg_bad)

    def test_v2_extract_features_preserves_cls_with_drop_first_token(self):
        """Fix 5: extract_features_and_logits passes cls_token_id even when drop_first_token=True."""
        from namethatllm.trainer import extract_features_and_logits

        seen_first_tokens: list[int] = []

        class _EncTok:
            pad_token_id = 0
            cls_token_id = 101
            sep_token_id = 102
            eos_token_id = 102

            def __call__(self, text, **kwargs):
                return {"input_ids": [200, 201, 202]}

        class _SpyClassifier(nn.Module):
            total_dim = 8
            num_classes = K

            def forward_features(self, input_ids, attention_mask, drop_first_token=True):
                seen_first_tokens.extend(input_ids[:, 0].tolist())
                return torch.ones((input_ids.shape[0], 8), dtype=torch.float32)

            def predict_plain(self, Z):
                return torch.zeros((Z.shape[0], K), dtype=torch.float32)

        extract_features_and_logits(
            _SpyClassifier(),
            ["sample encoder text"],
            _EncTok(),
            max_len=16,
            device=torch.device("cpu"),
            batch_size=2,
            drop_first_token=True,
            compute_dtype=torch.float32,
        )
        self.assertEqual(seen_first_tokens, [101])


if __name__ == "__main__":
    unittest.main()

