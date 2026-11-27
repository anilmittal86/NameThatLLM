"""
Verification Suite for Modular Backbone Adapters (`qwen`, `deberta`, `bge`),
Clean-Code AST Architecture Guards, and Thin Trainer Facade Parity.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig

from namethatllm.config import K, LABELS
from namethatllm.training.backbones import (
    BgeAdapter,
    DebertaAdapter,
    PrecisionSpec,
    QwenAdapter,
    resolve_backbone_adapter,
)
from namethatllm.training.embeddings import extract_and_export_embeddings
from namethatllm.training.modeling import MultiLayerLoRAClassifier


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 4, model_type: str = "qwen3"):
        super().__init__()
        self.config = PretrainedConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, model_type=model_type)
        self.embed = nn.Embedding(256, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed(input_ids)
        states = [h]
        for layer in self.layers:
            h = layer(h)
            states.append(h)
        h_final = self.norm(h)
        states[-1] = h_final
        return SimpleNamespace(hidden_states=states, last_hidden_state=h_final)


def test_registry_and_bge_stage1_stage2_routing():
    """Adapters resolve explicitly & cleanly; BGE with both stage1+stage2 stays on full Stage1->Stage2 flow."""
    assert isinstance(resolve_backbone_adapter({"backbone_family": "qwen"}), QwenAdapter)
    assert isinstance(resolve_backbone_adapter({"backbone_family": "deberta"}), DebertaAdapter)
    bge = resolve_backbone_adapter({"backbone_family": "bge", "backbone": "BAAI/bge-m3"})
    assert isinstance(bge, BgeAdapter)

    # Fallback inference when backbone_family is omitted
    assert isinstance(resolve_backbone_adapter({"backbone": "Qwen/Qwen3-14B-Base"}), QwenAdapter)
    assert isinstance(resolve_backbone_adapter({"backbone": "microsoft/deberta-v3-large"}), DebertaAdapter)
    assert isinstance(resolve_backbone_adapter({"backbone": "BAAI/bge-m3"}), BgeAdapter)

    # A BGE config containing BOTH stage1 and stage2 must NOT silently divert into run_encoder_stage1
    cfg_two_stage = {"backbone_family": "bge", "backbone": "BAAI/bge-m3", "stage1": {"enabled": True}, "stage2": {"enabled": True}}
    resolved = resolve_backbone_adapter(cfg_two_stage)
    assert isinstance(resolved, BgeAdapter)
    assert resolved.maybe_run_pipeline(cfg_two_stage) is None


def test_1024_dim_disambiguation_between_bge_and_deberta(tmp_path: Path):
    """1024-dim BgeAdapter drops <s> with cls_id != None & drop_first_token=True; 1024-dim DebertaAdapter keeps drop_first_token=False."""
    class _EncoderTok:
        pad_token_id = 0
        cls_token_id = 101
        sep_token_id = 102
        eos_token_id = 102

        def __call__(self, text, **kwargs):
            return {"input_ids": [200, 201, 202]}

    class _SpyModel(MultiLayerLoRAClassifier):
        def __init__(self, tok_spec, fam: str):
            bb = _DummyBackbone(hidden_size=1024, num_layers=3, model_type="custom")
            super().__init__(bb, hidden_size=1024, layers=[1, 2], num_classes=K, backbone_family=fam, tok_spec=tok_spec)
            self.seen_drop_flags: list[bool] = []
            self.seen_first_toks: list[int] = []

        def forward_features(self, input_ids, attention_mask, drop_first_token=True, **kwargs):
            self.seen_drop_flags.append(bool(drop_first_token))
            self.seen_first_toks.extend(input_ids[:, 0].tolist())
            bsz = input_ids.shape[0]
            z_cat = torch.ones((bsz, len(self.layers) * 1024), dtype=torch.float32) + torch.randn(bsz, len(self.layers) * 1024) * 0.1
            z_layers = torch.randn((bsz, len(self.layers), 1024), dtype=torch.float32)
            return (z_cat, z_layers) if kwargs.get("return_by_layer") else z_cat

    tok = _EncoderTok()
    df_pool = pd.DataFrame([{"text": f"Long synthetic sentence number {i} for extraction verification.", "model": LABELS[i % K]} for i in range(6)])
    df_tr = df_pool.copy()

    _, bge_spec = BgeAdapter().load_tokenizer("BAAI/bge-m3", {"mock_tokenizer": tok, "max_len": 32})
    assert bge_spec.cls_id == 101
    assert bge_spec.drop_first_token is True

    _, deb_spec = DebertaAdapter().load_tokenizer("microsoft/deberta-v3-large", {"mock_tokenizer": tok, "max_len": 32})
    assert deb_spec.cls_id == 101
    assert deb_spec.drop_first_token is False

    m_bge = _SpyModel(bge_spec, "bge")
    extract_and_export_embeddings(m_bge, tok, df_pool, df_tr, target="gold", output_dir=tmp_path / "bge_emb", batch_size=4, device=torch.device("cpu"))
    assert all(flag is True for flag in m_bge.seen_drop_flags)
    assert all(tid == 101 for tid in m_bge.seen_first_toks)

    m_deb = _SpyModel(deb_spec, "deberta")
    extract_and_export_embeddings(m_deb, tok, df_pool, df_tr, target="gold", output_dir=tmp_path / "deb_emb", batch_size=4, device=torch.device("cpu"))
    assert all(flag is False for flag in m_deb.seen_drop_flags)
    assert all(tid == 101 for tid in m_deb.seen_first_toks)


def test_bge_export_context_restores_layers_even_on_exception():
    """BgeAdapter.export_context restores full encoder layer count both normally and when an exception is raised."""
    class _FakeEncoder(nn.Module):
        def __init__(self, n_layers: int = 24):
            super().__init__()
            self.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(n_layers)])

    class _FakeBge(nn.Module):
        def __init__(self, n_layers: int = 24):
            super().__init__()
            self.encoder = _FakeEncoder(n_layers=n_layers)

    adapter = BgeAdapter()
    bb = _FakeBge(24)
    prec = PrecisionSpec(compute_dtype=torch.float32, use_bf16=False, use_fa2=False, total_vram_gb=16.0)
    adapter.prepare_backbone(bb, prec, {"architecture": {"tap_layers": [4, 7, 10]}}, is_eval_only=False)
    assert len(bb.encoder.layer) == 10

    with adapter.export_context(bb):
        assert len(bb.encoder.layer) == 24
    assert len(bb.encoder.layer) == 10

    with pytest.raises(RuntimeError, match="boom"):
        with adapter.export_context(bb):
            assert len(bb.encoder.layer) == 24
            raise RuntimeError("boom")
    assert len(bb.encoder.layer) == 10


def test_qwen_adapter_t4_safety_memory_eval_bs_and_fp32_lora():
    """QwenAdapter enforces {0: '10.0GiB', 1: '13.5GiB'}, eval_batch_size=4 under 24 GiB, and FP32 LoRA params."""
    adapter = QwenAdapter()
    prec_t4 = PrecisionSpec(compute_dtype=torch.float16, use_bf16=False, use_fa2=False, total_vram_gb=15.0)
    with patch("torch.cuda.is_available", return_value=True):
        bs1, accum1, eval_bs = adapter.adjust_stage1_batch_config({"batch_size": 32, "gradient_accumulation_steps": 1}, prec_t4)
    assert bs1 == 2
    assert accum1 == 16
    assert eval_bs == 4

    # 2xT4 max_memory verification via mocked AutoConfig + AutoModel
    captured_kwargs: dict = {}

    def _fake_from_pretrained(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _DummyBackbone(hidden_size=32, num_layers=27)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=15 * 1024**3)),
        patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(num_hidden_layers=27, hidden_size=32, use_cache=True)),
        patch("transformers.AutoModel.from_pretrained", side_effect=_fake_from_pretrained),
    ):
        adapter.load_backbone("Qwen/Qwen3-14B-Base", {}, prec_t4, torch.device("cpu"))
    assert captured_kwargs.get("max_memory") == {0: "10.0GiB", 1: "13.5GiB"}
    assert captured_kwargs.get("device_map") == "auto"

    # FP16 base weights wrapped by apply_lora must produce FP32 trainable LoRA weights
    half_bb = _DummyBackbone(hidden_size=16, num_layers=3).half()
    peft_m = adapter.apply_lora(half_bb, {"lora_r": 4, "lora_alpha": 8, "lora_target_modules": ["0"]}, is_eval_only=False)
    trainable = [p for p in peft_m.parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all(p.dtype == torch.float32 for p in trainable)


def test_resume_from_checkpoint_fail_fast_on_incompatible_heads():
    """run_training_pipeline raises RuntimeError immediately when resume heads.safetensors is incompatible."""
    from namethatllm.trainer import run_training_pipeline

    class _SimpleTok:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            return {"input_ids": [10, 11, 12, 13]}

        def __len__(self):
            return 256

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inp = root / "input"
        inp.mkdir()
        pd.DataFrame([{"prompt_id": "p0", "model": m, "text": f"Gold train text {m}"} for m in LABELS]).to_csv(inp / "train.csv", index=False)

        # Create valid initial checkpoint first, then corrupt only heads.safetensors
        bb_init = _DummyBackbone(hidden_size=32, num_layers=4)
        from peft import LoraConfig, get_peft_model

        ckpt_dir = root / "bad_resume_ckpt"
        ckpt_dir.mkdir()
        peft_init = get_peft_model(bb_init, LoraConfig(r=4, lora_alpha=8, target_modules=["0"], bias="none", task_type="FEATURE_EXTRACTION"))
        peft_init.save_pretrained(str(ckpt_dir / "adapter"))
        save_file({"plain.weight": torch.zeros((2, 4), dtype=torch.float32)}, str(ckpt_dir / "heads.safetensors"))

        cfg = {
            "backbone_family": "qwen",
            "backbone": "mock-qwen",
            "fresh_init": False,
            "resume_from_checkpoint": str(ckpt_dir),
            "input_dir": str(inp),
            "output_dir": str(root / "out"),
            "layers": [1, 2],
            "keep_layers": 2,
            "max_len": 16,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["0"],
            "eval_benchmark": False,
            "smoke": True,
            "mock_tokenizer": _SimpleTok(),
            "mock_backbone": _DummyBackbone(hidden_size=32, num_layers=4),
            "stage1": {"enabled": False},
            "stage2": {"enabled": False},
        }
        with pytest.raises(RuntimeError, match="Failed to load compatible heads"):
            run_training_pipeline(cfg)


def test_clean_code_ast_guard_no_backbone_heuristics_or_inline_stdlib():
    """AST guard ensures shared training modules contain zero model-name heuristics, 1024 checks, or inline stdlib imports."""
    pkg_dir = Path(__file__).resolve().parent.parent / "src" / "namethatllm" / "training"
    guarded_files = ("engine.py", "data.py", "losses.py", "evaluation.py", "embeddings.py", "gates.py")
    forbidden_tokens = ("deberta", "qwen", "bge")
    stdlib_names = {"copy", "functools", "gc", "hashlib", "json", "math", "os", "pathlib", "shutil", "time", "types"}

    for fname in guarded_files:
        src = (pkg_dir / fname).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=fname)

        # 1. Zero hardcoded model-family strings in AST constants & identifiers
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val_l = node.value.lower()
                for tok in forbidden_tokens:
                    assert tok not in val_l, f"{fname}:{node.lineno} contains forbidden string constant '{node.value}'"
            if isinstance(node, ast.Compare):
                ops_vals = [c.value for c in node.comparators if isinstance(c, ast.Constant)]
                if 1024 in ops_vals:
                    left_src = ast.unparse(node.left)
                    assert "hidden_size" not in left_src, f"{fname}:{node.lineno} contains hidden_size == 1024 heuristic"

        # 2. Zero inline stdlib re-imports inside function bodies
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            assert alias.name.split(".")[0] not in stdlib_names, f"{fname}:{child.lineno} has inline stdlib import {alias.name}"
                    elif isinstance(child, ast.ImportFrom) and child.module:
                        assert child.module.split(".")[0] not in stdlib_names, f"{fname}:{child.lineno} has inline stdlib from-import {child.module}"


def test_facade_completeness_and_smoke_across_all_three_adapters(tmp_path: Path):
    """Every legacy symbol imports from namethatllm.trainer and runs mock smoke across qwen, deberta, and bge."""
    import namethatllm.trainer as trainer_facade

    expected_symbols = [
        "MultiLayerLoRAClassifier", "compute_two_stage_loss", "PromptGroupBatchSampler",
        "export_compatible_checkpoint", "run_training_pipeline", "collate_text_batch",
        "_disable_final_norm", "_stabilize_fp16_backbone", "train_prompt_group_step",
        "family_margin_cross_entropy", "_build_sister_mask", "BucketedLengthBatchSampler",
        "compute_supcon_loss", "build_targeted_epoch_indices", "compute_detailed_metrics",
        "gpu_trace_ridge_eval", "solve_trace_ridge_wb", "_build_class_weight_tensor",
        "_normalize_match_weights", "_select_stage1_pool_files", "fit_ridge_heads",
        "_ManualFP16LossScaler", "_step_with_grad_guard", "extract_features_and_logits",
        "TextSequenceDataset", "compute_grouped_inference_logits", "_blend_head",
        "evaluate_stage2_oof_cv", "run_encoder_stage1", "run_bge_v2_stage1",
        "package_encoder_ridge_checkpoint", "train_family_gate", "train_family_gate_from_dataset",
        "FamilyMLPHead", "fit_family_mlp_head", "fit_family_gate_head", "run_sweep",
        "save_extracted_features_bundle", "extract_and_export_embeddings",
        "load_zero_leak_stage1_pool", "load_clean_stage1_pool", "build_supervised_family_subspace",
        "run_deberta_specialist_suite", "norm_text_hash", "PROBLEM_WATCH_CLASSES",
    ]
    for sym in expected_symbols:
        assert hasattr(trainer_facade, sym), f"Missing facade symbol: {sym}"

    # Verify trainer.py stays strictly under 120 lines
    facade_path = Path(__file__).resolve().parent.parent / "src" / "namethatllm" / "trainer.py"
    assert len(facade_path.read_text(encoding="utf-8").splitlines()) < 120

    class _SmokeTok:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            if isinstance(text, list):
                return {"input_ids": [[ord(c) % 80 + 10 for c in str(t)[:20]] for t in text]}
            return {"input_ids": [ord(c) % 80 + 10 for c in str(text)[:20]]}

        def __len__(self):
            return 256

    inp = tmp_path / "input"
    inp.mkdir()
    pd.DataFrame([{"prompt_id": f"p{r}", "model": m, "text": f"Smoke prompt {r} sample by {m} text payload."} for r in range(2) for m in LABELS]).to_csv(inp / "train.csv", index=False)
    pd.DataFrame([{"text": f"Smoke pool sample {r} by {m} sufficiently long sentence text payload.", "model": m, "match_weight": 1.0, "priority": 1} for r in range(2) for m in LABELS]).to_parquet(inp / "clean_master_pool_v2.parquet", index=False)
    pd.DataFrame([{"model": m, "text": f"Bench clean row {i} by {m} verification payload text."} for i, m in enumerate(LABELS)]).to_parquet(inp / "clean_benchmark_1800_rows.parquet", index=False)

    quotas = {m: 2 for m in LABELS}
    for fam in ("qwen", "deberta", "bge"):
        out_f = tmp_path / f"out_{fam}"
        res = trainer_facade.run_training_pipeline({
            "pipeline_mode": "full_lora",
            "backbone_family": fam,
            "backbone": f"mock-{fam}",
            "fresh_init": True,
            "input_dir": str(inp),
            "output_dir": str(out_f),
            "layers": [1, 2],
            "keep_layers": 2,
            "max_len": 24,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["0"],
            "smoke": True,
            "mock_tokenizer": _SmokeTok(),
            "mock_backbone": _DummyBackbone(hidden_size=32, num_layers=4, model_type=fam),
            "stage1": {"enabled": True, "batch_size": 4, "gradient_accumulation_steps": 1, "epochs": 1, "eval_steps": 4, "eval_subset_size": 18, "save_steps": 4, "class_quotas": quotas},
            "stage2": {"enabled": True, "batch_size": 4, "gradient_accumulation_steps": 1, "epochs": 1, "n_splits": 2, "swa": False},
        })
        assert (Path(res) / "heads.safetensors").exists()
        assert (Path(res) / "meta.json").exists()


def test_fit_family_gate_head_temp_scale_and_groupkfold_guard():
    """Verify Ridge gate calibrates intra-family logits to row_std ≈ 3.2 and blocks GroupKFold '-1' collapse."""
    from namethatllm.training.gates import fit_family_gate_head

    rng = np.random.RandomState(7)
    N = 100
    D = 16
    K_fam = 4
    Z = rng.randn(N, D).astype(np.float32)
    y = np.array([i % K_fam for i in range(N)], dtype=np.int64)
    for c in range(K_fam):
        Z[y == c, c] += 1.8

    best_w, best_b, *_ = fit_family_gate_head(
        Z_train=Z,
        y_train=y,
        num_gate_classes=K_fam,
        num_family_classes=K_fam,
        device=torch.device("cpu"),
        prompt_ids=None,
    )
    train_logits = (Z @ best_w + best_b)[:, :K_fam]
    row_std = float(np.std(train_logits, axis=1).mean())
    assert abs(row_std - 3.2) < 1e-3

    # 90% '-1' group + 10 other groups: unique_groups=11 >= 10, but max_group_frac=0.90 >= 0.05 -> must use StratifiedKFold
    collapsed_pids = ["-1"] * 90 + [f"g_{i}" for i in range(10)]
    with patch("namethatllm.training.gates.GroupKFold") as mock_gkf:
        fit_family_gate_head(
            Z_train=Z,
            y_train=y,
            num_gate_classes=K_fam,
            num_family_classes=K_fam,
            device=torch.device("cpu"),
            prompt_ids=collapsed_pids,
        )
        mock_gkf.assert_not_called()

    # Balanced groups (max_group_frac < 0.05, unique_groups >= 10) -> must use GroupKFold
    balanced_pids = [f"g_{i // 4}" for i in range(N)]
    with patch("namethatllm.training.gates.GroupKFold", wraps=__import__("sklearn.model_selection", fromlist=["GroupKFold"]).GroupKFold) as spy_gkf:
        fit_family_gate_head(
            Z_train=Z,
            y_train=y,
            num_gate_classes=K_fam,
            num_family_classes=K_fam,
            device=torch.device("cpu"),
            prompt_ids=balanced_pids,
        )
        spy_gkf.assert_called()


def test_prompt_group_batch_sampler_shuffles_within_groups():
    """Verify PromptGroupBatchSampler shuffles sample indices inside each prompt group every epoch."""
    from namethatllm.training.data import PromptGroupBatchSampler

    prompt_ids = ["p1"] * 12 + ["p2"] * 12
    sampler = PromptGroupBatchSampler(prompt_ids, batch_size=2, shuffle=True, seed=42)

    sampler.set_epoch(0)
    batches_ep0 = [list(b) for b in sampler]
    sampler.set_epoch(1)
    batches_ep1 = [list(b) for b in sampler]

    def _by_group(batches):
        out = {}
        for b in batches:
            key = prompt_ids[b[0]]
            out[key] = b
        return out

    g0 = _by_group(batches_ep0)
    g1 = _by_group(batches_ep1)
    assert sorted(g0["p1"]) == list(range(12))
    assert sorted(g1["p1"]) == list(range(12))
    assert g0["p1"] != list(range(12))
    assert g0["p1"] != g1["p1"]


def test_compute_two_stage_loss_legacy_3_tuple_api():
    """Verify compute_two_stage_loss preserves legacy `(model, Z, labels, prompt_ids=...) -> (total, plain, centered)` API."""
    from namethatllm.training.losses import compute_two_stage_loss
    from namethatllm.training.modeling import MultiLayerLoRAClassifier

    bb = _DummyBackbone(hidden_size=16, num_layers=3)
    model = MultiLayerLoRAClassifier(backbone=bb, hidden_size=16, layers=[1, 2], num_classes=4)
    Z = torch.randn(6, 32, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
    pids = ["p1", "p1", "p1", "p2", "p2", "p3"]

    total, l_plain, l_centered = compute_two_stage_loss(
        model, Z, labels, prompt_ids=pids, centered_weight=0.5, supcon_weight=0.1
    )
    assert isinstance(total, torch.Tensor)
    assert torch.isfinite(total) and torch.isfinite(l_plain) and l_centered is not None and torch.isfinite(l_centered)
    total.backward()
    assert model.plain_head.weight.grad is not None
    assert model.centered_head.weight.grad is not None

    tot_no_c, plain_no_c, cent_none = compute_two_stage_loss(model, Z.detach(), labels, prompt_ids=None)
    assert torch.isfinite(tot_no_c) and torch.isfinite(plain_no_c)
    assert cent_none is None


def test_find_compatible_champion_lora_rejects_qwen_for_bge(tmp_path: Path):
    """Verify find_compatible_champion_lora excludes Qwen adapters when resolving for BAAI/bge-m3 and handles token ID 0."""
    import json
    from safetensors.torch import save_file as st_save
    from namethatllm.training.backbones import resolve_backbone_adapter
    from namethatllm.training.modeling import find_compatible_champion_lora

    qwen_dir = tmp_path / "export_champion_qwen3_14b" / "adapter"
    qwen_dir.mkdir(parents=True)
    st_save({"w": torch.zeros(2)}, str(qwen_dir / "adapter_model.safetensors"))
    (qwen_dir / "adapter_config.json").write_text(
        json.dumps({"r": 32, "base_model_name_or_path": "Qwen/Qwen3-14B-Base", "target_modules": ["q_proj", "v_proj"]}),
        encoding="utf-8",
    )

    assert find_compatible_champion_lora(tmp_path, backbone_name="BAAI/bge-m3", lora_r=32) is None
    assert find_compatible_champion_lora(tmp_path, backbone_name="Qwen/Qwen3-14B-Base", lora_r=32) == qwen_dir

    bge_dir = tmp_path / "export_bge_m3_v2" / "adapter"
    bge_dir.mkdir(parents=True)
    st_save({"w": torch.zeros(2)}, str(bge_dir / "adapter_model.safetensors"))
    (bge_dir / "adapter_config.json").write_text(
        json.dumps({"r": 32, "base_model_name_or_path": "BAAI/bge-m3", "target_modules": ["query", "key"]}),
        encoding="utf-8",
    )
    assert find_compatible_champion_lora(tmp_path, backbone_name="BAAI/bge-m3", lora_r=32) == bge_dir
    assert find_compatible_champion_lora(tmp_path, backbone_name="Qwen/Qwen3-14B-Base", lora_r=32) == qwen_dir

    # Verify token ID 0 is not treated as falsy when resolving sep/eos/end_id
    class _ZeroTok:
        pad_token_id = 0
        cls_token_id = 0
        sep_token_id = 0
        eos_token_id = 2

    bge_adapter = resolve_backbone_adapter({"backbone_family": "bge"})
    spec_bge = bge_adapter.tok_spec_from_tokenizer(_ZeroTok(), max_len=128, end_id=0)
    assert spec_bge.cls_id == 0 and spec_bge.drop_first_token is True and spec_bge.end_id == 0

    deb_adapter = resolve_backbone_adapter({"backbone_family": "deberta"})
    spec_deb = deb_adapter.tok_spec_from_tokenizer(_ZeroTok(), max_len=128)
    assert spec_deb.end_id == 0 and spec_deb.pad_id == 0
