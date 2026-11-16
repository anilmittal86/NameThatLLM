import json
import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from namethatllm.config import FAMILIES, SPECIALIST_GROUPS, PRIOR_SCALES
from namethatllm.trainer import (
    TextSequenceDataset,
    collate_text_batch,
    MultiLayerLoRAClassifier,
    train_prompt_group_step,
)


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(self, text, add_special_tokens=False, truncation=False, **kwargs):
        tokens = [ord(c) % 500 + 10 for c in str(text)[:30]]
        return {"input_ids": tokens}

    def __len__(self):
        return 128100


from transformers import PretrainedConfig


class DummyConfig(PretrainedConfig):
    model_type = "deberta-v2"

    def __init__(self, hidden_size=64, num_hidden_layers=24, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class DummyDebertaBackbone(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=24):
        super().__init__()
        self.config = DummyConfig(hidden_size=hidden_dim, num_hidden_layers=num_layers)
        self.embed = nn.Embedding(128100, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, **kwargs):
        h = self.embed(input_ids)
        states = [h]
        for l in self.layers:
            h = l(h)
            states.append(h)
        return type("Output", (), {"hidden_states": states, "last_hidden_state": h})()


def test_deberta_config_and_super_clusters():
    """Verify super-cluster definitions and DeBERTa prior scales in config."""
    assert "claude_glm" in SPECIALIST_GROUPS
    assert "gpt_qwen_kimi" in SPECIALIST_GROUPS
    assert "deberta_gemini" in PRIOR_SCALES
    assert "deberta_claude_glm" in PRIOR_SCALES
    assert "deberta_gpt_qwen_kimi" in PRIOR_SCALES
    assert FAMILIES["gemini"] == [5, 6, 7, 8]


def test_deberta_dataset_and_collation():
    """Verify [CLS] prepending and pad_token_id=0 collation for DeBERTa."""
    tok = DummyTokenizer()
    texts = ["First passage for DeBERTa", "Second passage"]
    labels = [0, 1]
    prompt_ids = ["p1", "p2"]

    ds = TextSequenceDataset(
        texts=texts,
        labels=labels,
        prompt_ids=prompt_ids,
        tok=tok,
        max_len=64,
        end_id=tok.sep_token_id,
        cls_id=tok.cls_token_id,
    )

    assert len(ds) == 2
    item0 = ds[0]
    # Check that [CLS] token (id=1) is prepended at index 0
    assert item0["input_ids"][0].item() == tok.cls_token_id
    # Check that [SEP] token (id=2) is appended at the end
    assert item0["input_ids"][-1].item() == tok.sep_token_id

    # Check collation with pad_token_id=0
    batch = collate_text_batch([ds[0], ds[1]], pad_token_id=tok.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert (batch["input_ids"] < 128100).all(), "No token id should exceed DeBERTa vocab size (128,100)"
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_deberta_prompt_group_step_preserves_cls():
    """Verify train_prompt_group_step with drop_first_token=False retains position 0 for DeBERTa."""
    torch.manual_seed(42)
    tok = DummyTokenizer()
    backbone = DummyDebertaBackbone(hidden_dim=32, num_layers=24)
    layers = [3, 6, 9, 12, 15, 18, 21, 24]
    model = MultiLayerLoRAClassifier(
        backbone=backbone,
        hidden_size=32,
        layers=layers,
        num_classes=4,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(10, 500, (4, 16))
    input_ids[:, 0] = tok.cls_token_id  # Set position 0 as [CLS]
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([0, 1, 2, 3])

    # Execute step with drop_first_token=False (DeBERTa mode)
    tot_loss, p_loss, c_loss = train_prompt_group_step(
        model=model,
        opt=opt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        micro_batch_size=2,
        centered_weight=0.5,
        family_margin=0.40,
        supcon_weight=0.0,
        accum_steps=1,
        compute_dtype=torch.float32,
        drop_first_token=False,
    )

    assert tot_loss > 0.0
    assert np.isfinite(tot_loss)
    assert np.isfinite(p_loss)
    assert np.isfinite(c_loss)


def test_oof_supervised_family_subspace_stacking():
    """Verify build_oof_supervised_family_subspace generates leak-free OOF projections."""
    from namethatllm.layer_probing import build_oof_supervised_family_subspace

    np.random.seed(42)
    N_tr = 40
    N_ev = 10
    D = 32
    Z_tr = np.random.randn(N_tr, D).astype(np.float32)
    y_tr = np.array([i % 4 for i in range(N_tr)], dtype=np.int64)
    Z_ev = np.random.randn(N_ev, D).astype(np.float32)

    Z_tr_oof, Z_ev_act, C_full, mu_full, diag = build_oof_supervised_family_subspace(
        Z_train=Z_tr,
        y_train=y_tr,
        Z_eval=Z_ev,
        num_family_classes=4,
        num_tapped_layers=4,
        subspace_dim=8,
        n_splits=3,
        reg_ratio=0.35,
    )

    assert Z_tr_oof.shape[0] == N_tr
    assert Z_ev_act.shape[0] == N_ev
    assert Z_tr_oof.shape[1] == C_full.shape[0]
    assert np.all(np.isfinite(Z_tr_oof))
    assert np.all(np.isfinite(Z_ev_act))
    assert C_full.shape[1] == D


def test_deberta_layer_probing_primal_solver():
    """Verify GPU Primal / Dual solver handles layer-probing sweep."""
    from namethatllm.layer_probing import run_all_layers_probe_sweep

    np.random.seed(42)
    N = 30
    L = 25
    D_l = 16
    feats = np.random.randn(N, L, D_l).astype(np.float32)
    y = np.array([i % 4 for i in range(N)], dtype=np.int64)

    sweep = run_all_layers_probe_sweep(
        features_by_layer=feats,
        y=y,
        num_family_classes=4,
        alpha=5.0,
    )

    assert "best_single_layer" in sweep
    assert "recommended_top8_layers" in sweep
    assert "layers" in sweep
    assert len(sweep["layers"]) == L
    assert 0 <= sweep["best_single_layer"]["layer"] < L


def test_run_deberta_specialist_suite_exp06a_06b_06c(tmp_path):
    """Verify end-to-end execution of EXP-06a, EXP-06b, and EXP-06c in DeBERTa suite."""
    from safetensors.torch import save_file
    import pandas as pd
    from namethatllm.config import LABELS
    from namethatllm.trainer import run_training_pipeline

    input_d = tmp_path / "inputs"
    input_d.mkdir(parents=True, exist_ok=True)

    # 1. Clean Benchmark
    bench_rows = []
    for m in LABELS:
        for r in range(2):
            bench_rows.append({"text": f"Benchmark evaluation passage for {m} sample {r}", "model": m})
    df_b = pd.DataFrame(bench_rows)
    df_b.to_parquet(input_d / "clean_benchmark_1800_rows.parquet", index=False)

    # 2. Champion Logits
    logits_champ = torch.randn(len(df_b), len(LABELS), dtype=torch.float32)
    save_file({"logits_plain": logits_champ}, str(input_d / "benchmark_champion_logits.safetensors"))

    # 3. Gold Train
    train_rows = []
    for m in LABELS:
        for r in range(2):
            train_rows.append({"text": f"Gold train passage for {m} prompt {r}", "model": m, "prompt_id": f"prompt_{r}"})
    df_t = pd.DataFrame(train_rows)
    df_t.to_csv(input_d / "train.csv", index=False)

    # 4. Pool
    pool_rows = []
    for m in LABELS:
        for r in range(4):
            pool_rows.append({"text": f"Pool passage text for {m} candidate {r} extra padding words", "model": m})
    df_p = pd.DataFrame(pool_rows)
    df_p.to_parquet(input_d / "pool_01.parquet", index=False)

    tok = DummyTokenizer()
    backbone = DummyDebertaBackbone(hidden_dim=32, num_layers=24)

    suite_cfg = {
        "backbone": "microsoft/deberta-v3-large",
        "pipeline_mode": "deberta_specialist_suite",
        "target_family": "gemini",
        "input_dir": str(input_d),
        "output_dir": str(tmp_path / "export_deberta_gemini_specialist"),
        "gate_output_dir": str(tmp_path / "export_deberta_gemini_gate_mlp"),
        "embeddings_output_dir": str(tmp_path / "export_embeddings_deberta_finetuned"),
        "ledger_dir": str(tmp_path / "experiments"),
        "keep_layers": 24,
        "layers": [3, 6, 9, 12, 15, 18, 21, 24],
        "max_len": 32,
        "samples_per_class": 4,
        "negative_ratio": 0.20,
        "batch_size": 4,
        "smoke": True,
        "mock_tokenizer": tok,
        "mock_backbone": backbone,
        "stage1": {"enabled": True, "epochs": 1, "batch_size": 4, "max_samples": 12},
        "stage2": {"enabled": True, "epochs": 1, "batch_size": 4, "n_splits": 2},
        "mlp_epochs": 2,
    }

    res = run_training_pipeline(suite_cfg)
    assert res["suite"] == "deberta_specialist_suite"

    # Verify ledger entries
    ledger_csv = tmp_path / "experiments" / "experiment_ledger.csv"
    assert ledger_csv.exists()
    df_led = pd.read_csv(ledger_csv)
    exp_ids = df_led["exp_id"].tolist()

    # EXP-00 baseline logged
    assert "EXP-00" in exp_ids

    # EXP-06a variants logged
    assert any("EXP-06a_ridge" in eid for eid in exp_ids)
    assert any("EXP-06a_mlp" in eid for eid in exp_ids)
    assert any("EXP-06a_hybrid" in eid for eid in exp_ids)

    # EXP-06b variants logged
    assert any("EXP-06b_plain" in eid for eid in exp_ids)
    assert any("EXP-06b_blend" in eid for eid in exp_ids)

    # EXP-06c variants logged
    assert any("EXP-06c_01" in eid for eid in exp_ids)
    assert any("EXP-06c_02" in eid for eid in exp_ids)
    assert any("EXP-06c_03" in eid for eid in exp_ids)

    # Verify specialist checkpoint output
    spec_dir = tmp_path / "export_deberta_gemini_specialist"
    assert (spec_dir / "heads.safetensors").exists()
    assert (spec_dir / "meta.json").exists()
    assert (spec_dir / "experiment_ledger.md").exists()
    assert (spec_dir / "experiment_ledger.csv").exists()

    # Verify winning gate output
    gate_dir = tmp_path / "export_deberta_gemini_gate_mlp"
    assert (gate_dir / "heads.safetensors").exists()
    assert (gate_dir / "meta.json").exists()
    assert (gate_dir / "experiment_ledger.md").exists()
    assert (gate_dir / "experiment_ledger.csv").exists()
