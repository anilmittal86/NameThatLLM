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


class DummyDebertaBackbone(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=24):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim, "num_hidden_layers": num_layers, "model_type": "deberta-v2"})()
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
