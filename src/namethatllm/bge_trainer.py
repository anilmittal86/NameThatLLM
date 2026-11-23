"""
BGE-M3 Two-Stage Multi-Head LoRA Fine-Tuning Pipeline.
======================================================
Fine-tunes BGE-M3 (560M XLM-RoBERTa bidirectional encoder) for LLM author attribution:
- Stage 1: Multi-Corpus Pretraining on Master Pool (267k rows) with Family-Margin Cross-Entropy & Label Smoothing.
- Stage 2: Prompt-Grouped Contrastive Fine-Tuning on train.csv with Within-Prompt Centered Twin Head & SupCon Loss.
- Validation: Evaluated against the Clean 1,800-Row Unseen Benchmark.
- Export: Self-contained `heads.safetensors` and PEFT LoRA adapter for immediate drop-in integration.
"""

from __future__ import annotations

import os
import gc
import json
import math
import time
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import balanced_accuracy_score
from safetensors.torch import save_file

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from namethatllm.config import LABELS, LAB2I, K, FAMILIES

LABEL2ID = LAB2I


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_hash(t: str) -> str:
    return hashlib.md5(t.strip().encode("utf-8")).hexdigest()


class NTLDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, prompt_ids: np.ndarray, tokenizer: Any, max_len: int = 512):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.prompt_ids = np.asarray(prompt_ids)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "text": str(self.texts[idx]),
            "label": self.labels[idx],
            "prompt_id": self.prompt_ids[idx]
        }


def make_collate_fn(tokenizer: Any, max_len: int = 512):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        prompt_ids = [b["prompt_id"] for b in batch]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        encoded["labels"] = labels
        encoded["prompt_ids"] = prompt_ids
        return encoded
    return collate_fn


class PromptGroupBatchSampler(Sampler):
    """Batches samples strictly from the same prompt cluster for Stage 2."""
    def __init__(self, prompt_ids: np.ndarray, batch_size: int = 4, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        prompt_to_indices: Dict[Any, List[int]] = {}
        for idx, pid in enumerate(prompt_ids):
            prompt_to_indices.setdefault(pid, []).append(idx)

        self.batches: List[List[int]] = []
        for pid, indices in prompt_to_indices.items():
            if len(indices) < 2:
                continue
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                chunk = indices[i:i + batch_size]
                if len(chunk) >= 2:
                    self.batches.append(chunk)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


class BGEClassifier(nn.Module):
    """BGE-M3 Backbone with PEFT LoRA, L2-Normalized CLS Extraction, and Dual Heads."""
    def __init__(self, backbone: nn.Module, hidden_size: int = 1024, num_classes: int = 18):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.plain_head = nn.Linear(hidden_size, num_classes)
        self.centered_head = nn.Linear(hidden_size, num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])

    def forward_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_repr = outputs.last_hidden_state[:, 0]
        cls_norm = F.normalize(cls_repr, p=2, dim=-1)
        return cls_norm

    def predict_plain(self, Z: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.mean(torch.stack([self.plain_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.plain_head(Z)

    def predict_centered(self, Z: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.mean(torch.stack([self.centered_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.centered_head(Z)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = self.forward_features(input_ids, attention_mask)
        logits_p = self.predict_plain(Z)

        # In-batch mean centering (Prompt-centering proxy for Stage 2)
        mu_batch = Z.mean(dim=0, keepdim=True)
        Z_c = Z - mu_batch
        logits_c = self.predict_centered(Z_c)
        return Z, logits_p, logits_c


def build_family_mask(device: torch.device) -> torch.Tensor:
    mask = torch.zeros((K, K), dtype=torch.float32, device=device)
    for f_name, members in FAMILIES.items():
        for m1 in members:
            for m2 in members:
                if m1 != m2:
                    mask[m1, m2] = 1.0
    return mask


def family_margin_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    family_mask: torch.Tensor,
    family_margin: float = 0.35,
    label_smoothing: float = 0.05
) -> torch.Tensor:
    if family_margin > 0.0:
        margin_matrix = family_mask[labels] * family_margin
        logits = logits + margin_matrix
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def compute_supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    norms = F.normalize(features, p=2, dim=-1)
    sim = torch.matmul(norms, norms.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
    diag_mask = torch.eye(features.shape[0], device=features.device)
    mask = mask * (1.0 - diag_mask)
    pos_per_row = mask.sum(dim=1)
    valid = pos_per_row > 0
    if not valid.any():
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    exp_sim = torch.exp(sim - torch.max(sim, dim=1, keepdim=True)[0]) * (1.0 - diag_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / pos_per_row[valid]
    return -mean_log_prob_pos.mean()


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds_p, all_preds_c, all_preds_m, all_labels = [], [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                Z = model.forward_features(input_ids, attention_mask)
                logits_p = model.predict_plain(Z)
                Z_c = Z - Z.mean(dim=0, keepdim=True)
                logits_c = model.predict_centered(Z_c)

                prob_p = F.softmax(logits_p, dim=-1)
                prob_c = F.softmax(logits_c, dim=-1)
                prob_m = 0.5 * prob_p + 0.5 * prob_c

                loss = F.cross_entropy(logits_p, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds_p.append(prob_p.argmax(dim=-1).cpu().numpy())
            all_preds_c.append(prob_c.argmax(dim=-1).cpu().numpy())
            all_preds_m.append(prob_m.argmax(dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    bacc_p = balanced_accuracy_score(y_true, np.concatenate(all_preds_p))
    bacc_c = balanced_accuracy_score(y_true, np.concatenate(all_preds_c))
    bacc_m = balanced_accuracy_score(y_true, np.concatenate(all_preds_m))

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "bacc_plain": float(bacc_p),
        "bacc_centered": float(bacc_c),
        "bacc_merged": float(bacc_m),
    }


def run_bge_training_pipeline(
    config: Dict[str, Any],
    datasets_dir: Path,
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes full Two-Stage LoRA fine-tuning for BGE-M3.
    Returns dictionary with validation metrics and artifact paths.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[BGE Trainer] Running on device: {device}")
    seed_everything(42)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover Datasets
    bench_p = datasets_dir / "clean_benchmark_1800_rows.parquet"
    if not bench_p.exists():
        for cand in datasets_dir.glob("*benchmark*.parquet"):
            bench_p = cand
            break

    if bench_p.exists():
        df_unseen = pd.read_parquet(bench_p)
        print(f"[BGE Trainer] Loaded Unseen Benchmark: {len(df_unseen)} rows from {bench_p}")
        unseen_hashes = set(df_unseen["text"].map(compute_hash))
    else:
        raise FileNotFoundError(f"Clean benchmark dataset not found in {datasets_dir}")

    pool_p = datasets_dir / "clean_master_pool.parquet"
    if not pool_p.exists():
        pool_p = datasets_dir / "pool_120k.parquet"
    if not pool_p.exists():
        for cand in datasets_dir.glob("*pool*.parquet"):
            pool_p = cand
            break

    if pool_p.exists():
        df_pool_raw = pd.read_parquet(pool_p)
        df_pool_raw = df_pool_raw[df_pool_raw["model"].isin(LABELS)].reset_index(drop=True)
        pool_hashes = df_pool_raw["text"].map(compute_hash)
        df_pool = df_pool_raw[~pool_hashes.isin(unseen_hashes)].reset_index(drop=True)
        print(f"[BGE Trainer] Loaded Master Pool: {len(df_pool)} clean rows from {pool_p}")
    else:
        raise FileNotFoundError(f"Master pool dataset not found in {datasets_dir}")

    train_p = datasets_dir / "train.csv"
    if not train_p.exists():
        for cand in datasets_dir.glob("train*.csv"):
            train_p = cand
            break

    if train_p.exists():
        df_train_raw = pd.read_csv(train_p)
        df_train_raw = df_train_raw[df_train_raw["model"].isin(LABELS)].reset_index(drop=True)
        tr_hashes = df_train_raw["text"].map(compute_hash)
        df_train = df_train_raw[~tr_hashes.isin(unseen_hashes)].reset_index(drop=True)
        print(f"[BGE Trainer] Loaded Train CSV: {len(df_train)} clean rows from {train_p}")
    else:
        raise FileNotFoundError(f"train.csv not found in {datasets_dir}")

    # Stage 1 Dataset Slicing (Balanced sampling up to max_samples)
    s1_max = config.get("stage1", {}).get("max_samples", 36000)
    per_class = s1_max // K
    s1_subsets = []
    for m in LABELS:
        m_df = df_pool[df_pool["model"] == m]
        if len(m_df) > per_class:
            m_df = m_df.sample(n=per_class, random_state=42)
        s1_subsets.append(m_df)
    df_s1_train = pd.concat(s1_subsets, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    if "prompt_id" not in df_s1_train.columns:
        df_s1_train["prompt_id"] = -1
    print(f"[BGE Trainer] Stage 1 Pool Prepared: {len(df_s1_train)} rows across {K} classes.")

    # 2. Tokenizer & Model
    backbone_name = config.get("backbone", "BAAI/bge-m3")
    print(f"[BGE Trainer] Loading tokenizer & base model for: {backbone_name}...")
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    try:
        base_model = AutoModel.from_pretrained(backbone_name, dtype=torch.float16)
    except TypeError:
        base_model = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float16)

    # Enable gradient checkpointing for memory efficiency
    base_model.gradient_checkpointing_enable()

    # Wrap backbone with LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["query", "key", "value", "dense"]),
        bias="none",
    )
    peft_backbone = get_peft_model(base_model, lora_cfg)
    peft_backbone.print_trainable_parameters()

    model = BGEClassifier(backbone=peft_backbone, hidden_size=config.get("hidden_size", 1024), num_classes=K)
    model.to(device)

    # Validation DataLoader
    val_pids = (
        df_unseen["prompt_canon"].values
        if "prompt_canon" in df_unseen.columns
        else (df_unseen["prompt_id"].values if "prompt_id" in df_unseen.columns else np.zeros(len(df_unseen), dtype=np.int64))
    )
    val_dataset = NTLDataset(
        texts=df_unseen["text"].tolist(),
        labels=df_unseen["model"].map(LABEL2ID).values,
        prompt_ids=val_pids,
        tokenizer=tokenizer,
        max_len=config.get("max_len", 512),
    )
    collate = make_collate_fn(tokenizer, max_len=config.get("max_len", 512))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)

    family_mask = build_family_mask(device)
    val_history = []
    best_bacc = 0.0

    # 3. Stage 1 Training
    if config.get("stage1", {}).get("enabled", True):
        s1_cfg = config["stage1"]
        train_s1_dataset = NTLDataset(
            texts=df_s1_train["text"].tolist(),
            labels=df_s1_train["model"].map(LABEL2ID).values,
            prompt_ids=df_s1_train["prompt_id"].values,
            tokenizer=tokenizer,
            max_len=config.get("max_len", 512),
        )
        train_s1_loader = DataLoader(
            train_s1_dataset,
            batch_size=s1_cfg["batch_size"],
            shuffle=True,
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
        )

        optimizer_s1 = torch.optim.AdamW([
            {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad], "lr": s1_cfg["lr_lora"]},
            {"params": model.plain_head.parameters(), "lr": s1_cfg["lr_head"]}
        ], weight_decay=config.get("weight_decay", 0.01))

        total_steps_s1 = (len(train_s1_loader) // s1_cfg["gradient_accumulation_steps"]) * s1_cfg["epochs"]
        scheduler_s1 = get_cosine_schedule_with_warmup(optimizer_s1, num_warmup_steps=100, num_training_steps=total_steps_s1)
        scaler = torch.amp.GradScaler("cuda")

        print(f"\n[Stage 1] Beginning Master Pool Training: {len(df_s1_train)} samples | {s1_cfg['epochs']} epochs | {total_steps_s1} steps")
        start_s1 = time.time()

        for epoch in range(s1_cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            optimizer_s1.zero_grad()

            for step, batch in enumerate(train_s1_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    Z = model.forward_features(input_ids, attention_mask)
                    logits_p = model.predict_plain(Z)
                    loss = family_margin_cross_entropy(logits_p, labels, family_mask, config.get("family_margin", 0.35), config.get("label_smoothing", 0.05))
                    loss = loss / s1_cfg["gradient_accumulation_steps"]

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * s1_cfg["gradient_accumulation_steps"]

                if (step + 1) % s1_cfg["gradient_accumulation_steps"] == 0 or (step + 1) == len(train_s1_loader):
                    scaler.unscale_(optimizer_s1)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer_s1)
                    scaler.update()
                    optimizer_s1.zero_grad()
                    scheduler_s1.step()

                if (step + 1) % 200 == 0:
                    print(f"  [Stage 1] Step {step+1}/{len(train_s1_loader)} | Running Loss: {epoch_loss / (step + 1):.4f}")

            val_metrics = evaluate_model(model, val_loader, device)
            val_metrics["epoch"] = f"stage1_{epoch+1}"
            val_metrics["elapsed_min"] = (time.time() - start_s1) / 60.0
            val_history.append(val_metrics)
            print(f"[Stage 1 Epoch {epoch+1}] Val Merged BAcc: {val_metrics['bacc_merged']:.4f} | Plain: {val_metrics['bacc_plain']:.4f} | Centered: {val_metrics['bacc_centered']:.4f}")

            if val_metrics["bacc_plain"] > best_bacc:
                best_bacc = val_metrics["bacc_plain"]
                _save_checkpoint(model, output_dir, is_best=True)

    # 4. Stage 2 Training (Prompt-Grouped Contrastive Fine-Tuning)
    if config.get("stage2", {}).get("enabled", True):
        # Initialize centered_head directly from trained plain_head
        with torch.no_grad():
            model.centered_head.weight.copy_(model.plain_head.weight)
            model.centered_head.bias.copy_(model.plain_head.bias)
        print("\n[Stage 2] Initialized centered_head weights and biases directly from trained plain_head.")

        s2_cfg = config["stage2"]
        if "prompt_id" not in df_train.columns:
            if "prompt_canon" in df_train.columns:
                df_train["prompt_id"] = df_train["prompt_canon"]
            else:
                df_train["prompt_id"] = df_train.index
        prompt_counts = df_train["prompt_id"].value_counts()
        multi_prompts = set(prompt_counts[prompt_counts >= 2].index)
        df_s2 = df_train[df_train["prompt_id"].isin(multi_prompts)].reset_index(drop=True)
        print(f"[Stage 2] Grouped dataset: {len(df_s2)} rows across {df_s2['prompt_id'].nunique()} multi-sample prompts.")

        train_s2_dataset = NTLDataset(
            texts=df_s2["text"].tolist(),
            labels=df_s2["model"].map(LABEL2ID).values,
            prompt_ids=df_s2["prompt_id"].values,
            tokenizer=tokenizer,
            max_len=config.get("max_len", 512),
        )
        batch_sampler_s2 = PromptGroupBatchSampler(df_s2["prompt_id"].values, batch_size=s2_cfg["batch_size"], shuffle=True)
        train_s2_loader = DataLoader(
            train_s2_dataset,
            batch_sampler=batch_sampler_s2,
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
        )

        optimizer_s2 = torch.optim.AdamW([
            {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad], "lr": s2_cfg["lr_lora"]},
            {"params": list(model.plain_head.parameters()) + list(model.centered_head.parameters()), "lr": s2_cfg["lr_head"]}
        ], weight_decay=config.get("weight_decay", 0.01))

        total_steps_s2 = (len(train_s2_loader) // s2_cfg["gradient_accumulation_steps"]) * s2_cfg["epochs"]
        scheduler_s2 = get_cosine_schedule_with_warmup(optimizer_s2, num_warmup_steps=30, num_training_steps=total_steps_s2)
        scaler = torch.amp.GradScaler("cuda")
        start_s2 = time.time()

        for epoch in range(s2_cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            optimizer_s2.zero_grad()

            for step, batch in enumerate(train_s2_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    Z, logits_p, logits_c = model(input_ids, attention_mask)
                    loss_p = F.cross_entropy(logits_p, labels, label_smoothing=0.05)
                    loss_c = family_margin_cross_entropy(logits_c, labels, family_mask, config.get("family_margin", 0.35), label_smoothing=0.05)
                    loss_sup = compute_supcon_loss(Z, labels, temperature=0.07)

                    cw = s2_cfg.get("centered_weight", 0.65)
                    sw = s2_cfg.get("supcon_weight", 0.20)
                    pw = 1.0 - cw
                    loss = (pw * loss_p + cw * loss_c + sw * loss_sup) / s2_cfg["gradient_accumulation_steps"]

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * s2_cfg["gradient_accumulation_steps"]

                if (step + 1) % s2_cfg["gradient_accumulation_steps"] == 0 or (step + 1) == len(train_s2_loader):
                    scaler.unscale_(optimizer_s2)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer_s2)
                    scaler.update()
                    optimizer_s2.zero_grad()
                    scheduler_s2.step()

            val_metrics = evaluate_model(model, val_loader, device)
            val_metrics["epoch"] = f"stage2_{epoch+1}"
            val_metrics["elapsed_min"] = (time.time() - start_s2) / 60.0
            val_history.append(val_metrics)
            print(f"[Stage 2 Epoch {epoch+1}] Val Merged BAcc: {val_metrics['bacc_merged']:.4f} | Plain: {val_metrics['bacc_plain']:.4f} | Centered: {val_metrics['bacc_centered']:.4f}")

            if val_metrics["bacc_merged"] > best_bacc:
                best_bacc = val_metrics["bacc_merged"]
                _save_checkpoint(model, output_dir, is_best=True)

    # 5. Save Final Export
    _save_checkpoint(model, output_dir, is_best=False)

    meta = {
        "model_name": "bge-m3-lora-inhouse",
        "backbone": backbone_name,
        "hidden_size": config.get("hidden_size", 1024),
        "num_classes": K,
        "labels": LABELS,
        "families": FAMILIES,
        "config": config,
        "val_history": val_history,
        "best_val_bacc": float(best_bacc),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[BGE Trainer] 🎉 Completed successfully! Best Unseen BAcc: {best_bacc:.4f}")
    return meta


def _save_checkpoint(model: BGEClassifier, output_dir: Path, is_best: bool = False) -> None:
    # Save PEFT adapter
    model.backbone.save_pretrained(str(output_dir))

    # Save dual heads
    head_tensors = {
        "plain.weight": model.plain_head.weight.detach().cpu().float(),
        "plain.bias": model.plain_head.bias.detach().cpu().float(),
        "centered.weight": model.centered_head.weight.detach().cpu().float(),
        "centered.bias": model.centered_head.bias.detach().cpu().float(),
    }
    save_file(head_tensors, str(output_dir / "heads.safetensors"))
    if is_best:
        save_file(head_tensors, str(output_dir / "heads_best.safetensors"))
