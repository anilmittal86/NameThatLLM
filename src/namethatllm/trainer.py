"""
Two-Stage Multi-Layer LoRA Training Engine
==========================================
Implements production Two-Stage Multi-Layer LoRA training with 8 intermediate
LayerNorm heads, causal attention-sink masking (m[:, 0] = 0.0), prompt-grouped
contrastive centering, and standard checkpoint export.
"""

import os
import gc
import json
import math
import time
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Iterator

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Neutralize Kaggle's incompatible pre-installed torchao (<0.16.0) across all libraries
_orig_find_spec = importlib.util.find_spec
def _safe_find_spec(name, *args, **kwargs):
    if name == "torchao":
        return None
    return _orig_find_spec(name, *args, **kwargs)
importlib.util.find_spec = _safe_find_spec

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from .config import LABELS, K, LAB2I, resolve_dataset_paths, resolve_all_pool_files, find_base_model
from .wpca_gaussian import fit_shared_gaussian_lowram

# Disable incompatible torchao version check in Kaggle environments
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

class MultiLayerLoRAClassifier(nn.Module):
    """
    Universal Multi-Layer LoRA sequence classifier with dedicated per-layer
    LayerNorm heads and causal attention-sink masking (m[:, 0] = 0.0).
    """
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        layers: List[int],
        num_classes: int = K,
        keep_layers: Optional[int] = None
    ):
        super().__init__()
        self.backbone = backbone
        if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
            self.backbone.norm = nn.Identity()
        self.layers = list(layers)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.total_dim = len(layers) * hidden_size

        # Dedicated LayerNorm for each intermediate layer
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers])

        # Dual classification heads (Plain + Prompt-Centered Twin)
        self.plain_head = nn.Linear(self.total_dim, num_classes)
        self.centered_head = nn.Linear(self.total_dim, num_classes)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drop_first_token: bool = True
    ) -> torch.Tensor:
        """Forward pass with causal attention sink masking and multi-layer LayerNorm pooling."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        hidden_states = outputs.hidden_states

        m = attention_mask.clone().float()
        if drop_first_token and m.shape[1] > 1:
            m[:, 0] = 0.0
        den = m.sum(dim=1, keepdim=True).clamp(min=1e-5)

        first_device = input_ids.device
        pooled_list = []
        for idx, l in enumerate(self.layers):
            h = hidden_states[l].float()
            layer_dev = h.device
            m_l = m.to(layer_dev)
            den_l = den.to(layer_dev)
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
            pooled = (h * m_l.unsqueeze(-1)).sum(dim=1) / den_l
            norm = self.norms[idx]
            if next(norm.parameters()).device != layer_dev:
                norm = norm.to(layer_dev)
                self.norms[idx] = norm
            normed = norm(pooled)
            pooled_list.append(normed.to(first_device))

        return torch.cat(pooled_list, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drop_first_token: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = self.forward_features(input_ids, attention_mask, drop_first_token=drop_first_token)
        first_dev = Z.device
        if next(self.plain_head.parameters()).device != first_dev:
            self.plain_head = self.plain_head.to(first_dev)
        if next(self.centered_head.parameters()).device != first_dev:
            self.centered_head = self.centered_head.to(first_dev)
        logits_plain = self.plain_head(Z)
        logits_centered = self.centered_head(Z)
        return Z, logits_plain, logits_centered

def compute_two_stage_loss(
    model: MultiLayerLoRAClassifier,
    Z: torch.Tensor,
    labels: torch.Tensor,
    prompt_ids: Optional[List[Any]] = None,
    centered_weight: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Computes joint plain cross-entropy and within-prompt contrastive centered loss:
    L = L_plain + centered_weight * L_centered
    """
    dev = Z.device
    if next(model.plain_head.parameters()).device != dev:
        model.plain_head = model.plain_head.to(dev)
    if next(model.centered_head.parameters()).device != dev:
        model.centered_head = model.centered_head.to(dev)

    logits_plain = model.plain_head(Z)
    loss_plain = F.cross_entropy(logits_plain, labels.to(dev))

    if prompt_ids is None or centered_weight <= 0.0:
        return loss_plain, loss_plain, None

    # Exact within-prompt centering
    Z_c = Z.clone()
    prompt_arr = np.array(prompt_ids)
    unique_prompts = list(dict.fromkeys(prompt_ids))
    for p in unique_prompts:
        mask = torch.tensor(prompt_arr == p, device=dev, dtype=torch.bool)
        if mask.sum() > 1:
            Z_c[mask] = Z[mask] - Z[mask].mean(dim=0, keepdim=True)
        else:
            Z_c[mask] = Z[mask] - Z[mask]

    logits_centered = model.centered_head(Z_c)
    loss_centered = F.cross_entropy(logits_centered, labels.to(dev))
    total_loss = loss_plain + float(centered_weight) * loss_centered
    return total_loss, loss_plain, loss_centered

class PromptGroupBatchSampler(Sampler):
    """
    Groups samples sharing the same prompt_id into complete batches to ensure exact
    within-prompt contrastive centering during Stage 2 training.
    """
    def __init__(self, prompt_ids: List[Any], batch_size: int = 18, shuffle: bool = True, seed: int = 42):
        self.prompt_to_indices: Dict[Any, List[int]] = {}
        for idx, p in enumerate(prompt_ids):
            self.prompt_to_indices.setdefault(p, []).append(idx)
        self.prompts = list(self.prompt_to_indices.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        prompts = list(self.prompts)
        if self.shuffle:
            rng.shuffle(prompts)

        current_batch: List[int] = []
        for p in prompts:
            idxs = list(self.prompt_to_indices[p])
            if self.shuffle:
                rng.shuffle(idxs)
            if len(current_batch) + len(idxs) > self.batch_size and len(current_batch) > 0:
                yield current_batch
                current_batch = []
            current_batch.extend(idxs)
            if len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch = []
        if len(current_batch) > 0:
            yield current_batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self.prompt_to_indices.values())
        return math.ceil(total / max(1, self.batch_size))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class TextSequenceDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        prompt_ids: Optional[List[Any]],
        tok: Any,
        max_len: int = 447,
        end_id: int = 151643
    ):
        self.texts = texts
        self.labels = labels
        self.prompt_ids = prompt_ids
        self.tok = tok
        self.max_len = max_len
        self.end_id = end_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_ids = self.tok(str(self.texts[idx]), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        input_ids = raw_ids + [self.end_id]
        item: Dict[str, Any] = {"input_ids": torch.tensor(input_ids, dtype=torch.long)}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.prompt_ids is not None:
            item["prompt_id"] = self.prompt_ids[idx]
        return item

def collate_text_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = [item["input_ids"] for item in batch]
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (padded != 0).long()
    res: Dict[str, Any] = {"input_ids": padded, "attention_mask": attention_mask}
    if "label" in batch[0]:
        res["labels"] = torch.stack([item["label"] for item in batch])
    if "prompt_id" in batch[0]:
        res["prompt_ids"] = [item["prompt_id"] for item in batch]
    return res

def export_compatible_checkpoint(
    model: MultiLayerLoRAClassifier,
    output_dir: Path,
    layers: List[int],
    keep: int,
    max_len: int = 447,
    end_id: int = 151643,
    Z_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    prompts_train: Optional[np.ndarray] = None
) -> Path:
    """
    Exports standard checkpoint artifacts strictly compatible with inference loaders:
    1. adapter/ (LoRA adapter weights and config)
    2. heads.safetensors (norms.{i}.weight, norms.{i}.bias, plain.weight, plain.bias, centered.weight, centered.bias)
    3. meta.json (keep, layers, max_len, end_id, labels)
    4. wPCA_bank.npz (shared Gaussian covariance / whitening basis)
    """
    from safetensors.torch import save_file

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Adapter weights
    adapter_dir = out_p / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(str(adapter_dir))
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(str(adapter_dir))

    # 2. Heads dictionary
    heads_dict = {}
    for i, norm in enumerate(model.norms):
        heads_dict[f"norms.{i}.weight"] = norm.weight.detach().cpu().contiguous()
        if hasattr(norm, "bias") and norm.bias is not None:
            heads_dict[f"norms.{i}.bias"] = norm.bias.detach().cpu().contiguous()

    heads_dict["plain.weight"] = model.plain_head.weight.detach().cpu().contiguous()
    heads_dict["plain.bias"] = model.plain_head.bias.detach().cpu().contiguous()
    heads_dict["centered.weight"] = model.centered_head.weight.detach().cpu().contiguous()
    heads_dict["centered.bias"] = model.centered_head.bias.detach().cpu().contiguous()

    save_file(heads_dict, str(out_p / "heads.safetensors"))

    # 3. Meta JSON
    meta = {
        "labels": list(LABELS),
        "cfg": {
            "keep": int(keep),
            "layers": [int(x) for x in layers],
            "max_len": int(max_len),
            "end_token_id": int(end_id)
        }
    }
    (out_p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 4. wPCA Bank precomputation
    if Z_train is not None and y_train is not None and prompts_train is not None:
        d_target = min(256, Z_train.shape[1])
        res = fit_shared_gaussian_lowram(Z_train, y_train, prompts_train, d_target=d_target)
        np.savez_compressed(
            str(out_p / "wPCA_bank.npz"),
            mu=res["mx"].astype(np.float32),
            sd=res["sx"].astype(np.float32),
            W=res["W"].astype(np.float32),
            layers=np.array(layers, dtype=np.int32)
        )

    return out_p

def run_training_pipeline(train_cfg: Optional[Dict[str, Any]] = None) -> Path:
    """
    Executes complete Two-Stage Multi-Layer LoRA training pipeline:
    Stage 1: Pretraining on pool parquets.
    Stage 2: Prompt-grouped contrastive fine-tuning on train.csv.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    if train_cfg is None:
        train_cfg = {}

    backbone_name = train_cfg.get("backbone", "Qwen/Qwen3-8B-Base")
    keep_layers = int(train_cfg.get("keep_layers", 24))
    layers = list(train_cfg.get("layers", [2, 4, 6, 8, 12, 16, 20, 24]))
    max_len = int(train_cfg.get("max_len", 447))
    end_id = int(train_cfg.get("end_id", 151643))
    out_dir = Path(train_cfg.get("output_dir", "export_lora24"))

    input_dir = Path(train_cfg.get("input_dir", "/kaggle/input"))
    local_base = None
    if input_dir.exists():
        if "8b" in backbone_name.lower():
            local_base = find_base_model(input_dir, "qwen3", 4096, 36)
        elif "14b" in backbone_name.lower():
            local_base = find_base_model(input_dir, "qwen3", 5120, 40)
    model_source = str(local_base) if local_base else backbone_name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"[Trainer] Device: {device}, Available GPUs: {n_gpus}, Model Source: {model_source}, Keep: {keep_layers}")

    train_path, test_path, pool_file = resolve_dataset_paths(input_dir if input_dir.exists() else Path("."))
    all_pools = resolve_all_pool_files(input_dir if input_dir.exists() else Path("."))

    tok = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_cfg = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    base_cfg.num_hidden_layers = keep_layers

    # Multi-GPU 2xT4 memory balancing: balance across both GPUs so GPU 0 has plenty of headroom
    if n_gpus >= 2:
        max_memory = {i: "9.0GiB" for i in range(n_gpus)}
        base_model = AutoModel.from_pretrained(
            model_source,
            config=base_cfg,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True
        )
    elif n_gpus == 1:
        base_model = AutoModel.from_pretrained(
            model_source,
            config=base_cfg,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        base_model = AutoModel.from_pretrained(
            model_source,
            config=base_cfg,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    try:
        import peft.import_utils
        import peft.tuners.lora.torchao
        peft.import_utils.is_torchao_available = lambda: False
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
    except Exception:
        pass

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=int(train_cfg.get("lora_r", 16)),
        lora_alpha=int(train_cfg.get("lora_alpha", 32)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        target_modules=list(train_cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])),
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    base_model = get_peft_model(base_model, lora_cfg)

    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    model = MultiLayerLoRAClassifier(base_model, hidden_size=base_cfg.hidden_size, layers=layers, num_classes=K)
    # If device_map was used, do NOT call model.to() as it overrides device_map onto a single GPU
    if hasattr(base_model, "hf_device_map") and len(getattr(base_model, "hf_device_map", {})) > 1:
        model.plain_head = model.plain_head.to(device)
        model.centered_head = model.centered_head.to(device)
        for idx in range(len(model.norms)):
            model.norms[idx] = model.norms[idx].to(device)
    else:
        model = model.to(device)

    # Stage 1: Pool Pretraining
    stage1_cfg = train_cfg.get("stage1", {})
    if stage1_cfg.get("enabled", True) and (all_pools or pool_file):
        pool_p = all_pools[0] if all_pools else pool_file
        print(f"[Trainer] Stage 1 Pretraining on pool {pool_p}...")
        df_p = pd.read_parquet(pool_p, columns=["text", "model"])
        df_p = df_p[df_p["model"].isin(LABELS)].dropna().sample(frac=1.0, random_state=42).reset_index(drop=True)
        max_s = int(stage1_cfg.get("max_samples", 20000))
        if len(df_p) > max_s:
            df_p = df_p.iloc[:max_s]

        bs1 = int(stage1_cfg.get("batch_size", 4))
        accum1 = int(stage1_cfg.get("gradient_accumulation_steps", 2))
        ds1 = TextSequenceDataset(df_p["text"].tolist(), [LAB2I[m] for m in df_p["model"]], None, tok, max_len=max_len, end_id=end_id)
        loader1 = DataLoader(ds1, batch_size=bs1, shuffle=True, collate_fn=collate_text_batch)

        opt1 = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": float(stage1_cfg.get("lr_lora", 2e-4))},
            {"params": list(model.norms.parameters()) + list(model.plain_head.parameters()), "lr": float(stage1_cfg.get("lr_head", 1e-3))}
        ], weight_decay=float(train_cfg.get("weight_decay", 0.01)))
        sched1 = get_cosine_schedule_with_warmup(opt1, num_warmup_steps=int(stage1_cfg.get("warmup_steps", 100)), num_training_steps=math.ceil(len(loader1) / accum1))

        model.train()
        opt1.zero_grad()
        for step_idx, batch in enumerate(loader1):
            _, logits_p, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = F.cross_entropy(logits_p, batch["labels"].to(device)) / accum1
            loss.backward()
            if (step_idx + 1) % accum1 == 0 or (step_idx + 1) == len(loader1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt1.step()
                sched1.step()
                opt1.zero_grad()

        del ds1, loader1, opt1, sched1, df_p
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: Prompt-Grouped Fine-Tuning on train.csv
    stage2_cfg = train_cfg.get("stage2", {})
    df_tr = pd.read_csv(train_path)
    print(f"[Trainer] Stage 2 Fine-Tuning on {train_path} ({len(df_tr)} samples)...")
    bs2 = int(stage2_cfg.get("batch_size", 2))
    accum2 = int(stage2_cfg.get("gradient_accumulation_steps", 2))
    ds2 = TextSequenceDataset(df_tr["text"].tolist(), [LAB2I[m] for m in df_tr["model"]], df_tr["prompt_id"].tolist(), tok, max_len=max_len, end_id=end_id)
    sampler2 = PromptGroupBatchSampler(df_tr["prompt_id"].tolist(), batch_size=bs2, shuffle=True)
    loader2 = DataLoader(ds2, batch_sampler=sampler2, collate_fn=collate_text_batch)

    opt2 = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": float(stage2_cfg.get("lr_lora", 5e-5))},
        {"params": list(model.norms.parameters()) + list(model.plain_head.parameters()) + list(model.centered_head.parameters()), "lr": float(stage2_cfg.get("lr_head", 3e-4))}
    ], weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    epochs2 = int(stage2_cfg.get("epochs", 2))
    total_steps2 = math.ceil(len(loader2) / accum2) * epochs2
    sched2 = get_cosine_schedule_with_warmup(opt2, num_warmup_steps=int(stage2_cfg.get("warmup_steps", 50)), num_training_steps=total_steps2)

    model.train()
    for ep in range(epochs2):
        sampler2.set_epoch(ep)
        opt2.zero_grad()
        for step_idx, batch in enumerate(loader2):
            Z = model.forward_features(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            tot_loss, _, _ = compute_two_stage_loss(
                model, Z, batch["labels"].to(device),
                prompt_ids=batch.get("prompt_ids"),
                centered_weight=float(stage2_cfg.get("centered_weight", 0.5))
            )
            loss = tot_loss / accum2
            loss.backward()
            if (step_idx + 1) % accum2 == 0 or (step_idx + 1) == len(loader2):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()
                sched2.step()
                opt2.zero_grad()

    # Feature extraction for wPCA bank
    del loader2, opt2, sched2, sampler2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    loader_eval = DataLoader(ds2, batch_size=4, shuffle=False, collate_fn=collate_text_batch)
    Z_list = []
    with torch.no_grad():
        for batch in loader_eval:
            z_b = model.forward_features(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            Z_list.append(z_b.cpu().numpy())
    Z_train = np.vstack(Z_list)
    y_train = np.array([LAB2I[m] for m in df_tr["model"]])
    prompts_train = df_tr["prompt_id"].values

    export_compatible_checkpoint(
        model=model,
        output_dir=out_dir,
        layers=layers,
        keep=keep_layers,
        max_len=max_len,
        end_id=end_id,
        Z_train=Z_train,
        y_train=y_train,
        prompts_train=prompts_train
    )
    print(f"[Trainer] Checkpoint exported to {out_dir}")
    return out_dir
