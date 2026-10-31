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

from .config import LABELS, K, LAB2I, FAMILIES, resolve_dataset_paths, resolve_all_pool_files, find_base_model
from .wpca_gaussian import fit_shared_gaussian_lowram
from .ablation import _text_hash, build_unseen_holdout_dataset

# Disable incompatible torchao version check in Kaggle environments
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

def _disable_final_norm(mod: nn.Module) -> None:
    for path in ["norm", "model.norm", "base_model.norm", "base_model.model.norm", "base_model.model.model.norm"]:
        curr = mod
        parts = path.split(".")
        for p in parts[:-1]:
            curr = getattr(curr, p, None)
            if curr is None:
                break
        if curr is not None and hasattr(curr, parts[-1]) and getattr(curr, parts[-1]) is not None:
            setattr(curr, parts[-1], nn.Identity())

def _find_decoder_layer_list(backbone: nn.Module) -> Optional[nn.ModuleList]:
    """Finds the main sequence of transformer/decoder layer modules within a model."""
    for path in [
        "model.layers",
        "base_model.model.layers",
        "base_model.model.model.layers",
        "layers",
        "encoder.layer",
        "deberta.encoder.layer",
        "base_model.encoder.layer",
        "transformer.h",
        "base_model.transformer.h",
    ]:
        curr = backbone
        ok = True
        for p in path.split("."):
            if hasattr(curr, p):
                curr = getattr(curr, p)
            else:
                ok = False
                break
        if ok and isinstance(curr, (nn.ModuleList, list, tuple)):
            return curr
    return None

class MultiLayerLoRAClassifier(nn.Module):
    """
    Universal Multi-Layer LoRA sequence classifier with dedicated per-layer
    LayerNorm heads, causal attention-sink masking (m[:, 0] = 0.0), and
    5-branch Multi-Sample Dropout.
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
        _disable_final_norm(self.backbone)
        self.layers = list(layers)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.total_dim = len(layers) * hidden_size

        # Dedicated LayerNorm for each intermediate layer
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers])

        # Dual classification heads (Plain + Prompt-Centered Twin)
        self.plain_head = nn.Linear(self.total_dim, num_classes)
        self.centered_head = nn.Linear(self.total_dim, num_classes)

        # 5-branch Multi-Sample Dropout for superior regularization
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])

    def predict_plain(self, Z: torch.Tensor) -> torch.Tensor:
        first_dev = Z.device
        if next(self.plain_head.parameters()).device != first_dev:
            self.plain_head = self.plain_head.to(first_dev)
        if self.training:
            return torch.mean(torch.stack([self.plain_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.plain_head(Z)

    def predict_centered(self, Z: torch.Tensor) -> torch.Tensor:
        first_dev = Z.device
        if next(self.centered_head.parameters()).device != first_dev:
            self.centered_head = self.centered_head.to(first_dev)
        if self.training:
            return torch.mean(torch.stack([self.centered_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.centered_head(Z)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drop_first_token: bool = True
    ) -> torch.Tensor:
        """Forward pass with causal attention sink masking and multi-layer LayerNorm pooling."""
        m = attention_mask.clone().float()
        if drop_first_token and m.shape[1] > 1:
            m[:, 0] = 0.0
        den = m.sum(dim=1, keepdim=True).clamp(min=1e-5)
        first_device = input_ids.device

        dec_layers = _find_decoder_layer_list(self.backbone)
        can_use_hooks = (
            dec_layers is not None
            and all((0 <= (l - 1) < len(dec_layers)) for l in self.layers)
        )

        if can_use_hooks:
            hooked_pooled = {}
            handles = []
            for idx, l in enumerate(self.layers):
                norm = self.norms[idx]
                layer_mod = dec_layers[l - 1]

                def _make_hook(norm_layer, l_idx):
                    def _hook(mod, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        h = h.float()
                        ldev = h.device
                        m_l = m.to(ldev)
                        den_l = den.to(ldev)
                        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
                        pooled = (h * m_l.unsqueeze(-1)).sum(dim=1) / den_l
                        if next(norm_layer.parameters()).device != ldev:
                            norm_layer.to(ldev)
                        normed = norm_layer(pooled)
                        hooked_pooled[l_idx] = normed.to(first_device)
                    return _hook

                handles.append(layer_mod.register_forward_hook(_make_hook(norm, idx)))

            try:
                try:
                    self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False
                    )
                except TypeError:
                    self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for h in handles:
                    h.remove()

            if len(hooked_pooled) == len(self.layers):
                pooled_list = [hooked_pooled[idx] for idx in range(len(self.layers))]
                return torch.cat(pooled_list, dim=-1)

        # Fallback to output_hidden_states=True
        try:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
        except TypeError:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = getattr(outputs, "hidden_states", outputs)
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
        logits_plain = self.predict_plain(Z)
        logits_centered = self.predict_centered(Z)
        return Z, logits_plain, logits_centered

def _build_sister_mask(device: torch.device) -> torch.Tensor:
    """Builds an (18, 18) binary mask of sister models within the same model family."""
    M = torch.zeros((K, K), dtype=torch.float32, device=device)
    for fam_name, members in FAMILIES.items():
        for m1 in members:
            for m2 in members:
                if m1 != m2:
                    M[m1, m2] = 1.0
    return M

def family_margin_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    family_margin: float = 0.35,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Computes cross-entropy with intra-family margin penalty.
    Sister models within FAMILIES receive an additive logit penalty (+delta),
    forcing the network to establish a decisive decision boundary against sister LLMs.
    """
    if family_margin > 0.0:
        mask = _build_sister_mask(logits.device)
        boost = mask[labels] * family_margin
        logits = logits + boost
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

def compute_supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised Contrastive Cosine Loss on normalized representations.
    When duplicate classes exist, attracts same-class pairs while repelling different classes.
    When all classes in batch are distinct, penalizes positive cosine similarity among different models.
    """
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    z_norm = F.normalize(z, p=2, dim=-1)
    sim = torch.matmul(z_norm, z_norm.T) / max(temperature, 1e-4)
    diag_mask = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    label_mat = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = label_mat & ~diag_mask
    if pos_mask.sum() == 0:
        off_diag = ~diag_mask
        if off_diag.sum() > 0:
            return torch.clamp(sim[off_diag], min=0.0).mean()
        return torch.tensor(0.0, device=z.device)

    exp_sim = torch.exp(sim) * (~diag_mask).float()
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)
    log_prob = sim - torch.log(denom)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    valid_anchors = pos_mask.sum(dim=1) > 0
    return -mean_log_prob_pos[valid_anchors].mean()

def compute_two_stage_loss(
    model: MultiLayerLoRAClassifier,
    Z: torch.Tensor,
    labels: torch.Tensor,
    prompt_ids: Optional[List[Any]] = None,
    centered_weight: float = 0.5,
    family_margin: float = 0.35,
    supcon_weight: float = 0.0,
    label_smoothing: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Computes joint plain cross-entropy and within-prompt contrastive centered loss
    with intra-family margin penalties and optional supervised contrastive loss:
    L = L_plain + centered_weight * L_centered + supcon_weight * L_supcon
    """
    dev = Z.device
    labels_dev = labels.to(dev)

    if hasattr(model, "predict_plain"):
        logits_plain = model.predict_plain(Z)
    else:
        if next(model.plain_head.parameters()).device != dev:
            model.plain_head = model.plain_head.to(dev)
        logits_plain = model.plain_head(Z)

    loss_plain = family_margin_cross_entropy(logits_plain, labels_dev, family_margin=family_margin, label_smoothing=label_smoothing)

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

    if hasattr(model, "predict_centered"):
        logits_centered = model.predict_centered(Z_c)
    else:
        if next(model.centered_head.parameters()).device != dev:
            model.centered_head = model.centered_head.to(dev)
        logits_centered = model.centered_head(Z_c)

    loss_centered = family_margin_cross_entropy(logits_centered, labels_dev, family_margin=family_margin, label_smoothing=label_smoothing)

    total_loss = loss_plain + float(centered_weight) * loss_centered
    if supcon_weight > 0.0:
        loss_sup = compute_supcon_loss(Z_c, labels_dev)
        total_loss = total_loss + float(supcon_weight) * loss_sup

    return total_loss, loss_plain, loss_centered

class PromptGroupBatchSampler(Sampler):
    """
    Groups samples sharing the same prompt_id into complete batches (one prompt group per step)
    to enable two-pass micro-batched within-prompt contrastive centering during Stage 2 training.
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
        for p in prompts:
            idxs = list(self.prompt_to_indices[p])
            if self.shuffle:
                rng.shuffle(idxs)
            yield idxs

    def __len__(self) -> int:
        return len(self.prompts)

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
        item: Dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.prompt_ids is not None:
            item["prompt_id"] = self.prompt_ids[idx]
        return item

def collate_text_batch(batch: List[Dict[str, Any]], pad_token_id: int = 151643) -> Dict[str, Any]:
    input_ids = [item["input_ids"] for item in batch]
    masks = [item.get("attention_mask", torch.ones_like(item["input_ids"])) for item in batch]
    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    res: Dict[str, Any] = {"input_ids": padded_ids, "attention_mask": padded_mask}
    if "label" in batch[0]:
        res["labels"] = torch.stack([item["label"] for item in batch])
    if "prompt_id" in batch[0]:
        res["prompt_ids"] = [item["prompt_id"] for item in batch]
    return res

class BucketedLengthBatchSampler(Sampler):
    """
    Batches sequences of similar lengths together to minimize padding waste,
    while shuffling batches for stochastic gradient descent.
    """
    def __init__(self, lengths: List[int], batch_size: int, bucket_size_multiplier: int = 50, shuffle: bool = True, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            sorted_bucket = bucket[np.argsort([self.lengths[idx] for idx in bucket])]
            for b in range(0, len(sorted_bucket), self.batch_size):
                batches.append(list(sorted_bucket[b:b + self.batch_size]))

        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

def train_prompt_group_step(
    model: MultiLayerLoRAClassifier,
    opt: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    micro_batch_size: int = 2,
    centered_weight: float = 0.5,
    family_margin: float = 0.35,
    supcon_weight: float = 0.20,
    label_smoothing: float = 0.10
) -> Tuple[float, float, float]:
    """
    Two-pass micro-batched training step for a single prompt group.
    Pass 1: Computes exact within-prompt mean vector mu_prompt without gradients in chunks of 4.
    Pass 2: Computes plain and centered cross-entropy losses over micro-batches of micro_batch_size,
            accumulating gradients directly with dynamic sequence length slicing, AMP fp16,
            family margin penalties, and supervised contrastive loss.
    """
    N = input_ids.shape[0]
    dev = input_ids.device
    labels_dev = labels.to(dev)

    # Pass 1: Compute mu_prompt with no_grad in chunks of 4
    z_chunks = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
            for s in range(0, N, 4):
                e = min(s + 4, N)
                mb_mask = attention_mask[s:e]
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                z_c = model.forward_features(input_ids[s:e, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                z_chunks.append(z_c)
    Z_all_no_grad = torch.cat(z_chunks, dim=0)
    mu_prompt = Z_all_no_grad.mean(dim=0, keepdim=True)
    del z_chunks, Z_all_no_grad

    # Pass 2: Autograd micro-batches with AMP
    total_loss_val = 0.0
    plain_loss_val = 0.0
    centered_loss_val = 0.0

    model.train()
    for s in range(0, N, micro_batch_size):
        e = min(s + micro_batch_size, N)
        mb_mask = attention_mask[s:e]
        max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
        mb_ids = input_ids[s:e, :max_l]
        mb_mask_slice = mb_mask[:, :max_l]

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
            Z_mb = model.forward_features(mb_ids, mb_mask_slice, drop_first_token=True)
            mb_dev = Z_mb.device

            logits_p = model.predict_plain(Z_mb)
            loss_p = family_margin_cross_entropy(logits_p, labels_dev[s:e].to(mb_dev), family_margin=family_margin, label_smoothing=label_smoothing)

            Z_c_mb = Z_mb - mu_prompt.to(mb_dev)
            logits_c = model.predict_centered(Z_c_mb)
            loss_c = family_margin_cross_entropy(logits_c, labels_dev[s:e].to(mb_dev), family_margin=family_margin, label_smoothing=label_smoothing)

            loss_mb = loss_p + float(centered_weight) * loss_c
            if supcon_weight > 0.0 and (e - s) > 1:
                loss_sup = compute_supcon_loss(Z_c_mb, labels_dev[s:e].to(mb_dev))
                loss_mb = loss_mb + float(supcon_weight) * loss_sup

            weight_frac = (e - s) / float(N)
            scaled_loss = loss_mb * weight_frac

        scaled_loss.backward()

        total_loss_val += loss_mb.item() * weight_frac
        plain_loss_val += loss_p.item() * weight_frac
        centered_loss_val += loss_c.item() * weight_frac

    return total_loss_val, plain_loss_val, centered_loss_val

def fit_ridge_heads(
    Z: np.ndarray,
    y: np.ndarray,
    prompts: Optional[np.ndarray] = None,
    alpha: float = 10.0,
    label_smoothing: float = 0.10,
    num_classes: int = K
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits closed-form dual Ridge regression heads for plain and centered representations
    with label smoothing. Uses dual formulation (N x N) for ultra-fast, numerically stable solution.
    """
    N, D = Z.shape
    Y = np.zeros((N, num_classes), dtype=np.float32)
    for i, lab in enumerate(y):
        Y[i, lab] = 1.0
    if label_smoothing > 0.0:
        Y = (1.0 - label_smoothing) * Y + (label_smoothing / num_classes)

    # Plain head: Z_tilde = [Z, 1]
    Z_tilde = np.hstack([Z, np.ones((N, 1), dtype=np.float32)])
    K_mat = Z_tilde @ Z_tilde.T
    K_mat.flat[::N + 1] += alpha
    A = np.linalg.solve(K_mat, Y)
    W_tilde = (A.T @ Z_tilde).astype(np.float32)
    W_p = W_tilde[:, :-1]
    b_p = W_tilde[:, -1]

    # Centered head: Z_c
    Z_c = Z.copy()
    if prompts is not None:
        u_p = np.unique(prompts)
        for p in u_p:
            mask = (prompts == p)
            if mask.sum() > 1:
                Z_c[mask] = Z[mask] - Z[mask].mean(axis=0, keepdims=True)
            else:
                Z_c[mask] = 0.0

    Z_c_tilde = np.hstack([Z_c, np.ones((N, 1), dtype=np.float32)])
    K_mat_c = Z_c_tilde @ Z_c_tilde.T
    K_mat_c.flat[::N + 1] += alpha
    A_c = np.linalg.solve(K_mat_c, Y)
    W_c_tilde = (A_c.T @ Z_c_tilde).astype(np.float32)
    W_c = W_c_tilde[:, :-1]
    b_c = W_c_tilde[:, -1]

    return W_p, b_p, W_c, b_c

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
    Stage 1: Multi-corpus pretraining on up to 90k passages with family margin loss.
    Stage 2: Prompt-grouped contrastive fine-tuning on train.csv with SupCon loss.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    if train_cfg is None:
        train_cfg = {}

    backbone_name = train_cfg.get("backbone", "Qwen/Qwen3-14B-Base")
    keep_layers = int(train_cfg.get("keep_layers", 27))
    layers = list(train_cfg.get("layers", [2, 4, 6, 9, 13, 17, 22, 27]))
    max_len = int(train_cfg.get("max_len", 512))
    end_id = int(train_cfg.get("end_id", 151643))
    out_dir = Path(train_cfg.get("output_dir", "export_champion_qwen3_14b"))
    family_margin = float(train_cfg.get("family_margin", 0.35))
    supcon_weight = float(train_cfg.get("supcon_weight", 0.20))
    is_deberta = "deberta" in backbone_name.lower()

    input_dir = Path(train_cfg.get("input_dir", "/kaggle/input"))
    local_base = None
    if input_dir.exists():
        if is_deberta:
            for cfg_p in input_dir.rglob("config.json"):
                try:
                    c_data = json.load(open(cfg_p))
                    if "deberta" in str(c_data.get("model_type", "")).lower():
                        local_base = cfg_p.parent
                        break
                except Exception:
                    continue
        elif "2.5" in backbone_name.lower() or "qwen25" in backbone_name.lower():
            local_base = find_base_model(input_dir, "qwen2", 5120, 48) or find_base_model(input_dir, "qwen2", 3584, 28)
        elif "14b" in backbone_name.lower():
            local_base = find_base_model(input_dir, "qwen3", 5120, 40) or find_base_model(input_dir, "qwen2", 5120, 48)
        elif "8b" in backbone_name.lower():
            local_base = find_base_model(input_dir, "qwen3", 4096, 36)
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

    _disable_final_norm(base_model)

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    default_targets = (
        ["query_proj", "key_proj", "value_proj", "dense"]
        if is_deberta
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_cfg = LoraConfig(
        r=int(train_cfg.get("lora_r", 32)),
        lora_alpha=int(train_cfg.get("lora_alpha", 64)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        target_modules=list(train_cfg.get("lora_target_modules", default_targets)),
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    base_model = get_peft_model(base_model, lora_cfg)

    if hasattr(base_model, "gradient_checkpointing_enable"):
        try:
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base_model.gradient_checkpointing_enable()
    for mod in base_model.modules():
        if hasattr(mod, "gradient_checkpointing"):
            setattr(mod, "gradient_checkpointing", True)

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
        pool_list = all_pools if all_pools else ([pool_file] if pool_file else [])
        print(f"[Trainer] Stage 1 Pretraining on {len(pool_list)} pool file(s)...")
        dfs = []
        for p in pool_list:
            try:
                if str(p).endswith(".parquet"):
                    df_one = pd.read_parquet(p, columns=["text", "model"])
                else:
                    df_one = pd.read_csv(p, usecols=["text", "model"])
                dfs.append(df_one)
            except Exception as e:
                print(f"[Trainer] Could not read pool {p}: {e}")

        if dfs:
            df_p = pd.concat(dfs, ignore_index=True)
        else:
            df_p = pd.DataFrame(columns=["text", "model"])

        df_p = df_p.dropna(subset=["text", "model"])
        df_p["text"] = df_p["text"].astype(str)
        df_p = df_p[df_p["text"].str.strip().str.len() >= 40]
        df_p = df_p[df_p["model"].isin(LABELS)]

        df_p["_hash"] = [_text_hash(t) for t in df_p["text"]]
        df_p = df_p.drop_duplicates(subset=["_hash"]).reset_index(drop=True)

        # Build unseen holdout to quarantine holdout passages from Stage 1 pretraining
        try:
            _, holdout_hashes = build_unseen_holdout_dataset(input_dir, train_path)
            if holdout_hashes:
                mask_clean = ~df_p["_hash"].isin(holdout_hashes)
                n_dropped = len(df_p) - mask_clean.sum()
                if n_dropped > 0:
                    print(f"[Trainer] Quarantined {n_dropped} pool rows that overlapped with unseen holdout.")
                df_p = df_p[mask_clean].reset_index(drop=True)
        except Exception as e:
            print(f"[Trainer] Note on holdout quarantine: {e}")

        max_s = int(stage1_cfg.get("max_samples", 90000))
        if len(df_p) > max_s:
            target_per_c = max_s // K
            sampled = []
            remainder = []
            for c, group in df_p.groupby("model"):
                if len(group) <= target_per_c:
                    sampled.append(group)
                else:
                    s_grp = group.sample(n=target_per_c, random_state=42)
                    sampled.append(s_grp)
                    remainder.append(group.drop(s_grp.index))
            res_df = pd.concat(sampled, ignore_index=True)
            needed = max_s - len(res_df)
            if needed > 0 and remainder:
                rem_df = pd.concat(remainder, ignore_index=True)
                take_extra = rem_df.sample(n=min(needed, len(rem_df)), random_state=42)
                res_df = pd.concat([res_df, take_extra], ignore_index=True)
            df_p = res_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            df_p = df_p.sample(frac=1.0, random_state=42).reset_index(drop=True)

        lengths = [len(tok(t, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]) for t in df_p["text"]]
        bs1 = int(stage1_cfg.get("batch_size", 2))
        accum1 = int(stage1_cfg.get("gradient_accumulation_steps", 8))
        ds1 = TextSequenceDataset(df_p["text"].tolist(), [LAB2I[m] for m in df_p["model"]], None, tok, max_len=max_len, end_id=end_id)
        bucket_sampler = BucketedLengthBatchSampler(lengths, batch_size=bs1, shuffle=True)
        loader1 = DataLoader(ds1, batch_sampler=bucket_sampler, collate_fn=collate_text_batch)

        opt1 = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": float(stage1_cfg.get("lr_lora", 1.5e-4))},
            {"params": list(model.norms.parameters()) + list(model.plain_head.parameters()), "lr": float(stage1_cfg.get("lr_head", 8e-4))}
        ], weight_decay=float(train_cfg.get("weight_decay", 0.01)))
        sched1 = get_cosine_schedule_with_warmup(opt1, num_warmup_steps=int(stage1_cfg.get("warmup_steps", 100)), num_training_steps=math.ceil(len(loader1) / accum1))

        model.train()
        opt1.zero_grad()
        for step_idx, batch in enumerate(loader1):
            mb_ids = batch["input_ids"].to(device)
            mb_mask = batch["attention_mask"].to(device)
            max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
            mb_ids = mb_ids[:, :max_l]
            mb_mask = mb_mask[:, :max_l]
            labels_dev = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                _, logits_p, _ = model(mb_ids, mb_mask, drop_first_token=not is_deberta)
                loss = family_margin_cross_entropy(logits_p, labels_dev, family_margin=family_margin) / accum1

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
    ds2 = TextSequenceDataset(df_tr["text"].tolist(), [LAB2I[m] for m in df_tr["model"]], df_tr["prompt_id"].tolist(), tok, max_len=max_len, end_id=end_id)
    sampler2 = PromptGroupBatchSampler(df_tr["prompt_id"].tolist(), batch_size=bs2, shuffle=True)
    loader2 = DataLoader(ds2, batch_sampler=sampler2, collate_fn=collate_text_batch)

    opt2 = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": float(stage2_cfg.get("lr_lora", 3e-5))},
        {"params": list(model.norms.parameters()) + list(model.plain_head.parameters()) + list(model.centered_head.parameters()), "lr": float(stage2_cfg.get("lr_head", 2e-4))}
    ], weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    epochs2 = int(stage2_cfg.get("epochs", 2))
    total_steps2 = len(loader2) * epochs2
    sched2 = get_cosine_schedule_with_warmup(opt2, num_warmup_steps=int(stage2_cfg.get("warmup_steps", 50)), num_training_steps=total_steps2)

    model.train()
    for ep in range(epochs2):
        sampler2.set_epoch(ep)
        opt2.zero_grad()
        for step_idx, batch in enumerate(loader2):
            train_prompt_group_step(
                model=model,
                opt=opt2,
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                micro_batch_size=bs2,
                centered_weight=float(stage2_cfg.get("centered_weight", 0.60)),
                family_margin=family_margin,
                supcon_weight=supcon_weight,
                label_smoothing=0.10
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            sched2.step()
            opt2.zero_grad()

    # Feature extraction for wPCA bank & checkpoint export
    del loader2, opt2, sched2, sampler2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    loader_eval = DataLoader(ds2, batch_size=4, shuffle=False, collate_fn=collate_text_batch)
    Z_list = []
    with torch.no_grad():
        for batch in loader_eval:
            z_b = model.forward_features(batch["input_ids"].to(device), batch["attention_mask"].to(device), drop_first_token=not is_deberta)
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
