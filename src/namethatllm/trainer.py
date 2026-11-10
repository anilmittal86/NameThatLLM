"""
Two-Stage Multi-Layer LoRA Training Engine
==========================================
Implements production Two-Stage Multi-Layer LoRA training with 8 intermediate
LayerNorm heads, causal attention-sink masking (m[:, 0] = 0.0), prompt-grouped
contrastive centering, and standard checkpoint export.
"""

import gc
import importlib.util
import json
import math
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .ablation import _text_hash, build_unseen_holdout_dataset
from .config import (
    FAMILIES,
    HARD_CLASS_INDICES,
    LAB2I,
    LABELS,
    K,
    find_base_model,
    resolve_all_pool_files,
    resolve_dataset_paths,
    softmax,
)
from .postprocess import sinkhorn_balancing
from .wpca_gaussian import fit_shared_gaussian_lowram


def _setup_training_env() -> None:
    """Configures PyTorch memory allocator and neutralizes incompatible packages."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    _orig_find_spec = importlib.util.find_spec

    def _safe_find_spec(name, *args, **kwargs):
        if name == "torchao":
            return None
        return _orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _safe_find_spec

    try:
        import peft.import_utils

        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass


# Backward compatibility alias
_configure_training_environment = _setup_training_env


def _disable_final_norm(mod: nn.Module) -> None:
    """
    Replaces the model's final LayerNorm / RMSNorm with nn.Identity().

    Why This Is Done:
    -----------------
    In standard autoregressive language models (Qwen, Llama), the final normalization
    layer immediately precedes the vocabulary lm_head and is tuned specifically for
    next-token token prediction. For sequence classification using intermediate hidden states
    from earlier layers (e.g. layers 9, 13, 17, 21, 25), the final norm is either bypassed
    or distorts the pooled representation. Replacing it with Identity avoids unintended side-effects.
    """
    for path in ["norm", "model.norm", "base_model.norm", "base_model.model.norm", "base_model.model.model.norm"]:
        curr = mod
        parts = path.split(".")
        for p in parts[:-1]:
            curr = getattr(curr, p, None)
            if curr is None:
                break
        if curr is not None and hasattr(curr, parts[-1]) and getattr(curr, parts[-1]) is not None:
            setattr(curr, parts[-1], nn.Identity())


class MultiLayerLoRAClassifier(nn.Module):
    """
    Universal Multi-Layer LoRA Sequence Classifier.

    Key Architectural Components:
    ------------------------------
    1. Intermediate Layer Extraction:
       Instead of relying only on the final transformer layer, hidden states are pulled
       from multiple intermediate layers (e.g., 8 evenly spaced layers). Intermediate layers
       capture rich lexical, syntactic, and stylistic patterns before they get compressed
       into next-token representations.
    2. Per-Layer LayerNorm:
       Each selected layer has a dedicated nn.LayerNorm to normalize feature scales
       prior to concatenation into a single (num_layers * hidden_size) feature vector Z.
    3. Dual Classification Heads:
       - plain_head: Standard linear projection from Z -> num_classes.
       - centered_head: Linear projection trained on prompt-centered features (Z - Z_prompt_mean)
         to decouple pure LLM writing style from the prompt's topical content.
    4. 5-Branch Multi-Sample Dropout (MSD):
       During training, Z is passed through 5 distinct dropout masks (p = 0.1, 0.2, 0.3, 0.4, 0.5),
       and the outputs are averaged. This accelerates training convergence and acts as an
       implicit ensemble of regularized classifiers.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, layers: list[int], num_classes: int = K, keep_layers: int | None = None):
        super().__init__()
        self.backbone = backbone
        _disable_final_norm(self.backbone)
        self.layers = list(layers)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.total_dim = len(layers) * hidden_size

        # Dedicated LayerNorm for each extracted intermediate layer
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers])

        # Dual classification heads (Plain + Prompt-Centered Twin)
        self.plain_head = nn.Linear(self.total_dim, num_classes)
        self.centered_head = nn.Linear(self.total_dim, num_classes)

        # 5-branch Multi-Sample Dropout for superior regularization and gradient stability
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])

    def predict_plain(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Projects concatenated feature vector Z to logits via the plain head.
        Averages across 5 Multi-Sample Dropout branches during training.
        """
        first_dev = Z.device
        if next(self.plain_head.parameters()).device != first_dev:
            self.plain_head = self.plain_head.to(first_dev)
        if self.training:
            return torch.mean(torch.stack([self.plain_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.plain_head(Z)

    def predict_centered(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Projects prompt-centered feature vector Z to logits via the centered twin head.
        Averages across 5 Multi-Sample Dropout branches during training.
        """
        first_dev = Z.device
        if next(self.centered_head.parameters()).device != first_dev:
            self.centered_head = self.centered_head.to(first_dev)
        if self.training:
            return torch.mean(torch.stack([self.centered_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.centered_head(Z)

    def forward_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, drop_first_token: bool = True) -> torch.Tensor:
        """
        Extracts pooled multi-layer representations with causal attention-sink masking.

        Why Attention-Sink Masking (drop_first_token) Matters:
        ------------------------------------------------------
        In causal LLMs (Qwen, Llama), the first token (token index 0, e.g. BOS or leading word)
        acts as an 'attention sink' where the softmax mechanism dumps massive attention logits
        regardless of whether token 0 carries semantic meaning. If token 0 is included in
        mean pooling across the sequence length, it injects noise that masks subtle stylistic
        differences. Setting m[:, 0] = 0.0 cleans the representation.
        """
        m = attention_mask.clone().float()
        if drop_first_token and m.shape[1] > 1:
            m[:, 0] = 0.0  # Zero out attention sink token
        den = m.sum(dim=1, keepdim=True).clamp(min=1e-5)
        first_device = input_ids.device

        try:
            outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, use_cache=False
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
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, drop_first_token: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = self.forward_features(input_ids, attention_mask, drop_first_token=drop_first_token)
        logits_plain = self.predict_plain(Z)
        logits_centered = self.predict_centered(Z)
        return Z, logits_plain, logits_centered


def _build_sister_mask(device: torch.device, boosted: bool = False) -> torch.Tensor:
    """
    Builds an (18, 18) indicator matrix M where M[i, j] > 0.0 if class i and
    class j belong to the same LLM provider family (sisters), and 0.0 otherwise.
    When boosted=True, applies empirically validated sibling margin penalties to the highest-error sister pairs:
      - Gemini Flash Triad (3.7-flash <-> 3.8-flash @ 1.5x, 3.6-flash @ 1.4x)
      - GPT-5.6 Triad (luna <-> sol <-> terra @ 1.4x, sol <-> astra @ 1.3x)
      - GLM Sisters (glm-5.3 <-> glm-5.3-flash @ 1.4x)
    Used by family_margin_cross_entropy to identify and decisively separate competing sibling classes.
    """
    M = torch.zeros((K, K), dtype=torch.float32, device=device)
    for fam_name, members in FAMILIES.items():
        for m1 in members:
            for m2 in members:
                if m1 != m2:
                    M[m1, m2] = 1.0

    if boosted:
        # 1. Gemini Flash Triad (indices 6, 7, 8)
        M[7, 8] = 1.5
        M[8, 7] = 1.5
        M[6, 7] = 1.4
        M[7, 6] = 1.4
        M[6, 8] = 1.4
        M[8, 6] = 1.4

        # 2. GPT-5.6 Triad (indices 1, 2, 3) & Astra (4)
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i != j:
                    M[i, j] = 1.4
        M[2, 4] = 1.3
        M[4, 2] = 1.3
        M[1, 4] = 1.2
        M[4, 1] = 1.2

        # 3. GLM Sisters (indices 15, 16)
        M[15, 16] = 1.4
        M[16, 15] = 1.4

    return M


def family_margin_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    family_margin: float = 0.35,
    label_smoothing: float = 0.0,
    weight: torch.Tensor | None = None,
    boost_hard_sisters: bool = True,
) -> torch.Tensor:
    """
    Computes Cross-Entropy with Sibling Margin Penalty (Additive Margin on Sister Classes).

    Concept & Mathematical Intuition:
    ---------------------------------
    In LLM author identification, confusing a Gemini model with a Claude model is a rare
    mistake; the overwhelming source of error is confusing sister variants (e.g., Gemini 3.7
    vs Gemini 3.8, or GLM-5.3 vs GLM-5.3-flash). Standard cross-entropy treats all negative
    classes symmetrically.

    By adding a positive margin penalty (+family_margin) to the logits of sibling classes:
        logits[i, sister_j] += margin_boost
    the softmax denominator is artificially inflated by the sister's logits. This penalizes
    the network whenever it produces ambiguous or overlapping predictions between sisters,
    forcing the gradients to carve out a sharp, decisive separation boundary.

    Specialist Sibling Boost:
    -------------------------
    - For single-family specialists (e.g. Gemini 4-class), all classes are sisters.
    - We apply an extra 1.5x multiplier to the most difficult sibling pair:
      Gemini 3.7-flash <-> Gemini 3.8-flash, which prevents stylistic collapse.
    - For GPT 5-class specialist: 1.4x boost among the triad (Luna, Sol, Terra).
    """
    if family_margin > 0.0:
        num_cls = logits.shape[1]
        if num_cls == K:
            mask = _build_sister_mask(logits.device, boosted=boost_hard_sisters)
        else:
            # Single-family specialist: all classes are sisters of each other
            mask = 1.0 - torch.eye(num_cls, dtype=torch.float32, device=logits.device)
            # Extra margin boost on the hardest sibling pairs:
            if num_cls == 4:  # Gemini: boost 3.7-flash (2) <-> 3.8-flash (3)
                mask[2, 3] = 1.5
                mask[3, 2] = 1.5
            elif num_cls == 5:  # GPT: boost luna (1) <-> sol (2) <-> terra (3)
                for i in (1, 2, 3):
                    for j in (1, 2, 3):
                        if i != j:
                            mask[i, j] = 1.4
        boost = mask[labels] * family_margin
        logits = logits + boost
    return F.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing)


def compute_supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10, **kwargs) -> torch.Tensor:
    """
    Supervised contrastive loss with support for unique-class prompt groups.
    When multiple samples share the same class, performs standard supervised contrastive alignment.
    When all samples in the batch have distinct classes (pos_mask.sum() == 0, e.g. Stage 2 prompt groups),
    repels distinct authors using unshifted pairwise cosine penalty: clamp(cos + 0.05, min=0.0)^2.
    Also aligns against class prototypes if provided via prototypes/centered_head_weight.
    """
    if z.shape[0] < 2:
        return (z * 0.0).sum()

    dev = z.device
    z_norm = F.normalize(z.float(), p=2, dim=-1, eps=1e-6)
    raw_cos = torch.matmul(z_norm, z_norm.T)
    diag_mask = torch.eye(z.shape[0], device=dev, dtype=torch.bool)

    labels_dev = labels.to(dev)
    label_mat = labels_dev.unsqueeze(0) == labels_dev.unsqueeze(1)
    pos_mask = label_mat & ~diag_mask

    prototypes = kwargs.get("prototypes")
    if prototypes is None:
        prototypes = kwargs.get("centered_head_weight")

    if pos_mask.sum() == 0:
        off_diag = ~diag_mask
        if off_diag.sum() > 0:
            loss = torch.clamp(raw_cos[off_diag] + 0.05, min=0.0).pow(2).mean()
        else:
            loss = (z * 0.0).sum()

        if prototypes is not None:
            w_norm = F.normalize(prototypes.float().to(dev), p=2, dim=-1, eps=1e-6)
            sim_proto = (z_norm @ w_norm.T) / max(float(temperature), 1e-3)
            loss = loss + 0.5 * F.cross_entropy(sim_proto, labels_dev)

        Z_group = kwargs.get("Z_group_centered")
        if Z_group is not None and Z_group.shape[0] > z.shape[0]:
            g_norm = F.normalize(Z_group.detach().float().to(dev), p=2, dim=-1, eps=1e-6)
            sim_group = z_norm @ g_norm.T
            loss = loss + 0.10 * (sim_group.pow(2).mean())
        return loss

    sim = raw_cos / max(float(temperature), 1e-4)
    sim = sim.masked_fill(diag_mask, -1e4)
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim) * (~diag_mask).float()
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)
    log_prob = sim - torch.log(denom)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    valid_anchors = pos_mask.sum(dim=1) > 0
    loss = -mean_log_prob_pos[valid_anchors].mean()

    if prototypes is not None:
        w_norm = F.normalize(prototypes.float().to(dev), p=2, dim=-1, eps=1e-6)
        sim_proto = (z_norm @ w_norm.T) / max(float(temperature), 1e-3)
        loss = loss + 0.5 * F.cross_entropy(sim_proto, labels_dev)

    return loss


def compute_two_stage_loss(
    model: MultiLayerLoRAClassifier,
    Z: torch.Tensor,
    labels: torch.Tensor,
    prompt_ids: list[Any] | None = None,
    centered_weight: float = 0.5,
    family_margin: float = 0.35,
    supcon_weight: float = 0.0,
    label_smoothing: float = 0.0,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Computes joint plain cross-entropy and within-prompt contrastive centered loss
    with intra-family margin penalties, optional supervised contrastive loss, and hard-class loss weighting:
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

    loss_plain = family_margin_cross_entropy(logits_plain, labels_dev, family_margin=family_margin, label_smoothing=label_smoothing, weight=weight)

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

    loss_centered = family_margin_cross_entropy(logits_centered, labels_dev, family_margin=family_margin, label_smoothing=label_smoothing, weight=weight)

    total_loss = loss_plain + float(centered_weight) * loss_centered
    if supcon_weight > 0.0:
        loss_sup = compute_supcon_loss(Z_c, labels_dev, prototypes=model.centered_head.weight)
        total_loss = total_loss + float(supcon_weight) * loss_sup

    return total_loss, loss_plain, loss_centered


class PromptGroupBatchSampler(Sampler):
    """
    Groups samples sharing the same prompt_id into complete batches (one prompt group per step)
    to enable two-pass micro-batched within-prompt contrastive centering during Stage 2 training.
    """

    def __init__(self, prompt_ids: list[Any], batch_size: int = 18, shuffle: bool = True, seed: int = 42):
        self.prompt_to_indices: dict[Any, list[int]] = {}
        for idx, p in enumerate(prompt_ids):
            self.prompt_to_indices.setdefault(p, []).append(idx)
        self.prompts = list(self.prompt_to_indices.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
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
        self, texts: list[str], labels: list[int] | None, prompt_ids: list[Any] | None, tok: Any, max_len: int = 447, end_id: int = 151643
    ):
        self.texts = texts
        self.labels = labels
        self.prompt_ids = prompt_ids
        self.tok = tok
        self.max_len = max_len
        self.end_id = end_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw_ids = self.tok(str(self.texts[idx]), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        input_ids = raw_ids + [self.end_id]
        item: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.prompt_ids is not None:
            item["prompt_id"] = self.prompt_ids[idx]
        return item


def collate_text_batch(batch: list[dict[str, Any]], pad_token_id: int = 151643) -> dict[str, Any]:
    input_ids = [item["input_ids"] for item in batch]
    masks = [item.get("attention_mask", torch.ones_like(item["input_ids"])) for item in batch]
    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    res: dict[str, Any] = {"input_ids": padded_ids, "attention_mask": padded_mask}
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

    def __init__(self, lengths: list[int], batch_size: int, bucket_size_multiplier: int = 50, shuffle: bool = True, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i : i + self.bucket_size]
            sorted_bucket = bucket[np.argsort([self.lengths[idx] for idx in bucket])]
            for b in range(0, len(sorted_bucket), self.batch_size):
                batches.append(list(sorted_bucket[b : b + self.batch_size]))

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
    label_smoothing: float = 0.10,
    accum_steps: int = 1,
    compute_dtype: torch.dtype = torch.float16,
    weight: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """
    Two-pass micro-batched training step for a single prompt group.
    Pass 1: Computes exact within-prompt mean vector mu_prompt without gradients in chunks of 4.
    Pass 2: Computes plain and centered cross-entropy losses over micro-batches of micro_batch_size,
            accumulating gradients directly with dynamic sequence length slicing, AMP fp16/bf16,
            family margin penalties, hard-class loss upweighting, and supervised contrastive loss.
    """
    N = input_ids.shape[0]
    dev = input_ids.device
    labels_dev = labels.to(dev)

    # Pass 1: Compute mu_prompt with no_grad in chunks of 4
    z_chunks = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            for s in range(0, N, 4):
                e = min(s + 4, N)
                mb_mask = attention_mask[s:e]
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                z_c = model.forward_features(input_ids[s:e, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                z_chunks.append(z_c)
    Z_all_no_grad = torch.cat(z_chunks, dim=0)
    # Variable-Size Sub-Cluster Centering Augmentation:
    # When N >= 8, with 50% probability compute mu_prompt using a random subset of k ~ Uniform(4, min(N, 14))
    # rows from Z_all_no_grad so centered_head generalizes to smaller test-time prompt clusters.
    if N >= 8 and torch.rand(1).item() < 0.5:
        max_k = min(N, 14)
        k = int(torch.randint(4, max_k + 1, (1,)).item())
        sub_indices = torch.randperm(N, device=Z_all_no_grad.device)[:k]
        mu_prompt = Z_all_no_grad[sub_indices].mean(dim=0, keepdim=True)
    else:
        mu_prompt = Z_all_no_grad.mean(dim=0, keepdim=True)
    Z_group_centered = (Z_all_no_grad - mu_prompt).detach()
    del z_chunks

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

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            Z_mb = model.forward_features(mb_ids, mb_mask_slice, drop_first_token=True)
            mb_dev = Z_mb.device

            logits_p = model.predict_plain(Z_mb)
            loss_p = family_margin_cross_entropy(
                logits_p, labels_dev[s:e].to(mb_dev), family_margin=family_margin, label_smoothing=label_smoothing, weight=weight
            )

            Z_c_mb = Z_mb - mu_prompt.to(mb_dev)
            logits_c = model.predict_centered(Z_c_mb)
            loss_c = family_margin_cross_entropy(
                logits_c, labels_dev[s:e].to(mb_dev), family_margin=family_margin, label_smoothing=label_smoothing, weight=weight
            )

            loss_mb = loss_p + float(centered_weight) * loss_c
            if supcon_weight > 0.0:
                loss_sup = compute_supcon_loss(
                    Z_c_mb, labels_dev[s:e], temperature=0.10, prototypes=model.centered_head.weight, Z_group_centered=Z_group_centered
                )
                loss_mb = loss_mb + float(supcon_weight) * loss_sup

            weight_frac = (e - s) / float(N)
            scaled_loss = (loss_mb * weight_frac) / float(max(accum_steps, 1))

        scaled_loss.backward()

        total_loss_val += loss_mb.item() * weight_frac
        plain_loss_val += loss_p.item() * weight_frac
        centered_loss_val += loss_c.item() * weight_frac

    return total_loss_val, plain_loss_val, centered_loss_val


def fit_ridge_heads(
    Z: np.ndarray,
    y: np.ndarray,
    prompts: np.ndarray | None = None,
    alpha: float = 10.0,
    label_smoothing: float = 0.10,
    num_classes: int = K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    K_mat.flat[:: N + 1] += alpha
    A = np.linalg.solve(K_mat, Y)
    W_tilde = (A.T @ Z_tilde).astype(np.float32)
    W_p = W_tilde[:, :-1]
    b_p = W_tilde[:, -1]

    # Centered head: Z_c
    Z_c = Z.copy()
    if prompts is not None:
        u_p = np.unique(prompts)
        for p in u_p:
            mask = prompts == p
            if mask.sum() > 1:
                Z_c[mask] = Z[mask] - Z[mask].mean(axis=0, keepdims=True)
            else:
                Z_c[mask] = 0.0

    Z_c_tilde = np.hstack([Z_c, np.ones((N, 1), dtype=np.float32)])
    K_mat_c = Z_c_tilde @ Z_c_tilde.T
    K_mat_c.flat[:: N + 1] += alpha
    A_c = np.linalg.solve(K_mat_c, Y)
    W_c_tilde = (A_c.T @ Z_c_tilde).astype(np.float32)
    W_c = W_c_tilde[:, :-1]
    b_c = W_c_tilde[:, -1]

    return W_p, b_p, W_c, b_c


def _blend_head(head_linear: nn.Linear, W_r: np.ndarray, b_r: np.ndarray, w_neural: float = 0.65, w_ridge: float = 0.35) -> None:
    """Blends neural head with closed-form ridge head after matching Frobenius norm."""
    with torch.no_grad():
        W_n = head_linear.weight.detach().cpu().float().numpy()
        b_n = head_linear.bias.detach().cpu().float().numpy()
        norm_n = float(np.linalg.norm(W_n))
        norm_r = float(np.linalg.norm(W_r))
        if norm_r > 1e-6 and norm_n > 1e-6:
            W_r_scaled = W_r * (norm_n / norm_r)
        else:
            W_r_scaled = W_r
        W_blend = w_neural * W_n + w_ridge * W_r_scaled
        b_blend = w_neural * b_n + w_ridge * b_r
        head_linear.weight.copy_(torch.from_numpy(W_blend).to(head_linear.weight.device, dtype=head_linear.weight.dtype))
        head_linear.bias.copy_(torch.from_numpy(b_blend).to(head_linear.bias.device, dtype=head_linear.bias.dtype))


def evaluate_stage2_oof_cv(
    Z: np.ndarray,
    y: np.ndarray,
    prompts: np.ndarray,
    n_splits: int = 5,
    alpha: float = 10.0,
    label_smoothing: float = 0.10,
    num_classes: int = K,
    class_labels: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Executes 5-Stage GroupKFold Cross-Validation on Stage 2 neural representations
    grouped strictly by prompt_id to evaluate out-of-fold generalization and check overfitting.

    Why This Is Done:
    -----------------
    In authorship attribution, autoregressive models can memorize prompt topics rather than
    learning stylistic traits. Evaluating on the training prompts produces artificially high
    accuracy (e.g. 95%+). By partitioning prompts into 5 disjoint folds, we evaluate on
    passages from prompts the heads never saw during training, giving an honest, unbiased
    out-of-fold (OOF) accuracy and quantifying the exact Overfitting Gap (Train BAcc - Val BAcc).
    """
    u_prompts = np.unique(prompts)
    actual_splits = min(n_splits, len(u_prompts))
    if actual_splits < 2:
        print(f"[Stage 2 CV] Insufficient prompt groups ({len(u_prompts)}) for cross-validation; skipping.")
        return {}

    labels = class_labels or list(LABELS[:num_classes])
    gkf = GroupKFold(n_splits=actual_splits)

    oof_plain_lp = np.zeros((len(y), num_classes), dtype=np.float32)
    oof_centered_lp = np.zeros((len(y), num_classes), dtype=np.float32)
    oof_blend_lp = np.zeros((len(y), num_classes), dtype=np.float32)
    fold_results = []

    print("\n==========================================================================================")
    print(f" STAGE 2: {actual_splits}-STAGE GROUP-KFOLD CV & OVERFITTING AUDIT (grouped by prompt_id)")
    print("==========================================================================================")
    print(" Fold   | Train Prompts | Val Prompts | Train BAcc | Val Plain  | Val Center | Val Sink   | Overfit Gap")
    print("------------------------------------------------------------------------------------------")

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(Z, y, groups=prompts)):
        Z_tr, y_tr, p_tr = Z[tr_idx], y[tr_idx], prompts[tr_idx]
        Z_val, y_val, p_val = Z[val_idx], y[val_idx], prompts[val_idx]

        W_p, b_p, W_c, b_c = fit_ridge_heads(Z_tr, y_tr, p_tr, alpha=alpha, label_smoothing=label_smoothing, num_classes=num_classes)

        # Train evaluation (to measure overfitting gap)
        tr_plain = Z_tr @ W_p.T + b_p
        Z_tr_c = Z_tr.copy()
        for p in np.unique(p_tr):
            m = p_tr == p
            if m.sum() > 1:
                Z_tr_c[m] -= Z_tr_c[m].mean(axis=0, keepdims=True)
            else:
                Z_tr_c[m] = 0.0
        tr_centered = Z_tr_c @ W_c.T + b_c
        tr_blend = 0.5 * tr_plain + 0.5 * tr_centered
        train_bacc = float(balanced_accuracy_score(y_tr, tr_blend.argmax(axis=1)))

        # Validation evaluation (strictly unseen prompt groups)
        val_plain = Z_val @ W_p.T + b_p
        Z_val_c = Z_val.copy()
        for p in np.unique(p_val):
            m = p_val == p
            if m.sum() > 1:
                Z_val_c[m] -= Z_val_c[m].mean(axis=0, keepdims=True)
            else:
                Z_val_c[m] = 0.0
        val_centered = Z_val_c @ W_c.T + b_c
        val_blend = 0.5 * val_plain + 0.5 * val_centered

        val_plain_bacc = float(balanced_accuracy_score(y_val, val_plain.argmax(axis=1)))
        val_center_bacc = float(balanced_accuracy_score(y_val, val_centered.argmax(axis=1)))
        val_blend_bacc = float(balanced_accuracy_score(y_val, val_blend.argmax(axis=1)))

        # Val Soft Sinkhorn
        P_val = softmax(val_blend, axis=1)
        P_val_sk = sinkhorn_balancing(P_val, iters=60, sinkhorn_strength=0.65, target_mix=0.35)
        val_sink_bacc = float(balanced_accuracy_score(y_val, P_val_sk.argmax(axis=1)))

        gap = train_bacc - val_sink_bacc

        oof_plain_lp[val_idx] = val_plain
        oof_centered_lp[val_idx] = val_centered
        oof_blend_lp[val_idx] = val_blend

        fold_res = {
            "fold": fold_idx + 1,
            "train_prompts": int(len(np.unique(p_tr))),
            "val_prompts": int(len(np.unique(p_val))),
            "train_bacc": train_bacc,
            "val_plain_bacc": val_plain_bacc,
            "val_center_bacc": val_center_bacc,
            "val_blend_bacc": val_blend_bacc,
            "val_sink_bacc": val_sink_bacc,
            "overfit_gap": gap,
        }
        fold_results.append(fold_res)

        print(
            f" Fold {fold_idx + 1:<2} | {len(np.unique(p_tr)):<13} | {len(np.unique(p_val)):<11} | "
            f"{train_bacc * 100:5.2f}%    | {val_plain_bacc * 100:5.2f}%    | "
            f"{val_center_bacc * 100:5.2f}%    | {val_sink_bacc * 100:5.2f}%   | {gap * 100:+5.2f}%"
        )

    # Global OOF Metrics across all N samples
    oof_plain_bacc = float(balanced_accuracy_score(y, oof_plain_lp.argmax(axis=1)))
    oof_center_bacc = float(balanced_accuracy_score(y, oof_centered_lp.argmax(axis=1)))
    oof_blend_bacc = float(balanced_accuracy_score(y, oof_blend_lp.argmax(axis=1)))

    P_oof = softmax(oof_blend_lp, axis=1)
    P_oof_sk = sinkhorn_balancing(P_oof, iters=60, sinkhorn_strength=0.65, target_mix=0.35)
    oof_sink_bacc = float(balanced_accuracy_score(y, P_oof_sk.argmax(axis=1)))

    mean_tr_bacc = float(np.mean([f["train_bacc"] for f in fold_results]))
    total_overfit_gap = mean_tr_bacc - oof_sink_bacc

    if total_overfit_gap <= 0.08:
        verdict = "LOW (Excellent Generalization across Unseen Prompts)"
    elif total_overfit_gap <= 0.18:
        verdict = "MODERATE (Normal Stylistic Domain Variance)"
    else:
        verdict = "HIGH (Prompt Memorization Risk - Increase supcon/family margin)"

    print("------------------------------------------------------------------------------------------")
    print(
        f" OOF   | {len(u_prompts):<13} | {len(u_prompts):<11} | "
        f"{mean_tr_bacc * 100:5.2f}%    | {oof_plain_bacc * 100:5.2f}%    | "
        f"{oof_center_bacc * 100:5.2f}%    | {oof_sink_bacc * 100:5.2f}%   | {total_overfit_gap * 100:+5.2f}%"
    )
    print("==========================================================================================")
    print(f" Overfitting Assessment: {verdict}")
    print("==========================================================================================\n")

    report = {
        "n_splits": actual_splits,
        "n_samples": int(len(y)),
        "n_prompts": int(len(u_prompts)),
        "mean_train_bacc": mean_tr_bacc,
        "oof_plain_bacc": oof_plain_bacc,
        "oof_center_bacc": oof_center_bacc,
        "oof_blend_bacc": oof_blend_bacc,
        "oof_sink_bacc": oof_sink_bacc,
        "overfit_gap": total_overfit_gap,
        "verdict": verdict,
        "folds": fold_results,
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        rep_file = out_p / "stage2_cv_report.json"
        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[Stage 2 CV] Saved cross-validation and overfitting report to: {rep_file}")

        oof_npz = out_p / "stage2_oof_predictions.npz"
        np.savez_compressed(
            str(oof_npz),
            oof_plain_lp=oof_plain_lp,
            oof_centered_lp=oof_centered_lp,
            oof_blend_lp=oof_blend_lp,
            oof_sink_probs=P_oof_sk,
            y_true=y,
            prompts=prompts,
            classes=np.array(labels),
        )
        print(f"[Stage 2 CV] Saved out-of-fold predictions to: {oof_npz}")

    return report


def export_compatible_checkpoint(
    model: MultiLayerLoRAClassifier,
    output_dir: Path,
    layers: list[int],
    keep: int,
    max_len: int = 447,
    end_id: int = 151643,
    backbone_name: str | None = None,
    Z_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    prompts_train: np.ndarray | None = None,
    class_labels: list[str] | None = None,
    target_family: str | None = None,
) -> Path:
    """
    Exports standard checkpoint artifacts strictly compatible with inference loaders:
    1. adapter/ (LoRA adapter weights and config)
    2. heads.safetensors (norms.{i}.weight, norms.{i}.bias, plain.weight, plain.bias, centered.weight, centered.bias)
    3. meta.json (keep, layers, max_len, end_id, labels, backbone, classes, target_family)
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
    active_labels = list(class_labels) if class_labels is not None else list(LABELS)
    meta = {
        "labels": active_labels,
        "classes": active_labels,
        "target_family": str(target_family or ""),
        "backbone": str(backbone_name) if backbone_name else "",
        "cfg": {
            "keep": int(keep),
            "layers": [int(x) for x in layers],
            "max_len": int(max_len),
            "end_token_id": int(end_id),
            "backbone": str(backbone_name) if backbone_name else "",
            "target_family": str(target_family or ""),
        },
    }
    (out_p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 4. wPCA Bank precomputation (only when standard 18-class)
    if len(active_labels) == K and Z_train is not None and y_train is not None and prompts_train is not None:
        d_target = min(256, Z_train.shape[1])
        res = fit_shared_gaussian_lowram(Z_train, y_train, prompts_train, d_target=d_target)
        np.savez_compressed(
            str(out_p / "wPCA_bank.npz"),
            mx=res["mx"].astype(np.float32),
            sx=res["sx"].astype(np.float32),
            mu=res["mx"].astype(np.float32),
            sd=res["sx"].astype(np.float32),
            W=res["W"].astype(np.float32),
            m=res["m"].astype(np.float32),
            lam=res["lam"].astype(np.float32),
            V_top64=res["V_top64"].astype(np.float32),
            layers=np.array(layers, dtype=np.int32),
        )

    return out_p


def _get_loraplus_param_groups(
    model: MultiLayerLoRAClassifier,
    lr_lora: float,
    lr_head: float,
    lora_plus_ratio: float = 2.5,
    include_centered_head: bool = False,
) -> list[dict[str, Any]]:
    """
    Builds LoRA+ asymmetric parameter groups where lora_B learns at lora_plus_ratio * lr_lora
    relative to lora_A to accelerate feature learning in high-dimensional hidden layers.
    """
    lora_a_params = []
    lora_b_params = []
    other_backbone_params = []
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if "lora_b" in name_lower:
            lora_b_params.append(param)
        elif "lora_a" in name_lower:
            lora_a_params.append(param)
        else:
            other_backbone_params.append(param)

    head_params = list(model.norms.parameters()) + list(model.plain_head.parameters())
    if include_centered_head and hasattr(model, "centered_head"):
        head_params += list(model.centered_head.parameters())

    groups = []
    if lora_a_params:
        groups.append({"params": lora_a_params, "lr": float(lr_lora)})
    if lora_b_params:
        groups.append({"params": lora_b_params, "lr": float(lr_lora) * float(lora_plus_ratio)})
    if other_backbone_params:
        groups.append({"params": other_backbone_params, "lr": float(lr_lora)})
    if head_params:
        groups.append({"params": head_params, "lr": float(lr_head)})
    return groups


def run_training_pipeline(train_cfg: dict[str, Any] | None = None) -> Path:
    """
    Executes complete Two-Stage Multi-Layer LoRA training pipeline:
    Stage 1: Multi-corpus pretraining on up to 90k passages with family margin loss.
    Stage 2: Prompt-grouped contrastive fine-tuning on train.csv with SupCon loss.
    """
    _setup_training_env()

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    if train_cfg is None:
        train_cfg = {}

    backbone_name = train_cfg.get("backbone", "Qwen/Qwen3-14B-Base")
    keep_layers = int(train_cfg.get("keep_layers", 27))
    layers = list(train_cfg.get("layers", [2, 4, 6, 9, 13, 17, 22, 27]))
    max_len = int(train_cfg.get("max_len", 384))
    end_id = int(train_cfg.get("end_id", 151643))
    out_dir = Path(train_cfg.get("output_dir", "export_champion_qwen3_14b"))
    family_margin = float(train_cfg.get("family_margin", 0.35))
    supcon_weight = float(train_cfg.get("supcon_weight", 0.20))
    is_deberta = "deberta" in backbone_name.lower()

    input_dir = Path(train_cfg.get("input_dir", "/kaggle/input"))

    # Preserve existing 18-class in-house champion checkpoint if present in input_dir
    working_champ = Path("/kaggle/working/export_champion_qwen3_14b")
    if input_dir.exists() and not working_champ.exists():
        for cand in input_dir.rglob("export_champion_qwen3_14b"):
            if cand.is_dir() and (cand / "heads.safetensors").exists():
                try:
                    shutil.copytree(str(cand), str(working_champ))
                    print(f"[Trainer] Preserved existing in-house checkpoint from {cand} to {working_champ}")
                    break
                except Exception as e:
                    print(f"[Trainer] Note on preserving in-house checkpoint: {e}")

    # Sequential execution of specialist jobs if configured
    if train_cfg.get("specialist_jobs") and isinstance(train_cfg["specialist_jobs"], list):
        final_out = out_dir
        for job in train_cfg["specialist_jobs"]:
            job_cfg = {**train_cfg, **job}
            job_cfg.pop("specialist_jobs", None)
            final_out = run_training_pipeline(job_cfg)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return final_out

    target_family = train_cfg.get("target_family")
    is_specialist = bool(target_family and target_family in FAMILIES)
    if is_specialist:
        fam_cols = FAMILIES[target_family]
        active_labels = [LABELS[i] for i in fam_cols]
        sub_lab2i = {m: i for i, m in enumerate(active_labels)}
        num_classes = len(active_labels)
        print(f"[Trainer] Training {target_family} family specialist with {num_classes} classes: {active_labels}")
    else:
        active_labels = list(LABELS)
        sub_lab2i = dict(LAB2I)
        num_classes = K
    local_base = None
    if input_dir.exists():
        if is_deberta:
            for cfg_p in input_dir.rglob("config.json"):
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        c_data = json.load(f)
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

    # Hardware capability detection: Ampere/Hopper (A100/H100) vs. Turing/earlier (T4)
    is_ampere_or_newer = False
    has_high_vram_single_gpu = False
    total_vram_gb = 0.0
    if torch.cuda.is_available():
        major_cap = torch.cuda.get_device_capability(0)[0]
        is_ampere_or_newer = major_cap >= 8
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        total_vram_gb = total_vram_bytes / (1024**3)
        has_high_vram_single_gpu = bool(n_gpus == 1 and total_vram_bytes >= 40 * 1024**3)

    use_bf16 = bool(train_cfg.get("bf16", False) or is_ampere_or_newer)
    compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    use_fa2 = False
    if has_high_vram_single_gpu and use_bf16 and not is_deberta:
        try:
            import flash_attn  # noqa: F401

            use_fa2 = True
        except ImportError:
            use_fa2 = False

    print(
        f"[Trainer] Device: {device}, Available GPUs: {n_gpus} (VRAM: {total_vram_gb:.1f} GiB), "
        f"Dtype: {compute_dtype}, FlashAttn-2: {use_fa2}, Model Source: {model_source}, Keep: {keep_layers}"
    )

    train_path, test_path, pool_file = resolve_dataset_paths(input_dir if input_dir.exists() else Path("."))
    all_pools = resolve_all_pool_files(input_dir if input_dir.exists() else Path("."))

    tok = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab_sz = len(tok)
    if end_id >= vocab_sz:
        safe_end_id = int(tok.eos_token_id or tok.sep_token_id or 2)
        print(f"[Trainer] Config end_id ({end_id}) >= vocab size ({vocab_sz}); adjusting to {safe_end_id}")
        end_id = safe_end_id

    base_cfg = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    base_cfg.num_hidden_layers = keep_layers
    base_cfg.use_cache = False
    if getattr(base_cfg, "layer_types", None) is not None:
        base_cfg.layer_types = base_cfg.layer_types[:keep_layers]

    extra_model_kwargs = {}
    if use_fa2:
        extra_model_kwargs["attn_implementation"] = "flash_attention_2"

    # Memory strategy: Single high-VRAM GPU (A100/H100) vs. Multi-GPU 2xT4 balancing vs. Single T4 vs. CPU
    if has_high_vram_single_gpu:
        print(f"[Trainer] High-VRAM single GPU detected ({total_vram_gb:.1f} GiB). Loading directly on {device} in {compute_dtype}...")
        base_model = AutoModel.from_pretrained(
            model_source,
            config=base_cfg,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **extra_model_kwargs,
        ).to(device)
    elif n_gpus >= 2:
        max_memory = {0: "12.5GiB", 1: "13.5GiB"}
        base_model = AutoModel.from_pretrained(
            model_source,
            config=base_cfg,
            torch_dtype=compute_dtype,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    elif n_gpus == 1:
        base_model = AutoModel.from_pretrained(
            model_source,
            config=base_cfg,
            torch_dtype=compute_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModel.from_pretrained(
            model_source,
            config=base_cfg,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
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

    linear_mod_names = set()
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            linear_mod_names.add(name.split(".")[-1])

    configured_targets = list(train_cfg.get("lora_target_modules", []))
    if not configured_targets:
        configured_targets = (
            ["query_proj", "key_proj", "value_proj", "dense"]
            if is_deberta
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

    active_targets = [m for m in configured_targets if m in linear_mod_names]
    if not active_targets:
        active_targets = list(linear_mod_names)[:7] if linear_mod_names else configured_targets
        print(f"[Trainer] Configured LoRA targets not found; falling back to: {active_targets}")
    else:
        print(f"[Trainer] Active LoRA target modules: {active_targets}")

    lora_cfg = LoraConfig(
        r=int(train_cfg.get("lora_r", 32)),
        lora_alpha=int(train_cfg.get("lora_alpha", 64)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        target_modules=active_targets,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    base_model = get_peft_model(base_model, lora_cfg)

    if hasattr(base_model, "gradient_checkpointing_enable"):
        try:
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base_model.gradient_checkpointing_enable()
    for mod in base_model.modules():
        if hasattr(mod, "gradient_checkpointing"):
            mod.gradient_checkpointing = True

    model = MultiLayerLoRAClassifier(base_model, hidden_size=base_cfg.hidden_size, layers=layers, num_classes=num_classes)
    # If device_map was used, keep classification heads and layer norms on the main device (device / cuda:0)
    # to avoid unnecessary PCI-e device thrashing while preserving device_map on the backbone.
    if hasattr(base_model, "hf_device_map") and len(getattr(base_model, "hf_device_map", {})) > 1:
        model.plain_head = model.plain_head.to(device)
        model.centered_head = model.centered_head.to(device)
        model.dropouts = model.dropouts.to(device)
        for idx in range(len(model.norms)):
            model.norms[idx] = model.norms[idx].to(device)
    else:
        model = model.to(device)

    # Stage 1: Pool Pretraining
    stage1_cfg = train_cfg.get("stage1", {})
    if stage1_cfg.get("enabled", True) and (all_pools or pool_file):
        max_stage1_hours = float(stage1_cfg.get("max_hours", 12.0))
        s1_start = time.time()
        max_s1_sec = max_stage1_hours * 3600.0
        pool_list = all_pools if all_pools else ([pool_file] if pool_file else [])
        print(f"[Trainer] Stage 1 Pretraining on {len(pool_list)} pool file(s)...")
        dfs = []
        pool_source_counts = {}
        for p in pool_list:
            try:
                if str(p).endswith(".parquet"):
                    df_one = pd.read_parquet(p, columns=["text", "model"])
                else:
                    df_one = pd.read_csv(p, usecols=["text", "model"])
                pool_source_counts[Path(p).name] = len(df_one)
                dfs.append(df_one)
            except Exception as e:
                print(f"[Trainer] Could not read pool {p}: {e}")

        total_raw_rows = sum(len(df) for df in dfs)
        if dfs:
            df_p = pd.concat(dfs, ignore_index=True)
        else:
            df_p = pd.DataFrame(columns=["text", "model"])

        df_p = df_p.dropna(subset=["text", "model"])
        df_p["text"] = df_p["text"].astype(str)
        df_p = df_p[df_p["text"].str.strip().str.len() >= 40]
        df_p = df_p[df_p["model"].isin(active_labels)]

        df_p["_hash"] = [_text_hash(t) for t in df_p["text"]]
        df_p = df_p.drop_duplicates(subset=["_hash"]).reset_index(drop=True)
        dedup_rows = len(df_p)

        # Build unseen holdout to quarantine holdout passages from Stage 1 pretraining
        quarantine_dropped = 0
        try:
            _, holdout_hashes = build_unseen_holdout_dataset(input_dir, train_path)
            if holdout_hashes:
                mask_clean = ~df_p["_hash"].isin(holdout_hashes)
                quarantine_dropped = int(len(df_p) - mask_clean.sum())
                if quarantine_dropped > 0:
                    print(f"[Trainer] Quarantined {quarantine_dropped} pool rows that overlapped with unseen holdout.")
                df_p = df_p[mask_clean].reset_index(drop=True)
        except Exception as e:
            print(f"[Trainer] Note on holdout quarantine: {e}")

        # Sacred Data Integrity Report before Stage 1 begins
        word_lens = [len(t.split()) for t in df_p["text"]]
        tok_lens = [min(len(t) // 4 + 1, max_len) for t in df_p["text"]]
        class_dist = df_p["model"].value_counts().to_dict()

        def _calc_percentiles(vals: list[int]) -> dict[str, float]:
            if not vals:
                return {}
            v = np.array(vals)
            return {
                "p10": float(np.percentile(v, 10)),
                "p25": float(np.percentile(v, 25)),
                "p50": float(np.percentile(v, 50)),
                "p75": float(np.percentile(v, 75)),
                "p90": float(np.percentile(v, 90)),
                "p99": float(np.percentile(v, 99)),
            }

        audit_report = {
            "source_files": pool_source_counts,
            "total_raw_rows": int(total_raw_rows),
            "deduplicated_rows": int(dedup_rows),
            "quarantined_holdout_rows": int(quarantine_dropped),
            "retained_pool_rows": int(len(df_p)),
            "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
            "word_length_percentiles": _calc_percentiles(word_lens),
            "token_length_percentiles": _calc_percentiles(tok_lens),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        audit_file = out_dir / "data_audit_report.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

        print("\n==========================================================================================")
        print("                         SACRED DATA INTEGRITY AUDIT REPORT                               ")
        print("==========================================================================================")
        print(f" Source Files Ingested ({len(pool_source_counts)}):")
        for fname, cnt in pool_source_counts.items():
            print(f"   - {fname:<32}: {cnt:>8,} rows")
        print(f" Total Raw Pool Rows     : {total_raw_rows:,}")
        print(f" Deduplicated Pool Rows  : {dedup_rows:,}")
        print(f" Quarantined Holdout Rows: {quarantine_dropped:,}")
        print(f" Clean Retained Pool Rows: {len(df_p):,}")
        print(" Per-Class Distribution (Active Models):")
        for lbl in active_labels:
            c_cnt = class_dist.get(lbl, 0)
            pct = (c_cnt / max(len(df_p), 1)) * 100
            print(f"   - {lbl:<22}: {c_cnt:>6,} ({pct:5.2f}%)")
        w_p = audit_report["word_length_percentiles"]
        t_p = audit_report["token_length_percentiles"]
        if w_p:
            print(
                f" Word Lengths (words)   : p10={w_p['p10']:.0f} | p25={w_p['p25']:.0f} | p50={w_p['p50']:.0f} | p75={w_p['p75']:.0f} | p90={w_p['p90']:.0f} | p99={w_p['p99']:.0f}"
            )
        if t_p:
            print(
                f" Token Lengths (tokens) : p10={t_p['p10']:.0f} | p25={t_p['p25']:.0f} | p50={t_p['p50']:.0f} | p75={t_p['p75']:.0f} | p90={t_p['p90']:.0f} | p99={t_p['p99']:.0f}"
            )
        print(f" Report Saved To        : {audit_file}")
        print("==========================================================================================\n")

        # Raise default max_samples to 150000 and upweight HARD_CLASS_INDICES by 1.30x
        max_s = int(stage1_cfg.get("max_samples", 2000 * num_classes if is_specialist else 150000))
        if len(df_p) > max_s:
            hard_set = set(HARD_CLASS_INDICES)
            cls_multipliers = {lbl: (1.30 if sub_lab2i[lbl] in hard_set else 1.0) for lbl in active_labels}
            total_multiplier = sum(cls_multipliers.values())
            target_per_c = {lbl: int(max_s * (cls_multipliers[lbl] / total_multiplier)) for lbl in active_labels}
            sampled = []
            remainder = []
            for lbl, group in df_p.groupby("model"):
                target_k = target_per_c.get(lbl, max_s // num_classes)
                if len(group) <= target_k:
                    sampled.append(group)
                else:
                    s_grp = group.sample(n=target_k, random_state=42)
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

        lengths = [min(len(t) // 4 + 1, max_len) for t in df_p["text"]]
        bs1 = int(stage1_cfg.get("batch_size", 2))
        accum1 = int(stage1_cfg.get("gradient_accumulation_steps", 8))
        epochs1 = int(stage1_cfg.get("epochs", 2 if not is_specialist else 1))
        ds1 = TextSequenceDataset(df_p["text"].tolist(), [sub_lab2i[m] for m in df_p["model"]], None, tok, max_len=max_len, end_id=end_id)
        bucket_sampler = BucketedLengthBatchSampler(lengths, batch_size=bs1, shuffle=True)
        loader1 = DataLoader(ds1, batch_sampler=bucket_sampler, collate_fn=collate_text_batch)

        lora_ratio1 = float(stage1_cfg.get("lora_plus_ratio", 2.5))
        param_groups1 = _get_loraplus_param_groups(
            model,
            lr_lora=float(stage1_cfg.get("lr_lora", 1.5e-4)),
            lr_head=float(stage1_cfg.get("lr_head", 8e-4)),
            lora_plus_ratio=lora_ratio1,
            include_centered_head=False,
        )
        opt1 = torch.optim.AdamW(param_groups1, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
        total_steps1 = len(loader1) * epochs1
        sched1 = get_cosine_schedule_with_warmup(
            opt1, num_warmup_steps=int(stage1_cfg.get("warmup_steps", 100)), num_training_steps=math.ceil(total_steps1 / accum1)
        )

        stage1_hard_weight = float(stage1_cfg.get("hard_class_weight", train_cfg.get("hard_class_weight", 1.40)))
        stage1_class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
        for h_idx in HARD_CLASS_INDICES:
            if h_idx < num_classes:
                stage1_class_weights[h_idx] = stage1_hard_weight

        model.train()
        opt1.zero_grad()
        global_step1 = 0
        stop_stage1 = False
        for ep in range(epochs1):
            if stop_stage1:
                break
            bucket_sampler.set_epoch(ep)
            for step_idx, batch in enumerate(loader1):
                global_step1 += 1
                mb_ids = batch["input_ids"].to(device)
                mb_mask = batch["attention_mask"].to(device)
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                mb_ids = mb_ids[:, :max_l]
                mb_mask = mb_mask[:, :max_l]
                labels_dev = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
                    Z_s1, logits_p, _ = model(mb_ids, mb_mask, drop_first_token=not is_deberta)
                    loss_ce = family_margin_cross_entropy(logits_p, labels_dev, family_margin=family_margin, weight=stage1_class_weights)
                    loss_sup = compute_supcon_loss(Z_s1, labels_dev, temperature=0.10, prototypes=model.plain_head.weight)
                    loss = (loss_ce + 0.15 * loss_sup) / accum1

                loss.backward()
                if global_step1 % accum1 == 0 or global_step1 == total_steps1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt1.step()
                    sched1.step()
                    opt1.zero_grad()

                if global_step1 % 200 == 0:
                    elapsed_m = (time.time() - s1_start) / 60.0
                    print(
                        f"[Trainer Stage 1] Epoch {ep + 1}/{epochs1} Step {step_idx + 1}/{len(loader1)} (Global {global_step1}/{total_steps1}) | Loss: {loss.item() * accum1:.4f} | Elapsed: {elapsed_m:.1f}m"
                    )
                if (time.time() - s1_start) >= max_s1_sec:
                    print(
                        f"[Trainer Stage 1] Reached {stage1_cfg.get('max_hours', 12.0)}h time budget at global step {global_step1}; advancing to Stage 2."
                    )
                    stop_stage1 = True
                    break

        del ds1, loader1, opt1, sched1, df_p
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Intermediate checkpoint export after Stage 1
        try:
            export_compatible_checkpoint(
                model=model,
                output_dir=out_dir,
                layers=layers,
                keep=keep_layers,
                max_len=max_len,
                end_id=end_id,
                backbone_name=backbone_name,
                class_labels=active_labels,
                target_family=target_family if is_specialist else None,
            )
            print(f"[Trainer] Safe intermediate Stage 1 checkpoint exported to {out_dir}")
        except Exception as e:
            print(f"[Trainer] Warning: could not export Stage 1 intermediate checkpoint: {e}")

    # Stage 2: Prompt-Grouped Fine-Tuning on train.csv
    stage2_cfg = train_cfg.get("stage2", {})
    df_tr = pd.read_csv(train_path)
    if is_specialist:
        df_tr = df_tr[df_tr["model"].isin(active_labels)].reset_index(drop=True)
    print(f"[Trainer] Stage 2 Fine-Tuning on {train_path} ({len(df_tr)} samples, {num_classes} classes)...")
    bs2 = int(stage2_cfg.get("batch_size", 2))
    accum2 = int(stage2_cfg.get("gradient_accumulation_steps", 1))
    ds2 = TextSequenceDataset(
        df_tr["text"].tolist(), [sub_lab2i[m] for m in df_tr["model"]], df_tr["prompt_id"].tolist(), tok, max_len=max_len, end_id=end_id
    )
    sampler2 = PromptGroupBatchSampler(df_tr["prompt_id"].tolist(), batch_size=bs2, shuffle=True)
    loader2 = DataLoader(ds2, batch_sampler=sampler2, collate_fn=collate_text_batch)

    lora_ratio2 = float(stage2_cfg.get("lora_plus_ratio", 2.5))
    param_groups2 = _get_loraplus_param_groups(
        model,
        lr_lora=float(stage2_cfg.get("lr_lora", 3e-5)),
        lr_head=float(stage2_cfg.get("lr_head", 2e-4)),
        lora_plus_ratio=lora_ratio2,
        include_centered_head=True,
    )
    opt2 = torch.optim.AdamW(param_groups2, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    epochs2 = int(stage2_cfg.get("epochs", 2))
    total_steps2 = len(loader2) * epochs2
    sched2 = get_cosine_schedule_with_warmup(
        opt2, num_warmup_steps=int(stage2_cfg.get("warmup_steps", 50)), num_training_steps=math.ceil(total_steps2 / accum2)
    )

    # SWA setup: average parameters across the final epoch of Stage 2
    swa_enabled = bool(stage2_cfg.get("swa", True))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    swa_shadow = {p: torch.zeros_like(p.data, device="cpu") for p in trainable_params} if swa_enabled else {}
    swa_steps = 0

    # Hard-Class Loss Upweighting for Stage 2 fine-tuning
    stage2_hard_weight = float(stage2_cfg.get("hard_class_weight", train_cfg.get("hard_class_weight", 1.40)))
    stage2_class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    for h_idx in HARD_CLASS_INDICES:
        if h_idx < num_classes:
            stage2_class_weights[h_idx] = stage2_hard_weight

    model.train()
    global_step2 = 0
    for ep in range(epochs2):
        sampler2.set_epoch(ep)
        opt2.zero_grad()
        for step_idx, batch in enumerate(loader2):
            global_step2 += 1
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
                label_smoothing=0.10,
                accum_steps=accum2,
                compute_dtype=compute_dtype,
                weight=stage2_class_weights,
            )
            if global_step2 % accum2 == 0 or (step_idx + 1) == len(loader2):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()
                sched2.step()
                opt2.zero_grad()

                if swa_enabled and ep == (epochs2 - 1):
                    swa_steps += 1
                    for p in trainable_params:
                        swa_shadow[p].add_(p.data.detach().cpu())

    if swa_enabled and swa_steps > 0:
        print(f"[Trainer] Applying SWA weight averaging across {swa_steps} steps in final Stage 2 epoch...")
        for p in trainable_params:
            p.data.copy_((swa_shadow[p] / float(swa_steps)).to(p.device))
        del swa_shadow

    # Feature extraction for wPCA bank & checkpoint export
    del loader2, opt2, sched2, sampler2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    loader_eval = DataLoader(ds2, batch_size=4, shuffle=False, collate_fn=collate_text_batch)
    Z_list = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            for batch in loader_eval:
                b_ids = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                max_l = max(int(b_mask.sum(dim=1).max().item()), 2)
                z_b = model.forward_features(b_ids[:, :max_l], b_mask[:, :max_l], drop_first_token=not is_deberta)
                Z_list.append(z_b.cpu().numpy())
    Z_train = np.vstack(Z_list)
    y_train = np.array([sub_lab2i[m] for m in df_tr["model"]])
    prompts_train = df_tr["prompt_id"].values

    # 5-Stage Cross-Validation & Overfitting Audit on Stage 2 Representations
    n_cv_splits = int(stage2_cfg.get("n_splits", 5))
    if stage2_cfg.get("check_overfitting", True) and len(np.unique(prompts_train)) >= 2:
        evaluate_stage2_oof_cv(
            Z=Z_train,
            y=y_train,
            prompts=prompts_train,
            n_splits=n_cv_splits,
            alpha=8.0 if is_specialist else 10.0,
            label_smoothing=0.05 if is_specialist else 0.10,
            num_classes=num_classes,
            class_labels=active_labels,
            output_dir=out_dir,
        )

    # Closed-form dual Ridge head blending:
    # Combines neural margin boundaries with exact L2-regularized closed-form ridge centroids over Z_train.
    # Applied to both specialist models (0.65 neural / 0.35 ridge) and 18-class champion model (0.75 neural / 0.25 ridge).
    r_alpha = 8.0 if is_specialist else 10.0
    r_smoothing = 0.05 if is_specialist else 0.10
    w_n = 0.65 if is_specialist else 0.75
    w_r = 0.35 if is_specialist else 0.25

    W_p_r, b_p_r, W_c_r, b_c_r = fit_ridge_heads(
        Z_train, y_train, prompts_train, alpha=r_alpha, label_smoothing=r_smoothing, num_classes=num_classes
    )
    _blend_head(model.plain_head, W_p_r, b_p_r, w_neural=w_n, w_ridge=w_r)
    _blend_head(model.centered_head, W_c_r, b_c_r, w_neural=w_n, w_ridge=w_r)
    print(f"[Trainer] Successfully blended dual Ridge heads into neural classifier (w_neural={w_n:.2f}, w_ridge={w_r:.2f}).")

    export_compatible_checkpoint(
        model=model,
        output_dir=out_dir,
        layers=layers,
        keep=keep_layers,
        max_len=max_len,
        end_id=end_id,
        backbone_name=backbone_name,
        Z_train=Z_train,
        y_train=y_train,
        prompts_train=prompts_train,
        class_labels=active_labels,
        target_family=target_family if is_specialist else None,
    )
    print(f"[Trainer] Checkpoint exported to {out_dir}")
    return out_dir
