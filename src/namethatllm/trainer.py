"""
Name That LLM - Production Multi-Layer LoRA Training & Feature Extraction Engine
================================================================================
A unified, high-performance training and representation engine supporting three core execution modes:

1. Full LoRA Foundation Model Training with Dual Heads (Stage 1 & Stage 2):
   - Backbones: Qwen3-14B, Qwen2.5-14B, Qwen3-8B, DeBERTa-v3-large.
   - Intermediate Layer Extraction: Taps 8 intermediate hidden states (e.g., layers 2, 4, 6, 8, 12, 16, 20, 24)
     with dedicated per-layer LayerNorm projections to capture rich syntactic and stylistic representations.
   - Causal Attention-Sink Masking: Automatically masks the first token (m[:, 0] = 0.0) to remove BOS softmax artifacts.
   - Dual Classification Heads:
       * plain_head: Linear projection Z -> R^K for direct sequence-level attribution.
       * centered_head: Linear projection (Z - mu_prompt) -> R^K trained on prompt-centered features to decouple
         pure model writing style from the user's prompt topic.
   - Multi-Sample Dropout (MSD): 5-branch dropout (p in [0.1, 0.5]) during training for accelerated regularized convergence.
   - Stage 1 Pool Pre-Training: Pre-trains on up to 120,000+ deduplicated external samples with dynamic length slicing and AMP.
   - Stage 2 Gold Fine-Tuning: Trains on competition train.csv with prompt-group batch sampling, LoRA+ learning rate ratios,
     supervised contrastive loss (SupCon), and SWA parameter averaging across the final epoch.

2. Specialist Model & Family Gate Training (Phase 3):
   - Trains dedicated gates on frozen Champion 14B features for hard intra-family sister decoupling:
       * Gemini Family: gemini-3.7-flash vs gemini-3.8-flash (cosine similarity 0.9418).
       * GPT Family: gpt-5.6-sol vs gpt-5.6-terra vs gpt-5.6-luna vs gpt-5.5 vs gpt-6-astra.
       * GLM Family: glm-5.3 vs glm-5.3-flash.
       * Claude Family: claude-opus-4-8 vs claude-sonnet-5 vs claude-opus-4-6 vs claude-fable-5.
   - Head Types Supported:
       * Linear Ridge Regression Head: Closed-form dual solver with trace-scaled regularization across alpha sweeps.
       * Non-Linear MLP Head: 2-layer MLP (fc1 -> GELU -> LayerNorm -> Dropout -> fc2).
   - Zero-Cost Centroid Fold-In: Folds intra-family mean subtraction directly into Layer 1 bias:
       b1* = b1 - mu_family @ W1.T
     allowing downstream inference to evaluate raw Z vectors without runtime centering latency.
   - Zero-Leakage Holdout Quarantine: Automatically cross-references texts against clean_benchmark_1800_rows.parquet
     to guarantee 0.0% data leakage during gate training.

3. Configurable Embedding & Layer-by-Layer Feature Extraction (Dataset Export):
   - Pre-extracts and saves multi-layer representations as a reusable Kaggle Dataset so subsequent downstream
     gate training and model ablation take seconds instead of repeating 30-minute GPU forward passes.
   - Configurable Subsets ("target"):
       * "all": All 18 competition classes across pool and gold data.
       * "gemini": Gemini models (3.1-pro, 3.6-flash, 3.7-flash, 3.8-flash) + negative distractors.
       * "gpt": GPT models (5.5, 5.6-luna, 5.6-sol, 5.6-terra, 6-astra) + negative distractors.
       * "claude": Claude models (opus-4-6, opus-4-8, opus-5, sonnet-5, fable-5) + negative distractors.
       * "glm": GLM models (5.3, 5.3-flash) + negative distractors.
       * "gold": Pure gold competition training data (2,699 samples).
   - Configurable Layer Storage ("save_all_layers"):
       * false (default): Saves the 8 champion tapped layers ([N, 8, hidden_size] and [N, 40960]).
       * true: Saves all backbone hidden states ([N, num_layers, hidden_size], e.g. [N, 28, 5120]).
   - Export Artifacts Produced:
       * features_concat.safetensors: Full concatenated representation in FP16.
       * features_by_layer.safetensors: 3D tensor [N, L, hidden_size] in FP16.
       * metadata.parquet: Text hashes, sample IDs, prompt IDs, lengths, and model labels.
       * benchmark_features_concat.safetensors & benchmark_metadata.parquet: Exact holdout features.
       * dataset-metadata.json: Pre-formatted for instant 1-click `kaggle datasets create`.
       * features_summary.json: Human-readable inspection metadata.

Config Schema Summary:
----------------------
{
  "pipeline_mode": "full_lora" | "family_gate" | "export_embeddings",
  "backbone": "Qwen/Qwen3-14B-Base",
  "target_family": "gemini" | "gpt" | "claude" | "glm",
  "export_embeddings": {
    "enabled": true,
    "target": "all" | "gemini" | "gpt" | "claude" | "glm" | "gold",
    "save_all_layers": false,
    "output_dir": "export_embeddings",
    "samples_per_class": 3800
  }
}
"""

import copy
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
from .postprocess import apply_prior_and_sinkhorn as _apply_prior_and_sinkhorn, sinkhorn_balancing
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

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drop_first_token: bool = True,
        return_by_layer: bool = False,
        extract_all_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts pooled multi-layer representations with causal attention-sink masking.

        Why Attention-Sink Masking (drop_first_token) Matters:
        ------------------------------------------------------
        In causal LLMs (Qwen, Llama), the first token (token index 0, e.g. BOS or leading word)
        acts as an 'attention sink' where the softmax mechanism dumps massive attention logits
        regardless of whether token 0 carries semantic meaning. If token 0 is included in
        mean pooling across the sequence length, it injects noise that masks subtle stylistic
        differences. Setting m[:, 0] = 0.0 cleans the representation.

        Parameters:
        -----------
        return_by_layer: If True, returns (Z_cat, Z_layers).
        extract_all_layers: If True, Z_layers stacks all backbone hidden states [B, num_all_layers, hidden_size].
                            If False, Z_layers stacks the configured tapped layers [B, len(self.layers), hidden_size].
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

        all_layers_pooled = []
        if extract_all_layers:
            for l_idx in range(len(hidden_states)):
                h = hidden_states[l_idx].float()
                layer_dev = h.device
                m_l = m.to(layer_dev)
                den_l = den.to(layer_dev)
                h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
                p_l = (h * m_l.unsqueeze(-1)).sum(dim=1) / den_l
                p_l = F.normalize(p_l, p=2, dim=-1, eps=1e-6)
                all_layers_pooled.append(p_l.to(first_device))

        pooled_list = []
        for idx, l in enumerate(self.layers):
            h = hidden_states[l].float()
            layer_dev = h.device
            m_l = m.to(layer_dev)
            den_l = den.to(layer_dev)
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
            pooled = (h * m_l.unsqueeze(-1)).sum(dim=1) / den_l
            norm = self.norms[idx]
            norm_dev = next(norm.parameters()).device
            normed = norm(pooled.to(norm_dev))
            pooled_list.append(normed.to(first_device))

        Z_cat = torch.cat(pooled_list, dim=-1)
        if return_by_layer:
            if extract_all_layers:
                Z_layers = torch.stack(all_layers_pooled, dim=1)
            else:
                Z_layers = torch.stack(pooled_list, dim=1)
            return Z_cat, Z_layers

        return Z_cat

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
        # 1. Gemini Flash Triad (indices 6, 7, 8) & 3.1-pro (5) - Asymmetric protection for 3.7-flash (7) and 3.6-flash (6)
        M[7, 8] = 2.0  # True: gemini-3.7-flash -> False: gemini-3.8-flash
        M[8, 7] = 1.3  # True: gemini-3.8-flash -> False: gemini-3.7-flash
        M[6, 7] = 1.4
        M[7, 6] = 1.4
        M[6, 8] = 1.5
        M[8, 6] = 1.3
        M[6, 5] = 1.6  # True: gemini-3.6-flash -> False: gemini-3.1-pro-preview
        M[7, 5] = 1.5  # True: gemini-3.7-flash -> False: gemini-3.1-pro-preview

        # 2. GPT-5.6 Triad (indices 1, 2, 3) & GPT-6-Astra (4) - Asymmetric protection for Astra
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i != j:
                    M[i, j] = 1.4
        M[4, 2] = 1.8  # True: gpt-6-astra -> False: gpt-5.6-sol
        M[2, 4] = 1.2
        M[4, 1] = 1.7  # True: gpt-6-astra -> False: gpt-5.6-luna
        M[1, 4] = 1.2
        M[4, 3] = 1.5  # True: gpt-6-astra -> False: gpt-5.6-terra
        M[3, 4] = 1.2

        # 3. GLM Sisters (15, 16) & Cross-Family RLHF Sinks (Claude Sonnet 5 [12], Fable 5 [13], Kimi-K3 [17])
        M[15, 16] = 1.5  # True: glm-5.3 -> False: glm-5.3-flash
        M[16, 15] = 1.7  # True: glm-5.3-flash -> False: glm-5.3
        M[15, 12] = 1.7  # True: glm-5.3 -> False: claude-sonnet-5
        M[12, 15] = 1.2
        M[15, 13] = 1.5  # True: glm-5.3 -> False: claude-fable-5
        M[17, 13] = 1.4  # True: kimi-k3 -> False: claude-fable-5

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
        loss_sup = compute_supcon_loss(Z_c, labels_dev, prototypes=model.centered_head.weight.detach())
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
        full_ids = self.tok(str(self.texts[idx]), add_special_tokens=False, truncation=False)["input_ids"]
        if len(full_ids) > self.max_len:
            head_len = self.max_len // 2
            tail_len = self.max_len - head_len
            raw_ids = full_ids[:head_len] + full_ids[-tail_len:]
        else:
            raw_ids = full_ids
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
                    Z_c_mb, labels_dev[s:e], temperature=0.10, prototypes=model.centered_head.weight.detach(), Z_group_centered=Z_group_centered
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
    scale_by_trace: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits closed-form dual Ridge regression heads for plain and centered representations
    with label smoothing and feature-dimension trace scaling.
    Uses dual formulation (N x N) for ultra-fast, numerically stable solution.
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
    if scale_by_trace:
        mean_diag_p = float(np.mean(np.sum(Z * Z, axis=1)))
        eff_alpha_p = float(alpha) * max(mean_diag_p / 512.0, 1.0)
    else:
        eff_alpha_p = float(alpha)
    K_mat.flat[:: N + 1] += eff_alpha_p
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
    if scale_by_trace:
        mean_diag_c = float(np.mean(np.sum(Z_c * Z_c, axis=1)))
        eff_alpha_c = float(alpha) * max(mean_diag_c / 512.0, 1.0)
    else:
        eff_alpha_c = float(alpha)
    K_mat_c.flat[:: N + 1] += eff_alpha_c
    A_c = np.linalg.solve(K_mat_c, Y)
    W_c_tilde = (A_c.T @ Z_c_tilde).astype(np.float32)
    W_c = W_c_tilde[:, :-1]
    b_c = W_c_tilde[:, -1]

    return W_p, b_p, W_c, b_c


def compute_grouped_inference_logits(
    Z: np.ndarray,
    plain_weight: np.ndarray,
    plain_bias: np.ndarray,
    centered_weight: np.ndarray,
    centered_bias: np.ndarray,
    groups: list[Any] | np.ndarray | None = None,
    blend_centered: float = 0.20,
) -> dict[str, np.ndarray]:
    """
    Computes plain, gated prompt-group centered, and blended logits for inference.
    When groups are available, only centers groups that have >= 6 samples and >= 4 distinct
    plain-head predicted classes to prevent self-subtraction on homogeneous/small clusters.
    Uses shrinkage centering (0.75x mean subtraction) for valid groups.
    When groups are absent and batch size >= 18, centers globally across the batch.
    Computes prior-centered Sinkhorn-balanced probabilities on both logits_plain and logits_blend.
    """
    logits_plain = Z @ plain_weight.T + plain_bias
    if groups is not None and len(groups) == len(Z):
        Z_c = Z.copy()
        singleton_mask = np.zeros(len(Z), dtype=bool)
        grps_arr = np.asarray(groups)
        plain_preds = logits_plain.argmax(axis=1)
        for g in np.unique(grps_arr):
            mask = (grps_arr == g)
            distinct_classes = len(np.unique(plain_preds[mask]))
            if mask.sum() >= 6 and distinct_classes >= 4:
                Z_c[mask] = Z[mask] - 0.75 * Z[mask].mean(axis=0, keepdims=True)
            else:
                singleton_mask[mask] = True
        logits_centered = Z_c @ centered_weight.T + centered_bias
        if np.any(singleton_mask):
            logits_centered[singleton_mask] = logits_plain[singleton_mask]
    elif groups is None and len(Z) >= 18:
        Z_c = Z - 0.75 * Z.mean(axis=0, keepdims=True)
        logits_centered = Z_c @ centered_weight.T + centered_bias
    else:
        logits_centered = logits_plain.copy()

    # Standardize per-sample logit scale before blending
    row_mean_p = logits_plain.mean(axis=1, keepdims=True)
    row_std_p = np.clip(logits_plain.std(axis=1, keepdims=True), 1e-4, None)
    norm_plain = (logits_plain - row_mean_p) / row_std_p

    row_mean_c = logits_centered.mean(axis=1, keepdims=True)
    row_std_c = np.clip(logits_centered.std(axis=1, keepdims=True), 1e-4, None)
    norm_centered = (logits_centered - row_mean_c) / row_std_c

    logits_blend = (1.0 - blend_centered) * norm_plain + blend_centered * norm_centered

    probs_plain_sinkhorn = _apply_prior_and_sinkhorn(logits_plain)
    probs_sinkhorn = _apply_prior_and_sinkhorn(logits_blend)

    return {
        "logits_plain": logits_plain,
        "logits_centered": logits_centered,
        "logits_blend": logits_blend,
        "probs_plain_sinkhorn": probs_plain_sinkhorn,
        "probs_sinkhorn": probs_sinkhorn,
    }


def _blend_head(head_linear: nn.Linear, W_r: np.ndarray, b_r: np.ndarray, w_neural: float = 0.65, w_ridge: float = 0.35) -> None:
    """Blends neural head with closed-form ridge head after matching Frobenius norm."""
    with torch.no_grad():
        W_n = head_linear.weight.detach().cpu().float().numpy()
        b_n = head_linear.bias.detach().cpu().float().numpy()
        norm_n = float(np.linalg.norm(W_n))
        norm_r = float(np.linalg.norm(W_r))
        scale = (norm_n / norm_r) if (norm_r > 1e-6 and norm_n > 1e-6) else 1.0
        W_r_scaled = W_r * scale
        b_r_centered = (b_r - b_r.mean()) * scale + b_n.mean()
        W_blend = w_neural * W_n + w_ridge * W_r_scaled
        b_blend = w_neural * b_n + w_ridge * b_r_centered
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
        singleton_mask = np.zeros(len(p_val), dtype=bool)
        for p in np.unique(p_val):
            m = p_val == p
            if m.sum() > 1:
                Z_val_c[m] -= Z_val_c[m].mean(axis=0, keepdims=True)
            else:
                Z_val_c[m] = 0.0
                singleton_mask[m] = True
        val_centered = Z_val_c @ W_c.T + b_c
        if np.any(singleton_mask):
            val_centered[singleton_mask] = val_plain[singleton_mask]
        val_blend = 0.5 * val_plain + 0.5 * val_centered

        val_plain_bacc = float(balanced_accuracy_score(y_val, val_plain.argmax(axis=1)))
        val_center_bacc = float(balanced_accuracy_score(y_val, val_centered.argmax(axis=1)))
        val_blend_bacc = float(balanced_accuracy_score(y_val, val_blend.argmax(axis=1)))

        # Val Soft Sinkhorn (scale ridge outputs to standard logit std ~4.0 before softmax)
        val_std = float(np.std(val_blend, axis=1).mean())
        val_scale = (4.0 / val_std) if val_std > 1e-4 else 1.0
        P_val = softmax(val_blend * val_scale, axis=1)
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

    oof_std = float(np.std(oof_blend_lp, axis=1).mean())
    oof_scale = (4.0 / oof_std) if oof_std > 1e-4 else 1.0
    P_oof = softmax(oof_blend_lp * oof_scale, axis=1)
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


def run_training_pipeline(
    train_cfg: dict[str, Any] | None = None,
    step_callback: Any | None = None,
) -> Path:
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
    is_smoke = bool(train_cfg.get("smoke", False))
    checkpoints_volume = train_cfg.get("checkpoints_volume", None)

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
            job_cfg = copy.deepcopy(train_cfg)
            job_cfg.pop("specialist_jobs", None)
            for k, v in job.items():
                if isinstance(v, dict) and isinstance(job_cfg.get(k), dict):
                    job_cfg[k] = {**job_cfg[k], **v}
                else:
                    job_cfg[k] = v
            final_out = run_training_pipeline(job_cfg, step_callback=step_callback)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return final_out

    target_family = train_cfg.get("target_family")
    is_gate_or_export = bool(
        train_cfg.get("pipeline_mode") in ("family_gate", "export_embeddings", "extract_features")
        or train_cfg.get("mode") == "family_gate"
        or train_cfg.get("gate_training")
    )
    is_specialist = bool(target_family and target_family in FAMILIES and not is_gate_or_export)
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

    # Pipeline Mode: Feature & Layer-by-Layer Embedding Extraction (Dataset Export)
    if train_cfg.get("pipeline_mode") in ("export_embeddings", "extract_features"):
        exp_cfg = train_cfg.get("export_embeddings", {})
        if not isinstance(exp_cfg, dict):
            exp_cfg = {}
        target = exp_cfg.get("target") or train_cfg.get("embeddings_target") or target_family or "gemini"
        save_all = bool(exp_cfg.get("save_all_layers", train_cfg.get("save_all_layers", False)))
        emb_out = exp_cfg.get("output_dir") or train_cfg.get("embeddings_output_dir") or out_dir

        pool_list = all_pools if all_pools else ([pool_file] if pool_file else [])
        dfs = []
        for p in pool_list:
            try:
                if str(p).endswith(".parquet"):
                    df_one = pd.read_parquet(p, columns=["text", "model"])
                else:
                    df_one = pd.read_csv(p, usecols=["text", "model"])
                dfs.append(df_one)
            except Exception:
                pass
        df_p = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["text", "model"])

        df_tr = None
        if train_path and Path(train_path).exists():
            try:
                df_tr = pd.read_csv(train_path)
            except Exception:
                pass

        eval_texts, eval_labels = None, None
        bench_paths = list(input_dir.rglob("clean_benchmark_1800_rows.parquet"))
        if bench_paths:
            try:
                df_bench = pd.read_parquet(bench_paths[0])
                eval_texts = df_bench["text"].tolist()
                eval_labels = [LAB2I[m] for m in df_bench["model"].tolist()]
                print(f"[Trainer] Loaded {len(eval_texts)} benchmark samples from {bench_paths[0]}")
            except Exception as e:
                print(f"[Trainer] Note on reading clean benchmark: {e}")

        # If a champion LoRA checkpoint exists, load it
        lora_r_val = int(train_cfg.get("lora_r", 32))
        adapter_found = find_compatible_champion_lora(input_dir, backbone_name=backbone_name, lora_r=lora_r_val)
        if adapter_found is not None:
            try:
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                adapter_file = adapter_found / "adapter_model.safetensors"
                if not adapter_file.exists():
                    adapter_file = adapter_found / "adapter" / "adapter_model.safetensors"
                if not adapter_file.exists():
                    adapter_file = adapter_found / "adapter_model.bin"
                adapter_sd = load_file(str(adapter_file))
                set_peft_model_state_dict(base_model, adapter_sd)
                print(f"[Trainer] Loaded champion LoRA weights from {adapter_found} for extraction.")
            except Exception as e:
                print(f"[Trainer] Note on loading adapter for extraction: {e}")

        for cand in list(input_dir.rglob("heads.safetensors")) + list(input_dir.rglob("heads.pt")):
            if "export_champion" in str(cand) or "inhouse" in str(cand):
                if load_and_slice_champion_heads(cand, model, fam_cols=None):
                    print(f"[Trainer] Loaded champion norms & 18-class plain head from {cand}")
                    break

        extract_and_export_embeddings(
            model=model,
            tok=tok,
            df_pool=df_p,
            df_train=df_tr,
            target=target,
            save_all_layers=save_all,
            output_dir=emb_out,
            samples_per_class=int(train_cfg.get("samples_per_class", 0)),
            negative_ratio=float(train_cfg.get("negative_ratio", 0.10)),
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            batch_size=int(train_cfg.get("batch_size", 16)),
            max_len=int(train_cfg.get("max_len", 384)),
            end_id=end_id,
            device=device,
            checkpoints_volume=checkpoints_volume,
        )
        return Path(emb_out)

    # Phase 3: Family Gate Training Mode (Non-Linear MLP or Ridge)
    if train_cfg.get("pipeline_mode") == "family_gate" or train_cfg.get("mode") == "family_gate" or train_cfg.get("gate_training"):
        print(f"[Trainer] Running Phase 3 Family Gate Pipeline for target family: {target_family} (head_type={train_cfg.get('head_type', 'mlp')})...")

        lora_r_val = int(train_cfg.get("lora_r", 32))
        adapter_found = find_compatible_champion_lora(
            input_dir,
            backbone_name=backbone_name,
            lora_r=lora_r_val,
            target_family=target_family or "",
        )
        if adapter_found is not None:
            try:
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file

                adapter_file = adapter_found / "adapter_model.safetensors"
                if not adapter_file.exists():
                    adapter_file = adapter_found / "adapter" / "adapter_model.safetensors"
                if not adapter_file.exists():
                    adapter_file = adapter_found / "adapter_model.bin"
                adapter_sd = load_file(str(adapter_file))

                # Pre-validate tensor shapes against base_model
                model_sd = base_model.state_dict()
                mismatches = []
                for k_sd, v_sd in adapter_sd.items():
                    if k_sd in model_sd and model_sd[k_sd].shape != v_sd.shape:
                        mismatches.append(f"{k_sd}: checkpoint {v_sd.shape} vs model {model_sd[k_sd].shape}")

                if mismatches:
                    raise ValueError(f"Shape mismatches in adapter {adapter_found}: {mismatches[:3]}...")

                set_peft_model_state_dict(base_model, adapter_sd)
                print(f"[Trainer] Successfully loaded champion LoRA weights from {adapter_found}")
            except Exception as e:
                print(f"[Trainer] Error loading champion adapter weights: {e}")
                if not is_smoke:
                    raise RuntimeError(f"Failed to load champion LoRA adapter from {adapter_found}: {e}")
        else:
            print(f"[Trainer] Warning: No compatible champion LoRA adapter found for {backbone_name} (r={lora_r_val}) in {input_dir}")
            if not is_smoke and input_dir.exists():
                found_all = list(input_dir.rglob("adapter_model.safetensors"))
                if found_all:
                    raise RuntimeError(f"Discovered adapters on disk but none were compatible with {backbone_name} (r={lora_r_val}): {found_all}")

        head_loaded = False
        for cand in list(input_dir.rglob("heads.safetensors")) + list(input_dir.rglob("heads.pt")):
            if "export_champion" in str(cand) or "inhouse" in str(cand):
                if load_and_slice_champion_heads(cand, model, fam_cols=None):
                    print(f"[Trainer] Loaded champion norms & 18-class plain head from {cand}")
                    head_loaded = True
                    break

        pool_list = all_pools if all_pools else ([pool_file] if pool_file else [])
        dfs = []
        for p in pool_list:
            try:
                if str(p).endswith(".parquet"):
                    df_one = pd.read_parquet(p, columns=["text", "model"])
                else:
                    df_one = pd.read_csv(p, usecols=["text", "model"])
                dfs.append(df_one)
            except Exception:
                pass
        df_p = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["text", "model"])

        df_tr = None
        if train_path and Path(train_path).exists():
            try:
                df_tr = pd.read_csv(train_path)
            except Exception:
                pass

        eval_texts, eval_labels = None, None
        bench_paths = list(input_dir.rglob("clean_benchmark_1800_rows.parquet"))
        if bench_paths:
            try:
                df_bench = pd.read_parquet(bench_paths[0])
                eval_texts = df_bench["text"].tolist()
                eval_labels = [LAB2I[m] for m in df_bench["model"].tolist()]
                print(f"[Trainer] Loaded {len(eval_texts)} benchmark samples from {bench_paths[0]}")
            except Exception as e:
                print(f"[Trainer] Note on reading clean benchmark: {e}")

        exp_cfg = train_cfg.get("export_embeddings", {})
        if isinstance(exp_cfg, bool):
            exp_enabled = exp_cfg
            exp_save_all = bool(train_cfg.get("save_all_layers", False))
            exp_out = train_cfg.get("embeddings_output_dir")
        elif isinstance(exp_cfg, dict):
            exp_enabled = bool(exp_cfg.get("enabled", True))
            exp_save_all = bool(exp_cfg.get("save_all_layers", False))
            exp_out = exp_cfg.get("output_dir") or train_cfg.get("embeddings_output_dir")
        else:
            exp_enabled = bool(train_cfg.get("export_embeddings", False))
            exp_save_all = bool(train_cfg.get("save_all_layers", False))
            exp_out = train_cfg.get("embeddings_output_dir")

        gate_res = train_family_gate(
            model=model,
            tok=tok,
            df_pool=df_p,
            df_train=df_tr,
            target_family=target_family or "gemini",
            samples_per_class=int(train_cfg.get("samples_per_class", 3800)),
            negative_ratio=float(train_cfg.get("negative_ratio", 0.10)),
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            batch_size=int(train_cfg.get("batch_size", 32)),
            max_len=int(train_cfg.get("max_len", 384)),
            end_id=end_id,
            device=device,
            output_dir=out_dir,
            checkpoints_volume=checkpoints_volume,
            head_type=train_cfg.get("head_type", "mlp"),
            hidden_dim=int(train_cfg.get("hidden_dim", 256)),
            mlp_epochs=int(train_cfg.get("mlp_epochs", 20)),
            mlp_lr=float(train_cfg.get("mlp_lr", 3e-4)),
            mlp_dropout=float(train_cfg.get("mlp_dropout", 0.20)),
            export_embeddings=exp_enabled,
            save_all_layers=exp_save_all,
            embeddings_output_dir=exp_out,
        )
        print(f"[Trainer] Phase 3 Family Gate Training Complete: {gate_res}")
        return out_dir

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
        hard_set = set(HARD_CLASS_INDICES)
        if len(df_p) > max_s:
            cls_multipliers = {lbl: (1.30 if LAB2I[lbl] in hard_set else 1.0) for lbl in active_labels}
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
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 24.0 and bs1 > 4:
                old_bs1 = bs1
                bs1 = 2
                accum1 = max(accum1 * (old_bs1 // 2), 8)
                print(f"[Trainer] Low VRAM GPU ({vram_gb:.1f} GiB) detected: adjusted Stage 1 batch_size from {old_bs1} to {bs1}, accum_steps to {accum1}")
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

        stage1_hard_weight = float(stage1_cfg.get("hard_class_weight", train_cfg.get("hard_class_weight", 1.0)))
        stage1_class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
        for lbl, local_idx in sub_lab2i.items():
            if LAB2I.get(lbl) in hard_set:
                stage1_class_weights[local_idx] = stage1_hard_weight

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
                    loss_sup = compute_supcon_loss(Z_s1, labels_dev, temperature=0.10, prototypes=model.plain_head.weight.detach())
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

                save_steps = int(stage1_cfg.get("save_steps", 500))
                if global_step1 % save_steps == 0:
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
                        print(f"[Trainer Stage 1] Intermediate checkpoint auto-saved at step {global_step1} to {out_dir}")
                        if step_callback is not None:
                            step_callback("checkpoint_saved", global_step1)
                    except Exception as e:
                        print(f"[Trainer Stage 1 Warning] Auto-save failed at step {global_step1}: {e}")

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
            if step_callback is not None:
                step_callback("checkpoint_saved", "stage1_finished")
        except Exception as e:
            print(f"[Trainer] Warning: could not export Stage 1 intermediate checkpoint: {e}")

    # Stage 2: Prompt-Grouped Fine-Tuning on train.csv
    stage2_cfg = train_cfg.get("stage2", {})
    freeze_backbone_s2 = bool(stage2_cfg.get("freeze_backbone", False))
    if freeze_backbone_s2:
        print("[Trainer] Freezing backbone parameters for Stage 2 (exclusively training norms, plain_head, and centered_head)...")
        for p in model.backbone.parameters():
            p.requires_grad = False
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

    # SWA setup: average parameters across the final epoch of Stage 2 (accumulate in FP32)
    swa_enabled = bool(stage2_cfg.get("swa", True))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    swa_shadow = {p: torch.zeros_like(p.data, dtype=torch.float32, device="cpu") for p in trainable_params} if swa_enabled else {}
    swa_steps = 0

    # Hard-Class Loss Upweighting for Stage 2 fine-tuning (default to 1.0 to preserve unbiased prior margins when Sinkhorn is disabled)
    stage2_hard_weight = float(stage2_cfg.get("hard_class_weight", 1.0))
    stage2_class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    if stage2_hard_weight != 1.0:
        hard_set = set(HARD_CLASS_INDICES)
        for lbl, local_idx in sub_lab2i.items():
            if LAB2I.get(lbl) in hard_set:
                stage2_class_weights[local_idx] = stage2_hard_weight

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
            p.data.copy_((swa_shadow[p] / float(swa_steps)).to(p.device, p.dtype))
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
    if step_callback is not None:
        try:
            step_callback("checkpoint_saved", "training_complete")
        except Exception as e:
            print(f"[Trainer Callback Warning] {e}")
    print(f"[Trainer] Checkpoint exported to {out_dir}")
    return out_dir


def fit_family_gate_head(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    num_gate_classes: int,
    alphas: list[float] | None = None,
    label_smoothing: float = 0.05,
    Z_eval: np.ndarray | None = None,
    y_eval_true: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    num_family_classes: int | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray | None]:
    """
    Fits an L2-regularized Plain Ridge Head for intra-family gating using GPU-accelerated PyTorch solve.
    Returns: (best_W, best_b, best_alpha, best_eval_acc, best_eval_logits)
    """
    if alphas is None:
        alphas = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N_tr, D_feat = Z_train.shape
    Y_onehot = np.zeros((N_tr, num_gate_classes), dtype=np.float32)
    for i, lab in enumerate(y_train):
        Y_onehot[i, lab] = 1.0
    Y_smooth = (1.0 - label_smoothing) * Y_onehot + (label_smoothing / float(num_gate_classes))

    K_eval_classes = num_family_classes if num_family_classes is not None else num_gate_classes

    # Move to PyTorch tensors on GPU for blazing-fast linear algebra
    Z_t = torch.from_numpy(Z_train).to(device, dtype=torch.float32)
    Y_t = torch.from_numpy(Y_smooth).to(device, dtype=torch.float32)

    # Compute mu_Z strictly on the intra-family training rows so the shared family mean cancels out
    fam_mask = y_train < K_eval_classes
    fam_mask_t = torch.from_numpy(fam_mask).to(device)
    if fam_mask_t.sum() > 0:
        mu_Z = Z_t[fam_mask_t].mean(dim=0, keepdim=True)
    else:
        mu_Z = Z_t.mean(dim=0, keepdim=True)

    mu_Y = Y_t.mean(dim=0, keepdim=True)
    Z_c = Z_t - mu_Z
    Y_c = Y_t - mu_Y

    # trace(Z_c @ Z_c.T) is simply the sum of squares ||Z_c||_F^2!
    scale = float(torch.sum(Z_c ** 2).item() / N_tr)

    use_primal = (N_tr >= D_feat)
    if use_primal:
        Cov = Z_c.T @ Z_c
        Zt_Y = Z_c.T @ Y_c
    else:
        K_mat = Z_c @ Z_c.T

    best_w, best_b = None, None
    best_acc = 0.0
    best_alpha = 1.0
    best_eval_logits = None

    for alpha in alphas:
        eff_alpha = alpha * scale
        if use_primal:
            reg = eff_alpha * torch.eye(D_feat, device=device, dtype=torch.float32)
            W_head = torch.linalg.solve(Cov + reg, Zt_Y)
        else:
            reg = eff_alpha * torch.eye(N_tr, device=device, dtype=torch.float32)
            A_mat = torch.linalg.solve(K_mat + reg, Y_c)
            W_head = Z_c.T @ A_mat
        b_head = (mu_Y - mu_Z @ W_head).squeeze(0)

        W_np = W_head.cpu().numpy()
        b_np = b_head.cpu().numpy()

        fam_logits_tr = (Z_train[fam_mask] @ W_np + b_np)[:, :K_eval_classes]
        row_std = float(np.std(fam_logits_tr, axis=1).mean())
        temp_scale = (3.2 / row_std) if row_std > 1e-4 else 1.0
        W_np = (W_np * temp_scale).astype(np.float32)
        b_np = (b_np * temp_scale).astype(np.float32)

        if Z_eval is not None and y_eval_true is not None and eval_mask is not None:
            logits_ev = Z_eval @ W_np + b_np
            fam_pred = logits_ev[eval_mask, :K_eval_classes].argmax(axis=1)
            acc = float(balanced_accuracy_score(y_eval_true, fam_pred))
            if acc > best_acc or best_w is None:
                best_acc = acc
                best_w = W_np
                best_b = b_np
                best_alpha = alpha
                best_eval_logits = logits_ev
        else:
            if best_w is None:
                best_w = W_np
                best_b = b_np
                best_alpha = alpha

    # Clean up GPU memory
    del Z_t, Y_t, Z_c, Y_c
    if use_primal:
        del Cov, Zt_Y
    else:
        del K_mat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_w, best_b, best_alpha, best_acc, best_eval_logits


class FamilyMLPHead(nn.Module):
    """
    2-Layer Non-Linear MLP Head for intra-family sister decoupling (Phase 3):
    InputDropout(in_dropout) -> Linear(D -> hidden_dim) -> GELU -> LayerNorm(hidden_dim) -> Dropout(p) -> Linear(hidden_dim -> num_classes)
    """

    def __init__(
        self,
        in_features: int = 40960,
        hidden_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.25,
        in_dropout: float = 0.50,
    ):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.in_drop(x))
        h = self.act(h)
        h = self.ln(h)
        h = self.drop(h)
        return self.fc2(h)

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.ln(h)
        return self.fc2(h)


def fit_family_mlp_head(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    num_gate_classes: int,
    num_family_classes: int,
    hidden_dim: int = 128,
    epochs: int = 20,
    lr: float = 2e-4,
    dropout: float = 0.25,
    weight_decay: float = 0.05,
    batch_size: int = 64,
    label_smoothing: float = 0.05,
    Z_eval: np.ndarray | None = None,
    y_eval_true: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], float, np.ndarray | None]:
    """
    Trains a regularized 2-Layer Non-Linear MLP Head on standardized intra-family residuals
    with sibling margin penalties and folds both (mu_family, sd_family) into Layer 1 (W1*, b1*).
    """
    from sklearn.model_selection import StratifiedKFold

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N_tr, D_feat = Z_train.shape

    # 1. Compute intra-family centroid and per-feature standard deviation
    fam_mask_tr = y_train < num_family_classes
    if fam_mask_tr.sum() > 1:
        mu_family = Z_train[fam_mask_tr].mean(axis=0, keepdims=True).astype(np.float32)
        sd_family = Z_train[fam_mask_tr].std(axis=0, keepdims=True).clip(min=1e-3).astype(np.float32)
    else:
        mu_family = Z_train.mean(axis=0, keepdims=True).astype(np.float32)
        sd_family = Z_train.std(axis=0, keepdims=True).clip(min=1e-3).astype(np.float32)

    Z_tr_std = ((Z_train - mu_family) / sd_family).astype(np.float32)

    # 2. Construct an internal 15% stratified validation split on Z_train to prevent benchmark overfitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fit_idx, val_idx = next(skf.split(Z_tr_std, y_train))

    X_fit_t = torch.from_numpy(Z_tr_std[fit_idx]).float()
    y_fit_t = torch.from_numpy(y_train[fit_idx]).long()
    X_val_t = torch.from_numpy(Z_tr_std[val_idx]).float().to(device)
    y_val_np = y_train[val_idx]
    val_fam_mask = y_val_np < num_family_classes

    ds = torch.utils.data.TensorDataset(X_fit_t, y_fit_t)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    mlp = FamilyMLPHead(
        in_features=D_feat,
        hidden_dim=hidden_dim,
        num_classes=num_gate_classes,
        dropout=dropout,
        in_dropout=0.50,
    ).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    # Sibling margin mask for num_gate_classes (boost Gemini 3.7 <-> 3.8 or GPT-5.6 triad)
    sib_mask = torch.zeros((num_gate_classes, num_gate_classes), dtype=torch.float32, device=device)
    for i in range(num_family_classes):
        for j in range(num_family_classes):
            if i != j:
                sib_mask[i, j] = 1.0
    if num_family_classes == 4:  # Gemini: 3.7-flash (2) <-> 3.8-flash (3)
        sib_mask[2, 3] = 1.6
        sib_mask[3, 2] = 1.6
    elif num_family_classes == 5:  # GPT: 5.6 triad (1, 2, 3)
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i != j:
                    sib_mask[i, j] = 1.4

    best_val_acc = -1.0
    best_state = None

    for ep in range(epochs):
        mlp.train()
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = mlp(b_x)
            margin_boost = sib_mask[b_y] * 0.35
            loss = F.cross_entropy(out + margin_boost, b_y, label_smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        mlp.eval()
        with torch.no_grad():
            val_logits = mlp.forward_eval(X_val_t).cpu().numpy()
        if val_fam_mask.sum() > 0:
            val_acc = float(
                balanced_accuracy_score(
                    y_val_np[val_fam_mask],
                    val_logits[val_fam_mask, :num_family_classes].argmax(axis=1),
                )
            )
        else:
            val_acc = float(balanced_accuracy_score(y_val_np, val_logits.argmax(axis=1)))

        if val_acc >= best_val_acc or best_state is None:
            best_val_acc = val_acc
            best_state = copy.deepcopy(mlp.state_dict())

    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()

    # 3. Zero-Cost Affine Fold-In of (mu_family, sd_family) into fc1:
    # ((Z - mu) / sd) @ W1.T + b1 = Z @ (W1 / sd).T + (b1 - (mu / sd) @ W1.T)
    with torch.no_grad():
        W1 = mlp.fc1.weight.detach().float()
        b1 = mlp.fc1.bias.detach().float()
        mu_t = torch.from_numpy(mu_family).to(device=W1.device, dtype=torch.float32)
        sd_t = torch.from_numpy(sd_family).to(device=W1.device, dtype=torch.float32)
        W1_folded = (W1 / sd_t).contiguous()
        b1_folded = (b1 - ((mu_t / sd_t) @ W1.T).squeeze(0)).contiguous()
        mlp.fc1.weight.copy_(W1_folded)
        mlp.fc1.bias.copy_(b1_folded)

    # 4. Evaluate folded MLP directly on raw uncentered Z_eval (verifying exact fast-path equivalence)
    best_eval_acc = best_val_acc
    best_eval_logits = None
    if Z_eval is not None:
        with torch.no_grad():
            Z_ev_raw_t = torch.from_numpy(Z_eval).float().to(device)
            best_eval_logits = mlp.forward_eval(Z_ev_raw_t).cpu().numpy()
            del Z_ev_raw_t
        if y_eval_true is not None and eval_mask is not None and eval_mask.sum() > 0:
            fam_pred = best_eval_logits[eval_mask, :num_family_classes].argmax(axis=1)
            best_eval_acc = float(balanced_accuracy_score(y_eval_true, fam_pred))

    # 5. Fit closed-form Ridge head for a valid linear fallback in plain.weight / plain.bias
    W_ridge, b_ridge, _, _, _ = fit_family_gate_head(
        Z_train=Z_train,
        y_train=y_train,
        num_gate_classes=num_gate_classes,
        Z_eval=Z_eval,
        y_eval_true=y_eval_true,
        eval_mask=eval_mask,
        num_family_classes=num_family_classes,
        device=device,
    )

    head_dict = {
        "mlp_w1": mlp.fc1.weight.detach().cpu().to(torch.float32).contiguous(),
        "mlp_b1": mlp.fc1.bias.detach().cpu().to(torch.float32).contiguous(),
        "ln_weight": mlp.ln.weight.detach().cpu().to(torch.float32).contiguous(),
        "ln_bias": mlp.ln.bias.detach().cpu().to(torch.float32).contiguous(),
        "mlp_w2": mlp.fc2.weight.detach().cpu().to(torch.float32).contiguous(),
        "mlp_b2": mlp.fc2.bias.detach().cpu().to(torch.float32).contiguous(),
        "plain.weight": torch.from_numpy(W_ridge.T).to(torch.float32).contiguous(),
        "plain.bias": torch.from_numpy(b_ridge).to(torch.float32).contiguous(),
    }

    del X_fit_t, y_fit_t, X_val_t, ds, mlp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return head_dict, best_eval_acc, best_eval_logits


def find_compatible_champion_lora(
    input_dir: Path,
    backbone_name: str,
    lora_r: int = 32,
    target_family: str = "",
) -> Path | None:
    """
    Finds the matching fine-tuned LoRA adapter directory for the given backbone and rank.
    Pre-validates adapter_config.json to ensure:
    1. Rank r matches expected lora_r (e.g. 32 vs 16).
    2. Backbone family matches (e.g. Qwen3 vs Qwen2.5 vs DeBERTa vs Llama).
    3. Never greedily returns an incompatible adapter (e.g. Qwen2.5-14B when running Qwen3-14B).
    """
    if not input_dir or not Path(input_dir).exists():
        return None

    input_p = Path(input_dir)
    candidates = list(input_p.rglob("adapter_model.safetensors")) + list(input_p.rglob("adapter_model.bin"))
    if not candidates:
        return None

    b_lower = backbone_name.lower()
    is_qwen3 = "qwen3" in b_lower or ("qwen" in b_lower and "2" not in b_lower and "2.5" not in b_lower)
    is_qwen25 = "qwen2" in b_lower or "2.5" in b_lower
    is_deberta = "deberta" in b_lower

    matching_dirs: list[tuple[int, Path]] = []

    for cand in candidates:
        p_dir = cand.parent
        p_str = str(p_dir).lower().replace("\\", "/")

        if target_family:
            other_fams = [f for f in FAMILIES.keys() if f != target_family.lower()]
            if any(f"export_{f}" in p_str or f"{f}_gate" in p_str or f"{f}_spec" in p_str for f in other_fams):
                continue

        cfg_p = p_dir / "adapter_config.json"
        if not cfg_p.exists():
            cfg_p = p_dir.parent / "adapter_config.json"

        score = 0
        if cfg_p.exists():
            try:
                with open(cfg_p, "r", encoding="utf-8") as f:
                    c_data = json.load(f)

                r_found = int(c_data.get("r", 0))
                if lora_r > 0 and r_found > 0 and r_found != lora_r:
                    continue

                base_str = str(c_data.get("base_model_name_or_path", "")).lower()

                if is_qwen3 and ("qwen2" in base_str or "2.5" in base_str or "llama" in base_str):
                    continue
                if is_qwen25 and ("qwen3" in base_str or "llama" in base_str) and "2.5" not in base_str and "qwen2" not in base_str:
                    continue
                if is_deberta and "deberta" not in base_str and base_str:
                    continue

                if is_qwen3 and "qwen3" in base_str:
                    score += 10
                if is_qwen25 and ("qwen2" in base_str or "2.5" in base_str):
                    score += 10
            except Exception:
                pass

        if "export_champion" in p_str:
            score += 20
        if "inhouse" in p_str:
            score += 15
        if is_qwen3 and "qwen3_14b" in p_str:
            score += 10
        if is_qwen25 and ("qwen25" in p_str or "qwen2" in p_str):
            score += 10

        if is_qwen3 and ("qwen2" in p_str or "2.5" in p_str or "8b" in p_str):
            continue
        if is_qwen25 and ("qwen3" in p_str or "8b" in p_str):
            continue

        matching_dirs.append((score, p_dir))

    if not matching_dirs:
        return None

    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    return matching_dirs[0][1]


def load_and_slice_champion_heads(
    heads_path: Path,
    model: nn.Module,
    fam_cols: list[int] | None = None,
) -> bool:
    """
    Loads champion per-layer LayerNorms (model.norms) and plain_head weights into model.
    If the checkpoint contains an 18-class head and model.plain_head has len(fam_cols) rows,
    slices the target family classes; otherwise loads the full 18-class head.
    """
    if not Path(heads_path).exists():
        return False

    try:
        if str(heads_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            h_sd = load_file(str(heads_path))
        else:
            h_sd = torch.load(str(heads_path), map_location="cpu")

        # 1. Always load trained per-layer LayerNorm parameters
        if hasattr(model, "norms") and model.norms is not None:
            for i, norm in enumerate(model.norms):
                wk, bk = f"norms.{i}.weight", f"norms.{i}.bias"
                if wk in h_sd and norm.weight.shape == h_sd[wk].shape:
                    norm.weight.data.copy_(h_sd[wk].to(norm.weight.device, dtype=norm.weight.dtype))
                if bk in h_sd and hasattr(norm, "bias") and norm.bias is not None and norm.bias.shape == h_sd[bk].shape:
                    norm.bias.data.copy_(h_sd[bk].to(norm.bias.device, dtype=norm.bias.dtype))

        if not hasattr(model, "plain_head") or model.plain_head is None:
            return False

        w_ckpt = h_sd.get("plain.weight") if "plain.weight" in h_sd else h_sd.get("weight", None)
        b_ckpt = h_sd.get("plain.bias") if "plain.bias" in h_sd else h_sd.get("bias", None)

        if w_ckpt is None or b_ckpt is None:
            return False

        tgt_w = model.plain_head.weight
        tgt_b = model.plain_head.bias

        # Case A: Exact shape match (e.g., 18-class champion head into 18-class model)
        if tgt_w.shape == w_ckpt.shape and tgt_b.shape == b_ckpt.shape:
            tgt_w.data.copy_(w_ckpt.to(tgt_w.device, dtype=tgt_w.dtype))
            tgt_b.data.copy_(b_ckpt.to(tgt_b.device, dtype=tgt_b.dtype))
            return True

        # Case B: Checkpoint is 18-class, model is a Stage 1/2 specialist (e.g., 4-class)
        if fam_cols is not None and w_ckpt.shape[0] >= max(fam_cols) + 1:
            sliced_w = w_ckpt[fam_cols]
            sliced_b = b_ckpt[fam_cols]
            if sliced_w.shape == tgt_w.shape and sliced_b.shape == tgt_b.shape:
                tgt_w.data.copy_(sliced_w.to(tgt_w.device, dtype=tgt_w.dtype))
                tgt_b.data.copy_(sliced_b.to(tgt_b.device, dtype=tgt_b.dtype))
                return True

        return False
    except Exception as e:
        print(f"[Trainer] Note on loading/slicing champion head: {e}")
        return False


def save_extracted_features_bundle(
    output_dir: Path | str,
    target: str,
    features_concat: np.ndarray,
    features_by_layer: np.ndarray | None,
    texts: list[str],
    labels: list[int] | np.ndarray,
    model_names: list[str],
    prompt_ids: list[Any] | None = None,
    eval_features_concat: np.ndarray | None = None,
    eval_features_by_layer: np.ndarray | None = None,
    eval_texts: list[str] | None = None,
    eval_labels: list[int] | None = None,
    eval_model_names: list[str] | None = None,
    dataset_title: str | None = None,
    dataset_slug: str | None = None,
    checkpoints_volume: Any | None = None,
) -> Path:
    """
    Saves extracted features and metadata in Kaggle Dataset format:
    1. features_concat.safetensors: FP16 tensor [N, D]
    2. features_by_layer.safetensors: FP16 tensor [N, L, hidden_size] (if provided)
    3. metadata.parquet: [id, model, label, prompt_id, text_hash, char_len]
    4. benchmark_features_concat.safetensors & benchmark_metadata.parquet (if eval provided)
    5. dataset-metadata.json: For instant 1-click `kaggle datasets create`
    6. features_summary.json: Human-readable inspection stats
    """
    from safetensors.torch import save_file

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"\n[Features Export] Saving features bundle for target '{target}' to: {out_p}...")

    # 1. Save concatenated representations in FP16
    feat_cat_t = torch.from_numpy(features_concat).to(torch.float16)
    save_file({"features_concat": feat_cat_t}, str(out_p / "features_concat.safetensors"))
    sz_cat_mb = (out_p / "features_concat.safetensors").stat().st_size / (1024**2)
    print(f"  Saved features_concat.safetensors: shape={list(feat_cat_t.shape)}, size={sz_cat_mb:.1f} MB")

    # 2. Save layer-by-layer representations if provided
    sz_layer_mb = 0.0
    if features_by_layer is not None:
        feat_layers_t = torch.from_numpy(features_by_layer).to(torch.float16)
        save_file({"features_by_layer": feat_layers_t}, str(out_p / "features_by_layer.safetensors"))
        sz_layer_mb = (out_p / "features_by_layer.safetensors").stat().st_size / (1024**2)
        print(f"  Saved features_by_layer.safetensors: shape={list(feat_layers_t.shape)}, size={sz_layer_mb:.1f} MB")

    # 3. Save training metadata parquet
    meta_df = pd.DataFrame({
        "id": np.arange(len(texts), dtype=np.int64),
        "model": model_names,
        "label": np.asarray(labels, dtype=np.int64),
        "prompt_id": prompt_ids if prompt_ids is not None else [-1] * len(texts),
        "text_hash": [_text_hash(t) for t in texts],
        "char_len": [len(t) for t in texts],
    })
    meta_df.to_parquet(out_p / "metadata.parquet", index=False)
    print(f"  Saved metadata.parquet: {len(meta_df):,} rows")

    # 4. Save benchmark holdout representations if provided
    if eval_features_concat is not None and eval_texts is not None:
        ev_cat_t = torch.from_numpy(eval_features_concat).to(torch.float16)
        save_file({"benchmark_features_concat": ev_cat_t}, str(out_p / "benchmark_features_concat.safetensors"))
        if eval_features_by_layer is not None:
            ev_layer_t = torch.from_numpy(eval_features_by_layer).to(torch.float16)
            save_file({"benchmark_features_by_layer": ev_layer_t}, str(out_p / "benchmark_features_by_layer.safetensors"))

        eval_models = eval_model_names if eval_model_names is not None else [
            LABELS[i] if i < len(LABELS) else f"class_{i}" for i in (eval_labels if eval_labels is not None else [0] * len(eval_texts))
        ]
        bench_df = pd.DataFrame({
            "id": np.arange(len(eval_texts), dtype=np.int64),
            "model": eval_models,
            "label": np.asarray(eval_labels, dtype=np.int64) if eval_labels is not None else np.zeros(len(eval_texts), dtype=np.int64),
            "text_hash": [_text_hash(t) for t in eval_texts],
            "char_len": [len(t) for t in eval_texts],
        })
        bench_df.to_parquet(out_p / "benchmark_metadata.parquet", index=False)
        print(f"  Saved benchmark features & metadata: {len(bench_df):,} samples")

    # 5. Save Kaggle dataset-metadata.json
    slug = dataset_slug or f"ntl-{target.lower()}-features-qwen3-14b"
    title = dataset_title or f"NTL {target.upper()} Extracted Features (Qwen3-14B)"
    kaggle_meta = {
        "title": title,
        "id": f"anilmittal/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            f"Pre-extracted high-dimensional feature representations for {target.upper()} models "
            f"using Champion Multi-Layer Qwen3-14B backbone with causal attention-sink masking."
        ),
    }
    (out_p / "dataset-metadata.json").write_text(json.dumps(kaggle_meta, indent=2), encoding="utf-8")

    # 6. Save inspection summary JSON
    summary = {
        "target": target,
        "num_train_samples": len(texts),
        "num_classes": len(set(model_names)),
        "class_counts": {k: int(v) for k, v in pd.Series(model_names).value_counts().items()},
        "feature_dim_concat": int(features_concat.shape[1]),
        "features_by_layer_shape": list(features_by_layer.shape) if features_by_layer is not None else None,
        "has_benchmark": eval_features_concat is not None,
        "num_benchmark_samples": len(eval_texts) if eval_texts is not None else 0,
        "total_bundle_size_mb": round(sz_cat_mb + sz_layer_mb, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    (out_p / "features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if checkpoints_volume is not None and hasattr(checkpoints_volume, "commit"):
        try:
            checkpoints_volume.commit()
        except Exception:
            pass

    print(f"[Features Export] Complete! Bundle ready for Kaggle Dataset creation at: {out_p} ({summary['total_bundle_size_mb']:.1f} MB)\n")
    return out_p


def extract_and_export_embeddings(
    model: MultiLayerLoRAClassifier,
    tok: Any,
    df_pool: pd.DataFrame,
    df_train: pd.DataFrame | None = None,
    target: str = "gemini",
    save_all_layers: bool = False,
    output_dir: Path | str = "export_embeddings",
    samples_per_class: int = 0,
    negative_ratio: float = 0.10,
    eval_texts: list[str] | None = None,
    eval_labels: list[int] | None = None,
    eval_model_names: list[str] | None = None,
    batch_size: int = 16,
    max_len: int = 384,
    end_id: int = 151643,
    device: torch.device | None = None,
    dataset_title: str | None = None,
    dataset_slug: str | None = None,
    checkpoints_volume: Any | None = None,
) -> Path:
    """
    Configurable feature extraction and dataset export pipeline.
    Filters data for target ('all', 'gemini', 'gpt', 'claude', 'glm', 'gold'),
    extracts representations batch-by-batch across model layers, and packages into a Kaggle Dataset.
    """
    if device is None:
        device = next(model.parameters()).device

    target_clean = target.lower().strip()
    print("\n==========================================================================================")
    print(f" [Features Extraction] Target: '{target_clean}' | Save All Layers: {save_all_layers} | Max Len: {max_len}")
    print("==========================================================================================")

    # 1. Zero leakage quarantining against clean benchmark holdout
    if eval_texts is not None and len(eval_texts) > 0:
        bench_hashes = {_text_hash(t) for t in eval_texts}
        if df_pool is not None and len(df_pool) > 0:
            init_p = len(df_pool)
            pool_hashes = [_text_hash(t) for t in df_pool["text"].fillna("")]
            df_pool = df_pool[[h not in bench_hashes for h in pool_hashes]].copy()
            quar_p = init_p - len(df_pool)
            if quar_p > 0:
                print(f"[Features Extraction] Quarantined {quar_p} pool samples matching benchmark.")
        if df_train is not None and len(df_train) > 0:
            init_t = len(df_train)
            tr_hashes = [_text_hash(t) for t in df_train["text"].fillna("")]
            df_train = df_train[[h not in bench_hashes for h in tr_hashes]].copy()
            quar_t = init_t - len(df_train)
            if quar_t > 0:
                print(f"[Features Extraction] Quarantined {quar_t} gold train samples matching benchmark.")

    # 2. Data Selection according to target
    if target_clean == "gold" or target_clean == "train":
        assert df_train is not None and len(df_train) > 0, "No gold df_train provided for target='gold'"
        df_target_all = df_train.copy()
        df_neg_all = pd.DataFrame()
    elif target_clean == "all":
        # All 18 classes
        all_dfs = []
        for c in LABELS:
            sub_tr = df_train[df_train["model"] == c] if (df_train is not None and len(df_train) > 0) else pd.DataFrame()
            sub_pool = df_pool[df_pool["model"] == c] if (df_pool is not None and len(df_pool) > 0) else pd.DataFrame()
            if samples_per_class > 0:
                n_take = min(samples_per_class - len(sub_tr), len(sub_pool))
                sampled_pool = sub_pool.sample(n=max(n_take, 0), random_state=42) if n_take > 0 else pd.DataFrame()
            else:
                sampled_pool = sub_pool
            c_df = pd.concat([sub_tr, sampled_pool], ignore_index=True)
            all_dfs.append(c_df)
            print(f"  Class {c}: {len(c_df)} samples ({len(sub_tr)} gold, {len(c_df)-len(sub_tr)} pool)")
        df_target_all = pd.concat(all_dfs, ignore_index=True)
        df_neg_all = pd.DataFrame()
    elif target_clean in FAMILIES:
        target_cols = FAMILIES[target_clean]
        target_classes = [LABELS[i] for i in target_cols]
        cap = samples_per_class if samples_per_class > 0 else 3800
        fam_dfs = []
        for c in target_classes:
            sub_tr = df_train[df_train["model"] == c] if (df_train is not None and len(df_train) > 0) else pd.DataFrame()
            sub_pool = df_pool[df_pool["model"] == c] if (df_pool is not None and len(df_pool) > 0) else pd.DataFrame()
            n_take = min(cap - len(sub_tr), len(sub_pool))
            sampled_pool = sub_pool.sample(n=max(n_take, 0), random_state=42) if n_take > 0 else pd.DataFrame()
            c_df = pd.concat([sub_tr, sampled_pool], ignore_index=True)
            fam_dfs.append(c_df)
            print(f"  {target_clean.upper()} class {c}: {len(c_df)} samples ({len(sub_tr)} gold, {len(c_df)-len(sub_tr)} pool)")
        df_target_all = pd.concat(fam_dfs, ignore_index=True)

        # Distractor sampling if negative_ratio > 0
        if negative_ratio > 0.0 and df_pool is not None and len(df_pool) > 0:
            n_neg = int(len(df_target_all) * negative_ratio)
            df_non_fam = df_pool[~df_pool["model"].isin(target_classes)].copy()
            other_fams = [f for f in ["gpt", "gemini", "claude", "glm", "qwen", "kimi"] if f != target_clean]
            neg_dfs = []
            n_per_fam = max(n_neg // max(len(other_fams), 1), 1)
            for fam in other_fams:
                fam_sub = df_non_fam[df_non_fam["model"].str.startswith(fam)]
                if len(fam_sub) > 0:
                    neg_dfs.append(fam_sub.sample(n=min(n_per_fam, len(fam_sub)), random_state=42))
            df_neg_all = pd.concat(neg_dfs, ignore_index=True) if neg_dfs else pd.DataFrame()
            print(f"  Sampled {len(df_neg_all)} negative distractors across non-{target_clean.upper()} families.")
        else:
            df_neg_all = pd.DataFrame()
    else:
        raise ValueError(f"Unknown target '{target}'. Must be one of: 'all', 'gold', or a valid family {list(FAMILIES.keys())}.")

    # Assemble final extraction list
    train_texts = df_target_all["text"].tolist() + (df_neg_all["text"].tolist() if len(df_neg_all) > 0 else [])
    model_names = df_target_all["model"].tolist() + (["__OTHER__"] * len(df_neg_all) if len(df_neg_all) > 0 else [])

    if target_clean in FAMILIES:
        ordered_models = [LABELS[i] for i in FAMILIES[target_clean]] + (["__OTHER__"] if len(df_neg_all) > 0 else [])
    else:
        ordered_models = list(LABELS) + (["__OTHER__"] if len(df_neg_all) > 0 else [])
    lab_map = {m: i for i, m in enumerate(ordered_models)}
    train_labels = [lab_map[m] for m in model_names]

    target_pids = (
        df_target_all["prompt_id"].fillna("-1").astype(str).tolist()
        if "prompt_id" in df_target_all.columns
        else ["-1"] * len(df_target_all)
    )
    neg_pids = (
        df_neg_all["prompt_id"].fillna("-1").astype(str).tolist()
        if (len(df_neg_all) > 0 and "prompt_id" in df_neg_all.columns)
        else ["-1"] * len(df_neg_all)
    )
    prompt_ids = target_pids + neg_pids

    amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
        else torch.float16
    )

    print(f"[Features Extraction] Extracting features on {len(train_texts):,} samples (batch_size={batch_size})...")

    # 3. Extract Train Features
    ds_tr = TextSequenceDataset(train_texts, train_labels, None, tok, max_len=max_len, end_id=end_id)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, collate_fn=collate_text_batch)

    Z_cat_list = []
    Z_layers_list = []
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=amp_dtype):
            for batch in loader_tr:
                mb_ids = batch["input_ids"].to(device)
                mb_mask = batch["attention_mask"].to(device)
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                z_b, z_l = model.forward_features(
                    mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True,
                    return_by_layer=True, extract_all_layers=save_all_layers,
                )
                Z_cat_list.append(z_b.cpu().float().numpy())
                Z_layers_list.append(z_l.cpu().to(torch.float16).numpy())

    features_concat = np.vstack(Z_cat_list)
    features_by_layer = np.vstack(Z_layers_list)

    # 4. Extract Benchmark Features if provided
    eval_cat = None
    eval_layers = None
    if eval_texts is not None and len(eval_texts) > 0:
        print(f"[Features Extraction] Extracting benchmark features on {len(eval_texts)} holdout samples...")
        dummy_y = [0] * len(eval_texts)
        ds_ev = TextSequenceDataset(eval_texts, dummy_y, None, tok, max_len=max_len, end_id=end_id)
        loader_ev = DataLoader(ds_ev, batch_size=batch_size, shuffle=False, collate_fn=collate_text_batch)
        ev_cat_list = []
        ev_layers_list = []
        with torch.no_grad():
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                for batch in loader_ev:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    z_b, z_l = model.forward_features(
                        mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True,
                        return_by_layer=True, extract_all_layers=save_all_layers,
                    )
                    ev_cat_list.append(z_b.cpu().float().numpy())
                    ev_layers_list.append(z_l.cpu().to(torch.float16).numpy())
        eval_cat = np.vstack(ev_cat_list)
        eval_layers = np.vstack(ev_layers_list)

    # 5. Save features bundle
    out_dir_path = save_extracted_features_bundle(
        output_dir=output_dir,
        target=target_clean,
        features_concat=features_concat,
        features_by_layer=features_by_layer,
        texts=train_texts,
        labels=train_labels,
        model_names=model_names,
        prompt_ids=prompt_ids,
        eval_features_concat=eval_cat,
        eval_features_by_layer=eval_layers,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        eval_model_names=eval_model_names,
        dataset_title=dataset_title,
        dataset_slug=dataset_slug,
        checkpoints_volume=checkpoints_volume,
    )
    return out_dir_path


def train_family_gate(
    model: nn.Module,
    tok: Any,
    df_pool: pd.DataFrame,
    df_train: pd.DataFrame | None,
    target_family: str,
    samples_per_class: int = 0,
    negative_ratio: float = 0.10,
    eval_texts: list[str] | None = None,
    eval_labels: list[int] | None = None,
    batch_size: int = 32,
    max_len: int = 384,
    end_id: int = 151643,
    device: torch.device | None = None,
    output_dir: Path | str | None = None,
    checkpoints_volume: Any | None = None,
    head_type: str = "mlp",
    hidden_dim: int = 256,
    mlp_epochs: int = 20,
    mlp_lr: float = 3e-4,
    mlp_dropout: float = 0.20,
    export_embeddings: bool = False,
    save_all_layers: bool = False,
    embeddings_output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    End-to-end family gate training pipeline (HB-WRH):
    1. Quarantines benchmark overlap to guarantee 0% leakage.
    2. Curates balanced class samples + 10% negative distractors.
    3. Extracts Z_train and Z_eval on model forward_features.
    4. Solves closed-form L2 Ridge with trace-scaled regularization across alpha sweep.
    5. Evaluates within-family accuracy & sweeps blending weights into champion plain head.
    6. Saves heads.safetensors and meta.json if output_dir provided.
    """
    from safetensors.torch import save_file
    from scipy.special import softmax as sp_softmax

    if device is None:
        device = next(model.parameters()).device

    target_family = target_family.lower()
    assert target_family in FAMILIES, f"Unknown family {target_family}. Must be one of {list(FAMILIES.keys())}"
    family_cols = FAMILIES[target_family]
    family_classes = [LABELS[i] for i in family_cols]
    num_family_classes = len(family_classes)
    has_negatives = bool(negative_ratio > 0.0 and df_pool is not None and len(df_pool) > 0)
    num_gate_classes = num_family_classes + (1 if has_negatives else 0)

    if samples_per_class <= 0:
        default_samples = {
            "glm": 4000,
            "gemini": 3800,
            "gpt": 2500,
            "claude": 2500,
        }
        samples_per_class = default_samples.get(target_family, 2500)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 16.0 and batch_size > 8:
            print(f"[{target_family.upper()} Gate] Low VRAM ({vram_gb:.1f} GiB) detected: clamping extraction batch_size from {batch_size} to 8.")
            batch_size = 8

    # 1. Zero leakage quarantining
    if eval_texts is not None and len(eval_texts) > 0:
        bench_hashes = {_text_hash(t) for t in eval_texts}
        init_pool_len = len(df_pool)
        pool_hashes = [_text_hash(t) for t in df_pool["text"].fillna("")]
        df_pool = df_pool[[h not in bench_hashes for h in pool_hashes]].copy()
        if df_train is not None and len(df_train) > 0:
            tr_hashes = [_text_hash(t) for t in df_train["text"].fillna("")]
            df_train = df_train[[h not in bench_hashes for h in tr_hashes]].copy()
        quarantined = init_pool_len - len(df_pool)
        print(f"[{target_family.upper()} Gate] Quarantined {quarantined} benchmark overlap rows. Clean pool: {len(df_pool):,}.")

    # 2. Balanced intra-family sampling
    fam_dfs = []
    for c in family_classes:
        sub_tr = df_train[df_train["model"] == c] if (df_train is not None and len(df_train) > 0) else pd.DataFrame()
        sub_pool = df_pool[df_pool["model"] == c]
        needed = samples_per_class
        n_take = max(min(needed - len(sub_tr), len(sub_pool)), 0)
        sampled_pool = sub_pool.sample(n=n_take, random_state=42) if n_take > 0 else pd.DataFrame()
        c_df = pd.concat([sub_tr, sampled_pool], ignore_index=True)
        fam_dfs.append(c_df)
        print(f"  {target_family.upper()} class {c}: {len(c_df)} samples ({len(sub_tr)} train, {len(c_df)-len(sub_tr)} pool)")

    df_fam_all = pd.concat(fam_dfs, ignore_index=True)
    total_fam_samples = len(df_fam_all)

    # Negative distractor sampling from non-family models (if negative_ratio > 0)
    if has_negatives:
        n_neg = int(total_fam_samples * negative_ratio)
        df_non_fam = df_pool[~df_pool["model"].isin(family_classes)].copy()
        other_fams = [f for f in ["gpt", "gemini", "claude", "glm", "qwen", "kimi"] if f != target_family]
        neg_dfs = []
        n_per_fam = max(n_neg // max(len(other_fams), 1), 1)
        for fam in other_fams:
            fam_sub = df_non_fam[df_non_fam["model"].str.startswith(fam)]
            if len(fam_sub) > 0:
                neg_dfs.append(fam_sub.sample(n=min(n_per_fam, len(fam_sub)), random_state=42))
        df_neg_all = pd.concat(neg_dfs, ignore_index=True) if neg_dfs else pd.DataFrame()
        print(f"[{target_family.upper()} Gate] Sampled {len(df_neg_all)} negative distractors across {len(other_fams)} non-{target_family.upper()} families.")
    else:
        df_neg_all = pd.DataFrame()

    train_texts = df_fam_all["text"].tolist() + (df_neg_all["text"].tolist() if len(df_neg_all) > 0 else [])
    fam_map = {name: i for i, name in enumerate(family_classes)}
    train_labels = [fam_map[m] for m in df_fam_all["model"].tolist()] + ([num_family_classes] * len(df_neg_all) if len(df_neg_all) > 0 else [])
    other_str = " + 1 Other" if len(df_neg_all) > 0 else ""
    print(f"[{target_family.upper()} Gate] Total training dataset: {len(train_texts):,} samples ({num_gate_classes} classes: {num_family_classes} {target_family.upper()}{other_str}).")

    amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
        else torch.float16
    )

    # 3. Extract Z_train representations
    print(f"[{target_family.upper()} Gate] Extracting Z_train on {len(train_texts)} samples (batch_size={batch_size})...")
    ds_tr = TextSequenceDataset(train_texts, train_labels, None, tok, max_len=max_len, end_id=end_id)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, collate_fn=collate_text_batch)

    Z_tr_list = []
    Z_tr_layers_list = []
    need_layers = bool(export_embeddings or save_all_layers)
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=amp_dtype):
            for batch in loader_tr:
                mb_ids = batch["input_ids"].to(device)
                mb_mask = batch["attention_mask"].to(device)
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                if need_layers:
                    z_b, z_l = model.forward_features(
                        mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True,
                        return_by_layer=True, extract_all_layers=save_all_layers,
                    )
                    Z_tr_layers_list.append(z_l.cpu().to(torch.float16).numpy())
                else:
                    z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                Z_tr_list.append(z_b.cpu().float().numpy())
    Z_train = np.vstack(Z_tr_list)
    Z_train_layers = np.vstack(Z_tr_layers_list) if Z_tr_layers_list else None
    y_train = np.array(train_labels, dtype=np.int64)
    print(f"[{target_family.upper()} Gate] Extracted Z_train: shape={Z_train.shape}, dtype={Z_train.dtype}" +
          (f", layers_shape={Z_train_layers.shape}" if Z_train_layers is not None else ""))

    # 4. Extract Z_eval on benchmark samples
    Z_eval = None
    Z_eval_layers = None
    main_logits_eval = None
    if eval_texts is not None and len(eval_texts) > 0:
        print(f"[{target_family.upper()} Gate] Extracting Z_eval on {len(eval_texts)} benchmark samples...")
        dummy_eval_y = [0] * len(eval_texts)
        ds_ev = TextSequenceDataset(eval_texts, dummy_eval_y, None, tok, max_len=max_len, end_id=end_id)
        loader_ev = DataLoader(ds_ev, batch_size=batch_size, shuffle=False, collate_fn=collate_text_batch)
        Z_ev_list = []
        Z_ev_layers_list = []
        with torch.no_grad():
            with torch.amp.autocast(device.type, dtype=amp_dtype):
                for batch in loader_ev:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    if need_layers:
                        z_b, z_l = model.forward_features(
                            mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True,
                            return_by_layer=True, extract_all_layers=save_all_layers,
                        )
                        Z_ev_layers_list.append(z_l.cpu().to(torch.float16).numpy())
                    else:
                        z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                    Z_ev_list.append(z_b.cpu().float().numpy())
        Z_eval = np.vstack(Z_ev_list)
        Z_eval_layers = np.vstack(Z_ev_layers_list) if Z_ev_layers_list else None

        if hasattr(model, "plain_head"):
            W_p = model.plain_head.weight.detach().cpu().float().numpy()
            b_p = model.plain_head.bias.detach().cpu().float().numpy()
            main_logits_eval = Z_eval @ W_p.T + b_p

    # Optional: Export Extracted Features Dataset for Kaggle / Local
    if export_embeddings:
        emb_dir = Path(embeddings_output_dir) if embeddings_output_dir else (Path(output_dir) / "embeddings" if output_dir else Path("export_embeddings"))
        save_extracted_features_bundle(
            output_dir=emb_dir,
            target=target_family,
            features_concat=Z_train,
            features_by_layer=Z_train_layers,
            texts=train_texts,
            labels=train_labels,
            model_names=df_fam_all["model"].tolist() + ["__OTHER__"] * len(df_neg_all),
            eval_features_concat=Z_eval,
            eval_features_by_layer=Z_eval_layers,
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            dataset_title=f"NTL {target_family.upper()} Extracted Features (Qwen3-14B)",
            dataset_slug=f"ntl-{target_family.lower()}-features-qwen3-14b",
            checkpoints_volume=checkpoints_volume,
        )

    # 5. Fit Head (Non-Linear MLP or Linear Ridge)
    y_ev_arr = np.array(eval_labels) if eval_labels is not None else None
    fam_eval_mask = np.isin(y_ev_arr, family_cols) if y_ev_arr is not None else None
    col_to_local = {c: i for i, c in enumerate(family_cols)}
    y_fam_true = np.array([col_to_local[v] for v in y_ev_arr[fam_eval_mask]]) if fam_eval_mask is not None else None

    acc_plain_within = 0.0
    if main_logits_eval is not None and fam_eval_mask is not None:
        plain_fam_pred = main_logits_eval[fam_eval_mask][:, family_cols].argmax(axis=1)
        acc_plain_within = float(balanced_accuracy_score(y_fam_true, plain_fam_pred))

    # 5. Fit Head (Non-Linear MLP or Linear Ridge)
    if head_type.lower() == "mlp":
        print(f"[{target_family.upper()} Gate] Fitting {num_gate_classes}-class Non-Linear MLP Head (dim={hidden_dim}, epochs={mlp_epochs}, lr={mlp_lr})...")
        mlp_head_dict, best_fam_acc, best_fam_logits = fit_family_mlp_head(
            Z_train=Z_train,
            y_train=y_train,
            num_gate_classes=num_gate_classes,
            num_family_classes=num_family_classes,
            hidden_dim=hidden_dim,
            epochs=mlp_epochs,
            lr=mlp_lr,
            dropout=mlp_dropout,
            label_smoothing=0.05,
            Z_eval=Z_eval,
            y_eval_true=y_fam_true,
            eval_mask=fam_eval_mask,
            device=device,
        )
        best_alpha = 0.0
        best_w = mlp_head_dict["plain.weight"].T.cpu().numpy()
        best_b = mlp_head_dict["plain.bias"].cpu().numpy()
        print(f"[{target_family.upper()} Gate] Non-Linear MLP Head trained with {target_family.upper()} Acc: {best_fam_acc*100:.2f}% (Lift: {(best_fam_acc - acc_plain_within)*100:+.2f}%)")
    else:
        print(f"[{target_family.upper()} Gate] Fitting {num_gate_classes}-class Ridge Head on Z_train...")
        best_w, best_b, best_alpha, best_fam_acc, best_fam_logits = fit_family_gate_head(
            Z_train=Z_train,
            y_train=y_train,
            num_gate_classes=num_gate_classes,
            alphas=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            label_smoothing=0.05,
            Z_eval=Z_eval,
            y_eval_true=y_fam_true,
            eval_mask=fam_eval_mask,
            num_family_classes=num_family_classes,
            device=device,
        )
        mlp_head_dict = {
            "plain.weight": torch.from_numpy(best_w.T).contiguous(),
            "plain.bias": torch.from_numpy(best_b).contiguous(),
        }
        print(f"[{target_family.upper()} Gate] Optimal Ridge Head selected: alpha={best_alpha:.2f} with {target_family.upper()} Acc: {best_fam_acc*100:.2f}% (Lift: {(best_fam_acc - acc_plain_within)*100:+.2f}%)")

    # 6. Evaluate Full Pipeline on all benchmark samples
    pipe_results = {}
    if main_logits_eval is not None and best_fam_logits is not None and y_ev_arr is not None:
        P_main = sp_softmax(main_logits_eval, axis=1)
        acc_plain_solo = float(balanced_accuracy_score(y_ev_arr, P_main.argmax(axis=1)))
        print(f"[{target_family.upper()} Gate] Champion Plain Head Benchmark BAcc: {acc_plain_solo*100:.2f}%")

        best_pipe_acc = 0.0
        best_pipe_w = 0.0
        cand_mask = np.isin(P_main.argmax(axis=1), family_cols) | (P_main[:, family_cols].sum(axis=1) >= 0.10)

        for gw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
            P_fused = P_main.copy()
            spec_probs = sp_softmax(best_fam_logits[:, :num_family_classes], axis=1)
            for idx in np.flatnonzero(cand_mask):
                orig_mass = P_main[idx, family_cols].sum()
                fused_fam = (1.0 - gw) * P_main[idx, family_cols] + gw * spec_probs[idx]
                fused_fam = fused_fam / fused_fam.sum() * orig_mass
                P_fused[idx, family_cols] = fused_fam

            acc_gated = float(balanced_accuracy_score(y_ev_arr, P_fused.argmax(axis=1)))
            P_sk = sinkhorn_balancing(P_fused, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
            acc_gated_sk = float(balanced_accuracy_score(y_ev_arr, P_sk.argmax(axis=1)))
            print(f"  [Gate Blend Sweep] gw={gw:.2f} | Gate Alone BAcc: {acc_gated*100:.2f}% | Gate + Sinkhorn BAcc: {acc_gated_sk*100:.2f}%")
            if acc_gated_sk > best_pipe_acc:
                best_pipe_acc = acc_gated_sk
                best_pipe_w = gw

        pipe_results = {
            "plain_solo_bacc": acc_plain_solo,
            "best_pipe_bacc": best_pipe_acc,
            "best_gate_weight": best_pipe_w,
            "lift_over_plain": best_pipe_acc - acc_plain_solo,
        }

    # 7. Save Checkpoint
    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_file(mlp_head_dict, str(out_p / "heads.safetensors"))
        meta_fam = {
            "classes": family_classes + ["__OTHER__"],
            "labels": family_classes + ["__OTHER__"],
            "target_family": target_family,
            "head_type": head_type.lower(),
            "hidden_dim": hidden_dim if head_type.lower() == "mlp" else None,
            "alpha": float(best_alpha),
            "within_family_acc": float(best_fam_acc),
            "plain_within_family_acc": float(acc_plain_within),
        }
        (out_p / "meta.json").write_text(json.dumps(meta_fam, indent=2), encoding="utf-8")
        if checkpoints_volume is not None and hasattr(checkpoints_volume, "commit"):
            checkpoints_volume.commit()
        print(f"[{target_family.upper()} Gate] Checkpoint saved to {out_p}.")

    return {
        "target_family": target_family,
        "within_family_balanced_acc": float(best_fam_acc),
        "plain_within_family_acc": float(acc_plain_within),
        "delta_within_family": float(best_fam_acc - acc_plain_within),
        "best_alpha": float(best_alpha),
        "pipe_results": pipe_results,
        "spec_logits": best_fam_logits[:, :num_family_classes].tolist() if best_fam_logits is not None else [],
        "spec_classes": family_classes,
    }
