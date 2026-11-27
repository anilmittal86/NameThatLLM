"""
Training Loss Functions: Family-Margin Cross-Entropy, Supervised Contrastive Loss, and Dual-Head Stage-2 Loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import FAMILIES, K


def _build_sister_mask(device: torch.device, boosted: bool = False) -> torch.Tensor:
    """
    Builds an (18, 18) indicator matrix M where M[i, j] > 0.0 if class i and
    class j belong to the same LLM provider family (sisters), and 0.0 otherwise.
    When boosted=True, applies empirically validated sibling and cross-family sink margin penalties.
    """
    M = torch.zeros((K, K), dtype=torch.float32, device=device)
    for _, members in FAMILIES.items():
        for m1 in members:
            for m2 in members:
                if m1 != m2:
                    M[m1, m2] = 1.0

    if boosted:
        boost_pairs = {
            (7, 8): 1.9, (8, 7): 1.9,
            (6, 7): 1.6, (7, 6): 1.6,
            (6, 8): 1.6, (8, 6): 1.6,
            (6, 5): 1.5, (5, 6): 1.5,
            (7, 5): 1.5, (5, 7): 1.5,
            (8, 5): 1.4, (5, 8): 1.4,
            (1, 2): 1.85, (2, 1): 1.85,
            (1, 3): 1.7, (3, 1): 1.7,
            (2, 3): 1.7, (3, 2): 1.7,
            (3, 0): 1.3, (0, 3): 1.2,
            (2, 0): 1.3, (0, 2): 1.2,
            (4, 2): 1.5, (2, 4): 1.4,
            (4, 1): 1.5, (1, 4): 1.4,
            (4, 3): 1.4, (3, 4): 1.4,
            (9, 10): 1.6, (10, 9): 1.6,
            (9, 12): 1.5, (12, 9): 1.5,
            (9, 13): 1.6, (13, 9): 1.6,
            (9, 11): 1.3, (11, 9): 1.3,
            (13, 10): 1.5, (10, 13): 1.5,
            (15, 16): 1.8, (16, 15): 1.8,
            (17, 16): 1.85, (16, 17): 1.75,
            (17, 15): 1.6, (15, 17): 1.5,
            (17, 13): 1.6, (13, 17): 1.6,
            (16, 13): 1.7, (13, 16): 1.7,
            (15, 13): 1.6, (13, 15): 1.6,
            (17, 9): 1.5, (9, 17): 1.4,
            (15, 9): 1.5, (9, 15): 1.4,
            (16, 9): 1.5, (9, 16): 1.4,
            (8, 9): 1.3, (9, 8): 1.2,
            (15, 12): 1.5, (12, 15): 1.4,
            (15, 11): 1.4, (11, 15): 1.3,
            (16, 10): 1.4, (10, 16): 1.3,
            (14, 0): 1.3, (0, 14): 1.2,
            (14, 1): 1.4, (1, 14): 1.3,
        }
        for (i, j), val in boost_pairs.items():
            M[i, j] = float(val)

    return M


def family_margin_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    family_margin: float = 0.35,
    label_smoothing: float = 0.0,
    weight: torch.Tensor | None = None,
    boost_hard_sisters: bool = True,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes Cross-Entropy with Additive Sibling Margin Penalty."""
    labels_dev = labels.to(logits.device)
    if family_margin > 0.0:
        num_cls = logits.shape[1]
        if num_cls == K:
            mask = _build_sister_mask(logits.device, boosted=boost_hard_sisters)
        else:
            mask = 1.0 - torch.eye(num_cls, dtype=torch.float32, device=logits.device)
            if num_cls == 4:
                mask[2, 3] = 1.8
                mask[3, 2] = 1.8
            elif num_cls == 5:
                for i in (1, 2, 3):
                    for j in (1, 2, 3):
                        if i != j:
                            mask[i, j] = 1.6
        boost = mask[labels_dev] * family_margin
        logits = logits + boost
    w_dev = weight.to(logits.device, dtype=torch.float32) if weight is not None else None
    if sample_weight is not None:
        per_sample_loss = F.cross_entropy(logits, labels_dev, weight=w_dev, label_smoothing=label_smoothing, reduction="none")
        sw = sample_weight.to(logits.device, dtype=torch.float32)
        eff_w = sw * w_dev[labels_dev] if w_dev is not None else sw
        return (per_sample_loss * sw).sum() / eff_w.sum().clamp(min=1e-6)
    return F.cross_entropy(logits, labels_dev, weight=w_dev, label_smoothing=label_smoothing)


def compute_supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10, **kwargs) -> torch.Tensor:
    """Supervised contrastive loss with hinge-repulsion fallback for unique-class prompt groups."""
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
            g_tensor = Z_group.detach().float().to(dev)
            group_slice = kwargs.get("group_slice")
            if group_slice is not None:
                s_idx, e_idx = int(group_slice[0]), int(group_slice[1])
                keep_rows = torch.ones(g_tensor.shape[0], dtype=torch.bool, device=dev)
                keep_rows[max(0, s_idx) : min(g_tensor.shape[0], e_idx)] = False
                g_tensor = g_tensor[keep_rows]
            if g_tensor.shape[0] > 0:
                g_norm = F.normalize(g_tensor, p=2, dim=-1, eps=1e-6)
                sim_group = z_norm @ g_norm.T
                if group_slice is None:
                    sim_group = sim_group.masked_fill(sim_group > 0.995, 0.0)
                loss = loss + 0.10 * torch.clamp(sim_group + 0.05, min=0.0).pow(2).mean()
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
    model: Any,
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

    loss_plain = family_margin_cross_entropy(
        logits_plain, labels_dev, family_margin=family_margin, label_smoothing=label_smoothing, weight=weight
    )

    if prompt_ids is None or centered_weight <= 0.0:
        return loss_plain, loss_plain, None

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

    loss_centered = family_margin_cross_entropy(
        logits_centered, labels_dev, family_margin=family_margin, label_smoothing=label_smoothing, weight=weight
    )

    total_loss = loss_plain + float(centered_weight) * loss_centered
    if supcon_weight > 0.0:
        loss_sup = compute_supcon_loss(Z_c, labels_dev, prototypes=model.centered_head.weight.detach())
        total_loss = total_loss + float(supcon_weight) * loss_sup

    return total_loss, loss_plain, loss_centered
