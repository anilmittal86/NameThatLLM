"""
Layer-Probing, Activation Analysis, and ScalarMix Aggregation Module.

Enables fast empirical sweeps across intermediate transformer hidden states to:
1. Identify optimal layers for fine-grained LLM sister separation.
2. Quantify per-layer signal-to-noise ratio and out-of-fold balanced accuracy.
3. Fit learnable convex layer aggregation (ScalarMix) and stacked meta-probes.
4. Export structured layer-probing diagnostic reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from namethatllm.config import FAMILIES, LABELS


class LayerScalarMix(nn.Module):
    """
    Differentiable convex combination of layer-normalized hidden states.

    Formula:
        h_mix = gamma * sum_{l=0}^{L-1} softmax(w)_l * LayerNorm_l(h_l)
    """

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, elementwise_affine=False) for _ in range(num_layers)
        ])
        self.scalar_weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, layer_pooled_states: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        layer_pooled_states : torch.Tensor of shape [B, num_layers, hidden_dim]
                              or list of [B, hidden_dim] tensors.

        Returns
        -------
        torch.Tensor of shape [B, hidden_dim]
        """
        if isinstance(layer_pooled_states, torch.Tensor):
            normed = [
                self.layer_norms[l](layer_pooled_states[:, l, :])
                for l in range(layer_pooled_states.shape[1])
            ]
            stacked = torch.stack(normed, dim=1)
        else:
            normed = [self.layer_norms[l](h) for l, h in enumerate(layer_pooled_states)]
            stacked = torch.stack(normed, dim=1)

        probs = F.softmax(self.scalar_weights, dim=0).view(1, -1, 1)
        mixed = self.gamma * torch.sum(stacked * probs, dim=1)
        return mixed

    def get_layer_importance(self) -> dict[int, float]:
        probs = F.softmax(self.scalar_weights.detach(), dim=0).cpu().tolist()
        return {idx: round(p, 4) for idx, p in enumerate(probs)}


class LayerScalarMixWithHead(nn.Module):
    """
    ScalarMix layer aggregation combined with an end-to-end linear classification probe.
    """

    def __init__(self, num_layers: int, hidden_dim: int, num_classes: int, dropout: float = 0.10):
        super().__init__()
        self.mix = LayerScalarMix(num_layers, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, layer_pooled_states: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        mixed = self.mix(layer_pooled_states)
        return self.classifier(self.dropout(mixed))

    def get_layer_importance(self) -> dict[int, float]:
        return self.mix.get_layer_importance()


def fit_trace_scaled_ridge_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    label_smoothing: float = 0.05,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits closed-form trace-scaled Ridge regression in PyTorch.
    Returns (W, b) where W is [D, num_classes] and b is [num_classes].
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N, D = X_train.shape
    Y_onehot = np.zeros((N, num_classes), dtype=np.float32)
    for i, lab in enumerate(y_train):
        if 0 <= lab < num_classes:
            Y_onehot[i, lab] = 1.0
    Y_smooth = (1.0 - label_smoothing) * Y_onehot + (label_smoothing / float(num_classes))

    X_t = torch.from_numpy(X_train).to(device, dtype=torch.float32)
    Y_t = torch.from_numpy(Y_smooth).to(device, dtype=torch.float32)

    mu_X = X_t.mean(dim=0, keepdim=True)
    mu_Y = Y_t.mean(dim=0, keepdim=True)
    X_c = X_t - mu_X
    Y_c = Y_t - mu_Y

    scale = float(torch.sum(X_c ** 2).item() / max(N, 1))
    eff_alpha = max(alpha * scale, 1e-6)

    use_primal = (N >= D)
    if use_primal:
        Cov = X_c.T @ X_c
        reg = eff_alpha * torch.eye(D, device=device, dtype=torch.float32)
        W = torch.linalg.solve(Cov + reg, X_c.T @ Y_c)
    else:
        K_mat = X_c @ X_c.T
        reg = eff_alpha * torch.eye(N, device=device, dtype=torch.float32)
        A = torch.linalg.solve(K_mat + reg, Y_c)
        W = X_c.T @ A

    b = (mu_Y - mu_X @ W).squeeze(0)
    W_np = W.cpu().numpy().astype(np.float32)
    b_np = b.cpu().numpy().astype(np.float32)

    del X_t, Y_t, X_c, Y_c, W, b
    return W_np, b_np


def run_all_layers_probe_sweep(
    features_by_layer: np.ndarray,
    labels: np.ndarray | list[int],
    prompt_ids: list[Any] | None = None,
    family_cols: list[int] | None = None,
    target_family: str | None = None,
    alphas: list[float] | None = None,
    n_splits: int = 5,
    output_json: Path | str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Evaluates individual layer probe performance across all L hidden states.

    Parameters
    ----------
    features_by_layer : np.ndarray of shape [N, L, hidden_size]
    labels : labels array of shape [N]
    prompt_ids : optional prompt identifiers for GroupKFold validation
    family_cols : subset of global class indices to isolate (intra-family evaluation)
    target_family : name of target family ('gemini', 'gpt', 'claude', 'glm', etc.)
    alphas : candidate L2 regularization strengths
    n_splits : cross-validation folds (default 5)
    output_json : optional file path to save report JSON
    device : torch.device

    Returns
    -------
    dict containing layer ranking, per-layer balanced accuracy, and optimal layer subset
    """
    labels_arr = np.asarray(labels)
    N, L, D = features_by_layer.shape

    if alphas is None:
        alphas = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    # Filter to intra-family rows if specified
    if family_cols is not None:
        fam_mask = np.isin(labels_arr, family_cols)
        if fam_mask.sum() > 0:
            X_data = features_by_layer[fam_mask]
            y_raw = labels_arr[fam_mask]
            pids = [prompt_ids[i] for i in np.flatnonzero(fam_mask)] if prompt_ids is not None else None
            col_map = {c: i for i, c in enumerate(family_cols)}
            y_data = np.array([col_map[v] for v in y_raw], dtype=np.int64)
            num_classes = len(family_cols)
        else:
            X_data = features_by_layer
            y_data = labels_arr
            pids = prompt_ids
            num_classes = len(np.unique(labels_arr))
    else:
        X_data = features_by_layer
        y_data = labels_arr
        pids = prompt_ids
        num_classes = len(np.unique(labels_arr))

    N_sub = len(y_data)
    unique_groups = len(set(pids)) if pids is not None else 0
    if pids is not None and unique_groups >= n_splits:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X_data, y_data, groups=pids))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X_data, y_data))

    print("\n" + "=" * 88)
    fam_str = f" for '{target_family.upper()}'" if target_family else ""
    print(f" [Layer Probing Sweep] Evaluating {L} Hidden Layers on {N_sub:,} Samples ({num_classes} Classes){fam_str}")
    print("=" * 88)

    layer_results: list[dict[str, Any]] = []
    layer_oof_probs: list[np.ndarray] = []

    for l_idx in range(L):
        X_l = X_data[:, l_idx, :].astype(np.float32)

        # Tune alpha using 5-fold OOF CV
        best_alpha = alphas[0]
        best_oof_bacc = -1.0
        best_oof_logits: np.ndarray | None = None

        for alpha in alphas:
            oof_logits = np.zeros((N_sub, num_classes), dtype=np.float32)
            for tr_idx, val_idx in splits:
                W, b = fit_trace_scaled_ridge_probe(
                    X_l[tr_idx], y_data[tr_idx], num_classes=num_classes, alpha=alpha, device=device
                )
                oof_logits[val_idx] = X_l[val_idx] @ W + b

            preds = oof_logits.argmax(axis=1)
            bacc = float(balanced_accuracy_score(y_data, preds))
            if bacc > best_oof_bacc:
                best_oof_bacc = bacc
                best_alpha = alpha
                best_oof_logits = oof_logits

        # Compute softmax probabilities from best OOF logits
        exps = np.exp(best_oof_logits - best_oof_logits.max(axis=1, keepdims=True))
        probs_l = exps / exps.sum(axis=1, keepdims=True)
        layer_oof_probs.append(probs_l)

        random_baseline = 1.0 / max(num_classes, 1)
        lift = best_oof_bacc - random_baseline
        layer_results.append({
            "layer_idx": l_idx,
            "oof_balanced_acc": round(best_oof_bacc, 4),
            "lift_over_random": round(lift, 4),
            "best_alpha": best_alpha,
        })
        print(f"  Layer {l_idx:02d} | 5-Fold OOF BAcc: {best_oof_bacc*100:6.2f}% (Lift: {lift*100:+5.2f}%) | Alpha: {best_alpha}")

    # Rank layers by OOF accuracy
    ranked_layers = sorted(layer_results, key=lambda x: x["oof_balanced_acc"], reverse=True)
    best_single = ranked_layers[0]

    # Multi-Layer Stacked Meta-Ridge:
    # Concatenate top-K layer OOF log-odds and evaluate stacked combination
    top_4_indices = [r["layer_idx"] for r in ranked_layers[:min(4, L)]]
    top_8_indices = [r["layer_idx"] for r in ranked_layers[:min(8, L)]]

    stacked_features_4 = np.hstack([np.log(np.clip(layer_oof_probs[i], 1e-6, 1.0)) for i in top_4_indices])
    stacked_features_8 = np.hstack([np.log(np.clip(layer_oof_probs[i], 1e-6, 1.0)) for i in top_8_indices])

    meta_oof_4 = np.zeros((N_sub, num_classes), dtype=np.float32)
    meta_oof_8 = np.zeros((N_sub, num_classes), dtype=np.float32)

    for tr_idx, val_idx in splits:
        W4, b4 = fit_trace_scaled_ridge_probe(stacked_features_4[tr_idx], y_data[tr_idx], num_classes, alpha=1.0, device=device)
        meta_oof_4[val_idx] = stacked_features_4[val_idx] @ W4 + b4

        W8, b8 = fit_trace_scaled_ridge_probe(stacked_features_8[tr_idx], y_data[tr_idx], num_classes, alpha=1.0, device=device)
        meta_oof_8[val_idx] = stacked_features_8[val_idx] @ W8 + b8

    meta_acc_top4 = float(balanced_accuracy_score(y_data, meta_oof_4.argmax(axis=1)))
    meta_acc_top8 = float(balanced_accuracy_score(y_data, meta_oof_8.argmax(axis=1)))

    print("-" * 88)
    print(f"  Best Single Layer   : Layer {best_single['layer_idx']:02d} -> {best_single['oof_balanced_acc']*100:.2f}% BAcc")
    print(f"  Stacked Top-4 Layers: {top_4_indices} -> {meta_acc_top4*100:.2f}% BAcc (Lift: {(meta_acc_top4 - best_single['oof_balanced_acc'])*100:+.2f}%)")
    print(f"  Stacked Top-8 Layers: {top_8_indices} -> {meta_acc_top8*100:.2f}% BAcc (Lift: {(meta_acc_top8 - best_single['oof_balanced_acc'])*100:+.2f}%)")
    print("=" * 88 + "\n")

    report = {
        "target_family": target_family,
        "num_samples": N_sub,
        "num_classes": num_classes,
        "num_layers_evaluated": L,
        "best_single_layer": best_single,
        "top_4_layers": top_4_indices,
        "top_8_layers": top_8_indices,
        "meta_acc_top4": round(meta_acc_top4, 4),
        "meta_acc_top8": round(meta_acc_top8, 4),
        "ranked_layers": ranked_layers,
    }

    if output_json is not None:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[Layer Probing Sweep] Saved full report to {out_p}")

    return report
