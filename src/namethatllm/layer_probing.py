"""
Layer-by-Layer Probing, Pairwise Sister Fisher Subspace, and Multi-View Feature Engineering
===========================================================================================
Provides:
1. build_supervised_family_subspace:
   Constructs a whitened linear projection matrix C_eff (subspace_dim, 40960) and mean vector
   mu_fam (1, 40960) combining:
     - Per-layer Pairwise Sister Fisher Discriminant directions: (Sigma_W + reg * I)^-1 (mu_a - mu_b)
     - Inter-layer Residual Velocity Fisher directions on Delta h^(l) = h^(l) - h^(l-1)
     - Per-layer multi-class Ridge probe weights
     - Orthogonalized intra-family residual PCA components
   Because C_eff is a linear operator on 40,960-dim Z_cat, it folds with 100% mathematical
   exactness into mlp_w1 / mlp_b1 and plain.weight / plain.bias for zero-cost inference.

2. run_all_layers_probe_sweep:
   Automated transformer-heads style probe across all L layers ([N, L, hidden_size]),
   reporting overall family Balanced Accuracy and hardest-sister-pair accuracy per layer,
   plus learned ScalarMix layer aggregation weights.

3. engineer_multiview_features:
   Extracts rich tabular/neural features from exported Kaggle Dataset bundles
   (head-tail cosine drift, inter-layer velocity norms, token dispersion ratios, and OOF log-odds).

4. LayerScalarMix & LayerScalarMixWithHead:
   Differentiable convex combination of layer representations with learnable softmax weights.

5. fit_trace_scaled_ridge_probe:
   Closed-form trace-scaled Ridge regression in primal (N >= D) or dual (N < D) regimes.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


# ============================================================================
# Layer ScalarMix Modules & Ridge Probing
# ============================================================================

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W_np, b_np


# ============================================================================
# Supervised Within-Class Fisher Discriminant & Sister Decoupling Subspace
# ============================================================================

def _solve_woodbury_direction(
    R_sub: torch.Tensor,
    Delta_sub: torch.Tensor,
    reg_scale: float,
) -> torch.Tensor:
    N_s, D_s = R_sub.shape
    trace_val = float(torch.sum(R_sub * R_sub).item() / max(N_s, 1))
    lam = max(reg_scale * (trace_val / max(D_s, 1)), 1e-4)
    if N_s >= D_s:
        Cov_W = (R_sub.T @ R_sub) / float(max(N_s, 1))
        Cov_W.diagonal().add_(lam)
        W_sol = torch.linalg.solve(Cov_W, Delta_sub)
    else:
        K_gram = R_sub @ R_sub.T
        K_gram.diagonal().add_(float(max(N_s, 1)) * lam)
        R_Delta = R_sub @ Delta_sub
        Dual_sol = torch.linalg.solve(K_gram, R_Delta)
        W_sol = (Delta_sub - R_sub.T @ Dual_sol) / lam
    return torch.nn.functional.normalize(W_sol, p=2, dim=0, eps=1e-6)


def _solve_within_class_fisher_gpu(
    X_fam: np.ndarray,
    y_fam: np.ndarray,
    num_family_classes: int,
    reg_ratio: float = 0.15,
    reg_scales: tuple[float, ...] = (0.03, 0.12, 0.45),
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    100% Pure Supervised Multi-Scale Fisher Solver (Zero PCA):
    1. Solves multi-class OVR Fisher directions on pooled intra-family residuals.
    2. Solves pairwise sister Fisher directions on pair-specific residuals R_ab
       across multiple spectral regularization scales (reg_scales).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N, D = X_fam.shape
    X_t = torch.from_numpy(X_fam).to(device, dtype=torch.float32)
    mu_all = X_t.mean(dim=0, keepdim=True)

    centroids = []
    R_t = X_t.clone()
    for c in range(num_family_classes):
        mask = (y_fam == c)
        if mask.sum() > 0:
            idx_t = torch.from_numpy(np.flatnonzero(mask)).to(device)
            mu_c = X_t[idx_t].mean(dim=0, keepdim=True)
            centroids.append(mu_c.squeeze(0))
            R_t[idx_t] -= mu_c
        else:
            centroids.append(mu_all.squeeze(0))

    # 1. Multi-class OVR Fisher directions
    rhs_list = [centroids[c] - mu_all.squeeze(0) for c in range(num_family_classes)]
    Delta_ovr = torch.stack(rhs_list, dim=1) if rhs_list else torch.zeros((D, 1), device=device, dtype=torch.float32)
    W_ovr_t = _solve_woodbury_direction(R_t, Delta_ovr, reg_scale=reg_ratio)
    W_ovr = W_ovr_t.cpu().numpy().T  # (K, D)

    # 2. Pairwise Sister Fisher directions on pair-specific covariance across reg_scales
    pair_dirs = []
    for a, b in combinations(range(num_family_classes), 2):
        delta_ab = (centroids[a] - centroids[b]).unsqueeze(1)  # (D, 1)
        mask_ab = (y_fam == a) | (y_fam == b)
        if mask_ab.sum() >= 4:
            idx_ab = torch.from_numpy(np.flatnonzero(mask_ab)).to(device)
            R_ab = R_t[idx_ab]
        else:
            R_ab = R_t
        for r_scale in reg_scales:
            w_ab = _solve_woodbury_direction(R_ab, delta_ab, reg_scale=r_scale)
            pair_dirs.append(w_ab.squeeze(1).cpu().numpy())

    W_pairs = np.vstack(pair_dirs).astype(np.float32) if pair_dirs else np.zeros((0, D), dtype=np.float32)

    del X_t, R_t, Delta_ovr, W_ovr_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return W_ovr, W_pairs


def _compute_layer_oof_scores(
    Z_fam_c: np.ndarray,
    y_fam: np.ndarray,
    num_tapped_layers: int,
    num_family_classes: int,
    hard_sister_pair: tuple[int, int] | None = None,
    prompt_ids: list[Any] | np.ndarray | None = None,
    alpha: float = 5.0,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 5-fold OOF Ridge probe scores s_l = 0.5 * family_oof_bacc + 0.5 * hard_pair_oof_acc
    for each tapped layer block X_l in Z_fam_c.
    Returns (layer_weights, oof_baccs, composite_scores).
    """
    N, D_total = Z_fam_c.shape
    D_layer = D_total // max(num_tapped_layers, 1)

    min_class_count = min(int((y_fam == c).sum()) for c in range(num_family_classes)) if num_family_classes > 0 else 0
    if N < 5 or min_class_count < 2:
        uniform_w = np.full(num_tapped_layers, 1.0 / max(num_tapped_layers, 1), dtype=np.float32)
        ones_bacc = np.ones(num_tapped_layers, dtype=np.float32)
        return uniform_w, ones_bacc, ones_bacc

    n_splits_eff = max(min(n_splits, min_class_count), 2)
    p_arr = np.asarray(prompt_ids) if prompt_ids is not None else None
    use_groups = False
    if p_arr is not None and len(p_arr) == N:
        u_p = np.unique(p_arr)
        if len(u_p) >= n_splits_eff and len(u_p) < 0.8 * N:
            use_groups = True

    if use_groups:
        splitter = GroupKFold(n_splits=n_splits_eff).split(Z_fam_c[:, :D_layer], y_fam, groups=p_arr)
    else:
        splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=42).split(Z_fam_c[:, :D_layer], y_fam)
    folds = list(splitter)

    Y_oh = np.eye(num_family_classes, dtype=np.float32)[y_fam]
    Y_oh = 0.95 * Y_oh + (0.05 / float(max(num_family_classes, 1)))

    scores = []
    bacc_list = []
    for l_idx in range(num_tapped_layers):
        s_col = l_idx * D_layer
        e_col = (l_idx + 1) * D_layer
        X_l = Z_fam_c[:, s_col:e_col]

        oof_l = np.zeros((N, num_family_classes), dtype=np.float32)
        for tr_idx, val_idx in folds:
            X_tr, Y_tr = X_l[tr_idx], Y_oh[tr_idx]
            X_val = X_l[val_idx]

            mu_tr = X_tr.mean(axis=0, keepdims=True)
            X_tr_c = X_tr - mu_tr
            X_val_c = X_val - mu_tr

            N_tr, D_l = X_tr_c.shape
            scale = float(np.mean(np.sum(X_tr_c * X_tr_c, axis=1)))
            eff_alpha = alpha * max(scale / 512.0, 1e-3)

            if N_tr >= D_l:
                Cov = X_tr_c.T @ X_tr_c
                Cov.flat[:: D_l + 1] += eff_alpha
                W_l = np.linalg.solve(Cov, X_tr_c.T @ (Y_tr - Y_tr.mean(axis=0, keepdims=True)))
                b_l = Y_tr.mean(axis=0)
            else:
                K_mat = X_tr_c @ X_tr_c.T
                K_mat.flat[:: N_tr + 1] += eff_alpha
                A = np.linalg.solve(K_mat, Y_tr - Y_tr.mean(axis=0, keepdims=True))
                W_l = X_tr_c.T @ A
                b_l = Y_tr.mean(axis=0)

            oof_l[val_idx] = X_val_c @ W_l + b_l

        pred_labels = oof_l.argmax(axis=1)
        fam_bacc = float(balanced_accuracy_score(y_fam, pred_labels))
        bacc_list.append(fam_bacc)

        if hard_sister_pair is not None:
            s_a, s_b = hard_sister_pair
            pair_mask = (y_fam == s_a) | (y_fam == s_b)
            if pair_mask.sum() > 0 and len(np.unique(y_fam[pair_mask])) == 2:
                pair_preds = np.where(oof_l[pair_mask, s_a] >= oof_l[pair_mask, s_b], s_a, s_b)
                pair_acc = float(balanced_accuracy_score(y_fam[pair_mask], pair_preds))
            else:
                pair_acc = fam_bacc
        else:
            pair_acc = fam_bacc

        s_l = 0.5 * fam_bacc + 0.5 * pair_acc
        scores.append(s_l)

    scores_arr = np.array(scores, dtype=np.float32)
    shifted = (scores_arr - scores_arr.max()) * 12.0
    exp_s = np.exp(shifted)
    w_layers = exp_s / np.sum(exp_s)
    return w_layers, np.array(bacc_list, dtype=np.float32), scores_arr


def build_supervised_family_subspace(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    num_family_classes: int,
    num_tapped_layers: int = 8,
    subspace_dim: int | None = None,
    reg_ratio: float = 0.15,
    device: torch.device | None = None,
    include_velocity: bool = True,
    layer_weighting: bool = True,
    hard_sister_pair: tuple[int, int] | None = None,
    prompt_ids: list[Any] | np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Constructs a 100% Pure Supervised Fisher Subspace (Zero Unsupervised PCA):
      1. Part 1: Per-Layer Multi-Scale Pairwise Sister + OVR Fisher directions across all tapped layers,
         optionally weighted by layer importance w_l = softmax(12 * s_l).
      2. Part 2: Inter-Layer Residual Velocity Fisher directions on Delta X^(l) = X^(l) - X^(l-1),
         optionally enabled via include_velocity.
      3. Part 3: Economic QR orthonormalization (filtering collinear directions via diag_r > 1e-3 * max).
      4. Part 4: Unit-variance coordinate whitening by intra-family training std.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N_total, D_total = Z_train.shape
    D_layer = D_total // max(num_tapped_layers, 1)
    assert D_layer * num_tapped_layers == D_total, (
        f"D_total ({D_total}) must be divisible by num_tapped_layers ({num_tapped_layers})"
    )

    fam_mask = (y_train < num_family_classes)
    if fam_mask.sum() == 0:
        fam_mask = np.ones(len(y_train), dtype=bool)

    Z_fam = Z_train[fam_mask]
    y_fam = y_train[fam_mask]

    mu_fam = Z_fam.mean(axis=0, keepdims=True).astype(np.float32)
    Z_fam_c = (Z_fam - mu_fam).astype(np.float32)

    # Resolve default hard_sister_pair if None
    if hard_sister_pair is None:
        if num_family_classes == 4:
            hard_sister_pair = (2, 3)
        elif num_family_classes == 5:
            hard_sister_pair = (1, 2)
        elif num_family_classes == 2:
            hard_sister_pair = (0, 1)

    # Compute Layer-Probing Importance Weights if layer_weighting is enabled
    if layer_weighting:
        w_layers, oof_baccs, comp_scores = _compute_layer_oof_scores(
            Z_fam_c=Z_fam_c,
            y_fam=y_fam,
            num_tapped_layers=num_tapped_layers,
            num_family_classes=num_family_classes,
            hard_sister_pair=hard_sister_pair,
            prompt_ids=prompt_ids,
        )
    else:
        w_layers = np.full(num_tapped_layers, 1.0 / max(num_tapped_layers, 1), dtype=np.float32)
        oof_baccs = np.zeros(num_tapped_layers, dtype=np.float32)
        comp_scores = np.zeros(num_tapped_layers, dtype=np.float32)

    basis_rows: list[np.ndarray] = []
    layer_fisher_norms: dict[int, float] = {}

    # Part 1: Per-Layer Multi-Scale Fisher Directions
    for l_idx in range(num_tapped_layers):
        s_col = l_idx * D_layer
        e_col = (l_idx + 1) * D_layer
        X_l = Z_fam_c[:, s_col:e_col]

        alpha_l = float(np.sqrt(float(num_tapped_layers) * float(w_layers[l_idx])))

        W_ovr, W_pairs = _solve_within_class_fisher_gpu(
            X_l, y_fam, num_family_classes=num_family_classes, reg_ratio=reg_ratio,
            reg_scales=(0.03, 0.12, 0.45), device=device
        )
        stacked_dirs = np.vstack([W_ovr, W_pairs]) if len(W_pairs) > 0 else W_ovr
        for row in stacked_dirs:
            full_vec = np.zeros(D_total, dtype=np.float32)
            full_vec[s_col:e_col] = row * alpha_l
            basis_rows.append(full_vec)

        layer_fisher_norms[l_idx] = float(np.linalg.norm(W_pairs)) if len(W_pairs) > 0 else 0.0

    # Part 2: Inter-Layer Residual Velocity Fisher Directions: Delta X^(l) = X^(l) - X^(l-1)
    if include_velocity:
        for l_idx in range(1, num_tapped_layers):
            s_prev, e_prev = (l_idx - 1) * D_layer, l_idx * D_layer
            s_curr, e_curr = l_idx * D_layer, (l_idx + 1) * D_layer
            Delta_X = Z_fam_c[:, s_curr:e_curr] - Z_fam_c[:, s_prev:e_prev]

            w_avg = 0.5 * (float(w_layers[l_idx]) + float(w_layers[l_idx - 1]))
            alpha_vel = float(np.sqrt(float(num_tapped_layers) * w_avg))

            _, W_vel_pairs = _solve_within_class_fisher_gpu(
                Delta_X, y_fam, num_family_classes=num_family_classes, reg_ratio=reg_ratio * 1.5,
                reg_scales=(0.05, 0.20, 0.60), device=device
            )
            for row in W_vel_pairs:
                full_vec = np.zeros(D_total, dtype=np.float32)
                full_vec[s_curr:e_curr] = (row / np.sqrt(2.0)) * alpha_vel
                full_vec[s_prev:e_prev] = (-row / np.sqrt(2.0)) * alpha_vel
                basis_rows.append(full_vec)

    W_sup = np.vstack(basis_rows)  # (D_sup, 40960)

    # Part 3: Orthonormalize pure Fisher directions via reduced QR (rejecting float32 noise)
    Q_sup, R_sup = np.linalg.qr(W_sup.T, mode="reduced")
    diag_r = np.abs(np.diag(R_sup))
    r_thr = max(float(diag_r.max()) * 1e-3, 1e-4) if len(diag_r) > 0 else 1e-4
    valid_sup = diag_r > r_thr
    V_basis = Q_sup[:, valid_sup].T.astype(np.float32)  # (D_fisher, 40960)
    if subspace_dim is not None and subspace_dim > 0 and V_basis.shape[0] > subspace_dim:
        V_basis = V_basis[:subspace_dim]

    # Part 4: Whiten each Fisher coordinate by its intra-family training std
    Z_fam_proj = Z_fam_c @ V_basis.T  # (N_fam, D_fisher)
    sd_proj = Z_fam_proj.std(axis=0, keepdims=True).clip(min=1e-3).astype(np.float32)
    C_eff = (V_basis / sd_proj.T).astype(np.float32)

    diagnostics = {
        "n_supervised_dirs": int(V_basis.shape[0]),
        "n_residual_pca_dirs": 0,
        "total_subspace_dim": int(V_basis.shape[0]),
        "subspace_dim": int(V_basis.shape[0]),
        "layer_fisher_norms": layer_fisher_norms,
        "layer_scalar_weights": [float(w) for w in w_layers],
        "per_layer_oof_bacc": [float(b) for b in oof_baccs],
        "layer_composite_scores": [float(s) for s in comp_scores],
        "include_velocity": include_velocity,
        "layer_weighting": layer_weighting,
    }
    return C_eff, mu_fam, diagnostics


# ============================================================================
# Automated Layer-by-Layer Probing Sweep
# ============================================================================

def run_all_layers_probe_sweep(
    features_by_layer: np.ndarray,
    y: np.ndarray | None = None,
    num_family_classes: int = 4,
    hard_sister_pair: tuple[int, int] = (2, 3),
    prompts: np.ndarray | None = None,
    alpha: float = 5.0,
    n_splits: int = 5,
    output_dir: Path | str | None = None,
    labels: np.ndarray | list[int] | None = None,
    prompt_ids: list[Any] | None = None,
    output_json: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Automated layer-by-layer linear probing sweep across all L layers ([N, L, hidden_size]).
    Evaluates:
      1. Per-layer 5-fold OOF Balanced Accuracy across all family classes.
      2. Per-layer Binary Pairwise Accuracy on the hardest sister pair (default: indices (2, 3)
         corresponding to gemini-3.7-flash vs gemini-3.8-flash).
      3. Learned ScalarMix / Precision-Weighted Layer Ensemble accuracy.
    """
    if y is None and labels is not None:
        y = np.asarray(labels)
    if y is None:
        raise ValueError("Either `y` or `labels` must be provided.")

    if prompts is None and prompt_ids is not None:
        prompts = np.asarray(prompt_ids)

    if output_dir is None and output_json is not None:
        output_dir = Path(output_json).parent

    fam_mask = (y < num_family_classes)
    if fam_mask.sum() == 0:
        fam_mask = np.ones(len(y), dtype=bool)

    X_layers = features_by_layer[fam_mask].astype(np.float32)
    y_fam = y[fam_mask]
    N, L, D = X_layers.shape

    if prompts is not None:
        p_fam = prompts[fam_mask]
        u_p = np.unique(p_fam)
        use_groups = len(u_p) >= n_splits and len(u_p) < 0.8 * N
    else:
        p_fam = None
        use_groups = False

    n_splits_eff = max(min(n_splits, N // max(num_family_classes, 1)), 2)
    if use_groups:
        splitter = GroupKFold(n_splits=n_splits_eff).split(X_layers[:, 0, :], y_fam, groups=p_fam)
    else:
        splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=42).split(X_layers[:, 0, :], y_fam)
    folds = list(splitter)

    Y_oh = np.eye(num_family_classes, dtype=np.float32)[y_fam]
    Y_oh = 0.95 * Y_oh + 0.05 / max(num_family_classes, 1)

    layer_oof_logits = np.zeros((L, N, num_family_classes), dtype=np.float32)
    layer_metrics = []

    s_a, s_b = hard_sister_pair
    pair_mask = (y_fam == s_a) | (y_fam == s_b)

    print("\n================================================================================")
    print(f" AUTOMATED LAYER-BY-LAYER PROBING SWEEP ({L} Layers x {D} Dim | N={N})")
    print("================================================================================")
    print(f" Layer | Family OOF BAcc | Hard Sister Pair ({s_a} vs {s_b}) Acc | Signal Weight")
    print("--------------------------------------------------------------------------------")

    for l_idx in range(L):
        X_l = X_layers[:, l_idx, :]
        oof_l = np.zeros((N, num_family_classes), dtype=np.float32)

        for tr_idx, val_idx in folds:
            X_tr, Y_tr = X_l[tr_idx], Y_oh[tr_idx]
            X_val = X_l[val_idx]

            mu_tr = X_tr.mean(axis=0, keepdims=True)
            X_tr_c = X_tr - mu_tr
            X_val_c = X_val - mu_tr

            scale = float(np.mean(np.sum(X_tr_c * X_tr_c, axis=1)))
            eff_alpha = alpha * max(scale / 512.0, 1e-3)

            K_mat = X_tr_c @ X_tr_c.T
            K_mat.flat[:: len(tr_idx) + 1] += eff_alpha
            A = np.linalg.solve(K_mat, Y_tr - Y_tr.mean(axis=0, keepdims=True))
            W_l = X_tr_c.T @ A
            b_l = Y_tr.mean(axis=0)

            oof_l[val_idx] = X_val_c @ W_l + b_l

        layer_oof_logits[l_idx] = oof_l
        fam_bacc = float(balanced_accuracy_score(y_fam, oof_l.argmax(axis=1)))

        if pair_mask.sum() > 0:
            pair_preds = np.where(oof_l[pair_mask, s_a] >= oof_l[pair_mask, s_b], s_a, s_b)
            pair_acc = float(balanced_accuracy_score(y_fam[pair_mask], pair_preds))
        else:
            pair_acc = fam_bacc

        layer_metrics.append({
            "layer": int(l_idx),
            "layer_idx": int(l_idx),
            "family_oof_bacc": fam_bacc,
            "hard_pair_oof_acc": pair_acc,
            "oof_balanced_acc": fam_bacc,
        })

    # Compute precision-weighted ScalarMix weights across layers based on log-odds margin
    raw_scores = np.array([0.5 * m["family_oof_bacc"] + 0.5 * m["hard_pair_oof_acc"] for m in layer_metrics])
    centered_scores = (raw_scores - raw_scores.max()) * 15.0
    scalar_mix_weights = np.exp(centered_scores) / np.sum(np.exp(centered_scores))

    for l_idx, m in enumerate(layer_metrics):
        m["scalar_mix_weight"] = float(scalar_mix_weights[l_idx])
        print(
            f" L{l_idx:02d}   | {m['family_oof_bacc'] * 100:6.2f}%         | "
            f"{m['hard_pair_oof_acc'] * 100:6.2f}%                    | {scalar_mix_weights[l_idx]:6.4f}"
        )

    # Evaluate ScalarMix Ensemble across all layers
    mix_oof = np.tensordot(scalar_mix_weights.astype(np.float32), layer_oof_logits, axes=(0, 0))
    mix_bacc = float(balanced_accuracy_score(y_fam, mix_oof.argmax(axis=1)))
    if pair_mask.sum() > 0:
        mix_pair_preds = np.where(mix_oof[pair_mask, s_a] >= mix_oof[pair_mask, s_b], s_a, s_b)
        mix_pair_acc = float(balanced_accuracy_score(y_fam[pair_mask], mix_pair_preds))
    else:
        mix_pair_acc = mix_bacc

    ranked_layers = sorted(
        layer_metrics,
        key=lambda d: 0.5 * d["family_oof_bacc"] + 0.5 * d["hard_pair_oof_acc"],
        reverse=True,
    )
    top8_layers = sorted([d["layer"] for d in ranked_layers[:8]])

    print("--------------------------------------------------------------------------------")
    print(f" ScalarMix All-{L}-Layer OOF BAcc: {mix_bacc * 100:.2f}% | Hard Pair ({s_a} vs {s_b}): {mix_pair_acc * 100:.2f}%")
    print(f" Empirically Optimal Top-8 Layers: {top8_layers}")
    print("================================================================================\n")

    report = {
        "num_layers": int(L),
        "num_samples": int(N),
        "scalar_mix_oof_bacc": mix_bacc,
        "scalar_mix_hard_pair_acc": mix_pair_acc,
        "recommended_top8_layers": top8_layers,
        "layers": layer_metrics,
        "ranked_layers": ranked_layers,
        "top_4_layers": [d["layer"] for d in ranked_layers[:4]],
        "top_8_layers": top8_layers,
        "best_single_layer": ranked_layers[0],
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "layer_probe_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if output_json is not None:
        out_f = Path(output_json)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


# ============================================================================
# Offline Multi-View Feature Engineering
# ============================================================================

def engineer_multiview_features(
    features_concat: np.ndarray,
    features_by_layer: np.ndarray | None = None,
    head_tail_cos: np.ndarray | None = None,
    token_std_norms: np.ndarray | None = None,
    num_tapped_layers: int = 8,
) -> np.ndarray:
    """
    Engineers explicit structural & trajectory features for offline Kaggle Dataset modeling:
    1. Per-layer L2 activation norms across tapped layers [N, 8].
    2. Inter-layer cosine similarity & L2 velocity ||h^(l) - h^(l-1)||_2 across consecutive layers [N, 14].
    3. Head-to-Tail positional cosine drift [N, 8] (opening framing vs closing summary divergence).
    4. Token-level activation dispersion norms & ratios [N, 16] (distilled Flash vs full Pro burstiness).
    """
    N, D_total = features_concat.shape
    D_layer = D_total // max(num_tapped_layers, 1)
    feats: list[np.ndarray] = []

    # Reshape tapped layers: (N, 8, D_layer)
    blocks = features_concat.reshape(N, num_tapped_layers, D_layer).astype(np.float32)
    block_norms = np.linalg.norm(blocks, axis=2).clip(min=1e-6)  # (N, 8)
    feats.append(block_norms)

    # Inter-layer cosine similarity and L2 transition velocity between consecutive tapped layers
    blocks_unit = blocks / block_norms[:, :, None]
    inter_cos = np.sum(blocks_unit[:, 1:, :] * blocks_unit[:, :-1, :], axis=2)  # (N, 7)
    inter_vel = np.linalg.norm(blocks[:, 1:, :] - blocks[:, :-1, :], axis=2)    # (N, 7)
    feats.extend([inter_cos, inter_vel])

    # If full all-28-layer tensor is provided, add trajectory curvature across early/mid/late thirds
    if features_by_layer is not None and features_by_layer.ndim == 3:
        f_all = features_by_layer.astype(np.float32)
        f_norms = np.linalg.norm(f_all, axis=2).clip(min=1e-6)
        feats.append(f_norms)
        f_unit = f_all / f_norms[:, :, None]
        adj_cos = np.sum(f_unit[:, 1:, :] * f_unit[:, :-1, :], axis=2)  # (N, L_all - 1)
        feats.append(adj_cos)

    if head_tail_cos is not None:
        feats.append(head_tail_cos.astype(np.float32))

    if token_std_norms is not None:
        std_f = token_std_norms.astype(np.float32)
        feats.append(std_f)
        # Coefficient of variation (dispersion relative to layer norm)
        feats.append(std_f / block_norms)

    return np.hstack(feats).astype(np.float32)
