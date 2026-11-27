"""
Phase 3 Intra-Family Sister-Decoupling Gate Heads, Supervised Subspace Training, and Offline Sweeps.
"""

from __future__ import annotations

import copy
import functools
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scipy.special import softmax as sp_softmax
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..ablation import _text_hash
from ..config import FAMILIES, LABELS
from ..experiment_logger import ExperimentLedger
from ..layer_probing import (
    build_oof_supervised_family_subspace,
    build_supervised_family_subspace,
    engineer_multiview_features,
    run_all_layers_probe_sweep,
)
from ..postprocess import sinkhorn_balancing
from .backbones.base import resolve_backbone_adapter
from .data import TextSequenceDataset, collate_text_batch
from .embeddings import save_extracted_features_bundle


def fit_family_gate_head(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    num_gate_classes: int,
    alphas: list[float] | None = None,
    label_smoothing: float = 0.10,
    Z_eval: np.ndarray | None = None,
    y_eval_true: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    num_family_classes: int | None = None,
    device: torch.device | None = None,
    prompt_ids: list[Any] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray | None, float]:
    """
    Fits an L2-regularized Plain Ridge Head for intra-family gating using 5-fold CV on Z_train.
    Returns: (best_W, best_b, best_alpha, eval_or_oof_acc, eval_logits, best_oof_bacc)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if alphas is None:
        alphas = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N_tr, _ = Z_train.shape
    K_eval_classes = num_family_classes if num_family_classes is not None else num_gate_classes
    fam_mask_all = y_train < K_eval_classes

    def _solve_ridge_split(X_sub: np.ndarray, y_sub: np.ndarray, alpha_val: float) -> tuple[np.ndarray, np.ndarray]:
        N_s, D_s = X_sub.shape
        Y_oh = np.zeros((N_s, num_gate_classes), dtype=np.float32)
        for i_s, lab_s in enumerate(y_sub):
            if 0 <= lab_s < num_gate_classes:
                Y_oh[i_s, lab_s] = 1.0
        Y_sm = (1.0 - label_smoothing) * Y_oh + (label_smoothing / float(num_gate_classes))

        X_t_sub = torch.from_numpy(X_sub).to(device, dtype=torch.float32)
        Y_t_sub = torch.from_numpy(Y_sm).to(device, dtype=torch.float32)

        sub_fam_m = torch.from_numpy(y_sub < K_eval_classes).to(device)
        mu_X_s = X_t_sub[sub_fam_m].mean(dim=0, keepdim=True) if sub_fam_m.sum() > 0 else X_t_sub.mean(dim=0, keepdim=True)
        mu_Y_s = Y_t_sub.mean(dim=0, keepdim=True)

        X_c_s = X_t_sub - mu_X_s
        Y_c_s = Y_t_sub - mu_Y_s

        scale_s = float(torch.sum(X_c_s ** 2).item() / max(N_s, 1))
        eff_a = max(alpha_val * scale_s, 1e-6)

        if N_s >= D_s:
            Cov_s = X_c_s.T @ X_c_s
            reg_s = eff_a * torch.eye(D_s, device=device, dtype=torch.float32)
            W_s = torch.linalg.solve(Cov_s + reg_s, X_c_s.T @ Y_c_s)
        else:
            K_s = X_c_s @ X_c_s.T
            reg_s = eff_a * torch.eye(N_s, device=device, dtype=torch.float32)
            A_s = torch.linalg.solve(K_s + reg_s, Y_c_s)
            W_s = X_c_s.T @ A_s

        b_s = (mu_Y_s - mu_X_s @ W_s).squeeze(0)
        W_np_s = W_s.cpu().numpy()
        b_np_s = b_s.cpu().numpy()

        fam_logits_s = (X_sub[y_sub < K_eval_classes] @ W_np_s + b_np_s)[:, :K_eval_classes]
        row_std_s = float(np.std(fam_logits_s, axis=1).mean()) if len(fam_logits_s) > 0 else 1.0
        temp_scale_s = (3.2 / row_std_s) if row_std_s > 1e-4 else 1.0
        W_np_s = (W_np_s * temp_scale_s).astype(np.float32)
        b_np_s = (b_np_s * temp_scale_s).astype(np.float32)

        del X_t_sub, Y_t_sub, X_c_s, Y_c_s, W_s, b_s
        return W_np_s, b_np_s

    n_splits = 5
    pids_arr = np.asarray(prompt_ids) if prompt_ids is not None else None
    unique_groups = len(set(prompt_ids)) if prompt_ids is not None else 0
    max_group_frac = 1.0
    if pids_arr is not None and len(pids_arr) > 0:
        _, counts = np.unique(pids_arr, return_counts=True)
        max_group_frac = float(counts.max() / len(pids_arr))

    if pids_arr is not None and unique_groups >= 10 and max_group_frac < 0.05:
        splitter = list(GroupKFold(n_splits=n_splits).split(Z_train, y_train, groups=pids_arr))
    else:
        min_cls = min(int(np.bincount(y_train).min()) if len(y_train) > 0 else 0, n_splits)
        if min_cls >= 2:
            splitter = list(StratifiedKFold(n_splits=min_cls, shuffle=True, random_state=42).split(Z_train, y_train))
        else:
            fold_sizes = np.full(min(n_splits, N_tr), N_tr // min(n_splits, N_tr), dtype=int)
            fold_sizes[: N_tr % min(n_splits, N_tr)] += 1
            indices = np.arange(N_tr)
            splitter = []
            cur = 0
            for fs in fold_sizes:
                val_idx = indices[cur : cur + fs]
                tr_idx = np.concatenate([indices[:cur], indices[cur + fs :]])
                splitter.append((tr_idx, val_idx))
                cur += fs

    best_alpha = alphas[0]
    best_oof_bacc = -1.0
    for alpha in alphas:
        oof_logits = np.zeros((N_tr, num_gate_classes), dtype=np.float32)
        for tr_idx, val_idx in splitter:
            W_fold, b_fold = _solve_ridge_split(Z_train[tr_idx], y_train[tr_idx], alpha)
            oof_logits[val_idx] = Z_train[val_idx] @ W_fold + b_fold

        if fam_mask_all.sum() > 0:
            oof_preds = oof_logits[fam_mask_all, :K_eval_classes].argmax(axis=1)
            oof_acc = float(balanced_accuracy_score(y_train[fam_mask_all], oof_preds))
        else:
            oof_acc = float(balanced_accuracy_score(y_train, oof_logits.argmax(axis=1)))

        if oof_acc > best_oof_bacc:
            best_oof_bacc = oof_acc
            best_alpha = alpha

    best_w, best_b = _solve_ridge_split(Z_train, y_train, best_alpha)

    best_eval_logits = None
    if Z_eval is not None and y_eval_true is not None and eval_mask is not None and eval_mask.sum() > 0:
        best_eval_logits = Z_eval @ best_w + best_b
        fam_pred = best_eval_logits[eval_mask, :K_eval_classes].argmax(axis=1)
        best_acc = float(balanced_accuracy_score(y_eval_true, fam_pred))
    else:
        best_acc = best_oof_bacc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_w, best_b, best_alpha, best_acc, best_eval_logits, best_oof_bacc


class FamilyMLPHead(nn.Module):
    """2-Layer Non-Linear MLP Head for intra-family sister decoupling."""

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
    hidden_dim: int = 64,
    epochs: int = 12,
    lr: float = 2e-4,
    dropout: float = 0.35,
    weight_decay: float = 0.10,
    batch_size: int = 64,
    label_smoothing: float = 0.10,
    Z_eval: np.ndarray | None = None,
    y_eval_true: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    device: torch.device | None = None,
    pca_components: np.ndarray | None = None,
    pca_mean: np.ndarray | None = None,
    raw_Z_eval: np.ndarray | None = None,
) -> tuple[dict[str, torch.Tensor], float, np.ndarray | None, float]:
    """Trains a regularized 2-layer MLP head on intra-family features and folds PCA + centering into Layer 1."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N_tr, D_feat = Z_train.shape

    if pca_components is not None:
        Z_tr_std = Z_train.astype(np.float32)
        mu_family = None
        sd_family = None
    else:
        fam_mask_tr = y_train < num_family_classes
        if fam_mask_tr.sum() > 1:
            mu_family = Z_train[fam_mask_tr].mean(axis=0, keepdims=True).astype(np.float32)
            sd_family = Z_train[fam_mask_tr].std(axis=0, keepdims=True).clip(min=1e-3).astype(np.float32)
        else:
            mu_family = Z_train.mean(axis=0, keepdims=True).astype(np.float32)
            sd_family = Z_train.std(axis=0, keepdims=True).clip(min=1e-3).astype(np.float32)
        Z_tr_std = ((Z_train - mu_family) / sd_family).astype(np.float32)

    min_c = min(int((y_train == c).sum()) for c in range(num_gate_classes)) if num_gate_classes > 0 else 5
    n_splits_eff = max(min(5, min_c), 2)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=42)
    fit_idx, val_idx = next(skf.split(Z_tr_std, y_train))

    X_fit_t = torch.from_numpy(Z_tr_std[fit_idx]).float()
    y_fit_t = torch.from_numpy(y_train[fit_idx]).long()
    X_val_t = torch.from_numpy(Z_tr_std[val_idx]).float().to(device)
    y_val_np = y_train[val_idx]
    val_fam_mask = y_val_np < num_family_classes

    ds = TensorDataset(X_fit_t, y_fit_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    in_drop_rate = 0.15 if D_feat <= 256 else 0.45
    mlp = FamilyMLPHead(
        in_features=D_feat,
        hidden_dim=hidden_dim,
        num_classes=num_gate_classes,
        dropout=dropout,
        in_dropout=in_drop_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    sib_mask = torch.zeros((num_gate_classes, num_gate_classes), dtype=torch.float32, device=device)
    for i in range(num_family_classes):
        for j in range(num_family_classes):
            if i != j:
                sib_mask[i, j] = 1.0
    if num_family_classes == 4:
        sib_mask[2, 3] = 1.6
        sib_mask[3, 2] = 1.6
    elif num_family_classes == 5:
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i != j:
                    sib_mask[i, j] = 1.4

    best_val_acc = -1.0
    best_state = None
    for _ in range(epochs):
        mlp.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            bx = bx + 0.15 * torch.randn_like(bx)
            optimizer.zero_grad()
            logits = mlp(bx)
            margin_boost = sib_mask[by] * 0.35
            loss = F.cross_entropy(logits + margin_boost, by, label_smoothing=label_smoothing)
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

    with torch.no_grad():
        fam_mask = y_train < num_family_classes
        if fam_mask.sum() > 0:
            X_fam_t = torch.from_numpy(Z_tr_std[fam_mask]).float().to(device)
            fam_logits_tr = mlp.forward_eval(X_fam_t).cpu().numpy()[:, :num_family_classes]
            row_std = float(np.std(fam_logits_tr, axis=1).mean())
            temp_scale = (3.2 / row_std) if row_std > 1e-4 else 1.0
            mlp.fc2.weight.data.mul_(temp_scale)
            mlp.fc2.bias.data.mul_(temp_scale)
            del X_fam_t

    if pca_components is not None and pca_mean is not None:
        W1_64 = mlp.fc1.weight.detach().cpu().numpy().astype(np.float64)
        b1_64 = mlp.fc1.bias.detach().cpu().numpy().astype(np.float64)
        comp_64 = pca_components.astype(np.float64)
        mu_64 = pca_mean.reshape(1, -1).astype(np.float64)

        W1_folded_64 = W1_64 @ comp_64
        b1_folded_64 = b1_64 - (mu_64 @ W1_folded_64.T).squeeze(0)

        W1_folded = torch.from_numpy(W1_folded_64.astype(np.float32)).contiguous()
        b1_folded = torch.from_numpy(b1_folded_64.astype(np.float32)).contiguous()
    else:
        with torch.no_grad():
            W1 = mlp.fc1.weight.detach().float()
            b1 = mlp.fc1.bias.detach().float()
            mu_t = torch.from_numpy(mu_family).to(device=W1.device, dtype=torch.float32)
            sd_t = torch.from_numpy(sd_family).to(device=W1.device, dtype=torch.float32)
            W1_folded = (W1 / sd_t).contiguous()
            b1_folded = (b1 - ((mu_t / sd_t) @ W1.T).squeeze(0)).contiguous()

    best_eval_acc = best_val_acc
    best_eval_logits = None
    eval_target = raw_Z_eval if (raw_Z_eval is not None and pca_components is not None) else Z_eval
    if eval_target is not None:
        with torch.no_grad():
            Z_ev_raw_t = torch.from_numpy(eval_target).float().to(device)
            h = F.gelu(Z_ev_raw_t @ W1_folded.to(device).T + b1_folded.to(device))
            h = mlp.ln(h)
            best_eval_logits = mlp.fc2(h).cpu().numpy()
            del Z_ev_raw_t, h
        if y_eval_true is not None and eval_mask is not None and eval_mask.sum() > 0:
            fam_pred = best_eval_logits[eval_mask, :num_family_classes].argmax(axis=1)
            best_eval_acc = float(balanced_accuracy_score(y_eval_true, fam_pred))

    W_ridge, b_ridge, *_ = fit_family_gate_head(
        Z_train=Z_train,
        y_train=y_train,
        num_gate_classes=num_gate_classes,
        Z_eval=Z_eval,
        y_eval_true=y_eval_true,
        eval_mask=eval_mask,
        num_family_classes=num_family_classes,
        device=device,
    )

    if pca_components is not None and pca_mean is not None:
        comp_64 = pca_components.astype(np.float64)
        mu_64 = pca_mean.reshape(1, -1).astype(np.float64)
        W_r_64 = W_ridge.astype(np.float64)
        b_r_64 = b_ridge.astype(np.float64)
        W_r_eff_64 = comp_64.T @ W_r_64
        b_r_eff_64 = b_r_64 - (mu_64 @ W_r_eff_64).squeeze(0)
        plain_w = torch.from_numpy(W_r_eff_64.T.astype(np.float32)).contiguous()
        plain_b = torch.from_numpy(b_r_eff_64.astype(np.float32)).contiguous()
    else:
        plain_w = torch.from_numpy(W_ridge.T).to(torch.float32).contiguous()
        plain_b = torch.from_numpy(b_ridge).to(torch.float32).contiguous()

    head_dict = {
        "mlp_w1": W1_folded.cpu().to(torch.float32).contiguous(),
        "mlp_b1": b1_folded.cpu().to(torch.float32).contiguous(),
        "unfolded_w1": mlp.fc1.weight.detach().cpu().to(torch.float32).contiguous(),
        "unfolded_b1": mlp.fc1.bias.detach().cpu().to(torch.float32).contiguous(),
        "ln_weight": mlp.ln.weight.detach().cpu().to(torch.float32).contiguous(),
        "ln_bias": mlp.ln.bias.detach().cpu().to(torch.float32).contiguous(),
        "mlp_w2": mlp.fc2.weight.detach().cpu().to(torch.float32).contiguous(),
        "mlp_b2": mlp.fc2.bias.detach().cpu().to(torch.float32).contiguous(),
        "plain.weight": plain_w,
        "plain.bias": plain_b,
    }

    del X_fit_t, y_fit_t, X_val_t, ds, mlp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return head_dict, best_eval_acc, best_eval_logits, best_val_acc


def train_family_gate_from_dataset(
    dataset_dir: Path | str,
    target_family: str = "gemini",
    output_dir: Path | str | None = None,
    head_type: str = "mlp",
    use_supervised_subspace: bool = True,
    subspace_dim: int | None = None,
    pca_dim: int | None = None,
    hidden_dim: int = 64,
    mlp_epochs: int = 12,
    mlp_lr: float = 2e-4,
    mlp_dropout: float = 0.35,
    device: torch.device | None = None,
    run_probing_sweep: bool = False,
    checkpoints_volume: Any | None = None,
    gate_weight: float = 0.40,
    select_layers: list[int] | str | None = None,
    include_velocity: bool = True,
    layer_weighting: bool = True,
    use_multiview: bool = False,
    hard_sister_pair: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Offline entrypoint to train and evaluate any family gate directly from pre-extracted features."""
    if subspace_dim is None and pca_dim is not None:
        subspace_dim = pca_dim

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds_p = Path(dataset_dir)
    assert (ds_p / "features_concat.safetensors").exists(), f"features_concat.safetensors not found in {ds_p}"
    assert (ds_p / "metadata.parquet").exists(), f"metadata.parquet not found in {ds_p}"

    target_family = target_family.lower().strip()
    assert target_family in FAMILIES, f"Unknown family {target_family}. Must be one of {list(FAMILIES.keys())}"
    family_cols = FAMILIES[target_family]
    family_classes = [LABELS[i] for i in family_cols]
    num_family_classes = len(family_classes)

    print("\n" + "=" * 88)
    print(f" [Offline Dataset Gate Training] Loading pre-extracted features from {ds_p} for '{target_family.upper()}'...")
    print("=" * 88)

    feat_sd = load_file(str(ds_p / "features_concat.safetensors"))
    Z_train = feat_sd["features_concat"].cpu().float().numpy()

    Z_train_layers = None
    if (ds_p / "features_by_layer.safetensors").exists():
        feat_l_sd = load_file(str(ds_p / "features_by_layer.safetensors"))
        Z_train_layers = feat_l_sd["features_by_layer"].cpu().numpy()

    df_meta = pd.read_parquet(ds_p / "metadata.parquet")
    model_names = df_meta["model"].tolist()
    prompt_ids = df_meta["prompt_id"].tolist() if "prompt_id" in df_meta.columns else None

    has_negatives = any(m == "__OTHER__" for m in model_names)
    num_gate_classes = num_family_classes + (1 if has_negatives else 0)

    fam_map = {name: i for i, name in enumerate(family_classes)}
    y_train = np.array([fam_map.get(m, num_family_classes if has_negatives else 0) for m in model_names], dtype=np.int64)

    eff_hard_pair = hard_sister_pair
    if eff_hard_pair is None:
        if num_family_classes == 4:
            eff_hard_pair = (2, 3)
        elif num_family_classes == 5:
            eff_hard_pair = (1, 2)
        elif num_family_classes == 2:
            eff_hard_pair = (0, 1)

    if run_probing_sweep and Z_train_layers is not None:
        probe_out_dir = Path(output_dir) if output_dir else ds_p
        run_all_layers_probe_sweep(
            features_by_layer=Z_train_layers,
            y=y_train,
            num_family_classes=num_family_classes,
            hard_sister_pair=eff_hard_pair or (2, 3),
            prompts=np.asarray(prompt_ids) if prompt_ids is not None else None,
            output_dir=probe_out_dir,
        )

    chosen_layers = None
    if select_layers is not None and Z_train_layers is not None:
        if select_layers == "auto_top8":
            probe_out_dir = Path(output_dir) if output_dir else ds_p
            probe_rep = run_all_layers_probe_sweep(
                features_by_layer=Z_train_layers,
                y=y_train,
                num_family_classes=num_family_classes,
                hard_sister_pair=eff_hard_pair or (2, 3),
                prompts=np.asarray(prompt_ids) if prompt_ids is not None else None,
                output_dir=probe_out_dir,
            )
            chosen_layers = probe_rep["top_8_layers"]
        else:
            chosen_layers = list(select_layers)

        if chosen_layers is not None and len(chosen_layers) > 0:
            if max(chosen_layers) >= Z_train_layers.shape[1]:
                print(
                    f"[{target_family.upper()} Gate] Warning: select_layers {chosen_layers} exceeds available layers ({Z_train_layers.shape[1]}). Using all available layers."
                )
                chosen_layers = None
            else:
                print(f"[{target_family.upper()} Gate] Slicing select_layers={chosen_layers} from features_by_layer...")
                Z_train_sel = Z_train_layers[:, chosen_layers, :].astype(np.float32)
                m_l = Z_train_sel.mean(axis=-1, keepdims=True)
                s_l = Z_train_sel.std(axis=-1, keepdims=True) + 1e-5
                Z_train_sel = (Z_train_sel - m_l) / s_l
                Z_train = Z_train_sel.reshape(Z_train_sel.shape[0], -1).astype(np.float32)

    Z_eval, y_ev_arr, main_logits_eval = None, None, None
    Z_ev_layers = None
    if (ds_p / "benchmark_features_concat.safetensors").exists() and (ds_p / "benchmark_metadata.parquet").exists():
        bench_sd = load_file(str(ds_p / "benchmark_features_concat.safetensors"))
        Z_eval = bench_sd["benchmark_features_concat"].cpu().float().numpy()
        df_bench = pd.read_parquet(ds_p / "benchmark_metadata.parquet")
        y_ev_arr = df_bench["label"].to_numpy()

    if (ds_p / "benchmark_features_by_layer.safetensors").exists():
        bench_l_sd = load_file(str(ds_p / "benchmark_features_by_layer.safetensors"))
        Z_ev_layers = bench_l_sd["benchmark_features_by_layer"].cpu().numpy()

    if chosen_layers is not None and Z_ev_layers is not None and max(chosen_layers) < Z_ev_layers.shape[1]:
        Z_ev_sel = Z_ev_layers[:, chosen_layers, :].astype(np.float32)
        m_ev = Z_ev_sel.mean(axis=-1, keepdims=True)
        s_ev = Z_ev_sel.std(axis=-1, keepdims=True) + 1e-5
        Z_ev_sel = (Z_ev_sel - m_ev) / s_ev
        Z_eval = Z_ev_sel.reshape(Z_ev_sel.shape[0], -1).astype(np.float32)

    if (ds_p / "benchmark_champion_logits.safetensors").exists():
        champ_ev_sd = load_file(str(ds_p / "benchmark_champion_logits.safetensors"))
        if "logits_plain" in champ_ev_sd:
            main_logits_eval = champ_ev_sd["logits_plain"].cpu().float().numpy()
        elif "champion_logits" in champ_ev_sd:
            main_logits_eval = champ_ev_sd["champion_logits"].cpu().float().numpy()
        elif len(champ_ev_sd) > 0 and any("logit" in k.lower() for k in champ_ev_sd):
            for k in champ_ev_sd:
                if "logit" in k.lower():
                    main_logits_eval = champ_ev_sd[k].cpu().float().numpy()
                    break
    elif (ds_p / "champion_logits.safetensors").exists() and Z_eval is not None:
        champ_sd = load_file(str(ds_p / "champion_logits.safetensors"))
        if "logits_plain" in champ_sd and len(champ_sd["logits_plain"]) == len(Z_eval):
            main_logits_eval = champ_sd["logits_plain"].cpu().float().numpy()
        elif "champion_logits" in champ_sd and len(champ_sd["champion_logits"]) == len(Z_eval):
            main_logits_eval = champ_sd["champion_logits"].cpu().float().numpy()

    num_tapped = len(chosen_layers) if chosen_layers is not None else 8
    pca_components = None
    pca_mean = None
    if use_supervised_subspace:
        Z_tr_active, Z_ev_active, pca_components, pca_mean, _ = build_oof_supervised_family_subspace(
            Z_train=Z_train,
            y_train=y_train,
            Z_eval=Z_eval,
            num_family_classes=num_family_classes,
            subspace_dim=subspace_dim,
            num_tapped_layers=num_tapped,
            include_velocity=include_velocity,
            layer_weighting=layer_weighting,
            hard_sister_pair=eff_hard_pair,
            prompt_ids=prompt_ids,
            device=device,
        )
    else:
        Z_tr_active = Z_train
        Z_ev_active = Z_eval

    if use_multiview:
        ht_tr, std_tr = None, None
        if (ds_p / "champion_logits.safetensors").exists():
            c_sd = load_file(str(ds_p / "champion_logits.safetensors"))
            if "head_tail_cos" in c_sd:
                ht_tr = c_sd["head_tail_cos"].cpu().float().numpy()
            if "token_std_norms" in c_sd:
                std_tr = c_sd["token_std_norms"].cpu().float().numpy()

        f_by_layer_tr = Z_train_layers if (Z_eval is None or Z_ev_layers is not None) else None
        mv_tr = engineer_multiview_features(
            features_concat=Z_train,
            features_by_layer=f_by_layer_tr,
            head_tail_cos=ht_tr,
            token_std_norms=std_tr,
            num_tapped_layers=num_tapped,
        )
        mv_mean = mv_tr.mean(axis=0, keepdims=True)
        mv_std = mv_tr.std(axis=0, keepdims=True) + 1e-5
        mv_tr_norm = (mv_tr - mv_mean) / mv_std
        Z_tr_active = np.hstack([Z_tr_active, mv_tr_norm]).astype(np.float32)

        if Z_eval is not None:
            ht_ev, std_ev = None, None
            if (ds_p / "benchmark_champion_logits.safetensors").exists():
                b_sd = load_file(str(ds_p / "benchmark_champion_logits.safetensors"))
                if "head_tail_cos" in b_sd:
                    ht_ev = b_sd["head_tail_cos"].cpu().float().numpy()
                if "token_std_norms" in b_sd:
                    std_ev = b_sd["token_std_norms"].cpu().float().numpy()
            f_by_layer_ev = Z_ev_layers if (Z_train_layers is not None) else None
            mv_ev = engineer_multiview_features(
                features_concat=Z_eval,
                features_by_layer=f_by_layer_ev,
                head_tail_cos=ht_ev,
                token_std_norms=std_ev,
                num_tapped_layers=num_tapped,
            )
            mv_ev_norm = (mv_ev - mv_mean) / mv_std
            Z_ev_active = np.hstack([Z_ev_active, mv_ev_norm]).astype(np.float32)

        pca_components = None
        pca_mean = None

    fam_eval_mask = np.isin(y_ev_arr, family_cols) if y_ev_arr is not None else None
    col_to_local = {c: i for i, c in enumerate(family_cols)}
    y_fam_true = np.array([col_to_local[v] for v in y_ev_arr[fam_eval_mask]]) if fam_eval_mask is not None else None

    acc_plain_within = 0.0
    if main_logits_eval is not None and fam_eval_mask is not None and fam_eval_mask.sum() > 0:
        if main_logits_eval.shape[1] == len(LABELS):
            plain_fam_pred = main_logits_eval[fam_eval_mask][:, family_cols].argmax(axis=1)
        else:
            plain_fam_pred = main_logits_eval[fam_eval_mask][:, :num_family_classes].argmax(axis=1)
        acc_plain_within = float(balanced_accuracy_score(y_fam_true, plain_fam_pred))

    best_w_pca, best_b_pca, ridge_alpha, ridge_acc, ridge_logits_ev, ridge_oof_bacc = fit_family_gate_head(
        Z_train=Z_tr_active,
        y_train=y_train,
        num_gate_classes=num_gate_classes,
        alphas=[1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0],
        label_smoothing=0.10,
        Z_eval=Z_ev_active,
        y_eval_true=y_fam_true,
        eval_mask=fam_eval_mask,
        num_family_classes=num_family_classes,
        device=device,
        prompt_ids=prompt_ids,
    )

    if pca_components is not None:
        comp_64 = pca_components.astype(np.float64)
        mu_64 = pca_mean.reshape(1, -1).astype(np.float64)
        W_r_64 = best_w_pca.astype(np.float64)
        b_r_64 = best_b_pca.astype(np.float64)
        W_r_eff_64 = comp_64.T @ W_r_64
        b_r_eff_64 = b_r_64 - (mu_64 @ W_r_eff_64).squeeze(0)
        best_w = W_r_eff_64.astype(np.float32)
        best_b = b_r_eff_64.astype(np.float32)
    else:
        best_w = best_w_pca
        best_b = best_b_pca

    plain_w_tensor = torch.from_numpy(best_w.T).to(torch.float32).contiguous()
    plain_b_tensor = torch.from_numpy(best_b).to(torch.float32).contiguous()

    mlp_head_dict, mlp_acc, mlp_logits_ev, mlp_val_acc = fit_family_mlp_head(
        Z_train=Z_tr_active,
        y_train=y_train,
        num_gate_classes=num_gate_classes,
        num_family_classes=num_family_classes,
        hidden_dim=hidden_dim,
        epochs=mlp_epochs,
        lr=mlp_lr,
        dropout=mlp_dropout,
        label_smoothing=0.10,
        Z_eval=Z_ev_active,
        y_eval_true=y_fam_true,
        eval_mask=fam_eval_mask,
        device=device,
        pca_components=pca_components,
        pca_mean=pca_mean,
        raw_Z_eval=Z_eval if pca_components is not None else None,
    )

    hybrid_logits_ev = None
    if mlp_logits_ev is not None and ridge_logits_ev is not None:
        raw_hyb = 0.65 * mlp_logits_ev + 0.35 * ridge_logits_ev
        hyb_fam = raw_hyb[:, :num_family_classes]
        hyb_std = float(np.std(hyb_fam, axis=1).mean()) if len(hyb_fam) > 0 else 1.0
        hyb_scale = (3.2 / hyb_std) if hyb_std > 1e-4 else 1.0
        hybrid_logits_ev = (raw_hyb * hyb_scale).astype(np.float32)

    default_gw = float(gate_weight)

    def _eval_head_metrics(h_logits: np.ndarray | None):
        if h_logits is None:
            return 0.0, 0.0, {}
        ev_acc = 0.0
        if fam_eval_mask is not None and fam_eval_mask.sum() > 0 and y_fam_true is not None:
            fam_pred = h_logits[fam_eval_mask, :num_family_classes].argmax(axis=1)
            ev_acc = float(balanced_accuracy_score(y_fam_true, fam_pred))

        hard_acc = ev_acc
        if y_fam_true is not None and eff_hard_pair is not None and fam_eval_mask is not None and fam_eval_mask.sum() > 0:
            s_a, s_b = eff_hard_pair
            pair_mask_ev = (y_fam_true == s_a) | (y_fam_true == s_b)
            if pair_mask_ev.sum() > 0 and len(np.unique(y_fam_true[pair_mask_ev])) == 2:
                fam_l = h_logits[fam_eval_mask, :num_family_classes]
                spec_pair_preds = np.where(fam_l[pair_mask_ev, s_a] >= fam_l[pair_mask_ev, s_b], s_a, s_b)
                hard_acc = float(balanced_accuracy_score(y_fam_true[pair_mask_ev], spec_pair_preds))

        p_res = {}
        if main_logits_eval is not None and y_ev_arr is not None and main_logits_eval.shape[1] == len(LABELS):
            P_main = sp_softmax(main_logits_eval, axis=1)
            acc_plain_solo = float(balanced_accuracy_score(y_ev_arr, P_main.argmax(axis=1)))
            cand_mask = np.isin(P_main.argmax(axis=1), family_cols) | (P_main[:, family_cols].sum(axis=1) >= 0.10)
            spec_probs = sp_softmax(h_logits[:, :num_family_classes], axis=1)

            best_pipe_acc = 0.0
            best_pipe_w = default_gw
            best_raw_bacc = 0.0
            best_sk_bacc = 0.0
            default_raw_bacc = 0.0
            default_sk_bacc = 0.0
            sweep_weights = sorted(set([0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, default_gw]))

            for gw in sweep_weights:
                P_fused = P_main.copy()
                for idx in np.flatnonzero(cand_mask):
                    orig_mass = float(P_main[idx, family_cols].sum())
                    if orig_mass > 1e-6:
                        gw_eff = gw * min(orig_mass / 0.50, 1.0)
                        main_norm = P_main[idx, family_cols] / orig_mass
                        fused_fam = (1.0 - gw_eff) * main_norm + gw_eff * spec_probs[idx]
                        P_fused[idx, family_cols] = fused_fam * orig_mass

                acc_gated = float(balanced_accuracy_score(y_ev_arr, P_fused.argmax(axis=1)))
                P_sk = sinkhorn_balancing(P_fused, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
                acc_gated_sk = float(balanced_accuracy_score(y_ev_arr, P_sk.argmax(axis=1)))

                if abs(gw - default_gw) < 1e-4:
                    default_raw_bacc = acc_gated
                    default_sk_bacc = acc_gated_sk

                if acc_gated_sk > best_pipe_acc:
                    best_pipe_acc = acc_gated_sk
                    best_pipe_w = gw
                    best_raw_bacc = acc_gated
                    best_sk_bacc = acc_gated_sk

            p_res = {
                "plain_solo_bacc": acc_plain_solo,
                "best_pipe_bacc": best_pipe_acc,
                "best_gate_weight": best_pipe_w,
                "best_raw_bacc": best_raw_bacc,
                "best_sinkhorn_bacc": best_sk_bacc,
                "default_raw_bacc": default_raw_bacc,
                "default_sinkhorn_bacc": default_sk_bacc,
                "lift_over_plain": best_pipe_acc - acc_plain_solo,
            }
        return ev_acc, hard_acc, p_res

    r_ev_acc, r_hard_acc, r_pipe_res = _eval_head_metrics(ridge_logits_ev)
    m_ev_acc, m_hard_acc, m_pipe_res = _eval_head_metrics(mlp_logits_ev)
    h_ev_acc, h_hard_acc, h_pipe_res = _eval_head_metrics(hybrid_logits_ev)

    heads_breakdown = {
        "ridge": {
            "eval_fam_acc": r_ev_acc if ridge_logits_ev is not None else float(ridge_acc),
            "hard_pair_acc": r_hard_acc if ridge_logits_ev is not None else float(ridge_acc),
            "oof_acc": float(ridge_oof_bacc),
            "pipe_results": r_pipe_res,
            "logits": ridge_logits_ev,
        },
        "mlp": {
            "eval_fam_acc": m_ev_acc if mlp_logits_ev is not None else float(mlp_acc),
            "hard_pair_acc": m_hard_acc if mlp_logits_ev is not None else float(mlp_acc),
            "oof_acc": float(mlp_val_acc),
            "pipe_results": m_pipe_res,
            "logits": mlp_logits_ev,
        },
        "hybrid": {
            "eval_fam_acc": h_ev_acc if hybrid_logits_ev is not None else float(mlp_acc),
            "hard_pair_acc": h_hard_acc if hybrid_logits_ev is not None else float(mlp_acc),
            "oof_acc": float(0.65 * mlp_val_acc + 0.35 * ridge_oof_bacc),
            "pipe_results": h_pipe_res,
            "logits": hybrid_logits_ev,
        },
    }

    if head_type.lower() == "mlp":
        chosen_head = heads_breakdown["mlp"]
        best_fam_acc = chosen_head["eval_fam_acc"]
        best_fam_logits = mlp_logits_ev
        best_alpha = ridge_alpha
        final_head_dict = mlp_head_dict
        final_head_dict["plain.weight"] = plain_w_tensor
        final_head_dict["plain.bias"] = plain_b_tensor
    elif head_type.lower() in ("hybrid", "hybrid_65mlp_35ridge"):
        chosen_head = heads_breakdown["hybrid"]
        best_fam_acc = chosen_head["eval_fam_acc"]
        best_fam_logits = hybrid_logits_ev if hybrid_logits_ev is not None else mlp_logits_ev
        best_alpha = ridge_alpha
        final_head_dict = mlp_head_dict
        final_head_dict["plain.weight"] = plain_w_tensor
        final_head_dict["plain.bias"] = plain_b_tensor
    else:
        chosen_head = heads_breakdown["ridge"]
        best_fam_acc = chosen_head["eval_fam_acc"]
        best_fam_logits = ridge_logits_ev
        best_alpha = ridge_alpha
        final_head_dict = {
            "plain.weight": plain_w_tensor,
            "plain.bias": plain_b_tensor,
        }

    hard_pair_eval_acc = chosen_head["hard_pair_acc"]
    pipe_results = chosen_head["pipe_results"]
    best_pipe_w = pipe_results.get("best_gate_weight", default_gw)

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        norms_src = ds_p / "heads.safetensors"
        if norms_src.exists():
            try:
                src_sd = load_file(str(norms_src))
                for k, v in src_sd.items():
                    if k.startswith("norms."):
                        final_head_dict[k] = v
            except Exception:
                pass
        save_file(final_head_dict, str(out_p / "heads.safetensors"))
        meta_fam = {
            "classes": family_classes + (["__OTHER__"] if has_negatives else []),
            "labels": family_classes + (["__OTHER__"] if has_negatives else []),
            "target_family": target_family,
            "head_type": head_type.lower(),
            "default_gate_weight": default_gw,
            "best_sweep_gate_weight": float(best_pipe_w),
            "use_supervised_subspace": use_supervised_subspace,
            "subspace_dim": int(pca_components.shape[0]) if pca_components is not None else None,
            "pca_dim": pca_dim,
            "hidden_dim": hidden_dim if "mlp" in head_type.lower() or "hybrid" in head_type.lower() else None,
            "alpha": float(best_alpha),
            "ridge_alpha": float(ridge_alpha),
            "ridge_within_family_acc": float(ridge_acc),
            "mlp_within_family_acc": float(mlp_acc),
            "within_family_acc": float(best_fam_acc),
            "plain_within_family_acc": float(acc_plain_within),
            "hard_pair_eval_acc": float(hard_pair_eval_acc),
            "lift_over_plain": float(best_fam_acc - acc_plain_within),
        }
        (out_p / "meta.json").write_text(json.dumps(meta_fam, indent=2), encoding="utf-8")
        if checkpoints_volume is not None and hasattr(checkpoints_volume, "commit"):
            checkpoints_volume.commit()
        print(f"[{target_family.upper()} Gate] Checkpoint successfully saved to {out_p}.")

    return {
        "target_family": target_family,
        "within_family_balanced_acc": float(best_fam_acc),
        "plain_within_family_acc": float(acc_plain_within),
        "delta_within_family": float(best_fam_acc - acc_plain_within),
        "hard_pair_eval_acc": float(hard_pair_eval_acc),
        "best_alpha": float(best_alpha),
        "ridge_acc": float(ridge_acc),
        "mlp_acc": float(mlp_acc),
        "subspace_dim": int(pca_components.shape[0]) if pca_components is not None else 0,
        "pipe_results": pipe_results,
        "heads_breakdown": heads_breakdown,
        "spec_logits": best_fam_logits[:, :num_family_classes].tolist() if best_fam_logits is not None else [],
        "spec_classes": family_classes,
    }


def train_family_gate(
    model: nn.Module,
    tok: Any,
    df_pool: pd.DataFrame,
    df_train: pd.DataFrame | None,
    target_family: str,
    samples_per_class: int = 0,
    negative_ratio: float = 0.0,
    eval_texts: list[str] | None = None,
    eval_labels: list[int] | None = None,
    batch_size: int = 32,
    max_len: int | None = None,
    end_id: int | None = None,
    device: torch.device | None = None,
    output_dir: Path | str | None = None,
    checkpoints_volume: Any | None = None,
    head_type: str = "mlp",
    use_supervised_subspace: bool = True,
    subspace_dim: int | None = None,
    pca_dim: int | None = None,
    hidden_dim: int = 64,
    mlp_epochs: int = 12,
    mlp_lr: float = 2e-4,
    mlp_dropout: float = 0.35,
    export_embeddings: bool = False,
    save_all_layers: bool = False,
    embeddings_output_dir: Path | str | None = None,
    gate_weight: float = 0.40,
    train_cfg: dict[str, Any] | None = None,
    include_velocity: bool = True,
    layer_weighting: bool = True,
    hard_sister_pair: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """End-to-end family gate training pipeline extracting representations on model forward_features."""
    if subspace_dim is None and pca_dim is not None:
        subspace_dim = pca_dim

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
        default_samples = {"glm": 4000, "gemini": 3800, "gpt": 2500, "claude": 2500}
        samples_per_class = default_samples.get(target_family, 2500)

    fam_hint = getattr(model, "backbone_family", "unknown")
    adapter_cfg: dict[str, Any] = {}
    if fam_hint and fam_hint != "unknown":
        adapter_cfg["backbone_family"] = fam_hint
    else:
        bb_obj = getattr(model, "backbone", model)
        adapter_cfg["backbone"] = f"{type(bb_obj)} {getattr(bb_obj, 'config', '')}"
    adapter = resolve_backbone_adapter(adapter_cfg)

    tok_spec = getattr(model, "tok_spec", None)
    if tok_spec is None:
        tok_spec = adapter.tok_spec_from_tokenizer(tok, max_len=max_len, end_id=end_id)

    batch_size = adapter.clamp_extraction_batch_size(batch_size)

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

    fam_dfs = []
    for c in family_classes:
        sub_tr = df_train[df_train["model"] == c] if (df_train is not None and len(df_train) > 0) else pd.DataFrame()
        sub_pool = df_pool[df_pool["model"] == c]
        n_take = max(min(samples_per_class - len(sub_tr), len(sub_pool)), 0)
        sampled_pool = sub_pool.sample(n=n_take, random_state=42) if n_take > 0 else pd.DataFrame()
        c_df = pd.concat([sub_tr, sampled_pool], ignore_index=True)
        fam_dfs.append(c_df)
        print(f"  {target_family.upper()} class {c}: {len(c_df)} samples ({len(sub_tr)} train, {len(c_df)-len(sub_tr)} pool)")

    df_fam_all = pd.concat(fam_dfs, ignore_index=True)
    total_fam_samples = len(df_fam_all)

    if has_negatives:
        n_neg = int(total_fam_samples * negative_ratio)
        df_non_fam = df_pool[~df_pool["model"].isin(family_classes)].copy()
        other_fams = sorted({"".join(ch for ch in m.split("-")[0].split(".")[0] if ch.isalpha()) for m in LABELS if m not in family_classes})
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
    fam_pids = (
        df_fam_all["prompt_id"].fillna("-1").astype(str).tolist()
        if "prompt_id" in df_fam_all.columns
        else [f"pool_{i}" for i in range(len(df_fam_all))]
    )
    neg_pids = (
        df_neg_all["prompt_id"].fillna("-1").astype(str).tolist()
        if (len(df_neg_all) > 0 and "prompt_id" in df_neg_all.columns)
        else [f"neg_{i}" for i in range(len(df_neg_all))]
    )
    prompt_ids = fam_pids + neg_pids
    other_str = " + 1 Other" if len(df_neg_all) > 0 else ""
    print(f"[{target_family.upper()} Gate] Total training dataset: {len(train_texts):,} samples ({num_gate_classes} classes: {num_family_classes} {target_family.upper()}{other_str}).")

    amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
        else torch.float16
    )

    print(f"[{target_family.upper()} Gate] Extracting Z_train on {len(train_texts)} samples (batch_size={batch_size})...")
    pad_id = tok_spec.pad_id
    cls_id = tok_spec.cls_id
    eff_end_id = tok_spec.end_id if end_id is None else end_id
    eff_max_len = int(max_len if max_len is not None else tok_spec.max_len)
    drop_first_token = tok_spec.drop_first_token
    collate_fn_gate = functools.partial(collate_text_batch, pad_token_id=pad_id)

    ds_tr = TextSequenceDataset(train_texts, train_labels, None, tok, max_len=eff_max_len, end_id=eff_end_id, cls_id=cls_id)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gate)

    Z_tr_list = []
    Z_tr_layers_list = []
    tr_cos_list = []
    tr_std_list = []
    need_layers = bool(export_embeddings or save_all_layers)
    if save_all_layers and export_embeddings:
        try:
            emb_target_dir = Path(embeddings_output_dir) if embeddings_output_dir else (Path(output_dir) / "embeddings" if output_dir else Path("export_embeddings"))
            emb_target_dir.mkdir(parents=True, exist_ok=True)
            _, _, free_bytes = shutil.disk_usage(emb_target_dir)
            n_lay_est = adapter.default_layer_count_for_disk_estimate(model)
            h_dim_est = int(
                getattr(model, "hidden_size", 0)
                or getattr(getattr(getattr(model, "backbone", model), "config", None), "hidden_size", 0)
                or (getattr(model, "total_dim", 0) // max(len(getattr(model, "layers", [1])), 1))
            )
            est_all_layers_bytes = len(train_texts) * n_lay_est * max(h_dim_est, 1) * 2
            if est_all_layers_bytes > 0.80 * free_bytes:
                print(
                    f"\n[WARNING][Disk Safety Guard] Estimated {n_lay_est}-layer tensor size ({est_all_layers_bytes / (1024**3):.2f} GB) "
                    f"exceeds 80% of available free disk space ({free_bytes / (1024**3):.2f} GB) in {emb_target_dir}."
                )
                print(
                    f"  -> Automatically falling back to saving {len(getattr(model, 'layers', []))} tapped layers in "
                    f"features_by_layer.safetensors ([N, {len(getattr(model, 'layers', []))}, {h_dim_est}]).\n"
                )
                save_all_layers = False
        except Exception:
            pass
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=(amp_dtype == torch.bfloat16), dtype=amp_dtype):
            for batch in loader_tr:
                mb_ids = batch["input_ids"].to(device)
                mb_mask = batch["attention_mask"].to(device)
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                if need_layers:
                    res = model.forward_features(
                        mb_ids[:, :max_l],
                        mb_mask[:, :max_l],
                        drop_first_token=drop_first_token,
                        return_by_layer=True,
                        extract_all_layers=save_all_layers,
                        return_multiview=True,
                    )
                    if isinstance(res, tuple) and len(res) == 4:
                        z_b, z_l, ht_cos_t, std_norm_t = res
                        tr_cos_list.append(ht_cos_t.cpu().to(torch.float16).numpy())
                        tr_std_list.append(std_norm_t.cpu().to(torch.float16).numpy())
                    elif isinstance(res, tuple) and len(res) == 2:
                        z_b, z_l = res
                    Z_tr_layers_list.append(z_l.cpu().to(torch.float16).numpy())
                else:
                    z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=drop_first_token)

                if len(Z_tr_list) == 0:
                    d_chunk = max(z_b.shape[1] // max(len(model.layers), 1), 1)
                    chunk_stds = [float(z_b[:, i * d_chunk : (i + 1) * d_chunk].std().item()) for i in range(len(model.layers))]
                    layer_stds = (
                        [float(z_l[:, i, :].std().item()) for i in range(z_l.shape[1])]
                        if (need_layers and z_l is not None and z_l.ndim == 3)
                        else chunk_stds
                    )
                    dead_layers = [i for i, s in enumerate(layer_stds) if s < 1e-6 or not math.isfinite(s)]
                    print(
                        f"[Sanity Check Batch 0] Active layers: {len(layer_stds) - len(dead_layers)}/{len(layer_stds)} | "
                        f"Layer std range: [{min(layer_stds):.4f}, {max(layer_stds):.4f}] | "
                        f"Tapped {len(model.layers)}-layer std range: [{min(chunk_stds):.4f}, {max(chunk_stds):.4f}]",
                        flush=True,
                    )
                    if dead_layers:
                        raise RuntimeError(f"Batch 0 Sanity Check FAILED: Layers {dead_layers} have zero/NaN activations!")

                Z_tr_list.append(z_b.cpu().float().numpy())
    Z_train = np.vstack(Z_tr_list)
    Z_train_layers = np.vstack(Z_tr_layers_list) if Z_tr_layers_list else None
    head_tail_cos_tr = np.concatenate(tr_cos_list, axis=0) if tr_cos_list else None
    token_std_tr = np.concatenate(tr_std_list, axis=0) if tr_std_list else None
    y_train = np.array(train_labels, dtype=np.int64)
    print(
        f"[{target_family.upper()} Gate] Extracted Z_train: shape={Z_train.shape}, dtype={Z_train.dtype}"
        + (f", layers_shape={Z_train_layers.shape}" if Z_train_layers is not None else "")
    )

    Z_eval = None
    Z_eval_layers = None
    head_tail_cos_ev = None
    token_std_ev = None
    main_logits_eval = None
    main_logits_train = None
    centered_logits_train = None
    centered_logits_eval = None
    if eval_texts is not None and len(eval_texts) > 0:
        print(f"[{target_family.upper()} Gate] Extracting Z_eval on {len(eval_texts)} benchmark samples...")
        dummy_eval_y = [0] * len(eval_texts)
        ds_ev = TextSequenceDataset(eval_texts, dummy_eval_y, None, tok, max_len=eff_max_len, end_id=eff_end_id, cls_id=cls_id)
        loader_ev = DataLoader(ds_ev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gate)
        Z_ev_list = []
        Z_ev_layers_list = []
        ev_cos_list = []
        ev_std_list = []
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=(amp_dtype == torch.bfloat16), dtype=amp_dtype):
                for batch in loader_ev:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    if need_layers:
                        res = model.forward_features(
                            mb_ids[:, :max_l],
                            mb_mask[:, :max_l],
                            drop_first_token=drop_first_token,
                            return_by_layer=True,
                            extract_all_layers=save_all_layers,
                            return_multiview=True,
                        )
                        if isinstance(res, tuple) and len(res) == 4:
                            z_b, z_l, ht_cos_t, std_norm_t = res
                            ev_cos_list.append(ht_cos_t.cpu().to(torch.float16).numpy())
                            ev_std_list.append(std_norm_t.cpu().to(torch.float16).numpy())
                        elif isinstance(res, tuple) and len(res) == 2:
                            z_b, z_l = res
                        Z_ev_layers_list.append(z_l.cpu().to(torch.float16).numpy())
                    else:
                        z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=drop_first_token)
                    Z_ev_list.append(z_b.cpu().float().numpy())
        Z_eval = np.vstack(Z_ev_list)
        Z_eval_layers = np.vstack(Z_ev_layers_list) if Z_ev_layers_list else None
        head_tail_cos_ev = np.concatenate(ev_cos_list, axis=0) if ev_cos_list else None
        token_std_ev = np.concatenate(ev_std_list, axis=0) if ev_std_list else None

    if hasattr(model, "plain_head") and model.plain_head is not None:
        W_p = model.plain_head.weight.detach().cpu().float().numpy()
        b_p = model.plain_head.bias.detach().cpu().float().numpy()
        main_logits_train = Z_train @ W_p.T + b_p
        if Z_eval is not None:
            main_logits_eval = Z_eval @ W_p.T + b_p

    if hasattr(model, "centered_head") and model.centered_head is not None:
        W_c = model.centered_head.weight.detach().cpu().float().numpy()
        b_c = model.centered_head.bias.detach().cpu().float().numpy()
        centered_logits_train = Z_train @ W_c.T + b_c
        if Z_eval is not None:
            centered_logits_eval = Z_eval @ W_c.T + b_c

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
            prompt_ids=prompt_ids,
            champion_logits_plain=main_logits_train,
            champion_logits_centered=centered_logits_train,
            head_tail_cos=head_tail_cos_tr,
            token_std_norms=token_std_tr,
            eval_features_concat=Z_eval,
            eval_features_by_layer=Z_eval_layers,
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            eval_champion_logits_plain=main_logits_eval,
            eval_champion_logits_centered=centered_logits_eval,
            eval_head_tail_cos=head_tail_cos_ev,
            eval_token_std_norms=token_std_ev,
            dataset_title=f"NTL {target_family.upper()} Extracted Features",
            dataset_slug=f"ntl-{target_family.lower()}-features",
            checkpoints_volume=checkpoints_volume,
        )
        if hasattr(model, "norms") and model.norms is not None:
            norms_dict = {}
            for i, norm in enumerate(model.norms):
                norms_dict[f"norms.{i}.weight"] = norm.weight.detach().cpu().to(torch.float32).contiguous()
                if hasattr(norm, "bias") and norm.bias is not None:
                    norms_dict[f"norms.{i}.bias"] = norm.bias.detach().cpu().to(torch.float32).contiguous()
            save_file(norms_dict, str(emb_dir / "heads.safetensors"))

    norms = np.linalg.norm(Z_train, axis=1)
    q25, q50, q75 = np.percentile(norms, [25, 50, 75])
    iqr = q75 - q25
    norm_outliers_3iqr = int(((norms < q25 - 3.0 * iqr) | (norms > q75 + 3.0 * iqr)).sum())
    norm_outliers_15iqr = int(((norms < q25 - 1.5 * iqr) | (norms > q75 + 1.5 * iqr)).sum())

    num_tapped = len(getattr(model, "layers", [2, 4, 6, 9, 13, 17, 22, 27]))
    chunk_dim = Z_train.shape[1] // max(num_tapped, 1)

    fam_mask_stats = y_train < num_family_classes
    if fam_mask_stats.sum() > 0:
        mu_fam = Z_train[fam_mask_stats].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(Z_train[fam_mask_stats] - mu_fam, axis=1)
        dq25, dq50, dq75 = np.percentile(dists, [25, 50, 75])
        diqr = dq75 - dq25
        dist_outliers_3iqr = int((dists > dq75 + 3.0 * diqr).sum())
        dist_outliers_15iqr = int((dists > dq75 + 1.5 * diqr).sum())
    else:
        dists = np.zeros(len(Z_train))
        dq50, diqr = 0.0, 0.0
        dist_outliers_3iqr, dist_outliers_15iqr = 0, 0

    print("\n" + "=" * 80)
    print(f" [{target_family.upper()} Gate] Step 2: Quality Inspection, Normalization Check & Outlier Statistics:")
    print(f"  - Total training samples: {len(Z_train):,} ({len(df_fam_all)} intra-family across {num_family_classes} classes)")
    for i, c in enumerate(family_classes):
        print(f"    * Class {i} ({c}): {int((y_train == i).sum()):,} samples")
    print(f"  - Feature dimension: {Z_train.shape[1]:,} ({num_tapped} tapped layers x {chunk_dim}-dim LayerNormed)")
    print(f"  - Sample L2 norm stats: mean={norms.mean():.2f}, std={norms.std():.2f}, median={q50:.2f}, IQR={iqr:.2f}")
    print(f"  - Centroid distance stats: mean={dists.mean():.2f}, std={dists.std():.2f}, median={dq50:.2f}, IQR={diqr:.2f}")
    print(f"  - Outliers by norm (>1.5/3.0 IQR): {norm_outliers_15iqr}/{norm_outliers_3iqr} | dist: {dist_outliers_15iqr}/{dist_outliers_3iqr}")
    print("=" * 80 + "\n")

    if Z_train_layers is not None:
        hard_pair = (2, 3) if target_family == "gemini" else ((1, 2) if target_family == "gpt" else (0, 1))
        run_all_layers_probe_sweep(
            features_by_layer=Z_train_layers,
            y=y_train,
            num_family_classes=num_family_classes,
            hard_sister_pair=hard_pair,
            output_dir=output_dir,
            prompts=np.asarray(prompt_ids) if prompt_ids is not None else None,
        )

    subspace_components = None
    subspace_mean = None
    Z_tr_active = Z_train
    Z_ev_active = Z_eval

    if use_supervised_subspace:
        print(f"[{target_family.upper()} Gate] Step 3: Building Pure Supervised Sister-Decoupling Subspace...")
        Z_tr_active, Z_ev_active, subspace_components, subspace_mean, sub_diag = build_oof_supervised_family_subspace(
            Z_train=Z_train,
            y_train=y_train,
            Z_eval=Z_eval,
            num_family_classes=num_family_classes,
            num_tapped_layers=num_tapped,
            subspace_dim=subspace_dim,
            device=device,
            include_velocity=include_velocity,
            layer_weighting=layer_weighting,
            hard_sister_pair=hard_sister_pair,
            prompt_ids=prompt_ids,
        )
        print(f"[{target_family.upper()} Gate] Step 3 Complete: {sub_diag['total_subspace_dim']} whitened dims.")

    pca_components = subspace_components
    pca_mean = subspace_mean

    y_ev_arr = np.array(eval_labels) if eval_labels is not None else None
    fam_eval_mask = np.isin(y_ev_arr, family_cols) if y_ev_arr is not None else None
    col_to_local = {c: i for i, c in enumerate(family_cols)}
    y_fam_true = np.array([col_to_local[v] for v in y_ev_arr[fam_eval_mask]]) if fam_eval_mask is not None else None

    acc_plain_within = 0.0
    if main_logits_eval is not None and fam_eval_mask is not None and fam_eval_mask.sum() > 0:
        if main_logits_eval.shape[1] == len(LABELS):
            plain_fam_pred = main_logits_eval[fam_eval_mask][:, family_cols].argmax(axis=1)
        else:
            plain_fam_pred = main_logits_eval[fam_eval_mask][:, :num_family_classes].argmax(axis=1)
        acc_plain_within = float(balanced_accuracy_score(y_fam_true, plain_fam_pred))

    best_w_pca, best_b_pca, ridge_alpha, ridge_acc, ridge_logits_ev, _ = fit_family_gate_head(
        Z_train=Z_tr_active,
        y_train=y_train,
        num_gate_classes=num_gate_classes,
        alphas=[1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0],
        label_smoothing=0.10,
        Z_eval=Z_ev_active,
        y_eval_true=y_fam_true,
        eval_mask=fam_eval_mask,
        num_family_classes=num_family_classes,
        device=device,
        prompt_ids=prompt_ids,
    )

    if pca_components is not None:
        comp_64 = pca_components.astype(np.float64)
        mu_64 = pca_mean.reshape(1, -1).astype(np.float64)
        W_r_eff_64 = comp_64.T @ best_w_pca.astype(np.float64)
        b_r_eff_64 = best_b_pca.astype(np.float64) - (mu_64 @ W_r_eff_64).squeeze(0)
        best_w = W_r_eff_64.astype(np.float32)
        best_b = b_r_eff_64.astype(np.float32)
    else:
        best_w = best_w_pca
        best_b = best_b_pca

    plain_w_tensor = torch.from_numpy(best_w.T).to(torch.float32).contiguous()
    plain_b_tensor = torch.from_numpy(best_b).to(torch.float32).contiguous()

    mlp_head_dict, mlp_acc, mlp_logits_ev, _ = fit_family_mlp_head(
        Z_train=Z_tr_active,
        y_train=y_train,
        num_gate_classes=num_gate_classes,
        num_family_classes=num_family_classes,
        hidden_dim=hidden_dim,
        epochs=mlp_epochs,
        lr=mlp_lr,
        dropout=mlp_dropout,
        label_smoothing=0.10,
        Z_eval=Z_ev_active,
        y_eval_true=y_fam_true,
        eval_mask=fam_eval_mask,
        device=device,
        pca_components=pca_components,
        pca_mean=pca_mean,
        raw_Z_eval=Z_eval if pca_components is not None else None,
    )

    hybrid_logits_ev = None
    if mlp_logits_ev is not None and ridge_logits_ev is not None:
        raw_hyb = 0.65 * mlp_logits_ev + 0.35 * ridge_logits_ev
        hyb_fam = raw_hyb[:, :num_family_classes]
        hyb_std = float(np.std(hyb_fam, axis=1).mean()) if len(hyb_fam) > 0 else 1.0
        hyb_scale = (3.2 / hyb_std) if hyb_std > 1e-4 else 1.0
        hybrid_logits_ev = (raw_hyb * hyb_scale).astype(np.float32)

    if head_type.lower() == "mlp":
        best_fam_acc = mlp_acc
        best_fam_logits = mlp_logits_ev
        best_alpha = ridge_alpha
        final_head_dict = mlp_head_dict
        final_head_dict["plain.weight"] = plain_w_tensor
        final_head_dict["plain.bias"] = plain_b_tensor
    elif head_type.lower() in ("hybrid", "hybrid_65mlp_35ridge"):
        if hybrid_logits_ev is not None and fam_eval_mask is not None and fam_eval_mask.sum() > 0:
            fam_hyb_pred = hybrid_logits_ev[fam_eval_mask, :num_family_classes].argmax(axis=1)
            hybrid_acc = float(balanced_accuracy_score(y_fam_true, fam_hyb_pred))
        else:
            hybrid_acc = mlp_acc
        best_fam_acc = hybrid_acc
        best_fam_logits = hybrid_logits_ev if hybrid_logits_ev is not None else mlp_logits_ev
        best_alpha = ridge_alpha
        final_head_dict = mlp_head_dict
        final_head_dict["plain.weight"] = plain_w_tensor
        final_head_dict["plain.bias"] = plain_b_tensor
    else:
        best_fam_acc = ridge_acc
        best_fam_logits = ridge_logits_ev
        best_alpha = ridge_alpha
        final_head_dict = {"plain.weight": plain_w_tensor, "plain.bias": plain_b_tensor}

    default_gw = float(train_cfg.get("gate_weight", gate_weight)) if isinstance(train_cfg, dict) else float(gate_weight)
    best_pipe_w = default_gw
    pipe_results = {}
    if main_logits_eval is not None and best_fam_logits is not None and y_ev_arr is not None and main_logits_eval.shape[1] == len(LABELS):
        P_main = sp_softmax(main_logits_eval, axis=1)
        acc_plain_solo = float(balanced_accuracy_score(y_ev_arr, P_main.argmax(axis=1)))
        best_pipe_acc = 0.0
        cand_mask = np.isin(P_main.argmax(axis=1), family_cols) | (P_main[:, family_cols].sum(axis=1) >= 0.10)
        spec_probs = sp_softmax(best_fam_logits[:, :num_family_classes], axis=1)
        sweep_weights = sorted(set([0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, default_gw]))

        for gw in sweep_weights:
            P_fused = P_main.copy()
            for idx in np.flatnonzero(cand_mask):
                orig_mass = float(P_main[idx, family_cols].sum())
                if orig_mass > 1e-6:
                    gw_eff = gw * min(orig_mass / 0.50, 1.0)
                    main_norm = P_main[idx, family_cols] / orig_mass
                    fused_fam = (1.0 - gw_eff) * main_norm + gw_eff * spec_probs[idx]
                    P_fused[idx, family_cols] = fused_fam * orig_mass

            P_sk = sinkhorn_balancing(P_fused, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
            acc_gated_sk = float(balanced_accuracy_score(y_ev_arr, P_sk.argmax(axis=1)))
            if acc_gated_sk > best_pipe_acc:
                best_pipe_acc = acc_gated_sk
                best_pipe_w = gw

        pipe_results = {
            "plain_solo_bacc": acc_plain_solo,
            "best_pipe_bacc": best_pipe_acc,
            "best_gate_weight": best_pipe_w,
            "lift_over_plain": best_pipe_acc - acc_plain_solo,
        }

    hard_pair_eval_acc = float(best_fam_acc)
    pair_check = hard_sister_pair
    if pair_check is None:
        if num_family_classes == 4:
            pair_check = (2, 3)
        elif num_family_classes == 5:
            pair_check = (1, 2)
        elif num_family_classes == 2:
            pair_check = (0, 1)

    if y_fam_true is not None and best_fam_logits is not None and pair_check is not None and fam_eval_mask is not None and fam_eval_mask.sum() > 0:
        s_a, s_b = pair_check
        pair_mask_ev = (y_fam_true == s_a) | (y_fam_true == s_b)
        if pair_mask_ev.sum() > 0 and len(np.unique(y_fam_true[pair_mask_ev])) == 2:
            fam_logits = best_fam_logits[fam_eval_mask, :num_family_classes]
            spec_pair_preds = np.where(fam_logits[pair_mask_ev, s_a] >= fam_logits[pair_mask_ev, s_b], s_a, s_b)
            hard_pair_eval_acc = float(balanced_accuracy_score(y_fam_true[pair_mask_ev], spec_pair_preds))

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "norms") and model.norms is not None:
            for i, norm in enumerate(model.norms):
                final_head_dict[f"norms.{i}.weight"] = norm.weight.detach().cpu().to(torch.float32).contiguous()
                if hasattr(norm, "bias") and norm.bias is not None:
                    final_head_dict[f"norms.{i}.bias"] = norm.bias.detach().cpu().to(torch.float32).contiguous()
        save_file(final_head_dict, str(out_p / "heads.safetensors"))
        meta_fam = {
            "classes": family_classes + (["__OTHER__"] if has_negatives else []),
            "labels": family_classes + (["__OTHER__"] if has_negatives else []),
            "target_family": target_family,
            "head_type": head_type.lower(),
            "default_gate_weight": default_gw,
            "best_sweep_gate_weight": float(best_pipe_w),
            "use_supervised_subspace": use_supervised_subspace,
            "subspace_dim": int(pca_components.shape[0]) if pca_components is not None else None,
            "pca_dim": pca_dim,
            "hidden_dim": hidden_dim if ("mlp" in head_type.lower() or "hybrid" in head_type.lower()) else None,
            "alpha": float(best_alpha),
            "ridge_alpha": float(ridge_alpha),
            "ridge_within_family_acc": float(ridge_acc),
            "mlp_within_family_acc": float(mlp_acc),
            "within_family_acc": float(best_fam_acc),
            "plain_within_family_acc": float(acc_plain_within),
            "hard_pair_eval_acc": float(hard_pair_eval_acc),
            "lift_over_plain": float(best_fam_acc - acc_plain_within),
        }
        (out_p / "meta.json").write_text(json.dumps(meta_fam, indent=2), encoding="utf-8")
        if checkpoints_volume is not None and hasattr(checkpoints_volume, "commit"):
            checkpoints_volume.commit()
        print(f"[{target_family.upper()} Gate] Checkpoint saved to {out_p}.")

    try:
        del Z_train, Z_train_layers, Z_eval, Z_eval_layers, Z_tr_active, Z_ev_active
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    should_sweep = export_embeddings and (train_cfg.get("run_sweep", True) if isinstance(train_cfg, dict) else True)
    if should_sweep:
        print(f"\n[{target_family.upper()} Gate] Automatically launching offline feature & head sweep...")
        run_sweep(
            embeddings_dir=emb_dir,
            target_family=target_family,
            output_dir=output_dir,
            ledger_dir=Path(output_dir) / "experiments" if output_dir else "experiments",
            device=device,
        )

    return {
        "target_family": target_family,
        "within_family_balanced_acc": float(best_fam_acc),
        "plain_within_family_acc": float(acc_plain_within),
        "delta_within_family": float(best_fam_acc - acc_plain_within),
        "hard_pair_eval_acc": float(hard_pair_eval_acc),
        "best_alpha": float(best_alpha),
        "ridge_acc": float(ridge_acc),
        "mlp_acc": float(mlp_acc),
        "subspace_dim": int(pca_components.shape[0]) if pca_components is not None else 0,
        "pipe_results": pipe_results,
        "spec_logits": best_fam_logits[:, :num_family_classes].tolist() if best_fam_logits is not None else [],
        "spec_classes": family_classes,
    }


def run_sweep(
    embeddings_dir: Path | str,
    target_family: str,
    output_dir: Path | str | None = None,
    ledger_dir: Path | str = "experiments",
    device: torch.device | None = None,
    exp_prefix: str = "EXP",
) -> dict[str, Any]:
    """Executes offline ablation sweep across feature engineering variants and heads in <30s total."""
    ds_p = Path(embeddings_dir)
    target_family = target_family.lower().strip()
    if output_dir is None:
        output_dir = Path(f"export_{target_family}_gate_mlp")
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    ledger = ExperimentLedger(ledger_dir=ledger_dir)

    print("\n" + "=" * 92)
    print(f" [RUNNER] STARTING OFFLINE GATE EXPERIMENT SWEEP: '{target_family.upper()}' ({ds_p})")
    print("=" * 92)

    family_cols = FAMILIES[target_family]
    eff_hard_pair = (2, 3) if target_family == "gemini" else ((1, 2) if target_family in ("gpt", "claude") else (0, 1))

    y_ev_arr = None
    main_logits_eval = None
    baseline_raw_bacc = 0.0
    baseline_sk_bacc = 0.0
    fam_solo_bacc = 0.0
    hard_pair_solo_acc = 0.0

    if (ds_p / "benchmark_metadata.parquet").exists():
        df_bench = pd.read_parquet(ds_p / "benchmark_metadata.parquet")
        y_ev_arr = df_bench["label"].to_numpy()

    if (ds_p / "benchmark_champion_logits.safetensors").exists():
        champ_ev_sd = load_file(str(ds_p / "benchmark_champion_logits.safetensors"))
        if "logits_plain" in champ_ev_sd:
            main_logits_eval = champ_ev_sd["logits_plain"].cpu().float().numpy()
        elif "champion_logits" in champ_ev_sd:
            main_logits_eval = champ_ev_sd["champion_logits"].cpu().float().numpy()
        elif len(champ_ev_sd) > 0 and any("logit" in k.lower() for k in champ_ev_sd):
            for k in champ_ev_sd:
                if "logit" in k.lower():
                    main_logits_eval = champ_ev_sd[k].cpu().float().numpy()
                    break

    if y_ev_arr is not None and main_logits_eval is not None:
        if main_logits_eval.shape[1] == len(LABELS):
            P_main = sp_softmax(main_logits_eval, axis=1)
            baseline_raw_bacc = float(balanced_accuracy_score(y_ev_arr, P_main.argmax(axis=1)))
            P_sk = sinkhorn_balancing(P_main, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
            baseline_sk_bacc = float(balanced_accuracy_score(y_ev_arr, P_sk.argmax(axis=1)))

            fam_eval_mask = np.isin(y_ev_arr, family_cols)
            if fam_eval_mask.sum() > 0:
                col_to_local = {c: i for i, c in enumerate(family_cols)}
                y_fam_true = np.array([col_to_local[v] for v in y_ev_arr[fam_eval_mask]])
                plain_fam_pred = main_logits_eval[fam_eval_mask][:, family_cols].argmax(axis=1)
                fam_solo_bacc = float(balanced_accuracy_score(y_fam_true, plain_fam_pred))

                s_a, s_b = eff_hard_pair
                pair_mask = (y_fam_true == s_a) | (y_fam_true == s_b)
                if pair_mask.sum() > 0 and len(np.unique(y_fam_true[pair_mask])) == 2:
                    pair_preds = np.where(
                        main_logits_eval[fam_eval_mask][pair_mask, family_cols[s_a]]
                        >= main_logits_eval[fam_eval_mask][pair_mask, family_cols[s_b]],
                        s_a,
                        s_b,
                    )
                    hard_pair_solo_acc = float(balanced_accuracy_score(y_fam_true[pair_mask], pair_preds))
                else:
                    hard_pair_solo_acc = fam_solo_bacc
        else:
            fam_eval_mask = np.isin(y_ev_arr, family_cols)
            if fam_eval_mask.sum() > 0:
                col_to_local = {c: i for i, c in enumerate(family_cols)}
                y_fam_true = np.array([col_to_local[v] for v in y_ev_arr[fam_eval_mask]])
                plain_fam_pred = main_logits_eval[fam_eval_mask][:, : len(family_cols)].argmax(axis=1)
                fam_solo_bacc = float(balanced_accuracy_score(y_fam_true, plain_fam_pred))

                s_a, s_b = eff_hard_pair
                pair_mask = (y_fam_true == s_a) | (y_fam_true == s_b)
                if pair_mask.sum() > 0 and len(np.unique(y_fam_true[pair_mask])) == 2:
                    pair_preds = np.where(
                        main_logits_eval[fam_eval_mask][pair_mask, s_a] >= main_logits_eval[fam_eval_mask][pair_mask, s_b],
                        s_a,
                        s_b,
                    )
                    hard_pair_solo_acc = float(balanced_accuracy_score(y_fam_true[pair_mask], pair_preds))
                else:
                    hard_pair_solo_acc = fam_solo_bacc

    ledger.baseline_bacc = baseline_sk_bacc
    exp00_entry = {
        "exp_id": "EXP-00",
        "phase": "P1_baseline",
        "target_family": target_family,
        "feature_set": "F0_plain_solo",
        "subspace_dim": 0,
        "head_type": "plain",
        "gate_weight": 0.0,
        "oof_family_bacc": None,
        "holdout_family_bacc": fam_solo_bacc,
        "hard_sister_acc": hard_pair_solo_acc,
        "holdout_18cls_raw_bacc": baseline_raw_bacc,
        "holdout_18cls_sinkhorn_bacc": baseline_sk_bacc,
        "delta_vs_baseline_bacc": 0.0,
        "notes": "18-class Champion Plain Head solo baseline on benchmark holdout",
    }
    existing_df = ledger.get_dataframe()
    if existing_df is None or existing_df.empty or "exp_id" not in existing_df.columns or "EXP-00" not in existing_df["exp_id"].values:
        ledger.add_entry(exp00_entry)
    print(f"EXP-00 Complete: Solo BAcc={baseline_raw_bacc*100:.2f}% | Sinkhorn={baseline_sk_bacc*100:.2f}% | Family BAcc={fam_solo_bacc*100:.2f}%")

    pfx = exp_prefix if exp_prefix else "EXP"
    sep = "_" if ("_" in pfx or "-" in pfx) else "-"
    feature_specs = [
        (f"{pfx}{sep}01", "F1_fisher_per_layer", False, False, None, False, "Pure Per-Layer Fisher Subspace"),
        (f"{pfx}{sep}02", "F2_fisher_velocity", True, False, None, False, "+ Inter-Layer Velocity Fisher Subspace"),
        (f"{pfx}{sep}03", "F3_scalarmix_fisher", True, True, None, False, "+ ScalarMix Layer Importance Weighting"),
    ]
    has_by_layer = False
    if (ds_p / "features_by_layer.safetensors").exists():
        try:
            with safe_open(str(ds_p / "features_by_layer.safetensors"), framework="pt") as f:
                shp = f.get_slice("features_by_layer").get_shape()
                has_by_layer = len(shp) == 3 and shp[1] > 8
        except Exception:
            has_by_layer = False
    if has_by_layer:
        feature_specs.append((f"{pfx}{sep}04", "F4_auto_top8_fisher", True, True, "auto_top8", False, "+ Auto-Top-8 Layer Slicing"))
    feature_specs.append((f"{pfx}{sep}05", "F5_fisher_multiview", True, True, None, True, "+ Offline Multi-View Features (Upper Bound)"))

    sweep_results = []
    best_foldable_score = -1.0
    best_foldable_run = None
    suffix_map = {"ridge": "a", "mlp": "b", "hybrid": "c"}

    for base_id, feat_set, inc_vel, l_wt, sel_l, use_mv, desc in feature_specs:
        print(f"\n--- [{base_id}] Running {feat_set} ({desc}) ---")
        tmp_run_out = out_p / f"_tmp_{base_id.lower()}"
        try:
            res = train_family_gate_from_dataset(
                dataset_dir=ds_p,
                target_family=target_family,
                output_dir=tmp_run_out,
                head_type="hybrid",
                select_layers=sel_l,
                include_velocity=inc_vel,
                layer_weighting=l_wt,
                use_multiview=use_mv,
                device=device,
                gate_weight=0.40,
            )

            heads_bd = res.get("heads_breakdown", {})
            for h_type in ["ridge", "mlp", "hybrid"]:
                exp_id = f"{base_id}{suffix_map[h_type]}"
                h_info = heads_bd.get(h_type, {})
                p_res = h_info.get("pipe_results", {})
                hld_18_raw = p_res.get("best_raw_bacc", 0.0)
                hld_18_sk = p_res.get("best_pipe_bacc", 0.0)
                gw_best = p_res.get("best_gate_weight", 0.40)
                delta_lift = hld_18_sk - baseline_sk_bacc

                record = {
                    "exp_id": exp_id,
                    "phase": "P2_features" if h_type == "ridge" else "P3_heads",
                    "target_family": target_family,
                    "feature_set": feat_set,
                    "subspace_dim": res.get("subspace_dim", 0),
                    "head_type": h_type,
                    "gate_weight": gw_best,
                    "oof_family_bacc": h_info.get("oof_acc"),
                    "holdout_family_bacc": h_info.get("eval_fam_acc", 0.0),
                    "hard_sister_acc": h_info.get("hard_pair_acc", 0.0),
                    "holdout_18cls_raw_bacc": hld_18_raw,
                    "holdout_18cls_sinkhorn_bacc": hld_18_sk,
                    "delta_vs_baseline_bacc": delta_lift,
                    "notes": f"{desc} ({h_type.upper()}, gw={gw_best:.2f})",
                }
                arts = []
                if (tmp_run_out / "meta.json").exists():
                    arts.append(tmp_run_out / "meta.json")
                if (tmp_run_out / "layer_probe_report.json").exists():
                    arts.append(tmp_run_out / "layer_probe_report.json")
                ledger.add_entry(record, artifacts=arts)
                sweep_results.append((record, tmp_run_out))

                if sel_l is None and not use_mv and hld_18_sk > best_foldable_score:
                    best_foldable_score = hld_18_sk
                    best_foldable_run = (exp_id, tmp_run_out, record)

                print(
                    f"[{exp_id}] Result: Head={h_type.upper()} | Fam BAcc={record['holdout_family_bacc']*100:.2f}% | "
                    f"Hard Sis={record['hard_sister_acc']*100:.2f}% | 18c SK={hld_18_sk*100:.2f}% (d {delta_lift*100:+.2f}%)"
                )

        except Exception as e:
            print(f"[{base_id}] Encountered error: {e}")

    if best_foldable_run is not None:
        best_id, best_tmp_dir, best_rec = best_foldable_run
        print("\n" + "=" * 92)
        print(f" [WINNER] Selecting Best Foldable Checkpoint: {best_id} ({best_rec['feature_set']} + {best_rec['head_type']})")
        print("=" * 92)

        for src_f in best_tmp_dir.glob("*"):
            if src_f.is_file():
                shutil.copy2(str(src_f), str(out_p / src_f.name))
        meta_p = out_p / "meta.json"
        if meta_p.exists():
            try:
                m_data = json.loads(meta_p.read_text(encoding="utf-8"))
                m_data["winning_exp_id"] = best_id
                m_data["head_type"] = best_rec["head_type"]
                m_data["best_sweep_gate_weight"] = float(best_rec["gate_weight"])
                meta_p.write_text(json.dumps(m_data, indent=2), encoding="utf-8")
            except Exception:
                pass

    for _, tmp_dir in sweep_results:
        if tmp_dir.exists():
            try:
                shutil.rmtree(str(tmp_dir))
            except Exception:
                pass

    ledger.export_copy(out_p)
    print(f"\n[RUNNER] Sweep Complete. Exported champion gate and ledgers to: {out_p}")
    return {
        "target_family": target_family,
        "baseline_sk_bacc": baseline_sk_bacc,
        "best_foldable_score": best_foldable_score,
        "best_run": best_foldable_run[0] if best_foldable_run else None,
        "total_experiments": len(sweep_results) + 1,
    }
