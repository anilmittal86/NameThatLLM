"""
Evaluation, Cross-Validation, Ridge Head Solvers, Grouped Inference Logits, and Feature Extraction.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader

from ..config import FAMILIES, LABELS, K, softmax
from ..postprocess import apply_prior_and_sinkhorn, sinkhorn_balancing
from .data import PROBLEM_WATCH_CLASSES, TextSequenceDataset, collate_text_batch
from .modeling import MultiLayerLoRAClassifier


def _solve_centered_dual_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    scale_by_trace: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solves centered float64 dual Ridge regression with exact unpenalized intercept."""
    N = X.shape[0]
    mu_x = X.mean(axis=0, keepdims=True).astype(np.float64)
    mu_y = Y.mean(axis=0, keepdims=True).astype(np.float64)
    X_c = X.astype(np.float64) - mu_x
    Y_c = Y.astype(np.float64) - mu_y
    K_mat = X_c @ X_c.T
    if scale_by_trace:
        mean_diag = float(np.mean(np.sum(X_c * X_c, axis=1)))
        eff_alpha = float(alpha) * max(mean_diag / 512.0, 1.0)
    else:
        eff_alpha = float(alpha)
    K_mat.flat[:: N + 1] += max(eff_alpha, 1e-6)
    A = np.linalg.solve(K_mat, Y_c)
    W = (A.T @ X_c).astype(np.float32)
    b = (mu_y.ravel() - (mu_x @ W.astype(np.float64).T).ravel()).astype(np.float32)
    return W, b


def fit_ridge_heads(
    Z: np.ndarray,
    y: np.ndarray,
    prompts: np.ndarray | None = None,
    alpha: float = 10.0,
    label_smoothing: float = 0.10,
    num_classes: int = K,
    scale_by_trace: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits closed-form dual Ridge regression heads for plain and centered representations."""
    N, _ = Z.shape
    Y = np.zeros((N, num_classes), dtype=np.float32)
    for i, lab in enumerate(y):
        Y[i, lab] = 1.0
    if label_smoothing > 0.0:
        Y = (1.0 - label_smoothing) * Y + (label_smoothing / num_classes)

    W_p, b_p = _solve_centered_dual_ridge(Z, Y, alpha=alpha, scale_by_trace=scale_by_trace)

    Z_c = Z.copy()
    if prompts is not None:
        for p in np.unique(prompts):
            mask = prompts == p
            if mask.sum() > 1:
                Z_c[mask] = Z[mask] - Z[mask].mean(axis=0, keepdims=True)
            else:
                Z_c[mask] = 0.0

    W_c, b_c = _solve_centered_dual_ridge(Z_c, Y, alpha=alpha, scale_by_trace=scale_by_trace)
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
    """Computes plain, gated prompt-group centered, blended logits, and Sinkhorn-balanced probabilities."""
    logits_plain = Z @ plain_weight.T + plain_bias
    if groups is not None and len(groups) == len(Z):
        Z_c = Z.copy()
        singleton_mask = np.zeros(len(Z), dtype=bool)
        grps_arr = np.asarray(groups)
        plain_preds = logits_plain.argmax(axis=1)
        for g in np.unique(grps_arr):
            mask = grps_arr == g
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

    row_mean_p = logits_plain.mean(axis=1, keepdims=True)
    row_std_p = np.clip(logits_plain.std(axis=1, keepdims=True), 1e-4, None)
    norm_plain = (logits_plain - row_mean_p) / row_std_p

    row_mean_c = logits_centered.mean(axis=1, keepdims=True)
    row_std_c = np.clip(logits_centered.std(axis=1, keepdims=True), 1e-4, None)
    norm_centered = (logits_centered - row_mean_c) / row_std_c

    logits_blend = (1.0 - blend_centered) * norm_plain + blend_centered * norm_centered
    probs_plain_sinkhorn = apply_prior_and_sinkhorn(logits_plain)
    probs_sinkhorn = apply_prior_and_sinkhorn(logits_blend)

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
    """Executes GroupKFold CV on Stage 2 representations grouped by prompt_id to measure OOF BAcc & overfitting."""
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
        rep_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
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


def compute_detailed_metrics(logits: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    """Computes BAcc, NLL, Gemini 4-class metrics, sister pair accuracy, and per-class recall/precision."""
    preds = logits.argmax(axis=1)
    bacc = float(balanced_accuracy_score(y_true, preds))
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    nll = float(log_loss(y_true, probs, labels=list(range(K))))

    gem_idx = FAMILIES["gemini"]
    gem_mask = np.isin(y_true, gem_idx)
    gem_sub_preds = logits[gem_mask][:, gem_idx].argmax(axis=1)
    gem_y = np.array([gem_idx.index(int(v)) for v in y_true[gem_mask]])
    gem_within_bacc = float(balanced_accuracy_score(gem_y, gem_sub_preds))
    gem_ret = float(np.mean(np.isin(preds[gem_mask], gem_idx)))
    gem_cm = confusion_matrix(gem_y, gem_sub_preds, labels=list(range(len(gem_idx))))

    pair_mask = (y_true == 7) | (y_true == 8)
    pair_preds = np.where(logits[pair_mask, 7] >= logits[pair_mask, 8], 7, 8)
    sister_acc = float(balanced_accuracy_score(y_true[pair_mask], pair_preds)) if pair_mask.any() else 0.0

    cm = confusion_matrix(y_true, preds, labels=list(range(K)))
    per_class = []
    zero_recall = []
    for i, name in enumerate(LABELS):
        true_n = int(cm[i, :].sum())
        pred_n = int(cm[:, i].sum())
        correct_n = int(cm[i, i])
        if correct_n == 0:
            zero_recall.append(name)
        per_class.append({
            "idx": i,
            "model": name,
            "true": true_n,
            "preds": pred_n,
            "correct": correct_n,
            "recall": float(correct_n / max(true_n, 1)),
            "precision": float(correct_n / max(pred_n, 1)) if pred_n > 0 else 0.0,
            "bias": float(pred_n / max(true_n, 1)),
        })

    return {
        "bacc": bacc,
        "nll": nll,
        "gem_within_bacc": gem_within_bacc,
        "gem_retention": gem_ret,
        "sister_37_38_acc": sister_acc,
        "zero_recall_classes": zero_recall,
        "per_class": per_class,
        "gemini_4x4": gem_cm.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def print_prediction_table(metrics: dict[str, Any], header: str) -> None:
    """Prints a compact 18-class diagnostic breakdown table."""
    print(f"\n=== {header} ===")
    print(
        f"BAcc={metrics['bacc']*100:.2f}% | NLL={metrics['nll']:.4f} | "
        f"Gem4={metrics['gem_within_bacc']*100:.2f}% | GemRet={metrics['gem_retention']*100:.2f}% | "
        f"3.7v3.8={metrics['sister_37_38_acc']*100:.2f}% | ZeroRec={len(metrics['zero_recall_classes'])}"
    )
    print("  # | CLASS NAME               | PREDS | CORRECT | RECALL% |  PREC% |  BIAS")
    print("----+--------------------------+-------+---------+---------+--------+------")
    for row in metrics["per_class"]:
        print(
            f" {row['idx']:2d} | {row['model']:<24} | {row['preds']:5d} | {row['correct']:7d} | "
            f"{row['recall']*100:6.1f}% | {row['precision']*100:5.1f}% | {row['bias']:4.2f}x"
        )


@torch.no_grad()
def extract_features_and_logits(
    model: MultiLayerLoRAClassifier,
    texts: list[str],
    tok: Any,
    max_len: int,
    device: torch.device,
    batch_size: int = 64,
    drop_first_token: bool = True,
    compute_dtype: torch.dtype = torch.float16,
    end_id: int | None = None,
    cls_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Length-sorted evaluation using TextSequenceDataset head+tail truncation returning float32 Z and plain logits."""
    model.eval()
    n = len(texts)
    order = np.argsort([len(str(t)) for t in texts], kind="stable")
    feats = np.zeros((n, model.total_dim), dtype=np.float32)
    logits_out = np.zeros((n, model.num_classes), dtype=np.float32)
    tok_spec = getattr(model, "tok_spec", None)
    raw_cls_id = cls_id if cls_id is not None else (tok_spec.cls_id if tok_spec is not None else getattr(tok, "cls_token_id", None))
    eff_cls_id = int(raw_cls_id) if raw_cls_id is not None else None
    if end_id is not None:
        raw_end_id = end_id
    elif tok_spec is not None:
        raw_end_id = tok_spec.end_id
    else:
        tok_sep = getattr(tok, "sep_token_id", None)
        raw_end_id = tok_sep if tok_sep is not None else getattr(tok, "eos_token_id", None)
    eff_end_id = int(raw_end_id) if raw_end_id is not None else None
    if tok_spec is not None:
        pad_id = tok_spec.pad_id
    else:
        tok_pad = getattr(tok, "pad_token_id", None)
        pad_id = int(tok_pad) if tok_pad is not None else 0

    for s in range(0, n, batch_size):
        idx = order[s : s + batch_size]
        chunk = [str(texts[i]) for i in idx]
        ds_chunk = TextSequenceDataset(
            chunk,
            None,
            None,
            tok,
            max_len=max_len,
            end_id=eff_end_id,
            cls_id=eff_cls_id,
        )
        batch_items = [ds_chunk[k] for k in range(len(chunk))]
        enc = collate_text_batch(batch_items, pad_token_id=pad_id)
        mb_ids = enc["input_ids"].to(device)
        mb_mask = enc["attention_mask"].to(device)
        max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            Z = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=drop_first_token)
            lg = model.predict_plain(Z)
        feats[idx] = Z.float().cpu().numpy().astype(np.float32)
        logits_out[idx] = lg.float().cpu().numpy().astype(np.float32)

    return feats, logits_out


def _evaluate_bench_checkpoint(
    model: MultiLayerLoRAClassifier,
    tok: Any,
    texts: list[str],
    y_true: np.ndarray,
    max_len: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    drop_first_token: bool,
    tag: str,
    end_id: int | None = 151643,
    cls_id: int | None = None,
    pad_id: int = 151643,
    batch_size: int = 8,
    blend_centered: float = 0.0,
    out_dir: Path | None = None,
    history: list[dict[str, Any]] | None = None,
    eval_extras_out: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluates Bench-1800 (or subset) with train/inference tokenization parity and logs problem-class metrics."""
    ds_eval = TextSequenceDataset(texts, None, None, tok, max_len=max_len, end_id=end_id, cls_id=cls_id)
    collate_eval = functools.partial(collate_text_batch, pad_token_id=pad_id)
    loader_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)

    model.eval()
    z_parts = []
    logits_p_parts = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            for batch in loader_eval:
                b_ids = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                max_l = max(int(b_mask.sum(dim=1).max().item()), 2)
                z_b = model.forward_features(b_ids[:, :max_l], b_mask[:, :max_l], drop_first_token=drop_first_token)
                lg_b = model.predict_plain(z_b)
                z_parts.append(z_b.float().cpu().numpy())
                logits_p_parts.append(lg_b.float().cpu().numpy())

    Z_eval = np.vstack(z_parts)
    logits = np.vstack(logits_p_parts)
    if blend_centered > 0.0 or eval_extras_out is not None:
        w_p = model.plain_head.weight.detach().float().cpu().numpy()
        b_p = model.plain_head.bias.detach().float().cpu().numpy()
        w_c = model.centered_head.weight.detach().float().cpu().numpy()
        b_c = model.centered_head.bias.detach().float().cpu().numpy()
        eff_bc = blend_centered if blend_centered > 0.0 else 0.20
        res_grp = compute_grouped_inference_logits(Z_eval, w_p, b_p, w_c, b_c, groups=None, blend_centered=eff_bc)
        if eval_extras_out is not None:
            eval_extras_out["plain_sinkhorn"] = compute_detailed_metrics(np.log(np.clip(res_grp["probs_plain_sinkhorn"], 1e-8, 1.0)), y_true)
            eval_extras_out["blend_global_approx"] = compute_detailed_metrics(res_grp["logits_blend"], y_true)
            eval_extras_out["blend_global_approx_sinkhorn"] = compute_detailed_metrics(np.log(np.clip(res_grp["probs_sinkhorn"], 1e-8, 1.0)), y_true)
        if blend_centered > 0.0:
            logits = res_grp["logits_blend"]

    metrics = compute_detailed_metrics(logits, y_true)
    pc_map = {r["model"]: r for r in metrics["per_class"]}
    print(
        f"[Milestone {tag}] N={len(y_true)} | BAcc={metrics['bacc']*100:.2f}% | "
        f"NLL={metrics['nll']:.4f} | Gem4={metrics['gem_within_bacc']*100:.2f}% | "
        f"3.7v3.8={metrics['sister_37_38_acc']*100:.2f}%"
    )
    watch_parts = []
    for name in PROBLEM_WATCH_CLASSES:
        if name in pc_map:
            r = pc_map[name]
            f1 = (2.0 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-8)) if (r["precision"] + r["recall"]) > 0 else 0.0
            r["f1"] = float(f1)
            watch_parts.append(f"{name}:R{r['recall']*100:.0f}/P{r['precision']*100:.0f}/F{f1*100:.0f}({r['preds']})")
    if watch_parts:
        print("  [Watch Classes] " + " | ".join(watch_parts))

    entry = {
        "tag": tag,
        "n_samples": int(len(y_true)),
        "bacc": metrics["bacc"],
        "nll": metrics["nll"],
        "gem_within_bacc": metrics["gem_within_bacc"],
        "gem_retention": metrics["gem_retention"],
        "sister_37_38_acc": metrics["sister_37_38_acc"],
        "watch": {k: pc_map[k] for k in PROBLEM_WATCH_CLASSES if k in pc_map},
    }
    if history is not None:
        history.append(entry)
    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if history is not None:
                (out_dir / "milestone_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Trainer] Warning saving milestone history: {e}")
    model.train()
    return metrics, logits


def solve_trace_ridge_wb(
    Z_tr_np: np.ndarray,
    y_tr_np: np.ndarray,
    alphas: list[float],
    device: torch.device,
    label_smoothing: float = 0.05,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Canonical trace-scaled float64 closed-form Ridge solver (primal when N>=D else dual Gram)."""
    n_tr, dim = Z_tr_np.shape
    y_oh = np.zeros((n_tr, K), dtype=np.float32)
    y_oh[np.arange(n_tr), y_tr_np] = 1.0
    if label_smoothing > 0.0:
        y_oh = (1.0 - label_smoothing) * y_oh + (label_smoothing / float(K))

    X_tr = torch.from_numpy(Z_tr_np).to(device, dtype=torch.float32)
    Y_t = torch.from_numpy(y_oh).to(device, dtype=torch.float32)
    mu_x = X_tr.mean(dim=0, keepdim=True)
    mu_y = Y_t.mean(dim=0, keepdim=True)
    X_c = X_tr - mu_x
    Y_c = Y_t - mu_y
    trace_xx = float((X_c * X_c).sum().item())
    mean_eig = max(trace_xx / float(min(n_tr, dim)), 1e-8)

    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    if n_tr >= dim:
        X64, Y64 = X_c.double(), Y_c.double()
        xtx = X64.T @ X64
        xty = X64.T @ Y64
        evals, vecs = torch.linalg.eigh(xtx)
        evals = torch.clamp(evals, min=0.0)
        v_xty = vecs.T @ xty
        mu_x_f = mu_x.double()
        mu_y_f = mu_y.double()
        for a in alphas:
            inv_e = 1.0 / (evals + float(a) * mean_eig)
            W_t = vecs @ (inv_e.unsqueeze(1) * v_xty)
            b_t = (mu_y_f - mu_x_f @ W_t).squeeze(0)
            out[a] = (W_t.T.float().cpu().numpy().astype(np.float32), b_t.float().cpu().numpy().astype(np.float32))
    else:
        X64, Y64 = X_c.double(), Y_c.double()
        K_mat = X64 @ X64.T
        evals, vecs = torch.linalg.eigh(K_mat)
        evals = torch.clamp(evals, min=0.0)
        vt_y = vecs.T @ Y64
        x_t = X64.T
        mu_x_f = mu_x.double()
        mu_y_f = mu_y.double()
        for a in alphas:
            inv_k = 1.0 / (evals + float(a) * mean_eig)
            dual_w = vecs @ (inv_k.unsqueeze(1) * vt_y)
            W_t = x_t @ dual_w
            b_t = (mu_y_f - mu_x_f @ W_t).squeeze(0)
            out[a] = (W_t.T.float().cpu().numpy().astype(np.float32), b_t.float().cpu().numpy().astype(np.float32))
    return out


def gpu_trace_ridge_eval(
    Z_tr_np: np.ndarray,
    y_tr_np: np.ndarray,
    Z_te_np: np.ndarray,
    alphas: list[float],
    device: torch.device,
    label_smoothing: float = 0.05,
) -> dict[float, np.ndarray]:
    """Evaluates closed-form trace-scaled Ridge linear weights on Z_te_np across candidate alphas."""
    wb_map = solve_trace_ridge_wb(Z_tr_np, y_tr_np, alphas=alphas, device=device, label_smoothing=label_smoothing)
    X_te = torch.from_numpy(Z_te_np).to(device, dtype=torch.float32)
    return {a: (X_te @ torch.from_numpy(w).to(device).T + torch.from_numpy(b).to(device)).cpu().numpy() for a, (w, b) in wb_map.items()}
