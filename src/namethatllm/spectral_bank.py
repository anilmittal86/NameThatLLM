"""
Leak-Free Prior-Scaled 384-Dim Joint Spectral Bank
=================================================
Constructs full-rank cross-backbone spectral representations from unsupervised
prompt-centered principal components V_top64, scaled by out-of-sample prior scales.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

from .config import LABELS, K, align_cols, softmax, log_softmax
from .wpca_gaussian import split_oversized_clusters_dist

def project_spectral_block(Z: np.ndarray, res: dict, prior_scale: float) -> np.ndarray:
    """
    Projects a single representation matrix onto unsupervised within-prompt V_top64
    and scales by out-of-sample prior scale.
    """
    if len(Z) == 0:
        d_out = res["V_top64"].shape[1] if "V_top64" in res else 64
        return np.zeros((0, d_out), dtype=np.float32)
    mx, sx, V64 = res["mx"], res["sx"], res["V_top64"]
    S = (((Z - mx) / sx) @ V64).astype(np.float32)
    return (S / np.linalg.norm(S, axis=1, keepdims=True).clip(min=1e-6)) * float(prior_scale)


def make_spectral_blocks(Z_tr: np.ndarray, Z_te: np.ndarray, res: dict, prior_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper projecting train and test features onto V_top64.
    """
    return project_spectral_block(Z_tr, res, prior_scale), project_spectral_block(Z_te, res, prior_scale)


def fit_joint_spectral_bank(
    spectral_tr_blocks: List[np.ndarray],
    spectral_te_blocks: List[np.ndarray],
    y_sub: np.ndarray,
    prompts_sub: np.ndarray,
    D_tfidf: np.ndarray,
    D_neural_sum: np.ndarray,
    D_neural_weight_total: float,
    N_test: int,
    plain_C: float = 0.04,
    center_C: float = 0.03,
    center_blend: float = 0.35,
    max_cluster_size: int = 14,
    spectral_u_blocks: Optional[List[np.ndarray]] = None,
    D_tfidf_u: Optional[np.ndarray] = None,
    D_neural_sum_u: Optional[np.ndarray] = None,
    D_neural_weight_total_u: float = 0.0,
    N_unseen: int = 0,
    return_unseen: bool = False,
    save_path: Optional[Any] = None,
) -> Any:
    """
    Fits plain and prompt-centered logistic regressions on stacked unsupervised spectral features.
    Returns:
    - If return_unseen: (P_spectral, oof_bacc, P_spectral_u)
    - Otherwise: (P_spectral, oof_bacc)
    """
    if len(spectral_tr_blocks) == 0:
        p_te = np.ones((N_test, K), dtype=np.float64) / float(K)
        p_u = np.ones((N_unseen, K), dtype=np.float64) / float(K) if N_unseen > 0 else None
        return (p_te, 0.0, p_u) if return_unseen else (p_te, 0.0)

    X_spec_tr = np.hstack(spectral_tr_blocks)
    X_spec_te = np.hstack(spectral_te_blocks)

    # 5-fold GroupKFold OOF Balanced Accuracy (zero label leakage)
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y_sub), dtype=int)
    for tr_idx, val_idx in gkf.split(X_spec_tr, y_sub, groups=prompts_sub):
        clf_fold = LogisticRegression(C=plain_C, max_iter=2000, random_state=42)
        clf_fold.fit(X_spec_tr[tr_idx], y_sub[tr_idx])
        oof_preds[val_idx] = clf_fold.predict(X_spec_tr[val_idx])
    oof_bacc = float(balanced_accuracy_score(y_sub, oof_preds))

    # Plain Logistic Regression
    clf_plain = LogisticRegression(C=plain_C, max_iter=2000, random_state=42)
    clf_plain.fit(X_spec_tr, y_sub)
    lp_spec_plain = clf_plain.predict_log_proba(X_spec_te)
    if hasattr(clf_plain, 'classes_'):
        lp_spec_plain = align_cols(lp_spec_plain, [LABELS[c] for c in clf_plain.classes_])

    # Prompt-Centered Logistic Regression
    X_spec_tr_c = X_spec_tr.copy()
    for p in np.unique(prompts_sub):
        mask = (prompts_sub == p)
        X_spec_tr_c[mask] -= X_spec_tr_c[mask].mean(axis=0, keepdims=True)
    clf_center = LogisticRegression(C=center_C, max_iter=2000, random_state=42)
    clf_center.fit(X_spec_tr_c, y_sub)

    if save_path:
        spectral_params = {
            "W_plain": clf_plain.coef_,
            "b_plain": clf_plain.intercept_,
            "W_center": clf_center.coef_,
            "b_center": clf_center.intercept_,
            "center_blend": center_blend,
        }
        try:
            save_spectral_params(spectral_params, save_path)
            print(f"[Spectral Bank] Saved fitted spectral parameters to {save_path}")
        except Exception as e:
            print(f"[Spectral Bank] Warning: Could not save spectral params: {e}")

    # Prompt-centered test features
    X_spec_te_c = X_spec_te.copy()
    D_neural_final = D_neural_sum / max(1e-6, D_neural_weight_total) if D_neural_weight_total > 0 else D_tfidf
    D_hybrid_final = 0.55 * D_tfidf + 0.45 * D_neural_final
    g65_final = fcluster(linkage(squareform(D_hybrid_final, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
    safe_groups = split_oversized_clusters_dist(D_hybrid_final, g65_final, max_cluster_size=max_cluster_size)

    if N_test >= 2 and len(np.unique(safe_groups)) > 0:
        for g in np.unique(safe_groups):
            mask = (safe_groups == g)
            if mask.sum() > 1:
                X_spec_te_c[mask] -= X_spec_te_c[mask].mean(axis=0, keepdims=True)
            else:
                X_spec_te_c[mask] = 0.0
    else:
        X_spec_te_c = np.zeros_like(X_spec_te)

    lp_spec_center = clf_center.predict_log_proba(X_spec_te_c)
    if hasattr(clf_center, 'classes_'):
        lp_spec_center = align_cols(lp_spec_center, [LABELS[c] for c in clf_center.classes_])

    lp_spec_joint = lp_spec_plain + center_blend * lp_spec_center
    P_spectral = softmax(lp_spec_joint, axis=1)

    # Symmetrical evaluation on unseen holdout benchmark
    P_spectral_u = None
    if spectral_u_blocks and len(spectral_u_blocks) == len(spectral_tr_blocks) and N_unseen > 0:
        X_spec_u = np.hstack(spectral_u_blocks)
        lp_spec_u_plain = clf_plain.predict_log_proba(X_spec_u)
        if hasattr(clf_plain, 'classes_'):
            lp_spec_u_plain = align_cols(lp_spec_u_plain, [LABELS[c] for c in clf_plain.classes_])

        X_spec_u_c = X_spec_u.copy()
        D_neural_u = D_neural_sum_u / max(1e-6, D_neural_weight_total_u) if (D_neural_weight_total_u > 0 and D_neural_sum_u is not None) else (D_tfidf_u if D_tfidf_u is not None else np.zeros((N_unseen, N_unseen), dtype=np.float32))
        D_hybrid_u = 0.55 * (D_tfidf_u if D_tfidf_u is not None else D_neural_u) + 0.45 * D_neural_u
        if N_unseen >= 2 and D_hybrid_u.shape[0] == N_unseen:
            g65_u = fcluster(linkage(squareform(D_hybrid_u, checks=False), method='average'), t=0.65, criterion='distance')
            safe_groups_u = split_oversized_clusters_dist(D_hybrid_u, g65_u, max_cluster_size=max_cluster_size)
            for g in np.unique(safe_groups_u):
                mask = (safe_groups_u == g)
                if mask.sum() > 1:
                    X_spec_u_c[mask] -= X_spec_u_c[mask].mean(axis=0, keepdims=True)
                else:
                    X_spec_u_c[mask] = 0.0
        else:
            X_spec_u_c = np.zeros_like(X_spec_u)

        lp_spec_u_center = clf_center.predict_log_proba(X_spec_u_c)
        if hasattr(clf_center, 'classes_'):
            lp_spec_u_center = align_cols(lp_spec_u_center, [LABELS[c] for c in clf_center.classes_])

        lp_spec_u_joint = lp_spec_u_plain + center_blend * lp_spec_u_center
        P_spectral_u = softmax(lp_spec_u_joint, axis=1)

    if return_unseen:
        return P_spectral, oof_bacc, P_spectral_u
    return P_spectral, oof_bacc


def save_spectral_params(prm: dict, path: Any):
    """Serializes fitted Spectral Bank linear heads to safetensors."""
    from pathlib import Path
    import torch
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {
        "W_plain": torch.from_numpy(np.asarray(prm["W_plain"], dtype=np.float32)),
        "b_plain": torch.from_numpy(np.asarray(prm["b_plain"], dtype=np.float32)),
        "W_center": torch.from_numpy(np.asarray(prm["W_center"], dtype=np.float32)),
        "b_center": torch.from_numpy(np.asarray(prm["b_center"], dtype=np.float32)),
        "center_blend": torch.tensor([float(prm.get("center_blend", 0.35))], dtype=torch.float32),
    }
    save_file(sd, str(path))


def load_spectral_params(path: Any) -> dict:
    """Loads fitted Spectral Bank parameters from safetensors."""
    from safetensors.torch import load_file

    sd = load_file(str(path))
    return {
        "W_plain": sd["W_plain"].cpu().numpy(),
        "b_plain": sd["b_plain"].cpu().numpy(),
        "W_center": sd["W_center"].cpu().numpy(),
        "b_center": sd["b_center"].cpu().numpy(),
        "center_blend": float(sd["center_blend"].cpu().numpy()[0]),
        "labels": list(LABELS),
    }


def train_and_save_spectral_bank(
    spectral_tr_blocks: List[np.ndarray],
    y_sub: np.ndarray,
    prompts_sub: np.ndarray,
    output_path: Any,
    plain_C: float = 0.04,
    center_C: float = 0.03,
    center_blend: float = 0.35,
) -> dict:
    """
    Fits plain and prompt-centered LogisticRegression heads on stacked spectral training blocks
    and serializes the weights to safetensors. Zero GPU dependency.
    """
    X_spec_tr = np.hstack(spectral_tr_blocks)

    clf_plain = LogisticRegression(C=plain_C, max_iter=2000, random_state=42)
    clf_plain.fit(X_spec_tr, y_sub)
    W_p = clf_plain.coef_
    b_p = clf_plain.intercept_
    if hasattr(clf_plain, 'classes_'):
        classes = clf_plain.classes_
        if len(classes) < K:
            full_W = np.zeros((K, X_spec_tr.shape[1]), dtype=np.float32)
            full_b = np.zeros(K, dtype=np.float32)
            for idx, c in enumerate(classes):
                full_W[c] = W_p[idx]
                full_b[c] = b_p[idx]
            W_p, b_p = full_W, full_b

    X_spec_tr_c = X_spec_tr.copy()
    for p in np.unique(prompts_sub):
        mask = (prompts_sub == p)
        X_spec_tr_c[mask] -= X_spec_tr_c[mask].mean(axis=0, keepdims=True)
    clf_center = LogisticRegression(C=center_C, max_iter=2000, random_state=42)
    clf_center.fit(X_spec_tr_c, y_sub)

    W_c = clf_center.coef_
    b_c = clf_center.intercept_
    if hasattr(clf_center, 'classes_'):
        classes = clf_center.classes_
        if len(classes) < K:
            full_W = np.zeros((K, X_spec_tr.shape[1]), dtype=np.float32)
            full_b = np.zeros(K, dtype=np.float32)
            for idx, c in enumerate(classes):
                full_W[c] = W_c[idx]
                full_b[c] = b_c[idx]
            W_c, b_c = full_W, full_b

    prm = {
        "W_plain": W_p.astype(np.float32),
        "b_plain": b_p.astype(np.float32),
        "W_center": W_c.astype(np.float32),
        "b_center": b_c.astype(np.float32),
        "center_blend": center_blend,
        "labels": list(LABELS),
    }
    save_spectral_params(prm, output_path)
    print(f"[Spectral Bank Trainer] Saved fitted spectral heads to {output_path}")
    return prm


def predict_spectral_bank(
    spectral_blocks: List[np.ndarray],
    prm: dict,
    D_tfidf: np.ndarray,
    D_neural_sum: Optional[np.ndarray] = None,
    D_neural_weight_total: float = 0.0,
    max_cluster_size: int = 14,
) -> np.ndarray:
    """
    Pure inference evaluation of Spectral Bank using pre-trained weights.
    Zero training data required; executes in <1 ms on CPU.
    """
    if len(spectral_blocks) == 0:
        N = D_tfidf.shape[0] if D_tfidf is not None else 0
        return np.ones((N, K), dtype=np.float32) / float(K)

    X_spec = np.hstack(spectral_blocks)
    N = X_spec.shape[0]

    W_p, b_p = prm["W_plain"], prm["b_plain"]
    W_c, b_c = prm["W_center"], prm["b_center"]
    blend = float(prm.get("center_blend", 0.35))
    labels = prm.get("labels", list(LABELS))

    lp_plain = log_softmax(align_cols(X_spec @ W_p.T + b_p, labels))

    X_spec_c = X_spec.copy()
    D_neural = D_neural_sum / max(1e-6, D_neural_weight_total) if (D_neural_weight_total > 0 and D_neural_sum is not None) else D_tfidf
    D_hybrid = 0.55 * (D_tfidf if D_tfidf is not None else D_neural) + 0.45 * D_neural
    if N >= 2 and D_hybrid is not None and D_hybrid.shape[0] == N:
        g65 = fcluster(linkage(squareform(D_hybrid, checks=False), method='average'), t=0.65, criterion='distance')
        safe_groups = split_oversized_clusters_dist(D_hybrid, g65, max_cluster_size=max_cluster_size)
        for g in np.unique(safe_groups):
            mask = (safe_groups == g)
            if mask.sum() > 1:
                X_spec_c[mask] -= X_spec_c[mask].mean(axis=0, keepdims=True)
            else:
                X_spec_c[mask] = 0.0
    else:
        X_spec_c = np.zeros_like(X_spec)

    lp_center = log_softmax(align_cols(X_spec_c @ W_c.T + b_c, labels))
    return softmax(lp_plain + blend * lp_center, axis=1)


