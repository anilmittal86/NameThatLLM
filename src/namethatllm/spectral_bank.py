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

from .config import LABELS, K, align_cols, softmax
from .wpca_gaussian import split_oversized_clusters_dist

def make_spectral_blocks(Z_tr: np.ndarray, Z_te: np.ndarray, res: dict, prior_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects train and test features onto V_top64 and scales by out-of-sample prior scale.
    """
    mx, sx, V64 = res["mx"], res["sx"], res["V_top64"]
    S_tr = (((Z_tr - mx) / sx) @ V64).astype(np.float32)
    S_te = (((Z_te - mx) / sx) @ V64).astype(np.float32)
    S_tr = (S_tr / np.linalg.norm(S_tr, axis=1, keepdims=True).clip(min=1e-6)) * float(prior_scale)
    S_te = (S_te / np.linalg.norm(S_te, axis=1, keepdims=True).clip(min=1e-6)) * float(prior_scale)
    return S_tr, S_te

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
    max_cluster_size: int = 14
) -> Tuple[np.ndarray, float]:
    """
    Fits plain and prompt-centered logistic regressions on stacked unsupervised spectral features.
    Returns:
    - P_spectral: (N_test, K) predicted probabilities
    - oof_bacc: 5-fold GroupKFold balanced accuracy
    """
    if len(spectral_tr_blocks) == 0:
        return np.ones((N_test, K), dtype=np.float64) / float(K), 0.0

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

    return P_spectral, oof_bacc

