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

def fit_within_family_spectral_gates(
    spectral_tr_blocks: List[np.ndarray],
    spectral_te_blocks: List[np.ndarray],
    y_sub: np.ndarray,
    prompts_sub: np.ndarray,
    D_tfidf: np.ndarray,
    D_neural_sum: np.ndarray,
    D_neural_weight_total: float,
    N_test: int,
    max_cluster_size: int = 14,
) -> Dict[str, np.ndarray]:
    """
    Fits plain + prompt-centered LogisticRegression sibling specialist gates
    on the stacked cross-backbone spectral features for each provider family
    ('gpt', 'gemini', 'claude', 'glm').
    Returns dict mapping family_name -> (N_test, len(fam_cols)) log-probabilities.
    """
    from .config import SPECIALIST_GROUPS, FAMILIES, log_softmax

    if len(spectral_tr_blocks) == 0 or N_test == 0:
        return {}
    X_tr = np.hstack(spectral_tr_blocks)
    X_te = np.hstack(spectral_te_blocks)

    X_tr_c = X_tr.copy()
    for p in np.unique(prompts_sub):
        m = (prompts_sub == p)
        if m.sum() > 1:
            X_tr_c[m] -= X_tr_c[m].mean(axis=0, keepdims=True)

    X_te_c = X_te.copy()
    D_neural_final = D_neural_sum / max(1e-6, D_neural_weight_total) if D_neural_weight_total > 0 else D_tfidf
    D_hybrid_final = 0.55 * D_tfidf + 0.45 * D_neural_final
    g65 = fcluster(linkage(squareform(D_hybrid_final, checks=False), method="average"), t=0.65, criterion="distance") if N_test >= 2 else np.ones(N_test, dtype=int)
    safe_groups = split_oversized_clusters_dist(D_hybrid_final, g65, max_cluster_size=max_cluster_size)
    if N_test >= 2:
        for g in np.unique(safe_groups):
            m = (safe_groups == g)
            if m.sum() > 1:
                X_te_c[m] -= X_te_c[m].mean(axis=0, keepdims=True)
            else:
                X_te_c[m] = 0.0
    else:
        X_te_c = np.zeros_like(X_te)

    groups_dict = SPECIALIST_GROUPS if SPECIALIST_GROUPS else FAMILIES
    family_gates: Dict[str, np.ndarray] = {}
    for fam_name, fam_cols in groups_dict.items():
        mask_tr = np.isin(y_sub, fam_cols)
        min_samples = 20 if len(fam_cols) <= 3 else 30
        if mask_tr.sum() >= min_samples and len(np.unique(y_sub[mask_tr])) == len(fam_cols):
            sub_y = np.array([fam_cols.index(int(c)) for c in y_sub[mask_tr]], dtype=int)
            clf_p = LogisticRegression(C=0.10, max_iter=1000, class_weight="balanced", random_state=42)
            clf_p.fit(X_tr[mask_tr], sub_y)
            lp_p = clf_p.predict_log_proba(X_te)

            clf_c = LogisticRegression(C=0.06, max_iter=1000, class_weight="balanced", random_state=42)
            clf_c.fit(X_tr_c[mask_tr], sub_y)
            lp_c = clf_c.predict_log_proba(X_te_c)

            family_gates[fam_name] = log_softmax(0.65 * lp_p + 0.35 * lp_c, axis=1)
    return family_gates


