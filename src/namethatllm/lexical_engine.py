"""
Subprocess Dual-Output Lexical Engine
====================================
Subprocess memory isolation running 78,482-row 4-View LinearSVC (Output 1)
and 5-View BPE + SGD + Style Twin + Specialist Complementary Ensemble (Output 2).
"""

import os
import gc
import re
import sys
import time
import hashlib
import unicodedata
import collections
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

FAMILIES = {
    "gpt": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
}

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def _text_hash(text: str) -> str:
    """Normalized SHA1 hash of whitespace-collapsed text to guarantee zero exact/near-exact leakage."""
    norm = " ".join(str(text).strip().split())
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()

def _norm_text_key(value: Any) -> str:
    """Normalized NFKC string key for exact deduplication."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value or ''))).strip()

FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose "
    "this these those it its they them their theirs he him his she her hers we us our ours "
    "you your yours i me my mine in on at to from with without for of by over under between "
    "through into about is are was were be been being have has had do does did can could "
    "will would shall should may might must not no nor all any some each every both either "
    "neither other another such only also just very too more most less least here there "
    "when where why how what whether then now still yet even much many few little "
    "am having s t re ve ll d m don doesn didn isn aren wasn weren hasn haven hadn won wouldn couldn shouldn".split()
)
STOP_WORDS = FUNCTION_WORDS

def shape_text(t: str) -> str:
    """Replaces alphabetic character blocks with X (uppercase) and x (lowercase)."""
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m.group(0)[0].isupper() else "x", str(t))

def function_text(t: str) -> str:
    """Masks non-stopwords and non-punctuation to CONTENT, preserving function and punctuation syntax."""
    tokens = re.findall(r"(?u)\b\w+\b|[^\w\s]", str(t))
    return " ".join([w.lower() if (w.lower() in STOP_WORDS or not w.isalnum()) else "CONTENT" for w in tokens])

_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
_RE_STRAIGHT_CONTR = re.compile(r"[A-Za-z]'[A-Za-z]")
_RE_CURLY_CONTR = re.compile(r"[A-Za-z]\u2019[A-Za-z]")

def extract_handwriting_and_orthographic_features(texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-pass extractor returning:
    - feats: (N, 26) float32 continuous punctuation, contraction, and structural densities
    - rules: (N, 4) float32 binary indicators for the 4 discrete orthographic invariants:
        col 0: any curly quote (\u2018, \u2019, \u201c, \u201d)
        col 1: >= 2 em-dashes (\u2014) in non-code prose
        col 2: >= 2 straight contractions [A-Za-z]'[A-Za-z] in non-code prose AND 0 curly \u2019
        col 3: >= 1 curly contraction [A-Za-z]\u2019[A-Za-z] in non-code prose
    """
    N = len(texts)
    feats = np.zeros((N, 26), dtype=np.float32)
    rules = np.zeros((N, 4), dtype=np.float32)

    for i, raw_t in enumerate(texts):
        t = str(raw_t)
        prose = _RE_CODE_BLOCK.sub(" ", t)
        t_len = max(len(t), 1)
        words = t.split()
        n_words = max(len(words), 1)

        raw_curly_sq = t.count("\u2018") + t.count("\u2019")
        raw_curly_dq = t.count("\u201c") + t.count("\u201d")
        prose_em_dash = prose.count("\u2014")
        n_str_c = len(_RE_STRAIGHT_CONTR.findall(prose))
        n_cur_c = len(_RE_CURLY_CONTR.findall(prose))

        # 4 Discrete Structural Rule Flags
        rules[i, 0] = 1.0 if (raw_curly_sq + raw_curly_dq) > 0 else 0.0
        rules[i, 1] = 1.0 if prose_em_dash >= 2 else 0.0
        rules[i, 2] = 1.0 if (n_str_c >= 2 and raw_curly_sq == 0) else 0.0
        rules[i, 3] = 1.0 if n_cur_c >= 1 else 0.0

        # 26 Continuous Density Features
        c_colon = t.count(":") / n_words
        c_semi = t.count(";") / n_words
        c_em_dash = t.count("\u2014") / n_words
        c_double_hyph = t.count("--") / n_words
        c_curly_sq = raw_curly_sq / n_words
        c_straight_sq = t.count("'") / n_words
        c_curly_dq = raw_curly_dq / n_words
        c_straight_dq = t.count('"') / n_words
        c_paren = (t.count("(") + t.count(")")) / n_words
        c_comma = t.count(",") / n_words
        c_excl = t.count("!") / n_words
        c_ques = t.count("?") / n_words
        c_ellipsis = (t.count("...") + t.count("\u2026")) / n_words
        c_hyphen = t.count("-") / n_words
        n_straight_contr = n_str_c / n_words
        n_curly_contr = n_cur_c / n_words

        c_newline = t.count("\n") / n_words
        c_bullet = (t.count("\n- ") + t.count("\n* ")) / n_words
        c_upper = sum(1 for ch in t if ch.isupper()) / t_len
        c_digits = sum(1 for ch in t if ch.isdigit()) / t_len

        w_lens = [len(w) for w in words]
        mean_w_len = float(np.mean(w_lens)) if w_lens else 0.0
        std_w_len = float(np.std(w_lens)) if len(w_lens) > 1 else 0.0

        sents = [s.strip() for s in t.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        s_lens = [len(s.split()) for s in sents]
        mean_s_len = float(np.mean(s_lens)) if s_lens else 0.0
        std_s_len = float(np.std(s_lens)) if len(s_lens) > 1 else 0.0
        burstiness = float(std_s_len / max(mean_s_len, 1e-5))
        ttr = float(len(set(words)) / n_words)

        feats[i] = [
            c_colon, c_semi, c_em_dash, c_double_hyph, c_curly_sq, c_straight_sq,
            c_curly_dq, c_straight_dq, c_paren, c_comma, c_excl, c_ques,
            c_ellipsis, c_hyphen, n_straight_contr, n_curly_contr, c_newline,
            c_bullet, c_upper, c_digits, mean_w_len, std_w_len, mean_s_len,
            std_s_len, burstiness, ttr
        ]
    return feats, rules

def extract_handwriting_features(texts: List[str]) -> np.ndarray:
    """Backward-compatible wrapper returning only the (N, 26) continuous matrix."""
    feats, _ = extract_handwriting_and_orthographic_features(texts)
    return feats

def fit_handwriting_model(
    hw_tr: np.ndarray,
    fit_y: np.ndarray,
    hw_te: np.ndarray,
    df_tr: pd.DataFrame,
    groups_085: np.ndarray,
    twin_rows_085: np.ndarray,
    len_tr_raw: int
) -> np.ndarray:
    """
    Fits dedicated LogisticRegression plain (C=0.25) and prompt-centered (C=0.15)
    classifiers strictly on the 26 continuous handwriting features.
    Returns log_p_handwriting (N_test, 18).
    """
    clf_hw_plain = LogisticRegression(C=0.25, max_iter=1000, random_state=42, class_weight="balanced")
    clf_hw_plain.fit(hw_tr, fit_y)
    lp_hw = log_softmax(clf_hw_plain.decision_function(hw_te))

    if twin_rows_085.any() and len_tr_raw >= 18:
        hw_tr_c = hw_tr[:len_tr_raw].copy()
        for p in np.unique(df_tr["prompt_id"].values):
            m_p = (df_tr["prompt_id"].values == p)
            hw_tr_c[m_p] -= hw_tr_c[m_p].mean(axis=0, keepdims=True)
        clf_hw_twin = LogisticRegression(C=0.15, max_iter=1000, random_state=42, class_weight="balanced")
        clf_hw_twin.fit(hw_tr_c, fit_y[:len_tr_raw])

        hw_te_c = hw_te.copy()
        for g in np.unique(groups_085[twin_rows_085]):
            m_g = (groups_085 == g)
            hw_te_c[m_g] -= hw_te_c[m_g].mean(axis=0, keepdims=True)
        lp_hw_twin = log_softmax(clf_hw_twin.decision_function(hw_te_c[twin_rows_085]))
        lp_hw[twin_rows_085] = log_softmax(0.70 * lp_hw[twin_rows_085] + 0.30 * lp_hw_twin)

    return lp_hw

def train_lexical_engine(
    fit_texts: List[str],
    fit_y: np.ndarray,
    fit_sw: Optional[np.ndarray] = None,
    df_tr: Optional[pd.DataFrame] = None,
    qwen_tok_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains all TF-IDF vectorizers, linear classifiers, handwriting density models,
    and specialist refiners offline on train + synthetic pool data.
    Returns a self-contained bundle dictionary that can be saved with save_lexical_bundle.
    """
    if fit_sw is None:
        fit_sw = np.ones(len(fit_texts), dtype=np.float32)
    len_tr_raw = len(df_tr) if df_tr is not None else 0

    VIEWS_4V = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("function", 1, 3, 50000, 0.5),
    ]

    fit_sh = [shape_text(t) for t in fit_texts]
    fit_fn = [function_text(t) for t in fit_texts]

    vectorizers_4v = []
    tr_mats_4v = []
    for mode, mn, mx, mf, sc in VIEWS_4V:
        analyzer = "char" if mode == "shape" else ("word" if mode == "function" else mode)
        opts = dict(
            analyzer=analyzer,
            ngram_range=(mn, mx),
            lowercase=False,
            sublinear_tf=True,
            min_df=2 if len(fit_texts) >= 50 else 1,
            max_features=mf,
            dtype=np.float32,
        )
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"

        v = TfidfVectorizer(**opts)
        if mode == "shape":
            tr_m = v.fit_transform(fit_sh) * np.float32(sc)
        elif mode == "function":
            tr_m = v.fit_transform(fit_fn) * np.float32(sc)
        else:
            tr_m = v.fit_transform(fit_texts) * np.float32(sc)
        vectorizers_4v.append(v)
        tr_mats_4v.append(tr_m)

    X_tr_4v = sparse.hstack(tr_mats_4v, format="csr")

    # OUTPUT 1: Pure 4-View LinearSVC (C=0.5, class_weight='balanced', source sample weights)
    svc_main = LinearSVC(C=0.5, random_state=42, max_iter=5000, tol=1e-3, class_weight="balanced")
    svc_main.fit(X_tr_4v, fit_y, sample_weight=fit_sw)

    # OUTPUT 2: Stamatatos Text Distortion Model (keeps top-100 words, masks rest with '*')
    wre = re.compile(r"[A-Za-z’']+")
    cnt = collections.Counter(w.lower() for t in fit_texts for w in wre.findall(t))
    distort_keep = {w for w, _ in cnt.most_common(100)}
    dist_fn = lambda arr: [wre.sub(lambda m_: m_.group(0) if m_.group(0).lower() in distort_keep else '*', t) for t in arr]

    D_tr = dist_fn(fit_texts)
    vs_dist = [
        TfidfVectorizer(sublinear_tf=True, lowercase=False, analyzer='char', ngram_range=(2, 6), min_df=2 if len(fit_texts) >= 50 else 1),
        TfidfVectorizer(sublinear_tf=True, lowercase=True, token_pattern=r"[^\s]+", ngram_range=(1, 3), min_df=2 if len(fit_texts) >= 50 else 1)
    ]
    Xa_dist = sparse.hstack([v_.fit_transform(D_tr) for v_ in vs_dist], format="csr")

    lr_distort = LogisticRegression(C=10.0, max_iter=3000, random_state=42)
    lr_distort.fit(Xa_dist, fit_y, sample_weight=fit_sw)

    # OUTPUT 3: Continuous Handwriting Density Model
    hw_tr_raw, _ = extract_handwriting_and_orthographic_features(fit_texts)
    scaler = StandardScaler()
    hw_tr = np.clip(scaler.fit_transform(hw_tr_raw), -5.0, 5.0).astype(np.float32)

    clf_hw_plain = LogisticRegression(C=0.25, max_iter=1000, random_state=42, class_weight="balanced")
    clf_hw_plain.fit(hw_tr, fit_y)

    clf_hw_twin = None
    if df_tr is not None and len_tr_raw >= 18 and "prompt_id" in df_tr.columns:
        hw_tr_c = hw_tr[:len_tr_raw].copy()
        for p in np.unique(df_tr["prompt_id"].values):
            m_p = (df_tr["prompt_id"].values == p)
            hw_tr_c[m_p] -= hw_tr_c[m_p].mean(axis=0, keepdims=True)
        clf_hw_twin = LogisticRegression(C=0.15, max_iter=1000, random_state=42, class_weight="balanced")
        clf_hw_twin.fit(hw_tr_c, fit_y[:len_tr_raw])

    # 5th View: BPE or Character-n-gram
    tok = None
    if qwen_tok_dir and os.path.exists(qwen_tok_dir):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(qwen_tok_dir, use_fast=True)
        except Exception:
            tok = None

    if tok is not None:
        def batch_bpe_strings(texts: List[str], batch_sz: int = 4000) -> List[str]:
            out_strs = []
            for s in range(0, len(texts), batch_sz):
                batch_ids = tok(
                    texts[s : s + batch_sz],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512,
                )["input_ids"]
                out_strs.extend(" ".join(map(str, ids)) for ids in batch_ids)
            return out_strs

        fit_bpe_strs = batch_bpe_strings(fit_texts)
        v_5th = TfidfVectorizer(
            analyzer="word", token_pattern=r"\S+", lowercase=False,
            ngram_range=(1, 3), max_features=75000, sublinear_tf=True, dtype=np.float32
        )
        tr_5th = v_5th.fit_transform(fit_bpe_strs) * 0.8
        is_bpe = True
        del fit_bpe_strs
        gc.collect()
    else:
        v_5th = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=75000, sublinear_tf=True, dtype=np.float32)
        tr_5th = v_5th.fit_transform(fit_texts) * 0.8
        is_bpe = False

    X_tr_5v = sparse.hstack([X_tr_4v, tr_5th], format="csr")

    svc_comp = LinearSVC(C=0.35, random_state=42, max_iter=3000, tol=1e-3, class_weight="balanced")
    svc_comp.fit(X_tr_5v, fit_y, sample_weight=fit_sw)

    sgd_comp = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1500, random_state=42)
    sgd_comp.fit(X_tr_5v, fit_y, sample_weight=fit_sw)

    svc_twin = None
    if df_tr is not None and len_tr_raw >= 18 and "prompt_id" in df_tr.columns:
        X_train_only = X_tr_5v[:len_tr_raw]
        u_tr, inv_tr = np.unique(df_tr['prompt_id'].values, return_inverse=True)
        counts_tr = np.bincount(inv_tr)
        vals_tr = (1.0 / counts_tr[inv_tr]).astype(np.float32)
        G = sparse.csr_matrix((vals_tr, (np.arange(len(inv_tr)), inv_tr)), shape=(len(inv_tr), len(u_tr)), dtype=np.float32)
        X_train_c = X_train_only - G @ (G.T @ X_train_only)

        svc_twin = LinearSVC(C=0.30, random_state=42, max_iter=2500)
        svc_twin.fit(X_train_c, fit_y[:len_tr_raw])
        del X_train_only, G, X_train_c
        gc.collect()

    specialist_svcs = {}
    for fam_name, fam_cols in FAMILIES.items():
        fam_mask = np.isin(fit_y, fam_cols)
        if fam_mask.sum() > 50:
            sub_y = fit_y[fam_mask]
            sub_lab2i = {c: i for i, c in enumerate(fam_cols)}
            sub_y_mapped = np.array([sub_lab2i[y] for y in sub_y])

            spec_svc = LinearSVC(C=0.40, random_state=42, max_iter=2000, tol=1e-3, class_weight="balanced")
            spec_svc.fit(X_tr_5v[fam_mask], sub_y_mapped)
            specialist_svcs[fam_name] = spec_svc

    bundle = {
        "views_4v_specs": VIEWS_4V,
        "vectorizers_4v": vectorizers_4v,
        "svc_main": svc_main,
        "distort_keep": sorted(list(distort_keep)),
        "vs_dist": vs_dist,
        "lr_distort": lr_distort,
        "hw_scaler": scaler,
        "clf_hw_plain": clf_hw_plain,
        "clf_hw_twin": clf_hw_twin,
        "is_bpe": is_bpe,
        "v_5th": v_5th,
        "svc_comp": svc_comp,
        "sgd_comp": sgd_comp,
        "svc_twin": svc_twin,
        "specialist_svcs": specialist_svcs,
        "families": FAMILIES,
        "labels": LABELS,
    }
    return bundle

def save_lexical_bundle(bundle: Dict[str, Any], path: Path | str) -> None:
    """Serializes the lexical bundle to disk using compressed joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    sz_mb = path.stat().st_size / (1024 * 1024)
    print(f"[Lexical] Saved lexical bundle to {path} ({sz_mb:.2f} MB)")

def load_lexical_bundle(path: Path | str) -> Dict[str, Any]:
    """Loads a pre-trained lexical bundle from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexical bundle not found at {path}")
    t0 = time.time()
    bundle = joblib.load(path)
    print(f"[Lexical] Loaded lexical bundle from {path} in {time.time() - t0:.2f}s")
    return bundle

def predict_lexical_engine(
    bundle: Dict[str, Any],
    test_texts: List[str],
    unseen_texts: Optional[List[str]] = None,
    qwen_tok_dir: Optional[str] = None,
    working_dir: Optional[Path | str] = None,
) -> Dict[str, np.ndarray]:
    """
    Executes fast inference using a pre-trained lexical bundle on test_texts and optional unseen_texts.
    Takes ~10-20 seconds on CPU.
    """
    t0 = time.time()
    unseen_texts = unseen_texts or []

    # 1. Compute stopword TF-IDF distance matrix and topic clusters (t=0.85) on test
    if len(test_texts) >= 2:
        Xt_tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100000, dtype=np.float32).fit_transform(test_texts)
        D = np.clip(1.0 - (Xt_tfidf @ Xt_tfidf.T).toarray(), 0.0, None)
        np.fill_diagonal(D, 0.0)
        cond_d = squareform(D, checks=False)
        groups_085 = fcluster(linkage(cond_d, method='average'), 0.85, criterion='distance')
    else:
        D = np.zeros((len(test_texts), len(test_texts)), dtype=np.float32)
        groups_085 = np.ones(len(test_texts), dtype=int)

    sz_085 = np.bincount(groups_085)[groups_085] if len(groups_085) > 0 else np.array([], dtype=int)
    twin_rows_085 = (sz_085 >= 2) & (sz_085 <= 18) if len(sz_085) > 0 else np.zeros(0, dtype=bool)

    # Stopword TF-IDF and clusters on unseen holdout benchmark
    if len(unseen_texts) >= 2:
        Xu_tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100000, dtype=np.float32).fit_transform(unseen_texts)
        D_u = np.clip(1.0 - (Xu_tfidf @ Xu_tfidf.T).toarray(), 0.0, None)
        np.fill_diagonal(D_u, 0.0)
        cond_du = squareform(D_u, checks=False)
        groups_085_u = fcluster(linkage(cond_du, method='average'), 0.85, criterion='distance')
    else:
        D_u = np.zeros((len(unseen_texts), len(unseen_texts)), dtype=np.float32)
        groups_085_u = np.ones(len(unseen_texts), dtype=int)
    sz_085_u = np.bincount(groups_085_u)[groups_085_u] if len(groups_085_u) > 0 else np.array([], dtype=int)
    twin_rows_085_u = (sz_085_u >= 2) & (sz_085_u <= 18) if len(sz_085_u) > 0 else np.zeros(0, dtype=bool)

    # 2. 4-View Transform
    test_sh = [shape_text(t) for t in test_texts]
    unseen_sh = [shape_text(t) for t in unseen_texts] if unseen_texts else []
    test_fn = [function_text(t) for t in test_texts]
    unseen_fn = [function_text(t) for t in unseen_texts] if unseen_texts else []

    te_mats_4v = []
    unseen_mats_4v = []
    for (mode, mn, mx, mf, sc), v in zip(bundle["views_4v_specs"], bundle["vectorizers_4v"]):
        if mode == "shape":
            te_m = v.transform(test_sh) * np.float32(sc)
            u_m = v.transform(unseen_sh) * np.float32(sc) if unseen_texts else None
        elif mode == "function":
            te_m = v.transform(test_fn) * np.float32(sc)
            u_m = v.transform(unseen_fn) * np.float32(sc) if unseen_texts else None
        else:
            te_m = v.transform(test_texts) * np.float32(sc)
            u_m = v.transform(unseen_texts) * np.float32(sc) if unseen_texts else None
        te_mats_4v.append(te_m)
        if u_m is not None:
            unseen_mats_4v.append(u_m)

    X_te_4v = sparse.hstack(te_mats_4v, format="csr")
    X_unseen_4v = sparse.hstack(unseen_mats_4v, format="csr") if unseen_mats_4v else None

    # OUTPUT 1: Pure 4-View LinearSVC
    svc_main = bundle["svc_main"]
    logits_svc_main = svc_main.decision_function(X_te_4v)
    log_p_svc_main = log_softmax(logits_svc_main)

    if X_unseen_4v is not None and X_unseen_4v.shape[0] > 0:
        unseen_log_p_svc_main = log_softmax(svc_main.decision_function(X_unseen_4v))
    else:
        unseen_log_p_svc_main = np.zeros((0, K), dtype=np.float32)

    # OUTPUT 2: Text Distortion Model
    wre = re.compile(r"[A-Za-z’']+")
    distort_keep = set(bundle["distort_keep"])
    dist_fn = lambda arr: [wre.sub(lambda m_: m_.group(0) if m_.group(0).lower() in distort_keep else '*', t) for t in arr]

    D_te = dist_fn(test_texts)
    D_unseen = dist_fn(unseen_texts) if unseen_texts else []
    vs_dist = bundle["vs_dist"]
    lr_distort = bundle["lr_distort"]

    Xb_dist = sparse.hstack([v_.transform(D_te) for v_ in vs_dist], format="csr")
    Xu_dist = sparse.hstack([v_.transform(D_unseen) for v_ in vs_dist], format="csr") if unseen_texts else None

    P_distort = lr_distort.predict_proba(Xb_dist)
    log_p_distort = lr_distort.predict_log_proba(Xb_dist)
    if Xu_dist is not None and Xu_dist.shape[0] > 0:
        P_unseen_distort = lr_distort.predict_proba(Xu_dist)
        log_p_unseen_distort = lr_distort.predict_log_proba(Xu_dist)
    else:
        P_unseen_distort = None
        log_p_unseen_distort = None

    # OUTPUT 3: Continuous Handwriting & Orthographic Features
    scaler = bundle["hw_scaler"]
    clf_hw_plain = bundle["clf_hw_plain"]
    clf_hw_twin = bundle.get("clf_hw_twin")

    hw_te_raw, rule_flags_te = extract_handwriting_and_orthographic_features(test_texts)
    hw_te = np.clip(scaler.transform(hw_te_raw), -5.0, 5.0).astype(np.float32)
    lp_hw = log_softmax(clf_hw_plain.decision_function(hw_te))

    if clf_hw_twin is not None and twin_rows_085.any():
        hw_te_c = hw_te.copy()
        for g in np.unique(groups_085[twin_rows_085]):
            m_g = (groups_085 == g)
            hw_te_c[m_g] -= hw_te_c[m_g].mean(axis=0, keepdims=True)
        lp_hw_twin = log_softmax(clf_hw_twin.decision_function(hw_te_c[twin_rows_085]))
        lp_hw[twin_rows_085] = log_softmax(0.70 * lp_hw[twin_rows_085] + 0.30 * lp_hw_twin)

    if unseen_texts:
        hw_u_raw, rule_flags_u = extract_handwriting_and_orthographic_features(unseen_texts)
        hw_u = np.clip(scaler.transform(hw_u_raw), -5.0, 5.0).astype(np.float32)
        lp_hw_u = log_softmax(clf_hw_plain.decision_function(hw_u))
        if clf_hw_twin is not None and twin_rows_085_u.any():
            hw_u_c = hw_u.copy()
            for g in np.unique(groups_085_u[twin_rows_085_u]):
                m_g = (groups_085_u == g)
                hw_u_c[m_g] -= hw_u_c[m_g].mean(axis=0, keepdims=True)
            lp_hw_u_twin = log_softmax(clf_hw_twin.decision_function(hw_u_c[twin_rows_085_u]))
            lp_hw_u[twin_rows_085_u] = log_softmax(0.70 * lp_hw_u[twin_rows_085_u] + 0.30 * lp_hw_u_twin)
    else:
        rule_flags_u = np.zeros((0, 4), dtype=np.float32)
        lp_hw_u = np.zeros((0, K), dtype=np.float32)

    # 5th View Transform
    is_bpe = bundle.get("is_bpe", False)
    v_5th = bundle["v_5th"]

    if is_bpe:
        tok = None
        if qwen_tok_dir and os.path.exists(qwen_tok_dir):
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(qwen_tok_dir, use_fast=True)
            except Exception:
                tok = None

        if tok is not None:
            def batch_bpe_strings(texts: List[str], batch_sz: int = 4000) -> List[str]:
                out_strs = []
                for s in range(0, len(texts), batch_sz):
                    batch_ids = tok(
                        texts[s : s + batch_sz],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=512,
                    )["input_ids"]
                    out_strs.extend(" ".join(map(str, ids)) for ids in batch_ids)
                return out_strs

            test_bpe_strs = batch_bpe_strings(test_texts)
            unseen_bpe_strs = batch_bpe_strings(unseen_texts) if unseen_texts else []
            te_5th = v_5th.transform(test_bpe_strs) * 0.8
            unseen_5th = v_5th.transform(unseen_bpe_strs) * 0.8 if unseen_texts else None
            del test_bpe_strs, unseen_bpe_strs
        else:
            te_5th = v_5th.transform(test_texts) * 0.8
            unseen_5th = v_5th.transform(unseen_texts) * 0.8 if unseen_texts else None
    else:
        te_5th = v_5th.transform(test_texts) * 0.8
        unseen_5th = v_5th.transform(unseen_texts) * 0.8 if unseen_texts else None

    X_te_5v = sparse.hstack([X_te_4v, te_5th], format="csr")
    X_unseen_5v = sparse.hstack([X_unseen_4v, unseen_5th], format="csr") if unseen_5th is not None else None

    # Complementary Ensemble
    svc_comp = bundle["svc_comp"]
    sgd_comp = bundle["sgd_comp"]
    svc_twin = bundle.get("svc_twin")

    P_svc_comp = softmax(svc_comp.decision_function(X_te_5v), axis=1)
    P_sgd_comp = sgd_comp.predict_proba(X_te_5v)
    P_twin_comp = P_svc_comp.copy()

    if svc_twin is not None and twin_rows_085.any():
        X_te_twin = X_te_5v[twin_rows_085]
        u_t, inv_t = np.unique(groups_085[twin_rows_085], return_inverse=True)
        counts_t = np.bincount(inv_t)
        vals_t = (1.0 / counts_t[inv_t]).astype(np.float32)
        G_t = sparse.csr_matrix((vals_t, (np.arange(len(inv_t)), inv_t)), shape=(len(inv_t), len(u_t)), dtype=np.float32)
        X_te_twin_c = X_te_twin - G_t @ (G_t.T @ X_te_twin)
        P_twin_comp[twin_rows_085] = softmax(svc_twin.decision_function(X_te_twin_c), axis=1)
        del X_te_twin, G_t, X_te_twin_c

    P_lex_comp = 0.40 * P_svc_comp + 0.25 * P_sgd_comp + 0.15 * P_twin_comp + 0.20 * P_distort
    log_p_lex_comp = np.log(np.clip(P_lex_comp, 1e-9, 1.0))

    P_unseen_comp = None
    log_p_unseen_comp = None
    if X_unseen_5v is not None and X_unseen_5v.shape[0] > 0:
        P_unseen_svc = softmax(svc_comp.decision_function(X_unseen_5v), axis=1)
        P_unseen_sgd = sgd_comp.predict_proba(X_unseen_5v)
        P_twin_comp_u = P_unseen_svc.copy()
        if svc_twin is not None and len(twin_rows_085_u) > 0 and twin_rows_085_u.any():
            X_u_twin = X_unseen_5v[twin_rows_085_u]
            u_u, inv_u = np.unique(groups_085_u[twin_rows_085_u], return_inverse=True)
            counts_u = np.bincount(inv_u)
            vals_u = (1.0 / counts_u[inv_u]).astype(np.float32)
            G_u = sparse.csr_matrix((vals_u, (np.arange(len(inv_u)), inv_u)), shape=(len(inv_u), len(u_u)), dtype=np.float32)
            X_u_twin_c = X_u_twin - G_u @ (G_u.T @ X_u_twin)
            P_twin_comp_u[twin_rows_085_u] = softmax(svc_twin.decision_function(X_u_twin_c), axis=1)
            del X_u_twin, G_u, X_u_twin_c

        u_twin = P_twin_comp_u if P_twin_comp_u is not None else P_unseen_svc
        if P_unseen_distort is not None:
            P_unseen_comp = 0.40 * P_unseen_svc + 0.25 * P_unseen_sgd + 0.15 * u_twin + 0.20 * P_unseen_distort
        else:
            P_unseen_comp = 0.55 * P_unseen_svc + 0.45 * P_unseen_sgd
        log_p_unseen_comp = np.log(np.clip(P_unseen_comp, 1e-9, 1.0))

    # 4 Within-Family Specialist Refiners
    beta = 0.25
    specialist_svcs = bundle.get("specialist_svcs", {})
    families = bundle.get("families", FAMILIES)

    for fam_name, spec_svc in specialist_svcs.items():
        if fam_name not in families:
            continue
        fam_cols = families[fam_name]
        df_spec = spec_svc.decision_function(X_te_5v)
        logits_spec = np.column_stack([-df_spec, df_spec]) if df_spec.ndim == 1 else df_spec
        P_spec = softmax(logits_spec, axis=1)

        P_prior = P_lex_comp[:, fam_cols]
        P_prior_norm = P_prior / (P_prior.sum(axis=1, keepdims=True) + 1e-12)
        P_refined = (1.0 - beta) * P_prior_norm + beta * P_spec

        log_ratio = np.log(np.clip(P_refined, 1e-9, 1.0)) - np.log(np.clip(P_prior_norm, 1e-9, 1.0))
        log_p_lex_comp[:, fam_cols] += log_ratio

        if log_p_unseen_comp is not None and X_unseen_5v is not None and X_unseen_5v.shape[0] > 0:
            df_spec_u = spec_svc.decision_function(X_unseen_5v)
            logits_spec_u = np.column_stack([-df_spec_u, df_spec_u]) if df_spec_u.ndim == 1 else df_spec_u
            P_spec_u = softmax(logits_spec_u, axis=1)
            P_prior_u = P_unseen_comp[:, fam_cols]
            P_prior_u_norm = P_prior_u / (P_prior_u.sum(axis=1, keepdims=True) + 1e-12)
            P_refined_u = (1.0 - beta) * P_prior_u_norm + beta * P_spec_u
            log_ratio_u = np.log(np.clip(P_refined_u, 1e-9, 1.0)) - np.log(np.clip(P_prior_u_norm, 1e-9, 1.0))
            log_p_unseen_comp[:, fam_cols] += log_ratio_u

    P_lex_comp_final = softmax(log_p_lex_comp, axis=1)
    unseen_p_lex_comp_final = softmax(log_p_unseen_comp, axis=1) if log_p_unseen_comp is not None else np.zeros((0, K), dtype=np.float32)

    # Save to disk if working_dir provided
    if working_dir is not None:
        working_dir = Path(working_dir)
        os.makedirs(working_dir, exist_ok=True)
        np.save(working_dir / 'log_p_svc_main.npy', log_p_svc_main)
        np.save(working_dir / 'p_lex_comp.npy', P_lex_comp_final)
        np.save(working_dir / 'D_tfidf.npy', D)
        np.save(working_dir / 'groups_085.npy', groups_085)
        np.save(working_dir / 'log_p_handwriting.npy', lp_hw)
        np.save(working_dir / 'rule_flags_test.npy', rule_flags_te)
        np.save(working_dir / 'unseen_log_p_svc_main.npy', unseen_log_p_svc_main)
        np.save(working_dir / 'unseen_p_lex_comp.npy', unseen_p_lex_comp_final)
        if 'D_u' in locals() and D_u is not None:
            np.save(working_dir / 'D_tfidf_unseen.npy', D_u)
        if 'groups_085_u' in locals() and groups_085_u is not None:
            np.save(working_dir / 'groups_085_unseen.npy', groups_085_u)
        if 'lp_hw_u' in locals() and lp_hw_u is not None:
            np.save(working_dir / 'log_p_handwriting_unseen.npy', lp_hw_u)
        if 'rule_flags_u' in locals() and rule_flags_u is not None:
            np.save(working_dir / 'rule_flags_unseen.npy', rule_flags_u)
        if log_p_distort is not None:
            np.save(working_dir / 'log_p_distort.npy', log_p_distort)
        if log_p_unseen_distort is not None:
            np.save(working_dir / 'unseen_log_p_distort.npy', log_p_unseen_distort)

    print(f"[Lexical] predict_lexical_engine completed in {time.time() - t0:.2f}s")
    return {
        "log_p_svc_main": log_p_svc_main,
        "p_lex_comp": P_lex_comp_final,
        "D_tfidf": D,
        "groups_085": groups_085,
        "log_p_handwriting": lp_hw,
        "rule_flags_test": rule_flags_te,
        "unseen_log_p_svc_main": unseen_log_p_svc_main,
        "unseen_p_lex_comp": unseen_p_lex_comp_final,
        "D_tfidf_unseen": D_u if ('D_u' in locals() and D_u is not None) else None,
        "groups_085_unseen": groups_085_u if ('groups_085_u' in locals() and groups_085_u is not None) else None,
        "log_p_handwriting_unseen": lp_hw_u if ('lp_hw_u' in locals() and lp_hw_u is not None) else None,
        "rule_flags_unseen": rule_flags_u if ('rule_flags_u' in locals() and rule_flags_u is not None) else None,
        "log_p_distort": log_p_distort,
        "unseen_log_p_distort": log_p_unseen_distort,
    }

def run_lexical_engine(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    pool_file: Optional[Path] = None,
    qwen_tok_dir: str = "",
    working_dir: Optional[Path] = None,
    bundle_path: Optional[Path | str] = None,
) -> Dict[str, np.ndarray]:
    """
    Runs the lexical engine with automatic pre-trained bundle auto-discovery.
    If a pre-trained bundle exists, executes pure inference in ~15-30s.
    Otherwise fits models from scratch on train + pool data.
    """
    working_dir = Path(working_dir) if working_dir else Path(".")
    working_dir.mkdir(parents=True, exist_ok=True)

    # 1. Search for pre-trained bundle
    bundle = None
    bundle_candidates = []
    if bundle_path:
        bundle_candidates.append(Path(bundle_path))
    if working_dir:
        bundle_candidates.append(working_dir / "lexical_bundle.joblib")
    bundle_candidates.extend([
        Path("local_data/statistical_heads/lexical_bundle.joblib"),
        Path("/kaggle/input/ntl-statistical-heads/lexical_bundle.joblib"),
    ])
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("lexical_bundle.joblib"):
            if p not in bundle_candidates:
                bundle_candidates.append(p)

    for cand in bundle_candidates:
        if cand.exists():
            try:
                bundle = load_lexical_bundle(cand)
                print(f"[Lexical] Using pre-trained bundle from {cand}")
                break
            except Exception as e:
                print(f"[Lexical] Could not load candidate bundle {cand}: {e}")

    # 2. Extract test and unseen texts
    df_te = pd.read_csv(test_path) if test_path and Path(test_path).exists() else pd.DataFrame({"text": []})
    test_texts = df_te['text'].fillna('').astype(str).tolist()

    unseen_parquet = working_dir / "unseen_holdout.parquet"
    unseen_texts: List[str] = []
    unseen_hashes = set()
    if unseen_parquet.exists():
        try:
            df_unseen_holdout = pd.read_parquet(unseen_parquet)
            if len(df_unseen_holdout) > 0 and "text" in df_unseen_holdout.columns:
                unseen_texts = df_unseen_holdout["text"].fillna("").astype(str).tolist()
                if "_hash" in df_unseen_holdout.columns:
                    unseen_hashes = set(df_unseen_holdout["_hash"].tolist())
                else:
                    unseen_hashes = set(_text_hash(t) for t in unseen_texts)
                print(f"[Lexical] Found unseen holdout set: {len(unseen_texts)} rows quarantined from training.")
        except Exception as e:
            print(f"[Lexical] Note: Could not read {unseen_parquet}: {e}")

    # 3. If pre-trained bundle found, predict immediately!
    if bundle is not None:
        return predict_lexical_engine(
            bundle=bundle,
            test_texts=test_texts,
            unseen_texts=unseen_texts,
            qwen_tok_dir=qwen_tok_dir,
            working_dir=working_dir,
        )

    # 4. Fall back to training from scratch
    print("[Lexical] No pre-trained bundle found. Training lexical engine from scratch...")
    if train_path is None or not Path(train_path).exists():
        raise FileNotFoundError(f"Training path {train_path} not found and no pre-trained bundle available.")

    df_tr = pd.read_csv(train_path)
    fit_texts = df_tr['text'].fillna('').astype(str).tolist()
    fit_y = np.array([LAB2I[m] for m in df_tr['model']])
    fit_sw = np.ones(len(df_tr), dtype=np.float32)  # Official train rows get 1.00
    seen_texts = {_norm_text_key(t) for t in fit_texts}

    # Load pool parquet files if present with source weights
    pool_candidates = [pool_file] if isinstance(pool_file, (str, Path)) else (pool_file if isinstance(pool_file, list) else [])
    for pf in pool_candidates:
        if pf and os.path.exists(pf):
            try:
                p_df = pd.read_parquet(pf, columns=['text', 'model'])
                p_df = p_df[p_df['model'].isin(LABELS)].dropna(subset=['text', 'model']).copy()
                pf_name = str(pf).lower().replace('\\', '/')
                # Matched clean gets 0.50, general scraped pool gets 0.35
                src_w = 0.50 if ('matched' in pf_name or 'clean' in pf_name) else 0.35

                keep_mask = []
                for t in p_df['text'].values:
                    k = _norm_text_key(t)
                    h = _text_hash(t)
                    ok = bool(k) and (k not in seen_texts) and (h not in unseen_hashes)
                    keep_mask.append(ok)
                    if ok:
                        seen_texts.add(k)
                p_df = p_df.loc[keep_mask].reset_index(drop=True)
                if len(p_df) > 0:
                    p_texts = p_df['text'].fillna('').astype(str).tolist()
                    p_y = p_df['model'].map(LAB2I).values
                    p_sw = np.full(len(p_df), src_w, dtype=np.float32)
                    fit_texts.extend(p_texts)
                    fit_y = np.r_[fit_y, p_y]
                    fit_sw = np.r_[fit_sw, p_sw]
                    print(f"[Lexical] Added {len(p_df)} rows from {Path(pf).name} with source weight {src_w}")
            except Exception as e:
                print(f"[Lexical] Note: Could not read pool {pf}: {e}")

    bundle = train_lexical_engine(
        fit_texts=fit_texts,
        fit_y=fit_y,
        fit_sw=fit_sw,
        df_tr=df_tr,
        qwen_tok_dir=qwen_tok_dir,
    )

    # Optional: save the newly trained bundle to working_dir
    try:
        save_lexical_bundle(bundle, working_dir / "lexical_bundle.joblib")
    except Exception as e:
        print(f"[Lexical] Note: Could not save bundle: {e}")

    return predict_lexical_engine(
        bundle=bundle,
        test_texts=test_texts,
        unseen_texts=unseen_texts,
        qwen_tok_dir=qwen_tok_dir,
        working_dir=working_dir,
    )

if __name__ == "__main__":
    working_dir = Path("/kaggle/working")
    if not working_dir.exists():
        working_dir = Path("local_data")
    working_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        input_dir = Path(".")

    candidates = [
        input_dir / 'competitions' / 'name-that-llm',
        input_dir / 'name-that-llm',
        Path('local_data'),
        Path('.')
    ]
    train_path = None
    test_path = None
    for c in candidates:
        if (c / 'train.csv').exists() and train_path is None:
            train_path = c / 'train.csv'
        if (c / 'test.csv').exists() and test_path is None:
            test_path = c / 'test.csv'
    if train_path is None:
        train_path = Path('local_data') / 'train.csv'
    if test_path is None:
        test_path = Path('local_data') / 'test.csv'

    pool_files: List[Path] = []
    for pat in ['nlm_pool_v5.parquet', 'pool_v4.parquet', 'nlm_matched_clean.parquet', 'nlm_pool_v2.parquet']:
        for p in input_dir.rglob(pat):
            if p not in pool_files:
                pool_files.append(p)

    tok_paths = list(input_dir.rglob("vocab.json"))
    qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""

    print(f"[Module 1 CLI] Running lexical engine: train={train_path}, test={test_path}, pools={[p.name for p in pool_files]}")
    run_lexical_engine(train_path, test_path, pool_files, qwen_tok_dir, working_dir)
    print("[Module 1 CLI] Lexical engine finished successfully.")
