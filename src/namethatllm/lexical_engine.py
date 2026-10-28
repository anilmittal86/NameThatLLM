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
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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

STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself"
])

def shape_text(t: str) -> str:
    """Replaces alphabetic character blocks with X (uppercase) and x (lowercase)."""
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m.group(0)[0].isupper() else "x", str(t))

def function_text(t: str) -> str:
    """Masks non-stopwords and non-punctuation to CONTENT, preserving function and punctuation syntax."""
    tokens = re.findall(r"(?u)\b\w+\b|[^\w\s]", str(t))
    return " ".join([w.lower() if (w.lower() in STOP_WORDS or not w.isalnum()) else "CONTENT" for w in tokens])

def run_lexical_engine(train_path: Path, test_path: Path, pool_file: Optional[Path],
                       qwen_tok_dir: str, working_dir: Path) -> Dict[str, np.ndarray]:
    """
    Runs the complete dual-output lexical engine in-process or writes standalone script.
    Saves and returns:
    - log_p_svc_main.npy (Output 1 for LOGIT_MAIN)
    - p_lex_comp.npy (Output 2 for COMP_LIST)
    - D_tfidf.npy
    - groups_085.npy
    """
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    test_texts = df_te['text'].fillna('').astype(str).tolist()
    fit_texts = df_tr['text'].fillna('').astype(str).tolist()
    fit_y = np.array([LAB2I[m] for m in df_tr['model']])
    len_tr_raw = len(df_tr)

    # 1. Compute stopword TF-IDF distance matrix and topic clusters (t=0.85) on test
    Xt_tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100000, dtype=np.float32).fit_transform(test_texts)
    if len(test_texts) >= 2:
        D = np.clip(1.0 - (Xt_tfidf @ Xt_tfidf.T).toarray(), 0.0, None)
        np.fill_diagonal(D, 0.0)
        cond_d = squareform(D, checks=False)
        groups_085 = fcluster(linkage(cond_d, method='average'), 0.85, criterion='distance')
    else:
        D = np.zeros((1, 1), dtype=np.float32)
        groups_085 = np.ones(1, dtype=int)

    sz_085 = np.bincount(groups_085)[groups_085]
    twin_rows_085 = (sz_085 >= 2) & (sz_085 <= 18)

    # 2. Load pool_v4.parquet if present
    if pool_file and os.path.exists(pool_file):
        df_pool = pd.read_parquet(pool_file, columns=['text', 'model'])
        df_pool = df_pool[df_pool['model'].isin(LABELS)].dropna(subset=['text', 'model']).copy()
        fit_texts += df_pool['text'].fillna('').astype(str).tolist()
        fit_y = np.r_[fit_y, df_pool['model'].map(LAB2I).values]
        del df_pool
        gc.collect()

    # 3. Canonical 4 Views (from NTL two-stage LoRA part two)
    VIEWS_4V = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("function", 1, 3, 40000, 0.5),
    ]

    fit_sh = [shape_text(t) for t in fit_texts]
    test_sh = [shape_text(t) for t in test_texts]
    fit_fn = [function_text(t) for t in fit_texts]
    test_fn = [function_text(t) for t in test_texts]

    tr_mats_4v = []
    te_mats_4v = []
    for mode, mn, mx, mf, sc in VIEWS_4V:
        if mode == "shape":
            v = TfidfVectorizer(analyzer="char", ngram_range=(mn, mx), max_features=mf, sublinear_tf=True, dtype=np.float32)
            tr_m = v.fit_transform(fit_sh) * sc
            te_m = v.transform(test_sh) * sc
        elif mode == "function":
            v = TfidfVectorizer(analyzer="word", ngram_range=(mn, mx), max_features=mf, sublinear_tf=True, dtype=np.float32)
            tr_m = v.fit_transform(fit_fn) * sc
            te_m = v.transform(test_fn) * sc
        elif mode == "word":
            v = TfidfVectorizer(analyzer="word", ngram_range=(mn, mx), max_features=mf,
                                token_pattern=r"(?u)\b\w+\b|[^\w\s]", lowercase=False,
                                sublinear_tf=True, dtype=np.float32)
            tr_m = v.fit_transform(fit_texts) * sc
            te_m = v.transform(test_texts) * sc
        else:
            v = TfidfVectorizer(analyzer=mode, ngram_range=(mn, mx), max_features=mf, sublinear_tf=True, dtype=np.float32)
            tr_m = v.fit_transform(fit_texts) * sc
            te_m = v.transform(test_texts) * sc
        tr_mats_4v.append(tr_m)
        te_mats_4v.append(te_m)

    X_tr_4v = sparse.hstack(tr_mats_4v, format="csr")
    X_te_4v = sparse.hstack(te_mats_4v, format="csr")

    # OUTPUT 1: Pure 4-View LinearSVC (C=0.5, uniform 1.0 sample weights)
    svc_main = LinearSVC(C=0.5, random_state=42, max_iter=3000, tol=1e-3)
    svc_main.fit(X_tr_4v, fit_y)
    logits_svc_main = svc_main.decision_function(X_te_4v)
    log_p_svc_main = log_softmax(logits_svc_main)

    # OUTPUT 2: 5-View BPE + SGD + Style Twin + Specialist Complementary Ensemble
    tok = None
    if qwen_tok_dir and os.path.exists(qwen_tok_dir):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(qwen_tok_dir, use_fast=True)
        except Exception:
            tok = None

    if tok is not None:
        def bpe_tokenize(txt: str) -> List[str]:
            return [str(tok_id) for tok_id in tok.encode(str(txt), truncation=True, max_length=512)]
        v_bpe = TfidfVectorizer(tokenizer=bpe_tokenize, lowercase=False, token_pattern=None,
                                ngram_range=(1, 3), max_features=75000, sublinear_tf=True, dtype=np.float32)
        tr_bpe = v_bpe.fit_transform(fit_texts) * 0.8
        te_bpe = v_bpe.transform(test_texts) * 0.8
        X_tr_5v = sparse.hstack([X_tr_4v, tr_bpe], format="csr")
        X_te_5v = sparse.hstack([X_te_4v, te_bpe], format="csr")
    else:
        v_cw = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=75000, sublinear_tf=True, dtype=np.float32)
        tr_cw = v_cw.fit_transform(fit_texts) * 0.8
        te_cw = v_cw.transform(test_texts) * 0.8
        X_tr_5v = sparse.hstack([X_tr_4v, tr_cw], format="csr")
        X_te_5v = sparse.hstack([X_te_4v, te_cw], format="csr")

    svc_comp = LinearSVC(C=0.35, random_state=42, max_iter=3000, tol=1e-3)
    svc_comp.fit(X_tr_5v, fit_y)
    P_svc_comp = softmax(svc_comp.decision_function(X_te_5v), axis=1)

    sgd_comp = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1500, random_state=42)
    sgd_comp.fit(X_tr_5v, fit_y)
    P_sgd_comp = sgd_comp.predict_proba(X_te_5v)

    # Style Twin on groups_085
    P_twin_comp = P_svc_comp.copy()
    if twin_rows_085.any() and len_tr_raw >= 18:
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

        X_te_twin = X_te_5v[twin_rows_085]
        u_t, inv_t = np.unique(groups_085[twin_rows_085], return_inverse=True)
        counts_t = np.bincount(inv_t)
        vals_t = (1.0 / counts_t[inv_t]).astype(np.float32)
        G_t = sparse.csr_matrix((vals_t, (np.arange(len(inv_t)), inv_t)), shape=(len(inv_t), len(u_t)), dtype=np.float32)
        X_te_twin_c = X_te_twin - G_t @ (G_t.T @ X_te_twin)
        P_twin_comp[twin_rows_085] = softmax(svc_twin.decision_function(X_te_twin_c), axis=1)
        del svc_twin, X_te_twin, G_t, X_te_twin_c
        gc.collect()

    P_lex_comp = 0.50 * P_svc_comp + 0.35 * P_sgd_comp + 0.15 * P_twin_comp
    log_p_lex_comp = np.log(np.clip(P_lex_comp, 1e-9, 1.0))

    # 4 Within-Family Specialist Refiners
    beta = 0.25
    for fam_name, fam_cols in FAMILIES.items():
        fam_mask = np.isin(fit_y, fam_cols)
        if fam_mask.sum() > 50:
            sub_y = fit_y[fam_mask]
            sub_lab2i = {c: i for i, c in enumerate(fam_cols)}
            sub_y_mapped = np.array([sub_lab2i[y] for y in sub_y])

            spec_svc = LinearSVC(C=0.40, random_state=42, max_iter=2000, tol=1e-3, class_weight="balanced")
            spec_svc.fit(X_tr_5v[fam_mask], sub_y_mapped)
            df_spec = spec_svc.decision_function(X_te_5v)
            logits_spec = np.column_stack([-df_spec, df_spec]) if df_spec.ndim == 1 else df_spec
            P_spec = softmax(logits_spec, axis=1)

            P_prior = P_lex_comp[:, fam_cols]
            P_prior_norm = P_prior / (P_prior.sum(axis=1, keepdims=True) + 1e-12)
            P_refined = (1.0 - beta) * P_prior_norm + beta * P_spec

            log_ratio = np.log(np.clip(P_refined, 1e-9, 1.0)) - np.log(np.clip(P_prior_norm, 1e-9, 1.0))
            log_p_lex_comp[:, fam_cols] += log_ratio

    P_lex_comp_final = softmax(log_p_lex_comp, axis=1)

    # Save to disk
    os.makedirs(working_dir, exist_ok=True)
    np.save(working_dir / 'log_p_svc_main.npy', log_p_svc_main)
    np.save(working_dir / 'p_lex_comp.npy', P_lex_comp_final)
    np.save(working_dir / 'D_tfidf.npy', D)
    np.save(working_dir / 'groups_085.npy', groups_085)

    return {
        "log_p_svc_main": log_p_svc_main,
        "p_lex_comp": P_lex_comp_final,
        "D_tfidf": D,
        "groups_085": groups_085
    }

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

    pool_paths = list(input_dir.rglob('pool_v4.parquet'))
    pool_file = pool_paths[0] if pool_paths else None

    tok_paths = list(input_dir.rglob("vocab.json"))
    qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""

    print(f"[Module 1 CLI] Running lexical engine: train={train_path}, test={test_path}")
    run_lexical_engine(train_path, test_path, pool_file, qwen_tok_dir, working_dir)
    print("[Module 1 CLI] Lexical engine finished successfully.")
