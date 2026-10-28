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

from .config import LABELS, K, LAB2I, FAMILIES, log_softmax, softmax

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
    """Masks non-stopwords to CONTENT, preserving function syntax patterns."""
    tokens = re.findall(r"(?u)\b\w+\b|[^\w\s]", str(t))
    return " ".join([w.lower() if w.lower() in STOP_WORDS else "CONTENT" for w in tokens])

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

    # 2. Load 100% of pool_v4.parquet
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
        ("function", 1, 3, 50000, 0.5)
    ]

    A_blocks_4v, B_blocks_4v = [], []
    for view, lo, hi, max_feat, weight in VIEWS_4V:
        prep = shape_text if view == "shape" else function_text if view == "function" else None
        analyzer = "char" if view in ("char", "shape") else "word"
        opts = dict(
            analyzer=analyzer, ngram_range=(lo, hi), lowercase=False,
            sublinear_tf=True, min_df=2, max_features=max_feat, dtype=np.float32
        )
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        tr_iter = (prep(t) for t in fit_texts) if prep is not None else fit_texts
        te_iter = (prep(t) for t in test_texts) if prep is not None else test_texts
        X_tr_v = vec.fit_transform(tr_iter)
        X_te_v = vec.transform(te_iter)
        X_tr_v.data *= np.float32(weight)
        X_te_v.data *= np.float32(weight)
        A_blocks_4v.append(X_tr_v)
        B_blocks_4v.append(X_te_v)
        del vec, X_tr_v, X_te_v
        gc.collect()

    X_tr_4v = sparse.hstack(A_blocks_4v, format="csr", dtype=np.float32)
    X_te_4v = sparse.hstack(B_blocks_4v, format="csr", dtype=np.float32)

    # OUTPUT 1: Pure 4-View LinearSVC with uniform 1.0 sample weights
    sample_weights_uniform = np.ones(len(fit_y), dtype=np.float32)
    svc_main = LinearSVC(C=0.5, random_state=42, max_iter=3500, tol=1e-3, class_weight="balanced")
    svc_main.fit(X_tr_4v, fit_y, sample_weight=sample_weights_uniform)
    log_p_svc_main = log_softmax(svc_main.decision_function(X_te_4v), axis=1)
    del svc_main
    gc.collect()

    # 4. View 5: Native BPE Subword Token-ID N-Grams
    bpe_loaded = False
    if qwen_tok_dir and os.path.exists(qwen_tok_dir):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(qwen_tok_dir, use_fast=True)
            def to_bpe_strings(texts, batch_size=2048):
                res = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    enc = tok(batch, add_special_tokens=False, truncation=True, max_length=450)["input_ids"]
                    for row in enc:
                        res.append(" ".join(map(str, row)))
                return res
            bpe_tr = to_bpe_strings(fit_texts)
            bpe_te = to_bpe_strings(test_texts)
            bpe_loaded = True
            del tok
            gc.collect()
        except Exception:
            pass

    if not bpe_loaded:
        def hash_bpe_fallback(texts):
            return [" ".join("w" + str(hash(w) % 65536) for w in re.findall(r'\b\w+\b', t.lower())) for t in texts]
        bpe_tr = hash_bpe_fallback(fit_texts)
        bpe_te = hash_bpe_fallback(test_texts)

    bpe_vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 3), lowercase=False, sublinear_tf=True, min_df=5, max_features=80000, dtype=np.float32)
    X_tr_bpe = bpe_vec.fit_transform(bpe_tr)
    X_te_bpe = bpe_vec.transform(bpe_te)
    X_tr_bpe.data *= np.float32(0.65)
    X_te_bpe.data *= np.float32(0.65)
    del bpe_vec, bpe_tr, bpe_te
    gc.collect()

    X_tr_5v = sparse.hstack([X_tr_4v, X_tr_bpe], format="csr", dtype=np.float32)
    X_te_5v = sparse.hstack([X_te_4v, X_te_bpe], format="csr", dtype=np.float32)
    del X_tr_4v, X_te_4v, X_tr_bpe, X_te_bpe, A_blocks_4v, B_blocks_4v
    gc.collect()

    # OUTPUT 2: Complementary Lexical Ensemble (LinearSVC + SGDClassifier + Style Twin + Specialists)
    svc_comp = LinearSVC(C=0.35, random_state=42, max_iter=2500, tol=1e-3, class_weight="balanced")
    svc_comp.fit(X_tr_5v, fit_y, sample_weight=sample_weights_uniform)
    P_svc_comp = softmax(svc_comp.decision_function(X_te_5v), axis=1)
    del svc_comp
    gc.collect()

    sgd_comp = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-3, random_state=42, class_weight="balanced")
    sgd_comp.fit(X_tr_5v, fit_y, sample_weight=sample_weights_uniform)
    P_sgd_comp = softmax(sgd_comp.decision_function(X_te_5v), axis=1)
    del sgd_comp
    gc.collect()

    # Prompt-Centered Style Twin on train.csv
    P_twin_comp = np.copy(P_svc_comp)
    if twin_rows_085.any():
        X_train_only = X_tr_5v[:len_tr_raw]
        pu, pinv = np.unique(df_tr['prompt_id'].values, return_inverse=True)
        n_p = len(pu)
        counts = np.bincount(pinv)
        vals = (1.0 / counts[pinv]).astype(np.float32)
        G = sparse.csr_matrix((vals, (np.arange(len_tr_raw), pinv)), shape=(len_tr_raw, n_p), dtype=np.float32)
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
