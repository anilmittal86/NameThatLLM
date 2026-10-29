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

    # 2. Load pool parquet files if present (supports single path or list)
    pool_candidates = [pool_file] if isinstance(pool_file, (str, Path)) else (pool_file if isinstance(pool_file, list) else [])
    pool_dfs = []
    for pf in pool_candidates:
        if pf and os.path.exists(pf):
            try:
                p_df = pd.read_parquet(pf, columns=['text', 'model'])
                p_df = p_df[p_df['model'].isin(LABELS)].dropna(subset=['text', 'model']).copy()
                pool_dfs.append(p_df)
            except Exception as e:
                print(f"[Lexical] Note: Could not read pool {pf}: {e}")
    if pool_dfs:
        df_pool = pd.concat(pool_dfs, ignore_index=True).drop_duplicates(subset=['text'])
        print(f"[Lexical] Merged {len(pool_dfs)} pool file(s): {len(df_pool)} unique rows.")
        fit_texts += df_pool['text'].fillna('').astype(str).tolist()
        fit_y = np.r_[fit_y, df_pool['model'].map(LAB2I).values]
        del df_pool, pool_dfs
        gc.collect()

    # 3. Canonical 4 Views (from NTL two-stage LoRA part two)
    VIEWS_4V = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("function", 1, 3, 50000, 0.5),
    ]

    fit_sh = [shape_text(t) for t in fit_texts]
    test_sh = [shape_text(t) for t in test_texts]
    fit_fn = [function_text(t) for t in fit_texts]
    test_fn = [function_text(t) for t in test_texts]

    tr_mats_4v = []
    te_mats_4v = []
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
            te_m = v.transform(test_sh) * np.float32(sc)
        elif mode == "function":
            tr_m = v.fit_transform(fit_fn) * np.float32(sc)
            te_m = v.transform(test_fn) * np.float32(sc)
        else:
            tr_m = v.fit_transform(fit_texts) * np.float32(sc)
            te_m = v.transform(test_texts) * np.float32(sc)
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
    # Continuous Handwriting & Punctuation Density Features + Dedicated Classifier
    hw_tr_raw, _ = extract_handwriting_and_orthographic_features(fit_texts)
    hw_te_raw, rule_flags_te = extract_handwriting_and_orthographic_features(test_texts)
    scaler = StandardScaler()
    hw_tr = np.clip(scaler.fit_transform(hw_tr_raw), -5.0, 5.0).astype(np.float32)
    hw_te = np.clip(scaler.transform(hw_te_raw), -5.0, 5.0).astype(np.float32)

    lp_hw = fit_handwriting_model(hw_tr, fit_y, hw_te, df_tr, groups_085, twin_rows_085, len_tr_raw)

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
        test_bpe_strs = batch_bpe_strings(test_texts)
        v_bpe = TfidfVectorizer(
            analyzer="word", token_pattern=r"\S+", lowercase=False,
            ngram_range=(1, 3), max_features=75000, sublinear_tf=True, dtype=np.float32
        )
        tr_bpe = v_bpe.fit_transform(fit_bpe_strs) * 0.8
        te_bpe = v_bpe.transform(test_bpe_strs) * 0.8
        del fit_bpe_strs, test_bpe_strs, v_bpe
        gc.collect()
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
    np.save(working_dir / 'log_p_handwriting.npy', lp_hw)
    np.save(working_dir / 'rule_flags_test.npy', rule_flags_te)

    return {
        "log_p_svc_main": log_p_svc_main,
        "p_lex_comp": P_lex_comp_final,
        "D_tfidf": D,
        "groups_085": groups_085,
        "log_p_handwriting": lp_hw,
        "rule_flags_test": rule_flags_te
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
