"""
Subprocess Dual-Output Lexical Engine
====================================
Subprocess memory isolation running 78,482-row 4-View LinearSVC (Output 1)
and 5-View BPE + SGD + Style Twin + Specialist Complementary Ensemble (Output 2).
"""

import collections
import gc
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import (
    FAMILIES,
    HARD_CLASS_INDICES,
    LAB2I,
    LABELS,
    K,
    _text_hash,
    log_softmax,
    resolve_all_pool_files,
    resolve_dataset_paths,
    softmax,
)


def _norm_text_key(value: Any) -> str:
    """Normalized NFKC string key for exact deduplication."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


FUNCTION_WORDS = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "although",
        "because",
        "as",
        "so",
        "than",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "i",
        "me",
        "my",
        "mine",
        "in",
        "on",
        "at",
        "to",
        "from",
        "with",
        "without",
        "for",
        "of",
        "by",
        "over",
        "under",
        "between",
        "through",
        "into",
        "about",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "not",
        "no",
        "nor",
        "all",
        "any",
        "some",
        "each",
        "every",
        "both",
        "either",
        "neither",
        "other",
        "another",
        "such",
        "only",
        "also",
        "just",
        "very",
        "too",
        "more",
        "most",
        "less",
        "least",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "what",
        "whether",
        "then",
        "now",
        "still",
        "yet",
        "even",
        "much",
        "many",
        "few",
        "little",
        "am",
        "having",
        "s",
        "t",
        "re",
        "ve",
        "ll",
        "d",
        "m",
        "don",
        "doesn",
        "didn",
        "isn",
        "aren",
        "wasn",
        "weren",
        "hasn",
        "haven",
        "hadn",
        "won",
        "wouldn",
        "couldn",
        "shouldn",
    ]
)
STOP_WORDS = FUNCTION_WORDS


def shape_text(t: str) -> str:
    """
    Transforms text into structural casing patterns (shape view).
    Replaces capitalized words with 'X' and lowercase words with 'x'.
    Captures formatting capitalization rhythm independent of lexical vocabulary.
    """
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m.group(0)[0].isupper() else "x", str(t))


def function_text(t: str) -> str:
    """
    Transforms text into a pure function-word and syntactic skeleton (function view).
    Masks open-class content words (nouns, main verbs, adjectives) to 'CONTENT' while
    preserving closed-class function words (prepositions, conjunctions, pronouns, auxiliaries)
    and punctuation. Stylometric research demonstrates that function-word transition rates
    form one of the strongest, topic-invariant signatures of LLM generator identity.
    """
    tokens = re.findall(r"(?u)\b\w+\b|[^\w\s]", str(t))
    return " ".join([w.lower() if (w.lower() in STOP_WORDS or not w.isalnum()) else "CONTENT" for w in tokens])


_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
_RE_STRAIGHT_CONTR = re.compile(r"[A-Za-z]'[A-Za-z]")
_RE_CURLY_CONTR = re.compile(r"[A-Za-z]\u2019[A-Za-z]")


def extract_handwriting_and_orthographic_features(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
            c_colon,
            c_semi,
            c_em_dash,
            c_double_hyph,
            c_curly_sq,
            c_straight_sq,
            c_curly_dq,
            c_straight_dq,
            c_paren,
            c_comma,
            c_excl,
            c_ques,
            c_ellipsis,
            c_hyphen,
            n_straight_contr,
            n_curly_contr,
            c_newline,
            c_bullet,
            c_upper,
            c_digits,
            mean_w_len,
            std_w_len,
            mean_s_len,
            std_s_len,
            burstiness,
            ttr,
        ]
    return feats, rules


def extract_handwriting_features(texts: list[str]) -> np.ndarray:
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
    len_tr_raw: int,
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
            m_p = df_tr["prompt_id"].values == p
            hw_tr_c[m_p] -= hw_tr_c[m_p].mean(axis=0, keepdims=True)
        clf_hw_twin = LogisticRegression(C=0.15, max_iter=1000, random_state=42, class_weight="balanced")
        clf_hw_twin.fit(hw_tr_c, fit_y[:len_tr_raw])

        hw_te_c = hw_te.copy()
        for g in np.unique(groups_085[twin_rows_085]):
            m_g = groups_085 == g
            hw_te_c[m_g] -= hw_te_c[m_g].mean(axis=0, keepdims=True)
        lp_hw_twin = log_softmax(clf_hw_twin.decision_function(hw_te_c[twin_rows_085]))
        lp_hw[twin_rows_085] = log_softmax(0.70 * lp_hw[twin_rows_085] + 0.30 * lp_hw_twin)

    return lp_hw


def run_lexical_engine(
    train_path: Path, test_path: Path, pool_file: Path | None, qwen_tok_dir: str, working_dir: Path
) -> dict[str, np.ndarray]:
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

    test_texts = df_te["text"].fillna("").astype(str).tolist()
    fit_texts = df_tr["text"].fillna("").astype(str).tolist()
    fit_y = np.array([LAB2I[m] for m in df_tr["model"]])
    fit_sw = np.ones(len(df_tr), dtype=np.float32)  # Official train rows get 1.00
    seen_texts = {_norm_text_key(t) for t in fit_texts}
    len_tr_raw = len(df_tr)

    # 1. Compute stopword TF-IDF distance matrix and topic clusters (t=0.85) on test
    vec_kwargs = dict(
        stop_words="english",
        sublinear_tf=True,
        min_df=2 if len(test_texts) >= 20 else 1,
        max_df=0.25 if len(test_texts) >= 100 else 1.0,
        max_features=100000,
        dtype=np.float32,
    )
    try:
        Xt_tfidf = TfidfVectorizer(**vec_kwargs).fit_transform(test_texts)
    except ValueError:
        vec_kwargs["min_df"] = 1
        vec_kwargs["max_df"] = 1.0
        Xt_tfidf = TfidfVectorizer(**vec_kwargs).fit_transform(test_texts)
    if len(test_texts) >= 2:
        D = np.clip(1.0 - (Xt_tfidf @ Xt_tfidf.T).toarray(), 0.0, None)
        np.fill_diagonal(D, 0.0)
        cond_d = squareform(D, checks=False)
        groups_085 = fcluster(linkage(cond_d, method="average"), 0.85, criterion="distance")
    else:
        D = np.zeros((1, 1), dtype=np.float32)
        groups_085 = np.ones(1, dtype=int)

    sz_085 = np.bincount(groups_085)[groups_085]
    twin_rows_085 = (sz_085 >= 2) & (sz_085 <= 18)

    # 1.5. Check for quarantined unseen holdout dataset
    working_dir = Path(working_dir)
    unseen_parquet = working_dir / "unseen_holdout.parquet"
    unseen_texts: list[str] = []
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

    # 2. Load pool parquet files if present with source weights & stratified two-tier caps
    core_texts = list(fit_texts)
    core_y = fit_y.copy()
    core_sw = fit_sw.copy()
    extra_fit_texts: list[str] = []
    extra_fit_y: list[int] = []
    extra_fit_sw: list[float] = []

    MAX_CORE_POOL_PER_CLASS = 700
    MAX_TOTAL_POOL_PER_CLASS = 2400
    core_class_counts = np.zeros(K, dtype=int)
    total_class_counts = np.zeros(K, dtype=int)
    rng_pool = np.random.RandomState(42)

    pool_candidates = [pool_file] if isinstance(pool_file, (str, Path)) else (pool_file if isinstance(pool_file, list) else [])
    for pf in pool_candidates:
        if pf and os.path.exists(pf):
            try:
                p_df = pd.read_parquet(pf, columns=["text", "model"])
                p_df = p_df[p_df["model"].isin(LABELS)].dropna(subset=["text", "model"]).copy()
                pf_filename = Path(pf).name.lower()
                is_matched_clean = "matched" in pf_filename or "clean" in pf_filename
                src_w = 0.50 if is_matched_clean else 0.35

                # Filter to competition passage length (25..360 words) when pool is large
                if len(p_df) > 5000:
                    w_counts = p_df["text"].astype(str).str.split().str.len()
                    p_df = p_df[w_counts.between(25, 360)].copy()

                # Shuffle deterministically before stratified cap
                p_df = p_df.sample(frac=1.0, random_state=rng_pool).reset_index(drop=True)

                fit_add_t, fit_add_y, fit_add_sw = [], [], []
                core_add_t, core_add_y, core_add_sw = [], [], []

                for t_val, m_val in zip(p_df["text"].values, p_df["model"].values):
                    c_idx = LAB2I[m_val]
                    max_tot_c = int(1.4 * MAX_TOTAL_POOL_PER_CLASS) if c_idx in HARD_CLASS_INDICES else MAX_TOTAL_POOL_PER_CLASS
                    if total_class_counts[c_idx] >= max_tot_c:
                        continue
                    k = _norm_text_key(t_val)
                    if not k or k in seen_texts:
                        continue
                    if unseen_hashes and _text_hash(t_val) in unseen_hashes:
                        continue
                    seen_texts.add(k)
                    t_str = str(t_val)
                    total_class_counts[c_idx] += 1

                    max_core_c = int(1.5 * MAX_CORE_POOL_PER_CLASS) if c_idx in HARD_CLASS_INDICES else MAX_CORE_POOL_PER_CLASS
                    if core_class_counts[c_idx] < max_core_c:
                        core_class_counts[c_idx] += 1
                        core_add_t.append(t_str)
                        core_add_y.append(c_idx)
                        core_add_sw.append(src_w)
                    else:
                        fit_add_t.append(t_str)
                        fit_add_y.append(c_idx)
                        fit_add_sw.append(src_w)

                if core_add_t:
                    core_texts.extend(core_add_t)
                    core_y = np.r_[core_y, np.array(core_add_y, dtype=int)]
                    core_sw = np.r_[core_sw, np.array(core_add_sw, dtype=np.float32)]
                if fit_add_t:
                    extra_fit_texts.extend(fit_add_t)
                    extra_fit_y.extend(fit_add_y)
                    extra_fit_sw.extend(fit_add_sw)
                print(
                    f"[Lexical] Added {len(core_add_t) + len(fit_add_t)} rows (core={len(core_add_t)}) from {Path(pf).name} with source weight {src_w}"
                )
            except Exception as e:
                print(f"[Lexical] Note: Could not read pool {pf}: {e}")

    fit_texts = core_texts + extra_fit_texts
    fit_y = np.r_[core_y, np.array(extra_fit_y, dtype=int)] if extra_fit_y else core_y
    fit_sw = np.r_[core_sw, np.array(extra_fit_sw, dtype=np.float32)] if extra_fit_sw else core_sw

    # 3. Canonical 4 Views (from NTL two-stage LoRA part two)
    VIEWS_4V = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("function", 1, 3, 50000, 0.5),
    ]

    fit_sh = [shape_text(t) for t in fit_texts]
    test_sh = [shape_text(t) for t in test_texts]
    unseen_sh = [shape_text(t) for t in unseen_texts] if unseen_texts else []
    fit_fn = [function_text(t) for t in fit_texts]
    test_fn = [function_text(t) for t in test_texts]
    unseen_fn = [function_text(t) for t in unseen_texts] if unseen_texts else []

    tr_mats_4v = []
    te_mats_4v = []
    unseen_mats_4v = []
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
            u_m = v.transform(unseen_sh) * np.float32(sc) if unseen_texts else None
        elif mode == "function":
            tr_m = v.fit_transform(fit_fn) * np.float32(sc)
            te_m = v.transform(test_fn) * np.float32(sc)
            u_m = v.transform(unseen_fn) * np.float32(sc) if unseen_texts else None
        else:
            tr_m = v.fit_transform(fit_texts) * np.float32(sc)
            te_m = v.transform(test_texts) * np.float32(sc)
            u_m = v.transform(unseen_texts) * np.float32(sc) if unseen_texts else None
        tr_mats_4v.append(tr_m)
        te_mats_4v.append(te_m)
        if u_m is not None:
            unseen_mats_4v.append(u_m)

    X_tr_4v = sparse.hstack(tr_mats_4v, format="csr")
    X_te_4v = sparse.hstack(te_mats_4v, format="csr")
    X_unseen_4v = sparse.hstack(unseen_mats_4v, format="csr") if unseen_mats_4v else None
    X_core_4v = X_tr_4v[: len(core_texts)]

    # OUTPUT 1: Pure 4-View LinearSVC (C=0.5, class_weight='balanced', source sample weights)
    svc_main = LinearSVC(C=0.5, dual=True, random_state=42, max_iter=2000, tol=1e-3, class_weight="balanced")
    svc_main.fit(X_tr_4v, fit_y, sample_weight=fit_sw)
    logits_svc_main = svc_main.decision_function(X_te_4v)
    log_p_svc_main = log_softmax(logits_svc_main)

    if X_unseen_4v is not None and X_unseen_4v.shape[0] > 0:
        unseen_log_p_svc_main = log_softmax(svc_main.decision_function(X_unseen_4v))
    else:
        unseen_log_p_svc_main = np.zeros((0, K), dtype=np.float32)

    # OUTPUT 2: Stamatatos Text Distortion Model (trained strictly on core_texts with feature caps)
    wre = re.compile(r"[A-Za-z\u2019']+")
    cnt = collections.Counter(w.lower() for t in core_texts for w in wre.findall(t))
    distort_keep = {w for w, _ in cnt.most_common(100)}

    def dist_fn(arr):
        return [wre.sub(lambda m_: m_.group(0) if m_.group(0).lower() in distort_keep else "*", t) for t in arr]

    D_tr = dist_fn(core_texts)
    D_te = dist_fn(test_texts)
    D_unseen = dist_fn(unseen_texts) if unseen_texts else []

    vs_dist = [
        TfidfVectorizer(
            sublinear_tf=True,
            lowercase=False,
            analyzer="char",
            ngram_range=(2, 6),
            max_features=100000,
            dtype=np.float32,
            min_df=2 if len(core_texts) >= 50 else 1,
        ),
        TfidfVectorizer(
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"[^\s]+",
            ngram_range=(1, 3),
            max_features=60000,
            dtype=np.float32,
            min_df=2 if len(core_texts) >= 50 else 1,
        ),
    ]
    Xa_dist = sparse.hstack([v_.fit_transform(D_tr) for v_ in vs_dist], format="csr")
    Xb_dist = sparse.hstack([v_.transform(D_te) for v_ in vs_dist], format="csr")
    Xu_dist = sparse.hstack([v_.transform(D_unseen) for v_ in vs_dist], format="csr") if unseen_texts else None

    lr_distort = LogisticRegression(C=10.0, max_iter=500, random_state=42)
    lr_distort.fit(Xa_dist, core_y, sample_weight=core_sw)
    P_distort = lr_distort.predict_proba(Xb_dist)
    log_p_distort = lr_distort.predict_log_proba(Xb_dist)
    if Xu_dist is not None and Xu_dist.shape[0] > 0:
        P_unseen_distort = lr_distort.predict_proba(Xu_dist)
        log_p_unseen_distort = lr_distort.predict_log_proba(Xu_dist)
    else:
        P_unseen_distort = None
        log_p_unseen_distort = None

    # OUTPUT 3: 5-View BPE + SGD + Style Twin + Specialist Complementary Ensemble
    # Continuous Handwriting & Punctuation Density Features + Dedicated Classifier (fit strictly on df_tr train.csv)
    hw_core_raw, _ = extract_handwriting_and_orthographic_features(core_texts)
    hw_te_raw, rule_flags_te = extract_handwriting_and_orthographic_features(test_texts)
    hw_unseen_raw, _ = extract_handwriting_and_orthographic_features(unseen_texts) if unseen_texts else (None, None)
    scaler = StandardScaler()
    scaler.fit(hw_core_raw[:len_tr_raw])
    hw_core = np.clip(scaler.transform(hw_core_raw), -5.0, 5.0).astype(np.float32)
    hw_te = np.clip(scaler.transform(hw_te_raw), -5.0, 5.0).astype(np.float32)
    hw_unseen = np.clip(scaler.transform(hw_unseen_raw), -5.0, 5.0).astype(np.float32) if hw_unseen_raw is not None else None
    hw_tr = hw_core[:len_tr_raw]

    lp_hw = fit_handwriting_model(hw_tr, fit_y[:len_tr_raw], hw_te, df_tr, groups_085, twin_rows_085, len_tr_raw)

    tok = None
    if qwen_tok_dir and os.path.exists(qwen_tok_dir):
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(qwen_tok_dir, use_fast=True)
        except Exception:
            tok = None

    if tok is not None:

        def batch_bpe_strings(texts: list[str], batch_sz: int = 4000) -> list[str]:
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

        fit_bpe_strs = batch_bpe_strings(core_texts)
        test_bpe_strs = batch_bpe_strings(test_texts)
        unseen_bpe_strs = batch_bpe_strings(unseen_texts) if unseen_texts else []
        v_bpe = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",
            lowercase=False,
            ngram_range=(1, 3),
            max_features=75000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        tr_bpe = v_bpe.fit_transform(fit_bpe_strs) * 0.8
        te_bpe = v_bpe.transform(test_bpe_strs) * 0.8
        unseen_bpe = v_bpe.transform(unseen_bpe_strs) * 0.8 if unseen_texts else None
        del fit_bpe_strs, test_bpe_strs, unseen_bpe_strs, v_bpe
        gc.collect()
        X_core_5v = sparse.hstack([X_core_4v, tr_bpe], format="csr")
        X_te_5v = sparse.hstack([X_te_4v, te_bpe], format="csr")
        X_unseen_5v = sparse.hstack([X_unseen_4v, unseen_bpe], format="csr") if unseen_bpe is not None else None
    else:
        v_cw = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=75000, sublinear_tf=True, dtype=np.float32)
        tr_cw = v_cw.fit_transform(core_texts) * 0.8
        te_cw = v_cw.transform(test_texts) * 0.8
        unseen_cw = v_cw.transform(unseen_texts) * 0.8 if unseen_texts else None
        X_core_5v = sparse.hstack([X_core_4v, tr_cw], format="csr")
        X_te_5v = sparse.hstack([X_te_4v, te_cw], format="csr")
        X_unseen_5v = sparse.hstack([X_unseen_4v, unseen_cw], format="csr") if unseen_cw is not None else None

    comp_core_sw = core_sw.copy()
    # Boost non-Gemini hard classes, keeping 7 and 8 strictly symmetric
    hard_mask = np.isin(core_y, [c for c in HARD_CLASS_INDICES if c not in (7, 8)])
    comp_core_sw[hard_mask] *= 1.20

    svc_comp = LinearSVC(C=0.35, random_state=42, max_iter=3000, tol=1e-3, class_weight="balanced")
    svc_comp.fit(X_core_5v, core_y, sample_weight=comp_core_sw)
    P_svc_comp = softmax(svc_comp.decision_function(X_te_5v), axis=1)

    sgd_comp = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1500, random_state=42)
    sgd_comp.fit(X_core_5v, core_y, sample_weight=comp_core_sw)
    P_sgd_comp = sgd_comp.predict_proba(X_te_5v)

    # Compute baseline probabilities
    P_unseen_svc = (
        softmax(svc_comp.decision_function(X_unseen_5v), axis=1) if (X_unseen_5v is not None and X_unseen_5v.shape[0] > 0) else None
    )
    P_unseen_sgd = sgd_comp.predict_proba(X_unseen_5v) if (X_unseen_5v is not None and X_unseen_5v.shape[0] > 0) else None

    # Style Twin on groups_085
    P_twin_comp = P_svc_comp.copy()
    P_unseen_twin = P_unseen_svc.copy() if P_unseen_svc is not None else None
    if twin_rows_085.any() and len_tr_raw >= 18:
        X_train_only = X_core_5v[:len_tr_raw]
        u_tr, inv_tr = np.unique(df_tr["prompt_id"].values, return_inverse=True)
        counts_tr = np.bincount(inv_tr)
        vals_tr = (1.0 / counts_tr[inv_tr]).astype(np.float32)
        G = sparse.csr_matrix((vals_tr, (np.arange(len(inv_tr)), inv_tr)), shape=(len(inv_tr), len(u_tr)), dtype=np.float32)
        X_train_c = X_train_only - G @ (G.T @ X_train_only)

        svc_twin = LinearSVC(C=0.30, random_state=42, max_iter=2500)
        svc_twin.fit(X_train_c, core_y[:len_tr_raw])
        del X_train_only, G, X_train_c
        gc.collect()

        X_te_twin = X_te_5v[twin_rows_085]
        u_t, inv_t = np.unique(groups_085[twin_rows_085], return_inverse=True)
        counts_t = np.bincount(inv_t)
        vals_t = (1.0 / counts_t[inv_t]).astype(np.float32)
        G_t = sparse.csr_matrix((vals_t, (np.arange(len(inv_t)), inv_t)), shape=(len(inv_t), len(u_t)), dtype=np.float32)
        X_te_twin_c = X_te_twin - G_t @ (G_t.T @ X_te_twin)
        P_twin_comp[twin_rows_085] = softmax(svc_twin.decision_function(X_te_twin_c), axis=1)

        if X_unseen_5v is not None and X_unseen_5v.shape[0] > 0:
            P_unseen_twin = P_unseen_svc.copy()
            if (
                "df_unseen_holdout" in locals()
                and "prompt_id" in df_unseen_holdout.columns
                and len(df_unseen_holdout) == X_unseen_5v.shape[0]
            ):
                u_u, inv_u = np.unique(df_unseen_holdout["prompt_id"].values, return_inverse=True)
                counts_u = np.bincount(inv_u)
                twin_u_mask = (counts_u[inv_u] >= 2) & (counts_u[inv_u] <= 18)
                if np.any(twin_u_mask):
                    X_u_twin = X_unseen_5v[twin_u_mask]
                    u_u_sub, inv_u_sub = np.unique(df_unseen_holdout["prompt_id"].values[twin_u_mask], return_inverse=True)
                    counts_u_sub = np.bincount(inv_u_sub)
                    vals_u = (1.0 / counts_u_sub[inv_u_sub]).astype(np.float32)
                    G_u = sparse.csr_matrix((vals_u, (np.arange(len(inv_u_sub)), inv_u_sub)), shape=(len(inv_u_sub), len(u_u_sub)), dtype=np.float32)
                    X_unseen_twin_c = X_u_twin - G_u @ (G_u.T @ X_u_twin)
                    P_unseen_twin[twin_u_mask] = softmax(svc_twin.decision_function(X_unseen_twin_c), axis=1)
                    del G_u, X_unseen_twin_c, X_u_twin

        del svc_twin, X_te_twin, G_t, X_te_twin_c
        gc.collect()

    if X_unseen_5v is not None and X_unseen_5v.shape[0] > 0:
        if P_unseen_distort is not None:
            P_unseen_comp = 0.40 * P_unseen_svc + 0.25 * P_unseen_sgd + 0.15 * P_unseen_twin + 0.20 * P_unseen_distort
        else:
            P_unseen_comp = 0.55 * P_unseen_svc + 0.45 * P_unseen_sgd
        log_p_unseen_comp = np.log(np.clip(P_unseen_comp, 1e-9, 1.0))
    else:
        log_p_unseen_comp = None

    P_lex_comp = 0.40 * P_svc_comp + 0.25 * P_sgd_comp + 0.15 * P_twin_comp + 0.20 * P_distort
    P_lex_comp_final = P_lex_comp.copy()
    unseen_p_lex_comp_final = (
        P_unseen_comp.copy() if (X_unseen_5v is not None and X_unseen_5v.shape[0] > 0) else np.zeros((0, K), dtype=np.float32)
    )

    hw_c_sp = sparse.csr_matrix(hw_core * 0.25)
    hw_te_sp = sparse.csr_matrix(hw_te * 0.25)
    hw_u_sp = sparse.csr_matrix(hw_unseen * 0.25) if hw_unseen is not None else None

    # Disjoint Mass-Conserving Provider Family Refiners (prevents overlapping triple-counting)
    for grp_name, grp_cols in FAMILIES.items():
        grp_mask = np.isin(core_y, grp_cols)
        if grp_mask.sum() > 30 and len(np.unique(core_y[grp_mask])) == len(grp_cols):
            sub_y = core_y[grp_mask]
            sub_lab2i = {c: i for i, c in enumerate(grp_cols)}
            sub_y_mapped = np.array([sub_lab2i[y] for y in sub_y], dtype=int)

            c_val = 0.35 if grp_name == "glm" else 0.40
            X_tr_grp = sparse.hstack([X_core_5v[grp_mask], hw_c_sp[grp_mask]], format="csr")
            X_te_grp = sparse.hstack([X_te_5v, hw_te_sp], format="csr")
            X_u_grp = sparse.hstack([X_unseen_5v, hw_u_sp], format="csr") if X_unseen_5v is not None else None

            spec_svc = LinearSVC(C=c_val, random_state=42, max_iter=2000, tol=1e-3, class_weight="balanced")
            spec_svc.fit(X_tr_grp, sub_y_mapped, sample_weight=comp_core_sw[grp_mask])
            df_spec = spec_svc.decision_function(X_te_grp)
            logits_spec = np.column_stack([-df_spec, df_spec]) if df_spec.ndim == 1 else df_spec
            P_spec_plain = softmax(logits_spec, axis=1)

            train_mask = grp_mask[:len_tr_raw]
            has_centered_pair = train_mask.sum() >= 6 * len(grp_cols) and len(np.unique(core_y[:len_tr_raw][train_mask])) == len(grp_cols)
            if has_centered_pair:
                X_tr_pair = sparse.hstack([X_core_5v[:len_tr_raw][train_mask], hw_c_sp[:len_tr_raw][train_mask]], format="csr")
                y_tr_pair = np.array([sub_lab2i[y] for y in core_y[:len_tr_raw][train_mask]], dtype=int)
                prompts_pair = df_tr["prompt_id"].values[train_mask]
                u_p, inv_p = np.unique(prompts_pair, return_inverse=True)
                counts_p = np.bincount(inv_p)
                vals_p = (1.0 / np.maximum(counts_p[inv_p], 1)).astype(np.float32)
                G_p = sparse.csr_matrix((vals_p, (np.arange(len(inv_p)), inv_p)), shape=(len(inv_p), len(u_p)), dtype=np.float32)
                X_tr_pair_c = X_tr_pair - G_p @ (G_p.T @ X_tr_pair)

                clf_centered = LogisticRegression(C=0.25, class_weight="balanced", random_state=42, max_iter=1000)
                clf_centered.fit(X_tr_pair_c, y_tr_pair)

                P_spec = P_spec_plain.copy()
                if twin_rows_085.any():
                    X_te_sub = X_te_grp[twin_rows_085]
                    u_t, inv_t = np.unique(groups_085[twin_rows_085], return_inverse=True)
                    counts_t = np.bincount(inv_t)
                    vals_t = (1.0 / np.maximum(counts_t[inv_t], 1)).astype(np.float32)
                    G_t = sparse.csr_matrix((vals_t, (np.arange(len(inv_t)), inv_t)), shape=(len(inv_t), len(u_t)), dtype=np.float32)
                    X_te_pair_c = X_te_sub - G_t @ (G_t.T @ X_te_sub)
                    P_spec_centered = clf_centered.predict_proba(X_te_pair_c)
                    w_plain_s = 0.50 if grp_name == "glm" else 0.65
                    P_spec[twin_rows_085] = w_plain_s * P_spec_plain[twin_rows_085] + (1.0 - w_plain_s) * P_spec_centered
            else:
                P_spec = P_spec_plain

            beta_grp = 0.25
            P_prior = P_lex_comp_final[:, grp_cols]
            fam_mass = P_prior.sum(axis=1, keepdims=True)
            P_prior_norm = P_prior / (fam_mass + 1e-12)
            P_refined = (1.0 - beta_grp) * P_prior_norm + beta_grp * P_spec
            P_lex_comp_final[:, grp_cols] = fam_mass * P_refined

            if len(unseen_p_lex_comp_final) > 0 and X_u_grp is not None and X_u_grp.shape[0] > 0:
                df_spec_u = spec_svc.decision_function(X_u_grp)
                logits_spec_u = np.column_stack([-df_spec_u, df_spec_u]) if df_spec_u.ndim == 1 else df_spec_u
                P_spec_u_plain = softmax(logits_spec_u, axis=1)
                P_spec_u = P_spec_u_plain.copy()
                if (
                    has_centered_pair
                    and "df_unseen_holdout" in locals()
                    and "prompt_id" in df_unseen_holdout.columns
                    and len(df_unseen_holdout) == X_u_grp.shape[0]
                ):
                    u_u, inv_u = np.unique(df_unseen_holdout["prompt_id"].values, return_inverse=True)
                    counts_u = np.bincount(inv_u)
                    sz_u = counts_u[inv_u]
                    twin_u_mask = (sz_u >= 2) & (sz_u <= 18)
                    if twin_u_mask.any():
                        X_u_sub = X_u_grp[twin_u_mask]
                        u_t, inv_t = np.unique(df_unseen_holdout["prompt_id"].values[twin_u_mask], return_inverse=True)
                        counts_t = np.bincount(inv_t)
                        vals_t = (1.0 / np.maximum(counts_t[inv_t], 1)).astype(np.float32)
                        G_t = sparse.csr_matrix((vals_t, (np.arange(len(inv_t)), inv_t)), shape=(len(inv_t), len(u_t)), dtype=np.float32)
                        X_u_pair_c = X_u_sub - G_t @ (G_t.T @ X_u_sub)
                        P_spec_u_centered = clf_centered.predict_proba(X_u_pair_c)
                        w_plain_s = 0.50 if grp_name == "glm" else 0.65
                        P_spec_u[twin_u_mask] = w_plain_s * P_spec_u_plain[twin_u_mask] + (1.0 - w_plain_s) * P_spec_u_centered

                P_prior_u = unseen_p_lex_comp_final[:, grp_cols]
                fam_mass_u = P_prior_u.sum(axis=1, keepdims=True)
                P_prior_u_norm = P_prior_u / (fam_mass_u + 1e-12)
                P_refined_u = (1.0 - beta_grp) * P_prior_u_norm + beta_grp * P_spec_u
                unseen_p_lex_comp_final[:, grp_cols] = fam_mass_u * P_refined_u

    P_lex_comp_final /= P_lex_comp_final.sum(axis=1, keepdims=True).clip(min=1e-12)
    if len(unseen_p_lex_comp_final) > 0:
        unseen_p_lex_comp_final /= unseen_p_lex_comp_final.sum(axis=1, keepdims=True).clip(min=1e-12)

    # Save to disk
    os.makedirs(working_dir, exist_ok=True)
    np.save(working_dir / "hw_train.npy", hw_core[:len_tr_raw])
    np.save(working_dir / "hw_test.npy", hw_te)
    if hw_unseen is not None:
        np.save(working_dir / "hw_unseen.npy", hw_unseen)
    np.save(working_dir / "log_p_svc_main.npy", log_p_svc_main)
    np.save(working_dir / "p_lex_comp.npy", P_lex_comp_final)
    np.save(working_dir / "D_tfidf.npy", D)
    np.save(working_dir / "groups_085.npy", groups_085)
    np.save(working_dir / "log_p_handwriting.npy", lp_hw)
    np.save(working_dir / "rule_flags_test.npy", rule_flags_te)
    np.save(working_dir / "unseen_log_p_svc_main.npy", unseen_log_p_svc_main)
    np.save(working_dir / "unseen_p_lex_comp.npy", unseen_p_lex_comp_final)
    if log_p_distort is not None:
        np.save(working_dir / "log_p_distort.npy", log_p_distort)
    if log_p_unseen_distort is not None:
        np.save(working_dir / "unseen_log_p_distort.npy", log_p_unseen_distort)

    return {
        "log_p_svc_main": log_p_svc_main,
        "p_lex_comp": P_lex_comp_final,
        "D_tfidf": D,
        "groups_085": groups_085,
        "log_p_handwriting": lp_hw,
        "rule_flags_test": rule_flags_te,
        "unseen_log_p_svc_main": unseen_log_p_svc_main,
        "unseen_p_lex_comp": unseen_p_lex_comp_final,
        "hw_train": hw_core[:len_tr_raw],
        "hw_test": hw_te,
        "hw_unseen": hw_unseen,
        "log_p_distort": log_p_distort,
        "unseen_log_p_distort": log_p_unseen_distort,
    }


if __name__ == "__main__" and "run_submission_pipeline" not in globals() and "ENSEMBLE_CONFIG" not in globals():
    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        input_dir = Path(".")

    if len(sys.argv) >= 4:
        train_path = Path(sys.argv[1])
        test_path = Path(sys.argv[2])
        working_dir = Path(sys.argv[3])
    else:
        working_dir = Path("/kaggle/working")
        if not working_dir.exists():
            working_dir = Path("local_data")
        train_path, test_path, _ = resolve_dataset_paths(input_dir)

    working_dir.mkdir(parents=True, exist_ok=True)
    pool_files = resolve_all_pool_files(input_dir)

    tok_paths = list(input_dir.rglob("vocab.json"))
    qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""

    print(f"[Module 1 CLI] Running lexical engine: train={train_path}, test={test_path}, pools={[p.name for p in pool_files]}")
    run_lexical_engine(train_path, test_path, pool_files, qwen_tok_dir, working_dir)
    print("[Module 1 CLI] Lexical engine finished successfully.")
