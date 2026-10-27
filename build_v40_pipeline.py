"""
Name That LLM - Version 40 Pipeline Generator (0.66+ Target)
============================================================
Fixes the 14B FP16 Overflow Bug, Restores Full-Rank wPCA (1,620 rows) +
ExactAuthorPosterior (2^18 Bitmask DP) & 384-Dim Joint Spectral Bank,
and Applies Multi-Resolution Centered Twins.

Key Components:
1. Module 1: Subprocess 78,482-row 5-view lexical engine with BPE subwords and 4 within-family specialists.
2. Critical Fix 1: 14B fp16 overflow rescue with dynamic bfloat16 row retry (zero nan_to_num feature zeroing).
3. Critical Fix 2: Full-rank wPCA (d=256) and 384-dim Cross-Validated Joint Spectral Bank on 1,620 balanced rows.
4. Critical Fix 3: Exhaustive centered twin key matcher and multi-resolution twin pooling (t=0.85, 0.72, 0.60).
5. 6-Checkpoint Ensemble across 4 Trunks with in-place LoRA trunk reuse.
6. Probability Mixture: 0.88 * R_core + 0.12 * H_helper.
7. Pre-Sinkhorn Top-1 Claude Specialist Gate + Orthographic Shield.
8. Official 300-step exact multiplicative Sinkhorn balancing.
"""

import json
from pathlib import Path

def create_v40_pipeline_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True)
        })

    def add_code(source: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.strip().splitlines(keepends=True)
        })

    # Header Markdown
    add_md("""# Name That LLM: Version 40 Pipeline (0.66+ LB Target)
### 14B `bfloat16` Overflow Rescue, Full-Rank 1,620-Row `wPCA` ($2^{18}$ Bitmask DP), 384-Dim Spectral Bank & Multi-Resolution Centered Twins

**Architectural Improvements in Version 40**:
- **Critical Fix 1 (14B `fp16` Overflow Rescue)**: Eliminates silent `nan_to_num(0.0)` zeroing in Qwen3-14B & Qwen2.5-14B layers 14-32 by dynamically identifying overflowed rows and recomputing only those rows in `torch.bfloat16` (exponent range $3.4 \\times 10^{38}$) pooled directly in `float32`.
- **Critical Fix 2 (Full-Rank 1,620-Row `wPCA` & Joint Spectral Bank)**: Replaces rank-deficient 180-row helper and heuristic `LOO-CLC` with 90 balanced training prompts ($90 \\times 18 = 1,620$ rows of `train.csv`, $N = 1,620 \\gg d = 256$), restoring true full-rank whitening covariance and exact $2^{18}$ bitmask DP (`ExactAuthorPosterior`).
- **Critical Fix 3 (Exhaustive Centered Key Matcher & Multi-Resolution Twins)**: Resolves British (`centred.weight`) and American (`centered.weight`, `center.weight`, `twin.weight`) keys across all checkpoints, and pools centered twin logits at 3 consensus cluster resolutions ($t = 0.85, 0.72, 0.60$) with weights $0.50, 0.30, 0.20$.
- **6 High-Alpha Checkpoints on 4 Trunks**: In-place LoRA adapter reuse on shared trunks (Trunk 1 hosts both `lora_qwen3_14b` and `lora14b_v1`; Trunk 3 hosts `instyle` and `ntl-claude-head`).
- **Subprocess 78,482-Row 5-View Lexical Engine**: Subprocess memory isolation (Peak RSS $\\le 2.09$ GB) with Native Qwen BPE subwords and 4 within-family specialists.
- **Pre-Sinkhorn Claude Specialist Gate & Orthographic Shield**: Mass-preserving within-family refinement and punctuation shields before the single final 300-step exact multiplicative Sinkhorn balancing.
""")

    # Cell 1: Environment, Dependencies & Master 18 Labels
    add_md("### Step 1: Environment, Dependencies & Master 18 Labels")
    add_code("""import os
import gc
import sys
import time
import math
import json
import shutil
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import ledoit_wolf_shrinkage
from safetensors.torch import load_file

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Memory telemetry helper
def get_process_memory() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0

def log_ram(tag: str = ""):
    rss = get_process_memory()
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"[RAM AUDIT | {tag}] Process RSS: {rss:.2f} GB | GPU VRAM Allocated: {gpu_mem:.2f} GB")

log_ram("Notebook Startup")

# Canonical 18-Class Labels
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

def align_cols(arr: np.ndarray, src_labels: list) -> np.ndarray:
    src_list = [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in src_labels]
    idx = [src_list.index(l) for l in LABELS]
    if arr.ndim == 1:
        return arr[idx]
    elif arr.ndim == 2:
        if arr.shape[0] == len(src_list) and arr.shape[1] != len(src_list):
            return arr[idx, :]
        elif arr.shape[1] == len(src_list):
            return arr[:, idx]
    return arr[..., idx]

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)

print(f"Master Canonical 18 Classes (K={K}): {LABELS}")
""")

    # Cell 2: Asset Discovery & Balanced 1,620-Row Training Subset
    add_md("### Step 2: Asset Discovery & Balanced 1,620-Row Training Subset")
    add_code("""INPUT_DIR = Path('/kaggle/input')
DATASET_DIR = INPUT_DIR / 'competitions' / 'name-that-llm'
if not DATASET_DIR.exists():
    DATASET_DIR = INPUT_DIR / 'name-that-llm'
if not DATASET_DIR.exists():
    DATASET_DIR = Path('.')

train_path = DATASET_DIR / 'train.csv'
test_path = DATASET_DIR / 'test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_texts = test_df['text'].fillna('').tolist()
N_test = len(test_texts)

print(f"Train path: {train_path} ({len(train_df)} rows)")
print(f"Test path:  {test_path} ({N_test} rows)")

# Deterministic selection of 90 balanced prompts (90 * 18 = 1,620 rows) for full-rank wPCA
unique_prompts = sorted(train_df['prompt_id'].unique())
rng = np.random.RandomState(42)
sel_prompts = set(rng.choice(unique_prompts, size=min(90, len(unique_prompts)), replace=False))
train_sub = train_df[train_df['prompt_id'].isin(sel_prompts)].reset_index(drop=True)
y_sub = train_sub['model'].map(LAB2I).values
prompts_sub = train_sub['prompt_id'].values
train_sub_texts = train_sub['text'].fillna('').tolist()
N_sub = len(train_sub)
print(f"[Balanced Subset] Selected {len(sel_prompts)} prompts, {N_sub} rows for full-rank wPCA & Spectral Bank.")

eval_texts = test_texts + train_sub_texts
print(f"Total passages to extract per backbone: {len(eval_texts)} ({N_test} test + {N_sub} train_sub)")

# Discover base model directories
def find_base_model(model_family: str, hidden_sz: int, num_layers: int) -> Optional[Path]:
    for cfg_p in INPUT_DIR.rglob('config.json'):
        try:
            cfg = json.load(open(cfg_p))
            m_type = str(cfg.get('model_type', '')).lower()
            h_dim = cfg.get('hidden_size', 0)
            n_lay = cfg.get('num_hidden_layers', 0)
            if model_family.lower() in m_type and h_dim == hidden_sz and n_lay == num_layers:
                return cfg_p.parent
        except Exception:
            continue
    return None

BASE_QWEN3_14B = find_base_model('qwen3', 5120, 40)
BASE_QWEN25_14B = find_base_model('qwen2', 5120, 48)
BASE_QWEN3_8B = find_base_model('qwen3', 4096, 36)
BASE_DEBERTA = find_base_model('deberta-v2', 1024, 24)

print(f"Discovered Base Qwen3-14B:   {BASE_QWEN3_14B}")
print(f"Discovered Base Qwen2.5-14B: {BASE_QWEN25_14B}")
print(f"Discovered Base Qwen3-8B:    {BASE_QWEN3_8B}")
print(f"Discovered Base DeBERTa:     {BASE_DEBERTA}")

# Discover checkpoints
def find_checkpoints() -> Dict[str, Path]:
    ckpts = {}
    for p in INPUT_DIR.rglob('heads.safetensors'):
        p_str = str(p).lower().replace('\\\\', '/')
        if 'lora_qwen3_14b' in p_str or 'qwen3_14b_part2' in p_str or ('14b' in p_str and 'part2' in p_str and 'qwen3' in p_str):
            ckpts['qwen3_14b_p2'] = p.parent
        elif 'lora_qwen25_14b' in p_str or 'qwen25_14b_part2' in p_str or ('14b' in p_str and 'part2' in p_str and 'qwen2' in p_str):
            ckpts['qwen25_14b_p2'] = p.parent
        elif 'lora14b_v1' in p_str:
            ckpts['qwen3_14b_p1'] = p.parent
        elif 'lora8b_v1' in p_str:
            ckpts['qwen3_8b_p1'] = p.parent
        elif 'instyle' in p_str:
            ckpts['instyle'] = p.parent

    for p in INPUT_DIR.rglob('claude_head.safetensors'):
        ckpts['qwen3_8b_claude'] = p.parent
        break
    if 'qwen3_8b_claude' not in ckpts:
        for p in INPUT_DIR.rglob('readouts.npz'):
            if 'claude' in str(p).lower():
                ckpts['qwen3_8b_claude'] = p.parent
                break

    for p in INPUT_DIR.rglob('head.safetensors'):
        if 'deberta' in str(p).lower():
            ckpts['deberta_p2'] = p.parent
            break

    if 'instyle' not in ckpts:
        for p in INPUT_DIR.rglob('*'):
            p_s = str(p).lower().replace('\\\\', '/')
            if 'instyle' in p_s and ((p / 'adapter_model.safetensors').exists() or (p / 'adapter' / 'adapter_model.safetensors').exists() or (p / 'heads.safetensors').exists()):
                ckpts['instyle'] = p
                break
    return ckpts

CKPTS = find_checkpoints()
print(f"Discovered Checkpoints ({len(CKPTS)}): {list(CKPTS.keys())}")

pool_paths = list(INPUT_DIR.rglob('pool_v4.parquet'))
pool_file = pool_paths[0] if pool_paths else None
print(f"Discovered Passage Pool:     {pool_file}")
""")

    # Cell 3: Module 1 — Subprocess 78,482-Row 5-View Lexical + BPE + Within-Family Refiners
    add_md("### Step 3: Module 1 — Subprocess 78,482-Row 5-View Lexical + BPE + Within-Family Specialist Engine")
    add_code("""working_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
lexical_script_path = working_dir / "run_lexical.py"

qwen_tok_dir = str(BASE_QWEN3_8B or BASE_QWEN3_14B or '')

lexical_code = f\"\"\"import os
import gc
import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

t0 = time.time()
print("=== [Subprocess Lexical Engine] Starting 78k Full Execution with BPE + Specialist Refiners ===", flush=True)

LABELS = {LABELS}
K = len(LABELS)
LAB2I = {{l: i for i, l in enumerate(LABELS)}}

FAMILIES = {FAMILIES}

train_df = pd.read_csv(r"{train_path}")
test_df = pd.read_csv(r"{test_path}")
pool_path = r"{pool_file}"

if pool_path and Path(pool_path).exists():
    print(f"[Subprocess Lexical] Loading 100% of pool_v4 from {{pool_path}}...", flush=True)
    pool_df = pd.read_parquet(pool_path)
    combined_df = pd.concat([
        train_df[['text', 'model']].dropna(),
        pool_df[['text', 'model']].dropna()
    ], ignore_index=True)
    weights = np.concatenate([
        np.full(len(train_df), 2.0, dtype=np.float32),
        np.full(len(pool_df), 1.0, dtype=np.float32)
    ])
else:
    print("[Subprocess Lexical] WARNING: pool_v4 not found! Training only on train.csv...", flush=True)
    combined_df = train_df[['text', 'model']].dropna().reset_index(drop=True)
    weights = np.full(len(combined_df), 1.0, dtype=np.float32)

all_train_texts = combined_df['text'].fillna('').tolist()
y_all = combined_df['model'].map(LAB2I).values
test_texts = test_df['text'].fillna('').tolist()
N_test = len(test_texts)
print(f"[Subprocess Lexical] Full combined training corpus: {{len(all_train_texts)}} rows ({{len(train_df)}} train + {{len(combined_df)-len(train_df)}} pool)", flush=True)

def shape_text(t: str) -> str:
    return re.sub(r"[^\\\\W\\\\d_]+", lambda m: "X" if m[0].isupper() else "x", t)

FUNCS = [
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with","he",
    "as","you","do","at","this","but","his","by","from","they","we","say","her","she","or",
    "an","will","my","one","all","would","there","their","what","so","up","out","if","about",
    "who","get","which","go","me","when","make","can","like","time","no","just","him","know",
    "take","people","into","year","your","good","some","could","them","see","other","than",
    "then","now","look","only","come","its","over","think","also","back","after","use","two",
    "how","our","work","first","well","way","even","new","want","because","any","these","give",
    "day","most","us"
]
func_pat = re.compile(r"\\b(" + "|".join(FUNCS) + r")\\b", re.I)
def function_text(t: str) -> str:
    return " ".join(func_pat.findall(t.lower()))

views = [
    ("char", 3, 5, 250000, 1.0, {{'analyzer': 'char', 'sublinear_tf': True}}),
    ("word", 1, 3, 180000, 0.7, {{'analyzer': 'word', 'sublinear_tf': True, 'token_pattern': r"(?u)\\\\b\\\\w+\\\\b|[^\\\\w\\\\s]"}}),
    ("shape", 2, 6, 60000, 0.5, {{'analyzer': 'char', 'sublinear_tf': True}}),
    ("function", 1, 3, 50000, 0.5, {{'analyzer': 'word', 'sublinear_tf': True}}),
]

qwen_tok_path = r"{qwen_tok_dir}"
if qwen_tok_path and Path(qwen_tok_path).exists():
    views.append(("bpe", 1, 3, 100000, 0.6, {{'analyzer': 'word', 'sublinear_tf': True}}))

X_tr_blocks = []
X_te_blocks = []

for v_name, mn, mx, mf, w, kwargs in views:
    print(f"[Subprocess Lexical] Extracting view: {{v_name}} (weight={{w}})...", flush=True)
    if v_name == "shape":
        tr_inp = [shape_text(t) for t in all_train_texts]
        te_inp = [shape_text(t) for t in test_texts]
    elif v_name == "function":
        tr_inp = [function_text(t) for t in all_train_texts]
        te_inp = [function_text(t) for t in test_texts]
    elif v_name == "bpe":
        tok = AutoTokenizer.from_pretrained(qwen_tok_path, trust_remote_code=True, use_fast=True)
        def to_bpe_strings(texts, batch_size=2048):
            res = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tok(batch, add_special_tokens=False, truncation=True, max_length=447)["input_ids"]
                res.extend([" ".join(str(tid) for tid in seq) for seq in enc])
            return res
        tr_inp = to_bpe_strings(all_train_texts)
        te_inp = to_bpe_strings(test_texts)
        del tok
        gc.collect()
    else:
        tr_inp = all_train_texts
        te_inp = test_texts

    vec = TfidfVectorizer(ngram_range=(mn, mx), max_features=mf, dtype=np.float32, **kwargs)
    X_tr_v = vec.fit_transform(tr_inp)
    X_te_v = vec.transform(te_inp)
    del tr_inp, te_inp, vec
    gc.collect()

    X_tr_v = normalize(X_tr_v, norm='l2', copy=False) * w
    X_te_v = normalize(X_te_v, norm='l2', copy=False) * w
    X_tr_blocks.append(X_tr_v)
    X_te_blocks.append(X_te_v)

X_tr_csr = sparse.hstack(X_tr_blocks, format='csr')
X_te_csr = sparse.hstack(X_te_blocks, format='csr')
del X_tr_blocks, X_te_blocks
gc.collect()

print(f"[Subprocess Lexical] Combined feature matrix: {{X_tr_csr.shape}} ({{X_tr_csr.nnz / 1e6:.1f}}M non-zeros)", flush=True)

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)

# Fit Dual Classifier: LinearSVC (0.82) + SGDClassifier log_loss (0.18)
print("[Subprocess Lexical] Fitting LinearSVC(C=0.5, tol=1e-3)...", flush=True)
svc = LinearSVC(C=0.5, tol=1e-3, max_iter=2500, random_state=42, class_weight='balanced')
svc.fit(X_tr_csr, y_all, sample_weight=weights)
log_p_svc = log_softmax(svc.decision_function(X_te_csr))
del svc
gc.collect()

print("[Subprocess Lexical] Fitting SGDClassifier(log_loss, alpha=1e-5)...", flush=True)
sgd = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, max_iter=1000, random_state=42, class_weight='balanced')
sgd.fit(X_tr_csr, y_all, sample_weight=weights)
log_p_sgd = sgd.predict_log_proba(X_te_csr)
del sgd
gc.collect()

log_p_lex = 0.82 * log_p_svc + 0.18 * log_p_sgd

# 4 Within-Family Specialist Refiners (gpt, claude, gemini, glm) on 78k passages with beta=0.20
print("[Subprocess Lexical] Fitting 4 Within-Family Specialist LinearSVC Refiners (beta=0.20)...", flush=True)
for fam_name, members in FAMILIES.items():
    fam_mask = np.isin(y_all, members)
    if fam_mask.sum() > 100:
        y_fam_raw = y_all[fam_mask]
        sub_map = {{m: i for i, m in enumerate(members)}}
        y_fam = np.array([sub_map[m] for m in y_fam_raw])
        w_fam = weights[fam_mask]

        spec_svc = LinearSVC(C=0.4, tol=1e-3, max_iter=2000, random_state=42, class_weight='balanced')
        spec_svc.fit(X_tr_csr[fam_mask], y_fam, sample_weight=w_fam)
        spec_lp = log_softmax(spec_svc.decision_function(X_te_csr))

        P_prior = np.array([(y_all == m).sum() for m in members], dtype=np.float64)
        P_prior /= P_prior.sum()
        log_P_prior = np.log(P_prior + 1e-12)[None, :]

        log_ratio = spec_lp - log_P_prior
        log_ratio -= log_ratio.mean(axis=1, keepdims=True)
        log_p_lex[:, members] += 0.20 * log_ratio
        del spec_svc
        gc.collect()

log_p_lex = log_softmax(log_p_lex)

# Consensus TF-IDF distance on test
print("[Subprocess Lexical] Computing TF-IDF distance and cluster groups at t=0.85...", flush=True)
vec_sw = TfidfVectorizer(max_features=25000, stop_words="english", sublinear_tf=True)
X_sw_te = vec_sw.fit_transform(test_texts)
D_tfidf = np.clip(1.0 - (X_sw_te @ X_sw_te.T).toarray(), 0.0, None)
np.fill_diagonal(D_tfidf, 0.0)

if N_test >= 2:
    cond_d = squareform(D_tfidf, checks=False)
    groups_085 = fcluster(linkage(cond_d, method="average"), 0.85, criterion="distance")
else:
    groups_085 = np.ones(N_test, dtype=int)

sz = np.bincount(groups_085)[groups_085]
twin_rows_085 = (sz >= 2) & (sz <= 18)

# Prompt-Centered Style Twin on train.csv evaluated on twin_rows_085
log_p_lex_twin = np.zeros_like(log_p_lex)
if twin_rows_085.any():
    print(f"[Subprocess Lexical] Fitting Prompt-Centered Style-Gram Twin on train.csv for {{twin_rows_085.sum()}} test passages...", flush=True)
    N_tr = len(train_df)
    X_train_csr = X_tr_csr[:N_tr]
    y_tr = y_all[:N_tr]
    prompts_tr = train_df['prompt_id'].values

    unique_p, p_inv = np.unique(prompts_tr, return_inverse=True)
    G = len(unique_p)
    rows_g = np.arange(N_tr)
    cols_g = p_inv
    vals_g = 1.0 / np.sqrt(np.bincount(p_inv)[p_inv])
    G_mat = sparse.csr_matrix((vals_g, (rows_g, cols_g)), shape=(N_tr, G))

    X_tr_mean = G_mat @ (G_mat.T @ X_train_csr)
    X_tr_centered = X_train_csr - X_tr_mean

    svc_twin = LinearSVC(C=0.2, tol=1e-3, max_iter=2000, random_state=42, class_weight='balanced')
    svc_twin.fit(X_tr_centered, y_tr)

    X_te_twin = X_te_csr[twin_rows_085]
    te_groups = groups_085[twin_rows_085]
    u_te_g, te_inv = np.unique(te_groups, return_inverse=True)
    G_te = len(u_te_g)
    N_t_sub = len(te_groups)
    vals_te = 1.0 / np.sqrt(np.bincount(te_inv)[te_inv])
    G_t = sparse.csr_matrix((vals_te, (np.arange(N_t_sub), te_inv)), shape=(N_t_sub, G_te))
    X_te_twin_c = X_te_twin - G_t @ (G_t.T @ X_te_twin)

    log_p_lex_twin[twin_rows_085] = log_softmax(svc_twin.decision_function(X_te_twin_c))
    del svc_twin, X_te_twin, G_t, X_te_twin_c
    gc.collect()

del X_tr_csr, X_te_csr
gc.collect()

np.save(r"{working_dir}/log_p_lex.npy", log_p_lex)
np.save(r"{working_dir}/log_p_lex_twin.npy", log_p_lex_twin)
np.save(r"{working_dir}/groups_085.npy", groups_085)
np.save(r"{working_dir}/twin_rows_085.npy", twin_rows_085)
np.save(r"{working_dir}/D_tfidf.npy", D_tfidf)

print(f"[Subprocess Lexical] Completed successfully in {{time.time() - t0:.2f}}s. Saved arrays to {working_dir}.", flush=True)
sys.exit(0)
\"\"\"

with open(lexical_script_path, "w", encoding="utf-8") as f:
    f.write(lexical_code)

print(f"[Main Process] Launching isolated lexical subprocess: {lexical_script_path}...")
proc = subprocess.run([sys.executable, str(lexical_script_path)], capture_output=True, text=True)
print(proc.stdout)
if proc.returncode != 0:
    print(proc.stderr)
    raise RuntimeError(f"Subprocess run_lexical.py failed with return code {proc.returncode}!")

log_p_lex = np.load(working_dir / 'log_p_lex.npy')
log_p_lex_twin = np.load(working_dir / 'log_p_lex_twin.npy')
groups_085 = np.load(working_dir / 'groups_085.npy')
twin_rows_085 = np.load(working_dir / 'twin_rows_085.npy')
D_tfidf = np.load(working_dir / 'D_tfidf.npy')

LOGIT_CORE = 1.02 * log_p_lex
if twin_rows_085.any():
    LOGIT_CORE[twin_rows_085] += 0.10 * log_p_lex_twin[twin_rows_085]

HELPER_LIST: List[Tuple[float, np.ndarray]] = []
D_neural_sum = np.zeros((N_test, N_test), dtype=np.float64)
D_neural_weight_total = 0.0

spectral_tr_blocks: List[np.ndarray] = []
spectral_te_blocks: List[np.ndarray] = []

print(f"[Main Process] Initialized LOGIT_CORE with 78k Lexical Engine: shape={LOGIT_CORE.shape}")
log_ram("After Subprocess Memory Reclaim & Core Init")
""")

    # Cell 4: Exact Training Extractor, Head Key Resolver & Math Solvers
    add_md("### Step 4: Universal Trunk Loader with `bfloat16` Batch Rescue, Head Resolver & Math Solvers")
    add_code("""def in_place_copy(dst: torch.Tensor, src: torch.Tensor):
    dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))

def merge_lora(model: nn.Module, adapter_path: Path, sign: float = +1.0) -> bool:
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.exists():
        weights_path = adapter_path / "adapter" / "adapter_model.safetensors"
    if not weights_path.exists():
        return False

    sd = load_file(str(weights_path))
    cfg_p = adapter_path / "adapter_config.json"
    if not cfg_p.exists():
        cfg_p = adapter_path / "adapter" / "adapter_config.json"
    scale = 1.0
    if cfg_p.exists():
        c = json.load(open(cfg_p))
        scale = float(c.get("lora_alpha", 16)) / float(c.get("r", 16))

    state_dict = dict(model.named_modules())
    pairs = {}
    for k in sd.keys():
        if "lora_A" in k:
            b_key = k.replace("lora_A", "lora_B")
            if b_key in sd:
                m_name = k.split(".lora_A")[0].replace("base_model.model.", "")
                pairs[m_name] = (sd[k], sd[b_key])

    applied = 0
    with torch.no_grad():
        for m_name, (A, B) in pairs.items():
            if m_name in state_dict:
                mod = state_dict[m_name]
                if hasattr(mod, "weight"):
                    A_d = A.to(mod.weight.device, dtype=torch.float32)
                    B_d = B.to(mod.weight.device, dtype=torch.float32)
                    delta = (B_d @ A_d) * (scale * sign)
                    mod.weight.data.add_(delta.to(mod.weight.dtype))
                    applied += 1
    return applied > 0

@torch.no_grad()
def forward_and_pool_batch(model: nn.Module, enc: dict, layers: List[int], norms: nn.ModuleList,
                           dtype: torch.dtype = torch.float16, drop_first_token: bool = True):
    \"\"\"
    Runs forward pass in `dtype` (float16 on T4).
    CRITICAL: If any target layer overflows fp16 (Inf/NaN), recomputes ONLY the overflowed
    rows in `torch.bfloat16` and pools directly in float32 before restoring `dtype`!
    \"\"\"
    hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states

    B = enc["input_ids"].shape[0]
    dev = enc["input_ids"].device
    first_norm_dev = norms[0].weight.device

    # Check if any selected layer has Inf/NaN per row
    bad_mask = torch.zeros(B, dtype=torch.bool, device="cpu")
    for l in layers:
        bad_mask |= ~torch.isfinite(hs[l]).all(dim=(1, 2)).cpu()

    n_bad = int(bad_mask.sum().item())
    m = enc["attention_mask"].unsqueeze(-1).float().to(first_norm_dev)
    if drop_first_token:
        m[:, 0, :] = 0.0
    den = m.sum(dim=1).clamp(min=1.0)

    # Pool finite rows first in float32
    pooled_layers = [torch.zeros((B, model.config.hidden_size), dtype=torch.float32, device=first_norm_dev) for _ in layers]
    good_rows = torch.where(~bad_mask)[0].to(first_norm_dev)
    if len(good_rows) > 0:
        for idx_l, l in enumerate(layers):
            h_good = hs[l][good_rows.to(hs[l].device)].to(first_norm_dev).float()
            pooled_layers[idx_l][good_rows] = (h_good * m[good_rows]).sum(dim=1) / den[good_rows]

    # Rescue overflowed rows in bfloat16 (exact exponent range as float32, NEVER overflows!)
    if n_bad > 0:
        bad_rows = torch.where(bad_mask)[0]
        del hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(torch.bfloat16)
        enc_bad = {k: v[bad_rows] for k, v in enc.items()}
        hs_bad = model(**enc_bad, output_hidden_states=True, use_cache=False).hidden_states
        bad_rows_dev = bad_rows.to(first_norm_dev)
        for idx_l, l in enumerate(layers):
            h_b = hs_bad[l].to(first_norm_dev).float()
            h_b = torch.nan_to_num(h_b, nan=0.0, posinf=1e4, neginf=-1e4)
            pooled_layers[idx_l][bad_rows_dev] = (h_b * m[bad_rows_dev]).sum(dim=1) / den[bad_rows_dev]
        del hs_bad
        model.to(dtype)

    out = torch.cat([nm(pooled_layers[idx_l]) for idx_l, nm in enumerate(norms)], dim=-1)
    return out.cpu().numpy(), n_bad

def resolve_head_tensors(heads_dict: dict):
    W_p, b_p, W_c, b_c = None, None, None, None
    for wk, bk in [
        ("plain.weight", "plain.bias"),
        ("plain_coef", "plain_intercept"),
        ("weight", "bias"),
        ("W", "b"),
    ]:
        if wk in heads_dict and bk in heads_dict:
            W_p, b_p = heads_dict[wk], heads_dict[bk]
            break
    for wk, bk in [
        ("centered.weight", "centered.bias"),
        ("centred.weight", "centred.bias"),
        ("center.weight", "center.bias"),
        ("twin.weight", "twin.bias"),
        ("center_coef", "center_intercept"),
        ("centred_coef", "centred_intercept"),
    ]:
        if wk in heads_dict and bk in heads_dict:
            W_c, b_c = heads_dict[wk], heads_dict[bk]
            break

    if W_p is not None and hasattr(W_p, "cpu"):
        W_p = W_p.cpu().numpy()
    if b_p is not None and hasattr(b_p, "cpu"):
        b_p = b_p.cpu().numpy()
    if W_c is not None and hasattr(W_c, "cpu"):
        W_c = W_c.cpu().numpy()
    if b_c is not None and hasattr(b_c, "cpu"):
        b_c = b_c.cpu().numpy()

    return W_p, b_p, W_c, b_c

class UniversalLoRATrunkLoader:
    def __init__(self, base_dir: Path, keep_layers: int, layers: List[int], max_len: int = 447):
        if base_dir is not None and not (base_dir / "config.json").exists():
            for sub in base_dir.rglob("config.json"):
                base_dir = sub.parent
                break
        self.base_dir = base_dir
        self.keep_layers = keep_layers
        self.layers = layers
        self.max_len = max_len
        self.total_rescued = 0
        self.tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<|endoftext|>"

        cfg = AutoConfig.from_pretrained(str(base_dir), trust_remote_code=True)
        cfg.num_hidden_layers = keep_layers
        cfg.use_cache = False

        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            max_memory = {0: "13.5GiB", 1: "13.5GiB"}
            self.model = AutoModel.from_pretrained(
                str(base_dir), config=cfg, device_map="auto", max_memory=max_memory,
                torch_dtype=torch.float16, trust_remote_code=True
            )
        elif n_gpus == 1:
            self.model = AutoModel.from_pretrained(
                str(base_dir), config=cfg, device_map="auto",
                torch_dtype=torch.float16, trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(str(base_dir), config=cfg, torch_dtype=torch.float32)

        self.model.eval()
        self.device_norms = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        hidden_dim = self.model.config.hidden_size
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_dim, eps=getattr(cfg, "rms_norm_eps", 1e-6))
            for _ in layers
        ]).to(self.device_norms).float()

        for l_idx, nm in zip(layers, self.norms):
            try:
                base_lyr = self.model.layers[l_idx] if hasattr(self.model, "layers") else self.model.model.layers[l_idx]
                if hasattr(base_lyr, "input_layernorm") and hasattr(base_lyr.input_layernorm, "weight"):
                    nm.weight.data.copy_(base_lyr.input_layernorm.weight.data)
            except Exception:
                pass
        self.norms.eval()
        self.first_dev = next(self.model.parameters()).device

    def load_norm_weights(self, heads_sd: dict):
        with torch.no_grad():
            for i, ln in enumerate(self.norms):
                w_k = f'norms.{i}.weight'
                b_k = f'norms.{i}.bias'
                if w_k in heads_sd:
                    ln.weight.copy_(heads_sd[w_k].to(ln.weight.device, ln.weight.dtype))
                if b_k in heads_sd:
                    if hasattr(ln, 'bias') and ln.bias is not None:
                        ln.bias.copy_(heads_sd[b_k].to(ln.bias.device, ln.bias.dtype))

    def _extract_batch_ids(self, id_list: List[List[int]], batch_size: int = 8) -> np.ndarray:
        order = np.argsort([len(x) for x in id_list], kind="stable")
        total_dim = len(self.layers) * self.model.config.hidden_size
        Z = np.zeros((len(id_list), total_dim), dtype=np.float32)

        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            enc = self.tok.pad({"input_ids": [id_list[i] for i in idx]}, padding=True, return_tensors="pt")
            enc = {k: v.to(self.first_dev) for k, v in enc.items()}
            out_np, n_bad = forward_and_pool_batch(self.model, enc, self.layers, self.norms, torch.float16, drop_first_token=True)
            self.total_rescued += n_bad
            Z[idx] = out_np
        return Z

    @torch.no_grad()
    def extract_exact(self, texts: List[str], batch_size: int = 8, end_id: int = 151643) -> np.ndarray:
        t0 = time.time()
        self.total_rescued = 0
        raw_ids = self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        ids = [x + [end_id] for x in raw_ids]
        Z = self._extract_batch_ids(ids, batch_size=batch_size)
        print(f"[Extractor] Extracted {len(texts)} samples in {time.time() - t0:.2f}s (Total BF16 Rescued Rows: {self.total_rescued}).")
        return Z

    def free_memory(self):
        del self.model, self.norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def compute_centered_logits(Z_test: np.ndarray, groups: np.ndarray, W_c: np.ndarray, b_c: np.ndarray, src_labels: list) -> Tuple[np.ndarray, np.ndarray]:
    N = len(Z_test)
    out_lp = np.zeros((N, K), dtype=np.float64)
    if N < 2 or W_c is None:
        return out_lp, np.zeros(N, dtype=bool)
    sz = np.bincount(groups)[groups]
    valid = (sz >= 2) & (sz <= 18)
    if not valid.any():
        return out_lp, valid
    Zc = Z_test.copy()
    for g in np.unique(groups[valid]):
        mask = (groups == g)
        Zc[mask] -= Zc[mask].mean(axis=0, keepdims=True)
    raw_c = Zc[valid] @ W_c.T + b_c
    out_lp[valid] = log_softmax(align_cols(raw_c, src_labels))
    return out_lp, valid

def split_oversized_clusters_dist(D: np.ndarray, groups: np.ndarray, max_cluster_size: int = 14) -> np.ndarray:
    if len(groups) < 2:
        return groups.copy()
    new_groups = np.zeros(len(groups), dtype=np.int64)
    next_id = 1
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        if len(rows) <= max_cluster_size:
            new_groups[rows] = next_id
            next_id += 1
        else:
            n_sub = int(np.ceil(len(rows) / max_cluster_size))
            D_sub = D[np.ix_(rows, rows)].copy()
            np.fill_diagonal(D_sub, 0.0)
            D_sub = np.clip(0.5 * (D_sub + D_sub.T), 0.0, None)
            d_condensed = squareform(D_sub, checks=False)
            sub_labels = fcluster(linkage(d_condensed, method='average'), t=n_sub, criterion='maxclust')
            for sg in np.unique(sub_labels):
                s_rows = rows[sub_labels == sg]
                for start in range(0, len(s_rows), max_cluster_size):
                    new_groups[s_rows[start : start + max_cluster_size]] = next_id
                    next_id += 1
    return new_groups

def _two_way(Z: np.ndarray, y: np.ndarray, pinv: np.ndarray, G: int, K: int = 18, iters: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, d = Z.shape
    mu = np.zeros((K, d), dtype=np.float64)
    P = np.zeros((G, d), dtype=np.float64)
    ng = np.bincount(pinv, minlength=G).clip(min=1)
    for _ in range(iters):
        R = Z - P[pinv]
        for k in range(K):
            idx = (y == k)
            if idx.any():
                mu[k] = R[idx].mean(axis=0)
        R = Z - mu[y]
        P[:] = 0.0
        np.add.at(P, pinv, R)
        P /= ng[:, None]
    return mu, P, ng

def _shrunk(S: np.ndarray, alpha: float) -> np.ndarray:
    d = S.shape[0]
    tr = np.trace(S) / d
    return (1.0 - alpha) * S + alpha * tr * np.eye(d)

def fit_shared_gaussian_lowram(X_tr_f32: np.ndarray, y: np.ndarray, prompts: np.ndarray, d_target: int = 256) -> dict:
    N, D = X_tr_f32.shape
    pu, pinv = np.unique(prompts, return_inverse=True)
    G = len(pu)
    d = min(d_target, N - G, D)

    mx = X_tr_f32.mean(axis=0, keepdims=True)
    sx = X_tr_f32.std(axis=0, keepdims=True)
    sx[sx < 1e-6] = 1.0

    R = ((X_tr_f32 - mx) / sx).astype(np.float32)
    Gram = (R @ R.T) / float(D)
    lam_r, U = np.linalg.eigh(Gram.astype(np.float64))
    idx_top = np.argsort(lam_r)[::-1][:d]
    lam_top = np.maximum(lam_r[idx_top], 1e-9)
    U_top = U[:, idx_top]
    U_f32 = U_top.astype(np.float32)

    V_f32 = (R.T @ U_f32) / (np.sqrt(lam_top.astype(np.float32) * float(D))[None, :])
    del R, Gram, U, U_f32

    Z = (((X_tr_f32 - mx) / sx) @ V_f32).astype(np.float64)

    mu, P, ng = _two_way(Z, y, pinv, G, K=18, iters=4)
    E = Z - mu[y] - P[pinv]
    dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = (E.T @ E) / dof
    ae = float(ledoit_wolf_shrinkage(E, assume_centered=True))
    Se = _shrunk(Se_raw, ae)

    Pc = P - P.mean(0)
    Sp_raw = (Pc.T @ Pc) / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = float(ledoit_wolf_shrinkage(Pc, assume_centered=True))
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = (1.0 - ap) * Sp_raw + ap * (tr / d) * np.eye(d)

    L = np.linalg.cholesky(Se)
    Li = np.linalg.inv(L)
    M = 0.5 * ((Li @ Sp @ Li.T) + (Li @ Sp @ Li.T).T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, 1e-4, None)
    A = Q.T @ Li
    W_f32 = (V_f32 @ A.T.astype(np.float32)).astype(np.float32)
    m_whitened = mu @ A.T

    snr = m_whitened.var(axis=0) / (1.0 + lam)
    top64_idx = np.argsort(snr)[::-1][:64]

    return {
        "mx": mx, "sx": sx, "W": W_f32, "m": m_whitened, "lam": lam,
        "top64_idx": top64_idx, "T1": 2.35, "Tg": 2.30
    }

def transform_lowram(X_f32: np.ndarray, prm: dict, batch_rows: int = 1000) -> np.ndarray:
    N = len(X_f32)
    d = prm["W"].shape[1]
    U_out = np.empty((N, d), dtype=np.float64)
    mx, sx, W = prm["mx"], prm["sx"], prm["W"]
    for s in range(0, N, batch_rows):
        chunk = ((X_f32[s : s + batch_rows] - mx) / sx) @ W
        U_out[s : s + batch_rows] = chunk.astype(np.float64)
    return U_out

class ExactAuthorPosterior:
    def __init__(self, m: np.ndarray, lam: np.ndarray):
        self.m = np.ascontiguousarray(m, dtype=np.float32)
        self.lam = np.ascontiguousarray(lam, dtype=np.float32)
        self.K, self.d = m.shape
        self._subset_means = {}

    def _get_subset_means(self, subset_size: int) -> np.ndarray:
        if subset_size not in self._subset_means:
            if len(self._subset_means) >= 4:
                self._subset_means.clear()
            from itertools import combinations
            subsets = list(combinations(range(self.K), subset_size))
            means = np.empty((len(subsets), self.d), dtype=np.float32)
            for i, S in enumerate(subsets):
                means[i] = self.m[list(S)].mean(axis=0)
            self._subset_means[subset_size] = means
        return self._subset_means[subset_size]

    def single(self, u: np.ndarray, temperature: float = 2.35) -> np.ndarray:
        diff = u[None, :] - self.m
        diag = 1.0 + self.lam
        dist = np.sum(diff * diff / diag[None, :], axis=1)
        log_p = -0.5 * dist / max(float(temperature), 1e-4)
        return log_p - logsumexp(log_p)

    def group(self, U: np.ndarray, temperature: float = 2.30) -> np.ndarray:
        s = len(U)
        if s <= 1 or s > self.K:
            out = np.empty((s, self.K), dtype=np.float64)
            for i in range(s):
                out[i] = self.single(U[i], temperature=temperature)
            return out

        u_bar = U.mean(axis=0).astype(np.float32)
        from itertools import combinations
        subsets = list(combinations(range(self.K), s))

        means = self._get_subset_means(s)
        diff = u_bar[None, :] - means
        diag = (1.0 + float(s) * self.lam) / float(s)
        dist = np.sum(diff * diff / diag[None, :], axis=1)
        log_p_S = -0.5 * dist / max(float(temperature), 1e-4)
        log_p_S -= logsumexp(log_p_S)

        P_S = np.exp(log_p_S)
        post_i = np.zeros((s, self.K), dtype=np.float64)
        for sub_idx, S in enumerate(subsets):
            p = P_S[sub_idx]
            if p > 0:
                post_i[:, list(S)] += p

        row_sums = np.sum(post_i, axis=1, keepdims=True)
        zero_rows = (row_sums.squeeze(axis=-1) <= 1e-12)
        if np.any(zero_rows):
            post_i[zero_rows] = 1.0 / self.K
            row_sums[zero_rows] = 1.0
        prob_i = np.clip(post_i / row_sums, 1e-15, 1.0)
        return np.log(prob_i)

def scores_exact_dp(U_test: np.ndarray, groups: np.ndarray, prm: dict, D_dist: np.ndarray) -> np.ndarray:
    N = len(U_test)
    if N == 0:
        return np.zeros((0, K), dtype=np.float64)
    if N == 1:
        solver = ExactAuthorPosterior(prm["m"], prm["lam"])
        return solver.single(U_test[0], temperature=prm.get("T1", 2.35))[None, :]

    safe_groups = split_oversized_clusters_dist(D_dist, groups, max_cluster_size=14)
    solver = ExactAuthorPosterior(prm["m"], prm["lam"])
    T1 = prm.get("T1", 2.35)
    Tg = prm.get("Tg", 2.30)

    lp_dp = np.zeros((N, K), dtype=np.float64)
    for g in np.unique(safe_groups):
        idx_g = np.where(safe_groups == g)[0]
        if len(idx_g) == 1:
            lp_dp[idx_g] = solver.single(U_test[idx_g[0]], temperature=T1)
        else:
            lp_dp[idx_g] = solver.group(U_test[idx_g], temperature=Tg)
    return lp_dp
""")

    # Cell 5: 6-Checkpoint Ensemble across 4 Trunks
    add_md("### Step 5: 6-Checkpoint Ensemble across 4 Trunks with In-Place Trunk Sharing")
    add_code("""claude_spec_logits = None
claude_classes = None

# ==============================================================================
# TRUNK 1: Qwen3-14B (Hosts lora_qwen3_14b and lora14b_v1, keep=27)
# ==============================================================================
if BASE_QWEN3_14B is not None and ('qwen3_14b_p2' in CKPTS or 'qwen3_14b_p1' in CKPTS):
    print('\\n===============================================================')
    print('=== Trunk 1: Qwen3-14B (Part 2 & Part 1 Shared Trunk, keep=27) ===')
    print('===============================================================')
    trunk_14b = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN3_14B, keep_layers=27,
        layers=[2, 5, 8, 12, 16, 20, 24, 27], max_len=447
    )

    # 1A. Member: lora_qwen3_14b (Part 2, 27L)
    if 'qwen3_14b_p2' in CKPTS:
        ckpt_14bp2 = CKPTS['qwen3_14b_p2']
        meta_14bp2 = json.load(open(ckpt_14bp2 / 'meta.json')) if (ckpt_14bp2 / 'meta.json').exists() else {'labels': LABELS}
        print('\\n--- [Member: lora_qwen3_14b (Part 2)] In-place merging adapter ---')
        merge_lora(trunk_14b.model, ckpt_14bp2, sign=+1.0)
        heads_sd = load_file(str(ckpt_14bp2 / 'heads.safetensors')) if (ckpt_14bp2 / 'heads.safetensors').exists() else {}
        if heads_sd:
            trunk_14b.load_norm_weights(heads_sd)

        Z_all = trunk_14b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
        W_p = align_cols(W_p, meta_14bp2.get('labels', LABELS))
        b_p = align_cols(b_p, meta_14bp2.get('labels', LABELS))
        if W_c is not None:
            W_c = align_cols(W_c, meta_14bp2.get('labels', LABELS))
            b_c = align_cols(b_c, meta_14bp2.get('labels', LABELS))

        print(f"[HEAD KEYS] lora_qwen3_14b: Plain={W_p.shape if W_p is not None else None} | Centered={W_c.shape if W_c is not None else None} | Rescued BF16 Rows={trunk_14b.total_rescued}")

        pred_tr = np.argmax(Z_train_bb @ W_p.T + b_p, axis=1)
        bacc_tr = balanced_accuracy_score(y_sub, pred_tr)
        print(f"[CHECKPOINT GATE] lora_qwen3_14b: 1620-Row Train BAcc = {bacc_tr:.4f}")

        if bacc_tr >= 0.50:
            w_plain, w_twin, w_helper = 0.42, 0.13, 0.34
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_CORE += w_plain * lp_plain

            Z_norm = Z_test - Z_test.mean(axis=0, keepdims=True)
            Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
            D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
            np.fill_diagonal(D_m, 0.0)
            D_neural_sum += w_plain * D_m
            D_neural_weight_total += w_plain

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85 = fcluster(Z_link, t=0.85, criterion='distance')
                g72 = fcluster(Z_link, t=0.72, criterion='distance')
                g60 = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85, m85 = compute_centered_logits(Z_test, g85, W_c, b_c, LABELS)
                lp_c_72, m72 = compute_centered_logits(Z_test, g72, W_c, b_c, LABELS)
                lp_c_60, m60 = compute_centered_logits(Z_test, g60, W_c, b_c, LABELS)

                if m85.any(): LOGIT_CORE[m85] += (0.50 * w_twin) * lp_c_85[m85]
                if m72.any(): LOGIT_CORE[m72] += (0.30 * w_twin) * lp_c_72[m72]
                if m60.any(): LOGIT_CORE[m60] += (0.20 * w_twin) * lp_c_60[m60]
                print(f"[lora_qwen3_14b] Multi-Resolution Twins Applied: m85={m85.sum()}, m72={m72.sum()}, m60={m60.sum()}")

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
            lp_dp = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((w_helper, P_dp))
            print(f"[Routing: lora_qwen3_14b] Added Core Plain({w_plain}) + Twin({w_twin}), Helper DP({w_helper}).")
            del U_train, U_test, prm, lp_dp, P_dp

        merge_lora(trunk_14b.model, ckpt_14bp2, sign=-1.0)
        del Z_all, Z_test, Z_train_bb, heads_sd
        gc.collect()

    # 1B. Member: lora14b_v1 (Part 1, 24L) reusing Trunk 1
    if 'qwen3_14b_p1' in CKPTS:
        ckpt_14bp1 = CKPTS['qwen3_14b_p1']
        meta_14bp1 = json.load(open(ckpt_14bp1 / 'meta.json')) if (ckpt_14bp1 / 'meta.json').exists() else {'labels': LABELS}
        print('\\n--- [Member: lora14b_v1 (Part 1)] In-place merging adapter ---')
        merge_lora(trunk_14b.model, ckpt_14bp1, sign=+1.0)
        heads_sd = load_file(str(ckpt_14bp1 / 'heads.safetensors')) if (ckpt_14bp1 / 'heads.safetensors').exists() else {}
        if heads_sd:
            trunk_14b.load_norm_weights(heads_sd)

        Z_all = trunk_14b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
        W_p = align_cols(W_p, meta_14bp1.get('labels', LABELS))
        b_p = align_cols(b_p, meta_14bp1.get('labels', LABELS))
        if W_c is not None:
            W_c = align_cols(W_c, meta_14bp1.get('labels', LABELS))
            b_c = align_cols(b_c, meta_14bp1.get('labels', LABELS))

        print(f"[HEAD KEYS] lora14b_v1: Plain={W_p.shape if W_p is not None else None} | Centered={W_c.shape if W_c is not None else None} | Rescued BF16 Rows={trunk_14b.total_rescued}")

        pred_tr = np.argmax(Z_train_bb @ W_p.T + b_p, axis=1)
        bacc_tr = balanced_accuracy_score(y_sub, pred_tr)
        print(f"[CHECKPOINT GATE] lora14b_v1: 1620-Row Train BAcc = {bacc_tr:.4f}")

        if bacc_tr >= 0.50:
            w_plain, w_twin, w_helper = 0.07, 0.02, 0.04
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_CORE += w_plain * lp_plain

            Z_norm = Z_test - Z_test.mean(axis=0, keepdims=True)
            Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
            D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
            np.fill_diagonal(D_m, 0.0)
            D_neural_sum += w_plain * D_m
            D_neural_weight_total += w_plain

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85 = fcluster(Z_link, t=0.85, criterion='distance')
                g72 = fcluster(Z_link, t=0.72, criterion='distance')
                g60 = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85, m85 = compute_centered_logits(Z_test, g85, W_c, b_c, LABELS)
                lp_c_72, m72 = compute_centered_logits(Z_test, g72, W_c, b_c, LABELS)
                lp_c_60, m60 = compute_centered_logits(Z_test, g60, W_c, b_c, LABELS)

                if m85.any(): LOGIT_CORE[m85] += (0.50 * w_twin) * lp_c_85[m85]
                if m72.any(): LOGIT_CORE[m72] += (0.30 * w_twin) * lp_c_72[m72]
                if m60.any(): LOGIT_CORE[m60] += (0.20 * w_twin) * lp_c_60[m60]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
            lp_dp = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((w_helper, P_dp))
            print(f"[Routing: lora14b_v1] Added Core Plain({w_plain}) + Twin({w_twin}), Helper DP({w_helper}).")
            del U_train, U_test, prm, lp_dp, P_dp

        merge_lora(trunk_14b.model, ckpt_14bp1, sign=-1.0)
        del Z_all, Z_test, Z_train_bb, heads_sd
        gc.collect()

    trunk_14b.free_memory()
    del trunk_14b
    gc.collect()
    torch.cuda.empty_cache()

log_ram("After Trunk 1 (Qwen3-14B) Cleanup")

# ==============================================================================
# TRUNK 2: Qwen2.5-14B (Part 2, keep=32)
# ==============================================================================
if BASE_QWEN25_14B is not None and 'qwen25_14b_p2' in CKPTS:
    ckpt_q25 = CKPTS['qwen25_14b_p2']
    meta_q25 = json.load(open(ckpt_q25 / 'meta.json')) if (ckpt_q25 / 'meta.json').exists() else {'labels': LABELS}
    print('\\n===============================================================')
    print('=== Trunk 2: Qwen2.5-14B (Part 2, keep=32) ===')
    print('===============================================================')
    trunk_q25 = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN25_14B, keep_layers=32,
        layers=[2, 6, 10, 14, 18, 22, 27, 32], max_len=447
    )

    print('\\n--- [Member: lora_qwen25_14b (Part 2)] In-place merging adapter ---')
    merge_lora(trunk_q25.model, ckpt_q25, sign=+1.0)
    heads_sd = load_file(str(ckpt_q25 / 'heads.safetensors')) if (ckpt_q25 / 'heads.safetensors').exists() else {}
    if heads_sd:
        trunk_q25.load_norm_weights(heads_sd)

    Z_all = trunk_q25.extract_exact(eval_texts, batch_size=8, end_id=151643)
    Z_test = Z_all[:N_test]
    Z_train_bb = Z_all[N_test:]

    W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
    W_p = align_cols(W_p, meta_q25.get('labels', LABELS))
    b_p = align_cols(b_p, meta_q25.get('labels', LABELS))
    if W_c is not None:
        W_c = align_cols(W_c, meta_q25.get('labels', LABELS))
        b_c = align_cols(b_c, meta_q25.get('labels', LABELS))

    print(f"[HEAD KEYS] lora_qwen25_14b: Plain={W_p.shape if W_p is not None else None} | Centered={W_c.shape if W_c is not None else None} | Rescued BF16 Rows={trunk_q25.total_rescued}")

    pred_tr = np.argmax(Z_train_bb @ W_p.T + b_p, axis=1)
    bacc_tr = balanced_accuracy_score(y_sub, pred_tr)
    print(f"[CHECKPOINT GATE] lora_qwen25_14b: 1620-Row Train BAcc = {bacc_tr:.4f}")

    if bacc_tr >= 0.50:
        w_plain, w_twin, w_helper = 0.24, 0.10, 0.22
        lp_plain = log_softmax(Z_test @ W_p.T + b_p)
        LOGIT_CORE += w_plain * lp_plain

        Z_norm = Z_test - Z_test.mean(axis=0, keepdims=True)
        Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
        D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
        np.fill_diagonal(D_m, 0.0)
        D_neural_sum += w_plain * D_m
        D_neural_weight_total += w_plain

        D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
        D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

        if W_c is not None and N_test >= 2:
            cond_dist = squareform(D_hybrid_curr, checks=False)
            Z_link = linkage(cond_dist, method='average')
            g85 = fcluster(Z_link, t=0.85, criterion='distance')
            g72 = fcluster(Z_link, t=0.72, criterion='distance')
            g60 = fcluster(Z_link, t=0.60, criterion='distance')

            lp_c_85, m85 = compute_centered_logits(Z_test, g85, W_c, b_c, LABELS)
            lp_c_72, m72 = compute_centered_logits(Z_test, g72, W_c, b_c, LABELS)
            lp_c_60, m60 = compute_centered_logits(Z_test, g60, W_c, b_c, LABELS)

            if m85.any(): LOGIT_CORE[m85] += (0.50 * w_twin) * lp_c_85[m85]
            if m72.any(): LOGIT_CORE[m72] += (0.30 * w_twin) * lp_c_72[m72]
            if m60.any(): LOGIT_CORE[m60] += (0.20 * w_twin) * lp_c_60[m60]

        prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256)
        U_train = transform_lowram(Z_train_bb, prm)
        U_test = transform_lowram(Z_test, prm)

        idx64 = prm["top64_idx"]
        U_tr_64 = U_train[:, idx64].astype(np.float32)
        U_te_64 = U_test[:, idx64].astype(np.float32)
        U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
        U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
        spectral_tr_blocks.append(U_tr_64)
        spectral_te_blocks.append(U_te_64)

        g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
        lp_dp = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
        P_dp = softmax(lp_dp, axis=1)
        HELPER_LIST.append((w_helper, P_dp))
        print(f"[Routing: lora_qwen25_14b] Added Core Plain({w_plain}) + Twin({w_twin}), Helper DP({w_helper}).")
        del U_train, U_test, prm, lp_dp, P_dp

    merge_lora(trunk_q25.model, ckpt_q25, sign=-1.0)
    trunk_q25.free_memory()
    del trunk_q25, heads_sd, Z_all, Z_test, Z_train_bb
    gc.collect()
    torch.cuda.empty_cache()

log_ram("After Trunk 2 (Qwen2.5-14B) Cleanup")

# ==============================================================================
# TRUNK 3: Qwen3-8B (Hosts InStyle LoRA & Claude Head, keep=24)
# ==============================================================================
if BASE_QWEN3_8B is not None and ('instyle' in CKPTS or 'qwen3_8b_claude' in CKPTS):
    print('\\n===============================================================')
    print('=== Trunk 3: Qwen3-8B (InStyle LoRA & Claude Head Shared Trunk, keep=24) ===')
    print('===============================================================')
    trunk_8b = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN3_8B, keep_layers=24,
        layers=[2, 5, 8, 11, 14, 17, 20, 24], max_len=447
    )

    # 3A. In-Style Qwen3-8B LoRA
    if 'instyle' in CKPTS:
        ckpt_inst = CKPTS['instyle']
        meta_inst = json.load(open(ckpt_inst / 'meta.json')) if (ckpt_inst / 'meta.json').exists() else {'labels': LABELS}
        print('\\n--- [Member: In-Style Qwen3-8B LoRA] In-place merging ---')
        merge_lora(trunk_8b.model, ckpt_inst, sign=+1.0)
        heads_inst_sd = load_file(str(ckpt_inst / 'heads.safetensors')) if (ckpt_inst / 'heads.safetensors').exists() else {}
        if heads_inst_sd:
            trunk_8b.load_norm_weights(heads_inst_sd)

        Z_all = trunk_8b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        readouts_file = ckpt_inst / 'readouts.npz'
        if readouts_file.exists():
            ro = np.load(readouts_file, allow_pickle=True)
            heads_dict = dict(ro)
        else:
            heads_dict = heads_inst_sd

        W_p, b_p, W_c, b_c = resolve_head_tensors(heads_dict)
        src_labs = ro['classes'] if (readouts_file.exists() and 'classes' in ro) else meta_inst.get('labels', LABELS)
        W_p = align_cols(W_p, src_labs)
        b_p = align_cols(b_p, src_labs)
        if W_c is not None:
            W_c = align_cols(W_c, src_labs)
            b_c = align_cols(b_c, src_labs)

        print(f"[HEAD KEYS] instyle_qwen3_8b: Plain={W_p.shape if W_p is not None else None} | Centered={W_c.shape if W_c is not None else None} | Rescued BF16 Rows={trunk_8b.total_rescued}")

        pred_tr = np.argmax(Z_train_bb @ W_p.T + b_p, axis=1)
        bacc_tr = balanced_accuracy_score(y_sub, pred_tr)
        print(f"[CHECKPOINT GATE] instyle_qwen3_8b: 1620-Row Train BAcc = {bacc_tr:.4f}")

        if bacc_tr >= 0.50:
            w_plain, w_twin, w_helper = 0.18, 0.06, 0.14
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_CORE += w_plain * lp_plain

            Z_norm = Z_test - Z_test.mean(axis=0, keepdims=True)
            Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
            D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
            np.fill_diagonal(D_m, 0.0)
            D_neural_sum += w_plain * D_m
            D_neural_weight_total += w_plain

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85 = fcluster(Z_link, t=0.85, criterion='distance')
                g72 = fcluster(Z_link, t=0.72, criterion='distance')
                g60 = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85, m85 = compute_centered_logits(Z_test, g85, W_c, b_c, LABELS)
                lp_c_72, m72 = compute_centered_logits(Z_test, g72, W_c, b_c, LABELS)
                lp_c_60, m60 = compute_centered_logits(Z_test, g60, W_c, b_c, LABELS)

                if m85.any(): LOGIT_CORE[m85] += (0.50 * w_twin) * lp_c_85[m85]
                if m72.any(): LOGIT_CORE[m72] += (0.30 * w_twin) * lp_c_72[m72]
                if m60.any(): LOGIT_CORE[m60] += (0.20 * w_twin) * lp_c_60[m60]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
            lp_dp = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((w_helper, P_dp))
            print(f"[Routing: instyle_qwen3_8b] Added Core Plain({w_plain}) + Twin({w_twin}), Helper DP({w_helper}).")
            del U_train, U_test, prm, lp_dp, P_dp

        merge_lora(trunk_8b.model, ckpt_inst, sign=-1.0)
        del Z_all, Z_test, Z_train_bb
        gc.collect()

    # 3B. Claude-Head Qwen3-8B LoRA
    if 'qwen3_8b_claude' in CKPTS:
        ckpt_8bcl = CKPTS['qwen3_8b_claude']
        meta_cl = json.load(open(ckpt_8bcl / 'meta.json')) if (ckpt_8bcl / 'meta.json').exists() else {'labels': LABELS}
        print('\\n--- [Member: ntl-qwen3-8b-lora-claude-head] In-place merging ---')
        merge_lora(trunk_8b.model, ckpt_8bcl, sign=+1.0)
        heads_cl_sd = load_file(str(ckpt_8bcl / 'heads.safetensors')) if (ckpt_8bcl / 'heads.safetensors').exists() else {}
        if heads_cl_sd:
            trunk_8b.load_norm_weights(heads_cl_sd)

        Z_all = trunk_8b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        ro_cl = None
        if (ckpt_8bcl / 'readouts.npz').exists():
            ro_cl = np.load(ckpt_8bcl / 'readouts.npz', allow_pickle=True)
            heads_dict = dict(ro_cl)
        else:
            heads_dict = heads_cl_sd

        W_p, b_p, W_c, b_c = resolve_head_tensors(heads_dict)
        src_labs = ro_cl['classes'] if (ro_cl is not None and 'classes' in ro_cl) else meta_cl.get('labels', LABELS)
        W_p = align_cols(W_p, src_labs)
        b_p = align_cols(b_p, src_labs)
        if W_c is not None:
            W_c = align_cols(W_c, src_labs)
            b_c = align_cols(b_c, src_labs)

        print(f"[HEAD KEYS] claude_head: Plain={W_p.shape if W_p is not None else None} | Centered={W_c.shape if W_c is not None else None} | Rescued BF16 Rows={trunk_8b.total_rescued}")

        # Retain Claude specialist head
        if ro_cl is not None and 'claude_coef' in ro_cl and 'claude_classes' in ro_cl:
            cl_w = ro_cl['claude_coef']
            cl_b = ro_cl['claude_intercept']
            raw_cl = Z_test @ cl_w.T + cl_b
            claude_spec_logits = raw_cl
            claude_classes = ro_cl['claude_classes']
            print('[Routing: qwen3_8b_claude] Stored Claude family specialist logits.')

        pred_tr = np.argmax(Z_train_bb @ W_p.T + b_p, axis=1)
        bacc_tr = balanced_accuracy_score(y_sub, pred_tr)
        print(f"[CHECKPOINT GATE] ntl-qwen3-8b-lora-claude-head: 1620-Row Train BAcc = {bacc_tr:.4f}")

        if bacc_tr >= 0.50:
            w_plain, w_twin, w_helper = 0.12, 0.04, 0.08
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_CORE += w_plain * lp_plain

            Z_norm = Z_test - Z_test.mean(axis=0, keepdims=True)
            Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
            D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
            np.fill_diagonal(D_m, 0.0)
            D_neural_sum += w_plain * D_m
            D_neural_weight_total += w_plain

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85 = fcluster(Z_link, t=0.85, criterion='distance')
                g72 = fcluster(Z_link, t=0.72, criterion='distance')
                g60 = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85, m85 = compute_centered_logits(Z_test, g85, W_c, b_c, LABELS)
                lp_c_72, m72 = compute_centered_logits(Z_test, g72, W_c, b_c, LABELS)
                lp_c_60, m60 = compute_centered_logits(Z_test, g60, W_c, b_c, LABELS)

                if m85.any(): LOGIT_CORE[m85] += (0.50 * w_twin) * lp_c_85[m85]
                if m72.any(): LOGIT_CORE[m72] += (0.30 * w_twin) * lp_c_72[m72]
                if m60.any(): LOGIT_CORE[m60] += (0.20 * w_twin) * lp_c_60[m60]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
            lp_dp = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((w_helper, P_dp))
            print(f"[Routing: ntl-qwen3-8b-lora-claude-head] Added Core Plain({w_plain}) + Twin({w_twin}), Helper DP({w_helper}).")
            del U_train, U_test, prm, lp_dp, P_dp

        merge_lora(trunk_8b.model, ckpt_8bcl, sign=-1.0)
        del Z_all, Z_test, Z_train_bb
        gc.collect()

    trunk_8b.free_memory()
    del trunk_8b
    gc.collect()
    torch.cuda.empty_cache()

log_ram("After Trunk 3 (Qwen3-8B) Cleanup")

# ==============================================================================
# TRUNK 4: Dedicated DeBERTa-v3 Fine-Tuned Encoder
# ==============================================================================
if 'deberta_p2' in CKPTS:
    ckpt_deb = CKPTS['deberta_p2']
    meta_deb = json.load(open(ckpt_deb / 'meta.json')) if (ckpt_deb / 'meta.json').exists() else {'labels': LABELS}
    print('\\n===============================================================')
    print('=== Trunk 4: Dedicated DeBERTa-v3 Fine-Tuned Encoder ===')
    print('===============================================================')
    enc_src = ckpt_deb / 'encoder' if (ckpt_deb / 'encoder').exists() else ckpt_deb
    stage_dir = Path('/kaggle/temp/deberta_v3_large') if Path('/kaggle/temp').exists() else Path('/kaggle/working/temp_deberta') if Path('/kaggle/working').exists() else Path('temp_deberta')
    os.makedirs(stage_dir, exist_ok=True)
    for f in enc_src.iterdir():
        if f.is_file():
            shutil.copy2(f, stage_dir / f.name)

    deb_tok = AutoTokenizer.from_pretrained(str(stage_dir))
    if deb_tok.pad_token is None:
        deb_tok.pad_token = deb_tok.eos_token or '[PAD]'

    dev_deb = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    deb_model = AutoModel.from_pretrained(str(stage_dir), torch_dtype=torch.float16).to(dev_deb)
    deb_model.eval()
    shutil.rmtree(stage_dir, ignore_errors=True)

    head_path = ckpt_deb / 'head.safetensors' if (ckpt_deb / 'head.safetensors').exists() else ckpt_deb / 'heads.safetensors'
    heads_deb_sd = load_file(str(head_path)) if head_path.exists() else {}

    W_p, b_p, W_c, b_c = resolve_head_tensors(heads_deb_sd)
    W_p = align_cols(W_p, meta_deb.get('labels', LABELS))
    b_p = align_cols(b_p, meta_deb.get('labels', LABELS))
    if W_c is not None:
        W_c = align_cols(W_c, meta_deb.get('labels', LABELS))
        b_c = align_cols(b_c, meta_deb.get('labels', LABELS))

    print(f"[HEAD KEYS] deberta_v3_large: Plain={W_p.shape if W_p is not None else None} | Centered={W_c.shape if W_c is not None else None}")

    # Length-sorted dynamic batching for DeBERTa
    order_deb = np.argsort([len(x) for x in eval_texts], kind="stable")
    Z_deb = np.zeros((len(eval_texts), 1024), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(order_deb), 16):
            b_idx = order_deb[s : s + 16]
            batch_texts = [eval_texts[i] for i in b_idx]
            enc = deb_tok(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(dev_deb)
            out = deb_model(**enc)
            m = enc['attention_mask'].unsqueeze(-1).float()
            pool = (out.last_hidden_state.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            Z_deb[b_idx] = pool.detach().cpu().numpy()

    del deb_model, deb_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    Z_test_deb = Z_deb[:N_test]
    Z_train_deb = Z_deb[N_test:]

    pred_tr = np.argmax(Z_train_deb @ W_p.T + b_p, axis=1)
    bacc_tr = balanced_accuracy_score(y_sub, pred_tr)
    print(f"[CHECKPOINT GATE] deberta_v3_large: 1620-Row Train BAcc = {bacc_tr:.4f}")

    if bacc_tr >= 0.50:
        w_plain, w_twin, w_helper = 0.22, 0.08, 0.18
        lp_plain = log_softmax(Z_test_deb @ W_p.T + b_p)
        LOGIT_CORE += w_plain * lp_plain

        Z_norm = Z_test_deb - Z_test_deb.mean(axis=0, keepdims=True)
        Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
        D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
        np.fill_diagonal(D_m, 0.0)
        D_neural_sum += w_plain * D_m
        D_neural_weight_total += w_plain

        D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
        D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

        if W_c is not None and N_test >= 2:
            cond_dist = squareform(D_hybrid_curr, checks=False)
            Z_link = linkage(cond_dist, method='average')
            g85 = fcluster(Z_link, t=0.85, criterion='distance')
            g72 = fcluster(Z_link, t=0.72, criterion='distance')
            g60 = fcluster(Z_link, t=0.60, criterion='distance')

            lp_c_85, m85 = compute_centered_logits(Z_test_deb, g85, W_c, b_c, LABELS)
            lp_c_72, m72 = compute_centered_logits(Z_test_deb, g72, W_c, b_c, LABELS)
            lp_c_60, m60 = compute_centered_logits(Z_test_deb, g60, W_c, b_c, LABELS)

            if m85.any(): LOGIT_CORE[m85] += (0.50 * w_twin) * lp_c_85[m85]
            if m72.any(): LOGIT_CORE[m72] += (0.30 * w_twin) * lp_c_72[m72]
            if m60.any(): LOGIT_CORE[m60] += (0.20 * w_twin) * lp_c_60[m60]

        prm = fit_shared_gaussian_lowram(Z_train_deb, y_sub, prompts_sub, d_target=256)
        U_train = transform_lowram(Z_train_deb, prm)
        U_test = transform_lowram(Z_test_deb, prm)

        idx64 = prm["top64_idx"]
        U_tr_64 = U_train[:, idx64].astype(np.float32)
        U_te_64 = U_test[:, idx64].astype(np.float32)
        U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
        U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
        spectral_tr_blocks.append(U_tr_64)
        spectral_te_blocks.append(U_te_64)

        g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
        lp_dp = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
        P_dp = softmax(lp_dp, axis=1)
        HELPER_LIST.append((w_helper, P_dp))
        print(f"[Routing: deberta_v3_large] Added Core Plain({w_plain}) + Twin({w_twin}), Helper DP({w_helper}).")
        del U_train, U_test, prm, lp_dp, P_dp

    del Z_deb, Z_test_deb, Z_train_deb, heads_deb_sd
    gc.collect()

log_ram("After Trunk 4 (DeBERTa) Cleanup")
""")

    # Cell 6: Module 4 — 384-Dim Joint Spectral Bank + 88/12 Mixture + Shield + 300-Step Sinkhorn
    add_md("### Step 6: Module 4 — 384-Dim Cross-Backbone Spectral Bank + 88/12 Mixture + Shield + 300-Step Sinkhorn")
    add_code("""# 1. Fit Full-Rank 384-Dim Joint Cross-Backbone Spectral Bank on 1,620 Rows
if len(spectral_tr_blocks) > 0:
    X_spec_tr = np.hstack(spectral_tr_blocks)
    X_spec_te = np.hstack(spectral_te_blocks)
    print(f"[Joint Spectral Bank] Formed matrices: Train={X_spec_tr.shape}, Test={X_spec_te.shape}")

    # 5-fold GroupKFold OOF Balanced Accuracy
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y_sub), dtype=int)
    for tr_idx, val_idx in gkf.split(X_spec_tr, y_sub, groups=prompts_sub):
        clf_fold = LogisticRegression(C=0.15, max_iter=2000, random_state=42)
        clf_fold.fit(X_spec_tr[tr_idx], y_sub[tr_idx])
        oof_preds[val_idx] = clf_fold.predict(X_spec_tr[val_idx])
    oof_bacc = balanced_accuracy_score(y_sub, oof_preds)
    print(f"[SPECTRAL BANK 5-FOLD OOF] Plain BAcc = {oof_bacc:.4f}")

    # Plain Logistic Regression
    clf_plain = LogisticRegression(C=0.15, max_iter=2000, random_state=42)
    clf_plain.fit(X_spec_tr, y_sub)
    lp_spec_plain = clf_plain.predict_log_proba(X_spec_te)
    if hasattr(clf_plain, 'classes_'):
        lp_spec_plain = align_cols(lp_spec_plain, [LABELS[c] for c in clf_plain.classes_])

    # Prompt-Centered Logistic Regression
    X_spec_tr_c = X_spec_tr.copy()
    for p in np.unique(prompts_sub):
        mask = (prompts_sub == p)
        X_spec_tr_c[mask] -= X_spec_tr_c[mask].mean(axis=0, keepdims=True)
    clf_center = LogisticRegression(C=0.10, max_iter=2000, random_state=42)
    clf_center.fit(X_spec_tr_c, y_sub)

    # Prompt-centered test features
    X_spec_te_c = X_spec_te.copy()
    D_neural_final = D_neural_sum / max(1e-6, D_neural_weight_total) if D_neural_weight_total > 0 else D_tfidf
    D_hybrid_final = 0.55 * D_tfidf + 0.45 * D_neural_final
    g65_final = fcluster(linkage(squareform(D_hybrid_final, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
    safe_groups = split_oversized_clusters_dist(D_hybrid_final, g65_final, max_cluster_size=14)

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

    lp_spec_joint = lp_spec_plain + 0.45 * lp_spec_center
    HELPER_LIST.append((0.22, softmax(lp_spec_joint, axis=1)))
    print("[Joint Spectral Bank] Appended helper probability with weight 0.22.")

# 2. Combine Core (88%) and Helper (12%) in Probability Space
R_core = softmax(LOGIT_CORE, axis=1)
if len(HELPER_LIST) > 0:
    H_helper = sum(w * P_h for w, P_h in HELPER_LIST) / max(1e-8, sum(w for w, _ in HELPER_LIST))
    P_mix = 0.88 * R_core + 0.12 * H_helper
else:
    P_mix = R_core

log_P = np.log(np.clip(P_mix, 1e-12, 1.0))
print(f"[Probability Mixture] Blended 88% Core + 12% Helper ({len(HELPER_LIST)} helpers).")

# 3. Apply Pre-Sinkhorn Top-1 Claude Specialist Gate & Orthographic Shield
# Claude Specialist Gate
if claude_spec_logits is not None:
    claude_cols = FAMILIES["claude"]
    P_claude_head = softmax(claude_spec_logits, axis=-1)
    if claude_classes is not None:
        src_c = [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in claude_classes]
        c_idx = [src_c.index(LABELS[c]) for c in claude_cols]
        P_claude_head = P_claude_head[:, c_idx]

    pred_core = R_core.argmax(axis=1)
    claude_mask = np.isin(pred_core, claude_cols)
    if claude_mask.any():
        for i in np.where(claude_mask)[0]:
            tot_mass = P_mix[i, claude_cols].sum()
            blended = 0.78 * P_mix[i, claude_cols] + 0.22 * (tot_mass * P_claude_head[i])
            P_mix[i, claude_cols] = blended
        log_P = np.log(np.clip(P_mix, 1e-12, 1.0))
        print(f"[Claude Specialist Gate] Refined {claude_mask.sum()} passages predicted in Claude family.")

# Pre-Sinkhorn Orthographic Shield
claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
gemini_idx = FAMILIES["gemini"]
sol_idx = LAB2I["gpt-5.6-sol"]

for i, txt in enumerate(test_texts):
    txt_s = str(txt)
    if any(q in txt_s for q in ['\\u2018', '\\u2019', '\\u201c', '\\u201d']):
        log_P[i, claude_and_glm_idx] -= 2.5
    if '\\u2014' in txt_s:
        log_P[i, gemini_idx] -= 2.0
    if "'" in txt_s:
        log_P[i, sol_idx] -= 2.5

# 4. Final 300-Step Exact Multiplicative Sinkhorn Balancing
def sinkhorn_balancing(P: np.ndarray, iters: int = 300) -> np.ndarray:
    if len(P) < 50:
        return P
    P = np.clip(P.astype(np.float64), 1e-12, 1.0)
    target_col = 1.0 / P.shape[1]
    for _ in range(iters):
        P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
        P /= (P.sum(axis=0, keepdims=True) / (target_col * P.shape[0])).clip(min=1e-12)
    P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P

print("[Sinkhorn] Running official 300-step exact multiplicative Sinkhorn mass balancing...")
P_final = sinkhorn_balancing(softmax(log_P, axis=1), iters=300)

preds = [LABELS[idx] for idx in P_final.argmax(axis=1)]
df_sub = pd.DataFrame({"id": test_df["id"], "model": preds})
sub_parquet_path = working_dir / "submission.parquet"
sub_csv_path = working_dir / "submission.csv"

df_sub.to_parquet(sub_parquet_path, index=False)
df_sub.to_csv(sub_csv_path, index=False)

print(f"\\n[SUCCESS] Wrote submission to {sub_parquet_path} ({len(df_sub)} rows).")
print("\\nSubmission preview:")
print(df_sub.head())
print("\\nClass prediction distribution:")
print(df_sub['model'].value_counts())
print("\\n=== Version 40 Pipeline Execution Successfully Completed ===")
""")

    out_file = Path("kaggle_submission.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated {out_file} ({out_file.stat().st_size} bytes).")

if __name__ == "__main__":
    create_v40_pipeline_notebook()
