"""
Name That LLM - Version 42 Pipeline Generator (0.645–0.655+ LB Target)
======================================================================
Eliminates the 8B In-Sample Stacking Trap (0.9142 OOF Leakage) & Deploys
Pure 85% Part-Two Main Stack + 15% Prior-Scaled Complementary Stack.

Key Architectural Improvements in Version 42:
1. Pure Two-Tier Separation (85% Main Stack + 15% Complementary Stack):
   - LOGIT_MAIN (85%): Strictly the 4 official Part Two members:
     * 4-View LinearSVC(C=0.5, uniform sample weights 1.0) -> log_p_svc_main (w = 1.00)
     * lora_qwen3_14b (Part 2, 27L): Plain (w = 0.40) + Twin (w = 0.10, groups_085 only)
     * lora_qwen25_14b (Part 2, 32L): Plain (w = 0.20) + Twin (w = 0.10, groups_085 only)
     * deberta_v3_large (Part 2, 24L): Plain (w = 0.25) (no twin)
   - COMP_LIST (15%):
     * Secondary LoRAs & multi-resolution twins (w = 0.35)
     * Complementary lexical (BPE subwords + char_wb + SGD + LinearSVC + style twin, w = 0.20)
     * Regularized wPCA + ExactAuthorPosterior (2^18 Bitmask DP with Ledoit-Wolf shrinkage floors, w = 0.30)
     * Leak-free prior-scaled 384-dim Joint Spectral Bank (w = 0.15)
2. Module 1 Dual-Output Subprocess Lexical Engine (run_lexical.py):
   - Output 1: Pure 4-view LinearSVC(C=0.5) with uniform weights -> log_p_svc_main.npy for LOGIT_MAIN.
   - Output 2: 5-view BPE + char_wb + SGD + LinearSVC(C=0.35) + style twin -> p_lex_comp.npy for COMP_LIST.
   - Saves D_tfidf.npy and groups_085.npy.
3. Leak-Free Unsupervised 384-Dim Joint Spectral Bank:
   - Features extracted from unsupervised prompt-centered principal components V_f32[:, :64] (zero label leakage!).
   - Multiplied by Out-of-Sample Prior Scales (q3_14b: 1.00, q25_14b: 0.85, deb: 0.80, part1: 0.50, instyle: 0.45, claude_head: 0.30).
   - 5-fold GroupKFold OOF BAcc verification (realistic 0.63–0.68).
4. Regularized ExactAuthorPosterior with Ledoit-Wolf Shrinkage Floors:
   - se_floor = 0.10 for Part Two checkpoints.
   - se_floor = 0.35 for overfit 8B and Part 1 checkpoints.
5. Postprocessing:
   - Gentle mass-preserving Claude specialist gate (0.15 * P_claude_head within Claude family).
   - Gentle zero-occurrence guard on Claude/GLM curly quotes (np.exp(-1.0)) followed by simplex normalization.
   - Official 300-step exact multiplicative Sinkhorn mass balancing.
"""

import json
from pathlib import Path

def create_v42_pipeline_notebook():
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
    add_md("""# Name That LLM: Version 42 Pipeline (0.645–0.655+ LB Target)
### Pure 85% Part-Two Main Stack + 15% Prior-Scaled Complementary Stack, Leak-Free 384-Dim Spectral Bank & Regularized ExactAuthorPosterior

**Core Architectural Upgrades in Version 42**:
1. **Elimination of the 8B In-Sample Stacking Trap & Label Leakage**:
   - `instyle_qwen3_8b` (0.9401) and `ntl-claude-head` (0.9377) memorized `train.csv`. In v40, they hijacked 80%+ weight of the Spectral Bank, suppressing `lora_qwen3_14b` (0.8006) and `lora_qwen25_14b` (0.7506).
   - In v42, spectral features are projected onto **unsupervised prompt-centered principal components `V_f32[:, :64]`** (zero labels used, zero leakage!), and each block is weighted by its **Out-of-Sample Prior Scale** (`q3_14b`: 1.00, `q25_14b`: 0.85, `deberta`: 0.80, `part1_14b`: 0.50, `instyle`: 0.45, `claude_head`: 0.30).
2. **Pure Two-Tier Stack Separation (85% Main Stack + 15% Complementary Stack)**:
   - **Main Stack (`LOGIT_MAIN`, 85% Mass)**: Strictly the 4 official `NTL two-stage LoRA part two` members (guaranteed `0.620+` baseline):
     * 1. 4-View LinearSVC(C=0.5, uniform sample weights 1.0) on 78k corpus (`w = 1.00`, twin = `0.00`)
     * 2. `lora_qwen3_14b` (Part 2, 27L): Plain (`w = 0.40`) + Twin (`w = 0.10`, `groups_085` only)
     * 3. `lora_qwen25_14b` (Part 2, 32L): Plain (`w = 0.20`) + Twin (`w = 0.10`, `groups_085` only)
     * 4. `deberta_v3_large` (Part 2, 24L): Plain (`w = 0.25`) (twin = `0.00`)
   - **Complementary Stack (`COMP_LIST`, 15% Mass)**:
     * Secondary LoRAs & multi-resolution twins (`w = 0.35`)
     * Complementary lexical (BPE subwords + char_wb + SGD + LinearSVC + style twin, `w = 0.20`)
     * Regularized `wPCA` + `ExactAuthorPosterior` ($2^{18}$ Bitmask DP with Ledoit-Wolf shrinkage floors, `w = 0.30`)
     * Leak-free prior-scaled 384-dim Joint Spectral Bank (`w = 0.15`)
3. **Module 1 Subprocess Dual-Output Lexical Engine (`run_lexical.py`)**:
   - `Output 1`: Pure 4-view LinearSVC with uniform 1.0 sample weights -> `log_p_svc_main.npy` for `LOGIT_MAIN`.
   - `Output 2`: 5-view BPE + SGDClassifier + LinearSVC(C=0.35) + style twin -> `p_lex_comp.npy` for `COMP_LIST`.
   - Saves `D_tfidf.npy` and `groups_085.npy`.
4. **Regularized `ExactAuthorPosterior` with Ledoit-Wolf Shrinkage Floors**:
   - `se_floor = 0.10` for Part Two models.
   - `se_floor = 0.35` for overfit 8B and Part 1 models to prevent overconfident posteriors.
5. **Postprocessing Shields & Sinkhorn**:
   - Gentle mass-preserving Claude specialist gate (`0.15 * P_claude_head` within Claude family).
   - Gentle zero-occurrence guard on Claude/GLM curly quotes (`np.exp(-1.0)`) followed by row normalization.
   - Official 300-step exact multiplicative Sinkhorn mass balancing.
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

# Master Canonical 18-Class Labels
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

    # Cell 3: Module 1 — Dual-Output Subprocess Lexical Engine (run_lexical.py)
    add_md("### Step 3: Module 1 — Dual-Output Subprocess Lexical Engine (`run_lexical.py`)")
    add_code("""# Write standalone script run_lexical.py and execute via subprocess for 100% OS RAM reclamation
working_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
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
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

working_dir = Path(r'{working_dir}')
LABELS = {LABELS}
K = len(LABELS)
LAB2I = {{l: i for i, l in enumerate(LABELS)}}

FAMILIES = {{
    "gpt": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
}}

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
    return re.sub(r"[^\\\\W\\\\d_]+", lambda m: "X" if m.group(0)[0].isupper() else "x", str(t))

def function_text(t: str) -> str:
    tokens = re.findall(r"(?u)\\\\b\\\\w+\\\\b|[^\\\\w\\\\s]", str(t))
    return " ".join([w.lower() if w.lower() in STOP_WORDS else "CONTENT" for w in tokens])

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)

t0 = time.time()
print('=== [Subprocess Lexical Engine] Starting Dual-Output 78k Full Execution ===', flush=True)

df_tr = pd.read_csv(r'{train_path}')
df_te = pd.read_csv(r'{test_path}')

test_texts = df_te['text'].fillna('').astype(str).tolist()
fit_texts = df_tr['text'].fillna('').astype(str).tolist()
fit_y = np.array([LAB2I[m] for m in df_tr['model']])
len_tr_raw = len(df_tr)

# 1. Compute stopword TF-IDF Distance Matrix and groups_085 on Test
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
print(f'[Subprocess Lexical] Topic clusters computed (t=0.85): {{len(np.unique(groups_085))}} clusters, twin rows: {{twin_rows_085.sum()}}/{{len(test_texts)}}', flush=True)

np.save(r'{working_dir}/D_tfidf.npy', D)
np.save(r'{working_dir}/groups_085.npy', groups_085)
del Xt_tfidf
gc.collect()

# 2. Load 100% of pool_v4.parquet
pool_path = r'{pool_file}'
if pool_path and os.path.exists(pool_path):
    print(f'[Subprocess Lexical] Loading 100% of pool_v4 from {{pool_path}}...', flush=True)
    df_pool = pd.read_parquet(pool_path, columns=['text', 'model'])
    df_pool = df_pool[df_pool['model'].isin(LABELS)].dropna(subset=['text', 'model']).copy()
    fit_texts += df_pool['text'].fillna('').astype(str).tolist()
    fit_y = np.r_[fit_y, df_pool['model'].map(LAB2I).values]
    print(f'[Subprocess Lexical] Full combined training corpus: {{len(fit_texts)}} rows ({{len_tr_raw}} train + {{len(df_pool)}} pool)', flush=True)
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
    print(f'[Subprocess Lexical] Extracting canonical view: {{view}} (weight={{weight}})...', flush=True)
    prep = shape_text if view == "shape" else function_text if view == "function" else None
    analyzer = "char" if view in ("char", "shape") else "word"
    opts = dict(
        analyzer=analyzer, ngram_range=(lo, hi), lowercase=False,
        sublinear_tf=True, min_df=2, max_features=max_feat, dtype=np.float32
    )
    if analyzer == "word":
        opts["token_pattern"] = r"(?u)\\\\b\\\\w+\\\\b|[^\\\\w\\\\s]"
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

print(f'[Subprocess Lexical] Canonical 4-view CSR: X_tr={{X_tr_4v.shape}}, X_te={{X_te_4v.shape}} (nnz={{X_tr_4v.nnz}})', flush=True)

# OUTPUT 1: Pure 4-View LinearSVC with Uniform 1.0 Sample Weights for Main Stack
sample_weights_uniform = np.ones(len(fit_y), dtype=np.float32)
print('[Subprocess Lexical] Fitting Pure 4-View LinearSVC(C=0.5) with uniform weights (1.0) for LOGIT_MAIN...', flush=True)
svc_main = LinearSVC(C=0.5, random_state=42, max_iter=3500, tol=1e-3, class_weight="balanced")
svc_main.fit(X_tr_4v, fit_y, sample_weight=sample_weights_uniform)
log_p_svc_main = log_softmax(svc_main.decision_function(X_te_4v), axis=1)
np.save(r'{working_dir}/log_p_svc_main.npy', log_p_svc_main)
print(f'[Subprocess Lexical] Saved Output 1: log_p_svc_main.npy (shape={{log_p_svc_main.shape}}).', flush=True)
del svc_main
gc.collect()

# 4. View 5: Native BPE Subword Token-ID N-Grams for Complementary Stack
qwen_tok_path = r'{qwen_tok_dir}'
print(f'[Subprocess Lexical] Extracting view 5: Native BPE Subwords (weight=0.65)...', flush=True)
bpe_loaded = False
if qwen_tok_path and os.path.exists(qwen_tok_path):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(qwen_tok_path, use_fast=True)
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
    except Exception as e:
        print(f'[Subprocess Lexical] Fast tokenizer load failed ({{e}}), using fallback hash BPE.', flush=True)

if not bpe_loaded:
    def hash_bpe_fallback(texts):
        return [" ".join("w" + str(hash(w) % 65536) for w in re.findall(r'\\\\b\\\\w+\\\\b', t.lower())) for t in texts]
    bpe_tr = hash_bpe_fallback(fit_texts)
    bpe_te = hash_bpe_fallback(test_texts)

bpe_vec = TfidfVectorizer(token_pattern=r"\\\\S+", ngram_range=(1, 3), lowercase=False, sublinear_tf=True, min_df=5, max_features=80000, dtype=np.float32)
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

print(f'[Subprocess Lexical] Complementary 5-view CSR: X_tr={{X_tr_5v.shape}}, X_te={{X_te_5v.shape}} (nnz={{X_tr_5v.nnz}})', flush=True)

# OUTPUT 2: Complementary Lexical Ensemble (LinearSVC(C=0.35) + SGDClassifier + Style Twin)
print('[Subprocess Lexical] Fitting Complementary LinearSVC(C=0.35) on 5-view CSR...', flush=True)
svc_comp = LinearSVC(C=0.35, random_state=42, max_iter=2500, tol=1e-3, class_weight="balanced")
svc_comp.fit(X_tr_5v, fit_y, sample_weight=sample_weights_uniform)
P_svc_comp = softmax(svc_comp.decision_function(X_te_5v), axis=1)
del svc_comp
gc.collect()

print('[Subprocess Lexical] Fitting Complementary SGDClassifier (log_loss) on 5-view CSR...', flush=True)
sgd_comp = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-3, random_state=42, class_weight="balanced")
sgd_comp.fit(X_tr_5v, fit_y, sample_weight=sample_weights_uniform)
P_sgd_comp = softmax(sgd_comp.decision_function(X_te_5v), axis=1)
del sgd_comp
gc.collect()

# Prompt-Centered Style Twin on train.csv
P_twin_comp = np.copy(P_svc_comp)
if twin_rows_085.any():
    print('[Subprocess Lexical] Fitting Prompt-Centered Style-Gram Twin on train.csv...', flush=True)
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

# Blend into P_lex_comp
P_lex_comp = 0.50 * P_svc_comp + 0.35 * P_sgd_comp + 0.15 * P_twin_comp
log_p_lex_comp = np.log(np.clip(P_lex_comp, 1e-9, 1.0))

# 4 Within-Family Specialist Refiners on Complementary Lexical Stream
print('[Subprocess Lexical] Fitting 4 Within-Family Specialist Refiners on Complementary Stream...', flush=True)
beta = 0.25
for fam_name, fam_cols in FAMILIES.items():
    fam_mask = np.isin(fit_y, fam_cols)
    if fam_mask.sum() > 50:
        sub_y = fit_y[fam_mask]
        sub_lab2i = {{c: i for i, c in enumerate(fam_cols)}}
        sub_y_mapped = np.array([sub_lab2i[y] for y in sub_y])

        spec_svc = LinearSVC(C=0.40, random_state=42, max_iter=2000, tol=1e-3, class_weight="balanced")
        spec_svc.fit(X_tr_5v[fam_mask], sub_y_mapped)
        df_spec = spec_svc.decision_function(X_te_5v)
        if df_spec.ndim == 1:
            logits_spec = np.column_stack([-df_spec, df_spec])
        else:
            logits_spec = df_spec
        P_spec = softmax(logits_spec, axis=1)

        P_prior = P_lex_comp[:, fam_cols]
        P_prior_norm = P_prior / (P_prior.sum(axis=1, keepdims=True) + 1e-12)
        P_refined = (1.0 - beta) * P_prior_norm + beta * P_spec

        log_ratio = np.log(np.clip(P_refined, 1e-9, 1.0)) - np.log(np.clip(P_prior_norm, 1e-9, 1.0))
        log_p_lex_comp[:, fam_cols] += log_ratio
        print(f'[Subprocess Lexical] Refined complementary family {{fam_name}} (cols={{fam_cols}}, train_rows={{fam_mask.sum()}}).', flush=True)

P_lex_comp_final = softmax(log_p_lex_comp, axis=1)
np.save(r'{working_dir}/p_lex_comp.npy', P_lex_comp_final)
print(f'[Subprocess Lexical] Saved Output 2: p_lex_comp.npy (shape={{P_lex_comp_final.shape}}).', flush=True)

del X_tr_5v, X_te_5v, fit_texts
gc.collect()
print(f'[Subprocess Lexical] Completed in {{time.time() - t0:.2f}}s. All outputs saved successfully.', flush=True)
sys.exit(0)
\"\"\"

with open(lexical_script_path, "w", encoding="utf-8") as f:
    f.write(lexical_code)

print(f'[Main Process] Launching isolated lexical subprocess: {lexical_script_path}...')
t_lex0 = time.time()
subprocess.run([sys.executable, str(lexical_script_path)], check=True)
print(f'[Main Process] Lexical subprocess completed in {time.time() - t_lex0:.2f}s.')

log_p_svc_main = np.load(working_dir / 'log_p_svc_main.npy')
p_lex_comp = np.load(working_dir / 'p_lex_comp.npy')
D_tfidf = np.load(working_dir / 'D_tfidf.npy')
groups_085 = np.load(working_dir / 'groups_085.npy')

print(f'Loaded log_p_svc_main: shape={log_p_svc_main.shape}')
print(f'Loaded p_lex_comp:     shape={p_lex_comp.shape}')
print(f'Loaded D_tfidf:        shape={D_tfidf.shape}')
print(f'Loaded groups_085:      shape={groups_085.shape}')
log_ram("After Lexical Subprocess Memory Reclamation")

# Initialize LOGIT_MAIN with canonical 1.00 * log_p_svc_main (100% official Part Two baseline)
LOGIT_MAIN = 1.00 * log_p_svc_main

# Initialize Complementary Containers
LOGIT_COMP_LORA = np.zeros((N_test, K), dtype=np.float64)
COMP_LIST: List[Tuple[float, np.ndarray]] = []
# Complementary lexical stream gets weight 0.20
COMP_LIST.append((0.20, p_lex_comp))

D_neural_sum = np.zeros((N_test, N_test), dtype=np.float64)
D_neural_weight_total = 0.0

spectral_tr_blocks: List[np.ndarray] = []
spectral_te_blocks: List[np.ndarray] = []

dp_posteriors: Dict[str, np.ndarray] = {}

print(f"[Main Process] Initialized LOGIT_MAIN: shape={LOGIT_MAIN.shape}")
log_ram("Main and Complementary Containers Initialized")
""")

    # Cell 4: Universal Trunk Loader, Head Resolver & Math Solvers
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

def fit_shared_gaussian_lowram(X_tr_f32: np.ndarray, y: np.ndarray, prompts: np.ndarray,
                               d_target: int = 256, se_floor: float = 0.10) -> dict:
    \"\"\"
    Fits shared Gaussian whitener with explicit Ledoit-Wolf shrinkage floor `se_floor`.
    Returns unsupervised top-64 principal components V_top64 (zero label leakage!).
    \"\"\"
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

    # Unsupervised top-64 principal components (zero label leakage)
    V_top64 = V_f32[:, :min(64, d)].copy()

    Z = (((X_tr_f32 - mx) / sx) @ V_f32).astype(np.float64)

    mu, P, ng = _two_way(Z, y, pinv, G, K=18, iters=4)
    E = Z - mu[y] - P[pinv]
    dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = (E.T @ E) / dof
    ae = max(float(ledoit_wolf_shrinkage(E, assume_centered=True)), se_floor)
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

    return {
        "mx": mx, "sx": sx, "W": W_f32, "m": m_whitened, "lam": lam,
        "V_top64": V_top64, "T1": 2.40, "Tg": 2.35
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

    def single(self, u: np.ndarray, temperature: float = 2.40) -> np.ndarray:
        diff = u[None, :] - self.m
        diag = 1.0 + self.lam
        dist = np.sum(diff * diff / diag[None, :], axis=1)
        log_p = -0.5 * dist / max(float(temperature), 1e-4)
        return log_p - logsumexp(log_p)

    def group(self, U: np.ndarray, temperature: float = 2.35) -> np.ndarray:
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
        return solver.single(U_test[0], temperature=prm.get("T1", 2.40))[None, :]

    safe_groups = split_oversized_clusters_dist(D_dist, groups, max_cluster_size=14)
    solver = ExactAuthorPosterior(prm["m"], prm["lam"])
    T1 = prm.get("T1", 2.40)
    Tg = prm.get("Tg", 2.35)

    lp_dp = np.zeros((N, K), dtype=np.float64)
    for g in np.unique(safe_groups):
        idx_g = np.where(safe_groups == g)[0]
        if len(idx_g) == 1:
            lp_dp[idx_g] = solver.single(U_test[idx_g[0]], temperature=T1)
        else:
            lp_dp[idx_g] = solver.group(U_test[idx_g], temperature=Tg)
    return lp_dp
""")

    # Cell 5: 6-Checkpoint Ensemble across 4 Trunks with Tier 1 & Tier 2 Separation
    add_md("### Step 5: 6-Checkpoint Ensemble across 4 Trunks with Tier 1 & Tier 2 Separation")
    add_code("""claude_spec_logits = None
claude_classes = None

# Prior scaling multipliers for unsupervised spectral bank features
PRIOR_SCALES = {
    "qwen3_14b_p2": 1.00,
    "qwen25_14b_p2": 0.85,
    "deberta_p2": 0.80,
    "qwen3_14b_p1": 0.50,
    "instyle": 0.45,
    "qwen3_8b_claude": 0.30
}

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
            # 1. MAIN STACK (85% Mass): Canonical weights w_plain = 0.40, w_twin = 0.10 (groups_085 only)
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_MAIN += 0.40 * lp_plain

            if W_c is not None and N_test >= 2:
                lp_c_85, m85 = compute_centered_logits(Z_test, groups_085, W_c, b_c, LABELS)
                if m85.any():
                    LOGIT_MAIN[m85] += 0.10 * lp_c_85[m85]
                print(f"[lora_qwen3_14b -> LOGIT_MAIN] Added Plain (0.40) + Twin (0.10, {m85.sum()} rows).")

            # Update consensus distance
            Z_norm = Z_test - Z_test.mean(axis=0, keepdims=True)
            Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
            D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
            np.fill_diagonal(D_m, 0.0)
            D_neural_sum += 0.40 * D_m
            D_neural_weight_total += 0.40

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            # 2. COMPLEMENTARY LORA STACK: Plain (0.30) + Multi-Res Twins (0.10)
            LOGIT_COMP_LORA += 0.30 * lp_plain
            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85_c = fcluster(Z_link, t=0.85, criterion='distance')
                g72_c = fcluster(Z_link, t=0.72, criterion='distance')
                g60_c = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85_m, m85_c = compute_centered_logits(Z_test, g85_c, W_c, b_c, LABELS)
                lp_c_72_m, m72_c = compute_centered_logits(Z_test, g72_c, W_c, b_c, LABELS)
                lp_c_60_m, m60_c = compute_centered_logits(Z_test, g60_c, W_c, b_c, LABELS)

                if m85_c.any(): LOGIT_COMP_LORA[m85_c] += (0.50 * 0.10) * lp_c_85_m[m85_c]
                if m72_c.any(): LOGIT_COMP_LORA[m72_c] += (0.30 * 0.10) * lp_c_72_m[m72_c]
                if m60_c.any(): LOGIT_COMP_LORA[m60_c] += (0.20 * 0.10) * lp_c_60_m[m60_c]

            # 3. Regularized DP with se_floor = 0.10
            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256, se_floor=0.10)
            U_test = transform_lowram(Z_test, prm)

            # Unsupervised top-64 principal components (zero label leakage)
            scale = PRIOR_SCALES["qwen3_14b_p2"]
            S_tr_64 = (((Z_train_bb - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_te_64 = (((Z_test - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_tr_64 = (S_tr_64 / np.linalg.norm(S_tr_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            S_te_64 = (S_te_64 / np.linalg.norm(S_te_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            spectral_tr_blocks.append(S_tr_64)
            spectral_te_blocks.append(S_te_64)

            g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
            dp_posteriors["q3_14b"] = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
            print(f"[lora_qwen3_14b -> COMP_LIST] Stored Regularized DP Posterior & Prior-Scaled 64D Spectral Block (scale={scale}).")
            del U_test, prm

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
            # Notice: NOT added to LOGIT_MAIN! (Zero Part 1 dilution of Main Stack)
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_COMP_LORA += 0.14 * lp_plain

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total) if D_neural_weight_total > 0 else D_tfidf
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85_c = fcluster(Z_link, t=0.85, criterion='distance')
                g72_c = fcluster(Z_link, t=0.72, criterion='distance')
                g60_c = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85_m, m85_c = compute_centered_logits(Z_test, g85_c, W_c, b_c, LABELS)
                lp_c_72_m, m72_c = compute_centered_logits(Z_test, g72_c, W_c, b_c, LABELS)
                lp_c_60_m, m60_c = compute_centered_logits(Z_test, g60_c, W_c, b_c, LABELS)

                if m85_c.any(): LOGIT_COMP_LORA[m85_c] += (0.50 * 0.05) * lp_c_85_m[m85_c]
                if m72_c.any(): LOGIT_COMP_LORA[m72_c] += (0.30 * 0.05) * lp_c_72_m[m72_c]
                if m60_c.any(): LOGIT_COMP_LORA[m60_c] += (0.20 * 0.05) * lp_c_60_m[m60_c]

            # Regularized DP with se_floor = 0.35 (protects against overfitting)
            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256, se_floor=0.35)
            U_test = transform_lowram(Z_test, prm)

            scale = PRIOR_SCALES["qwen3_14b_p1"]
            S_tr_64 = (((Z_train_bb - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_te_64 = (((Z_test - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_tr_64 = (S_tr_64 / np.linalg.norm(S_tr_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            S_te_64 = (S_te_64 / np.linalg.norm(S_te_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            spectral_tr_blocks.append(S_tr_64)
            spectral_te_blocks.append(S_te_64)

            g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
            dp_posteriors["part1_14b"] = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
            print(f"[lora14b_v1 -> COMP_LIST] Stored Regularized DP Posterior & Prior-Scaled 64D Spectral Block (scale={scale}).")
            del U_test, prm

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
        # 1. MAIN STACK (85% Mass): Canonical weights w_plain = 0.20, w_twin = 0.10 (groups_085 only)
        lp_plain = log_softmax(Z_test @ W_p.T + b_p)
        LOGIT_MAIN += 0.20 * lp_plain

        if W_c is not None and N_test >= 2:
            lp_c_85, m85 = compute_centered_logits(Z_test, groups_085, W_c, b_c, LABELS)
            if m85.any():
                LOGIT_MAIN[m85] += 0.10 * lp_c_85[m85]
            print(f"[lora_qwen25_14b -> LOGIT_MAIN] Added Plain (0.20) + Twin (0.10, {m85.sum()} rows).")

        # Update consensus distance
        Z_norm = Z_test - Z_test.mean(axis=0, keepdims=True)
        Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
        D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
        np.fill_diagonal(D_m, 0.0)
        D_neural_sum += 0.20 * D_m
        D_neural_weight_total += 0.20

        D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
        D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

        # 2. COMPLEMENTARY LORA STACK: Plain (0.20) + Multi-Res Twins (0.08)
        LOGIT_COMP_LORA += 0.20 * lp_plain
        if W_c is not None and N_test >= 2:
            cond_dist = squareform(D_hybrid_curr, checks=False)
            Z_link = linkage(cond_dist, method='average')
            g85_c = fcluster(Z_link, t=0.85, criterion='distance')
            g72_c = fcluster(Z_link, t=0.72, criterion='distance')
            g60_c = fcluster(Z_link, t=0.60, criterion='distance')

            lp_c_85_m, m85_c = compute_centered_logits(Z_test, g85_c, W_c, b_c, LABELS)
            lp_c_72_m, m72_c = compute_centered_logits(Z_test, g72_c, W_c, b_c, LABELS)
            lp_c_60_m, m60_c = compute_centered_logits(Z_test, g60_c, W_c, b_c, LABELS)

            if m85_c.any(): LOGIT_COMP_LORA[m85_c] += (0.50 * 0.08) * lp_c_85_m[m85_c]
            if m72_c.any(): LOGIT_COMP_LORA[m72_c] += (0.30 * 0.08) * lp_c_72_m[m72_c]
            if m60_c.any(): LOGIT_COMP_LORA[m60_c] += (0.20 * 0.08) * lp_c_60_m[m60_c]

        # 3. Regularized DP with se_floor = 0.10
        prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256, se_floor=0.10)
        U_test = transform_lowram(Z_test, prm)

        scale = PRIOR_SCALES["qwen25_14b_p2"]
        S_tr_64 = (((Z_train_bb - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
        S_te_64 = (((Z_test - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
        S_tr_64 = (S_tr_64 / np.linalg.norm(S_tr_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
        S_te_64 = (S_te_64 / np.linalg.norm(S_te_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
        spectral_tr_blocks.append(S_tr_64)
        spectral_te_blocks.append(S_te_64)

        g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
        dp_posteriors["q25_14b"] = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
        print(f"[lora_qwen25_14b -> COMP_LIST] Stored Regularized DP Posterior & Prior-Scaled 64D Spectral Block (scale={scale}).")
        del U_test, prm

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
            # Notice: NOT added to LOGIT_MAIN! Routed strictly to Complementary Stack
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_COMP_LORA += 0.22 * lp_plain

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total) if D_neural_weight_total > 0 else D_tfidf
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85_c = fcluster(Z_link, t=0.85, criterion='distance')
                g72_c = fcluster(Z_link, t=0.72, criterion='distance')
                g60_c = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85_m, m85_c = compute_centered_logits(Z_test, g85_c, W_c, b_c, LABELS)
                lp_c_72_m, m72_c = compute_centered_logits(Z_test, g72_c, W_c, b_c, LABELS)
                lp_c_60_m, m60_c = compute_centered_logits(Z_test, g60_c, W_c, b_c, LABELS)

                if m85_c.any(): LOGIT_COMP_LORA[m85_c] += (0.50 * 0.08) * lp_c_85_m[m85_c]
                if m72_c.any(): LOGIT_COMP_LORA[m72_c] += (0.30 * 0.08) * lp_c_72_m[m72_c]
                if m60_c.any(): LOGIT_COMP_LORA[m60_c] += (0.20 * 0.08) * lp_c_60_m[m60_c]

            # Regularized DP with se_floor = 0.35
            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256, se_floor=0.35)
            U_test = transform_lowram(Z_test, prm)

            scale = PRIOR_SCALES["instyle"]
            S_tr_64 = (((Z_train_bb - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_te_64 = (((Z_test - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_tr_64 = (S_tr_64 / np.linalg.norm(S_tr_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            S_te_64 = (S_te_64 / np.linalg.norm(S_te_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            spectral_tr_blocks.append(S_tr_64)
            spectral_te_blocks.append(S_te_64)

            g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
            dp_posteriors["instyle"] = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
            print(f"[instyle_qwen3_8b -> COMP_LIST] Stored Regularized DP Posterior & Prior-Scaled 64D Spectral Block (scale={scale}).")
            del U_test, prm

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
            # Routed strictly to Complementary Stack
            lp_plain = log_softmax(Z_test @ W_p.T + b_p)
            LOGIT_COMP_LORA += 0.14 * lp_plain

            D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total) if D_neural_weight_total > 0 else D_tfidf
            D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

            if W_c is not None and N_test >= 2:
                cond_dist = squareform(D_hybrid_curr, checks=False)
                Z_link = linkage(cond_dist, method='average')
                g85_c = fcluster(Z_link, t=0.85, criterion='distance')
                g72_c = fcluster(Z_link, t=0.72, criterion='distance')
                g60_c = fcluster(Z_link, t=0.60, criterion='distance')

                lp_c_85_m, m85_c = compute_centered_logits(Z_test, g85_c, W_c, b_c, LABELS)
                lp_c_72_m, m72_c = compute_centered_logits(Z_test, g72_c, W_c, b_c, LABELS)
                lp_c_60_m, m60_c = compute_centered_logits(Z_test, g60_c, W_c, b_c, LABELS)

                if m85_c.any(): LOGIT_COMP_LORA[m85_c] += (0.50 * 0.05) * lp_c_85_m[m85_c]
                if m72_c.any(): LOGIT_COMP_LORA[m72_c] += (0.30 * 0.05) * lp_c_72_m[m72_c]
                if m60_c.any(): LOGIT_COMP_LORA[m60_c] += (0.20 * 0.05) * lp_c_60_m[m60_c]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_sub, prompts_sub, d_target=256, se_floor=0.35)
            scale = PRIOR_SCALES["qwen3_8b_claude"]
            S_tr_64 = (((Z_train_bb - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_te_64 = (((Z_test - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
            S_tr_64 = (S_tr_64 / np.linalg.norm(S_tr_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            S_te_64 = (S_te_64 / np.linalg.norm(S_te_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
            spectral_tr_blocks.append(S_tr_64)
            spectral_te_blocks.append(S_te_64)
            print(f"[ntl-claude-head -> COMP_LIST] Stored Prior-Scaled 64D Spectral Block (scale={scale}).")
            del prm

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
        # 1. MAIN STACK (85% Mass): Canonical weight w_plain = 0.25 (twin = 0.00)
        lp_plain = log_softmax(Z_test_deb @ W_p.T + b_p)
        LOGIT_MAIN += 0.25 * lp_plain
        print("[deberta_v3_large -> LOGIT_MAIN] Added Plain (0.25).")

        # Update consensus distance
        Z_norm = Z_test_deb - Z_test_deb.mean(axis=0, keepdims=True)
        Z_norm /= np.linalg.norm(Z_norm, axis=1, keepdims=True).clip(min=1e-8)
        D_m = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
        np.fill_diagonal(D_m, 0.0)
        D_neural_sum += 0.25 * D_m
        D_neural_weight_total += 0.25

        D_neural_curr = D_neural_sum / max(1e-6, D_neural_weight_total)
        D_hybrid_curr = 0.55 * D_tfidf + 0.45 * D_neural_curr

        # 2. COMPLEMENTARY LORA STACK: Plain (0.18) + Multi-Res Twins (0.06)
        LOGIT_COMP_LORA += 0.18 * lp_plain
        if W_c is not None and N_test >= 2:
            cond_dist = squareform(D_hybrid_curr, checks=False)
            Z_link = linkage(cond_dist, method='average')
            g85_c = fcluster(Z_link, t=0.85, criterion='distance')
            g72_c = fcluster(Z_link, t=0.72, criterion='distance')
            g60_c = fcluster(Z_link, t=0.60, criterion='distance')

            lp_c_85_m, m85_c = compute_centered_logits(Z_test_deb, g85_c, W_c, b_c, LABELS)
            lp_c_72_m, m72_c = compute_centered_logits(Z_test_deb, g72_c, W_c, b_c, LABELS)
            lp_c_60_m, m60_c = compute_centered_logits(Z_test_deb, g60_c, W_c, b_c, LABELS)

            if m85_c.any(): LOGIT_COMP_LORA[m85_c] += (0.50 * 0.06) * lp_c_85_m[m85_c]
            if m72_c.any(): LOGIT_COMP_LORA[m72_c] += (0.30 * 0.06) * lp_c_72_m[m72_c]
            if m60_c.any(): LOGIT_COMP_LORA[m60_c] += (0.20 * 0.06) * lp_c_60_m[m60_c]

        # 3. Regularized DP with se_floor = 0.10
        prm = fit_shared_gaussian_lowram(Z_train_deb, y_sub, prompts_sub, d_target=256, se_floor=0.10)
        U_test = transform_lowram(Z_test_deb, prm)

        scale = PRIOR_SCALES["deberta_p2"]
        S_tr_64 = (((Z_train_deb - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
        S_te_64 = (((Z_test_deb - prm["mx"]) / prm["sx"]) @ prm["V_top64"]).astype(np.float32)
        S_tr_64 = (S_tr_64 / np.linalg.norm(S_tr_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
        S_te_64 = (S_te_64 / np.linalg.norm(S_te_64, axis=1, keepdims=True).clip(min=1e-6)) * scale
        spectral_tr_blocks.append(S_tr_64)
        spectral_te_blocks.append(S_te_64)

        g65 = fcluster(linkage(squareform(D_hybrid_curr, checks=False), method='average'), t=0.65, criterion='distance') if N_test >= 2 else np.ones(N_test, dtype=int)
        dp_posteriors["deberta"] = scores_exact_dp(U_test, g65, prm, D_hybrid_curr)
        print(f"[deberta_v3_large -> COMP_LIST] Stored Regularized DP Posterior & Prior-Scaled 64D Spectral Block (scale={scale}).")
        del U_test, prm

    del Z_deb, Z_test_deb, Z_train_deb, heads_deb_sd
    gc.collect()

log_ram("After Trunk 4 (DeBERTa) Cleanup")

# Append Secondary LoRAs to COMP_LIST (weight = 0.35)
P_comp_lora = softmax(LOGIT_COMP_LORA, axis=1)
COMP_LIST.append((0.35, P_comp_lora))
print(f"[COMP_LIST] Added Secondary LoRAs & Multi-Resolution Twins (weight=0.35).")

# Append Blended Regularized DP Posteriors to COMP_LIST (weight = 0.30)
dp_weights = {
    "q3_14b": 0.38,
    "q25_14b": 0.24,
    "deberta": 0.20,
    "instyle": 0.10,
    "part1_14b": 0.08
}
active_dp_keys = [k for k in dp_weights.keys() if k in dp_posteriors]
if len(active_dp_keys) > 0:
    sum_w_dp = sum(dp_weights[k] for k in active_dp_keys)
    lp_dp_blend = sum((dp_weights[k] / sum_w_dp) * dp_posteriors[k] for k in active_dp_keys)
    P_dp_blend = softmax(lp_dp_blend, axis=1)
    COMP_LIST.append((0.30, P_dp_blend))
    print(f"[COMP_LIST] Added Blended Regularized DP Posteriors (weight=0.30, components={active_dp_keys}).")
""")

    # Cell 6: Module 4 — Leak-Free 384-Dim Joint Spectral Bank, 85/15 Mixture, Shields & 300-Step Sinkhorn
    add_md("### Step 6: Module 4 — Leak-Free 384-Dim Spectral Bank, 85/15 Mixture, Shields & 300-Step Sinkhorn")
    add_code("""# 1. Fit Leak-Free Full-Rank 384-Dim Joint Spectral Bank on 1,620 Rows (Unsupervised Features)
if len(spectral_tr_blocks) > 0:
    X_spec_tr = np.hstack(spectral_tr_blocks)
    X_spec_te = np.hstack(spectral_te_blocks)
    print(f"[Joint Spectral Bank] Formed matrices: Train={X_spec_tr.shape}, Test={X_spec_te.shape}")

    # 5-fold GroupKFold OOF Balanced Accuracy (zero label leakage!)
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y_sub), dtype=int)
    for tr_idx, val_idx in gkf.split(X_spec_tr, y_sub, groups=prompts_sub):
        clf_fold = LogisticRegression(C=0.04, max_iter=2000, random_state=42)
        clf_fold.fit(X_spec_tr[tr_idx], y_sub[tr_idx])
        oof_preds[val_idx] = clf_fold.predict(X_spec_tr[val_idx])
    oof_bacc = balanced_accuracy_score(y_sub, oof_preds)
    print(f"[LEAK-FREE SPECTRAL BANK 5-FOLD OOF] Realistic Generalization BAcc = {oof_bacc:.4f}")

    # Plain Logistic Regression
    clf_plain = LogisticRegression(C=0.04, max_iter=2000, random_state=42)
    clf_plain.fit(X_spec_tr, y_sub)
    lp_spec_plain = clf_plain.predict_log_proba(X_spec_te)
    if hasattr(clf_plain, 'classes_'):
        lp_spec_plain = align_cols(lp_spec_plain, [LABELS[c] for c in clf_plain.classes_])

    # Prompt-Centered Logistic Regression
    X_spec_tr_c = X_spec_tr.copy()
    for p in np.unique(prompts_sub):
        mask = (prompts_sub == p)
        X_spec_tr_c[mask] -= X_spec_tr_c[mask].mean(axis=0, keepdims=True)
    clf_center = LogisticRegression(C=0.03, max_iter=2000, random_state=42)
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

    lp_spec_joint = lp_spec_plain + 0.35 * lp_spec_center
    COMP_LIST.append((0.15, softmax(lp_spec_joint, axis=1)))
    print("[COMP_LIST] Added Leak-Free Prior-Scaled Spectral Bank (weight=0.15).")

# 2. Combine 85% Main Stack and 15% Complementary Stack in Probability Space
R_main = softmax(LOGIT_MAIN, axis=1)
if len(COMP_LIST) > 0:
    comp_weight_total = sum(w for w, _ in COMP_LIST)
    P_comp = sum(w * P_c for w, P_c in COMP_LIST) / max(1e-8, comp_weight_total)
    P_mix = 0.85 * R_main + 0.15 * P_comp
    print(f"[Probability Mixture] Convex blend: 0.85 * R_main + 0.15 * P_comp ({len(COMP_LIST)} complementary streams).")
else:
    P_mix = R_main
    print("[Probability Mixture] Pure R_main (no complementary streams).")

# 3. Apply Gentle Pre-Sinkhorn Claude Specialist Gate & Orthographic Shield
# Gentle Claude Specialist Gate (mass-preserving within Claude family)
if claude_spec_logits is not None:
    claude_cols = FAMILIES["claude"]
    P_claude_head = softmax(claude_spec_logits, axis=-1)
    if claude_classes is not None:
        src_c = [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in claude_classes]
        c_idx = [src_c.index(LABELS[c]) for c in claude_cols]
        P_claude_head = P_claude_head[:, c_idx]

    pred_main = R_main.argmax(axis=1)
    claude_mask = np.isin(pred_main, claude_cols)
    if claude_mask.any():
        for i in np.where(claude_mask)[0]:
            tot_mass = P_mix[i, claude_cols].sum()
            blended = 0.85 * P_mix[i, claude_cols] + 0.15 * (tot_mass * P_claude_head[i])
            P_mix[i, claude_cols] = blended
        print(f"[Claude Specialist Gate] Refined {claude_mask.sum()} passages predicted in Claude family.")

# Gentle Curly Quote Guard on Claude/GLM
claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
for i, txt in enumerate(test_texts):
    txt_s = str(txt)
    if any(q in txt_s for q in ['\\u2018', '\\u2019', '\\u201c', '\\u201d']):
        P_mix[i, claude_and_glm_idx] *= np.exp(-1.0)

# Re-normalize rows to simplex
P_mix /= P_mix.sum(axis=1, keepdims=True).clip(min=1e-12)

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
P_final = sinkhorn_balancing(P_mix, iters=300)

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
print("\\n=== Version 42 Pipeline Execution Successfully Completed ===")
""")

    out_file = Path("kaggle_submission.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated {out_file} ({out_file.stat().st_size} bytes).")

if __name__ == "__main__":
    create_v42_pipeline_notebook()
