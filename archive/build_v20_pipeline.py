import json
from pathlib import Path

def create_v20_pipeline_notebook():
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
            "source": [l + "\n" for l in source.splitlines()]
        })

    def add_code(source: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in source.splitlines()]
        })

    # Header Cell
    add_md("""# Name That LLM: v20 (0.65+ Target) Pipeline
### 6-Checkpoint In-Place Trunk Reuse, Fine-Tuned DeBERTa Loader, Joint Spectral Bank, Exact 2^18 Subset DP & 50-Step Sinkhorn
- **6-Checkpoint Multi-Backbone Ensemble**: Qwen3-14B (Part 2 + Part 1), Qwen2.5-14B (Part 2), Qwen3-8B (Claude Head + Part 1), and DeBERTa-v3-Large Fine-Tuned Encoder.
- **In-Place Trunk Reuse**: Shared 14B and 8B trunks loaded from disk only once; adapters merged and unmerged in-place (+delta, -delta) with zero VRAM waste.
- **Dedicated DeBERTa Fine-Tuned Encoder Loader**: Native encoder loader directly utilizing full model weights and calibrated pooling.
- **Exact 2^18 Bitmask Subset Dynamic Programming (ExactAuthorPosterior)**: Forward-backward DP over 262,144 author subsets guaranteeing exact prompt uniqueness without log-prob calibration destruction.
- **Joint Cross-Backbone Spectral Readout Bank**: Unified L2-normalized representations driving whitened ridge readouts with zero extra GPU forward cost.
- **50-Step Log-Domain Sinkhorn-Knopp & Post-Sinkhorn Bayesian Shield**: Global marginal balancing followed by calibrated -2.0 orthographic constraints.
""")

    # Cell 1: Environment, Dependencies & Canonical 18 Labels Order
    add_md("### Step 1: Environment, Dependencies & Canonical 18 Labels Order")
    add_code("""import os
import gc
import sys
import time
import json
import re
import subprocess
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import logsumexp
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn

# Bypass torchao version mismatch in Kaggle container
subprocess.run(['pip', 'uninstall', '-y', 'torchao'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

N_GPUS = torch.cuda.device_count()
print(f'CUDA Available: {torch.cuda.is_available()} | Device Count: {N_GPUS}')
for i in range(N_GPUS):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)} | VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB')

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def log_softmax(L):
    return L - logsumexp(L, axis=1, keepdims=True)

# Master Canonical 18 Labels Order
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}
ID2LAB = {i: l for i, l in enumerate(LABELS)}

def align_to_master(arr, src_labels, dst_labels=LABELS):
    src_list = list(src_labels)
    idx = [src_list.index(l) for l in dst_labels]
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[idx]
    elif arr.ndim == 2:
        if arr.shape[0] == len(src_list) and arr.shape[1] != len(src_list):
            return arr[idx, :]
        elif arr.shape[1] == len(src_list):
            return arr[:, idx]
        else:
            return arr[:, idx]
    return arr
""")

    # Cell 2: Data & Model Path Discovery (Deep config.json Inspection)
    add_md("### Step 2: Data & Model Path Discovery")
    add_code("""INPUT_DIR = Path('/kaggle/input')

# Locate train.csv and test.csv
train_paths = list(INPUT_DIR.rglob('train.csv'))
test_paths = list(INPUT_DIR.rglob('test.csv'))

train_path = train_paths[0] if train_paths else Path('train.csv')
test_path = test_paths[0] if test_paths else Path('test.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

N_train = len(df_train)
N_test = len(df_test)
print(f'Train path: {train_path} ({N_train} rows)')
print(f'Test path:  {test_path} ({N_test} rows)')

y_train = np.array([LAB2I[m] for m in df_train['model']])
prompts_train = df_train['prompt_id'].values
test_texts = df_test['text'].fillna('').tolist()

# Robust Base Model Discovery via config.json inspection (resolves exact leaf folders)
def find_base_model(target: str) -> Optional[Path]:
    for p in INPUT_DIR.rglob('config.json'):
        p_str = str(p).lower()
        if 'adapter' in p_str or 'checkpoint' in p_str:
            continue
        try:
            cfg = json.load(open(p, 'r'))
            m_type = str(cfg.get('model_type', '')).lower()
            h_size = int(cfg.get('hidden_size', 0))
            n_layers = int(cfg.get('num_hidden_layers', 0))

            if target == 'qwen3_14b':
                if 'qwen' in m_type and h_size == 5120 and n_layers == 40:
                    return p.parent
                if ('qwen-3' in p_str or 'qwen3' in p_str) and '14b' in p_str:
                    return p.parent
            elif target == 'qwen25_14b':
                if 'qwen' in m_type and h_size == 5120 and n_layers == 48:
                    return p.parent
                if ('qwen2.5' in p_str or 'qwen-2.5' in p_str) and '14b' in p_str:
                    return p.parent
            elif target == 'qwen3_8b':
                if 'qwen' in m_type and h_size == 4096 and n_layers == 36:
                    return p.parent
                if ('qwen-3' in p_str or 'qwen3' in p_str) and '8b' in p_str:
                    return p.parent
            elif target == 'deberta':
                if ('deberta' in m_type or 'deberta' in p_str) and 'encoder' not in p_str:
                    return p.parent
        except Exception:
            pass
    return None

BASE_QWEN3_14B = find_base_model('qwen3_14b')
BASE_QWEN25_14B = find_base_model('qwen25_14b')
BASE_QWEN3_8B = find_base_model('qwen3_8b')
BASE_DEBERTA = find_base_model('deberta')

print(f'Discovered Base Qwen3-14B:   {BASE_QWEN3_14B}')
print(f'Discovered Base Qwen2.5-14B: {BASE_QWEN25_14B}')
print(f'Discovered Base Qwen3-8B:    {BASE_QWEN3_8B}')
print(f'Discovered Base DeBERTa:     {BASE_DEBERTA}')

# Checkpoints discovery across mounted datasets
CKPTS = {}
for p in INPUT_DIR.rglob('*'):
    if not p.is_dir():
        continue
    sp = str(p).lower()
    if 'lora_qwen3_14b' in sp and (p / 'heads.safetensors').exists():
        CKPTS['qwen3_14b_p2'] = p
    elif 'lora14b_v1' in sp and (p / 'heads.safetensors').exists():
        CKPTS['qwen3_14b_p1'] = p
    elif 'lora_qwen25_14b' in sp and (p / 'heads.safetensors').exists():
        CKPTS['qwen25_14b_p2'] = p
    elif ('ntl-qwen3-8b-lora-claude-head' in sp or 'claude-head' in sp) and (p / 'readouts.npz').exists():
        CKPTS['qwen3_8b_claude'] = p
    elif 'lora8b_v1' in sp and (p / 'heads.safetensors').exists():
        CKPTS['qwen3_8b_p1'] = p
    elif 'deberta_v3_large' in sp and ((p / 'head.safetensors').exists() or (p / 'encoder').exists()):
        CKPTS['deberta_p2'] = p

print(f'Discovered Checkpoints ({len(CKPTS)}): {list(CKPTS.keys())}')

# Locate public pool
pool_file = None
for name in ['pool_v4.parquet', 'pool_v3.parquet', 'pool_v2.parquet', 'pool_v1.parquet']:
    m = list(INPUT_DIR.rglob(name))
    if m:
        pool_file = m[0]
        break
print(f'Discovered Passage Pool:     {pool_file}')
""")

    # Cell 3: Module 1 — 4-View Topic-Blind LinearSVC (78k Passages)
    add_md("### Step 3: Module 1 — 4-View Topic-Blind LinearSVC on train.csv + pool_v4.parquet")
    add_code("""print('\\n=== [Module 1] Training 4-View Topic-Blind LinearSVC ===')
fit_texts = df_train['text'].fillna('').astype(str).tolist()
fit_y = y_train.copy()

if pool_file is not None and pool_file.exists():
    print(f'[Module 1] Loading public pool from {pool_file}...')
    df_pool = pd.read_parquet(pool_file)
    df_pool = df_pool[df_pool['model'].isin(LABELS)].copy()
    if N_test < 50:
        # Fast interactive commit mode: sample pool rows
        df_pool = df_pool.sample(min(8000, len(df_pool)), random_state=RANDOM_SEED)
    fit_texts += df_pool['text'].fillna('').astype(str).tolist()
    fit_y = np.r_[fit_y, df_pool['model'].map(LAB2I).values]
    print(f'[Module 1] Combined training passages: {len(fit_texts)} across 18 classes.')

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

def to_shape(t: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m.group(0)[0].isupper() else "x", t)

def to_function(t: str) -> str:
    tokens = re.findall(r"(?u)\b\w+\b|[^\w\s]", t)
    return " ".join([w if w.lower() in STOP_WORDS else "CONTENT" for w in tokens])

t0 = time.time()
vec_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=250000, lowercase=False, sublinear_tf=True, min_df=2)
vec_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=180000, lowercase=False, sublinear_tf=True, min_df=2)
vec_shape = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), max_features=60000, lowercase=False, sublinear_tf=True, min_df=2)
vec_func = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=50000, lowercase=False, sublinear_tf=True, min_df=2)

print('[Module 1] Extracting 4-view TF-IDF features on train passages...')
X_tr_c = vec_char.fit_transform(fit_texts)
X_tr_w = vec_word.fit_transform(fit_texts)
X_tr_s = vec_shape.fit_transform([to_shape(t) for t in fit_texts])
X_tr_f = vec_func.fit_transform([to_function(t) for t in fit_texts])

X_fit_svc = sparse.hstack([1.0 * X_tr_c, 0.7 * X_tr_w, 0.5 * X_tr_s, 0.5 * X_tr_f], format='csr')

svc_4view = LinearSVC(C=0.5, random_state=RANDOM_SEED, max_iter=5000, class_weight='balanced')
svc_4view.fit(X_fit_svc, fit_y)
print(f'[Module 1] 4-View LinearSVC fitted on {X_fit_svc.shape[1]} features in {time.time() - t0:.2f}s.')

# Extract on test texts
X_te_c = vec_char.transform(test_texts)
X_te_w = vec_word.transform(test_texts)
X_te_s = vec_shape.transform([to_shape(t) for t in test_texts])
X_te_f = vec_func.transform([to_function(t) for t in test_texts])
X_test_svc = sparse.hstack([1.0 * X_te_c, 0.7 * X_te_w, 0.5 * X_te_s, 0.5 * X_te_f], format='csr')

log_p_4view_svc = log_softmax(svc_4view.decision_function(X_test_svc))

LOGIT = 1.05 * log_p_4view_svc
print('[Module 1] Initialized master LOGIT with 1.05 * 4-View LinearSVC.')
""")

    # Cell 4: Module 2 — Dual-Resolution Topic Clustering
    add_md("### Step 4: Module 2 — Dual-Resolution Topic Clustering (0.85 & 0.65)")
    add_code("""tfidf_content = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100000)
Xt_tfidf = tfidf_content.fit_transform(test_texts)
D_tfidf = np.clip(1.0 - (Xt_tfidf @ Xt_tfidf.T).toarray(), 0.0, None)
np.fill_diagonal(D_tfidf, 0.0)

if len(test_texts) > 1:
    Z_link = linkage(squareform(D_tfidf, checks=False), method="average")
    # 1. Standard 0.85 clustering for Two-Stage centered.weight heads
    groups_085 = fcluster(Z_link, t=0.85, criterion="distance")
    # 2. High-precision 0.65 clustering for ExactAuthorPosterior
    groups_065 = fcluster(Z_link, t=0.65, criterion="distance")
else:
    groups_085 = np.array([1])
    groups_065 = np.array([1])

sz_085 = np.bincount(groups_085)[groups_085]
twin_rows_085 = (sz_085 >= 2) & (sz_085 <= 18)
print(f'Test groups (0.85): {len(np.unique(groups_085))} | Twin-eligible rows (2 <= size <= 18): {twin_rows_085.sum()}/{len(test_texts)}')
print(f'Test groups (0.65): {len(np.unique(groups_065))}')
""")

    # Cell 5: Module 4 — Exact 2^18 Bitmask Subset DP & wPCA Gaussian
    add_md("### Step 5: Module 4 — Exact 2^18 Bitmask Subset DP (ExactAuthorPosterior) & wPCA Gaussian Readout")
    add_code("""@lru_cache(maxsize=4)
def _bitmask_levels(k: int = 18):
    masks = np.arange(1 << k, dtype=np.int64)
    counts = np.fromiter((int(x).bit_count() for x in masks), dtype=np.uint8, count=len(masks))
    return tuple(masks[counts == n] for n in range(k + 1))

class ExactAuthorPosterior:
    \"\"\"
    Exact shared-prompt Gaussian posteriors under author uniqueness (1-to-1 assignment)
    via forward-backward dynamic programming over the 2^18 = 262,144 author subsets.
    \"\"\"
    def __init__(self, means: np.ndarray, prompt_variances: np.ndarray):
        self.m = np.asarray(means, dtype=np.float64)
        self.lam = np.asarray(prompt_variances, dtype=np.float64)
        self.k, self.d = self.m.shape
        self.levels = _bitmask_levels(self.k)
        self.subset_means = {}

    def group(self, U: np.ndarray, temperature: float = 1.0, extra_log_scores: np.ndarray = None,
              model_weight: float = 1.0, extra_weight: float = 0.0) -> np.ndarray:
        U = np.asarray(U, dtype=np.float64)
        n = len(U)
        if n == 1 or n > self.k:
            val = -0.5 * ((U[:, None, :] - self.m[None, :, :]) ** 2 / (1.0 + self.lam)).sum(2) / temperature
            val *= model_weight
            if extra_log_scores is not None and extra_weight > 0:
                val += extra_weight * extra_log_scores
            return val - logsumexp(val, axis=1, keepdims=True)

        if n not in self.subset_means:
            masks = self.levels[n]
            membership = ((masks[:, None] >> np.arange(self.k)) & 1).astype(np.float64)
            self.subset_means[n] = membership @ self.m

        terminal = self.levels[n]
        gamma = self.lam / (1.0 + n * self.lam)
        residual = U.sum(0)[None, :] - self.subset_means[n]
        correction = 0.5 * np.einsum("nd,d,nd->n", residual, gamma, residual) / temperature
        correction *= model_weight
        correction -= correction.max()

        A = -0.5 * ((U[:, None, :] - self.m[None, :, :]) ** 2).sum(2) / temperature
        A *= model_weight
        if extra_log_scores is not None and extra_weight > 0:
            A += extra_weight * extra_log_scores
        A -= A.max(1, keepdims=True)

        forward = np.full(1 << self.k, -np.inf)
        forward[0] = 0.0
        for count in range(1, n + 1):
            masks = self.levels[count]
            for author in range(self.k):
                bit = 1 << author
                chosen = masks[(masks & bit) != 0]
                forward[chosen] = np.logaddexp(forward[chosen], forward[chosen ^ bit] + A[count - 1, author])

        backward = np.full_like(forward, -np.inf)
        backward[terminal] = correction
        for count in range(n - 1, -1, -1):
            masks = self.levels[count]
            for author in range(self.k):
                bit = 1 << author
                chosen = masks[(masks & bit) == 0]
                backward[chosen] = np.logaddexp(backward[chosen], A[count, author] + backward[chosen | bit])

        result = np.empty((n, self.k), dtype=np.float64)
        for row in range(n):
            masks = self.levels[row]
            for author in range(self.k):
                bit = 1 << author
                chosen = masks[(masks & bit) == 0]
                result[row, author] = logsumexp(forward[chosen] + A[row, author] + backward[chosen | bit]) - backward[0]

        result -= logsumexp(result, axis=1, keepdims=True)
        return result

def _shrunk(S, a):
    d = S.shape[0]
    return (1.0 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y, pinv, G, K=18, iters=4):
    P = np.zeros((G, Z.shape[1]))
    cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y)
    mu = np.zeros((K, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls:
            mu[c] = R[y == c].mean(0)
        R = Z - mu[y]
        P = np.zeros_like(P)
        np.add.at(P, pinv, R)
        P /= np.maximum(cnt_p[:, None], 1.0)
    return mu, P, cnt_p

def fit_shared_gaussian(X, y, prompts, cfg=None):
    \"\"\"Fits within-prompt whitened PCA and two-way additive author-prompt decomposition.\"\"\"
    c = {"d": 256, "se_shrink": "lw", "sp_shrink": "lw", "lam_floor": 1e-4, "sp_scale": 1.0, "T1": 2.35, "Tg": 2.30}
    c.update(cfg or {})
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True)
    G = len(pu)
    mx = X.mean(0)
    sx = np.where(X.std(0) > 1e-8, X.std(0), 1.0)
    Xs = (X - mx) / sx

    # Within-Prompt Whitened PCA (wPCA)
    Pm = np.zeros((G, D))
    np.add.at(Pm, pinv, Xs)
    Pm /= np.maximum(np.bincount(pinv)[:, None], 1.0)
    R = Xs - Pm[pinv]
    d = int(min(c["d"], max(N - G - 1, 1), D))

    if N < D:
        w, U = np.linalg.eigh(R @ R.T)
        o = np.argsort(w)[::-1][:d]
        w, U = np.clip(w[o], 1e-12, None), U[:, o]
        V = (R.T @ U) / np.sqrt(w)
    else:
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt[:d].T

    Z = Xs @ V
    d = Z.shape[1]
    mu, P, ng = _two_way(Z, y, pinv, G, K=len(LABELS))
    E = Z - mu[y] - P[pinv]
    dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = (E.T @ E) / dof
    ae = float(ledoit_wolf_shrinkage(E, assume_centered=True)) if c["se_shrink"] == "lw" else 0.1
    Se = _shrunk(Se_raw, ae)

    Pc = P - P.mean(0)
    Sp_raw = (Pc.T @ Pc) / max(G - 1, 1) - np.mean(1.0 / np.maximum(ng, 1.0)) * Se_raw
    ap = float(ledoit_wolf_shrinkage(Pc, assume_centered=True)) if c["sp_shrink"] == "lw" else 0.1
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = ((1.0 - ap) * Sp_raw + ap * (tr / d) * np.eye(d)) * c["sp_scale"]

    L = np.linalg.cholesky(Se + 1e-6 * np.eye(d))
    Li = np.linalg.inv(L)
    M = 0.5 * ((Li @ Sp @ Li.T) + (Li @ Sp @ Li.T).T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, c["lam_floor"], None)
    A = Q.T @ Li
    W = V @ A.T
    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": c["T1"], "Tg": c["Tg"], "cfg": c}
""")

    # Cell 6: Module 3 — In-Place LoRA Merger & 6-Checkpoint Extractor
    add_md("### Step 6: Module 3 — In-Place LoRA Merger, Dedicated DeBERTa Loader & 6-Checkpoint Extractor")
    add_code("""@torch.no_grad()
def merge_lora(model, adapter_dir: Union[str, Path], sign: float = 1.0):
    adapter_path = Path(adapter_dir)
    cfg_file = adapter_path / "adapter_config.json"
    if not cfg_file.exists():
        cfg_file = adapter_path / "adapter" / "adapter_config.json"
    if not cfg_file.exists():
        print(f"[LoRA Merge] Warning: No adapter_config.json found in {adapter_dir}")
        return False

    acfg = json.load(open(cfg_file))
    scale = float(sign) * (float(acfg.get("lora_alpha", 32)) / float(acfg.get("r", 16)))

    st_file = adapter_path / "adapter_model.safetensors"
    if not st_file.exists():
        st_file = adapter_path / "adapter" / "adapter_model.safetensors"
    if not st_file.exists():
        print(f"[LoRA Merge] Warning: No adapter_model.safetensors found in {adapter_dir}")
        return False

    sd = load_file(str(st_file))
    modules = dict(model.named_modules())
    n_merged = 0
    for k, A in sd.items():
        if ".lora_A." in k:
            name = k.split(".lora_A.")[0]
            rel = name[name.index("layers."):] if "layers." in name else name
            target_mod = None
            if rel in modules:
                target_mod = modules[rel]
            elif f"model.{rel}" in modules:
                target_mod = modules[f"model.{rel}"]
            elif name in modules:
                target_mod = modules[name]

            if target_mod is not None and hasattr(target_mod, "weight"):
                kB = k.replace(".lora_A.", ".lora_B.")
                if kB in sd:
                    Bm = sd[kB]
                    dev = target_mod.weight.device
                    dtype = target_mod.weight.dtype
                    delta = (Bm.float().to(dev) @ A.float().to(dev)) * scale
                    target_mod.weight.data.add_(delta.to(dtype))
                    n_merged += 1
    action = "Merged" if sign > 0 else "Unmerged"
    print(f"[LoRA {action}] {n_merged} linear LoRA weights in-place with scale={scale:.4f}.")
    return True

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
        self.tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<|endoftext|>"

        cfg = AutoConfig.from_pretrained(str(base_dir), trust_remote_code=True)
        cfg.num_hidden_layers = keep_layers

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

        hidden_size = cfg.hidden_size
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=getattr(cfg, "rms_norm_eps", 1e-6))
            for _ in layers
        ]).to(self.device_norms).float()

        for l_idx, nm in zip(layers, self.norms):
            try:
                base_lyr = self.model.layers[l_idx] if hasattr(self.model, "layers") else self.model.model.layers[l_idx]
                if hasattr(base_lyr, "post_attention_layernorm"):
                    nm.weight.data.copy_(base_lyr.post_attention_layernorm.weight.data.to(self.device_norms).float())
                elif hasattr(base_lyr, "input_layernorm"):
                    nm.weight.data.copy_(base_lyr.input_layernorm.weight.data.to(self.device_norms).float())
            except Exception:
                pass

    @torch.no_grad()
    def extract(self, texts: List[str], batch_size: int = 8, end_id: Optional[int] = None) -> np.ndarray:
        if end_id is None:
            end_id = getattr(self.tok, "eos_token_id", 151643) or 151643

        raw_ids = self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        ids = [x + [end_id] for x in raw_ids]

        H = self.norms[0].weight.shape[0]
        D_out = len(self.layers) * H
        Z = np.empty((len(ids), D_out), dtype=np.float32)

        t0 = time.time()
        for i in range(0, len(ids), batch_size):
            b_ids = ids[i:i + batch_size]
            enc = self.tok.pad({"input_ids": b_ids}, padding=True, return_tensors="pt")
            idx = np.arange(i, i + len(b_ids))

            inp = enc["input_ids"].to(self.model.device)
            attn_mask = enc["attention_mask"].to(self.model.device)

            out = self.model(inp, attention_mask=attn_mask, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states

            # Attention-Sink Zeroing: zero position 0
            m = attn_mask.unsqueeze(-1).float().to(self.device_norms)
            m[:, 0, :] = 0.0
            den = m.sum(dim=1).clamp(min=1.0)

            pooled_parts = []
            for l_idx, nm in zip(self.layers, self.norms):
                h_l = hs[l_idx].to(self.device_norms).float()
                if not torch.isfinite(h_l).all():
                    h_l = torch.nan_to_num(h_l, nan=0.0, posinf=100.0, neginf=-100.0)
                mean_vec = (h_l * m).sum(dim=1) / den
                pooled_parts.append(nm(mean_vec))

            Z[idx] = torch.cat(pooled_parts, dim=-1).cpu().numpy()

        print(f"[Extractor] Extracted {len(ids)} samples in {time.time() - t0:.2f}s ({len(ids) / max(0.1, time.time() - t0):.1f} samples/s).")
        return Z

    def free_memory(self):
        print("[Extractor] Freeing GPU VRAM...")
        del self.model, self.norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Prepare train set to extract
if N_test < 50:
    prompts_for_tr = df_train['prompt_id'].unique()[:10]
    df_tr_eval = df_train[df_train['prompt_id'].isin(prompts_for_tr)].copy().reset_index(drop=True)
    y_tr_eval = np.array([LAB2I[m] for m in df_tr_eval['model']])
    prompts_tr_eval = df_tr_eval['prompt_id'].values
else:
    df_tr_eval = df_train.copy().reset_index(drop=True)
    y_tr_eval = y_train.copy()
    prompts_tr_eval = prompts_train.copy()

eval_texts = test_texts + df_tr_eval['text'].fillna('').tolist()
print(f'Total passages to extract per backbone: {len(eval_texts)} ({N_test} test + {len(df_tr_eval)} train)')

# Joint Spectral Bank storage
Z_tr_norm_list = []
Z_te_norm_list = []

# Refinement cluster groups
groups_hybrid = groups_065.copy()
first_14b_extracted = False

# References for exact solver and Claude specialist
solver_14b = None
U_test_14b = None
Z_test_qwen3_8b = None
qwen3_8b_ro = None

# ==============================================================================
# TRUNK 1: Qwen3-14B-Base (Hosts qwen3_14b_p2 and qwen3_14b_p1)
# ==============================================================================
if BASE_QWEN3_14B is not None and ('qwen3_14b_p2' in CKPTS or 'qwen3_14b_p1' in CKPTS):
    print(f'\\n===============================================================')
    print(f'=== Loading Base Trunk: Qwen3-14B (keep=27) ===')
    print(f'===============================================================')
    trunk_14b = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN3_14B,
        keep_layers=27,
        layers=[2, 4, 6, 9, 13, 17, 22, 27],
        max_len=447
    )

    # 1A. lora_qwen3_14b (Part 2)
    if 'qwen3_14b_p2' in CKPTS:
        ckpt_p2 = CKPTS['qwen3_14b_p2']
        print(f'\\n--- [Member: lora_qwen3_14b (Part 2)] In-place merging adapter ---')
        merge_lora(trunk_14b.model, ckpt_p2, sign=+1.0)
        Z_all = trunk_14b.extract(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        Z_tr_norm_list.append(Z_train_bb / np.linalg.norm(Z_train_bb, axis=1, keepdims=True))
        Z_te_norm_list.append(Z_test_bb / np.linalg.norm(Z_test_bb, axis=1, keepdims=True))

        if not first_14b_extracted and len(test_texts) > 1:
            Z_te_n = Z_test_bb / np.linalg.norm(Z_test_bb, axis=1, keepdims=True)
            D_cos = np.clip(1.0 - (Z_te_n @ Z_te_n.T), 0.0, None)
            D_hybrid = 0.60 * D_tfidf + 0.40 * D_cos
            np.fill_diagonal(D_hybrid, 0.0)
            Z_link_hyb = linkage(squareform(D_hybrid, checks=False), method="average")
            groups_hybrid = fcluster(Z_link_hyb, t=0.62, criterion="distance")
            first_14b_extracted = True
            print(f'[Hybrid Clustering] Refined groups_hybrid with 14B cosine dist: {len(np.unique(groups_hybrid))} clusters.')

        heads_sd = load_file(str(ckpt_p2 / 'heads.safetensors'))
        meta = json.load(open(ckpt_p2 / 'meta.json'))
        plain_w = align_to_master(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
        plain_b = align_to_master(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
        center_w = align_to_master(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
        center_b = align_to_master(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT VERIFY] lora_qwen3_14b: Train Top-1 BAcc = {bacc_tr:.4f}')
        if N_test >= 50:
            assert bacc_tr >= 0.40, f"Validation gate failed for lora_qwen3_14b: {bacc_tr:.4f} < 0.40"

        log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
        LOGIT += 0.40 * log_p_plain

        if twin_rows_085.any():
            Zc = Z_test_bb.copy()
            for g in np.unique(groups_085[twin_rows_085]):
                m_g = (groups_085 == g)
                Zc[m_g] -= Zc[m_g].mean(axis=0, keepdims=True)
            log_p_center = log_softmax(Zc[twin_rows_085] @ center_w.T + center_b)
            LOGIT[twin_rows_085] += 0.12 * log_p_center

        gauss_prm = fit_shared_gaussian(Z_train_bb, y_tr_eval, prompts_tr_eval, cfg={"d": 256, "T1": 2.35, "Tg": 2.30})
        U_test = ((Z_test_bb - gauss_prm["mx"]) / gauss_prm["sx"]) @ gauss_prm["W"]
        solver_14b = ExactAuthorPosterior(gauss_prm["m"], gauss_prm["lam"])
        U_test_14b = U_test.copy()

        log_p_gauss = np.zeros_like(log_p_plain)
        for g in np.unique(groups_hybrid):
            idx_g = np.where(groups_hybrid == g)[0]
            log_p_gauss[idx_g] = solver_14b.group(U_test[idx_g], temperature=gauss_prm["Tg"])
        LOGIT += 0.12 * log_p_gauss
        print('[Backbone: lora_qwen3_14b] Accumulated: Plain(0.40) + Twin(0.12) + ExactDP(0.12).')

        merge_lora(trunk_14b.model, ckpt_p2, sign=-1.0)

    # 1B. lora14b_v1 (Part 1)
    if 'qwen3_14b_p1' in CKPTS:
        ckpt_p1 = CKPTS['qwen3_14b_p1']
        print(f'\\n--- [Member: lora14b_v1 (Part 1)] In-place merging adapter ---')
        merge_lora(trunk_14b.model, ckpt_p1, sign=+1.0)
        Z_all = trunk_14b.extract(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        Z_tr_norm_list.append(Z_train_bb / np.linalg.norm(Z_train_bb, axis=1, keepdims=True))
        Z_te_norm_list.append(Z_test_bb / np.linalg.norm(Z_test_bb, axis=1, keepdims=True))

        heads_sd = load_file(str(ckpt_p1 / 'heads.safetensors'))
        meta = json.load(open(ckpt_p1 / 'meta.json'))
        plain_w = align_to_master(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
        plain_b = align_to_master(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
        center_w = align_to_master(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
        center_b = align_to_master(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT VERIFY] lora14b_v1: Train Top-1 BAcc = {bacc_tr:.4f}')
        if N_test >= 50:
            assert bacc_tr >= 0.40, f"Validation gate failed for lora14b_v1: {bacc_tr:.4f} < 0.40"

        log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
        LOGIT += 0.20 * log_p_plain
        if twin_rows_085.any():
            Zc = Z_test_bb.copy()
            for g in np.unique(groups_085[twin_rows_085]):
                m_g = (groups_085 == g)
                Zc[m_g] -= Zc[m_g].mean(axis=0, keepdims=True)
            log_p_center = log_softmax(Zc[twin_rows_085] @ center_w.T + center_b)
            LOGIT[twin_rows_085] += 0.07 * log_p_center

        gauss_prm = fit_shared_gaussian(Z_train_bb, y_tr_eval, prompts_tr_eval, cfg={"d": 256, "T1": 2.35, "Tg": 2.30})
        U_test = ((Z_test_bb - gauss_prm["mx"]) / gauss_prm["sx"]) @ gauss_prm["W"]
        solver_p1 = ExactAuthorPosterior(gauss_prm["m"], gauss_prm["lam"])
        log_p_gauss = np.zeros_like(log_p_plain)
        for g in np.unique(groups_hybrid):
            idx_g = np.where(groups_hybrid == g)[0]
            log_p_gauss[idx_g] = solver_p1.group(U_test[idx_g], temperature=gauss_prm["Tg"])
        LOGIT += 0.07 * log_p_gauss
        print('[Backbone: lora14b_v1] Accumulated: Plain(0.20) + Twin(0.07) + ExactDP(0.07).')

    trunk_14b.free_memory()
    del trunk_14b
    gc.collect()
    torch.cuda.empty_cache()

# ==============================================================================
# TRUNK 2: Qwen2.5-14B (Hosts lora_qwen25_14b)
# ==============================================================================
if BASE_QWEN25_14B is not None and 'qwen25_14b_p2' in CKPTS:
    ckpt_q25 = CKPTS['qwen25_14b_p2']
    print(f'\\n===============================================================')
    print(f'=== Loading Base Trunk: Qwen2.5-14B (keep=32) ===')
    print(f'===============================================================')
    trunk_q25 = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN25_14B,
        keep_layers=32,
        layers=[3, 5, 8, 11, 16, 21, 27, 32],
        max_len=447
    )
    merge_lora(trunk_q25.model, ckpt_q25, sign=+1.0)
    Z_all = trunk_q25.extract(eval_texts, batch_size=8, end_id=151643)
    Z_test_bb = Z_all[:N_test]
    Z_train_bb = Z_all[N_test:]

    Z_tr_norm_list.append(Z_train_bb / np.linalg.norm(Z_train_bb, axis=1, keepdims=True))
    Z_te_norm_list.append(Z_test_bb / np.linalg.norm(Z_test_bb, axis=1, keepdims=True))

    heads_sd = load_file(str(ckpt_q25 / 'heads.safetensors'))
    meta = json.load(open(ckpt_q25 / 'meta.json'))
    plain_w = align_to_master(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
    plain_b = align_to_master(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
    center_w = align_to_master(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
    center_b = align_to_master(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

    pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
    bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
    print(f'[CHECKPOINT VERIFY] lora_qwen25_14b: Train Top-1 BAcc = {bacc_tr:.4f}')
    if N_test >= 50:
        assert bacc_tr >= 0.40, f"Validation gate failed for lora_qwen25_14b: {bacc_tr:.4f} < 0.40"

    log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
    LOGIT += 0.22 * log_p_plain
    if twin_rows_085.any():
        Zc = Z_test_bb.copy()
        for g in np.unique(groups_085[twin_rows_085]):
            m_g = (groups_085 == g)
            Zc[m_g] -= Zc[m_g].mean(axis=0, keepdims=True)
        log_p_center = log_softmax(Zc[twin_rows_085] @ center_w.T + center_b)
        LOGIT[twin_rows_085] += 0.08 * log_p_center

    gauss_prm = fit_shared_gaussian(Z_train_bb, y_tr_eval, prompts_tr_eval, cfg={"d": 256, "T1": 2.35, "Tg": 2.30})
    U_test = ((Z_test_bb - gauss_prm["mx"]) / gauss_prm["sx"]) @ gauss_prm["W"]
    solver_q25 = ExactAuthorPosterior(gauss_prm["m"], gauss_prm["lam"])
    log_p_gauss = np.zeros_like(log_p_plain)
    for g in np.unique(groups_hybrid):
        idx_g = np.where(groups_hybrid == g)[0]
        log_p_gauss[idx_g] = solver_q25.group(U_test[idx_g], temperature=gauss_prm["Tg"])
    LOGIT += 0.08 * log_p_gauss
    print('[Backbone: lora_qwen25_14b] Accumulated: Plain(0.22) + Twin(0.08) + ExactDP(0.08).')

    trunk_q25.free_memory()
    del trunk_q25
    gc.collect()
    torch.cuda.empty_cache()

# ==============================================================================
# TRUNK 3: Qwen3-8B-Base (Hosts qwen3_8b_claude and qwen3_8b_p1)
# ==============================================================================
if BASE_QWEN3_8B is not None and ('qwen3_8b_claude' in CKPTS or 'qwen3_8b_p1' in CKPTS):
    print(f'\\n===============================================================')
    print(f'=== Loading Base Trunk: Qwen3-8B (keep=24) ===')
    print(f'===============================================================')
    trunk_8b = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN3_8B,
        keep_layers=24,
        layers=[2, 4, 6, 8, 12, 16, 20, 24],
        max_len=384
    )

    # 3A. ntl-qwen3-8b-lora-claude-head
    if 'qwen3_8b_claude' in CKPTS:
        ckpt_cl = CKPTS['qwen3_8b_claude']
        print(f'\\n--- [Member: ntl-qwen3-8b-lora-claude-head] In-place merging adapter ---')
        merge_lora(trunk_8b.model, ckpt_cl, sign=+1.0)
        Z_all = trunk_8b.extract(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        Z_test_qwen3_8b = Z_test_bb.copy()

        Z_tr_norm_list.append(Z_train_bb / np.linalg.norm(Z_train_bb, axis=1, keepdims=True))
        Z_te_norm_list.append(Z_test_bb / np.linalg.norm(Z_test_bb, axis=1, keepdims=True))

        if (ckpt_cl / 'readouts.npz').exists():
            ro = np.load(ckpt_cl / 'readouts.npz', allow_pickle=True)
            qwen3_8b_ro = ro
            plain_w = align_to_master(ro['plain_coef'], ro['classes'])
            plain_b = align_to_master(ro['plain_intercept'], ro['classes'])
            center_w = align_to_master(ro['center_coef'], ro['classes'])
            center_b = align_to_master(ro['center_intercept'], ro['classes'])
        else:
            heads_sd = load_file(str(ckpt_cl / 'heads.safetensors'))
            meta = json.load(open(ckpt_cl / 'meta.json'))
            plain_w = align_to_master(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
            plain_b = align_to_master(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
            center_w = align_to_master(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
            center_b = align_to_master(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT VERIFY] ntl-qwen3-8b-lora-claude-head: Train Top-1 BAcc = {bacc_tr:.4f}')
        if N_test >= 50:
            assert bacc_tr >= 0.40, f"Validation gate failed for qwen3_8b_claude: {bacc_tr:.4f} < 0.40"

        log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
        LOGIT += 0.18 * log_p_plain
        if twin_rows_085.any():
            Zc = Z_test_bb.copy()
            for g in np.unique(groups_085[twin_rows_085]):
                m_g = (groups_085 == g)
                Zc[m_g] -= Zc[m_g].mean(axis=0, keepdims=True)
            log_p_center = log_softmax(Zc[twin_rows_085] @ center_w.T + center_b)
            LOGIT[twin_rows_085] += 0.06 * log_p_center

        gauss_prm = fit_shared_gaussian(Z_train_bb, y_tr_eval, prompts_tr_eval, cfg={"d": 256, "T1": 2.35, "Tg": 2.30})
        U_test = ((Z_test_bb - gauss_prm["mx"]) / gauss_prm["sx"]) @ gauss_prm["W"]
        solver_cl = ExactAuthorPosterior(gauss_prm["m"], gauss_prm["lam"])
        log_p_gauss = np.zeros_like(log_p_plain)
        for g in np.unique(groups_hybrid):
            idx_g = np.where(groups_hybrid == g)[0]
            log_p_gauss[idx_g] = solver_cl.group(U_test[idx_g], temperature=gauss_prm["Tg"])
        LOGIT += 0.06 * log_p_gauss
        print('[Backbone: ntl-qwen3-8b-lora-claude-head] Accumulated: Plain(0.18) + Twin(0.06) + ExactDP(0.06).')

        merge_lora(trunk_8b.model, ckpt_cl, sign=-1.0)

    # 3B. lora8b_v1 (Part 1)
    if 'qwen3_8b_p1' in CKPTS:
        ckpt_8bp1 = CKPTS['qwen3_8b_p1']
        print(f'\\n--- [Member: lora8b_v1 (Part 1)] In-place merging adapter ---')
        merge_lora(trunk_8b.model, ckpt_8bp1, sign=+1.0)
        Z_all = trunk_8b.extract(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        Z_tr_norm_list.append(Z_train_bb / np.linalg.norm(Z_train_bb, axis=1, keepdims=True))
        Z_te_norm_list.append(Z_test_bb / np.linalg.norm(Z_test_bb, axis=1, keepdims=True))

        heads_sd = load_file(str(ckpt_8bp1 / 'heads.safetensors'))
        meta = json.load(open(ckpt_8bp1 / 'meta.json'))
        plain_w = align_to_master(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
        plain_b = align_to_master(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
        center_w = align_to_master(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
        center_b = align_to_master(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT VERIFY] lora8b_v1: Train Top-1 BAcc = {bacc_tr:.4f}')
        if N_test >= 50:
            assert bacc_tr >= 0.40, f"Validation gate failed for lora8b_v1: {bacc_tr:.4f} < 0.40"

        log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
        LOGIT += 0.12 * log_p_plain
        if twin_rows_085.any():
            Zc = Z_test_bb.copy()
            for g in np.unique(groups_085[twin_rows_085]):
                m_g = (groups_085 == g)
                Zc[m_g] -= Zc[m_g].mean(axis=0, keepdims=True)
            log_p_center = log_softmax(Zc[twin_rows_085] @ center_w.T + center_b)
            LOGIT[twin_rows_085] += 0.04 * log_p_center

        gauss_prm = fit_shared_gaussian(Z_train_bb, y_tr_eval, prompts_tr_eval, cfg={"d": 256, "T1": 2.35, "Tg": 2.30})
        U_test = ((Z_test_bb - gauss_prm["mx"]) / gauss_prm["sx"]) @ gauss_prm["W"]
        solver_8bp1 = ExactAuthorPosterior(gauss_prm["m"], gauss_prm["lam"])
        log_p_gauss = np.zeros_like(log_p_plain)
        for g in np.unique(groups_hybrid):
            idx_g = np.where(groups_hybrid == g)[0]
            log_p_gauss[idx_g] = solver_8bp1.group(U_test[idx_g], temperature=gauss_prm["Tg"])
        LOGIT += 0.04 * log_p_gauss
        print('[Backbone: lora8b_v1] Accumulated: Plain(0.12) + Twin(0.04) + ExactDP(0.04).')

    trunk_8b.free_memory()
    del trunk_8b
    gc.collect()
    torch.cuda.empty_cache()

# ==============================================================================
# TRUNK 4: Dedicated DeBERTa-v3 Fine-Tuned Encoder Loader
# ==============================================================================
if 'deberta_p2' in CKPTS:
    ckpt_deb = CKPTS['deberta_p2']
    print(f'\\n===============================================================')
    print(f'=== Loading Dedicated DeBERTa-v3 Fine-Tuned Encoder ===')
    print(f'===============================================================')
    enc_dir = ckpt_deb / 'encoder' if (ckpt_deb / 'encoder').exists() else ckpt_deb
    try:
        deb_tok = AutoTokenizer.from_pretrained(str(enc_dir))
    except Exception:
        deb_tok = AutoTokenizer.from_pretrained(str(BASE_DEBERTA)) if BASE_DEBERTA is not None else None
        
    if deb_tok is not None and deb_tok.pad_token is None:
        deb_tok.pad_token = deb_tok.eos_token or '[PAD]'

    dev_deb = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    deb_model = AutoModel.from_pretrained(str(enc_dir), torch_dtype=torch.float16).to(dev_deb)
    deb_model.eval()

    drop_first = False
    head_file = ckpt_deb / 'head.safetensors'
    if head_file.exists() and deb_tok is not None:
        hsd = load_file(str(head_file))
        meta_deb = json.load(open(ckpt_deb / 'meta.json'))
        w_d = align_to_master(hsd['weight'].cpu().numpy(), meta_deb['labels'])
        b_d = align_to_master(hsd['bias'].cpu().numpy(), meta_deb['labels'])

        sample_txt = df_tr_eval['text'].fillna('').tolist()[:36]
        sample_y = y_tr_eval[:36]
        with torch.no_grad():
            enc_s = deb_tok(sample_txt, padding=True, truncation=True, max_length=384, return_tensors='pt').to(dev_deb)
            out_s = deb_model(**enc_s)
            h_s = out_s.last_hidden_state.float()

            m_a = enc_s['attention_mask'].unsqueeze(-1).float()
            pool_a = ((h_s * m_a).sum(1) / m_a.sum(1).clamp(min=1.0)).cpu().numpy()
            acc_a = (np.argmax(pool_a @ w_d.T + b_d, axis=1) == sample_y).mean()

            m_b = m_a.clone()
            m_b[:, 0, :] = 0.0
            pool_b = ((h_s * m_b).sum(1) / m_b.sum(1).clamp(min=1.0)).cpu().numpy()
            acc_b = (np.argmax(pool_b @ w_d.T + b_d, axis=1) == sample_y).mean()

            print(f'[DeBERTa Mask Check] Full mask Acc={acc_a:.4f} | Drop-first Acc={acc_b:.4f}')
            drop_first = (acc_b > acc_a)

    print(f'[DeBERTa Extractor] Extracting {len(eval_texts)} passages with drop_first={drop_first}...')
    all_deb_e = []
    with torch.no_grad():
        for i in range(0, len(eval_texts), 16):
            enc = deb_tok(eval_texts[i:i+16], padding=True, truncation=True, max_length=384, return_tensors='pt').to(dev_deb)
            out = deb_model(**enc)
            m = enc['attention_mask'].unsqueeze(-1).float()
            if drop_first:
                m[:, 0, :] = 0.0
            pool = (out.last_hidden_state.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            all_deb_e.append(pool.detach().cpu().numpy())

    Z_deb = np.vstack(all_deb_e)
    Z_test_bb = Z_deb[:N_test]
    Z_train_bb = Z_deb[N_test:]

    Z_tr_norm_list.append(Z_train_bb / np.linalg.norm(Z_train_bb, axis=1, keepdims=True))
    Z_te_norm_list.append(Z_test_bb / np.linalg.norm(Z_test_bb, axis=1, keepdims=True))

    del deb_model, deb_tok
    gc.collect()
    torch.cuda.empty_cache()

    pred_tr = np.argmax(Z_train_bb @ w_d.T + b_d, axis=1)
    bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
    print(f'[CHECKPOINT VERIFY] deberta_v3_large: Train Top-1 BAcc = {bacc_tr:.4f}')
    if N_test >= 50:
        assert bacc_tr >= 0.40, f"Validation gate failed for deberta_v3_large: {bacc_tr:.4f} < 0.40"

    log_p_plain = log_softmax(Z_test_bb @ w_d.T + b_d)
    LOGIT += 0.20 * log_p_plain

    if twin_rows_085.any():
        pu, pinv = np.unique(prompts_tr_eval, return_inverse=True)
        Pm = np.zeros((len(pu), Z_train_bb.shape[1]))
        np.add.at(Pm, pinv, Z_train_bb)
        Pm /= np.maximum(np.bincount(pinv)[:, None], 1.0)
        Z_deb_tr_c = Z_train_bb - Pm[pinv]
        lr_deb_c = LinearSVC(C=0.1, class_weight='balanced', random_state=RANDOM_SEED, max_iter=2000)
        lr_deb_c.fit(Z_deb_tr_c, y_tr_eval)

        Z_deb_te_c = Z_test_bb.copy()
        for g in np.unique(groups_085[twin_rows_085]):
            m_g = (groups_085 == g)
            Z_deb_te_c[m_g] -= Z_deb_te_c[m_g].mean(axis=0, keepdims=True)
        log_p_center = log_softmax(lr_deb_c.decision_function(Z_deb_te_c[twin_rows_085]))
        LOGIT[twin_rows_085] += 0.07 * log_p_center

    gauss_prm = fit_shared_gaussian(Z_train_bb, y_tr_eval, prompts_tr_eval, cfg={"d": 256, "T1": 2.35, "Tg": 2.30})
    U_test = ((Z_test_bb - gauss_prm["mx"]) / gauss_prm["sx"]) @ gauss_prm["W"]
    solver_deb = ExactAuthorPosterior(gauss_prm["m"], gauss_prm["lam"])
    log_p_gauss = np.zeros_like(log_p_plain)
    for g in np.unique(groups_hybrid):
        idx_g = np.where(groups_hybrid == g)[0]
        log_p_gauss[idx_g] = solver_deb.group(U_test[idx_g], temperature=gauss_prm["Tg"])
    LOGIT += 0.07 * log_p_gauss
    print('[Backbone: deberta_v3_large] Accumulated: Plain(0.20) + Twin(0.07) + ExactDP(0.07).')
""")

    # Cell 7: Module 5 — Joint Cross-Backbone Spectral Style Readout Bank
    add_md("### Step 7: Module 5 — Joint Cross-Backbone Spectral Style Readout Bank")
    add_code("""print('\\n=== [Module 5] Fitting Joint Cross-Backbone Spectral Readout Bank ===')
if len(Z_tr_norm_list) >= 2:
    X_joint_train = np.hstack(Z_tr_norm_list)
    X_joint_test = np.hstack(Z_te_norm_list)
    print(f'[Joint Spectral] Concatenated {len(Z_tr_norm_list)} backbones: Dimension D = {X_joint_train.shape[1]}.')

    pu, pinv = np.unique(prompts_tr_eval, return_inverse=True)
    G = len(pu)
    Pm = np.zeros((G, X_joint_train.shape[1]))
    np.add.at(Pm, pinv, X_joint_train)
    Pm /= np.maximum(np.bincount(pinv)[:, None], 1.0)
    R_joint = X_joint_train - Pm[pinv]

    d_joint = min(512, max(len(X_joint_train) - G - 1, 10), X_joint_train.shape[1])
    if R_joint.shape[0] < R_joint.shape[1]:
        w, U = np.linalg.eigh(R_joint @ R_joint.T)
        o = np.argsort(w)[::-1][:d_joint]
        w, U = np.clip(w[o], 1e-12, None), U[:, o]
        V_joint = (R_joint.T @ U) / np.sqrt(w)
    else:
        _, _, Vt = np.linalg.svd(R_joint, full_matrices=False)
        V_joint = Vt[:d_joint].T

    X_j_tr_proj = X_joint_train @ V_joint
    X_j_te_proj = X_joint_test @ V_joint

    # 1. Plain Joint Spectral Readout
    lr_joint_plain = LogisticRegression(C=0.15, max_iter=2000, random_state=RANDOM_SEED)
    lr_joint_plain.fit(X_j_tr_proj, y_tr_eval)
    log_p_joint_plain = log_softmax(lr_joint_plain.decision_function(X_j_te_proj))
    LOGIT += 0.20 * log_p_joint_plain
    print('[Joint Spectral] Plain joint spectral readout added with weight 0.20.')

    # 2. Centered Joint Spectral Readout
    if twin_rows_085.any():
        X_j_tr_c = X_joint_train - Pm[pinv]
        lr_joint_center = LogisticRegression(C=0.10, max_iter=2000, random_state=RANDOM_SEED)
        lr_joint_center.fit(X_j_tr_c @ V_joint, y_tr_eval)

        X_j_te_c = X_joint_test.copy()
        for g in np.unique(groups_085[twin_rows_085]):
            m_g = (groups_085 == g)
            X_j_te_c[m_g] -= X_j_te_c[m_g].mean(axis=0, keepdims=True)

        log_p_joint_center = log_softmax(lr_joint_center.decision_function(X_j_te_c[twin_rows_085] @ V_joint))
        LOGIT[twin_rows_085] += 0.10 * log_p_joint_center
        print('[Joint Spectral] Centered joint spectral readout added with weight 0.10 on twin rows.')
""")

    # Cell 8: Module 6 — Global Exact Uniqueness, 50-Step Sinkhorn, Shield & Output
    add_md("### Step 8: Module 6 — Global Exact Uniqueness, 50-Step Sinkhorn-Knopp, Sovereign Shield & Output")
    add_code("""# Step 1: Global Exact Uniqueness Refinement across hybrid prompt clusters
print('\\n=== [Module 6] Global Exact Uniqueness Refinement ===')
sz_hyb = np.bincount(groups_hybrid)[groups_hybrid]
refine_rows = (sz_hyb >= 2) & (sz_hyb <= 18)

if refine_rows.any() and solver_14b is not None:
    print(f'[Module 6] Refining {refine_rows.sum()} cluster rows using ExactAuthorPosterior forward-backward DP...')
    for g in np.unique(groups_hybrid[refine_rows]):
        idx_g = np.where(groups_hybrid == g)[0]
        U_grp = U_test_14b[idx_g] if U_test_14b is not None else np.zeros((len(idx_g), solver_14b.d))
        scores_g = solver_14b.group(
            U_grp, temperature=1.0, extra_log_scores=LOGIT[idx_g],
            model_weight=0.15, extra_weight=1.0
        )
        LOGIT[idx_g] = scores_g

# Step 2: 50-Iteration Log-Domain Sinkhorn-Knopp Marginal Balancing
print('\\n=== [Module 6] 50-Step Log-Domain Sinkhorn-Knopp ===')
log_P = LOGIT - logsumexp(LOGIT, axis=1, keepdims=True)

if N_test >= 50:
    print(f'[Sinkhorn] Running 50 log-domain balancing iterations on {N_test} rows...')
    for _ in range(50):
        # Dampened column marginal equalization toward uniform 1/18 prior
        col_lse = logsumexp(log_P, axis=0, keepdims=True) - np.log(len(log_P))
        log_P = log_P - 0.50 * col_lse
        # Exact row normalization
        log_P = log_P - logsumexp(log_P, axis=1, keepdims=True)
else:
    print('[Sinkhorn] Interactive commit mode (N_test < 50): bypassed Sinkhorn.')

# Step 3: Calibrated Post-Sinkhorn Orthographic Log-Penalty (-2.0)
print('\\n=== [Module 6] Applying Calibrated Sovereign Orthographic Bayesian Shield ===')
claude_classes = ["claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5"]
glm_classes = ["glm-5.3", "glm-5.3-flash"]
gemini_classes = ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
sol_class = "gpt-5.6-sol"

claude_idx = [LABELS.index(c) for c in claude_classes if c in LABELS]
glm_idx = [LABELS.index(c) for c in glm_classes if c in LABELS]
gemini_idx = [LABELS.index(c) for c in gemini_classes if c in LABELS]
sol_idx = LABELS.index(sol_class) if sol_class in LABELS else None

curly_chars = {"\u2018", "\u2019", "\u201c", "\u201d"}
n_curly, n_em, n_apos = 0, 0, 0

for i, t in enumerate(test_texts):
    if any(c in t for c in curly_chars):
        log_P[i, claude_idx] -= 2.0
        log_P[i, glm_idx] -= 2.0
        n_curly += 1

    if "\u2014" in t:
        log_P[i, gemini_idx] -= 2.0
        n_em += 1

    if "'" in t and sol_idx is not None:
        log_P[i, sol_idx] -= 2.0
        n_apos += 1

print(f'[Shield] Applied {n_curly} curly penalties (-2.0), {n_em} em-dash penalties (-2.0), {n_apos} straight apostrophe penalties (-2.0).')

pred_idx = np.argmax(log_P, axis=1)
initial_pred_idx = pred_idx.copy()

# Step 4: Top-1 Gated Claude Specialist
is_claude = np.isin(pred_idx, claude_idx)
if is_claude.any():
    claude_rows = np.where(is_claude)[0]
    print(f'\\n=== [Module 6] Top-1 Gated Claude Specialist on {len(claude_rows)} rows ===')

    ro_claude_classes = claude_classes
    df_claude = df_train[df_train['model'].isin(ro_claude_classes)].copy()
    y_c_tr = np.array([ro_claude_classes.index(m) for m in df_claude['model']])
    X_c_tr = sparse.hstack([vec_char.transform(df_claude['text']), vec_word.transform(df_claude['text'])], format='csr')
    svc_claude = LinearSVC(C=0.5, class_weight='balanced', random_state=RANDOM_SEED)
    svc_claude.fit(X_c_tr, y_c_tr)

    X_c_te = sparse.hstack([
        vec_char.transform([test_texts[i] for i in claude_rows]),
        vec_word.transform([test_texts[i] for i in claude_rows])
    ], format='csr')
    log_p_c_svc = log_softmax(svc_claude.decision_function(X_c_te))

    if qwen3_8b_ro is not None and Z_test_qwen3_8b is not None:
        logits_c_lora = Z_test_qwen3_8b[claude_rows] @ qwen3_8b_ro['claude_coef'].T + qwen3_8b_ro['claude_intercept']
        ro_src = list(qwen3_8b_ro['claude_classes'])
        logits_c_lora = align_to_master(logits_c_lora, ro_src, ro_claude_classes)
        log_p_c_lora = log_softmax(logits_c_lora)
        log_p_claude_comb = 0.65 * log_p_c_lora + 0.35 * log_p_c_svc
    else:
        log_p_claude_comb = log_p_c_svc

    best_c = np.argmax(log_p_claude_comb, axis=1)
    for r_idx, c_best in zip(claude_rows, best_c):
        best_name = ro_claude_classes[c_best]
        pred_idx[r_idx] = LABELS.index(best_name)

    assert np.array_equal(pred_idx[~is_claude], initial_pred_idx[~is_claude]), "Safety Check Failed: Non-Claude predictions modified!"
    print('[Module 6] Claude specialist completed. Invariant verified: Non-Claude predictions 100% untouched.')

# Step 5: Format & Output Submission
sub = pd.DataFrame({
    'id': df_test['id'],
    'model': [ID2LAB[i] for i in pred_idx]
})

out_dir = Path('/kaggle/working')
out_dir.mkdir(parents=True, exist_ok=True)
sub_path = out_dir / 'submission.parquet'
sub.to_parquet(sub_path, index=False)

print(f'\\n[SUCCESS] Wrote valid submission to {sub_path} ({sub_path.stat().st_size} bytes)!')
print(f'Submission shape: {sub.shape}')
assert len(sub) == len(df_test), f"Row count mismatch: {len(sub)} != {len(df_test)}"
assert sub['model'].isnull().sum() == 0, "Submission contains null predictions!"

print('\\nFirst 5 Predictions:')
print(sub.head())

print('\\nPredicted Model Distribution:')
print(sub['model'].value_counts())
""")

    out_path = Path("kaggle_submission.ipynb")
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Generated {out_path} with {len(nb['cells'])} cells successfully!")

if __name__ == "__main__":
    create_v20_pipeline_notebook()
