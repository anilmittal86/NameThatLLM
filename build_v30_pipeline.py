import json
from pathlib import Path

def create_v30_pipeline_notebook():
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
    add_md("""# Name That LLM: Version 30 (0.65 Target) Pipeline
### Roll Back v28/v29 Regressions to Proven v26 Foundation (0.605) + Crash-Proof Oversized Cluster Splitting in `ExactAuthorPosterior` + Pre-Sinkhorn Orthographic Shield + Auto-Integrate High-Alpha Assets
- **Roll Back Regressions**: Strips transductive pseudo-labeling (`TTCA`), removes logit division by 1.85, and eliminates double DP passes that caused v29 macro-recall degradation.
- **Restore Proven 88/12 Probability Mixture**: Reinstates $P_{\\text{mix}} = 0.88 \\cdot R_{\\text{core}} + 0.12 \\cdot H_{\\text{helper}}$ from v26 (`0.605` public score foundation).
- **Permanent Oversized Cluster Splitting**: Fixes the v28 exception by dynamically splitting multi-passage clusters with $n > 14$ into balanced sub-clusters via cosine average linkage, guaranteeing cluster size $\\le 14 \\le 18$ so `ExactAuthorPosterior` never encounters index or recursion errors.
- **Pre-Sinkhorn Sovereign Orthographic Shield & Claude Specialist**: Applies orthographic penalties ($-3.0$ curly quotes, $-2.5$ em-dash, $-3.0$ straight apostrophe) and top-1 gated Claude head *before* Sinkhorn, allowing Sinkhorn's column normalization to redistribute mass cleanly across all 18 classes without starvation.
- **Single Final 50-Step Sinkhorn Balancing**: Runs log-domain Sinkhorn-Knopp marginal balancing once at the end (`BALANCE_ALPHA = 0.50`) on the full test set.
- **Fail-Loud Discovery & Checkpoint Gate**: Discovers base models strictly by `(model_type, hidden_size, num_hidden_layers)` and enforces a mandatory `BAcc >= 0.48` gate on full `train.csv`.
- **Auto-Discovery of High-Alpha Assets**: Seamlessly integrates `instyle` (Qwen3-8B LoRA), `synthetic4404.joblib` (paired spectral readout bank), and `ultrabert` if attached in the sidebar.
- **Subprocess Memory Isolation**: Maintains Peak Process RSS $\\le 2.05$ GB with OS memory reclamation after the 78k lexical engine.
""")

    # Cell 1: Environment, Dependencies & Master 18 Labels
    add_md("### Step 1: Environment, Dependencies & Master 18 Labels")
    add_code("""import os
import gc
import sys
import time
import json
import re
import psutil
import shutil
import subprocess
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
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

def log_ram(tag: str):
    rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    print(f'[RAM AUDIT | {tag}] Process RSS: {rss_gb:.2f} GB', flush=True)

log_ram('Notebook Startup')

# Canonical 18 Target Labels (exact Part 2 canonical order)
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

def align_cols(arr: np.ndarray, src_labels: list) -> np.ndarray:
    idx = [list(src_labels).index(l) for l in LABELS]
    return arr[..., idx]

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)
""")

    # Cell 2: Dataset Discovery & Base Paths
    add_md("### Step 2: Dataset Discovery, Strict Base Matcher & Checkpoint Inventory")
    add_code("""INPUT_DIR = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('.')

def find_file(name: str) -> Optional[Path]:
    for p in INPUT_DIR.rglob(name):
        return p
    return None

train_file = find_file('train.csv')
test_file = find_file('test.csv')

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

train_texts = df_train['text'].fillna('').astype(str).tolist()
test_texts = df_test['text'].fillna('').astype(str).tolist()

y_train = np.array([LAB2I[m] for m in df_train['model']])
prompts_train = df_train['prompt_id'].values

N_train = len(df_train)
N_test = len(df_test)

print(f'Train path: {train_file} ({N_train} rows)')
print(f'Test path:  {test_file} ({N_test} rows)')

def find_base_models() -> Dict[str, Optional[Path]]:
    bases = {"qwen3_14b": None, "qwen25_14b": None, "qwen3_8b": None, "deberta_v3_large": None}
    configs = list(INPUT_DIR.rglob("config.json"))
    found_qwen2_sizes = []
    for cfg_p in configs:
        try:
            with open(cfg_p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            m_type = cfg.get("model_type", "")
            h_size = cfg.get("hidden_size", 0)
            n_layers = cfg.get("num_hidden_layers", 0)
            base_dir = cfg_p.parent
            if m_type == "qwen3" and h_size == 5120 and n_layers == 40:
                bases["qwen3_14b"] = base_dir
            elif m_type == "qwen2" and h_size == 5120 and n_layers == 48:
                bases["qwen25_14b"] = base_dir
            elif m_type == "qwen2":
                found_qwen2_sizes.append((h_size, n_layers, base_dir))
            elif m_type == "qwen3" and h_size == 4096 and n_layers == 36:
                bases["qwen3_8b"] = base_dir
            elif m_type == "deberta-v2" and h_size == 1024 and n_layers == 24:
                bases["deberta_v3_large"] = base_dir
        except Exception:
            continue
    if bases["qwen25_14b"] is None:
        print("[MISSING BASE] Qwen2.5-14B base not found; attach Qwen2.5 14b variation in sidebar!", flush=True)
        if found_qwen2_sizes:
            print(f"  Found other Qwen2 variants: {found_qwen2_sizes}", flush=True)
    return bases

BASES = find_base_models()
BASE_QWEN3_14B = BASES["qwen3_14b"]
BASE_QWEN25_14B = BASES["qwen25_14b"]
BASE_QWEN3_8B = BASES["qwen3_8b"]
BASE_DEBERTA = BASES["deberta_v3_large"]

print(f'Discovered Base Qwen3-14B:   {BASE_QWEN3_14B}')
print(f'Discovered Base Qwen2.5-14B: {BASE_QWEN25_14B}')
print(f'Discovered Base Qwen3-8B:    {BASE_QWEN3_8B}')
print(f'Discovered Base DeBERTa:     {BASE_DEBERTA}')

def find_checkpoints() -> Dict[str, Path]:
    ckpts = {}
    for p in INPUT_DIR.rglob('heads.safetensors'):
        p_str = str(p).lower().replace('\\\\', '/')
        if 'lora_qwen3_14b' in p_str:
            ckpts['qwen3_14b_p2'] = p.parent
        elif 'lora14b_v1' in p_str:
            ckpts['qwen3_14b_p1'] = p.parent
        elif 'lora_qwen25_14b' in p_str:
            ckpts['qwen25_14b_p2'] = p.parent
        elif 'lora8b_v1' in p_str:
            ckpts['qwen3_8b_p1'] = p.parent
        elif 'instyle' in p_str:
            ckpts['instyle'] = p.parent
    for p in INPUT_DIR.rglob('readouts.npz'):
        if 'qwen3-8b' in str(p).lower() or 'claude-head' in str(p).lower():
            ckpts['qwen3_8b_claude'] = p.parent
            break
    for p in INPUT_DIR.rglob('head.safetensors'):
        if 'deberta' in str(p).lower():
            ckpts['deberta_p2'] = p.parent
            break
    if 'instyle' not in ckpts:
        for p in INPUT_DIR.rglob('*'):
            if 'instyle' in str(p).lower() and ((p / 'adapter_model.safetensors').exists() or (p / 'adapter' / 'adapter_model.safetensors').exists()):
                ckpts['instyle'] = p
                break
    for p in INPUT_DIR.rglob('synthetic4404.joblib'):
        ckpts['synthetic4404'] = p.parent
        break
    for p in INPUT_DIR.rglob('*'):
        if 'ultrabert' in str(p).lower() and (p / 'fold0' / 'config.json').exists():
            ckpts['ultrabert'] = p
            break
    return ckpts

CKPTS = find_checkpoints()
print(f'Discovered Checkpoints ({len(CKPTS)}): {list(CKPTS.keys())}')

pool_paths = list(INPUT_DIR.rglob('pool_v4.parquet'))
pool_file = pool_paths[0] if pool_paths else None
print(f'Discovered Passage Pool:     {pool_file}')
""")

    # Cell 3: Module 1 — Subprocess-Isolated 78,482-Row Dual Lexical Engine
    add_md("### Step 3: Module 1 — Subprocess-Isolated 78,482-Row Dual Lexical Engine")
    add_code("""# Write self-contained run_lexical.py for total memory isolation (Peak RSS <= 2.05 GB)
lexical_script_path = Path('/kaggle/working/run_lexical.py') if Path('/kaggle/working').exists() else Path('run_lexical.py')
working_dir = lexical_script_path.parent

lexical_code = f\"\"\"import os
import sys
import gc
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

LABELS = {LABELS}
K = len(LABELS)
LAB2I = {{l: i for i, l in enumerate(LABELS)}}

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

t0 = time.time()
print('=== [Subprocess Lexical Engine] Starting 78k Full Execution ===', flush=True)

df_tr = pd.read_csv(r'{train_file}')
df_te = pd.read_csv(r'{test_file}')

test_texts = df_te['text'].fillna('').astype(str).tolist()
fit_texts = df_tr['text'].fillna('').astype(str).tolist()
fit_y = np.array([LAB2I[m] for m in df_tr['model']])
len_tr_raw = len(df_tr)

# Compute t=0.85 Topic Clusters on Test
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

np.save(r'{working_dir}/groups_085.npy', groups_085)
np.save(r'{working_dir}/twin_rows_085.npy', twin_rows_085)
np.save(r'{working_dir}/D_tfidf.npy', D)
del Xt_tfidf, D
gc.collect()

# Load 100% of pool_v4.parquet
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

# 4 Canonical Views from NTL two-stage LoRA part two
VIEWS = [
    ("char", 3, 5, 250000, 1.0),
    ("word", 1, 3, 180000, 0.7),
    ("shape", 2, 6, 60000, 0.5),
    ("function", 1, 3, 50000, 0.5)
]

A_blocks, B_blocks = [], []
for view, lo, hi, max_feat, weight in VIEWS:
    print(f'[Subprocess Lexical] Extracting view: {{view}} (weight={{weight}})...', flush=True)
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
    A_blocks.append(X_tr_v)
    B_blocks.append(X_te_v)
    del vec, X_tr_v, X_te_v
    gc.collect()

del fit_texts
gc.collect()

X_tr_csr = sparse.hstack(A_blocks, format="csr", dtype=np.float32)
del A_blocks
gc.collect()

X_te_csr = sparse.hstack(B_blocks, format="csr", dtype=np.float32)
del B_blocks
gc.collect()

print(f'[Subprocess Lexical] Built full sparse CSR matrices: X_tr={{X_tr_csr.shape}}, X_te={{X_te_csr.shape}} (nnz={{X_tr_csr.nnz}})', flush=True)

# 1. LinearSVC on full 78,482 rows
print('[Subprocess Lexical] Fitting LinearSVC on full 78k corpus...', flush=True)
svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
svc.fit(X_tr_csr, fit_y)
log_p_svc = log_softmax(svc.decision_function(X_te_csr))
del svc
gc.collect()

log_p_lex_twin = np.zeros_like(log_p_svc)

# 2. Prompt-Centered Sparse Style Twin on train.csv for twin_rows_085
if twin_rows_085.any():
    print('[Subprocess Lexical] Fitting Prompt-Centered Style-Gram Twin on train.csv...', flush=True)
    X_train_only = X_tr_csr[:len_tr_raw]
    pu, pinv = np.unique(df_tr['prompt_id'].values, return_inverse=True)
    n_p = len(pu)
    counts = np.bincount(pinv)
    vals = (1.0 / counts[pinv]).astype(np.float32)
    G = sparse.csr_matrix((vals, (np.arange(len_tr_raw), pinv)), shape=(len_tr_raw, n_p), dtype=np.float32)
    X_train_c = X_train_only - G @ (G.T @ X_train_only)

    svc_twin = LinearSVC(C=0.3, random_state=42, max_iter=3000)
    svc_twin.fit(X_train_c, fit_y[:len_tr_raw])
    del X_train_only, G, X_train_c
    gc.collect()

    X_te_twin = X_te_csr[twin_rows_085]
    u_t, inv_t = np.unique(groups_085[twin_rows_085], return_inverse=True)
    counts_t = np.bincount(inv_t)
    vals_t = (1.0 / counts_t[inv_t]).astype(np.float32)
    G_t = sparse.csr_matrix((vals_t, (np.arange(len(inv_t)), inv_t)), shape=(len(inv_t), len(u_t)), dtype=np.float32)
    X_te_twin_c = X_te_twin - G_t @ (G_t.T @ X_te_twin)
    log_p_lex_twin[twin_rows_085] = log_softmax(svc_twin.decision_function(X_te_twin_c))
    del svc_twin, X_te_twin, G_t, X_te_twin_c
    gc.collect()

del X_tr_csr, X_te_csr
gc.collect()

np.save(r'{working_dir}/log_p_svc.npy', log_p_svc)
np.save(r'{working_dir}/log_p_lex_twin.npy', log_p_lex_twin)
# Alias for compatibility
np.save(r'{working_dir}/logit_lex_test.npy', log_p_svc)
np.save(r'{working_dir}/logit_lex_twin_test.npy', log_p_lex_twin)

print(f'[Subprocess Lexical] Completed in {{time.time() - t0:.2f}}s. Saved arrays to {{working_dir}}.', flush=True)
sys.exit(0)
\"\"\"

with open(lexical_script_path, "w", encoding="utf-8") as f:
    f.write(lexical_code)

try:
    print(f'[Main Process] Launching isolated lexical subprocess: {lexical_script_path}...')
    t0_sub = time.time()
    subprocess.run([sys.executable, str(lexical_script_path)], check=True)
    print(f'[Main Process] Lexical subprocess completed in {time.time() - t0_sub:.2f}s! Memory reclaimed by OS.')
except Exception as e:
    print(f'[Main Process] Subprocess launch bypassed ({e}). Executing directly in-process with gc.collect() fallback...')
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_lexical", str(lexical_script_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass

# Load outputs in main process
log_p_svc = np.load(working_dir / 'log_p_svc.npy')
log_p_lex_twin = np.load(working_dir / 'log_p_lex_twin.npy')
groups_085 = np.load(working_dir / 'groups_085.npy')
twin_rows_085 = np.load(working_dir / 'twin_rows_085.npy')
D_tfidf = np.load(working_dir / 'D_tfidf.npy')

# Compute groups_065 on test for exact author DP
if len(test_texts) >= 2:
    groups_065 = fcluster(linkage(squareform(D_tfidf, checks=False), method='average'), 0.65, criterion='distance')
else:
    groups_065 = np.ones(len(test_texts), dtype=int)

# Initialize strict Core logit accumulator and Helper probability list
LOGIT_CORE = 1.02 * log_p_svc
if twin_rows_085.any():
    LOGIT_CORE[twin_rows_085] += 0.10 * log_p_lex_twin[twin_rows_085]

HELPER_LIST: List[Tuple[float, np.ndarray]] = []

print(f'[Main Process] Initialized LOGIT_CORE with 78k Lexical Engine: shape={LOGIT_CORE.shape}')
log_ram('After Subprocess Memory Reclaim & Core Init')
""")

    # Cell 4: In-Place LoRA Merger, Low-RAM wPCA, Crash-Proof Exact DP & Centered Twin
    add_md("### Step 4: Reversible LoRA Merger, Low-RAM wPCA, Crash-Proof Cluster Splitting & Exact DP")
    add_code("""def merge_lora(model, ckpt_dir: Path, sign: float = +1.0) -> bool:
    target_file = None
    for fname in ["adapter_model.safetensors", "model.safetensors"]:
        if (ckpt_dir / fname).exists():
            target_file = ckpt_dir / fname
            break
        elif (ckpt_dir / "adapter" / fname).exists():
            target_file = ckpt_dir / "adapter" / fname
            break
    if target_file is None:
        return False
    sd = load_file(str(target_file))
    meta_path = ckpt_dir / "meta.json"
    alpha = 32.0
    rank = 16.0
    if meta_path.exists():
        try:
            m = json.load(open(meta_path))
            alpha = float(m.get("lora_alpha", 32.0))
            rank = float(m.get("lora_r", 16.0))
        except Exception:
            pass
    scale = (alpha / rank) * sign
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

def _two_way(Z: np.ndarray, y: np.ndarray, pinv: np.ndarray, G: int, K: int = 18, iters: int = 4):
    N, d = Z.shape
    ng = np.bincount(pinv, minlength=G).astype(np.float64)
    P = np.zeros((G, d), dtype=np.float64)
    mu = np.zeros((K, d), dtype=np.float64)
    for _ in range(iters):
        R = Z - P[pinv]
        for c in range(K):
            m_c = (y == c)
            if m_c.any():
                mu[c] = R[m_c].mean(axis=0)
        R2 = Z - mu[y]
        P.fill(0.0)
        np.add.at(P, pinv, R2)
        P /= np.maximum(ng[:, None], 1.0)
    return mu, P, ng

def _shrunk(S: np.ndarray, alpha: float) -> np.ndarray:
    d = S.shape[0]
    tr = max(np.trace(S), 1e-8)
    return (1.0 - alpha) * S + alpha * (tr / d) * np.eye(d, dtype=S.dtype)

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

def compute_centered_logits(Z_test: np.ndarray, groups: np.ndarray, W_c: np.ndarray, b_c: np.ndarray, src_labels: list) -> Tuple[np.ndarray, np.ndarray]:
    \"\"\"Returns full-shape (N_test, 18) log-probabilities and boolean mask of valid cluster rows.\"\"\"
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

def split_oversized_clusters(U: np.ndarray, groups: np.ndarray, max_cluster_size: int = 14) -> np.ndarray:
    \"\"\"
    Guarantees every cluster has size 1 <= n <= max_cluster_size (<= 18) so ExactAuthorPosterior
    NEVER raises an IndexError on n > 18 and never drops large clusters to singletons.
    \"\"\"
    new_groups = np.zeros(len(groups), dtype=np.int64)
    next_id = 1
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        if len(rows) <= max_cluster_size:
            new_groups[rows] = next_id
            next_id += 1
        else:
            n_sub = int(np.ceil(len(rows) / max_cluster_size))
            U_sub = U[rows]
            norms = np.linalg.norm(U_sub, axis=1, keepdims=True).clip(min=1e-8)
            d_sub = pdist(U_sub / norms, metric="cosine")
            d_sub = np.nan_to_num(d_sub, nan=0.0, posinf=1.0, neginf=0.0)
            sub_labels = fcluster(linkage(d_sub, method="average"), t=n_sub, criterion="maxclust")
            for sg in np.unique(sub_labels):
                s_rows = rows[sub_labels == sg]
                for start in range(0, len(s_rows), max_cluster_size):
                    new_groups[s_rows[start : start + max_cluster_size]] = next_id
                    next_id += 1
    return new_groups

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

    def group(self, U: np.ndarray, temperature: float = 2.30,
              extra_log_scores: Optional[np.ndarray] = None,
              model_weight: float = 1.0, extra_weight: float = 1.0) -> np.ndarray:
        s = len(U)
        if s <= 1 or s > self.K:
            out = np.empty((s, self.K), dtype=np.float64)
            for i in range(s):
                lp = self.single(U[i], temperature=temperature)
                if extra_log_scores is not None:
                    lp = model_weight * lp + extra_weight * extra_log_scores[i]
                out[i] = lp - logsumexp(lp)
            return out

        u_bar = U.mean(axis=0).astype(np.float32)
        from itertools import combinations
        subsets = list(combinations(range(self.K), s))
        n_subsets = len(subsets)

        means = self._get_subset_means(s)
        diff = u_bar[None, :] - means
        diag = (1.0 + float(s) * self.lam) / float(s)
        dist = np.sum(diff * diff / diag[None, :], axis=1)
        log_p_S = -0.5 * dist / max(float(temperature), 1e-4)
        log_p_S -= logsumexp(log_p_S)

        P_S = np.exp(log_p_S)
        post_i = np.zeros((s, self.K), dtype=np.float64)

        if extra_log_scores is not None:
            extra_exp = np.exp(extra_log_scores - np.max(extra_log_scores, axis=1, keepdims=True))
            for i in range(s):
                weights = np.empty(n_subsets, dtype=np.float64)
                for sub_idx, S in enumerate(subsets):
                    weights[sub_idx] = P_S[sub_idx] * extra_exp[i, list(S)].sum()
                for sub_idx, S in enumerate(subsets):
                    w = weights[sub_idx]
                    if w > 0:
                        post_i[i, list(S)] += w
        else:
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

def scores_exact_dp(U_test: np.ndarray, groups: np.ndarray, prm: dict) -> np.ndarray:
    \"\"\"
    Guarantees cluster sizes <= 14 <= 18 and computes exact author subset posteriors.
    \"\"\"
    N = len(U_test)
    if N == 0:
        return np.zeros((0, K), dtype=np.float64)
    if N == 1:
        solver = ExactAuthorPosterior(prm["m"], prm["lam"])
        T1 = prm.get("T1", 2.35)
        return solver.single(U_test[0], temperature=T1)[None, :]

    safe_groups = split_oversized_clusters(U_test, groups, max_cluster_size=14)
    max_c = np.bincount(safe_groups).max() if len(safe_groups) > 0 else 0
    assert max_c <= 18, f"Max cluster size {max_c} exceeds 18!"

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

    # Cell 5: Exact Training Extractor
    add_md("### Step 5: Exact Training Extractor (Strictly ids[:max_len] + [end_id])")
    add_code("""class UniversalLoRATrunkLoader:
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

        # Build GPU norms
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

    def _extract_batch_ids(self, id_list: List[List[int]], batch_size: int = 8) -> np.ndarray:
        order = np.argsort([len(x) for x in id_list], kind="stable")
        total_dim = len(self.layers) * self.model.config.hidden_size
        Z = np.zeros((len(id_list), total_dim), dtype=np.float32)

        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            enc = self.tok.pad({"input_ids": [id_list[i] for i in idx]}, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.first_dev)
            attn_mask = enc["attention_mask"].to(self.first_dev)

            out = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            hs = out.hidden_states

            # Attention-sink zeroing (m[:, 0] = 0)
            m = attn_mask.unsqueeze(-1).float().to(self.device_norms)
            m[:, 0, :] = 0.0
            den = m.sum(dim=1).clamp(min=1.0)

            pooled = []
            for l_idx, nm in zip(self.layers, self.norms):
                h_l = hs[l_idx].to(self.device_norms).float()
                if not torch.isfinite(h_l).all():
                    h_l = torch.nan_to_num(h_l, nan=0.0, posinf=100.0, neginf=-100.0)
                mean_v = (h_l * m).sum(dim=1) / den
                pooled.append(nm(mean_v))

            Z[idx] = torch.cat(pooled, dim=-1).cpu().numpy()
        return Z

    @torch.no_grad()
    def extract_exact(self, texts: List[str], batch_size: int = 8, end_id: int = 151643) -> np.ndarray:
        t0 = time.time()
        raw_ids = self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        ids = [x + [end_id] for x in raw_ids]
        Z = self._extract_batch_ids(ids, batch_size=batch_size)
        print(f"[Extractor] Extracted {len(texts)} samples in {time.time() - t0:.2f}s ({len(texts) / max(0.1, time.time() - t0):.1f} samples/s).")
        return Z

    def free_memory(self):
        print("[Extractor] Freeing GPU VRAM...")
        del self.model, self.norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
""")

    # Cell 6: Checkpoint Execution & Routing
    add_md("### Step 6: Checkpoint Execution, Strict Gate (BAcc >= 0.48) & Routing")
    add_code("""if N_test < 50:
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

spectral_tr_blocks = []
spectral_te_blocks = []
gate_results = []
Z_test_qwen3_8b = None

# ==============================================================================
# TRUNK 1: Qwen3-14B-Base (Hosts lora_qwen3_14b and lora14b_v1)
# ==============================================================================
if BASE_QWEN3_14B is not None and ('qwen3_14b_p2' in CKPTS or 'qwen3_14b_p1' in CKPTS):
    print(f'\\n===============================================================')
    print(f'=== Loading Base Trunk 1: Qwen3-14B (keep=27) ===')
    print(f'===============================================================')
    trunk_14b = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN3_14B, keep_layers=27,
        layers=[2, 5, 8, 12, 16, 20, 24, 27], max_len=447
    )

    # 1A. Member: lora_qwen3_14b (Part 2, 27L)
    if 'qwen3_14b_p2' in CKPTS:
        ckpt_14bp2 = CKPTS['qwen3_14b_p2']
        print(f'\\n--- [Member: lora_qwen3_14b (Part 2)] In-place merging adapter ---')
        merge_lora(trunk_14b.model, ckpt_14bp2, sign=+1.0)
        Z_all = trunk_14b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        heads_sd = load_file(str(ckpt_14bp2 / 'heads.safetensors'))
        meta = json.load(open(ckpt_14bp2 / 'meta.json'))
        plain_w = align_cols(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
        plain_b = align_cols(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
        center_w = align_cols(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
        center_b = align_cols(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT GATE] lora_qwen3_14b: train.csv BAcc = {bacc_tr:.4f}')
        gate_results.append({"name": "lora_qwen3_14b", "bacc": bacc_tr, "passed": (bacc_tr >= 0.48)})

        if bacc_tr >= 0.48:
            log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
            LOGIT_CORE += 0.42 * log_p_plain
            out_lp_twin, valid_twin = compute_centered_logits(Z_test_bb, groups_085, center_w, center_b, LABELS)
            if valid_twin.any():
                LOGIT_CORE[valid_twin] += 0.12 * out_lp_twin[valid_twin]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_tr_eval, prompts_tr_eval, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test_bb, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            lp_dp = scores_exact_dp(U_test, groups_065, prm)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((0.35, P_dp))
            print('[Routing: lora_qwen3_14b] Added Core Plain(0.42) + Twin(0.12), Helper DP(0.35).')
            del U_train, U_test, prm, lp_dp, P_dp

        del Z_all, Z_train_bb, Z_test_bb, heads_sd, plain_w, plain_b, center_w, center_b
        gc.collect()
        merge_lora(trunk_14b.model, ckpt_14bp2, sign=-1.0)

    # 1B. Member: lora14b_v1 (Part 1, 24L)
    if 'qwen3_14b_p1' in CKPTS:
        ckpt_14bp1 = CKPTS['qwen3_14b_p1']
        print(f'\\n--- [Member: lora14b_v1 (Part 1)] In-place merging adapter ---')
        merge_lora(trunk_14b.model, ckpt_14bp1, sign=+1.0)
        Z_all = trunk_14b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        heads_sd = load_file(str(ckpt_14bp1 / 'heads.safetensors'))
        meta = json.load(open(ckpt_14bp1 / 'meta.json'))
        plain_w = align_cols(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
        plain_b = align_cols(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
        center_w = align_cols(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
        center_b = align_cols(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT GATE] lora14b_v1: train.csv BAcc = {bacc_tr:.4f}')
        gate_results.append({"name": "lora14b_v1", "bacc": bacc_tr, "passed": (bacc_tr >= 0.48)})

        if bacc_tr >= 0.48:
            log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
            LOGIT_CORE += 0.08 * log_p_plain
            out_lp_twin, valid_twin = compute_centered_logits(Z_test_bb, groups_085, center_w, center_b, LABELS)
            if valid_twin.any():
                LOGIT_CORE[valid_twin] += 0.03 * out_lp_twin[valid_twin]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_tr_eval, prompts_tr_eval, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test_bb, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            lp_dp = scores_exact_dp(U_test, groups_065, prm)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((0.06, P_dp))
            print('[Routing: lora14b_v1] Added Core Plain(0.08) + Twin(0.03), Helper DP(0.06).')
            del U_train, U_test, prm, lp_dp, P_dp

        del Z_all, Z_train_bb, Z_test_bb, heads_sd, plain_w, plain_b, center_w, center_b
        gc.collect()
        merge_lora(trunk_14b.model, ckpt_14bp1, sign=-1.0)

    trunk_14b.free_memory()
    del trunk_14b
    gc.collect()
    torch.cuda.empty_cache()
    log_ram('After Trunk 1 (Qwen3-14B) Full Cleanup')

# ==============================================================================
# TRUNK 2: Qwen2.5-14B-Base (Hosts lora_qwen25_14b Part 2)
# ==============================================================================
if BASE_QWEN25_14B is not None and 'qwen25_14b_p2' in CKPTS:
    print(f'\\n===============================================================')
    print(f'=== Loading Base Trunk 2: Qwen2.5-14B (keep=32) ===')
    print(f'===============================================================')
    trunk_q25 = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN25_14B, keep_layers=32,
        layers=[2, 6, 10, 15, 20, 24, 28, 32], max_len=447
    )

    ckpt_q25 = CKPTS['qwen25_14b_p2']
    print(f'\\n--- [Member: lora_qwen25_14b (Part 2)] In-place merging adapter ---')
    merge_lora(trunk_q25.model, ckpt_q25, sign=+1.0)
    Z_all = trunk_q25.extract_exact(eval_texts, batch_size=8, end_id=151643)
    Z_test_bb = Z_all[:N_test]
    Z_train_bb = Z_all[N_test:]

    heads_sd = load_file(str(ckpt_q25 / 'heads.safetensors'))
    meta = json.load(open(ckpt_q25 / 'meta.json'))
    plain_w = align_cols(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
    plain_b = align_cols(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
    center_w = align_cols(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
    center_b = align_cols(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

    pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
    bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
    print(f'[CHECKPOINT GATE] lora_qwen25_14b: train.csv BAcc = {bacc_tr:.4f}')
    gate_results.append({"name": "lora_qwen25_14b", "bacc": bacc_tr, "passed": (bacc_tr >= 0.48)})

    if bacc_tr >= 0.48:
        log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
        LOGIT_CORE += 0.22 * log_p_plain
        out_lp_twin, valid_twin = compute_centered_logits(Z_test_bb, groups_085, center_w, center_b, LABELS)
        if valid_twin.any():
            LOGIT_CORE[valid_twin] += 0.09 * out_lp_twin[valid_twin]

        prm = fit_shared_gaussian_lowram(Z_train_bb, y_tr_eval, prompts_tr_eval, d_target=256)
        U_train = transform_lowram(Z_train_bb, prm)
        U_test = transform_lowram(Z_test_bb, prm)

        idx64 = prm["top64_idx"]
        U_tr_64 = U_train[:, idx64].astype(np.float32)
        U_te_64 = U_test[:, idx64].astype(np.float32)
        U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
        U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
        spectral_tr_blocks.append(U_tr_64)
        spectral_te_blocks.append(U_te_64)

        lp_dp = scores_exact_dp(U_test, groups_065, prm)
        P_dp = softmax(lp_dp, axis=1)
        HELPER_LIST.append((0.22, P_dp))
        print('[Routing: lora_qwen25_14b] Added Core Plain(0.22) + Twin(0.09), Helper DP(0.22).')
        del U_train, U_test, prm, lp_dp, P_dp

    del Z_all, Z_train_bb, Z_test_bb, heads_sd, plain_w, plain_b, center_w, center_b
    gc.collect()
    merge_lora(trunk_q25.model, ckpt_q25, sign=-1.0)
    trunk_q25.free_memory()
    del trunk_q25
    gc.collect()
    torch.cuda.empty_cache()
    log_ram('After Trunk 2 (Qwen2.5-14B) Full Cleanup')

# ==============================================================================
# TRUNK 3: Qwen3-8B-Base (Hosts Claude Head, lora8b_v1, and InStyle LoRA)
# ==============================================================================
if BASE_QWEN3_8B is not None and ('qwen3_8b_claude' in CKPTS or 'qwen3_8b_p1' in CKPTS or 'instyle' in CKPTS):
    print(f'\\n===============================================================')
    print(f'=== Loading Base Trunk 3: Qwen3-8B (keep=24) ===')
    print(f'===============================================================')
    trunk_8b = UniversalLoRATrunkLoader(
        base_dir=BASE_QWEN3_8B, keep_layers=24,
        layers=[2, 5, 8, 11, 14, 17, 20, 24], max_len=447
    )

    # 3A. Member: ntl-qwen3-8b-lora-claude-head
    if 'qwen3_8b_claude' in CKPTS:
        ckpt_8bcl = CKPTS['qwen3_8b_claude']
        print(f'\\n--- [Member: ntl-qwen3-8b-lora-claude-head] In-place merging adapter ---')
        merge_lora(trunk_8b.model, ckpt_8bcl, sign=+1.0)
        Z_all = trunk_8b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        # Retain for Claude specialist gate & synthetic4404 hook
        Z_test_qwen3_8b = Z_test_bb.copy()

        ro = np.load(ckpt_8bcl / 'readouts.npz')
        meta = json.load(open(ckpt_8bcl / 'meta.json'))
        plain_w = align_cols(ro['W_plain'].T, meta['labels'])
        plain_b = align_cols(ro['b_plain'], meta['labels'])
        center_w = align_cols(ro['W_center'].T, meta['labels'])
        center_b = align_cols(ro['b_center'], meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT GATE] ntl-qwen3-8b-lora-claude-head: train.csv BAcc = {bacc_tr:.4f}')
        gate_results.append({"name": "ntl-qwen3-8b-lora-claude-head", "bacc": bacc_tr, "passed": (bacc_tr >= 0.48)})

        if bacc_tr >= 0.48:
            log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
            LOGIT_CORE += 0.16 * log_p_plain
            out_lp_twin, valid_twin = compute_centered_logits(Z_test_bb, groups_085, center_w, center_b, LABELS)
            if valid_twin.any():
                LOGIT_CORE[valid_twin] += 0.06 * out_lp_twin[valid_twin]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_tr_eval, prompts_tr_eval, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test_bb, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            lp_dp = scores_exact_dp(U_test, groups_065, prm)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((0.13, P_dp))
            print('[Routing: ntl-qwen3-8b-lora-claude-head] Added Core Plain(0.16) + Twin(0.06), Helper DP(0.13).')
            del U_train, U_test, prm, lp_dp, P_dp

        del Z_all, Z_train_bb, Z_test_bb, ro, plain_w, plain_b, center_w, center_b
        gc.collect()
        merge_lora(trunk_8b.model, ckpt_8bcl, sign=-1.0)

    # 3B. Member: lora8b_v1 (Part 1, 24L)
    if 'qwen3_8b_p1' in CKPTS:
        ckpt_8bp1 = CKPTS['qwen3_8b_p1']
        print(f'\\n--- [Member: lora8b_v1 (Part 1)] In-place merging adapter ---')
        merge_lora(trunk_8b.model, ckpt_8bp1, sign=+1.0)
        Z_all = trunk_8b.extract_exact(eval_texts, batch_size=8, end_id=151643)
        Z_test_bb = Z_all[:N_test]
        Z_train_bb = Z_all[N_test:]

        heads_sd = load_file(str(ckpt_8bp1 / 'heads.safetensors'))
        meta = json.load(open(ckpt_8bp1 / 'meta.json'))
        plain_w = align_cols(heads_sd['plain.weight'].cpu().numpy(), meta['labels'])
        plain_b = align_cols(heads_sd['plain.bias'].cpu().numpy(), meta['labels'])
        center_w = align_cols(heads_sd['centered.weight'].cpu().numpy(), meta['labels'])
        center_b = align_cols(heads_sd['centered.bias'].cpu().numpy(), meta['labels'])

        pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
        bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
        print(f'[CHECKPOINT GATE] lora8b_v1: train.csv BAcc = {bacc_tr:.4f}')
        gate_results.append({"name": "lora8b_v1", "bacc": bacc_tr, "passed": (bacc_tr >= 0.48)})

        if bacc_tr >= 0.48:
            log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
            LOGIT_CORE += 0.05 * log_p_plain
            out_lp_twin, valid_twin = compute_centered_logits(Z_test_bb, groups_085, center_w, center_b, LABELS)
            if valid_twin.any():
                LOGIT_CORE[valid_twin] += 0.02 * out_lp_twin[valid_twin]

            prm = fit_shared_gaussian_lowram(Z_train_bb, y_tr_eval, prompts_tr_eval, d_target=256)
            U_train = transform_lowram(Z_train_bb, prm)
            U_test = transform_lowram(Z_test_bb, prm)

            idx64 = prm["top64_idx"]
            U_tr_64 = U_train[:, idx64].astype(np.float32)
            U_te_64 = U_test[:, idx64].astype(np.float32)
            U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
            U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
            spectral_tr_blocks.append(U_tr_64)
            spectral_te_blocks.append(U_te_64)

            lp_dp = scores_exact_dp(U_test, groups_065, prm)
            P_dp = softmax(lp_dp, axis=1)
            HELPER_LIST.append((0.04, P_dp))
            print('[Routing: lora8b_v1] Added Core Plain(0.05) + Twin(0.02), Helper DP(0.04).')
            del U_train, U_test, prm, lp_dp, P_dp

        del Z_all, Z_train_bb, Z_test_bb, heads_sd, plain_w, plain_b, center_w, center_b
        gc.collect()
        merge_lora(trunk_8b.model, ckpt_8bp1, sign=-1.0)

    # 3C. Optional Auto-Hook: zbigniewkirchner/ntl-qwen3-8b-lora-instyle
    if 'instyle' in CKPTS:
        ckpt_inst = CKPTS['instyle']
        print(f'\\n--- [Optional Member: In-Style Qwen3-8B LoRA] In-place merging from {ckpt_inst} ---')
        merged_inst = merge_lora(trunk_8b.model, ckpt_inst, sign=+1.0)
        if merged_inst:
            try:
                Z_all = trunk_8b.extract_exact(eval_texts, batch_size=8, end_id=151643)
                Z_test_bb = Z_all[:N_test]
                Z_train_bb = Z_all[N_test:]

                heads_file = ckpt_inst / 'heads.safetensors' if (ckpt_inst / 'heads.safetensors').exists() else ckpt_inst / 'head.safetensors'
                if heads_file.exists():
                    h_sd = load_file(str(heads_file))
                    meta_inst = json.load(open(ckpt_inst / 'meta.json')) if (ckpt_inst / 'meta.json').exists() else {'labels': LABELS}
                    pw_key = 'plain.weight' if 'plain.weight' in h_sd else 'weight'
                    pb_key = 'plain.bias' if 'plain.bias' in h_sd else 'bias'
                    plain_w = align_cols(h_sd[pw_key].cpu().numpy(), meta_inst['labels'])
                    plain_b = align_cols(h_sd[pb_key].cpu().numpy(), meta_inst['labels'])
                    center_w = align_cols(h_sd['centered.weight'].cpu().numpy(), meta_inst['labels']) if 'centered.weight' in h_sd else None
                    center_b = align_cols(h_sd['centered.bias'].cpu().numpy(), meta_inst['labels']) if 'centered.bias' in h_sd else None

                    pred_tr = np.argmax(Z_train_bb @ plain_w.T + plain_b, axis=1)
                    bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
                    print(f'[CHECKPOINT GATE] instyle_qwen3_8b: train.csv BAcc = {bacc_tr:.4f}')
                    gate_results.append({"name": "instyle_qwen3_8b", "bacc": bacc_tr, "passed": (bacc_tr >= 0.48)})

                    if bacc_tr >= 0.48:
                        log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
                        LOGIT_CORE += 0.20 * log_p_plain
                        if center_w is not None:
                            out_lp_twin, valid_twin = compute_centered_logits(Z_test_bb, groups_085, center_w, center_b, LABELS)
                            if valid_twin.any():
                                LOGIT_CORE[valid_twin] += 0.07 * out_lp_twin[valid_twin]
                        print('[Routing: instyle_qwen3_8b] Added Core Plain(0.20) + Twin(0.07).')
                del Z_all, Z_train_bb, Z_test_bb
            except Exception as e:
                print(f'[instyle_qwen3_8b] Skipped due to error: {e}')
            finally:
                merge_lora(trunk_8b.model, ckpt_inst, sign=-1.0)
                gc.collect()
    else:
        print('[Optional In-Style LoRA] zbigniewkirchner/ntl-qwen3-8b-lora-instyle not mounted; skipping.')

    trunk_8b.free_memory()
    del trunk_8b
    gc.collect()
    torch.cuda.empty_cache()
    log_ram('After Trunk 3 (Qwen3-8B) Full Cleanup')

# ==============================================================================
# TRUNK 4: Dedicated DeBERTa-v3 Fine-Tuned Encoder
# ==============================================================================
if 'deberta_p2' in CKPTS:
    ckpt_deb = CKPTS['deberta_p2']
    print(f'\\n===============================================================')
    print(f'=== Loading Dedicated DeBERTa-v3 Fine-Tuned Encoder ===')
    print(f'===============================================================')
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

    # Immediate disk cleanup
    shutil.rmtree(stage_dir, ignore_errors=True)

    head_path = ckpt_deb / 'head.safetensors' if (ckpt_deb / 'head.safetensors').exists() else ckpt_deb / 'heads.safetensors'
    heads_deb_sd = load_file(str(head_path))
    meta_deb = json.load(open(ckpt_deb / 'meta.json')) if (ckpt_deb / 'meta.json').exists() else {'labels': LABELS}
    if 'plain.weight' in heads_deb_sd:
        w_d = align_cols(heads_deb_sd['plain.weight'].cpu().numpy(), meta_deb['labels'])
        b_d = align_cols(heads_deb_sd['plain.bias'].cpu().numpy(), meta_deb['labels'])
    else:
        w_d = align_cols(heads_deb_sd['weight'].cpu().numpy(), meta_deb['labels'])
        b_d = align_cols(heads_deb_sd['bias'].cpu().numpy(), meta_deb['labels'])

    if 'centered.weight' in heads_deb_sd:
        w_cd = align_cols(heads_deb_sd['centered.weight'].cpu().numpy(), meta_deb['labels'])
        b_cd = align_cols(heads_deb_sd['centered.bias'].cpu().numpy(), meta_deb['labels'])
    else:
        w_cd, b_cd = None, None

    # Probe on 180 train passages for attention mask and tokenization mode
    probe_texts = df_tr_eval['text'].fillna('').tolist()[:180]
    probe_y = y_tr_eval[:180]

    best_mode = 1
    best_bacc = -1.0
    best_drop_first = False
    best_add_special = True

    for mode_idx, (add_spec, drop_first) in enumerate([
        (True, False),   # Mode 1: add_special=True, m[:, 0]=1
        (True, True),    # Mode 2: add_special=True, m[:, 0]=0
        (False, True),   # Mode 3: add_special=False, m[:, 0]=0
        (False, False),  # Mode 4: add_special=False, m[:, 0]=1
    ], 1):
        with torch.no_grad():
            if add_spec:
                enc_p = deb_tok(probe_texts, padding=True, truncation=True, max_length=384, return_tensors='pt').to(dev_deb)
            else:
                p_ids = deb_tok(probe_texts, add_special_tokens=False, truncation=True, max_length=383)["input_ids"]
                p_end = deb_tok.eos_token_id or deb_tok.pad_token_id or 0
                p_ids = [x + [p_end] for x in p_ids]
                enc_p = deb_tok.pad({"input_ids": p_ids}, padding=True, return_tensors="pt").to(dev_deb)

            out_p = deb_model(**enc_p)
            m_p = enc_p['attention_mask'].unsqueeze(-1).float()
            if drop_first:
                m_p[:, 0, :] = 0.0
            pool_p = ((out_p.last_hidden_state.float() * m_p).sum(dim=1) / m_p.sum(dim=1).clamp(min=1.0)).cpu().numpy()
            pred_p = np.argmax(pool_p @ w_d.T + b_d, axis=1)
            bacc_p = balanced_accuracy_score(probe_y, pred_p)
            print(f'[DEBERTA PROBE] Mode {mode_idx} (add_special={add_spec}, drop_first={drop_first}) -> Probe BAcc = {bacc_p:.4f}')
            if bacc_p > best_bacc:
                best_bacc = bacc_p
                best_mode = mode_idx
                best_add_special = add_spec
                best_drop_first = drop_first

    print(f'[DEBERTA PROBE] Winning Mode = Mode {best_mode} (add_special={best_add_special}, drop_first={best_drop_first}) | Probe BAcc = {best_bacc:.4f}')

    print(f'[DeBERTa Extractor] Extracting {len(eval_texts)} passages with Mode {best_mode}...')
    all_deb_e = []
    with torch.no_grad():
        for i in range(0, len(eval_texts), 16):
            batch_texts = eval_texts[i : i + 16]
            if best_add_special:
                enc = deb_tok(batch_texts, padding=True, truncation=True, max_length=384, return_tensors='pt').to(dev_deb)
            else:
                b_ids = deb_tok(batch_texts, add_special_tokens=False, truncation=True, max_length=383)["input_ids"]
                b_end = deb_tok.eos_token_id or deb_tok.pad_token_id or 0
                b_ids = [x + [b_end] for x in b_ids]
                enc = deb_tok.pad({"input_ids": b_ids}, padding=True, return_tensors="pt").to(dev_deb)

            out = deb_model(**enc)
            m = enc['attention_mask'].unsqueeze(-1).float()
            if best_drop_first:
                m[:, 0, :] = 0.0
            pool = (out.last_hidden_state.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            all_deb_e.append(pool.detach().cpu().numpy())

    Z_deb = np.vstack(all_deb_e)
    Z_test_bb = Z_deb[:N_test]
    Z_train_bb = Z_deb[N_test:]

    del deb_model, deb_tok, all_deb_e, Z_deb
    gc.collect()
    torch.cuda.empty_cache()

    pred_tr = np.argmax(Z_train_bb @ w_d.T + b_d, axis=1)
    bacc_tr = balanced_accuracy_score(y_tr_eval, pred_tr)
    print(f'[CHECKPOINT GATE] deberta_v3_large: train.csv BAcc = {bacc_tr:.4f}')
    gate_results.append({"name": "deberta_v3_large", "bacc": bacc_tr, "passed": (bacc_tr >= 0.48)})

    if bacc_tr >= 0.48:
        log_p_plain = log_softmax(Z_test_bb @ w_d.T + b_d)
        LOGIT_CORE += 0.22 * log_p_plain
        if w_cd is not None:
            out_lp_twin, valid_twin = compute_centered_logits(Z_test_bb, groups_085, w_cd, b_cd, LABELS)
            if valid_twin.any():
                LOGIT_CORE[valid_twin] += 0.09 * out_lp_twin[valid_twin]

        prm = fit_shared_gaussian_lowram(Z_train_bb, y_tr_eval, prompts_tr_eval, d_target=256)
        U_train = transform_lowram(Z_train_bb, prm)
        U_test = transform_lowram(Z_test_bb, prm)

        idx64 = prm["top64_idx"]
        U_tr_64 = U_train[:, idx64].astype(np.float32)
        U_te_64 = U_test[:, idx64].astype(np.float32)
        U_tr_64 /= np.linalg.norm(U_tr_64, axis=1, keepdims=True).clip(min=1e-6)
        U_te_64 /= np.linalg.norm(U_te_64, axis=1, keepdims=True).clip(min=1e-6)
        spectral_tr_blocks.append(U_tr_64)
        spectral_te_blocks.append(U_te_64)

        lp_dp = scores_exact_dp(U_test, groups_065, prm)
        P_dp = softmax(lp_dp, axis=1)
        HELPER_LIST.append((0.20, P_dp))
        print('[Routing: deberta_v3_large] Added Core Plain(0.22) + Twin(0.09), Helper DP(0.20).')
        del U_train, U_test, prm, lp_dp, P_dp

    del Z_train_bb, Z_test_bb, heads_deb_sd, w_d, b_d
    if w_cd is not None:
        del w_cd, b_cd
    gc.collect()

# ==============================================================================
# Optional Auto-Hook: pjmathematician/ultrabert-380m-ntl (5-Fold Inference)
# ==============================================================================
if 'ultrabert' in CKPTS:
    ckpt_ub = CKPTS['ultrabert']
    print(f'\\n--- [Optional Member: UltraBERT-380M 5-fold] Scanning {ckpt_ub} ---')
    try:
        ub_folds = [ckpt_ub / f"fold{i}" for i in range(5) if (ckpt_ub / f"fold{i}").exists()]
        if ub_folds:
            log_p_ub_folds = []
            for f_dir in ub_folds:
                ub_tok = AutoTokenizer.from_pretrained(str(f_dir), trust_remote_code=True)
                ub_mod = AutoModel.from_pretrained(str(f_dir), torch_dtype=torch.float16, trust_remote_code=True).cuda()
                ub_mod.eval()
                enc_ub = ub_tok(test_texts, padding=True, truncation=True, max_length=384, return_tensors='pt').cuda()
                with torch.no_grad():
                    out_ub = ub_mod(**enc_ub)
                    logits_ub = out_ub.logits if hasattr(out_ub, 'logits') else out_ub[0]
                    log_p_ub_folds.append(log_softmax(logits_ub.cpu().numpy()))
                del ub_tok, ub_mod, enc_ub
                gc.collect()
                torch.cuda.empty_cache()
            log_p_ub_mean = np.mean(log_p_ub_folds, axis=0)
            LOGIT_CORE += 0.12 * log_p_ub_mean
            print(f'[UltraBERT Hook] Successfully averaged {len(ub_folds)} folds and added with weight 0.12 to LOGIT_CORE.')
    except Exception as e:
        print(f'[UltraBERT Hook] Skipped due to error: {e}')
else:
    print('[UltraBERT Hook] pjmathematician/ultrabert-380m-ntl not mounted; skipping.')
""")

    # Cell 7: Joint 384-Dim Compact Spectral Bank & Optional Synthetic Hook
    add_md("### Step 7: Joint 384-Dim Compact Spectral Bank (Added to HELPER_LIST)")
    add_code("""print('\\n=== [Module 3] Fitting Joint Cross-Backbone 384-Dim Compact Spectral Bank ===')
if len(spectral_tr_blocks) >= 2:
    X_spec_tr = np.hstack(spectral_tr_blocks)
    X_spec_te = np.hstack(spectral_te_blocks)
    print(f'[Joint Spectral] Concatenated {len(spectral_tr_blocks)} backbones: Compact Dimension D = {X_spec_tr.shape[1]} (only {X_spec_tr.nbytes / (1024**2):.2f} MB RAM).')

    lr_joint_plain = LogisticRegression(C=0.15, max_iter=2000, random_state=RANDOM_SEED)
    lr_joint_plain.fit(X_spec_tr, y_tr_eval)
    lp_spec_plain = log_softmax(lr_joint_plain.decision_function(X_spec_te))

    lp_spec_center = np.zeros_like(lp_spec_plain)
    if twin_rows_085.any():
        pu, pinv = np.unique(prompts_tr_eval, return_inverse=True)
        Pm = np.zeros((len(pu), X_spec_tr.shape[1]), dtype=np.float32)
        np.add.at(Pm, pinv, X_spec_tr)
        Pm /= np.maximum(np.bincount(pinv)[:, None], 1.0)
        X_spec_tr_c = X_spec_tr - Pm[pinv]

        lr_joint_center = LogisticRegression(C=0.10, max_iter=2000, random_state=RANDOM_SEED)
        lr_joint_center.fit(X_spec_tr_c, y_tr_eval)

        X_spec_te_c = X_spec_te.copy()
        for g in np.unique(groups_085[twin_rows_085]):
            m_g = (groups_085 == g)
            X_spec_te_c[m_g] -= X_spec_te_c[m_g].mean(axis=0, keepdims=True)

        raw_sc = lr_joint_center.decision_function(X_spec_te_c[twin_rows_085])
        lp_spec_center[twin_rows_085] = log_softmax(raw_sc)
        del X_spec_tr_c, X_spec_te_c, lr_joint_center, Pm

    P_spec = softmax(lp_spec_plain + 0.4 * lp_spec_center, axis=1)
    HELPER_LIST.append((0.20, P_spec))
    print('[Joint Spectral] Plain + Centered Joint Spectral Bank added to HELPER_LIST (w=0.20).')
    del X_spec_tr, X_spec_te, lr_joint_plain
    gc.collect()

# Optional Auto-Hook for synthetic4404.joblib (kirilltym/ntl-spectral-style-heads-paired-4404)
synth_files = list(INPUT_DIR.rglob('synthetic4404.joblib'))
if synth_files:
    print(f'[Synthetic Hook] Found {synth_files[0]}. Loading external paired spectral head...')
    try:
        import joblib
        synth_clf = joblib.load(synth_files[0])
        if Z_test_qwen3_8b is not None:
            if hasattr(synth_clf, "predict_proba"):
                P_synth = synth_clf.predict_proba(Z_test_qwen3_8b)
            elif hasattr(synth_clf, "decision_function"):
                P_synth = softmax(synth_clf.decision_function(Z_test_qwen3_8b), axis=1)
            else:
                P_synth = None
            if P_synth is not None:
                HELPER_LIST.append((0.10, P_synth))
                print('[Synthetic Hook] Integrated synthetic4404 head into HELPER_LIST (w=0.10).')
    except Exception as e:
        print(f'[Synthetic Hook] Skipped: {e}')
else:
    print('[Synthetic Hook] No external synthetic4404.joblib mounted. Proceeding with pure internal bank.')

log_ram('After Joint Spectral Bank Completion')
""")

    # Cell 8: Module 4 — 88/12 Probability Mixture, Pre-Sinkhorn Shield, Claude Gate & Sinkhorn
    add_md("### Step 8: Module 4 — 88/12 Probability Mixture, Pre-Sinkhorn Shield, Claude Gate & Single Final Sinkhorn")
    add_code("""print('\\n=== [Module 4] Bounded 88/12 Core-to-Helper Probability Mixture ===')
R_core = softmax(LOGIT_CORE, axis=1)

if len(HELPER_LIST) > 0:
    total_hw = sum(w for w, _ in HELPER_LIST)
    print(f'[Probability Mixture] Blending {len(HELPER_LIST)} helpers (sum_w={total_hw:.2f}) into R_core...')
    H_helper = sum(w * P for w, P in HELPER_LIST) / total_hw
    P_mix = 0.88 * R_core + 0.12 * H_helper
else:
    print('[Probability Mixture] No helpers populated; R_core used directly.')
    P_mix = R_core

log_P = np.log(np.clip(P_mix, 1e-12, 1.0))

# ==============================================================================
# Sovereign Orthographic Shield (Applied BEFORE Sinkhorn)
# ==============================================================================
claude_classes = [
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "claude-fable-5"
]
glm_classes = ["glm-5.3", "glm-5.3-flash"]
gemini_classes = [
    "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash"
]
sol_class = "gpt-5.6-sol"

claude_idx = [LAB2I[c] for c in claude_classes if c in LAB2I]
glm_idx = [LAB2I[c] for c in glm_classes if c in LAB2I]
claude_and_glm_idx = claude_idx + glm_idx
gemini_idx = [LAB2I[c] for c in gemini_classes if c in LAB2I]
sol_idx = LAB2I.get(sol_class, None)

curly_chars = {"\\u2018", "\\u2019", "\\u201c", "\\u201d"}
n_curly, n_em, n_apos = 0, 0, 0

for i, t in enumerate(test_texts):
    # Rule 1: Curly Quotes (Claude & GLM have 0.00% in train.csv)
    if any(c in t for c in curly_chars):
        log_P[i, claude_and_glm_idx] -= 3.0
        n_curly += 1

    # Rule 2: Em-Dash (Gemini has only 0.17% in train.csv)
    if "\\u2014" in t:
        log_P[i, gemini_idx] -= 2.5
        n_em += 1

    # Rule 3: Straight Apostrophe (gpt-5.6-sol has 0.00% in train.csv)
    if "'" in t and sol_idx is not None:
        log_P[i, sol_idx] -= 3.0
        n_apos += 1

print(f'[Shield] Applied {n_curly} curly penalties (-3.0), {n_em} em-dash penalties (-2.5), {n_apos} straight apostrophe penalties (-3.0) BEFORE Sinkhorn.')

# ==============================================================================
# Top-1 Gated Claude Specialist (Applied BEFORE Sinkhorn)
# ==============================================================================
top1_preds = np.argmax(R_core, axis=1)
claude_rows = np.where(np.isin(top1_preds, claude_idx))[0]
print(f'\\n=== [Module 4] Top-1 Gated Claude Specialist on {len(claude_rows)} rows ===')

if len(claude_rows) > 0 and Z_test_qwen3_8b is not None:
    qwen3_8b_ckpt = CKPTS.get('qwen3_8b_claude')
    claude_head_file = None
    if qwen3_8b_ckpt:
        for fname in ['claude_head.safetensors', 'heads.safetensors']:
            if (qwen3_8b_ckpt / fname).exists():
                claude_head_file = qwen3_8b_ckpt / fname
                break
    if claude_head_file is None:
        for p in INPUT_DIR.rglob('claude_head.safetensors'):
            claude_head_file = p
            break

    if claude_head_file is not None and claude_head_file.exists():
        ch_sd = load_file(str(claude_head_file))
        ch_meta = json.load(open(claude_head_file.parent / 'meta.json')) if (claude_head_file.parent / 'meta.json').exists() else {'claude_labels': claude_classes}
        ch_labs = ch_meta.get('claude_labels', claude_classes)
        idx_ch = [ch_labs.index(c) for c in claude_classes]
        w_k = 'weight' if 'weight' in ch_sd else 'claude_weight'
        b_k = 'bias' if 'bias' in ch_sd else 'claude_bias'
        ch_w = ch_sd[w_k].cpu().numpy()[idx_ch]
        ch_b = ch_sd[b_k].cpu().numpy()[idx_ch]

        claude_scores = Z_test_qwen3_8b[claude_rows] @ ch_w.T + ch_b
        P_claude_head = softmax(claude_scores, axis=1)

        P_current = softmax(log_P[claude_rows], axis=1)
        for r_local, r_global in enumerate(claude_rows):
            mass_claude = np.sum(P_current[r_local, claude_idx])
            if mass_claude > 1e-12:
                P_claude_core = P_current[r_local, claude_idx] / mass_claude
                P_claude_rel = 0.78 * P_claude_core + 0.22 * P_claude_head[r_local]
                P_current[r_local, claude_idx] = mass_claude * P_claude_rel
            log_P[r_global] = np.log(np.clip(P_current[r_local], 1e-12, 1.0))

        print('[Module 4] Gated Claude head blended (0.78 core + 0.22 specialist) while preserving total Claude family mass.')
    else:
        print('[Module 4] Claude head file not found; skipped specialist blend.')

del Z_test_qwen3_8b
gc.collect()

# ==============================================================================
# Single Final 50-Step Log-Domain Sinkhorn-Knopp Marginal Balancing
# ==============================================================================
log_P = log_P - logsumexp(log_P, axis=1, keepdims=True)
if len(test_texts) >= 50:
    print(f'\\n[Sinkhorn] Running single final 50-step log-domain Sinkhorn marginal balancing (alpha=0.50) on {len(test_texts)} rows...')
    for _ in range(50):
        col_lse = logsumexp(log_P, axis=0, keepdims=True) - np.log(len(log_P))
        log_P = log_P - 0.50 * col_lse
        log_P = log_P - logsumexp(log_P, axis=1, keepdims=True)
else:
    print(f'\\n[Sinkhorn] Small test set ({len(test_texts)} < 50), skipping Sinkhorn balancing.')

# ==============================================================================
# Final Verification Gate Diagnostics Summary
# ==============================================================================
print('\\n' + '=' * 65)
print('             [CHECKPOINT VERIFICATION GATE SUMMARY]')
print('=' * 65)
print(f'{"Checkpoint Name":<32} | {"Train BAcc":<10} | {"Status":<8}')
print('-' * 65)
for r in gate_results:
    st = "PASSED" if r["passed"] else "DROPPED"
    print(f'{r["name"]:<32} | {r["bacc"]:<10.4f} | {st:<8}')
print('=' * 65)

# Save Final Predictions
final_pred_idx = np.argmax(log_P, axis=1)
final_pred_labels = [LABELS[i] for i in final_pred_idx]

sub_df = pd.DataFrame({
    'id': df_test['id'],
    'model': final_pred_labels
})
sub_df.to_parquet('submission.parquet', index=False)
print(f'\\n[SUCCESS] Wrote valid submission to /kaggle/working/submission.parquet ({Path(\"submission.parquet\").stat().st_size} bytes)!')
print(f'Submission shape: {sub_df.shape}')
print(f'\\nFirst 5 Predictions:')
print(sub_df.head())
print(f'\\nPredicted Model Distribution:')
print(sub_df['model'].value_counts())

log_ram('Final Complete Notebook Execution')
""")

    # Write notebook
    out_path = Path("kaggle_submission.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[SUCCESS] Generated Version 30 pipeline notebook: {out_path} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    create_v30_pipeline_notebook()
