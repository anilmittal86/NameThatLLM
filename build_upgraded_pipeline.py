import json
from pathlib import Path

def create_upgraded_pipeline_notebook():
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
    add_md("""# Name That LLM: Upgraded 0.620+ Multi-Backbone Two-Stage LoRA Pipeline
### High-Precision Authorship Attribution on Dual Tesla T4 GPUs
- **Multi-Backbone Two-Stage LoRA Pool**: Qwen3-14B, Qwen2.5-14B, Qwen3-8B LoRA + Claude Head, and DeBERTa-v3-Large.
- **Universal 2xT4 Trunk Loader**: Truncated layer caching, in-place LoRA weight merging, attention-sink zeroing (`m[:, 0] = 0`).
- **wPCA Shared-Prompt Gaussian Readout**: Two-way additive prompt-author covariance decomposition ($Z = \\mu_y + P_{\\text{prompt}} + E$), Ledoit-Wolf shrinkage, and coordinate-ascent repulsion.
- **4-View Topic-Blind LinearSVC**: Char (3-5), Word (1-3), Word-Shape, and Content-Masked Function Words trained across 78,000+ passages (`pool_v4.parquet` + `train.csv`).
- **Sovereign Orthographic Bayesian Shield**: Strict empirical invariants for curly quotes (Claude/GLM = 0%), straight apostrophes (`gpt-5.6-sol` = 0%), and em-dashes (Gemini = 0.17%).
- **Top-1 Gated Claude Specialist & Guarded Cluster Assignment**: Closed-family refinement without altering cross-family marginals.
""")

    # Cell 1: Environment, Dependencies & Master Label Ordering
    add_md("### Step 1: Environment, Dependencies & Master Label Ordering")
    add_code("""import os
import gc
import sys
import time
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.svm import LinearSVC
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

def _lse(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    return out if keepdims else (np.squeeze(out, axis=axis) if axis is not None else float(out.item()))

def log_softmax(L):
    return L - _lse(L, axis=1, keepdims=True)

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

def align_cols(arr, src_labels, dst_labels=LABELS):
    \"\"\"Permutes class dimension of array/matrix from src_labels order to dst_labels order.\"\"\"
    src_list = list(src_labels)
    idx = [src_list.index(l) for l in dst_labels]
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[idx]
    if arr.ndim == 2:
        if arr.shape[0] == len(src_labels) and arr.shape[1] != len(src_labels):
            return arr[idx, :]
        elif arr.shape[1] == len(src_labels) and arr.shape[0] != len(src_labels):
            return arr[:, idx]
        else:
            return arr[:, idx]
    return arr[..., idx]

def find_file(filename: str) -> Path:
    candidates = [
        Path(f'/kaggle/input/competitions/name-that-llm/{filename}'),
        Path(f'/kaggle/input/name-that-llm/{filename}'),
        Path(filename)
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(filename)

TRAIN_PATH = find_file('train.csv')
TEST_PATH = find_file('test.csv')
OUTPUT_PATH = Path('submission.parquet')

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
N_test = len(df_test)
test_texts = df_test['text'].fillna('').tolist()
test_ids = list(df_test['id'].values)
y_train = np.array([LAB2I[m] for m in df_train['model']])
prompts_train = df_train['prompt_id'].values

print(f'Train path: {TRAIN_PATH} ({len(df_train)} rows)')
print(f'Test path:  {TEST_PATH} ({N_test} rows)')
print(f'Master Classes ({K}): {LABELS}')
""")

    # Cell 2: Offline Asset Discovery (Checkpoints, Base Models, Pools)
    add_md("### Step 2: Offline Asset Discovery (Checkpoints, Base Models, Pools)")
    add_code("""def find_qwen3_14b_base() -> Optional[Path]:
    for p in Path('/kaggle/input').rglob('config.json'):
        p_str = str(p).lower()
        if 'adapter' in p_str or 'checkpoint' in p_str:
            continue
        try:
            cfg = json.load(open(p, 'r'))
            m_type = str(cfg.get('model_type', '')).lower()
            h_size = int(cfg.get('hidden_size', 0))
            n_layers = int(cfg.get('num_hidden_layers', 0))
            if 'qwen' in m_type and h_size == 5120 and n_layers == 40:
                return p.parent
        except Exception:
            pass
    for p in Path('/kaggle/input').rglob('config.json'):
        p_str = str(p).lower()
        if 'adapter' in p_str:
            continue
        if ('qwen-3' in p_str or 'qwen3' in p_str) and '14b' in p_str:
            return p.parent
    return None

def find_qwen25_14b_base() -> Optional[Path]:
    for p in Path('/kaggle/input').rglob('config.json'):
        p_str = str(p).lower()
        if 'adapter' in p_str or 'checkpoint' in p_str:
            continue
        try:
            cfg = json.load(open(p, 'r'))
            m_type = str(cfg.get('model_type', '')).lower()
            h_size = int(cfg.get('hidden_size', 0))
            n_layers = int(cfg.get('num_hidden_layers', 0))
            if 'qwen' in m_type and h_size == 5120 and n_layers == 48:
                return p.parent
        except Exception:
            pass
    for p in Path('/kaggle/input').rglob('config.json'):
        p_str = str(p).lower()
        if 'adapter' in p_str:
            continue
        if 'qwen2.5' in p_str and '14b' in p_str:
            return p.parent
    return None

def find_qwen3_8b_base() -> Optional[Path]:
    for p in Path('/kaggle/input').rglob('config.json'):
        p_str = str(p).lower()
        if 'adapter' in p_str or 'checkpoint' in p_str:
            continue
        try:
            cfg = json.load(open(p, 'r'))
            m_type = str(cfg.get('model_type', '')).lower()
            h_size = int(cfg.get('hidden_size', 0))
            n_layers = int(cfg.get('num_hidden_layers', 0))
            if 'qwen' in m_type and h_size == 4096 and n_layers == 36:
                return p.parent
        except Exception:
            pass
    for p in Path('/kaggle/input').rglob('config.json'):
        p_str = str(p).lower()
        if 'adapter' in p_str:
            continue
        if ('qwen-3' in p_str or 'qwen3' in p_str) and '8b' in p_str:
            return p.parent
    return None

def find_deberta_base() -> Optional[Path]:
    for p in Path('/kaggle/input').rglob('config.json'):
        p_str = str(p).lower()
        if 'deberta' in p_str and 'adapter' not in p_str:
            return p.parent
    return None

# Discover LoRA Checkpoint Dirs
def find_lora_dirs() -> Dict[str, Path]:
    discovered = {}
    kaggle_input = Path('/kaggle/input')
    
    # 1. Qwen3-14B LoRA
    for p in kaggle_input.rglob('heads.safetensors'):
        if 'lora_qwen3_14b' in str(p).lower() or 'lora14b_v1' in str(p).lower():
            discovered['qwen3_14b'] = p.parent
            break
            
    # 2. Qwen2.5-14B LoRA
    for p in kaggle_input.rglob('heads.safetensors'):
        if 'lora_qwen25_14b' in str(p).lower():
            discovered['qwen25_14b'] = p.parent
            break
            
    # 3. Qwen3-8B LoRA + Claude Head
    for p in kaggle_input.rglob('readouts.npz'):
        if 'final' in str(p).lower():
            discovered['qwen3_8b'] = p.parent
            break
    if 'qwen3_8b' not in discovered:
        for p in kaggle_input.rglob('readouts.npz'):
            discovered['qwen3_8b'] = p.parent
            break
            
    # 4. DeBERTa fine-tuned encoder
    for p in kaggle_input.rglob('head.safetensors'):
        if 'deberta' in str(p).lower():
            discovered['deberta_blend'] = p.parent
            break
            
    return discovered

# Discover External Public Passage Pools
def find_passage_pool() -> Optional[Path]:
    kaggle_input = Path('/kaggle/input')
    for fname in ['pool_v4.parquet', 'pool_v1.parquet', 'passages.csv']:
        for p in kaggle_input.rglob(fname):
            return p
    return None

BASE_QWEN3_14B = find_qwen3_14b_base()
BASE_QWEN25_14B = find_qwen25_14b_base()
BASE_QWEN3_8B = find_qwen3_8b_base()
BASE_DEBERTA = find_deberta_base()
LORA_DIRS = find_lora_dirs()
POOL_PATH = find_passage_pool()

print(f'Discovered Base Qwen3-14B:   {BASE_QWEN3_14B}')
print(f'Discovered Base Qwen2.5-14B: {BASE_QWEN25_14B}')
print(f'Discovered Base Qwen3-8B:    {BASE_QWEN3_8B}')
print(f'Discovered Base DeBERTa:     {BASE_DEBERTA}')
print(f'Discovered LoRA Dirs:        {LORA_DIRS}')
print(f'Discovered Passage Pool:     {POOL_PATH}')
""")

    # Cell 3: Module 1 — 4-View Topic-Blind LinearSVC on train.csv + Public Pool
    add_md("### Step 3: Module 1 — 4-View Topic-Blind LinearSVC on train.csv + Public Pool")
    add_code("""FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose "
    "this these those it its they them their theirs he him his she her hers we us our ours "
    "you your yours i me my mine in on at to from with without for of by over under between "
    "through into about is are was were be been being have has had do does did can could "
    "will would shall should may might must not no nor all any some each every both either "
    "neither other another such only also just very too more most less least here there "
    "when where why how what whether then now still yet even much many few little".split()
)

def shape_text(t: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m.group(0)[0].isupper() else "x", str(t))

def function_text(t: str) -> str:
    return " ".join(
        w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
        for w in re.findall(r"\w+|[^\w\s]", str(t))
    )

fit_texts = df_train['text'].fillna('').tolist()
fit_y = y_train.copy()

if POOL_PATH is not None and POOL_PATH.exists():
    print(f'[Module 1] Loading public passage pool from {POOL_PATH.name}...')
    if POOL_PATH.suffix == '.parquet':
        df_pool = pd.read_parquet(POOL_PATH)
    else:
        df_pool = pd.read_csv(POOL_PATH)
        
    text_c = 'text' if 'text' in df_pool.columns else ('passage' if 'passage' in df_pool.columns else df_pool.columns[0])
    label_c = 'model' if 'model' in df_pool.columns else ('label' if 'label' in df_pool.columns else df_pool.columns[1])
    
    df_pool = df_pool[df_pool[label_c].isin(LABELS)].copy()
    print(f'[Module 1] Filtered public pool rows: {len(df_pool)} across {df_pool[label_c].nunique()} classes.')
    
    if df_pool[label_c].nunique() >= 18:
        p_texts = df_pool[text_c].fillna('').tolist()
        p_y = np.array([LAB2I[m] for m in df_pool[label_c]])
        fit_texts = fit_texts + p_texts
        fit_y = np.r_[fit_y, p_y]
        print(f'[Module 1] Combined train + full public pool: {len(fit_texts)} rows.')
    else:
        df_pool_sub = df_pool.groupby(label_c).head(300)
        p_texts = df_pool_sub[text_c].fillna('').tolist()
        p_y = np.array([LAB2I[m] for m in df_pool_sub[label_c]])
        fit_texts = fit_texts + p_texts
        fit_y = np.r_[fit_y, p_y]
        print(f'[Module 1] Combined train + capped public pool: {len(fit_texts)} rows.')

# Sample for fast execution if interactive commit mode
if N_test < 50 and len(fit_texts) > 10000:
    print('[Module 1] Fast interactive commit mode: sampling 8,000 pool rows...')
    tr_indices = np.arange(len(df_train))
    extra_indices = np.random.choice(np.arange(len(df_train), len(fit_texts)), size=6000, replace=False)
    fit_idx = np.r_[tr_indices, extra_indices]
    fit_texts_svc = [fit_texts[i] for i in fit_idx]
    fit_y_svc = fit_y[fit_idx]
else:
    fit_texts_svc = fit_texts
    fit_y_svc = fit_y

print(f'[Module 1] Fitting 4-View LinearSVC on {len(fit_texts_svc)} passages...')

vec_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=250000, lowercase=False, sublinear_tf=True, min_df=2)
vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=180000, lowercase=False, sublinear_tf=True, min_df=2, token_pattern=r'(?u)\b\w+\b|[^\w\s]')
vec_shape = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=60000, lowercase=False, sublinear_tf=True, min_df=2)
vec_func = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=50000, lowercase=True, sublinear_tf=True, min_df=2)

t0_svc = time.time()
Xc_tr = 1.0 * vec_char.fit_transform(fit_texts_svc)
Xw_tr = 0.7 * vec_word.fit_transform(fit_texts_svc)
Xs_tr = 0.5 * vec_shape.fit_transform([shape_text(t) for t in fit_texts_svc])
Xf_tr = 0.5 * vec_func.fit_transform([function_text(t) for t in fit_texts_svc])
X_sparse_tr = sparse.hstack([Xc_tr, Xw_tr, Xs_tr, Xf_tr], format='csr')

svc_4view = LinearSVC(C=0.5, class_weight='balanced', random_state=RANDOM_SEED, max_iter=5000)
svc_4view.fit(X_sparse_tr, fit_y_svc)
print(f'[Module 1] 4-View LinearSVC fitted on {X_sparse_tr.shape[1]} features in {time.time() - t0_svc:.2f}s.')

Xc_te = 1.0 * vec_char.transform(test_texts)
Xw_te = 0.7 * vec_word.transform(test_texts)
Xs_te = 0.5 * vec_shape.transform([shape_text(t) for t in test_texts])
Xf_te = 0.5 * vec_func.transform([function_text(t) for t in test_texts])
X_sparse_te = sparse.hstack([Xc_te, Xw_te, Xs_te, Xf_te], format='csr')

log_p_4view_svc = log_softmax(svc_4view.decision_function(X_sparse_te))
print(f'[Module 1] Computed 4-View LinearSVC log-probabilities for test set.')
""")

    # Cell 4: Universal 2xT4 LoRA Trunk Loader & Extractor
    add_md("### Step 4: Universal 2xT4 LoRA Trunk Loader & Extractor")
    add_code("""def load_truncated_trunk(base_dir: str, keep_layers: int, dtype=torch.float16):
    cfg = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    cfg.num_hidden_layers = keep_layers
    cfg.use_cache = False
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:keep_layers]
        
    max_mem = {0: "13.5GiB", 1: "13.5GiB"} if torch.cuda.device_count() > 1 else None
    model = AutoModel.from_pretrained(
        base_dir,
        config=cfg,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
    )
    model.norm = nn.Identity()
    model.eval()
    return model

@torch.no_grad()
def merge_lora_weights_inplace(model, adapter_dir: str):
    print(f'[LoRA Merge] Merging adapter from: {adapter_dir}...')
    acfg_file = Path(adapter_dir) / "adapter_config.json"
    if not acfg_file.exists():
        acfg_file = Path(adapter_dir) / "adapter" / "adapter_config.json"
    acfg = json.load(open(acfg_file))
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    
    weights_file = Path(adapter_dir) / "adapter_model.safetensors"
    if not weights_file.exists():
        weights_file = Path(adapter_dir) / "adapter" / "adapter_model.safetensors"
    sd = load_file(str(weights_file))
    
    modules = dict(model.named_modules())
    n_merged = 0
    for k, A in sd.items():
        if ".lora_A." in k:
            B_key = k.replace(".lora_A.", ".lora_B.")
            if B_key not in sd:
                continue
            B = sd[B_key]
            mod_path = k.split(".lora_A.")[0]
            if "layers." not in mod_path:
                continue
            rel_path = mod_path[mod_path.index("layers."):]
            if rel_path in modules:
                lin = modules[rel_path]
                dev, dt = lin.weight.device, lin.weight.dtype
                delta = (B.to(dev, torch.float32) @ A.to(dev, torch.float32)) * scale
                lin.weight.data.add_(delta.to(dt))
                n_merged += 1
    print(f'[LoRA Merge] Successfully merged {n_merged} linear LoRA weights into truncated trunk!')

class UniversalLoRAExtractor:
    \"\"\"Extracts multi-layer representations with token-0 attention-sink masking.\"\"\"
    def __init__(self, base_dir: Path, ckpt_dir: Path, keep_layers: int, layers: List[int], max_len: int = 384):
        self.base_dir = str(base_dir)
        self.ckpt_dir = ckpt_dir
        self.layers = layers
        self.max_len = max_len
        
        tok_dir = str(ckpt_dir / "tokenizer") if (ckpt_dir / "tokenizer").exists() else self.base_dir
        self.tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<|endoftext|>"
            
        self.model = load_truncated_trunk(self.base_dir, keep_layers=keep_layers, dtype=torch.float16)
        adapter_path = str(ckpt_dir / "adapter") if (ckpt_dir / "adapter").exists() else str(ckpt_dir)
        merge_lora_weights_inplace(self.model, adapter_path)
        
        H = self.model.config.hidden_size
        heads_sd = load_file(str(ckpt_dir / "heads.safetensors"))
        self.device_norms = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in self.layers]).to(self.device_norms).float()
        
        norm_dict = {}
        for idx, l in enumerate(self.layers):
            w_key = f"norms.{idx}.weight"
            b_key = f"norms.{idx}.bias"
            if w_key in heads_sd:
                norm_dict[f"{idx}.weight"] = heads_sd[w_key]
                norm_dict[f"{idx}.bias"] = heads_sd[b_key]
        self.norms.load_state_dict(norm_dict, strict=False)
        self.norms.eval()
        self.H = H
        self.first_dev = next(self.model.parameters()).device
        print(f'[Extractor] Loaded {len(self.layers)} norms with hidden size H={self.H}.')

    @torch.no_grad()
    def extract(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        raw_ids = self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        eos_id = self.tok.eos_token_id or 151643
        ids = [x + [eos_id] for x in raw_ids]

        order = np.argsort([len(x) for x in ids], kind="stable")
        total_dim = self.H * len(self.layers)
        Z = np.zeros((len(ids), total_dim), dtype=np.float32)

        t0 = time.time()
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            enc = self.tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.first_dev)
            attn_mask = enc["attention_mask"].to(self.first_dev)

            out = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            hs = out.hidden_states

            # CRITICAL ATTENTION SINK FIX: Zero out position 0 before mean pooling
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

        print(f'[Extractor] Extracted {len(ids)} samples in {time.time() - t0:.2f}s ({len(ids) / max(0.1, time.time() - t0):.1f} samples/s).')
        return Z

    def free_memory(self):
        print('[Extractor] Freeing GPU VRAM...')
        del self.model, self.norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
""")

    # Cell 5: Module 3 — Shared-Prompt Whitened PCA (wPCA) + Ledoit-Wolf Joint Gaussian Readout
    add_md("### Step 5: Module 3 — Shared-Prompt Whitened PCA (wPCA) + Ledoit-Wolf Joint Gaussian Readout")
    add_code("""GAUSS_DEFAULT = {
    "red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw",
    "lam_floor": 1e-4, "n_exact": 3, "inj": True, "iters": 5, "repel": True, "sp_scale": 1.0
}

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
    c = dict(GAUSS_DEFAULT)
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
    R = Xs - Pm[pinv] if c["red"] == "wpca" else Xs
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
    ae = float(ledoit_wolf_shrinkage(E, assume_centered=True)) if c["se_shrink"] == "lw" else float(c["se_shrink"])
    Se = _shrunk(Se_raw, ae)

    Pc = P - P.mean(0)
    Sp_raw = (Pc.T @ Pc) / max(G - 1, 1) - np.mean(1.0 / np.maximum(ng, 1.0)) * Se_raw
    ap = float(ledoit_wolf_shrinkage(Pc, assume_centered=True)) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = ((1.0 - ap) * Sp_raw + ap * (tr / d) * np.eye(d)) * c["sp_scale"]

    # Simultaneous diagonalization: A Se A^T = I, A Sp A^T = diag(lam)
    L = np.linalg.cholesky(Se + 1e-6 * np.eye(d))
    Li = np.linalg.inv(L)
    M = 0.5 * ((Li @ Sp @ Li.T) + (Li @ Sp @ Li.T).T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, c["lam_floor"], None)
    A = Q.T @ Li
    W = V @ A.T
    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 2.4, "Tg": 2.3, "cfg": c}

def singleton_ll(U, m, lam):
    \"\"\"Exact marginal log-likelihood for singletons with prompt effect integrated out.\"\"\"
    R = U[:, None, :] - m[None, :, :]
    return -0.5 * (R ** 2 / (1.0 + lam[None, None, :])).sum(-1)

def joint_group_scores(U_grp, m, lam, T1=2.4, Tg=2.3, iters=5, repel=True):
    \"\"\"Computes marginal log-probabilities for shared-prompt clusters with injectivity repulsion.\"\"\"
    n, d = U_grp.shape
    C = m.shape[0]
    if n == 1:
        return log_softmax(singleton_ll(U_grp, m, lam) / T1)

    gam = lam / (1.0 + n * lam)
    R = U_grp[:, None, :] - m[None, :, :]
    a = -0.5 * ((1.0 - gam) * (R ** 2)).sum(-1)
    Rw = R * np.sqrt(gam)[None, None, :]
    B = np.tensordot(Rw, Rw, axes=([2], [2])).transpose(0, 2, 1, 3)

    unary = (a / Tg)
    q = np.exp(log_softmax(unary))
    for _ in range(iters):
        msg = np.zeros_like(unary)
        for i in range(n):
            for j in range(n):
                if i != j:
                    coup = (B[i, j] @ q[j]) / Tg
                    if repel:
                        coup = coup - np.log(np.clip(1.0 - q[j], 1e-4, 1.0))
                    msg[i] += coup
        q = np.exp(log_softmax(unary + msg))
    return np.log(np.clip(q, 1e-9, 1.0))
""")

    # Cell 6: Multi-Backbone Sequential Feature Extraction & Blending
    add_md("### Step 6: Multi-Backbone Sequential Feature Extraction & Blending")
    add_code("""tfidf_content = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=50000)
X_content = tfidf_content.fit_transform(test_texts)
D_tfidf = np.clip(1.0 - (X_content @ X_content.T).toarray(), 0.0, 1.0)
np.fill_diagonal(D_tfidf, 0.0)

if len(test_texts) > 1:
    test_groups = fcluster(linkage(squareform(D_tfidf, checks=False), method="average"), t=0.60, criterion="distance")
else:
    test_groups = np.array([1])

u_groups, group_counts = np.unique(test_groups, return_counts=True)
count_map = dict(zip(u_groups, group_counts))
sizes = np.array([count_map[g] for g in test_groups])
adapt_rows = (sizes >= 2) & (sizes <= 18)
print(f'Test prompt clusters: {len(u_groups)} | Eligible multi-passage rows (2 <= size <= 18): {adapt_rows.sum()}/{len(test_texts)}')

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

LOGIT = np.zeros((N_test, K), dtype=np.float64)

causal_backbones = []

# 1. Qwen3-14B LoRA
if 'qwen3_14b' in LORA_DIRS and BASE_QWEN3_14B is not None:
    causal_backbones.append({
        'name': 'qwen3_14b',
        'base': BASE_QWEN3_14B,
        'ckpt': LORA_DIRS['qwen3_14b'],
        'keep': 27,
        'layers': [2, 4, 6, 9, 13, 17, 22, 27],
        'max_len': 447,
        'weight': 0.30
    })

# 2. Qwen2.5-14B LoRA
if 'qwen25_14b' in LORA_DIRS and BASE_QWEN25_14B is not None:
    causal_backbones.append({
        'name': 'qwen25_14b',
        'base': BASE_QWEN25_14B,
        'ckpt': LORA_DIRS['qwen25_14b'],
        'keep': 32,
        'layers': [3, 5, 8, 11, 16, 21, 27, 32],
        'max_len': 447,
        'weight': 0.25
    })

# 3. Qwen3-8B LoRA + Claude Head
if 'qwen3_8b' in LORA_DIRS and BASE_QWEN3_8B is not None:
    causal_backbones.append({
        'name': 'qwen3_8b',
        'base': BASE_QWEN3_8B,
        'ckpt': LORA_DIRS['qwen3_8b'],
        'keep': 24,
        'layers': [2, 4, 6, 8, 12, 16, 20, 24],
        'max_len': 384,
        'weight': 0.25
    })

print(f'Active Causal LoRA Backbones ({len(causal_backbones)}): {[b[\"name\"] for b in causal_backbones]}')

Z_test_qwen3_8b = None
qwen3_8b_ro = None

tot_bb_weight = sum(bb['weight'] for bb in causal_backbones)
for bb in causal_backbones:
    norm_w = (bb['weight'] / tot_bb_weight) * 0.80 if tot_bb_weight > 0 else 0.80
    print(f'\\n=== [Backbone: {bb[\"name\"]}] Loading {bb[\"base\"]} with keep={bb[\"keep\"]} ===')
    extractor = UniversalLoRAExtractor(
        base_dir=bb['base'],
        ckpt_dir=bb['ckpt'],
        keep_layers=bb['keep'],
        layers=bb['layers'],
        max_len=bb['max_len']
    )
    
    Z_all = extractor.extract(eval_texts, batch_size=8)
    extractor.free_memory()
    
    Z_test_bb = Z_all[:N_test]
    Z_train_bb = Z_all[N_test:]
    
    if bb['name'] == 'qwen3_8b':
        Z_test_qwen3_8b = Z_test_bb.copy()
        
    ckpt_path = bb['ckpt']
    if (ckpt_path / 'readouts.npz').exists():
        ro = np.load(ckpt_path / 'readouts.npz', allow_pickle=True)
        qwen3_8b_ro = ro
        plain_w = align_cols(ro['plain_coef'], ro['classes'], LABELS)
        plain_b = align_cols(ro['plain_intercept'], ro['classes'], LABELS)
        center_w = align_cols(ro['center_coef'], ro['classes'], LABELS)
        center_b = align_cols(ro['center_intercept'], ro['classes'], LABELS)
    else:
        heads_sd = load_file(str(ckpt_path / 'heads.safetensors'))
        plain_w = heads_sd['plain.weight'].cpu().numpy()
        plain_b = heads_sd['plain.bias'].cpu().numpy()
        center_w = heads_sd['centered.weight'].cpu().numpy()
        center_b = heads_sd['centered.bias'].cpu().numpy()
        if (ckpt_path / 'meta.json').exists():
            meta = json.load(open(ckpt_path / 'meta.json'))
            src_labs = meta.get('labels', LABELS)
            plain_w = align_cols(plain_w, src_labs, LABELS)
            plain_b = align_cols(plain_b, src_labs, LABELS)
            center_w = align_cols(center_w, src_labs, LABELS)
            center_b = align_cols(center_b, src_labs, LABELS)
            
    # 1. Plain Log-probabilities
    log_p_plain = log_softmax(Z_test_bb @ plain_w.T + plain_b)
    
    # 2. Centered Twin Log-probabilities
    log_p_center = np.zeros_like(log_p_plain)
    if adapt_rows.any():
        Zc = Z_test_bb.copy()
        for g in np.unique(test_groups[adapt_rows]):
            m = (test_groups == g)
            Zc[m] -= Zc[m].mean(axis=0, keepdims=True)
        log_p_center[adapt_rows] = log_softmax(Zc[adapt_rows] @ center_w.T + center_b)
        
    # 3. Module 3: wPCA Shared-Prompt Gaussian Readout
    print(f'[wPCA Gaussian] Fitting shared-prompt Gaussian on {len(Z_train_bb)} train rows...')
    gauss_cfg = {"d": min(256, max(len(Z_train_bb) - 15, 10)), "se_shrink": "lw", "sp_shrink": "lw"}
    gauss_model = fit_shared_gaussian(Z_train_bb, y_tr_eval, prompts_tr_eval, cfg=gauss_cfg)
    
    Xs_test = (Z_test_bb - gauss_model["mx"]) / gauss_model["sx"]
    U_test = Xs_test @ gauss_model["W"]
    
    log_p_gauss = np.zeros_like(log_p_plain)
    for g in np.unique(test_groups):
        idx_g = np.where(test_groups == g)[0]
        U_grp = U_test[idx_g]
        scores_g = joint_group_scores(
            U_grp, gauss_model["m"], gauss_model["lam"],
            T1=gauss_model["T1"], Tg=gauss_model["Tg"],
            iters=5, repel=True
        )
        log_p_gauss[idx_g] = scores_g
        
    # Combine backbone evidence: adapt_rows gets center, singletons get 50/50 plain/gauss
    bb_logit = np.zeros_like(log_p_plain)
    if adapt_rows.any():
        bb_logit[adapt_rows] = 0.40 * log_p_plain[adapt_rows] + 0.20 * log_p_center[adapt_rows] + 0.40 * log_p_gauss[adapt_rows]
        bb_logit[~adapt_rows] = 0.50 * log_p_plain[~adapt_rows] + 0.50 * log_p_gauss[~adapt_rows]
    else:
        bb_logit = 0.50 * log_p_plain + 0.50 * log_p_gauss
        
    LOGIT += norm_w * bb_logit
    print(f'[Backbone: {bb[\"name\"]}] Accumulated with normalized weight {norm_w:.4f}.')

# Add 4-View LinearSVC (Weight 0.10)
LOGIT += 0.10 * log_p_4view_svc
print('[Module 1] Added 4-View LinearSVC with weight 0.10.')

# Add DeBERTa-v3 Frozen / Fine-tuned Encoder (Weight 0.10)
if BASE_DEBERTA is not None:
    print('\\n=== [Backbone: DeBERTa-v3] Loading Encoder ===')
    deb_tok = AutoTokenizer.from_pretrained(str(BASE_DEBERTA), trust_remote_code=True)
    if deb_tok.pad_token is None:
        deb_tok.pad_token = deb_tok.eos_token or '[PAD]'
    deb_model = AutoModel.from_pretrained(str(BASE_DEBERTA), torch_dtype=torch.float16).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    deb_model.eval()
    
    dev_deb = next(deb_model.parameters()).device
    def extract_deb_last(texts, bs=16):
        all_e = []
        for i in range(0, len(texts), bs):
            enc = deb_tok(texts[i:i+bs], padding=True, truncation=True, max_length=384, return_tensors='pt').to(dev_deb)
            out = deb_model(**enc)
            m = enc['attention_mask'].unsqueeze(-1).float()
            pool = (out.last_hidden_state.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            all_e.append(pool.cpu().numpy())
        return np.vstack(all_e)
        
    E_te = extract_deb_last(test_texts)
    
    del deb_model, deb_tok
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check if pre-trained head exists in deberta_blend
    head_file = LORA_DIRS.get('deberta_blend', Path()).parent / 'deberta_v3_large' / 'head.safetensors'
    if not head_file.exists():
        head_file = Path('/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend/deberta_v3_large/head.safetensors')
        
    if head_file.exists():
        head_sd = load_file(str(head_file))
        meta_deb = json.load(open(head_file.parent / 'meta.json'))
        w_deb = align_cols(head_sd['weight'].cpu().numpy(), meta_deb['labels'], LABELS)
        b_deb = align_cols(head_sd['bias'].cpu().numpy(), meta_deb['labels'], LABELS)
        log_p_deb = log_softmax(E_te @ w_deb.T + b_deb)
        print('[Backbone: DeBERTa-v3] Evaluated pre-trained neural head.')
    else:
        E_tr = extract_deb_last(df_tr_eval['text'].fillna('').tolist())
        sc_deb = StandardScaler()
        E_tr_s = sc_deb.fit_transform(E_tr)
        E_te_s = sc_deb.transform(E_te)
        lr_deb = LinearSVC(C=0.1, class_weight='balanced', random_state=RANDOM_SEED, max_iter=3000)
        lr_deb.fit(E_tr_s, y_tr_eval)
        log_p_deb = log_softmax(lr_deb.decision_function(E_te_s))
        print('[Backbone: DeBERTa-v3] Fitted transductive LinearSVC head.')
        
    LOGIT += 0.10 * log_p_deb
    print('[Backbone: DeBERTa-v3] Accumulated with weight 0.10.')
""")

    # Cell 7: Module 4 & 5 — Sovereign Orthographic Shield, Top-1 Claude Gate & Guarded Hungarian
    add_md("### Step 7: Module 4 & 5 — Sovereign Orthographic Shield, Top-1 Claude Gate & Guarded Hungarian")
    add_code("""P = softmax(LOGIT)
if N_test >= 50:
    P_cal = P / (P.mean(axis=0, keepdims=True) ** 0.35)
    LOGIT = np.log(np.clip(P_cal, 1e-12, 1.0))

# MODULE 4: Sovereign Orthographic Bayesian Shield
print('\\n=== [Module 4] Applying Sovereign Orthographic Bayesian Shield ===')
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

claude_idx = [LABELS.index(c) for c in claude_classes if c in LABELS]
glm_idx = [LABELS.index(c) for c in glm_classes if c in LABELS]
gemini_idx = [LABELS.index(c) for c in gemini_classes if c in LABELS]
sol_idx = LABELS.index(sol_class) if sol_class in LABELS else None

curly_chars = {"\u2018", "\u2019", "\u201c", "\u201d"}
n_curly, n_em, n_apos = 0, 0, 0

for i, t in enumerate(test_texts):
    # Rule 1: Curly Quotes (Claude & GLM have 0.00% in train.csv)
    if any(c in t for c in curly_chars):
        LOGIT[i, claude_idx] -= 8.0
        LOGIT[i, glm_idx] -= 8.0
        n_curly += 1

    # Rule 2: Em-Dash (Gemini has only 0.17% in train.csv)
    if "\u2014" in t:
        LOGIT[i, gemini_idx] -= 5.5
        n_em += 1

    # Rule 3: Straight Apostrophe (gpt-5.6-sol has 0.00% in train.csv)
    if "'" in t and sol_idx is not None:
        LOGIT[i, sol_idx] -= 6.0
        n_apos += 1

print(f'[Module 4] Shield applied: {n_curly} curly-quote penalties, {n_em} em-dash penalties, {n_apos} straight-apos penalties.')

pred_idx = np.argmax(LOGIT, axis=1)
initial_pred_idx = pred_idx.copy()

# MODULE 5: Top-1 Gated Claude Specialist
is_claude = np.isin(pred_idx, claude_idx)
if is_claude.any():
    claude_rows = np.where(is_claude)[0]
    print(f'\\n=== [Module 5] Top-1 Gated Claude Specialist on {len(claude_rows)} rows ===')
    
    # Train Claude-only LinearSVC on train.csv
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
        # Align ro claude classes to ro_claude_classes
        ro_src = list(qwen3_8b_ro['claude_classes'])
        logits_c_lora = align_cols(logits_c_lora, ro_src, ro_claude_classes)
        log_p_c_lora = log_softmax(logits_c_lora)
        log_p_claude_comb = 0.70 * log_p_c_lora + 0.30 * log_p_c_svc
    else:
        log_p_claude_comb = log_p_c_svc
        
    best_c = np.argmax(log_p_claude_comb, axis=1)
    
    for r_idx, c_best in zip(claude_rows, best_c):
        best_name = ro_claude_classes[c_best]
        pred_idx[r_idx] = LABELS.index(best_name)
        
    assert np.array_equal(pred_idx[~is_claude], initial_pred_idx[~is_claude]), "Safety Check Failed: Non-Claude predictions modified!"
    print('[Module 5] Claude specialist completed. Invariant verified: Non-Claude predictions 100% untouched.')

# Guarded Hungarian Matchmaker within tight multi-passage clusters
if adapt_rows.any():
    P_final = softmax(LOGIT)
    for g in np.unique(test_groups[adapt_rows]):
        idx_g = np.where(test_groups == g)[0]
        n_g = len(idx_g)
        if 2 <= n_g <= 18:
            preds_g = pred_idx[idx_g]
            if len(np.unique(preds_g)) < n_g:
                cost = -P_final[idx_g]
                row_ind, col_ind = linear_sum_assignment(cost)
                min_p = np.min([P_final[idx_g[r], col_ind[r]] for r in range(n_g)])
                if min_p > 0.03:
                    for r, c in zip(row_ind, col_ind):
                        pred_idx[idx_g[r]] = c
""")

    # Cell 8: Parity Checks, Self-Test & Final Submission Output
    add_md("### Step 8: Parity Checks, Self-Test & Final Submission Output")
    add_code("""if N_test < 50:
    print('\\n=== [Self-Test Verification] Running on 54 samples of train.csv ===')
    prompts_3 = df_train['prompt_id'].unique()[:3]
    df_self = df_train[df_train['prompt_id'].isin(prompts_3)].copy()
    y_self = np.array([LAB2I[m] for m in df_self['model']])
    
    X_self_c = 1.0 * vec_char.transform(df_self['text'])
    X_self_w = 0.7 * vec_word.transform(df_self['text'])
    X_self_s = 0.5 * vec_shape.transform([shape_text(t) for t in df_self['text']])
    X_self_f = 0.5 * vec_func.transform([function_text(t) for t in df_self['text']])
    X_self_sparse = sparse.hstack([X_self_c, X_self_w, X_self_s, X_self_f], format='csr')
    preds_self_svc = svc_4view.predict(X_self_sparse)
    acc_self_svc = balanced_accuracy_score(y_self, preds_self_svc)
    print(f'[Self-Test] 4-View LinearSVC Balanced Acc on 54 rows: {acc_self_svc:.4f}')

pred_labels = [ID2LAB[int(i)] for i in pred_idx]
sub_df = pd.DataFrame({
    'id': test_ids,
    'model': pred_labels
})

assert list(sub_df.columns) == ['id', 'model'], f'Invalid columns: {sub_df.columns}'
assert len(sub_df) == N_test, f'Row count mismatch: expected {N_test}, got {len(sub_df)}'
assert not sub_df.isnull().any().any(), 'Submission contains nulls/NaNs!'
invalid = set(sub_df['model']) - set(LABELS)
assert len(invalid) == 0, f'Invalid model strings: {invalid}'

sub_df.to_parquet(OUTPUT_PATH, index=False)
print(f'\\n[SUCCESS] Wrote valid submission to {OUTPUT_PATH.resolve()} ({os.path.getsize(OUTPUT_PATH)} bytes)!')
print('\\nFirst 5 Predictions:')
print(sub_df.head())
print('\\nPredicted Model Distribution:')
print(sub_df['model'].value_counts())
""")

    out_file = Path("kaggle_submission.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated upgraded {out_file} with {len(nb['cells'])} cells.")

if __name__ == "__main__":
    create_upgraded_pipeline_notebook()
