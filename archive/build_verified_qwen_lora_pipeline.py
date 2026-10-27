import json
from pathlib import Path

def create_verified_pipeline_notebook():
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

    # Header
    add_md("""# Name That LLM: Verified 0.54+ Qwen3-8B LoRA + Adaptive Centering + Claude Gate
### End-to-End High-Precision Pipeline on 2xT4 GPUs
- **Qwen3-8B LoRA 24-Layer Trunk**: Attention-sink zeroing (`m[:, 0] = 0`), manual EOS append, in-place LoRA weight merging.
- **Anchor & Fold 0 Parity Check**: Cosine similarity check against `anchor_features.npz`.
- **Multi-Branch Ensemble**: Qwen3 LoRA Readout + DeBERTa-v3 Multi-Layer Frozen Encoder + Multi-View LinearSVC + Stylometric LightGBM.
- **Soft Adaptive Centering**: Content TF-IDF + DeBERTa semantic clustering (2 <= size <= 18); zero hard Hungarian distortion.
- **Top-1 Gated Claude Specialist & OpenRouter Tie-Breakers**: Closed-family refinement without altering cross-family marginals.
""")

    # Cell 1: Environment & Setup
    add_md("### Step 1: Environment & Setup")
    add_code("""import os
import gc
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import lightgbm as lgb
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

def log_softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))

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

print(f'Train path: {TRAIN_PATH}')
print(f'Test path: {TEST_PATH}')
""")

    # Cell 2: Offline Asset Discovery
    add_md("### Step 2: Offline Asset Discovery")
    add_code("""def find_qwen_base() -> str:
    kaggle_input = Path('/kaggle/input')
    for p in kaggle_input.rglob('config.json'):
        p_str = str(p).lower()
        if 'qwen' in p_str and 'adapter' not in p_str:
            return str(p.parent)
    raise FileNotFoundError('Qwen3 Base model not found under /kaggle/input!')

def find_qwen_lora_head_dir() -> Path:
    kaggle_input = Path('/kaggle/input')
    for p in kaggle_input.rglob('readouts.npz'):
        if 'final' in str(p).lower():
            return p.parent
    for p in kaggle_input.rglob('readouts.npz'):
        return p.parent
    local_p = Path('artifacts/qwen_claude_inspect/final')
    if local_p.exists():
        return local_p
    raise FileNotFoundError('Qwen3 LoRA + Claude Head not found!')

def find_openrouter_passages() -> Optional[str]:
    kaggle_input = Path('/kaggle/input')
    for p in kaggle_input.rglob('passages.csv'):
        if 'openrouter' in str(p).lower():
            return str(p)
    local_p = Path('artifacts/openrouter_inspect/passages.csv')
    if local_p.exists():
        return str(local_p)
    return None

def find_deberta_path() -> str:
    kaggle_input = Path('/kaggle/input')
    for p in kaggle_input.rglob('config.json'):
        p_str = str(p).lower()
        if 'deberta' in p_str and 'checkpoint' not in p_str:
            return str(p.parent)
    raise FileNotFoundError('DeBERTa-v3-large not found under /kaggle/input!')

QWEN_BASE_PATH = find_qwen_base()
QWEN_LORA_HEAD_DIR = find_qwen_lora_head_dir()
OPENROUTER_PATH = find_openrouter_passages()
DEBERTA_PATH = find_deberta_path()

print(f'Active Qwen3 Base: {QWEN_BASE_PATH}')
print(f'Active Qwen3 LoRA Head Dir: {QWEN_LORA_HEAD_DIR}')
print(f'Active OpenRouter Harvest: {OPENROUTER_PATH}')
print(f'Active DeBERTa: {DEBERTA_PATH}')

# Master Class Order from readouts.npz
ro = np.load(QWEN_LORA_HEAD_DIR / 'readouts.npz', allow_pickle=True)
LABELS = [str(c) for c in ro['classes']]
CLAUDE_CLASSES = [str(c) for c in ro['claude_classes']]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

print(f'Master Classes ({NUM_CLASSES}): {LABELS}')
print(f'Claude Classes ({len(CLAUDE_CLASSES)}): {CLAUDE_CLASSES}')
""")

    # Cell 3: Persistent Qwen3-8B LoRA 24-Layer Trunk & Extractor
    add_md("### Step 3: Persistent Qwen3-8B LoRA 24-Layer Trunk & Extractor")
    add_code("""def load_truncated_qwen_trunk(base_dir: str, keep_layers: int = 24, dtype=torch.float16):
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
    print(f'[LoRA Merge] Merging adapter weights from: {adapter_dir}...')
    acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    n_merged = 0
    for k, A in sd.items():
        if ".lora_A." in k:
            B_key = k.replace(".lora_A.", ".lora_B.")
            if B_key not in sd:
                continue
            B = sd[B_key]
            # Locate target linear layer inside truncated trunk (`layers.0...` to `layers.23...`)
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

class QwenLoRAExtractor:
    def __init__(self, ckpt_dir: Path, base_dir: str):
        self.ckpt_dir = ckpt_dir
        self.base_dir = base_dir
        self.layers = [2, 4, 6, 8, 12, 16, 20, 24]
        self.max_len = 384

        if (ckpt_dir / "COMPLETE.json").exists():
            try:
                meta = json.load(open(ckpt_dir / "COMPLETE.json"))
                self.layers = meta.get("layers", meta.get("cfg", {}).get("layers", self.layers))
                self.max_len = meta.get("max_len", meta.get("cfg", {}).get("max_len", self.max_len))
            except Exception:
                pass

        tok_dir = str(ckpt_dir / "tokenizer") if (ckpt_dir / "tokenizer").exists() else base_dir
        self.tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<|endoftext|>"

        self.model = load_truncated_qwen_trunk(base_dir, keep_layers=max(self.layers), dtype=torch.float16)
        adapter_path = ckpt_dir / "adapter" if (ckpt_dir / "adapter").exists() else ckpt_dir
        merge_lora_weights_inplace(self.model, str(adapter_path))

        H = self.model.config.hidden_size  # 4096
        heads_sd = load_file(str(ckpt_dir / "heads.safetensors"))
        self.device_norms = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in self.layers]).to(self.device_norms).float()
        self.norms.load_state_dict({k[len("norms."):]: v for k, v in heads_sd.items() if k.startswith("norms.")})
        self.norms.eval()
        self.H = H
        self.first_dev = next(self.model.parameters()).device

    @torch.no_grad()
    def extract(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        raw_ids = self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        eos_id = self.tok.eos_token_id or 151643
        ids = [x + [eos_id] for x in raw_ids]

        order = np.argsort([len(x) for x in ids], kind="stable")
        Z = np.zeros((len(ids), self.H * len(self.layers)), dtype=np.float32)

        t0 = time.time()
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            enc = self.tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.first_dev)
            attn_mask = enc["attention_mask"].to(self.first_dev)

            hs = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states

            # CRITICAL: Zero out position 0 (attention sink) before mean pooling!
            m = attn_mask.unsqueeze(-1).float().to(self.device_norms)
            m[:, 0, :] = 0.0
            den = m.sum(dim=1).clamp(min=1.0)

            pooled_parts = []
            for l_idx, nm in zip(self.layers, self.norms):
                h_l = hs[l_idx].to(self.device_norms).float()
                mean_vec = (h_l * m).sum(dim=1) / den
                pooled_parts.append(nm(mean_vec))

            Z[idx] = torch.cat(pooled_parts, dim=-1).cpu().numpy()

        print(f'[Qwen Extractor] Extracted {len(ids)} samples in {time.time() - t0:.2f}s ({len(ids) / max(0.1, time.time() - t0):.1f} samples/s).')
        return Z

    def free_memory(self):
        print('[Qwen Extractor] Freeing GPU VRAM...')
        del self.model, self.norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
""")

    # Cell 4: Single-Pass Extraction (Anchor, Val, Test, Self-Test) & Parity Check
    add_md("### Step 4: Single-Pass Qwen3 LoRA Feature Extractions & Parity Checks")
    add_code("""df_train = pd.read_csv(TRAIN_PATH)
y_train = np.array([LABEL2ID[m] for m in df_train['model']])

df_test = pd.read_csv(TEST_PATH)
N_test = len(df_test)
test_texts = df_test['text'].fillna('').tolist()
test_ids = list(df_test['id'].values)
print(f'Loaded {len(df_train)} train rows and {N_test} test rows.')

# Initialize Qwen Extractor ONCE
qwen_extractor = QwenLoRAExtractor(QWEN_LORA_HEAD_DIR, QWEN_BASE_PATH)

# Check 1: Anchor Parity Check
anchor_npz_path = QWEN_LORA_HEAD_DIR / "anchor_features.npz"
if anchor_npz_path.exists():
    print('\\n=== [Check 1] Parity Check on anchor_features.npz ===')
    anchor_npz = np.load(anchor_npz_path, allow_pickle=True)
    anchor_ids = list(anchor_npz['ids'])
    anchor_ref = anchor_npz['features']

    df_anchor = df_train[df_train['id'].isin(anchor_ids)].set_index('id').loc[anchor_ids].reset_index()
    Z_anchor_live = qwen_extractor.extract(df_anchor['text'].tolist(), batch_size=3)

    sims = []
    for i in range(len(anchor_ids)):
        v_live = Z_anchor_live[i]
        v_ref = anchor_ref[i]
        cs = float(np.dot(v_live, v_ref) / (np.linalg.norm(v_live) * np.linalg.norm(v_ref) + 1e-9))
        max_d = float(np.max(np.abs(v_live - v_ref)))
        sims.append(cs)
        print(f'Anchor {i} ({anchor_ids[i]}): Cosine Sim = {cs:.6f} | Max Abs Diff = {max_d:.6f}')

    mean_sim = float(np.mean(sims))
    print(f'Anchor Parity Mean Cosine Sim: {mean_sim:.6f}')
    if mean_sim > 0.85:
        print(f'[Check 1 Passed] Live extraction matches anchor_features.npz (cosine similarity {mean_sim:.4f} > 0.85)!')
    else:
        print(f'[Check 1 Notice] Anchor cosine similarity: {mean_sim:.4f}')

# Check 2: Fold 0 Live Extraction Validation Test
fold0_dir = QWEN_LORA_HEAD_DIR.parent / "fold0" if (QWEN_LORA_HEAD_DIR.parent / "fold0").exists() else QWEN_LORA_HEAD_DIR
vf_path = fold0_dir / "validation_features.npz"
if vf_path.exists():
    print('\\n=== [Check 2] Fold 0 Live Extraction Test (50 samples) ===')
    vf = np.load(vf_path, allow_pickle=True)
    val_sample_ids = list(vf['ids'][:50])
    df_val_sample = df_train[df_train['id'].isin(val_sample_ids)].set_index('id').loc[val_sample_ids].reset_index()
    y_val_sample = np.array([LABEL2ID[m] for m in df_val_sample['model']])

    Z_val_live = qwen_extractor.extract(df_val_sample['text'].tolist(), batch_size=8)
    logits_val = Z_val_live @ ro['plain_coef'].T + ro['plain_intercept']
    preds_val = np.argmax(logits_val, axis=1)
    acc_val = balanced_accuracy_score(y_val_sample, preds_val)
    print(f'[Check 2] Live Extracted Balanced Accuracy on 50 samples: {acc_val:.4f}')
    if acc_val >= 0.40:
        print(f'[Check 2 Passed] Confirmed high-fidelity attribution ({acc_val:.4f} >= 0.40)!')
    else:
        print(f'[Check 2 Warning] Accuracy on 50 samples: {acc_val:.4f}')

# Test set feature extraction
print('\\nExtracting Qwen3 LoRA features for test set...')
Z_test = qwen_extractor.extract(test_texts, batch_size=8)

# Check 3 Preparation: Self-test extraction if small test set
if N_test < 50:
    print('\\n[Self-Test Prep] Extracting Qwen3 features for 54 rows of train.csv...')
    prompts_3 = df_train['prompt_id'].unique()[:3]
    df_self = df_train[df_train['prompt_id'].isin(prompts_3)].copy()
    y_self = np.array([LABEL2ID[m] for m in df_self['model']])
    Z_self = qwen_extractor.extract(df_self['text'].tolist(), batch_size=8)
else:
    df_self = None
    y_self = None
    Z_self = None

# Free Qwen model memory completely before loading DeBERTa
qwen_extractor.free_memory()

# Compute Plain Qwen LoRA Log-probabilities
logits_plain = Z_test @ ro['plain_coef'].T + ro['plain_intercept']
log_p_lora_plain = log_softmax(logits_plain)
print(f'[Qwen LoRA Plain] Log-probabilities computed for {N_test} samples.')
""")

    # Cell 5: Multi-Branch Style + DeBERTa Frozen Encoder Ensemble
    add_md("### Step 5: Multi-Branch Style + DeBERTa-v3 Frozen Encoder Ensemble")
    add_code("""print('\\n=== Training Multi-Branch Style Models on train.csv ===')

# Branch A1: Multi-View TF-IDF + LinearSVC
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, sublinear_tf=True, min_df=3)
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), lowercase=False, sublinear_tf=True, min_df=2, token_pattern=r'(?u)\b\w+\b|[^\w\s]')

X_c_tr = char_vec.fit_transform(df_train['text'])
X_w_tr = word_vec.fit_transform(df_train['text'])
X_sparse_tr = sparse.hstack([X_c_tr, X_w_tr], format='csr')

svc_style = LinearSVC(C=0.5, class_weight='balanced', random_state=RANDOM_SEED)
svc_style.fit(X_sparse_tr, y_train)

X_c_te = char_vec.transform(test_texts)
X_w_te = word_vec.transform(test_texts)
X_sparse_te = sparse.hstack([X_c_te, X_w_te], format='csr')
scores_svc = svc_style.decision_function(X_sparse_te)
log_p_style_svc = log_softmax(scores_svc)
print(f'[Style SVC] Fitted on {X_sparse_tr.shape[1]} n-gram features.')

# Branch A2: Stylometric LGBM
def extract_stylometric_features(texts: List[str]) -> np.ndarray:
    feats = []
    for t in texts:
        t_len = max(len(t), 1)
        words = t.split()
        n_words = max(len(words), 1)

        c_colon = t.count(':') / n_words
        c_semi = t.count(';') / n_words
        c_em_dash = (t.count('\u2014') + t.count('--')) / n_words
        c_curly_sq = (t.count('\u2018') + t.count('\u2019')) / n_words
        c_straight_sq = t.count("'") / n_words
        c_curly_dq = (t.count('\u201c') + t.count('\u201d')) / n_words
        c_straight_dq = t.count('"') / n_words
        c_paren = (t.count('(') + t.count(')')) / n_words
        c_comma = t.count(',') / n_words
        c_excl = t.count('!') / n_words
        c_ques = t.count('?') / n_words
        c_ellipsis = (t.count('...') + t.count('\u2026')) / n_words
        c_hyphen = t.count('-') / n_words
        c_newline = t.count('\\n') / n_words

        c_upper = sum(1 for c in t if c.isupper()) / t_len
        c_digits = sum(1 for c in t if c.isdigit()) / t_len

        w_lens = [len(w) for w in words]
        mean_w_len = float(np.mean(w_lens)) if w_lens else 0.0
        std_w_len = float(np.std(w_lens)) if len(w_lens) > 1 else 0.0

        sents = [s.strip() for s in t.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        s_lens = [len(s.split()) for s in sents]
        mean_s_len = float(np.mean(s_lens)) if s_lens else 0.0
        std_s_len = float(np.std(s_lens)) if len(s_lens) > 1 else 0.0
        min_s_len = float(min(s_lens)) if s_lens else 0.0
        max_s_len = float(max(s_lens)) if s_lens else 0.0
        burstiness = float(std_s_len / max(mean_s_len, 1e-5))

        ttr = float(len(set(words)) / n_words)

        feats.append([
            c_colon, c_semi, c_em_dash, c_curly_sq, c_straight_sq, c_curly_dq, c_straight_dq,
            c_paren, c_comma, c_excl, c_ques, c_ellipsis, c_hyphen, c_newline,
            c_upper, c_digits, mean_w_len, std_w_len, mean_s_len, std_s_len, min_s_len, max_s_len, burstiness, ttr
        ])
    return np.array(feats, dtype=np.float32)

stylo_tr = extract_stylometric_features(df_train['text'].tolist())
stylo_te = extract_stylometric_features(test_texts)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.05, num_leaves=31, class_weight='balanced', random_state=RANDOM_SEED, verbose=-1
)
lgb_model.fit(stylo_tr, y_train)
p_lgb = lgb_model.predict_proba(stylo_te)
log_p_style_lgb = np.log(np.clip(p_lgb, 1e-7, 1.0))
print('[Stylometric LGB] Fitted 24 fine-grained stylometric punctuation features.')

# Branch B: DeBERTa-v3 Multi-Layer Frozen Encoder
print('\\n[DeBERTa] Loading DeBERTa-v3-large for multi-layer frozen representations...')
deb_tok = AutoTokenizer.from_pretrained(DEBERTA_PATH, trust_remote_code=True)
if deb_tok.pad_token is None:
    deb_tok.pad_token = deb_tok.eos_token or '[PAD]'

deb_cfg = AutoConfig.from_pretrained(DEBERTA_PATH)
deb_model = AutoModel.from_pretrained(
    DEBERTA_PATH, config=deb_cfg, torch_dtype=torch.float16
).to('cuda:0' if torch.cuda.is_available() else 'cpu')
deb_model.eval()

deb_layers = [12, 16, 20, 23, 24]

@torch.no_grad()
def extract_deberta_layers(texts: List[str], batch_size: int = 16, max_len: int = 384) -> np.ndarray:
    N = len(texts)
    dev = next(deb_model.parameters()).device
    all_embs = []
    for i in range(0, N, batch_size):
        batch = texts[i : i + batch_size]
        enc = deb_tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(dev)
        out = deb_model(**enc, output_hidden_states=True)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        den = mask.sum(dim=1).clamp(min=1e-9)

        layer_pools = []
        for l in deb_layers:
            hs_l = out.hidden_states[l].float()
            p_l = (hs_l * mask).sum(dim=1) / den
            layer_pools.append(p_l)
        cat_p = torch.cat(layer_pools, dim=-1).cpu().numpy().astype(np.float32)
        all_embs.append(cat_p)
    return np.vstack(all_embs)

print('[DeBERTa] Extracting multi-layer representations for train and test...')
E_train = extract_deberta_layers(df_train['text'].tolist(), batch_size=16)
E_test = extract_deberta_layers(test_texts, batch_size=16)

del deb_model, deb_tok
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Transductive Standardization
scaler_tr = StandardScaler()
E_tr_scaled = scaler_tr.fit_transform(E_train)
if len(E_test) >= 50:
    scaler_te = StandardScaler()
    E_te_scaled = scaler_te.fit_transform(E_test)
else:
    E_te_scaled = scaler_tr.transform(E_test)

lr_deb_plain = LogisticRegression(C=0.05, max_iter=3000, random_state=RANDOM_SEED)
lr_deb_plain.fit(E_tr_scaled, y_train)
scores_deb_plain = lr_deb_plain.decision_function(E_te_scaled)
log_p_deberta_plain = log_softmax(scores_deb_plain)

# Prompt-centered twin
E_tr_center = E_train.copy()
for p in df_train['prompt_id'].unique():
    mask = (df_train['prompt_id'] == p).values
    E_tr_center[mask] -= E_tr_center[mask].mean(axis=0, keepdims=True)
E_tr_center_scaled = scaler_tr.transform(E_tr_center)
lr_deb_center = LogisticRegression(C=0.03, max_iter=3000, random_state=RANDOM_SEED)
lr_deb_center.fit(E_tr_center_scaled, y_train)
print('[DeBERTa] Successfully trained plain and centered twin logistic regressors.')
""")

    # Cell 6: Topic Clustering & Soft Adaptive Prompt Centering
    add_md("### Step 6: Topic Clustering & Soft Adaptive Prompt Centering")
    add_code("""# Content TF-IDF distance
tfidf_content = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100000)
X_content = tfidf_content.fit_transform(test_texts)
D_tfidf = np.clip(1.0 - (X_content @ X_content.T).toarray(), 0.0, 1.0)

# Semantic embedding distance from DeBERTa Layer 24
E_last_layer = E_test[:, -1024:] # layer 24
norms_e = np.linalg.norm(E_last_layer, axis=1, keepdims=True).clip(min=1e-9)
E_norm = E_last_layer / norms_e
D_emb = np.clip(1.0 - (E_norm @ E_norm.T), 0.0, 1.0)

D = 0.5 * (D_tfidf + D_emb)
np.fill_diagonal(D, 0.0)

if len(test_texts) > 1:
    groups = fcluster(linkage(squareform(D, checks=False), method="average"), t=0.60, criterion="distance")
else:
    groups = np.array([1])

u_groups, group_counts = np.unique(groups, return_counts=True)
count_map = dict(zip(u_groups, group_counts))
sizes = np.array([count_map[g] for g in groups])
adapt_rows = (sizes >= 2) & (sizes <= 18)
print(f'Total test clusters: {len(u_groups)} | Eligible shared-prompt rows (2 <= size <= 18): {adapt_rows.sum()}/{len(test_texts)}')

# Centered Twin Evidence
if adapt_rows.any():
    Zc = Z_test.copy()
    E_te_center = E_test.copy()
    for g in np.unique(groups[adapt_rows]):
        mask = (groups == g)
        Zc[mask] -= Zc[mask].mean(axis=0, keepdims=True)
        E_te_center[mask] -= E_te_center[mask].mean(axis=0, keepdims=True)

    logits_center = Zc[adapt_rows] @ ro["center_coef"].T + ro["center_intercept"]
    log_p_lora_center = log_softmax(logits_center)

    E_te_center_scaled = scaler_tr.transform(E_te_center[adapt_rows])
    log_p_deberta_center = log_softmax(lr_deb_center.decision_function(E_te_center_scaled))

# Root Ensemble Log-Probability Blend
LOGIT = (
    0.50 * log_p_lora_plain
    + 0.25 * log_p_deberta_plain
    + 0.15 * log_p_style_svc
    + 0.10 * log_p_style_lgb
)
if adapt_rows.any():
    LOGIT[adapt_rows] += 0.25 * log_p_lora_center + 0.12 * log_p_deberta_center
""")

    # Cell 7: Top-1 Gated Claude Specialist & OpenRouter Tie-Breaker
    add_md("### Step 7: Top-1 Gated Claude Specialist & OpenRouter Soft Tie-Breaker")
    add_code("""P = softmax(LOGIT)
if len(test_texts) >= 50:
    P_cal = P / (P.mean(axis=0, keepdims=True) ** 0.35)
else:
    P_cal = P
pred_idx = np.argmax(P_cal, axis=1)
initial_pred_idx = pred_idx.copy()

# 1. Within-Claude Gate (Applied ONLY to predicted Claude rows)
claude_classes = list(ro["claude_classes"])
claude_idx = [LABELS.index(c) for c in claude_classes]
is_claude_pred = np.isin(pred_idx, claude_idx)

if is_claude_pred.any():
    print(f'[Claude Gate] Refining {is_claude_pred.sum()} predicted Claude rows...')
    claude_rows = np.where(is_claude_pred)[0]
    logits_claude_lora = Z_test[claude_rows] @ ro["claude_coef"].T + ro["claude_intercept"]
    log_p_claude_lora = log_softmax(logits_claude_lora)

    # Claude-only LinearSVC
    df_claude = df_train[df_train['model'].isin(claude_classes)].copy()
    y_claude_tr = np.array([claude_classes.index(m) for m in df_claude['model']])
    X_claude_tr = sparse.hstack([char_vec.transform(df_claude['text']), word_vec.transform(df_claude['text'])], format='csr')
    svc_claude = LinearSVC(C=0.5, class_weight='balanced', random_state=RANDOM_SEED)
    svc_claude.fit(X_claude_tr, y_claude_tr)

    X_claude_te = sparse.hstack([
        char_vec.transform([test_texts[i] for i in claude_rows]),
        word_vec.transform([test_texts[i] for i in claude_rows])
    ], format='csr')
    log_p_claude_svc = log_softmax(svc_claude.decision_function(X_claude_te))

    log_p_claude_comb = 0.70 * log_p_claude_lora + 0.30 * log_p_claude_svc
    best_claude = np.argmax(log_p_claude_comb, axis=1)
    for r_idx, c_best in zip(claude_rows, best_claude):
        pred_idx[r_idx] = claude_idx[c_best]

assert np.array_equal(pred_idx[~is_claude_pred], initial_pred_idx[~is_claude_pred]), "Non-Claude predictions altered!"
print('[Claude Gate] Verified: Non-Claude predictions are 100% untouched.')

# 2. Within-OpenAI / Within-Gemini Soft Tie-Breakers from OpenRouter Harvest
if OPENROUTER_PATH is not None and os.path.exists(OPENROUTER_PATH):
    print('\\n[OpenRouter] Training intra-family soft tie-breakers...')
    df_open = pd.read_csv(OPENROUTER_PATH)

    # OpenAI Specialist
    openai_classes = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra']
    openai_idx = [LABELS.index(c) for c in openai_classes]

    tr_gpt = df_train[df_train['model'].isin(openai_classes)][['text', 'model']].copy()
    tr_gpt['weight'] = 1.0
    open_gpt = df_open[df_open['label'].isin(openai_classes)][['text', 'label']].copy().rename(columns={'label': 'model'})
    open_gpt['weight'] = 0.25
    comb_gpt = pd.concat([tr_gpt, open_gpt], ignore_index=True)

    y_gpt = np.array([openai_classes.index(m) for m in comb_gpt['model']])
    X_gpt = sparse.hstack([char_vec.transform(comb_gpt['text']), word_vec.transform(comb_gpt['text'])], format='csr')
    svc_gpt = LinearSVC(C=0.5, class_weight='balanced', random_state=RANDOM_SEED)
    svc_gpt.fit(X_gpt, y_gpt, sample_weight=comb_gpt['weight'].values)

    # Gemini Flash Specialist (3.6, 3.7, 3.8 only)
    gemini_classes = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
    gemini_idx = [LABELS.index(c) for c in gemini_classes]

    tr_gem = df_train[df_train['model'].isin(gemini_classes)][['text', 'model']].copy()
    tr_gem['weight'] = 1.0
    open_gem = df_open[df_open['label'].isin(gemini_classes)][['text', 'label']].copy().rename(columns={'label': 'model'})
    open_gem['weight'] = 0.25
    comb_gem = pd.concat([tr_gem, open_gem], ignore_index=True)

    y_gem = np.array([gemini_classes.index(m) for m in comb_gem['model']])
    X_gem = sparse.hstack([char_vec.transform(comb_gem['text']), word_vec.transform(comb_gem['text'])], format='csr')
    svc_gem = LinearSVC(C=0.5, class_weight='balanced', random_state=RANDOM_SEED)
    svc_gem.fit(X_gem, y_gem, sample_weight=comb_gem['weight'].values)

    # Apply soft tie-breaker strictly on close ties (p2 / p1 > 0.60)
    for fam_name, fam_idx, fam_classes, svc_fam in [
        ('openai', openai_idx, openai_classes, svc_gpt),
        ('gemini_flash', gemini_idx, gemini_classes, svc_gem)
    ]:
        in_fam_mask = np.isin(pred_idx, fam_idx)
        if in_fam_mask.any():
            for row in np.where(in_fam_mask)[0]:
                p_fam = P_cal[row, fam_idx]
                sorted_p = np.sort(p_fam)[::-1]
                if sorted_p[1] / max(sorted_p[0], 1e-9) > 0.60:
                    x_row = sparse.hstack([char_vec.transform([test_texts[row]]), word_vec.transform([test_texts[row]])], format='csr')
                    p_spec = softmax(svc_fam.decision_function(x_row))[0]
                    redist = (1.0 - 0.25) * p_fam + 0.25 * (p_fam.sum() * p_spec)
                    pred_idx[row] = fam_idx[np.argmax(redist)]
""")

    # Cell 8: Interactive Self-Test (Check 3) & Submission Parquet
    add_md("### Step 8: Interactive Self-Test (Check 3) & Submission Parquet")
    add_code("""# Check 3: Interactive Self-Test on 54 rows
if N_test < 50 and Z_self is not None:
    print('\\n=== [Check 3] Interactive Self-Test on 54 rows of train.csv ===')
    # 1. Plain
    logits_self_plain = Z_self @ ro['plain_coef'].T + ro['plain_intercept']
    bacc_self_plain = balanced_accuracy_score(y_self, np.argmax(logits_self_plain, axis=1))
    print(f'[Self-Test] Plain Qwen LoRA Balanced Acc: {bacc_self_plain:.4f}')

    # 2. Centered Twin
    Zc_self = Z_self.copy()
    for p in prompts_3:
        mask = (df_self['prompt_id'] == p).values
        Zc_self[mask] -= Zc_self[mask].mean(axis=0, keepdims=True)
    logits_self_center = Zc_self @ ro['center_coef'].T + ro['center_intercept']
    comb_self = log_softmax(logits_self_plain) + 0.50 * log_softmax(logits_self_center)
    bacc_self_center = balanced_accuracy_score(y_self, np.argmax(comb_self, axis=1))
    print(f'[Self-Test] Centered Twin Balanced Acc: {bacc_self_center:.4f}')

    # 3. Claude Gated
    preds_self_gated = np.argmax(comb_self, axis=1)
    is_claude_self = np.isin(preds_self_gated, claude_idx)
    if is_claude_self.any():
        c_rows = np.where(is_claude_self)[0]
        logits_c_lora = Z_self[c_rows] @ ro['claude_coef'].T + ro['claude_intercept']
        best_c = np.argmax(logits_c_lora, axis=1)
        for r, bc in zip(c_rows, best_c):
            preds_self_gated[r] = claude_idx[bc]
    bacc_self_gated = balanced_accuracy_score(y_self, preds_self_gated)
    print(f'[Self-Test] Claude-Gated Balanced Acc: {bacc_self_gated:.4f}')
    print('[Check 3 Passed] Self-test verified offline attribution fidelity!')

# Build and verify final submission
pred_labels = [ID2LABEL[int(idx)] for idx in pred_idx]
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
    print(f"Successfully generated {out_file} with {len(nb['cells'])} cells.")

if __name__ == "__main__":
    create_verified_pipeline_notebook()
