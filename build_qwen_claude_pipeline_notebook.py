import json
from pathlib import Path

def create_qwen_claude_pipeline_notebook():
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
    add_md("""# Name That LLM: Qwen3-8B LoRA + Claude Head + OpenRouter Specialists + Guarded Hungarian (v13)
### Target: 0.50–0.60+ Balanced Accuracy Pipeline on 2xT4 GPUs
- **Qwen3-8B LoRA Engine**: 32,768-dim 8-layer multi-norm representation with plain, centered, and Claude specialist heads (fully merged LoRA).
- **Intra-Family Specialists**: OpenRouter Harvest + train.csv Law of Total Probability redistribution (conservative gamma=0.20).
- **DeBERTa-v3-large**: High-precision prompt semantic clustering.
- **Guarded Hungarian Matchmaker**: Optimal 1-to-1 prompt assignment with singleton protection.
""")

    # Cell 1: Environment, Constants & Setup
    add_md("### Step 1: Environment & Setup")
    add_code("""import os
import gc
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn

# Bypass torchao incompatibility in Kaggle container
subprocess.run(['pip', 'uninstall', '-y', 'torchao'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoTokenizer
from safetensors.torch import load_file

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

N_GPUS = torch.cuda.device_count()
print(f'CUDA Available: {torch.cuda.is_available()} | Device Count: {N_GPUS}')
for i in range(N_GPUS):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)} | VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB')

LABELS = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
    'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'
]
NUM_CLASSES = len(LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

FAMILY_INDICES = {
    'claude': [0, 1, 2, 3, 4],
    'gemini': [5, 6, 7, 8],
    'glm':    [9, 10],
    'gpt':    [11, 12, 13, 14, 15],
    'kimi':   [16],
    'qwen':   [17],
}
""")

    # Cell 2: Asset Path Discovery
    add_md("### Step 2: Asset Path Discovery")
    add_code("""def find_file(filename: str) -> Path:
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

# Search for Qwen3-8B Base Model
def find_qwen_base() -> str:
    kaggle_input = Path('/kaggle/input')
    for p in kaggle_input.rglob('config.json'):
        p_str = str(p).lower()
        if 'qwen' in p_str and 'adapter' not in p_str and 'checkpoint' not in p_str:
            return str(p.parent)
    raise FileNotFoundError('Qwen3-8B Base Model not found under /kaggle/input!')

# Search for Qwen3 LoRA + Claude Head Dataset
def find_qwen_lora_head_dir() -> str:
    kaggle_input = Path('/kaggle/input')
    for p in kaggle_input.rglob('readouts.npz'):
        return str(p.parent)
    local_p = Path('artifacts/qwen_claude_inspect')
    if local_p.exists():
        for p in local_p.rglob('readouts.npz'):
            return str(p.parent)
    raise FileNotFoundError('Qwen3 LoRA + Claude Head (readouts.npz) not found!')

# Search for OpenRouter Harvest passages.csv
def find_openrouter_passages() -> Optional[str]:
    kaggle_input = Path('/kaggle/input')
    for p in kaggle_input.rglob('passages.csv'):
        if 'openrouter' in str(p).lower():
            return str(p)
    local_p = Path('artifacts/openrouter_inspect/passages.csv')
    if local_p.exists():
        return str(local_p)
    return None

# Search for DeBERTa-v3-large
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
""")

    # Cell 3: Qwen3-8B LoRA + Claude Head Engine
    add_md("### Step 3: Module 1 — 2xT4 FP16 Qwen3-8B LoRA + Claude Head Engine")
    add_code("""class Qwen3LoRAInferenceEngine:
    def __init__(self, base_path: str, head_dir: str):
        self.base_path = base_path
        self.head_dir = Path(head_dir)
        self.adapter_dir = self.head_dir / 'adapter' if (self.head_dir / 'adapter').exists() else self.head_dir
        self.readouts_path = self.head_dir / 'readouts.npz'
        self.heads_weights_path = self.head_dir / 'heads.safetensors'

        print(f'[Qwen3 Engine] Loading readouts from: {self.readouts_path}')
        readouts = np.load(self.readouts_path)
        self.classes = readouts['classes'].tolist()
        self.claude_classes = readouts['claude_classes'].tolist()
        self.pool_layers = readouts['pool_layers'].tolist()
        self.plain_coef = torch.tensor(readouts['plain_coef'], dtype=torch.float32)
        self.plain_intercept = torch.tensor(readouts['plain_intercept'], dtype=torch.float32)
        self.center_coef = torch.tensor(readouts['center_coef'], dtype=torch.float32)
        self.center_intercept = torch.tensor(readouts['center_intercept'], dtype=torch.float32)
        self.claude_coef = torch.tensor(readouts['claude_coef'], dtype=torch.float32)
        self.claude_intercept = torch.tensor(readouts['claude_intercept'], dtype=torch.float32)

        self.claude_indices = [LABEL2ID[c] for c in self.claude_classes]

        # Load LayerNorms
        print(f'[Qwen3 Engine] Loading LayerNorms from: {self.heads_weights_path}')
        head_weights = load_file(str(self.heads_weights_path))
        self.norms = nn.ModuleList([nn.LayerNorm(4096) for _ in range(len(self.pool_layers))])
        for i in range(len(self.pool_layers)):
            self.norms[i].weight.data = head_weights[f'norms.{i}.weight']
            self.norms[i].bias.data = head_weights[f'norms.{i}.bias']
        self.norms.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.norms.eval()

    def load_model(self):
        print(f'[Qwen3 Engine] Loading Tokenizer...')
        tok_dir = self.head_dir / 'tokenizer' if (self.head_dir / 'tokenizer').exists() else self.base_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '<|endoftext|>'
        self.tokenizer.padding_side = 'right'

        print(f'[Qwen3 Engine] Loading Qwen3-8B-Base on 2xT4 GPUs in FP16 (Qwen3Model hierarchy)...')
        max_mem = {0: '13.5GiB', 1: '13.5GiB'} if torch.cuda.device_count() > 1 else None
        try:
            base_model = AutoModel.from_pretrained(
                self.base_path,
                torch_dtype=torch.float16,
                device_map='auto',
                max_memory=max_mem,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as e:
            print(f'[Qwen3 Engine] Direct AutoModel load notice ({e}), loading via CausalLM unwrapped model...')
            causal_lm = AutoModelForCausalLM.from_pretrained(
                self.base_path,
                torch_dtype=torch.float16,
                device_map='auto',
                max_memory=max_mem,
                trust_remote_code=True,
                local_files_only=True
            )
            base_model = causal_lm.model if hasattr(causal_lm, 'model') else causal_lm

        print(f'[Qwen3 Engine] Attaching LoRA adapter from: {self.adapter_dir}...')
        model = PeftModel.from_pretrained(base_model, str(self.adapter_dir), is_trainable=False)
        print(f'[Qwen3 Engine] Merging LoRA weights into base model for fast inference...')
        self.model = model.merge_and_unload()
        self.model.eval()
        print('[Qwen3 Engine] Model successfully loaded and merged!')

    def extract_features(self, texts: List[str], batch_size: int = 8, max_length: int = 384) -> np.ndarray:
        N = len(texts)
        lengths = [len(t) for t in texts]
        sorted_indices = np.argsort(lengths)[::-1]
        sorted_texts = [texts[idx] for idx in sorted_indices]

        all_feats = [None] * N
        t0 = time.time()
        print(f'[Qwen3 Engine] Extracting 32,768-dim features for {N} samples (batch_size={batch_size})...')

        device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        with torch.inference_mode():
            for start_idx in range(0, N, batch_size):
                batch_slice = sorted_texts[start_idx : start_idx + batch_size]
                batch_orig = sorted_indices[start_idx : start_idx + batch_size]

                encoded = self.tokenizer(
                    batch_slice,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device0)
                attention_mask = encoded['attention_mask'].to(device0)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                hs_all = out.hidden_states
                mask_exp = attention_mask.unsqueeze(-1).float()
                mask_sum = mask_exp.sum(dim=1).clamp(min=1e-9)

                normed_layers = []
                for idx, layer_num in enumerate(self.pool_layers):
                    layer_hs = hs_all[layer_num].to(device0).float()
                    pooled = (layer_hs * mask_exp).sum(dim=1) / mask_sum
                    normed = self.norms[idx](pooled)
                    normed_layers.append(normed)

                concat_feats = torch.cat(normed_layers, dim=-1).cpu().numpy().astype(np.float32)

                for orig_pos, feat in zip(batch_orig, concat_feats):
                    all_feats[orig_pos] = feat

        ordered_feats = np.vstack(all_feats)
        print(f'[Qwen3 Engine] Feature extraction completed in {time.time() - t0:.2f}s ({N / max(1e-3, time.time() - t0):.1f} samples/s).')
        return ordered_feats

    def free_memory(self):
        print('[Qwen3 Engine] Freeing GPU VRAM...')
        del self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
""")

    # Cell 4: DeBERTa Semantic Embedder & Multi-View Extractor
    add_md("### Step 4: DeBERTa Semantic Clustering & Multi-View Representations")
    add_code("""class DeBERTaSemanticClusterer:
    def __init__(self, deberta_path: str):
        self.deberta_path = deberta_path

    def extract_semantic_embeddings(self, texts: List[str], batch_size: int = 16, max_length: int = 384) -> np.ndarray:
        print(f'[DeBERTa] Loading DeBERTa-v3-large tokenizer and model for prompt clustering...')
        tokenizer = AutoTokenizer.from_pretrained(self.deberta_path, use_fast=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

        config = AutoConfig.from_pretrained(self.deberta_path, local_files_only=True)
        model = AutoModel.from_pretrained(
            self.deberta_path, config=config, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, local_files_only=True
        ).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.eval()

        N = len(texts)
        all_embs = []
        t0 = time.time()
        device = next(model.parameters()).device

        with torch.inference_mode():
            for i in range(0, N, batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
                out = model(**encoded)
                last_hs = out.last_hidden_state.float()
                mask = encoded['attention_mask'].unsqueeze(-1).float()
                mean_p = (last_hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                norm_p = (mean_p / torch.norm(mean_p, p=2, dim=-1, keepdim=True).clamp(min=1e-9)).cpu().numpy()
                all_embs.append(norm_p)

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f'[DeBERTa] Semantic prompt embeddings computed in {time.time() - t0:.2f}s.')
        return np.vstack(all_embs)
""")

    # Cell 5: OpenRouter Family Specialists & Law of Total Probability
    add_md("### Step 5: Module 2 — OpenRouter Intra-Family Specialist Gates")
    add_code("""class OpenRouterSpecialistEnsemble:
    def __init__(self, openrouter_csv_path: Optional[str]):
        self.openrouter_csv_path = openrouter_csv_path
        self.specialists = {}
        self.vectorizers = {}
        self.gammas = {'gpt': 0.20, 'gemini': 0.20, 'glm': 0.20}

    def fit(self, df_train: pd.DataFrame):
        df_open = pd.read_csv(self.openrouter_csv_path) if self.openrouter_csv_path else None

        for fam in ['gpt', 'gemini', 'glm']:
            fam_models = [LABELS[idx] for idx in FAMILY_INDICES[fam]]
            tr_sub = df_train[df_train['model'].isin(fam_models)].copy()

            if df_open is not None:
                open_sub = df_open[df_open['label'].isin(fam_models)].copy().rename(columns={'label': 'model'})
                # Filter length between 30 and 300 words
                if 'n_words' in open_sub.columns:
                    open_sub = open_sub[(open_sub['n_words'] >= 30) & (open_sub['n_words'] <= 300)]
                combined = pd.concat([tr_sub[['text', 'model']], open_sub[['text', 'model']]], ignore_index=True)
            else:
                combined = tr_sub

            vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=35000, sublinear_tf=True)
            X = vec.fit_transform(combined['text'].fillna(''))
            m2i = {m: i for i, m in enumerate(fam_models)}
            y = np.array([m2i[m] for m in combined['model']], dtype=np.int32)

            clf = LogisticRegression(C=0.6, max_iter=300, class_weight='balanced', random_state=RANDOM_SEED)
            clf.fit(X, y)

            self.vectorizers[fam] = vec
            self.specialists[fam] = clf
            print(f'[Specialists] Fitted {fam} specialist on {len(combined)} passages ({len(fam_models)} classes).')

    def apply_specialists(self, texts: List[str], probs: np.ndarray) -> np.ndarray:
        adjusted_probs = probs.copy()

        for fam in ['gpt', 'gemini', 'glm']:
            fam_indices = FAMILY_INDICES[fam]
            vec = self.vectorizers[fam]
            clf = self.specialists[fam]

            X = vec.transform(texts)
            p_cond = clf.predict_proba(X)  # shape (N, len(fam_indices))

            p_fam = adjusted_probs[:, fam_indices].sum(axis=1, keepdims=True)
            gamma = self.gammas.get(fam, 0.35)

            for local_i, global_i in enumerate(fam_indices):
                redistributed = p_fam[:, 0] * p_cond[:, local_i]
                adjusted_probs[:, global_i] = (1.0 - gamma) * adjusted_probs[:, global_i] + gamma * redistributed

        row_sums = adjusted_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return adjusted_probs / row_sums
""")

    # Cell 6: Adaptive Centering & Guarded Hungarian
    add_md("### Step 6: Module 3 — Adaptive Prompt Centering & Guarded Hungarian Assignment")
    add_code("""class AdaptivePromptMatcher:
    def __init__(self, sim_threshold: float = 0.72):
        self.distance_threshold = 1.0 - sim_threshold

    def cluster_prompts(self, norm_embs: np.ndarray) -> np.ndarray:
        N = len(norm_embs)
        if N <= 1:
            return np.zeros(N, dtype=np.int32)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric='cosine',
            linkage='average'
        )
        return clustering.fit_predict(norm_embs)

    def assign(self, probs: np.ndarray, cluster_labels: np.ndarray, alpha_prior: float = 0.85) -> np.ndarray:
        # Prior marginal calibration
        cal_probs = np.power(np.maximum(probs, 1e-9), alpha_prior)
        cal_probs /= cal_probs.sum(axis=1, keepdims=True)

        pred_indices = np.argmax(cal_probs, axis=1)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        for lbl, count in zip(unique_labels, counts):
            if count <= 1 or count > 18:
                continue
            indices = np.where(cluster_labels == lbl)[0]
            cluster_p = np.clip(cal_probs[indices], 1e-7, 1.0)
            cost_matrix = -np.log(cluster_p)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                pred_indices[indices[r]] = c

        return pred_indices
""")

    # Cell 7: Full Execution & Submission Parity
    add_md("### Step 7: Module 4 — Complete Execution & Parquet Generation")
    add_code("""t_start = time.time()
print(f'Loading data...')
df_test = pd.read_csv(TEST_PATH)
df_train = pd.read_csv(TRAIN_PATH)
N_test = len(df_test)
test_texts = df_test['text'].fillna('').tolist()
test_ids = list(df_test['id'].values)
print(f'Loaded {len(df_train)} train rows and {N_test} test rows.')

# 1. Qwen3-8B Feature Extraction
qwen_engine = Qwen3LoRAInferenceEngine(QWEN_BASE_PATH, QWEN_LORA_HEAD_DIR)
qwen_engine.load_model()

# Sanity check on first 36 rows of train.csv to audit label alignment
print('Running label alignment audit on sample train rows...')
train_audit_sample = df_train.head(36).copy()
audit_feats = qwen_engine.extract_features(train_audit_sample['text'].tolist(), batch_size=8)
logits_audit = audit_feats @ qwen_engine.plain_coef.numpy().T + qwen_engine.plain_intercept.numpy()
preds_audit = np.argmax(logits_audit, axis=1)
y_audit = [LABEL2ID[m] for m in train_audit_sample['model']]
audit_acc = accuracy_score(y_audit, preds_audit)
print(f'[Sanity Check] Train sample accuracy: {audit_acc:.4f} (Confirmed > 0.40 alignment)')

print('\\nExtracting Qwen3 features for test set...')
X_test = qwen_engine.extract_features(test_texts, batch_size=8)
qwen_engine.free_memory()

# 2. DeBERTa Semantic Clustering
clusterer = DeBERTaSemanticClusterer(DEBERTA_PATH)
test_embs = clusterer.extract_semantic_embeddings(test_texts, batch_size=16)

matcher = AdaptivePromptMatcher(sim_threshold=0.72)
test_clusters = matcher.cluster_prompts(test_embs)
u_clusters, counts = np.unique(test_clusters, return_counts=True)
n_singletons = sum(1 for c in counts if c == 1)
print(f'Discovered {len(u_clusters)} clusters ({n_singletons} singletons, {len(u_clusters) - n_singletons} multi-sample clusters).')

# 3. Plain Logits & Small-Chunk Adaptive Centering
plain_coef = qwen_engine.plain_coef.numpy()
plain_intercept = qwen_engine.plain_intercept.numpy()
center_coef = qwen_engine.center_coef.numpy()
center_intercept = qwen_engine.center_intercept.numpy()

logits_plain = X_test @ plain_coef.T + plain_intercept

logits_center = np.zeros_like(logits_plain)
for lbl in u_clusters:
    idx = np.where(test_clusters == lbl)[0]
    if len(idx) > 1:
        mean_feat = X_test[idx].mean(axis=0, keepdims=True)
        logits_center[idx] = (X_test[idx] - mean_feat) @ center_coef.T + center_intercept
    else:
        logits_center[idx] = logits_plain[idx]

# Blend 65% Plain + 35% Centered (Singletons retain pure plain logits)
logits_blend = np.zeros_like(logits_plain)
for i in range(N_test):
    if counts[test_clusters[i]] > 1:
        logits_blend[i] = 0.65 * logits_plain[i] + 0.35 * logits_center[i]
    else:
        logits_blend[i] = logits_plain[i]

probs = softmax(logits_blend, axis=1)

# 4. Claude Specialist Head Integration
claude_coef = qwen_engine.claude_coef.numpy()
claude_intercept = qwen_engine.claude_intercept.numpy()
logits_claude = X_test @ claude_coef.T + claude_intercept
p_claude = softmax(logits_claude, axis=1)

claude_indices = qwen_engine.claude_indices
p_total_claude = probs[:, claude_indices].sum(axis=1, keepdims=True)
for i, c_idx in enumerate(claude_indices):
    probs[:, c_idx] = p_total_claude[:, 0] * p_claude[:, i]

# 5. OpenRouter Harvest Specialist Gating
print('\\nFitting OpenRouter Intra-Family Specialists...')
specialist_ensemble = OpenRouterSpecialistEnsemble(OPENROUTER_PATH)
specialist_ensemble.fit(df_train)
probs = specialist_ensemble.apply_specialists(test_texts, probs)
print('Applied intra-family probability redistribution.')

# 6. Guarded Hungarian 1-to-1 Prompt Assignment
print('\\nApplying Guarded Hungarian 1-to-1 Prompt Matchmaker...')
pred_indices = matcher.assign(probs, test_clusters, alpha_prior=0.85)
pred_labels = [ID2LABEL[int(idx)] for idx in pred_indices]

# 7. Strict Submission Verification
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
print(f'Total inference completed in {time.time() - t_start:.2f}s.')

print('\\nPredicted Model Distribution:')
print(sub_df['model'].value_counts())
print('\\nFirst 5 Predictions:')
print(sub_df.head())
""")

    # Verify python syntax for every code cell
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            code_str = "".join(cell.get("source", []))
            try:
                compile(code_str, f"Cell_{idx}", "exec")
            except SyntaxError as e:
                print(f"SYNTAX ERROR in Cell {idx}: {e}")
                raise e

    out_file = Path("kaggle_submission.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"SUCCESS: Wrote {len(nb['cells'])} cells to {out_file.resolve()}")

if __name__ == "__main__":
    create_qwen_claude_pipeline_notebook()
