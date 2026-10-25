import json
from pathlib import Path

def create_submission_notebook():
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
    add_md("""# Name That LLM: Fine-Tuned LoRA Sequence Classifier + Within-Family Specialists + Guarded Hungarian Matchmaker
### High-Precision Multi-Task Authorship Attribution Pipeline
- **DeBERTa-v3-large LoRA Classifier**: Trained specifically for 18-class LLM authorship discrimination.
- **Within-Family ANOVA Specialists**: Dedicated sub-models to untangle intra-family sibling confusions.
- **Guarded Hungarian Prompt Matchmaker**: Enforces 1-to-1 prompt assignment within semantic clusters ($\tau = 0.75$).
""")

    # Cell 1: Environment & Setup
    add_md("### Step 1: Hardware & Setup")
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
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Fix Kaggle container torchao 0.10.0 incompatibility with peft
subprocess.run(['pip', 'uninstall', '-y', 'torchao'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: PEFT not installed. Will attempt standard loading.")

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FP16 = torch.cuda.is_available()
print(f'Device: {DEVICE} (fp16: {USE_FP16})')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Total VRAM: {vram_gb:.2f} GB')

TARGET_MODELS = [
    'claude-fable-5',
    'claude-opus-4-6',
    'claude-opus-4-8',
    'claude-opus-5',
    'claude-sonnet-5',
    'gemini-3.1-pro-preview',
    'gemini-3.6-flash',
    'gemini-3.7-flash',
    'gemini-3.8-flash',
    'glm-5.3',
    'glm-5.3-flash',
    'gpt-5.5',
    'gpt-5.6-luna',
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'gpt-6-astra',
    'kimi-k3',
    'qwen3.8-max'
]
NUM_CLASSES = len(TARGET_MODELS)
MODEL_TO_IDX = {m: i for i, m in enumerate(TARGET_MODELS)}
IDX_TO_MODEL = {i: m for i, m in enumerate(TARGET_MODELS)}

PROVIDER_FAMILIES = {
    'claude': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'],
    'gemini': ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
    'glm': ['glm-5.3', 'glm-5.3-flash'],
    'gpt': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'],
    'kimi': ['kimi-k3'],
    'qwen': ['qwen3.8-max']
}
FAMILY_NAMES = sorted(list(PROVIDER_FAMILIES.keys()))
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}
FAMILY_MODEL_INDICES = {f: [MODEL_TO_IDX[m] for m in models] for f, models in PROVIDER_FAMILIES.items()}
MODEL_TO_FAMILY_IDX = {m: FAMILY_TO_IDX[f] for f, models in PROVIDER_FAMILIES.items() for m in models}

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

    # Cell 2: Offline Model & Adapter Discovery + Batched Inference
    add_md("### Step 2: Offline Model & LoRA Adapter Discovery")
    add_code("""def find_offline_base_model() -> Optional[str]:
    kaggle_input = Path('/kaggle/input')
    if not kaggle_input.exists():
        return None
    for cand in kaggle_input.glob('**/config.json'):
        cand_str = str(cand).lower()
        if 'adapter' not in cand_str and 'checkpoint' not in cand_str:
            p_str = str(cand.parent)
            print(f'Found Offline Base Model at: {p_str}')
            return p_str
    return None

def find_offline_adapter_dir() -> Optional[str]:
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        for cand in kaggle_input.rglob('adapter_config.json'):
            p_str = str(cand.parent)
            print(f'Found LoRA Adapter Directory at: {p_str}')
            return p_str
    local_p = Path('artifacts')
    if local_p.exists():
        for cand in local_p.rglob('adapter_config.json'):
            p_str = str(cand.parent)
            print(f'Found Local LoRA Adapter at: {p_str}')
            return p_str
    return None

BASE_MODEL_PATH = find_offline_base_model()
ADAPTER_PATH = find_offline_adapter_dir()
print(f'Active Base Model: {BASE_MODEL_PATH}')
print(f'Active Adapter Path: {ADAPTER_PATH}')

def load_inference_pipeline(base_path: str, adapter_path: Optional[str]):
    print(f'Loading tokenizer from: {adapter_path or base_path}')
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_path, use_fast=True, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

    config = AutoConfig.from_pretrained(
        base_path,
        num_labels=NUM_CLASSES,
        id2label=IDX_TO_MODEL,
        label2id=MODEL_TO_IDX,
        local_files_only=True
    )
    config.pad_token_id = tokenizer.pad_token_id

    dtype = torch.float16 if USE_FP16 else torch.float32
    print(f'Loading base sequence classification model ({dtype})...')
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        config=config,
        torch_dtype=dtype,
        local_files_only=True
    ).to(DEVICE)

    if adapter_path is not None and HAS_PEFT:
        print(f'Applying fine-tuned LoRA weights from {adapter_path}...')
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print('Using base model sequence classification head.')
        model = base_model

    model.eval()
    return model, tokenizer

def predict_lora_and_embeds(
    df: pd.DataFrame,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 384
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(df)
    if N == 0:
        return np.empty((0, NUM_CLASSES)), np.empty((0, 1024))

    texts = df['text'].fillna('').astype(str).tolist()
    lengths = [len(t) for t in texts]
    sorted_indices = np.argsort(lengths)[::-1]
    sorted_texts = [texts[idx] for idx in sorted_indices]

    all_probs = [None] * N
    all_embeds = [None] * N
    t0 = time.time()

    print(f'Running length-sorted batched inference on {N} samples (batch_size={batch_size})...')
    with torch.inference_mode():
        for start_idx in range(0, N, batch_size):
            batch_slice = sorted_texts[start_idx : start_idx + batch_size]
            batch_orig = sorted_indices[start_idx : start_idx + batch_size]

            encoded = tokenizer(
                batch_slice,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = encoded['attention_mask'].to(DEVICE, non_blocking=True)

            if USE_FP16:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            logits = outputs.logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)

            # Extract normalized sentence embeddings for prompt clustering (mean pooling of last hidden state)
            last_hs = outputs.hidden_states[-1].float()
            mask_exp = attention_mask.unsqueeze(-1).expand_as(last_hs).float()
            mask_sum = mask_exp.sum(dim=1).clamp(min=1e-9)
            pooled = torch.sum(last_hs * mask_exp, dim=1) / mask_sum
            normed_embs = (pooled / torch.norm(pooled, p=2, dim=-1, keepdim=True).clamp(min=1e-9)).cpu().numpy().astype(np.float32)

            for orig_pos, p_val, e_val in zip(batch_orig, probs, normed_embs):
                all_probs[orig_pos] = p_val
                all_embeds[orig_pos] = e_val

    ordered_probs = np.vstack(all_probs)
    ordered_embeds = np.vstack(all_embeds)

    # Normalize row sums strictly to 1.0
    row_sums = ordered_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    ordered_probs /= row_sums

    print(f'LoRA inference completed in {time.time() - t0:.2f}s ({N / max(1e-3, time.time() - t0):.1f} samples/s).')
    return ordered_probs, ordered_embeds
""")

    # Cell 3: Stylometrics & TF-IDF
    add_md("### Step 3: Multi-View Stylometric & TF-IDF Representations")
    add_code("""class StylometricExtractor(BaseEstimator):
    def __init__(self):
        self.scaler = StandardScaler()

    def _extract_row(self, text: str) -> np.ndarray:
        n_chars = max(1, len(text))
        words = text.split()
        n_words = max(1, len(words))

        colons = text.count(':') / n_words
        semicolons = text.count(';') / n_words
        em_dashes = (text.count('\\u2014') + text.count('--')) / n_words
        quotes = (text.count('"') + text.count("'") + text.count('\\u201c') + text.count('\\u201d')) / n_words
        hyphens = text.count('-') / n_words
        parens = (text.count('(') + text.count(')')) / n_words
        linebreaks = text.count('\\n') / n_words

        digits = sum(1 for c in text if c.isdigit()) / n_chars
        uppers = sum(1 for c in text if c.isupper()) / n_chars

        word_lengths = [len(w) for w in words]
        mean_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
        std_word_len = float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0

        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        sent_lengths = [len(s.split()) for s in sentences]
        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

        ttr = len(set(words)) / n_words

        # Log1p on unbounded counts to compress extreme outliers
        return np.array([
            colons, semicolons, em_dashes, quotes, hyphens, parens,
            digits, uppers, linebreaks,
            mean_word_len, std_word_len, mean_sent_len, std_sent_len,
            ttr, np.log1p(n_words), np.log1p(n_chars / n_words)
        ], dtype=np.float32)

    def fit(self, texts: List[str]):
        raw = np.array([self._extract_row(t) for t in texts], dtype=np.float32)
        self.scaler.fit(raw)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        raw = np.array([self._extract_row(t) for t in texts], dtype=np.float32)
        return self.scaler.transform(raw).astype(np.float32)

class MultiViewFeatureExtractor:
    def __init__(self):
        self.char_vec = TfidfVectorizer(
            analyzer='char', ngram_range=(2, 5), min_df=3, max_df=0.98, sublinear_tf=True, max_features=25000
        )
        self.char_wb_vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=15000
        )
        self.word_case_vec = TfidfVectorizer(
            analyzer='word', lowercase=False, ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=20000
        )
        self.word_lower_vec = TfidfVectorizer(
            analyzer='word', lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=15000
        )
        self.stylo = StylometricExtractor()

    def fit(self, texts: List[str]):
        self.char_vec.fit(texts)
        self.char_wb_vec.fit(texts)
        self.word_case_vec.fit(texts)
        self.word_lower_vec.fit(texts)
        self.stylo.fit(texts)
        return self

    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        X_c = self.char_vec.transform(texts)
        X_cwb = self.char_wb_vec.transform(texts)
        X_wc = self.word_case_vec.transform(texts)
        X_wl = self.word_lower_vec.transform(texts)
        X_sty = sparse.csr_matrix(self.stylo.transform(texts))
        return sparse.hstack([X_c, X_cwb, X_wc, X_wl, X_sty], format='csr')
""")

    # Cell 4: Within-Family ANOVA Specialists
    add_md("### Step 4: Within-Family ANOVA Specialists")
    add_code("""class WithinFamilySpecialist(BaseEstimator):
    def __init__(self, family_name: str, percentile: int = 60):
        self.family_name = family_name
        self.percentile = percentile
        self.model_indices = FAMILY_MODEL_INDICES[family_name]
        self.n_models = len(self.model_indices)
        self.var_filter = VarianceThreshold(threshold=0.0)
        self.selector = SelectPercentile(f_classif, percentile=self.percentile)
        self.clf = LogisticRegression(C=0.6, max_iter=300, tol=1e-3, solver='lbfgs', random_state=RANDOM_SEED)

    def fit(self, X_sparse: sparse.csr_matrix, y_model_idx: np.ndarray):
        if self.n_models <= 1:
            return self
        local_mapping = {g: i for i, g in enumerate(self.model_indices)}
        y_local = np.array([local_mapping[g] for g in y_model_idx], dtype=np.int32)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                X_var = self.var_filter.fit_transform(X_sparse)
                if X_var.shape[1] > 0:
                    X_filt = self.selector.fit_transform(X_var, y_local)
                else:
                    X_filt = None
            except Exception:
                X_filt = None

        if X_filt is None or X_filt.shape[1] == 0:
            X_filt = X_sparse
            self.selector = None
            self.var_filter = None

        self.clf.fit(X_filt, y_local)
        return self

    def predict_proba(self, X_sparse: sparse.csr_matrix) -> np.ndarray:
        if self.n_models <= 1:
            return np.ones((X_sparse.shape[0], 1), dtype=np.float32)
        if self.var_filter is not None and self.selector is not None:
            try:
                X_var = self.var_filter.transform(X_sparse)
                X_filt = self.selector.transform(X_var)
            except Exception:
                X_filt = X_sparse
        else:
            X_filt = X_sparse
        return self.clf.predict_proba(X_filt)

class FamilySpecialistEnsemble(BaseEstimator):
    def __init__(self):
        self.specialists: Dict[str, WithinFamilySpecialist] = {}
        self.gammas: Dict[str, float] = {'claude': 0.40, 'gemini': 0.45, 'glm': 0.50, 'gpt': 0.40}

    def fit(self, X_sparse: sparse.csr_matrix, y_model_idx: np.ndarray):
        y_family = np.array([MODEL_TO_FAMILY_IDX[TARGET_MODELS[i]] for i in y_model_idx])
        for fam_name, models in PROVIDER_FAMILIES.items():
            if len(models) <= 1:
                continue
            fam_idx = FAMILY_TO_IDX[fam_name]
            fam_mask = (y_family == fam_idx)
            if np.sum(fam_mask) == 0:
                continue
            spec = WithinFamilySpecialist(family_name=fam_name)
            spec.fit(X_sparse[fam_mask], y_model_idx[fam_mask])
            self.specialists[fam_name] = spec
        return self

    def apply_gated_blend(self, X_sparse: sparse.csr_matrix, base_probs: np.ndarray) -> np.ndarray:
        N = X_sparse.shape[0]
        blended = base_probs.copy()
        family_probs = np.zeros((N, len(FAMILY_NAMES)), dtype=np.float32)
        for fam_name, fam_indices in FAMILY_MODEL_INDICES.items():
            fam_idx = FAMILY_TO_IDX[fam_name]
            family_probs[:, fam_idx] = base_probs[:, fam_indices].sum(axis=1)

        for fam_name, spec in self.specialists.items():
            fam_idx = FAMILY_TO_IDX[fam_name]
            p_fam = family_probs[:, fam_idx]
            cond_p = spec.predict_proba(X_sparse)
            g = self.gammas.get(fam_name, 0.40)
            for local_i, global_i in enumerate(spec.model_indices):
                hier_p = p_fam * cond_p[:, local_i]
                blended[:, global_i] = (1.0 - g) * base_probs[:, global_i] + g * hier_p

        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (blended / row_sums).astype(np.float32)
""")

    # Cell 5: Guarded Hungarian Matchmaker
    add_md("### Step 5: Guarded Hungarian Matchmaker")
    add_code("""class GuardedPromptAssigner:
    def __init__(self, sim_threshold: float = 0.75):
        self.distance_threshold = 1.0 - sim_threshold

    def find_clusters(self, norm_embs: np.ndarray) -> np.ndarray:
        N = norm_embs.shape[0]
        if N <= 1:
            return np.zeros(N, dtype=np.int32)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold, metric='cosine', linkage='average'
        )
        return clustering.fit_predict(norm_embs)

    def assign(self, probs: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        pred_indices = np.argmax(probs, axis=1)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        for lbl, k in zip(unique_labels, counts):
            if k <= 1 or k > 18:
                continue
            indices = np.where(cluster_labels == lbl)[0]
            cluster_p = np.clip(probs[indices], 1e-7, 1.0)
            cost_matrix = -np.log(cluster_p)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                pred_indices[indices[r]] = c

        return pred_indices
""")

    # Cell 6: Train Baseline Components & Load LoRA Pipeline
    add_md("### Step 6: Train Baseline Components & Load Fine-Tuned LoRA Adapter")
    add_code("""t_start = time.time()
print(f'Loading {TRAIN_PATH}...')
df_train = pd.read_csv(TRAIN_PATH)
print(f'Loaded {len(df_train)} rows across {df_train["prompt_id"].nunique()} unique prompts.')

train_texts = df_train['text'].fillna('').tolist()
y_train = np.array([MODEL_TO_IDX[m] for m in df_train['model']], dtype=np.int32)

print('Fitting MultiViewFeatureExtractor on train passages...')
fe = MultiViewFeatureExtractor()
X_train = fe.fit(train_texts).transform(train_texts)
print(f'TF-IDF feature matrix shape: {X_train.shape}')

print('Fitting baseline Root Authorship Classifier...')
clf_flat = LogisticRegression(C=1.2, max_iter=300, tol=1e-3, solver='lbfgs', random_state=RANDOM_SEED)
clf_flat.fit(X_train, y_train)

print('Fitting Within-Family Specialists...')
family_ensemble = FamilySpecialistEnsemble()
family_ensemble.fit(X_train, y_train)

print('\\nLoading Fine-Tuned LoRA Model & Tokenizer...')
model, tokenizer = load_inference_pipeline(BASE_MODEL_PATH, ADAPTER_PATH)

print(f'Setup and preparation completed in {time.time() - t_start:.2f}s!')
""")

    # Cell 7: Test Inference, Blend & Submission Generation
    add_md("### Step 7: Test Inference, Blend & Guarded Assignment")
    add_code("""t_infer_start = time.time()
print(f'Loading {TEST_PATH}...')
df_test = pd.read_csv(TEST_PATH)
n_test = len(df_test)
test_ids = list(df_test['id'].values)
test_texts = df_test['text'].fillna('').tolist()
print(f'Loaded {n_test} test passages.')

# 1. High-Precision LoRA Inference on GPU
p_lora, test_embs = predict_lora_and_embeds(df_test, model, tokenizer, batch_size=16, max_length=384)

# 2. Baseline TF-IDF + Within-Family Specialists
X_test = fe.transform(test_texts)
raw_tfidf_probs = clf_flat.predict_proba(X_test)
base_tfidf = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
for i, c in enumerate(clf_flat.classes_):
    base_tfidf[:, c] = raw_tfidf_probs[:, i]

gated_tfidf = family_ensemble.apply_gated_blend(X_test, base_tfidf)

# 3. Blend: 45% Deep LoRA Sequence Classifier + 55% Stylometric Specialists (Empirically optimal)
p_blend = 0.45 * p_lora + 0.55 * gated_tfidf
row_sums = p_blend.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
p_blend /= row_sums
print('Successfully blended LoRA sequence probabilities with stylometric specialists!')

# 4. Calibration: Temperature / Prior Power Scaling
BEST_ALPHA = 0.85
calibrated_probs = np.power(np.maximum(p_blend, 1e-9), BEST_ALPHA)
calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)

# 5. Semantic Clustering & Guarded Hungarian Matching
if n_test > 1:
    print('Discovering test prompt clusters...')
    assigner = GuardedPromptAssigner(sim_threshold=0.75)
    test_clusters = assigner.find_clusters(test_embs)
    u_clusters, counts = np.unique(test_clusters, return_counts=True)
    n_singletons = sum(1 for c in counts if c == 1)
    print(f'Found {len(u_clusters)} clusters: {n_singletons} singletons, {len(u_clusters) - n_singletons} multi-sample clusters.')

    pred_indices = assigner.assign(calibrated_probs, test_clusters)
else:
    pred_indices = np.argmax(calibrated_probs, axis=1)

pred_labels = [IDX_TO_MODEL[int(idx)] for idx in pred_indices]

# 6. Build and verify submission DataFrame
sub_df = pd.DataFrame({
    'id': test_ids,
    'model': pred_labels
})

assert list(sub_df.columns) == ['id', 'model'], f'Invalid columns: {sub_df.columns}'
assert len(sub_df) == n_test, f'Row count mismatch: expected {n_test}, got {len(sub_df)}'
assert not sub_df.isnull().any().any(), 'Submission contains nulls/NaNs!'
invalid = set(sub_df['model']) - set(TARGET_MODELS)
assert len(invalid) == 0, f'Invalid model strings: {invalid}'

sub_df.to_parquet(OUTPUT_PATH, index=False)
print(f'\\n[SUCCESS] Wrote valid submission to {OUTPUT_PATH.resolve()} ({os.path.getsize(OUTPUT_PATH)} bytes)!')
print(f'Inference finished in {time.time() - t_infer_start:.2f}s.')

print('\\nClass distribution in predictions:')
print(sub_df['model'].value_counts())
print('\\nFirst 5 rows:')
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
    create_submission_notebook()
