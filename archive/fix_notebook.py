import json
from pathlib import Path

nb_path = Path("kaggle_submission.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: Hardware & Setup
cell1_code = '''import os
import gc
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Memory allocation guard
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

# Path resolution
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
'''

# Cell 2: Offline Transformer Feature Extractor
cell2_code = '''from transformers import AutoConfig, AutoModel, AutoTokenizer

def find_offline_model_path() -> Optional[str]:
    kaggle_input = Path('/kaggle/input')
    if not kaggle_input.exists():
        return None
    for cand in kaggle_input.glob('**/config.json'):
        parent_str = str(cand.parent)
        if 'checkpoint' not in parent_str.lower():
            print(f'Discovered offline transformer model at: {parent_str}')
            return parent_str
    return None

TRANSFORMER_PATH = find_offline_model_path()
print(f'Active Transformer Path: {TRANSFORMER_PATH}')

if Path('/kaggle/input').exists() and TRANSFORMER_PATH is None:
    dirs_found = [str(p) for p in Path('/kaggle/input').glob('*')]
    raise FileNotFoundError(
        f"CRITICAL ERROR: No offline transformer model (config.json) found under /kaggle/input!\\n"
        f"Available directories under /kaggle/input: {dirs_found}.\\n"
        f"Please verify 'model_sources' in kernel-metadata.json."
    )

class OfflineTransformerFeatureExtractor:
    """High-throughput multi-layer transformer feature extractor optimized for T4 GPU."""
    def __init__(self, model_path: Optional[str] = TRANSFORMER_PATH, batch_size: int = 16, max_length: int = 384):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.hidden_dim = None
        self.batch_size = batch_size
        self.max_length = max_length

    def load(self) -> bool:
        if self.model_path is None:
            raise FileNotFoundError("CRITICAL ERROR: model_path is None! DeBERTa cannot be loaded.")
        try:
            config = AutoConfig.from_pretrained(self.model_path, output_hidden_states=True, local_files_only=True)
            self.hidden_dim = getattr(config, 'hidden_size', 1024)
            print(f'[Transformer] Loaded config: hidden_dim={self.hidden_dim}, batch_size={self.batch_size}, max_length={self.max_length}.')

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

            dtype = torch.float16 if USE_FP16 else torch.float32
            self.model = AutoModel.from_pretrained(
                self.model_path, config=config, torch_dtype=dtype, local_files_only=True
            ).to(DEVICE)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            
            print(f'[Transformer] Successfully loaded offline model from {self.model_path} (dim {self.hidden_dim})')
            return True
        except Exception as e:
            print(f'[Transformer] FATAL ERROR: Could not load offline model from {self.model_path} ({e}).')
            raise e

    def extract(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        N = len(texts)
        if self.model is None:
            self.load()

        lengths = [len(t) for t in texts]
        sorted_indices = np.argsort(lengths)[::-1]
        sorted_texts = [texts[i] for i in sorted_indices]

        all_style = [None] * N
        all_semantic = [None] * N
        t0 = time.time()

        try:
            # Use torch.inference_mode() for maximum GPU inference speed
            with torch.inference_mode():
                for start_idx in range(0, N, self.batch_size):
                    batch_texts = sorted_texts[start_idx : start_idx + self.batch_size]
                    batch_orig = sorted_indices[start_idx : start_idx + self.batch_size]
                    encoded = self.tokenizer(
                        batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
                    )
                    
                    input_ids = encoded['input_ids'].to(DEVICE, non_blocking=True)
                    attention_mask = encoded['attention_mask'].to(DEVICE, non_blocking=True)

                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    hs = out.hidden_states
                    
                    # Compute pooling mask once outside layer loop
                    mask_exp = attention_mask.unsqueeze(-1).expand_as(hs[0]).float()
                    mask_sum = mask_exp.sum(dim=1).clamp(min=1e-9)

                    # Intermediate + late layers: mean and standard deviation pooling
                    layer_indices = [-1, -3, -5, -7] if len(hs) >= 8 else [-1, -2]
                    pooled_layers = []
                    for l in layer_indices:
                        lf = hs[l].float()
                        mean_p = torch.sum(lf * mask_exp, dim=1) / mask_sum
                        diff = (lf - mean_p.unsqueeze(1)) ** 2
                        std_p = torch.sqrt((torch.sum(diff * mask_exp, dim=1) / mask_sum).clamp(min=1e-9))
                        pooled_layers.extend([mean_p, std_p])

                    b_style = torch.cat(pooled_layers, dim=-1).cpu().numpy().astype(np.float32)
                    
                    # Unit-normalized final layer embedding for semantic clustering
                    final_m = torch.sum(hs[-1].float() * mask_exp, dim=1) / mask_sum
                    b_sem = (final_m / torch.norm(final_m, p=2, dim=-1, keepdim=True).clamp(min=1e-9)).cpu().numpy().astype(np.float32)

                    for orig_pos, s_v, sem_v in zip(batch_orig, b_style, b_sem):
                        all_style[orig_pos] = s_v
                        all_semantic[orig_pos] = sem_v

            print(f'Extracted transformer features for {N} samples in {time.time() - t0:.1f}s.')
            return np.vstack(all_style), np.vstack(all_semantic)

        except Exception as e:
            print(f'[Transformer] FATAL ERROR during feature extraction: {e}')
            raise e
'''

# Cell 3: Stylometric & TF-IDF
cell3_code = '''class StylometricExtractor(BaseEstimator):
    """Fast, robust stylometric feature extraction (punctuation, lengths, casing)."""
    def __init__(self):
        self.means = None
        self.stds = None

    def _extract_row(self, text: str) -> np.ndarray:
        n_chars = max(1, len(text))
        words = text.split()
        n_words = max(1, len(words))
        
        # Punctuation counts
        colons = text.count(':') / n_words
        semicolons = text.count(';') / n_words
        em_dashes = (text.count('\u2014') + text.count('--')) / n_words
        quotes = (text.count('"') + text.count("'") + text.count('\u201c') + text.count('\u201d')) / n_words
        hyphens = text.count('-') / n_words
        parens = (text.count('(') + text.count(')')) / n_words
        linebreaks = text.count('\\n') / n_words

        # Character characteristics
        digits = sum(1 for c in text if c.isdigit()) / n_chars
        uppers = sum(1 for c in text if c.isupper()) / n_chars

        # Word length statistics
        word_lengths = [len(w) for w in words]
        mean_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
        std_word_len = float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0

        # Sentence length statistics
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        sent_lengths = [len(s.split()) for s in sentences]
        mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

        # Type-token ratio
        ttr = len(set(words)) / n_words

        return np.array([
            colons, semicolons, em_dashes, quotes, hyphens, parens,
            digits, uppers, linebreaks,
            mean_word_len, std_word_len, mean_sent_len, std_sent_len,
            ttr, n_words, n_chars / n_words
        ], dtype=np.float32)

    def fit(self, texts: List[str]):
        raw_feats = np.array([self._extract_row(t) for t in texts], dtype=np.float32)
        self.means = np.mean(raw_feats, axis=0)
        self.stds = np.std(raw_feats, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        raw_feats = np.array([self._extract_row(t) for t in texts], dtype=np.float32)
        scaled = (raw_feats - self.means) / self.stds
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

class MultiViewFeatureExtractor:
    """Multi-view TF-IDF and stylometric feature representation."""
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
'''

# Cell 4: Within-Family Specialists with Variance Pre-filtering (No divide-by-zero warnings)
cell4_code = '''class WithinFamilySpecialist(BaseEstimator):
    """Specialist for resolving models inside a single provider family."""
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
                # Pre-filter constant columns to prevent zero-variance warnings and speed up ANOVA
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
'''

# Cell 5: Clustering & Centering
cell5_code = '''from scipy.optimize import linear_sum_assignment

class GuardedPromptAssigner:
    """Singleton-protected optimal Hungarian prompt matching."""
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
                # Singleton or oversized cluster guardrail: keep unconstrained argmax
                continue
            indices = np.where(cluster_labels == lbl)[0]
            cluster_p = np.clip(probs[indices], 1e-7, 1.0)
            cost_matrix = -np.log(cluster_p)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                pred_indices[indices[r]] = c

        return pred_indices
'''

# Cell 6: Training Full Pipeline
cell6_code = '''t_start = time.time()
print(f'Loading {TRAIN_PATH}...')
df_train = pd.read_csv(TRAIN_PATH)
print(f'Loaded {len(df_train)} rows across {df_train["prompt_id"].nunique()} unique prompts.')

train_texts = df_train['text'].fillna('').tolist()
y_train = np.array([MODEL_TO_IDX[m] for m in df_train['model']], dtype=np.int32)

# 1. Multi-view TF-IDF & stylometrics
print('Fitting MultiViewFeatureExtractor...')
fe = MultiViewFeatureExtractor()
X_train = fe.fit(train_texts).transform(train_texts)
print(f'Extracted sparse feature matrix shape: {X_train.shape}')

# 2. Root 18-class model on TF-IDF
print('Training Root Authorship Classifier on TF-IDF...')
clf_flat = LogisticRegression(C=1.2, max_iter=300, tol=1e-3, solver='lbfgs', random_state=RANDOM_SEED)
clf_flat.fit(X_train, y_train)

# 3. Within-Family Specialist Ensemble (ANOVA subspaced)
print('Training Within-Family Specialists for multi-model families (claude, gemini, glm, gpt)...')
family_ensemble = FamilySpecialistEnsemble()
family_ensemble.fit(X_train, y_train)

# 4. Offline transformer extraction & Neural Classifier Head
transformer_extractor = OfflineTransformerFeatureExtractor()
E_style_train, E_sem_train = transformer_extractor.extract(train_texts)

assert E_style_train is not None, "FATAL ERROR: E_style_train is None! DeBERTa extraction failed."

print('Training dedicated Neural Classifier Head directly on transformer hidden states...')
scaler_style = StandardScaler()
E_style_tr_scaled = scaler_style.fit_transform(E_style_train)
clf_neural = RidgeClassifier(alpha=15.0, random_state=RANDOM_SEED)
clf_neural.fit(E_style_tr_scaled, y_train)
print('Neural Classifier Head successfully trained!')

print(f'Full v2 pipeline trained in {time.time() - t_start:.2f}s!')
'''

# Cell 7: Test Inference
cell7_code = '''t_infer_start = time.time()
print(f'Loading {TEST_PATH}...')
df_test = pd.read_csv(TEST_PATH)
n_test = len(df_test)
test_ids = list(df_test['id'].values)
test_texts = df_test['text'].fillna('').tolist()
print(f'Loaded {n_test} test passages.')

# 1. TF-IDF & Stylometric transform
X_test = fe.transform(test_texts)

# 2. Root 18-class probabilities from TF-IDF
raw_probs = clf_flat.predict_proba(X_test)
base_probs = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
for i, c in enumerate(clf_flat.classes_):
    base_probs[:, c] = raw_probs[:, i]

# 3. Deep Neural Classifier predictions on GPU
assert clf_neural is not None and scaler_style is not None, "FATAL ERROR: Neural head not trained!"

print('Extracting transformer features for test passages...')
E_style_test, test_embs = transformer_extractor.extract(test_texts)
assert E_style_test is not None, "FATAL ERROR: Test transformer features are None!"

E_style_test_scaled = scaler_style.transform(E_style_test)
dec = clf_neural.decision_function(E_style_test_scaled)
p_neural = softmax(dec * 0.7, axis=1)

# Direct neural blend: 40% TF-IDF + 60% Deep Neural Representations
base_probs = 0.40 * base_probs + 0.60 * p_neural
print('Successfully blended Neural Transformer probabilities with TF-IDF baseline!')

# 4. Within-Family Specialist Gating
print('Applying within-family specialist synthesis...')
gated_probs = family_ensemble.apply_gated_blend(X_test, base_probs)

# 5. Prior-shift power scaling calibration
BEST_ALPHA = 0.85
calibrated_probs = np.power(np.maximum(gated_probs, 1e-9), BEST_ALPHA)
calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)

# 6. Semantic clustering & Guarded Hungarian Assignment
if n_test > 1:
    print('Discovering latent test prompt clusters...')
    assigner = GuardedPromptAssigner(sim_threshold=0.75)
    test_clusters = assigner.find_clusters(test_embs)
    u_clusters, counts = np.unique(test_clusters, return_counts=True)
    n_singletons = sum(1 for c in counts if c == 1)
    print(f'Found {len(u_clusters)} clusters: {n_singletons} singletons, {len(u_clusters) - n_singletons} multi-sample clusters.')

    pred_indices = assigner.assign(calibrated_probs, test_clusters)
else:
    pred_indices = np.argmax(calibrated_probs, axis=1)

pred_labels = [IDX_TO_MODEL[int(idx)] for idx in pred_indices]

# 8. Build submission DataFrame
sub_df = pd.DataFrame({
    'id': test_ids,
    'model': pred_labels
})

# Strict Verification
assert list(sub_df.columns) == ['id', 'model'], f'Invalid columns: {sub_df.columns}'
assert len(sub_df) == n_test, f'Row count mismatch: expected {n_test}, got {len(sub_df)}'
assert not sub_df.isnull().any().any(), 'Submission contains nulls/NaNs!'
invalid = set(sub_df['model']) - set(TARGET_MODELS)
assert len(invalid) == 0, f'Invalid model strings: {invalid}'

# Save submission.parquet
sub_df.to_parquet(OUTPUT_PATH, index=False)
print(f'\\n[SUCCESS] Wrote valid submission to {OUTPUT_PATH.resolve()} ({os.path.getsize(OUTPUT_PATH)} bytes)!')
print(f'Inference finished in {time.time() - t_infer_start:.2f}s.')

print('\\nClass distribution in predictions:')
print(sub_df['model'].value_counts())
print('\\nFirst 5 rows:')
print(sub_df.head())
'''

# Find cells by signature and replace with clean optimized code
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "TARGET_MODELS =" in src:
        cell["source"] = [l + "\n" for l in cell1_code.splitlines()]
    elif "def find_offline_model_path" in src:
        cell["source"] = [l + "\n" for l in cell2_code.splitlines()]
    elif "class StylometricExtractor" in src:
        cell["source"] = [l + "\n" for l in cell3_code.splitlines()]
    elif "class WithinFamilySpecialist" in src:
        cell["source"] = [l + "\n" for l in cell4_code.splitlines()]
    elif "class HighPrecisionPromptClusterer" in src:
        cell["source"] = [l + "\n" for l in cell5_code.splitlines()]
    elif "Training Root Authorship Classifier on TF-IDF" in src:
        cell["source"] = [l + "\n" for l in cell6_code.splitlines()]
    elif "t_infer_start = time.time()" in src:
        cell["source"] = [l + "\n" for l in cell7_code.splitlines()]

# Validate every single code cell for Python syntax!
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        code_str = "".join(cell.get("source", []))
        try:
            compile(code_str, f"Cell_{idx}", "exec")
        except SyntaxError as e:
            print(f"SYNTAX ERROR in Cell {idx}: {e}")
            raise e

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("SUCCESS: Full notebook optimized, cleaned, and verified with zero syntax errors.")
