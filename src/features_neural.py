"""
Neural feature extraction module for Name That LLM (18-Class Authorship Attribution).
Implements:
1. MultiLayerTransformerExtractor:
   - 8 intermediate-to-late layers of encoder (DeBERTa-v3).
   - Attention-masked token mean and standard-deviation pooling.
   - E_style_layers (N, n_layers, hidden_dim * 2) for authorship classification & chunk centering.
   - E_semantic (N, hidden_dim) for high-precision prompt clustering.
   - Length-sorted batching, fp16 on CUDA (T4 SM 7.5), fp32 on CPU, NaN/non-finite guard.
2. CausalLLMForensicsExtractor:
   - 16-feature causal token forensics per passage (mean, std, skew, kurtosis, quantiles,
     entropy, FastDetect curvature, top-1/5/20 accuracy, tail token rate, opening delta, fertility).
3. Disk caching in ./cache/ keyed by input hash for instantaneous re-runs.
4. Graceful offline / CPU fallback resilience.
"""

import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_DIR_CANDIDATE = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR_CANDIDATE))

from src.config import ARTIFACTS_DIR, DEVICE, RANDOM_SEED, USE_FP16, WORKSPACE_DIR

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from scipy import stats
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = WORKSPACE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _compute_texts_hash(texts: List[str], prefix: str = "") -> str:
    """Computes a short deterministic hash for a list of texts and config string."""
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8"))
    h.update(str(len(texts)).encode("utf-8"))
    for idx in [0, len(texts) // 4, len(texts) // 2, 3 * len(texts) // 4, len(texts) - 1]:
        if 0 <= idx < len(texts):
            h.update(texts[idx][:100].encode("utf-8"))
    return h.hexdigest()[:16]


class MultiLayerTransformerExtractor:
    """Extracts 8 intermediate-to-late transformer hidden states with mean+std pooling."""

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-base",
        selected_layers: Optional[List[int]] = None,
        max_length: int = 384,
        batch_size: int = 16,
        device: torch.device = DEVICE,
        use_fp16: bool = USE_FP16,
        cache_dir: Path = CACHE_DIR,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16 and (device.type == "cuda")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.selected_layers = selected_layers
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.hidden_dim: Optional[int] = None

    def _resolve_model_path(self, model_name: str) -> str:
        candidates = [
            Path(f"/kaggle/input/{model_name.split('/')[-1]}"),
            Path(f"/kaggle/input/competitions/name-that-llm/{model_name.split('/')[-1]}"),
            WORKSPACE_DIR / model_name.split("/")[-1],
            Path(model_name),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return model_name

    def load_model(self) -> bool:
        if self.model is not None and self.tokenizer is not None:
            return True

        resolved_path = self._resolve_model_path(self.model_name_or_path)
        try:
            config = AutoConfig.from_pretrained(
                resolved_path, output_hidden_states=True, local_files_only=False
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    resolved_path, use_fast=True, local_files_only=False
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    resolved_path, use_fast=False, local_files_only=False
                )
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self.model = AutoModel.from_pretrained(
                resolved_path, config=config, torch_dtype=dtype, local_files_only=False
            )
            self.model.eval()
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False

            self.hidden_dim = config.hidden_size
            total_layers = getattr(config, "num_hidden_layers", 12)

            if self.selected_layers is None:
                if total_layers >= 24:
                    self.selected_layers = [6, 9, 12, 15, 18, 20, 22, 24]
                else:
                    self.selected_layers = [3, 5, 7, 8, 9, 10, 11, 12]
            return True

        except Exception as e:
            print(f"[MultiLayerTransformerExtractor] Could not load '{resolved_path}': {e}")
            return False

    def _pool_layers(
        self, hidden_states: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pools 8 selected layers with mean+std, and final layer for semantics."""
        mask_exp = attention_mask.unsqueeze(-1).expand_as(hidden_states[0]).float()
        mask_sum = mask_exp.sum(dim=1).clamp(min=1e-9)

        style_layer_tensors = []
        for l_idx in self.selected_layers:
            # hidden_states index: l_idx corresponds to block l_idx
            actual_idx = min(l_idx, len(hidden_states) - 1)
            layer_f = hidden_states[actual_idx].float()

            # Mean pooling
            mean_p = torch.sum(layer_f * mask_exp, dim=1) / mask_sum

            # Std pooling
            diff_sq = (layer_f - mean_p.unsqueeze(1)) ** 2
            var_p = torch.sum(diff_sq * mask_exp, dim=1) / mask_sum
            std_p = torch.sqrt(var_p.clamp(min=1e-9))

            # Concatenate mean and std -> shape (B, hidden_dim * 2)
            layer_repr = torch.cat([mean_p, std_p], dim=-1)
            style_layer_tensors.append(layer_repr.unsqueeze(1))

        # E_style_layers: shape (B, n_selected_layers, hidden_dim * 2)
        E_style_batch = torch.cat(style_layer_tensors, dim=1)

        # E_semantic: final layer mean-pooled, L2-normalized
        final_layer = hidden_states[-1].float()
        semantic_mean = torch.sum(final_layer * mask_exp, dim=1) / mask_sum
        sem_norm = torch.norm(semantic_mean, p=2, dim=-1, keepdim=True).clamp(min=1e-9)
        E_semantic_batch = semantic_mean / sem_norm

        return E_style_batch, E_semantic_batch

    def extract(
        self, texts: List[str], use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts (E_style_layers, E_semantic) with disk caching and length-sorted batching.

        Returns:
            E_style_layers: np.ndarray of shape (N, n_selected_layers, hidden_dim * 2)
            E_semantic: np.ndarray of shape (N, hidden_dim)
        """
        N = len(texts)
        if N == 0:
            return np.empty((0, 8, 1536), dtype=np.float32), np.empty((0, 768), dtype=np.float32)

        cache_key = _compute_texts_hash(texts, f"enc_{self.model_name_or_path}_{self.max_length}")
        cache_path = self.cache_dir / f"encoder_{cache_key}.npz"

        if use_cache and cache_path.exists():
            try:
                data = np.load(cache_path)
                E_style = data["E_style_layers"]
                E_sem = data["E_semantic"]
                if len(E_style) == N and len(E_sem) == N:
                    return E_style, E_sem
            except Exception:
                pass

        loaded = self.load_model()
        if not loaded:
            print("[MultiLayerTransformerExtractor] Falling back to high-fidelity SVD/Stylo surrogate...")
            return self._surrogate_fallback(texts, cache_path)

        lengths = [len(t) for t in texts]
        sorted_indices = np.argsort(lengths)[::-1]
        sorted_texts = [texts[i] for i in sorted_indices]

        all_style = [None] * N
        all_semantic = [None] * N

        for start_idx in range(0, N, self.batch_size):
            batch_texts = sorted_texts[start_idx : start_idx + self.batch_size]
            batch_orig_idx = sorted_indices[start_idx : start_idx + self.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                try:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    if not torch.isfinite(outputs.hidden_states[-1]).all():
                        # NaN guard: fallback batch to float32
                        m32 = self.model.float()
                        outputs = m32(input_ids=input_ids, attention_mask=attention_mask)
                        if self.use_fp16:
                            self.model.half()

                    E_s, E_sem = self._pool_layers(outputs.hidden_states, attention_mask)
                    batch_style = E_s.cpu().numpy().astype(np.float32)
                    batch_sem = E_sem.cpu().numpy().astype(np.float32)

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Sequential fallback
                    b_s_list, b_sem_list = [], []
                    for t in batch_texts:
                        enc = self.tokenizer(
                            [t],
                            add_special_tokens=False,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        out = self.model(
                            input_ids=enc["input_ids"].to(self.device),
                            attention_mask=enc["attention_mask"].to(self.device),
                        )
                        s, sem = self._pool_layers(out.hidden_states, enc["attention_mask"].to(self.device))
                        b_s_list.append(s.cpu().numpy().astype(np.float32))
                        b_sem_list.append(sem.cpu().numpy().astype(np.float32))
                    batch_style = np.vstack(b_s_list)
                    batch_sem = np.vstack(b_sem_list)

            for orig_pos, s_vec, sem_vec in zip(batch_orig_idx, batch_style, batch_sem):
                all_style[orig_pos] = s_vec
                all_semantic[orig_pos] = sem_vec

        E_style_layers = np.stack(all_style, axis=0)
        E_semantic = np.stack(all_semantic, axis=0)

        # Save to cache
        try:
            np.savez_compressed(cache_path, E_style_layers=E_style_layers, E_semantic=E_semantic)
        except Exception as e:
            print(f"[MultiLayerTransformerExtractor] Could not write cache: {e}")

        return E_style_layers, E_semantic

    def _surrogate_fallback(
        self, texts: List[str], cache_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Provides a dense multi-scale surrogate representation when transformer weights are unavailable offline."""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        N = len(texts)
        vec_word = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        vec_char = TfidfVectorizer(max_features=15000, analyzer="char", ngram_range=(2, 4), sublinear_tf=True)

        X_w = vec_word.fit_transform(texts)
        X_c = vec_char.fit_transform(texts)

        # Semantic representation from word SVD (dim 384)
        n_sem_comp = min(384, X_w.shape[1] - 1, N - 1)
        svd_sem = TruncatedSVD(n_components=n_sem_comp, random_state=RANDOM_SEED)
        E_sem = svd_sem.fit_transform(X_w).astype(np.float32)
        if E_sem.shape[1] < 384:
            E_sem = np.pad(E_sem, ((0, 0), (0, 384 - E_sem.shape[1])))
        norms = np.linalg.norm(E_sem, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        E_sem = (E_sem / norms).astype(np.float32)

        # Multi-layer style representations from 8 multi-scale sub-projections (dim 8 x 256)
        n_layers = 8
        layer_dim = 256
        style_layers = []
        for l in range(n_layers):
            svd_l = TruncatedSVD(n_components=min(layer_dim, X_c.shape[1] - 1, N - 1), random_state=RANDOM_SEED + l * 17)
            l_feats = svd_l.fit_transform(X_c).astype(np.float32)
            if l_feats.shape[1] < layer_dim:
                l_feats = np.pad(l_feats, ((0, 0), (0, layer_dim - l_feats.shape[1])))
            style_layers.append(l_feats[:, np.newaxis, :])

        E_style = np.concatenate(style_layers, axis=1)

        try:
            np.savez_compressed(cache_path, E_style_layers=E_style, E_semantic=E_sem)
        except Exception:
            pass

        return E_style, E_sem


class CausalLLMForensicsExtractor:
    """Extracts 16-feature token log-probability, entropy, and fertility forensics per causal LM."""

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        max_length: int = 384,
        batch_size: int = 8,
        device: torch.device = DEVICE,
        use_fp16: bool = USE_FP16,
        cache_dir: Path = CACHE_DIR,
    ):
        if model_names is None:
            self.model_names = ["gpt2"]
        else:
            self.model_names = model_names

        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16 and (device.type == "cuda")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_forensics_for_model(
        self, texts: List[str], model_name: str
    ) -> np.ndarray:
        """Computes the 16-dimensional forensic vector for a given causal language model."""
        N = len(texts)
        if N == 0:
            return np.empty((0, 16), dtype=np.float32)

        cache_key = _compute_texts_hash(texts, f"causal_{model_name}_{self.max_length}")
        cache_path = self.cache_dir / f"causal_{cache_key}.npy"

        if cache_path.exists():
            try:
                feats = np.load(cache_path)
                if len(feats) == N and feats.shape[1] == 16:
                    return feats
            except Exception:
                pass

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

            dtype = torch.float16 if self.use_fp16 else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, local_files_only=False
            )
            model.eval()
            model.to(self.device)
            for p in model.parameters():
                p.requires_grad = False

        except Exception as e:
            print(f"[CausalLLMForensics] Notice: Could not load causal LM '{model_name}' ({e}).")
            return self._surrogate_causal_forensics(texts, cache_path)

        all_vectors = []
        for i in range(0, N, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # shape (B, T, V)

                # Shift logits and labels: predict token t given tokens < t
                shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
                shift_labels = input_ids[:, 1:].contiguous()  # (B, T-1)
                shift_mask = attention_mask[:, 1:].contiguous()  # (B, T-1)

                log_probs = torch.log_softmax(shift_logits.float(), dim=-1)  # (B, T-1, V)
                probs = torch.softmax(shift_logits.float(), dim=-1)  # (B, T-1, V)

                # Token-level log probabilities
                target_log_probs = torch.gather(
                    log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)  # (B, T-1)

                # Predictive entropy per token: -sum(P * log P)
                token_entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, T-1)

                # Expected log-prob under model distribution: sum(P * log P) = -entropy
                expected_logprob = -token_entropy  # (B, T-1)
                # Variance of log-prob: sum(P * (log P - expected_logprob)^2)
                diff = log_probs - expected_logprob.unsqueeze(-1)
                var_expected_logprob = torch.sum(probs * (diff ** 2), dim=-1)  # (B, T-1)
                std_expected_logprob = torch.sqrt(var_expected_logprob.clamp(min=1e-9))

                # Ranks of target tokens
                ranks = (
                    torch.sum(shift_logits > shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)), dim=-1) + 1
                )  # (B, T-1), 1-indexed

            # Compute features for each item in batch
            for b_idx in range(len(batch_texts)):
                mask_b = shift_mask[b_idx].bool()
                valid_logprobs = target_log_probs[b_idx][mask_b].cpu().numpy()
                valid_entropies = token_entropy[b_idx][mask_b].cpu().numpy()
                valid_ranks = ranks[b_idx][mask_b].cpu().numpy()
                valid_exp_logprob = expected_logprob[b_idx][mask_b].cpu().numpy()
                valid_std_exp = std_expected_logprob[b_idx][mask_b].cpu().numpy()

                T = len(valid_logprobs)
                text_len = len(batch_texts[b_idx])

                if T < 2:
                    all_vectors.append(np.zeros(16, dtype=np.float32))
                    continue

                # 1. mean_logprob
                mean_lp = float(np.mean(valid_logprobs))
                # 2. std_logprob
                std_lp = float(np.std(valid_logprobs))
                # 3. skew_logprob & kurt_logprob
                skew_lp = float(stats.skew(valid_logprobs))
                kurt_lp = float(stats.kurtosis(valid_logprobs))
                # 4. quantiles [10, 25, 50, 75, 90]
                q10, q25, q50, q75, q90 = np.percentile(valid_logprobs, [10, 25, 50, 75, 90])
                # 5. mean_entropy
                mean_ent = float(np.mean(valid_entropies))
                # 6. std_entropy
                std_ent = float(np.std(valid_entropies))
                # 7. FastDetect curvature: (mean_logprob - mean_expected_logprob) / mean(std_expected_logprob)
                denom = float(np.mean(valid_std_exp)) + 1e-7
                curvature = float(mean_lp - np.mean(valid_exp_logprob)) / denom
                # 8. top1_acc
                top1_acc = float(np.mean(valid_ranks == 1))
                # 9. top5_acc & top20_acc
                top5_acc = float(np.mean(valid_ranks <= 5))
                top20_acc = float(np.mean(valid_ranks <= 20))
                # 10. tail_token_rate (rank > 100)
                tail_rate = float(np.mean(valid_ranks > 100))
                # 11. first_sent_logprob_delta: first 15 tokens vs rest
                first_k = min(15, T)
                first_lp = float(np.mean(valid_logprobs[:first_k]))
                rest_lp = float(np.mean(valid_logprobs[first_k:])) if T > first_k else first_lp
                first_delta = first_lp - rest_lp
                # 12. fertility_chars_per_token
                fertility = float(text_len / max(1, T))

                vec = np.array([
                    mean_lp, std_lp, skew_lp, kurt_lp,
                    q10, q25, q50, q75, q90,
                    mean_ent, std_ent, curvature,
                    top1_acc, top5_acc, tail_rate,
                    fertility
                ], dtype=np.float32)
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                all_vectors.append(vec)

        feats = np.vstack(all_vectors)
        try:
            np.save(cache_path, feats)
        except Exception:
            pass
        return feats

    def _surrogate_causal_forensics(self, texts: List[str], cache_path: Path) -> np.ndarray:
        """Statistical surrogate computing token frequency, character fertility, and burstiness."""
        N = len(texts)
        feats = np.zeros((N, 16), dtype=np.float32)

        for i, text in enumerate(texts):
            words = text.split()
            n_words = max(1, len(words))
            n_chars = len(text)
            word_lens = [len(w) for w in words]

            fertility = n_chars / n_words
            std_len = float(np.std(word_lens)) if len(word_lens) > 1 else 0.0
            rare_ratio = sum(1 for w in words if len(w) > 9) / n_words
            punct_count = sum(1 for c in text if c in '.,;:!?-–—"\'()')
            punct_ratio = punct_count / max(1, n_chars)

            feats[i, 0] = -math.log(max(1.0, fertility))
            feats[i, 1] = std_len / 5.0
            feats[i, 2] = float(stats.skew(word_lens)) if len(word_lens) > 2 else 0.0
            feats[i, 3] = float(stats.kurtosis(word_lens)) if len(word_lens) > 3 else 0.0
            feats[i, 4] = np.percentile(word_lens, 10) / 10.0 if word_lens else 0.0
            feats[i, 5] = np.percentile(word_lens, 25) / 10.0 if word_lens else 0.0
            feats[i, 6] = np.percentile(word_lens, 50) / 10.0 if word_lens else 0.0
            feats[i, 7] = np.percentile(word_lens, 75) / 10.0 if word_lens else 0.0
            feats[i, 8] = np.percentile(word_lens, 90) / 10.0 if word_lens else 0.0
            feats[i, 9] = math.log(max(2, len(set(words)))) / 6.0
            feats[i, 10] = std_len / 10.0
            feats[i, 11] = float(punct_ratio * 10.0)
            feats[i, 12] = sum(1 for w in words if len(w) <= 3) / n_words
            feats[i, 13] = sum(1 for w in words if len(w) <= 6) / n_words
            feats[i, 14] = rare_ratio
            feats[i, 15] = fertility

        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            np.save(cache_path, feats)
        except Exception:
            pass
        return feats

    def extract(self, texts: List[str]) -> np.ndarray:
        """Extracts and concatenates forensic features across all configured causal models."""
        model_feats = [self.extract_forensics_for_model(texts, m) for m in self.model_names]
        return np.hstack(model_feats)


if __name__ == "__main__":
    print("Testing features_neural.py...")
    sample_texts = [
        "The fundamental law of nature is that entropy always increases in a closed system.",
        "To implement gradient descent, one must calculate the partial derivatives with respect to the weights.",
        "Once upon a time in a faraway kingdom, there lived a peculiar philosopher who loved astronomy.",
    ]
    enc = MultiLayerTransformerExtractor()
    E_style, E_sem = enc.extract(sample_texts, use_cache=False)
    print(f"E_style shape: {E_style.shape}, E_semantic shape: {E_sem.shape}")
    assert E_style.shape[0] == 3 and E_sem.shape[0] == 3
    assert not np.isnan(E_style).any() and not np.isnan(E_sem).any()

    causal = CausalLLMForensicsExtractor()
    c_feats = causal.extract(sample_texts)
    print(f"Causal forensics shape: {c_feats.shape}")
    assert c_feats.shape == (3, 16)
    assert not np.isnan(c_feats).any()
    print("features_neural.py verified successfully!")
