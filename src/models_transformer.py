"""
Transformer feature extraction and inference module for Name That LLM.
Implements:
1. Multi-layer frozen hidden state extraction (e.g. DeBERTa-v3 / Qwen) in fp16.
2. Hardware-tailored inference for Kaggle T4 GPUs (SM 7.5):
   - Strict torch.float16 with fp32 fallback on non-finite outputs.
   - Length-sorted batching (descending by text length) to minimize padding waste.
3. Offline-resilient loading for offline Kaggle environments.
"""

import gc
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.config import DEVICE, RANDOM_SEED, USE_FP16


class TextDataset(Dataset):
    """Simple Dataset holding sorted texts with their original indices."""

    def __init__(self, texts: List[str], indices: List[int]):
        self.texts = texts
        self.indices = indices

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.indices[idx]


class MultiLayerTransformerExtractor:
    """Extracts multi-layer hidden state representations from a transformer backbone.

    Adheres strictly to Kaggle T4 GPU constraints:
    - Runs in torch.float16 when CUDA is available.
    - Uses length-sorted batching to maximize throughput and minimize padding.
    - NaN guard: falls back to float32 if non-finite logits/features occur.
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-base",
        num_layers: int = 4,
        max_length: int = 512,
        batch_size: int = 16,
        device: torch.device = DEVICE,
        use_fp16: bool = USE_FP16,
    ):
        self.model_name_or_path = model_name_or_path
        self.num_layers = num_layers
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.hidden_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    def _resolve_model_path(self, model_name: str) -> str:
        """Checks local paths, Kaggle input directories, or returns the huggingface hub name."""
        kaggle_candidate = Path(f"/kaggle/input/{model_name.split('/')[-1]}")
        if kaggle_candidate.exists():
            return str(kaggle_candidate)

        kaggle_cand2 = Path(f"/kaggle/input/competitions/name-that-llm/{model_name.split('/')[-1]}")
        if kaggle_cand2.exists():
            return str(kaggle_cand2)

        local_candidate = Path(model_name)
        if local_candidate.exists():
            return str(local_candidate)

        return model_name

    def load_model(self) -> bool:
        """Loads model and tokenizer, configuring fp16 and frozen weights."""
        resolved_path = self._resolve_model_path(self.model_name_or_path)
        try:
            config = AutoConfig.from_pretrained(
                resolved_path,
                output_hidden_states=True,
                local_files_only=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                use_fast=True,
                local_files_only=False,
            )

            # Choose dtype: fp16 on CUDA (T4 GPU), fp32 on CPU
            dtype = torch.float16 if self.use_fp16 and self.device.type == "cuda" else torch.float32

            self.model = AutoModel.from_pretrained(
                resolved_path,
                config=config,
                torch_dtype=dtype,
                local_files_only=False,
            )
            self.model.eval()
            self.model.to(self.device)

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            self.hidden_dim = config.hidden_size
            # Concatenating mean-pooled last `num_layers` hidden states
            self.output_dim = self.hidden_dim * self.num_layers
            return True

        except Exception as e:
            print(f"[TransformerExtractor] Notice: Could not load '{resolved_path}' ({e}).")
            return False

    def _pool_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pools multi-layer hidden states using attention mask (excluding pad tokens)."""
        # Take the last `num_layers` hidden layers
        layers_to_pool = hidden_states[-self.num_layers :]
        pooled_layers = []

        mask_expanded = attention_mask.unsqueeze(-1).expand_as(layers_to_pool[0]).float()
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)

        for layer in layers_to_pool:
            layer_float = layer.float()
            # Mean pooling over non-padded tokens
            sum_embeddings = torch.sum(layer_float * mask_expanded, dim=1)
            mean_pooled = sum_embeddings / mask_sum
            pooled_layers.append(mean_pooled)

        # Concatenate multi-layer pooled representations: shape (B, hidden_dim * num_layers)
        return torch.cat(pooled_layers, dim=-1)

    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extracts multi-layer embeddings for texts with length-sorting & NaN guard."""
        if self.model is None or self.tokenizer is None:
            loaded = self.load_model()
            if not loaded:
                raise RuntimeError("Model could not be loaded and no offline cache is available.")

        n_samples = len(texts)
        if n_samples == 0:
            return np.empty((0, self.output_dim or 768), dtype=np.float32)

        # ---------------------------------------------------------------------
        # 1. Length-Sorted Batching: Sort by descending text length
        # ---------------------------------------------------------------------
        text_lens = [len(t) for t in texts]
        sorted_indices = np.argsort(text_lens)[::-1]
        sorted_texts = [texts[i] for i in sorted_indices]

        all_embeddings = [None] * n_samples

        for start_idx in range(0, n_samples, self.batch_size):
            batch_texts = sorted_texts[start_idx : start_idx + self.batch_size]
            batch_orig_indices = sorted_indices[start_idx : start_idx + self.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    pooled = self._pool_hidden_states(outputs.hidden_states, attention_mask)

                    # NaN guard: Check for non-finite values
                    if not torch.isfinite(pooled).all():
                        print("[TransformerExtractor] Warning: Non-finite values detected in batch. Falling back to fp32.")
                        # Fallback to float32
                        model_fp32 = self.model.float()
                        outputs = model_fp32(input_ids=input_ids, attention_mask=attention_mask)
                        pooled = self._pool_hidden_states(outputs.hidden_states, attention_mask)
                        if self.use_fp16 and self.device.type == "cuda":
                            self.model.half()

                    batch_embs = pooled.cpu().numpy().astype(np.float32)

                except torch.cuda.OutOfMemoryError:
                    print("[TransformerExtractor] OOM during batch inference. Clearing CUDA cache and retrying in smaller batches...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Process sequentially
                    batch_embs_list = []
                    for t in batch_texts:
                        enc_single = self.tokenizer([t], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                        out_single = self.model(
                            input_ids=enc_single["input_ids"].to(self.device),
                            attention_mask=enc_single["attention_mask"].to(self.device),
                        )
                        p_single = self._pool_hidden_states(out_single.hidden_states, enc_single["attention_mask"].to(self.device))
                        batch_embs_list.append(p_single.cpu().numpy().astype(np.float32))
                    batch_embs = np.vstack(batch_embs_list)

            # Store in original positions
            for orig_pos, emb in zip(batch_orig_indices, batch_embs):
                all_embeddings[orig_pos] = emb

        result = np.vstack(all_embeddings)
        assert result.shape[0] == n_samples, "Mismatch in extracted embedding count"
        return result


class OfflineSemanticProjector:
    """Offline semantic feature generator.

    Provides high-dimensional dense representations using TruncatedSVD on TF-IDF
    when heavy transformer weights are not available offline.
    """

    def __init__(self, n_components: int = 256, random_state: int = RANDOM_SEED):
        from sklearn.decomposition import TruncatedSVD
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.is_fitted = False

    def fit(self, X_sparse):
        max_comps = min(self.n_components, max(1, X_sparse.shape[0] - 1), X_sparse.shape[1])
        if self.svd.n_components != max_comps:
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=max_comps, random_state=self.random_state)
        self.svd.fit(X_sparse)
        self.is_fitted = True
        return self

    def transform(self, X_sparse) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("OfflineSemanticProjector is not fitted.")
        emb = self.svd.transform(X_sparse)
        # Normalize to unit sphere for cosine clustering
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (emb / norms).astype(np.float32)

    def fit_transform(self, X_sparse) -> np.ndarray:
        return self.fit(X_sparse).transform(X_sparse)


def load_saved_embeddings(split: str = "train", artifacts_dir: Path = None) -> Optional[np.ndarray]:
    """Loads pre-extracted transformer embeddings if available from Colab or offline run."""
    if artifacts_dir is None:
        from src.config import ARTIFACTS_DIR
        artifacts_dir = ARTIFACTS_DIR

    candidate_names = [
        f"{split}_transformer_embeddings.npy",
        f"{split}_deberta_v3_embeddings.npy",
        f"{split}_embeddings.npy",
    ]
    for name in candidate_names:
        p = Path(artifacts_dir) / name
        if p.exists():
            try:
                data = np.load(p)
                print(f"[Transformer] Found pre-extracted {split} embeddings: {p} (shape {data.shape})")
                return data
            except Exception as e:
                print(f"[Transformer] Warning: Failed to load {p}: {e}")
    return None


if __name__ == "__main__":
    print("Testing MultiLayerTransformerExtractor & OfflineSemanticProjector...")
    projector = OfflineSemanticProjector(n_components=16)
    from scipy.sparse import csr_matrix
    dummy_sparse = csr_matrix(np.random.randn(30, 200).astype(np.float32))
    proj_feats = projector.fit_transform(dummy_sparse)
    print(f"Projector output shape: {proj_feats.shape}")
    assert proj_feats.shape == (30, 16), f"Projector dimension mismatch: {proj_feats.shape}"
    assert np.allclose(np.linalg.norm(proj_feats, axis=1), 1.0, atol=1e-4), "Projected feats must be unit-norm"
    print("OfflineSemanticProjector verified successfully!")
