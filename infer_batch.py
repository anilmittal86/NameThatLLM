"""
Module 2: infer_batch.py (Step 3 - Fast FP16 Length-Sorted Batched Inference)

High-throughput inference module using length-sorted dynamic batching in torch.float16
to score validation and test passages on 2xT4 GPUs without OOM or redundant padding.

Exposes:
- load_fold_model(base_model_id: str, adapter_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]
- predict_probabilities(df: pd.DataFrame, model, tokenizer, batch_size: int = 16) -> np.ndarray
- predict_ensemble_folds(df: pd.DataFrame, base_model_id: str, adapter_paths: List[str], batch_size: int = 16) -> np.ndarray
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    PeftModel = None
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("infer_batch")

LABELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
NUM_CLASSES = len(LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def resolve_path(path_str: str) -> str:
    """Resolves local paths or Kaggle input mount directories."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    candidates = [
        Path(f"/kaggle/input/{path_str}"),
        Path(f"/kaggle/input/models/{path_str}"),
        Path(f"/kaggle/input/{path_str.split('/')[-1]}"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return path_str


def load_fold_model(base_model_id: str, adapter_path: str) -> Tuple[nn.Module, AutoTokenizer]:
    """Loads base sequence classification model with fine-tuned LoRA adapter."""
    resolved_base = resolve_path(base_model_id)
    resolved_adapter = resolve_path(adapter_path)

    logger.info(f"Loading Base Model: {resolved_base}")
    logger.info(f"Loading LoRA Adapter: {resolved_adapter}")

    # Load tokenizer from adapter directory or base
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_adapter, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(resolved_base, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    tokenizer.padding_side = "right"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    config = AutoConfig.from_pretrained(
        resolved_base,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    config.pad_token_id = tokenizer.pad_token_id

    device_map = "auto" if torch.cuda.device_count() > 1 else None
    base_model = AutoModelForSequenceClassification.from_pretrained(
        resolved_base,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if torch.cuda.is_available() and device_map is None:
        base_model.to("cuda")

    # Load LoRA adapter
    if not HAS_PEFT:
        raise ImportError(
            "The 'peft' package is required to load LoRA adapters. "
            "Please install it with 'pip install peft' or run in Kaggle where peft is preinstalled."
        )
    model = PeftModel.from_pretrained(base_model, resolved_adapter)
    model.eval()

    logger.info("Successfully loaded model and LoRA adapter weights.")
    return model, tokenizer


def predict_probabilities(
    df: pd.DataFrame,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 384,
) -> np.ndarray:
    """Computes predicted probabilities using length-sorted dynamic batching.

    Returns:
        np.ndarray: Probability matrix of shape (len(df), 18) with dtype float64,
                    zero NaNs, and all rows strictly summing to 1.0.
    """
    N = len(df)
    if N == 0:
        return np.empty((0, NUM_CLASSES), dtype=np.float64)

    texts = df["text"].fillna("").astype(str).tolist()

    # 1. Compute text lengths and argsort descending (longest first to verify peak VRAM on batch #1)
    lengths = [len(t) for t in texts]
    sorted_indices = np.argsort(lengths)[::-1]
    sorted_texts = [texts[idx] for idx in sorted_indices]

    all_sorted_probs = [None] * N
    t0 = time.time()

    device = next(model.parameters()).device
    use_cuda = torch.cuda.is_available() and device.type == "cuda"

    logger.info(f"Scoring {N} passages with length-sorted dynamic batching (batch_size={batch_size}, device={device})...")

    # 2. Process in dynamic batches
    for start_idx in range(0, N, batch_size):
        batch_slice = sorted_texts[start_idx : start_idx + batch_size]
        batch_encoded = tokenizer(
            batch_slice,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = batch_encoded["input_ids"].to(device)
        attention_mask = batch_encoded["attention_mask"].to(device)

        with torch.no_grad():
            if use_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            # Upcast logits to float32 before softmax to prevent fp16 underflow/NaNs
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)

        for i, p in enumerate(probs):
            all_sorted_probs[start_idx + i] = p

    # 3. Inverse-permute the output array back to the exact original DataFrame order
    ordered_probs = np.zeros((N, NUM_CLASSES), dtype=np.float64)
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        ordered_probs[orig_idx] = all_sorted_probs[sorted_pos]

    # Verify no NaNs and normalize row sums strictly to 1.0
    assert not np.isnan(ordered_probs).any(), "FATAL: Predicted probabilities contain NaN!"
    row_sums = ordered_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    ordered_probs = ordered_probs / row_sums

    elapsed = time.time() - t0
    logger.info(f"Inference completed for {N} samples in {elapsed:.2f}s ({N / max(1e-3, elapsed):.1f} samples/s).")
    return ordered_probs


def predict_ensemble_folds(
    df: pd.DataFrame,
    base_model_id: str,
    adapter_paths: List[str],
    batch_size: int = 16,
    max_length: int = 384,
) -> np.ndarray:
    """Sequentially loads each fold adapter, predicts probabilities, and averages ensemble outputs.

    Frees GPU VRAM cleanly between fold evaluations.
    """
    N = len(df)
    n_folds = len(adapter_paths)
    if n_folds == 0:
        raise ValueError("No adapter paths provided for ensemble prediction!")

    accumulated_probs = np.zeros((N, NUM_CLASSES), dtype=np.float64)

    for i, adapter_path in enumerate(adapter_paths):
        logger.info(f"--- [Ensemble Fold {i+1}/{n_folds}] Processing: {adapter_path} ---")
        model, tokenizer = load_fold_model(base_model_id, adapter_path)
        fold_probs = predict_probabilities(df, model, tokenizer, batch_size=batch_size, max_length=max_length)
        accumulated_probs += fold_probs

        # Explicitly clean up GPU VRAM before next fold
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_probs = accumulated_probs / float(n_folds)
    ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
    return ensemble_probs
