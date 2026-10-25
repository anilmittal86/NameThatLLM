"""
Module 1: train_lora.py (Step 2 - Fine-Tuning Sequence Classifier)

Offline Sequence Classification fine-tuning with PEFT LoRA on a base causal LLM
(default: Qwen/Qwen2.5-7B or local weights in /kaggle/input/...).
Features:
- Canonical 18-class label mapping.
- GroupKFold(n_splits=5) partitioned strictly by prompt_id.
- PEFT LoRA (SEQ_CLS, r=16, alpha=32, target_modules, modules_to_save=['score']).
- Gradient checkpointing (non-reentrant) + enable_input_require_grads().
- Epoch-by-epoch logging of Train Loss, Train BAcc, Val Loss, Val BAcc, and per-class recall.
- Saves best adapter checkpoint and validation OOF probabilities per fold.
"""

import argparse
import gc
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn

# Transformers & PEFT
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = None

from torch.utils.data import Dataset as TorchDataset

class TextClassificationDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lora")

# -----------------------------------------------------------------------------
# Canonical Label Mapping (18 Classes)
# -----------------------------------------------------------------------------
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

DEFAULT_TARGET_MODULES = [
    "query_proj",
    "key_proj",
    "value_proj",
    "dense",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_hardware():
    logger.info("=" * 60)
    logger.info("Hardware & Environment Diagnostics:")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    if cuda_available:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            free_vram = torch.cuda.mem_get_info(i)[0] / (1024**3)
            total_vram = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}: {props.name} | VRAM: {free_vram:.2f} GB free / {total_vram:.2f} GB total")
    else:
        logger.warning("Running on CPU. CUDA is strongly recommended for fine-tuning!")
    logger.info("=" * 60)


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)

    bacc = balanced_accuracy_score(labels, preds)
    acc = accuracy_score(labels, preds)

    # Per-class recall
    recalls = recall_score(labels, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    per_class_recalls = {f"recall_{LABELS[i]}": float(recalls[i]) for i in range(NUM_CLASSES)}

    metrics = {
        "balanced_accuracy": float(bacc),
        "accuracy": float(acc),
    }
    metrics.update(per_class_recalls)
    return metrics


class EpochLoggerCallback(TrainerCallback):
    """Callback printing clean epoch-by-epoch loss & balanced accuracy."""

    def __init__(self, train_dataset, tokenizer):
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.epoch_start_time = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = state.epoch if state.epoch is not None else 0.0
        val_loss = metrics.get("eval_loss", 0.0)
        val_bacc = metrics.get("eval_balanced_accuracy", 0.0)

        # Retrieve latest train loss logged in state.log_history
        tr_loss = 0.0
        for log in reversed(state.log_history):
            if "loss" in log:
                tr_loss = log["loss"]
                break

        elapsed = time.time() - self.epoch_start_time
        print(
            f"Epoch {epoch:.1f} ({elapsed:.1f}s) | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val BAcc: {val_bacc:.4f}"
        )


def build_training_args(output_dir: Path, epochs: int, lr: float, batch_size: int, grad_accum: int) -> TrainingArguments:
    args_dict = {
        "output_dir": str(output_dir),
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": max(1, batch_size * 2),
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
    # Compatibility check for eval_strategy vs evaluation_strategy
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        args_dict["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig.parameters:
        args_dict["evaluation_strategy"] = "epoch"

    if "warmup_ratio" in sig.parameters:
        args_dict["warmup_ratio"] = 0.1
    elif "warmup_steps" in sig.parameters:
        args_dict["warmup_steps"] = 50

    valid_params = set(sig.parameters.keys())
    filtered_args = {k: v for k, v in args_dict.items() if k in valid_params}
    return TrainingArguments(**filtered_args)


def prepare_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int = 384):
    texts = df["text"].fillna("").astype(str).tolist()
    labels = [LABEL2ID[m] for m in df["model"].tolist()]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    if HAS_DATASETS and Dataset is not None:
        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "label": labels,
        }
        return Dataset.from_dict(data)
    else:
        return TextClassificationDataset(encodings, labels)


def resolve_model_path(model_id: str) -> str:
    """Finds local Kaggle input directory or returns Hugging Face model id."""
    candidates = [
        Path(model_id),
        Path(f"/kaggle/input/{model_id}"),
        Path(f"/kaggle/input/models/{model_id}"),
        Path(f"/kaggle/input/{model_id.split('/')[-1]}"),
    ]
    for c in candidates:
        if c.exists():
            # Check for config.json
            if (c / "config.json").exists():
                return str(c)
            # Search subdirectories for config.json
            sub_configs = list(c.glob("**/config.json"))
            if sub_configs:
                return str(sub_configs[0].parent)
    return model_id


def train_single_fold(
    fold: int,
    df_train: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_id: str,
    output_dir: Path,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_length: int = 384,
    extra_data_paths: Optional[List[str]] = None,
) -> Tuple[float, np.ndarray]:
    """Trains a LoRA sequence classification model on a single GroupKFold split."""
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_adapter_dir = fold_dir / "best_adapter"

    logger.info(f"\n{'='*70}\n[FOLD {fold}] Initiating Training (Train samples: {len(train_idx)}, Val samples: {len(val_idx)})\n{'='*70}")

    resolved_path = resolve_model_path(model_id)
    logger.info(f"Loading Base Model and Tokenizer from: {resolved_path}")

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    tokenizer.padding_side = "right"

    # Training data preparation
    df_tr = df_train.iloc[train_idx].copy()
    if extra_data_paths:
        for ep in extra_data_paths:
            if Path(ep).exists():
                extra_df = pd.read_csv(ep)
                valid_extra = extra_df[extra_df["model"].isin(LABELS)].copy()
                logger.info(f"Appending {len(valid_extra)} extra external passages to fold {fold} training set.")
                df_tr = pd.concat([df_tr, valid_extra], ignore_index=True)

    df_val = df_train.iloc[val_idx].copy()

    train_ds = prepare_hf_dataset(df_tr, tokenizer, max_length=max_length)
    val_ds = prepare_hf_dataset(df_val, tokenizer, max_length=max_length)

    # Base Model setup: keep trainable LoRA weights in float32 so GradScaler can unscale gradients during fp16 training
    torch_dtype = torch.float32
    config = AutoConfig.from_pretrained(
        resolved_path,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    config.pad_token_id = tokenizer.pad_token_id

    device_map = "auto" if torch.cuda.device_count() > 1 else None
    base_model = AutoModelForSequenceClassification.from_pretrained(
        resolved_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if torch.cuda.is_available() and device_map is None:
        base_model.to("cuda")

    # Enable gradient checkpointing and require input gradients for PEFT
    try:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_model.enable_input_require_grads()
    except Exception as e:
        logger.warning(f"Notice on gradient checkpointing configuration: {e}")

    # Inspect model architecture to set appropriate target modules
    named_modules = [n for n, _ in base_model.named_modules()]
    target_modules = [m for m in DEFAULT_TARGET_MODULES if any(m in nm for nm in named_modules)]
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]  # fallback to standard attention projections

    save_candidates = ["score", "classifier", "pooler"]
    modules_to_save = [c for c in save_candidates if any(c == nm.split(".")[-1] for nm in named_modules)]
    if not modules_to_save:
        modules_to_save = ["classifier"] if "deberta" in resolved_path.lower() else ["score"]

    logger.info(f"LoRA Target Modules: {target_modules} | Modules to Save: {modules_to_save}")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments & Trainer
    training_args = build_training_args(
        output_dir=fold_dir / "checkpoints",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
        "compute_metrics": compute_metrics_fn,
        "callbacks": [EpochLoggerCallback(train_ds, tokenizer)],
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # Train
    train_result = trainer.train()

    # Final evaluation on validation fold
    eval_metrics = trainer.evaluate()
    val_bacc = eval_metrics.get("eval_balanced_accuracy", 0.0)
    logger.info(f"[FOLD {fold} COMPLETE] Best Validation Balanced Accuracy: {val_bacc:.4f}")

    # Save best LoRA adapter and tokenizer
    logger.info(f"Saving best adapter checkpoint to: {best_adapter_dir}")
    model.save_pretrained(str(best_adapter_dir))
    tokenizer.save_pretrained(str(best_adapter_dir))

    # Predict validation probabilities
    preds_output = trainer.predict(val_ds)
    val_logits = preds_output.predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    val_probs = torch.softmax(torch.tensor(val_logits, dtype=torch.float32), dim=-1).numpy().astype(np.float64)

    # Save OOF probabilities
    oof_path = fold_dir / "oof_probs.npy"
    np.save(oof_path, val_probs)
    logger.info(f"Saved fold OOF probabilities to: {oof_path} (shape: {val_probs.shape})")

    # Cleanup VRAM
    del model, base_model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_bacc, val_probs


def main():
    parser = argparse.ArgumentParser(description="Train Sequence Classifier with PEFT LoRA (GroupKFold by prompt_id)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B", help="Base model ID or local path")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 4)")
    parser.add_argument("--all_folds", action="store_true", help="Train all 5 folds sequentially")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=384, help="Max sequence length")
    parser.add_argument("--extra_data_paths", nargs="*", default=None, help="Paths to extra train passages")
    args = parser.parse_args()

    seed_everything(42)
    verify_hardware()

    if not HAS_PEFT:
        raise ImportError(
            "The 'peft' package is required for LoRA training.\n"
            "Please install it using: pip install peft\n"
            "(On Kaggle GPUs, 'peft' is already pre-installed)."
        )

    train_path = Path(args.train_csv)
    if not train_path.exists():
        train_path = Path(f"/kaggle/input/competitions/name-that-llm/{args.train_csv}")
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find training CSV at: {args.train_csv}")

    df_train = pd.read_csv(train_path)
    logger.info(f"Loaded {len(df_train)} training rows across {df_train['prompt_id'].nunique()} unique prompts.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5-fold GroupKFold by prompt_id
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df_train, groups=df_train["prompt_id"].values))

    folds_to_run = list(range(5)) if args.all_folds else [args.fold]
    fold_scores = {}

    for f in folds_to_run:
        tr_idx, val_idx = splits[f]
        bacc, _ = train_single_fold(
            fold=f,
            df_train=df_train,
            train_idx=tr_idx,
            val_idx=val_idx,
            model_id=args.model_id,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
            extra_data_paths=args.extra_data_paths,
        )
        fold_scores[f] = bacc

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING SUMMARY ACROSS FOLDS:")
    for f, score in fold_scores.items():
        logger.info(f"  Fold {f}: Balanced Accuracy = {score:.4f}")
    if len(fold_scores) > 1:
        logger.info(f"  Mean BAcc: {np.mean(list(fold_scores.values())):.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
