import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4"},
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    },
    "cells": [],
}


def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in text.splitlines()],
    })


def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in text.splitlines()],
    })


add_md("""# Name That LLM: LoRA Sequence Classifier Training Pipeline 
### Fine-tuning DeBERTa-v3-large with PEFT LoRA on 18-Class Authorship Attribution

**Key Highlights:**
- **GroupKFold (n_splits=5) partitioned strictly by `prompt_id`** (zero prompt leakage).
- **PEFT LoRA (TaskType.SEQ_CLS, r=16, alpha=32)** targeting attention projections + classifier head.
- **Non-reentrant gradient checkpointing** for optimal T4 GPU memory efficiency.
- **Balanced Accuracy macro-recall tracking** epoch-by-epoch.
- **Saves best adapter weights & OOF probabilities** ready for ensembling and Guarded Hungarian post-processing.
""")

add_md("### Step 1: Install Dependencies & Verify GPU Hardware")
add_code("""import os
# Pin strictly to GPU 0 to prevent PyTorch DataParallel deadlock with LoRA & gradient checkpointing
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import gc
import time
from pathlib import Path

# Uninstall incompatible pre-installed torchao (0.10.0 incompatible with PEFT), then install PEFT & Accelerate
!pip uninstall -y torchao 2>/dev/null
!pip install -q peft accelerate

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Total VRAM: {vram_gb:.2f} GB')
else:
    print('WARNING: CUDA is not available. Please enable GPU accelerator in Kaggle settings!')
""")

add_md("### Step 2: Canonical Label Mapping & Data Paths")
add_code("""LABELS = [
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
NUM_CLASSES = len(LABELS)
LABEL2ID = {m: i for i, m in enumerate(LABELS)}
ID2LABEL = {i: m for i, m in enumerate(LABELS)}

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
print(f'Train dataset path: {TRAIN_PATH}')
df_train = pd.read_csv(TRAIN_PATH)
print(f'Loaded {len(df_train)} training passages across {df_train["prompt_id"].nunique()} unique prompts.')
""")

add_md("### Step 3: Base Model Path Discovery")
add_code("""def find_base_model_path() -> str:
    candidates = [
        '/kaggle/input/models/angelndevil2/deberta-v3-large/transformers/default/1',
        '/kaggle/input/deberta-v3-large',
        'microsoft/deberta-v3-large'
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            if (p / 'config.json').exists():
                return str(p)
            sub = list(p.glob('**/config.json'))
            if sub:
                return str(sub[0].parent)
    return 'microsoft/deberta-v3-large'

MODEL_PATH = find_base_model_path()
print(f'Discovered base model path: {MODEL_PATH}')
""")

add_md("### Step 4: Dataset Tokenization & Metrics")
add_code("""from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainerCallback

class TextDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    bacc = balanced_accuracy_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {'balanced_accuracy': float(bacc), 'accuracy': float(acc)}

class EpochPrintCallback(TrainerCallback):
    def __init__(self):
        self.start_t = time.time()
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_t = time.time()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            step = state.global_step
            loss = logs['loss']
            lr = logs.get('learning_rate', 0.0)
            dt = time.time() - self.start_t
            print(f'>>> [Step {step}] Loss: {loss:.4f} | LR: {lr:.2e} | Elapsed: {dt:.1f}s')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            ep = state.epoch if state.epoch else 0.0
            dt = time.time() - self.start_t
            val_loss = metrics.get('eval_loss', 0.0)
            val_bacc = metrics.get('eval_balanced_accuracy', 0.0)
            print(f'>>> [Epoch {ep:.1f} EVAL] ({dt:.1f}s) | Val Loss: {val_loss:.4f} | Val Balanced Acc: {val_bacc:.4f}')
""")

add_md("### Step 5: LoRA Model Builder & Trainer Configuration")
add_code("""from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

def build_lora_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
        
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    config.pad_token_id = tokenizer.pad_token_id
    
    # Trainable LoRA weights must be float32 so GradScaler can unscale gradients during mixed-precision fp16
    torch_dtype = torch.float32
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype
    )
    if torch.cuda.is_available():
        base_model.to('cuda')
        
    try:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base_model.enable_input_require_grads()
    except Exception as e:
        print(f'Notice on gradient checkpointing: {e}')
        
    named_modules = [n for n, _ in base_model.named_modules()]
    target_candidates = ['query_proj', 'key_proj', 'value_proj', 'dense', 'q_proj', 'v_proj']
    target_modules = [m for m in target_candidates if any(m in nm for nm in named_modules)]
    if not target_modules:
        target_modules = ['query_proj', 'value_proj']
        
    save_candidates = ['classifier', 'pooler', 'score']
    modules_to_save = [c for c in save_candidates if any(c == nm.split('.')[-1] for nm in named_modules)]
    
    print(f'Target Modules: {target_modules} | Modules to Save: {modules_to_save}')
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        modules_to_save=modules_to_save
    )
    
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer
""")

add_md("### Step 6: Execute Fold 0 Training (GroupKFold by prompt_id)")
add_code("""from sklearn.model_selection import GroupKFold
import inspect

torch.manual_seed(42)
np.random.seed(42)

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(df_train, groups=df_train['prompt_id'].values))

FOLD = 0
train_idx, val_idx = splits[FOLD]
print(f'--- Initiating Fold {FOLD} Training ---')
print(f'Train Samples: {len(train_idx)} | Val Samples: {len(val_idx)}')

df_tr = df_train.iloc[train_idx].copy()
df_val = df_train.iloc[val_idx].copy()

model, tokenizer = build_lora_model(MODEL_PATH)

MAX_LENGTH = 384
tr_enc = tokenizer(df_tr['text'].fillna('').tolist(), truncation=True, max_length=MAX_LENGTH, padding=False)
val_enc = tokenizer(df_val['text'].fillna('').tolist(), truncation=True, max_length=MAX_LENGTH, padding=False)

tr_labels = [LABEL2ID[m] for m in df_tr['model'].tolist()]
val_labels = [LABEL2ID[m] for m in df_val['model'].tolist()]

train_dataset = TextDataset(tr_enc, tr_labels)
val_dataset = TextDataset(val_enc, val_labels)

OUTPUT_DIR = Path('/kaggle/working/deberta_lora_fold0')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args_dict = {
    'output_dir': str(OUTPUT_DIR / 'checkpoints'),
    'learning_rate': 3e-5,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 2,
    'num_train_epochs': 3,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'save_strategy': 'epoch',
    'save_total_limit': 1,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'balanced_accuracy',
    'greater_is_better': True,
    'fp16': torch.cuda.is_available(),
    'report_to': 'none',
    'seed': 42
}

sig = inspect.signature(TrainingArguments.__init__)
if 'eval_strategy' in sig.parameters:
    training_args_dict['eval_strategy'] = 'epoch'
elif 'evaluation_strategy' in sig.parameters:
    training_args_dict['evaluation_strategy'] = 'epoch'

if 'warmup_ratio' in sig.parameters:
    training_args_dict['warmup_ratio'] = 0.1
elif 'warmup_steps' in sig.parameters:
    training_args_dict['warmup_steps'] = 50

valid_params = set(sig.parameters.keys())
filtered_args = {k: v for k, v in training_args_dict.items() if k in valid_params}
training_args = TrainingArguments(**filtered_args)

trainer_kwargs = {
    'model': model,
    'args': training_args,
    'train_dataset': train_dataset,
    'eval_dataset': val_dataset,
    'data_collator': DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
    'compute_metrics': compute_metrics,
    'callbacks': [EpochPrintCallback()]
}
trainer_sig = inspect.signature(Trainer.__init__)
if 'processing_class' in trainer_sig.parameters:
    trainer_kwargs['processing_class'] = tokenizer
elif 'tokenizer' in trainer_sig.parameters:
    trainer_kwargs['tokenizer'] = tokenizer

trainer = Trainer(**trainer_kwargs)

print('Starting Trainer.train()...')
trainer.train()

eval_results = trainer.evaluate()
print('\\n[FOLD 0 FINAL EVALUATION]:')
print(f'Best Validation Balanced Accuracy: {eval_results.get("eval_balanced_accuracy", 0.0):.4f}')
print(f'Best Validation Accuracy: {eval_results.get("eval_accuracy", 0.0):.4f}')

ADAPTER_SAVE_PATH = OUTPUT_DIR / 'best_adapter'
model.save_pretrained(str(ADAPTER_SAVE_PATH))
tokenizer.save_pretrained(str(ADAPTER_SAVE_PATH))
print(f'\\nSuccessfully saved best LoRA adapter to: {ADAPTER_SAVE_PATH}')

preds = trainer.predict(val_dataset)
logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
np.save(OUTPUT_DIR / 'oof_probs_fold0.npy', probs)
print(f'Saved OOF probabilities to: {OUTPUT_DIR / "oof_probs_fold0.npy"} (shape: {probs.shape})')
""")

add_md("### Step 7: Verify Saved Artifacts & Export")
add_code("""import os

print('Files generated in output directory:')
for root, dirs, files in os.walk('/kaggle/working'):
    for f in files:
        full_p = os.path.join(root, f)
        sz_kb = os.path.getsize(full_p) / 1024
        print(f'  {full_p} ({sz_kb:.1f} KB)')

print('\\n[COMPLETE] LoRA adapter training finished successfully!')
""")

out_path = Path("train_kernel/lora_training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"SUCCESS: Wrote {len(nb['cells'])} cells to {out_path.resolve()}")
