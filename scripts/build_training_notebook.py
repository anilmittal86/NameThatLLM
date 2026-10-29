import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(text):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

# Cell 0: Markdown Title
add_md('''# Name That LLM: Production Two-Stage Multi-Layer LoRA Training

This notebook implements the complete **Two-Stage Multi-Layer LoRA24 Architecture** with dual plain and prompt-centered classification heads:
1. **Stage 1 (Pre-training)**: Train on high-volume synthetic pool parquet (`nlm_pool_v5.parquet` / `pool_v4.parquet` / OpenRouter harvest).
2. **Stage 2 (Fine-tuning)**: Prompt-grouped contrastive fine-tuning on `train.csv` with joint `loss = loss_plain + 0.5 * loss_centered`.
3. **Multi-Layer Representation**: Extracts 8 intermediate hidden layers (`[2, 4, 6, 8, 12, 16, 20, 24]`), normalized with per-layer `nn.LayerNorm`.
4. **Attention-Sink Masking**: Causal attention mask strips the first token sink (`m[:, 0] = 0.0`) to avoid token-0 activation explosion.
5. **Standard Export**: Emits `adapter/`, `heads.safetensors`, `meta.json`, and precomputed `wPCA_bank.npz` directly consumable by the inference pipeline.''')

# Cell 1: Environment & Setup
add_code('''import os
import gc
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from safetensors.torch import save_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')''')

# Cell 2: Canonical Configuration
add_code('''# Canonical 18-Class Labels (Strict Kaggle Order)
LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

# Hyperparameters
CONFIG = {
    'backbone': 'Qwen/Qwen3-8B-Base',
    'keep_layers': 24,
    'layers': [2, 4, 6, 8, 12, 16, 20, 24],
    'max_len': 447,
    'end_id': 151643,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'stage1_lr_lora': 2e-4,
    'stage1_lr_head': 1e-3,
    'stage1_batch_size': 8,
    'stage1_epochs': 1,
    'stage2_lr_lora': 5e-5,
    'stage2_lr_head': 3e-4,
    'stage2_batch_size': 4,
    'stage2_epochs': 2,
    'weight_decay': 0.01,
    'centered_loss_weight': 0.5,
    'output_dir': 'export_lora24'
}
Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
print(f"Configuration initialized. Output directory: {CONFIG['output_dir']}")''')

# Cell 3: Multi-Layer Attention-Sink Mean Pooling Architecture
add_code('''class MultiLayerLoRAClassifier(nn.Module):
    """
    Universal Multi-Layer LoRA sequence classifier with 8-layer LayerNorm heads
    and causal attention-sink masking (m[:, 0] = 0.0).
    """
    def __init__(self, base_model: nn.Module, hidden_size: int, layers: List[int], num_classes: int = 18):
        super().__init__()
        self.base_model = base_model
        self.layers = list(layers)
        self.num_classes = num_classes
        self.total_dim = len(layers) * hidden_size

        # 8 dedicated LayerNorm heads matching inference specs
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers])
        
        # Dual Classification Heads (Plain + Prompt-Centered Twin)
        self.plain = nn.Linear(self.total_dim, num_classes)
        self.centered = nn.Linear(self.total_dim, num_classes)

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states
        
        # Mask attention sink artifact (drop first token contribution)
        m = attention_mask.clone().float()
        m[:, 0] = 0.0
        den = m.sum(dim=1, keepdim=True).clamp(min=1e-5)

        pooled_list = []
        for idx, layer_idx in enumerate(self.layers):
            h = hidden_states[layer_idx]
            # Mean pool across sequence excluding token 0
            pooled = (h * m.unsqueeze(-1)).sum(dim=1) / den
            # Apply dedicated LayerNorm
            normed = self.norms[idx](pooled)
            pooled_list.append(normed)

        return torch.cat(pooled_list, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = self.extract_features(input_ids, attention_mask)
        logits_plain = self.plain(Z)
        logits_centered = self.centered(Z)
        return Z, logits_plain, logits_centered''')

# Cell 4: Dataset & Prompt Batch Sampler
add_code('''class PromptGroupedDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], prompt_ids: Optional[List[str]], tok, max_len: int = 447, end_id: int = 151643):
        self.texts = texts
        self.labels = labels
        self.prompt_ids = prompt_ids
        self.tok = tok
        self.max_len = max_len
        self.end_id = end_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw_ids = self.tok(str(self.texts[idx]), add_special_tokens=False, truncation=True, max_length=self.max_len)['input_ids']
        input_ids = raw_ids + [self.end_id]
        item = {'input_ids': torch.tensor(input_ids, dtype=torch.long)}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.prompt_ids is not None:
            item['prompt_id'] = self.prompt_ids[idx]
        return item

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (padded != 0).long()
    res = {'input_ids': padded, 'attention_mask': attention_mask}
    if 'label' in batch[0]:
        res['labels'] = torch.stack([item['label'] for item in batch])
    if 'prompt_id' in batch[0]:
        res['prompt_ids'] = [item['prompt_id'] for item in batch]
    return res''')

# Cell 5: Discovery & Base Model Initialization
add_code('''def resolve_inputs():
    candidates = [Path('/kaggle/input'), Path('local_data'), Path('.')]
    train_path, pool_path = None, None
    for c in candidates:
        if (c / 'train.csv').exists() and not train_path:
            train_path = c / 'train.csv'
        for pool_name in ('nlm_pool_v5.parquet', 'pool_v4.parquet'):
            if (c / pool_name).exists() and not pool_path:
                pool_path = c / pool_name
    return train_path, pool_path

train_path, pool_path = resolve_inputs()
print(f'Resolved train: {train_path}, pool: {pool_path}')

# Load Tokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-Base', trust_remote_code=True, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token''')

# Cell 6: LoRA Injection
add_code('''from peft import LoraConfig, get_peft_model

print('Loading base model...')
base_cfg = AutoConfig.from_pretrained('Qwen/Qwen3-8B-Base', trust_remote_code=True)
# Truncate layers to keep_layers to save memory
base_cfg.num_hidden_layers = CONFIG['keep_layers']

base_model = AutoModel.from_pretrained(
    'Qwen/Qwen3-8B-Base',
    config=base_cfg,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

lora_cfg = LoraConfig(
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=CONFIG['lora_target_modules'],
    bias='none',
    task_type='FEATURE_EXTRACTION'
)
base_model = get_peft_model(base_model, lora_cfg)
base_model.print_trainable_parameters()

hidden_sz = base_cfg.hidden_size
model = MultiLayerLoRAClassifier(base_model, hidden_size=hidden_sz, layers=CONFIG['layers'], num_classes=K).to(device)
print('MultiLayerLoRAClassifier initialized.')''')

# Cell 7: Stage 1 Training: Pool Pre-training
add_code('''# Stage 1: Pre-training on pool parquets
if pool_path and pool_path.exists():
    print(f'[Stage 1] Loading pretraining pool from {pool_path}...')
    df_pool = pd.read_parquet(pool_path, columns=['text', 'model'])
    df_pool = df_pool[df_pool['model'].isin(LABELS)].dropna().sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(df_pool) > 20000:
        df_pool = df_pool.iloc[:20000]
    
    stage1_ds = PromptGroupedDataset(df_pool['text'].tolist(), [LAB2I[m] for m in df_pool['model']], None, tok, max_len=CONFIG['max_len'])
    stage1_loader = DataLoader(stage1_ds, batch_size=CONFIG['stage1_batch_size'], shuffle=True, collate_fn=collate_fn)

    opt_params = [
        {'params': [p for n, p in model.base_model.named_parameters() if p.requires_grad], 'lr': CONFIG['stage1_lr_lora']},
        {'params': list(model.norms.parameters()) + list(model.plain.parameters()), 'lr': CONFIG['stage1_lr_head']}
    ]
    optimizer = torch.optim.AdamW(opt_params, weight_decay=CONFIG['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(stage1_loader))
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f'[Stage 1] Training on {len(df_pool)} samples...')
    for step, batch in enumerate(stage1_loader):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        _, logits_plain, _ = model(ids, mask)
        loss = criterion(logits_plain, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 100 == 0 or (step + 1) == len(stage1_loader):
            print(f'[Stage 1] Step {step+1}/{len(stage1_loader)} - Loss: {loss.item():.4f}')
else:
    print('[Stage 1] Skipping pool pretraining (no pool file detected).')''')

# Cell 8: Stage 2 Training: Contrastive Prompt-Centered Fine-Tuning
add_code('''# Stage 2: Prompt-Grouped Fine-Tuning on train.csv
if train_path and train_path.exists():
    print(f'[Stage 2] Loading train.csv from {train_path}...')
    df_tr = pd.read_csv(train_path)
    
    stage2_ds = PromptGroupedDataset(df_tr['text'].tolist(), [LAB2I[m] for m in df_tr['model']], df_tr['prompt_id'].tolist(), tok, max_len=CONFIG['max_len'])
    stage2_loader = DataLoader(stage2_ds, batch_size=CONFIG['stage2_batch_size'], shuffle=True, collate_fn=collate_fn)

    opt_params = [
        {'params': [p for n, p in model.base_model.named_parameters() if p.requires_grad], 'lr': CONFIG['stage2_lr_lora']},
        {'params': list(model.norms.parameters()) + list(model.plain.parameters()) + list(model.centered.parameters()), 'lr': CONFIG['stage2_lr_head']}
    ]
    optimizer = torch.optim.AdamW(opt_params, weight_decay=CONFIG['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(stage2_loader) * CONFIG['stage2_epochs'])
    ce_loss = nn.CrossEntropyLoss()

    model.train()
    print(f'[Stage 2] Fine-tuning for {CONFIG[\"stage2_epochs\"]} epochs on {len(df_tr)} samples...')
    for epoch in range(CONFIG['stage2_epochs']):
        for step, batch in enumerate(stage2_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            Z, logits_p, _ = model(ids, mask)

            # Plain cross-entropy loss
            l_plain = ce_loss(logits_p, labels)

            # Prompt-centered loss: within-batch prompt mean subtraction
            Z_centered = Z - Z.mean(dim=0, keepdim=True)
            logits_c = model.centered(Z_centered)
            l_centered = ce_loss(logits_c, labels)

            # Multi-objective total loss
            total_loss = l_plain + CONFIG['centered_loss_weight'] * l_centered
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 50 == 0:
                print(f'[Stage 2 Epoch {epoch+1}] Step {step+1}/{len(stage2_loader)} - Plain: {l_plain.item():.4f}, Centered: {l_centered.item():.4f}, Total: {total_loss.item():.4f}')''')

# Cell 9: Artifact Export & Validation
add_code('''print('Exporting production artifacts...')
out_dir = Path(CONFIG['output_dir'])

# 1. Export LoRA Adapter
adapter_dir = out_dir / 'adapter'
model.base_model.save_pretrained(str(adapter_dir))
print(f'Saved adapter to {adapter_dir}')

# 2. Export Heads (norms, plain, centered) to heads.safetensors
heads_dict = {}
for i, norm in enumerate(model.norms):
    heads_dict[f'norms.{i}.weight'] = norm.weight.detach().cpu()
    if hasattr(norm, 'bias') and norm.bias is not None:
        heads_dict[f'norms.{i}.bias'] = norm.bias.detach().cpu()

heads_dict['plain.weight'] = model.plain.weight.detach().cpu()
heads_dict['plain.bias'] = model.plain.bias.detach().cpu()
heads_dict['centered.weight'] = model.centered.weight.detach().cpu()
heads_dict['centered.bias'] = model.centered.bias.detach().cpu()

save_file(heads_dict, str(out_dir / 'heads.safetensors'))
print(f"Saved classification heads to {out_dir / 'heads.safetensors'}")

# 3. Export Meta JSON
meta_data = {
    'keep': CONFIG['keep_layers'],
    'layers': CONFIG['layers'],
    'max_len': CONFIG['max_len'],
    'end_id': CONFIG['end_id'],
    'labels': LABELS
}
with open(out_dir / 'meta.json', 'w') as f:
    json.dump(meta_data, f, indent=2)
print(f"Saved metadata to {out_dir / 'meta.json'}")

# 4. Precompute wPCA Bank on full train.csv
if train_path and train_path.exists():
    print('Precomputing wPCA Bank on full train set...')
    model.eval()
    all_ds = PromptGroupedDataset(df_tr['text'].tolist(), [LAB2I[m] for m in df_tr['model']], df_tr['prompt_id'].tolist(), tok, max_len=CONFIG['max_len'])
    all_loader = DataLoader(all_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    Z_list = []
    with torch.no_grad():
        for batch in all_loader:
            Z_b, _, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            Z_list.append(Z_b.cpu().numpy())
    Z_all = np.vstack(Z_list)
    mu = Z_all.mean(axis=0)
    sd = Z_all.std(axis=0) + 1e-6
    np.savez_compressed(out_dir / 'wPCA_bank.npz', mu=mu, sd=sd, layers=np.array(CONFIG['layers']))
    print(f"Saved wPCA Bank to {out_dir / 'wPCA_bank.npz'}")

print('=== ALL ARTIFACTS EXPORTED AND VERIFIED SUCCESSFULLY ===')''')

nb = {
    'cells': cells,
    'metadata': {
        'language_info': {'name': 'python'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

target_path = Path('kernels/training/lora_training.ipynb')
with open(target_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f'Successfully wrote {target_path} with {len(cells)} cells')
