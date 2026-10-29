"""
Canonical Configuration, Constants & Path Resolution
====================================================
Master canonical label ordering, provider families, simplex helpers,
and Kaggle filesystem asset discovery.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Canonical 18-Class Labels (Strict order from NTL two-stage LoRA part two)
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

FAMILIES = {
    "gpt": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
}

PRIOR_SCALES = {
    "qwen3_14b_p2": 1.00,
    "qwen25_14b_p2": 0.85,
    "deberta_p2": 0.80,
    "qwen3_14b_p1": 0.50,
    "instyle": 0.45,
    "qwen3_8b_claude": 0.30
}

def align_cols(arr: np.ndarray, src_labels: list) -> np.ndarray:
    """Aligns columns of array to canonical 18-class order."""
    src_list = [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in src_labels]
    idx = [src_list.index(l) for l in LABELS]
    if arr.ndim == 1:
        return arr[idx]
    elif arr.ndim == 2:
        if arr.shape[0] == len(src_list) and arr.shape[1] != len(src_list):
            return arr[idx, :]
        elif arr.shape[1] == len(src_list):
            return arr[:, idx]
    return arr[..., idx]

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)

def resolve_dataset_paths(input_dir: Optional[Path] = None) -> Tuple[Path, Path, Optional[Path]]:
    """Resolves train.csv, test.csv, and pool_v4.parquet across local and Kaggle environments."""
    if input_dir is None:
        input_dir = Path('/kaggle/input')
        if not input_dir.exists():
            input_dir = Path('.')

    candidates = [
        input_dir / 'competitions' / 'name-that-llm',
        input_dir / 'name-that-llm',
        Path('local_data'),
        Path('.')
    ]
    train_path = None
    test_path = None
    for c in candidates:
        if (c / 'train.csv').exists() and train_path is None:
            train_path = c / 'train.csv'
        if (c / 'test.csv').exists() and test_path is None:
            test_path = c / 'test.csv'

    if train_path is None:
        train_path = Path('local_data') / 'train.csv'
    if test_path is None:
        test_path = Path('local_data') / 'test.csv'

    pool_files = resolve_all_pool_files(input_dir)
    pool_file = pool_files[0] if pool_files else None

    return train_path, test_path, pool_file

def resolve_all_pool_files(input_dir: Optional[Path] = None) -> List[Path]:
    """Discovers all pretraining/unlabeled pool parquet files in priority order."""
    if input_dir is None:
        input_dir = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('.')
    pool_files: List[Path] = []
    pool_patterns = ['nlm_pool_v5.parquet', 'pool_v4.parquet', 'nlm_matched_clean.parquet', 'nlm_pool_v2.parquet']
    for pat in pool_patterns:
        for p in input_dir.rglob(pat):
            if p not in pool_files:
                pool_files.append(p)
    return pool_files

def find_base_model(input_dir: Path, model_family: str, hidden_sz: int, num_layers: int) -> Optional[Path]:
    """Discovers base model directories from Hugging Face config.json inspection."""
    for cfg_p in input_dir.rglob('config.json'):
        try:
            cfg = json.load(open(cfg_p))
            m_type = str(cfg.get('model_type', '')).lower()
            h_dim = cfg.get('hidden_size', 0)
            n_lay = cfg.get('num_hidden_layers', 0)
            if model_family.lower() in m_type and h_dim == hidden_sz and n_lay == num_layers:
                return cfg_p.parent
        except Exception:
            continue
    return None

def find_checkpoints(input_dir: Path) -> Dict[str, Path]:
    """Finds all fine-tuned LoRA and head checkpoints across safetensors and PyTorch formats."""
    ckpts = {}

    # 1. Check for heads.safetensors and heads.pt files
    head_candidates = list(input_dir.rglob('heads.safetensors')) + list(input_dir.rglob('heads.pt'))
    for p in head_candidates:
        p_str = str(p).lower().replace('\\', '/')
        if 'lora_qwen3_14b' in p_str or 'qwen3_14b_part2' in p_str or ('14b' in p_str and 'part2' in p_str and 'qwen3' in p_str):
            ckpts['qwen3_14b_p2'] = p.parent
        elif 'lora_qwen25_14b' in p_str or 'qwen25_14b_part2' in p_str or ('14b' in p_str and 'part2' in p_str and 'qwen2' in p_str):
            ckpts['qwen25_14b_p2'] = p.parent
        elif 'q14v5' in p_str or 'ts_q3b14' in p_str:
            ckpts['qwen3_14b_ts'] = p.parent
        elif 'lora14b_v1' in p_str:
            ckpts['qwen3_14b_p1'] = p.parent
        elif 'l24x_q3b8' in p_str or 'ts8_s7' in p_str or 'ts_q3b8' in p_str or 'll8v5' in p_str:
            ckpts['qwen3_8b_ts'] = p.parent
        elif 'lora8b_v1' in p_str:
            ckpts['qwen3_8b_p1'] = p.parent
        elif 'l24x_q25_7b' in p_str or 'ts_q25_7b' in p_str:
            ckpts['qwen25_7b_ts'] = p.parent
        elif 'instyle' in p_str:
            ckpts['instyle'] = p.parent

    # 2. Check for Claude specialist head
    for p in input_dir.rglob('claude_head.safetensors'):
        ckpts['qwen3_8b_claude'] = p.parent
        break
    if 'qwen3_8b_claude' not in ckpts:
        for p in input_dir.rglob('readouts.npz'):
            if 'claude' in str(p).lower():
                ckpts['qwen3_8b_claude'] = p.parent
                break

    # 3. Check for DeBERTa sequence classification head
    for p in list(input_dir.rglob('head.safetensors')) + list(input_dir.rglob('heads.safetensors')):
        if 'deberta' in str(p).lower():
            ckpts['deberta_p2'] = p.parent
            break

    # 4. Fallback search for instyle
    if 'instyle' not in ckpts:
        for p in input_dir.rglob('*'):
            p_s = str(p).lower().replace('\\', '/')
            if 'instyle' in p_s and ((p / 'adapter_model.safetensors').exists() or (p / 'adapter' / 'adapter_model.safetensors').exists() or (p / 'heads.safetensors').exists() or (p / 'heads.pt').exists()):
                ckpts['instyle'] = p
                break

    # 5. Fallback search for q14v5 / ts_q3b14 adapter
    if 'qwen3_14b_ts' not in ckpts:
        for p in input_dir.rglob('adapter_model.safetensors'):
            p_s = str(p).lower().replace('\\', '/')
            if 'q14v5' in p_s or 'ts_q3b14' in p_s:
                ckpts['qwen3_14b_ts'] = p.parent
                break

    return ckpts
