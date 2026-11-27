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
LABEL2ID = LAB2I

FAMILIES = {
    "gpt": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
}

HARD_CLASS_INDICES = [1, 2, 3, 6, 7, 8, 13, 14, 15, 16, 17]

SPECIALIST_GROUPS = {
    "gpt": [0, 1, 2, 3, 4],
    "gpt_56_triad": [1, 2, 3],
    "gemini": [5, 6, 7, 8],
    "gemini_flash_triad": [6, 7, 8],
    "gemini_37_38_pair": [7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
    "glm_kimi_triad": [15, 16, 17],
    "frontier_and_fable": [10, 13, 14, 15, 16, 17],
    "kimi_gpt55_qwen": [0, 14, 17],
    "claude_glm": [9, 10, 11, 12, 13, 15, 16],
    "gpt_qwen_kimi": [0, 1, 2, 3, 4, 14, 17],
}

PRIOR_SCALES = {
    "qwen3_14b_p2": 1.00,
    "qwen3_14b_inhouse": 1.00,
    "qwen3_14b_inhouse_v2": 1.00,
    "qwen25_14b_p2": 0.85,
    "qwen25_14b_inhouse": 0.85,
    "deberta_p2": 0.80,
    "deberta_inhouse": 0.80,
    "deberta_gemini": 0.40,
    "deberta_glm": 0.35,
    "deberta_gpt": 0.35,
    "deberta_claude": 0.35,
    "deberta_claude_glm": 0.35,
    "deberta_gpt_qwen_kimi": 0.35,
    "qwen3_14b_p1": 0.50,
    "instyle": 0.45,
    "qwen3_8b_claude": 0.30,
    "qwen3_8b_inhouse": 0.30,
    "gemma2_9b_ts": 0.40,
    "llama31_8b_ts": 0.40,
    "mistral_7b_ts": 0.40,
    "bge_m3": 0.60,
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
    pool_patterns = ['clean_master_pool_v2.parquet', 'clean_master_pool.parquet', 'ntl_clean_master_pool.parquet', 'nlm_pool_v5.parquet', 'pool_v4.parquet', 'nlm_matched_clean.parquet', 'nlm_pool_v2.parquet']
    for pat in pool_patterns:
        for p in input_dir.rglob(pat):
            if p not in pool_files:
                pool_files.append(p)
    return pool_files

def find_base_model(input_dir: Path, model_family: str, hidden_sz: int, num_layers: int) -> Optional[Path]:
    """Discovers base model directories from Hugging Face config.json inspection."""
    candidates = []
    fam_target = model_family.lower()
    for cfg_p in input_dir.rglob('config.json'):
        try:
            with open(cfg_p, encoding='utf-8') as f:
                cfg = json.load(f)
            m_type = str(cfg.get('model_type', '')).lower()
            h_dim = cfg.get('hidden_size', 0)
            n_lay = cfg.get('num_hidden_layers', 0)
            text_cfg = cfg.get('text_config', {})
            if isinstance(text_cfg, dict):
                if not h_dim:
                    h_dim = text_cfg.get('hidden_size', 0)
                if not n_lay:
                    n_lay = text_cfg.get('num_hidden_layers', 0)
                if not m_type or m_type == 'none':
                    m_type = str(text_cfg.get('model_type', '')).lower()

            p_str = str(cfg_p.parent).lower().replace('\\', '/')
            fam_match = (
                fam_target in m_type or
                m_type in fam_target or
                ("gemma" in fam_target and ("gemma" in m_type or "gemma" in p_str)) or
                ("llama" in fam_target and ("llama" in m_type or "llama" in p_str)) or
                ("qwen" in fam_target and ("qwen" in m_type or "qwen" in p_str)) or
                ("mistral" in fam_target and ("mistral" in m_type or "mistral" in p_str))
            )
            if fam_match and h_dim == hidden_sz and n_lay == num_layers:
                p_str_norm = p_str.replace('-', '').replace('_', '')
                fam_target_norm = fam_target.replace('-', '').replace('_', '')
                if fam_target_norm in p_str_norm or fam_target in p_str or fam_target in m_type:
                    return cfg_p.parent
                candidates.append(cfg_p.parent)
        except Exception:
            continue
    return candidates[0] if candidates else None

def _inspect_checkpoint_dim_and_keep(ckpt_dir: Path) -> Tuple[int, int, str]:
    """Returns (head_feature_dim, keep_layers, backbone) without loading model weights."""
    dim = 0
    keep = 0
    backbone = ""
    meta_p = ckpt_dir / "meta.json"
    if meta_p.exists():
        try:
            with open(meta_p, encoding='utf-8') as f:
                m = json.load(f)
            cfg = m.get("cfg", m)
            keep = int(cfg.get("keep", 0))
            backbone = str(m.get("backbone", cfg.get("backbone", ""))).lower()
        except Exception:
            pass

    adapter_p = ckpt_dir / "adapter_config.json"
    if adapter_p.exists():
        try:
            with open(adapter_p, encoding='utf-8') as f:
                ac = json.load(f)
            base_model = str(ac.get("base_model_name_or_path", "")).lower()
            if not backbone and base_model:
                backbone = base_model
        except Exception:
            pass

    heads_p = ckpt_dir / "heads.safetensors"
    if heads_p.exists():
        try:
            from safetensors import safe_open
            with safe_open(str(heads_p), framework="pt", device="cpu") as f:
                for k in f.keys():
                    shape = f.get_tensor(k).shape
                    if len(shape) == 2 and (shape[0] == 18 or shape[1] == 18):
                        dim = int(shape[1] if shape[0] == 18 else shape[0])
                        break
        except Exception:
            pass
    elif (ckpt_dir / "heads.pt").exists():
        try:
            import torch
            sd = torch.load(ckpt_dir / "heads.pt", map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                for k, v in sd.items():
                    if hasattr(v, "shape") and len(v.shape) == 2 and (v.shape[0] == 18 or v.shape[1] == 18):
                        dim = int(v.shape[1] if v.shape[0] == 18 else v.shape[0])
                        break
        except Exception:
            pass
    return dim, keep, backbone

_FOLD_MARKER_TOKENS = ("_f0", "_f1", "_f2", "_f3", "_f4")
_INHOUSE_TOKENS = ("inhouse", "ntl-inhouse", "export_lora24", "export_champion", "lora-peft-training", "peft_training")
_FOLD_PRIORITY_RULES = (
    (("q14v5",), "qwen3_14b_v5"),
    (("ts_q3b14",), "qwen3_14b_ts"),
    (("q8v5",), "qwen3_8b_v5"),
    (("ts_q3b8", "ts8_s7", "l24x_q3b8"), "qwen3_8b_ts"),
    (("ll8v5",), "llama3_8b_v5"),
    (("ts_llama8",), "llama3_8b_ts"),
    (("ts_q25_7b", "l24x_q25_7b"), "qwen25_7b_ts"),
    (("gemma2", "gemma_2", "gemma-2", "ts_gemma", "ts-gemma"), "gemma2_9b_ts"),
    (("llama31", "llama_31", "llama-3.1", "ts_llama31", "ts-llama31"), "llama31_8b_ts"),
    (("mistral", "ts_mistral", "ts-mistral"), "mistral_7b_ts"),
    (("bge_m3", "bge-m3", "bgem3"), "bge_m3"),
)


def _classify_inhouse_checkpoint(ckpt_dir: Path, p_str: str) -> Optional[str]:
    """Routes in-house fine-tuned checkpoints using dim/keep/backbone metadata."""
    if any(skip in p_str for skip in ("family_gate", "/gate", "\\gate", "stage1_best", "stage1_last")):
        return None
    if any(tok in p_str for tok in ("bge_m3", "bge-m3", "bgem3")):
        return "bge_m3"
    dim, keep, backbone = _inspect_checkpoint_dim_and_keep(ckpt_dir)
    if "2.5" in backbone or "qwen25" in backbone or (dim == 40960 and (keep >= 30 or "25" in p_str or "qwen25" in p_str)):
        return "qwen25_14b_inhouse"
    if dim == 40960 or "14b" in p_str or "14b" in backbone:
        if any(v2_tok in p_str for v2_tok in ("inhouse_v2", "inhouse-v2", "14b_v2", "14b-v2")):
            return "qwen3_14b_inhouse_v2"
        return "qwen3_14b_inhouse"
    if dim == 8192 or "deberta" in p_str or "deberta" in backbone:
        return "deberta_inhouse"
    if dim in (32768, 28672) or ("8b" in p_str or "8b" in backbone) or ("14b" not in p_str and "14b" not in backbone):
        return "qwen3_8b_inhouse"
    return None


def find_checkpoints(input_dir: Path) -> Dict[str, Path]:
    """Finds all fine-tuned LoRA and head checkpoints across safetensors and PyTorch formats."""
    ckpts: Dict[str, Path] = {}

    head_candidates = list(input_dir.rglob("heads.safetensors")) + list(input_dir.rglob("heads.pt"))
    for p in head_candidates:
        p_str = str(p).lower().replace("\\", "/")
        is_fold = any(tok in p_str for tok in _FOLD_MARKER_TOKENS)

        if "lora_qwen3_14b" in p_str or "qwen3_14b_part2" in p_str or ("14b" in p_str and "part2" in p_str and "qwen3" in p_str):
            ckpts["qwen3_14b_p2"] = p.parent
        elif "lora_qwen25_14b" in p_str or "qwen25_14b_part2" in p_str or ("14b" in p_str and "part2" in p_str and "qwen2" in p_str):
            ckpts["qwen25_14b_p2"] = p.parent
        elif "lora14b_v1" in p_str:
            ckpts["qwen3_14b_p1"] = p.parent
        elif "lora8b_v1" in p_str:
            ckpts["qwen3_8b_p1"] = p.parent
        elif any(token in p_str for token in _INHOUSE_TOKENS):
            key = _classify_inhouse_checkpoint(p.parent, p_str)
            if key:
                ckpts[key] = p.parent
        elif "instyle" in p_str:
            ckpts["instyle"] = p.parent
        else:
            for tokens, key in _FOLD_PRIORITY_RULES:
                if any(tok in p_str for tok in tokens):
                    if key not in ckpts or not is_fold:
                        ckpts[key] = p.parent
                    break

    # 2. Check for Claude specialist head
    for p in input_dir.rglob("claude_head.safetensors"):
        ckpts["qwen3_8b_claude"] = p.parent
        break
    if "qwen3_8b_claude" not in ckpts:
        for p in input_dir.rglob("readouts.npz"):
            if "claude" in str(p).lower():
                ckpts["qwen3_8b_claude"] = p.parent
                break

    # 3. Check for DeBERTa sequence classification head
    for p in list(input_dir.rglob("head.safetensors")) + list(input_dir.rglob("heads.safetensors")):
        if "deberta" in str(p).lower():
            ckpts["deberta_p2"] = p.parent
            break

    # 4. Fallback adapter/config discovery
    if "instyle" not in ckpts:
        for p in input_dir.rglob("*"):
            p_s = str(p).lower().replace("\\", "/")
            if "instyle" in p_s and any(
                (p / rel).exists()
                for rel in ("adapter_model.safetensors", "adapter/adapter_model.safetensors", "heads.safetensors", "heads.pt")
            ):
                ckpts["instyle"] = p
                break

    if "qwen3_14b_ts" not in ckpts:
        for p in input_dir.rglob("adapter_model.safetensors"):
            p_s = str(p).lower().replace("\\", "/")
            if "q14v5" in p_s or "ts_q3b14" in p_s:
                ckpts["qwen3_14b_ts"] = p.parent
                break

    if "bge_m3" not in ckpts:
        for p in input_dir.rglob("config.json"):
            p_s = str(p.parent).lower().replace("\\", "/")
            if any(t in p_s for t in ("bge-m3", "bge_m3", "bgem3")):
                ckpts["bge_m3"] = p.parent
                break

    if "gemma2_9b_ts" not in ckpts:
        for fname in ("heads.safetensors", "adapter_model.safetensors"):
            for p in input_dir.rglob(fname):
                p_s = str(p.parent).lower().replace("\\", "/")
                if any(t in p_s for t in ("gemma2", "gemma_2", "gemma-2", "ts_gemma", "ts-gemma")):
                    ckpts["gemma2_9b_ts"] = p.parent.parent if p.parent.name == "adapter" else p.parent
                    break
            if "gemma2_9b_ts" in ckpts:
                break

    return ckpts

def find_fold_adapters(input_dir: Optional[Path], prefix: str) -> List[Path]:
    """Scans input_dir for all adapter/head directories matching prefix, prefix_f*, prefix_fold*, prefix-f*."""
    if input_dir is None:
        input_dir = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('.')

    matches: set[Path] = set()
    prefix_lower = prefix.lower()

    candidate_files = (
        list(input_dir.rglob('heads.safetensors')) +
        list(input_dir.rglob('heads.pt')) +
        list(input_dir.rglob('adapter_model.safetensors'))
    )
    for f in candidate_files:
        p = f.parent
        p_name = p.name.lower()
        if p_name in ('adapter', 'checkpoint'):
            p = p.parent
            p_name = p.name.lower()

        p_str = str(p).lower().replace('\\', '/')
        if prefix_lower in p_name or prefix_lower in p_str:
            # Check pattern: exact match or prefix followed by _ or - or fold
            tokens = p_name.replace('-', '_').split('_')
            matched = False
            for t in tokens:
                if t == prefix_lower or t.startswith(prefix_lower):
                    matched = True
                    break
            if not matched and prefix_lower in p_str:
                matched = True

            if matched:
                matches.add(p)

    return sorted(list(matches), key=lambda x: str(x).lower())

def validate_production_assets(
    input_dir: Path,
    train_path: Path,
    test_path: Path,
    pool_files: List[Path],
    ckpts: Dict[str, Path],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Fail-closed pre-flight validation for production inference runs (when NTL_SMOKE != '1').
    Raises FileNotFoundError immediately at startup if any required dataset, pool parquet,
    base model trunk, or checkpoint is missing.
    Returns a dict of verified base_model paths keyed by name.
    """
    if cfg is None:
        cfg = {}
    req = cfg.get("required_assets", {})
    missing: List[str] = []

    # 1. Verify competition CSVs
    if not train_path or not Path(train_path).exists():
        missing.append(f"Competition train.csv not found at: {train_path}")
    if not test_path or not Path(test_path).exists():
        missing.append(f"Competition test.csv not found at: {test_path}")

    # 2. Verify public pool parquet(s)
    require_pool = req.get("require_pool_parquet", True)
    valid_pools = [p for p in pool_files if p is not None and Path(p).exists()]
    if require_pool and len(valid_pools) == 0:
        missing.append(f"No public pool parquet (pool_v4.parquet / nlm_pool_v5.parquet) found under {input_dir}")

    # 3. Verify all required base model trunks
    default_bases = [
        {"name": "qwen3_14b", "family": "qwen3", "hidden_size": 5120, "num_layers": 40},
        {"name": "qwen25_14b", "family": "qwen2", "hidden_size": 5120, "num_layers": 48},
        {"name": "qwen3_8b", "family": "qwen3", "hidden_size": 4096, "num_layers": 36},
        {"name": "deberta_v3_large", "family": "deberta", "hidden_size": 1024, "num_layers": 24},
    ]
    if "required_base_models" in req:
        base_specs = req["required_base_models"]
    elif "checkpoints" in cfg or "checkpoints" in req:
        # Dynamically infer required base trunks from active checkpoints
        active_cks = cfg.get("checkpoints", req.get("checkpoints", []))
        inferred_bases = []
        if any("qwen3_14b" in str(c) for c in active_cks):
            inferred_bases.append({"name": "qwen3_14b", "family": "qwen3", "hidden_size": 5120, "num_layers": 40})
        if any("qwen25_14b" in str(c) for c in active_cks):
            inferred_bases.append({"name": "qwen25_14b", "family": "qwen2", "hidden_size": 5120, "num_layers": 48})
        if any("qwen3_8b" in str(c) for c in active_cks):
            inferred_bases.append({"name": "qwen3_8b", "family": "qwen3", "hidden_size": 4096, "num_layers": 36})
        if any("deberta" in str(c) for c in active_cks):
            inferred_bases.append({"name": "deberta_v3_large", "family": "deberta", "hidden_size": 1024, "num_layers": 24})
        if any("gemma2" in str(c) or "gemma" in str(c) for c in active_cks):
            inferred_bases.append({"name": "gemma2_9b", "family": "gemma2", "hidden_size": 3584, "num_layers": 42})
        if any("llama" in str(c) for c in active_cks):
            inferred_bases.append({"name": "llama3_8b", "family": "llama", "hidden_size": 4096, "num_layers": 32})
        if any("mistral" in str(c) for c in active_cks):
            inferred_bases.append({"name": "mistral_7b", "family": "mistral", "hidden_size": 4096, "num_layers": 32})
        base_specs = inferred_bases
    else:
        base_specs = default_bases
    verified_bases: Dict[str, Path] = {}
    for b_spec in base_specs:
        b_path = find_base_model(
            input_dir,
            b_spec["family"],
            int(b_spec["hidden_size"]),
            int(b_spec["num_layers"]),
        )
        if b_path is None or not Path(b_path).exists():
            missing.append(
                f"Required base model '{b_spec['name']}' (family={b_spec['family']}, "
                f"hidden={b_spec['hidden_size']}, layers={b_spec['num_layers']}) not found under {input_dir}"
            )
        else:
            verified_bases[b_spec["name"]] = b_path

    # 4. Verify fine-tuned checkpoints
    default_req_ckpts = [
        "qwen3_14b_p2", "qwen25_14b_p2",
        "deberta_p2", "qwen3_8b_claude"
    ]
    if "checkpoints" in req or "checkpoints" in cfg:
        # Unified single checkpoints list: ensure valid checkpoints were discovered under input_dir
        if len(ckpts) == 0:
            missing.append(f"No fine-tuned checkpoints found under {input_dir}")
    else:
        req_ckpts = req.get("required_checkpoints", default_req_ckpts)
        for ck_name in req_ckpts:
            if ck_name not in ckpts or not Path(ckpts[ck_name]).exists():
                missing.append(f"Required checkpoint '{ck_name}' not found under {input_dir} (discovered: {list(ckpts.keys())})")

    if missing:
        error_msg = (
            "[PRE-FLIGHT ASSET AUDIT FAILED] Missing required inference assets:\n  - "
            + "\n  - ".join(missing)
        )
        raise FileNotFoundError(error_msg)

    print(
        f"[Pre-Flight Audit PASSED] Verified {len(valid_pools)} pool file(s), "
        f"{len(verified_bases)} base trunks, and {len(ckpts)} checkpoints: {list(ckpts.keys())}"
    )
    return verified_bases
