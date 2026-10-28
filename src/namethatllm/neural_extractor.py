"""
Universal Neural Trunk Loader & Feature Extractor
=================================================
Manages multi-GPU sharding (2xT4 GPUs, max 13.5 GiB), layer truncation (keep_layers),
dynamic bfloat16 row-level retry for 14B fp16 overflow rescue, LayerNorm heads,
and bit-exact CPU fp16 snapshot/restore for shared trunks.
"""

import gc
import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file

from .config import LABELS, K, align_cols, log_softmax

def in_place_copy(dst: torch.Tensor, src: torch.Tensor):
    """In-place weight copy."""
    dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))

def _resolve_module_name(k: str, state_dict: dict) -> Optional[str]:
    """Resolves module name across various AutoModel and PEFT naming schemes."""
    raw = k.split(".lora_A")[0]
    candidates = [raw, raw.replace("base_model.model.", "")]
    if "layers." in raw:
        rel = raw[raw.index("layers."):]
        candidates.extend([rel, f"model.{rel}"])
    for c in candidates:
        if c in state_dict and hasattr(state_dict[c], "weight"):
            return c
    return None

def merge_lora(model: nn.Module, adapter_path: Path, sign: float = +1.0) -> bool:
    """In-place merges or unmerges (+1.0 or -1.0) a LoRA adapter directly into trunk weights."""
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.exists():
        weights_path = adapter_path / "adapter" / "adapter_model.safetensors"
    if not weights_path.exists():
        return False

    sd = load_file(str(weights_path))
    cfg_p = adapter_path / "adapter_config.json"
    if not cfg_p.exists():
        cfg_p = adapter_path / "adapter" / "adapter_config.json"
    scale = 1.0
    if cfg_p.exists():
        c = json.load(open(cfg_p))
        scale = float(c.get("lora_alpha", 16)) / float(c.get("r", 16))

    state_dict = dict(model.named_modules())
    pairs = {}
    for k in sd.keys():
        if "lora_A" in k:
            b_key = k.replace("lora_A", "lora_B")
            if b_key in sd:
                m_name = _resolve_module_name(k, state_dict)
                if m_name is not None:
                    pairs[m_name] = (sd[k], sd[b_key])

    applied = 0
    with torch.no_grad():
        for m_name, (A, B) in pairs.items():
            if m_name in state_dict:
                mod = state_dict[m_name]
                if hasattr(mod, "weight"):
                    A_d = A.to(mod.weight.device, dtype=torch.float32)
                    B_d = B.to(mod.weight.device, dtype=torch.float32)
                    delta = (B_d @ A_d) * (scale * sign)
                    mod.weight.data.add_(delta.to(mod.weight.dtype))
                    applied += 1
    return applied > 0

@torch.no_grad()
def forward_and_pool_batch(model: nn.Module, enc: dict, layers: List[int], norms: nn.ModuleList,
                           dtype: torch.dtype = torch.float16, drop_first_token: bool = True):
    """
    Runs forward pass in `dtype` (float16 on T4).
    CRITICAL: If any target layer overflows fp16 (Inf/NaN), recomputes ONLY the overflowed
    rows in `torch.bfloat16` and pools directly in float32 before restoring `dtype`!
    """
    hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states

    B = enc["input_ids"].shape[0]
    first_norm_dev = norms[0].weight.device

    # Check if any selected layer has Inf/NaN per row
    bad_mask = torch.zeros(B, dtype=torch.bool, device="cpu")
    for l in layers:
        bad_mask |= ~torch.isfinite(hs[l]).all(dim=(1, 2)).cpu()

    n_bad = int(bad_mask.sum().item())
    m = enc["attention_mask"].unsqueeze(-1).float().to(first_norm_dev)
    if drop_first_token:
        m[:, 0, :] = 0.0
    den = m.sum(dim=1).clamp(min=1.0)

    # Pool finite rows first in float32
    pooled_layers = [torch.zeros((B, model.config.hidden_size), dtype=torch.float32, device=first_norm_dev) for _ in layers]
    good_rows = torch.where(~bad_mask)[0].to(first_norm_dev)
    if len(good_rows) > 0:
        for idx_l, l in enumerate(layers):
            h_good = hs[l][good_rows.to(hs[l].device)].to(first_norm_dev).float()
            pooled_layers[idx_l][good_rows] = (h_good * m[good_rows]).sum(dim=1) / den[good_rows]

    # Rescue overflowed rows in bfloat16 (exact exponent range as float32, NEVER overflows!)
    if n_bad > 0:
        bad_rows = torch.where(bad_mask)[0]
        del hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(torch.bfloat16)
        enc_bad = {k: v[bad_rows] for k, v in enc.items()}
        hs_bad = model(**enc_bad, output_hidden_states=True, use_cache=False).hidden_states
        bad_rows_dev = bad_rows.to(first_norm_dev)
        for idx_l, l in enumerate(layers):
            h_b = hs_bad[l].to(first_norm_dev).float()
            h_b = torch.nan_to_num(h_b, nan=0.0, posinf=1e4, neginf=-1e4)
            pooled_layers[idx_l][bad_rows_dev] = (h_b * m[bad_rows_dev]).sum(dim=1) / den[bad_rows_dev]
        del hs_bad
        model.to(dtype)

    out = torch.cat([nm(pooled_layers[idx_l]) for idx_l, nm in enumerate(norms)], dim=-1)
    return out.cpu().numpy(), n_bad

def resolve_head_tensors(heads_dict: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Resolves plain and centered head weights and biases matching American and British naming."""
    W_p, b_p, W_c, b_c = None, None, None, None
    for wk, bk in [
        ("plain.weight", "plain.bias"),
        ("plain_coef", "plain_intercept"),
        ("weight", "bias"),
        ("W", "b"),
    ]:
        if wk in heads_dict and bk in heads_dict:
            W_p, b_p = heads_dict[wk], heads_dict[bk]
            break
    for wk, bk in [
        ("centered.weight", "centered.bias"),
        ("centred.weight", "centred.bias"),
        ("center.weight", "center.bias"),
        ("twin.weight", "twin.bias"),
        ("center_coef", "center_intercept"),
        ("centred_coef", "centred_intercept"),
    ]:
        if wk in heads_dict and bk in heads_dict:
            W_c, b_c = heads_dict[wk], heads_dict[bk]
            break

    if W_p is not None and hasattr(W_p, "cpu"):
        W_p = W_p.cpu().numpy()
    if b_p is not None and hasattr(b_p, "cpu"):
        b_p = b_p.cpu().numpy()
    if W_c is not None and hasattr(W_c, "cpu"):
        W_c = W_c.cpu().numpy()
    if b_c is not None and hasattr(b_c, "cpu"):
        b_c = b_c.cpu().numpy()

    return W_p, b_p, W_c, b_c

def compute_centered_logits(Z_test: np.ndarray, groups: np.ndarray, W_c: np.ndarray, b_c: np.ndarray, src_labels: list) -> Tuple[np.ndarray, np.ndarray]:
    """Computes prompt-topic centered twin logits for valid multi-sample clusters."""
    N = len(Z_test)
    out_lp = np.zeros((N, K), dtype=np.float64)
    if N < 2 or W_c is None:
        return out_lp, np.zeros(N, dtype=bool)
    sz = np.bincount(groups)[groups]
    valid = (sz >= 2) & (sz <= 18)
    if not valid.any():
        return out_lp, valid
    Zc = Z_test.copy()
    for g in np.unique(groups[valid]):
        mask = (groups == g)
        Zc[mask] -= Zc[mask].mean(axis=0, keepdims=True)
    raw_c = Zc[valid] @ W_c.T + b_c
    out_lp[valid] = log_softmax(align_cols(raw_c, src_labels))
    return out_lp, valid

class UniversalLoRATrunkLoader:
    def __init__(self, base_dir: Path, keep_layers: int, layers: List[int], max_len: int = 447):
        if base_dir is not None and not (base_dir / "config.json").exists():
            for sub in base_dir.rglob("config.json"):
                base_dir = sub.parent
                break
        self.base_dir = base_dir
        self.keep_layers = keep_layers
        self.layers = layers
        self.max_len = max_len
        self.total_rescued = 0
        self.tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "<|endoftext|>"

        cfg = AutoConfig.from_pretrained(str(base_dir), trust_remote_code=True)
        cfg.num_hidden_layers = keep_layers
        if getattr(cfg, "layer_types", None) is not None:
            cfg.layer_types = cfg.layer_types[:keep_layers]
        cfg.use_cache = False

        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            max_memory = {0: "13.5GiB", 1: "13.5GiB"}
            self.model = AutoModel.from_pretrained(
                str(base_dir), config=cfg, device_map="auto", max_memory=max_memory,
                torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
            )
        elif n_gpus == 1:
            self.model = AutoModel.from_pretrained(
                str(base_dir), config=cfg, device_map="auto",
                torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModel.from_pretrained(str(base_dir), config=cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        # Disable final trunk norm so last layer is not double normalized
        if hasattr(self.model, "norm"):
            self.model.norm = nn.Identity()
        elif hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            self.model.model.norm = nn.Identity()

        self.model.eval()
        self.device_norms = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # nn.LayerNorm with both weight and bias matching heads.safetensors
        hidden_dim = self.model.config.hidden_size
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in layers
        ]).to(self.device_norms).float()

        self.norms.eval()
        self.first_dev = next(self.model.parameters()).device

    def snapshot_weights(self, adapter_path: Path) -> dict:
        """Saves CPU fp16 copies of trunk modules modified by adapter."""
        weights_path = adapter_path / "adapter_model.safetensors"
        if not weights_path.exists():
            weights_path = adapter_path / "adapter" / "adapter_model.safetensors"
        if not weights_path.exists():
            return {}
        sd = load_file(str(weights_path))
        state_dict = dict(self.model.named_modules())
        backup = {}
        for k in sd.keys():
            if "lora_A" in k:
                m_name = _resolve_module_name(k, state_dict)
                if m_name is not None and m_name in state_dict and hasattr(state_dict[m_name], "weight"):
                    backup[m_name] = state_dict[m_name].weight.detach().cpu().half().clone()
        return backup

    def restore_weights(self, backup: dict):
        """Restores base trunk weights bit-for-bit without FP16 subtraction drift."""
        state_dict = dict(self.model.named_modules())
        with torch.no_grad():
            for m_name, orig in backup.items():
                if m_name in state_dict and hasattr(state_dict[m_name], "weight"):
                    state_dict[m_name].weight.data.copy_(orig.to(state_dict[m_name].weight.device))

    def set_active_layers(self, layers: List[int], max_len: Optional[int] = None):
        """Updates active pooled layers and instantiates LayerNorms matching the checkpoint."""
        self.layers = list(layers)
        if max_len is not None:
            self.max_len = max_len
        hidden_dim = self.model.config.hidden_size
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in self.layers
        ]).to(self.device_norms).float()
        self.norms.eval()

    def load_norm_weights(self, heads_sd: dict):
        """Loads LayerNorm weights and biases from head state dict."""
        with torch.no_grad():
            for i, ln in enumerate(self.norms):
                w_k = f'norms.{i}.weight'
                b_k = f'norms.{i}.bias'
                if w_k in heads_sd:
                    ln.weight.copy_(heads_sd[w_k].to(ln.weight.device, ln.weight.dtype))
                if b_k in heads_sd:
                    if hasattr(ln, 'bias') and ln.bias is not None:
                        ln.bias.copy_(heads_sd[b_k].to(ln.bias.device, ln.bias.dtype))

    def _extract_batch_ids(self, id_list: List[List[int]], batch_size: int = 8) -> np.ndarray:
        order = np.argsort([len(x) for x in id_list], kind="stable")
        total_dim = len(self.layers) * self.model.config.hidden_size
        Z = np.zeros((len(id_list), total_dim), dtype=np.float32)

        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            enc = self.tok.pad({"input_ids": [id_list[i] for i in idx]}, padding=True, return_tensors="pt")
            enc = {k: v.to(self.first_dev) for k, v in enc.items()}
            out_np, n_bad = forward_and_pool_batch(self.model, enc, self.layers, self.norms, torch.float16, drop_first_token=True)
            self.total_rescued += n_bad
            Z[idx] = out_np
        return Z

    @torch.no_grad()
    def extract_exact(self, texts: List[str], batch_size: int = 8, end_id: int = 151643) -> np.ndarray:
        t0 = time.time()
        self.total_rescued = 0
        raw_ids = self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        ids = [x + [end_id] for x in raw_ids]
        Z = self._extract_batch_ids(ids, batch_size=batch_size)
        print(f"[Extractor] Extracted {len(texts)} samples in {time.time() - t0:.2f}s (Total BF16 Rescued Rows: {self.total_rescued}).")
        return Z

    def free_memory(self):
        del self.model, self.norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def read_checkpoint_meta(ckpt_dir: Path, default_keep: int, default_layers: List[int], default_max_len: int = 447) -> dict:
    """Reads keep, layers, max_len, end_id, and labels dynamically from meta.json."""
    meta_p = ckpt_dir / "meta.json"
    labels = list(LABELS)
    keep = default_keep
    layers = list(default_layers)
    max_len = default_max_len
    end_id = 151643
    if meta_p.exists():
        try:
            m = json.load(open(meta_p))
            labels = m.get("labels", m.get("classes", labels))
            cfg = m.get("cfg", m)
            keep = int(cfg.get("keep", keep))
            layers = list(cfg.get("layers", layers))
            max_len = int(cfg.get("max_len", max_len))
            for ek in ("end_token_id", "eos_id", "eos_token_id", "end_token"):
                if ek in cfg and isinstance(cfg[ek], int):
                    end_id = int(cfg[ek])
                    break
        except Exception:
            pass
    return {"keep": keep, "layers": layers, "max_len": max_len, "end_id": end_id, "labels": labels}

def extract_deberta_checkpoint(
    ckpt_dir: Path,
    base_deb_dir: Optional[Path],
    texts_test: List[str],
    texts_train: List[str],
    y_train: np.ndarray,
    max_len: int = 512,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], list]:
    """
    Dedicated feature extractor for DeBERTa-v3-large sequence classification checkpoints.
    Probes drop_first_token and tokenization settings on train.csv to guarantee optimal accuracy.
    """
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_dir = str(base_deb_dir if base_deb_dir else ckpt_dir)
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

    head_p = ckpt_dir / "heads.safetensors"
    if not head_p.exists():
        head_p = ckpt_dir / "head.safetensors"
    heads_sd = load_file(str(head_p))
    W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

    meta = read_checkpoint_meta(ckpt_dir, default_keep=24, default_layers=[3, 6, 9, 12, 15, 18, 21, 24], default_max_len=max_len)
    ckpt_labels = meta["labels"]
    layers = meta["layers"]
    max_len = meta["max_len"]

    if (ckpt_dir / "encoder" / "config.json").exists():
        model_src = ckpt_dir / "encoder"
    elif (ckpt_dir / "config.json").exists() and ((ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists()):
        model_src = ckpt_dir
    elif base_deb_dir and (base_deb_dir / "config.json").exists():
        model_src = base_deb_dir
    else:
        model_src = ckpt_dir

    stage_dir = Path("/kaggle/temp/deberta_p2_stage")
    staged = False
    load_src = model_src
    if Path("/kaggle/temp").exists() and os.access("/kaggle/temp", os.W_OK):
        try:
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            shutil.copytree(model_src, stage_dir)
            load_src = stage_dir
            staged = True
        except Exception:
            load_src = model_src

    try:
        model = AutoModel.from_pretrained(str(load_src), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(dev)
    finally:
        if staged and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    model.eval()

    hidden_dim = model.config.hidden_size
    if W_p is not None and W_p.shape[1] > 0:
        expected_layers = W_p.shape[1] // hidden_dim
        if len(layers) != expected_layers:
            norm_keys = [k for k in heads_sd if k.startswith("norms.") and k.endswith(".weight")]
            if len(norm_keys) == expected_layers:
                layers = layers[:expected_layers] if len(layers) > expected_layers else list(range(expected_layers))
            elif len(layers) > expected_layers:
                layers = layers[:expected_layers]

    norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in layers]).to(dev).float()
    with torch.no_grad():
        for i, ln in enumerate(norms):
            w_k = f"norms.{i}.weight"
            b_k = f"norms.{i}.bias"
            if w_k in heads_sd:
                ln.weight.copy_(heads_sd[w_k].to(dev, ln.weight.dtype))
            if b_k in heads_sd:
                ln.bias.copy_(heads_sd[b_k].to(dev, ln.bias.dtype))
    norms.eval()

    def pool_texts(txt_list: List[str], drop_first: bool = True) -> np.ndarray:
        if len(txt_list) == 0:
            return np.zeros((0, len(layers) * hidden_dim), dtype=np.float32)
        order = np.argsort([len(t) for t in txt_list], kind="stable")
        all_vecs = np.zeros((len(txt_list), len(layers) * hidden_dim), dtype=np.float32)
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            chunk = [txt_list[i] for i in idx]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
                m = enc["attention_mask"].unsqueeze(-1).float()
                if drop_first:
                    m[:, 0, :] = 0.0
                den = m.sum(dim=1).clamp(min=1.0)
                pooled = [((torch.nan_to_num(hs[l].float(), nan=0.0, posinf=1e4, neginf=-1e4) * m).sum(dim=1) / den) for l in layers]
                out = torch.cat([norms[i](pooled[i]) for i in range(len(layers))], dim=-1)
                all_vecs[idx] = out.cpu().numpy()
        return all_vecs

    # Probe drop_first_token on train.csv[:180]
    probe_n = min(180, len(texts_train))
    best_drop = True
    if probe_n > 0 and W_p is not None:
        best_acc = -1.0
        for cand_drop in [True, False]:
            z_pr = pool_texts(texts_train[:probe_n], drop_first=cand_drop)
            preds = align_cols(z_pr @ W_p.T + b_p, ckpt_labels).argmax(axis=1)
            acc = float((preds == y_train[:probe_n]).mean())
            if acc > best_acc:
                best_acc = acc
                best_drop = cand_drop
        print(f"[DeBERTa Extractor] Probed drop_first={best_drop} (accuracy = {best_acc:.4f})")

    Z_test = pool_texts(texts_test, drop_first=best_drop)
    Z_train = pool_texts(texts_train, drop_first=best_drop) if len(texts_train) > 0 else np.zeros((0, len(layers) * hidden_dim), dtype=np.float32)

    del model, norms
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Z_test, Z_train, W_p, b_p, W_c, b_c, ckpt_labels
