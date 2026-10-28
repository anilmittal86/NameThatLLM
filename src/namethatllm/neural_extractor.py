"""
Universal Neural Trunk Loader & Feature Extractor
=================================================
Manages multi-GPU sharding (2xT4 GPUs, max 13.5 GiB), layer truncation (keep_layers),
dynamic bfloat16 row-level retry for 14B fp16 overflow rescue, and head tensor resolution.
"""

import gc
import json
import time
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
                m_name = k.split(".lora_A")[0].replace("base_model.model.", "")
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
        cfg.use_cache = False

        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            max_memory = {0: "13.5GiB", 1: "13.5GiB"}
            self.model = AutoModel.from_pretrained(
                str(base_dir), config=cfg, device_map="auto", max_memory=max_memory,
                torch_dtype=torch.float16, trust_remote_code=True
            )
        elif n_gpus == 1:
            self.model = AutoModel.from_pretrained(
                str(base_dir), config=cfg, device_map="auto",
                torch_dtype=torch.float16, trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(str(base_dir), config=cfg, torch_dtype=torch.float32)

        self.model.eval()
        self.device_norms = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        hidden_dim = self.model.config.hidden_size
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_dim, eps=getattr(cfg, "rms_norm_eps", 1e-6))
            for _ in layers
        ]).to(self.device_norms).float()

        for l_idx, nm in zip(layers, self.norms):
            try:
                base_lyr = self.model.layers[l_idx] if hasattr(self.model, "layers") else self.model.model.layers[l_idx]
                if hasattr(base_lyr, "input_layernorm") and hasattr(base_lyr.input_layernorm, "weight"):
                    nm.weight.data.copy_(base_lyr.input_layernorm.weight.data)
            except Exception:
                pass
        self.norms.eval()
        self.first_dev = next(self.model.parameters()).device

    def load_norm_weights(self, heads_sd: dict):
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
