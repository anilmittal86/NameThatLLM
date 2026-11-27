"""
Qwen Backbone Adapter (Qwen3-14B, Qwen2.5-14B, Qwen3-8B) and FP16 / Norm Stabilization Helpers.
"""

from __future__ import annotations

import copy
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..modeling import _ManualFP16LossScaler
from ...config import find_base_model
from .base import BaseBackboneAdapter, PrecisionSpec, TokenizationSpec, register_backbone


def _disable_final_norm(mod: nn.Module) -> None:
    """Replaces an autoregressive LM's final LayerNorm / RMSNorm with nn.Identity()."""
    for path in ["norm", "model.norm", "base_model.norm", "base_model.model.norm", "base_model.model.model.norm"]:
        curr = mod
        parts = path.split(".")
        for p in parts[:-1]:
            curr = getattr(curr, p, None)
            if curr is None:
                break
        if curr is not None and hasattr(curr, parts[-1]) and getattr(curr, parts[-1]) is not None:
            setattr(curr, parts[-1], nn.Identity())


def _stabilize_fp16_backbone(mod: nn.Module) -> None:
    """Prevents float16 overflow (>65,504 -> Inf -> NaN) in Qwen2.5/Qwen3 on Turing T4 GPUs."""
    for _, sub in mod.named_modules():
        cls_name = sub.__class__.__name__.lower()
        if "rmsnorm" in cls_name and not getattr(sub, "_fp32_norm_patched", False):
            orig_norm_fwd = sub.forward

            def _safe_rms_fwd(self_norm, x, _orig=orig_norm_fwd):
                x_clean = torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
                variance = x_clean.pow(2).mean(-1, keepdim=True)
                x_normed = x_clean * torch.rsqrt(variance + getattr(self_norm, "variance_epsilon", 1e-6))
                w = self_norm.weight
                return (w.float() * x_normed).to(x.dtype)

            sub.forward = types.MethodType(_safe_rms_fwd, sub)
            sub._fp32_norm_patched = True

        if (
            hasattr(sub, "gate_proj")
            and hasattr(sub, "up_proj")
            and hasattr(sub, "down_proj")
            and hasattr(sub, "act_fn")
            and not getattr(sub, "_fp32_mlp_patched", False)
        ):
            def _safe_mlp_fwd(self_mlp, x):
                in_dtype = x.dtype
                x_safe = torch.nan_to_num(x, nan=0.0, posinf=65000.0, neginf=-65000.0)
                gate_f = self_mlp.act_fn(self_mlp.gate_proj(x_safe).float())
                up_f = self_mlp.up_proj(x_safe).float()
                inter_f = gate_f * up_f
                max_val = inter_f.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
                scale = (max_val / 16384.0).clamp(min=1.0).detach()
                inter_scaled = (inter_f / scale).to(in_dtype)
                down_out = self_mlp.down_proj(inter_scaled).float() * scale
                return torch.clamp(down_out, min=-65000.0, max=65000.0).to(in_dtype)

            sub.forward = types.MethodType(_safe_mlp_fwd, sub)
            sub._fp32_mlp_patched = True

        if "decoderlayer" in cls_name and not getattr(sub, "_fp16_clamp_hooked", False):
            def _clamp_layer_out(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    return torch.nan_to_num(output, nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(-65000.0, 65000.0)
                if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    h0 = torch.nan_to_num(output[0], nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(-65000.0, 65000.0)
                    return (h0,) + output[1:]
                return output

            sub.register_forward_hook(_clamp_layer_out)
            sub._fp16_clamp_hooked = True


@register_backbone("qwen")
class QwenAdapter(BaseBackboneAdapter):
    """Strategy adapter for Qwen causal decoder backbones (Qwen3-14B, Qwen2.5-14B, Qwen3-8B)."""

    family_name = "qwen"
    default_backbone = "Qwen/Qwen3-14B-Base"
    default_output_dir = "export_champion_qwen3_14b"
    default_keep_layers = 27
    default_layers = (2, 4, 6, 9, 13, 17, 22, 27)
    default_max_len = 384
    default_end_id = 151643
    preserves_working_checkpoint = True

    def default_lora_targets(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def resolve_model_source(self, cfg: dict[str, Any], input_dir: Path) -> str:
        backbone_name = str(cfg.get("backbone", self.default_backbone))
        local_base = None
        if input_dir.exists():
            b_lower = backbone_name.lower()
            if "2.5" in b_lower or "qwen25" in b_lower:
                local_base = find_base_model(input_dir, "qwen2", 5120, 48) or find_base_model(input_dir, "qwen2", 3584, 28)
            elif "14b" in b_lower:
                local_base = find_base_model(input_dir, "qwen3", 5120, 40) or find_base_model(input_dir, "qwen2", 5120, 48)
            elif "8b" in b_lower:
                local_base = find_base_model(input_dir, "qwen3", 4096, 36)
        return str(local_base) if local_base else backbone_name

    def resolve_precision(self, cfg: dict[str, Any], device: torch.device) -> PrecisionSpec:
        base = super().resolve_precision(cfg, device)
        n_gpus = torch.cuda.device_count()
        has_high_vram_single_gpu = bool(
            torch.cuda.is_available() and n_gpus == 1 and torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
        )
        use_fa2 = False
        if has_high_vram_single_gpu and base.use_bf16:
            try:
                import flash_attn  # noqa: F401

                use_fa2 = True
            except ImportError:
                use_fa2 = False
        return PrecisionSpec(
            compute_dtype=base.compute_dtype,
            use_bf16=base.use_bf16,
            use_fa2=use_fa2,
            total_vram_gb=base.total_vram_gb,
        )

    def tok_spec_from_tokenizer(
        self, tok: Any, max_len: int | None = None, end_id: int | None = None
    ) -> TokenizationSpec:
        tok_pad = getattr(tok, "pad_token_id", None)
        pad_id = int(tok_pad) if tok_pad is not None else self.default_end_id
        eff_end_id = int(end_id) if end_id is not None else self.default_end_id
        try:
            vocab_sz = len(tok)
            if eff_end_id >= vocab_sz:
                tok_eos = getattr(tok, "eos_token_id", None)
                tok_sep = getattr(tok, "sep_token_id", None)
                if tok_eos is not None:
                    eff_end_id = int(tok_eos)
                elif tok_sep is not None:
                    eff_end_id = int(tok_sep)
                else:
                    eff_end_id = 2
        except Exception:
            pass
        return TokenizationSpec(
            max_len=int(max_len if max_len is not None else self.default_max_len),
            pad_id=pad_id,
            cls_id=None,
            end_id=eff_end_id,
            drop_first_token=True,
        )

    def load_tokenizer(self, model_source: str, cfg: dict[str, Any], is_smoke: bool = False) -> tuple[Any, TokenizationSpec]:
        from transformers import AutoTokenizer

        mock_tok = cfg.get("mock_tokenizer")
        if mock_tok is not None:
            tok = mock_tok
        else:
            try:
                tok = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
            except Exception:
                if is_smoke:
                    class _FallbackTok:
                        pad_token_id = 0
                        cls_token_id = None
                        sep_token_id = None
                        eos_token_id = 2
                        pad_token = "<pad>"
                        eos_token = "</s>"

                        def __call__(self, text, **kwargs):
                            return {"input_ids": [ord(c) % 500 + 10 for c in str(text)[:30]]}

                        def __len__(self):
                            return 128100

                    tok = _FallbackTok()
                else:
                    raise
        max_len = int(cfg.get("max_len", self.default_max_len))
        cfg_end = cfg.get("end_id")
        end_id = int(cfg_end) if cfg_end is not None else self.default_end_id
        return tok, self.tok_spec_from_tokenizer(tok, max_len=max_len, end_id=end_id)

    def load_backbone(
        self, model_source: str, cfg: dict[str, Any], precision: PrecisionSpec, device: torch.device
    ) -> tuple[nn.Module, int, int, list[int]]:
        from transformers import AutoConfig, AutoModel

        keep_layers = int(cfg.get("keep_layers", self.default_keep_layers))
        layers = list(cfg.get("layers", self.default_layers))
        mock_bb = cfg.get("mock_backbone")
        if mock_bb is not None:
            base_model = copy.deepcopy(mock_bb).to(device)
            base_cfg = getattr(base_model, "config", type("Config", (), {"hidden_size": 32, "num_hidden_layers": 24})())
            return base_model, int(base_cfg.hidden_size), keep_layers, layers

        base_cfg = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        max_backbone_layers = int(getattr(base_cfg, "num_hidden_layers", self.default_keep_layers))
        keep_layers = min(keep_layers, max_backbone_layers)
        if any(l > keep_layers for l in layers):
            layers = [l for l in layers if l <= keep_layers]
        base_cfg.num_hidden_layers = keep_layers
        base_cfg.use_cache = False
        if getattr(base_cfg, "layer_types", None) is not None:
            base_cfg.layer_types = base_cfg.layer_types[:keep_layers]

        extra_kwargs: dict[str, Any] = {}
        if precision.use_fa2:
            extra_kwargs["attn_implementation"] = "flash_attention_2"

        n_gpus = torch.cuda.device_count()
        has_high_vram_single_gpu = bool(
            torch.cuda.is_available() and n_gpus == 1 and torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
        )
        if has_high_vram_single_gpu:
            print(f"[Trainer] High-VRAM single GPU detected ({precision.total_vram_gb:.1f} GiB). Loading directly on {device} in {precision.compute_dtype}...")
            base_model = AutoModel.from_pretrained(
                model_source, config=base_cfg, torch_dtype=precision.compute_dtype, low_cpu_mem_usage=True, trust_remote_code=True, **extra_kwargs
            ).to(device)
        elif n_gpus >= 2:
            max_memory = {0: "10.0GiB", 1: "13.5GiB"}
            base_model = AutoModel.from_pretrained(
                model_source, config=base_cfg, torch_dtype=precision.compute_dtype, device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True, trust_remote_code=True
            )
        elif n_gpus == 1:
            base_model = AutoModel.from_pretrained(
                model_source, config=base_cfg, torch_dtype=precision.compute_dtype, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True
            )
        else:
            base_model = AutoModel.from_pretrained(
                model_source, config=base_cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
            )
        return base_model, int(base_cfg.hidden_size), keep_layers, layers

    def prepare_backbone(
        self, base_model: nn.Module, precision: PrecisionSpec, cfg: dict[str, Any], is_eval_only: bool
    ) -> nn.Module:
        _disable_final_norm(base_model)
        if (
            precision.compute_dtype == torch.float16
            and cfg.get("mock_backbone") is None
            and bool(cfg.get("stabilize_fp16", True))
        ):
            _stabilize_fp16_backbone(base_model)
            print("[Trainer] Stabilized FP16 backbone with detached SwiGLU dynamic scaling and residual clamping.")
        return base_model

    def build_scalers(self, precision: PrecisionSpec, device: torch.device) -> tuple[Any | None, Any | None]:
        enabled = bool(precision.compute_dtype == torch.float16 and device.type == "cuda" and torch.cuda.is_available())
        return None, _ManualFP16LossScaler(enabled=enabled)

    def adjust_stage1_batch_config(
        self, stage1_cfg: dict[str, Any], precision: PrecisionSpec
    ) -> tuple[int, int, int]:
        bs1 = int(stage1_cfg.get("batch_size", 2))
        accum1 = int(stage1_cfg.get("gradient_accumulation_steps", 8))
        is_low_vram = torch.cuda.is_available() and precision.total_vram_gb < 24.0
        if is_low_vram and bs1 > 4:
            old_bs1 = bs1
            bs1 = 2
            accum1 = max(accum1 * (old_bs1 // 2), 8)
            print(f"[Trainer] Low VRAM GPU ({precision.total_vram_gb:.1f} GiB) detected: adjusted Stage 1 batch_size from {old_bs1} to {bs1}, accum_steps to {accum1}")
        default_eval_bs = 4 if is_low_vram else 8
        eval_bs = int(stage1_cfg.get("eval_batch_size", default_eval_bs))
        return bs1, accum1, eval_bs

    def clamp_extraction_batch_size(self, batch_size: int) -> int:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 18.0 and batch_size > 8:
                print(f"[Features Extraction] Low VRAM GPU ({vram_gb:.1f} GiB) detected: clamping batch_size from {batch_size} to 8.")
                return 8
        return batch_size
