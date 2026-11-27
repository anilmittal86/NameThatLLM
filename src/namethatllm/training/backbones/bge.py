"""
BGE Backbone Adapter (BAAI/bge-m3), Export Context Layer Restorer, and Encoder Ridge Packager.
"""

from __future__ import annotations

import copy
import gc
import importlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ...config import LAB2I, LABELS, K
from ...wpca_gaussian import fit_shared_gaussian_lowram
from ..data import load_zero_leak_stage1_pool
from ..evaluation import compute_detailed_metrics, extract_features_and_logits, solve_trace_ridge_wb
from ..modeling import MultiLayerLoRAClassifier
from .base import BaseBackboneAdapter, PrecisionSpec, TokenizationSpec, register_backbone


@register_backbone("bge")
class BgeAdapter(BaseBackboneAdapter):
    """Strategy adapter for bidirectional BGE encoder backbones (BAAI/bge-m3)."""

    family_name = "bge"
    default_backbone = "BAAI/bge-m3"
    default_output_dir = "checkpoints/export_bge_m3_v2"
    default_keep_layers = 24
    default_layers = (4, 7, 10, 13, 16, 19, 22, 24)
    default_max_len = 512
    default_end_id = 2

    def default_lora_targets(self) -> list[str]:
        return ["query", "key", "value", "dense"]

    def resolve_model_source(self, cfg: dict[str, Any], input_dir: Path) -> str:
        backbone_name = str(cfg.get("backbone", self.default_backbone))
        if input_dir.exists():
            for cfg_p in input_dir.rglob("config.json"):
                try:
                    c_data = json.loads(cfg_p.read_text(encoding="utf-8"))
                    m_type = str(c_data.get("model_type", "")).lower()
                    if "xlm-roberta" in m_type or "bge" in str(cfg_p.parent).lower():
                        return str(cfg_p.parent)
                except Exception:
                    continue
        return backbone_name

    def tok_spec_from_tokenizer(
        self, tok: Any, max_len: int | None = None, end_id: int | None = None
    ) -> TokenizationSpec:
        tok_pad = getattr(tok, "pad_token_id", None)
        pad_id = int(tok_pad) if tok_pad is not None else 0
        raw_cls = getattr(tok, "cls_token_id", None)
        cls_id = int(raw_cls) if raw_cls is not None else None
        if end_id is not None:
            raw_end = end_id
        else:
            tok_eos = getattr(tok, "eos_token_id", None)
            tok_sep = getattr(tok, "sep_token_id", None)
            if tok_eos is not None:
                raw_end = tok_eos
            elif tok_sep is not None:
                raw_end = tok_sep
            else:
                raw_end = self.default_end_id
        eff_end_id = int(raw_end)
        return TokenizationSpec(
            max_len=int(max_len if max_len is not None else self.default_max_len),
            pad_id=pad_id,
            cls_id=cls_id,
            end_id=eff_end_id,
            drop_first_token=True,
        )

    def load_tokenizer(self, model_source: str, cfg: dict[str, Any], is_smoke: bool = False) -> tuple[Any, TokenizationSpec]:
        mock_tok = cfg.get("mock_tokenizer")
        if mock_tok is not None:
            tok = mock_tok
        else:
            try:
                tok = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
                if tok.pad_token is None and tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
            except Exception:
                if is_smoke:
                    class _FallbackTok:
                        pad_token_id = 0
                        cls_token_id = 0
                        sep_token_id = 2
                        eos_token_id = 2
                        pad_token = "<pad>"
                        eos_token = "</s>"

                        def __call__(self, text, **kwargs):
                            return {"input_ids": [ord(c) % 500 + 10 for c in str(text)[:30]]}

                        def __len__(self):
                            return 250002

                    tok = _FallbackTok()
                else:
                    raise
        arch_cfg = cfg.get("architecture", {})
        max_len = int(cfg.get("max_len", self.default_max_len))
        end_id = cfg.get("end_id", None)
        spec = self.tok_spec_from_tokenizer(tok, max_len=max_len, end_id=end_id)
        drop_first = bool(arch_cfg.get("drop_first_token", cfg.get("drop_first_token", True)))
        if drop_first != spec.drop_first_token:
            spec = TokenizationSpec(
                max_len=spec.max_len,
                pad_id=spec.pad_id,
                cls_id=spec.cls_id,
                end_id=spec.end_id,
                drop_first_token=drop_first,
            )
        return tok, spec

    def load_backbone(
        self, model_source: str, cfg: dict[str, Any], precision: PrecisionSpec, device: torch.device
    ) -> tuple[nn.Module, int, int, list[int]]:
        arch_cfg = cfg.get("architecture", {})
        layers = [int(x) for x in arch_cfg.get("tap_layers", cfg.get("layers", self.default_layers))]
        keep_layers = int(cfg.get("keep_layers", max(layers) if layers else self.default_keep_layers))
        mock_bb = cfg.get("mock_backbone")
        if mock_bb is not None:
            base_model = copy.deepcopy(mock_bb).to(device)
            base_cfg = getattr(base_model, "config", type("Config", (), {"hidden_size": int(cfg.get("hidden_size", 32)), "num_hidden_layers": keep_layers})())
            return base_model, int(getattr(base_cfg, "hidden_size", cfg.get("hidden_size", 32))), keep_layers, layers

        base_cfg = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        dtype = torch.bfloat16 if precision.use_bf16 else torch.float32
        base_model = AutoModel.from_pretrained(model_source, config=base_cfg, torch_dtype=dtype, trust_remote_code=True).to(device)
        return base_model, int(base_cfg.hidden_size), keep_layers, layers

    def prepare_backbone(
        self, base_model: nn.Module, precision: PrecisionSpec, cfg: dict[str, Any], is_eval_only: bool
    ) -> nn.Module:
        arch_cfg = cfg.get("architecture", {})
        tap_layers = [int(x) for x in arch_cfg.get("tap_layers", cfg.get("layers", self.default_layers))]
        max_active_block = max(tap_layers) if tap_layers else int(cfg.get("keep_layers", self.default_keep_layers))
        encoder = getattr(base_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer") and len(encoder.layer) > max_active_block:
            base_model._orig_encoder_layers = encoder.layer
            base_model._max_active_block = max_active_block
            encoder.layer = nn.ModuleList(list(encoder.layer)[:max_active_block])
        return base_model

    @staticmethod
    def _find_encoder_holder(backbone: nn.Module) -> nn.Module | None:
        cands = [
            backbone,
            getattr(backbone, "base_model", None),
            getattr(getattr(backbone, "base_model", None), "model", None),
        ]
        if hasattr(backbone, "get_base_model"):
            try:
                cands.append(backbone.get_base_model())
            except Exception:
                pass
        for cand in cands:
            if cand is not None and hasattr(cand, "_orig_encoder_layers") and hasattr(cand, "encoder"):
                return cand
        return None

    @contextmanager
    def export_context(self, backbone: nn.Module) -> Iterator[None]:
        holder = self._find_encoder_holder(backbone)
        if holder is None:
            yield
            return
        orig_layers = holder._orig_encoder_layers
        max_active = int(getattr(holder, "_max_active_block", len(orig_layers)))
        holder.encoder.layer = orig_layers
        try:
            yield
        finally:
            holder.encoder.layer = nn.ModuleList(list(orig_layers)[:max_active])

    def apply_lora(self, base_model: nn.Module, cfg: dict[str, Any], is_eval_only: bool) -> nn.Module:
        lora_block = cfg.get("lora")
        if not isinstance(lora_block, dict):
            return super().apply_lora(base_model, cfg, is_eval_only=is_eval_only)

        lora_cfg = LoraConfig(
            r=int(lora_block.get("r", 32)),
            lora_alpha=int(lora_block.get("alpha", lora_block.get("lora_alpha", 64))),
            lora_dropout=float(lora_block.get("dropout", 0.05)),
            bias="none",
            target_modules=list(lora_block.get("target_modules", self.default_lora_targets())),
            layers_to_transform=[int(i) for i in lora_block["layers_to_transform"]] if "layers_to_transform" in lora_block else None,
            layers_pattern=str(lora_block.get("layers_pattern", "layer")) if "layers_to_transform" in lora_block else None,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        for p in peft_model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        if is_eval_only:
            peft_model.eval()
        return peft_model

    def build_scalers(self, precision: PrecisionSpec, device: torch.device) -> tuple[Any | None, Any | None]:
        enabled = bool(precision.compute_dtype == torch.float16 and device.type == "cuda" and torch.cuda.is_available())
        return torch.amp.GradScaler("cuda", enabled=enabled), None

    def maybe_run_pipeline(
        self, train_cfg: dict[str, Any], step_callback: Any | None = None, **kwargs: Any
    ) -> Any | None:
        pipeline_mode = train_cfg.get("pipeline_mode")
        if pipeline_mode in ("encoder_stage1", "bge_v2_stage1") or (
            pipeline_mode is None and "stage1" in train_cfg and "stage2" not in train_cfg
        ):
            run_encoder_stage1_fn = kwargs.get("run_encoder_stage1_fn")
            if run_encoder_stage1_fn is None:
                run_encoder_stage1_fn = globals().get("run_encoder_stage1") or importlib.import_module("namethatllm.training.engine").run_encoder_stage1

            input_dir = Path(train_cfg.get("input_dir", "/kaggle/input"))
            out_dir = Path(train_cfg.get("output_dir", self.default_output_dir))
            checkpoints_volume = train_cfg.get("checkpoints_volume", None)
            res_enc = run_encoder_stage1_fn(
                config=train_cfg,
                datasets_dir=Path(train_cfg.get("datasets_dir", input_dir)),
                checkpoint_base=out_dir if "output_dir" in train_cfg else None,
                on_checkpoint_saved=getattr(checkpoints_volume, "commit", None),
            )
            return Path(res_enc.get("checkpoint_dir", out_dir))
        return None


def package_encoder_ridge_checkpoint(
    cfg: dict[str, Any],
    datasets_dir: Path,
    checkpoint_root: Path | None = None,
    on_checkpoint_saved: Any = None,
) -> dict[str, Any]:
    """Merges an encoder Stage-1 LoRA checkpoint to FP16 and calibrates Plain + Centered trace-scaled Ridge heads."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tap_layers = [int(x) for x in cfg["architecture"]["tap_layers"]]
    max_len = int(cfg["max_len"])
    hidden_size = int(cfg["hidden_size"])
    backbone_name = str(cfg["backbone"])

    ckpt_root = Path(checkpoint_root) if checkpoint_root is not None else Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints/export_bge_m3_v2"))
    stage1_dir = ckpt_root / "stage1_best"
    final_dir = ckpt_root / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(backbone_name)
    base_f32 = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float32)
    peft_m = PeftModel.from_pretrained(base_f32, str(stage1_dir / "adapter"))
    merged = peft_m.merge_and_unload().half()
    merged.save_pretrained(str(final_dir), safe_serialization=True)
    tok.save_pretrained(str(final_dir))
    del peft_m, base_f32, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded = AutoModel.from_pretrained(str(final_dir), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device).eval()
    stage1_heads = load_file(str(stage1_dir / "heads.safetensors"), device="cpu")
    model = MultiLayerLoRAClassifier(backbone=reloaded, hidden_size=hidden_size, layers=tap_layers, num_classes=K, backbone_family="bge").to(device).eval()
    model.norms.float()
    with torch.no_grad():
        for i in range(len(tap_layers)):
            model.norms[i].weight.copy_(stage1_heads[f"norms.{i}.weight"].to(device, dtype=torch.float32))
            model.norms[i].bias.copy_(stage1_heads[f"norms.{i}.bias"].to(device, dtype=torch.float32))
    W_neural = stage1_heads["plain.weight"].float().numpy().astype(np.float32)
    b_neural = stage1_heads["plain.bias"].float().numpy().astype(np.float32)

    df_pool, df_train, df_bench = load_zero_leak_stage1_pool(datasets_dir)
    y_bench = np.array([LAB2I[m] for m in df_bench["model"]], dtype=np.int64)
    y_gold = np.array([LAB2I[m] for m in df_train["model"]], dtype=np.int64)
    prompts_gold = df_train["prompt_id"].values
    pool_labels = df_pool["label_idx"].to_numpy()

    rng = np.random.RandomState(42)
    n_per = max(1, int(cfg.get("output", {}).get("ridge_probe_samples", 36000)) // K)
    probe_idxs = np.concatenate([rng.choice(np.where(pool_labels == c_i)[0], size=min(n_per, int((pool_labels == c_i).sum())), replace=False) for c_i in range(K)])
    Z_bench, _ = extract_features_and_logits(model, df_bench["text"].astype(str).tolist(), tok, max_len=max_len, device=device)
    Z_gold, _ = extract_features_and_logits(model, df_train["text"].astype(str).tolist(), tok, max_len=max_len, device=device)
    Z_probe, _ = extract_features_and_logits(model, [str(df_pool["text"].iloc[int(i)]) for i in probe_idxs], tok, max_len=max_len, device=device)

    alpha_p = 0.25
    W_rp, b_rp = solve_trace_ridge_wb(Z_probe, pool_labels[probe_idxs], alphas=[alpha_p], device=device)[alpha_p]
    scale_rp = float(np.linalg.norm(W_neural)) / max(float(np.linalg.norm(W_rp)), 1e-8)
    W_rp_s, b_rp_s = W_rp * scale_rp, (b_rp - b_rp.mean()) * scale_rp + b_neural.mean()
    best_wn, W_plain, b_plain = max(
        ((wn, wn * W_neural + (1.0 - wn) * W_rp_s, wn * b_neural + (1.0 - wn) * b_rp_s) for wn in [0.0, 0.35, 0.50, 0.65, 0.80, 1.0]),
        key=lambda item: balanced_accuracy_score(y_bench, (Z_bench @ item[1].T + item[2]).argmax(axis=1)),
    )
    blend_detail = compute_detailed_metrics(Z_bench @ W_plain.T + b_plain, y_bench)

    Z_gold_cent = Z_gold.copy()
    for pid in np.unique(prompts_gold):
        m = prompts_gold == pid
        if m.sum() > 1:
            Z_gold_cent[m] -= Z_gold_cent[m].mean(axis=0, keepdims=True)
    cand_alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    splits = list(GroupKFold(n_splits=5).split(Z_gold, y_gold, groups=prompts_gold))
    oof_p_map = {a: np.zeros((len(y_gold), K), dtype=np.float32) for a in cand_alphas}
    oof_c_map = {a: np.zeros((len(y_gold), K), dtype=np.float32) for a in cand_alphas}
    for tr_i, val_i in splits:
        wp = solve_trace_ridge_wb(Z_gold[tr_i], y_gold[tr_i], cand_alphas, device=device)
        wc = solve_trace_ridge_wb(Z_gold_cent[tr_i], y_gold[tr_i], cand_alphas, device=device)
        for a in cand_alphas:
            oof_p_map[a][val_i] = Z_gold[val_i] @ wp[a][0].T + wp[a][1]
            oof_c_map[a][val_i] = Z_gold_cent[val_i] @ wc[a][0].T + wc[a][1]
    oof_p_best = max(float(balanced_accuracy_score(y_gold, oof_p_map[a].argmax(axis=1))) for a in cand_alphas)
    best_alpha_c, oof_c_best = max(
        ((a, float(balanced_accuracy_score(y_gold, oof_c_map[a].argmax(axis=1)))) for a in cand_alphas),
        key=lambda item: item[1],
    )
    W_cent_raw, b_cent_raw = solve_trace_ridge_wb(Z_gold_cent, y_gold, [best_alpha_c], device=device)[best_alpha_c]
    scale_c = float(np.linalg.norm(W_plain)) / max(float(np.linalg.norm(W_cent_raw)), 1e-8)

    heads_dict = {
        **{f"norms.{i}.weight": ln.weight.detach().cpu().contiguous() for i, ln in enumerate(model.norms)},
        **{f"norms.{i}.bias": ln.bias.detach().cpu().contiguous() for i, ln in enumerate(model.norms)},
        "plain.weight": torch.from_numpy(np.ascontiguousarray(W_plain)).float(),
        "plain.bias": torch.from_numpy(np.ascontiguousarray(b_plain)).float(),
        "centered.weight": torch.from_numpy(np.ascontiguousarray((W_cent_raw * scale_c).astype(np.float32))).float(),
        "centered.bias": torch.from_numpy(np.ascontiguousarray((b_cent_raw * scale_c).astype(np.float32))).float(),
    }
    save_file(heads_dict, str(final_dir / "heads.safetensors"))

    wpca = fit_shared_gaussian_lowram(Z_gold, y_gold, prompts_gold, d_target=min(256, Z_gold.shape[1]))
    np.savez_compressed(
        str(final_dir / "wPCA_bank.npz"),
        mx=wpca["mx"].astype(np.float32),
        sx=wpca["sx"].astype(np.float32),
        mu=wpca["mx"].astype(np.float32),
        sd=wpca["sx"].astype(np.float32),
        W=wpca["W"].astype(np.float32),
        m=wpca["m"].astype(np.float32),
        lam=wpca["lam"].astype(np.float32),
        V_top64=wpca["V_top64"].astype(np.float32),
        layers=np.array(tap_layers, dtype=np.int32),
    )

    metrics = {
        "bench_neural_bacc": float(balanced_accuracy_score(y_bench, (Z_bench @ W_neural.T + b_neural).argmax(axis=1))),
        "bench_plain_ridge_bacc": float(balanced_accuracy_score(y_bench, (Z_bench @ W_rp.T + b_rp).argmax(axis=1))),
        "bench_plain_blend_bacc": float(blend_detail["bacc"]),
        "bench_plain_blend_nll": float(blend_detail["nll"]),
        "bench_plain_blend_gem4": float(blend_detail["gem_within_bacc"]),
        "bench_plain_blend_gem_ret": float(blend_detail["gem_retention"]),
        "bench_plain_blend_sister_37_38": float(blend_detail["sister_37_38_acc"]),
        "plain_blend_w_neural": float(best_wn),
        "plain_ridge_alpha": float(alpha_p),
        "gold_oof5_plain_bacc": oof_p_best,
        "gold_oof5_centered_bacc": oof_c_best,
        "centered_ridge_alpha": float(best_alpha_c),
    }
    meta_json = {
        "labels": list(LABELS),
        "classes": list(LABELS),
        "target_family": "",
        "backbone": backbone_name,
        "cfg": {
            "keep": len(tap_layers),
            "layers": tap_layers,
            "max_len": max_len,
            "end_token_id": int(tok.eos_token_id) if getattr(tok, "eos_token_id", None) is not None else 2,
            "backbone": backbone_name,
            "pooling": "mean_drop_cls",
            "drop_first_token": True,
            "per_layer_layernorm": True,
            "feature_dim": len(tap_layers) * hidden_size,
            "metrics": metrics,
        },
    }
    (final_dir / "meta.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    (final_dir / "training_report.json").write_text(json.dumps({"metrics": metrics, "bench_blend_detail": blend_detail}, indent=2), encoding="utf-8")
    if on_checkpoint_saved is not None:
        on_checkpoint_saved()
    return {"final_dir": str(final_dir), "metrics": metrics, "meta_json": meta_json, "training_report": {"metrics": metrics, "bench_blend_detail": blend_detail}}
