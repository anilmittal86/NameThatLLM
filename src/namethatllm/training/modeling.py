"""
Shared Multi-Layer LoRA Architecture, FP16 Dynamic Loss Scaling, Optimizer Grouping, and Checkpoint I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from ..config import FAMILIES, HARD_CLASS_INDICES, LAB2I, LABELS, K
from ..wpca_gaussian import fit_shared_gaussian_lowram
from .backbones.base import TokenizationSpec


class MultiLayerLoRAClassifier(nn.Module):
    """Universal Multi-Layer LoRA Sequence Classifier with Plain and Prompt-Centered Dual Heads."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        layers: list[int],
        num_classes: int = K,
        keep_layers: int | None = None,
        backbone_family: str = "unknown",
        tok_spec: TokenizationSpec | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.layers = list(layers)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.total_dim = len(layers) * hidden_size
        self.backbone_family = str(backbone_family)
        self.tok_spec = tok_spec

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in layers])
        self.plain_head = nn.Linear(self.total_dim, num_classes)
        self.centered_head = nn.Linear(self.total_dim, num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])

    def predict_plain(self, Z: torch.Tensor) -> torch.Tensor:
        Z = Z.to(self.plain_head.weight.device)
        if self.training:
            return torch.mean(torch.stack([self.plain_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.plain_head(Z)

    def predict_centered(self, Z: torch.Tensor) -> torch.Tensor:
        Z = Z.to(self.centered_head.weight.device)
        if self.training:
            return torch.mean(torch.stack([self.centered_head(d(Z)) for d in self.dropouts], dim=0), dim=0)
        return self.centered_head(Z)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drop_first_token: bool = True,
        return_by_layer: bool = False,
        extract_all_layers: bool = False,
        return_multiview: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        m = attention_mask.clone().float()
        if drop_first_token and m.shape[1] > 1:
            m[:, 0] = 0.0
        den = m.sum(dim=1, keepdim=True).clamp(min=1e-5)
        first_device = input_ids.device

        if return_multiview:
            seq_idx = torch.cumsum(m, dim=1)
            half_len = (den * 0.5).clamp(min=1.0)
            m_head = m * (seq_idx <= half_len).float()
            m_tail = m * (seq_idx > half_len).float()
            den_head = m_head.sum(dim=1, keepdim=True).clamp(min=1e-5)
            den_tail = m_tail.sum(dim=1, keepdim=True).clamp(min=1e-5)

        try:
            outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, use_cache=False
            )
        except TypeError:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = getattr(outputs, "hidden_states", outputs)

        all_layers_pooled = []
        if extract_all_layers:
            for l_idx in range(len(hidden_states)):
                h = hidden_states[l_idx].float()
                layer_dev = h.device
                m_l = m.to(layer_dev)
                den_l = den.to(layer_dev)
                p_l = (h * m_l.unsqueeze(-1)).sum(dim=1) / den_l
                p_l = torch.nan_to_num(p_l, nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(min=-65000.0, max=65000.0)
                all_layers_pooled.append(p_l.to(first_device))

        pooled_list = []
        head_tail_cos_list = []
        token_std_norm_list = []

        for idx, l in enumerate(self.layers):
            h = hidden_states[l].float()
            layer_dev = h.device
            m_l = m.to(layer_dev)
            den_l = den.to(layer_dev)
            pooled = (h * m_l.unsqueeze(-1)).sum(dim=1) / den_l
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(min=-65000.0, max=65000.0)

            if return_multiview:
                mh_l = m_head.to(layer_dev).unsqueeze(-1)
                mt_l = m_tail.to(layer_dev).unsqueeze(-1)
                p_head = (h * mh_l).sum(dim=1) / den_head.to(layer_dev)
                p_tail = (h * mt_l).sum(dim=1) / den_tail.to(layer_dev)
                ht_cos = F.cosine_similarity(p_head, p_tail, dim=-1, eps=1e-6)
                head_tail_cos_list.append(ht_cos.to(first_device))

                diff_sq = ((h - pooled.unsqueeze(1)) ** 2) * m_l.unsqueeze(-1)
                tok_std = torch.sqrt((diff_sq.sum(dim=1) / den_l).clamp(min=1e-6))
                std_norm = torch.norm(tok_std, p=2, dim=-1)
                token_std_norm_list.append(std_norm.to(first_device))

            norm = self.norms[idx]
            norm_dev = next(norm.parameters()).device
            normed = norm(pooled.to(norm_dev))
            pooled_list.append(normed.to(first_device))

        Z_cat = torch.cat(pooled_list, dim=-1)

        if return_by_layer:
            Z_layers = torch.stack(all_layers_pooled if extract_all_layers else pooled_list, dim=1)
            if return_multiview:
                ht_cos_t = torch.stack(head_tail_cos_list, dim=1)
                std_norm_t = torch.stack(token_std_norm_list, dim=1)
                return Z_cat, Z_layers, ht_cos_t, std_norm_t
            return Z_cat, Z_layers

        return Z_cat

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, drop_first_token: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Z = self.forward_features(input_ids, attention_mask, drop_first_token=drop_first_token)
        logits_plain = self.predict_plain(Z)
        logits_centered = self.predict_centered(Z)
        return Z, logits_plain, logits_centered


class _ManualFP16LossScaler:
    """Device-agnostic dynamic FP16 loss scaler safe across multi-GPU device_map='auto' pipelines."""

    def __init__(
        self,
        enabled: bool = True,
        scale: float = 256.0,
        min_scale: float = 1.0,
        max_scale: float = 4096.0,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 200,
    ) -> None:
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.backoff_factor = float(backoff_factor)
        self.growth_factor = float(growth_factor)
        self.growth_interval = int(growth_interval)
        self._growth_tracker = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale if self.enabled else loss

    def unscale_grads_(self, model: nn.Module) -> None:
        if not self.enabled or self.scale == 0.0:
            return
        inv_scale = 1.0 / self.scale
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.mul_(inv_scale)

    def update(self, found_inf: bool) -> None:
        if not self.enabled:
            return
        if found_inf:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self._growth_tracker = 0


def _step_with_grad_guard(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: Any,
    scaler: Any | None = None,
    max_grad_norm: float = 1.0,
    manual_scaler: _ManualFP16LossScaler | None = None,
) -> bool:
    """Steps optimizer/scheduler safely, skipping update if any trainable gradient is non-finite."""
    use_manual = manual_scaler is not None and manual_scaler.is_enabled()
    use_scaler = (not use_manual) and scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
    if use_manual:
        manual_scaler.unscale_grads_(model)
    elif use_scaler:
        scaler.unscale_(opt)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
            if use_manual:
                manual_scaler.update(found_inf=True)
                print(
                    f"[Trainer Warning] Non-finite gradient detected; skipping optimizer step "
                    f"(new manual FP16 scale={manual_scaler.scale:.1f})."
                )
            else:
                print("[Trainer Warning] Non-finite gradient detected; skipping optimizer step to protect weights.")
                if use_scaler:
                    scaler.update()
            opt.zero_grad(set_to_none=True)
            return False
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if use_scaler:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    sched.step()
    if use_manual:
        manual_scaler.update(found_inf=False)
    opt.zero_grad(set_to_none=True)
    return True


def _get_loraplus_param_groups(
    model: MultiLayerLoRAClassifier,
    lr_lora: float,
    lr_head: float,
    lora_plus_ratio: float = 2.5,
    include_centered_head: bool = False,
) -> list[dict[str, Any]]:
    """Builds LoRA+ asymmetric parameter groups with higher learning rate on lora_B."""
    lora_a_params = []
    lora_b_params = []
    other_backbone_params = []
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if "lora_b" in name_lower:
            lora_b_params.append(param)
        elif "lora_a" in name_lower:
            lora_a_params.append(param)
        else:
            other_backbone_params.append(param)

    head_params = list(model.norms.parameters()) + list(model.plain_head.parameters())
    if include_centered_head and hasattr(model, "centered_head"):
        head_params += list(model.centered_head.parameters())

    groups = []
    if lora_a_params:
        groups.append({"params": lora_a_params, "lr": float(lr_lora)})
    if lora_b_params:
        groups.append({"params": lora_b_params, "lr": float(lr_lora) * float(lora_plus_ratio)})
    if other_backbone_params:
        groups.append({"params": other_backbone_params, "lr": float(lr_lora)})
    if head_params:
        groups.append({"params": head_params, "lr": float(lr_head)})
    return groups


def _build_class_weight_tensor(
    active_labels: list[str],
    sub_lab2i: dict[str, int],
    stage_cfg: dict[str, Any],
    default_hard_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Builds per-class CE loss weights from explicit class_weights map and hard_class_weight fallback."""
    num_classes = len(active_labels)
    hard_set = set(HARD_CLASS_INDICES)
    base_hard = float(stage_cfg.get("hard_class_weight", default_hard_weight))
    cls_map = stage_cfg.get("class_weights") or {}
    weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    for lbl, local_idx in sub_lab2i.items():
        if lbl in cls_map:
            weights[local_idx] = float(cls_map[lbl])
        elif LAB2I.get(lbl) in hard_set and base_hard != 1.0:
            weights[local_idx] = base_hard
    return weights


def export_compatible_checkpoint(
    model: MultiLayerLoRAClassifier,
    output_dir: Path,
    layers: list[int],
    keep: int,
    max_len: int = 447,
    end_id: int = 151643,
    backbone_name: str | None = None,
    Z_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    prompts_train: np.ndarray | None = None,
    class_labels: list[str] | None = None,
    target_family: str | None = None,
) -> Path:
    """Exports standard checkpoint artifacts: adapter/, heads.safetensors, meta.json, and optional wPCA_bank.npz."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    adapter_dir = out_p / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.backbone, "save_pretrained"):
        try:
            model.backbone.save_pretrained(str(adapter_dir))
        except Exception as e:
            print(f"[Trainer] Note on saving adapter: {e}")
    elif hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(str(adapter_dir))
        except Exception as e:
            print(f"[Trainer] Note on saving model: {e}")

    heads_dict = {}
    for i, norm in enumerate(model.norms):
        heads_dict[f"norms.{i}.weight"] = norm.weight.detach().cpu().contiguous()
        if hasattr(norm, "bias") and norm.bias is not None:
            heads_dict[f"norms.{i}.bias"] = norm.bias.detach().cpu().contiguous()

    heads_dict["plain.weight"] = model.plain_head.weight.detach().cpu().contiguous()
    heads_dict["plain.bias"] = model.plain_head.bias.detach().cpu().contiguous()
    heads_dict["centered.weight"] = model.centered_head.weight.detach().cpu().contiguous()
    heads_dict["centered.bias"] = model.centered_head.bias.detach().cpu().contiguous()

    save_file(heads_dict, str(out_p / "heads.safetensors"))

    active_labels = list(class_labels) if class_labels is not None else list(LABELS)
    meta = {
        "labels": active_labels,
        "classes": active_labels,
        "target_family": str(target_family or ""),
        "backbone": str(backbone_name) if backbone_name else "",
        "cfg": {
            "keep": int(keep),
            "layers": [int(x) for x in layers],
            "max_len": int(max_len),
            "end_token_id": int(end_id),
            "backbone": str(backbone_name) if backbone_name else "",
            "target_family": str(target_family or ""),
        },
    }
    (out_p / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if len(active_labels) == K and Z_train is not None and y_train is not None and prompts_train is not None:
        d_target = min(256, Z_train.shape[1])
        res = fit_shared_gaussian_lowram(Z_train, y_train, prompts_train, d_target=d_target)
        np.savez_compressed(
            str(out_p / "wPCA_bank.npz"),
            mx=res["mx"].astype(np.float32),
            sx=res["sx"].astype(np.float32),
            mu=res["mx"].astype(np.float32),
            sd=res["sx"].astype(np.float32),
            W=res["W"].astype(np.float32),
            m=res["m"].astype(np.float32),
            lam=res["lam"].astype(np.float32),
            V_top64=res["V_top64"].astype(np.float32),
            layers=np.array(layers, dtype=np.int32),
        )

    return out_p


def find_compatible_champion_lora(
    input_dir: Path,
    backbone_name: str,
    lora_r: int = 32,
    target_family: str = "",
) -> Path | None:
    """Finds the matching fine-tuned LoRA adapter directory for the given backbone and rank."""
    if not input_dir or not Path(input_dir).exists():
        return None

    input_p = Path(input_dir)
    candidates = list(input_p.rglob("adapter_model.safetensors")) + list(input_p.rglob("adapter_model.bin"))
    if not candidates:
        return None

    b_lower = backbone_name.lower()
    is_qwen3 = "qwen3" in b_lower or ("qwen" in b_lower and "2" not in b_lower and "2.5" not in b_lower)
    is_qwen25 = "qwen2" in b_lower or "2.5" in b_lower
    is_deberta = "deberta" in b_lower
    is_bge = "bge" in b_lower or "xlm-roberta" in b_lower

    matching_dirs: list[tuple[int, Path]] = []

    for cand in candidates:
        p_dir = cand.parent
        p_str = str(p_dir).lower().replace("\\", "/")

        if target_family:
            other_fams = [f for f in FAMILIES.keys() if f != target_family.lower()]
            if any(f"export_{f}" in p_str or f"{f}_gate" in p_str or f"{f}_spec" in p_str for f in other_fams):
                continue

        if is_bge and any(x in p_str for x in ("qwen", "deberta", "llama")) and "bge" not in p_str and "xlm-roberta" not in p_str:
            continue
        if (is_qwen3 or is_qwen25 or is_deberta) and ("bge" in p_str or "xlm-roberta" in p_str):
            if not any(x in p_str for x in ("qwen", "deberta")):
                continue

        cfg_p = p_dir / "adapter_config.json"
        if not cfg_p.exists():
            cfg_p = p_dir.parent / "adapter_config.json"

        score = 0
        if cfg_p.exists():
            try:
                c_data = json.loads(cfg_p.read_text(encoding="utf-8"))
                r_found = int(c_data.get("r", 0))
                if lora_r > 0 and r_found > 0 and r_found != lora_r:
                    continue

                base_str = str(c_data.get("base_model_name_or_path", "")).lower()
                if is_qwen3 and ("qwen2" in base_str or "2.5" in base_str or "llama" in base_str or "deberta" in base_str or "bge" in base_str or "xlm-roberta" in base_str):
                    continue
                if is_qwen25 and ("qwen3" in base_str or "llama" in base_str or "deberta" in base_str or "bge" in base_str or "xlm-roberta" in base_str) and "2.5" not in base_str and "qwen2" not in base_str:
                    continue
                if is_deberta and base_str and "deberta" not in base_str:
                    continue
                if is_bge and base_str and "bge" not in base_str and "xlm-roberta" not in base_str:
                    continue

                targets = {str(t).lower() for t in (c_data.get("target_modules") or [])}
                if is_bge and ("q_proj" in targets or "query_proj" in targets):
                    continue

                if is_qwen3 and "qwen3" in base_str:
                    score += 10
                if is_qwen25 and ("qwen2" in base_str or "2.5" in base_str):
                    score += 10
                if is_deberta and "deberta" in base_str:
                    score += 10
                if is_bge and ("bge" in base_str or "xlm-roberta" in base_str):
                    score += 10
            except Exception:
                pass

        if "export_champion" in p_str:
            score += 20
        if "inhouse" in p_str:
            score += 15
        if is_qwen3 and "qwen3_14b" in p_str:
            score += 10
        if is_qwen25 and ("qwen25" in p_str or "qwen2" in p_str):
            score += 10
        if is_bge and ("bge" in p_str or "xlm-roberta" in p_str):
            score += 10

        if is_qwen3 and ("qwen2" in p_str or "2.5" in p_str or "8b" in p_str):
            continue
        if is_qwen25 and ("qwen3" in p_str or "8b" in p_str):
            continue

        matching_dirs.append((score, p_dir))

    if not matching_dirs:
        return None

    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    return matching_dirs[0][1]


def load_and_slice_champion_heads(
    heads_path: Path,
    model: nn.Module,
    fam_cols: list[int] | None = None,
) -> bool:
    """Loads champion per-layer LayerNorms, plain_head, and centered_head after verifying shape compatibility."""
    if not Path(heads_path).exists():
        return False

    try:
        if str(heads_path).endswith(".safetensors"):
            h_sd = load_file(str(heads_path))
        else:
            h_sd = torch.load(str(heads_path), map_location="cpu")

        if not hasattr(model, "plain_head") or model.plain_head is None:
            return False

        w_ckpt = h_sd.get("plain.weight") if "plain.weight" in h_sd else h_sd.get("weight", None)
        b_ckpt = h_sd.get("plain.bias") if "plain.bias" in h_sd else h_sd.get("bias", None)
        if w_ckpt is None or b_ckpt is None:
            return False

        tgt_w = model.plain_head.weight
        tgt_b = model.plain_head.bias

        is_exact = tgt_w.shape == w_ckpt.shape and tgt_b.shape == b_ckpt.shape
        is_sliced = False
        sliced_w, sliced_b = None, None
        if not is_exact:
            if fam_cols is not None and w_ckpt.shape[0] >= max(fam_cols) + 1:
                sliced_w = w_ckpt[fam_cols]
                sliced_b = b_ckpt[fam_cols]
                if sliced_w.shape == tgt_w.shape and sliced_b.shape == tgt_b.shape:
                    is_sliced = True
            if not is_sliced:
                return False

        if is_exact:
            tgt_w.data.copy_(w_ckpt.to(tgt_w.device, dtype=tgt_w.dtype))
            tgt_b.data.copy_(b_ckpt.to(tgt_b.device, dtype=tgt_b.dtype))
        else:
            tgt_w.data.copy_(sliced_w.to(tgt_w.device, dtype=tgt_w.dtype))
            tgt_b.data.copy_(sliced_b.to(tgt_b.device, dtype=tgt_b.dtype))

        if hasattr(model, "norms") and model.norms is not None:
            for i, norm in enumerate(model.norms):
                wk, bk = f"norms.{i}.weight", f"norms.{i}.bias"
                if wk in h_sd and norm.weight.shape == h_sd[wk].shape:
                    norm.weight.data.copy_(h_sd[wk].to(norm.weight.device, dtype=norm.weight.dtype))
                if bk in h_sd and hasattr(norm, "bias") and norm.bias is not None and norm.bias.shape == h_sd[bk].shape:
                    norm.bias.data.copy_(h_sd[bk].to(norm.bias.device, dtype=norm.bias.dtype))

        if hasattr(model, "centered_head") and model.centered_head is not None:
            cw = h_sd.get("centered.weight", None)
            cb = h_sd.get("centered.bias", None)
            if cw is not None and cb is not None:
                tgt_cw = model.centered_head.weight
                tgt_cb = model.centered_head.bias
                if tgt_cw.shape == cw.shape and tgt_cb.shape == cb.shape:
                    tgt_cw.data.copy_(cw.to(tgt_cw.device, dtype=tgt_cw.dtype))
                    tgt_cb.data.copy_(cb.to(tgt_cb.device, dtype=tgt_cb.dtype))
                elif fam_cols is not None and cw.shape[0] >= max(fam_cols) + 1:
                    cw_sl = cw[fam_cols]
                    cb_sl = cb[fam_cols]
                    if cw_sl.shape == tgt_cw.shape and cb_sl.shape == tgt_cb.shape:
                        tgt_cw.data.copy_(cw_sl.to(tgt_cw.device, dtype=tgt_cw.dtype))
                        tgt_cb.data.copy_(cb_sl.to(tgt_cb.device, dtype=tgt_cb.dtype))

        return True
    except Exception as e:
        print(f"[Trainer] Note on loading/slicing champion head: {e}")
        return False
