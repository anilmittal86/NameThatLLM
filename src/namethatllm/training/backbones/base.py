"""
Backbone Adapter Base Contracts, Precision/Tokenization Specs, and Registry.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TokenizationSpec:
    """Normalized tokenization and sequence boundary specification for a backbone family."""

    max_len: int
    pad_id: int
    cls_id: int | None
    end_id: int | None
    drop_first_token: bool


@dataclass(frozen=True)
class PrecisionSpec:
    """Resolved compute dtype and hardware capability profile."""

    compute_dtype: torch.dtype
    use_bf16: bool
    use_fa2: bool
    total_vram_gb: float


class BaseBackboneAdapter(ABC):
    """Strategy contract encapsulating backbone-specific loading, precision, tokenization, and LoRA rules."""

    family_name: str = "base"
    default_backbone: str = ""
    default_output_dir: str = "export_checkpoint"
    default_keep_layers: int = 24
    default_layers: tuple[int, ...] = (3, 6, 9, 12, 15, 18, 21, 24)
    default_max_len: int = 384
    default_end_id: int = 0
    preserves_working_checkpoint: bool = False

    def default_lora_targets(self) -> list[str]:
        return []

    @abstractmethod
    def resolve_model_source(self, cfg: dict[str, Any], input_dir: Path) -> str:
        """Resolves a local cached backbone path under `input_dir` or falls back to HF ID."""

    def resolve_precision(self, cfg: dict[str, Any], device: torch.device) -> PrecisionSpec:
        is_ampere_or_newer = False
        total_vram_gb = 0.0
        if torch.cuda.is_available():
            major_cap = torch.cuda.get_device_capability(0)[0]
            is_ampere_or_newer = major_cap >= 8
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        use_bf16 = bool(cfg.get("bf16", False) or is_ampere_or_newer)
        compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
        return PrecisionSpec(
            compute_dtype=compute_dtype,
            use_bf16=use_bf16,
            use_fa2=False,
            total_vram_gb=total_vram_gb,
        )

    @abstractmethod
    def load_tokenizer(self, model_source: str, cfg: dict[str, Any], is_smoke: bool = False) -> tuple[Any, TokenizationSpec]:
        """Loads tokenizer (or mock/fallback) and constructs its TokenizationSpec."""

    def tok_spec_from_tokenizer(
        self, tok: Any, max_len: int | None = None, end_id: int | None = None
    ) -> TokenizationSpec:
        """Builds a TokenizationSpec from an already-instantiated tokenizer (fallback for legacy raw mocks)."""
        tok_pad = getattr(tok, "pad_token_id", None)
        pad_id = int(tok_pad) if tok_pad is not None else 0
        raw_cls = getattr(tok, "cls_token_id", None)
        cls_id = int(raw_cls) if raw_cls is not None else None
        if end_id is not None:
            raw_end = end_id
        else:
            tok_sep = getattr(tok, "sep_token_id", None)
            raw_end = tok_sep if tok_sep is not None else getattr(tok, "eos_token_id", None)
        eff_end_id = int(raw_end) if raw_end is not None else None
        return TokenizationSpec(
            max_len=int(max_len if max_len is not None else self.default_max_len),
            pad_id=pad_id,
            cls_id=cls_id,
            end_id=eff_end_id,
            drop_first_token=cls_id is None,
        )

    @abstractmethod
    def load_backbone(
        self, model_source: str, cfg: dict[str, Any], precision: PrecisionSpec, device: torch.device
    ) -> tuple[nn.Module, int, int, list[int]]:
        """Loads backbone weights and returns `(base_model, hidden_size, keep_layers, layers)`."""

    def prepare_backbone(
        self, base_model: nn.Module, precision: PrecisionSpec, cfg: dict[str, Any], is_eval_only: bool
    ) -> nn.Module:
        """Applies optional model-specific modifications prior to wrapping with LoRA."""
        return base_model

    def apply_lora(self, base_model: nn.Module, cfg: dict[str, Any], is_eval_only: bool) -> nn.Module:
        """Wraps backbone with PEFT LoRA, enforces FP32 trainable params, and sets checkpointing/eval mode."""
        from peft import LoraConfig, get_peft_model

        try:
            import peft.import_utils
            import peft.tuners.lora.torchao

            peft.import_utils.is_torchao_available = lambda: False
            peft.tuners.lora.torchao.is_torchao_available = lambda: False
        except Exception:
            pass

        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

        linear_mod_names = {
            name.split(".")[-1]
            for name, module in base_model.named_modules()
            if isinstance(module, nn.Linear)
        }
        configured_targets = list(cfg.get("lora_target_modules", [])) or self.default_lora_targets()
        active_targets = [m for m in configured_targets if m in linear_mod_names]
        if not active_targets:
            active_targets = list(linear_mod_names)[:7] if linear_mod_names else configured_targets
            print(f"[Trainer] Configured LoRA targets not found; falling back to: {active_targets}")
        else:
            print(f"[Trainer] Active LoRA target modules: {active_targets}")

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 64)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=active_targets,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        for p in peft_model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()

        if not is_eval_only:
            if hasattr(peft_model, "gradient_checkpointing_enable"):
                try:
                    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    peft_model.gradient_checkpointing_enable()
            for mod in peft_model.modules():
                if hasattr(mod, "gradient_checkpointing"):
                    mod.gradient_checkpointing = True
        else:
            peft_model.eval()
        return peft_model

    @abstractmethod
    def build_scalers(self, precision: PrecisionSpec, device: torch.device) -> tuple[Any | None, Any | None]:
        """Returns `(grad_scaler, manual_scaler)` appropriate for this backbone and precision."""

    def adjust_stage1_batch_config(
        self, stage1_cfg: dict[str, Any], precision: PrecisionSpec
    ) -> tuple[int, int, int]:
        """Returns `(batch_size, accum_steps, eval_batch_size)`."""
        bs1 = int(stage1_cfg.get("batch_size", 2))
        accum1 = int(stage1_cfg.get("gradient_accumulation_steps", 8))
        eval_bs = int(stage1_cfg.get("eval_batch_size", 32))
        return bs1, accum1, eval_bs

    def clamp_extraction_batch_size(self, batch_size: int) -> int:
        return batch_size

    def stage2_eval_batch_size(self) -> int:
        return 4

    def specialist_phase_tag(self, target_family: str) -> str:
        return f"P3_{target_family}_finetuned"

    def specialist_feature_set_tag(self, target_family: str) -> str:
        return f"F6_{target_family}_tapped"

    def specialist_notes_label(self) -> str:
        return self.family_name.capitalize()

    def maybe_run_pipeline(
        self,
        train_cfg: dict[str, Any],
        step_callback: Any | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Optional hook allowing a backbone adapter to handle specialized pipeline modes."""
        return None

    @contextmanager
    def export_context(self, backbone: nn.Module) -> Iterator[None]:
        """Context manager bracketing checkpoint export (default no-op)."""
        yield

    def default_layer_count_for_disk_estimate(self, model: nn.Module) -> int:
        """Dynamically infers total hidden state layer count for disk estimation."""
        bb = getattr(model, "backbone", model)
        cfg = getattr(bb, "config", None)
        n_hidden = getattr(cfg, "num_hidden_layers", None) if cfg is not None else None
        if n_hidden is not None:
            return int(n_hidden) + 1
        tapped = getattr(model, "layers", None)
        if tapped:
            return max(len(tapped), 1)
        return 1


_BACKBONE_REGISTRY: dict[str, type[BaseBackboneAdapter]] = {}


def register_backbone(name: str) -> Callable[[type[BaseBackboneAdapter]], type[BaseBackboneAdapter]]:
    """Class decorator registering a backbone adapter by canonical family name."""

    def _decorator(cls: type[BaseBackboneAdapter]) -> type[BaseBackboneAdapter]:
        _BACKBONE_REGISTRY[name.lower().strip()] = cls
        return cls

    return _decorator


def resolve_backbone_adapter(cfg: dict[str, Any] | None = None) -> BaseBackboneAdapter:
    """
    Resolves the concrete BaseBackboneAdapter for a configuration dict.
    Prefers explicit `backbone_family`; falls back ONLY here to inspecting `backbone`.
    """
    cfg = cfg or {}
    fam = cfg.get("backbone_family")
    if fam:
        key = str(fam).lower().strip()
        if key in _BACKBONE_REGISTRY:
            return _BACKBONE_REGISTRY[key]()

    b_name = str(cfg.get("backbone", "")).lower()
    if "deberta" in b_name and "deberta" in _BACKBONE_REGISTRY:
        return _BACKBONE_REGISTRY["deberta"]()
    if "bge" in b_name and "bge" in _BACKBONE_REGISTRY:
        return _BACKBONE_REGISTRY["bge"]()
    return _BACKBONE_REGISTRY["qwen"]()
