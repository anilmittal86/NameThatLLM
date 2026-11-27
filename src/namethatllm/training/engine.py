"""
Model-Agnostic Training Engine: Stage 1 & Stage 2 Loops, Resume Support, and Encoder Stage 1 Entrypoint.
"""

from __future__ import annotations

import copy
import functools
import gc
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_cosine_schedule_with_warmup

from ..ablation import _text_hash, build_unseen_holdout_dataset
from ..config import (
    FAMILIES,
    HARD_CLASS_INDICES,
    LAB2I,
    LABELS,
    K,
    resolve_all_pool_files,
    resolve_dataset_paths,
    softmax,
)
from ..experiment_logger import ExperimentLedger
from ..postprocess import sinkhorn_balancing
from .backbones import resolve_backbone_adapter
from .backbones.base import BaseBackboneAdapter, PrecisionSpec, TokenizationSpec
from .data import (
    PROBLEM_WATCH_CLASSES,
    BucketedLengthBatchSampler,
    PromptGroupBatchSampler,
    TextSequenceDataset,
    _load_benchmark_holdout,
    _select_stage1_pool_files,
    build_targeted_epoch_indices,
    collate_text_batch,
    load_clean_stage1_pool,
    load_zero_leak_stage1_pool,
    norm_text_hash,
)
from .embeddings import extract_and_export_embeddings
from .evaluation import (
    _blend_head,
    _evaluate_bench_checkpoint,
    compute_detailed_metrics,
    compute_grouped_inference_logits,
    evaluate_stage2_oof_cv,
    extract_features_and_logits,
    fit_ridge_heads,
    gpu_trace_ridge_eval,
    print_prediction_table,
)
from .gates import train_family_gate, train_family_gate_from_dataset
from .losses import compute_supcon_loss, family_margin_cross_entropy
from .modeling import (
    MultiLayerLoRAClassifier,
    _build_class_weight_tensor,
    _get_loraplus_param_groups,
    _step_with_grad_guard,
    export_compatible_checkpoint,
    find_compatible_champion_lora,
    load_and_slice_champion_heads,
)


def _setup_training_env() -> None:
    """Configures PyTorch memory allocator and neutralizes incompatible optional packages."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    orig_find_spec = importlib.util.find_spec

    def _safe_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torchao":
            return None
        return orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _safe_find_spec
    try:
        import peft.import_utils

        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass


_configure_training_environment = _setup_training_env


def train_prompt_group_step(
    model: MultiLayerLoRAClassifier,
    opt: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    micro_batch_size: int = 2,
    centered_weight: float = 0.5,
    family_margin: float = 0.35,
    supcon_weight: float = 0.20,
    label_smoothing: float = 0.10,
    accum_steps: int = 1,
    compute_dtype: torch.dtype = torch.float16,
    weight: torch.Tensor | None = None,
    drop_first_token: bool = True,
    scaler: Any | None = None,
    boost_hard_sisters: bool = True,
    manual_scaler: Any | None = None,
) -> tuple[float, float, float]:
    """Two-pass micro-batched training step for a single prompt group with singleton guard and FP16 loss scaling."""
    N = input_ids.shape[0]
    if N == 0:
        return 0.0, 0.0, 0.0
    dev = input_ids.device
    labels_dev = labels.to(dev)
    use_manual = manual_scaler is not None and getattr(manual_scaler, "is_enabled", lambda: False)()
    use_grad_scaler = (not use_manual) and scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    if N <= 1:
        model.train()
        mb_mask = attention_mask[:1]
        max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            Z_mb = model.forward_features(input_ids[:1, :max_l], mb_mask[:, :max_l], drop_first_token=drop_first_token)
            mb_dev = Z_mb.device
            logits_p = model.predict_plain(Z_mb)
            loss_p = family_margin_cross_entropy(
                logits_p,
                labels_dev[:1].to(mb_dev),
                family_margin=family_margin,
                label_smoothing=label_smoothing,
                weight=weight,
                boost_hard_sisters=boost_hard_sisters,
            )
            scaled_loss = loss_p / float(max(accum_steps, 1))
            loss_to_bw = manual_scaler.scale_loss(scaled_loss) if use_manual else scaled_loss
        if use_grad_scaler:
            scaler.scale(loss_to_bw).backward()
        else:
            loss_to_bw.backward()
        loss_p_val = float(loss_p.item())
        return loss_p_val, loss_p_val, 0.0

    was_training = model.training
    model.eval()
    z_chunks = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            for s in range(0, N, 4):
                e = min(s + 4, N)
                mb_mask = attention_mask[s:e]
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                z_c = model.forward_features(input_ids[s:e, :max_l], mb_mask[:, :max_l], drop_first_token=drop_first_token)
                z_chunks.append(z_c)
    Z_all_no_grad = torch.cat(z_chunks, dim=0)
    if N >= 8 and torch.rand(1).item() < 0.25:
        min_k = min(N, max(8, N // 2))
        k = int(torch.randint(min_k, N + 1, (1,)).item())
        sub_indices = torch.randperm(N, device=Z_all_no_grad.device)[:k]
        mu_prompt = Z_all_no_grad[sub_indices].mean(dim=0, keepdim=True)
    else:
        mu_prompt = Z_all_no_grad.mean(dim=0, keepdim=True)
    Z_group_centered = (Z_all_no_grad - mu_prompt).detach()
    del z_chunks

    total_loss_val = 0.0
    plain_loss_val = 0.0
    centered_loss_val = 0.0

    if was_training or not model.training:
        model.train()
    for s in range(0, N, micro_batch_size):
        e = min(s + micro_batch_size, N)
        mb_mask = attention_mask[s:e]
        max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
        mb_ids = input_ids[s:e, :max_l]
        mb_mask_slice = mb_mask[:, :max_l]

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=compute_dtype):
            Z_mb = model.forward_features(mb_ids, mb_mask_slice, drop_first_token=drop_first_token)
            mb_dev = Z_mb.device

            logits_p = model.predict_plain(Z_mb)
            loss_p = family_margin_cross_entropy(
                logits_p,
                labels_dev[s:e].to(mb_dev),
                family_margin=family_margin,
                label_smoothing=label_smoothing,
                weight=weight,
                boost_hard_sisters=boost_hard_sisters,
            )

            Z_c_mb = Z_mb - mu_prompt.to(mb_dev)
            logits_c = model.predict_centered(Z_c_mb)
            loss_c = family_margin_cross_entropy(
                logits_c,
                labels_dev[s:e].to(mb_dev),
                family_margin=family_margin,
                label_smoothing=label_smoothing,
                weight=weight,
                boost_hard_sisters=boost_hard_sisters,
            )

            loss_mb = loss_p + float(centered_weight) * loss_c
            if supcon_weight > 0.0:
                loss_sup = compute_supcon_loss(
                    Z_c_mb,
                    labels_dev[s:e],
                    temperature=0.10,
                    prototypes=model.centered_head.weight.detach(),
                    Z_group_centered=Z_group_centered,
                    group_slice=(s, e),
                )
                loss_mb = loss_mb + float(supcon_weight) * loss_sup

            weight_frac = (e - s) / float(N)
            scaled_loss = (loss_mb * weight_frac) / float(max(accum_steps, 1))
            loss_to_bw = manual_scaler.scale_loss(scaled_loss) if use_manual else scaled_loss

        if use_grad_scaler:
            scaler.scale(loss_to_bw).backward()
        else:
            loss_to_bw.backward()

        total_loss_val += loss_mb.item() * weight_frac
        plain_loss_val += loss_p.item() * weight_frac
        centered_loss_val += loss_c.item() * weight_frac

    return total_loss_val, plain_loss_val, centered_loss_val


def _load_pool_and_bench_for_gate(
    input_dir: Path,
    all_pools: list[Path],
    pool_file: Path | None,
    train_path: Path | None,
    meta_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str] | None, list[int] | None]:
    """Loads pool, gold train, and clean benchmark dataframe for export or gate modes."""
    pool_list = all_pools if all_pools else ([pool_file] if pool_file else [])
    dfs = []
    for p in pool_list:
        try:
            df_one = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
            keep_cols = [c for c in meta_cols if c in df_one.columns]
            dfs.append(df_one[keep_cols])
        except Exception:
            pass
    df_p = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["text", "model"])

    df_tr = None
    if train_path and Path(train_path).exists():
        try:
            df_tr = pd.read_csv(train_path)
        except Exception:
            pass

    eval_texts, eval_labels = None, None
    bench_paths = list(input_dir.rglob("clean_benchmark_1800_rows.parquet")) if input_dir.exists() else []
    if bench_paths:
        try:
            df_bench = pd.read_parquet(bench_paths[0])
            eval_texts = df_bench["text"].tolist()
            eval_labels = [LAB2I[m] for m in df_bench["model"].tolist()]
            print(f"[Trainer] Loaded {len(eval_texts)} benchmark samples from {bench_paths[0]}")
        except Exception as e:
            print(f"[Trainer] Note on reading clean benchmark: {e}")
    return df_p, df_tr, eval_texts, eval_labels


def _load_champion_adapter_and_heads(
    model: MultiLayerLoRAClassifier,
    input_dir: Path,
    backbone_name: str,
    lora_r_val: int,
    target_family: str | None,
    merge_adapter: bool,
    fail_on_error: bool,
    adapter_obj: BaseBackboneAdapter,
    precision: PrecisionSpec,
    train_cfg: dict[str, Any],
) -> None:
    """Finds and loads compatible champion LoRA adapter and head weights for extraction/gate pipelines."""
    adapter_found = find_compatible_champion_lora(
        input_dir,
        backbone_name=backbone_name,
        lora_r=lora_r_val,
        target_family=target_family or "",
    )
    if adapter_found is not None:
        try:
            cand_files = [
                adapter_found / "adapter_model.safetensors",
                adapter_found / "adapter" / "adapter_model.safetensors",
                adapter_found / "adapter_model.bin",
            ]
            adapter_file = next((p for p in cand_files if p.exists()), cand_files[0])
            adapter_sd = load_file(str(adapter_file)) if str(adapter_file).endswith(".safetensors") else torch.load(str(adapter_file), map_location="cpu", weights_only=True)

            model_sd = model.backbone.state_dict()
            mismatches = [
                f"{k_sd}: checkpoint {v_sd.shape} vs model {model_sd[k_sd].shape}"
                for k_sd, v_sd in adapter_sd.items()
                if k_sd in model_sd and model_sd[k_sd].shape != v_sd.shape
            ]
            if mismatches:
                raise ValueError(f"Shape mismatches in adapter {adapter_found}: {mismatches[:3]}...")

            set_peft_model_state_dict(model.backbone, adapter_sd)
            print(f"[Trainer] Loaded champion LoRA weights from {adapter_found}.")
            if merge_adapter and hasattr(model.backbone, "merge_and_unload"):
                try:
                    model.backbone = model.backbone.merge_and_unload()
                    print("[Trainer] Successfully merged and unloaded champion LoRA weights into base model.")
                except Exception as e:
                    print(f"[Trainer] Note on merge_and_unload: {e}")
        except Exception as e:
            print(f"[Trainer] Error loading champion adapter weights: {e}")
            if fail_on_error:
                raise RuntimeError(f"Failed to load champion LoRA adapter from {adapter_found}: {e}") from e
    elif fail_on_error and input_dir.exists():
        found_all = list(input_dir.rglob("adapter_model.safetensors"))
        if found_all:
            raise RuntimeError(f"Discovered adapters on disk but none were compatible with {backbone_name} (r={lora_r_val}): {found_all}")

    head_candidates: list[Path] = []
    if adapter_found is not None:
        head_candidates.extend([
            adapter_found / "heads.safetensors",
            adapter_found.parent / "heads.safetensors",
            adapter_found / "heads.pt",
            adapter_found.parent / "heads.pt",
        ])
    all_heads = (list(input_dir.rglob("heads.safetensors")) + list(input_dir.rglob("heads.pt"))) if input_dir.exists() else []
    priority_terms = ("champion", "inhouse", "stage1", "backbone", "18class", "full")
    head_candidates.extend([c for c in all_heads if any(term in str(c).lower() for term in priority_terms)])
    head_candidates.extend(all_heads)

    for cand in head_candidates:
        if cand.exists() and load_and_slice_champion_heads(cand, model, fam_cols=None):
            print(f"[Trainer] Loaded champion norms & heads from {cand}")
            break


def _resume_checkpoint(
    model: MultiLayerLoRAClassifier,
    resume_ckpt: str | Path,
    target_family: str | None,
    is_specialist: bool,
) -> None:
    """Restores adapter weights and verified head tensors from a compatible checkpoint directory."""
    resume_dir = Path(str(resume_ckpt))
    adapter_candidates = [
        resume_dir / "adapter" / "adapter_model.safetensors",
        resume_dir / "adapter_model.safetensors",
        resume_dir / "adapter" / "adapter_model.bin",
        resume_dir / "adapter_model.bin",
    ]
    heads_candidates = [
        resume_dir / "heads.safetensors",
        resume_dir / "heads.pt",
    ]
    adapter_file = next((p for p in adapter_candidates if p.exists()), None)
    heads_file = next((p for p in heads_candidates if p.exists()), None)
    if not resume_dir.exists() or adapter_file is None or heads_file is None:
        raise FileNotFoundError(
            f"resume_from_checkpoint={resume_ckpt!r} must contain adapter/ (or adapter_model.safetensors) and heads.safetensors"
        )
    if str(adapter_file).endswith(".safetensors"):
        adapter_sd = load_file(str(adapter_file))
    else:
        adapter_sd = torch.load(str(adapter_file), map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model.backbone, adapter_sd)
    heads_ok = load_and_slice_champion_heads(
        heads_file,
        model,
        fam_cols=FAMILIES[target_family] if is_specialist and target_family in FAMILIES else None,
    )
    if not heads_ok:
        raise RuntimeError(f"Failed to load compatible heads from {heads_file}")
    print(f"[Trainer] Resumed checkpoint from {resume_dir} (adapter={adapter_file.name}, heads={heads_file.name}).")


def _run_stage1(
    model: MultiLayerLoRAClassifier,
    tok: Any,
    tok_spec: TokenizationSpec,
    adapter: BaseBackboneAdapter,
    precision: PrecisionSpec,
    device: torch.device,
    train_cfg: dict[str, Any],
    stage1_cfg: dict[str, Any],
    input_dir: Path,
    out_dir: Path,
    train_path: Path | None,
    pool_file: Path | None,
    all_pools: list[Path],
    bench_bundle: tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray] | None,
    active_labels: list[str],
    sub_lab2i: dict[str, int],
    layers: list[int],
    keep_layers: int,
    backbone_name: str,
    target_family: str | None,
    is_specialist: bool,
    family_margin: float,
    milestone_history: list[dict[str, Any]],
    step_callback: Any | None = None,
) -> None:
    """Executes Stage 1 pool pretraining with per-class weighting, opt-step milestones, and clean budget exit."""
    max_stage1_hours = float(stage1_cfg.get("max_hours", 12.0))
    s1_start = time.time()
    max_s1_sec = max_stage1_hours * 3600.0
    raw_pool_list = all_pools if all_pools else ([pool_file] if pool_file else [])
    use_v2_only = bool(stage1_cfg.get("use_v2_only", True))
    pool_list = _select_stage1_pool_files(raw_pool_list, use_v2_only=use_v2_only)
    print(f"[Trainer] Stage 1 Pretraining on {len(pool_list)} pool file(s) (use_v2_only={use_v2_only}): {[Path(p).name for p in pool_list]}")

    use_mw = bool(stage1_cfg.get("use_match_weight", True))
    mw_clip = tuple(stage1_cfg.get("match_weight_clip", (0.5, 2.5)))
    df_p, pool_stats = load_clean_stage1_pool(
        pool_list=pool_list,
        input_dir=input_dir,
        train_path=train_path,
        bench_bundle=bench_bundle,
        active_labels=active_labels,
        exclude_train_overlap=bool(stage1_cfg.get("exclude_train_overlap", True)),
        weight_clip=mw_clip,
        use_match_weight=use_mw,
        min_char_len=40,
        include_unseen_holdout=True,
    )

    word_lens = [len(t.split()) for t in df_p["text"]]
    tok_lens = [min(len(t) // 4 + 1, tok_spec.max_len) for t in df_p["text"]]
    class_dist = df_p["model"].value_counts().to_dict()

    def _calc_percentiles(vals: list[int]) -> dict[str, float]:
        if not vals:
            return {}
        v = np.array(vals)
        return {k: float(np.percentile(v, p)) for k, p in (("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("p99", 99))}

    audit_report = {
        "source_files": pool_stats["source_counts"],
        "total_raw_rows": pool_stats["total_raw_rows"],
        "deduplicated_rows": pool_stats["dedup_rows"],
        "quarantined_holdout_rows": pool_stats["quarantine_dropped"],
        "retained_pool_rows": int(len(df_p)),
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "word_length_percentiles": _calc_percentiles(word_lens),
        "token_length_percentiles": _calc_percentiles(tok_lens),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_file = out_dir / "data_audit_report.json"
    audit_file.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")

    print("\n==========================================================================================")
    print("                         SACRED DATA INTEGRITY AUDIT REPORT                               ")
    print("==========================================================================================")
    for fname, cnt in pool_stats["source_counts"].items():
        print(f"   - {fname:<32}: {cnt:>8,} rows")
    print(f" Clean Retained Pool Rows: {len(df_p):,} | Quarantined: {pool_stats['quarantine_dropped']:,}")
    for lbl in active_labels:
        c_cnt = class_dist.get(lbl, 0)
        print(f"   - {lbl:<22}: {c_cnt:>6,} ({(c_cnt / max(len(df_p), 1)) * 100:5.2f}%)")
    print("==========================================================================================\n")

    epochs1 = int(stage1_cfg.get("epochs", 1))
    num_classes = len(active_labels)
    max_s = int(stage1_cfg.get("max_samples", 2000 * num_classes if is_specialist else 150000))
    explicit_quotas = stage1_cfg.get("class_quotas")
    oversample_to_target = bool(stage1_cfg.get("oversample_to_target", explicit_quotas is not None))
    present_labels = [lbl for lbl in active_labels if class_dist.get(lbl, 0) > 0]
    present_lab2i = {lbl: idx for idx, lbl in enumerate(present_labels)}
    present_y_arr = np.array([present_lab2i[m] for m in df_p["model"]], dtype=np.int64)

    hard_set = set(HARD_CLASS_INDICES)
    if explicit_quotas:
        quotas_map = {lbl: int(explicit_quotas.get(lbl, max_s // max(len(present_labels), 1))) for lbl in present_labels}
        if "max_samples" in stage1_cfg:
            quota_sum = sum(int(v) for v in explicit_quotas.values())
            if quota_sum != int(stage1_cfg["max_samples"]):
                print(
                    f"[Trainer Warning] stage1.class_quotas sum ({quota_sum}) != stage1.max_samples ({int(stage1_cfg['max_samples'])}); using class_quotas."
                )
        epoch_plans = [
            build_targeted_epoch_indices(present_y_arr, quotas_map, epoch=ep, seed=42, class_labels=present_labels)
            for ep in range(epochs1)
        ]
    elif len(df_p) > max_s or oversample_to_target:
        cls_multipliers = {lbl: (1.30 if LAB2I.get(lbl) in hard_set else 1.0) for lbl in present_labels}
        total_multiplier = max(sum(cls_multipliers.values()), 1e-6)
        quotas_map = {lbl: max(1, int(round(max_s * (cls_multipliers[lbl] / total_multiplier)))) for lbl in present_labels}
        epoch_plans = [
            build_targeted_epoch_indices(present_y_arr, quotas_map, epoch=ep, seed=42, class_labels=present_labels)
            for ep in range(epochs1)
        ]
    else:
        epoch_plans = [np.random.RandomState(42 + ep).permutation(len(df_p)) for ep in range(epochs1)]

    print("[Trainer Stage 1] Epoch-1 Scheduled Sequence Counts:", {lbl: int((df_p.iloc[epoch_plans[0]]["model"] == lbl).sum()) for lbl in present_labels})

    bs1, accum1, bench_eval_bs = adapter.adjust_stage1_batch_config(stage1_cfg, precision)
    collate_fn_s1 = functools.partial(collate_text_batch, pad_token_id=tok_spec.pad_id)

    lora_ratio1 = float(stage1_cfg.get("lora_plus_ratio", 2.5))
    param_groups1 = _get_loraplus_param_groups(
        model,
        lr_lora=float(stage1_cfg.get("lr_lora", 1.5e-4)),
        lr_head=float(stage1_cfg.get("lr_head", 8e-4)),
        lora_plus_ratio=lora_ratio1,
        include_centered_head=False,
    )
    opt1 = torch.optim.AdamW(param_groups1, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    total_steps1 = sum(math.ceil(len(ep_idx) / bs1) for ep_idx in epoch_plans)
    total_opt_steps1 = max(1, sum(math.ceil(math.ceil(len(ep_idx) / bs1) / accum1) for ep_idx in epoch_plans))
    warmup1 = min(int(stage1_cfg.get("warmup_steps", 65)), max(1, total_opt_steps1 // 5))
    sched1 = get_cosine_schedule_with_warmup(opt1, num_warmup_steps=warmup1, num_training_steps=total_opt_steps1)

    stage1_class_weights = _build_class_weight_tensor(
        active_labels,
        sub_lab2i,
        stage1_cfg,
        default_hard_weight=float(train_cfg.get("hard_class_weight", 1.0)),
        device=device,
    )
    boost_hard = bool(train_cfg.get("boost_hard_sisters", True))
    s1_supcon_w = float(stage1_cfg.get("supcon_weight", 0.0))
    s1_label_smooth = float(stage1_cfg.get("label_smoothing", 0.0))

    scaler1, manual_scaler1 = adapter.build_scalers(precision, device)
    if manual_scaler1 is not None and manual_scaler1.is_enabled():
        print(f"[Trainer Stage 1] Enabled _ManualFP16LossScaler (initial scale={manual_scaler1.scale:.1f})")
    eval_steps1 = int(stage1_cfg.get("eval_steps", 0))
    save_steps = int(stage1_cfg.get("save_steps", 500))
    best_s1_bacc = -1.0
    best_s1_state: dict[str, torch.Tensor] | None = None

    def _snapshot_trainable() -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    model.train()
    opt1.zero_grad(set_to_none=True)
    global_step1 = 0
    opt_step1 = 0
    skipped_steps1 = 0
    stop_stage1 = False
    for ep in range(epochs1):
        if stop_stage1:
            break
        df_ep = df_p.iloc[epoch_plans[ep]].reset_index(drop=True)
        ep_lengths = [min(len(t) // 4 + 1, tok_spec.max_len) for t in df_ep["text"]]
        ep_weights = df_ep["_sample_weight"].astype(float).tolist() if "_sample_weight" in df_ep.columns else None
        ds_ep = TextSequenceDataset(
            df_ep["text"].tolist(),
            [sub_lab2i[m] for m in df_ep["model"]],
            None,
            tok,
            tok_spec=tok_spec,
            sample_weights=ep_weights,
        )
        bucket_sampler = BucketedLengthBatchSampler(ep_lengths, batch_size=bs1, shuffle=True, seed=42 + ep)
        bucket_sampler.set_epoch(ep)
        loader1 = DataLoader(ds_ep, batch_sampler=bucket_sampler, collate_fn=collate_fn_s1)
        n_batches1 = len(loader1)
        opt1.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(loader1):
            global_step1 += 1
            mb_ids = batch["input_ids"].to(device)
            mb_mask = batch["attention_mask"].to(device)
            max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
            mb_ids = mb_ids[:, :max_l]
            mb_mask = mb_mask[:, :max_l]
            labels_dev = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=precision.compute_dtype):
                Z_s1, logits_p, _ = model(mb_ids, mb_mask, drop_first_token=tok_spec.drop_first_token)
                sw_s1 = batch.get("sample_weights")
                loss_ce = family_margin_cross_entropy(
                    logits_p,
                    labels_dev,
                    family_margin=family_margin,
                    label_smoothing=s1_label_smooth,
                    weight=stage1_class_weights,
                    boost_hard_sisters=boost_hard,
                    sample_weight=sw_s1,
                )
                loss_sup = (
                    compute_supcon_loss(Z_s1, labels_dev, temperature=0.10, prototypes=model.plain_head.weight.detach())
                    if s1_supcon_w > 0.0
                    else (Z_s1 * 0.0).sum()
                )
                eff_accum = (
                    min(accum1, n_batches1 - (step_idx - (step_idx % accum1)))
                    if (step_idx // accum1) == ((n_batches1 - 1) // accum1) and (n_batches1 % accum1 != 0)
                    else accum1
                )
                loss = (loss_ce + s1_supcon_w * loss_sup) / max(eff_accum, 1)

            if scaler1 is not None and getattr(scaler1, "is_enabled", lambda: False)():
                scaler1.scale(loss).backward()
            elif manual_scaler1 is not None and manual_scaler1.is_enabled():
                manual_scaler1.scale_loss(loss).backward()
            else:
                loss.backward()

            did_step_now = (step_idx + 1) % accum1 == 0 or (step_idx + 1) == n_batches1
            if did_step_now:
                did_step = _step_with_grad_guard(
                    model,
                    opt1,
                    sched1,
                    scaler=scaler1,
                    manual_scaler=manual_scaler1,
                    max_grad_norm=1.0,
                )
                skipped_steps1 += int(not did_step)
                opt_step1 += 1

                if bench_bundle is not None and eval_steps1 > 0 and opt_step1 % eval_steps1 == 0 and (total_opt_steps1 - opt_step1) > 25:
                    _, _, sub_texts, sub_y, _ = bench_bundle
                    m_sub, _ = _evaluate_bench_checkpoint(
                        model=model,
                        tok=tok,
                        texts=sub_texts,
                        y_true=sub_y,
                        max_len=tok_spec.max_len,
                        device=device,
                        compute_dtype=precision.compute_dtype,
                        drop_first_token=tok_spec.drop_first_token,
                        tag=f"S1_OptStep{opt_step1}",
                        end_id=tok_spec.end_id,
                        cls_id=tok_spec.cls_id,
                        pad_id=tok_spec.pad_id,
                        batch_size=bench_eval_bs,
                        out_dir=out_dir,
                        history=milestone_history,
                    )
                    last_sub_bacc = float(m_sub["bacc"])
                    if last_sub_bacc > best_s1_bacc:
                        best_s1_bacc = last_sub_bacc
                        best_s1_state = _snapshot_trainable()
                        with adapter.export_context(model.backbone):
                            export_compatible_checkpoint(
                                model=model,
                                output_dir=out_dir / "stage1_best",
                                layers=layers,
                                keep=keep_layers,
                                max_len=tok_spec.max_len,
                                end_id=tok_spec.end_id if tok_spec.end_id is not None else 0,
                                backbone_name=backbone_name,
                                class_labels=active_labels,
                                target_family=target_family if is_specialist else None,
                            )
                        print(f"[Trainer Stage 1] Saved new best milestone checkpoint (Sub-Bench BAcc={best_s1_bacc*100:.2f}%)")

                if save_steps > 0 and opt_step1 % save_steps == 0:
                    try:
                        with adapter.export_context(model.backbone):
                            export_compatible_checkpoint(
                                model=model,
                                output_dir=out_dir,
                                layers=layers,
                                keep=keep_layers,
                                max_len=tok_spec.max_len,
                                end_id=tok_spec.end_id if tok_spec.end_id is not None else 0,
                                backbone_name=backbone_name,
                                class_labels=active_labels,
                                target_family=target_family if is_specialist else None,
                            )
                        print(f"[Trainer Stage 1] Intermediate checkpoint auto-saved at opt step {opt_step1} (global {global_step1}) to {out_dir}")
                        if step_callback is not None:
                            step_callback("checkpoint_saved", opt_step1)
                    except Exception as e:
                        print(f"[Trainer Stage 1 Warning] Auto-save failed at opt step {opt_step1}: {e}")

            if global_step1 % 200 == 0:
                elapsed_m = (time.time() - s1_start) / 60.0
                lr_now = sched1.get_last_lr()[0] if hasattr(sched1, "get_last_lr") else 0.0
                scale_str = f" | Scale: {manual_scaler1.scale:.0f}" if (manual_scaler1 is not None and manual_scaler1.is_enabled()) else ""
                print(
                    f"[Trainer Stage 1] Epoch {ep + 1}/{epochs1} Step {step_idx + 1}/{n_batches1} (Opt {opt_step1}/{total_opt_steps1}, Global {global_step1}/{total_steps1}) "
                    f"| Loss: {loss.item() * max(eff_accum, 1):.4f} | LR: {lr_now:.2e}{scale_str} | Skipped: {skipped_steps1} | Elapsed: {elapsed_m:.1f}m"
                )

            if (time.time() - s1_start) >= max_s1_sec:
                print(
                    f"[Trainer Stage 1] Reached {stage1_cfg.get('max_hours', 12.0)}h time budget at opt step {opt_step1} (global step {global_step1}); discarding partial accum window and advancing to Stage 2."
                )
                opt1.zero_grad(set_to_none=True)
                stop_stage1 = True
                break

        del ds_ep, loader1, bucket_sampler

    opt1.zero_grad(set_to_none=True)
    if bench_bundle is not None:
        full_texts, full_y, _, sub_y, sub_idx_arr = bench_bundle
        _, full_lg = _evaluate_bench_checkpoint(
            model=model,
            tok=tok,
            texts=full_texts,
            y_true=full_y,
            max_len=tok_spec.max_len,
            device=device,
            compute_dtype=precision.compute_dtype,
            drop_first_token=tok_spec.drop_first_token,
            tag="S1_End_Full",
            end_id=tok_spec.end_id,
            cls_id=tok_spec.cls_id,
            pad_id=tok_spec.pad_id,
            batch_size=bench_eval_bs,
            out_dir=out_dir,
            history=milestone_history,
        )
        sub_idx_arr = np.asarray(sub_idx_arr, dtype=np.int64)
        end_sub_bacc = float(balanced_accuracy_score(sub_y, full_lg[sub_idx_arr].argmax(axis=1)))
        if best_s1_state is not None and bool(stage1_cfg.get("restore_best_before_stage2", True)) and best_s1_bacc > end_sub_bacc + 0.005:
            print(
                f"[Trainer Stage 1] Restoring best milestone weights (Sub BAcc={best_s1_bacc*100:.2f}% > End Sub BAcc={end_sub_bacc*100:.2f}%) before Stage 2."
            )
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in best_s1_state:
                        p.data.copy_(best_s1_state[n].to(p.device, dtype=p.dtype))
    del opt1, sched1, df_p, best_s1_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        with adapter.export_context(model.backbone):
            export_compatible_checkpoint(
                model=model,
                output_dir=out_dir,
                layers=layers,
                keep=keep_layers,
                max_len=tok_spec.max_len,
                end_id=tok_spec.end_id if tok_spec.end_id is not None else 0,
                backbone_name=backbone_name,
                class_labels=active_labels,
                target_family=target_family if is_specialist else None,
            )
        print(f"[Trainer] Safe intermediate Stage 1 checkpoint exported to {out_dir}")
        if step_callback is not None:
            step_callback("checkpoint_saved", "stage1_finished")
    except Exception as e:
        print(f"[Trainer] Warning: could not export Stage 1 intermediate checkpoint: {e}")


def _run_stage2(
    model: MultiLayerLoRAClassifier,
    tok: Any,
    tok_spec: TokenizationSpec,
    adapter: BaseBackboneAdapter,
    precision: PrecisionSpec,
    device: torch.device,
    train_cfg: dict[str, Any],
    stage2_cfg: dict[str, Any],
    train_path: Path | None,
    out_dir: Path,
    bench_bundle: tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray] | None,
    active_labels: list[str],
    sub_lab2i: dict[str, int],
    layers: list[int],
    keep_layers: int,
    backbone_name: str,
    target_family: str | None,
    is_specialist: bool,
    family_margin: float,
    supcon_weight: float,
    step_callback: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Executes Stage 2 prompt-grouped contrastive fine-tuning, SWA averaging, and OOF/Ridge calibration."""
    if bool(stage2_cfg.get("freeze_backbone", False)):
        print("[Trainer] Freezing backbone parameters for Stage 2 (exclusively training norms, plain_head, and centered_head)...")
        for p in model.backbone.parameters():
            p.requires_grad = False

    df_tr = pd.read_csv(train_path)
    if bench_bundle is not None:
        bench_norm_set = {norm_text_hash(t) for t in bench_bundle[0]}
        before_s2 = len(df_tr)
        df_tr = df_tr[~df_tr["text"].astype(str).map(norm_text_hash).isin(bench_norm_set)].reset_index(drop=True)
        if len(df_tr) < before_s2:
            print(f"[Trainer Stage 2] Quarantined {before_s2 - len(df_tr)} overlapping benchmark rows from train.csv.")
    if is_specialist:
        df_tr = df_tr[df_tr["model"].isin(active_labels)].reset_index(drop=True)
    num_classes = len(active_labels)
    print(f"[Trainer] Stage 2 Fine-Tuning on {train_path} ({len(df_tr)} samples, {num_classes} classes)...")

    bs2 = int(stage2_cfg.get("batch_size", 2))
    accum2 = int(stage2_cfg.get("gradient_accumulation_steps", 1))
    collate_fn = functools.partial(collate_text_batch, pad_token_id=tok_spec.pad_id)
    ds2 = TextSequenceDataset(
        df_tr["text"].tolist(),
        [sub_lab2i[m] for m in df_tr["model"]],
        df_tr["prompt_id"].tolist(),
        tok,
        tok_spec=tok_spec,
    )

    if bool(stage2_cfg.get("enabled", True)):
        if bool(stage2_cfg.get("warmstart_centered_from_plain", bool(train_cfg.get("fresh_init", True)))):
            with torch.no_grad():
                w_p_init = model.plain_head.weight.detach().float()
                row_norms = w_p_init.norm(dim=1, keepdim=True).clamp(min=1e-6)
                target_norm = float(stage2_cfg.get("centered_init_norm", 0.65))
                w_c_init = (w_p_init / row_norms) * target_norm
                model.centered_head.weight.copy_(w_c_init.to(model.centered_head.weight.device, dtype=model.centered_head.weight.dtype))
                model.centered_head.bias.zero_()
            print(f"[Trainer Stage 2] Warm-started centered_head from unit-normalized plain_head directions (row_norm={target_norm:.2f}).")

        sampler2 = PromptGroupBatchSampler(df_tr["prompt_id"].tolist(), batch_size=bs2, shuffle=True)
        loader2 = DataLoader(ds2, batch_sampler=sampler2, collate_fn=collate_fn)
        n_batches2 = len(loader2)

        lora_ratio2 = float(stage2_cfg.get("lora_plus_ratio", 2.5))
        param_groups2 = _get_loraplus_param_groups(
            model,
            lr_lora=float(stage2_cfg.get("lr_lora", 3e-5)),
            lr_head=float(stage2_cfg.get("lr_head", 2e-4)),
            lora_plus_ratio=lora_ratio2,
            include_centered_head=True,
        )
        opt2 = torch.optim.AdamW(param_groups2, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
        epochs2 = int(stage2_cfg.get("epochs", 2))
        total_steps2 = max(1, n_batches2 * epochs2)
        total_opt_steps2 = max(1, epochs2 * max(1, math.ceil(n_batches2 / max(accum2, 1))))
        warmup2 = min(int(stage2_cfg.get("warmup_steps", 20)), max(1, total_opt_steps2 // 4))
        sched2 = get_cosine_schedule_with_warmup(opt2, num_warmup_steps=warmup2, num_training_steps=total_opt_steps2)

        swa_enabled = bool(stage2_cfg.get("swa", True))
        swa_start_frac = float(stage2_cfg.get("swa_start_frac", 0.5))
        swa_min_step = 0 if n_batches2 < 4 else int(n_batches2 * swa_start_frac)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        swa_shadow = {p: torch.zeros_like(p.data, dtype=torch.float32, device="cpu") for p in trainable_params} if swa_enabled else {}
        swa_steps = 0

        stage2_class_weights = _build_class_weight_tensor(
            active_labels,
            sub_lab2i,
            stage2_cfg,
            default_hard_weight=float(stage2_cfg.get("hard_class_weight", 1.0)),
            device=device,
        )
        supcon_s2_target = float(stage2_cfg.get("supcon_weight", supcon_weight))
        supcon_warmup_batches = max(1, int(total_steps2 * float(stage2_cfg.get("supcon_warmup_frac", 0.25))))
        s2_label_smooth = float(stage2_cfg.get("label_smoothing", 0.05 if is_specialist else 0.10))
        s2_centered_w = float(stage2_cfg.get("centered_weight", 0.50))

        scaler2, manual_scaler2 = adapter.build_scalers(precision, device)
        if manual_scaler2 is not None and manual_scaler2.is_enabled():
            print(f"[Trainer Stage 2] Enabled _ManualFP16LossScaler (initial scale={manual_scaler2.scale:.1f})")
        s2_boost_hard = bool(train_cfg.get("boost_hard_sisters", True))
        model.train()
        global_step2 = 0
        skipped_steps2 = 0
        s2_start = time.time()
        for ep in range(epochs2):
            sampler2.set_epoch(ep)
            opt2.zero_grad(set_to_none=True)
            for step_idx, batch in enumerate(loader2):
                global_step2 += 1
                win_start = step_idx - (step_idx % max(accum2, 1))
                eff_accum2 = min(accum2, n_batches2 - win_start) if (accum2 > 1 and win_start + accum2 > n_batches2) else max(accum2, 1)
                sup_ramp = min(1.0, float(global_step2) / float(supcon_warmup_batches))
                _, loss2_p, loss2_c = train_prompt_group_step(
                    model=model,
                    opt=opt2,
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    micro_batch_size=bs2,
                    centered_weight=s2_centered_w,
                    family_margin=family_margin,
                    supcon_weight=supcon_s2_target * sup_ramp,
                    label_smoothing=s2_label_smooth,
                    accum_steps=eff_accum2,
                    compute_dtype=precision.compute_dtype,
                    weight=stage2_class_weights,
                    drop_first_token=tok_spec.drop_first_token,
                    scaler=scaler2,
                    boost_hard_sisters=s2_boost_hard,
                    manual_scaler=manual_scaler2,
                )
                if (step_idx + 1) % max(accum2, 1) == 0 or (step_idx + 1) == n_batches2:
                    did_step = _step_with_grad_guard(
                        model, opt2, sched2, scaler=scaler2, manual_scaler=manual_scaler2, max_grad_norm=1.0
                    )
                    skipped_steps2 += int(not did_step)
                    if did_step and swa_enabled and ep == (epochs2 - 1) and step_idx >= swa_min_step:
                        swa_steps += 1
                        for p in trainable_params:
                            swa_shadow[p].add_(p.data.detach().float().cpu())

                if global_step2 % 25 == 0:
                    elapsed_s2_m = (time.time() - s2_start) / 60.0
                    lr2_now = sched2.get_last_lr()[0] if hasattr(sched2, "get_last_lr") else 0.0
                    scale2_str = f" | Scale: {manual_scaler2.scale:.0f}" if (manual_scaler2 is not None and manual_scaler2.is_enabled()) else ""
                    print(
                        f"[Trainer Stage 2] Epoch {ep + 1}/{epochs2} Group {step_idx + 1}/{n_batches2} (Global {global_step2}/{total_steps2}) "
                        f"| Plain: {loss2_p:.4f} | Centered: {loss2_c:.4f} | LR: {lr2_now:.2e}{scale2_str} "
                        f"| Skipped: {skipped_steps2} | Elapsed: {elapsed_s2_m:.1f}m"
                    )

        if swa_enabled and swa_steps > 0:
            print(f"[Trainer] Applying SWA weight averaging across {swa_steps} steps in final Stage 2 epoch...")
            for p in trainable_params:
                p.data.copy_((swa_shadow[p] / float(swa_steps)).to(p.device, p.dtype))
            del swa_shadow

        del loader2, opt2, sched2, sampler2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            with adapter.export_context(model.backbone):
                export_compatible_checkpoint(
                    model=model,
                    output_dir=out_dir,
                    layers=layers,
                    keep=keep_layers,
                    max_len=tok_spec.max_len,
                    end_id=tok_spec.end_id if tok_spec.end_id is not None else 0,
                    backbone_name=backbone_name,
                    class_labels=active_labels,
                    target_family=target_family if is_specialist else None,
                )
            print(f"[Trainer] Saved intermediate Stage 2 SWA checkpoint to {out_dir}")
            if step_callback is not None:
                step_callback("checkpoint_saved", "stage2_swa")
        except Exception as e:
            print(f"[Trainer] Warning saving Stage 2 intermediate checkpoint: {e}")

    print("[Trainer] Extracting Stage 2 embeddings for calibration & 5-Stage OOF Cross-Validation...")
    model.eval()
    loader_eval = DataLoader(ds2, batch_size=adapter.stage2_eval_batch_size(), shuffle=False, collate_fn=collate_fn)
    Z_list = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=precision.compute_dtype):
            for batch in loader_eval:
                b_ids = batch["input_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                max_l = max(int(b_mask.sum(dim=1).max().item()), 2)
                z_b = model.forward_features(b_ids[:, :max_l], b_mask[:, :max_l], drop_first_token=tok_spec.drop_first_token)
                Z_list.append(z_b.float().cpu().numpy())
    Z_train = np.vstack(Z_list)
    y_train = np.array([sub_lab2i[m] for m in df_tr["model"]], dtype=np.int64)
    prompts_train = df_tr["prompt_id"].to_numpy()

    del loader_eval, ds2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    default_ridge_w = 0.35 if is_specialist else 0.0
    default_alpha = 8.0 if is_specialist else 10.0
    default_ls = 0.05 if is_specialist else 0.10

    w_ridge_p = float(stage2_cfg.get("plain_ridge_weight", default_ridge_w))
    w_ridge_c = float(stage2_cfg.get("centered_ridge_weight", default_ridge_w))
    if w_ridge_p > 0.0 or w_ridge_c > 0.0:
        ridge_alpha = float(stage2_cfg.get("ridge_alpha", default_alpha))
        ridge_ls = float(stage2_cfg.get("ridge_label_smoothing", stage2_cfg.get("label_smoothing", default_ls)))
        W_p, b_p, W_c, b_c = fit_ridge_heads(Z_train, y_train, prompts_train, alpha=ridge_alpha, label_smoothing=ridge_ls, num_classes=num_classes)
        if w_ridge_p > 0.0:
            _blend_head(model.plain_head, W_p, b_p, w_neural=1.0 - w_ridge_p, w_ridge=w_ridge_p)
        if w_ridge_c > 0.0:
            _blend_head(model.centered_head, W_c, b_c, w_neural=1.0 - w_ridge_c, w_ridge=w_ridge_c)

    oof_report: dict[str, Any] = {}
    if bool(stage2_cfg.get("check_overfitting", True)) and len(np.unique(prompts_train)) >= 2:
        try:
            oof_report = evaluate_stage2_oof_cv(
                Z_train,
                y_train,
                prompts_train,
                n_splits=int(stage2_cfg.get("n_splits", 5)),
                alpha=float(stage2_cfg.get("cv_alpha", default_alpha)),
                label_smoothing=float(stage2_cfg.get("cv_label_smoothing", default_ls)),
                num_classes=num_classes,
                class_labels=active_labels,
                output_dir=out_dir,
            )
        except Exception as e:
            print(f"[Stage 2 CV Warning] OOF cross-validation failed ({e}); continuing to checkpoint export.")

    with adapter.export_context(model.backbone):
        export_compatible_checkpoint(
            model=model,
            output_dir=out_dir,
            layers=layers,
            keep=keep_layers,
            max_len=tok_spec.max_len,
            end_id=tok_spec.end_id if tok_spec.end_id is not None else 0,
            backbone_name=backbone_name,
            Z_train=Z_train,
            y_train=y_train,
            prompts_train=prompts_train,
            class_labels=active_labels,
            target_family=target_family if is_specialist else None,
        )
    return Z_train, oof_report


def run_training_pipeline(
    train_cfg: dict[str, Any] | None = None,
    step_callback: Any | None = None,
) -> Any:
    """Executes the unified multi-layer LoRA training pipeline delegated by backbone strategy adapter."""
    _setup_training_env()
    train_cfg = train_cfg or {}

    adapter = resolve_backbone_adapter(train_cfg)
    backbone_name = str(train_cfg.get("backbone", adapter.default_backbone))
    out_dir = Path(train_cfg.get("output_dir", adapter.default_output_dir))
    family_margin = float(train_cfg.get("family_margin", 0.35))
    supcon_weight = float(train_cfg.get("supcon_weight", 0.20))
    is_smoke = bool(train_cfg.get("smoke", False))
    checkpoints_volume = train_cfg.get("checkpoints_volume", None)
    input_dir = Path(train_cfg.get("input_dir", "/kaggle/input"))
    target_family = train_cfg.get("target_family")

    special_res = adapter.maybe_run_pipeline(
        train_cfg,
        step_callback=step_callback,
        run_pipeline_fn=run_training_pipeline,
        run_encoder_stage1_fn=run_encoder_stage1,
    )
    if special_res is not None:
        return special_res

    is_gate_or_export = bool(
        train_cfg.get("pipeline_mode") in ("family_gate", "export_embeddings", "extract_features")
        or train_cfg.get("mode") == "family_gate"
        or train_cfg.get("gate_training")
    )

    fresh_init = bool(train_cfg.get("fresh_init", False))
    working_champ = Path("/kaggle/working") / adapter.default_output_dir
    if adapter.preserves_working_checkpoint and not fresh_init and not is_gate_or_export and input_dir.exists() and not working_champ.exists():
        for cand in input_dir.rglob(adapter.default_output_dir):
            if cand.is_dir() and (cand / "heads.safetensors").exists():
                try:
                    shutil.copytree(str(cand), str(working_champ))
                    print(f"[Trainer] Preserved existing checkpoint from {cand} to {working_champ}")
                    break
                except Exception as e:
                    print(f"[Trainer] Note on preserving checkpoint: {e}")

    if train_cfg.get("specialist_jobs") and isinstance(train_cfg["specialist_jobs"], list):
        final_out = out_dir
        for job in train_cfg["specialist_jobs"]:
            job_cfg = copy.deepcopy(train_cfg)
            job_cfg.pop("specialist_jobs", None)
            for k, v in job.items():
                if isinstance(v, dict) and isinstance(job_cfg.get(k), dict):
                    job_cfg[k] = {**job_cfg[k], **v}
                else:
                    job_cfg[k] = v
            final_out = run_training_pipeline(job_cfg, step_callback=step_callback)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return final_out

    from_emb_dir = train_cfg.get("from_embeddings_dir")
    if from_emb_dir:
        print(f"[Trainer] Fast offline dataset gate training mode triggered from: {from_emb_dir}")
        train_family_gate_from_dataset(
            dataset_dir=Path(from_emb_dir),
            target_family=train_cfg.get("target_family", "gemini"),
            output_dir=out_dir,
            head_type=train_cfg.get("head_type", "mlp"),
            subspace_dim=train_cfg.get("subspace_dim", train_cfg.get("pca_dim", None)),
            hidden_dim=int(train_cfg.get("hidden_dim", 64)),
            mlp_epochs=int(train_cfg.get("mlp_epochs", 12)),
            mlp_lr=float(train_cfg.get("mlp_lr", 2e-4)),
            mlp_dropout=float(train_cfg.get("mlp_dropout", 0.35)),
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            checkpoints_volume=checkpoints_volume,
        )
        return out_dir

    is_specialist = bool(target_family and target_family in FAMILIES and not is_gate_or_export)
    if is_specialist:
        fam_cols = FAMILIES[target_family]
        active_labels = [LABELS[i] for i in fam_cols]
        sub_lab2i = {m: i for i, m in enumerate(active_labels)}
        num_classes = len(active_labels)
        print(f"[Trainer] Training {target_family} family specialist with {num_classes} classes: {active_labels}")
    else:
        active_labels = list(LABELS)
        sub_lab2i = dict(LAB2I)
        num_classes = K

    model_source = adapter.resolve_model_source(train_cfg, input_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = adapter.resolve_precision(train_cfg, device)
    train_path, _, pool_file = resolve_dataset_paths(input_dir if input_dir.exists() else Path("."))
    all_pools = resolve_all_pool_files(input_dir if input_dir.exists() else Path("."))

    tok, tok_spec = adapter.load_tokenizer(model_source, train_cfg, is_smoke=is_smoke)
    base_model, hidden_size, keep_layers, layers = adapter.load_backbone(model_source, train_cfg, precision, device)
    print(
        f"[Trainer] Family: {adapter.family_name}, Device: {device}, GPUs: {torch.cuda.device_count()} "
        f"(VRAM: {precision.total_vram_gb:.1f} GiB), Dtype: {precision.compute_dtype}, Source: {model_source}, Keep: {keep_layers}"
    )

    base_model = adapter.prepare_backbone(base_model, precision, train_cfg, is_eval_only=is_gate_or_export)
    base_model = adapter.apply_lora(base_model, train_cfg, is_eval_only=is_gate_or_export)

    model = MultiLayerLoRAClassifier(
        base_model,
        hidden_size=hidden_size,
        layers=layers,
        num_classes=num_classes,
        backbone_family=adapter.family_name,
        tok_spec=tok_spec,
    )
    if is_gate_or_export:
        model.eval()
    if hasattr(base_model, "hf_device_map") and len(getattr(base_model, "hf_device_map", {})) > 1:
        model.plain_head = model.plain_head.to(device)
        model.centered_head = model.centered_head.to(device)
        model.dropouts = model.dropouts.to(device)
        for idx in range(len(model.norms)):
            model.norms[idx] = model.norms[idx].to(device)
    else:
        model = model.to(device)

    if train_cfg.get("pipeline_mode") in ("export_embeddings", "extract_features"):
        exp_cfg = train_cfg.get("export_embeddings", {})
        if not isinstance(exp_cfg, dict):
            exp_cfg = {}
        target = exp_cfg.get("target") or train_cfg.get("embeddings_target") or target_family or "gemini"
        save_all = bool(exp_cfg.get("save_all_layers", train_cfg.get("save_all_layers", False)))
        save_mv = bool(exp_cfg.get("save_multiview", train_cfg.get("save_multiview", False)))
        emb_out = exp_cfg.get("output_dir") or train_cfg.get("embeddings_output_dir") or out_dir
        samples_per_cls = int(exp_cfg.get("samples_per_class", train_cfg.get("samples_per_class", 0)))
        neg_ratio = float(exp_cfg.get("negative_ratio", train_cfg.get("negative_ratio", 0.10)))
        ext_bs = int(exp_cfg.get("batch_size", train_cfg.get("batch_size", 16)))
        ext_max_len = int(exp_cfg.get("max_len", train_cfg.get("max_len", tok_spec.max_len)))
        ds_title = exp_cfg.get("dataset_title", train_cfg.get("dataset_title"))
        ds_slug = exp_cfg.get("dataset_slug", train_cfg.get("dataset_slug"))

        df_p, df_tr, eval_texts, eval_labels = _load_pool_and_bench_for_gate(
            input_dir, all_pools, pool_file, train_path, ["text", "model", "prompt_id"]
        )
        _load_champion_adapter_and_heads(
            model, input_dir, backbone_name, int(train_cfg.get("lora_r", 32)), target_family, True, False, adapter, precision, train_cfg
        )
        extract_and_export_embeddings(
            model=model,
            tok=tok,
            df_pool=df_p,
            df_train=df_tr,
            target=target,
            save_all_layers=save_all,
            save_multiview=save_mv,
            output_dir=emb_out,
            samples_per_class=samples_per_cls,
            negative_ratio=neg_ratio,
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            batch_size=ext_bs,
            max_len=ext_max_len,
            end_id=tok_spec.end_id,
            device=device,
            dataset_title=ds_title,
            dataset_slug=ds_slug,
            checkpoints_volume=checkpoints_volume,
        )
        return Path(emb_out)

    if is_gate_or_export:
        print(f"[Trainer] Running Phase 3 Family Gate Pipeline for target family: {target_family} (head_type={train_cfg.get('head_type', 'mlp')})...")
        _load_champion_adapter_and_heads(
            model, input_dir, backbone_name, int(train_cfg.get("lora_r", 32)), target_family, True, not is_smoke, adapter, precision, train_cfg
        )
        df_p, df_tr, eval_texts, eval_labels = _load_pool_and_bench_for_gate(
            input_dir, all_pools, pool_file, train_path, ["text", "model", "prompt_id"]
        )
        exp_cfg = train_cfg.get("export_embeddings", {})
        if isinstance(exp_cfg, bool):
            exp_enabled, exp_save_all, exp_out = exp_cfg, bool(train_cfg.get("save_all_layers", False)), train_cfg.get("embeddings_output_dir")
        elif isinstance(exp_cfg, dict):
            exp_enabled = bool(exp_cfg.get("enabled", True))
            exp_save_all = bool(exp_cfg.get("save_all_layers", train_cfg.get("save_all_layers", False)))
            exp_out = exp_cfg.get("output_dir") or train_cfg.get("embeddings_output_dir")
        else:
            exp_enabled = bool(train_cfg.get("export_embeddings", False))
            exp_save_all = bool(train_cfg.get("save_all_layers", False))
            exp_out = train_cfg.get("embeddings_output_dir")

        gate_res = train_family_gate(
            model=model,
            tok=tok,
            df_pool=df_p,
            df_train=df_tr,
            target_family=target_family or "gemini",
            samples_per_class=int(train_cfg.get("samples_per_class", 3800)),
            negative_ratio=float(train_cfg.get("negative_ratio", 0.10)),
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            batch_size=int(train_cfg.get("batch_size", 32)),
            max_len=int(train_cfg.get("max_len", tok_spec.max_len)),
            end_id=tok_spec.end_id,
            device=device,
            output_dir=out_dir,
            checkpoints_volume=checkpoints_volume,
            head_type=train_cfg.get("head_type", "mlp"),
            use_supervised_subspace=bool(train_cfg.get("use_supervised_subspace", True)),
            subspace_dim=train_cfg.get("subspace_dim", train_cfg.get("pca_dim", None)),
            hidden_dim=int(train_cfg.get("hidden_dim", 64)),
            mlp_epochs=int(train_cfg.get("mlp_epochs", 12)),
            mlp_lr=float(train_cfg.get("mlp_lr", 2e-4)),
            mlp_dropout=float(train_cfg.get("mlp_dropout", 0.35)),
            export_embeddings=exp_enabled,
            save_all_layers=exp_save_all,
            embeddings_output_dir=exp_out,
            gate_weight=float(train_cfg.get("gate_weight", 0.40)),
            train_cfg=train_cfg,
        )
        print(f"[Trainer] Phase 3 Family Gate Training Complete: {gate_res}")
        return out_dir

    milestone_history: list[dict[str, Any]] = []
    stage1_cfg = train_cfg.get("stage1", {})
    _, _, bench_eval_bs = adapter.adjust_stage1_batch_config(stage1_cfg, precision)
    bench_sub_size = int(stage1_cfg.get("eval_subset_size", 540))
    bench_bundle = (
        _load_benchmark_holdout(input_dir, subset_size=bench_sub_size, seed=42)
        if not is_specialist and bool(train_cfg.get("eval_benchmark", True))
        else None
    )

    resume_ckpt = train_cfg.get("resume_from_checkpoint")
    if resume_ckpt:
        _resume_checkpoint(model, resume_ckpt, target_family, is_specialist)
    else:
        print(f"[Trainer] Starting fresh training initialization (fresh_init={bool(train_cfg.get('fresh_init', True))}).")

    if stage1_cfg.get("enabled", True) and (all_pools or pool_file):
        _run_stage1(
            model=model,
            tok=tok,
            tok_spec=tok_spec,
            adapter=adapter,
            precision=precision,
            device=device,
            train_cfg=train_cfg,
            stage1_cfg=stage1_cfg,
            input_dir=input_dir,
            out_dir=out_dir,
            train_path=train_path,
            pool_file=pool_file,
            all_pools=all_pools,
            bench_bundle=bench_bundle,
            active_labels=active_labels,
            sub_lab2i=sub_lab2i,
            layers=layers,
            keep_layers=keep_layers,
            backbone_name=backbone_name,
            target_family=target_family,
            is_specialist=is_specialist,
            family_margin=family_margin,
            milestone_history=milestone_history,
            step_callback=step_callback,
        )

    stage2_cfg = train_cfg.get("stage2", {})
    Z_train, oof_report = _run_stage2(
        model=model,
        tok=tok,
        tok_spec=tok_spec,
        adapter=adapter,
        precision=precision,
        device=device,
        train_cfg=train_cfg,
        stage2_cfg=stage2_cfg,
        train_path=train_path,
        out_dir=out_dir,
        bench_bundle=bench_bundle,
        active_labels=active_labels,
        sub_lab2i=sub_lab2i,
        layers=layers,
        keep_layers=keep_layers,
        backbone_name=backbone_name,
        target_family=target_family,
        is_specialist=is_specialist,
        family_margin=family_margin,
        supcon_weight=supcon_weight,
        step_callback=step_callback,
    )

    if is_specialist:
        try:
            bench_paths = list(input_dir.rglob("clean_benchmark_1800_rows.parquet"))
            if not bench_paths and Path("clean_benchmark_1800_rows.parquet").exists():
                bench_paths = [Path("clean_benchmark_1800_rows.parquet")]
            if bench_paths:
                df_bench = pd.read_parquet(bench_paths[0])
                bench_texts = df_bench["text"].tolist()
                bench_labels = [LAB2I[m] for m in df_bench["model"].tolist()]
                bench_ds = TextSequenceDataset(bench_texts, bench_labels, None, tok, tok_spec=tok_spec)
                collate_fn = functools.partial(collate_text_batch, pad_token_id=tok_spec.pad_id)
                bench_loader = DataLoader(bench_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
                model.eval()
                Z_b_list = []
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=precision.compute_dtype):
                        for batch in bench_loader:
                            b_ids = batch["input_ids"].to(device)
                            b_mask = batch["attention_mask"].to(device)
                            max_l = max(int(b_mask.sum(dim=1).max().item()), 2)
                            z_b = model.forward_features(b_ids[:, :max_l], b_mask[:, :max_l], drop_first_token=tok_spec.drop_first_token)
                            Z_b_list.append(z_b.float().cpu().numpy())
                Z_bench = np.vstack(Z_b_list)
                W_p_curr = model.plain_head.weight.detach().float().cpu().numpy()
                b_p_curr = model.plain_head.bias.detach().float().cpu().numpy()
                logits_spec = Z_bench @ W_p_curr.T + b_p_curr
                W_c_curr = model.centered_head.weight.detach().float().cpu().numpy()
                b_c_curr = model.centered_head.bias.detach().float().cpu().numpy()
                logits_blend = compute_grouped_inference_logits(
                    Z_bench, W_p_curr, b_p_curr, W_c_curr, b_c_curr, groups=None, blend_centered=0.20
                )["logits_blend"]

                fam_cols = FAMILIES[target_family]
                fam_bench_mask = np.isin(bench_labels, fam_cols)
                holdout_fam_bacc_plain = 0.0
                hard_sister_acc_plain = 0.0
                holdout_fam_bacc_blend = 0.0
                hard_sister_acc_blend = 0.0

                if fam_bench_mask.sum() > 0:
                    y_fam_true = np.array([fam_cols.index(l) for l in np.array(bench_labels)[fam_bench_mask]])
                    holdout_fam_bacc_plain = float(balanced_accuracy_score(y_fam_true, logits_spec[fam_bench_mask, : len(fam_cols)].argmax(axis=1)))
                    holdout_fam_bacc_blend = float(balanced_accuracy_score(y_fam_true, logits_blend[fam_bench_mask, : len(fam_cols)].argmax(axis=1)))
                    eff_pair = (2, 3) if (target_family == "gemini" and len(fam_cols) == 4) else (0, 1)
                    s_a, s_b = eff_pair
                    pair_mask = (y_fam_true == s_a) | (y_fam_true == s_b)
                    if pair_mask.sum() > 0 and len(np.unique(y_fam_true[pair_mask])) == 2:
                        p_preds_plain = np.where(logits_spec[fam_bench_mask][pair_mask, s_a] >= logits_spec[fam_bench_mask][pair_mask, s_b], s_a, s_b)
                        hard_sister_acc_plain = float(balanced_accuracy_score(y_fam_true[pair_mask], p_preds_plain))
                        p_preds_blend = np.where(logits_blend[fam_bench_mask][pair_mask, s_a] >= logits_blend[fam_bench_mask][pair_mask, s_b], s_a, s_b)
                        hard_sister_acc_blend = float(balanced_accuracy_score(y_fam_true[pair_mask], p_preds_blend))

                champ_paths = (
                    list(input_dir.rglob("benchmark_champion_logits.safetensors"))
                    + list(input_dir.rglob("champion_logits.safetensors"))
                    + list(Path(".").rglob("*champion_logits.safetensors"))
                )

                def _eval_gate_fused(spec_lg: np.ndarray) -> tuple[float, float, float, float]:
                    b_gw, b_raw, b_sk, base_sk = 0.35, 0.0, 0.0, 0.5294
                    if champ_paths:
                        try:
                            champ_sd = load_file(str(champ_paths[0]))
                            c_key = "logits_plain" if "logits_plain" in champ_sd else list(champ_sd.keys())[0]
                            logits_champ = champ_sd[c_key].cpu().float().numpy()
                            if len(logits_champ) == len(bench_labels):
                                P_main = softmax(logits_champ, axis=1)
                                P_main_sk = sinkhorn_balancing(P_main, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
                                base_sk = float(balanced_accuracy_score(bench_labels, P_main_sk.argmax(axis=1)))
                                cand_mask = np.isin(P_main.argmax(axis=1), fam_cols) | (P_main[:, fam_cols].sum(axis=1) >= 0.10)
                                s_slice = spec_lg[:, : len(fam_cols)]
                                s_std = float(np.std(s_slice, axis=1).mean()) if len(s_slice) > 0 else 1.0
                                spec_probs = softmax(s_slice * ((3.2 / s_std) if s_std > 1e-4 else 1.0), axis=1)
                                for gw in [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
                                    P_fused = P_main.copy()
                                    for idx in np.flatnonzero(cand_mask):
                                        orig_mass = float(P_main[idx, fam_cols].sum())
                                        if orig_mass > 1e-6:
                                            gw_eff = gw * min(orig_mass / 0.50, 1.0)
                                            main_norm = P_main[idx, fam_cols] / orig_mass
                                            P_fused[idx, fam_cols] = ((1.0 - gw_eff) * main_norm + gw_eff * spec_probs[idx]) * orig_mass
                                    r_bacc = float(balanced_accuracy_score(bench_labels, P_fused.argmax(axis=1)))
                                    p_sk = sinkhorn_balancing(P_fused, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
                                    s_bacc = float(balanced_accuracy_score(bench_labels, p_sk.argmax(axis=1)))
                                    if s_bacc > b_sk:
                                        b_sk, b_raw, b_gw = s_bacc, r_bacc, gw
                        except Exception as e:
                            print(f"[Trainer] Gate fusion eval note: {e}")
                    return b_gw, b_raw, b_sk, base_sk

                best_gw_p, best_raw_p, best_sink_p, base_sk_p = _eval_gate_fused(logits_spec)
                best_gw_b, best_raw_b, best_sink_b, base_sk_b = _eval_gate_fused(logits_blend)
                oof_plain = float(oof_report.get("oof_plain_bacc", oof_report.get("oof_blend_bacc", 0.0)))
                oof_blend = float(oof_report.get("oof_blend_bacc", oof_report.get("oof_sink_bacc", 0.0)))
                ledger = ExperimentLedger(ledger_dir=train_cfg.get("ledger_dir", "experiments"))
                exp_pfx = str(train_cfg.get("exp_prefix") or "EXP-06")
                phase_tag = adapter.specialist_phase_tag(target_family)
                feat_tag = adapter.specialist_feature_set_tag(target_family)
                notes_lbl = adapter.specialist_notes_label()

                ledger.add_entry({
                    "exp_id": f"{exp_pfx}_plain",
                    "phase": phase_tag,
                    "target_family": target_family,
                    "feature_set": feat_tag,
                    "subspace_dim": int(Z_train.shape[1]),
                    "head_type": "plain",
                    "gate_weight": best_gw_p,
                    "oof_family_bacc": oof_plain,
                    "holdout_family_bacc": holdout_fam_bacc_plain,
                    "hard_sister_acc": hard_sister_acc_plain,
                    "holdout_18cls_raw_bacc": best_raw_p,
                    "holdout_18cls_sinkhorn_bacc": best_sink_p,
                    "delta_vs_baseline_bacc": (best_sink_p - base_sk_p) if best_sink_p > 0 else 0.0,
                    "notes": f"{notes_lbl} {target_family} specialist direct plain head",
                })
                ledger.add_entry({
                    "exp_id": f"{exp_pfx}_blend",
                    "phase": phase_tag,
                    "target_family": target_family,
                    "feature_set": feat_tag,
                    "subspace_dim": int(Z_train.shape[1]),
                    "head_type": "blend",
                    "gate_weight": best_gw_b,
                    "oof_family_bacc": oof_blend,
                    "holdout_family_bacc": holdout_fam_bacc_blend,
                    "hard_sister_acc": hard_sister_acc_blend,
                    "holdout_18cls_raw_bacc": best_raw_b,
                    "holdout_18cls_sinkhorn_bacc": best_sink_b,
                    "delta_vs_baseline_bacc": (best_sink_b - base_sk_b) if best_sink_b > 0 else 0.0,
                    "notes": f"{notes_lbl} {target_family} specialist grouped inference blend head",
                })
                if ledger.md_path.exists():
                    shutil.copy(str(ledger.md_path), str(out_dir / "experiment_ledger.md"))

                meta_p = out_dir / "meta.json"
                if meta_p.exists():
                    try:
                        m_d = json.loads(meta_p.read_text(encoding="utf-8"))
                        m_d.update({
                            "oof_plain_bacc": oof_plain,
                            "oof_sink_bacc": oof_blend,
                            "within_family_acc": max(holdout_fam_bacc_plain, holdout_fam_bacc_blend),
                            "hard_pair_eval_acc": max(hard_sister_acc_plain, hard_sister_acc_blend),
                            "holdout_family_bacc": max(holdout_fam_bacc_plain, holdout_fam_bacc_blend),
                            "holdout_18cls_sinkhorn_bacc": max(best_sink_p, best_sink_b),
                            "default_gate_weight": best_gw_b if best_sink_b >= best_sink_p else best_gw_p,
                            "best_gate_weight": best_gw_b if best_sink_b >= best_sink_p else best_gw_p,
                        })
                        meta_p.write_text(json.dumps(m_d, indent=2), encoding="utf-8")
                    except Exception as e:
                        print(f"[Trainer] Note updating meta.json: {e}")
        except Exception as e:
            print(f"[Trainer] Benchmark holdout evaluation note: {e}")
    elif not is_specialist and bench_bundle is not None:
        try:
            full_texts, full_y, _, _, _ = bench_bundle
            bench_extras: dict[str, Any] = {}
            m_plain, _ = _evaluate_bench_checkpoint(
                model=model,
                tok=tok,
                texts=full_texts,
                y_true=full_y,
                max_len=tok_spec.max_len,
                device=device,
                compute_dtype=precision.compute_dtype,
                drop_first_token=tok_spec.drop_first_token,
                tag="S2_Final_Plain",
                end_id=tok_spec.end_id,
                cls_id=tok_spec.cls_id,
                pad_id=tok_spec.pad_id,
                batch_size=bench_eval_bs,
                blend_centered=0.0,
                out_dir=out_dir,
                history=milestone_history,
                eval_extras_out=bench_extras,
            )
            print_prediction_table(m_plain, header="FINAL STAGE 2 BENCH-1800 (PLAIN HEAD)")
            m_sk = bench_extras.get("plain_sinkhorn", {})
            m_bl = bench_extras.get("blend_global_approx", {})
            m_bsk = bench_extras.get("blend_global_approx_sinkhorn", {})
            if m_sk:
                print(
                    f"[Bench-1800 Calibration] Plain={m_plain['bacc']*100:.2f}% | "
                    f"Plain+Sink={m_sk['bacc']*100:.2f}% (Gem4={m_sk['gem_within_bacc']*100:.2f}%) | "
                    f"Blend(GlobalApprox)={m_bl['bacc']*100:.2f}% | Blend(GlobalApprox)+Sink={m_bsk['bacc']*100:.2f}% (3.7v3.8={m_bsk['sister_37_38_acc']*100:.2f}%)"
                )
            cm_arr = np.asarray(m_plain["confusion_matrix"], dtype=np.int64)
            print("\n=== PROBLEM-CLASS TOP CONFUSION SINKS (BENCH-1800) ===")
            for cname in PROBLEM_WATCH_CLASSES:
                c_i = LAB2I.get(cname)
                if c_i is None:
                    continue
                row = cm_arr[c_i]
                top_sinks = sorted(
                    ((LABELS[j], int(row[j])) for j in range(K) if j != c_i and row[j] > 0),
                    key=lambda x: -x[1],
                )[:3]
                print(f"  {cname:<18} | true={int(row.sum()):3d} pred={int(cm_arr[:, c_i].sum()):3d} correct={int(row[c_i]):3d} | top sinks: {top_sinks}")

            eval_out = {"plain": m_plain, **bench_extras, "milestone_history": milestone_history, "oof_report": oof_report}
            (out_dir / "benchmark_1800_eval.json").write_text(json.dumps(eval_out, indent=2), encoding="utf-8")
            meta_p = out_dir / "meta.json"
            if meta_p.exists():
                m_d = json.loads(meta_p.read_text(encoding="utf-8"))
                m_d["bench_1800_plain_bacc"] = float(m_plain["bacc"])
                m_d["bench_1800_gem4_bacc"] = float(m_plain["gem_within_bacc"])
                m_d["bench_1800_37_38_acc"] = float(m_plain["sister_37_38_acc"])
                if m_sk:
                    m_d["bench_1800_plain_sink_bacc"] = float(m_sk["bacc"])
                if m_bsk:
                    m_d["bench_1800_blend_sink_bacc"] = float(m_bsk["bacc"])
                meta_p.write_text(json.dumps(m_d, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Trainer] 18-class benchmark evaluation note: {e}")

    if step_callback is not None:
        try:
            step_callback("checkpoint_saved", "training_complete")
        except Exception as e:
            print(f"[Trainer Callback Warning] {e}")
    print(f"[Trainer] Checkpoint exported to {out_dir}")
    return out_dir


def run_encoder_stage1(
    config: dict[str, Any],
    datasets_dir: Path,
    checkpoint_base: Path | None = None,
    on_checkpoint_saved: Any = None,
) -> dict[str, Any]:
    """Unified Stage-1 Multi-Layer LoRA trainer for bidirectional encoder backbones."""
    t0 = time.time()
    s1_cfg = config["stage1"]
    seed = int(s1_cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    adapter = resolve_backbone_adapter(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = adapter.resolve_precision(config, device)
    backbone_name = str(config["backbone"])
    loss_cfg = config["loss"]
    out_cfg = config["output"]

    save_root = Path(checkpoint_base) if checkpoint_base is not None else Path(out_cfg["checkpoint_dir"])
    save_root.mkdir(parents=True, exist_ok=True)
    df_pool, df_train, df_bench = load_zero_leak_stage1_pool(
        datasets_dir=datasets_dir,
        weight_clip=tuple(loss_cfg["match_weight_clip"]),
    )
    y_bench = np.array([LAB2I[m] for m in df_bench["model"]], dtype=np.int64)
    y_train_gold = np.array([LAB2I[m] for m in df_train["model"]], dtype=np.int64)
    prompts_gold = df_train["prompt_id"].values
    pool_labels = df_pool["label_idx"].to_numpy()
    pool_weights = df_pool["sample_weight"].to_numpy()
    pool_texts = df_pool["text"].astype(str).tolist()

    model_source = adapter.resolve_model_source(config, datasets_dir)
    tok, tok_spec = adapter.load_tokenizer(model_source, config, is_smoke=bool(config.get("smoke", False)))
    pool_tok_lens: list[int] = []
    for i in range(0, len(pool_texts), 4096):
        pool_tok_lens.extend(tok(pool_texts[i : i + 4096], truncation=True, max_length=tok_spec.max_len, return_length=True)["length"])

    base_model, hidden_size, _, tap_layers = adapter.load_backbone(model_source, config, precision, device)
    base_model = adapter.prepare_backbone(base_model, precision, config, is_eval_only=False)
    peft_backbone = adapter.apply_lora(base_model, config, is_eval_only=False)

    model = MultiLayerLoRAClassifier(
        backbone=peft_backbone,
        hidden_size=hidden_size,
        layers=tap_layers,
        num_classes=K,
        backbone_family=adapter.family_name,
        tok_spec=tok_spec,
    ).to(device)
    model.norms.float()
    model.plain_head.float()
    model.centered_head.float()
    for p in model.centered_head.parameters():
        p.requires_grad = False

    def _safe_export(out_p: Path, Z_tr: np.ndarray | None = None, y_tr: np.ndarray | None = None, p_tr: np.ndarray | None = None) -> None:
        with adapter.export_context(model.backbone):
            export_compatible_checkpoint(
                model=model,
                output_dir=out_p,
                layers=tap_layers,
                keep=len(tap_layers),
                max_len=tok_spec.max_len,
                end_id=tok_spec.end_id if tok_spec.end_id is not None else 2,
                backbone_name=backbone_name,
                Z_train=Z_tr,
                y_train=y_tr,
                prompts_train=p_tr,
            )

    lora_cfg = config["lora"]
    epochs = int(s1_cfg["epochs"])
    batch_size = int(s1_cfg["batch_size"])
    max_toks_fwd = int(s1_cfg.get("max_tokens_per_forward", 4096))
    class_quotas = dict(s1_cfg["class_quotas"])
    epoch_plans = [build_targeted_epoch_indices(pool_labels, class_quotas, epoch=e, seed=seed) for e in range(epochs)]
    total_steps = sum(math.ceil(len(ep_idx) / batch_size) for ep_idx in epoch_plans)
    warmup_steps = max(1, int(round(total_steps * float(s1_cfg["warmup_ratio"]))))

    param_groups = _get_loraplus_param_groups(
        model,
        lr_lora=float(s1_cfg["lr_lora"]),
        lr_head=float(s1_cfg["lr_head"]),
        lora_plus_ratio=float(lora_cfg["lora_plus_ratio"]),
        include_centered_head=False,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(loss_cfg["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler_enc, manual_scaler_enc = adapter.build_scalers(precision, device)
    max_grad_norm = float(loss_cfg["max_grad_norm"])

    eval_every_samples = int(s1_cfg.get("eval_every_samples", 0))
    bench_texts_list = df_bench["text"].astype(str).tolist()
    watch_models = tuple(PROBLEM_WATCH_CLASSES[:8])

    epoch_logs, scaling_curve = [], []
    best_bacc, best_epoch, best_samples_seen = -1.0, -1, -1
    total_samples_seen, global_step = 0, 0
    next_eval_target = eval_every_samples if eval_every_samples > 0 else 10**12
    best_dir, last_dir = save_root / "stage1_best", save_root / "stage1_last"

    def _record_scaling_point(tag: str, win_loss: float, ep_num: int) -> dict[str, Any]:
        nonlocal best_bacc, best_epoch, best_samples_seen
        _, b_logits = extract_features_and_logits(
            model,
            bench_texts_list,
            tok,
            max_len=tok_spec.max_len,
            device=device,
            batch_size=64,
            drop_first_token=tok_spec.drop_first_token,
            compute_dtype=precision.compute_dtype,
            end_id=tok_spec.end_id,
            cls_id=tok_spec.cls_id,
        )
        m = compute_detailed_metrics(b_logits, y_bench)
        pc_map = {r["model"]: r for r in m["per_class"]}
        pt = {
            "tag": tag,
            "epoch": ep_num,
            "step": global_step,
            "samples_seen": total_samples_seen,
            "elapsed_sec": float(time.time() - t0),
            "train_loss": float(win_loss),
            "bacc": m["bacc"],
            "nll": m["nll"],
            "gem_within_bacc": m["gem_within_bacc"],
            "gem_retention": m["gem_retention"],
            "sister_37_38_acc": m["sister_37_38_acc"],
            "zero_recall_classes": m["zero_recall_classes"],
            "watch_recall": {k: pc_map[k]["recall"] for k in watch_models if k in pc_map},
            "watch_bias": {k: pc_map[k]["bias"] for k in watch_models if k in pc_map},
        }
        scaling_curve.append(pt)
        if m["bacc"] > best_bacc:
            best_bacc, best_epoch, best_samples_seen = m["bacc"], ep_num, total_samples_seen
            _safe_export(best_dir)
            print(f"  -> [New Best Saved] {tag} ({total_samples_seen:,} samples): BAcc={best_bacc*100:.2f}% -> {best_dir}", flush=True)
        model.train()
        return m

    for ep in range(epochs):
        ep_t0 = time.time()
        ep_indices = epoch_plans[ep]
        sampler = BucketedLengthBatchSampler(
            lengths=[pool_tok_lens[int(i)] for i in ep_indices],
            batch_size=batch_size,
            bucket_size_multiplier=60,
            shuffle=True,
            seed=seed + ep * 31,
        )
        model.train()
        running_loss = 0.0
        n_batches = len(sampler)
        for step_i, rel_batch in enumerate(sampler):
            abs_batch = ep_indices[np.asarray(rel_batch, dtype=np.int64)]
            b_texts = [pool_texts[int(i)] for i in abs_batch]
            b_labels = torch.from_numpy(pool_labels[abs_batch]).to(device, dtype=torch.long)
            b_weights = torch.from_numpy(pool_weights[abs_batch]).to(device, dtype=torch.float32)
            ds_batch = TextSequenceDataset(b_texts, None, None, tok, tok_spec=tok_spec)
            enc = collate_text_batch([ds_batch[k] for k in range(len(b_texts))], pad_token_id=tok_spec.pad_id)
            seq_len, bsz = int(enc["input_ids"].shape[1]), len(abs_batch)
            optimizer.zero_grad(set_to_none=True)

            chunk_sz = max(2, min(bsz, max_toks_fwd // max(seq_len, 1)))
            total_w = b_weights.sum().clamp(min=1e-6)
            step_loss_val = 0.0
            for start in range(0, bsz, chunk_sz):
                end = min(start + chunk_sz, bsz)
                sub_ids = enc["input_ids"][start:end].to(device)
                sub_mask = enc["attention_mask"][start:end].to(device)
                sub_y, sub_w = b_labels[start:end], b_weights[start:end]
                w_frac = sub_w.sum() / total_w
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=precision.compute_dtype):
                    feats = model.forward_features(sub_ids, sub_mask, drop_first_token=tok_spec.drop_first_token)
                    loss = family_margin_cross_entropy(
                        model.predict_plain(feats),
                        sub_y,
                        family_margin=float(loss_cfg["family_margin"]),
                        label_smoothing=float(loss_cfg["label_smoothing"]),
                        boost_hard_sisters=bool(loss_cfg["boost_hard_sisters"]),
                        sample_weight=sub_w,
                    )
                if scaler_enc is not None and getattr(scaler_enc, "is_enabled", lambda: False)():
                    scaler_enc.scale(loss * w_frac).backward()
                elif manual_scaler_enc is not None and manual_scaler_enc.is_enabled():
                    manual_scaler_enc.scale_loss(loss * w_frac).backward()
                else:
                    (loss * w_frac).backward()
                step_loss_val += float(loss.item()) * float(w_frac.item())

            _step_with_grad_guard(
                model,
                optimizer,
                scheduler,
                scaler=scaler_enc,
                manual_scaler=manual_scaler_enc,
                max_grad_norm=max_grad_norm,
            )
            running_loss += step_loss_val
            global_step += 1
            total_samples_seen += bsz

            if total_samples_seen >= next_eval_target and (step_i + 1) < (n_batches - 15):
                _record_scaling_point(f"{total_samples_seen//1000}k", running_loss / float(step_i + 1), ep + 1)
                next_eval_target += eval_every_samples

        ep_metrics = _record_scaling_point(f"Ep{ep+1}_End", running_loss / max(n_batches, 1), ep + 1)
        ep_metrics["epoch"] = ep + 1
        ep_metrics["train_avg_loss"] = float(running_loss / max(n_batches, 1))
        ep_metrics["epoch_sec"] = float(time.time() - ep_t0)
        print_prediction_table(ep_metrics, header=f"Epoch {ep+1} Bench-1800")
        epoch_logs.append(ep_metrics)
        _safe_export(last_dir)
        if on_checkpoint_saved is not None:
            on_checkpoint_saved()

    if best_samples_seen != total_samples_seen and (best_dir / "heads.safetensors").exists():
        del model, peft_backbone, base_model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dtype_re = torch.bfloat16 if precision.use_bf16 else torch.float32
        base_re = AutoModel.from_pretrained(model_source, torch_dtype=dtype_re)
        peft_re = PeftModel.from_pretrained(base_re, str(best_dir / "adapter"))
        adapter.prepare_backbone(base_re, precision, config, is_eval_only=True)
        model = MultiLayerLoRAClassifier(
            backbone=peft_re,
            hidden_size=hidden_size,
            layers=tap_layers,
            num_classes=K,
            backbone_family=adapter.family_name,
            tok_spec=tok_spec,
        ).to(device)
        model.norms.float()
        model.plain_head.float()
        model.centered_head.float()
        hd = load_file(str(best_dir / "heads.safetensors"), device=str(device))
        for i in range(len(tap_layers)):
            model.norms[i].weight.data.copy_(hd[f"norms.{i}.weight"].float())
            if f"norms.{i}.bias" in hd:
                model.norms[i].bias.data.copy_(hd[f"norms.{i}.bias"].float())
        model.plain_head.weight.data.copy_(hd["plain.weight"].float())
        model.plain_head.bias.data.copy_(hd["plain.bias"].float())

    n_probe_per_cls = max(1, int(out_cfg.get("ridge_probe_samples", 36000)) // K)
    rng_pr = np.random.RandomState(42)
    probe_idxs = np.concatenate([
        rng_pr.choice(np.where(pool_labels == c_i)[0], size=min(n_probe_per_cls, int((pool_labels == c_i).sum())), replace=False)
        for c_i in range(K)
    ])
    Z_bench, lg_bench_neural = extract_features_and_logits(
        model, bench_texts_list, tok, max_len=tok_spec.max_len, device=device, batch_size=64, drop_first_token=tok_spec.drop_first_token, compute_dtype=precision.compute_dtype, end_id=tok_spec.end_id, cls_id=tok_spec.cls_id
    )
    Z_gold, lg_gold_neural = extract_features_and_logits(
        model, df_train["text"].astype(str).tolist(), tok, max_len=tok_spec.max_len, device=device, batch_size=64, drop_first_token=tok_spec.drop_first_token, compute_dtype=precision.compute_dtype, end_id=tok_spec.end_id, cls_id=tok_spec.cls_id
    )
    Z_probe, _ = extract_features_and_logits(
        model, [pool_texts[int(i)] for i in probe_idxs], tok, max_len=tok_spec.max_len, device=device, batch_size=64, drop_first_token=tok_spec.drop_first_token, compute_dtype=precision.compute_dtype, end_id=tok_spec.end_id, cls_id=tok_spec.cls_id
    )

    ridge_map = gpu_trace_ridge_eval(Z_probe, pool_labels[probe_idxs], Z_bench, alphas=[0.05, 0.25, 1.0, 5.0], device=device)
    best_r_alpha, best_r_logits = max(ridge_map.items(), key=lambda kv: balanced_accuracy_score(y_bench, kv[1].argmax(axis=1)))

    def _row_z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(axis=1, keepdims=True)) / np.clip(x.std(axis=1, keepdims=True), 1e-5, None)

    best_w_neural, best_blend_logits = max(
        ((w_n, w_n * _row_z(lg_bench_neural) + (1.0 - w_n) * _row_z(best_r_logits)) for w_n in [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]),
        key=lambda item: balanced_accuracy_score(y_bench, item[1].argmax(axis=1)),
    )

    Z_gold_cent = Z_gold.copy()
    for pid in np.unique(prompts_gold):
        m = prompts_gold == pid
        if m.sum() > 1:
            Z_gold_cent[m] -= Z_gold_cent[m].mean(axis=0, keepdims=True)

    oof_alphas = [0.05, 0.25, 1.0, 5.0]
    oof_p_map = {a: np.zeros((len(y_train_gold), K), dtype=np.float32) for a in oof_alphas}
    oof_c_map = {a: np.zeros((len(y_train_gold), K), dtype=np.float32) for a in oof_alphas}
    for tr_i, val_i in GroupKFold(n_splits=5).split(Z_gold, y_train_gold, groups=prompts_gold):
        rp = gpu_trace_ridge_eval(Z_gold[tr_i], y_train_gold[tr_i], Z_gold[val_i], oof_alphas, device=device)
        rc = gpu_trace_ridge_eval(Z_gold_cent[tr_i], y_train_gold[tr_i], Z_gold_cent[val_i], oof_alphas, device=device)
        for a in oof_alphas:
            oof_p_map[a][val_i] = rp[a]
            oof_c_map[a][val_i] = rc[a]

    _safe_export(best_dir, Z_tr=Z_gold, y_tr=y_train_gold, p_tr=prompts_gold)
    best_oof_c = max(oof_c_map.values(), key=lambda lg: balanced_accuracy_score(y_train_gold, lg.argmax(axis=1)))
    return {
        "checkpoint_dir": str(best_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": float(time.time() - t0),
        "config": config,
        "best_epoch": best_epoch,
        "best_samples_seen": best_samples_seen,
        "best_neural_bench_bacc": float(best_bacc),
        "scaling_curve": scaling_curve,
        "epoch_logs": epoch_logs,
        "bench_ridge": {"best_alpha": float(best_r_alpha), **compute_detailed_metrics(best_r_logits, y_bench)},
        "bench_neural_ridge_blend": {"w_neural": float(best_w_neural), **compute_detailed_metrics(best_blend_logits, y_bench)},
        "gold_train_eval": {
            "neural_plain_bacc": float(balanced_accuracy_score(y_train_gold, lg_gold_neural.argmax(axis=1))),
            "oof5_ridge_plain_bacc": max(float(balanced_accuracy_score(y_train_gold, oof_p_map[a].argmax(axis=1))) for a in oof_alphas),
            "oof5_ridge_centered_bacc": max(float(balanced_accuracy_score(y_train_gold, oof_c_map[a].argmax(axis=1))) for a in oof_alphas),
        },
        "arrays": {
            "y_bench": y_bench.tolist(),
            "bench_neural_logits": lg_bench_neural.astype(np.float16).tolist(),
            "bench_ridge_logits": best_r_logits.astype(np.float16).tolist(),
            "y_gold": y_train_gold.tolist(),
            "gold_oof_centered_logits": best_oof_c.astype(np.float16).tolist(),
        },
    }
