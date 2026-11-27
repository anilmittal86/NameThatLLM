"""
Feature and Layer-by-Layer Embedding Extraction and Kaggle Dataset Bundle Export.
"""

from __future__ import annotations

import functools
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from ..ablation import _text_hash
from ..config import FAMILIES, LABELS
from .backbones.base import resolve_backbone_adapter
from .data import TextSequenceDataset, collate_text_batch
from .modeling import MultiLayerLoRAClassifier


def save_extracted_features_bundle(
    output_dir: Path | str,
    target: str,
    features_concat: np.ndarray,
    features_by_layer: np.ndarray | None,
    texts: list[str],
    labels: list[int] | np.ndarray,
    model_names: list[str],
    prompt_ids: list[Any] | None = None,
    eval_features_concat: np.ndarray | None = None,
    eval_features_by_layer: np.ndarray | None = None,
    eval_texts: list[str] | None = None,
    eval_labels: list[int] | None = None,
    eval_model_names: list[str] | None = None,
    dataset_title: str | None = None,
    dataset_slug: str | None = None,
    checkpoints_volume: Any | None = None,
    champion_logits_plain: np.ndarray | None = None,
    champion_logits_centered: np.ndarray | None = None,
    head_tail_cos: np.ndarray | None = None,
    token_std_norms: np.ndarray | None = None,
    layer_norms: np.ndarray | None = None,
    eval_champion_logits_plain: np.ndarray | None = None,
    eval_champion_logits_centered: np.ndarray | None = None,
    eval_head_tail_cos: np.ndarray | None = None,
    eval_token_std_norms: np.ndarray | None = None,
    eval_layer_norms: np.ndarray | None = None,
) -> Path:
    """Saves extracted features, structural multiview tensors, and metadata in Kaggle Dataset format."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"\n[Features Export] Saving features bundle for target '{target}' to: {out_p}...")

    feat_cat_t = torch.from_numpy(features_concat).to(torch.float16)
    save_file({"features_concat": feat_cat_t}, str(out_p / "features_concat.safetensors"))
    sz_cat_mb = (out_p / "features_concat.safetensors").stat().st_size / (1024**2)
    print(f"  Saved features_concat.safetensors: shape={list(feat_cat_t.shape)}, size={sz_cat_mb:.1f} MB")

    sz_layer_mb = 0.0
    if features_by_layer is not None:
        feat_layers_t = torch.from_numpy(features_by_layer).to(torch.float16)
        save_file({"features_by_layer": feat_layers_t}, str(out_p / "features_by_layer.safetensors"))
        sz_layer_mb = (out_p / "features_by_layer.safetensors").stat().st_size / (1024**2)
        print(f"  Saved features_by_layer.safetensors: shape={list(feat_layers_t.shape)}, size={sz_layer_mb:.1f} MB")

    champ_dict = {}
    if champion_logits_plain is not None:
        champ_dict["logits_plain"] = torch.from_numpy(champion_logits_plain).to(torch.float32)
    if champion_logits_centered is not None:
        champ_dict["logits_centered"] = torch.from_numpy(champion_logits_centered).to(torch.float32)
    if head_tail_cos is not None:
        champ_dict["head_tail_cos"] = torch.from_numpy(head_tail_cos).to(torch.float32)
    if token_std_norms is not None:
        champ_dict["token_std_norms"] = torch.from_numpy(token_std_norms).to(torch.float32)
    if champ_dict:
        save_file(champ_dict, str(out_p / "champion_logits.safetensors"))
        print(f"  Saved champion_logits.safetensors: keys={list(champ_dict.keys())}")

    if layer_norms is not None:
        save_file({"layer_norms": torch.from_numpy(layer_norms).to(torch.float32)}, str(out_p / "layer_norms.safetensors"))

    meta_df = pd.DataFrame({
        "id": np.arange(len(texts), dtype=np.int64),
        "model": model_names,
        "label": np.asarray(labels, dtype=np.int64),
        "prompt_id": prompt_ids if prompt_ids is not None else [-1] * len(texts),
        "text_hash": [_text_hash(t) for t in texts],
        "char_len": [len(t) for t in texts],
    })
    meta_df.to_parquet(out_p / "metadata.parquet", index=False)
    print(f"  Saved metadata.parquet: {len(meta_df):,} rows")

    if eval_features_concat is not None and eval_texts is not None:
        ev_cat_t = torch.from_numpy(eval_features_concat).to(torch.float16)
        save_file({"benchmark_features_concat": ev_cat_t}, str(out_p / "benchmark_features_concat.safetensors"))
        if eval_features_by_layer is not None:
            ev_layer_t = torch.from_numpy(eval_features_by_layer).to(torch.float16)
            save_file({"benchmark_features_by_layer": ev_layer_t}, str(out_p / "benchmark_features_by_layer.safetensors"))

        eval_champ_dict = {}
        if eval_champion_logits_plain is not None:
            eval_champ_dict["logits_plain"] = torch.from_numpy(eval_champion_logits_plain).to(torch.float32)
        if eval_champion_logits_centered is not None:
            eval_champ_dict["logits_centered"] = torch.from_numpy(eval_champion_logits_centered).to(torch.float32)
        if eval_head_tail_cos is not None:
            eval_champ_dict["head_tail_cos"] = torch.from_numpy(eval_head_tail_cos).to(torch.float32)
        if eval_token_std_norms is not None:
            eval_champ_dict["token_std_norms"] = torch.from_numpy(eval_token_std_norms).to(torch.float32)
        if eval_champ_dict:
            save_file(eval_champ_dict, str(out_p / "benchmark_champion_logits.safetensors"))
            print(f"  Saved benchmark_champion_logits.safetensors: keys={list(eval_champ_dict.keys())}")

        if eval_layer_norms is not None:
            save_file({"layer_norms": torch.from_numpy(eval_layer_norms).to(torch.float32)}, str(out_p / "benchmark_layer_norms.safetensors"))

        eval_models = eval_model_names if eval_model_names is not None else [
            LABELS[i] if i < len(LABELS) else f"class_{i}" for i in (eval_labels if eval_labels is not None else [0] * len(eval_texts))
        ]
        bench_df = pd.DataFrame({
            "id": np.arange(len(eval_texts), dtype=np.int64),
            "model": eval_models,
            "label": np.asarray(eval_labels, dtype=np.int64) if eval_labels is not None else np.zeros(len(eval_texts), dtype=np.int64),
            "text_hash": [_text_hash(t) for t in eval_texts],
            "char_len": [len(t) for t in eval_texts],
        })
        bench_df.to_parquet(out_p / "benchmark_metadata.parquet", index=False)
        print(f"  Saved benchmark features & metadata: {len(bench_df):,} samples")

    slug = dataset_slug or f"ntl-{target.lower()}-features"
    title = dataset_title or f"NTL {target.upper()} Extracted Features"
    kaggle_meta = {
        "title": title,
        "id": f"anilmittal/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            f"Pre-extracted high-dimensional feature representations for {target.upper()} models "
            f"using Multi-Layer backbone representations."
        ),
    }
    (out_p / "dataset-metadata.json").write_text(json.dumps(kaggle_meta, indent=2), encoding="utf-8")

    summary = {
        "target": target,
        "num_train_samples": len(texts),
        "num_classes": len(set(model_names)),
        "class_counts": {k: int(v) for k, v in pd.Series(model_names).value_counts().items()},
        "feature_dim_concat": int(features_concat.shape[1]),
        "features_by_layer_shape": list(features_by_layer.shape) if features_by_layer is not None else None,
        "has_benchmark": eval_features_concat is not None,
        "num_benchmark_samples": len(eval_texts) if eval_texts is not None else 0,
        "total_bundle_size_mb": round(sz_cat_mb + sz_layer_mb, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    (out_p / "features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if checkpoints_volume is not None and hasattr(checkpoints_volume, "commit"):
        try:
            checkpoints_volume.commit()
        except Exception:
            pass

    print(f"[Features Export] Complete! Bundle ready for Kaggle Dataset creation at: {out_p} ({summary['total_bundle_size_mb']:.1f} MB)\n")
    return out_p


def extract_and_export_embeddings(
    model: MultiLayerLoRAClassifier,
    tok: Any,
    df_pool: pd.DataFrame,
    df_train: pd.DataFrame | None = None,
    target: str = "gemini",
    save_all_layers: bool = False,
    save_multiview: bool = False,
    output_dir: Path | str = "export_embeddings",
    samples_per_class: int = 0,
    negative_ratio: float = 0.10,
    eval_texts: list[str] | None = None,
    eval_labels: list[int] | None = None,
    eval_model_names: list[str] | None = None,
    batch_size: int = 16,
    max_len: int | None = None,
    end_id: int | None = None,
    device: torch.device | None = None,
    dataset_title: str | None = None,
    dataset_slug: str | None = None,
    checkpoints_volume: Any | None = None,
    eval_champion_logits_plain: np.ndarray | None = None,
) -> Path:
    """Extracts multi-layer embeddings batch-by-batch across a model and writes a Kaggle dataset bundle."""
    if device is None:
        device = next(model.parameters()).device

    fam_hint = getattr(model, "backbone_family", "unknown")
    adapter_cfg: dict[str, Any] = {}
    if fam_hint and fam_hint != "unknown":
        adapter_cfg["backbone_family"] = fam_hint
    else:
        bb_obj = getattr(model, "backbone", model)
        adapter_cfg["backbone"] = f"{type(bb_obj)} {getattr(bb_obj, 'config', '')}"
    adapter = resolve_backbone_adapter(adapter_cfg)

    tok_spec = getattr(model, "tok_spec", None)
    if tok_spec is None:
        tok_spec = adapter.tok_spec_from_tokenizer(tok, max_len=max_len, end_id=end_id)

    eff_max_len = int(max_len if max_len is not None else tok_spec.max_len)
    batch_size = adapter.clamp_extraction_batch_size(batch_size)

    target_clean = target.lower().strip()
    print("\n==========================================================================================")
    print(f" [Features Extraction] Target: '{target_clean}' | Save All Layers: {save_all_layers} | Multiview: {save_multiview} | Max Len: {eff_max_len}")
    print("==========================================================================================")

    if eval_texts is not None and len(eval_texts) > 0:
        bench_hashes = {_text_hash(t) for t in eval_texts}
        if df_pool is not None and len(df_pool) > 0:
            init_p = len(df_pool)
            pool_hashes = [_text_hash(t) for t in df_pool["text"].fillna("")]
            df_pool = df_pool[[h not in bench_hashes for h in pool_hashes]].copy()
            quar_p = init_p - len(df_pool)
            if quar_p > 0:
                print(f"[Features Extraction] Quarantined {quar_p} pool samples matching benchmark.")
        if df_train is not None and len(df_train) > 0:
            init_t = len(df_train)
            tr_hashes = [_text_hash(t) for t in df_train["text"].fillna("")]
            df_train = df_train[[h not in bench_hashes for h in tr_hashes]].copy()
            quar_t = init_t - len(df_train)
            if quar_t > 0:
                print(f"[Features Extraction] Quarantined {quar_t} gold train samples matching benchmark.")

    if target_clean in ("gold", "train"):
        assert df_train is not None and len(df_train) > 0, "No gold df_train provided for target='gold'"
        df_target_all = df_train.copy()
        df_neg_all = pd.DataFrame()
    elif target_clean == "all":
        all_dfs = []
        for c in LABELS:
            sub_tr = df_train[df_train["model"] == c] if (df_train is not None and len(df_train) > 0) else pd.DataFrame()
            sub_pool = df_pool[df_pool["model"] == c] if (df_pool is not None and len(df_pool) > 0) else pd.DataFrame()
            if samples_per_class > 0:
                n_take = min(samples_per_class - len(sub_tr), len(sub_pool))
                sampled_pool = sub_pool.sample(n=max(n_take, 0), random_state=42) if n_take > 0 else pd.DataFrame()
            else:
                sampled_pool = sub_pool
            c_df = pd.concat([sub_tr, sampled_pool], ignore_index=True)
            all_dfs.append(c_df)
            print(f"  Class {c}: {len(c_df)} samples ({len(sub_tr)} gold, {len(c_df)-len(sub_tr)} pool)")
        df_target_all = pd.concat(all_dfs, ignore_index=True)
        df_neg_all = pd.DataFrame()
    elif target_clean in FAMILIES:
        target_cols = FAMILIES[target_clean]
        target_classes = [LABELS[i] for i in target_cols]
        cap = samples_per_class if samples_per_class > 0 else 3800
        fam_dfs = []
        for c in target_classes:
            sub_tr = df_train[df_train["model"] == c] if (df_train is not None and len(df_train) > 0) else pd.DataFrame()
            sub_pool = df_pool[df_pool["model"] == c] if (df_pool is not None and len(df_pool) > 0) else pd.DataFrame()
            n_take = min(cap - len(sub_tr), len(sub_pool))
            sampled_pool = sub_pool.sample(n=max(n_take, 0), random_state=42) if n_take > 0 else pd.DataFrame()
            c_df = pd.concat([sub_tr, sampled_pool], ignore_index=True)
            fam_dfs.append(c_df)
            print(f"  {target_clean.upper()} class {c}: {len(c_df)} samples ({len(sub_tr)} gold, {len(c_df)-len(sub_tr)} pool)")
        df_target_all = pd.concat(fam_dfs, ignore_index=True)

        if negative_ratio > 0.0 and df_pool is not None and len(df_pool) > 0:
            n_neg = int(len(df_target_all) * negative_ratio)
            df_non_fam = df_pool[~df_pool["model"].isin(target_classes)].copy()
            other_fams = sorted({"".join(ch for ch in m.split("-")[0].split(".")[0] if ch.isalpha()) for m in LABELS if m not in target_classes})
            neg_dfs = []
            n_per_fam = max(n_neg // max(len(other_fams), 1), 1)
            for fam in other_fams:
                fam_sub = df_non_fam[df_non_fam["model"].str.startswith(fam)]
                if len(fam_sub) > 0:
                    neg_dfs.append(fam_sub.sample(n=min(n_per_fam, len(fam_sub)), random_state=42))
            df_neg_all = pd.concat(neg_dfs, ignore_index=True) if neg_dfs else pd.DataFrame()
            print(f"  Sampled {len(df_neg_all)} negative distractors across non-{target_clean.upper()} families.")
        else:
            df_neg_all = pd.DataFrame()
    else:
        raise ValueError(f"Unknown target '{target}'. Must be one of: 'all', 'gold', or a valid family {list(FAMILIES.keys())}.")

    train_texts = df_target_all["text"].tolist() + (df_neg_all["text"].tolist() if len(df_neg_all) > 0 else [])
    model_names = df_target_all["model"].tolist() + (["__OTHER__"] * len(df_neg_all) if len(df_neg_all) > 0 else [])

    if target_clean in FAMILIES:
        ordered_models = [LABELS[i] for i in FAMILIES[target_clean]] + (["__OTHER__"] if len(df_neg_all) > 0 else [])
    else:
        ordered_models = list(LABELS) + (["__OTHER__"] if len(df_neg_all) > 0 else [])
    lab_map = {m: i for i, m in enumerate(ordered_models)}
    train_labels = [lab_map[m] for m in model_names]

    target_pids = (
        df_target_all["prompt_id"].fillna("-1").astype(str).tolist()
        if "prompt_id" in df_target_all.columns
        else ["-1"] * len(df_target_all)
    )
    neg_pids = (
        df_neg_all["prompt_id"].fillna("-1").astype(str).tolist()
        if (len(df_neg_all) > 0 and "prompt_id" in df_neg_all.columns)
        else ["-1"] * len(df_neg_all)
    )
    prompt_ids = target_pids + neg_pids

    amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
        else torch.float16
    )

    has_plain = hasattr(model, "plain_head") and model.plain_head is not None
    has_centered = hasattr(model, "centered_head") and model.centered_head is not None

    print(f"[Features Extraction] Extracting features on {len(train_texts):,} samples (batch_size={batch_size})...")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    if save_all_layers:
        try:
            _, _, free_bytes = shutil.disk_usage(out_p)
            n_lay_est = adapter.default_layer_count_for_disk_estimate(model)
            h_dim_est = int(
                getattr(model, "hidden_size", 0)
                or getattr(getattr(getattr(model, "backbone", model), "config", None), "hidden_size", 0)
                or (getattr(model, "total_dim", 0) // max(len(getattr(model, "layers", [1])), 1))
            )
            est_all_layers_bytes = len(train_texts) * n_lay_est * max(h_dim_est, 1) * 2
            if est_all_layers_bytes > 0.80 * free_bytes:
                print(
                    f"\n[WARNING][Disk Safety Guard] Estimated {n_lay_est}-layer tensor size ({est_all_layers_bytes / (1024**3):.2f} GB) "
                    f"exceeds 80% of available free disk space ({free_bytes / (1024**3):.2f} GB) in {out_p}."
                )
                print(
                    f"  -> Automatically falling back to saving {len(getattr(model, 'layers', []))} tapped layers in "
                    f"features_by_layer.safetensors ([N, {len(getattr(model, 'layers', []))}, {h_dim_est}]) to prevent OSError.\n"
                )
                save_all_layers = False
        except Exception as e:
            print(f"[Features Extraction] Note: Could not query disk usage ({e}); proceeding with requested save_all_layers={save_all_layers}.")

    pad_id = tok_spec.pad_id
    cls_id = tok_spec.cls_id
    eff_end_id = tok_spec.end_id if end_id is None else end_id
    drop_first_token = tok_spec.drop_first_token
    collate_fn_ext = functools.partial(collate_text_batch, pad_token_id=pad_id)

    ds_tr = TextSequenceDataset(train_texts, train_labels, None, tok, max_len=eff_max_len, end_id=eff_end_id, cls_id=cls_id)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ext)

    Z_cat_list = []
    Z_layers_list = []
    cos_drift_list = []
    std_norms_list = []
    logits_p_list = []
    logits_c_list = []

    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=(amp_dtype == torch.bfloat16), dtype=amp_dtype):
            for batch in loader_tr:
                mb_ids = batch["input_ids"].to(device)
                mb_mask = batch["attention_mask"].to(device)
                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                if save_multiview:
                    res = model.forward_features(
                        mb_ids[:, :max_l],
                        mb_mask[:, :max_l],
                        drop_first_token=drop_first_token,
                        return_by_layer=True,
                        extract_all_layers=save_all_layers,
                        return_multiview=True,
                    )
                    if len(res) == 4:
                        z_b, z_l, ht_cos_t, std_norm_t = res
                        cos_drift_list.append(ht_cos_t.cpu().float().numpy())
                        std_norms_list.append(std_norm_t.cpu().float().numpy())
                    else:
                        z_b, z_l, mv_d = res
                        cos_drift_list.append(mv_d["head_tail_cos"].cpu().float().numpy())
                        std_norms_list.append(mv_d["token_std_norms"].cpu().float().numpy())
                else:
                    z_b, z_l = model.forward_features(
                        mb_ids[:, :max_l],
                        mb_mask[:, :max_l],
                        drop_first_token=drop_first_token,
                        return_by_layer=True,
                        extract_all_layers=save_all_layers,
                        return_multiview=False,
                    )

                if len(Z_cat_list) == 0:
                    d_chunk = max(z_b.shape[1] // max(len(model.layers), 1), 1)
                    chunk_stds = [float(z_b[:, i * d_chunk : (i + 1) * d_chunk].std().item()) for i in range(len(model.layers))]
                    layer_stds = (
                        [float(z_l[:, i, :].std().item()) for i in range(z_l.shape[1])]
                        if (save_all_layers and z_l is not None and z_l.ndim == 3)
                        else chunk_stds
                    )
                    dead_layers = [i for i, s in enumerate(layer_stds) if s < 1e-6 or not math.isfinite(s)]
                    print(
                        f"[Sanity Check Batch 0] Active layers: {len(layer_stds) - len(dead_layers)}/{len(layer_stds)} | "
                        f"Layer std range: [{min(layer_stds):.4f}, {max(layer_stds):.4f}] | "
                        f"Tapped {len(model.layers)}-layer std range: [{min(chunk_stds):.4f}, {max(chunk_stds):.4f}]",
                        flush=True,
                    )
                    if dead_layers:
                        raise RuntimeError(f"Batch 0 Sanity Check FAILED: Layers {dead_layers} have zero/NaN activations!")

                Z_cat_list.append(z_b.cpu().float().numpy())
                Z_layers_list.append(z_l.cpu().to(torch.float16).numpy())
                if has_plain:
                    logits_p_list.append(model.predict_plain(z_b).cpu().float().numpy())
                if has_centered:
                    logits_c_list.append(model.predict_centered(z_b).cpu().float().numpy())

    features_concat = np.vstack(Z_cat_list)
    features_by_layer = np.vstack(Z_layers_list)
    champ_p = np.vstack(logits_p_list) if logits_p_list else None
    champ_c = np.vstack(logits_c_list) if logits_c_list else None
    ht_cos = np.vstack(cos_drift_list) if cos_drift_list else None
    t_std = np.vstack(std_norms_list) if std_norms_list else None

    eval_cat = None
    eval_layers = None
    eval_champ_p = None
    eval_champ_c = None
    eval_ht_cos = None
    eval_t_std = None

    if eval_texts is not None and len(eval_texts) > 0:
        print(f"[Features Extraction] Extracting benchmark features on {len(eval_texts)} holdout samples...")
        dummy_y = [0] * len(eval_texts)
        ds_ev = TextSequenceDataset(eval_texts, dummy_y, None, tok, max_len=eff_max_len, end_id=eff_end_id, cls_id=cls_id)
        loader_ev = DataLoader(ds_ev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_ext)
        ev_cat_list = []
        ev_layers_list = []
        ev_cos_list = []
        ev_std_list = []
        ev_lp_list = []
        ev_lc_list = []

        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=(amp_dtype == torch.bfloat16), dtype=amp_dtype):
                for batch in loader_ev:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    if save_multiview:
                        res = model.forward_features(
                            mb_ids[:, :max_l],
                            mb_mask[:, :max_l],
                            drop_first_token=drop_first_token,
                            return_by_layer=True,
                            extract_all_layers=save_all_layers,
                            return_multiview=True,
                        )
                        if len(res) == 4:
                            z_b, z_l, ht_cos_t, std_norm_t = res
                            ev_cos_list.append(ht_cos_t.cpu().float().numpy())
                            ev_std_list.append(std_norm_t.cpu().float().numpy())
                        else:
                            z_b, z_l, mv_d = res
                            ev_cos_list.append(mv_d["head_tail_cos"].cpu().float().numpy())
                            ev_std_list.append(mv_d["token_std_norms"].cpu().float().numpy())
                    else:
                        z_b, z_l = model.forward_features(
                            mb_ids[:, :max_l],
                            mb_mask[:, :max_l],
                            drop_first_token=drop_first_token,
                            return_by_layer=True,
                            extract_all_layers=save_all_layers,
                            return_multiview=False,
                        )
                    ev_cat_list.append(z_b.cpu().float().numpy())
                    ev_layers_list.append(z_l.cpu().to(torch.float16).numpy())
                    if has_plain:
                        ev_lp_list.append(model.predict_plain(z_b).cpu().float().numpy())
                    if has_centered:
                        ev_lc_list.append(model.predict_centered(z_b).cpu().float().numpy())

        eval_cat = np.vstack(ev_cat_list)
        eval_layers = np.vstack(ev_layers_list)
        eval_champ_p = np.vstack(ev_lp_list) if ev_lp_list else None
        eval_champ_c = np.vstack(ev_lc_list) if ev_lc_list else None
        eval_ht_cos = np.vstack(ev_cos_list) if ev_cos_list else None
        eval_t_std = np.vstack(ev_std_list) if ev_std_list else None

    out_dir_path = save_extracted_features_bundle(
        output_dir=output_dir,
        target=target_clean,
        features_concat=features_concat,
        features_by_layer=features_by_layer,
        texts=train_texts,
        labels=train_labels,
        model_names=model_names,
        prompt_ids=prompt_ids,
        eval_features_concat=eval_cat,
        eval_features_by_layer=eval_layers,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        eval_model_names=eval_model_names,
        dataset_title=dataset_title,
        dataset_slug=dataset_slug,
        checkpoints_volume=checkpoints_volume,
        champion_logits_plain=champ_p if getattr(model, "num_classes", 0) == len(LABELS) else None,
        champion_logits_centered=champ_c if getattr(model, "num_classes", 0) == len(LABELS) else None,
        head_tail_cos=ht_cos,
        token_std_norms=t_std,
        layer_norms=None,
        eval_champion_logits_plain=eval_champion_logits_plain if eval_champion_logits_plain is not None else (eval_champ_p if getattr(model, "num_classes", 0) == len(LABELS) else None),
        eval_champion_logits_centered=eval_champ_c if getattr(model, "num_classes", 0) == len(LABELS) else None,
        eval_head_tail_cos=eval_ht_cos,
        eval_token_std_norms=eval_t_std,
        eval_layer_norms=None,
    )
    if hasattr(model, "norms") and model.norms is not None:
        norms_dict = {}
        for i, norm in enumerate(model.norms):
            norms_dict[f"norms.{i}.weight"] = norm.weight.detach().cpu().to(torch.float32).contiguous()
            if hasattr(norm, "bias") and norm.bias is not None:
                norms_dict[f"norms.{i}.bias"] = norm.bias.detach().cpu().to(torch.float32).contiguous()
        save_file(norms_dict, str(Path(output_dir) / "heads.safetensors"))
    return out_dir_path
