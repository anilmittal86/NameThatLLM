"""
Dataset Loading, Zero-Leak Quarantine, Conflict-Free Deduplication, Sample Weight Normalization, and Samplers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from ..ablation import _text_hash, build_unseen_holdout_dataset
from ..config import LAB2I, LABELS
from .backbones.base import TokenizationSpec


PROBLEM_WATCH_CLASSES: tuple[str, ...] = (
    "kimi-k3",
    "glm-5.3-flash",
    "glm-5.3",
    "gpt-5.6-sol",
    "claude-fable-5",
    "gpt-5.6-terra",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    LABELS[14],
)


def norm_text_hash(text: str) -> str:
    """Normalized whitespace SHA-1 hash for strict zero-leak deduplication."""
    norm = " ".join(str(text).strip().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _select_stage1_pool_files(pool_list: list[Path], use_v2_only: bool = True) -> list[Path]:
    """Prefers clean_master_pool_v2 files exclusively when present so v1 rows are not mixed back in."""
    if use_v2_only:
        v2_files = [p for p in pool_list if "clean_master_pool_v2" in Path(p).name.lower()]
        if v2_files:
            return v2_files
    return list(pool_list)


def _normalize_match_weights(
    values: Any,
    weight_clip: tuple[float, float] = (0.5, 2.5),
    groups: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    """Clips numeric sample match weights into a safe range and normalizes mean to 1.0 (per group if provided)."""
    lo, hi = float(weight_clip[0]), float(weight_clip[1])
    raw_w = pd.to_numeric(values, errors="coerce")
    if isinstance(raw_w, pd.Series):
        raw_arr = raw_w.fillna(1.0).to_numpy(dtype=np.float32)
    else:
        raw_arr = np.nan_to_num(np.asarray(raw_w, dtype=np.float32), nan=1.0)
    clipped = np.clip(raw_arr, lo, hi).astype(np.float32)
    if groups is not None:
        grp_arr = np.asarray(groups)
        if len(grp_arr) == len(clipped) and len(clipped) > 0:
            out = clipped.copy()
            for g in np.unique(grp_arr):
                mask = grp_arr == g
                if mask.any():
                    out[mask] = clipped[mask] / max(float(clipped[mask].mean()), 1e-6)
            return out.astype(np.float32)
    return (clipped / max(float(clipped.mean()), 1e-6)).astype(np.float32)


def _load_stage1_pool_df(pool_list: list[Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Loads Stage-1 pool files preserving optional match_weight, fidelity, and prompt metadata."""
    opt_cols = ["text", "model", "match_weight", "source_fidelity", "prompt_canon", "priority"]
    dfs = []
    pool_source_counts: dict[str, int] = {}
    for p in pool_list:
        try:
            p_str = str(p)
            if p_str.endswith(".parquet"):
                if pq is not None:
                    avail_cols = set(pq.read_schema(p).names)
                    cols = [c for c in opt_cols if c in avail_cols]
                    df_one = pd.read_parquet(p, columns=cols if cols else ["text", "model"])
                else:
                    df_one = pd.read_parquet(p)
                    cols = [c for c in opt_cols if c in df_one.columns]
                    if cols:
                        df_one = df_one[cols]
            else:
                hdr = pd.read_csv(p, nrows=0).columns
                cols = [c for c in opt_cols if c in hdr]
                df_one = pd.read_csv(p, usecols=cols if cols else ["text", "model"])
            pool_source_counts[Path(p).name] = len(df_one)
            dfs.append(df_one)
        except Exception as e:
            print(f"[Trainer] Could not read pool {p}: {e}")
    df_p = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["text", "model"])
    return df_p, pool_source_counts


def load_clean_stage1_pool(
    pool_list: list[Path],
    input_dir: Path | None = None,
    train_path: Path | None = None,
    bench_bundle: tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray] | None = None,
    active_labels: list[str] | None = None,
    exclude_train_overlap: bool = True,
    weight_clip: tuple[float, float] = (0.5, 2.5),
    use_match_weight: bool = True,
    min_char_len: int = 40,
    include_unseen_holdout: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Unified Stage-1 pool loader: ingests pool files, filters valid rows, drops cross-label
    `_norm_hash` conflicts, deterministically deduplicates by `(priority asc, match_weight desc)`,
    quarantines all unseen holdout / Bench-1800 / train.csv overlaps, and normalizes `match_weight` per class.
    """
    df_p, pool_source_counts = _load_stage1_pool_df(pool_list)
    total_raw_rows = sum(pool_source_counts.values())

    allowed = set(active_labels) if active_labels is not None else set(LABELS)
    df_p = df_p.dropna(subset=["text", "model"])
    df_p["text"] = df_p["text"].astype(str)
    if min_char_len > 0:
        df_p = df_p[df_p["text"].str.strip().str.len() >= min_char_len]
    df_p = df_p[df_p["model"].isin(allowed)].reset_index(drop=True)

    df_p["_hash"] = [_text_hash(t) for t in df_p["text"]]
    df_p["_norm_hash"] = [norm_text_hash(t) for t in df_p["text"]]
    label_nunique = df_p.groupby("_norm_hash")["model"].transform("nunique")
    n_conflict_rows = int((label_nunique > 1).sum())
    if n_conflict_rows > 0:
        n_conflict_hashes = int(df_p.loc[label_nunique > 1, "_norm_hash"].nunique())
        print(
            f"[Trainer] Dropped {n_conflict_rows} rows across {n_conflict_hashes} cross-label _norm_hash conflicts before deduplication."
        )
        df_p = df_p[label_nunique == 1].reset_index(drop=True)

    sort_cols: list[str] = []
    sort_asc: list[bool] = []
    if "priority" in df_p.columns:
        sort_cols.append("priority")
        sort_asc.append(True)
    if "match_weight" in df_p.columns:
        sort_cols.append("match_weight")
        sort_asc.append(False)
    if sort_cols:
        df_p = df_p.sort_values(by=sort_cols, ascending=sort_asc, kind="mergesort")
    df_p = df_p.drop_duplicates(subset=["_norm_hash"], keep="first").reset_index(drop=True)
    dedup_rows = len(df_p)

    quarantine_dropped = 0
    try:
        holdout_raw: set[str] = set()
        holdout_norm: set[str] = set()
        if include_unseen_holdout and input_dir is not None:
            _, holdout_hashes = build_unseen_holdout_dataset(input_dir, train_path)
            if holdout_hashes:
                holdout_raw.update(holdout_hashes)
                holdout_norm.update(holdout_hashes)
        if bench_bundle is not None:
            for t_b in bench_bundle[0]:
                holdout_raw.add(_text_hash(t_b))
                holdout_norm.add(norm_text_hash(t_b))
        elif input_dir is not None and input_dir.exists():
            bench_paths = list(input_dir.rglob("clean_benchmark_1800_rows.parquet"))
            if bench_paths:
                df_bench_q = pd.read_parquet(bench_paths[0])
                for t_b in df_bench_q["text"].dropna():
                    holdout_raw.add(_text_hash(t_b))
                    holdout_norm.add(norm_text_hash(t_b))
        if train_path and Path(train_path).exists() and exclude_train_overlap:
            df_tr_q = pd.read_csv(train_path, usecols=["text"])
            for t_q in df_tr_q["text"].dropna():
                holdout_raw.add(_text_hash(t_q))
                holdout_norm.add(norm_text_hash(t_q))

        if holdout_raw or holdout_norm:
            mask_clean = (~df_p["_hash"].isin(holdout_raw)) & (~df_p["_norm_hash"].isin(holdout_norm))
            quarantine_dropped = int(len(df_p) - mask_clean.sum())
            if quarantine_dropped > 0:
                print(
                    f"[Trainer] Quarantined {quarantine_dropped} pool rows that overlapped with unseen holdout / clean benchmark / train."
                )
            df_p = df_p[mask_clean].reset_index(drop=True)
    except Exception as e:
        print(f"[Trainer] Note on holdout quarantine: {e}")

    if use_match_weight and ("match_weight" in df_p.columns):
        df_p["_sample_weight"] = _normalize_match_weights(
            df_p["match_weight"], weight_clip=weight_clip, groups=df_p["model"]
        )
        print(
            f"[Trainer] Loaded & clipped per-class Stage-1 match_weight (clip={weight_clip}, "
            f"mean={df_p['_sample_weight'].mean():.3f}, max={df_p['_sample_weight'].max():.3f})"
        )

    stats = {
        "source_counts": pool_source_counts,
        "total_raw_rows": int(total_raw_rows),
        "dedup_rows": int(dedup_rows),
        "quarantine_dropped": int(quarantine_dropped),
        "conflict_rows": int(n_conflict_rows),
    }
    return df_p, stats


def load_zero_leak_stage1_pool(
    datasets_dir: Path,
    weight_clip: tuple[float, float] = (0.5, 2.5),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads clean_benchmark_1800_rows.parquet, train.csv, and builds the deduplicated
    Stage-1 pool strictly excluding all normalized text hashes in Bench-1800 or train.csv.
    """
    bench_path = datasets_dir / "clean_benchmark_1800_rows.parquet"
    df_bench = pd.read_parquet(bench_path)
    df_bench = df_bench[df_bench["model"].isin(LABELS)].reset_index(drop=True)
    bench_hashes = set(df_bench["text"].map(norm_text_hash))

    train_path = datasets_dir / "train.csv"
    df_train = pd.read_csv(train_path)
    df_train = df_train[df_train["model"].isin(LABELS)].reset_index(drop=True)
    df_train = df_train[~df_train["text"].map(norm_text_hash).isin(bench_hashes)].reset_index(drop=True)
    train_hashes = set(df_train["text"].map(norm_text_hash))

    excl_hashes = bench_hashes | train_hashes
    v2_path = datasets_dir / "clean_master_pool_v2.parquet"
    v1_path = datasets_dir / "clean_master_pool.parquet"
    master_path = v2_path if v2_path.exists() else v1_path

    df_master, _ = load_clean_stage1_pool(
        pool_list=[master_path],
        input_dir=datasets_dir,
        train_path=train_path,
        bench_bundle=(df_bench["text"].astype(str).tolist(), np.array([]), [], np.array([]), np.array([])),
        active_labels=list(LABELS),
        exclude_train_overlap=True,
        weight_clip=weight_clip,
        use_match_weight=False,
        min_char_len=0,
        include_unseen_holdout=False,
    )
    if "priority" not in df_master.columns:
        df_master["priority"] = 1
    if "match_weight" not in df_master.columns:
        df_master["match_weight"] = 1.0
    df_master["nh"] = df_master["_norm_hash"]

    master_hashes = set(df_master["nh"])
    parts = [df_master[["text", "model", "priority", "match_weight", "nh"]]]
    pool_v4_path = datasets_dir / "pool_v4.parquet"
    if not v2_path.exists() and pool_v4_path.exists():
        df_v4 = pd.read_parquet(pool_v4_path, columns=["text", "model"])
        df_v4 = df_v4[df_v4["model"].isin(LABELS)].copy()
        df_v4["nh"] = df_v4["text"].map(norm_text_hash)
        df_extra = df_v4[~df_v4["nh"].isin(excl_hashes | master_hashes)].drop_duplicates("nh").copy()
        if len(df_extra) > 0:
            df_extra["priority"] = 2
            df_extra["match_weight"] = 1.0
            parts.append(df_extra[["text", "model", "priority", "match_weight", "nh"]])

    df_pool = pd.concat(parts, ignore_index=True).drop(columns=["nh"]).reset_index(drop=True)
    df_pool["sample_weight"] = _normalize_match_weights(df_pool["match_weight"], weight_clip=weight_clip, groups=df_pool["model"])
    df_pool["label_idx"] = np.array([LAB2I[m] for m in df_pool["model"]], dtype=np.int64)

    pool_hashes = set(df_pool["text"].map(norm_text_hash))
    assert len(pool_hashes & bench_hashes) == 0, "Leak detected against Bench-1800!"
    assert len(pool_hashes & train_hashes) == 0, "Leak detected against train.csv!"
    return df_pool, df_train, df_bench


def build_targeted_epoch_indices(
    labels: np.ndarray,
    class_quotas: dict[str, int],
    epoch: int,
    seed: int = 42,
    class_labels: list[str] | None = None,
) -> np.ndarray:
    """Builds one epoch's index list matching per-class quotas via modular sliding slices and clean oversampling."""
    active_names = class_labels if class_labels is not None else list(LABELS)
    chosen_parts: list[np.ndarray] = []
    rng_ep = np.random.RandomState(seed + 1009 * (epoch + 1))
    for c_idx, name in enumerate(active_names):
        pool_idx = np.where(labels == c_idx)[0]
        n_avail = len(pool_idx)
        assert n_avail > 0, f"Class {name} has 0 examples in Stage-1 pool!"
        quota = int(class_quotas.get(name, n_avail))
        if quota <= 0:
            continue
        perm = pool_idx[np.random.RandomState(seed + 97 * c_idx).permutation(n_avail)]
        if quota <= n_avail:
            start = (epoch * quota) % n_avail
            if start + quota <= n_avail:
                sel = perm[start : start + quota]
            else:
                first = perm[start:]
                sel = np.concatenate([first, perm[: quota - len(first)]])
        else:
            reps = [perm] * (quota // n_avail)
            rem = quota % n_avail
            if rem > 0:
                reps.append(rng_ep.choice(perm, size=rem, replace=False))
            sel = np.concatenate(reps)
        chosen_parts.append(sel)

    out = np.concatenate(chosen_parts) if chosen_parts else np.array([], dtype=np.int64)
    rng_ep.shuffle(out)
    return out


def _load_benchmark_holdout(
    input_dir: Path, subset_size: int = 0, seed: int = 42
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray] | None:
    """Loads clean_benchmark_1800_rows.parquet and optionally builds a deterministic stratified slice."""
    bench_paths = list(input_dir.rglob("clean_benchmark_1800_rows.parquet")) if input_dir.exists() else []
    if not bench_paths:
        for cand in (Path("clean_benchmark_1800_rows.parquet"), Path("local_data/clean_benchmark_1800_rows.parquet")):
            if cand.exists():
                bench_paths = [cand]
                break
    if not bench_paths:
        return None
    try:
        df_b = pd.read_parquet(bench_paths[0])
        df_b = df_b[df_b["model"].isin(LABELS)].reset_index(drop=True)
        if df_b.empty:
            return None
        full_texts = df_b["text"].astype(str).tolist()
        full_y = np.array([LAB2I[m] for m in df_b["model"]], dtype=np.int64)
        if subset_size > 0 and subset_size < len(full_y):
            rng = np.random.RandomState(seed)
            per_cls = max(1, subset_size // len(LABELS))
            idx_parts = []
            for c_i in range(len(LABELS)):
                c_idx = np.where(full_y == c_i)[0]
                if len(c_idx) > 0:
                    take = min(per_cls, len(c_idx))
                    idx_parts.append(rng.choice(c_idx, size=take, replace=False))
            sub_idx = np.sort(np.concatenate(idx_parts)).astype(np.int64)
            sub_texts = [full_texts[int(i)] for i in sub_idx]
            sub_y = full_y[sub_idx]
        else:
            sub_idx = np.arange(len(full_y), dtype=np.int64)
            sub_texts, sub_y = full_texts, full_y
        return full_texts, full_y, sub_texts, sub_y, sub_idx
    except Exception as e:
        print(f"[Trainer] Note loading benchmark holdout: {e}")
        return None


class PromptGroupBatchSampler(Sampler[list[int]]):
    """Yields entire prompt clusters per step so within-prompt centering matches inference."""

    def __init__(self, prompt_ids: list[Any], batch_size: int | None = None, shuffle: bool = True, seed: int = 42):
        super().__init__()
        self.groups: dict[Any, list[int]] = {}
        for idx, pid in enumerate(prompt_ids):
            self.groups.setdefault(pid, []).append(idx)
        self.group_keys = list(self.groups.keys())
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, ep: int) -> None:
        self.epoch = ep

    def __len__(self) -> int:
        return len(self.group_keys)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        keys = list(self.group_keys)
        if self.shuffle:
            rng.shuffle(keys)
        for k in keys:
            idxs = list(self.groups[k])
            if self.shuffle:
                rng.shuffle(idxs)
            yield idxs


class TextSequenceDataset(Dataset):
    """Dataset applying exact head+tail truncation matching neural_extractor.UniversalLoRATrunkLoader."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        prompt_ids: list[Any] | None,
        tok: Any,
        max_len: int = 384,
        end_id: int | None = 151643,
        cls_id: int | None = None,
        sample_weights: list[float] | None = None,
        tok_spec: TokenizationSpec | None = None,
    ):
        self.texts = texts
        self.labels = labels
        self.prompt_ids = prompt_ids
        self.tok = tok
        self.tok_spec = tok_spec
        self.max_len = tok_spec.max_len if tok_spec is not None else max_len
        self.end_id = tok_spec.end_id if tok_spec is not None else end_id
        self.cls_id = tok_spec.cls_id if tok_spec is not None else cls_id
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        full_ids = self.tok(str(self.texts[idx]), add_special_tokens=False, truncation=False)["input_ids"]
        if len(full_ids) > self.max_len:
            head_len = self.max_len // 2
            tail_len = self.max_len - head_len
            raw_ids = full_ids[:head_len] + full_ids[-tail_len:]
        else:
            raw_ids = full_ids
        input_ids = ([self.cls_id] if self.cls_id is not None else []) + raw_ids
        if self.end_id is not None:
            input_ids = input_ids + [self.end_id]
        item: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.prompt_ids is not None:
            item["prompt_id"] = self.prompt_ids[idx]
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)
        return item


def collate_text_batch(batch: list[dict[str, Any]], pad_token_id: int = 151643) -> dict[str, Any]:
    input_ids = [item["input_ids"] for item in batch]
    masks = [item.get("attention_mask", torch.ones_like(item["input_ids"])) for item in batch]
    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    res: dict[str, Any] = {"input_ids": padded_ids, "attention_mask": padded_mask}
    if "label" in batch[0]:
        res["labels"] = torch.stack([item["label"] for item in batch])
    if "prompt_id" in batch[0]:
        res["prompt_ids"] = [item["prompt_id"] for item in batch]
    if "sample_weight" in batch[0]:
        res["sample_weights"] = torch.stack([item["sample_weight"] for item in batch])
    return res


class BucketedLengthBatchSampler(Sampler[list[int]]):
    """Groups indices into mega-buckets sorted by length with jittered length sorting to minimize padding."""

    def __init__(self, lengths: list[int], batch_size: int, bucket_size_multiplier: int = 50, shuffle: bool = True, seed: int = 42):
        super().__init__()
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, ep: int) -> None:
        self.epoch = ep

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        for s in range(0, len(indices), self.bucket_size):
            bucket = indices[s : s + self.bucket_size]
            lens = np.array([self.lengths[i] for i in bucket], dtype=np.float64)
            if self.shuffle and len(bucket) > 1:
                lens = lens + rng.uniform(-12.0, 12.0, size=len(bucket))
            sorted_bucket = bucket[np.argsort(lens)]
            for b in range(0, len(sorted_bucket), self.batch_size):
                batches.append(sorted_bucket[b : b + self.batch_size].tolist())
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch
