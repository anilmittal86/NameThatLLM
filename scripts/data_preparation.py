"""
Data Preparation, Fidelity Filtering, and Master Pool V2 Assembly
================================================================
1. Loads official holdouts (`clean_benchmark_1800_rows.parquet` and `train.csv`) for zero-leak exclusion.
2. Evaluates `crtproduction` v17 (`ntl_open_pool.parquet`, `fidelity_by_source.csv`, `gemini37_unverified.parquet`)
   and strictly purges all sources with `fidelity_ratio < min_fidelity` (default 0.70) as well as CRT risk flags.
3. Propagates a global normalized-text blacklist from rejected/low-fidelity/wrong-reasoning sources across all
   upstream pools (`prvsiyan`, `seyitkaangunes`, `prvsiyan_instyle`) so polluted records cannot re-enter.
4. Fixes the `prvsiyan` Priority-1 `kimi-k3` non-thinking bug and `gemini-3.7-flash` low-fidelity proxy bug.
5. Saves unified `local_data/clean_master_pool_v2.parquet` with `prompt_canon` and `source_fidelity` metadata.
6. Generates the prioritized `~3,000` high-coverage `prompt_canon` gap target files for clean API completion.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "src"))

from namethatllm.config import LABELS

LABEL_SET = set(LABELS)
DEFAULT_WEAK_TARGET_MODELS: tuple[str, ...] = (
    "kimi-k3",
    "gemini-3.7-flash",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "gemini-3.6-flash",
    "claude-opus-4-8",
    "claude-fable-5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
)
_WORD_HINT_RE = re.compile(r"(?:\bwords?\b|\bsingle paragraph\b|\(words?\s+\d+)", re.IGNORECASE)
_MD_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s)", re.MULTILINE)


def normalize_text(text: object) -> str:
    """Normalizes whitespace for deterministic cross-source text deduplication and leak checks."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def load_fidelity_map(fidelity_csv: Path) -> dict[tuple[str, str], float]:
    """Loads `(source_dataset, model) -> fidelity_ratio` from `fidelity_by_source.csv`."""
    if not fidelity_csv.exists():
        return {}
    df = pd.read_csv(fidelity_csv, usecols=["source_dataset", "model", "fidelity_ratio"])
    out: dict[tuple[str, str], float] = {}
    for src, mdl, ratio in zip(df["source_dataset"], df["model"], df["fidelity_ratio"]):
        if pd.notna(src) and pd.notna(mdl) and pd.notna(ratio):
            out[(str(src).strip(), str(mdl).strip())] = float(ratio)
    return out


def is_clean_single_paragraph(raw_text: str, norm_text: str, min_words: int = 36, max_words: int = 275) -> bool:
    """Checks competition single-paragraph constraints: no newlines, no markdown list/heading, valid length."""
    if not norm_text or "\n" in raw_text or "\r" in raw_text:
        return False
    if _MD_LINE_RE.match(raw_text):
        return False
    wc = len(norm_text.split())
    return min_words <= wc <= max_words


def filter_text_rows(
    df_in: pd.DataFrame,
    source_name: str,
    priority: int,
    excluded_texts: set[str],
    blacklisted_texts: set[str],
    good_gemini37_texts: Optional[set[str]] = None,
    default_weight: float = 1.0,
    weight_col: Optional[str] = None,
    fidelity_map: Optional[dict[tuple[str, str], float]] = None,
    sub_source_col: Optional[str] = None,
    sub_source_override: Optional[str] = None,
    prompt_canon_map: Optional[dict[str, str]] = None,
    min_fidelity: float = 0.70,
    require_known_fidelity: bool = False,
    min_words: int = 36,
    max_words: int = 275,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standardizes and strictly filters one source DataFrame into the canonical pool schema."""
    stats = {"input": len(df_in), "kept": 0, "skipped_leak": 0, "skipped_blacklist": 0, "skipped_fidelity": 0, "skipped_format": 0}
    if len(df_in) == 0:
        return pd.DataFrame(), stats

    rows = []
    texts = df_in["text"].tolist()
    models = df_in["model"].astype(str).str.strip().tolist()
    weights = df_in[weight_col].tolist() if (weight_col and weight_col in df_in.columns) else [default_weight] * len(df_in)
    pcanons = df_in["prompt_canon"].tolist() if "prompt_canon" in df_in.columns else [None] * len(df_in)
    sub_srcs = (
        df_in[sub_source_col].astype(str).str.strip().tolist()
        if (sub_source_col and sub_source_col in df_in.columns)
        else [sub_source_override or source_name] * len(df_in)
    )

    for raw_text, model, raw_w, raw_pc, sub_src in zip(texts, models, weights, pcanons, sub_srcs):
        if model not in LABEL_SET or not isinstance(raw_text, str):
            stats["skipped_format"] += 1
            continue
        norm = normalize_text(raw_text)
        if not is_clean_single_paragraph(raw_text, norm, min_words=min_words, max_words=max_words):
            stats["skipped_format"] += 1
            continue
        if norm in excluded_texts:
            stats["skipped_leak"] += 1
            continue
        if norm in blacklisted_texts:
            stats["skipped_blacklist"] += 1
            continue
        if model == "gemini-3.7-flash" and good_gemini37_texts is not None and norm not in good_gemini37_texts:
            stats["skipped_blacklist"] += 1
            continue

        fid_val: Optional[float] = None
        if fidelity_map is not None:
            fid_val = fidelity_map.get((sub_src, model))
            if fid_val is None and require_known_fidelity:
                stats["skipped_fidelity"] += 1
                continue
            if fid_val is not None and fid_val < min_fidelity:
                stats["skipped_fidelity"] += 1
                continue

        try:
            weight = float(raw_w) if pd.notna(raw_w) else default_weight
        except (TypeError, ValueError):
            weight = default_weight

        pc = str(raw_pc).strip() if pd.notna(raw_pc) and str(raw_pc).strip() else None
        if pc is None and prompt_canon_map is not None:
            pc = prompt_canon_map.get(norm)

        rows.append(
            {
                "text": raw_text,
                "norm_text": norm,
                "model": model,
                "dataset_source": source_name,
                "sub_source": sub_src,
                "prompt_canon": pc,
                "priority": int(priority),
                "match_weight": float(weight),
                "source_fidelity": float(fid_val) if fid_val is not None else np.nan,
                "word_count": len(norm.split()),
            }
        )
        stats["kept"] += 1

    return pd.DataFrame(rows), stats


def load_crt_v17_reference(
    crt_dir: Path,
    min_fidelity: float = 0.70,
) -> tuple[pd.DataFrame, dict[tuple[str, str], float], set[str], set[str], dict[str, str]]:
    """
    Loads CRT v17 (`ntl_open_pool.parquet`, `fidelity_by_source.csv`, `gemini37_unverified.parquet`).
    Returns `(df_crt, fidelity_map, bad_texts, good_gemini37_texts, prompt_canon_map)`.
    """
    fidelity_map = load_fidelity_map(crt_dir / "fidelity_by_source.csv")
    op_path = crt_dir / "ntl_open_pool.parquet"
    cols = [
        "text",
        "model",
        "source_dataset",
        "prompt_canon",
        "match_weight",
        "flag_gemini37_routing_risk",
        "flag_train_prompt_regen",
        "flag_near_train_prompt",
        "flag_kb_proxy_after_ruling",
    ]
    df_crt = pd.read_parquet(op_path, columns=cols)
    norms = [normalize_text(t) for t in df_crt["text"]]
    df_crt["norm_text"] = norms

    fids = [fidelity_map.get((str(s).strip(), str(m).strip())) for s, m in zip(df_crt["source_dataset"], df_crt["model"])]
    df_crt["src_fid"] = fids

    flag_bad = (
        df_crt["flag_gemini37_routing_risk"].fillna(False).astype(bool)
        | df_crt["flag_train_prompt_regen"].fillna(False).astype(bool)
        | df_crt["flag_near_train_prompt"].fillna(False).astype(bool)
        | df_crt["flag_kb_proxy_after_ruling"].fillna(False).astype(bool)
    )
    low_fid_or_unrated = pd.Series(fids).isna().to_numpy() | (
        pd.Series(fids).fillna(-1.0).to_numpy() < float(min_fidelity)
    )

    bad_texts = set(df_crt.loc[flag_bad.to_numpy() | low_fid_or_unrated, "norm_text"])

    bad_gem_path = crt_dir / "gemini37_unverified.parquet"
    if bad_gem_path.exists():
        df_bad_gem = pd.read_parquet(bad_gem_path, columns=["text"])
        bad_texts.update(normalize_text(t) for t in df_bad_gem["text"])

    good_gem_mask = (df_crt["model"] == "gemini-3.7-flash") & (~flag_bad) & (~low_fid_or_unrated)
    good_gemini37_texts = set(df_crt.loc[good_gem_mask, "norm_text"])

    prompt_canon_map: dict[str, str] = {}
    for nh, pc in zip(norms, df_crt["prompt_canon"]):
        if nh and pd.notna(pc) and str(pc).strip():
            prompt_canon_map[nh] = str(pc).strip()

    clean_crt = df_crt.loc[~flag_bad.to_numpy() & ~low_fid_or_unrated].copy()
    return clean_crt, fidelity_map, bad_texts, good_gemini37_texts, prompt_canon_map


def prepare_prvsiyan_direct(
    prv_path: Path,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Loads `prvsiyan/passages.csv` safely:
    - Keeps ONLY `passes_funnel == True`.
    - Drops the misattributed non-thinking / Alibaba `kimi-k3` rows and low-fidelity `gemini-3.7-flash` rows
      (`0.334` fidelity in CRT), returning them in `rejected_norms` so they cannot sneak in via instyle pools.
    """
    usecols = ["text", "model", "passes_funnel", "reasoning_setting", "served_provider"]
    df_prv = pd.read_csv(prv_path, usecols=usecols)
    df_prv["norm_text"] = [normalize_text(t) for t in df_prv["text"]]

    is_funnel_ok = df_prv["passes_funnel"].fillna(False).astype(bool)
    is_bad_kimi = (df_prv["model"] == "kimi-k3") & (
        (~df_prv["served_provider"].astype(str).str.contains("Moonshot", case=False, na=False))
        | (~df_prv["reasoning_setting"].astype(str).str.contains("high", case=False, na=False))
    )
    is_bad_gem37 = df_prv["model"] == "gemini-3.7-flash"
    keep_mask = is_funnel_ok & (~is_bad_kimi) & (~is_bad_gem37)

    rejected_norms = set(df_prv.loc[~keep_mask, "norm_text"])
    return df_prv.loc[keep_mask].copy(), rejected_norms


def build_prompt_canon_gap_targets(
    df_pool: pd.DataFrame,
    prompts_parquet: Path,
    out_targets_parquet: Path,
    out_targets_csv: Path,
    out_slots_csv: Path,
    train_word_medians: dict[str, int],
    n_panels: int = 3000,
    weak_models: Iterable[str] = DEFAULT_WEAK_TARGET_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes verified prompt-panel coverage inside the purified pool against `prompts.parquet`
    and selects the highest-coverage incomplete `prompt_canon` panels missing weak target models.
    """
    prompts_df = pd.read_parquet(prompts_parquet, columns=["prompt_canon", "prompt_text", "panel_labels", "passages"])
    weak_set = set(weak_models)

    valid_pc = df_pool[df_pool["prompt_canon"].notna()]
    present_by_pc = valid_pc.groupby("prompt_canon")["model"].agg(lambda s: set(s)).to_dict()

    all_labels_sorted = sorted(LABEL_SET)
    panel_rows = []
    for pc, pt, crt_pl, crt_pass in zip(
        prompts_df["prompt_canon"], prompts_df["prompt_text"], prompts_df["panel_labels"], prompts_df["passages"]
    ):
        pc_str = str(pc).strip()
        pt_str = str(pt)
        present = present_by_pc.get(pc_str, set()) & LABEL_SET
        missing = [m for m in all_labels_sorted if m not in present]
        if not missing:
            continue
        missing_weak = [m for m in DEFAULT_WEAK_TARGET_MODELS if m in missing]
        has_hint = bool(_WORD_HINT_RE.search(pt_str))
        panel_rows.append(
            {
                "prompt_canon": pc_str,
                "prompt_text": pt_str,
                "has_word_constraint": has_hint,
                "n_present_clean": len(present),
                "n_missing": len(missing),
                "n_missing_weak": len(missing_weak),
                "crt_panel_labels": int(crt_pl),
                "crt_passages": int(crt_pass),
                "present_models": ",".join(sorted(present)),
                "missing_models": ",".join(missing),
                "missing_weak_models": ",".join(missing_weak),
            }
        )

    df_gaps = pd.DataFrame(panel_rows)
    # Prioritize panels that already have high verified clean coverage (best contrastive utility),
    # are missing weak target models, and already carry a competition word-count constraint.
    df_gaps = df_gaps[df_gaps["n_missing_weak"] > 0].sort_values(
        by=["n_present_clean", "n_missing_weak", "has_word_constraint", "crt_panel_labels"],
        ascending=[False, False, False, False],
    )
    df_top = df_gaps.head(n_panels).reset_index(drop=True)

    slot_rows = []
    for _, row in df_top.iterrows():
        pc_str = row["prompt_canon"]
        pt_str = row["prompt_text"]
        has_hint = bool(row["has_word_constraint"])
        seed_int = int(hashlib.sha256(pc_str.encode("utf-8")).hexdigest()[:8], 16)
        missing_list = [m for m in row["missing_models"].split(",") if m]
        for idx_m, m in enumerate(missing_list):
            med = int(train_word_medians.get(m, 105))
            # Deterministic target word count around the model's train median clipped to [45, 220]
            offset = ((seed_int + idx_m * 31) % 61) - 30
            target_n = int(np.clip(med + offset, 45, 220))
            prompt_to_send = (
                pt_str
                if has_hint
                else f"{pt_str.rstrip()}\n\nWrite a single paragraph of about {target_n} words, with no title or headings."
            )
            slot_rows.append(
                {
                    "prompt_canon": pc_str,
                    "missing_model": m,
                    "is_weak_target": m in weak_set,
                    "n_present_clean": int(row["n_present_clean"]),
                    "n_missing_panel": int(row["n_missing"]),
                    "has_word_constraint": has_hint,
                    "suggested_target_words": target_n,
                    "prompt_to_send": prompt_to_send,
                }
            )

    df_slots = pd.DataFrame(slot_rows)
    df_top.to_parquet(out_targets_parquet, index=False)
    df_top.to_csv(out_targets_csv, index=False)
    df_slots.to_csv(out_slots_csv, index=False)
    return df_top, df_slots


def build_clean_master_pool_v2(
    data_dir: Path = Path("local_data"),
    min_fidelity: float = 0.70,
    min_words: int = 36,
    max_words: int = 275,
    include_legacy_pools: bool = False,
    n_gap_panels: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds `clean_master_pool_v2.parquet` and prioritized `prompt_canon` gap targets."""
    print("=" * 72)
    print(f"BUILDING CLEAN MASTER POOL V2 (min_fidelity >= {min_fidelity:.2f})")
    print("=" * 72)

    bench_path = data_dir / "clean_benchmark_1800_rows.parquet"
    train_path = data_dir / "train.csv"
    crt_dir = data_dir / "external_datasets" / "crtproduction"

    df_b1800 = pd.read_parquet(bench_path, columns=["text"])
    df_train = pd.read_csv(train_path, usecols=["text", "model"])
    b18_norms = {normalize_text(t) for t in df_b1800["text"]}
    train_norms = {normalize_text(t) for t in df_train["text"]}
    excluded_texts = b18_norms | train_norms
    print(f"Holdouts excluded: bench1800={len(b18_norms)}, train={len(train_norms)}, total={len(excluded_texts)}")

    train_word_medians = {
        str(m): int(round(float(wc.median())))
        for m, wc in df_train.assign(wc=df_train["text"].astype(str).str.split().str.len()).groupby("model")["wc"]
    }

    clean_crt, fidelity_map, bad_crt_texts, good_gem37_texts, prompt_canon_map = load_crt_v17_reference(
        crt_dir=crt_dir,
        min_fidelity=min_fidelity,
    )
    print(
        f"CRT v17 loaded: clean_rows={len(clean_crt):,}, blacklisted_texts={len(bad_crt_texts):,}, "
        f"verified_gemini37={len(good_gem37_texts):,}, prompt_canons_mapped={len(prompt_canon_map):,}"
    )

    blacklisted_texts = set(bad_crt_texts)
    parts: list[pd.DataFrame] = []

    # Priority 1A: prvsiyan direct API (kimi non-high & proxy gemini-3.7 purged)
    prv_path = data_dir / "external_datasets" / "prvsiyan" / "passages.csv"
    if prv_path.exists():
        df_prv_ok, prv_rejected = prepare_prvsiyan_direct(prv_path)
        blacklisted_texts.update(prv_rejected)
        p_prv, st_prv = filter_text_rows(
            df_prv_ok,
            source_name="prvsiyan_direct_api",
            priority=1,
            excluded_texts=excluded_texts,
            blacklisted_texts=blacklisted_texts,
            good_gemini37_texts=good_gem37_texts,
            default_weight=2.0,
            prompt_canon_map=prompt_canon_map,
            min_words=min_words,
            max_words=max_words,
        )
        parts.append(p_prv)
        print(f"  [prvsiyan_direct_api] {st_prv}")

    # Priority 1B: seyitkaangunes Claude API (fidelity >= min_fidelity enforced per sibling model)
    sey_path = data_dir / "external_datasets" / "seyitkaangunes" / "ntl_extra_passages_claude.csv"
    if sey_path.exists():
        df_sey = pd.read_csv(sey_path, usecols=["text", "model"])
        p_sey, st_sey = filter_text_rows(
            df_sey,
            source_name="seyitkaangunes_claude_api",
            priority=1,
            excluded_texts=excluded_texts,
            blacklisted_texts=blacklisted_texts,
            good_gemini37_texts=good_gem37_texts,
            default_weight=2.0,
            fidelity_map=fidelity_map,
            sub_source_override="seyitkaangunes/ntl-extra-api-passages",
            prompt_canon_map=prompt_canon_map,
            min_fidelity=min_fidelity,
            require_known_fidelity=False,
            min_words=min_words,
            max_words=max_words,
        )
        parts.append(p_sey)
        print(f"  [seyitkaangunes_claude_api] {st_sey}")

    # Priority 2: prvsiyan in-style pool (fidelity >= min_fidelity enforced via `source` column)
    inst_path = data_dir / "external_datasets" / "prvsiyan_instyle" / "pool_v1.parquet"
    if inst_path.exists():
        df_inst = pd.read_parquet(inst_path, columns=["text", "model", "source"])
        p_inst, st_inst = filter_text_rows(
            df_inst,
            source_name="prvsiyan_instyle_pool_v1",
            priority=2,
            excluded_texts=excluded_texts,
            blacklisted_texts=blacklisted_texts,
            good_gemini37_texts=good_gem37_texts,
            default_weight=1.5,
            fidelity_map=fidelity_map,
            sub_source_col="source",
            prompt_canon_map=prompt_canon_map,
            min_fidelity=min_fidelity,
            require_known_fidelity=False,
            min_words=min_words,
            max_words=max_words,
        )
        parts.append(p_inst)
        print(f"  [prvsiyan_instyle_pool_v1] {st_inst}")

    # Priority 3: crtproduction v17 open pool (strictly >= min_fidelity + zero bad flags)
    p_crt, st_crt = filter_text_rows(
        clean_crt,
        source_name="crtproduction_open_pool_v17",
        priority=3,
        excluded_texts=excluded_texts,
        blacklisted_texts=blacklisted_texts,
        good_gemini37_texts=good_gem37_texts,
        default_weight=1.0,
        weight_col="match_weight",
        fidelity_map=fidelity_map,
        sub_source_col="source_dataset",
        prompt_canon_map=prompt_canon_map,
        min_fidelity=min_fidelity,
        require_known_fidelity=True,
        min_words=min_words,
        max_words=max_words,
    )
    parts.append(p_crt)
    print(f"  [crtproduction_open_pool_v17] {st_crt}")

    if include_legacy_pools:
        for legacy_name, legacy_path, prio in [
            ("ankiteshtiwari_curated_pool", data_dir / "external_datasets" / "ankiteshtiwari" / "curated_pool.parquet", 4),
            ("pool_120k_cleaned", data_dir / "pool_120k_cleaned.parquet", 5),
        ]:
            if legacy_path.exists():
                df_leg = pd.read_parquet(legacy_path, columns=["text", "model"])
                p_leg, st_leg = filter_text_rows(
                    df_leg,
                    source_name=legacy_name,
                    priority=prio,
                    excluded_texts=excluded_texts,
                    blacklisted_texts=blacklisted_texts,
                    good_gemini37_texts=good_gem37_texts,
                    default_weight=0.75,
                    prompt_canon_map=prompt_canon_map,
                    min_words=min_words,
                    max_words=max_words,
                )
                parts.append(p_leg)
                print(f"  [{legacy_name}] {st_leg}")

    df_all = pd.concat(parts, ignore_index=True)
    df_all = df_all.sort_values(by=["priority", "match_weight"], ascending=[True, False])
    df_dedup = df_all.drop_duplicates(subset=["norm_text"], keep="first").reset_index(drop=True)

    # Hard verification assertions
    b18_leaks = int(df_dedup["norm_text"].isin(b18_norms).sum())
    train_leaks = int(df_dedup["norm_text"].isin(train_norms).sum())
    black_leaks = int(df_dedup["norm_text"].isin(blacklisted_texts).sum())
    assert b18_leaks == 0, f"FATAL: Bench-1800 leak detected ({b18_leaks})"
    assert train_leaks == 0, f"FATAL: train.csv leak detected ({train_leaks})"
    assert black_leaks == 0, f"FATAL: Blacklisted row leak detected ({black_leaks})"
    bad_g37_leaks = int(((df_dedup["model"] == "gemini-3.7-flash") & (~df_dedup["norm_text"].isin(good_gem37_texts))).sum())
    assert bad_g37_leaks == 0, f"FATAL: Unverified Gemini 3.7 leak detected ({bad_g37_leaks})"

    df_dedup["id"] = [f"clean_pool_v2_{i:07d}" for i in range(len(df_dedup))]
    output_cols = [
        "id",
        "text",
        "model",
        "dataset_source",
        "sub_source",
        "prompt_canon",
        "priority",
        "match_weight",
        "source_fidelity",
        "word_count",
    ]
    df_final = df_dedup[output_cols].copy()

    out_path = data_dir / "clean_master_pool_v2.parquet"
    df_final.to_parquet(out_path, index=False)
    print(f"\nSaved purified pool: {len(df_final):,} rows -> {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print("Per-class counts in clean_master_pool_v2:")
    print(df_final["model"].value_counts().reindex(LABELS).fillna(0).astype(int).to_string())

    out_targets_pq = data_dir / f"prompt_canon_gap_targets_{n_gap_panels}.parquet"
    out_targets_csv = data_dir / f"prompt_canon_gap_targets_{n_gap_panels}.csv"
    out_slots_csv = data_dir / f"prompt_canon_gap_slots_{n_gap_panels}.csv"
    df_top, df_slots = build_prompt_canon_gap_targets(
        df_pool=df_final,
        prompts_parquet=crt_dir / "prompts.parquet",
        out_targets_parquet=out_targets_pq,
        out_targets_csv=out_targets_csv,
        out_slots_csv=out_slots_csv,
        train_word_medians=train_word_medians,
        n_panels=n_gap_panels,
    )
    print(
        f"\nSaved {len(df_top):,} high-coverage prompt gap panels ({len(df_slots):,} total missing model slots) "
        f"-> {out_targets_csv} & {out_slots_csv}"
    )
    if len(df_slots) > 0:
        print("Missing slots per model across top target panels:")
        print(df_slots["missing_model"].value_counts().to_string())

    return df_final, df_top, df_slots


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build purified clean_master_pool_v2.parquet and prompt_canon gap targets.")
    parser.add_argument("--data-dir", type=Path, default=Path("local_data"))
    parser.add_argument("--min-fidelity", type=float, default=0.70)
    parser.add_argument("--min-words", type=int, default=36)
    parser.add_argument("--max-words", type=int, default=275)
    parser.add_argument("--include-legacy-pools", action="store_true")
    parser.add_argument("--gap-panels", type=int, default=3000)
    args = parser.parse_args(argv)

    build_clean_master_pool_v2(
        data_dir=args.data_dir,
        min_fidelity=args.min_fidelity,
        min_words=args.min_words,
        max_words=args.max_words,
        include_legacy_pools=args.include_legacy_pools,
        n_gap_panels=args.gap_panels,
    )


if __name__ == "__main__":
    main()
