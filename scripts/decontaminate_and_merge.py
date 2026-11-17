"""
Decontamination and Master Pool Assembly Script
==============================================
1. Purges corrupted Gemini 3.7 records and proxy misroutings from pool_120k.
2. Ingests high-fidelity external datasets:
   - crtproduction/name-that-llm-open-pool
   - prvsiyan/prvsiyans-extended-name-that-llm-dataset (with non-thinking kimi-k3)
   - seyitkaangunes/ntl-extra-api-passages (Anthropic Claude siblings)
   - prvsiyan/ntl-public-in-style-pool-v1
   - ankiteshtiwari/ntl-curated-pool-routed
3. Strictly removes any overlap with clean_benchmark_1800_rows.parquet.
4. Performs global deduplication with priority ranking and style match weighting.
5. Saves clean decontaminated master pool to local_data/clean_master_pool.parquet.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Canonical 18-class labels
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
LABEL_SET = set(LABELS)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())

def main():
    print("=" * 70)
    print("STARTING DATASET DECONTAMINATION AND UNIFICATION")
    print("=" * 70)

    # 1. Load benchmark holdout (must NEVER leak into training)
    benchmark_path = Path("local_data/clean_benchmark_1800_rows.parquet")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")
    df_b1800 = pd.read_parquet(benchmark_path)
    b_norm_texts = set(df_b1800["text"].apply(normalize_text))
    print(f"Loaded benchmark holdout: {len(df_b1800)} rows ({len(b_norm_texts)} unique normalized texts)")

    # 2. Load corrupt Gemini reference
    bad_gem_path = Path("local_data/external_datasets/crtproduction/gemini37_unverified.parquet")
    if bad_gem_path.exists():
        df_bad_gem = pd.read_parquet(bad_gem_path)
        bad_gem_texts = set(df_bad_gem["text"].apply(normalize_text))
        print(f"Loaded corrupt Gemini reference: {len(df_bad_gem)} rows ({len(bad_gem_texts)} unique texts)")
    else:
        bad_gem_texts = set()
        print("WARNING: gemini37_unverified.parquet not found!")

    # 3. Clean legacy pool_120k.parquet
    p120_path = Path("local_data/pool_120k.parquet")
    if not p120_path.exists():
        raise FileNotFoundError(f"pool_120k not found at {p120_path}")
    df_p120 = pd.read_parquet(p120_path)
    print(f"\n--- Processing Legacy pool_120k ({len(df_p120)} rows) ---")
    df_p120["norm_text"] = df_p120["text"].apply(normalize_text)

    # Find verified Gemini in crtproduction open pool for cross-referencing
    op_path = Path("local_data/external_datasets/crtproduction/ntl_open_pool.parquet")
    good_gem_texts = set()
    if op_path.exists():
        df_op = pd.read_parquet(op_path, columns=["model", "text"])
        good_gem = df_op[df_op["model"] == "gemini-3.7-flash"]
        good_gem_texts = set(good_gem["text"].apply(normalize_text))
        print(f"Verified Gemini 3.7 texts in crtproduction open pool: {len(good_gem_texts)}")

    # Flag corrupt gemini in pool_120k:
    # A) in bad_gem_texts
    # B) gemini-3.7-flash from proxy sources that is not verified in good_gem_texts
    is_gem37 = df_p120["model"] == "gemini-3.7-flash"
    matches_bad_gem = df_p120["norm_text"].isin(bad_gem_texts)
    unverified_proxy_gem = is_gem37 & (~df_p120["norm_text"].isin(good_gem_texts))
    corrupt_gem_mask = is_gem37 & (matches_bad_gem | unverified_proxy_gem)

    # Benchmark leak check
    leak_b1800_mask = df_p120["norm_text"].isin(b_norm_texts)
    
    # Save purged corrupt rows
    df_purged = df_p120[corrupt_gem_mask].copy()
    os.makedirs("local_data/quarantine", exist_ok=True)
    purged_path = Path("local_data/quarantine/purged_gemini37_corrupt.parquet")
    df_purged.to_parquet(purged_path, index=False)
    all_bad_gemini_texts = bad_gem_texts.union(set(df_purged["norm_text"]))
    print(f"Purged {len(df_purged)} corrupt/unverified Gemini 3.7 rows -> saved to {purged_path}")
    print(f"Total blacklisted Gemini 3.7 texts across all sources: {len(all_bad_gemini_texts)}")
    print(f"Overlap with benchmark holdout in pool_120k: {leak_b1800_mask.sum()} rows")

    # Cleaned pool_120k
    df_p120_clean = df_p120[~corrupt_gem_mask & ~leak_b1800_mask].copy()
    # Deduplicate within pool_120k
    df_p120_clean = df_p120_clean.drop_duplicates(subset=["norm_text"])
    clean_p120_path = Path("local_data/pool_120k_cleaned.parquet")
    df_p120_clean.to_parquet(clean_p120_path, index=False)
    print(f"Cleaned pool_120k saved: {len(df_p120_clean)} rows -> {clean_p120_path}")

    # Standardize columns for merging
    # Target columns: ['text', 'norm_text', 'model', 'dataset_source', 'priority', 'match_weight']
    records = []

    # Helper to add records
    def add_records(df_in, source_name, priority, default_weight=1.0, text_col="text", model_col="model", weight_col=None):
        count_added = 0
        count_skipped_b1800 = 0
        count_skipped_invalid = 0

        for _, row in df_in.iterrows():
            raw_text = row[text_col]
            norm = normalize_text(raw_text)
            model = str(row[model_col]).strip()

            if model not in LABEL_SET:
                count_skipped_invalid += 1
                continue

            words = len(norm.split())
            if words < 25 or words > 350:
                count_skipped_invalid += 1
                continue

            # Strict benchmark exclusion
            if norm in b_norm_texts:
                count_skipped_b1800 += 1
                continue

            # Don't add known bad or unverified proxy gemini
            if model == "gemini-3.7-flash" and norm in all_bad_gemini_texts:
                continue

            weight = default_weight
            if weight_col and weight_col in row and pd.notna(row[weight_col]):
                try:
                    weight = float(row[weight_col])
                except (ValueError, TypeError):
                    weight = default_weight

            records.append({
                "text": raw_text,
                "norm_text": norm,
                "model": model,
                "dataset_source": source_name,
                "priority": priority,
                "match_weight": weight,
                "word_count": words
            })
            count_added += 1

        print(f"  [{source_name}] Added: {count_added}, Skipped (B1800 leak): {count_skipped_b1800}, Skipped (invalid/len): {count_skipped_invalid}")

    # Priority 1: Direct Provider APIs (prvsiyan & seyitkaangunes) - Highest fidelity, 2.0x weight
    print("\n--- Ingesting Priority 1: Direct Provider APIs ---")
    
    # 1A. prvsiyan passages
    prv_path = Path("local_data/external_datasets/prvsiyan/passages.csv")
    if prv_path.exists():
        df_prv = pd.read_csv(prv_path)
        # Include passes_funnel == True OR non-thinking kimi-k3
        is_kimi_nothink = (df_prv["model"] == "kimi-k3") & (df_prv["reasoning_setting"].astype(str).str.contains("enabled\": false"))
        df_prv_valid = df_prv[df_prv["passes_funnel"] | is_kimi_nothink].copy()
        add_records(df_prv_valid, "prvsiyan_direct_api", priority=1, default_weight=2.0)

    # 1B. seyitkaangunes Claude API
    sey_path = Path("local_data/external_datasets/seyitkaangunes/ntl_extra_passages_claude.csv")
    if sey_path.exists():
        df_sey = pd.read_csv(sey_path)
        add_records(df_sey, "seyitkaangunes_claude_api", priority=1, default_weight=2.0)

    # Priority 2: In-Style Filtered Pool (prvsiyan_instyle)
    print("\n--- Ingesting Priority 2: In-Style Punctuation Pool ---")
    inst_path = Path("local_data/external_datasets/prvsiyan_instyle/pool_v1.parquet")
    if inst_path.exists():
        df_inst = pd.read_parquet(inst_path)
        add_records(df_inst, "prvsiyan_instyle_pool_v1", priority=2, default_weight=1.5)

    # Priority 3: crtproduction Open Pool (224k+)
    print("\n--- Ingesting Priority 3: crtproduction Open Pool ---")
    if op_path.exists():
        df_op = pd.read_parquet(op_path)
        # Filter flags
        clean_op = df_op[
            (~df_op["flag_gemini37_routing_risk"]) &
            (~df_op["flag_train_prompt_regen"]) &
            (~df_op["flag_near_train_prompt"])
        ].copy()
        add_records(clean_op, "crtproduction_open_pool", priority=3, default_weight=1.0, weight_col="match_weight")

    # Priority 4: ankiteshtiwari Curated Pool
    print("\n--- Ingesting Priority 4: ankiteshtiwari Curated Pool ---")
    ank_path = Path("local_data/external_datasets/ankiteshtiwari/curated_pool.parquet")
    if ank_path.exists():
        df_ank = pd.read_parquet(ank_path)
        add_records(df_ank, "ankiteshtiwari_curated_pool", priority=4, default_weight=1.0)

    # Priority 5: Cleaned pool_120k
    print("\n--- Ingesting Priority 5: Cleaned Legacy pool_120k ---")
    add_records(df_p120_clean, "pool_120k_cleaned", priority=5, default_weight=1.0)

    print(f"\nTotal raw collected records: {len(records)}")
    df_all = pd.DataFrame(records)

    # Deduplication across datasets
    print("\n--- Performing Global Deduplication ---")
    # Sort by priority (ascending: 1=highest priority) so drop_duplicates keeps the highest priority source
    df_all = df_all.sort_values(by=["priority", "match_weight"], ascending=[True, False])
    df_dedup = df_all.drop_duplicates(subset=["norm_text"], keep="first").copy()
    print(f"Records after global text deduplication: {len(df_dedup)}")

    # Verification: Check benchmark overlap
    final_leak = df_dedup["norm_text"].isin(b_norm_texts).sum()
    print(f"VERIFICATION: Benchmark holdout leakage count = {final_leak} (MUST BE ZERO)")
    assert final_leak == 0, f"FATAL: Benchmark leakage detected ({final_leak} rows)!"

    # Verification: Check bad Gemini overlap
    final_bad_gem = (df_dedup["model"] == "gemini-3.7-flash") & (df_dedup["norm_text"].isin(all_bad_gemini_texts))
    print(f"VERIFICATION: Corrupt / Unverified Gemini 3.7 leakage count = {final_bad_gem.sum()} (MUST BE ZERO)")
    assert final_bad_gem.sum() == 0, f"FATAL: Bad Gemini leakage detected ({final_bad_gem.sum()} rows)!"

    # Final cleanup of columns
    df_dedup["id"] = [f"clean_pool_{i:07d}" for i in range(len(df_dedup))]
    output_cols = ["id", "text", "model", "dataset_source", "priority", "match_weight", "word_count"]
    df_final = df_dedup[output_cols].copy()

    # Save to disk
    out_master_path = Path("local_data/clean_master_pool.parquet")
    df_final.to_parquet(out_master_path, index=False)
    print(f"\nSuccessfully wrote unified clean master pool: {len(df_final)} rows -> {out_master_path}")
    print(f"File size: {out_master_path.stat().st_size / 1e6:.2f} MB")

    # Summary statistics
    print("\n--- Final Master Pool Distribution by Model ---")
    print(df_final["model"].value_counts())

    print("\n--- Final Master Pool Distribution by Source ---")
    print(df_final["dataset_source"].value_counts())

    print("=" * 70)
    print("DECONTAMINATION AND UNIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
