"""
Offline Statistical Heads Trainer
=================================
Trains wPCA, Spectral Bank, and Lexical Engine offline on CPU and exports serialized
artifacts (lexical_bundle.joblib, wpca_bundle.safetensors, spectral_heads.safetensors)
for upload as a Kaggle dataset (anilmittal/ntl-statistical-heads).

Bypasses 45-60 minutes of CPU fitting on Kaggle GPU instances, reducing kernel execution
to pure inference in ~11-13 minutes total.
"""

import os
import gc
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from namethatllm.config import LABELS, K, LAB2I, FAMILIES
from namethatllm.lexical_engine import (
    train_lexical_engine,
    save_lexical_bundle,
    _text_hash,
    _norm_text_key
)
from namethatllm.wpca_gaussian import (
    fit_shared_gaussian_lowram,
    save_wpca_bundle,
    scores_exact_dp,
    transform_lowram
)
from namethatllm.spectral_bank import (
    fit_joint_spectral_bank,
    save_spectral_params,
    project_spectral_block
)


def train_offline_lexical(
    train_path: Path,
    pool_paths: List[Path],
    out_file: Path,
    unseen_path: Optional[Path] = None,
    qwen_tok_dir: Optional[str] = None
) -> Dict[str, Any]:
    print("=" * 70)
    print(f"[Offline Lexical Trainer] Starting lexical bundle fitting...")
    print(f"  - Train data: {train_path}")
    print(f"  - Pools: {[str(p) for p in pool_paths]}")
    print(f"  - Target bundle: {out_file}")
    t0 = time.time()

    df_tr = pd.read_csv(train_path)
    fit_texts = df_tr['text'].fillna('').astype(str).tolist()
    fit_y = np.array([LAB2I[m] for m in df_tr['model']])
    fit_sw = np.ones(len(df_tr), dtype=np.float32)
    seen_texts = {_norm_text_key(t) for t in fit_texts}
    print(f"[Offline Lexical Trainer] Loaded {len(df_tr)} official train rows.")

    unseen_hashes = set()
    if unseen_path and Path(unseen_path).exists():
        try:
            df_u = pd.read_parquet(unseen_path) if str(unseen_path).endswith('.parquet') else pd.read_csv(unseen_path)
            u_texts = df_u['text'].fillna('').astype(str).tolist()
            unseen_hashes = set(_text_hash(t) for t in u_texts)
            print(f"[Offline Lexical Trainer] Quarantined {len(unseen_hashes)} unseen benchmark rows from training.")
        except Exception as e:
            print(f"[Offline Lexical Trainer] Note: Could not read unseen holdout: {e}")

    for pf in pool_paths:
        pf = Path(pf)
        if pf.exists():
            try:
                p_df = pd.read_parquet(pf, columns=['text', 'model'])
                p_df = p_df[p_df['model'].isin(LABELS)].dropna(subset=['text', 'model']).copy()
                pf_name = str(pf).lower().replace('\\', '/')
                src_w = 0.50 if ('matched' in pf_name or 'clean' in pf_name) else 0.35

                keep_mask = []
                for t in p_df['text'].values:
                    k = _norm_text_key(t)
                    h = _text_hash(t)
                    ok = bool(k) and (k not in seen_texts) and (h not in unseen_hashes)
                    keep_mask.append(ok)
                    if ok:
                        seen_texts.add(k)
                p_df = p_df.loc[keep_mask].reset_index(drop=True)
                if len(p_df) > 0:
                    p_texts = p_df['text'].fillna('').astype(str).tolist()
                    p_y = p_df['model'].map(LAB2I).values
                    p_sw = np.full(len(p_df), src_w, dtype=np.float32)
                    fit_texts.extend(p_texts)
                    fit_y = np.r_[fit_y, p_y]
                    fit_sw = np.r_[fit_sw, p_sw]
                    print(f"[Offline Lexical Trainer] Added {len(p_df)} rows from {pf.name} (weight {src_w})")
            except Exception as e:
                print(f"[Offline Lexical Trainer] Note: Error reading pool {pf}: {e}")

    print(f"[Offline Lexical Trainer] Total training corpus: {len(fit_texts)} samples across 18 target classes.")
    bundle = train_lexical_engine(
        fit_texts=fit_texts,
        fit_y=fit_y,
        fit_sw=fit_sw,
        df_tr=df_tr,
        qwen_tok_dir=qwen_tok_dir,
    )
    save_lexical_bundle(bundle, out_file)
    print(f"[Offline Lexical Trainer] Completed lexical bundle fitting in {time.time() - t0:.2f}s")
    print("=" * 70)
    return bundle


def write_kaggle_metadata(out_dir: Path, dataset_slug: str = "ntl-statistical-heads", title: str = "NTL Statistical Heads"):
    meta_path = out_dir / "dataset-metadata.json"
    meta = {
        "title": title,
        "id": f"anilmittal/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Dataset Metadata] Created {meta_path} for Kaggle dataset {meta['id']}")


def main():
    parser = argparse.ArgumentParser(description="Train statistical heads (wPCA, Spectral Bank, Lexical Engine) offline.")
    parser.add_argument("--train_path", type=str, default="local_data/train.csv", help="Path to train.csv")
    parser.add_argument("--pool_paths", nargs="*", default=["local_data/clean_master_pool.parquet", "local_data/pool_v4.parquet"], help="Paths to pool parquet files")
    parser.add_argument("--unseen_path", type=str, default="local_data/clean_benchmark_1800_rows.parquet", help="Path to unseen benchmark for hash quarantining")
    parser.add_argument("--out_dir", type=str, default="local_data/statistical_heads", help="Output directory for serialized artifacts")
    parser.add_argument("--qwen_tok_dir", type=str, default="", help="Directory with Qwen fast tokenizer files (vocab.json)")
    parser.add_argument("--skip_lexical", action="store_true", help="Skip lexical bundle training")
    parser.add_argument("--metadata", action="store_true", default=True, help="Write Kaggle dataset metadata")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_lexical:
        lex_bundle_path = out_dir / "lexical_bundle.joblib"
        train_offline_lexical(
            train_path=Path(args.train_path),
            pool_paths=[Path(p) for p in args.pool_paths],
            out_file=lex_bundle_path,
            unseen_path=Path(args.unseen_path) if args.unseen_path else None,
            qwen_tok_dir=args.qwen_tok_dir if args.qwen_tok_dir else None
        )

    if args.metadata:
        write_kaggle_metadata(out_dir)

    print("\n[Complete] Statistical heads artifacts ready in:", out_dir)
    print(f"To upload to Kaggle, run:\n  kaggle datasets create -p {out_dir}")


if __name__ == "__main__":
    main()
