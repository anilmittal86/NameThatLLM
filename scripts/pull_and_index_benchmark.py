#!/usr/bin/env python3
"""
scripts/pull_and_index_benchmark.py
===================================
Automated Downloader & Indexer for Kaggle Submission Outputs and Z Feature Matrices.

Workflow:
1. Checks status of `anilmittal/name-that-llm-v2-submission`.
2. Downloads kernel outputs into `kaggle_logs/submission_v{VERSION}/`.
3. If `benchmark_1800_features.npz` (or `local_eval_cache.npz`) is present,
   extracts, indexes, and organizes into:
   local_data/benchmark_features/v{VERSION}_{YYYYMMDD}/
     ├── metadata.json
     ├── benchmark_1800_features_all_models_v{VERSION}_{YYYYMMDD}.npz
     └── Z_unseen_{MODEL}_v{VERSION}_{YYYYMMDD}.npz
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR / "scripts"))

from benchmark_feature_manager import save_benchmark_feature_bundle


def pull_and_index(version_tag: str, kernel_id: str = "anilmittal/name-that-llm-v2-submission"):
    clean_v = version_tag.lower() if version_tag.lower().startswith("v") else f"v{version_tag}"
    dest_dir = WORKSPACE_DIR / "kaggle_logs" / f"submission_{clean_v}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Step 1: Downloading outputs from Kaggle ({kernel_id}) to {dest_dir} ===")
    res = subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", str(dest_dir)], capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr)

    print(f"\n=== Step 2: Indexing Benchmark Features ===")
    feat_file = dest_dir / "benchmark_1800_features.npz"
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    if feat_file.exists():
        data = np.load(str(feat_file), allow_pickle=True)
        labels = data["labels"]
        groups = data["groups"]
        features = {k: data[k] for k in data.files if k not in ("labels", "groups")}

        bundle_path = save_benchmark_feature_bundle(
            features_dict=features,
            labels=labels,
            groups=groups,
            version=clean_v,
            date_str=date_str,
        )
        print(f"[SUCCESS] Features successfully indexed to: {bundle_path}")
    else:
        print(f"[Notice] {feat_file} not found in downloaded outputs.")
        # Check if local_eval_cache has Z matrices
        cache_file = dest_dir / "local_eval_cache.npz"
        if cache_file.exists():
            cdata = np.load(str(cache_file), allow_pickle=True)
            z_keys = [k for k in cdata.files if k.startswith("Z_") or "features" in k]
            if z_keys:
                labels = cdata.get("y_unseen", np.array([]))
                groups = cdata.get("groups_unseen", np.array([]))
                features = {k: cdata[k] for k in z_keys}
                save_benchmark_feature_bundle(features, labels, groups, clean_v, date_str)


def main():
    parser = argparse.ArgumentParser(description="Pull and Index Benchmark Features")
    parser.add_argument("--version", type=str, default="v83", help="Version tag (e.g. v83, v84)")
    parser.add_argument("--kernel", type=str, default="anilmittal/name-that-llm-v2-submission", help="Kaggle kernel slug")
    args = parser.parse_args()

    pull_and_index(args.version, args.kernel)


if __name__ == "__main__":
    main()
