#!/usr/bin/env python3
"""
scripts/benchmark_feature_manager.py
====================================
Standardized Management, Packaging & Indexing for 1,800-Row Benchmark Z Matrices.

Provides:
1. Strict naming convention:
   local_data/benchmark_features/v{VERSION}_{YYYYMMDD}/
     ├── metadata.json
     ├── benchmark_1800_features_all_models_v{VERSION}_{YYYYMMDD}.npz
     └── Z_unseen_{MODEL}_v{VERSION}_{YYYYMMDD}.npz
2. Safe FP16 compression and validation.
3. High-level loader API for local zero-GPU modeling head experiments:
   Z_dict, y, groups = load_benchmark_features(version="v84")
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_DIR = WORKSPACE_DIR / "local_data" / "benchmark_features"


def format_bundle_dirname(version: str, date_str: Optional[str] = None) -> str:
    """Returns standardized directory name: v{VERSION}_{YYYYMMDD}."""
    clean_v = version.lower() if version.lower().startswith("v") else f"v{version}"
    if date_str is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
    return f"{clean_v}_{date_str}"


def save_benchmark_feature_bundle(
    features_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    version: str,
    date_str: Optional[str] = None,
    output_root: Optional[Path] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves all Z_unseen matrices into a standardized, versioned folder with per-model
    files, master consolidated file, and metadata.json.
    """
    if output_root is None:
        output_root = DEFAULT_FEATURE_DIR

    dirname = format_bundle_dirname(version, date_str)
    bundle_dir = output_root / dirname
    bundle_dir.mkdir(parents=True, exist_ok=True)

    date_tag = dirname.split("_", 1)[1] if "_" in dirname else datetime.datetime.now().strftime("%Y%m%d")
    clean_v = dirname.split("_")[0]

    models_meta: Dict[str, Any] = {}
    master_dict: Dict[str, Any] = {
        "labels": np.asarray(labels, dtype=np.int32),
        "groups": np.asarray(groups, dtype=object),
    }

    print(f"\n[Feature Manager] Saving benchmark feature bundle to {bundle_dir}...")

    for raw_name, Z_raw in features_dict.items():
        # Canonicalize model name: strip leading 'Z_unseen_' if already present
        model_name = raw_name[9:] if raw_name.startswith("Z_unseen_") else raw_name
        model_key = f"Z_unseen_{model_name}"

        # Ensure FP16 for optimal disk and RAM efficiency
        Z_fp16 = np.asarray(Z_raw, dtype=np.float16)

        # 1. Save individual model file
        model_filename = f"{model_key}_{clean_v}_{date_tag}.npz"
        model_path = bundle_dir / model_filename
        np.savez_compressed(
            model_path,
            Z_unseen=Z_fp16,
            labels=np.asarray(labels, dtype=np.int32),
            groups=np.asarray(groups, dtype=object),
            model_name=model_name,
            version=clean_v,
            date=date_tag,
        )

        master_dict[model_key] = Z_fp16
        models_meta[model_name] = {
            "key": model_key,
            "filename": model_filename,
            "shape": list(Z_fp16.shape),
            "dtype": str(Z_fp16.dtype),
            "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        }
        print(f"  ✓ {model_name:<25}: shape {Z_fp16.shape} -> {model_filename} ({models_meta[model_name]['size_mb']} MB)")

    # 2. Save master consolidated archive
    master_filename = f"benchmark_1800_features_all_models_{clean_v}_{date_tag}.npz"
    master_path = bundle_dir / master_filename
    np.savez_compressed(master_path, **master_dict)
    master_size_mb = round(master_path.stat().st_size / (1024 * 1024), 2)
    print(f"  ✓ Master Archive         : {master_filename} ({master_size_mb} MB)")

    # 3. Save metadata.json
    meta_payload = {
        "version": clean_v,
        "date": date_tag,
        "created_at": datetime.datetime.now().isoformat(),
        "num_samples": len(labels),
        "num_models": len(features_dict),
        "models": models_meta,
        "master_archive": master_filename,
        "master_size_mb": master_size_mb,
    }
    if extra_metadata:
        meta_payload["extra"] = extra_metadata

    meta_path = bundle_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(f"  ✓ Metadata Index         : metadata.json")
    print(f"[Feature Manager] Successfully created bundle at: {bundle_dir}\n")

    return bundle_dir


def load_benchmark_features(
    version: Optional[str] = None,
    date_str: Optional[str] = None,
    model_name: Optional[str] = None,
    features_dir: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads benchmark features for local head prototyping.
    If version is None, loads the most recently created bundle.
    Returns: (features_dict, labels, groups, metadata)
    """
    if features_dir is None:
        features_dir = DEFAULT_FEATURE_DIR

    if not features_dir.exists():
        raise FileNotFoundError(f"Benchmark features directory does not exist: {features_dir}")

    # Discover candidate bundles
    bundles = sorted([d for d in features_dir.iterdir() if d.is_dir() and "_" in d.name], key=lambda d: d.stat().st_mtime)
    if not bundles:
        raise FileNotFoundError(f"No benchmark feature bundles found in {features_dir}")

    target_bundle = None
    if version is not None:
        clean_v = version.lower() if version.lower().startswith("v") else f"v{version}"
        for b in bundles:
            if b.name.startswith(clean_v):
                if date_str is None or date_str in b.name:
                    target_bundle = b
                    break
        if target_bundle is None:
            raise FileNotFoundError(f"No bundle found matching version={version}, date={date_str} in {features_dir}")
    else:
        target_bundle = bundles[-1]

    meta_file = target_bundle / "metadata.json"
    metadata = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    # Load specific model or all models
    if model_name is not None:
        clean_m = model_name[9:] if model_name.startswith("Z_unseen_") else model_name
        model_candidates = list(target_bundle.glob(f"Z_unseen_{clean_m}_*.npz"))
        if not model_candidates:
            raise FileNotFoundError(f"Model {model_name} not found in bundle {target_bundle}")
        data = np.load(str(model_candidates[0]), allow_pickle=True)
        return {clean_m: data["Z_unseen"]}, data["labels"], data["groups"], metadata

    # Load master archive
    master_file = target_bundle / metadata.get("master_archive", f"benchmark_1800_features_all_models_{target_bundle.name}.npz")
    if not master_file.exists():
        candidates = list(target_bundle.glob("benchmark_1800_features_all_models_*.npz"))
        if candidates:
            master_file = candidates[0]
        else:
            raise FileNotFoundError(f"Master archive not found in {target_bundle}")

    data = np.load(str(master_file), allow_pickle=True)
    labels = data["labels"]
    groups = data["groups"]
    features = {k: data[k] for k in data.files if k not in ("labels", "groups")}

    return features, labels, groups, metadata


def main():
    parser = argparse.ArgumentParser(description="Benchmark Feature Manager CLI")
    parser.add_argument("--list", action="store_true", help="List all available benchmark feature bundles")
    parser.add_argument("--bundle-dir", type=str, default=str(DEFAULT_FEATURE_DIR), help="Root directory for bundles")
    args = parser.parse_args()

    root = Path(args.bundle_dir)
    if not root.exists():
        print(f"Directory {root} does not exist yet.")
        return

    bundles = sorted([d for d in root.iterdir() if d.is_dir() and "_" in d.name])
    print(f"\n=== Available Benchmark Feature Bundles ({len(bundles)}) ===")
    for b in bundles:
        meta_p = b / "metadata.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            print(f"- {b.name:<20} | Models: {meta.get('num_models', 0)} | Samples: {meta.get('num_samples', 0)} | Total: {meta.get('master_size_mb', 0)} MB")
        else:
            print(f"- {b.name:<20} (No metadata.json)")
    print()


if __name__ == "__main__":
    main()
