"""
Modal Cloud Training Launcher: Two-Stage LoRA for Authorship Attribution
========================================================================
Thin cloud launcher running training on NVIDIA A100-80GB via Modal.
Mounts namethatllm and configs, delegating execution to run_training_pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure local package is discoverable when launching from any working directory
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("ntl-inhouse-trainer")

    trainer_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "torch>=2.4.0,<2.5.0",
            "transformers>=4.45.0,<5.0.0",
            "peft>=0.12.0",
            "accelerate>=0.34.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "pyarrow>=14.0.0",
            "safetensors>=0.4.0",
            "huggingface_hub>=0.24.0",
            "fastparquet>=2024.5.0",
            "pytest>=8.0.0",
        )
        .pip_install(
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
        )
        .add_local_python_source("namethatllm")
        .add_local_dir(str(repo_root / "configs"), remote_path="/root/configs")
        .add_local_dir(str(repo_root / "tests"), remote_path="/root/tests")
    )

    checkpoints_volume = modal.Volume.from_name("ntl-checkpoints", create_if_missing=True)
    datasets_volume = modal.Volume.from_name("ntl-datasets", create_if_missing=True)

    data_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("kaggle", "six", "pandas", "pyarrow", "fastparquet")
    )

    @app.function(
        image=data_image,
        volumes={"/datasets": datasets_volume},
        secrets=[
            modal.Secret.from_dict({
                "KAGGLE_USERNAME": "anilmittal",
                "KAGGLE_KEY": "KGAT_acf1831322ad0b003fcda339f7eb3f3e",
            })
        ],
        timeout=600,
    )
    def sync_cloud_datasets() -> int:
        """Downloads official Kaggle and external pool datasets directly into /datasets on Modal."""
        import subprocess
        import zipfile
        import pandas as pd
        from pathlib import Path

        kg_dir = Path.home() / ".kaggle"
        kg_dir.mkdir(parents=True, exist_ok=True)
        kg_file = kg_dir / "kaggle.json"
        kg_file.write_text('{"username": "anilmittal", "key": "KGAT_acf1831322ad0b003fcda339f7eb3f3e"}')
        kg_file.chmod(0o600)

        ds_root = Path("/datasets")
        ds_root.mkdir(parents=True, exist_ok=True)

        def _run_kg(cmd_list):
            res = subprocess.run(cmd_list, capture_output=True, text=True)
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
            if res.returncode != 0:
                print(f"[Data Sync Warning] Command failed: {' '.join(cmd_list)}")
            return res.returncode == 0

        print("[Data Sync] Downloading official competition data (train.csv, test.csv)...")
        _run_kg(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", "/datasets"])

        print("[Data Sync] Downloading nlm-matched-clean from Kaggle...")
        _run_kg(["kaggle", "datasets", "download", "-d", "dhruvtibarewal/nlm-matched-clean", "-p", "/datasets", "--unzip"])

        print("[Data Sync] Downloading pool_v4.parquet from Kaggle...")
        _run_kg(["kaggle", "datasets", "download", "-d", "nishantdahal/ntl-handwriting-blend", "-f", "pool_v4.parquet", "-p", "/datasets"])

        # Unpack any downloaded zip archives
        for z in list(ds_root.glob("*.zip")):
            print(f"[Data Sync] Extracting {z.name}...")
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall("/datasets")
            z.unlink()

        print("[Data Sync] Assembling canonical pool_120k.parquet from 4 pool files...")
        pool_files = ["nlm_matched_clean.parquet", "pool_v4.parquet", "nlm_pool_v5.parquet", "nlm_pool_v2.parquet"]
        dfs = []
        for fn in pool_files:
            fp = ds_root / fn
            if fp.exists():
                df = pd.read_parquet(fp)
                print(f"  [Pool Source] Ingested {fn}: {len(df):,} rows")
                dfs.append(df)
            else:
                print(f"  [Pool Source] Warning: {fn} not found in /datasets")

        if dfs:
            df_combined = pd.concat(dfs, ignore_index=True)
            raw_n = len(df_combined)
            df_dedup = df_combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
            out_p = ds_root / "pool_120k.parquet"
            df_dedup.to_parquet(out_p, index=False)
            print(f"[Data Sync] Canonical pool_120k.parquet created: {raw_n:,} raw -> {len(df_dedup):,} unique passages.")
            n_rows = len(df_dedup)
        else:
            n_rows = 0

        datasets_volume.commit()
        print("[Data Sync] Datasets volume successfully committed!")
        return n_rows

    @app.function(
        gpu="A100-80GB:1",
        timeout=600,  # 10 minutes max for test suite
        image=trainer_image,
    )
    def test_cloud_worker() -> str:
        """Runs the full pytest test suite inside the Modal container on an A100 GPU to catch any environment bugs."""
        import subprocess

        cmd = [sys.executable, "-m", "pytest", "-v", "/root/tests"]
        print(f"[Modal Test] Executing pytest inside Modal A100 container: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"Pytest failed on Modal cloud worker with exit code {res.returncode}")
        print("[Modal Test] All unit and integration tests passed cleanly on Modal A100!")
        return "ALL_TESTS_PASSED"

    @app.function(
        gpu="A100-80GB:1",
        timeout=1200,  # 20 minutes max for smoke run
        image=trainer_image,
        volumes={
            "/checkpoints": checkpoints_volume,
            "/datasets": datasets_volume,
        },
    )
    def smoke_cloud_worker(backbone: str = "Qwen/Qwen2.5-0.5B", keep_layers: int = 4) -> str:
        """
        Runs an ultra-fast 2-minute end-to-end smoke training pipeline on A100 to catch any
        bugs (CUDA, FlashAttention-2, BF16, SWA, Ridge blending, volumes) before full training.
        """
        import pandas as pd
        from namethatllm.config import LABELS
        from namethatllm.trainer import run_training_pipeline

        smoke_dir = Path("/tmp/ntl_smoke")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        train_p = smoke_dir / "train.csv"
        pool_p = smoke_dir / "pool_120k.parquet"

        dataset_root = Path("/datasets")
        if (dataset_root / "train.csv").exists():
            input_dir = str(dataset_root)
        else:
            print("[Modal Smoke] No train.csv in /datasets; synthesizing 72 smoke passages (18 classes x 4 prompts)...")
            smoke_rows = []
            for p_idx in range(4):
                p_id = f"smoke_prompt_{p_idx}"
                for c_idx in range(18):
                    model_name = LABELS[c_idx]
                    text = f"This is an automated smoke test passage for model {model_name} on prompt {p_id}. " * 5
                    smoke_rows.append({"prompt_id": p_id, "model": model_name, "text": text})
            df_smoke = pd.DataFrame(smoke_rows)
            df_smoke.to_csv(train_p, index=False)
            df_smoke.to_parquet(pool_p, index=False)
            input_dir = str(smoke_dir)

        smoke_cfg = {
            "backbone": backbone,
            "keep_layers": keep_layers,
            "layers": [2, keep_layers] if keep_layers >= 2 else [1],
            "max_len": 128,
            "input_dir": input_dir,
            "output_dir": "/checkpoints/export_smoke_test",
            "bf16": True,
            "family_margin": 0.20,
            "supcon_weight": 0.15,
            "stage1": {
                "enabled": True,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "epochs": 1,
                "max_samples": 72,
                "warmup_steps": 2,
            },
            "stage2": {
                "enabled": True,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "epochs": 1,
                "n_splits": 2,
                "check_overfitting": True,
                "warmup_steps": 2,
                "swa": True,
            },
        }

        print(f"[Modal Smoke] Launching 2-minute smoke pipeline on A100 with {backbone}...")
        out_path = run_training_pipeline(smoke_cfg)
        checkpoints_volume.commit()
        print(f"[Modal Smoke] SUCCESS! End-to-end smoke test passed cleanly. Output: {out_path}")
        return str(out_path)

    @app.function(
        gpu="A100-80GB:1",
        timeout=43200,  # 12 hours maximum execution budget
        image=trainer_image,
        volumes={
            "/checkpoints": checkpoints_volume,
            "/datasets": datasets_volume,
        },
    )
    def train_cloud_worker(
        config_path: str = "configs/training_lora.json",
        config_overrides: dict[str, Any] | None = None,
    ) -> str:
        """Executes full Two-Stage training pipeline on an NVIDIA A100-80GB in Modal."""
        from namethatllm.trainer import run_training_pipeline

        cfg_file = Path(config_path)
        if not cfg_file.is_absolute() and not cfg_file.exists():
            remote_cand = Path("/root") / config_path
            if remote_cand.exists():
                cfg_file = remote_cand

        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}

        if config_overrides:
            cfg.update(config_overrides)

        cfg["input_dir"] = "/datasets"
        out_sub = cfg.get("output_dir", "export_champion_qwen3_14b")
        cfg["output_dir"] = str(Path("/checkpoints") / Path(out_sub).name)
        cfg["bf16"] = True

        # Modal A100-80GB high-throughput defaults
        if "stage1" not in cfg:
            cfg["stage1"] = {}
        cfg["stage1"].setdefault("batch_size", 16)
        cfg["stage1"].setdefault("gradient_accumulation_steps", 2)
        cfg["stage1"].setdefault("epochs", 2)
        cfg["stage1"].setdefault("max_samples", 150000)

        out_path = run_training_pipeline(cfg)
        checkpoints_volume.commit()
        print(f"[Modal Worker] Training successfully completed. Checkpoints committed to: {out_path}")
        return str(out_path)

    @app.local_entrypoint()
    def main(
        mode: str = "run",
        config: str = "configs/training_lora.json",
        backbone: str = "Qwen/Qwen2.5-0.5B",
    ):
        """Entrypoint for: modal run scripts/modal_trainer.py [--mode test|smoke|sync|run] [--config ...] [--backbone ...]"""
        if mode == "test":
            print("[Modal Launcher] Dispatching cloud test suite to Modal A100...")
            res = test_cloud_worker.remote()
            print(f"[Modal Launcher] Test result: {res}")
        elif mode == "smoke":
            print(f"[Modal Launcher] Dispatching fast 2-minute smoke test to Modal A100 (backbone: {backbone})...")
            res = smoke_cloud_worker.remote(backbone=backbone)
            print(f"[Modal Launcher] Smoke test succeeded! Checkpoint: {res}")
        elif mode == "sync":
            print("[Modal Launcher] Syncing and downloading all datasets (train.csv, pool files, pool_120k)...")
            res = sync_cloud_datasets.remote()
            print(f"[Modal Launcher] Datasets ready in /datasets volume! Deduplicated pool rows: {res:,}")
        else:
            print(f"[Modal Launcher] Launching A100-80GB cloud training with config: {config}")
            res = train_cloud_worker.remote(config_path=config)
            print(f"[Modal Launcher] Cloud job finished successfully. Result: {res}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modal Cloud Launcher for NameThatLLM Training")
    parser.add_argument("cmd", nargs="?", default="run", choices=["run", "smoke", "test", "sync"], help="Command (run, smoke, test, sync)")
    parser.add_argument("--config", default="configs/training_lora.json", help="Path to training config json")
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B", help="Backbone model for smoke test")
    args = parser.parse_args()

    if modal is None:
        print("Error: modal package not installed. Run 'pip install modal' and 'modal setup'.")
        sys.exit(1)

    with modal.enable_output(), app.run():
        if args.cmd == "test":
            test_cloud_worker.remote()
        elif args.cmd == "smoke":
            smoke_cloud_worker.remote(backbone=args.backbone)
        elif args.cmd == "sync":
            sync_cloud_datasets.remote()
        else:
            train_cloud_worker.remote(config_path=args.config)
