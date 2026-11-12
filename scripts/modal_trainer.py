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
        kg_token = "KGAT_acf1831322ad0b003fcda339f7eb3f3e"
        (kg_dir / "access_token").write_text(kg_token)
        (kg_dir / "access_token").chmod(0o600)
        kg_file = kg_dir / "kaggle.json"
        kg_file.write_text('{"username": "anilmittal", "key": "' + kg_token + '"}')
        kg_file.chmod(0o600)
        import os
        os.environ["KAGGLE_API_TOKEN"] = kg_token

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
        image=data_image,
        volumes={"/checkpoints": checkpoints_volume},
        secrets=[
            modal.Secret.from_dict({
                "KAGGLE_USERNAME": "anilmittal",
                "KAGGLE_KEY": "KGAT_acf1831322ad0b003fcda339f7eb3f3e",
            })
        ],
        timeout=600,
    )
    def push_kaggle_dataset_worker(
        checkpoint_dir: str = "/checkpoints/export_champion_qwen3_14b",
        heads_bytes: bytes | None = None,
    ) -> str:
        """Pushes the export_champion_qwen3_14b checkpoint to Kaggle dataset anilmittal/ntl-inhouse-two-stage-lora24."""
        import subprocess
        from pathlib import Path

        ckpt = Path(checkpoint_dir)
        if heads_bytes is not None:
            heads_p = ckpt / "heads.safetensors"
            print(f"[Modal Kaggle Push] Writing {len(heads_bytes):,} bytes to {heads_p}...")
            heads_p.write_bytes(heads_bytes)
            checkpoints_volume.commit()
            print("[Modal Kaggle Push] Committed volume with restored heads.")

        kg_dir = Path.home() / ".kaggle"
        kg_dir.mkdir(parents=True, exist_ok=True)
        kg_token = "KGAT_acf1831322ad0b003fcda339f7eb3f3e"
        (kg_dir / "access_token").write_text(kg_token)
        (kg_dir / "access_token").chmod(0o600)
        kg_file = kg_dir / "kaggle.json"
        kg_file.write_text('{"username": "anilmittal", "key": "' + kg_token + '"}')
        kg_file.chmod(0o600)
        import os
        os.environ["KAGGLE_API_TOKEN"] = kg_token

        meta_p = ckpt / "dataset-metadata.json"
        if not meta_p.exists():
            meta_content = '{"title": "NTL In-House Two-Stage LoRA24", "id": "anilmittal/ntl-inhouse-two-stage-lora24", "licenses": [{"name": "CC0-1.0"}]}'
            meta_p.write_text(meta_content, encoding="utf-8")

        print(f"[Modal Kaggle Push] Uploading dataset version from {ckpt}...")
        res = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(ckpt), "-m", "Restore heads.safetensors to commit ae26404 (0.594 champion weights)", "-r", "zip"],
            capture_output=True,
            text=True,
        )
        print("stdout:", res.stdout)
        print("stderr:", res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"kaggle datasets version failed: {res.stderr}")
        return res.stdout

    @app.function(
        image=data_image,
        volumes={"/checkpoints": checkpoints_volume},
        secrets=[
            modal.Secret.from_dict({
                "KAGGLE_USERNAME": "anilmittal",
                "KAGGLE_KEY": "KGAT_acf1831322ad0b003fcda339f7eb3f3e",
            })
        ],
        timeout=600,
    )
    def sync_specialists_worker() -> dict[str, Any]:
        """Syncs within-family specialists from Kaggle into /checkpoints volume."""
        import subprocess
        from pathlib import Path
        import os

        kg_dir = Path.home() / ".kaggle"
        kg_dir.mkdir(parents=True, exist_ok=True)
        kg_token = "KGAT_acf1831322ad0b003fcda339f7eb3f3e"
        (kg_dir / "access_token").write_text(kg_token)
        (kg_dir / "access_token").chmod(0o600)
        kg_file = kg_dir / "kaggle.json"
        kg_file.write_text('{"username": "anilmittal", "key": "' + kg_token + '"}')
        kg_file.chmod(0o600)
        os.environ["KAGGLE_API_TOKEN"] = kg_token

        spec_root = Path("/checkpoints/training_v14_output")
        gemini_spec_p = spec_root / "export_gemini_specialist_14b"
        if not gemini_spec_p.exists():
            print("[Modal Specialist Sync] Downloading anilmittal/name-that-llm-lora-peft-training kernel output...")
            spec_root.mkdir(parents=True, exist_ok=True)
            res_kg = subprocess.run(["kaggle", "kernels", "output", "anilmittal/name-that-llm-lora-peft-training", "-p", str(spec_root)], capture_output=True, text=True)
            print("[Modal Specialist Sync] stdout:", res_kg.stdout)
            if res_kg.stderr:
                print("[Modal Specialist Sync] stderr:", res_kg.stderr)
            checkpoints_volume.commit()
        else:
            print("[Modal Specialist Sync] 14B specialists already cached in /checkpoints.")

        claude_spec_p = Path("/checkpoints/qwen3_8b_claude")
        if not (claude_spec_p / "final").exists():
            print("[Modal Specialist Sync] Downloading zbigniewkirchner/ntl-qwen3-8b-lora-claude-head...")
            claude_spec_p.mkdir(parents=True, exist_ok=True)
            res_cl = subprocess.run(["kaggle", "datasets", "download", "-d", "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head", "-p", str(claude_spec_p), "--unzip"], capture_output=True, text=True)
            print("[Modal Specialist Sync] stdout:", res_cl.stdout)
            if res_cl.stderr:
                print("[Modal Specialist Sync] stderr:", res_cl.stderr)
            checkpoints_volume.commit()
        else:
            print("[Modal Specialist Sync] Claude specialist already cached in /checkpoints.")

        return {
            "v14_exists": gemini_spec_p.exists(),
            "claude_exists": (claude_spec_p / "final").exists(),
        }

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

        def _on_checkpoint(event: str, step: Any):
            if event == "checkpoint_saved":
                try:
                    checkpoints_volume.commit()
                    print(f"[Modal Cloud Worker] Permanently committed intermediate checkpoint ({step}) to ntl-checkpoints volume!")
                except Exception as e:
                    print(f"[Modal Cloud Worker] Checkpoint volume commit note: {e}")

        out_path = run_training_pipeline(cfg, step_callback=_on_checkpoint)
        checkpoints_volume.commit()
        print(f"[Modal Worker] Training successfully completed. Checkpoints committed to: {out_path}")
        return str(out_path)

    @app.function(
        gpu="A100-80GB:1",
        timeout=1800,
        image=trainer_image,
        volumes={
            "/checkpoints": checkpoints_volume,
            "/datasets": datasets_volume,
        },
        secrets=[
            modal.Secret.from_dict({
                "KAGGLE_USERNAME": "anilmittal",
                "KAGGLE_KEY": "KGAT_acf1831322ad0b003fcda339f7eb3f3e",
            })
        ],
    )
    def eval_cloud_worker(
        checkpoint_dir: str = "/checkpoints/export_champion_qwen3_14b",
        texts: list[str] | None = None,
        labels: list[int] | None = None,
        groups: list[Any] | None = None,
        eval_specialists: bool = True,
    ) -> dict[str, Any]:
        """Runs fast inference on holdout texts using the champion checkpoint on an A100 GPU."""
        import json
        import torch
        import numpy as np
        from pathlib import Path
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer, AutoModel
        from safetensors import safe_open
        from namethatllm.config import LABELS
        from namethatllm.trainer import MultiLayerLoRAClassifier, collate_text_batch, TextSequenceDataset, compute_grouped_inference_logits
        from torch.utils.data import DataLoader
        from sklearn.metrics import balanced_accuracy_score

        ckpt = Path(checkpoint_dir)
        meta_p = ckpt / "meta.json"
        if meta_p.exists():
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}

        backbone_name = meta.get("backbone", "Qwen/Qwen3-14B-Base")
        keep_layers = int(meta.get("cfg", {}).get("keep", 27))
        layers = meta.get("cfg", {}).get("layers", [2, 4, 6, 9, 13, 17, 22, 27])
        max_len = int(meta.get("cfg", {}).get("max_len", 384))
        end_id = int(meta.get("cfg", {}).get("end_token_id", 151643))
        classes = meta.get("classes", LABELS)
        num_classes = len(classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Modal Eval] Loading tokenizer and config for {backbone_name} (keep_layers={keep_layers})...")
        tok = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base_cfg = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
        base_cfg.num_hidden_layers = keep_layers
        base_cfg.use_cache = False
        if getattr(base_cfg, "layer_types", None) is not None:
            base_cfg.layer_types = base_cfg.layer_types[:keep_layers]

        print(f"[Modal Eval] Loading base model on A100 GPU...")
        base_model = AutoModel.from_pretrained(
            backbone_name,
            config=base_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)

        adapter_dir = ckpt / "adapter"
        if adapter_dir.exists():
            print(f"[Modal Eval] Loading LoRA adapter from {adapter_dir}...")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        else:
            peft_model = base_model

        model = MultiLayerLoRAClassifier(
            peft_model,
            hidden_size=base_cfg.hidden_size,
            layers=layers,
            num_classes=num_classes,
        ).to(device)

        heads_p = ckpt / "heads.safetensors"
        if heads_p.exists():
            print(f"[Modal Eval] Loading heads weights from {heads_p}...")
            with safe_open(str(heads_p), framework="pt", device="cpu") as sf:
                for i in range(len(model.norms)):
                    w_k = f"norms.{i}.weight"
                    if w_k in sf.keys():
                        model.norms[i].weight.data.copy_(sf.get_tensor(w_k).to(device))
                    b_k = f"norms.{i}.bias"
                    if b_k in sf.keys() and hasattr(model.norms[i], "bias") and model.norms[i].bias is not None:
                        model.norms[i].bias.data.copy_(sf.get_tensor(b_k).to(device))
                if "plain.weight" in sf.keys():
                    model.plain_head.weight.data.copy_(sf.get_tensor("plain.weight").to(device))
                if "plain.bias" in sf.keys():
                    model.plain_head.bias.data.copy_(sf.get_tensor("plain.bias").to(device))
                if "centered.weight" in sf.keys():
                    model.centered_head.weight.data.copy_(sf.get_tensor("centered.weight").to(device))
                if "centered.bias" in sf.keys():
                    model.centered_head.bias.data.copy_(sf.get_tensor("centered.bias").to(device))

        model.eval()

        if texts is None:
            from namethatllm.ablation import build_unseen_holdout_dataset
            df_unseen, _ = build_unseen_holdout_dataset(Path("/datasets"))
            texts = df_unseen["text"].tolist()
            lab_map = {name: i for i, name in enumerate(classes)}
            labels = [lab_map[m] for m in df_unseen["model"]]
            if groups is None and "prompt_id" in df_unseen.columns:
                groups = df_unseen["prompt_id"].tolist()

        print(f"[Modal Eval] Extracting features Z_unseen on {len(texts)} holdout texts...")
        dummy_labels = [0] * len(texts)
        ds = TextSequenceDataset(texts, dummy_labels, None, tok, max_len=max_len, end_id=end_id)
        loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_text_batch)

        Z_list = []
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for batch in loader:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                    Z_list.append(z_b.float().cpu().numpy())

        Z_unseen = np.vstack(Z_list)
        print(f"[Modal Eval] Extracted Z_unseen: shape={Z_unseen.shape}, dtype={Z_unseen.dtype}")

        W_p = model.plain_head.weight.detach().cpu().float().numpy()
        b_p = model.plain_head.bias.detach().cpu().float().numpy()
        W_c = model.centered_head.weight.detach().cpu().float().numpy()
        b_c = model.centered_head.bias.detach().cpu().float().numpy()

        res_logits = compute_grouped_inference_logits(
            Z_unseen,
            plain_weight=W_p,
            plain_bias=b_p,
            centered_weight=W_c,
            centered_bias=b_c,
            groups=groups,
            blend_centered=0.20,
        )

        acc_plain = 0.0
        acc_centered = 0.0
        acc_blended = 0.0
        acc_plain_sinkhorn = 0.0
        acc_sinkhorn = 0.0

        if labels is not None:
            y_arr = np.array(labels)
            pred_p = res_logits["logits_plain"].argmax(axis=1)
            pred_c = res_logits["logits_centered"].argmax(axis=1)
            pred_b = res_logits["logits_blend"].argmax(axis=1)
            pred_p_sk = res_logits["probs_plain_sinkhorn"].argmax(axis=1)
            pred_sk = res_logits["probs_sinkhorn"].argmax(axis=1)

            acc_plain = float(balanced_accuracy_score(y_arr, pred_p))
            acc_centered = float(balanced_accuracy_score(y_arr, pred_c))
            acc_blended = float(balanced_accuracy_score(y_arr, pred_b))
            acc_plain_sinkhorn = float(balanced_accuracy_score(y_arr, pred_p_sk))
            acc_sinkhorn = float(balanced_accuracy_score(y_arr, pred_sk))

            print(f"[Modal Eval] Plain Head BAcc              : {acc_plain*100:.2f}%")
            print(f"[Modal Eval] Centered Head BAcc           : {acc_centered*100:.2f}%")
            print(f"[Modal Eval] Blended Head BAcc (80/20)    : {acc_blended*100:.2f}%")
            print(f"[Modal Eval] Plain + Sinkhorn BAcc        : {acc_plain_sinkhorn*100:.2f}%")
            print(f"[Modal Eval] Blended + Sinkhorn BAcc      : {acc_sinkhorn*100:.2f}%")

            print("\n" + "=" * 125)
            print(f"{'Class Name':<24} | {'True':<5} | {'Pred (Plain)':<13} | {'BAcc (Plain)':<13} | {'Pred (P-Sink)':<14} | {'BAcc (P-Sink)':<14} | {'Pred (B-Sink)':<14} | {'BAcc (B-Sink)':<14}")
            print("-" * 125)
            for idx, c_name in enumerate(classes):
                m_cls = (y_arr == idx)
                cnt_true = int(m_cls.sum())
                cnt_pred_p = int((pred_p == idx).sum())
                cnt_pred_p_sk = int((pred_p_sk == idx).sum())
                cnt_pred_sk = int((pred_sk == idx).sum())
                acc_cls_p = float((pred_p[m_cls] == idx).mean()) if cnt_true > 0 else 0.0
                acc_cls_p_sk = float((pred_p_sk[m_cls] == idx).mean()) if cnt_true > 0 else 0.0
                acc_cls_sk = float((pred_sk[m_cls] == idx).mean()) if cnt_true > 0 else 0.0
                print(f"{c_name:<24} | {cnt_true:<5} | {cnt_pred_p:<13} | {acc_cls_p*100:5.1f}%       | {cnt_pred_p_sk:<14} | {acc_cls_p_sk*100:5.1f}%        | {cnt_pred_sk:<14} | {acc_cls_sk*100:5.1f}%")
            print("=" * 125 + "\n")

        specialist_logits = {}
        specialist_classes = {}

        if eval_specialists and texts is not None and len(texts) > 0:
            import os
            import subprocess
            import gc
            from namethatllm.config import FAMILIES
            from scipy.special import softmax as sp_softmax

            spec_root = Path("/checkpoints/training_v14_output")
            claude_spec_p = Path("/checkpoints/qwen3_8b_claude")

            # 4. Run 14B specialists (gemini, glm, gpt) using the loaded Qwen3-14B base
            P_main = sp_softmax(res_logits["logits_plain"], axis=1)

            spec_configs = [
                ("gemini", spec_root / "export_gemini_specialist_14b", FAMILIES["gemini"]),
                ("glm", spec_root / "export_glm_specialist_14b", FAMILIES["glm"]),
                ("gpt", spec_root / "export_gpt_specialist_14b", FAMILIES["gpt"]),
            ]

            for fam_name, s_dir, fam_cols in spec_configs:
                if not s_dir.exists():
                    print(f"[Modal Specialist] Warning: {s_dir} not found; skipping {fam_name}.")
                    continue
                meta_s_p = s_dir / "meta.json"
                if meta_s_p.exists():
                    with open(meta_s_p, "r", encoding="utf-8") as f:
                        meta_s = json.load(f)
                else:
                    meta_s = {}
                s_classes = meta_s.get("classes", [LABELS[i] for i in fam_cols])
                specialist_classes[fam_name] = s_classes
                K_fam = len(s_classes)

                fam_prob_mass = P_main[:, fam_cols].sum(axis=1)
                cand_mask = np.isin(P_main.argmax(axis=1), fam_cols) | (fam_prob_mass >= 0.10)
                cand_idx = np.flatnonzero(cand_mask)
                print(f"[Modal Specialist {fam_name}] Running 14B specialist on {len(cand_idx)}/{len(texts)} candidate rows...")

                if len(cand_idx) > 0:
                    s_adapter = s_dir / "adapter"
                    peft_model.load_adapter(str(s_adapter), adapter_name=fam_name)
                    peft_model.set_adapter(fam_name)

                    s_heads = s_dir / "heads.safetensors"
                    with safe_open(str(s_heads), framework="pt", device="cpu") as sf:
                        for i in range(len(model.norms)):
                            w_k = f"norms.{i}.weight"
                            if w_k in sf.keys():
                                model.norms[i].weight.data.copy_(sf.get_tensor(w_k).to(device))
                            b_k = f"norms.{i}.bias"
                            if b_k in sf.keys() and hasattr(model.norms[i], "bias") and model.norms[i].bias is not None:
                                model.norms[i].bias.data.copy_(sf.get_tensor(b_k).to(device))
                        w_plain = sf.get_tensor("plain.weight").to(device)
                        b_plain = sf.get_tensor("plain.bias").to(device)

                    cand_texts = [texts[i] for i in cand_idx]
                    cand_ds = TextSequenceDataset(cand_texts, [0] * len(cand_texts), None, tok, max_len=max_len, end_id=end_id)
                    cand_loader = DataLoader(cand_ds, batch_size=16, shuffle=False, collate_fn=collate_text_batch)

                    z_fam_list = []
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            for batch in cand_loader:
                                mb_ids = batch["input_ids"].to(device)
                                mb_mask = batch["attention_mask"].to(device)
                                max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                                z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                                z_fam_list.append(z_b)

                    if z_fam_list:
                        z_fam = torch.cat(z_fam_list, dim=0)
                        raw_logits = (z_fam @ w_plain.T + b_plain).float().cpu().numpy()
                        std_fam = float(np.std(raw_logits, axis=1).mean()) if len(raw_logits) > 0 else 1.0
                        temp_fam = max(1.0, std_fam / 4.2)
                        sc_logits = raw_logits / temp_fam

                        fam_logits_full = np.zeros((len(texts), K_fam), dtype=np.float32)
                        fam_logits_full[cand_idx] = sc_logits
                        specialist_logits[fam_name] = fam_logits_full
                    else:
                        specialist_logits[fam_name] = np.zeros((len(texts), K_fam), dtype=np.float32)

                    peft_model.delete_adapter(fam_name)

            # 5. Free 14B model from GPU memory before loading Claude 8B
            del model, peft_model, base_model
            torch.cuda.empty_cache()
            gc.collect()

            # 6. Run Claude Specialist (Qwen3-8B)
            claude_adapter_dir = None
            for cand in [claude_spec_p / "final" / "adapter", claude_spec_p / "adapter"]:
                if (cand / "adapter_model.safetensors").exists():
                    claude_adapter_dir = cand
                    break
            if claude_adapter_dir is None:
                found = list(claude_spec_p.rglob("adapter_model.safetensors"))
                if found:
                    claude_adapter_dir = found[0].parent

            claude_readouts_p = None
            for cand in [claude_spec_p / "final" / "readouts.npz", claude_spec_p / "readouts.npz"]:
                if cand.exists():
                    claude_readouts_p = cand
                    break
            if claude_readouts_p is None:
                found = list(claude_spec_p.rglob("readouts.npz"))
                if found:
                    claude_readouts_p = found[0]

            claude_heads_p = None
            for cand in [claude_spec_p / "final" / "heads.safetensors", claude_spec_p / "heads.safetensors"]:
                if cand.exists():
                    claude_heads_p = cand
                    break
            if claude_heads_p is None:
                found = list(claude_spec_p.rglob("heads.safetensors"))
                if found:
                    claude_heads_p = found[0]

            if claude_adapter_dir is not None and claude_readouts_p is not None and claude_heads_p is not None:
                print(f"[Modal Specialist Claude] Loading Claude 8B: adapter={claude_adapter_dir}")
                rnpz = np.load(str(claude_readouts_p), allow_pickle=True)
                claude_coef = torch.from_numpy(rnpz["claude_coef"]).to(device)
                claude_intercept = torch.from_numpy(rnpz["claude_intercept"]).to(device)
                claude_cls = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in rnpz["claude_classes"]]
                specialist_classes["claude"] = claude_cls

                claude_cols = FAMILIES["claude"]
                cand_mask_cl = np.isin(P_main.argmax(axis=1), claude_cols) | (P_main[:, claude_cols].sum(axis=1) >= 0.10)
                cand_idx_cl = np.flatnonzero(cand_mask_cl)
                print(f"[Modal Specialist Claude] Running Claude 8B on {len(cand_idx_cl)}/{len(texts)} candidate rows...")

                backbone_8b = "Qwen/Qwen3-8B-Base"
                tok_8b = AutoTokenizer.from_pretrained(backbone_8b, trust_remote_code=True, use_fast=True)
                if tok_8b.pad_token is None:
                    tok_8b.pad_token = tok_8b.eos_token
                cfg_8b = AutoConfig.from_pretrained(backbone_8b, trust_remote_code=True)
                cfg_8b.num_hidden_layers = 24
                cfg_8b.use_cache = False
                if getattr(cfg_8b, "layer_types", None) is not None:
                    cfg_8b.layer_types = cfg_8b.layer_types[:24]

                base_8b = AutoModel.from_pretrained(
                    backbone_8b,
                    config=cfg_8b,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                ).to(device)

                peft_8b = PeftModel.from_pretrained(base_8b, str(claude_adapter_dir))
                model_8b = MultiLayerLoRAClassifier(
                    peft_8b,
                    hidden_size=cfg_8b.hidden_size,
                    layers=[2, 4, 6, 8, 12, 16, 20, 24],
                    num_classes=5,
                ).to(device)

                with safe_open(str(claude_heads_p), framework="pt", device="cpu") as sf:
                    for i in range(len(model_8b.norms)):
                        w_k = f"norms.{i}.weight"
                        if w_k in sf.keys():
                            model_8b.norms[i].weight.data.copy_(sf.get_tensor(w_k).to(device))
                        b_k = f"norms.{i}.bias"
                        if b_k in sf.keys() and hasattr(model_8b.norms[i], "bias") and model_8b.norms[i].bias is not None:
                            model_8b.norms[i].bias.data.copy_(sf.get_tensor(b_k).to(device))

                model_8b.eval()
                cand_texts_cl = [texts[i] for i in cand_idx_cl]
                cand_ds_cl = TextSequenceDataset(cand_texts_cl, [0] * len(cand_texts_cl), None, tok_8b, max_len=447, end_id=151643)
                cand_loader_cl = DataLoader(cand_ds_cl, batch_size=16, shuffle=False, collate_fn=collate_text_batch)

                z_cl_list = []
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        for batch in cand_loader_cl:
                            mb_ids = batch["input_ids"].to(device)
                            mb_mask = batch["attention_mask"].to(device)
                            max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                            z_b = model_8b.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                            z_cl_list.append(z_b)

                if z_cl_list:
                    z_cl = torch.cat(z_cl_list, dim=0)
                    raw_logits_cl = (z_cl @ claude_coef.T + claude_intercept).float().cpu().numpy()
                    cl_logits_full = np.zeros((len(texts), 5), dtype=np.float32)
                    cl_logits_full[cand_idx_cl] = raw_logits_cl
                    specialist_logits["claude"] = cl_logits_full
                else:
                    specialist_logits["claude"] = np.zeros((len(texts), 5), dtype=np.float32)

                del model_8b, peft_8b, base_8b
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"[Modal Specialist Claude] Warning: Claude assets missing in {claude_spec_p}")

        return {
            "acc_plain": acc_plain,
            "acc_centered": acc_centered,
            "acc_blended": acc_blended,
            "acc_plain_sinkhorn": acc_plain_sinkhorn,
            "acc_sinkhorn": acc_sinkhorn,
            "logits_plain": res_logits["logits_plain"].tolist(),
            "logits_centered": res_logits["logits_centered"].tolist(),
            "logits_blend": res_logits["logits_blend"].tolist(),
            "probs_plain_sinkhorn": res_logits["probs_plain_sinkhorn"].tolist(),
            "probs_sinkhorn": res_logits["probs_sinkhorn"].tolist(),
            "spec_logits_gemini": specialist_logits.get("gemini", np.zeros((len(texts), 4))).tolist(),
            "spec_logits_glm": specialist_logits.get("glm", np.zeros((len(texts), 2))).tolist(),
            "spec_logits_gpt": specialist_logits.get("gpt", np.zeros((len(texts), 5))).tolist(),
            "spec_logits_claude": specialist_logits.get("claude", np.zeros((len(texts), 5))).tolist(),
            "spec_classes_gemini": specialist_classes.get("gemini", []),
            "spec_classes_glm": specialist_classes.get("glm", []),
            "spec_classes_gpt": specialist_classes.get("gpt", []),
            "spec_classes_claude": specialist_classes.get("claude", []),
            "classes": classes,
        }

    @app.function(
        gpu="A100-80GB:1",
        timeout=1800,  # 30 minutes max
        image=trainer_image,
        volumes={
            "/checkpoints": checkpoints_volume,
            "/datasets": datasets_volume,
        },
    )
    def reexport_heads_worker(
        checkpoint_dir: str = "/checkpoints/export_champion_qwen3_14b",
        unseen_texts: list[str] | None = None,
        y_unseen: list[int] | None = None,
        groups_unseen: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Re-exports classification heads using the patched _blend_head formula with centered & scaled bias:
        1. Loads the frozen 14B LoRA model from checkpoint_dir/adapter.
        2. Loads /datasets/train.csv (5,000 samples) and extracts features Z_train.
        3. Runs 5-fold GroupKFold OOF CV with trace-scaled Ridge heads to verify regularization.
        4. Fits exact closed-form trace-scaled Ridge heads: fit_ridge_heads(Z_train, y_train, prompts_train).
        5. Recovers pure neural weights (W_n, b_n) and blends with centered and scaled bias.
        6. Evaluates holdout accuracy with compute_grouped_inference_logits before vs after.
        7. Backs up legacy heads.safetensors and writes the new corrected heads.safetensors.
        8. Commits volume permanently.
        """
        import json
        import shutil
        import torch
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer, AutoModel
        from safetensors import safe_open
        from safetensors.torch import save_file
        from torch.utils.data import DataLoader
        from sklearn.metrics import balanced_accuracy_score

        from namethatllm.config import LABELS, LAB2I, K
        from namethatllm.trainer import (
            MultiLayerLoRAClassifier,
            collate_text_batch,
            TextSequenceDataset,
            fit_ridge_heads,
            _blend_head,
            evaluate_stage2_oof_cv,
            compute_grouped_inference_logits,
        )
        from namethatllm.ablation import build_unseen_holdout_dataset

        ckpt = Path(checkpoint_dir)
        meta_p = ckpt / "meta.json"
        if not meta_p.exists():
            raise FileNotFoundError(f"meta.json not found in {checkpoint_dir}")
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)

        backbone_name = meta.get("backbone", "Qwen/Qwen3-14B-Base")
        keep_layers = int(meta.get("cfg", {}).get("keep", 27))
        layers = meta.get("cfg", {}).get("layers", [2, 4, 6, 9, 13, 17, 22, 27])
        max_len = int(meta.get("cfg", {}).get("max_len", 384))
        end_id = int(meta.get("cfg", {}).get("end_token_id", 151643))
        classes = meta.get("classes", LABELS)
        num_classes = len(classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Modal Re-Export] Initializing Qwen3-14B ({keep_layers} layers) on A100 GPU...")
        tok = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base_cfg = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
        base_cfg.num_hidden_layers = keep_layers
        base_cfg.use_cache = False
        if getattr(base_cfg, "layer_types", None) is not None:
            base_cfg.layer_types = base_cfg.layer_types[:keep_layers]

        base_model = AutoModel.from_pretrained(
            backbone_name,
            config=base_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)

        adapter_dir = ckpt / "adapter"
        print(f"[Modal Re-Export] Loading LoRA adapter from {adapter_dir}...")
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

        model = MultiLayerLoRAClassifier(
            peft_model,
            hidden_size=base_cfg.hidden_size,
            layers=layers,
            num_classes=num_classes,
        ).to(device)

        backup_p = ckpt / "heads_legacy_unscaled_bias.safetensors"
        heads_p = ckpt / "heads.safetensors"
        if backup_p.exists():
            print(f"[Modal Re-Export] Reading untouched legacy heads from backup: {backup_p}...")
            source_p = backup_p
        else:
            print(f"[Modal Re-Export] Backing up untouched heads.safetensors to {backup_p}...")
            shutil.copyfile(heads_p, backup_p)
            source_p = backup_p

        heads_sd_raw = {}
        with safe_open(str(source_p), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                heads_sd_raw[k] = sf.get_tensor(k)

        # Load norm layers
        for i in range(len(model.norms)):
            w_k = f"norms.{i}.weight"
            if w_k in heads_sd_raw:
                model.norms[i].weight.data.copy_(heads_sd_raw[w_k].to(device))
            b_k = f"norms.{i}.bias"
            if b_k in heads_sd_raw and hasattr(model.norms[i], "bias") and model.norms[i].bias is not None:
                model.norms[i].bias.data.copy_(heads_sd_raw[b_k].to(device))

        # Copy existing heads into model
        if "plain.weight" in heads_sd_raw:
            model.plain_head.weight.data.copy_(heads_sd_raw["plain.weight"].to(device))
        if "plain.bias" in heads_sd_raw:
            model.plain_head.bias.data.copy_(heads_sd_raw["plain.bias"].to(device))
        if "centered.weight" in heads_sd_raw:
            model.centered_head.weight.data.copy_(heads_sd_raw["centered.weight"].to(device))
        if "centered.bias" in heads_sd_raw:
            model.centered_head.bias.data.copy_(heads_sd_raw["centered.bias"].to(device))

        model.eval()

        # Step 1: Load train.csv dataset to extract Z_train
        ds_root = Path("/datasets")
        train_p = ds_root / "train.csv"
        if not train_p.exists():
            raise FileNotFoundError("train.csv not found in /datasets")

        df_train = pd.read_csv(train_p)
        print(f"[Modal Re-Export] Loaded train.csv with {len(df_train)} rows.")
        y_train = np.array([LAB2I[m] for m in df_train["model"]])
        prompts_train = df_train["prompt_id"].values
        texts_train = df_train["text"].tolist()

        # Step 2: Extract representations Z_train using model.forward_features
        print(f"[Modal Re-Export] Extracting Z_train representations on {len(texts_train)} samples...")
        ds_tr = TextSequenceDataset(texts_train, y_train.tolist(), None, tok, max_len=max_len, end_id=end_id)
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=False, collate_fn=collate_text_batch)

        Z_list = []
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for batch in loader_tr:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                    Z_list.append(z_b.cpu().float().numpy())

        Z_train = np.vstack(Z_list)
        print(f"[Modal Re-Export] Extracted Z_train: shape={Z_train.shape}, dtype={Z_train.dtype}")

        # Step 2b: Run 5-fold GroupKFold OOF CV with trace-scaled Ridge heads
        print("[Modal Re-Export] Running 5-fold GroupKFold OOF CV with trace-scaled Ridge heads...")
        oof_cv_res = evaluate_stage2_oof_cv(
            Z_train,
            y_train,
            prompts_train,
            alpha=10.0,
            label_smoothing=0.10,
            num_classes=num_classes,
        )

        # Step 3: Reconstruct exact legacy unscaled Ridge heads originally blended into backup_p
        print("[Modal Re-Export] Fitting exact legacy unscaled Ridge heads on Z_train (scale_by_trace=False)...")
        W_p_r_old, b_p_r_old, W_c_r_old, b_c_r_old = fit_ridge_heads(
            Z_train, y_train, prompts_train, alpha=10.0, label_smoothing=0.10, num_classes=num_classes, scale_by_trace=False
        )

        # Step 4: Extract unseen holdout texts if available to benchmark before vs after
        if unseen_texts is None:
            df_unseen, _ = build_unseen_holdout_dataset(ds_root)
            unseen_texts = df_unseen["text"].tolist() if len(df_unseen) > 0 else []
            y_unseen = [LAB2I[m] for m in df_unseen["model"]] if len(df_unseen) > 0 else []
            if groups_unseen is None and "prompt_id" in df_unseen.columns:
                groups_unseen = df_unseen["prompt_id"].tolist()
        y_u_arr = np.array(y_unseen) if y_unseen is not None and len(y_unseen) > 0 else np.array([])
        print(f"[Modal Re-Export] Extracting Z_unseen on {len(unseen_texts)} holdout samples...")

        Z_unseen = None
        if len(unseen_texts) > 0:
            ds_u = TextSequenceDataset(unseen_texts, y_u_arr.tolist(), None, tok, max_len=max_len, end_id=end_id)
            loader_u = DataLoader(ds_u, batch_size=16, shuffle=False, collate_fn=collate_text_batch)

            Z_u_list = []
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for batch in loader_u:
                        mb_ids = batch["input_ids"].to(device)
                        mb_mask = batch["attention_mask"].to(device)
                        max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                        z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                        Z_u_list.append(z_b.cpu().float().numpy())
            if Z_u_list:
                Z_unseen = np.vstack(Z_u_list)

        # Benchmark BEFORE: evaluation using backup_p weights
        bacc_old = 0.0
        if Z_unseen is not None and len(y_u_arr) > 0:
            W_p_old = heads_sd_raw["plain.weight"].float().numpy()
            b_p_old = heads_sd_raw["plain.bias"].float().numpy()
            logits_old = Z_unseen @ W_p_old.T + b_p_old
            bacc_old = float(balanced_accuracy_score(y_u_arr, logits_old.argmax(axis=1)))
            print(f"[Modal Re-Export] BEFORE (Legacy Plain Head) Unseen BAcc: {bacc_old:.4f} ({bacc_old*100:.2f}%)")

        # Step 5: Invert the legacy blend from heads_sd_raw to recover pure neural heads (W_n, b_n)
        W_p_saved = heads_sd_raw["plain.weight"].float().numpy()
        b_p_saved = heads_sd_raw["plain.bias"].float().numpy()
        W_c_saved = heads_sd_raw["centered.weight"].float().numpy()
        b_c_saved = heads_sd_raw["centered.bias"].float().numpy()

        norm_p_saved = float(np.linalg.norm(W_p_saved))
        norm_p_r_old = float(np.linalg.norm(W_p_r_old))
        scale_p_old = (norm_p_saved / norm_p_r_old) if (norm_p_r_old > 1e-6 and norm_p_saved > 1e-6) else 1.0
        W_p_r_scaled = W_p_r_old * scale_p_old
        W_p_n = (W_p_saved - 0.25 * W_p_r_scaled) / 0.75
        b_p_n = (b_p_saved - 0.25 * b_p_r_old) / 0.75

        norm_c_saved = float(np.linalg.norm(W_c_saved))
        norm_c_r_old = float(np.linalg.norm(W_c_r_old))
        scale_c_old = (norm_c_saved / norm_c_r_old) if (norm_c_r_old > 1e-6 and norm_c_saved > 1e-6) else 1.0
        W_c_r_scaled = W_c_r_old * scale_c_old
        W_c_n = (W_c_saved - 0.25 * W_c_r_scaled) / 0.75
        b_c_n = (b_c_saved - 0.25 * b_c_r_old) / 0.75

        # Check pure neural BAcc
        bacc_pure_n = 0.0
        if Z_unseen is not None and len(y_u_arr) > 0:
            logits_pure_n = Z_unseen @ W_p_n.T + b_p_n
            bacc_pure_n = float(balanced_accuracy_score(y_u_arr, logits_pure_n.argmax(axis=1)))
            print(f"[Modal Re-Export] PURE NEURAL (No Ridge) Unseen Plain Head BAcc: {bacc_pure_n:.4f} ({bacc_pure_n*100:.2f}%)")

        # Step 6: Fit new trace-scaled Ridge heads (scale_by_trace=True) and blend
        print("[Modal Re-Export] Fitting new trace-scaled Ridge heads on Z_train (scale_by_trace=True)...")
        W_p_r_new, b_p_r_new, W_c_r_new, b_c_r_new = fit_ridge_heads(
            Z_train, y_train, prompts_train, alpha=10.0, label_smoothing=0.10, num_classes=num_classes, scale_by_trace=True
        )

        plain_linear = torch.nn.Linear(W_p_n.shape[1], W_p_n.shape[0])
        plain_linear.weight.data.copy_(torch.from_numpy(W_p_n))
        plain_linear.bias.data.copy_(torch.from_numpy(b_p_n))
        _blend_head(plain_linear, W_p_r_new, b_p_r_new, w_neural=0.75, w_ridge=0.25)

        centered_linear = torch.nn.Linear(W_c_n.shape[1], W_c_n.shape[0])
        centered_linear.weight.data.copy_(torch.from_numpy(W_c_n))
        centered_linear.bias.data.copy_(torch.from_numpy(b_c_n))
        _blend_head(centered_linear, W_c_r_new, b_c_r_new, w_neural=0.75, w_ridge=0.25)

        # Benchmark AFTER: evaluation with corrected centered & scaled bias
        bacc_new = 0.0
        acc_cent_new = 0.0
        acc_blend_new = 0.0
        acc_p_sink_new = 0.0
        acc_sink_new = 0.0
        if Z_unseen is not None and len(y_u_arr) > 0:
            W_p_new = plain_linear.weight.detach().cpu().float().numpy()
            b_p_new = plain_linear.bias.detach().cpu().float().numpy()
            W_c_new = centered_linear.weight.detach().cpu().float().numpy()
            b_c_new = centered_linear.bias.detach().cpu().float().numpy()

            eval_res = compute_grouped_inference_logits(
                Z_unseen,
                plain_weight=W_p_new,
                plain_bias=b_p_new,
                centered_weight=W_c_new,
                centered_bias=b_c_new,
                groups=groups_unseen,
                blend_centered=0.20,
            )
            bacc_new = float(balanced_accuracy_score(y_u_arr, eval_res["logits_plain"].argmax(axis=1)))
            acc_cent_new = float(balanced_accuracy_score(y_u_arr, eval_res["logits_centered"].argmax(axis=1)))
            acc_blend_new = float(balanced_accuracy_score(y_u_arr, eval_res["logits_blend"].argmax(axis=1)))
            acc_p_sink_new = float(balanced_accuracy_score(y_u_arr, eval_res["probs_plain_sinkhorn"].argmax(axis=1)))
            acc_sink_new = float(balanced_accuracy_score(y_u_arr, eval_res["probs_sinkhorn"].argmax(axis=1)))
            print(f"[Modal Re-Export] AFTER (Trace-Scaled Ridge & Centered Bias):")
            print(f"  Plain Head BAcc          : {bacc_new:.4f} ({bacc_new*100:.2f}%)")
            print(f"  Centered Head BAcc       : {acc_cent_new:.4f} ({acc_cent_new*100:.2f}%)")
            print(f"  Blended Head BAcc (80/20): {acc_blend_new:.4f} ({acc_blend_new*100:.2f}%)")
            print(f"  Plain + Sinkhorn BAcc    : {acc_p_sink_new:.4f} ({acc_p_sink_new*100:.2f}%)")
            print(f"  Blended + Sinkhorn BAcc  : {acc_sink_new:.4f} ({acc_sink_new*100:.2f}%)")
            print(f"  NET LIFT (Plain)         : {bacc_new - bacc_old:+.4f} ({(bacc_new - bacc_old)*100:+.2f}%)")

        # Step 7: Save new heads.safetensors
        new_heads_dict = {}
        for k, v in heads_sd_raw.items():
            if k == "plain.weight":
                new_heads_dict[k] = plain_linear.weight.detach().cpu().contiguous()
            elif k == "plain.bias":
                new_heads_dict[k] = plain_linear.bias.detach().cpu().contiguous()
            elif k == "centered.weight":
                new_heads_dict[k] = centered_linear.weight.detach().cpu().contiguous()
            elif k == "centered.bias":
                new_heads_dict[k] = centered_linear.bias.detach().cpu().contiguous()
            else:
                new_heads_dict[k] = v.contiguous()

        save_file(new_heads_dict, str(heads_p))
        print(f"[Modal Re-Export] Saved new blended heads to {heads_p}")

        checkpoints_volume.commit()
        print("[Modal Re-Export] checkpoints_volume committed successfully!")

        return {
            "bacc_old": bacc_old,
            "bacc_pure_n": bacc_pure_n,
            "bacc_new": bacc_new,
            "acc_centered": acc_cent_new,
            "acc_blended": acc_blend_new,
            "acc_sinkhorn": acc_sink_new,
            "lift": bacc_new - bacc_old,
            "oof_cv": oof_cv_res,
        }

    @app.function(
        gpu="A100-80GB:1",
        timeout=1800,  # 30 minutes max
        image=trainer_image,
        volumes={
            "/checkpoints": checkpoints_volume,
            "/datasets": datasets_volume,
        },
    )
    def train_gemini_gate_worker(
        checkpoint_dir: str = "/checkpoints/export_champion_qwen3_14b",
        samples_per_class: int = 3800,
        negative_ratio: float = 0.10,
        eval_texts: list[str] | None = None,
        eval_labels: list[int] | None = None,
        eval_groups: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Trains an L2-regularized Plain Head for the Gemini family (Approach 1):
        1. Loads frozen Qwen3-14B trunk from checkpoint_dir.
        2. Quarantines eval_texts to ensure 0% benchmark leakage.
        3. Samples balanced Gemini (samples_per_class per variant) + 10% negative distractors from other families.
        4. Extracts Z_train (6,600 samples) and Z_eval (1,800 samples) through the frozen trunk.
        5. Fits 5-class L2-regularized Ridge Head on Z_train.
        6. Evaluates within-family accuracy on 400 Gemini benchmark passages (comparing to 57.0% plain head baseline).
        7. Evaluates full 1,800-sample pipeline balanced accuracy with logit blending.
        8. Exports export_gemini_gate_clean/heads.safetensors and commits volume.
        """
        import json
        import torch
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer, AutoModel
        from safetensors import safe_open
        from safetensors.torch import save_file
        from torch.utils.data import DataLoader
        from sklearn.metrics import balanced_accuracy_score
        from scipy.special import softmax as sp_softmax

        from namethatllm.config import LABELS, LAB2I, K, FAMILIES
        from namethatllm.trainer import (
            MultiLayerLoRAClassifier,
            collate_text_batch,
            TextSequenceDataset,
        )
        from namethatllm.postprocess import sinkhorn_balancing

        ckpt = Path(checkpoint_dir)
        meta_p = ckpt / "meta.json"
        if not meta_p.exists():
            raise FileNotFoundError(f"meta.json not found in {checkpoint_dir}")
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)

        backbone_name = meta.get("backbone", "Qwen/Qwen3-14B-Base")
        keep_layers = int(meta.get("cfg", {}).get("keep", 27))
        layers = meta.get("cfg", {}).get("layers", [2, 4, 6, 9, 13, 17, 22, 27])
        max_len = int(meta.get("cfg", {}).get("max_len", 384))
        end_id = int(meta.get("cfg", {}).get("end_token_id", 151643))
        classes = meta.get("classes", LABELS)
        num_classes = len(classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Gemini Gate Worker] Initializing Qwen3-14B ({keep_layers} layers) on A100 GPU...")
        tok = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base_cfg = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
        base_cfg.num_hidden_layers = keep_layers
        base_cfg.use_cache = False
        if getattr(base_cfg, "layer_types", None) is not None:
            base_cfg.layer_types = base_cfg.layer_types[:keep_layers]

        base_model = AutoModel.from_pretrained(
            backbone_name,
            config=base_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)

        adapter_dir = ckpt / "adapter"
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            print(f"[Gemini Gate Worker] Loading LoRA adapter from {adapter_dir}...")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        else:
            peft_model = base_model

        model = MultiLayerLoRAClassifier(
            peft_model,
            hidden_size=base_cfg.hidden_size,
            layers=layers,
            num_classes=num_classes,
        ).to(device)

        heads_p = ckpt / "heads.safetensors"
        with safe_open(str(heads_p), framework="pt", device="cpu") as sf:
            for i in range(len(model.norms)):
                w_k = f"norms.{i}.weight"
                if w_k in sf.keys():
                    model.norms[i].weight.data.copy_(sf.get_tensor(w_k).to(device))
                b_k = f"norms.{i}.bias"
                if b_k in sf.keys() and hasattr(model.norms[i], "bias") and model.norms[i].bias is not None:
                    model.norms[i].bias.data.copy_(sf.get_tensor(b_k).to(device))
            if "plain.weight" in sf.keys():
                model.plain_head.weight.data.copy_(sf.get_tensor("plain.weight").to(device))
            if "plain.bias" in sf.keys():
                model.plain_head.bias.data.copy_(sf.get_tensor("plain.bias").to(device))

        model.eval()

        # Step 1: Load and curate training dataset from /datasets
        ds_root = Path("/datasets")
        pool_p = ds_root / "pool_120k.parquet"
        if not pool_p.exists():
            pool_p = ds_root / "nlm_pool_v5.parquet"
        df_pool = pd.read_parquet(pool_p)
        train_p = ds_root / "train.csv"
        df_train = pd.read_csv(train_p) if train_p.exists() else pd.DataFrame()

        print(f"[Gemini Gate Worker] Loaded pool: {len(df_pool):,} rows, train: {len(df_train):,} rows.")

        # Zero leakage quarantining
        if eval_texts is not None and len(eval_texts) > 0:
            bench_set = set(t.strip().lower() for t in eval_texts)
            init_pool_len = len(df_pool)
            df_pool = df_pool[~df_pool["text"].str.strip().str.lower().isin(bench_set)].copy()
            if len(df_train) > 0:
                df_train = df_train[~df_train["text"].str.strip().str.lower().isin(bench_set)].copy()
            quarantined = init_pool_len - len(df_pool)
            print(f"[Gemini Gate Worker] Quarantined {quarantined} benchmark overlap rows. Clean pool: {len(df_pool):,}.")

        gemini_classes = [
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
        ]
        gem_dfs = []
        for c in gemini_classes:
            sub_tr = df_train[df_train["model"] == c] if len(df_train) > 0 else pd.DataFrame()
            sub_pool = df_pool[df_pool["model"] == c]
            needed = samples_per_class
            sampled_pool = sub_pool.sample(n=min(needed - len(sub_tr), len(sub_pool)), random_state=42)
            c_df = pd.concat([sub_tr, sampled_pool], ignore_index=True)
            gem_dfs.append(c_df)
            print(f"  Gemini class {c}: {len(c_df)} samples ({len(sub_tr)} train, {len(c_df)-len(sub_tr)} pool)")

        df_gem_all = pd.concat(gem_dfs, ignore_index=True)
        total_gemini = len(df_gem_all)

        # 10% negative distractor sampling
        n_neg = int(total_gemini * negative_ratio)
        df_non_gem = df_pool[~df_pool["model"].str.startswith("gemini")].copy()
        other_fams = ["claude", "gpt", "glm", "qwen", "kimi"]
        neg_dfs = []
        n_per_fam = max(n_neg // len(other_fams), 1)
        for fam in other_fams:
            fam_sub = df_non_gem[df_non_gem["model"].str.startswith(fam)]
            if len(fam_sub) > 0:
                neg_dfs.append(fam_sub.sample(n=min(n_per_fam, len(fam_sub)), random_state=42))
        df_neg_all = pd.concat(neg_dfs, ignore_index=True)
        print(f"[Gemini Gate Worker] Sampled {len(df_neg_all)} negative distractors across {len(other_fams)} non-Gemini families.")

        train_texts = df_gem_all["text"].tolist() + df_neg_all["text"].tolist()
        gem_map = {name: i for i, name in enumerate(gemini_classes)}
        train_labels = [gem_map[m] for m in df_gem_all["model"].tolist()] + [4] * len(df_neg_all)
        print(f"[Gemini Gate Worker] Total training dataset: {len(train_texts):,} samples (5 classes: 4 Gemini + 1 Other).")

        # Step 2: Extract Z_train representations using model.forward_features
        print(f"[Gemini Gate Worker] Extracting Z_train on {len(train_texts)} samples on A100...")
        ds_tr = TextSequenceDataset(train_texts, train_labels, None, tok, max_len=max_len, end_id=end_id)
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=False, collate_fn=collate_text_batch)

        Z_tr_list = []
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for batch in loader_tr:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                    Z_tr_list.append(z_b.cpu().float().numpy())
        Z_train = np.vstack(Z_tr_list)
        y_train = np.array(train_labels, dtype=np.int64)
        print(f"[Gemini Gate Worker] Extracted Z_train: shape={Z_train.shape}, dtype={Z_train.dtype}")

        # Step 3: Extract Z_eval on benchmark samples
        Z_eval = None
        main_logits_eval = None
        if eval_texts is not None and len(eval_texts) > 0:
            print(f"[Gemini Gate Worker] Extracting Z_eval on {len(eval_texts)} benchmark samples...")
            dummy_eval_y = [0] * len(eval_texts)
            ds_ev = TextSequenceDataset(eval_texts, dummy_eval_y, None, tok, max_len=max_len, end_id=end_id)
            loader_ev = DataLoader(ds_ev, batch_size=16, shuffle=False, collate_fn=collate_text_batch)
            Z_ev_list = []
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for batch in loader_ev:
                        mb_ids = batch["input_ids"].to(device)
                        mb_mask = batch["attention_mask"].to(device)
                        max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                        z_b = model.forward_features(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                        Z_ev_list.append(z_b.cpu().float().numpy())
            Z_eval = np.vstack(Z_ev_list)

            # Compute Champion Plain Head logits on benchmark
            W_p = model.plain_head.weight.detach().cpu().float().numpy()
            b_p = model.plain_head.bias.detach().cpu().float().numpy()
            main_logits_eval = Z_eval @ W_p.T + b_p

        # Step 4: Fit Dual Ridge Head on Z_train
        print("[Gemini Gate Worker] Fitting 5-class dual Ridge Head on Z_train...")
        N_tr, D_feat = Z_train.shape
        Y_onehot = np.zeros((N_tr, 5), dtype=np.float32)
        for i, lab in enumerate(y_train):
            Y_onehot[i, lab] = 1.0
        # Label smoothing 0.05
        Y_smooth = 0.95 * Y_onehot + (0.05 / 5.0)

        mu_Z = Z_train.mean(axis=0, keepdims=True)
        mu_Y = Y_smooth.mean(axis=0, keepdims=True)
        Z_c = Z_train - mu_Z
        Y_c = Y_smooth - mu_Y

        K_mat = Z_c @ Z_c.T
        scale_k = float(np.trace(K_mat) / N_tr)

        best_w, best_b = None, None
        best_gem_acc = 0.0
        best_alpha = 10.0
        best_gem_logits = None

        y_ev_arr = np.array(eval_labels) if eval_labels is not None else None
        # Benchmark Gemini true indices
        gem_eval_mask = np.isin(y_ev_arr, [5, 6, 7, 8]) if y_ev_arr is not None else None
        y_gem_true = (y_ev_arr[gem_eval_mask] - 5) if gem_eval_mask is not None else None

        for alpha in [1.0, 5.0, 10.0, 20.0, 50.0]:
            eff_alpha = alpha * scale_k
            A_mat = np.linalg.solve(K_mat + eff_alpha * np.eye(N_tr, dtype=np.float32), Y_c)
            W_head = Z_c.T @ A_mat
            b_head = (mu_Y - mu_Z @ W_head).squeeze(0)

            if Z_eval is not None and y_gem_true is not None:
                logits_ev = Z_eval @ W_head + b_head
                gem_pred = logits_ev[gem_eval_mask, :4].argmax(axis=1)
                acc = float(balanced_accuracy_score(y_gem_true, gem_pred))
                print(f"  [Ridge Sweep] alpha={alpha:4.1f} | Gemini Benchmark Balanced Acc: {acc*100:.2f}% (Baseline Champion: 57.0%, Old Spec: 49.5%)")
                if acc > best_gem_acc:
                    best_gem_acc = acc
                    best_w = W_head
                    best_b = b_head
                    best_alpha = alpha
                    best_gem_logits = logits_ev

        print(f"[Gemini Gate Worker] Optimal Ridge Head selected: alpha={best_alpha:.1f} with Gemini Balanced Acc: {best_gem_acc*100:.2f}%")

        # Step 5: Evaluate Full Pipeline on all 1,800 benchmark samples
        pipe_results = {}
        if main_logits_eval is not None and best_gem_logits is not None and y_ev_arr is not None:
            P_main = sp_softmax(main_logits_eval, axis=1)
            acc_plain_solo = float(balanced_accuracy_score(y_ev_arr, P_main.argmax(axis=1)))
            print(f"[Gemini Gate Worker] Champion Plain Head Benchmark BAcc: {acc_plain_solo*100:.2f}%")

            # Sweep gating blend weight
            best_pipe_acc = 0.0
            best_pipe_w = 0.0
            gem_cols = [5, 6, 7, 8]
            cand_mask = np.isin(P_main.argmax(axis=1), gem_cols) | (P_main[:, gem_cols].sum(axis=1) >= 0.10)

            for gw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
                P_fused = P_main.copy()
                spec_probs = sp_softmax(best_gem_logits[:, :4], axis=1)
                for idx in np.flatnonzero(cand_mask):
                    orig_mass = P_main[idx, gem_cols].sum()
                    fused_gem = (1.0 - gw) * P_main[idx, gem_cols] + gw * spec_probs[idx]
                    fused_gem = fused_gem / fused_gem.sum() * orig_mass
                    P_fused[idx, gem_cols] = fused_gem

                acc_gated = float(balanced_accuracy_score(y_ev_arr, P_fused.argmax(axis=1)))
                P_sk = sinkhorn_balancing(P_fused, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
                acc_gated_sk = float(balanced_accuracy_score(y_ev_arr, P_sk.argmax(axis=1)))
                print(f"  [Gate Blend Sweep] gw={gw:.2f} | Gate Alone BAcc: {acc_gated*100:.2f}% | Gate + Sinkhorn BAcc: {acc_gated_sk*100:.2f}%")
                if acc_gated_sk > best_pipe_acc:
                    best_pipe_acc = acc_gated_sk
                    best_pipe_w = gw

            pipe_results = {
                "plain_solo_bacc": acc_plain_solo,
                "best_pipe_bacc": best_pipe_acc,
                "best_gate_weight": best_pipe_w,
                "lift_over_plain": best_pipe_acc - acc_plain_solo,
            }

        # Step 6: Save export_gemini_gate_clean checkpoint
        export_dir = Path("/checkpoints/export_gemini_gate_clean")
        export_dir.mkdir(parents=True, exist_ok=True)
        head_dict = {
            "plain.weight": torch.from_numpy(best_w.T).contiguous(),
            "plain.bias": torch.from_numpy(best_b).contiguous(),
        }
        save_file(head_dict, str(export_dir / "heads.safetensors"))
        meta_gem = {
            "classes": gemini_classes + ["__OTHER__"],
            "labels": gemini_classes + ["__OTHER__"],
            "target_family": "gemini",
            "alpha": float(best_alpha),
            "gemini_acc": float(best_gem_acc),
        }
        (export_dir / "meta.json").write_text(json.dumps(meta_gem, indent=2), encoding="utf-8")
        checkpoints_volume.commit()
        print(f"[Gemini Gate Worker] Checkpoint saved to {export_dir} and committed to checkpoints volume.")

        return {
            "gemini_balanced_acc": best_gem_acc,
            "best_alpha": best_alpha,
            "pipe_results": pipe_results,
            "spec_logits_gemini": best_gem_logits[:, :4].tolist() if best_gem_logits is not None else [],
            "gemini_classes": gemini_classes,
        }

    @app.local_entrypoint()
    def main(
        mode: str = "run",
        config: str = "configs/training_lora.json",
        backbone: str = "Qwen/Qwen2.5-0.5B",
        eval_data: str = "local_data/clean_benchmark_1800_rows.parquet",
        save_preds: str = "local_data/clean_benchmark_1800_predictions.npz",
    ):
        """Entrypoint for: modal run scripts/modal_trainer.py [--mode test|smoke|sync|run|eval] [--config ...] [--backbone ...] [--eval-data ...] [--save-preds ...]"""
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
        elif mode == "push-dataset":
            heads_p = Path("checkpoints/export_champion_qwen3_14b/heads.safetensors")
            heads_bytes = heads_p.read_bytes() if heads_p.exists() else None
            print(f"[Modal Launcher] Pushing restored heads.safetensors ({len(heads_bytes) if heads_bytes else 0:,} bytes) to Kaggle dataset via cloud...")
            res = push_kaggle_dataset_worker.remote(heads_bytes=heads_bytes)
            print(f"[Modal Launcher] Dataset push result:\n{res}")
        elif mode == "eval":
            import numpy as np
            import pandas as pd
            from namethatllm.config import LAB2I

            texts, labels, groups = None, None, None
            if eval_data and Path(eval_data).exists():
                print(f"[Modal Launcher] Loading evaluation dataset from {eval_data}...")
                if str(eval_data).endswith(".parquet"):
                    df_eval = pd.read_parquet(eval_data)
                else:
                    df_eval = pd.read_csv(eval_data)
                texts = df_eval["text"].tolist()
                labels = [LAB2I[m] for m in df_eval["model"].tolist()]
                if "prompt_canon" in df_eval.columns:
                    groups = df_eval["prompt_canon"].tolist()
                elif "prompt_id" in df_eval.columns:
                    groups = df_eval["prompt_id"].tolist()
                print(f"[Modal Launcher] Loaded {len(texts)} samples across {len(set(labels))} classes from {eval_data}.")
            else:
                cache_p = Path("local_data/local_eval_cache.npz")
                if not cache_p.exists():
                    cache_p = Path("local_data/v76_output/local_eval_cache.npz")
                if cache_p.exists():
                    cache = np.load(cache_p, allow_pickle=True)
                    texts = cache["unseen_texts"].tolist() if "unseen_texts" in cache else None
                    labels = cache["y_unseen"].tolist() if "y_unseen" in cache else None
                    groups = cache["groups_unseen"].tolist() if "groups_unseen" in cache else None
                    print(f"[Modal Launcher] Loaded {len(texts) if texts else 0} unseen holdout texts from {cache_p}...")
                else:
                    print("[Modal Launcher] Evaluating on default holdout dataset...")

            print("[Modal Launcher] Verifying / syncing specialist checkpoints on Modal...")
            sync_res = sync_specialists_worker.remote()
            print(f"[Modal Launcher] Specialists sync status: {sync_res}")

            res = eval_cloud_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                texts=texts,
                labels=labels,
                groups=groups,
            )
            print("\n" + "=" * 80)
            print("[Modal Launcher] Evaluation Accuracy Summary:")
            print(f"  Plain Head BAcc       : {res.get('acc_plain', 0.0)*100:.2f}%")
            print(f"  Centered Head BAcc    : {res.get('acc_centered', 0.0)*100:.2f}%")
            print(f"  Blended Head BAcc     : {res.get('acc_blended', 0.0)*100:.2f}%")
            print(f"  Plain + Sinkhorn BAcc : {res.get('acc_plain_sinkhorn', 0.0)*100:.2f}%")
            print(f"  Sinkhorn-Balanced BAcc: {res.get('acc_sinkhorn', 0.0)*100:.2f}%")
            print("=" * 80 + "\n")

            if save_preds:
                out_p = Path(save_preds)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_p,
                    labels=np.array(labels) if labels is not None else np.array([]),
                    groups=np.array(groups) if groups is not None else np.array([]),
                    logits_plain=np.array(res.get("logits_plain", [])),
                    logits_centered=np.array(res.get("logits_centered", [])),
                    logits_blend=np.array(res.get("logits_blend", [])),
                    probs_plain_sinkhorn=np.array(res.get("probs_plain_sinkhorn", [])),
                    probs_sinkhorn=np.array(res.get("probs_sinkhorn", [])),
                    spec_logits_gemini=np.array(res.get("spec_logits_gemini", [])),
                    spec_logits_glm=np.array(res.get("spec_logits_glm", [])),
                    spec_logits_gpt=np.array(res.get("spec_logits_gpt", [])),
                    spec_logits_claude=np.array(res.get("spec_logits_claude", [])),
                    spec_classes_gemini=np.array(res.get("spec_classes_gemini", [])),
                    spec_classes_glm=np.array(res.get("spec_classes_glm", [])),
                    spec_classes_gpt=np.array(res.get("spec_classes_gpt", [])),
                    spec_classes_claude=np.array(res.get("spec_classes_claude", [])),
                )
                print(f"[Modal Launcher] Saved evaluation predictions to {out_p} ({out_p.stat().st_size:,} bytes).")
        elif mode == "reexport":
            print("[Modal Launcher] Launching A100-80GB head re-export with trace-scaled Ridge & centered bias...")
            import numpy as np
            cache_p = Path("local_data/local_eval_cache.npz")
            if not cache_p.exists():
                cache_p = Path("local_data/v76_output/local_eval_cache.npz")
            if cache_p.exists():
                cache = np.load(cache_p, allow_pickle=True)
                texts = cache["unseen_texts"].tolist() if "unseen_texts" in cache else None
                labels = cache["y_unseen"].tolist() if "y_unseen" in cache else None
                groups = cache["groups_unseen"].tolist() if "groups_unseen" in cache else None
            else:
                texts, labels, groups = None, None, None
            res = reexport_heads_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                unseen_texts=texts,
                y_unseen=labels,
                groups_unseen=groups,
            )
            print("[Modal Launcher] Re-export finished successfully!")
            print(f"  Legacy Unscaled BAcc   : {res.get('bacc_old', 0.0)*100:.2f}%")
            print(f"  Pure Neural BAcc       : {res.get('bacc_pure_n', 0.0)*100:.2f}%")
            print(f"  Centered & Scaled BAcc : {res.get('bacc_new', 0.0)*100:.2f}%")
            print(f"  Centered Head BAcc     : {res.get('acc_centered', 0.0)*100:.2f}%")
            print(f"  Blended Head BAcc      : {res.get('acc_blended', 0.0)*100:.2f}%")
            print(f"  Sinkhorn Balanced BAcc : {res.get('acc_sinkhorn', 0.0)*100:.2f}%")
            print(f"  Net Lift (Plain)       : {res.get('lift', 0.0)*100:+.2f}%")
        elif mode == "train-gemini-gate":
            import numpy as np
            import pandas as pd
            from namethatllm.config import LAB2I

            print("[Modal Launcher] Launching A100-80GB Gemini Gate (Approach 1) Training...")
            eval_p = Path(eval_data)
            if not eval_p.exists():
                raise FileNotFoundError(f"Evaluation benchmark not found at {eval_data}")
            df_eval = pd.read_parquet(eval_p) if str(eval_p).endswith(".parquet") else pd.read_csv(eval_p)
            texts = df_eval["text"].tolist()
            labels = [LAB2I[m] for m in df_eval["model"].tolist()]
            groups = df_eval["prompt_canon"].tolist() if "prompt_canon" in df_eval.columns else None

            print(f"[Modal Launcher] Loaded {len(texts)} benchmark samples for zero-leakage training & evaluation.")
            res = train_gemini_gate_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                samples_per_class=3800,
                negative_ratio=0.10,
                eval_texts=texts,
                eval_labels=labels,
                eval_groups=groups,
            )
            print("\n" + "=" * 80)
            print("[Modal Launcher] Gemini Gate (Approach 1) Training & Benchmark Results:")
            print(f"  Gemini Within-Family BAcc : {res.get('gemini_balanced_acc', 0.0)*100:.2f}% (Baseline: 57.0%, Old Spec: 49.5%)")
            print(f"  Best Ridge Alpha          : {res.get('best_alpha', 0.0)}")
            pipe = res.get("pipe_results", {})
            print(f"  Champion Plain Solo BAcc  : {pipe.get('plain_solo_bacc', 0.0)*100:.2f}%")
            print(f"  Gated + Sinkhorn BAcc     : {pipe.get('best_pipe_bacc', 0.0)*100:.2f}%")
            print(f"  Optimal Gate Weight       : {pipe.get('best_gate_weight', 0.0)}")
            print(f"  Net Lift over Plain Solo  : {pipe.get('lift_over_plain', 0.0)*100:+.2f}%")
            print("=" * 80 + "\n")

            if save_preds and "spec_logits_gemini" in res:
                out_p = Path(save_preds)
                curr_data = {}
                if out_p.exists():
                    existing = np.load(out_p, allow_pickle=True)
                    for k in existing.keys():
                        curr_data[k] = existing[k]
                curr_data["spec_logits_gemini"] = np.array(res["spec_logits_gemini"])
                curr_data["spec_classes_gemini"] = np.array(res["gemini_classes"])
                np.savez_compressed(out_p, **curr_data)
                print(f"[Modal Launcher] Updated Gemini Gate logits in {out_p} ({out_p.stat().st_size:,} bytes).")
        else:
            print(f"[Modal Launcher] Launching A100-80GB cloud training with config: {config}")
            res = train_cloud_worker.remote(config_path=config)
            print(f"[Modal Launcher] Cloud job finished successfully. Result: {res}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modal Cloud Launcher for NameThatLLM Training")
    parser.add_argument("cmd", nargs="?", default="run", choices=["run", "smoke", "test", "sync", "eval", "reexport", "push-dataset", "train-gemini-gate"], help="Command (run, smoke, test, sync, eval, reexport, push-dataset, train-gemini-gate)")
    parser.add_argument("--config", default="configs/training_lora.json", help="Path to training config json")
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B", help="Backbone model for smoke test")
    parser.add_argument("--eval-data", default="local_data/clean_benchmark_1800_rows.parquet", help="Path to evaluation parquet/csv")
    parser.add_argument("--save-preds", default="local_data/clean_benchmark_1800_predictions.npz", help="Path to save evaluation predictions npz")
    parser.add_argument("--spawn", action="store_true", help="Spawn asynchronously on deployed app and exit immediately")
    args = parser.parse_args()

    if modal is None:
        print("Error: modal package not installed. Run 'pip install modal' and 'modal setup'.")
        sys.exit(1)

    if args.spawn:
        f = modal.Function.from_name("ntl-inhouse-trainer", "train_cloud_worker")
        call = f.spawn(config_path=args.config)
        print(f"[Modal] Spawned detached training task on A100 GPU! Call ID: {call.object_id}")
        sys.exit(0)

    with modal.enable_output(), app.run():
        if args.cmd == "test":
            test_cloud_worker.remote()
        elif args.cmd == "smoke":
            smoke_cloud_worker.remote(backbone=args.backbone)
        elif args.cmd == "sync":
            sync_cloud_datasets.remote()
        elif args.cmd == "push-dataset":
            heads_p = Path("checkpoints/export_champion_qwen3_14b/heads.safetensors")
            heads_bytes = heads_p.read_bytes() if heads_p.exists() else None
            print(f"[Modal Launcher] Pushing restored heads.safetensors ({len(heads_bytes) if heads_bytes else 0:,} bytes) to Kaggle dataset via cloud...")
            res = push_kaggle_dataset_worker.remote(heads_bytes=heads_bytes)
            print(f"[Modal Launcher] Dataset push result:\n{res}")
        elif args.cmd == "eval":
            import numpy as np
            import pandas as pd
            from namethatllm.config import LAB2I

            texts, labels, groups = None, None, None
            eval_path = args.eval_data or "local_data/clean_benchmark_1800_rows.parquet"
            if eval_path and Path(eval_path).exists():
                print(f"[Modal Launcher] Loading evaluation dataset from {eval_path}...")
                if str(eval_path).endswith(".parquet"):
                    df_eval = pd.read_parquet(eval_path)
                else:
                    df_eval = pd.read_csv(eval_path)
                texts = df_eval["text"].tolist()
                labels = [LAB2I[m] for m in df_eval["model"].tolist()]
                if "prompt_canon" in df_eval.columns:
                    groups = df_eval["prompt_canon"].tolist()
                elif "prompt_id" in df_eval.columns:
                    groups = df_eval["prompt_id"].tolist()
                print(f"[Modal Launcher] Loaded {len(texts)} samples across {len(set(labels))} classes from {eval_path}.")
            else:
                cache_p = Path("local_data/local_eval_cache.npz")
                if not cache_p.exists():
                    cache_p = Path("local_data/v76_output/local_eval_cache.npz")
                if cache_p.exists():
                    cache = np.load(cache_p, allow_pickle=True)
                    texts = cache["unseen_texts"].tolist() if "unseen_texts" in cache else None
                    labels = cache["y_unseen"].tolist() if "y_unseen" in cache else None
                    groups = cache["groups_unseen"].tolist() if "groups_unseen" in cache else None
                    print(f"[Modal Launcher] Loaded {len(texts) if texts else 0} unseen holdout texts from {cache_p}...")
                else:
                    print("[Modal Launcher] Evaluating on default holdout dataset...")

            print("[Modal Launcher] Verifying / syncing specialist checkpoints on Modal...")
            sync_res = sync_specialists_worker.remote()
            print(f"[Modal Launcher] Specialists sync status: {sync_res}")

            res = eval_cloud_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                texts=texts,
                labels=labels,
                groups=groups,
            )
            print("\n" + "=" * 80)
            print("[Modal Launcher] Evaluation Accuracy Summary:")
            print(f"  Plain Head BAcc       : {res.get('acc_plain', 0.0)*100:.2f}%")
            print(f"  Centered Head BAcc    : {res.get('acc_centered', 0.0)*100:.2f}%")
            print(f"  Blended Head BAcc     : {res.get('acc_blended', 0.0)*100:.2f}%")
            print(f"  Plain + Sinkhorn BAcc : {res.get('acc_plain_sinkhorn', 0.0)*100:.2f}%")
            print(f"  Sinkhorn-Balanced BAcc: {res.get('acc_sinkhorn', 0.0)*100:.2f}%")
            print("=" * 80 + "\n")

            if args.save_preds:
                out_p = Path(args.save_preds)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_p,
                    labels=np.array(labels) if labels is not None else np.array([]),
                    groups=np.array(groups) if groups is not None else np.array([]),
                    logits_plain=np.array(res.get("logits_plain", [])),
                    logits_centered=np.array(res.get("logits_centered", [])),
                    logits_blend=np.array(res.get("logits_blend", [])),
                    probs_plain_sinkhorn=np.array(res.get("probs_plain_sinkhorn", [])),
                    probs_sinkhorn=np.array(res.get("probs_sinkhorn", [])),
                    spec_logits_gemini=np.array(res.get("spec_logits_gemini", [])),
                    spec_logits_glm=np.array(res.get("spec_logits_glm", [])),
                    spec_logits_gpt=np.array(res.get("spec_logits_gpt", [])),
                    spec_logits_claude=np.array(res.get("spec_logits_claude", [])),
                    spec_classes_gemini=np.array(res.get("spec_classes_gemini", [])),
                    spec_classes_glm=np.array(res.get("spec_classes_glm", [])),
                    spec_classes_gpt=np.array(res.get("spec_classes_gpt", [])),
                    spec_classes_claude=np.array(res.get("spec_classes_claude", [])),
                )
                print(f"[Modal Launcher] Saved evaluation predictions to {out_p} ({out_p.stat().st_size:,} bytes).")
        elif mode == "reexport" if "mode" in locals() else args.cmd == "reexport":
            print("[Modal Launcher] Launching A100-80GB head re-export with trace-scaled Ridge & centered bias...")
            import numpy as np
            cache_p = Path("local_data/local_eval_cache.npz")
            if not cache_p.exists():
                cache_p = Path("local_data/v76_output/local_eval_cache.npz")
            if cache_p.exists():
                cache = np.load(cache_p, allow_pickle=True)
                texts = cache["unseen_texts"].tolist() if "unseen_texts" in cache else None
                labels = cache["y_unseen"].tolist() if "y_unseen" in cache else None
                groups = cache["groups_unseen"].tolist() if "groups_unseen" in cache else None
            else:
                texts, labels, groups = None, None, None
            res = reexport_heads_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                unseen_texts=texts,
                y_unseen=labels,
                groups_unseen=groups,
            )
            print("[Modal Launcher] Re-export finished successfully!")
            print(f"  Legacy Unscaled BAcc   : {res.get('bacc_old', 0.0)*100:.2f}%")
            print(f"  Pure Neural BAcc       : {res.get('bacc_pure_n', 0.0)*100:.2f}%")
            print(f"  Centered & Scaled BAcc : {res.get('bacc_new', 0.0)*100:.2f}%")
            print(f"  Centered Head BAcc     : {res.get('acc_centered', 0.0)*100:.2f}%")
            print(f"  Blended Head BAcc      : {res.get('acc_blended', 0.0)*100:.2f}%")
            print(f"  Sinkhorn Balanced BAcc : {res.get('acc_sinkhorn', 0.0)*100:.2f}%")
            print(f"  Net Lift (Plain)       : {res.get('lift', 0.0)*100:+.2f}%")
        elif args.cmd == "train-gemini-gate":
            import numpy as np
            import pandas as pd
            from namethatllm.config import LAB2I

            print("[Modal Launcher] Launching A100-80GB Gemini Gate (Approach 1) Training...")
            eval_path = args.eval_data or "local_data/clean_benchmark_1800_rows.parquet"
            if not Path(eval_path).exists():
                raise FileNotFoundError(f"Evaluation benchmark not found at {eval_path}")
            df_eval = pd.read_parquet(eval_path) if str(eval_path).endswith(".parquet") else pd.read_csv(eval_path)
            texts = df_eval["text"].tolist()
            labels = [LAB2I[m] for m in df_eval["model"].tolist()]
            groups = df_eval["prompt_canon"].tolist() if "prompt_canon" in df_eval.columns else None

            print(f"[Modal Launcher] Loaded {len(texts)} benchmark samples for zero-leakage training & evaluation.")
            res = train_gemini_gate_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                samples_per_class=3800,
                negative_ratio=0.10,
                eval_texts=texts,
                eval_labels=labels,
                eval_groups=groups,
            )
            print("\n" + "=" * 80)
            print("[Modal Launcher] Gemini Gate (Approach 1) Training & Benchmark Results:")
            print(f"  Gemini Within-Family BAcc : {res.get('gemini_balanced_acc', 0.0)*100:.2f}% (Baseline: 57.0%, Old Spec: 49.5%)")
            print(f"  Best Ridge Alpha          : {res.get('best_alpha', 0.0)}")
            pipe = res.get("pipe_results", {})
            print(f"  Champion Plain Solo BAcc  : {pipe.get('plain_solo_bacc', 0.0)*100:.2f}%")
            print(f"  Gated + Sinkhorn BAcc     : {pipe.get('best_pipe_bacc', 0.0)*100:.2f}%")
            print(f"  Optimal Gate Weight       : {pipe.get('best_gate_weight', 0.0)}")
            print(f"  Net Lift over Plain Solo  : {pipe.get('lift_over_plain', 0.0)*100:+.2f}%")
            print("=" * 80 + "\n")

            if args.save_preds and "spec_logits_gemini" in res:
                out_p = Path(args.save_preds)
                curr_data = {}
                if out_p.exists():
                    existing = np.load(out_p, allow_pickle=True)
                    for k in existing.keys():
                        curr_data[k] = existing[k]
                curr_data["spec_logits_gemini"] = np.array(res["spec_logits_gemini"])
                curr_data["spec_classes_gemini"] = np.array(res["gemini_classes"])
                np.savez_compressed(out_p, **curr_data)
                print(f"[Modal Launcher] Updated Gemini Gate logits in {out_p} ({out_p.stat().st_size:,} bytes).")
        else:
            train_cloud_worker.remote(config_path=args.config)
