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
        timeout=600,
        image=trainer_image,
        volumes={
            "/checkpoints": checkpoints_volume,
            "/datasets": datasets_volume,
        },
    )
    def eval_cloud_worker(
        checkpoint_dir: str = "/checkpoints/export_champion_qwen3_14b",
        texts: list[str] | None = None,
        labels: list[int] | None = None,
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
        from namethatllm.trainer import MultiLayerLoRAClassifier, collate_text_batch, TextSequenceDataset
        from torch.utils.data import DataLoader

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

        print(f"[Modal Eval] Running forward pass on {len(texts)} holdout texts...")
        dummy_labels = [0] * len(texts)
        ds = TextSequenceDataset(texts, dummy_labels, None, tok, max_len=max_len, end_id=end_id)
        loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_text_batch)

        plain_lps = []
        centered_lps = []
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for batch in loader:
                    mb_ids = batch["input_ids"].to(device)
                    mb_mask = batch["attention_mask"].to(device)
                    max_l = max(int(mb_mask.sum(dim=1).max().item()), 2)
                    _, lp_p, lp_c = model(mb_ids[:, :max_l], mb_mask[:, :max_l], drop_first_token=True)
                    plain_lps.append(lp_p.float().cpu().numpy())
                    centered_lps.append(lp_c.float().cpu().numpy())

        p_mat = np.vstack(plain_lps)
        c_mat = np.vstack(centered_lps)

        def _to_logp(m):
            m_max = np.max(m, axis=1, keepdims=True)
            e = np.exp(m - m_max)
            p = e / np.sum(e, axis=1, keepdims=True)
            return np.log(np.clip(p, 1e-12, 1.0))

        plain_logp = _to_logp(p_mat)
        centered_logp = _to_logp(c_mat)
        blended_logp = _to_logp(0.75 * c_mat + 0.25 * p_mat)

        out_res = {
            "plain_logp": plain_logp.tolist(),
            "centered_logp": centered_logp.tolist(),
            "blended_logp": blended_logp.tolist(),
        }

        if labels is not None:
            y_arr = np.array(labels)
            for name, lp in [("plain", plain_logp), ("centered", centered_logp), ("blended", blended_logp)]:
                pred = np.argmax(lp, axis=1)
                acc = float(np.mean(pred == y_arr))
                out_res[f"acc_{name}"] = acc
                print(f"[Modal Eval] Model ({name}): Top-1 Accuracy = {acc*100:.2f}%")

        return out_res

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
    ) -> dict[str, Any]:
        """
        Re-exports classification heads using the patched _blend_head formula with centered & scaled bias:
        1. Loads the frozen 14B LoRA model from checkpoint_dir/adapter.
        2. Loads /datasets/train.csv (5,000 samples) and extracts features Z_train.
        3. Fits exact closed-form Ridge heads: fit_ridge_heads(Z_train, y_train, prompts_train).
        4. Recovers pure neural weights (W_n, b_n) and blends with centered and scaled bias.
        5. Evaluates holdout accuracy on df_unseen_holdout before vs after.
        6. Backs up legacy heads.safetensors and writes the new corrected heads.safetensors.
        7. Commits volume permanently.
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

        heads_p = ckpt / "heads.safetensors"
        print(f"[Modal Re-Export] Reading existing heads weights from {heads_p}...")
        heads_sd_raw = {}
        with safe_open(str(heads_p), framework="pt", device="cpu") as sf:
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

        # Step 3: Fit closed-form Ridge heads on Z_train
        print("[Modal Re-Export] Fitting exact closed-form Ridge heads on Z_train...")
        W_p_r, b_p_r, W_c_r, b_c_r = fit_ridge_heads(
            Z_train, y_train, prompts_train, alpha=10.0, label_smoothing=0.10, num_classes=num_classes
        )

        # Step 4: Extract unseen holdout texts if available to benchmark before vs after
        df_unseen, _ = build_unseen_holdout_dataset(ds_root)
        unseen_texts = df_unseen["text"].tolist() if len(df_unseen) > 0 else []
        y_unseen = np.array([LAB2I[m] for m in df_unseen["model"]]) if len(df_unseen) > 0 else np.array([])
        print(f"[Modal Re-Export] Extracting Z_unseen on {len(unseen_texts)} holdout samples...")

        Z_unseen = None
        if len(unseen_texts) > 0:
            ds_u = TextSequenceDataset(unseen_texts, y_unseen.tolist(), None, tok, max_len=max_len, end_id=end_id)
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

        # Benchmark BEFORE: evaluation using existing heads.safetensors
        bacc_old = 0.0
        if Z_unseen is not None and len(y_unseen) > 0:
            W_p_old = heads_sd_raw["plain.weight"].float().numpy()
            b_p_old = heads_sd_raw["plain.bias"].float().numpy()
            logits_old = Z_unseen @ W_p_old.T + b_p_old
            bacc_old = float(balanced_accuracy_score(y_unseen, logits_old.argmax(axis=1)))
            print(f"[Modal Re-Export] BEFORE (Unscaled Bias) Unseen Plain Head BAcc: {bacc_old:.4f} ({bacc_old*100:.2f}%)")

        # Step 5: Invert the blend to recover the true pure neural heads (W_n, b_n)
        W_p_saved = heads_sd_raw["plain.weight"].float().numpy()
        b_p_saved = heads_sd_raw["plain.bias"].float().numpy()
        W_c_saved = heads_sd_raw["centered.weight"].float().numpy()
        b_c_saved = heads_sd_raw["centered.bias"].float().numpy()

        norm_p_saved = float(np.linalg.norm(W_p_saved))
        norm_p_r = float(np.linalg.norm(W_p_r))
        scale_p_old = (norm_p_saved / norm_p_r) if (norm_p_r > 1e-6 and norm_p_saved > 1e-6) else 1.0
        W_p_r_scaled = W_p_r * scale_p_old
        W_p_n = (W_p_saved - 0.25 * W_p_r_scaled) / 0.75
        b_p_n = (b_p_saved - 0.25 * b_p_r) / 0.75

        norm_c_saved = float(np.linalg.norm(W_c_saved))
        norm_c_r = float(np.linalg.norm(W_c_r))
        scale_c_old = (norm_c_saved / norm_c_r) if (norm_c_r > 1e-6 and norm_c_saved > 1e-6) else 1.0
        W_c_r_scaled = W_c_r * scale_c_old
        W_c_n = (W_c_saved - 0.25 * W_c_r_scaled) / 0.75
        b_c_n = (b_c_saved - 0.25 * b_c_r) / 0.75

        # Check pure neural BAcc
        bacc_pure_n = 0.0
        if Z_unseen is not None and len(y_unseen) > 0:
            logits_pure_n = Z_unseen @ W_p_n.T + b_p_n
            bacc_pure_n = float(balanced_accuracy_score(y_unseen, logits_pure_n.argmax(axis=1)))
            print(f"[Modal Re-Export] PURE NEURAL (No Ridge) Unseen Plain Head BAcc: {bacc_pure_n:.4f} ({bacc_pure_n*100:.2f}%)")

        # Step 6: Compute the CORRECT centered & scaled blend using _blend_head
        plain_linear = torch.nn.Linear(W_p_n.shape[1], W_p_n.shape[0])
        plain_linear.weight.data.copy_(torch.from_numpy(W_p_n))
        plain_linear.bias.data.copy_(torch.from_numpy(b_p_n))
        _blend_head(plain_linear, W_p_r, b_p_r, w_neural=0.75, w_ridge=0.25)

        centered_linear = torch.nn.Linear(W_c_n.shape[1], W_c_n.shape[0])
        centered_linear.weight.data.copy_(torch.from_numpy(W_c_n))
        centered_linear.bias.data.copy_(torch.from_numpy(b_c_n))
        _blend_head(centered_linear, W_c_r, b_c_r, w_neural=0.75, w_ridge=0.25)

        # Benchmark AFTER: evaluation with corrected centered & scaled bias
        bacc_new = 0.0
        if Z_unseen is not None and len(y_unseen) > 0:
            W_p_new = plain_linear.weight.detach().cpu().numpy()
            b_p_new = plain_linear.bias.detach().cpu().numpy()
            logits_new = Z_unseen @ W_p_new.T + b_p_new
            bacc_new = float(balanced_accuracy_score(y_unseen, logits_new.argmax(axis=1)))
            print(f"[Modal Re-Export] AFTER (Centered & Scaled Bias) Unseen Plain Head BAcc: {bacc_new:.4f} ({bacc_new*100:.2f}%)")
            print(f"[Modal Re-Export] NET LIFT: {bacc_new - bacc_old:+.4f} ({(bacc_new - bacc_old)*100:+.2f}%)")
        print(f"[Modal Re-Export] NET LIFT: {bacc_new - bacc_old:+.4f} ({(bacc_new - bacc_old)*100:+.2f}%)")

        # Step 7: Backup old heads.safetensors and save new heads.safetensors
        backup_p = ckpt / "heads_legacy_unscaled_bias.safetensors"
        if not backup_p.exists():
            shutil.copyfile(heads_p, backup_p)
            print(f"[Modal Re-Export] Backed up legacy heads to {backup_p}")

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
        print(f"[Modal Re-Export] Saved corrected heads to {heads_p}")

        checkpoints_volume.commit()
        print("[Modal Re-Export] Volume ntl-checkpoints committed permanently!")

        return {
            "bacc_old": bacc_old,
            "bacc_pure_n": bacc_pure_n,
            "bacc_new": bacc_new,
            "lift": bacc_new - bacc_old,
        }

    @app.local_entrypoint()
    def main(
        mode: str = "run",
        config: str = "configs/training_lora.json",
        backbone: str = "Qwen/Qwen2.5-0.5B",
    ):
        """Entrypoint for: modal run scripts/modal_trainer.py [--mode test|smoke|sync|run|eval] [--config ...] [--backbone ...]"""
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
        elif mode == "eval":
            import numpy as np
            cache_p = Path("local_data/local_eval_cache.npz")
            if cache_p.exists():
                cache = np.load(cache_p, allow_pickle=True)
                texts = cache["unseen_texts"].tolist()
                labels = cache["y_unseen"].tolist()
                print(f"[Modal Launcher] Loaded {len(texts)} unseen holdout texts from {cache_p}...")
            else:
                texts = None
                labels = None
                print("[Modal Launcher] Evaluating on default holdout dataset...")

            res = eval_cloud_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                texts=texts,
                labels=labels,
            )
            print("[Modal Launcher] Holdout Evaluation Results:")
            print(f"  Plain Head Accuracy   : {res.get('acc_plain', 0.0)*100:.2f}%")
            print(f"  Centered Head Accuracy: {res.get('acc_centered', 0.0)*100:.2f}%")
            print(f"  Blended Head Accuracy : {res.get('acc_blended', 0.0)*100:.2f}%")
        elif mode == "reexport":
            print("[Modal Launcher] Launching A100-80GB head re-export with centered & scaled bias...")
            res = reexport_heads_worker.remote(checkpoint_dir="/checkpoints/export_champion_qwen3_14b")
            print("[Modal Launcher] Re-export finished successfully!")
            print(f"  Legacy Unscaled BAcc   : {res.get('bacc_old', 0.0)*100:.2f}%")
            print(f"  Pure Neural BAcc       : {res.get('bacc_pure_n', 0.0)*100:.2f}%")
            print(f"  Centered & Scaled BAcc : {res.get('bacc_new', 0.0)*100:.2f}%")
            print(f"  Net Lift               : {res.get('lift', 0.0)*100:+.2f}%")
        else:
            print(f"[Modal Launcher] Launching A100-80GB cloud training with config: {config}")
            res = train_cloud_worker.remote(config_path=config)
            print(f"[Modal Launcher] Cloud job finished successfully. Result: {res}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modal Cloud Launcher for NameThatLLM Training")
    parser.add_argument("cmd", nargs="?", default="run", choices=["run", "smoke", "test", "sync", "eval", "reexport"], help="Command (run, smoke, test, sync, eval, reexport)")
    parser.add_argument("--config", default="configs/training_lora.json", help="Path to training config json")
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B", help="Backbone model for smoke test")
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
        elif args.cmd == "eval":
            import numpy as np
            cache_p = Path("local_data/local_eval_cache.npz")
            if cache_p.exists():
                cache = np.load(cache_p, allow_pickle=True)
                texts = cache["unseen_texts"].tolist()
                labels = cache["y_unseen"].tolist()
            else:
                texts = None
                labels = None
            res = eval_cloud_worker.remote(
                checkpoint_dir="/checkpoints/export_champion_qwen3_14b",
                texts=texts,
                labels=labels,
            )
            print("[Modal Launcher] Holdout Evaluation Results:")
            print(f"  Plain Head Accuracy   : {res.get('acc_plain', 0.0)*100:.2f}%")
            print(f"  Centered Head Accuracy: {res.get('acc_centered', 0.0)*100:.2f}%")
            print(f"  Blended Head Accuracy : {res.get('acc_blended', 0.0)*100:.2f}%")
        elif args.cmd == "reexport":
            print("[Modal Launcher] Launching A100-80GB head re-export with centered & scaled bias...")
            res = reexport_heads_worker.remote(checkpoint_dir="/checkpoints/export_champion_qwen3_14b")
            print("[Modal Launcher] Re-export finished successfully!")
            print(f"  Legacy Unscaled BAcc   : {res.get('bacc_old', 0.0)*100:.2f}%")
            print(f"  Pure Neural BAcc       : {res.get('bacc_pure_n', 0.0)*100:.2f}%")
            print(f"  Centered & Scaled BAcc : {res.get('bacc_new', 0.0)*100:.2f}%")
            print(f"  Net Lift               : {res.get('lift', 0.0)*100:+.2f}%")
        else:
            train_cloud_worker.remote(config_path=args.config)
