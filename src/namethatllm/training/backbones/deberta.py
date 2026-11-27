"""
DeBERTa Backbone Adapter (DeBERTa-v3-large) and Specialist Experiment Suite.
"""

from __future__ import annotations

import copy
import gc
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.special import softmax as sp_softmax
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig

from ...ablation import _text_hash
from ...config import FAMILIES, LAB2I, LABELS, resolve_all_pool_files, resolve_dataset_paths
from ...experiment_logger import ExperimentLedger
from ...postprocess import sinkhorn_balancing
from ..embeddings import extract_and_export_embeddings
from ..gates import run_sweep, train_family_gate_from_dataset
from ..modeling import MultiLayerLoRAClassifier, load_and_slice_champion_heads
from .base import BaseBackboneAdapter, PrecisionSpec, TokenizationSpec, register_backbone


@register_backbone("deberta")
class DebertaAdapter(BaseBackboneAdapter):
    """Strategy adapter for DeBERTa encoder backbones (microsoft/deberta-v3-large)."""

    family_name = "deberta"
    default_backbone = "microsoft/deberta-v3-large"
    default_output_dir = "export_champion_deberta"
    default_keep_layers = 24
    default_layers = (3, 6, 9, 12, 15, 18, 21, 24)
    default_max_len = 512
    default_end_id = 0

    def default_lora_targets(self) -> list[str]:
        return ["query_proj", "key_proj", "value_proj", "pos_key_proj", "pos_query_proj", "dense"]

    def resolve_model_source(self, cfg: dict[str, Any], input_dir: Path) -> str:
        backbone_name = str(cfg.get("backbone", self.default_backbone))
        if input_dir.exists():
            for cfg_p in input_dir.rglob("config.json"):
                try:
                    c_data = json.loads(cfg_p.read_text(encoding="utf-8"))
                    if "deberta" in str(c_data.get("model_type", "")).lower():
                        return str(cfg_p.parent)
                except Exception:
                    continue
        return backbone_name

    def tok_spec_from_tokenizer(
        self, tok: Any, max_len: int | None = None, end_id: int | None = None
    ) -> TokenizationSpec:
        tok_pad = getattr(tok, "pad_token_id", None)
        pad_id = int(tok_pad) if tok_pad is not None else 0
        raw_cls = getattr(tok, "cls_token_id", None)
        cls_id = int(raw_cls) if raw_cls is not None else None
        raw_sep = getattr(tok, "sep_token_id", None)
        if raw_sep is not None:
            eff_end_id = int(raw_sep)
        elif end_id is not None:
            eff_end_id = int(end_id)
        else:
            eff_end_id = self.default_end_id
        try:
            vocab_sz = len(tok)
            if eff_end_id >= vocab_sz:
                tok_eos = getattr(tok, "eos_token_id", None)
                tok_sep = getattr(tok, "sep_token_id", None)
                if tok_eos is not None:
                    eff_end_id = int(tok_eos)
                elif tok_sep is not None:
                    eff_end_id = int(tok_sep)
                else:
                    eff_end_id = 2
        except Exception:
            pass
        return TokenizationSpec(
            max_len=int(max_len if max_len is not None else self.default_max_len),
            pad_id=pad_id,
            cls_id=cls_id,
            end_id=eff_end_id,
            drop_first_token=False,
        )

    def load_tokenizer(self, model_source: str, cfg: dict[str, Any], is_smoke: bool = False) -> tuple[Any, TokenizationSpec]:
        mock_tok = cfg.get("mock_tokenizer")
        if mock_tok is not None:
            tok = mock_tok
        else:
            try:
                tok = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
            except Exception:
                if is_smoke:
                    class _FallbackTok:
                        pad_token_id = 0
                        cls_token_id = 1
                        sep_token_id = 2
                        eos_token_id = 2
                        pad_token = "<pad>"
                        eos_token = "</s>"

                        def __call__(self, text, **kwargs):
                            return {"input_ids": [ord(c) % 500 + 10 for c in str(text)[:30]]}

                        def __len__(self):
                            return 128100

                    tok = _FallbackTok()
                else:
                    raise
        max_len = int(cfg.get("max_len", self.default_max_len))
        cfg_end = cfg.get("end_id")
        end_id = int(cfg_end) if cfg_end is not None else self.default_end_id
        return tok, self.tok_spec_from_tokenizer(tok, max_len=max_len, end_id=end_id)

    def load_backbone(
        self, model_source: str, cfg: dict[str, Any], precision: PrecisionSpec, device: torch.device
    ) -> tuple[nn.Module, int, int, list[int]]:
        keep_layers = int(cfg.get("keep_layers", self.default_keep_layers))
        layers = list(cfg.get("layers", self.default_layers))
        mock_bb = cfg.get("mock_backbone")
        if mock_bb is not None:
            base_model = copy.deepcopy(mock_bb).to(device)
            base_cfg = getattr(base_model, "config", type("Config", (), {"hidden_size": 32, "num_hidden_layers": 24})())
            return base_model, int(base_cfg.hidden_size), keep_layers, layers

        base_cfg = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        max_backbone_layers = int(getattr(base_cfg, "num_hidden_layers", 24))
        keep_layers = min(keep_layers, max_backbone_layers, 24)
        if any(l > keep_layers for l in layers) or len(layers) == 0:
            layers = list(self.default_layers)
        base_cfg.num_hidden_layers = keep_layers
        base_cfg.use_cache = False

        print(f"[Trainer] DeBERTa model detected. Loading in torch.float32 directly on {device} (bypassing device_map='auto')...")
        base_model = AutoModel.from_pretrained(
            model_source, config=base_cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device)
        return base_model, int(base_cfg.hidden_size), keep_layers, layers

    def build_scalers(self, precision: PrecisionSpec, device: torch.device) -> tuple[Any | None, Any | None]:
        enabled = bool(precision.compute_dtype == torch.float16 and device.type == "cuda" and torch.cuda.is_available())
        return torch.amp.GradScaler("cuda", enabled=enabled), None

    def adjust_stage1_batch_config(
        self, stage1_cfg: dict[str, Any], precision: PrecisionSpec
    ) -> tuple[int, int, int]:
        bs1 = int(stage1_cfg.get("batch_size", 2))
        accum1 = int(stage1_cfg.get("gradient_accumulation_steps", 8))
        eval_bs = int(stage1_cfg.get("eval_batch_size", 32))
        return bs1, accum1, eval_bs

    def stage2_eval_batch_size(self) -> int:
        return 16

    def specialist_phase_tag(self, target_family: str) -> str:
        return "P3_deberta_finetuned"

    def specialist_feature_set_tag(self, target_family: str) -> str:
        return "F6b_finetuned_deberta_direct"

    def specialist_notes_label(self) -> str:
        return "DeBERTa"

    def maybe_run_pipeline(
        self, train_cfg: dict[str, Any], step_callback: Any | None = None, **kwargs: Any
    ) -> Any | None:
        if train_cfg.get("pipeline_mode") == "deberta_specialist_suite":
            return run_deberta_specialist_suite(train_cfg, step_callback=step_callback, run_pipeline_fn=kwargs.get("run_pipeline_fn"))
        return None


def run_deberta_specialist_suite(
    train_cfg: dict[str, Any],
    step_callback: Any = None,
    run_pipeline_fn: Any = None,
) -> dict[str, Any]:
    """Executes the High-Sample DeBERTa-v3-Large Specialist Experiment Suite (EXP-06a, EXP-06b, EXP-06c)."""
    if run_pipeline_fn is None:
        run_pipeline_fn = globals().get("run_training_pipeline") or importlib.import_module("namethatllm.training.engine").run_training_pipeline

    adapter = DebertaAdapter()
    backbone_name = str(train_cfg.get("backbone", adapter.default_backbone))
    target_family = str(train_cfg.get("target_family", "gemini")).lower().strip()
    keep_layers = min(int(train_cfg.get("keep_layers", 24)), 24)
    layers = list(train_cfg.get("layers", adapter.default_layers))
    if any(l > keep_layers for l in layers):
        layers = list(adapter.default_layers)
    max_len = int(train_cfg.get("max_len", 446))
    samples_per_class = int(train_cfg.get("samples_per_class", 8500))
    negative_ratio = float(train_cfg.get("negative_ratio", 0.10))
    batch_size = int(train_cfg.get("batch_size", 32))
    out_dir = Path(train_cfg.get("output_dir", "export_deberta_gemini_specialist"))
    gate_out_dir = Path(train_cfg.get("gate_output_dir", "export_deberta_gemini_gate_mlp"))
    emb_out_dir = Path(train_cfg.get("embeddings_output_dir", "export_embeddings_deberta_finetuned"))
    input_dir = Path(train_cfg.get("input_dir", "/kaggle/input"))
    is_smoke = bool(train_cfg.get("smoke", False))
    device = torch.device(train_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoints_volume = train_cfg.get("checkpoints_volume", None)
    ledger_dir = Path(train_cfg.get("ledger_dir", "experiments"))
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger(ledger_dir=ledger_dir)

    fam_cols = FAMILIES[target_family]
    fam_classes = [LABELS[i] for i in fam_cols]
    num_fam_classes = len(fam_classes)
    eff_hard_pair = (2, 3) if target_family == "gemini" else ((1, 2) if target_family in ("gpt", "claude") else (0, 1))

    print("\n" + "=" * 92)
    print(f" [DEBERTA SPECIALIST SUITE] High-Sample ({samples_per_class}/class) Suite for '{target_family.upper()}'")
    print(f" Output Specialist: {out_dir} | Output Gate: {gate_out_dir} | Embeddings: {emb_out_dir}")
    print("=" * 92)

    bench_paths = list(input_dir.rglob("clean_benchmark_1800_rows.parquet"))
    if not bench_paths and Path("clean_benchmark_1800_rows.parquet").exists():
        bench_paths = [Path("clean_benchmark_1800_rows.parquet")]

    df_bench = None
    if bench_paths and bench_paths[0].exists():
        try:
            df_bench = pd.read_parquet(bench_paths[0])
        except Exception as e:
            print(f"[DeBERTa Suite] Note reading benchmark parquet: {e}")

    if df_bench is None or len(df_bench) == 0:
        if is_smoke:
            syn_records = [{"text": f"Benchmark sample passage text for {m} sample {rep} test", "model": m} for m in LABELS for rep in range(2)]
            df_bench = pd.DataFrame(syn_records)
        else:
            raise FileNotFoundError("clean_benchmark_1800_rows.parquet not found for DeBERTa specialist suite.")

    eval_texts = df_bench["text"].tolist()
    eval_labels = [LAB2I[m] for m in df_bench["model"].tolist()]
    eval_model_names = df_bench["model"].tolist()
    bench_hashes = {_text_hash(t) for t in eval_texts}
    print(f"[DeBERTa Suite] Loaded clean benchmark with {len(eval_texts)} passages ({len(bench_hashes)} unique hashes).")

    champ_paths = (
        list(input_dir.rglob("benchmark_champion_logits.safetensors"))
        + list(input_dir.rglob("champion_logits.safetensors"))
        + list(Path(".").rglob("*champion_logits.safetensors"))
    )
    champ_main_logits = None
    baseline_raw_bacc = 0.0
    baseline_sk_bacc = 0.0
    fam_solo_bacc = 0.0
    hard_pair_solo_acc = 0.0

    if champ_paths:
        try:
            champ_sd = load_file(str(champ_paths[0]))
            c_key = "logits_plain" if "logits_plain" in champ_sd else list(champ_sd.keys())[0]
            cand_l = champ_sd[c_key].cpu().float().numpy()
            if len(cand_l) == len(eval_texts) and cand_l.shape[1] == len(LABELS):
                champ_main_logits = cand_l
                P_main = sp_softmax(champ_main_logits, axis=1)
                baseline_raw_bacc = float(balanced_accuracy_score(eval_labels, P_main.argmax(axis=1)))
                P_sk = sinkhorn_balancing(P_main, iters=50, sinkhorn_strength=0.65, target_mix=0.35)
                baseline_sk_bacc = float(balanced_accuracy_score(eval_labels, P_sk.argmax(axis=1)))

                fam_eval_mask = np.isin(eval_labels, fam_cols)
                if fam_eval_mask.sum() > 0:
                    y_fam_true = np.array([fam_cols.index(c) for c in np.array(eval_labels)[fam_eval_mask]])
                    solo_fam_pred = champ_main_logits[fam_eval_mask][:, fam_cols].argmax(axis=1)
                    fam_solo_bacc = float(balanced_accuracy_score(y_fam_true, solo_fam_pred))
                    pair_mask = np.isin(y_fam_true, [eff_hard_pair[0], eff_hard_pair[1]])
                    if pair_mask.sum() > 0 and len(np.unique(y_fam_true[pair_mask])) == 2:
                        s_a, s_b = eff_hard_pair
                        fam_logits_pair = champ_main_logits[fam_eval_mask][pair_mask][:, fam_cols]
                        pair_preds = np.where(fam_logits_pair[:, s_a] >= fam_logits_pair[:, s_b], s_a, s_b)
                        hard_pair_solo_acc = float(balanced_accuracy_score(y_fam_true[pair_mask], pair_preds))
                print(f"[DeBERTa Suite] Baseline Champion: Raw={baseline_raw_bacc*100:.2f}% | Sinkhorn={baseline_sk_bacc*100:.2f}% | Gemini Solo={fam_solo_bacc*100:.2f}%")
        except Exception as e:
            print(f"[DeBERTa Suite] Note on reading champion logits: {e}")

    ledger.add_entry({
        "exp_id": "EXP-00",
        "phase": "P0_baseline",
        "target_family": target_family,
        "feature_set": "F0_plain_solo",
        "subspace_dim": 0,
        "head_type": "plain",
        "gate_weight": 0.0,
        "oof_family_bacc": 0.0,
        "holdout_family_bacc": fam_solo_bacc,
        "hard_sister_acc": hard_pair_solo_acc,
        "holdout_18cls_raw_bacc": baseline_raw_bacc,
        "holdout_18cls_sinkhorn_bacc": baseline_sk_bacc,
        "delta_vs_baseline_bacc": 0.0,
        "notes": "18-Class In-House Champion Plain Solo Baseline",
    })

    train_path, _, pool_file = resolve_dataset_paths(input_dir if input_dir.exists() else Path("."))
    all_pools = resolve_all_pool_files(input_dir if input_dir.exists() else Path("."))

    df_train = None
    if train_path and train_path.exists():
        try:
            df_train = pd.read_csv(train_path)
            tr_hashes = [_text_hash(t) for t in df_train["text"].fillna("")]
            df_train = df_train[[h not in bench_hashes for h in tr_hashes]].reset_index(drop=True)
            print(f"[DeBERTa Suite] Retained {len(df_train)} clean non-quarantined train.csv rows.")
        except Exception as e:
            print(f"[DeBERTa Suite] Note reading train_path: {e}")

    if df_train is None or len(df_train) == 0:
        if is_smoke:
            syn_tr = [{"text": f"Train passage for {m} prompt {rep} text content", "model": m, "prompt_id": f"p_{rep}"} for m in LABELS for rep in range(3)]
            df_train = pd.DataFrame(syn_tr)
        else:
            raise FileNotFoundError("train.csv not found for specialist suite.")

    dfs_pool = []
    pool_list = all_pools if all_pools else ([pool_file] if pool_file else [])
    for p in pool_list:
        try:
            df_one = pd.read_parquet(p, columns=["text", "model"]) if str(p).endswith(".parquet") else pd.read_csv(p, usecols=["text", "model"])
            dfs_pool.append(df_one)
        except Exception:
            pass

    if dfs_pool:
        df_pool = pd.concat(dfs_pool, ignore_index=True)
    elif is_smoke:
        df_pool = pd.DataFrame([{"text": f"Pool sample passage text for {m} item {rep} with enough characters to be valid", "model": m} for m in LABELS for rep in range(8)])
    else:
        df_pool = pd.DataFrame(columns=["text", "model"])

    if len(df_pool) > 0:
        df_pool = df_pool.dropna(subset=["text", "model"])
        df_pool["text"] = df_pool["text"].astype(str)
        p_hashes = [_text_hash(t) for t in df_pool["text"]]
        df_pool = df_pool[[h not in bench_hashes for h in p_hashes]].reset_index(drop=True)
        print(f"[DeBERTa Suite] Retained {len(df_pool)} clean non-quarantined pool rows (0.0% overlap with benchmark).")

    model_source = adapter.resolve_model_source(train_cfg, input_dir)
    mock_backbone = train_cfg.get("mock_backbone")
    tok, tok_spec = adapter.load_tokenizer(model_source, {**train_cfg, "max_len": max_len}, is_smoke=is_smoke)

    def _make_fallback_deberta(h_dim: int = 32, n_l: int = 24) -> nn.Module:
        class _FallbackConfig(PretrainedConfig):
            model_type = "deberta-v2"

            def __init__(self, hidden_size=32, num_hidden_layers=24, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers

        class _FallbackDeberta(nn.Module):
            def __init__(self, hd=32, nl=24):
                super().__init__()
                self.config = _FallbackConfig(hidden_size=hd, num_hidden_layers=nl)
                self.embed = nn.Embedding(128100, hd)
                self.layers = nn.ModuleList([nn.Linear(hd, hd) for _ in range(nl)])

            def forward(self, input_ids, **kwargs):
                h = self.embed(input_ids)
                states = [h]
                for l in self.layers:
                    h = l(h)
                    states.append(h)
                return type("Output", (), {"hidden_states": states, "last_hidden_state": h})()

        return _FallbackDeberta(h_dim, n_l).to(device)

    # EXP-06a: Frozen Public DeBERTa
    print("\n" + "=" * 88)
    print(" [EXP-06a] Extracting Frozen Public DeBERTa Embeddings & Training Heads...")
    print("=" * 88)
    exp_6a_emb_dir = Path("export_embeddings_deberta_public_frozen")
    found_pre_6a = None
    if (exp_6a_emb_dir / "features_concat.safetensors").exists() and (exp_6a_emb_dir / "metadata.parquet").exists():
        found_pre_6a = exp_6a_emb_dir
    elif input_dir.exists():
        for cand in input_dir.rglob("features_concat.safetensors"):
            cand_dir = cand.parent
            if (cand_dir / "metadata.parquet").exists() and "deberta" in str(cand_dir).lower():
                found_pre_6a = cand_dir
                break

    if found_pre_6a is not None:
        exp_6a_emb_dir = found_pre_6a
    else:
        exp_6a_emb_dir.mkdir(parents=True, exist_ok=True)
        if mock_backbone is not None:
            base_model_frozen = mock_backbone
            hidden_sz = getattr(base_model_frozen.config, "hidden_size", 64)
        else:
            try:
                base_cfg_f = AutoConfig.from_pretrained(model_source)
                base_cfg_f.num_hidden_layers = keep_layers
                base_model_frozen = AutoModel.from_pretrained(model_source, config=base_cfg_f, torch_dtype=torch.float32).to(device)
                hidden_sz = base_cfg_f.hidden_size
            except Exception:
                if is_smoke:
                    base_model_frozen = _make_fallback_deberta(32, keep_layers)
                    hidden_sz = 32
                else:
                    raise

        model_frozen = MultiLayerLoRAClassifier(
            base_model_frozen,
            hidden_size=hidden_sz,
            layers=layers,
            num_classes=num_fam_classes,
            backbone_family=adapter.family_name,
            tok_spec=tok_spec,
        ).to(device)
        model_frozen.eval()

        extract_and_export_embeddings(
            model=model_frozen,
            tok=tok,
            df_pool=df_pool,
            df_train=df_train,
            target=target_family,
            save_all_layers=False,
            save_multiview=False,
            output_dir=exp_6a_emb_dir,
            samples_per_class=min(samples_per_class, 8 if is_smoke else samples_per_class),
            negative_ratio=negative_ratio,
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            eval_model_names=eval_model_names,
            batch_size=min(batch_size, 4 if is_smoke else batch_size),
            max_len=min(max_len, 64 if is_smoke else max_len),
            end_id=tok_spec.end_id,
            device=device,
            eval_champion_logits_plain=champ_main_logits,
        )

    res_6a = train_family_gate_from_dataset(
        dataset_dir=exp_6a_emb_dir,
        target_family=target_family,
        output_dir=None,
        mlp_epochs=2 if is_smoke else int(train_cfg.get("mlp_epochs", 12)),
        gate_weight=float(train_cfg.get("gate_weight", 0.40)),
        device=device,
    )

    heads_bk = res_6a.get("heads_breakdown", {})
    for h_name in ["ridge", "mlp", "hybrid"]:
        if h_name in heads_bk:
            h_info = heads_bk[h_name]
            p_res = h_info.get("pipe_results", {})
            h_sk = p_res.get("best_sinkhorn_bacc", 0.0)
            h_raw = p_res.get("best_raw_bacc", 0.0)
            gw_val = p_res.get("best_gate_weight", 0.40)
            ledger.add_entry({
                "exp_id": f"EXP-06a_{h_name}",
                "phase": "P2_deberta_frozen",
                "target_family": target_family,
                "feature_set": "F6a_frozen_public_deberta",
                "subspace_dim": res_6a.get("subspace_dim", 0),
                "head_type": h_name,
                "gate_weight": gw_val,
                "oof_family_bacc": h_info.get("oof_acc", 0.0),
                "holdout_family_bacc": h_info.get("eval_fam_acc", 0.0),
                "hard_sister_acc": h_info.get("hard_pair_acc", 0.0),
                "holdout_18cls_raw_bacc": h_raw,
                "holdout_18cls_sinkhorn_bacc": h_sk,
                "delta_vs_baseline_bacc": (h_sk - baseline_sk_bacc) if baseline_sk_bacc > 0 else 0.0,
                "notes": f"Frozen Public DeBERTa + OOF Fisher ({h_name.upper()}, gw={gw_val:.2f})",
            })

    if "model_frozen" in locals():
        del model_frozen
    if "base_model_frozen" in locals():
        del base_model_frozen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if found_pre_6a is None:
        shutil.rmtree(str(exp_6a_emb_dir), ignore_errors=True)

    # EXP-06b: Fine-Tuned DeBERTa Specialist Model
    print("\n" + "=" * 88)
    print(" [EXP-06b] Fine-Tuning DeBERTa-v3-Large Specialist Model...")
    print("=" * 88)
    cfg_b = copy.deepcopy(train_cfg)
    cfg_b["pipeline_mode"] = "specialist"
    cfg_b["backbone_family"] = "deberta"
    cfg_b["output_dir"] = str(out_dir)
    cfg_b["exp_prefix"] = "EXP-06b"
    if is_smoke:
        cfg_b["smoke"] = True
        s1 = cfg_b.setdefault("stage1", {})
        s1["epochs"] = 1
        s1["max_samples"] = 16
        s2 = cfg_b.setdefault("stage2", {})
        s2["epochs"] = 1
        s2["n_splits"] = 2
    run_pipeline_fn(cfg_b, step_callback=step_callback)

    # EXP-06c: Fine-Tuned Embeddings + Probing Sweep
    print("\n" + "=" * 88)
    print(" [EXP-06c] Extracting Multi-Layer Embeddings from Fine-Tuned Specialist...")
    print("=" * 88)
    emb_out_dir.mkdir(parents=True, exist_ok=True)
    if mock_backbone is not None:
        merged_backbone = mock_backbone
        h_sz = getattr(merged_backbone.config, "hidden_size", 64)
    else:
        try:
            base_cfg_c = AutoConfig.from_pretrained(model_source)
            base_cfg_c.num_hidden_layers = keep_layers
            base_c = AutoModel.from_pretrained(model_source, config=base_cfg_c, torch_dtype=torch.float32).to(device)
            from peft import PeftModel

            adapter_load_dir = out_dir / "adapter" if (out_dir / "adapter" / "adapter_config.json").exists() else out_dir
            peft_m = PeftModel.from_pretrained(base_c, str(adapter_load_dir))
            merged_backbone = peft_m.merge_and_unload()
            h_sz = base_cfg_c.hidden_size
        except Exception:
            if is_smoke:
                merged_backbone = _make_fallback_deberta(32, keep_layers)
                h_sz = 32
            else:
                raise

    model_finetuned = MultiLayerLoRAClassifier(
        merged_backbone,
        hidden_size=h_sz,
        layers=layers,
        num_classes=num_fam_classes,
        backbone_family=adapter.family_name,
        tok_spec=tok_spec,
    ).to(device)
    heads_ckpt = out_dir / "heads.safetensors"
    if heads_ckpt.exists():
        load_and_slice_champion_heads(heads_ckpt, model_finetuned, fam_cols=None)
    model_finetuned.eval()

    extract_and_export_embeddings(
        model=model_finetuned,
        tok=tok,
        df_pool=df_pool,
        df_train=df_train,
        target=target_family,
        save_all_layers=True,
        save_multiview=True,
        output_dir=emb_out_dir,
        samples_per_class=min(samples_per_class, 8 if is_smoke else samples_per_class),
        negative_ratio=negative_ratio,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        eval_model_names=eval_model_names,
        batch_size=min(batch_size, 4 if is_smoke else batch_size),
        max_len=min(max_len, 64 if is_smoke else max_len),
        end_id=tok_spec.end_id,
        device=device,
        eval_champion_logits_plain=champ_main_logits,
    )

    del model_finetuned, merged_backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sweep_summary = run_sweep(
        embeddings_dir=emb_out_dir,
        target_family=target_family,
        output_dir=gate_out_dir,
        ledger_dir=ledger_dir,
        device=device,
        exp_prefix="EXP-06c",
    )

    ledger.export_copy(out_dir)
    ledger.export_copy(gate_out_dir)
    return {
        "suite": "deberta_specialist_suite",
        "specialist_dir": str(out_dir),
        "gate_dir": str(gate_out_dir),
        "embeddings_dir": str(emb_out_dir),
        "sweep_summary": sweep_summary,
    }
