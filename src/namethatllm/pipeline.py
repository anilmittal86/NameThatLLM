"""
Top-Level Pipeline Orchestrator
===============================
Coordinates the complete end-to-end execution flow:
- Module 1: Dual-Output Lexical Engine (subprocess isolated or in-process)
- Module 2: Neural Trunk Extraction Loop (All 4 trunks and 6 checkpoints)
- Module 3: wPCA DP Posterior (ExactAuthorPosterior) + 384-Dim Spectral Bank
- Module 4: Two-Tier Simplex Blend + Vectorized Claude Gate + Sinkhorn Balancing
"""

import os
import gc
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    LABELS, K, LAB2I, FAMILIES, PRIOR_SCALES,
    softmax, log_softmax, align_cols,
    resolve_dataset_paths, resolve_all_pool_files, find_base_model, find_checkpoints,
    find_fold_adapters,
    validate_production_assets
)
from .lexical_engine import run_lexical_engine
from .neural_extractor import (
    UniversalLoRATrunkLoader,
    load_head_file,
    resolve_head_tensors,
    compute_centered_logits,
    merge_lora,
    read_checkpoint_meta,
    extract_deberta_checkpoint
)
from .wpca_gaussian import (
    fit_shared_gaussian_lowram,
    transform_lowram,
    split_oversized_clusters_dist,
    scores_exact_dp,
    load_wpca_bundle,
    save_wpca_bundle
)
from .spectral_bank import (
    fit_joint_spectral_bank,
    make_spectral_blocks,
    project_spectral_block,
    load_spectral_params,
    predict_spectral_bank
)
from .postprocess import (
    apply_claude_specialist_gate,
    damped_sinkhorn,
    run_full_inference_pipeline,
    generate_submission
)
from .ablation import build_unseen_holdout_dataset, UnseenStepwiseAuditor, _fast_sinkhorn_bacc


def _auto_align_labels_on_unseen(
    Z_unseen: np.ndarray,
    W_p: Optional[np.ndarray],
    b_p: Optional[np.ndarray],
    W_c: Optional[np.ndarray],
    b_c: Optional[np.ndarray],
    default_labels: list,
    tracker_unseen: Any,
    y_unseen: np.ndarray,
    w_p: float,
    w_c: float,
    blend_weights: Tuple[float, float, float],
    compute_centered_fn: Any,
) -> Tuple[list, np.ndarray, float]:
    """Selects canonical vs alphabetical label order maximizing solo BAcc on unseen holdout."""
    if W_p is None or len(Z_unseen) == 0 or len(y_unseen) == 0:
        return list(default_labels), np.zeros((0, K), dtype=np.float32), 0.0

    cand_orders = [list(default_labels)]
    for candidate in (sorted(LABELS), list(LABELS)):
        if candidate not in cand_orders:
            cand_orders.append(candidate)

    best_order = list(default_labels)
    best_bacc = -1.0
    best_lp = None
    for cand_o in cand_orders:
        try:
            trial_lp_p = log_softmax(align_cols(Z_unseen @ W_p.T + b_p, cand_o))
            if W_c is not None and tracker_unseen is not None and w_c > 0.0:
                trial_lp_c = compute_centered_fn(Z_unseen, tracker_unseen.groups, W_c, b_c, cand_o, weights=blend_weights)
                trial_lp = _combine_plain_and_twin_lp(trial_lp_p, trial_lp_c, w_p, w_c)
            else:
                trial_lp = trial_lp_p
            acc = float(balanced_accuracy_score(y_unseen, trial_lp.argmax(axis=1)))
            if acc > best_bacc:
                best_bacc = acc
                best_order = cand_o
                best_lp = trial_lp
        except Exception:
            continue

    if best_lp is None:
        best_lp = log_softmax(align_cols(Z_unseen @ W_p.T + b_p, default_labels))
        best_bacc = float(balanced_accuracy_score(y_unseen, best_lp.argmax(axis=1)))
    return best_order, best_lp, max(best_bacc, 0.0)


def _extract_bge_features(
    bge_model: Any,
    bge_tok: Any,
    bge_dev: torch.device,
    texts: List[str],
    heads_sd: dict,
    meta_bge: dict,
    bsize: int = 32,
) -> Tuple[np.ndarray, bool]:
    """
    Extracts BGE-M3 representations:
    - v2 Multi-Layer: Drop-<s> Mean-Pool across tapped layers + per-layer LayerNorm(H) -> D = len(layers)*H.
    - v1 Fallback (legacy mocks / CLS): L2-normalized last_hidden_state[:, 0].
    """
    if len(texts) == 0:
        return np.zeros((0, 1024), dtype=np.float32), False

    raw_layers = meta_bge.get("layers", [])
    has_norms = any(k.startswith("norms.0.") or k.startswith("layer_norms.0.") for k in heads_sd.keys())
    W_p_shape = heads_sd.get("plain.weight").shape if "plain.weight" in heads_sd else None
    is_v2 = (isinstance(raw_layers, list) and len(raw_layers) > 1) or has_norms or (
        W_p_shape is not None and max(W_p_shape) > 1024
    )

    if not is_v2:
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                enc = bge_tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                if hasattr(enc, "to"):
                    enc = enc.to(bge_dev)
                out = bge_model(**enc)
                cls_repr = out.last_hidden_state[:, 0]
                cls_norm = F.normalize(cls_repr, p=2, dim=-1)
                embs.append(cls_norm.cpu().float().numpy())
        return (np.vstack(embs) if embs else np.zeros((0, 1024), dtype=np.float32)), False

    tap_layers = [int(l) for l in raw_layers] if raw_layers else [5, 6, 8, 10, 11, 15, 17, 20]
    max_len = int(meta_bge.get("max_len", 512))
    drop_first = bool(meta_bge.get("drop_first_token", True))
    hidden_sz = int(getattr(getattr(bge_model, "config", None), "hidden_size", 1024))
    norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_sz) for _ in tap_layers]).to(bge_dev).float().eval()
    with torch.no_grad():
        for li, ln in enumerate(norms):
            for pref in (f"norms.{li}", f"layer_norms.{li}"):
                if f"{pref}.weight" in heads_sd:
                    ln.weight.copy_(heads_sd[f"{pref}.weight"].to(device=bge_dev, dtype=torch.float32))
                if f"{pref}.bias" in heads_sd:
                    ln.bias.copy_(heads_sd[f"{pref}.bias"].to(device=bge_dev, dtype=torch.float32))

    dim = len(tap_layers) * hidden_sz
    order = np.argsort([len(str(t)) for t in texts], kind="stable")
    Z_out = np.zeros((len(texts), dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), bsize):
            idx = order[s : s + bsize]
            chunk = [str(texts[i]) for i in idx]
            enc = bge_tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(bge_dev) if hasattr(v, "to") else v for k, v in enc.items()}
            hs = bge_model(**enc, output_hidden_states=True).hidden_states
            m = enc["attention_mask"].unsqueeze(-1).float()
            if drop_first and m.shape[1] > 1:
                m[:, 0, :] = 0.0
            den = m.sum(dim=1).clamp(min=1e-5)
            parts = []
            for li, l_idx in enumerate(tap_layers):
                pooled = (hs[l_idx].float() * m).sum(dim=1) / den
                pooled = torch.nan_to_num(pooled, nan=0.0, posinf=65000.0, neginf=-65000.0)
                parts.append(norms[li](pooled))
            Z_out[idx] = torch.cat(parts, dim=-1).cpu().numpy().astype(np.float32)
    return Z_out, True

def verify_checkpoint_gate(
    Z_tr: np.ndarray,
    W_p: Optional[np.ndarray],
    b_p: Optional[np.ndarray],
    ckpt_labels: list,
    y_sub: np.ndarray,
    name: str,
    min_bacc: float = 0.50
) -> bool:
    """
    Verifies that a checkpoint's plain classification head achieves >= min_bacc
    on the prompt-heldout validation partition (train_sub) before adding its logits.
    Guarantees that corrupt or misaligned heads cannot degrade Main or Comp stack.
    """
    if W_p is None or b_p is None or len(Z_tr) == 0:
        print(f"[Gate] {name}: No plain head or empty Z_tr; skipping head logits.")
        return False
    if W_p.shape[1] != Z_tr.shape[1] or W_p.shape[0] != K:
        print(f"[Gate] {name}: Shape mismatch W_p={W_p.shape} vs Z_tr={Z_tr.shape}; skipping head logits.")
        return False
    logits_tr = align_cols(Z_tr @ W_p.T + b_p, ckpt_labels)
    preds_tr = logits_tr.argmax(axis=1)
    acc = float((preds_tr == y_sub).mean())
    passed = acc >= min_bacc
    print(f"[Gate] {name}: train_sub accuracy = {acc:.4f} (threshold={min_bacc:.2f}) -> {'PASSED' if passed else 'REJECTED'}")
    return passed

def auto_align_and_verify_head(
    Z_tr: np.ndarray,
    W_p: Optional[np.ndarray],
    b_p: Optional[np.ndarray],
    W_c: Optional[np.ndarray],
    b_c: Optional[np.ndarray],
    ckpt_labels: list,
    y_sub: np.ndarray,
    prompts_sub: np.ndarray,
    name: str,
    min_bacc: float = 0.50,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], list, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score

    if len(Z_tr) == 0:
        return False, W_p, b_p, W_c, b_c, ckpt_labels, 0.0

    if W_p is not None and b_p is not None and W_p.shape == (K, Z_tr.shape[1]):
        best_order = ckpt_labels
        best_acc = -1.0
        for _, cand_order in [("canonical", ckpt_labels), ("alphabetical", sorted(LABELS))]:
            try:
                logits_tr = align_cols(Z_tr @ W_p.T + b_p, cand_order)
                acc = float(balanced_accuracy_score(y_sub, logits_tr.argmax(axis=1)))
                if acc > best_acc:
                    best_acc = acc
                    best_order = cand_order
            except Exception:
                continue
        if best_acc >= min_bacc:
            print(f"[Gate] {name}: Pre-trained head PASSED with BAcc = {best_acc:.4f} (threshold={min_bacc:.2f})")
            return True, W_p, b_p, W_c, b_c, best_order, best_acc
        print(f"[Gate] {name}: Pre-trained head BAcc = {best_acc:.4f} < {min_bacc:.2f}; probing 5-fold OOF LogReg readout...")

    if len(y_sub) >= 90 and len(np.unique(prompts_sub)) >= 5:
        scale = float(np.sqrt(Z_tr.shape[1])) / 8.0
        Z_scaled = Z_tr / scale
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(y_sub), dtype=int)
        for tr_i, val_i in gkf.split(Z_scaled, y_sub, groups=prompts_sub):
            clf = LogisticRegression(C=0.20, max_iter=1000, random_state=42)
            clf.fit(Z_scaled[tr_i], y_sub[tr_i])
            oof_preds[val_i] = clf.predict(Z_scaled[val_i])
        oof_bacc = float(balanced_accuracy_score(y_sub, oof_preds))
        if oof_bacc >= min_bacc:
            clf_full = LogisticRegression(C=0.20, max_iter=1000, random_state=42)
            clf_full.fit(Z_scaled, y_sub)
            W_p_new = (clf_full.coef_ / scale).astype(np.float32)
            b_p_new = clf_full.intercept_.astype(np.float32)

            Z_c = Z_scaled.copy()
            for p in np.unique(prompts_sub):
                m = (prompts_sub == p)
                if m.sum() > 1:
                    Z_c[m] -= Z_c[m].mean(axis=0, keepdims=True)
            clf_c = LogisticRegression(C=0.12, max_iter=1000, random_state=42)
            clf_c.fit(Z_c, y_sub)
            W_c_new = (clf_c.coef_ / scale).astype(np.float32)
            b_c_new = clf_c.intercept_.astype(np.float32)
            fit_labels = [LABELS[c] for c in clf_full.classes_]
            print(f"[Gate] {name}: 5-Fold OOF Readout PASSED with BAcc = {oof_bacc:.4f}!")
            return True, W_p_new, b_p_new, W_c_new, b_c_new, fit_labels, oof_bacc

    print(f"[Gate] {name}: REJECTED (failed both pre-trained and 5-fold OOF gates).")
    return False, None, None, None, None, ckpt_labels, 0.0

def _combine_plain_and_twin_lp(
    lp_plain: np.ndarray,
    lp_twin: Optional[np.ndarray],
    w_plain: float,
    w_twin: float,
) -> np.ndarray:
    """
    Combines plain and prompt-centered twin log-probabilities proportionally:
    On rows where lp_twin is active (non-zero), blends with weights (w_plain, w_twin)
    and normalizes via log_softmax. On singleton rows where lp_twin is zero,
    scales by w_plain / (w_plain + w_twin) so downstream multiplication by (w_plain + w_twin)
    yields the exact w_plain * lp_plain contribution.
    """
    lp_p = log_softmax(lp_plain, axis=1)
    if lp_twin is None or w_twin <= 0.0 or len(lp_twin) != len(lp_plain):
        return lp_p
    w_tot = float(w_plain) + float(w_twin)
    if w_tot <= 1e-8:
        return lp_p
    has_twin = np.ptp(lp_twin, axis=1) > 1e-6
    out = np.empty_like(lp_p, dtype=np.float64)
    # Singletons: scale so w_tot * out == w_plain * lp_p
    out[~has_twin] = (float(w_plain) / w_tot) * lp_p[~has_twin]
    # Multi-sample clusters: exact weighted log-probability sum
    if has_twin.any():
        lp_t = log_softmax(lp_twin[has_twin], axis=1)
        out[has_twin] = (float(w_plain) * lp_p[has_twin] + float(w_twin) * lp_t) / w_tot
    return out.astype(np.float32)

def run_submission_pipeline(cfg: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Path]:
    """
    Executes the complete production or smoke-test submission pipeline.
    Returns (submission_dataframe, submission_parquet_path).
    """
    if cfg is None:
        cfg = {}

    start_time = time.time()
    is_smoke = os.environ.get("NTL_SMOKE") == "1"

    # 1. Environment & Paths
    if "working_dir" in cfg:
        working_dir = Path(cfg["working_dir"])
    else:
        working_dir = Path("/kaggle/working")
        if not working_dir.exists():
            working_dir = Path("local_data")
    working_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(cfg.get("input_dir", "/kaggle/input"))
    if not input_dir.exists():
        input_dir = Path(".")

    if "train_path" in cfg and "test_path" in cfg:
        train_path = Path(cfg["train_path"])
        test_path = Path(cfg["test_path"])
        pool_file = Path(cfg["pool_file"]) if "pool_file" in cfg else None
        all_pool_files = [pool_file] if pool_file else resolve_all_pool_files(input_dir)
        if pool_file is None and all_pool_files:
            pool_file = all_pool_files[0]
    else:
        train_path, test_path, pool_file = resolve_dataset_paths(input_dir)
        all_pool_files = resolve_all_pool_files(input_dir)

    enable_ablation_probe = bool(cfg.get("enable_ablation_probe", True)) if cfg else True
    probe_gate_min_bacc = float(cfg.get("probe_gate_min_bacc", 0.40)) if cfg else 0.40

    df_test_raw = pd.read_csv(test_path) if test_path and Path(test_path).exists() else pd.DataFrame({"id": [], "text": []})
    test_ids = df_test_raw["id"] if "id" in df_test_raw.columns else pd.Series(range(len(df_test_raw)))

    # -------------------------------------------------------------------------
    # Decoupled Execution Controls:
    # run_benchmark : Evaluates on clean 1,800 unseen samples (fast experiment evaluation).
    # run_submission: Evaluates on competition test.csv (for official Kaggle leaderboard scoring).
    # -------------------------------------------------------------------------
    run_benchmark = bool(cfg.get("run_benchmark", True)) if cfg else True
    run_submission = bool(cfg.get("run_submission", len(df_test_raw) > 10)) if cfg else (len(df_test_raw) > 10)

    # When competition test.csv has <= 10 dummy rows on Kaggle, auto-disable submission inference to safeguard GPU quota
    if len(df_test_raw) <= 10 and run_benchmark:
        run_submission = False

    mode_label = "DUAL" if (run_benchmark and run_submission) else ("BENCHMARK_ONLY" if run_benchmark else ("SUBMISSION_ONLY" if run_submission else "IDLE"))

    test_texts = df_test_raw["text"].fillna("").astype(str).tolist() if (run_submission and "text" in df_test_raw.columns) else []
    N_test = len(test_texts)

    # Build Zero-Leakage Unseen Holdout Benchmark (N_unseen = 1800 rows default)
    df_unseen = pd.DataFrame()
    unseen_texts: List[str] = []
    N_unseen = 0
    y_unseen = np.array([], dtype=int)
    groups_unseen = np.array([], dtype=int)

    if run_benchmark and not is_smoke:
        df_unseen, holdout_hashes = build_unseen_holdout_dataset(input_dir, train_path, samples_per_class=int(cfg.get("unseen_samples_per_class", 100)))
        if len(df_unseen) > 0:
            unseen_parquet_path = working_dir / "unseen_holdout.parquet"
            df_unseen.to_parquet(unseen_parquet_path, index=False)
            unseen_texts = df_unseen["text"].fillna("").astype(str).tolist()
            N_unseen = len(unseen_texts)
            y_unseen = np.array([LAB2I[m] for m in df_unseen["model"].values], dtype=int) if "model" in df_unseen.columns else np.array([], dtype=int)
            if "cluster_id" in df_unseen.columns:
                groups_unseen = df_unseen["cluster_id"].values
            elif "prompt_id" in df_unseen.columns:
                groups_unseen = df_unseen["prompt_id"].values
            elif "prompt_canon" in df_unseen.columns:
                groups_unseen = (pd.factorize(df_unseen["prompt_canon"])[0] + 1).astype(int)
            elif "prompt_text" in df_unseen.columns:
                groups_unseen = (pd.factorize(df_unseen["prompt_text"])[0] + 1).astype(int)
            else:
                groups_unseen = np.ones(N_unseen, dtype=int)
            print(f"[Pipeline] Loaded clean benchmark dataset: {N_unseen} samples ({len(set(y_unseen))} classes).")
    else:
        if not run_benchmark:
            print("[Pipeline] Frozen Production Mode (run_benchmark=False): Skipping unseen holdout extraction to save GPU time.")

    print("=" * 70)
    print(f"[Pipeline] EXECUTION MODE: {mode_label}")
    print(f"  - run_benchmark  : {run_benchmark} ({N_unseen} unseen samples)")
    print(f"  - run_submission : {run_submission} ({N_test} test rows)")
    print("=" * 70)

    auditor = UnseenStepwiseAuditor(y_unseen=y_unseen, groups_unseen=groups_unseen, N_test=N_test)

    # Discover fine-tuned checkpoints early and honor explicit config checkpoint selection
    ckpts = find_checkpoints(input_dir)
    configured_ckpts = cfg.get("checkpoints")
    if configured_ckpts:
        allowed_ckpts = set(configured_ckpts)
        if float(cfg.get("postprocess", {}).get("claude_specialist_weight", 0.15)) > 0.0:
            allowed_ckpts.add("qwen3_8b_claude")
        ckpts = {k: v for k, v in ckpts.items() if k in allowed_ckpts}
    print(f"[Pipeline] Active checkpoints ({len(ckpts)}): {list(ckpts.keys())}")

    if not is_smoke:
        verified_bases = validate_production_assets(
            input_dir=input_dir,
            train_path=train_path,
            test_path=test_path,
            pool_files=all_pool_files or ([pool_file] if pool_file else []),
            ckpts=ckpts,
            cfg=cfg,
        )
    else:
        verified_bases = {}

    # 2. MODULE 1: Subprocess Dual-Output Lexical Engine
    run_lex_script = working_dir / "run_lexical.py"
    if run_lex_script.exists() and not is_smoke:
        print("[Pipeline] Module 1: Launching subprocess run_lexical.py...")
        subprocess.run([sys.executable, str(run_lex_script)], check=True)

        def _load_optional_npy(p: Path, default=None) -> Optional[np.ndarray]:
            return np.load(p) if p.exists() else default

        log_p_svc_main = np.load(working_dir / "log_p_svc_main.npy")
        p_lex_comp = np.load(working_dir / "p_lex_comp.npy")
        D_tfidf = np.load(working_dir / "D_tfidf.npy")
        log_p_handwriting = _load_optional_npy(working_dir / "log_p_handwriting.npy")
        rule_flags_test = _load_optional_npy(working_dir / "rule_flags_test.npy")
        unseen_log_p_svc_main = _load_optional_npy(working_dir / "unseen_log_p_svc_main.npy", default=np.zeros((0, K), dtype=np.float32))
        unseen_p_lex_comp = _load_optional_npy(working_dir / "unseen_p_lex_comp.npy", default=np.zeros((0, K), dtype=np.float32))
        D_tfidf_u = _load_optional_npy(working_dir / "D_tfidf_unseen.npy")
        log_p_handwriting_unseen = _load_optional_npy(working_dir / "log_p_handwriting_unseen.npy")
        rule_flags_unseen = _load_optional_npy(working_dir / "rule_flags_unseen.npy")

        if (run_benchmark and unseen_log_p_svc_main.shape[0] != N_unseen) or (run_submission and log_p_svc_main.shape[0] != N_test):
            print(f"[Pipeline] Subprocess lexical cache shape mismatch (test: {log_p_svc_main.shape[0]} vs {N_test}, unseen: {unseen_log_p_svc_main.shape[0]} vs {N_unseen}). Falling back to in-process run_lexical_engine...")
            tok_paths = list(input_dir.rglob("vocab.json"))
            qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""
            lex_out = run_lexical_engine(train_path, test_path, all_pool_files or pool_file, qwen_tok_dir, working_dir)
            log_p_svc_main = lex_out["log_p_svc_main"]
            p_lex_comp = lex_out["p_lex_comp"]
            D_tfidf = lex_out["D_tfidf"]
            log_p_handwriting = lex_out.get("log_p_handwriting")
            rule_flags_test = lex_out.get("rule_flags_test")
            unseen_log_p_svc_main = lex_out.get("unseen_log_p_svc_main", np.zeros((0, K), dtype=np.float32))
            unseen_p_lex_comp = lex_out.get("unseen_p_lex_comp", np.zeros((0, K), dtype=np.float32))
            D_tfidf_u = lex_out.get("D_tfidf_unseen")
            log_p_handwriting_unseen = lex_out.get("log_p_handwriting_unseen")
            rule_flags_unseen = lex_out.get("rule_flags_unseen")
    else:
        print("[Pipeline] Module 1: Executing run_lexical_engine directly...")
        tok_paths = list(input_dir.rglob("vocab.json"))
        qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""
        lex_out = run_lexical_engine(train_path, test_path, all_pool_files or pool_file, qwen_tok_dir, working_dir)
        log_p_svc_main = lex_out["log_p_svc_main"]
        p_lex_comp = lex_out["p_lex_comp"]
        D_tfidf = lex_out["D_tfidf"]
        log_p_handwriting = lex_out.get("log_p_handwriting")
        rule_flags_test = lex_out.get("rule_flags_test")
        unseen_log_p_svc_main = lex_out.get("unseen_log_p_svc_main", np.zeros((0, K), dtype=np.float32))
        unseen_p_lex_comp = lex_out.get("unseen_p_lex_comp", np.zeros((0, K), dtype=np.float32))
        D_tfidf_u = lex_out.get("D_tfidf_unseen")
        log_p_handwriting_unseen = lex_out.get("log_p_handwriting_unseen")
        rule_flags_unseen = lex_out.get("rule_flags_unseen")

    print(f"[Pipeline] Module 1 complete: log_p_svc_main shape = {log_p_svc_main.shape}, unseen_log_p_svc_main shape = {unseen_log_p_svc_main.shape}")

    # Unified model specifications and configuration
    models_cfg = cfg.get("models", cfg.get("model_weights", {}))
    model_weights = models_cfg  # Backward compatibility
    post_cfg = cfg.get("postprocess", {})
    main_cfg = cfg.get("main_stack", {})
    comp_cfg = cfg.get("complementary_stack", {})
    main_members = main_cfg.get("members", {})
    comp_streams = comp_cfg.get("stream_weights", {})
    comp_members = comp_cfg.get("lora_members", {})
    se_floors_legacy = comp_cfg.get("se_floors", {})
    prior_scales_legacy = comp_cfg.get("prior_scales", {})
    dp_weights_legacy = comp_cfg.get("dp_weights", {})

    def get_model_weight(model_key: str, default: float = 0.0) -> float:
        m_entry = models_cfg.get(model_key, None)
        if m_entry is None:
            if model_key == "svc_4view":
                return float(main_cfg.get("linear_svc_weight", default))
            if model_key == "lexical_5v":
                return float(comp_streams.get("lexical", default))
            return float(default)
        if isinstance(m_entry, (int, float)):
            return float(m_entry)
        if isinstance(m_entry, dict):
            if "weight" in m_entry:
                return float(m_entry["weight"])
            p_w = float(m_entry.get("plain_weight", 0.0))
            t_w = float(m_entry.get("twin_weight", 0.0))
            if p_w + t_w > 0.0:
                return p_w + t_w
        return float(default)

    def get_model_twin_ratio(model_key: str, default: float = 0.25) -> float:
        m_entry = models_cfg.get(model_key, {})
        if isinstance(m_entry, dict):
            if "twin_ratio" in m_entry:
                return float(m_entry["twin_ratio"])
            p_w = float(m_entry.get("plain_weight", 0.0))
            t_w = float(m_entry.get("twin_weight", 0.0))
            if p_w + t_w > 0.0:
                return t_w / (p_w + t_w)
        return float(default)

    def get_model_spec(model_key: str, param: str, default: Any = 0.0) -> Any:
        m_entry = models_cfg.get(model_key, {})
        if isinstance(m_entry, dict) and param in m_entry:
            return m_entry[param]
        if isinstance(m_entry, (int, float)) and param in ("plain_weight", "weight"):
            return float(m_entry)
        if param == "dp_weight" and model_key in dp_weights_legacy:
            return dp_weights_legacy[model_key]
        if param == "se_floor" and model_key in se_floors_legacy:
            return se_floors_legacy[model_key]
        if param == "prior_scale" and model_key in prior_scales_legacy:
            return prior_scales_legacy[model_key]
        return default

    default_se = {
        "qwen3_14b_p2": 0.10, "qwen25_14b_p2": 0.10, "deberta_p2": 0.10,
        "qwen3_14b_p1": 0.35, "instyle": 0.35, "qwen3_8b_claude": 0.35
    }
    default_dp = {
        "q3_14b": 0.38, "q25_14b": 0.24, "deberta": 0.20, "instyle": 0.10, "part1_14b": 0.08
    }
    all_known_keys = set(list(ckpts.keys()) + list(models_cfg.keys()) + list(default_se.keys()) + list(PRIOR_SCALES.keys()))
    se_floors = {k: get_model_spec(k, "se_floor", default_se.get(k, 0.10)) for k in all_known_keys}
    prior_scales = {k: get_model_spec(k, "prior_scale", PRIOR_SCALES.get(k, 0.50)) for k in all_known_keys}
    dp_weights = {k: get_model_spec(k, "dp_weight", default_dp.get(k, 0.10)) for k in all_known_keys}

    twin_res = comp_cfg.get("twin_resolutions", {"t85": 0.50, "t72": 0.30, "t60": 0.20})
    w85 = float(twin_res.get("t85", 0.50))
    w72 = float(twin_res.get("t72", 0.30))
    w60 = float(twin_res.get("t60", 0.20))
    max_cl = post_cfg.get("max_cluster_size", 14)

    svc_weight = get_model_weight("svc_4view", default=0.30)
    lex_comp_weight = get_model_weight("lexical_5v", default=0.10)

    u_lp_svc = unseen_log_p_svc_main if (run_benchmark and N_unseen > 0 and len(unseen_log_p_svc_main) == N_unseen) else np.zeros((N_unseen, K), dtype=np.float32)
    te_lp_svc = log_p_svc_main if (run_submission and N_test > 0 and len(log_p_svc_main) == N_test) else np.zeros((N_test, K), dtype=np.float32)
    auditor.add_model(
        name="LinearSVC (4-View)",
        weight=svc_weight,
        unseen_log_probs=u_lp_svc,
        test_log_probs=te_lp_svc,
    )
    u_p_lex = unseen_p_lex_comp if (run_benchmark and N_unseen > 0 and len(unseen_p_lex_comp) == N_unseen) else (np.ones((N_unseen, K), dtype=np.float32) / float(K))
    te_p_lex = p_lex_comp if (run_submission and N_test > 0 and len(p_lex_comp) == N_test) else (np.ones((N_test, K), dtype=np.float32) / float(K))
    auditor.add_model(
        name="Lexical Comp (5-View)",
        weight=lex_comp_weight,
        unseen_log_probs=np.log(np.clip(u_p_lex, 1e-9, 1.0)),
        test_log_probs=np.log(np.clip(te_p_lex, 1e-9, 1.0)),
    )
    claude_spec_logits = None
    claude_spec_logits_u = None
    claude_classes = None
    features_unseen_dict: Dict[str, np.ndarray] = {}

    def compute_multires_clusters(
        D_mat: np.ndarray,
        thresholds: Tuple[float, ...] = (0.85, 0.72, 0.65, 0.60),
        max_cluster_sz: int = 14
    ) -> Dict[float, np.ndarray]:
        n_samples = D_mat.shape[0] if (D_mat is not None and D_mat.ndim == 2) else 0
        if n_samples < 2:
            return {t: np.ones(max(n_samples, 1), dtype=int) for t in thresholds}
        cond_d = squareform(D_mat, checks=False)
        Z_link = linkage(cond_d, method="average")
        return {
            t: split_oversized_clusters_dist(D_mat, fcluster(Z_link, t=t, criterion="distance"), max_cluster_size=max_cluster_sz)
            for t in thresholds
        }

    class HybridDistanceTracker:
        """Symmetric multi-resolution distance and topic cluster tracker for test and unseen benchmark."""
        def __init__(self, D_init: Optional[np.ndarray], texts: Optional[List[str]] = None, max_cluster_sz: int = 14):
            self.max_cluster_size = max_cluster_sz
            if D_init is not None and D_init.ndim == 2 and D_init.shape[0] >= 2:
                self.D_tfidf = D_init.copy()
            elif texts is not None and len(texts) >= 2:
                Xt = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100000, dtype=np.float32).fit_transform(texts)
                d = np.clip(1.0 - (Xt @ Xt.T).toarray(), 0.0, None)
                np.fill_diagonal(d, 0.0)
                self.D_tfidf = d
            elif texts is not None and len(texts) == 1:
                self.D_tfidf = np.zeros((1, 1), dtype=np.float32)
            else:
                self.D_tfidf = np.zeros((0, 0), dtype=np.float32)

            self.N = self.D_tfidf.shape[0]
            self.D_neural_sum = np.zeros((self.N, self.N), dtype=np.float64) if self.N > 0 else np.zeros((0, 0), dtype=np.float64)
            self.D_neural_weight_total = 0.0
            self.D_hybrid = self.D_tfidf.copy()
            self.groups = compute_multires_clusters(self.D_hybrid, max_cluster_sz=max_cluster_sz)

        def update(self, Z_mat: np.ndarray, ckpt_scale: float) -> None:
            if self.N < 2 or Z_mat is None or len(Z_mat) != self.N:
                return
            Z_c = Z_mat - Z_mat.mean(axis=0, keepdims=True)
            Z_norm = Z_c / np.linalg.norm(Z_c, axis=1, keepdims=True).clip(min=1e-6)
            D_cos = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
            np.fill_diagonal(D_cos, 0.0)
            w_d = float(ckpt_scale)
            self.D_neural_sum += w_d * D_cos
            self.D_neural_weight_total += w_d
            D_neural_avg = self.D_neural_sum / max(1e-6, self.D_neural_weight_total)
            self.D_hybrid = 0.55 * self.D_tfidf + 0.45 * D_neural_avg
            self.groups = compute_multires_clusters(self.D_hybrid, max_cluster_sz=self.max_cluster_size)

    def compute_multires_centered_blend(
        Z: np.ndarray,
        groups_map: Dict[float, np.ndarray],
        W_c: Optional[np.ndarray],
        b_c: Optional[np.ndarray],
        labels: list,
        weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
    ) -> np.ndarray:
        if W_c is None or b_c is None or len(Z) == 0:
            return np.zeros((len(Z), K), dtype=np.float32)
        w85, w72, w60 = weights
        lp85, _ = compute_centered_logits(Z, groups_map.get(0.85, np.ones(len(Z), dtype=int)), W_c, b_c, labels)
        lp72, _ = compute_centered_logits(Z, groups_map.get(0.72, np.ones(len(Z), dtype=int)), W_c, b_c, labels)
        lp60, _ = compute_centered_logits(Z, groups_map.get(0.60, np.ones(len(Z), dtype=int)), W_c, b_c, labels)
        return (w85 * lp85 + w72 * lp72 + w60 * lp60).astype(np.float32)

    tracker_test = HybridDistanceTracker(D_tfidf, texts=test_texts, max_cluster_sz=max_cl) if (run_submission and N_test > 0) else None
    tracker_unseen = HybridDistanceTracker(D_tfidf_u if ('D_tfidf_u' in locals() and D_tfidf_u is not None) else None, texts=unseen_texts, max_cluster_sz=max_cl) if (run_benchmark and N_unseen > 0) else None

    # Backward-compatible references (using active tracker)
    active_tracker = tracker_test if tracker_test is not None else tracker_unseen
    D_neural_sum = active_tracker.D_neural_sum if active_tracker else np.zeros((0, 0), dtype=np.float32)
    D_neural_weight_total = active_tracker.D_neural_weight_total if active_tracker else 0.0
    D_hybrid = active_tracker.D_hybrid if active_tracker else np.zeros((0, 0), dtype=np.float32)
    groups_085 = active_tracker.groups[0.85] if active_tracker else np.zeros(0, dtype=int)
    groups_072 = active_tracker.groups[0.72] if active_tracker else np.zeros(0, dtype=int)
    groups_065 = active_tracker.groups[0.65] if active_tracker else np.zeros(0, dtype=int)
    groups_060 = active_tracker.groups[0.60] if active_tracker else np.zeros(0, dtype=int)

    def update_hybrid_groups(Z_te_mat: Optional[np.ndarray], ckpt_scale: float, Z_u_mat: Optional[np.ndarray] = None):
        nonlocal D_hybrid, groups_085, groups_072, groups_065, groups_060, D_neural_weight_total
        if tracker_test is not None and Z_te_mat is not None and len(Z_te_mat) > 0:
            tracker_test.update(Z_te_mat, ckpt_scale)
            D_hybrid = tracker_test.D_hybrid
            D_neural_weight_total = tracker_test.D_neural_weight_total
            groups_085 = tracker_test.groups[0.85]
            groups_072 = tracker_test.groups[0.72]
            groups_065 = tracker_test.groups[0.65]
            groups_060 = tracker_test.groups[0.60]
        if tracker_unseen is not None and Z_u_mat is not None and len(Z_u_mat) > 0:
            tracker_unseen.update(Z_u_mat, ckpt_scale)
            if tracker_test is None:
                D_hybrid = tracker_unseen.D_hybrid
                D_neural_weight_total = tracker_unseen.D_neural_weight_total
                groups_085 = tracker_unseen.groups[0.85]
                groups_072 = tracker_unseen.groups[0.72]
                groups_065 = tracker_unseen.groups[0.65]
                groups_060 = tracker_unseen.groups[0.60]

    has_neural = len(ckpts) > 0 and not is_smoke
    spectral_blocks_tr: List[np.ndarray] = []
    spectral_blocks_te: List[np.ndarray] = []
    spectral_blocks_u: List[np.ndarray] = []
    dp_posteriors: List[Tuple[float, np.ndarray]] = []
    dp_posteriors_u: List[Tuple[float, np.ndarray]] = []
    P_spectral = None
    P_spectral_u = None

    df_train = pd.read_csv(train_path) if train_path and Path(train_path).exists() else None
    extract_train_wpca = comp_cfg.get("extract_train_wpca", False)
    train_sub_texts: List[str] = []
    y_sub: np.ndarray = np.array([], dtype=int)
    prompts_sub: np.ndarray = np.array([])
    N_tr_sub: int = 0

    if has_neural:
        import torch

        if extract_train_wpca and df_train is not None:
            rng = np.random.RandomState(42)
            u_p = df_train["prompt_id"].unique()
            sel_prompts = rng.choice(u_p, size=min(90, len(u_p)), replace=False)
            sub_idx = df_train[df_train["prompt_id"].isin(sel_prompts)].index.values
            train_sub = df_train.loc[sub_idx].reset_index(drop=True)
            train_sub_texts = train_sub["text"].fillna("").astype(str).tolist()
            y_sub = np.array([LAB2I[m] for m in train_sub["model"]])
            prompts_sub = train_sub["prompt_id"].values
            N_tr_sub = len(train_sub)
            print(f"[Pipeline] Selected {N_tr_sub} training rows ({len(sel_prompts)} prompts) for wPCA")

        # Discover pre-trained wPCA bundle
        wpca_bundle = None
        wpca_candidates = [
            working_dir / "wpca_bundle.safetensors",
            Path("local_data/statistical_heads/wpca_bundle.safetensors"),
            Path("/kaggle/input/ntl-statistical-heads/wpca_bundle.safetensors"),
        ]
        if Path("/kaggle/input").exists():
            for p in Path("/kaggle/input").rglob("wpca_bundle.safetensors"):
                if p not in wpca_candidates:
                    wpca_candidates.append(p)
        for cand in wpca_candidates:
            if cand.exists():
                try:
                    wpca_bundle = load_wpca_bundle(cand)
                    print(f"[Pipeline] Loaded pre-trained wPCA bundle from {cand}")
                    break
                except Exception as e:
                    print(f"[Pipeline] Note: Could not load candidate wPCA bundle {cand}: {e}")

        # Discover pre-trained Spectral Bank params
        spectral_params = None
        spectral_candidates = [
            working_dir / "spectral_heads.safetensors",
            Path("local_data/statistical_heads/spectral_heads.safetensors"),
            Path("/kaggle/input/ntl-statistical-heads/spectral_heads.safetensors"),
        ]
        if Path("/kaggle/input").exists():
            for p in Path("/kaggle/input").rglob("spectral_heads.safetensors"):
                if p not in spectral_candidates:
                    spectral_candidates.append(p)
        for cand in spectral_candidates:
            if cand.exists():
                try:
                    spectral_params = load_spectral_params(cand)
                    print(f"[Pipeline] Loaded pre-trained Spectral Bank params from {cand}")
                    break
                except Exception as e:
                    print(f"[Pipeline] Note: Could not load candidate Spectral Bank params {cand}: {e}")

        wpca_bundle_export: Dict[str, dict] = {}

        def _extract_trunk(trunk_obj, end_id=None):
            if run_submission and N_test > 0 and run_benchmark and N_unseen > 0:
                Z_all = trunk_obj.extract_exact(test_texts + unseen_texts, batch_size=16, end_id=end_id)
                Z_te = Z_all[:N_test]
                Z_u = Z_all[N_test : N_test + N_unseen]
            elif run_submission and N_test > 0:
                Z_te = trunk_obj.extract_exact(test_texts, batch_size=16, end_id=end_id)
                Z_u = np.zeros((0, Z_te.shape[1]), dtype=np.float32)
            elif run_benchmark and N_unseen > 0:
                Z_u = trunk_obj.extract_exact(unseen_texts, batch_size=16, end_id=end_id)
                Z_te = np.zeros((0, Z_u.shape[1]), dtype=np.float32)
            else:
                Z_te = np.zeros((0, 4096), dtype=np.float32)
                Z_u = np.zeros((0, 4096), dtype=np.float32)
            dim = Z_u.shape[1] if len(Z_u) > 0 else (Z_te.shape[1] if len(Z_te) > 0 else 4096)
            Z_tr = np.zeros((0, dim), dtype=np.float32)
            return Z_te, Z_u, Z_tr

        use_stat_heads = (
            (not models_cfg)
            or get_model_weight("wpca_dp", 0.0) > 0.0
            or get_model_weight("spectral_bank", 0.0) > 0.0
        )

        def apply_wpca_and_spectral(
            model_key: str,
            Z_te_mat: np.ndarray,
            Z_tr_mat: np.ndarray,
            Z_u_mat: Optional[np.ndarray],
            dp_w: float,
            prior_scale: float,
            d_target: int = 256,
            se_floor: float = 0.10,
        ) -> None:
            if not use_stat_heads:
                return
            res = None
            if wpca_bundle is not None and model_key in wpca_bundle:
                res = wpca_bundle[model_key]
            elif len(Z_tr_mat) > 0:
                res = fit_shared_gaussian_lowram(Z_tr_mat, y_sub, prompts_sub, d_target=d_target, se_floor=se_floor)
                wpca_bundle_export[model_key] = res

            if res is not None:
                if run_submission and len(Z_te_mat) > 0 and tracker_test is not None:
                    U_te = transform_lowram(Z_te_mat, res)
                    lp_dp = scores_exact_dp(U_te, tracker_test.groups[0.65], res, tracker_test.D_hybrid)
                    dp_posteriors.append((dp_w, softmax(lp_dp, axis=1)))
                    spectral_blocks_te.append(project_spectral_block(Z_te_mat, res, prior_scale))
                    del U_te

                if len(Z_tr_mat) > 0:
                    spectral_blocks_tr.append(project_spectral_block(Z_tr_mat, res, prior_scale))

                if run_benchmark and Z_u_mat is not None and len(Z_u_mat) > 0 and tracker_unseen is not None:
                    U_u = transform_lowram(Z_u_mat, res)
                    lp_dp_u = scores_exact_dp(U_u, tracker_unseen.groups[0.65], res, tracker_unseen.D_hybrid)
                    dp_posteriors_u.append((dp_w, softmax(lp_dp_u, axis=1)))
                    spectral_blocks_u.append(project_spectral_block(Z_u_mat, res, prior_scale))
                    del U_u
                del res

        def _discover_fold_dirs(ckpt_key: str, fallback_dir: Path) -> List[Path]:
            prefix = "q14v5" if "14b_v5" in ckpt_key else "ts_q3b14" if "14b_ts" in ckpt_key else "q8v5" if "8b_v5" in ckpt_key else "ts_q3b8" if "8b_ts" in ckpt_key else ckpt_key
            found = find_fold_adapters(input_dir, prefix)
            def _is_fold(p: Path) -> bool:
                nm = p.name.lower()
                return any(f"_f{i}" in nm or f"-f{i}" in nm or f"_fold{i}" in nm for i in range(10))
            folds_only = sorted([p for p in found if _is_fold(p)], key=lambda x: str(x).lower())
            full_only = [p for p in found if not _is_fold(p)] or [fallback_dir]
            return folds_only + full_only

        def _register_checkpoint_outputs(
            model_key: str,
            disp_name: str,
            Z_te: np.ndarray,
            Z_u: np.ndarray,
            Z_tr: np.ndarray,
            W_p: Optional[np.ndarray],
            b_p: Optional[np.ndarray],
            W_c: Optional[np.ndarray],
            b_c: Optional[np.ndarray],
            ckpt_labels: list,
            w_tot: float,
            dp_key: Optional[str] = None,
            dp_default: float = 0.10,
            prior_default: float = 0.50,
            se_default: float = 0.25,
            auto_align: bool = True,
        ) -> None:
            twin_ratio = get_model_twin_ratio(model_key, default=0.25)
            w_p = w_tot * (1.0 - twin_ratio)
            w_c = w_tot * twin_ratio

            if N_unseen > 0 and len(Z_u) == N_unseen:
                features_unseen_dict[model_key] = Z_u.astype(np.float16)

            eff_labels = list(ckpt_labels)
            lp_u = np.zeros((0, K), dtype=np.float32)
            gate_passed = True

            if W_p is not None and run_benchmark and N_unseen > 0 and len(Z_u) == N_unseen:
                if auto_align:
                    eff_labels, lp_u, solo_bacc = _auto_align_labels_on_unseen(
                        Z_u, W_p, b_p, W_c, b_c, ckpt_labels,
                        tracker_unseen, y_unseen, w_p, w_c, (w85, w72, w60), compute_multires_centered_blend,
                    )
                    align_tag = "canonical" if eff_labels == list(LABELS) else "alphabetical" if eff_labels == sorted(LABELS) else "custom"
                    print(f"[Pipeline] {model_key}: solo unseen BAcc = {solo_bacc:.4f} (alignment: {align_tag})")
                    if solo_bacc <= 1.0 / float(K):
                        print(f"[Pipeline] {model_key}: degenerate head (solo={solo_bacc:.4f} <= chance) -> excluding from active blend.")
                        gate_passed = False
                else:
                    lp_u_p = log_softmax(align_cols(Z_u @ W_p.T + b_p, eff_labels))
                    lp_u_c = compute_multires_centered_blend(Z_u, tracker_unseen.groups, W_c, b_c, eff_labels, weights=(w85, w72, w60)) if (W_c is not None and tracker_unseen is not None) else None
                    lp_u = _combine_plain_and_twin_lp(lp_u_p, lp_u_c, w_p, w_c)

            pr_sc = prior_scales.get(model_key, prior_default)
            if gate_passed:
                update_hybrid_groups(Z_te, pr_sc, Z_u if (N_unseen > 0 and len(Z_u) == N_unseen) else None)
                dp_w = dp_weights.get(dp_key or model_key, dp_default)
                apply_wpca_and_spectral(
                    model_key=model_key,
                    Z_te_mat=Z_te,
                    Z_tr_mat=Z_tr,
                    Z_u_mat=Z_u if (N_unseen > 0 and len(Z_u) == N_unseen) else None,
                    dp_w=dp_w,
                    prior_scale=pr_sc,
                    d_target=256,
                    se_floor=se_floors.get(model_key, se_default),
                )
                if W_p is not None:
                    lp_m_te = np.zeros((0, K), dtype=np.float32)
                    if run_submission and len(Z_te) == N_test and N_test > 0:
                        lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, eff_labels))
                        lp_c_te = compute_multires_centered_blend(Z_te, tracker_test.groups, W_c, b_c, eff_labels, weights=(w85, w72, w60)) if (W_c is not None and tracker_test is not None) else None
                        lp_m_te = _combine_plain_and_twin_lp(lp_p, lp_c_te, w_p, w_c)
                    lp_m_u = lp_u if (run_benchmark and N_unseen > 0 and len(lp_u) == N_unseen) else np.zeros((0, K), dtype=np.float32)
                    auditor.add_model(disp_name, weight=w_tot, unseen_log_probs=lp_m_u, test_log_probs=lp_m_te)
            elif W_p is not None and len(lp_u) == N_unseen:
                auditor.add_model(disp_name, weight=0.0, unseen_log_probs=lp_u, test_log_probs=np.zeros((N_test, K), dtype=np.float32))

        def _extract_shared_lora_ckpt(trunk_obj: Any, ckpt_key: str, ckpt_dir: Path, disp_name: str, default_keep: int, default_layers: List[int], weight_def: float, dp_key: Optional[str] = None, dp_def: float = 0.10, prior_def: float = 0.50, se_def: float = 0.25, allow_multifold: bool = False, auto_align: bool = True, required: bool = False) -> None:
            fold_dirs = _discover_fold_dirs(ckpt_key, ckpt_dir) if allow_multifold else [ckpt_dir]
            target_dir = fold_dirs[-1]
            meta = read_checkpoint_meta(target_dir, default_keep=default_keep, default_layers=default_layers, default_max_len=447)

            if allow_multifold and len(fold_dirs) > 1:
                w_tot = get_model_weight(ckpt_key, default=weight_def)
                twin_ratio = get_model_twin_ratio(ckpt_key, default=0.25)
                w_p, w_c = w_tot * (1.0 - twin_ratio), w_tot * twin_ratio
                P_p_acc = np.zeros((N_test, K), dtype=np.float64)
                P_c_acc = np.zeros((N_test, K), dtype=np.float64)
                P_u_p_acc = np.zeros((N_unseen, K), dtype=np.float64)
                P_u_c_acc = np.zeros((N_unseen, K), dtype=np.float64)
                has_any_c = False
                Z_te_last = np.zeros((0, 4096), dtype=np.float32)
                Z_u_last = np.zeros((0, 4096), dtype=np.float32)
                Z_tr_last = np.zeros((0, 4096), dtype=np.float32)
                print(f"[Pipeline] {ckpt_key}: extracting & averaging across {len(fold_dirs)} fold adapters...")
                for f_idx, f_dir in enumerate(fold_dirs):
                    is_last = f_idx == len(fold_dirs) - 1
                    f_meta = read_checkpoint_meta(f_dir, default_keep=default_keep, default_layers=default_layers, default_max_len=447)
                    trunk_obj.set_active_layers(f_meta["layers"], f_meta["max_len"])
                    trunk_obj.set_final_norm(not f_meta["no_final_norm"])
                    f_backup = trunk_obj.snapshot_weights(f_dir)
                    if not merge_lora(trunk_obj.model, f_dir, sign=+1.0):
                        trunk_obj.restore_weights(f_backup)
                        continue
                    f_heads = load_head_file(f_dir)
                    trunk_obj.load_norm_weights(f_heads)
                    f_Wp, f_bp, f_Wc, f_bc = resolve_head_tensors(f_heads)
                    f_lbls = f_meta["labels"]
                    f_Z_te, f_Z_u, f_Z_tr = _extract_trunk(trunk_obj, end_id=f_meta.get("end_id"))
                    if is_last:
                        Z_te_last, Z_u_last, Z_tr_last = f_Z_te, f_Z_u, f_Z_tr
                        if N_unseen > 0 and len(f_Z_u) == N_unseen:
                            features_unseen_dict[ckpt_key] = f_Z_u.astype(np.float16)
                    if f_idx == 0 or is_last:
                        update_hybrid_groups(f_Z_te, prior_scales.get(ckpt_key, prior_def), f_Z_u if N_unseen > 0 else None)
                    if run_submission and len(f_Z_te) > 0:
                        if f_Wp is not None:
                            P_p_acc += softmax(align_cols(f_Z_te @ f_Wp.T + f_bp, f_lbls), axis=1) / len(fold_dirs)
                        if f_Wc is not None and tracker_test is not None:
                            has_any_c = True
                            P_c_acc += softmax(compute_multires_centered_blend(f_Z_te, tracker_test.groups, f_Wc, f_bc, f_lbls, weights=(w85, w72, w60)), axis=1) / len(fold_dirs)
                    if run_benchmark and len(f_Z_u) > 0:
                        if f_Wp is not None:
                            P_u_p_acc += softmax(align_cols(f_Z_u @ f_Wp.T + f_bp, f_lbls), axis=1) / len(fold_dirs)
                        if f_Wc is not None and tracker_unseen is not None:
                            has_any_c = True
                            P_u_c_acc += softmax(compute_multires_centered_blend(f_Z_u, tracker_unseen.groups, f_Wc, f_bc, f_lbls, weights=(w85, w72, w60)), axis=1) / len(fold_dirs)
                    trunk_obj.restore_weights(f_backup)
                    del f_backup, f_heads
                trunk_obj.set_final_norm(False)
                apply_wpca_and_spectral(ckpt_key, Z_te_last, Z_tr_last, Z_u_last if N_unseen > 0 else None, dp_weights.get(dp_key or ckpt_key, dp_def), prior_scales.get(ckpt_key, prior_def), se_floor=se_floors.get(ckpt_key, se_def))
                lp_m_te = _combine_plain_and_twin_lp(np.log(np.clip(P_p_acc, 1e-9, 1.0)), np.log(np.clip(P_c_acc, 1e-9, 1.0)) if has_any_c else None, w_p, w_c) if (run_submission and N_test > 0) else np.zeros((0, K), dtype=np.float32)
                lp_m_u = _combine_plain_and_twin_lp(np.log(np.clip(P_u_p_acc, 1e-9, 1.0)), np.log(np.clip(P_u_c_acc, 1e-9, 1.0)) if has_any_c else None, w_p, w_c) if (run_benchmark and N_unseen > 0) else np.zeros((0, K), dtype=np.float32)
                auditor.add_model(disp_name, weight=w_tot, unseen_log_probs=lp_m_u, test_log_probs=lp_m_te)
                return

            trunk_obj.set_active_layers(meta["layers"], meta["max_len"])
            trunk_obj.set_final_norm(not meta["no_final_norm"])
            backup = trunk_obj.snapshot_weights(target_dir)
            merged = merge_lora(trunk_obj.model, target_dir, sign=+1.0)
            if not merged:
                trunk_obj.restore_weights(backup)
                trunk_obj.set_final_norm(False)
                if required and not is_smoke:
                    raise RuntimeError(f"Failed to merge required LoRA adapter at {target_dir}")
                print(f"[Pipeline] Optional checkpoint {ckpt_key} failed to merge LoRA; skipping.")
                return
            heads_sd = load_head_file(target_dir)
            trunk_obj.load_norm_weights(heads_sd)
            W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
            if required and W_p is None and not is_smoke:
                raise RuntimeError(f"Required classification head W_p is None for {target_dir}")
            Z_te, Z_u, Z_tr = _extract_trunk(trunk_obj, end_id=meta.get("end_id"))
            _register_checkpoint_outputs(
                model_key=ckpt_key, disp_name=disp_name,
                Z_te=Z_te, Z_u=Z_u, Z_tr=Z_tr, W_p=W_p, b_p=b_p, W_c=W_c, b_c=b_c,
                ckpt_labels=meta["labels"], w_tot=get_model_weight(ckpt_key, default=weight_def),
                dp_key=dp_key, dp_default=dp_def, prior_default=prior_def, se_default=se_def, auto_align=auto_align,
            )
            trunk_obj.restore_weights(backup)
            trunk_obj.set_final_norm(False)
            del backup, heads_sd, Z_te, Z_u, Z_tr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 1: Qwen3-14B (40 layers, hidden=5120)
        # ==============================================================================
        base_q3_14b = verified_bases.get("qwen3_14b") or find_base_model(input_dir, "qwen3", 5120, 40)
        q3_14b_specs = [
            ("qwen3_14b_p2", "Qwen3-14B Part-2", 27, [2, 4, 6, 9, 13, 17, 22, 27], 0.54, "q3_14b", 0.38, 1.00, 0.10, False, True),
            ("qwen3_14b_inhouse", "qwen3_14b_inhouse", 24, [2, 4, 6, 8, 12, 16, 20, 24], 0.45, None, 0.10, 0.50, 0.25, True, False),
            ("qwen3_14b_ts", "qwen3_14b_ts", 24, [2, 4, 6, 8, 12, 16, 20, 24], 0.45, "q3_14b_ts", 0.10, 0.50, 0.25, True, False),
            ("qwen3_14b_v5", "qwen3_14b_v5", 24, [2, 4, 6, 8, 12, 16, 20, 24], 0.45, "q3_14b_ts", 0.10, 0.50, 0.25, True, False),
            ("qwen3_14b_p1", "Qwen3-14B Part-1", 24, [2, 4, 6, 9, 13, 17, 21, 24], 0.16, "part1_14b", 0.08, 0.50, 0.35, True, False),
        ]
        if base_q3_14b and any(k in ckpts for k, *_ in q3_14b_specs):
            print(f"[Pipeline] Loading Trunk 1: Qwen3-14B from {base_q3_14b}...")
            trunk1 = UniversalLoRATrunkLoader(base_q3_14b, keep_layers=27, layers=[2, 4, 6, 9, 13, 17, 22, 27], max_len=447)
            for k, disp, k_def, l_def, w_def, dp_k, dp_d, pr_d, se_d, align_f, req_f in q3_14b_specs:
                if k in ckpts:
                    _extract_shared_lora_ckpt(trunk1, k, ckpts[k], disp, k_def, l_def, w_def, dp_k, dp_d, pr_d, se_d, allow_multifold=False, auto_align=align_f, required=req_f)
            trunk1.free_memory()
            del trunk1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 2: Qwen2.5-14B (48 layers, hidden=5120)
        # ==============================================================================
        base_q25_14b = verified_bases.get("qwen25_14b") or find_base_model(input_dir, "qwen2", 5120, 48)
        q25_specs = [
            ("qwen25_14b_p2", "Qwen2.5-14B Part-2", True),
            ("qwen25_14b_inhouse", "Qwen2.5-14B In-House", False),
        ]
        if base_q25_14b and any(k in ckpts for k, _, _ in q25_specs):
            print(f"[Pipeline] Loading Trunk 2: Qwen2.5-14B from {base_q25_14b}...")
            trunk2 = UniversalLoRATrunkLoader(base_q25_14b, keep_layers=32, layers=[3, 5, 8, 11, 16, 21, 27, 32], max_len=447)
            for k, disp, align_f in q25_specs:
                if k in ckpts:
                    _extract_shared_lora_ckpt(trunk2, k, ckpts[k], disp, 32, [3, 5, 8, 11, 16, 21, 27, 32], 0.34, "q25_14b", 0.24, 0.85, 0.10, allow_multifold=False, auto_align=align_f, required=True)
            trunk2.free_memory()
            del trunk2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 3: Qwen3-8B (36 layers, hidden=4096)
        # ==============================================================================
        base_q3_8b = verified_bases.get("qwen3_8b") or find_base_model(input_dir, "qwen3", 4096, 36)
        q3_8b_specs = [
            ("instyle", "Qwen3-8B In-Style", 0.22, "instyle", 0.10, 0.45, 0.35, False),
            ("qwen3_8b_ts", "qwen3_8b_ts", 0.22, "q3_8b_ts", 0.10, 0.40, 0.35, True),
            ("qwen3_8b_v5", "qwen3_8b_v5", 0.22, "q3_8b_ts", 0.10, 0.40, 0.35, True),
            ("qwen3_8b_inhouse", "qwen3_8b_inhouse", 0.22, "q3_8b_ts", 0.10, 0.35, 0.40, True),
        ]
        if base_q3_8b and (any(k in ckpts for k, *_ in q3_8b_specs) or "qwen3_8b_claude" in ckpts):
            print(f"[Pipeline] Loading Trunk 3: Qwen3-8B from {base_q3_8b}...")
            trunk3 = UniversalLoRATrunkLoader(base_q3_8b, keep_layers=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], max_len=447)
            for k, disp, w_def, dp_k, dp_d, pr_d, se_d, mf_f in q3_8b_specs:
                if k in ckpts:
                    _extract_shared_lora_ckpt(trunk3, k, ckpts[k], disp, 24, [2, 4, 6, 8, 12, 16, 20, 24], w_def, dp_k, dp_d, pr_d, se_d, allow_multifold=mf_f, auto_align=True, required=False)

            # Claude Specialist Gate LoRA
            if "qwen3_8b_claude" in ckpts:
                claude_dir = ckpts["qwen3_8b_claude"]
                meta_cl = read_checkpoint_meta(claude_dir, default_keep=24, default_layers=[2, 4, 6, 8, 12, 16, 20, 24], default_max_len=447)
                trunk3.set_active_layers(meta_cl["layers"], meta_cl["max_len"])
                if not merge_lora(trunk3.model, claude_dir, sign=+1.0) and not is_smoke:
                    raise RuntimeError(f"Failed to merge required LoRA adapter at {claude_dir}")
                heads_sd_cl = load_head_file(claude_dir)
                if heads_sd_cl:
                    trunk3.load_norm_weights(heads_sd_cl)
                W_p, b_p, W_c, b_c = None, None, None, None
                W_spec, b_spec, ro_labels = None, None, None
                readouts_file = claude_dir / "readouts.npz"
                if readouts_file.exists():
                    with np.load(str(readouts_file), allow_pickle=True) as rnpz:
                        for lk in ("classes", "labels"):
                            if lk in rnpz:
                                ro_labels = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in rnpz[lk]]
                                break
                        for wk, bk in [("plain_coef", "plain_intercept"), ("plain.weight", "plain.bias")]:
                            if wk in rnpz and bk in rnpz and rnpz[wk].shape[0] == 18:
                                W_p, b_p = np.array(rnpz[wk]), np.array(rnpz[bk])
                                break
                        for wk, bk in [("center_coef", "center_intercept"), ("center.weight", "center.bias")]:
                            if wk in rnpz and bk in rnpz and rnpz[wk].shape[0] == 18:
                                W_c, b_c = np.array(rnpz[wk]), np.array(rnpz[bk])
                                break
                        if "claude_coef" in rnpz and "claude_intercept" in rnpz and rnpz["claude_coef"].shape[0] == 5:
                            W_spec, b_spec = np.array(rnpz["claude_coef"]), np.array(rnpz["claude_intercept"])
                            if "claude_classes" in rnpz:
                                claude_classes = list(rnpz["claude_classes"])
                if W_p is None and heads_sd_cl:
                    cw, cb, cwc, cbc = resolve_head_tensors(heads_sd_cl)
                    if cw is not None and cw.shape[0] == 18:
                        W_p, b_p = cw, cb
                    if cwc is not None and cwc.shape[0] == 18:
                        W_c, b_c = cwc, cbc
                if ro_labels is None or len(ro_labels) != 18 or set(ro_labels) != set(LABELS):
                    cand_m = meta_cl.get("labels", [])
                    ro_labels = list(cand_m) if (len(cand_m) == 18 and set(cand_m) == set(LABELS)) else sorted(LABELS)
                if W_spec is None and (claude_dir / "claude_head.safetensors").exists():
                    ch_sd = load_head_file(claude_dir / "claude_head.safetensors")
                    for wk, bk in [("weight", "bias"), ("plain.weight", "plain.bias")]:
                        if wk in ch_sd and ch_sd[wk].shape[0] == 5:
                            W_spec = ch_sd[wk].cpu().numpy()
                            b_spec = ch_sd[bk].cpu().numpy() if bk in ch_sd else np.zeros(5, dtype=np.float32)
                            break
                if W_spec is None and not is_smoke:
                    raise RuntimeError(f"Required specialist head W_spec is None for {claude_dir}")
                if claude_classes is None:
                    claude_classes = [LABELS[i] for i in FAMILIES["claude"]]
                Z_te, Z_u, Z_tr = _extract_trunk(trunk3, end_id=meta_cl.get("end_id"))
                if W_spec is not None and W_spec.shape[0] == 5:
                    claude_spec_logits = (Z_te @ W_spec.T + b_spec) if (run_submission and len(Z_te) > 0) else None
                    claude_spec_logits_u = (Z_u @ W_spec.T + b_spec) if (run_benchmark and len(Z_u) > 0) else None
                _register_checkpoint_outputs(
                    model_key="qwen3_8b_claude", disp_name="Qwen3-8B Claude Head",
                    Z_te=Z_te, Z_u=Z_u, Z_tr=Z_tr, W_p=W_p, b_p=b_p, W_c=W_c, b_c=b_c,
                    ckpt_labels=ro_labels, w_tot=get_model_weight("qwen3_8b_claude", default=0.0),
                    dp_key="claude_8b", dp_default=0.04, prior_default=0.30, se_default=0.35, auto_align=False,
                )
                merge_lora(trunk3.model, claude_dir, sign=-1.0)
                del Z_te, Z_u, Z_tr

            trunk3.free_memory()
            del trunk3
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 4: DeBERTa-v3-Large (Optional / Fallback)
        # ==============================================================================
        for deb_key in ("deberta_p2", "deberta_inhouse"):
            if deb_key not in ckpts:
                continue
            deb_dir = ckpts[deb_key]
            base_deb_dir = verified_bases.get("deberta_v3_large") or find_base_model(input_dir, "deberta", 1024, 24)
            print(f"[Pipeline] Loading Trunk 4: DeBERTa-v3 ({deb_key}) from {deb_dir}...")
            if run_submission and N_test > 0 and run_benchmark and N_unseen > 0:
                Z_all, Z_tr_deb, W_p_deb, b_p_deb, W_c_deb, b_c_deb, deb_labels = extract_deberta_checkpoint(deb_dir, base_deb_dir, test_texts + unseen_texts, train_sub_texts, y_sub)
                Z_te_deb, Z_u_deb = Z_all[:N_test], Z_all[N_test : N_test + N_unseen]
            elif run_submission and N_test > 0:
                Z_te_deb, Z_tr_deb, W_p_deb, b_p_deb, W_c_deb, b_c_deb, deb_labels = extract_deberta_checkpoint(deb_dir, base_deb_dir, test_texts, train_sub_texts, y_sub)
                Z_u_deb = np.zeros((0, Z_te_deb.shape[1]), dtype=np.float32)
            elif run_benchmark and N_unseen > 0:
                Z_u_deb, Z_tr_deb, W_p_deb, b_p_deb, W_c_deb, b_c_deb, deb_labels = extract_deberta_checkpoint(deb_dir, base_deb_dir, unseen_texts, train_sub_texts, y_sub)
                Z_te_deb = np.zeros((0, Z_u_deb.shape[1]), dtype=np.float32)
            else:
                continue
            if W_p_deb is None and not is_smoke:
                raise RuntimeError(f"Required classification head W_p is None for {deb_dir}")
            _register_checkpoint_outputs(
                model_key=deb_key, disp_name="DeBERTa-v3-Large" if deb_key == "deberta_p2" else "DeBERTa-InHouse",
                Z_te=Z_te_deb, Z_u=Z_u_deb, Z_tr=Z_tr_deb, W_p=W_p_deb, b_p=b_p_deb, W_c=W_c_deb, b_c=b_c_deb,
                ckpt_labels=deb_labels, w_tot=get_model_weight(deb_key, default=0.28),
                dp_key="deberta", dp_default=0.20, prior_default=0.80, se_default=0.10, auto_align=False,
            )
            del Z_tr_deb, Z_te_deb, Z_u_deb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # OPTIONAL CAUSAL LLM TRUNKS: Gemma-2-9B, Llama-3.1-8B, Mistral-7B-v0.3
        # ==============================================================================
        opt_causal_specs = [
            ("gemma2_9b_ts", "gemma2", 3584, 42, 24, "Gemma-2-9B", "gemma2_9b"),
            ("llama31_8b_ts", "llama", 4096, 32, 24, "Llama-3.1-8B", "llama3_8b"),
            ("llama3_8b_v5", "llama", 4096, 32, 24, "Llama-3-8B-v5", "llama3_8b"),
            ("mistral_7b_ts", "mistral", 4096, 32, 24, "Mistral-7B", "mistral_7b"),
        ]
        for opt_key, opt_fam, opt_h, opt_l, opt_keep, opt_disp, base_k in opt_causal_specs:
            if opt_key not in ckpts:
                continue
            base_opt_dir = verified_bases.get(opt_key) or verified_bases.get(base_k) or find_base_model(input_dir, opt_fam, opt_h, opt_l)
            if not base_opt_dir:
                print(f"[Pipeline] Base model for {opt_disp} not found; skipping {opt_key}.")
                continue
            print(f"[Pipeline] Loading Trunk: {opt_disp} ({opt_key}) from base {base_opt_dir}...")
            meta_opt = read_checkpoint_meta(ckpts[opt_key], default_keep=opt_keep, default_layers=[2, 4, 6, 8, 12, 16, 20, 24], default_max_len=447)
            opt_trunk = UniversalLoRATrunkLoader(base_opt_dir, keep_layers=meta_opt["keep"], layers=meta_opt["layers"], max_len=meta_opt["max_len"])
            _extract_shared_lora_ckpt(opt_trunk, opt_key, ckpts[opt_key], opt_disp, opt_keep, [2, 4, 6, 8, 12, 16, 20, 24], 0.20, opt_key, 0.05, 0.40, 0.25, allow_multifold=False, auto_align=True, required=False)
            opt_trunk.free_memory()
            del opt_trunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 8: BGE-M3 (v2 8-Layer Drop-<s> Mean-Pool + Ridge, or v1 CLS fallback)
        # ==============================================================================
        if "bge_m3" in ckpts:
            bge_dir = ckpts["bge_m3"]
            print(f"[Pipeline] Loading Trunk 8: BGE-M3 from {bge_dir}...")
            try:
                from transformers import AutoTokenizer, AutoModel
                raw_meta_bge = json.loads((bge_dir / "meta.json").read_text(encoding="utf-8")) if (bge_dir / "meta.json").exists() else {}
                heads_sd = load_head_file(bge_dir) if ((bge_dir / "heads.safetensors").exists() or (bge_dir / "heads.pt").exists()) else {}
                bge_tok = AutoTokenizer.from_pretrained(str(bge_dir))
                bge_model = AutoModel.from_pretrained(str(bge_dir), torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                bge_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                bge_model = bge_model.to(bge_dev)
                bge_model.eval()

                Z_bge_te, is_bge_v2 = _extract_bge_features(bge_model, bge_tok, bge_dev, test_texts, heads_sd, raw_meta_bge) if (run_submission and N_test > 0) else (np.zeros((0, 1024), dtype=np.float32), False)
                Z_bge_u, is_bge_v2_u = _extract_bge_features(bge_model, bge_tok, bge_dev, unseen_texts, heads_sd, raw_meta_bge) if (run_benchmark and N_unseen > 0) else (np.zeros((0, 1024), dtype=np.float32), is_bge_v2)
                is_bge_v2 = is_bge_v2 or is_bge_v2_u
                Z_bge_tr, _ = _extract_bge_features(bge_model, bge_tok, bge_dev, train_sub_texts, heads_sd, raw_meta_bge) if (extract_train_wpca and N_tr_sub > 0) else (np.zeros((0, Z_bge_u.shape[1] if len(Z_bge_u) > 0 else 1024), dtype=np.float32), is_bge_v2)

                del bge_model, bge_tok
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd) if heads_sd else (None, None, None, None)
                bge_labels = raw_meta_bge.get("labels", list(LABELS)) if isinstance(raw_meta_bge, dict) else list(LABELS)
                if W_p is not None:
                    _register_checkpoint_outputs(
                        model_key="bge_m3",
                        disp_name="BGE-M3-v2" if is_bge_v2 else "BGE-M3",
                        Z_te=Z_bge_te, Z_u=Z_bge_u, Z_tr=Z_bge_tr,
                        W_p=W_p, b_p=b_p, W_c=W_c, b_c=b_c,
                        ckpt_labels=bge_labels,
                        w_tot=get_model_weight("bge_m3", default=0.23),
                        dp_key="bge_m3", dp_default=0.10, prior_default=0.60, se_default=0.20,
                        auto_align=False,
                    )
                else:
                    print("[Pipeline] BGE-M3 classification head not found; embeddings utilized for distance graph.")
                    update_hybrid_groups(Z_bge_te, prior_scales.get("bge_m3", 0.60), Z_bge_u if N_unseen > 0 else None)
                del Z_bge_tr, Z_bge_te, Z_bge_u, heads_sd
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[Pipeline] Warning: BGE-M3 extraction failed ({e}); continuing with remaining models.")

        # Export fitted wPCA bundle if any models were fitted from scratch
        if len(wpca_bundle_export) > 0 and not (working_dir / "wpca_bundle.safetensors").exists():
            try:
                save_wpca_bundle(wpca_bundle_export, working_dir / "wpca_bundle.safetensors")
            except Exception as e:
                print(f"[Pipeline] Note: Could not export wpca_bundle: {e}")

        # Joint Spectral Bank (Predict from pre-trained params or fit)
        P_spectral = None
        P_spectral_u = None
        sb_cfg = comp_cfg.get("spectral_bank", {})
        plain_C = sb_cfg.get("plain_C", 0.04)
        center_C = sb_cfg.get("center_C", 0.03)
        center_blend = sb_cfg.get("center_blend", 0.35)

        if spectral_params is not None:
            print("[Pipeline] Computing Spectral Bank inference from pre-trained parameters...")
            if run_submission and spectral_blocks_te and tracker_test is not None:
                P_spectral = predict_spectral_bank(
                    spectral_blocks=spectral_blocks_te,
                    prm=spectral_params,
                    D_tfidf=tracker_test.D_tfidf,
                    D_neural_sum=tracker_test.D_neural_sum,
                    D_neural_weight_total=tracker_test.D_neural_weight_total,
                    max_cluster_size=max_cl,
                )
            if run_benchmark and spectral_blocks_u and tracker_unseen is not None:
                P_spectral_u = predict_spectral_bank(
                    spectral_blocks=spectral_blocks_u,
                    prm=spectral_params,
                    D_tfidf=tracker_unseen.D_tfidf,
                    D_neural_sum=tracker_unseen.D_neural_sum,
                    D_neural_weight_total=tracker_unseen.D_neural_weight_total,
                    max_cluster_size=max_cl,
                )
        elif spectral_blocks_tr:
            res_spec = fit_joint_spectral_bank(
                spectral_tr_blocks=spectral_blocks_tr,
                spectral_te_blocks=spectral_blocks_te if (run_submission and spectral_blocks_te) else [np.zeros((N_test, b.shape[1]), dtype=np.float32) for b in spectral_blocks_tr],
                y_sub=y_sub,
                prompts_sub=prompts_sub,
                D_tfidf=tracker_test.D_tfidf if tracker_test is not None else np.zeros((0, 0), dtype=np.float32),
                D_neural_sum=tracker_test.D_neural_sum if tracker_test is not None else np.zeros((0, 0), dtype=np.float32),
                D_neural_weight_total=tracker_test.D_neural_weight_total if tracker_test is not None else 0.0,
                N_test=N_test if run_submission else 0,
                plain_C=plain_C,
                center_C=center_C,
                center_blend=center_blend,
                max_cluster_size=max_cl,
                spectral_u_blocks=spectral_blocks_u if (run_benchmark and spectral_blocks_u) else None,
                D_tfidf_u=tracker_unseen.D_tfidf if tracker_unseen is not None else None,
                D_neural_sum_u=tracker_unseen.D_neural_sum if tracker_unseen is not None else None,
                D_neural_weight_total_u=tracker_unseen.D_neural_weight_total if tracker_unseen is not None else 0.0,
                N_unseen=N_unseen if run_benchmark else 0,
                return_unseen=run_benchmark,
                save_path=working_dir / "spectral_heads.safetensors",
            )
            if run_benchmark:
                P_spectral, oof_score, P_spectral_u = res_spec
            else:
                P_spectral, oof_score = res_spec
                P_spectral_u = None
            print(f"[Pipeline] Joint Spectral Bank fitted (OOF BAcc = {oof_score:.4f})")

    elif is_smoke:
        print("[Pipeline] Smoke mode active: synthesizing high-fidelity mock representations...")
        np.random.seed(42)
        mock_p_dp = softmax(np.random.randn(N_test, K), axis=1) if N_test > 0 else np.zeros((0, K), dtype=np.float32)
        mock_p_spec = softmax(np.random.randn(N_test, K), axis=1) if N_test > 0 else np.zeros((0, K), dtype=np.float32)
        mock_p_dp_u = softmax(np.random.randn(N_unseen, K), axis=1) if N_unseen > 0 else np.zeros((0, K), dtype=np.float32)
        mock_p_spec_u = softmax(np.random.randn(N_unseen, K), axis=1) if N_unseen > 0 else np.zeros((0, K), dtype=np.float32)
        if N_test > 0:
            dp_posteriors.append((0.30, mock_p_dp))
            P_spectral = mock_p_spec
        if N_unseen > 0:
            dp_posteriors_u.append((0.30, mock_p_dp_u))
            P_spectral_u = mock_p_spec_u

    # Register statistical heads directly into auditor as first-class ensemble members
    w_dp = get_model_weight("wpca_dp", default=0.25)
    lp_dp_te = np.zeros((N_test, K), dtype=np.float32)
    if run_submission and dp_posteriors and N_test > 0:
        tot_dp_w = sum(w for w, _ in dp_posteriors)
        P_dp_comb = sum(w * P_dp for w, P_dp in dp_posteriors) / max(1e-8, tot_dp_w)
        lp_dp_te = np.log(np.clip(P_dp_comb, 1e-9, 1.0))

    lp_dp_u = np.zeros((0, K), dtype=np.float32)
    if run_benchmark and dp_posteriors_u and N_unseen > 0:
        tot_dp_u_w = sum(w for w, _ in dp_posteriors_u)
        P_dp_u_comb = sum(w * P_dp for w, P_dp in dp_posteriors_u) / max(1e-8, tot_dp_u_w)
        lp_dp_u = np.log(np.clip(P_dp_u_comb, 1e-9, 1.0))

    if (run_submission and len(lp_dp_te) == N_test and N_test > 0) or (run_benchmark and len(lp_dp_u) == N_unseen and N_unseen > 0):
        auditor.add_model(
            name="wPCA DP Posterior",
            weight=w_dp,
            unseen_log_probs=lp_dp_u,
            test_log_probs=lp_dp_te,
        )

    w_sb = get_model_weight("spectral_bank", default=0.10)
    lp_sb_te = np.zeros((N_test, K), dtype=np.float32)
    if run_submission and "P_spectral" in locals() and P_spectral is not None and N_test > 0:
        lp_sb_te = np.log(np.clip(P_spectral, 1e-9, 1.0))

    lp_sb_u = np.zeros((0, K), dtype=np.float32)
    if run_benchmark and "P_spectral_u" in locals() and P_spectral_u is not None and N_unseen > 0:
        lp_sb_u = np.log(np.clip(P_spectral_u, 1e-9, 1.0))

    if (run_submission and len(lp_sb_te) == N_test and N_test > 0) or (run_benchmark and len(lp_sb_u) == N_unseen and N_unseen > 0):
        auditor.add_model(
            name="Spectral Bank",
            weight=w_sb,
            unseen_log_probs=lp_sb_u,
            test_log_probs=lp_sb_te,
        )

    # 4. MODULE 4: Calibrated L-BFGS Log-Linear Ensemble & Postprocessing
    print("[Pipeline] Module 4: Calibrated L-BFGS Log-Linear Ensemble...")
    lock_prior = bool(cfg.get("lock_prior_weights", False))
    optimize_on_unseen = (N_unseen >= 36 and not is_smoke and not lock_prior)
    post_cfg_eff = dict(post_cfg)
    if len(auditor.models) > 0:
        P_mix, active_weights_map, df_ablation = auditor.optimize_weights_and_blend(
            working_dir=working_dir,
            optimize_on_unseen=optimize_on_unseen,
            lock_prior_weights=lock_prior,
            calibration_cfg=cfg,
            claude_spec_logits_u=claude_spec_logits_u if "claude_spec_logits_u" in locals() else None,
            claude_classes=claude_classes if "claude_classes" in locals() else None,
        )
        norm_map = getattr(auditor, "normalized_weights_map", active_weights_map)
        shares_str = ", ".join([f"'{k}': {w*100.0:.1f}%" for k, w in norm_map.items() if w > 0.0])
        print(f"[Pipeline] Calibrated ensemble active weights: {{{shares_str}}}")
        fitted_art = getattr(auditor, "fitted_artifact", None)
        if optimize_on_unseen and isinstance(fitted_art, dict) and claude_spec_logits_u is not None and "mean_gw" in fitted_art:
            post_cfg_eff["claude_specialist_weight"] = float(fitted_art["mean_gw"])
        R_main = P_mix
        P_mix_u = getattr(auditor, "P_mix_unseen", np.zeros((0, K), dtype=np.float64)) if run_benchmark else np.zeros((0, K), dtype=np.float64)
    else:
        P_mix = np.ones((N_test, K), dtype=np.float32) / float(K)
        R_main = P_mix
        P_mix_u = np.zeros((0, K), dtype=np.float64)

    train_prior = np.bincount(y_sub, minlength=18).astype(np.float64) + 1.0 if len(y_sub) > 0 else None
    train_ref_texts = train_sub_texts if len(train_sub_texts) > 0 else (df_train["text"].fillna("").astype(str).tolist() if df_train is not None and "text" in df_train.columns else [])
    train_ref_y = y_sub if len(y_sub) > 0 else (df_train["model"].values if df_train is not None and "model" in df_train.columns else np.array([], dtype=int))

    # Run full inference pipeline on TEST (if run_submission)
    if run_submission and N_test > 0:
        P_final = run_full_inference_pipeline(
            P_mix=P_mix,
            texts=test_texts,
            post_cfg=post_cfg_eff,
            R_main=R_main,
            claude_spec_logits=claude_spec_logits if "claude_spec_logits" in locals() else None,
            claude_classes=claude_classes if "claude_classes" in locals() else None,
            log_p_handwriting=log_p_handwriting if "log_p_handwriting" in locals() else None,
            rule_flags=rule_flags_test if "rule_flags_test" in locals() else None,
            train_texts=train_ref_texts,
            y_train=train_ref_y,
            train_prior=train_prior,
        )
    else:
        P_final = np.ones((N_test, K), dtype=np.float32) / float(K)

    # Run full inference pipeline on UNSEEN BENCHMARK (if run_benchmark)
    P_final_u = np.zeros((0, K), dtype=np.float32)
    if run_benchmark and N_unseen > 0:
        P_final_u = run_full_inference_pipeline(
            P_mix=P_mix_u,
            texts=unseen_texts,
            post_cfg=post_cfg_eff,
            R_main=P_mix_u,
            claude_spec_logits=claude_spec_logits_u if "claude_spec_logits_u" in locals() else None,
            claude_classes=claude_classes if "claude_classes" in locals() else None,
            log_p_handwriting=log_p_handwriting_unseen if "log_p_handwriting_unseen" in locals() else None,
            rule_flags=rule_flags_unseen if "rule_flags_unseen" in locals() else None,
            train_texts=train_ref_texts,
            y_train=train_ref_y,
            train_prior=train_prior,
        )

        raw_unseen_bacc = float(balanced_accuracy_score(y_unseen, P_mix_u.argmax(axis=1)))
        post_unseen_bacc = float(balanced_accuracy_score(y_unseen, P_final_u.argmax(axis=1)))
        lift_post = post_unseen_bacc - raw_unseen_bacc
        print("=" * 70)
        print("[Pipeline Benchmark Validation] Symmetrical Pipeline Parity Evaluation:")
        print(f"  - Pre-Postprocessing Ensemble Raw BAcc : {raw_unseen_bacc:.4f} ({raw_unseen_bacc*100:.2f}%)")
        print(f"  - Full Postprocessed Benchmark BAcc     : {post_unseen_bacc:.4f} ({post_unseen_bacc*100:.2f}%)")
        print(f"  - Downstream Postprocessing Lift       : {lift_post:+.4f} ({lift_post*100:+.2f}%)")
        print("=" * 70)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_unseen, P_final_u.argmax(axis=1), labels=range(K))
        print("[Pipeline Benchmark Validation] Class-by-Class Accuracies:")
        for c_idx, c_name in enumerate(LABELS):
            denom = cm[c_idx].sum()
            c_acc = cm[c_idx, c_idx] / denom if denom > 0 else 0.0
            print(f"  [{c_idx:02d}] {c_name:<28s}: {c_acc*100:5.1f}% ({cm[c_idx, c_idx]}/{denom})")

        cm_err = cm.copy()
        np.fill_diagonal(cm_err, 0)
        top_err_indices = np.dstack(np.unravel_index(np.argsort(cm_err.ravel())[::-1], cm_err.shape))[0][:10]
        print("[Pipeline Benchmark] Top 10 confused class pairs:")
        for r_idx, c_idx in top_err_indices:
            if cm_err[r_idx, c_idx] > 0:
                print(f"    True: {LABELS[r_idx]:<22} -> Pred: {LABELS[c_idx]:<22} : {cm_err[r_idx, c_idx]} errors")

        # Save clean_benchmark_1800_predictions.npz
        bench_pred_file = working_dir / "clean_benchmark_1800_predictions.npz"
        np.savez_compressed(
            bench_pred_file,
            P_mix=P_mix_u.astype(np.float32),
            P_final=P_final_u.astype(np.float32),
            y_unseen=y_unseen.astype(np.int32),
            raw_bacc=np.array([raw_unseen_bacc], dtype=np.float32),
            post_bacc=np.array([post_unseen_bacc], dtype=np.float32),
            lift=np.array([lift_post], dtype=np.float32),
            labels=np.array(LABELS, dtype=object),
        )
        print(f"[Pipeline Benchmark] Exported clean benchmark predictions: {bench_pred_file} ({bench_pred_file.stat().st_size} bytes)")

    # Auditing and report generation
    test_final_preds = P_final.argmax(axis=1) if len(P_final) > 0 else np.array([])
    auditor.print_and_save_report(working_dir, final_test_preds=test_final_preds)

    # Final Benchmark Accuracy Comparison on strictly unseen samples
    if run_benchmark and N_unseen > 0 and len(y_unseen) > 0:
        print("\n" + "=" * 95)
        print(f" FINAL BENCHMARK ACCURACY COMPARISON ON {len(y_unseen)} STRICTLY UNSEEN SAMPLES")
        print("=" * 95)
        print(f"{'Ensemble Configuration':<30} | {'Models Used':<12} | {'Raw BAcc':<10} | {'+Sinkhorn':<10} | {'Full Post-Processed'}")
        print("-" * 95)
        raw_full = float(balanced_accuracy_score(y_unseen, P_mix_u.argmax(axis=1))) if len(P_mix_u) > 0 else 0.0
        sink_alpha = float(post_cfg_eff.get("sinkhorn_alpha", 0.50))
        sink_full = float(balanced_accuracy_score(y_unseen, damped_sinkhorn(P_mix_u, alpha=sink_alpha).argmax(axis=1))) if len(P_mix_u) >= 18 else raw_full
        post_full = float(balanced_accuracy_score(y_unseen, P_final_u.argmax(axis=1))) if len(P_final_u) > 0 else 0.0
        n_active = sum(1 for m in auditor.models if m.get("weight", 0.0) > 0.0)
        print(f"{'Calibrated Ensemble':<30} | {n_active:>2d} / {len(auditor.models):<7d} | {raw_full:<10.4f} | {sink_full:<10.4f} | {post_full:<10.4f} (+{(post_full - raw_full)*100:+.2f}%)")
        fitted_art = getattr(auditor, "fitted_artifact", None)
        if isinstance(fitted_art, dict) and "oof_bacc_raw" in fitted_art:
            print(
                f"{'5-Fold OOF L-BFGS':<30} | {len(fitted_art['model_names']):>2d} / {len(auditor.models):<7d} | "
                f"{fitted_art['oof_bacc_raw']:<10.4f} | {fitted_art['oof_bacc_sink']:<10.4f} | "
                f"{fitted_art['oof_bacc_nested_claude']:<10.4f} (NLL={fitted_art['oof_nll_raw']:.4f})"
            )
        print("=" * 95 + "\n")

    # Generate Submission Artifacts
    df_sub, out_parquet, out_csv = generate_submission(P_final, test_ids, working_dir)
    elapsed = time.time() - start_time
    print(f"[Pipeline] Submission generated successfully in {elapsed:.2f}s: {out_parquet} ({len(df_sub)} rows)")

    # Export compact local evaluation cache for sub-second offline local tuning
    try:
        cache_dict = {
            "N_test": np.array([N_test], dtype=np.int32),
            "N_unseen": np.array([N_unseen], dtype=np.int32),
            "y_unseen": y_unseen.astype(np.int32) if len(y_unseen) > 0 else np.array([], dtype=np.int32),
            "groups_unseen": tracker_unseen.groups.get(0.85, np.zeros(0, dtype=np.int32)) if tracker_unseen else np.zeros(0, dtype=np.int32),
            "groups_085_u": tracker_unseen.groups.get(0.85, np.zeros(0, dtype=np.int32)) if tracker_unseen else np.zeros(0, dtype=np.int32),
            "groups_072_u": tracker_unseen.groups.get(0.72, np.zeros(0, dtype=np.int32)) if tracker_unseen else np.zeros(0, dtype=np.int32),
            "groups_065_u": tracker_unseen.groups.get(0.65, np.zeros(0, dtype=np.int32)) if tracker_unseen else np.zeros(0, dtype=np.int32),
            "groups_060_u": tracker_unseen.groups.get(0.60, np.zeros(0, dtype=np.int32)) if tracker_unseen else np.zeros(0, dtype=np.int32),
            "D_tfidf": tracker_test.D_tfidf.astype(np.float32) if (tracker_test and tracker_test.D_tfidf is not None) else np.zeros((0, 0), dtype=np.float32),
            "D_hybrid": tracker_test.D_hybrid.astype(np.float32) if (tracker_test and tracker_test.D_hybrid is not None) else np.zeros((0, 0), dtype=np.float32),
            "D_tfidf_u": tracker_unseen.D_tfidf.astype(np.float32) if (tracker_unseen and tracker_unseen.D_tfidf is not None) else np.zeros((0, 0), dtype=np.float32),
            "D_hybrid_u": tracker_unseen.D_hybrid.astype(np.float32) if (tracker_unseen and tracker_unseen.D_hybrid is not None) else np.zeros((0, 0), dtype=np.float32),
            "groups_085": tracker_test.groups.get(0.85, np.zeros(0, dtype=np.int32)) if tracker_test else np.zeros(0, dtype=np.int32),
            "groups_072": tracker_test.groups.get(0.72, np.zeros(0, dtype=np.int32)) if tracker_test else np.zeros(0, dtype=np.int32),
            "groups_065": tracker_test.groups.get(0.65, np.zeros(0, dtype=np.int32)) if tracker_test else np.zeros(0, dtype=np.int32),
            "groups_060": tracker_test.groups.get(0.60, np.zeros(0, dtype=np.int32)) if tracker_test else np.zeros(0, dtype=np.int32),
            "model_names": np.array([m["name"] for m in auditor.models], dtype=object),
            "model_tiers": np.array([m["tier"] for m in auditor.models], dtype=object),
            "model_prior_weights": np.array([m.get("prior_weight", m["weight"]) for m in auditor.models], dtype=np.float64),
            "model_learned_weights": np.array([m["weight"] for m in auditor.models], dtype=np.float64),
            "P_mix": P_mix.astype(np.float32),
            "P_final": P_final.astype(np.float32),
            "P_mix_unseen": P_mix_u.astype(np.float32) if len(P_mix_u) > 0 else np.zeros((0, K), dtype=np.float32),
            "P_final_unseen": P_final_u.astype(np.float32) if len(P_final_u) > 0 else np.zeros((0, K), dtype=np.float32),
            "test_texts": np.array(test_texts, dtype=object),
            "unseen_texts": np.array(unseen_texts, dtype=object),
        }
        for idx, m in enumerate(auditor.models):
            cache_dict[f"m_{idx}_unseen_lp"] = m["unseen_lp"].astype(np.float32)
            cache_dict[f"m_{idx}_test_lp"] = m["test_lp"].astype(np.float32)
        if "rule_flags_test" in locals() and rule_flags_test is not None:
            cache_dict["rule_flags_test"] = rule_flags_test.astype(np.float32)
        if "rule_flags_unseen" in locals() and rule_flags_unseen is not None:
            cache_dict["rule_flags_unseen"] = rule_flags_unseen.astype(np.float32)
        if "log_p_handwriting" in locals() and log_p_handwriting is not None:
            cache_dict["log_p_handwriting"] = log_p_handwriting.astype(np.float32)
        if "log_p_handwriting_unseen" in locals() and log_p_handwriting_unseen is not None:
            cache_dict["log_p_handwriting_unseen"] = log_p_handwriting_unseen.astype(np.float32)
        if "claude_spec_logits" in locals() and claude_spec_logits is not None:
            cache_dict["claude_spec_logits"] = claude_spec_logits.astype(np.float32)
        if "claude_spec_logits_u" in locals() and claude_spec_logits_u is not None:
            cache_dict["claude_spec_logits_u"] = claude_spec_logits_u.astype(np.float32)
        if "claude_classes" in locals() and claude_classes is not None:
            cache_dict["claude_classes"] = np.array(claude_classes, dtype=object)
        if "fitted_art" in locals() and isinstance(fitted_art, dict):
            cache_dict["cal_model_names"] = np.array(fitted_art.get("model_names", []), dtype=object)
            cache_dict["cal_T_m"] = np.asarray(fitted_art.get("mean_T_m", []), dtype=np.float64)
            cache_dict["cal_w_m"] = np.asarray(fitted_art.get("mean_w_m", []), dtype=np.float64)
            cache_dict["cal_T_global"] = np.array([float(fitted_art.get("mean_T_global", 1.0))], dtype=np.float64)
            cache_dict["cal_gw"] = np.array([float(fitted_art.get("mean_gw", 0.0))], dtype=np.float64)
        cache_dict["num_extra"] = np.array([0], dtype=np.int32)

        cache_file_path = working_dir / "local_eval_cache.npz"
        np.savez_compressed(cache_file_path, **cache_dict)
        print(f"[Pipeline] Exported compact local evaluation cache: {cache_file_path} ({cache_file_path.stat().st_size} bytes)")

        # Export versioned, standardized benchmark feature bundle
        if len(features_unseen_dict) > 0 and N_unseen > 0:
            version_str = cfg.get("version") or os.environ.get("EXPERIMENT_VERSION") or os.environ.get("PIPELINE_VERSION") or "latest"
            date_str = time.strftime("%Y%m%d")
            clean_v = version_str.lower() if version_str.lower().startswith("v") else f"v{version_str}"
            bundle_dir_name = f"{clean_v}_{date_str}"

            # Standard consolidated features file for working_dir
            feat_save_dict = {
                "labels": y_unseen.astype(np.int32),
                "groups_085": tracker_unseen.groups.get(0.85, np.zeros(0, dtype=np.int32)),
                "groups_072": tracker_unseen.groups.get(0.72, np.zeros(0, dtype=np.int32)),
                "groups_065": tracker_unseen.groups.get(0.65, np.zeros(0, dtype=np.int32)),
                "groups_060": tracker_unseen.groups.get(0.60, np.zeros(0, dtype=np.int32)),
                "D_tfidf": tracker_unseen.D_tfidf.astype(np.float32) if tracker_unseen.D_tfidf is not None else np.zeros((0, 0), dtype=np.float32),
                "D_hybrid": tracker_unseen.D_hybrid.astype(np.float32) if tracker_unseen.D_hybrid is not None else np.zeros((0, 0), dtype=np.float32),
            }
            for k, v in features_unseen_dict.items():
                feat_save_dict[f"Z_unseen_{k}"] = v
            feat_file_path = working_dir / "benchmark_1800_features.npz"
            np.savez_compressed(feat_file_path, **feat_save_dict)
            print(f"[Pipeline] Exported {len(features_unseen_dict)} benchmark feature representations to: {feat_file_path} ({feat_file_path.stat().st_size} bytes)")

            # Save versioned bundle per model to target directory if available
            target_dirs = [working_dir / "benchmark_features" / bundle_dir_name]
            try:
                this_file = Path(__file__).resolve() if "__file__" in globals() or "__file__" in locals() else None
                local_features_root = (this_file.parent.parent.parent / "local_data" / "benchmark_features") if this_file is not None else Path("local_data/benchmark_features")
                if local_features_root.parent.exists():
                    target_dirs.append(local_features_root / bundle_dir_name)
            except Exception:
                pass

            for bundle_dir in target_dirs:
                try:
                    bundle_dir.mkdir(parents=True, exist_ok=True)
                    models_meta = {}
                    for model_name, Z_val in features_unseen_dict.items():
                        model_file = f"Z_unseen_{model_name}_{clean_v}_{date_str}.npz"
                        m_path = bundle_dir / model_file
                        np.savez_compressed(
                            m_path,
                            Z_unseen=Z_val.astype(np.float16),
                            labels=y_unseen.astype(np.int32),
                            groups=tracker_unseen.groups.get(0.85, np.zeros(0, dtype=np.int32)),
                            model_name=model_name,
                            version=clean_v,
                            date=date_str,
                        )
                        models_meta[model_name] = {
                            "filename": model_file,
                            "shape": list(Z_val.shape),
                            "dtype": str(Z_val.dtype),
                            "size_mb": round(m_path.stat().st_size / (1024 * 1024), 2),
                        }

                    # Consolidated bundle archive
                    master_file = f"benchmark_1800_features_all_models_{clean_v}_{date_str}.npz"
                    np.savez_compressed(bundle_dir / master_file, **feat_save_dict)

                    # Metadata JSON
                    meta_content = {
                        "version": clean_v,
                        "date": date_str,
                        "N_samples": int(N_unseen),
                        "models": models_meta,
                        "consolidated_file": master_file,
                    }
                    (bundle_dir / "metadata.json").write_text(json.dumps(meta_content, indent=2), encoding="utf-8")
                    print(f"[Pipeline] Standardized benchmark feature bundle saved to: {bundle_dir}")
                except Exception as ex_bundle:
                    print(f"[Pipeline] Note: Could not save bundle to {bundle_dir}: {ex_bundle}")
    except Exception as e:
        print(f"[Pipeline] Note: Could not write local evaluation outputs: {e}")

    return df_sub, out_parquet
