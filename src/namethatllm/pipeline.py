r"""
Top-Level Pipeline Orchestrator
===============================
This module orchestrates the complete end-to-end inference and evaluation pipeline:

Architectural Dataflow:
-----------------------
  [ Frozen Neural Readers ]              [ Fit Live in Pipeline ]
  • Qwen3-14B Part 2                     • Style SVC (4 views)
  • Qwen3-14B Part 1                     • Lexical Model (5 views)
  • Qwen2.5-14B Part 2                   • Exact Prompt Group Search (TF-IDF linkage)
  • DeBERTa-v3-Large                     • Spectral Bank (joint multi-view features)
  • 8B Patches (Claude, In-Style, LoRA24)• wPCA-DP Posterior (shared Gaussian DP)
                    \                           /
                     \                         /
                      ▼                       ▼
            ┌───────────────────────────────────────────────┐
            │ Stage 1: Quick Probe and Gate                 │
            │ Evaluates solo balanced accuracy on strictly   │
            │ unseen 360-row holdout set. Drops any reader  │
            │ scoring below min threshold (0.40 BAcc).      │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │ Stage 2: Flat Ensemble / Greedy Stepwise      │
            │ Computes optimal combination weights using    │
            │ Leave-One-Out (LOO) attribution. Non-helpers  │
            │ receive weight 0.0 (pruned).                  │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │ Stage 3: Neural Specialist Gates Layer        │
            │ Mass-preserving neural gating inside families: │
            │ • Claude Head     (Model 3 head, weight 0.15) │
            │ • Gemini LoRA     (Qwen3-14B, weight 0.65)    │
            │ • GLM LoRA        (Qwen3-14B, weight 0.70)    │
            │ • GPT LoRA        (Qwen3-14B, weight 0.55)    │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │ Stage 4: Orthographic Punctuation Shield      │
            │ Bayesian tokenization penalties: curly quotes │
            │ get -1.0 log penalty for Claude and GLM.      │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │ Stage 5: Cluster Uniqueness Repulsion         │
            │ Cavity mean-field repulsion: at-most-one text │
            │ per model within identical test prompt group. │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │ Stage 6: Multiplicative Sinkhorn Balancing    │
            │ 300 alternating row/column normalization steps│
            │ matching uniform 1/18 global class priors.    │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │ Output: submission.parquet & submission.csv   │
            └───────────────────────────────────────────────┘
"""

import os
import gc
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import balanced_accuracy_score

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
    compute_family_centered_logits,
    merge_lora,
    read_checkpoint_meta,
    extract_deberta_checkpoint
)
from .wpca_gaussian import (
    fit_shared_gaussian_lowram,
    transform_lowram,
    split_oversized_clusters_dist,
    scores_exact_dp
)
from .spectral_bank import fit_joint_spectral_bank, make_spectral_blocks, fit_within_family_spectral_gates
from .postprocess import (
    blend_main_and_complementary,
    apply_claude_specialist_gate,
    apply_family_specialist_lora_gate,
    apply_all_family_confidence_gates,
    apply_cluster_uniqueness_repulsion,
    apply_unified_handwriting_shield,
    apply_curly_quote_shield,
    sinkhorn_balancing,
    apply_typobayes_postprocess,
    generate_submission
)
from .ablation import build_unseen_holdout_dataset, UnseenStepwiseAuditor

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
    Combines plain and prompt-centered twin log-probabilities without compressing
    logit temperature on singleton rows (where lp_twin == 0):
        log_softmax(lp_plain + (w_twin / w_plain) * lp_twin)
    """
    if lp_twin is None or w_twin <= 0.0 or len(lp_twin) != len(lp_plain):
        return log_softmax(lp_plain, axis=1)
    ratio = float(w_twin) / max(float(w_plain), 1e-6)
    return log_softmax(lp_plain + ratio * lp_twin, axis=1)

def extract_family_specialist_logits(
    trunk1: UniversalLoRATrunkLoader,
    spec_dir: Path,
    family_name: str,
    test_texts: List[str],
    unseen_texts: List[str],
    LOGIT_MAIN: np.ndarray,
    auditor: Any,
    groups_test_multi: Tuple[np.ndarray, np.ndarray, np.ndarray],
    groups_unseen: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Selectively runs inference for a Qwen3-14B family specialist adapter
    (GPT 5-class, Gemini 4-class, or GLM 2-class) on candidate test and unseen passages.

    Performance & Resource Optimization:
    ------------------------------------
    Rather than running the massive 14B model over ALL test samples, this function
    applies selective family masking:
    1. Only passages where the ensemble indicates membership in this family
       (either top-1 prediction is in family OR cumulative family probability >= 0.15)
       are selected for specialist inference.
    2. Dynamically hot-swaps LoRA adapter weights on the already loaded Qwen3-14B trunk
       without re-loading the base model from disk (saving ~2 minutes per specialist).
    3. Restores original trunk weights immediately upon completion.
    """
    fam_cols = FAMILIES[family_name]
    target_labels = [LABELS[i] for i in fam_cols]
    K_fam = len(target_labels)
    N_test = len(test_texts)
    N_unseen = len(unseen_texts)

    spec_logits_te = np.zeros((N_test, K_fam), dtype=np.float32)
    spec_logits_u = np.zeros((N_unseen, K_fam), dtype=np.float32)

    # 1. Selective Family Masking for Test set: filter to candidate rows
    if N_test <= 60:
        te_mask = np.ones(N_test, dtype=bool)
    else:
        P_cur = softmax(LOGIT_MAIN, axis=1)
        te_mask = np.isin(P_cur.argmax(axis=1), fam_cols) | (P_cur[:, fam_cols].sum(axis=1) >= 0.15)

    # 2. Selective Family Masking for Unseen holdout set
    if N_unseen <= 60:
        u_mask = np.ones(N_unseen, dtype=bool) if N_unseen > 0 else np.zeros(0, dtype=bool)
    else:
        u_mask = np.zeros(N_unseen, dtype=bool)
        if hasattr(auditor, "models") and auditor.models:
            for m in auditor.models:
                u_lp = m.get("unseen_lp")
                if u_lp is not None and len(u_lp) == N_unseen:
                    p_u = softmax(u_lp, axis=-1)
                    u_mask |= np.isin(p_u.argmax(axis=1), fam_cols) | (p_u[:, fam_cols].sum(axis=1) >= 0.15)
        if not u_mask.any() and N_unseen > 0:
            u_mask[:] = True

    sel_te_idx = np.flatnonzero(te_mask)
    sel_u_idx = np.flatnonzero(u_mask)
    n_sel_te = len(sel_te_idx)
    n_sel_u = len(sel_u_idx)

    print(f"[Specialist {family_name}] Masked inference: {n_sel_te}/{N_test} test rows, {n_sel_u}/{N_unseen} unseen rows.")
    if n_sel_te == 0 and n_sel_u == 0:
        return spec_logits_te, spec_logits_u, target_labels

    meta_spec = read_checkpoint_meta(
        spec_dir,
        default_keep=27,
        default_layers=[2, 4, 6, 9, 13, 17, 22, 27],
        default_max_len=447
    )
    # Step 1: Configure active layers and take a snapshot of base model weights
    trunk1.set_active_layers(meta_spec["layers"], meta_spec["max_len"])
    backup_spec = trunk1.snapshot_weights(spec_dir)

    # Step 2: Merge specialist LoRA weights directly into Qwen3-14B attention and MLP matrices
    merged = merge_lora(trunk1.model, spec_dir, sign=+1.0)
    heads_sd = load_head_file(spec_dir)
    trunk1.load_norm_weights(heads_sd)

    # Step 3: Resolve linear heads (plain head W_p, b_p and centered twin head W_c, b_c)
    W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd, num_classes=K_fam)
    src_labels = meta_spec.get("labels", target_labels)
    src_list = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in src_labels]
    perm = None
    if len(src_list) == K_fam and set(src_list) == set(target_labels):
        perm = [src_list.index(l) for l in target_labels]

    # Step 4: Extract multi-layer representations for selected test & holdout passages
    sel_texts = [test_texts[i] for i in sel_te_idx] + [unseen_texts[j] for j in sel_u_idx]
    Z_sel = trunk1.extract_exact(sel_texts, batch_size=8, end_id=meta_spec["end_id"])

    Z_sel_te = Z_sel[:n_sel_te]
    Z_sel_u = Z_sel[n_sel_te:]

    g85, g72, g65 = groups_test_multi

    # Step 5: Compute temperature-scaled plain logits for candidate test passages
    # (Do NOT prompt-center across family-masked subsets, as full-cluster mean is unavailable on masked rows)
    if n_sel_te > 0 and W_p is not None:
        raw_p_te = Z_sel_te @ W_p.T + b_p
        if perm is not None:
            raw_p_te = raw_p_te[:, perm]
        std_te = float(np.std(raw_p_te, axis=1).mean()) if len(raw_p_te) > 0 else 1.0
        temp_te = max(1.0, std_te / 4.2)
        spec_logits_te[sel_te_idx] = log_softmax(raw_p_te / temp_te, axis=1).astype(np.float32)

    # Step 6: Compute plain logits for candidate unseen holdout passages & validate within-family BAcc
    if n_sel_u > 0 and W_p is not None:
        raw_p_u = Z_sel_u @ W_p.T + b_p
        if perm is not None:
            raw_p_u = raw_p_u[:, perm]
        std_u = float(np.std(raw_p_u, axis=1).mean()) if len(raw_p_u) > 0 else 1.0
        temp_u = max(1.0, std_u / 4.2)
        spec_logits_u[sel_u_idx] = log_softmax(raw_p_u / temp_u, axis=1).astype(np.float32)

        if hasattr(auditor, "y_unseen") and len(auditor.y_unseen) == N_unseen:
            fam_true_mask = np.isin(auditor.y_unseen, fam_cols) & u_mask
            if fam_true_mask.sum() >= len(fam_cols) * 2:
                y_fam_true = np.array([fam_cols.index(int(y)) for y in auditor.y_unseen[fam_true_mask]])
                fam_preds = spec_logits_u[fam_true_mask].argmax(axis=1)
                fam_bacc = float(balanced_accuracy_score(y_fam_true, fam_preds))
                print(f"[Specialist {family_name}] Unseen within-family BAcc = {fam_bacc:.4f}")
                if fam_bacc < (1.0 / len(fam_cols)) + 0.10:
                    print(f"[Specialist {family_name}] REJECTED (within-family BAcc {fam_bacc:.4f} too low); zeroing specialist logits.")
                    spec_logits_te.fill(0.0)
                    spec_logits_u.fill(0.0)

    # Step 7: Cleanly restore original base weights and free temporary tensors
    trunk1.restore_weights(backup_spec)
    trunk1.set_final_norm(False)
    del backup_spec, heads_sd, Z_sel
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return spec_logits_te, spec_logits_u, target_labels

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
    print(f"[Pipeline] Startup: is_smoke={is_smoke}, enable_ablation_probe={enable_ablation_probe}, probe_gate_min_bacc={probe_gate_min_bacc:.2f}")
    print(f"[Pipeline] Paths: train={train_path}, test={test_path}, pool={pool_file}, all_pools={len(all_pool_files)}")

    df_test = pd.read_csv(test_path)
    test_ids = df_test["id"]
    test_texts = df_test["text"].fillna("").astype(str).tolist()
    N_test = len(test_texts)
    print(f"[Pipeline] Loaded test set: {N_test} rows")

    # Build Zero-Leakage Unseen Holdout Benchmark if enable_ablation_probe is True
    if enable_ablation_probe and not is_smoke and not (N_test > 100 and bool(cfg.get("lock_prior_weights", True))):
        ablation_spc = int(cfg.get("ablation_samples_per_class", 20))
        df_unseen, holdout_hashes = build_unseen_holdout_dataset(input_dir, train_path, samples_per_class=ablation_spc)
        unseen_parquet_path = working_dir / "unseen_holdout.parquet"
        df_unseen.to_parquet(unseen_parquet_path, index=False)
        unseen_texts = df_unseen["text"].fillna("").astype(str).tolist() if len(df_unseen) > 0 else []
        N_unseen = len(unseen_texts)
        y_unseen = np.array([LAB2I[m] for m in df_unseen["model"].values], dtype=int) if N_unseen > 0 else np.array([], dtype=int)
        groups_unseen = df_unseen["prompt_id"].values if N_unseen > 0 and "prompt_id" in df_unseen.columns else np.ones(N_unseen, dtype=int)
    else:
        if N_test > 100 and bool(cfg.get("lock_prior_weights", True)):
            print("[Pipeline] Hidden Leaderboard Rerun (N_test > 100 with locked priors): Skipping unseen holdout extraction to maximize GPU throughput.")
        elif not enable_ablation_probe:
            print("[Pipeline] Frozen Production Mode (enable_ablation_probe=False): Skipping unseen holdout extraction to save GPU time.")
        df_unseen = pd.DataFrame()
        unseen_texts = []
        N_unseen = 0
        y_unseen = np.array([], dtype=int)
        groups_unseen = np.array([], dtype=int)

    auditor = UnseenStepwiseAuditor(y_unseen=y_unseen, groups_unseen=groups_unseen, N_test=N_test)

    # Discover fine-tuned checkpoints early for pre-flight validation
    ckpts = find_checkpoints(input_dir)
    print(f"[Pipeline] Discovered checkpoints ({len(ckpts)}): {list(ckpts.keys())}")

    req_assets = cfg.get("required_assets", {})
    cfg_model_weights = cfg.get("model_weights", {})
    DEFAULT_ALLOWED_CKPTS = {
        "qwen3_14b_p2", "qwen3_14b_v5", "qwen3_14b_inhouse",
        "deberta_p2", "deberta_inhouse",
        "qwen3_8b_claude",
        "qwen3_14b_gpt", "qwen3_14b_gemini", "qwen3_14b_glm",
    }
    if "required_checkpoints" in req_assets or "optional_checkpoints" in req_assets:
        allowed_ckpts = set(req_assets.get("required_checkpoints", [])) | set(req_assets.get("optional_checkpoints", []))
    elif cfg_model_weights:
        allowed_ckpts = set(cfg_model_weights.keys())
    else:
        allowed_ckpts = DEFAULT_ALLOWED_CKPTS

    for k in list(ckpts.keys()):
        if k not in allowed_ckpts:
            print(f"[Pipeline] Pruning non-whitelisted checkpoint '{k}' at {ckpts[k]}")
            ckpts.pop(k, None)
    print(f"[Pipeline] Active whitelisted checkpoints ({len(ckpts)}): {list(ckpts.keys())}")

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

    family_spec_logits_te: Dict[str, np.ndarray] = {}
    family_spec_logits_unseen: Dict[str, np.ndarray] = {}
    family_spec_classes: Dict[str, list] = {}

    # 2. MODULE 1: Subprocess Dual-Output Lexical Engine
    run_lex_script = working_dir / "run_lexical.py"
    if run_lex_script.exists() and not is_smoke:
        print("[Pipeline] Module 1: Launching subprocess run_lexical.py...")
        subprocess.run([sys.executable, str(run_lex_script)], check=True)
        log_p_svc_main = np.load(working_dir / "log_p_svc_main.npy")
        p_lex_comp = np.load(working_dir / "p_lex_comp.npy")
        D_tfidf = np.load(working_dir / "D_tfidf.npy")
        groups_085 = np.load(working_dir / "groups_085.npy")
        log_p_handwriting = np.load(working_dir / "log_p_handwriting.npy") if (working_dir / "log_p_handwriting.npy").exists() else None
        rule_flags_test = np.load(working_dir / "rule_flags_test.npy") if (working_dir / "rule_flags_test.npy").exists() else None
        unseen_log_p_svc_main = np.load(working_dir / "unseen_log_p_svc_main.npy") if (working_dir / "unseen_log_p_svc_main.npy").exists() else np.zeros((0, K), dtype=np.float32)
        unseen_p_lex_comp = np.load(working_dir / "unseen_p_lex_comp.npy") if (working_dir / "unseen_p_lex_comp.npy").exists() else np.zeros((0, K), dtype=np.float32)
    else:
        print("[Pipeline] Module 1: Executing run_lexical_engine directly...")
        tok_paths = list(input_dir.rglob("vocab.json"))
        qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""
        lex_out = run_lexical_engine(train_path, test_path, all_pool_files or pool_file, qwen_tok_dir, working_dir)
        log_p_svc_main = lex_out["log_p_svc_main"]
        p_lex_comp = lex_out["p_lex_comp"]
        D_tfidf = lex_out["D_tfidf"]
        groups_085 = lex_out["groups_085"]
        log_p_handwriting = lex_out.get("log_p_handwriting")
        rule_flags_test = lex_out.get("rule_flags_test")
        unseen_log_p_svc_main = lex_out.get("unseen_log_p_svc_main", np.zeros((0, K), dtype=np.float32))
        unseen_p_lex_comp = lex_out.get("unseen_p_lex_comp", np.zeros((0, K), dtype=np.float32))

    print(f"[Pipeline] Module 1 complete: log_p_svc_main shape = {log_p_svc_main.shape}")

    # Read configuration hierarchies from configs/production_ensemble.json
    main_cfg = cfg.get("main_stack", {})
    comp_cfg = cfg.get("complementary_stack", {})
    fusion_cfg = cfg.get("fusion", {})
    post_cfg = cfg.get("postprocess", {})
    model_weights = cfg.get("model_weights", {})

    svc_weight = model_weights.get("svc_4view", {}).get("plain_weight", main_cfg.get("linear_svc_weight", 1.00))
    main_members = main_cfg.get("members", {})
    comp_streams = comp_cfg.get("stream_weights", {})
    comp_members = comp_cfg.get("lora_members", {})
    lex_comp_weight = model_weights.get("lexical_5v", {}).get("plain_weight", comp_streams.get("lexical", 0.20))
    se_floors = comp_cfg.get("se_floors", {
        "qwen3_14b_p2": 0.10, "qwen25_14b_p2": 0.10, "deberta_p2": 0.10,
        "qwen3_14b_p1": 0.35, "instyle": 0.35, "qwen3_8b_claude": 0.35
    })
    prior_scales = comp_cfg.get("prior_scales", PRIOR_SCALES)
    dp_weights = comp_cfg.get("dp_weights", {
        "q3_14b": 0.38, "q25_14b": 0.24, "deberta": 0.20, "instyle": 0.10, "part1_14b": 0.08
    })
    twin_res = comp_cfg.get("twin_resolutions", {
        "t85": 0.50, "t72": 0.30, "t60": 0.20
    })
    w85 = float(twin_res.get("t85", 0.50))
    w72 = float(twin_res.get("t72", 0.30))
    w60 = float(twin_res.get("t60", 0.20))
    max_cl = post_cfg.get("max_cluster_size", 14)

    LOGIT_MAIN = log_p_svc_main * svc_weight
    COMP_LIST: List[Tuple[float, np.ndarray]] = [
        (lex_comp_weight, p_lex_comp)
    ]
    u_lp_svc = unseen_log_p_svc_main if (N_unseen > 0 and len(unseen_log_p_svc_main) == N_unseen) else np.zeros((N_unseen, K), dtype=np.float32)
    auditor.add_model(
        name="svc_4view",
        tier="Main",
        weight=svc_weight,
        unseen_log_probs=u_lp_svc,
        test_log_probs=log_p_svc_main,
    )
    u_p_lex = unseen_p_lex_comp if (N_unseen > 0 and len(unseen_p_lex_comp) == N_unseen) else (np.ones((N_unseen, K), dtype=np.float32) / float(K))
    auditor.add_model(
        name="lexical_5v",
        tier="Comp",
        weight=lex_comp_weight,
        unseen_log_probs=np.log(np.clip(u_p_lex, 1e-9, 1.0)),
        test_log_probs=np.log(np.clip(p_lex_comp, 1e-9, 1.0)),
    )
    claude_spec_logits = None
    claude_classes = None

    D_neural_sum = np.zeros((N_test, N_test), dtype=np.float64)
    D_neural_weight_total = 0.0
    D_hybrid = D_tfidf.copy()

    if N_test >= 2:
        Z_link_init = linkage(squareform(D_tfidf, checks=False), method="average")
        groups_085 = split_oversized_clusters_dist(D_tfidf, fcluster(Z_link_init, t=0.85, criterion="distance"), max_cluster_size=max_cl)
        groups_072 = split_oversized_clusters_dist(D_tfidf, fcluster(Z_link_init, t=0.72, criterion="distance"), max_cluster_size=max_cl)
        groups_065 = split_oversized_clusters_dist(D_tfidf, fcluster(Z_link_init, t=0.65, criterion="distance"), max_cluster_size=max_cl)
        groups_060 = split_oversized_clusters_dist(D_tfidf, fcluster(Z_link_init, t=0.60, criterion="distance"), max_cluster_size=max_cl)
    else:
        groups_085 = np.ones(N_test, dtype=int)
        groups_072 = np.ones(N_test, dtype=int)
        groups_065 = np.ones(N_test, dtype=int)
        groups_060 = np.ones(N_test, dtype=int)

    def update_hybrid_groups(Z_te_mat: np.ndarray, ckpt_scale: float):
        nonlocal D_neural_sum, D_neural_weight_total, D_hybrid, groups_085, groups_072, groups_065, groups_060
        if N_test >= 2:
            Z_c = Z_te_mat - Z_te_mat.mean(axis=0, keepdims=True)
            Z_norm = Z_c / np.linalg.norm(Z_c, axis=1, keepdims=True).clip(min=1e-6)
            D_cos = np.clip(1.0 - (Z_norm @ Z_norm.T), 0.0, None)
            np.fill_diagonal(D_cos, 0.0)
            w_d = float(ckpt_scale)
            D_neural_sum += w_d * D_cos
            D_neural_weight_total += w_d
            D_neural_avg = D_neural_sum / max(1e-6, D_neural_weight_total)
            D_hybrid = 0.55 * D_tfidf + 0.45 * D_neural_avg
            Z_link = linkage(squareform(D_hybrid, checks=False), method="average")
            groups_085 = split_oversized_clusters_dist(D_hybrid, fcluster(Z_link, t=0.85, criterion="distance"), max_cluster_size=max_cl)
            groups_072 = split_oversized_clusters_dist(D_hybrid, fcluster(Z_link, t=0.72, criterion="distance"), max_cluster_size=max_cl)
            groups_065 = split_oversized_clusters_dist(D_hybrid, fcluster(Z_link, t=0.65, criterion="distance"), max_cluster_size=max_cl)
            groups_060 = split_oversized_clusters_dist(D_hybrid, fcluster(Z_link, t=0.60, criterion="distance"), max_cluster_size=max_cl)

    has_neural = len(ckpts) > 0 and not is_smoke
    spectral_blocks_tr: List[np.ndarray] = []
    spectral_blocks_te: List[np.ndarray] = []
    spectral_blocks_unseen: List[np.ndarray] = []
    dp_posteriors: List[Tuple[float, np.ndarray]] = []
    dp_posteriors_unseen: List[Tuple[float, np.ndarray]] = []
    comp_lora_posteriors: List[Tuple[float, np.ndarray]] = []
    claude_spec_logits_unseen: Optional[np.ndarray] = None
    P_spectral = None
    P_spectral_unseen = None

    df_train = pd.read_csv(train_path)
    train_sub_texts: List[str] = []
    y_sub: np.ndarray = np.array([LAB2I[m] for m in df_train["model"].values], dtype=int) if "model" in df_train.columns else np.array([], dtype=int)
    prompts_sub: np.ndarray = df_train["prompt_id"].values if "prompt_id" in df_train.columns else np.array([])
    N_tr_sub: int = 0

    if has_neural:
        import torch

        rng = np.random.RandomState(42)
        u_p = df_train["prompt_id"].unique()
        max_wpca_prompts = int(cfg.get("wpca_max_prompts", len(u_p)))
        sel_prompts = u_p if len(u_p) <= max_wpca_prompts else rng.choice(u_p, size=max_wpca_prompts, replace=False)
        sub_idx = df_train[df_train["prompt_id"].isin(sel_prompts)].index.values
        train_sub = df_train.loc[sub_idx].reset_index(drop=True)
        train_sub_texts = train_sub["text"].fillna("").astype(str).tolist()
        y_sub = np.array([LAB2I[m] for m in train_sub["model"]])
        prompts_sub = train_sub["prompt_id"].values
        N_tr_sub = len(train_sub)
        print(f"[Pipeline] Selected {N_tr_sub} training rows ({len(sel_prompts)} prompts) for wPCA")

        spectral_blocks_tr: List[np.ndarray] = []
        spectral_blocks_te: List[np.ndarray] = []
        spectral_blocks_unseen: List[np.ndarray] = []
        dp_posteriors: List[Tuple[float, np.ndarray]] = []
        dp_posteriors_unseen: List[Tuple[float, np.ndarray]] = []
        comp_lora_posteriors: List[Tuple[float, np.ndarray]] = []
        family_spec_logits_te: Dict[str, np.ndarray] = {}
        family_spec_logits_unseen: Dict[str, np.ndarray] = {}
        family_spec_classes: Dict[str, list] = {}

        # ==============================================================================
        # TRUNK 1: Qwen3-14B (40 layers, hidden=5120)
        # ==============================================================================
        base_q3_14b = verified_bases.get("qwen3_14b") or find_base_model(input_dir, "qwen3", 5120, 40)
        if base_q3_14b and any(k in ckpts for k in ("qwen3_14b_p2", "qwen3_14b_v5", "qwen3_14b_inhouse", "qwen3_14b_gpt", "qwen3_14b_gemini", "qwen3_14b_glm")):
            print(f"[Pipeline] Loading Trunk 1: Qwen3-14B from {base_q3_14b}...")
            trunk1 = UniversalLoRATrunkLoader(base_q3_14b, keep_layers=27, layers=[2, 4, 6, 9, 13, 17, 22, 27], max_len=447)

            # Checkpoint 1A: Part 2 (lora_qwen3_14b)
            if "qwen3_14b_p2" in ckpts:
                p2_dir = ckpts["qwen3_14b_p2"]
                meta_p2 = read_checkpoint_meta(p2_dir, default_keep=27, default_layers=[2, 4, 6, 9, 13, 17, 22, 27], default_max_len=447)
                trunk1.set_active_layers(meta_p2["layers"], meta_p2["max_len"])
                backup_t1 = trunk1.snapshot_weights(p2_dir)
                merged = merge_lora(trunk1.model, p2_dir, sign=+1.0)
                if not merged and not is_smoke:
                    raise RuntimeError(f"Failed to merge required LoRA adapter at {p2_dir}")
                heads_sd = load_head_file(p2_dir)
                trunk1.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
                if W_p is None and not is_smoke:
                    raise RuntimeError(f"Required classification head W_p is None for {p2_dir}")
                p2_labels = meta_p2["labels"]

                Z_all = trunk1.extract_exact(test_texts + unseen_texts + train_sub_texts, batch_size=8, end_id=meta_p2["end_id"])
                Z_te = Z_all[:N_test]
                Z_unseen = Z_all[N_test : N_test + N_unseen]
                Z_tr = Z_all[N_test + N_unseen :]
                update_hybrid_groups(Z_te, prior_scales.get("qwen3_14b_p2", 1.00))

                # Main Stack contributions
                w_main_p = model_weights.get("qwen3_14b_p2", {}).get("plain_weight", main_members.get("lora_qwen3_14b", {}).get("plain_weight", 0.42))
                w_main_c = model_weights.get("qwen3_14b_p2", {}).get("twin_weight", main_members.get("lora_qwen3_14b", {}).get("twin_weight", 0.12))
                lp_u_p = np.zeros((0, K), dtype=np.float32)
                lp_u_c = np.zeros((0, K), dtype=np.float32)
                lp_c_blend = np.zeros((N_test, K), dtype=np.float32)
                if W_p is not None:
                    lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, p2_labels))
                    LOGIT_MAIN += lp_p * w_main_p
                    w_comp_p = comp_members.get("lora_qwen3_14b", {}).get("plain_weight", 0.30)
                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                    if N_unseen > 0:
                        lp_u_p = log_softmax(align_cols(Z_unseen @ W_p.T + b_p, p2_labels))

                if W_c is not None:
                    lp_c85, valid_c85 = compute_centered_logits(Z_te, groups_085, W_c, b_c, p2_labels)
                    LOGIT_MAIN[valid_c85] += lp_c85[valid_c85] * w_main_c
                    lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, p2_labels)
                    lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, p2_labels)
                    lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                    w_comp_c = comp_members.get("lora_qwen3_14b", {}).get("twin_weight", 0.10)
                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))
                    if N_unseen > 0:
                        lp_u_c, _ = compute_centered_logits(Z_unseen, groups_unseen, W_c, b_c, p2_labels)

                if W_p is not None:
                    w_tot_14b = w_main_p + (w_main_c if W_c is not None else 0.0)
                    lp_model_u = _combine_plain_and_twin_lp(lp_u_p, lp_u_c if W_c is not None else None, w_main_p, w_main_c) if N_unseen > 0 else np.zeros((0, K), dtype=np.float32)
                    lp_model_te = _combine_plain_and_twin_lp(lp_p, lp_c_blend if W_c is not None else None, w_main_p, w_main_c)
                    auditor.add_model("qwen3_14b_p2", tier="Main", weight=w_tot_14b, unseen_log_probs=lp_model_u, test_log_probs=lp_model_te)

                # wPCA + DP Posterior + Spectral Bank
                res_p2 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen3_14b_p2", 0.10))
                U_te = transform_lowram(Z_te, res_p2)
                lp_dp = scores_exact_dp(U_te, groups_065, res_p2, D_hybrid)
                dp_posteriors.append((dp_weights.get("q3_14b", 0.38), softmax(lp_dp, axis=1)))

                S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_p2, prior_scales.get("qwen3_14b_p2", 1.00))
                spectral_blocks_tr.append(S_tr)
                spectral_blocks_te.append(S_te)
                if N_unseen > 0 and "Z_unseen" in locals() and Z_unseen is not None:
                    U_u = transform_lowram(Z_unseen, res_p2)
                    lp_dp_u = scores_exact_dp(U_u, groups_unseen, res_p2, None)
                    dp_posteriors_unseen.append((dp_weights.get("q3_14b", 0.38), softmax(lp_dp_u, axis=1)))
                    _, S_u = make_spectral_blocks(Z_tr, Z_unseen, res_p2, prior_scales.get("qwen3_14b_p2", 1.00))
                    spectral_blocks_unseen.append(S_u)
                    del U_u, S_u

                trunk1.restore_weights(backup_t1)
                del backup_t1, Z_all, Z_tr, Z_te, heads_sd, res_p2, U_te
                if "Z_unseen" in locals():
                    del Z_unseen
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Checkpoint 1B / Optional Whitelisted Checkpoints on Trunk 1
            for opt_14b_key in ("qwen3_14b_v5", "qwen3_14b_inhouse"):
                if opt_14b_key in ckpts:
                    primary_ts14_dir = ckpts[opt_14b_key]
                    prefix_14b = "q14v5" if "v5" in opt_14b_key else "ts_q3b14" if "ts" in opt_14b_key else opt_14b_key
                    found_14b_folds = find_fold_adapters(input_dir, prefix_14b)

                    def _is_fold_14b(p: Path) -> bool:
                        nm = p.name.lower()
                        return any(f"_f{i}" in nm or f"-f{i}" in nm or f"_fold{i}" in nm for i in range(10))

                    folds_only_14b = sorted([p for p in found_14b_folds if _is_fold_14b(p)], key=lambda x: str(x).lower())
                    full_only_14b = [p for p in found_14b_folds if not _is_fold_14b(p)]
                    if not full_only_14b:
                        full_only_14b = [primary_ts14_dir]
                    fold_dirs_14b = folds_only_14b + full_only_14b
                    ts14_dir = fold_dirs_14b[-1]

                    def_keep_14b = 27 if opt_14b_key == "qwen3_14b_inhouse" else 24
                    def_layers_14b = [2, 4, 6, 9, 13, 17, 22, 27] if opt_14b_key == "qwen3_14b_inhouse" else [2, 4, 6, 8, 12, 16, 20, 24]
                    meta_ts14 = read_checkpoint_meta(ts14_dir, default_keep=def_keep_14b, default_layers=def_layers_14b, default_max_len=447)
                    if opt_14b_key == "qwen3_14b_inhouse" and meta_ts14.get("keep", 24) < 27:
                        print(f"[Pipeline] Skipping qwen3_14b_inhouse because keep={meta_ts14.get('keep', 24)} < 27 (old 24L checkpoint)")
                        continue

                    trunk1.set_active_layers(meta_ts14["layers"], meta_ts14["max_len"])
                    trunk1.set_final_norm(not meta_ts14["no_final_norm"])
                    backup_t1_ts = trunk1.snapshot_weights(ts14_dir)
                    merged = merge_lora(trunk1.model, ts14_dir, sign=+1.0)
                    if not merged:
                        print(f"[Pipeline] Optional checkpoint {opt_14b_key} failed to merge LoRA; skipping.")
                        trunk1.restore_weights(backup_t1_ts)
                        trunk1.set_final_norm(False)
                    else:
                        heads_sd = load_head_file(ts14_dir)
                        trunk1.load_norm_weights(heads_sd)
                        W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
                        ts14_labels = meta_ts14["labels"]
                        p_comp_def = 0.12 if opt_14b_key == "qwen3_14b_p1" else 0.22
                        c_comp_def = 0.04 if opt_14b_key == "qwen3_14b_p1" else 0.08
                        comp_alias = "lora14b_v1" if opt_14b_key == "qwen3_14b_p1" else opt_14b_key
                        w_comp_p = model_weights.get(opt_14b_key, {}).get("plain_weight", comp_members.get(comp_alias, {}).get("plain_weight", p_comp_def))
                        w_comp_c = model_weights.get(opt_14b_key, {}).get("twin_weight", comp_members.get(comp_alias, {}).get("twin_weight", c_comp_def))
                        w_comp_tot = w_comp_p + w_comp_c

                        # Stage A: Fast 360-row probe on df_unseen
                        if N_unseen >= 36 and W_p is not None:
                            Z_unseen = trunk1.extract_exact(unseen_texts, batch_size=8, end_id=meta_ts14["end_id"])

                            # Auto-align label order: test canonical, alphabetical, and meta labels
                            cand_orders = [ts14_labels]
                            if sorted(LABELS) not in cand_orders:
                                cand_orders.append(sorted(LABELS))
                            if list(LABELS) not in cand_orders:
                                cand_orders.append(list(LABELS))

                            best_align_order = ts14_labels
                            best_solo_bacc = -1.0
                            best_lp_u = None
                            for cand_o in cand_orders:
                                try:
                                    trial_lp_p = log_softmax(align_cols(Z_unseen @ W_p.T + b_p, cand_o))
                                    if W_c is not None:
                                        trial_lp_c, _ = compute_centered_logits(Z_unseen, groups_unseen, W_c, b_c, cand_o)
                                        trial_lp = _combine_plain_and_twin_lp(trial_lp_p, trial_lp_c, w_comp_p, w_comp_c)
                                    else:
                                        trial_lp = trial_lp_p
                                    trial_acc = float(balanced_accuracy_score(y_unseen, trial_lp.argmax(axis=1)))
                                    if trial_acc > best_solo_bacc:
                                        best_solo_bacc = trial_acc
                                        best_align_order = cand_o
                                        best_lp_u = trial_lp
                                except Exception:
                                    continue

                            solo_bacc = best_solo_bacc if best_solo_bacc >= 0 else 0.0
                            lp_u = best_lp_u if best_lp_u is not None else log_softmax(align_cols(Z_unseen @ W_p.T + b_p, ts14_labels))
                            ts14_labels = best_align_order
                            align_name = "canonical" if best_align_order == list(LABELS) else "alphabetical" if best_align_order == sorted(LABELS) else "custom"
                            print(f"[Fast Probe] {opt_14b_key}: solo unseen BAcc = {solo_bacc:.4f} (alignment: {align_name})")

                            if solo_bacc < probe_gate_min_bacc:
                                if opt_14b_key in ("qwen3_14b_ts", "qwen3_14b_v5", "qwen3_14b_p1"):
                                    print(f"[Fast Probe] {opt_14b_key}: solo BAcc {solo_bacc:.4f} < {probe_gate_min_bacc:.2f} -> deferring to auto_align_and_verify_head on Z_tr...")
                                    W_p = None
                                    W_c = None
                                else:
                                    print(f"[Fast Probe Pruning] {opt_14b_key}: solo BAcc {solo_bacc:.4f} < {probe_gate_min_bacc:.2f} -> SKIPPING 5,620-row test/train extraction!")
                                    trunk1.restore_weights(backup_t1_ts)
                                    trunk1.set_final_norm(False)
                                    del backup_t1_ts, heads_sd, Z_unseen
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue
                            elif len(auditor.models) > 0:
                                lift = auditor.evaluate_marginal_lift(candidate_unseen_lp=lp_u, weight=w_comp_tot)
                                if lift < -0.002:
                                    print(f"[Fast Probe Pruning] {opt_14b_key}: marginal lift {lift:+.4f} < -0.002 -> SKIPPING 5,620-row test/train extraction!")
                                    trunk1.restore_weights(backup_t1_ts)
                                    trunk1.set_final_norm(False)
                                    del backup_t1_ts, heads_sd, Z_unseen
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue
                                print(f"[Fast Probe] {opt_14b_key}: passed probe (solo={solo_bacc:.4f}, lift={lift:+.4f}) -> extracting test/train...")

                        # Stage B: Extract test and train (with multi-fold averaging if multiple folds discovered)
                        num_14b_folds = len(fold_dirs_14b)
                        if num_14b_folds > 1:
                            trunk1.restore_weights(backup_t1_ts)
                            del backup_t1_ts

                            lp_p_acc = np.zeros((N_test, K), dtype=np.float64)
                            lp_c_acc = np.zeros((N_test, K), dtype=np.float64)
                            lp_u_p_acc = np.zeros((N_unseen, K), dtype=np.float64) if N_unseen > 0 else None
                            lp_u_c_acc = np.zeros((N_unseen, K), dtype=np.float64) if N_unseen > 0 else None
                            has_any_c = False
                            Z_te = None
                            Z_tr = None
                            print(f"[Pipeline] {opt_14b_key}: extracting & averaging across {num_14b_folds} fold adapters in log-space...")

                            for f_idx, f_dir in enumerate(fold_dirs_14b):
                                is_last = (f_idx == num_14b_folds - 1)
                                f_meta = read_checkpoint_meta(f_dir, default_keep=def_keep_14b, default_layers=def_layers_14b, default_max_len=447)
                                trunk1.set_active_layers(f_meta["layers"], f_meta["max_len"])
                                trunk1.set_final_norm(not f_meta["no_final_norm"])
                                f_backup = trunk1.snapshot_weights(f_dir)
                                f_merged = merge_lora(trunk1.model, f_dir, sign=+1.0)
                                if not f_merged:
                                    trunk1.restore_weights(f_backup)
                                    continue
                                f_heads_sd = load_head_file(f_dir)
                                trunk1.load_norm_weights(f_heads_sd)
                                f_W_p, f_b_p, f_W_c, f_b_c = resolve_head_tensors(f_heads_sd)
                                f_labels = f_meta["labels"]

                                if is_last:
                                    if "Z_unseen" in locals() and Z_unseen is not None and len(Z_unseen) == N_unseen and N_unseen > 0:
                                        extract_batch = test_texts + train_sub_texts
                                        Z_te_tr = trunk1.extract_exact(extract_batch, batch_size=8, end_id=f_meta["end_id"])
                                        f_Z_te = Z_te_tr[:N_test]
                                        f_Z_u = Z_unseen
                                        Z_tr = Z_te_tr[N_test:]
                                        Z_te = f_Z_te
                                        del Z_te_tr
                                    else:
                                        extract_batch = test_texts + (unseen_texts if N_unseen > 0 else []) + train_sub_texts
                                        Z_te_u_tr = trunk1.extract_exact(extract_batch, batch_size=8, end_id=f_meta["end_id"])
                                        f_Z_te = Z_te_u_tr[:N_test]
                                        f_Z_u = Z_te_u_tr[N_test : N_test + N_unseen] if N_unseen > 0 else None
                                        Z_tr = Z_te_u_tr[N_test + N_unseen :]
                                        Z_te = f_Z_te
                                        del Z_te_u_tr
                                else:
                                    extract_batch = test_texts + (unseen_texts if N_unseen > 0 else [])
                                    Z_te_u = trunk1.extract_exact(extract_batch, batch_size=8, end_id=f_meta["end_id"])
                                    f_Z_te = Z_te_u[:N_test]
                                    f_Z_u = Z_te_u[N_test : N_test + N_unseen] if N_unseen > 0 else None
                                    del Z_te_u

                                if f_W_p is not None:
                                    raw_p_f = align_cols(f_Z_te @ f_W_p.T + f_b_p, f_labels)
                                    std_f = float(np.std(raw_p_f, axis=1).mean())
                                    temp_f = max(1.0, std_f / 4.2)
                                    lp_p_acc += log_softmax(raw_p_f / temp_f, axis=1) / num_14b_folds
                                    if N_unseen > 0 and f_Z_u is not None:
                                        raw_u_f = align_cols(f_Z_u @ f_W_p.T + f_b_p, f_labels)
                                        lp_u_p_acc += log_softmax(raw_u_f / temp_f, axis=1) / num_14b_folds
                                if f_W_c is not None:
                                    has_any_c = True
                                    lp_c85_f, _ = compute_centered_logits(f_Z_te, groups_085, f_W_c, f_b_c, f_labels)
                                    lp_c72_f, _ = compute_centered_logits(f_Z_te, groups_072, f_W_c, f_b_c, f_labels)
                                    lp_c60_f, _ = compute_centered_logits(f_Z_te, groups_060, f_W_c, f_b_c, f_labels)
                                    c_blend_f = w85 * lp_c85_f + w72 * lp_c72_f + w60 * lp_c60_f
                                    lp_c_acc += c_blend_f / num_14b_folds
                                    if N_unseen > 0 and f_Z_u is not None:
                                        lp_u_c_f, _ = compute_centered_logits(f_Z_u, groups_unseen, f_W_c, f_b_c, f_labels)
                                        lp_u_c_acc += lp_u_c_f / num_14b_folds

                                trunk1.restore_weights(f_backup)
                                del f_backup, f_heads_sd, f_Z_te
                                if "f_Z_u" in locals() and f_Z_u is not None:
                                    del f_Z_u
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                            trunk1.set_final_norm(False)
                            lp_p = lp_p_acc
                            lp_c_blend = lp_c_acc if has_any_c else np.zeros((N_test, K), dtype=np.float32)
                            if N_unseen > 0 and lp_u_p_acc is not None:
                                lp_u = _combine_plain_and_twin_lp(lp_u_p_acc, lp_u_c_acc if has_any_c else None, w_comp_p, w_comp_c)
                            gate_passed = True
                            backup_t1_ts = None
                        else:
                            if N_unseen >= 36 and W_p is not None:
                                Z_te_tr = trunk1.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=meta_ts14["end_id"])
                                Z_te = Z_te_tr[:N_test]
                                Z_tr = Z_te_tr[N_test:]
                                del Z_te_tr
                            else:
                                Z_all = trunk1.extract_exact(test_texts + unseen_texts + train_sub_texts, batch_size=8, end_id=meta_ts14["end_id"])
                                Z_te = Z_all[:N_test]
                                Z_unseen = Z_all[N_test : N_test + N_unseen]
                                Z_tr = Z_all[N_test + N_unseen :]
                                del Z_all

                            gate_passed = True
                            if not is_smoke and len(y_sub) >= 180:
                                gate_passed, W_p, b_p, W_c, b_c, ts14_labels, _ = auto_align_and_verify_head(
                                    Z_tr=Z_tr,
                                    W_p=W_p,
                                    b_p=b_p,
                                    W_c=W_c,
                                    b_c=b_c,
                                    ckpt_labels=ts14_labels,
                                    y_sub=y_sub,
                                    prompts_sub=prompts_sub,
                                    name=opt_14b_key,
                                    min_bacc=probe_gate_min_bacc,
                                )

                            if gate_passed and N_unseen > 0 and W_p is not None and "Z_unseen" in locals():
                                raw_u_p = align_cols(Z_unseen @ W_p.T + b_p, ts14_labels)
                                std_u = float(np.std(raw_u_p, axis=1).mean())
                                temp_u = max(1.0, std_u / 4.2)
                                lp_u_p_fit = log_softmax(raw_u_p / temp_u, axis=1)
                                if W_c is not None:
                                    lp_u_c_fit, _ = compute_centered_logits(Z_unseen, groups_unseen, W_c, b_c, ts14_labels)
                                    lp_u_fitted = _combine_plain_and_twin_lp(lp_u_p_fit, lp_u_c_fit, w_comp_p, w_comp_c)
                                else:
                                    lp_u_fitted = lp_u_p_fit
                                lp_u = lp_u_fitted
                                fitted_bacc = float(balanced_accuracy_score(y_unseen, lp_u_fitted.argmax(axis=1)))
                                print(f"[Post-Fit Probe] {opt_14b_key}: unseen BAcc = {fitted_bacc:.4f}")
                                if fitted_bacc < probe_gate_min_bacc:
                                    print(f"[Post-Fit Pruning] {opt_14b_key}: fitted unseen BAcc {fitted_bacc:.4f} < {probe_gate_min_bacc:.2f} -> rejecting checkpoint!")
                                    gate_passed = False
                                elif len(auditor.models) > 0:
                                    fitted_lift = auditor.evaluate_marginal_lift(candidate_unseen_lp=lp_u_fitted, weight=w_comp_tot)
                                    if fitted_lift < -0.002:
                                        print(f"[Post-Fit Pruning] {opt_14b_key}: fitted marginal lift {fitted_lift:+.4f} < -0.002 -> rejecting checkpoint!")
                                        gate_passed = False

                            if gate_passed:
                                if W_p is not None:
                                    raw_te = align_cols(Z_te @ W_p.T + b_p, ts14_labels)
                                    std_te = float(np.std(raw_te, axis=1).mean())
                                    temp_te = max(1.0, std_te / 4.2)
                                    lp_p = log_softmax(raw_te / temp_te, axis=1)
                                else:
                                    lp_p = None
                                if W_c is not None:
                                    lp_c85, _ = compute_centered_logits(Z_te, groups_085, W_c, b_c, ts14_labels)
                                    lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, ts14_labels)
                                    lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, ts14_labels)
                                    lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                                else:
                                    lp_c_blend = np.zeros((N_test, K), dtype=np.float32)

                        if gate_passed:
                            pr_sc = prior_scales.get(opt_14b_key, 0.50)
                            update_hybrid_groups(Z_te, pr_sc)
                            if num_14b_folds > 1:
                                comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                                if has_any_c:
                                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))
                            else:
                                if W_p is not None and lp_p is not None:
                                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                                if W_c is not None:
                                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))

                            def_dp_14b = 0.08 if opt_14b_key == "qwen3_14b_p1" else 0.10
                            def_se_14b = 0.35 if opt_14b_key == "qwen3_14b_p1" else 0.25
                            res_ts14 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get(opt_14b_key, def_se_14b))
                            U_te = transform_lowram(Z_te, res_ts14)
                            lp_dp = scores_exact_dp(U_te, groups_065, res_ts14, D_hybrid)
                            dp_posteriors.append((dp_weights.get(opt_14b_key, dp_weights.get("part1_14b" if opt_14b_key == "qwen3_14b_p1" else "q3_14b_ts", def_dp_14b)), softmax(lp_dp, axis=1)))

                            S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_ts14, pr_sc)
                            spectral_blocks_tr.append(S_tr)
                            spectral_blocks_te.append(S_te)
                            if N_unseen > 0 and "Z_unseen" in locals() and Z_unseen is not None:
                                U_u = transform_lowram(Z_unseen, res_ts14)
                                lp_dp_u = scores_exact_dp(U_u, groups_unseen, res_ts14, None)
                                dp_w_ts14 = dp_weights.get(opt_14b_key, dp_weights.get("part1_14b" if opt_14b_key == "qwen3_14b_p1" else "q3_14b_ts", def_dp_14b))
                                dp_posteriors_unseen.append((dp_w_ts14, softmax(lp_dp_u, axis=1)))
                                _, S_u = make_spectral_blocks(Z_tr, Z_unseen, res_ts14, pr_sc)
                                spectral_blocks_unseen.append(S_u)
                                del U_u, S_u
                            del res_ts14, U_te

                            if lp_p is not None:
                                w_tot_ts14 = w_comp_p + (w_comp_c if (has_any_c if num_14b_folds > 1 else W_c is not None) else 0.0)
                                lp_m_u = lp_u if N_unseen > 0 and 'lp_u' in locals() else np.zeros((0, K), dtype=np.float32)
                                lp_m_te = _combine_plain_and_twin_lp(lp_p, lp_c_blend if (has_any_c if num_14b_folds > 1 else W_c is not None) else None, w_comp_p, w_comp_c)
                                aud_name_14b = opt_14b_key
                                auditor.add_model(name=aud_name_14b, tier="Comp", weight=w_tot_ts14, unseen_log_probs=lp_m_u, test_log_probs=lp_m_te)
                        else:
                            print(f"[Pipeline] Optional checkpoint {opt_14b_key} rejected by gate; excluded from logits and spectral bank.")

                        if backup_t1_ts is not None:
                            trunk1.restore_weights(backup_t1_ts)
                            del backup_t1_ts
                        trunk1.set_final_norm(False)
                        del Z_tr, Z_te, heads_sd
                        if "Z_unseen" in locals():
                            del Z_unseen
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # Execute Family Specialists on Trunk 1 (Qwen3-14B)
            spec_candidates = [
                ("qwen3_14b_gpt", "gpt"),
                ("qwen3_14b_gemini", "gemini"),
                ("qwen3_14b_glm", "glm"),
            ]
            for ckpt_key, fam_name in spec_candidates:
                if ckpt_key in ckpts:
                    spec_p = ckpts[ckpt_key]
                    print(f"[Pipeline] Running Trunk 1 Family Specialist: {ckpt_key} ({fam_name}) from {spec_p}...")
                    te_l, u_l, cls_l = extract_family_specialist_logits(
                        trunk1=trunk1,
                        spec_dir=spec_p,
                        family_name=fam_name,
                        test_texts=test_texts,
                        unseen_texts=unseen_texts if N_unseen > 0 else [],
                        LOGIT_MAIN=LOGIT_MAIN,
                        auditor=auditor,
                        groups_test_multi=(groups_085, groups_072, groups_065),
                        groups_unseen=groups_unseen,
                    )
                    family_spec_logits_te[fam_name] = te_l
                    family_spec_logits_unseen[fam_name] = u_l
                    family_spec_classes[fam_name] = cls_l

            trunk1.free_memory()
            del trunk1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



        # ==============================================================================
        # TRUNK 3: Qwen3-8B (36 layers, hidden=4096)
        # ==============================================================================
        base_q3_8b = verified_bases.get("qwen3_8b") or find_base_model(input_dir, "qwen3", 4096, 36)
        if base_q3_8b and "qwen3_8b_claude" in ckpts:
            print(f"[Pipeline] Loading Trunk 3: Qwen3-8B from {base_q3_8b}...")
            trunk3 = UniversalLoRATrunkLoader(base_q3_8b, keep_layers=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], max_len=447)

            # Claude Head LoRA
            if "qwen3_8b_claude" in ckpts:
                claude_dir = ckpts["qwen3_8b_claude"]
                meta_cl = read_checkpoint_meta(claude_dir, default_keep=24, default_layers=[2, 4, 6, 8, 12, 16, 20, 24], default_max_len=447)
                trunk3.set_active_layers(meta_cl["layers"], meta_cl["max_len"])
                merged = merge_lora(trunk3.model, claude_dir, sign=+1.0)
                if not merged and not is_smoke:
                    raise RuntimeError(f"Failed to merge required LoRA adapter at {claude_dir}")

                heads_sd_cl = load_head_file(claude_dir)
                if heads_sd_cl:
                    trunk3.load_norm_weights(heads_sd_cl)

                W_p, b_p, W_c, b_c = None, None, None, None
                W_spec, b_spec = None, None
                ro_labels = None
                readouts_file = claude_dir / "readouts.npz"
                if readouts_file.exists():
                    with np.load(str(readouts_file), allow_pickle=True) as rnpz:
                        if "classes" in rnpz:
                            ro_labels = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in rnpz["classes"]]
                        elif "labels" in rnpz:
                            ro_labels = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in rnpz["labels"]]
                        for wk, bk in [("plain_coef", "plain_intercept"), ("plain.weight", "plain.bias")]:
                            if wk in rnpz and bk in rnpz and rnpz[wk].shape[0] == 18:
                                W_p, b_p = np.array(rnpz[wk]), np.array(rnpz[bk])
                                break
                        for wk, bk in [("center_coef", "center_intercept"), ("center.weight", "center.bias")]:
                            if wk in rnpz and bk in rnpz and rnpz[wk].shape[0] == 18:
                                W_c, b_c = np.array(rnpz[wk]), np.array(rnpz[bk])
                                break
                        if "claude_coef" in rnpz and "claude_intercept" in rnpz:
                            if rnpz["claude_coef"].shape[0] == 5:
                                W_spec, b_spec = np.array(rnpz["claude_coef"]), np.array(rnpz["claude_intercept"])
                                if "claude_classes" in rnpz:
                                    claude_classes = list(rnpz["claude_classes"])
                if W_p is None and heads_sd_cl:
                    cand_W, cand_b, cand_Wc, cand_bc = resolve_head_tensors(heads_sd_cl)
                    if cand_W is not None and cand_W.shape[0] == 18:
                        W_p, b_p = cand_W, cand_b
                        if cand_Wc is not None and cand_Wc.shape[0] == 18:
                            W_c, b_c = cand_Wc, cand_bc

                # Ensure ro_labels is a valid 18-class label list (defaults to sorted(LABELS) alphabetical order)
                if ro_labels is None or len(ro_labels) != 18 or set(ro_labels) != set(LABELS):
                    cand_meta = meta_cl.get("labels", [])
                    if len(cand_meta) == 18 and set(cand_meta) == set(LABELS):
                        ro_labels = list(cand_meta)
                    else:
                        ro_labels = sorted(LABELS)

                if W_spec is None:
                    spec_file = claude_dir / "claude_head.safetensors"
                    if spec_file.exists():
                        ch_sd = load_head_file(spec_file)
                        for wk, bk in [("weight", "bias"), ("plain.weight", "plain.bias")]:
                            if wk in ch_sd and ch_sd[wk].shape[0] == 5:
                                W_spec = ch_sd[wk].cpu().numpy()
                                b_spec = ch_sd[bk].cpu().numpy() if bk in ch_sd else np.zeros(5, dtype=np.float32)
                                break
                if W_spec is None and not is_smoke:
                    raise RuntimeError(f"Required specialist head W_spec is None for {claude_dir}")
                if claude_classes is None:
                    claude_classes = [LABELS[i] for i in FAMILIES["claude"]]

                Z_all = trunk3.extract_exact(test_texts + unseen_texts + train_sub_texts, batch_size=8, end_id=meta_cl["end_id"])
                Z_te = Z_all[:N_test]
                Z_unseen = Z_all[N_test : N_test + N_unseen]
                Z_tr = Z_all[N_test + N_unseen :]
                update_hybrid_groups(Z_te, prior_scales.get("qwen3_8b_claude", 0.30))

                if W_spec is not None and W_spec.shape[0] == 5:
                    claude_spec_logits = Z_te @ W_spec.T + b_spec
                    if N_unseen > 0 and "Z_unseen" in locals() and Z_unseen is not None:
                        claude_spec_logits_unseen = Z_unseen @ W_spec.T + b_spec

                if W_p is not None and W_p.shape[0] == 18:
                    w_comp_p = model_weights.get("qwen3_8b_claude", {}).get("plain_weight", comp_members.get("ntl_claude_head", {}).get("plain_weight", 0.0))
                    w_comp_c = model_weights.get("qwen3_8b_claude", {}).get("twin_weight", comp_members.get("ntl_claude_head", {}).get("twin_weight", 0.0))
                    lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, ro_labels))
                    if w_comp_p > 0.0:
                        comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                    if N_unseen > 0:
                        lp_u_p = log_softmax(align_cols(Z_unseen @ W_p.T + b_p, ro_labels))
                    else:
                        lp_u_p = np.zeros((0, K), dtype=np.float32)

                    if W_c is not None and W_c.shape[0] == 18:
                        lp_c85, _ = compute_centered_logits(Z_te, groups_085, W_c, b_c, ro_labels)
                        lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, ro_labels)
                        lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, ro_labels)
                        lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                        if w_comp_c > 0.0:
                            comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))
                        if N_unseen > 0:
                            lp_u_c, _ = compute_centered_logits(Z_unseen, groups_unseen, W_c, b_c, ro_labels)
                        else:
                            lp_u_c = np.zeros((0, K), dtype=np.float32)
                    else:
                        lp_c85 = np.zeros((N_test, K), dtype=np.float32)
                        lp_c_blend = np.zeros((N_test, K), dtype=np.float32)
                        lp_u_c = np.zeros((0, K), dtype=np.float32)

                    w_tot_cl = w_comp_p + (w_comp_c if (W_c is not None and W_c.shape[0] == 18) else 0.0)
                    if w_tot_cl > 0.0:
                        twin_c_u = lp_u_c if (W_c is not None and W_c.shape[0] == 18) else None
                        twin_c_te = lp_c_blend if (W_c is not None and W_c.shape[0] == 18) else None
                        lp_m_u = _combine_plain_and_twin_lp(lp_u_p, twin_c_u, w_comp_p, w_comp_c) if N_unseen > 0 else np.zeros((0, K), dtype=np.float32)
                        lp_m_te = _combine_plain_and_twin_lp(lp_p, twin_c_te, w_comp_p, w_comp_c)
                        auditor.add_model("qwen3_8b_claude", tier="Comp", weight=w_tot_cl, unseen_log_probs=lp_m_u, test_log_probs=lp_m_te)

                res_cl = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen3_8b_claude", 0.35))
                U_te = transform_lowram(Z_te, res_cl)
                lp_dp = scores_exact_dp(U_te, groups_065, res_cl, D_hybrid)
                dp_posteriors.append((dp_weights.get("claude_8b", 0.04), softmax(lp_dp, axis=1)))

                S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_cl, prior_scales.get("qwen3_8b_claude", 0.30))
                spectral_blocks_tr.append(S_tr)
                spectral_blocks_te.append(S_te)
                if N_unseen > 0 and "Z_unseen" in locals() and Z_unseen is not None:
                    U_u = transform_lowram(Z_unseen, res_cl)
                    lp_dp_u = scores_exact_dp(U_u, groups_unseen, res_cl, None)
                    dp_posteriors_unseen.append((dp_weights.get("claude_8b", 0.04), softmax(lp_dp_u, axis=1)))
                    _, S_u = make_spectral_blocks(Z_tr, Z_unseen, res_cl, prior_scales.get("qwen3_8b_claude", 0.30))
                    spectral_blocks_unseen.append(S_u)
                    del U_u, S_u

                merge_lora(trunk3.model, claude_dir, sign=-1.0)
                del Z_all, Z_tr, Z_te, res_cl, U_te
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            trunk3.free_memory()
            del trunk3
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 4: DeBERTa-v3-Large
        # ==============================================================================
        for deb_key in ("deberta_p2",):
            if deb_key not in ckpts:
                continue
            deb_dir = ckpts[deb_key]
            base_deb_dir = verified_bases.get("deberta_v3_large") or find_base_model(input_dir, "deberta", 1024, 24)
            print(f"[Pipeline] Loading Trunk 4: DeBERTa-v3 ({deb_key}) from {deb_dir}...")
            Z_test_all, Z_tr_deb, W_p_deb, b_p_deb, W_c_deb, b_c_deb, deb_labels = extract_deberta_checkpoint(
                deb_dir, base_deb_dir, test_texts + unseen_texts, train_sub_texts, y_sub
            )
            Z_te_deb = Z_test_all[:N_test]
            Z_unseen_deb = Z_test_all[N_test : N_test + N_unseen]

            if W_p_deb is None and not is_smoke:
                raise RuntimeError(f"Required classification head W_p is None for {deb_dir}")
            update_hybrid_groups(Z_te_deb, prior_scales.get(deb_key, 0.80))

            w_main_deb = model_weights.get(deb_key, {}).get("plain_weight", main_members.get("deberta_v3_large", {}).get("plain_weight", 0.25))
            w_comp_deb = comp_members.get("deberta_v3_large", {}).get("plain_weight", 0.18)
            w_comp_c = model_weights.get(deb_key, {}).get("twin_weight", comp_members.get("deberta_v3_large", {}).get("twin_weight", 0.06))

            if W_c_deb is None and len(Z_tr_deb) >= 90 and len(np.unique(prompts_sub)) >= 5:
                from sklearn.linear_model import LogisticRegression
                Z_c_deb = Z_tr_deb.astype(np.float64).copy()
                for p in np.unique(prompts_sub):
                    m_p = (prompts_sub == p)
                    if m_p.sum() > 1:
                        Z_c_deb[m_p] -= Z_c_deb[m_p].mean(axis=0, keepdims=True)
                sd_c = Z_c_deb.std(axis=0) + 1e-6
                clf_deb_c = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
                clf_deb_c.fit(Z_c_deb / sd_c, y_sub)
                if len(clf_deb_c.classes_) == K:
                    W_c_canon = (clf_deb_c.coef_ / sd_c[None, :]).astype(np.float32)
                    b_c_canon = clf_deb_c.intercept_.astype(np.float32)
                    inv_perm = [LABELS.index(l) for l in deb_labels]
                    W_c_deb = W_c_canon[inv_perm]
                    b_c_deb = b_c_canon[inv_perm]
                    print(f"[DeBERTa] Fitted prompt-centered twin readout on {len(Z_tr_deb)} rows (twin_weight={w_comp_c:.2f})")
            lp_p_deb = None
            lp_u_deb = np.zeros((0, K), dtype=np.float32)
            lp_c85_deb = np.zeros((N_test, K), dtype=np.float32)
            lp_c_blend = np.zeros((N_test, K), dtype=np.float32)
            lp_u_c_deb = np.zeros((0, K), dtype=np.float32)

            if W_p_deb is not None:
                lp_p_deb = log_softmax(align_cols(Z_te_deb @ W_p_deb.T + b_p_deb, deb_labels))
                LOGIT_MAIN += lp_p_deb * w_main_deb
                comp_lora_posteriors.append((w_comp_deb, softmax(lp_p_deb, axis=1)))
                if N_unseen > 0:
                    lp_u_deb = log_softmax(align_cols(Z_unseen_deb @ W_p_deb.T + b_p_deb, deb_labels))

            if W_c_deb is not None:
                lp_c85_deb, _ = compute_centered_logits(Z_te_deb, groups_085, W_c_deb, b_c_deb, deb_labels)
                lp_c72, _ = compute_centered_logits(Z_te_deb, groups_072, W_c_deb, b_c_deb, deb_labels)
                lp_c60, _ = compute_centered_logits(Z_te_deb, groups_060, W_c_deb, b_c_deb, deb_labels)
                lp_c_blend = w85 * lp_c85_deb + w72 * lp_c72 + w60 * lp_c60
                comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))
                if N_unseen > 0:
                    lp_u_c_deb, _ = compute_centered_logits(Z_unseen_deb, groups_unseen, W_c_deb, b_c_deb, deb_labels)

            if W_p_deb is not None and lp_p_deb is not None:
                w_tot_deb = w_main_deb + (w_comp_c if W_c_deb is not None else 0.0)
                lp_m_u = _combine_plain_and_twin_lp(lp_u_deb, lp_u_c_deb if W_c_deb is not None else None, w_main_deb, w_comp_c) if N_unseen > 0 else np.zeros((0, K), dtype=np.float32)
                lp_m_te = _combine_plain_and_twin_lp(lp_p_deb, lp_c_blend if W_c_deb is not None else None, w_main_deb, w_comp_c)
                aud_name = deb_key
                auditor.add_model(aud_name, tier="Main", weight=w_tot_deb, unseen_log_probs=lp_m_u, test_log_probs=lp_m_te)

            res_deb = fit_shared_gaussian_lowram(Z_tr_deb, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get(deb_key, 0.10))
            U_te = transform_lowram(Z_te_deb, res_deb)
            lp_dp = scores_exact_dp(U_te, groups_065, res_deb, D_hybrid)
            dp_posteriors.append((dp_weights.get("deberta", 0.20), softmax(lp_dp, axis=1)))

            S_tr, S_te = make_spectral_blocks(Z_tr_deb, Z_te_deb, res_deb, prior_scales.get(deb_key, 0.80))
            spectral_blocks_tr.append(S_tr)
            spectral_blocks_te.append(S_te)
            if N_unseen > 0 and "Z_unseen_deb" in locals() and Z_unseen_deb is not None:
                U_u = transform_lowram(Z_unseen_deb, res_deb)
                lp_dp_u = scores_exact_dp(U_u, groups_unseen, res_deb, None)
                dp_posteriors_unseen.append((dp_weights.get("deberta", 0.20), softmax(lp_dp_u, axis=1)))
                _, S_u = make_spectral_blocks(Z_tr_deb, Z_unseen_deb, res_deb, prior_scales.get(deb_key, 0.80))
                spectral_blocks_unseen.append(S_u)
                del U_u, S_u

            del Z_tr_deb, Z_te_deb, Z_test_all, res_deb, U_te
            if "Z_unseen_deb" in locals():
                del Z_unseen_deb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if "Z_unseen_deb" in locals():
                del Z_unseen_deb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Add Secondary LoRAs to COMP_LIST
        if comp_lora_posteriors:
            tot_w = sum(w for w, _ in comp_lora_posteriors)
            P_lora_comb = sum(w * P for w, P in comp_lora_posteriors) / max(1e-8, tot_w)
            COMP_LIST.append((comp_streams.get("lora_and_twins", 0.35), P_lora_comb))

        # Add DP Posteriors to COMP_LIST
        if dp_posteriors:
            tot_dp_w = sum(w for w, _ in dp_posteriors)
            P_dp_comb = sum(w * P_dp for w, P_dp in dp_posteriors) / max(1e-8, tot_dp_w)
            COMP_LIST.append((comp_streams.get("dp_posteriors", 0.30), P_dp_comb))

        # Fit Joint Spectral Bank
        if spectral_blocks_tr and spectral_blocks_te:
            sb_cfg = comp_cfg.get("spectral_bank", {})
            plain_C = sb_cfg.get("plain_C", 0.04)
            center_C = sb_cfg.get("center_C", 0.03)
            center_blend = sb_cfg.get("center_blend", 0.35)

            P_spectral, oof_score = fit_joint_spectral_bank(
                spectral_tr_blocks=spectral_blocks_tr,
                spectral_te_blocks=spectral_blocks_te,
                y_sub=y_sub,
                prompts_sub=prompts_sub,
                D_tfidf=D_tfidf,
                D_neural_sum=D_neural_sum,
                D_neural_weight_total=D_neural_weight_total,
                N_test=N_test,
                plain_C=plain_C,
                center_C=center_C,
                center_blend=center_blend,
                max_cluster_size=max_cl
            )
            print(f"[Pipeline] Joint Spectral Bank fitted (OOF BAcc = {oof_score:.4f})")
            COMP_LIST.append((comp_streams.get("spectral_bank", 0.15), P_spectral))
            if len(spectral_blocks_unseen) > 0 and N_unseen > 0:
                P_spectral_unseen, _ = fit_joint_spectral_bank(
                    spectral_blocks_tr,
                    spectral_blocks_unseen,
                    y_sub,
                    prompts_sub,
                    D_tfidf=None,
                    D_neural_sum=None,
                    D_neural_weight_total=0.0,
                    N_test=N_unseen,
                    plain_C=plain_C,
                    center_C=center_C,
                    center_blend=center_blend,
                    max_cluster_size=max_cl,
                    groups_override=groups_unseen,
                    compute_oof=False,
                )

    elif is_smoke:
        print("[Pipeline] Smoke mode active: synthesizing high-fidelity mock representations...")
        np.random.seed(42)
        mock_p_lora = softmax(np.random.randn(N_test, K), axis=1)
        mock_p_dp = softmax(np.random.randn(N_test, K), axis=1)
        mock_p_spec = softmax(np.random.randn(N_test, K), axis=1)
        COMP_LIST.append((comp_streams.get("lora_and_twins", 0.35), mock_p_lora))
        COMP_LIST.append((comp_streams.get("dp_posteriors", 0.30), mock_p_dp))
        COMP_LIST.append((comp_streams.get("spectral_bank", 0.15), mock_p_spec))
        dp_posteriors.append((comp_streams.get("dp_posteriors", 0.30), mock_p_dp))
        P_spectral = mock_p_spec

    # 4. MODULE 4: Flat Per-Model Log-Space Ensemble & Dynamic Unseen Calibration
    print("[Pipeline] Module 4: Flat Per-Model Log-Space Ensemble & Dynamic Unseen Calibration...")
    extra_streams: List[Tuple[str, float, np.ndarray]] = []
    extra_streams_unseen: List[Tuple[str, float, np.ndarray]] = []
    w_dp = model_weights.get("wpca_dp", {}).get("plain_weight", 0.35)
    if dp_posteriors:
        tot_dp_w = sum(w for w, _ in dp_posteriors)
        P_dp_comb = sum(w * P_dp for w, P_dp in dp_posteriors) / max(1e-8, tot_dp_w)
        extra_streams.append(("wpca_dp", w_dp, np.log(np.clip(P_dp_comb, 1e-9, 1.0))))
    elif is_smoke:
        extra_streams.append(("wpca_dp", w_dp, np.log(np.clip(mock_p_dp, 1e-9, 1.0))))

    if dp_posteriors_unseen:
        tot_dp_u_w = sum(w for w, _ in dp_posteriors_unseen)
        P_dp_u_comb = sum(w * P_dp for w, P_dp in dp_posteriors_unseen) / max(1e-8, tot_dp_u_w)
        extra_streams_unseen.append(("wpca_dp", w_dp, np.log(np.clip(P_dp_u_comb, 1e-9, 1.0))))

    w_sb = model_weights.get("spectral_bank", {}).get("plain_weight", 0.20)
    if "P_spectral" in locals() and P_spectral is not None:
        extra_streams.append(("spectral_bank", w_sb, np.log(np.clip(P_spectral, 1e-9, 1.0))))
    elif is_smoke:
        extra_streams.append(("spectral_bank", w_sb, np.log(np.clip(mock_p_spec, 1e-9, 1.0))))

    if "P_spectral_unseen" in locals() and P_spectral_unseen is not None:
        extra_streams_unseen.append(("spectral_bank", w_sb, np.log(np.clip(P_spectral_unseen, 1e-9, 1.0))))

    if len(auditor.models) > 0:
        optimize_on_unseen = (N_unseen >= 36 and not is_smoke)
        lock_priors = bool(cfg.get("lock_prior_weights", True))
        P_mix, active_weights_map, df_ablation = auditor.optimize_weights_and_blend(
            working_dir=working_dir,
            extra_test_log_probs=extra_streams,
            optimize_on_unseen=optimize_on_unseen,
            lock_prior_weights=lock_priors,
        )
        print(f"[Pipeline] Flat log-space ensemble active weights: {active_weights_map}")
        R_main = P_mix
    else:
        main_ratio = fusion_cfg.get("main_ratio", 0.85)
        comp_ratio = fusion_cfg.get("comp_ratio", 0.15)
        P_mix, R_main = blend_main_and_complementary(LOGIT_MAIN, COMP_LIST, main_ratio=main_ratio, comp_ratio=comp_ratio)

    # 4-Family Within-Family Spectral Gates (Vectorized)
    family_gates = {}
    family_gates_unseen = {}

    # -------------------------------------------------------------------------
    # STAGE 3A: Claude Specialist Gate (Vectorized Head)
    # -------------------------------------------------------------------------
    gate_weight = post_cfg.get("claude_specialist_weight", 0.15)
    P_mix = apply_claude_specialist_gate(
        P_mix=P_mix,
        R_main=R_main,
        claude_spec_logits=claude_spec_logits,
        claude_classes=claude_classes,
        labels=LABELS,
        claude_cols=FAMILIES["claude"],
        gate_weight=gate_weight
    )

    # -------------------------------------------------------------------------
    # STAGE 3B: Deep Neural Family-Specialist LoRA Gates (GPT, Gemini, GLM)
    # -------------------------------------------------------------------------
    for fam_k, default_w in [("gpt", 0.55), ("gemini", 0.65), ("glm", 0.70)]:
        spec_te_logits = family_spec_logits_te.get(fam_k)
        if spec_te_logits is not None:
            spec_cls = family_spec_classes.get(fam_k)
            w_gate = float(post_cfg.get(f"{fam_k}_specialist_weight", default_w))
            P_mix = apply_family_specialist_lora_gate(
                P_mix=P_mix,
                R_main=R_main,
                spec_logits=spec_te_logits,
                spec_classes=spec_cls,
                labels=LABELS,
                family_cols=FAMILIES[fam_k],
                gate_weight=w_gate,
            )

    # -------------------------------------------------------------------------
    # STAGE 4: Unified Handwriting & Orthographic Shield
    # -------------------------------------------------------------------------
    default_curly_multiplier = float(np.exp(-1.0))
    multiplier = post_cfg.get("curly_quote_multiplier", default_curly_multiplier)
    multi_em_dash_log_pen = post_cfg.get("multi_em_dash_log_penalty", 0.0)
    straight_contr_log_pen = post_cfg.get("straight_contr_log_penalty", 0.0)
    curly_contr_log_pen = post_cfg.get("curly_contr_log_penalty", 0.0)
    continuous_hw_weight = post_cfg.get("continuous_hw_weight", 0.0)

    P_mix = apply_unified_handwriting_shield(
        P_mix=P_mix,
        test_texts=test_texts,
        log_p_handwriting=log_p_handwriting,
        rule_flags=rule_flags_test,
        curly_quote_multiplier=multiplier,
        multi_em_dash_log_penalty=multi_em_dash_log_pen,
        straight_contr_log_penalty=straight_contr_log_pen,
        curly_contr_log_penalty=curly_contr_log_pen,
        continuous_hw_weight=continuous_hw_weight,
    )

    # -------------------------------------------------------------------------
    # STAGE 5: Cavity Cluster Uniqueness Repulsion
    # -------------------------------------------------------------------------
    P_mix = apply_cluster_uniqueness_repulsion(
        P_mix=P_mix,
        groups=groups_065,
        strength=post_cfg.get("cluster_repulsion_strength", 0.35),
        iters=3
    )

    # -------------------------------------------------------------------------
    # STAGE 6: Multiplicative Sinkhorn Column Mass Balancing
    # -------------------------------------------------------------------------
    sinkhorn_iters = post_cfg.get("sinkhorn_iterations", 100)
    sinkhorn_strength = float(post_cfg.get("sinkhorn_strength", 0.65))
    sinkhorn_target_mix = float(post_cfg.get("sinkhorn_target_mix", 0.35))
    train_prior = np.bincount(y_sub, minlength=18).astype(np.float64) + 1.0 if len(y_sub) > 0 else None
    P_final = sinkhorn_balancing(
        P_mix,
        iters=sinkhorn_iters,
        sinkhorn_strength=sinkhorn_strength,
        target_mix=sinkhorn_target_mix,
        train_prior=train_prior
    )

    # -------------------------------------------------------------------------
    # STAGE 7: TypoBayes 37-Marker Hierarchical Beta-Bernoulli Postprocessing
    # -------------------------------------------------------------------------
    if post_cfg.get("enable_typobayes", True):
        P_final = apply_typobayes_postprocess(
            P_final,
            train_texts=df_train["text"].fillna("").astype(str).tolist(),
            y_train=np.array([LAB2I[m] for m in df_train["model"].values], dtype=int),
            test_texts=test_texts,
            labels=LABELS,
            kappa=float(post_cfg.get("typobayes_kappa", 24.0)),
            evidence_scale=float(post_cfg.get("typobayes_evidence_scale", 1.2)),
            evidence_cap=float(post_cfg.get("typobayes_evidence_cap", 0.30)),
        )

    # Auditing and forward stepwise ablation report
    test_final_preds = P_final.argmax(axis=1) if len(P_final) > 0 else np.array([])
    auditor.print_and_save_report(working_dir, final_test_preds=test_final_preds)

    # Generate Submission Artifacts
    df_sub, out_parquet, out_csv = generate_submission(P_final, test_ids, working_dir)
    elapsed = time.time() - start_time
    print(f"[Pipeline] Submission generated successfully in {elapsed:.2f}s: {out_parquet} ({len(df_sub)} rows)")

    # Export compact local evaluation cache for sub-second offline local tuning
    try:
        if len(groups_unseen) > 0:
            try:
                g_u = np.asarray(groups_unseen, dtype=np.int32)
            except (ValueError, TypeError):
                g_u = (pd.factorize(groups_unseen)[0] + 1).astype(np.int32)
        else:
            g_u = np.array([], dtype=np.int32)

        cache_dict = {
            "N_test": np.array([N_test], dtype=np.int32),
            "N_unseen": np.array([N_unseen], dtype=np.int32),
            "y_unseen": y_unseen.astype(np.int32) if len(y_unseen) > 0 else np.array([], dtype=np.int32),
            "groups_unseen": g_u,
            "D_tfidf": D_tfidf.astype(np.float32) if ("D_tfidf" in locals() and D_tfidf is not None) else np.zeros((0, 0), dtype=np.float32),
            "D_hybrid": D_hybrid.astype(np.float32) if ("D_hybrid" in locals() and D_hybrid is not None) else np.zeros((0, 0), dtype=np.float32),
            "groups_085": groups_085.astype(np.int32) if ("groups_085" in locals() and groups_085 is not None) else np.zeros(0, dtype=np.int32),
            "groups_065": groups_065.astype(np.int32) if ("groups_065" in locals() and groups_065 is not None) else np.zeros(0, dtype=np.int32),
            "model_names": np.array([m["name"] for m in auditor.models], dtype=object),
            "model_tiers": np.array([m["tier"] for m in auditor.models], dtype=object),
            "model_prior_weights": np.array([m.get("prior_weight", m["weight"]) for m in auditor.models], dtype=np.float64),
            "model_learned_weights": np.array([m["weight"] for m in auditor.models], dtype=np.float64),
            "P_final": P_final.astype(np.float32),
            "test_texts": np.array(test_texts, dtype=object),
        }
        for idx, m in enumerate(auditor.models):
            cache_dict[f"m_{idx}_unseen_lp"] = m["unseen_lp"].astype(np.float32)
            cache_dict[f"m_{idx}_test_lp"] = m["test_lp"].astype(np.float32)
        if "rule_flags_test" in locals() and rule_flags_test is not None:
            cache_dict["rule_flags_test"] = rule_flags_test.astype(np.float32)
        if "log_p_handwriting" in locals() and log_p_handwriting is not None:
            cache_dict["log_p_handwriting"] = log_p_handwriting.astype(np.float32)
        if "claude_spec_logits" in locals() and claude_spec_logits is not None:
            cache_dict["claude_spec_logits"] = claude_spec_logits.astype(np.float32)
        if "claude_classes" in locals() and claude_classes is not None:
            cache_dict["claude_classes"] = np.array(claude_classes, dtype=object)
        if "family_gates" in locals() and family_gates:
            for fam_k, fam_lp in family_gates.items():
                cache_dict[f"gate_{fam_k}_lp"] = fam_lp.astype(np.float32)
        for ex_idx, (ex_name, ex_w, ex_lp) in enumerate(extra_streams):
            cache_dict[f"extra_{ex_idx}_name"] = np.array([ex_name], dtype=object)
            cache_dict[f"extra_{ex_idx}_weight"] = np.array([ex_w], dtype=np.float64)
            cache_dict[f"extra_{ex_idx}_lp"] = ex_lp.astype(np.float32)
            if ex_idx < len(extra_streams_unseen):
                cache_dict[f"extra_{ex_idx}_unseen_lp"] = extra_streams_unseen[ex_idx][2].astype(np.float32)
        cache_dict["num_extra"] = np.array([len(extra_streams)], dtype=np.int32)
        if "claude_spec_logits_unseen" in locals() and claude_spec_logits_unseen is not None:
            cache_dict["claude_spec_logits_unseen"] = claude_spec_logits_unseen.astype(np.float32)
        for fam_k in ("gpt", "gemini", "glm"):
            if "family_spec_logits_te" in locals() and fam_k in family_spec_logits_te and family_spec_logits_te[fam_k] is not None:
                cache_dict[f"{fam_k}_spec_logits"] = family_spec_logits_te[fam_k].astype(np.float32)
            if "family_spec_logits_unseen" in locals() and fam_k in family_spec_logits_unseen and family_spec_logits_unseen[fam_k] is not None:
                cache_dict[f"{fam_k}_spec_logits_unseen"] = family_spec_logits_unseen[fam_k].astype(np.float32)
            if "family_spec_classes" in locals() and fam_k in family_spec_classes and family_spec_classes[fam_k] is not None:
                cache_dict[f"{fam_k}_classes"] = np.array(family_spec_classes[fam_k], dtype=object)
        if "family_gates_unseen" in locals() and family_gates_unseen:
            for fam_k, fam_lp in family_gates_unseen.items():
                cache_dict[f"gate_{fam_k}_unseen_lp"] = fam_lp.astype(np.float32)
        if "unseen_texts" in locals() and unseen_texts:
            cache_dict["unseen_texts"] = np.array(unseen_texts, dtype=object)
        if "train_sub_texts" in locals() and train_sub_texts:
            cache_dict["train_texts"] = np.array(train_sub_texts, dtype=object)
        if "y_sub" in locals() and y_sub is not None:
            cache_dict["y_train"] = np.asarray(y_sub, dtype=np.int32)

        cache_file_path = working_dir / "local_eval_cache.npz"
        np.savez_compressed(cache_file_path, **cache_dict)
        print(f"[Pipeline] Exported compact local evaluation cache: {cache_file_path} ({cache_file_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"[Pipeline] Note: Could not write local_eval_cache.npz: {e}")

    return df_sub, out_parquet
