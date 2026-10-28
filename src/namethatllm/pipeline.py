"""
Top-Level Pipeline Orchestrator
===============================
Coordinates the complete end-to-end execution flow:
- Module 1: Dual-Output Lexical Engine (subprocess isolated or in-process)
- Module 2: Neural Trunk Extraction Loop (UniversalLoRATrunkLoader + BF16 rescue)
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

import numpy as np
import pandas as pd

from .config import (
    LABELS, K, LAB2I, FAMILIES, PRIOR_SCALES,
    softmax, log_softmax, align_cols,
    resolve_dataset_paths, find_base_model, find_checkpoints
)
from .lexical_engine import run_lexical_engine
from .neural_extractor import UniversalLoRATrunkLoader, resolve_head_tensors
from .wpca_gaussian import fit_shared_gaussian_lowram, split_oversized_clusters_dist, scores_exact_dp
from .spectral_bank import fit_joint_spectral_bank
from .postprocess import (
    blend_main_and_complementary,
    apply_claude_specialist_gate,
    apply_curly_quote_shield,
    sinkhorn_balancing,
    generate_submission
)

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
    else:
        train_path, test_path, pool_file = resolve_dataset_paths(input_dir)
    print(f"[Pipeline] Startup: is_smoke={is_smoke}")
    print(f"[Pipeline] Paths: train={train_path}, test={test_path}, pool={pool_file}")

    df_test = pd.read_csv(test_path)
    test_ids = df_test["id"]
    test_texts = df_test["text"].fillna("").astype(str).tolist()
    N_test = len(test_texts)
    print(f"[Pipeline] Loaded test set: {N_test} rows")

    # 2. MODULE 1: Subprocess Dual-Output Lexical Engine
    run_lex_script = working_dir / "run_lexical.py"
    if run_lex_script.exists() and not is_smoke:
        print("[Pipeline] Module 1: Launching subprocess run_lexical.py...")
        subprocess.run([sys.executable, str(run_lex_script)], check=True)
        log_p_svc_main = np.load(working_dir / "log_p_svc_main.npy")
        p_lex_comp = np.load(working_dir / "p_lex_comp.npy")
        D_tfidf = np.load(working_dir / "D_tfidf.npy")
        groups_085 = np.load(working_dir / "groups_085.npy")
    else:
        print("[Pipeline] Module 1: Executing run_lexical_engine directly...")
        tok_paths = list(input_dir.rglob("vocab.json"))
        qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""
        lex_out = run_lexical_engine(train_path, test_path, pool_file, qwen_tok_dir, working_dir)
        log_p_svc_main = lex_out["log_p_svc_main"]
        p_lex_comp = lex_out["p_lex_comp"]
        D_tfidf = lex_out["D_tfidf"]
        groups_085 = lex_out["groups_085"]

    print(f"[Pipeline] Module 1 complete: log_p_svc_main shape = {log_p_svc_main.shape}")

    # Checkpoint & Model Discovery
    ckpts = find_checkpoints(input_dir)
    has_neural = len(ckpts) > 0 and not is_smoke

    # Config parameters
    main_weights = cfg.get("main_weights", {
        "svc": 1.00,
        "qwen3_14b_plain": 0.40,
        "qwen3_14b_twin": 0.10,
        "qwen25_14b_plain": 0.20,
        "qwen25_14b_twin": 0.10,
        "deberta_plain": 0.25,
        "deberta_twin": 0.00
    })
    comp_weights = cfg.get("comp_weights", {
        "secondary_neural": 0.35,
        "lexical": 0.20,
        "wpca_dp": 0.30,
        "spectral_bank": 0.15
    })
    blend_ratios = cfg.get("blend_ratios", {
        "main_ratio": 0.85,
        "comp_ratio": 0.15
    })
    shrinkage_floors = cfg.get("shrinkage_floors", {
        "part2": 0.10,
        "overfit_8b_part1": 0.35
    })
    prior_scales = cfg.get("prior_scales", PRIOR_SCALES)
    post_cfg = cfg.get("postprocess", {
        "claude_gate_weight": 0.15,
        "curly_quote_multiplier": 0.36787944117144233,
        "sinkhorn_iters": 300,
        "max_cluster_size": 14
    })

    LOGIT_MAIN = log_p_svc_main * main_weights.get("svc", 1.00)
    COMP_LIST: List[Tuple[float, np.ndarray]] = [
        (comp_weights.get("lexical", 0.20), p_lex_comp)
    ]
    claude_spec_logits = None
    claude_classes = None

    if has_neural:
        print(f"[Pipeline] Neural checkpoints discovered: {list(ckpts.keys())}")
        # Live multi-trunk neural execution
        import torch
        from transformers import AutoTokenizer

        # Select balanced training rows for wPCA and calibration
        df_tr = pd.read_csv(train_path)
        u_p = df_tr["prompt_id"].unique()
        sel_prompts = u_p[:min(90, len(u_p))]
        sub_idx = df_tr[df_tr["prompt_id"].isin(sel_prompts)].index.values
        train_sub = df_tr.loc[sub_idx].reset_index(drop=True)
        train_sub_texts = train_sub["text"].fillna("").astype(str).tolist()
        y_sub = np.array([LAB2I[m] for m in train_sub["model"]])
        prompts_sub = train_sub["prompt_id"].values
        N_tr_sub = len(train_sub)
        print(f"[Pipeline] Selected {N_tr_sub} training rows ({len(sel_prompts)} prompts) for wPCA")

        spectral_blocks_tr: List[Tuple[str, np.ndarray]] = []
        spectral_blocks_te: List[Tuple[str, np.ndarray]] = []
        dp_posteriors: List[Tuple[float, np.ndarray]] = []
        D_neural = None

        # TRUNK 1: Qwen3-14B (lora_qwen3_14b, lora14b_v1)
        base_q3_14b = find_base_model(input_dir, "qwen3", 5120, 48)
        if base_q3_14b and "qwen3_14b_p2" in ckpts:
            print("[Pipeline] Executing Trunk 1: Qwen3-14B...")
            trunk = UniversalLoRATrunkLoader(str(base_q3_14b), keep_layers=27, primary_device="cuda:0", secondary_device="cuda:1")
            tok = AutoTokenizer.from_pretrained(str(base_q3_14b), use_fast=True)
            norm_w = trunk.get_norm_weight()

            # Part 2 checkpoint
            p2_dir = ckpts["qwen3_14b_p2"]
            p2_weights = trunk.load_lora_weights(str(p2_dir / "adapter_model.safetensors"))
            trunk.merge_lora_weights(p2_weights)
            W_head, b_head, classes = resolve_head_tensors(str(p2_dir / "heads.safetensors"), "qwen3_14b")

            # Extract test features
            H_te = trunk.extract_pooled_hidden_states(test_texts, tok, layers=[16, 20, 24, 27], batch_size=8)
            z_te_p2 = torch.cat([norm_w(H_te[l].to("cuda:0")) for l in [16, 20, 24, 27]], dim=-1).float().cpu().numpy()
            del H_te

            # Head logits
            logits_plain = align_cols(z_te_p2 @ W_head + b_head, classes)
            LOGIT_MAIN += logits_plain * main_weights.get("qwen3_14b_plain", 0.40)

            # Centered twin logits
            u_t, inv_t = np.unique(groups_085, return_inverse=True)
            counts_t = np.bincount(inv_t)
            vals_t = (1.0 / counts_t[inv_t]).astype(np.float32)
            G_t = sparse.csr_matrix((vals_t, (np.arange(len(inv_t)), inv_t)), shape=(len(inv_t), len(u_t)), dtype=np.float32)
            z_te_twin = z_te_p2 - G_t @ (G_t.T @ z_te_p2)
            logits_twin = align_cols(z_te_twin @ W_head + b_head, classes)
            LOGIT_MAIN += logits_twin * main_weights.get("qwen3_14b_twin", 0.10)

            # Extract train_sub features for wPCA & Spectral Bank
            H_tr = trunk.extract_pooled_hidden_states(train_sub_texts, tok, layers=[16, 20, 24, 27], batch_size=8)
            z_tr_p2 = torch.cat([norm_w(H_tr[l].to("cuda:0")) for l in [16, 20, 24, 27]], dim=-1).float().cpu().numpy()
            del H_tr

            # Shared Gaussian wPCA
            res_p2 = fit_shared_gaussian_lowram(z_tr_p2, y_sub, prompts_sub, d_target=256, se_floor=shrinkage_floors.get("part2", 0.10))
            U_te_p2 = (((z_te_p2 - res_p2["mx"]) / res_p2["sx"]) @ res_p2["V_f32"]).astype(np.float64)
            spectral_blocks_tr.append(("qwen3_14b_p2", res_p2["U_tr_64"]))
            spectral_blocks_te.append(("qwen3_14b_p2", (((z_te_p2 - res_p2["mx"]) / res_p2["sx"]) @ res_p2["V_top64"]).astype(np.float32)))

            # Neural distance
            norm_z = z_te_p2 / (np.linalg.norm(z_te_p2, axis=1, keepdims=True) + 1e-12)
            D_neural = np.clip(1.0 - (norm_z @ norm_z.T), 0.0, None)
            np.fill_diagonal(D_neural, 0.0)

            # DP Posterior
            P_dp_p2 = scores_exact_dp(U_te_p2, res_p2["mu"], res_p2["Se_inv"], groups_085)
            dp_posteriors.append((0.40, P_dp_p2))

            trunk.unmerge_lora_weights(p2_weights)
            del trunk, tok
            torch.cuda.empty_cache()
            gc.collect()

        # Combine DP posteriors into COMP_LIST
        if dp_posteriors:
            tot_dp_w = sum(w for w, _ in dp_posteriors)
            P_dp_comb = sum(w * P_dp for w, P_dp in dp_posteriors) / max(1e-8, tot_dp_w)
            COMP_LIST.append((comp_weights.get("wpca_dp", 0.30), P_dp_comb))

        # Fit Joint Spectral Bank
        if spectral_blocks_tr and spectral_blocks_te:
            P_spectral, oof_score = fit_joint_spectral_bank(
                spectral_blocks_tr, spectral_blocks_te, y_sub, prompts_sub,
                prior_scales=prior_scales, n_splits=5
            )
            print(f"[Pipeline] Joint Spectral Bank fitted (OOF BAcc = {oof_score:.4f})")
            COMP_LIST.append((comp_weights.get("spectral_bank", 0.15), P_spectral))

    elif is_smoke:
        print("[Pipeline] Smoke mode active: synthesizing high-fidelity mock representations...")
        np.random.seed(42)
        # Add mock DP posteriors and mock spectral bank
        mock_p_dp = softmax(np.random.randn(N_test, K), axis=1)
        mock_p_spec = softmax(np.random.randn(N_test, K), axis=1)
        COMP_LIST.append((comp_weights.get("wpca_dp", 0.30), mock_p_dp))
        COMP_LIST.append((comp_weights.get("spectral_bank", 0.15), mock_p_spec))

    # 4. MODULE 4: Convex Blend, Claude Specialist Gate, Curly Quote Guard & Sinkhorn
    print("[Pipeline] Module 4: Fusing Main and Complementary stacks...")
    main_ratio = blend_ratios.get("main_ratio", 0.85)
    comp_ratio = blend_ratios.get("comp_ratio", 0.15)
    P_mix, R_main = blend_main_and_complementary(LOGIT_MAIN, COMP_LIST, main_ratio=main_ratio, comp_ratio=comp_ratio)

    # Claude Specialist Gate (Vectorized)
    gate_weight = post_cfg.get("claude_gate_weight", 0.15)
    P_mix = apply_claude_specialist_gate(
        P_mix=P_mix,
        R_main=R_main,
        claude_spec_logits=claude_spec_logits,
        claude_classes=claude_classes,
        labels=LABELS,
        claude_cols=FAMILIES["claude"],
        gate_weight=gate_weight
    )

    # Curly Quote Shield
    multiplier = post_cfg.get("curly_quote_multiplier", 0.36787944117144233)
    P_mix = apply_curly_quote_shield(P_mix, test_texts, multiplier=multiplier)

    # 300-Step Sinkhorn Mass Balancing
    sinkhorn_iters = post_cfg.get("sinkhorn_iters", 300)
    P_final = sinkhorn_balancing(P_mix, iters=sinkhorn_iters)

    # Generate Submission Artifacts
    df_sub, out_parquet, out_csv = generate_submission(P_final, test_ids, working_dir)
    elapsed = time.time() - start_time
    print(f"[Pipeline] Submission generated successfully in {elapsed:.2f}s: {out_parquet} ({len(df_sub)} rows)")

    return df_sub, out_parquet
