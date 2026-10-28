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

import numpy as np
import pandas as pd
from scipy import sparse

from .config import (
    LABELS, K, LAB2I, FAMILIES, PRIOR_SCALES,
    softmax, log_softmax, align_cols,
    resolve_dataset_paths, find_base_model, find_checkpoints
)
from .lexical_engine import run_lexical_engine
from .neural_extractor import (
    UniversalLoRATrunkLoader,
    resolve_head_tensors,
    compute_centered_logits,
    merge_lora,
    extract_deberta_checkpoint
)
from .wpca_gaussian import (
    fit_shared_gaussian_lowram,
    transform_lowram,
    split_oversized_clusters_dist,
    scores_exact_dp
)
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

    # Read configuration hierarchies from configs/v42_ensemble.json
    main_cfg = cfg.get("main_stack", {})
    comp_cfg = cfg.get("complementary_stack", {})
    fusion_cfg = cfg.get("fusion", {})
    post_cfg = cfg.get("postprocess", {})

    svc_weight = main_cfg.get("linear_svc_weight", 1.00)
    main_members = main_cfg.get("members", {})
    comp_streams = comp_cfg.get("stream_weights", {})
    comp_members = comp_cfg.get("lora_members", {})
    se_floors = comp_cfg.get("se_floors", {
        "qwen3_14b_p2": 0.10, "qwen25_14b_p2": 0.10, "deberta_p2": 0.10,
        "qwen3_14b_p1": 0.35, "instyle": 0.35, "qwen3_8b_claude": 0.35
    })
    prior_scales = comp_cfg.get("prior_scales", PRIOR_SCALES)

    LOGIT_MAIN = log_p_svc_main * svc_weight
    COMP_LIST: List[Tuple[float, np.ndarray]] = [
        (comp_streams.get("lexical", 0.20), p_lex_comp)
    ]
    claude_spec_logits = None
    claude_classes = None

    ckpts = find_checkpoints(input_dir)
    has_neural = len(ckpts) > 0 and not is_smoke

    if has_neural:
        import torch
        from safetensors.torch import load_file

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
        comp_lora_posteriors: List[Tuple[float, np.ndarray]] = []

        # ==============================================================================
        # TRUNK 1: Qwen3-14B (40 layers, hidden=5120)
        # ==============================================================================
        base_q3_14b = find_base_model(input_dir, "qwen3", 5120, 40)
        if base_q3_14b and ("qwen3_14b_p2" in ckpts or "qwen3_14b_p1" in ckpts):
            print(f"[Pipeline] Loading Trunk 1: Qwen3-14B from {base_q3_14b}...")
            trunk1 = UniversalLoRATrunkLoader(base_q3_14b, keep_layers=27, layers=[2, 4, 6, 9, 13, 17, 22, 27])

            # Checkpoint 1A: Part 2 (lora_qwen3_14b)
            if "qwen3_14b_p2" in ckpts:
                p2_dir = ckpts["qwen3_14b_p2"]
                backup_p2 = trunk1.snapshot_weights(p2_dir)
                merge_lora(trunk1.model, p2_dir, sign=+1.0)
                heads_sd = load_file(str(p2_dir / "heads.safetensors"))
                trunk1.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

                meta_p = p2_dir / "meta.json"
                p2_labels = LABELS
                if meta_p.exists():
                    try:
                        p2_labels = json.load(open(meta_p)).get("labels", LABELS)
                    except Exception:
                        pass

                Z_all = trunk1.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=151643)
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]

                # Main Stack contributions
                w_main_p = main_members.get("lora_qwen3_14b", {}).get("plain_weight", 0.40)
                w_main_c = main_members.get("lora_qwen3_14b", {}).get("twin_weight", 0.10)
                lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, p2_labels))
                LOGIT_MAIN += lp_p * w_main_p
                if W_c is not None:
                    lp_c, valid_c = compute_centered_logits(Z_te, groups_085, W_c, b_c, p2_labels)
                    LOGIT_MAIN[valid_c] += lp_c[valid_c] * w_main_c

                # Complementary LoRA stream
                w_comp_p = comp_members.get("lora_qwen3_14b", {}).get("plain_weight", 0.30)
                w_comp_c = comp_members.get("lora_qwen3_14b", {}).get("twin_weight", 0.10)
                comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                if W_c is not None:
                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c, axis=1)))

                # wPCA + DP Posterior + Spectral Bank
                res_p2 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen3_14b_p2", 0.10))
                U_te = transform_lowram(Z_te, res_p2)
                lp_dp = scores_exact_dp(U_te, groups_085, res_p2, D_tfidf)
                dp_posteriors.append((0.38, softmax(lp_dp, axis=1)))

                spectral_blocks_tr.append(("qwen3_14b_p2", res_p2["V_top64"]))
                spectral_blocks_te.append(("qwen3_14b_p2", (((Z_te - res_p2["mx"]) / res_p2["sx"]) @ res_p2["V_top64"]).astype(np.float32)))

                trunk1.restore_weights(backup_p2)

            # Checkpoint 1B: Part 1 (lora14b_v1)
            if "qwen3_14b_p1" in ckpts:
                p1_dir = ckpts["qwen3_14b_p1"]
                backup_p1 = trunk1.snapshot_weights(p1_dir)
                merge_lora(trunk1.model, p1_dir, sign=+1.0)
                heads_sd = load_file(str(p1_dir / "heads.safetensors"))
                trunk1.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

                meta_p = p1_dir / "meta.json"
                p1_labels = LABELS
                if meta_p.exists():
                    try:
                        p1_labels = json.load(open(meta_p)).get("labels", LABELS)
                    except Exception:
                        pass

                Z_all = trunk1.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=151643)
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]

                w_comp_p = comp_members.get("lora14b_v1", {}).get("plain_weight", 0.14)
                w_comp_c = comp_members.get("lora14b_v1", {}).get("twin_weight", 0.05)
                lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, p1_labels))
                comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                if W_c is not None:
                    lp_c, valid_c = compute_centered_logits(Z_te, groups_085, W_c, b_c, p1_labels)
                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c, axis=1)))

                res_p1 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen3_14b_p1", 0.35))
                U_te = transform_lowram(Z_te, res_p1)
                lp_dp = scores_exact_dp(U_te, groups_085, res_p1, D_tfidf)
                dp_posteriors.append((0.08, softmax(lp_dp, axis=1)))

                spectral_blocks_tr.append(("qwen3_14b_p1", res_p1["V_top64"]))
                spectral_blocks_te.append(("qwen3_14b_p1", (((Z_te - res_p1["mx"]) / res_p1["sx"]) @ res_p1["V_top64"]).astype(np.float32)))

                trunk1.restore_weights(backup_p1)

            trunk1.free_memory()

        # ==============================================================================
        # TRUNK 2: Qwen2.5-14B (48 layers, hidden=5120)
        # ==============================================================================
        base_q25_14b = find_base_model(input_dir, "qwen2", 5120, 48)
        if base_q25_14b and "qwen25_14b_p2" in ckpts:
            print(f"[Pipeline] Loading Trunk 2: Qwen2.5-14B from {base_q25_14b}...")
            trunk2 = UniversalLoRATrunkLoader(base_q25_14b, keep_layers=32, layers=[3, 6, 10, 14, 18, 23, 27, 32])
            p2_dir = ckpts["qwen25_14b_p2"]
            merge_lora(trunk2.model, p2_dir, sign=+1.0)
            heads_sd = load_file(str(p2_dir / "heads.safetensors"))
            trunk2.load_norm_weights(heads_sd)
            W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

            meta_p = p2_dir / "meta.json"
            p2_labels = LABELS
            if meta_p.exists():
                try:
                    p2_labels = json.load(open(meta_p)).get("labels", LABELS)
                except Exception:
                    pass

            Z_all = trunk2.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=151643)
            Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]

            w_main_p = main_members.get("lora_qwen25_14b", {}).get("plain_weight", 0.20)
            w_main_c = main_members.get("lora_qwen25_14b", {}).get("twin_weight", 0.10)
            lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, p2_labels))
            LOGIT_MAIN += lp_p * w_main_p
            if W_c is not None:
                lp_c, valid_c = compute_centered_logits(Z_te, groups_085, W_c, b_c, p2_labels)
                LOGIT_MAIN[valid_c] += lp_c[valid_c] * w_main_c

            w_comp_p = comp_members.get("lora_qwen25_14b", {}).get("plain_weight", 0.20)
            w_comp_c = comp_members.get("lora_qwen25_14b", {}).get("twin_weight", 0.08)
            comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
            if W_c is not None:
                comp_lora_posteriors.append((w_comp_c, softmax(lp_c, axis=1)))

            res_q25 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen25_14b_p2", 0.10))
            U_te = transform_lowram(Z_te, res_q25)
            lp_dp = scores_exact_dp(U_te, groups_085, res_q25, D_tfidf)
            dp_posteriors.append((0.24, softmax(lp_dp, axis=1)))

            spectral_blocks_tr.append(("qwen25_14b_p2", res_q25["V_top64"]))
            spectral_blocks_te.append(("qwen25_14b_p2", (((Z_te - res_q25["mx"]) / res_q25["sx"]) @ res_q25["V_top64"]).astype(np.float32)))

            trunk2.free_memory()

        # ==============================================================================
        # TRUNK 3: Qwen3-8B (36 layers, hidden=4096)
        # ==============================================================================
        base_q3_8b = find_base_model(input_dir, "qwen3", 4096, 36)
        if base_q3_8b and ("instyle" in ckpts or "qwen3_8b_claude" in ckpts):
            print(f"[Pipeline] Loading Trunk 3: Qwen3-8B from {base_q3_8b}...")
            trunk3 = UniversalLoRATrunkLoader(base_q3_8b, keep_layers=24, layers=[2, 5, 8, 11, 14, 18, 21, 24])

            # In-Style LoRA
            if "instyle" in ckpts:
                instyle_dir = ckpts["instyle"]
                backup_ins = trunk3.snapshot_weights(instyle_dir)
                merge_lora(trunk3.model, instyle_dir, sign=+1.0)
                heads_sd = load_file(str(instyle_dir / "heads.safetensors")) if (instyle_dir / "heads.safetensors").exists() else {}
                trunk3.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

                Z_all = trunk3.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=151643)
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]

                if W_p is not None:
                    w_comp_p = comp_members.get("instyle", {}).get("plain_weight", 0.22)
                    lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, LABELS))
                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))

                res_ins = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("instyle", 0.35))
                U_te = transform_lowram(Z_te, res_ins)
                lp_dp = scores_exact_dp(U_te, groups_085, res_ins, D_tfidf)
                dp_posteriors.append((0.10, softmax(lp_dp, axis=1)))

                spectral_blocks_tr.append(("instyle", res_ins["V_top64"]))
                spectral_blocks_te.append(("instyle", (((Z_te - res_ins["mx"]) / res_ins["sx"]) @ res_ins["V_top64"]).astype(np.float32)))

                trunk3.restore_weights(backup_ins)

            # Claude Head LoRA
            if "qwen3_8b_claude" in ckpts:
                claude_dir = ckpts["qwen3_8b_claude"]
                backup_cl = trunk3.snapshot_weights(claude_dir)
                merge_lora(trunk3.model, claude_dir, sign=+1.0)
                head_file = claude_dir / "claude_head.safetensors"
                if not head_file.exists():
                    head_file = claude_dir / "heads.safetensors"
                heads_sd = load_file(str(head_file)) if head_file.exists() else {}
                trunk3.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

                meta_p = claude_dir / "meta.json"
                if meta_p.exists():
                    try:
                        claude_classes = json.load(open(meta_p)).get("classes", None)
                    except Exception:
                        pass

                Z_all = trunk3.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=151643)
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]

                if W_p is not None:
                    claude_spec_logits = Z_te @ W_p.T + b_p
                    w_comp_p = comp_members.get("ntl_claude_head", {}).get("plain_weight", 0.14)
                    lp_p = log_softmax(align_cols(claude_spec_logits, claude_classes or LABELS))
                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))

                trunk3.restore_weights(backup_cl)

            trunk3.free_memory()

        # ==============================================================================
        # TRUNK 4: DeBERTa-v3-Large
        # ==============================================================================
        if "deberta_p2" in ckpts:
            deb_dir = ckpts["deberta_p2"]
            base_deb_dir = find_base_model(input_dir, "deberta", 1024, 24)
            print(f"[Pipeline] Loading Trunk 4: DeBERTa-v3 from {deb_dir}...")
            Z_te_deb, Z_tr_deb, W_p_deb, b_p_deb, W_c_deb, b_c_deb, deb_labels = extract_deberta_checkpoint(
                deb_dir, base_deb_dir, test_texts, train_sub_texts, y_sub
            )
            if W_p_deb is not None:
                w_main_deb = main_members.get("deberta_v3_large", {}).get("plain_weight", 0.25)
                lp_deb = log_softmax(align_cols(Z_te_deb @ W_p_deb.T + b_p_deb, deb_labels))
                LOGIT_MAIN += lp_deb * w_main_deb

                w_comp_deb = comp_members.get("deberta_v3_large", {}).get("plain_weight", 0.18)
                comp_lora_posteriors.append((w_comp_deb, softmax(lp_deb, axis=1)))

            res_deb = fit_shared_gaussian_lowram(Z_tr_deb, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("deberta_p2", 0.10))
            U_te = transform_lowram(Z_te_deb, res_deb)
            lp_dp = scores_exact_dp(U_te, groups_085, res_deb, D_tfidf)
            dp_posteriors.append((0.20, softmax(lp_dp, axis=1)))

            spectral_blocks_tr.append(("deberta_p2", res_deb["V_top64"]))
            spectral_blocks_te.append(("deberta_p2", (((Z_te_deb - res_deb["mx"]) / res_deb["sx"]) @ res_deb["V_top64"]).astype(np.float32)))

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
            P_spectral, oof_score = fit_joint_spectral_bank(
                spectral_blocks_tr, spectral_blocks_te, y_sub, prompts_sub,
                prior_scales=prior_scales, n_splits=5
            )
            print(f"[Pipeline] Joint Spectral Bank fitted (OOF BAcc = {oof_score:.4f})")
            COMP_LIST.append((comp_streams.get("spectral_bank", 0.15), P_spectral))

    elif is_smoke:
        print("[Pipeline] Smoke mode active: synthesizing high-fidelity mock representations...")
        np.random.seed(42)
        mock_p_lora = softmax(np.random.randn(N_test, K), axis=1)
        mock_p_dp = softmax(np.random.randn(N_test, K), axis=1)
        mock_p_spec = softmax(np.random.randn(N_test, K), axis=1)
        COMP_LIST.append((comp_streams.get("lora_and_twins", 0.35), mock_p_lora))
        COMP_LIST.append((comp_streams.get("dp_posteriors", 0.30), mock_p_dp))
        COMP_LIST.append((comp_streams.get("spectral_bank", 0.15), mock_p_spec))

    # 4. MODULE 4: Convex Blend, Claude Specialist Gate, Curly Quote Guard & Sinkhorn
    print("[Pipeline] Module 4: Fusing Main and Complementary stacks...")
    main_ratio = fusion_cfg.get("main_ratio", 0.85)
    comp_ratio = fusion_cfg.get("comp_ratio", 0.15)
    P_mix, R_main = blend_main_and_complementary(LOGIT_MAIN, COMP_LIST, main_ratio=main_ratio, comp_ratio=comp_ratio)

    # Claude Specialist Gate (Vectorized)
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

    # Curly Quote Shield (gentle -1.0 penalty in log-space = exp(-1.0) probability multiplier)
    default_curly_multiplier = float(np.exp(-1.0))
    multiplier = post_cfg.get("curly_quote_multiplier", default_curly_multiplier)
    P_mix = apply_curly_quote_shield(P_mix, test_texts, multiplier=multiplier)

    # 300-Step Sinkhorn Mass Balancing
    sinkhorn_iters = post_cfg.get("sinkhorn_iterations", 300)
    P_final = sinkhorn_balancing(P_mix, iters=sinkhorn_iters)

    # Generate Submission Artifacts
    df_sub, out_parquet, out_csv = generate_submission(P_final, test_ids, working_dir)
    elapsed = time.time() - start_time
    print(f"[Pipeline] Submission generated successfully in {elapsed:.2f}s: {out_parquet} ({len(df_sub)} rows)")

    return df_sub, out_parquet
