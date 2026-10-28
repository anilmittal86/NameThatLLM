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
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

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
    read_checkpoint_meta,
    extract_deberta_checkpoint
)
from .wpca_gaussian import (
    fit_shared_gaussian_lowram,
    transform_lowram,
    split_oversized_clusters_dist,
    scores_exact_dp
)
from .spectral_bank import fit_joint_spectral_bank, make_spectral_blocks
from .postprocess import (
    blend_main_and_complementary,
    apply_claude_specialist_gate,
    apply_unified_handwriting_shield,
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
        log_p_handwriting = np.load(working_dir / "log_p_handwriting.npy") if (working_dir / "log_p_handwriting.npy").exists() else None
        rule_flags_test = np.load(working_dir / "rule_flags_test.npy") if (working_dir / "rule_flags_test.npy").exists() else None
    else:
        print("[Pipeline] Module 1: Executing run_lexical_engine directly...")
        tok_paths = list(input_dir.rglob("vocab.json"))
        qwen_tok_dir = str(tok_paths[0].parent) if tok_paths else ""
        lex_out = run_lexical_engine(train_path, test_path, pool_file, qwen_tok_dir, working_dir)
        log_p_svc_main = lex_out["log_p_svc_main"]
        p_lex_comp = lex_out["p_lex_comp"]
        D_tfidf = lex_out["D_tfidf"]
        groups_085 = lex_out["groups_085"]
        log_p_handwriting = lex_out.get("log_p_handwriting")
        rule_flags_test = lex_out.get("rule_flags_test")

    print(f"[Pipeline] Module 1 complete: log_p_svc_main shape = {log_p_svc_main.shape}")

    # Read configuration hierarchies from configs/production_ensemble.json
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
        (comp_streams.get("lexical", 0.20), p_lex_comp)
    ]
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

    ckpts = find_checkpoints(input_dir)
    has_neural = len(ckpts) > 0 and not is_smoke

    if has_neural:
        import torch
        from safetensors.torch import load_file

        df_tr = pd.read_csv(train_path)
        rng = np.random.RandomState(42)
        u_p = df_tr["prompt_id"].unique()
        sel_prompts = rng.choice(u_p, size=min(90, len(u_p)), replace=False)
        sub_idx = df_tr[df_tr["prompt_id"].isin(sel_prompts)].index.values
        train_sub = df_tr.loc[sub_idx].reset_index(drop=True)
        train_sub_texts = train_sub["text"].fillna("").astype(str).tolist()
        y_sub = np.array([LAB2I[m] for m in train_sub["model"]])
        prompts_sub = train_sub["prompt_id"].values
        N_tr_sub = len(train_sub)
        print(f"[Pipeline] Selected {N_tr_sub} training rows ({len(sel_prompts)} prompts) for wPCA")

        spectral_blocks_tr: List[np.ndarray] = []
        spectral_blocks_te: List[np.ndarray] = []
        dp_posteriors: List[Tuple[float, np.ndarray]] = []
        comp_lora_posteriors: List[Tuple[float, np.ndarray]] = []

        # ==============================================================================
        # TRUNK 1: Qwen3-14B (40 layers, hidden=5120)
        # ==============================================================================
        base_q3_14b = find_base_model(input_dir, "qwen3", 5120, 40)
        if base_q3_14b and ("qwen3_14b_p2" in ckpts or "qwen3_14b_p1" in ckpts):
            print(f"[Pipeline] Loading Trunk 1: Qwen3-14B from {base_q3_14b}...")
            trunk1 = UniversalLoRATrunkLoader(base_q3_14b, keep_layers=27, layers=[2, 4, 6, 9, 13, 17, 22, 27], max_len=447)

            # Checkpoint 1A: Part 2 (lora_qwen3_14b)
            if "qwen3_14b_p2" in ckpts:
                p2_dir = ckpts["qwen3_14b_p2"]
                meta_p2 = read_checkpoint_meta(p2_dir, default_keep=27, default_layers=[2, 4, 6, 9, 13, 17, 22, 27], default_max_len=447)
                trunk1.set_active_layers(meta_p2["layers"], meta_p2["max_len"])
                backup_t1 = trunk1.snapshot_weights(p2_dir)
                merge_lora(trunk1.model, p2_dir, sign=+1.0)
                heads_sd = load_file(str(p2_dir / "heads.safetensors")) if (p2_dir / "heads.safetensors").exists() else {}
                trunk1.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
                p2_labels = meta_p2["labels"]

                Z_all = trunk1.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=meta_p2["end_id"])
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]
                update_hybrid_groups(Z_te, prior_scales.get("qwen3_14b_p2", 1.00))

                # Main Stack contributions
                w_main_p = main_members.get("lora_qwen3_14b", {}).get("plain_weight", 0.40)
                w_main_c = main_members.get("lora_qwen3_14b", {}).get("twin_weight", 0.10)
                if W_p is not None:
                    lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, p2_labels))
                    LOGIT_MAIN += lp_p * w_main_p
                    w_comp_p = comp_members.get("lora_qwen3_14b", {}).get("plain_weight", 0.30)
                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))

                if W_c is not None:
                    lp_c85, valid_c85 = compute_centered_logits(Z_te, groups_085, W_c, b_c, p2_labels)
                    LOGIT_MAIN[valid_c85] += lp_c85[valid_c85] * w_main_c
                    lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, p2_labels)
                    lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, p2_labels)
                    lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                    w_comp_c = comp_members.get("lora_qwen3_14b", {}).get("twin_weight", 0.10)
                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))

                # wPCA + DP Posterior + Spectral Bank
                res_p2 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen3_14b_p2", 0.10))
                U_te = transform_lowram(Z_te, res_p2)
                lp_dp = scores_exact_dp(U_te, groups_065, res_p2, D_hybrid)
                dp_posteriors.append((dp_weights.get("q3_14b", 0.38), softmax(lp_dp, axis=1)))

                S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_p2, prior_scales.get("qwen3_14b_p2", 1.00))
                spectral_blocks_tr.append(S_tr)
                spectral_blocks_te.append(S_te)

                trunk1.restore_weights(backup_t1)
                del backup_t1, Z_all, Z_tr, Z_te, heads_sd, res_p2, U_te
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Checkpoint 1B: Part 1 (lora14b_v1)
            if "qwen3_14b_p1" in ckpts:
                p1_dir = ckpts["qwen3_14b_p1"]
                meta_p1 = read_checkpoint_meta(p1_dir, default_keep=24, default_layers=[2, 4, 6, 9, 13, 17, 21, 24], default_max_len=447)
                trunk1.set_active_layers(meta_p1["layers"], meta_p1["max_len"])
                merge_lora(trunk1.model, p1_dir, sign=+1.0)
                heads_sd = load_file(str(p1_dir / "heads.safetensors")) if (p1_dir / "heads.safetensors").exists() else {}
                trunk1.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
                p1_labels = meta_p1["labels"]

                Z_all = trunk1.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=meta_p1["end_id"])
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]
                update_hybrid_groups(Z_te, prior_scales.get("qwen3_14b_p1", 0.50))

                w_comp_p = comp_members.get("lora14b_v1", {}).get("plain_weight", 0.14)
                w_comp_c = comp_members.get("lora14b_v1", {}).get("twin_weight", 0.05)
                if W_p is not None:
                    lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, p1_labels))
                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                if W_c is not None:
                    lp_c85, _ = compute_centered_logits(Z_te, groups_085, W_c, b_c, p1_labels)
                    lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, p1_labels)
                    lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, p1_labels)
                    lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))

                res_p1 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen3_14b_p1", 0.35))
                U_te = transform_lowram(Z_te, res_p1)
                lp_dp = scores_exact_dp(U_te, groups_065, res_p1, D_hybrid)
                dp_posteriors.append((dp_weights.get("part1_14b", 0.08), softmax(lp_dp, axis=1)))

                S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_p1, prior_scales.get("qwen3_14b_p1", 0.50))
                spectral_blocks_tr.append(S_tr)
                spectral_blocks_te.append(S_te)

                merge_lora(trunk1.model, p1_dir, sign=-1.0)
                del Z_all, Z_tr, Z_te, heads_sd, res_p1, U_te
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            trunk1.free_memory()
            del trunk1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 2: Qwen2.5-14B (48 layers, hidden=5120)
        # ==============================================================================
        base_q25_14b = find_base_model(input_dir, "qwen2", 5120, 48)
        if base_q25_14b and "qwen25_14b_p2" in ckpts:
            print(f"[Pipeline] Loading Trunk 2: Qwen2.5-14B from {base_q25_14b}...")
            p2_dir = ckpts["qwen25_14b_p2"]
            meta_q25 = read_checkpoint_meta(p2_dir, default_keep=32, default_layers=[3, 5, 8, 11, 16, 21, 27, 32], default_max_len=447)
            trunk2 = UniversalLoRATrunkLoader(base_q25_14b, keep_layers=meta_q25["keep"], layers=meta_q25["layers"], max_len=meta_q25["max_len"])
            merge_lora(trunk2.model, p2_dir, sign=+1.0)
            heads_sd = load_file(str(p2_dir / "heads.safetensors")) if (p2_dir / "heads.safetensors").exists() else {}
            trunk2.load_norm_weights(heads_sd)
            W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)
            p2_labels = meta_q25["labels"]

            Z_all = trunk2.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=meta_q25["end_id"])
            Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]
            update_hybrid_groups(Z_te, prior_scales.get("qwen25_14b_p2", 0.85))

            w_main_p = main_members.get("lora_qwen25_14b", {}).get("plain_weight", 0.20)
            w_main_c = main_members.get("lora_qwen25_14b", {}).get("twin_weight", 0.10)
            if W_p is not None:
                lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, p2_labels))
                LOGIT_MAIN += lp_p * w_main_p
                w_comp_p = comp_members.get("lora_qwen25_14b", {}).get("plain_weight", 0.20)
                comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))

            if W_c is not None:
                lp_c85, valid_c85 = compute_centered_logits(Z_te, groups_085, W_c, b_c, p2_labels)
                LOGIT_MAIN[valid_c85] += lp_c85[valid_c85] * w_main_c
                lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, p2_labels)
                lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, p2_labels)
                lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                w_comp_c = comp_members.get("lora_qwen25_14b", {}).get("twin_weight", 0.08)
                comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))

            res_q25 = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen25_14b_p2", 0.10))
            U_te = transform_lowram(Z_te, res_q25)
            lp_dp = scores_exact_dp(U_te, groups_065, res_q25, D_hybrid)
            dp_posteriors.append((dp_weights.get("q25_14b", 0.24), softmax(lp_dp, axis=1)))

            S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_q25, prior_scales.get("qwen25_14b_p2", 0.85))
            spectral_blocks_tr.append(S_tr)
            spectral_blocks_te.append(S_te)

            trunk2.free_memory()
            del trunk2, Z_all, Z_tr, Z_te, heads_sd, res_q25, U_te
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================================================
        # TRUNK 3: Qwen3-8B (36 layers, hidden=4096)
        # ==============================================================================
        base_q3_8b = find_base_model(input_dir, "qwen3", 4096, 36)
        if base_q3_8b and ("instyle" in ckpts or "qwen3_8b_claude" in ckpts):
            print(f"[Pipeline] Loading Trunk 3: Qwen3-8B from {base_q3_8b}...")
            trunk3 = UniversalLoRATrunkLoader(base_q3_8b, keep_layers=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], max_len=447)

            # In-Style LoRA
            if "instyle" in ckpts:
                instyle_dir = ckpts["instyle"]
                meta_ins = read_checkpoint_meta(instyle_dir, default_keep=24, default_layers=[2, 4, 6, 8, 12, 16, 20, 24], default_max_len=447)
                trunk3.set_active_layers(meta_ins["layers"], meta_ins["max_len"])
                backup_t3 = trunk3.snapshot_weights(instyle_dir)
                merge_lora(trunk3.model, instyle_dir, sign=+1.0)
                heads_sd = load_file(str(instyle_dir / "heads.safetensors")) if (instyle_dir / "heads.safetensors").exists() else {}
                trunk3.load_norm_weights(heads_sd)
                W_p, b_p, W_c, b_c = resolve_head_tensors(heads_sd)

                Z_all = trunk3.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=meta_ins["end_id"])
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]
                update_hybrid_groups(Z_te, prior_scales.get("instyle", 0.45))

                if W_p is not None:
                    w_comp_p = comp_members.get("instyle", {}).get("plain_weight", 0.22)
                    lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, meta_ins["labels"]))
                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))

                if W_c is not None:
                    lp_c85, _ = compute_centered_logits(Z_te, groups_085, W_c, b_c, meta_ins["labels"])
                    lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, meta_ins["labels"])
                    lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, meta_ins["labels"])
                    lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                    w_comp_c = comp_members.get("instyle", {}).get("twin_weight", 0.08)
                    comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))

                res_ins = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("instyle", 0.35))
                U_te = transform_lowram(Z_te, res_ins)
                lp_dp = scores_exact_dp(U_te, groups_065, res_ins, D_hybrid)
                dp_posteriors.append((dp_weights.get("instyle", 0.10), softmax(lp_dp, axis=1)))

                S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_ins, prior_scales.get("instyle", 0.45))
                spectral_blocks_tr.append(S_tr)
                spectral_blocks_te.append(S_te)

                trunk3.restore_weights(backup_t3)
                del backup_t3, Z_all, Z_tr, Z_te, heads_sd, res_ins, U_te
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Claude Head LoRA
            if "qwen3_8b_claude" in ckpts:
                claude_dir = ckpts["qwen3_8b_claude"]
                meta_cl = read_checkpoint_meta(claude_dir, default_keep=24, default_layers=[2, 4, 6, 8, 12, 16, 20, 24], default_max_len=447)
                trunk3.set_active_layers(meta_cl["layers"], meta_cl["max_len"])
                merge_lora(trunk3.model, claude_dir, sign=+1.0)

                heads_file = claude_dir / "heads.safetensors"
                if heads_file.exists():
                    trunk3.load_norm_weights(load_file(str(heads_file)))

                W_p, b_p, W_c, b_c = None, None, None, None
                readouts_file = claude_dir / "readouts.npz"
                if readouts_file.exists():
                    rnpz = np.load(str(readouts_file), allow_pickle=True)
                    for wk, bk in [("plain_coef", "plain_intercept"), ("plain.weight", "plain.bias")]:
                        if wk in rnpz and bk in rnpz and rnpz[wk].shape[0] == 18:
                            W_p, b_p = rnpz[wk], rnpz[bk]
                            break
                    for wk, bk in [("center_coef", "center_intercept"), ("center.weight", "center.bias")]:
                        if wk in rnpz and bk in rnpz and rnpz[wk].shape[0] == 18:
                            W_c, b_c = rnpz[wk], rnpz[bk]
                            break
                if W_p is None and heads_file.exists():
                    cand_W, cand_b, cand_Wc, cand_bc = resolve_head_tensors(load_file(str(heads_file)))
                    if cand_W is not None and cand_W.shape[0] == 18:
                        W_p, b_p = cand_W, cand_b
                        if cand_Wc is not None and cand_Wc.shape[0] == 18:
                            W_c, b_c = cand_Wc, cand_bc

                W_spec, b_spec = None, None
                if readouts_file.exists():
                    rnpz = np.load(str(readouts_file), allow_pickle=True)
                    if "claude_coef" in rnpz and "claude_intercept" in rnpz:
                        if rnpz["claude_coef"].shape[0] == 5:
                            W_spec, b_spec = rnpz["claude_coef"], rnpz["claude_intercept"]
                            if "claude_classes" in rnpz:
                                claude_classes = list(rnpz["claude_classes"])
                if W_spec is None:
                    spec_file = claude_dir / "claude_head.safetensors"
                    if spec_file.exists():
                        ch_sd = load_file(str(spec_file))
                        for wk, bk in [("weight", "bias"), ("plain.weight", "plain.bias")]:
                            if wk in ch_sd and ch_sd[wk].shape[0] == 5:
                                W_spec = ch_sd[wk].cpu().numpy()
                                b_spec = ch_sd[bk].cpu().numpy() if bk in ch_sd else np.zeros(5, dtype=np.float32)
                                break
                if claude_classes is None:
                    claude_classes = [LABELS[i] for i in FAMILIES["claude"]]

                Z_all = trunk3.extract_exact(test_texts + train_sub_texts, batch_size=8, end_id=meta_cl["end_id"])
                Z_te, Z_tr = Z_all[:N_test], Z_all[N_test:]
                update_hybrid_groups(Z_te, prior_scales.get("qwen3_8b_claude", 0.30))

                if W_spec is not None and W_spec.shape[0] == 5:
                    claude_spec_logits = Z_te @ W_spec.T + b_spec

                if W_p is not None and W_p.shape[0] == 18:
                    w_comp_p = comp_members.get("ntl_claude_head", {}).get("plain_weight", 0.14)
                    lp_p = log_softmax(align_cols(Z_te @ W_p.T + b_p, LABELS))
                    comp_lora_posteriors.append((w_comp_p, softmax(lp_p, axis=1)))
                    if W_c is not None and W_c.shape[0] == 18:
                        w_comp_c = comp_members.get("ntl_claude_head", {}).get("twin_weight", 0.05)
                        lp_c85, _ = compute_centered_logits(Z_te, groups_085, W_c, b_c, LABELS)
                        lp_c72, _ = compute_centered_logits(Z_te, groups_072, W_c, b_c, LABELS)
                        lp_c60, _ = compute_centered_logits(Z_te, groups_060, W_c, b_c, LABELS)
                        lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                        comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))

                res_cl = fit_shared_gaussian_lowram(Z_tr, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("qwen3_8b_claude", 0.35))
                U_te = transform_lowram(Z_te, res_cl)
                lp_dp = scores_exact_dp(U_te, groups_065, res_cl, D_hybrid)
                dp_posteriors.append((dp_weights.get("claude_8b", 0.04), softmax(lp_dp, axis=1)))

                S_tr, S_te = make_spectral_blocks(Z_tr, Z_te, res_cl, prior_scales.get("qwen3_8b_claude", 0.30))
                spectral_blocks_tr.append(S_tr)
                spectral_blocks_te.append(S_te)

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
        if "deberta_p2" in ckpts:
            deb_dir = ckpts["deberta_p2"]
            base_deb_dir = find_base_model(input_dir, "deberta", 1024, 24)
            print(f"[Pipeline] Loading Trunk 4: DeBERTa-v3 from {deb_dir}...")
            Z_te_deb, Z_tr_deb, W_p_deb, b_p_deb, W_c_deb, b_c_deb, deb_labels = extract_deberta_checkpoint(
                deb_dir, base_deb_dir, test_texts, train_sub_texts, y_sub
            )
            update_hybrid_groups(Z_te_deb, prior_scales.get("deberta_p2", 0.80))

            if W_p_deb is not None:
                w_main_deb = main_members.get("deberta_v3_large", {}).get("plain_weight", 0.25)
                lp_deb = log_softmax(align_cols(Z_te_deb @ W_p_deb.T + b_p_deb, deb_labels))
                LOGIT_MAIN += lp_deb * w_main_deb

                w_comp_deb = comp_members.get("deberta_v3_large", {}).get("plain_weight", 0.18)
                comp_lora_posteriors.append((w_comp_deb, softmax(lp_deb, axis=1)))

            if W_c_deb is not None:
                lp_c85, _ = compute_centered_logits(Z_te_deb, groups_085, W_c_deb, b_c_deb, deb_labels)
                lp_c72, _ = compute_centered_logits(Z_te_deb, groups_072, W_c_deb, b_c_deb, deb_labels)
                lp_c60, _ = compute_centered_logits(Z_te_deb, groups_060, W_c_deb, b_c_deb, deb_labels)
                lp_c_blend = w85 * lp_c85 + w72 * lp_c72 + w60 * lp_c60
                w_comp_c = comp_members.get("deberta_v3_large", {}).get("twin_weight", 0.06)
                comp_lora_posteriors.append((w_comp_c, softmax(lp_c_blend, axis=1)))

            res_deb = fit_shared_gaussian_lowram(Z_tr_deb, y_sub, prompts_sub, d_target=256, se_floor=se_floors.get("deberta_p2", 0.10))
            U_te = transform_lowram(Z_te_deb, res_deb)
            lp_dp = scores_exact_dp(U_te, groups_065, res_deb, D_hybrid)
            dp_posteriors.append((dp_weights.get("deberta", 0.20), softmax(lp_dp, axis=1)))

            S_tr, S_te = make_spectral_blocks(Z_tr_deb, Z_te_deb, res_deb, prior_scales.get("deberta_p2", 0.80))
            spectral_blocks_tr.append(S_tr)
            spectral_blocks_te.append(S_te)

            del Z_tr_deb, Z_te_deb, res_deb, U_te
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

    # Unified Handwriting & Orthographic Shield (discrete Bayesian rules + continuous handwriting density)
    default_curly_multiplier = float(np.exp(-1.0))
    multiplier = post_cfg.get("curly_quote_multiplier", default_curly_multiplier)
    multi_em_dash_log_pen = post_cfg.get("multi_em_dash_log_penalty", -0.45)
    straight_contr_log_pen = post_cfg.get("straight_contr_log_penalty", -0.60)
    curly_contr_log_pen = post_cfg.get("curly_contr_log_penalty", -0.40)
    continuous_hw_weight = post_cfg.get("continuous_hw_weight", 0.04)

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

    # 300-Step Sinkhorn Mass Balancing
    sinkhorn_iters = post_cfg.get("sinkhorn_iterations", 300)
    P_final = sinkhorn_balancing(P_mix, iters=sinkhorn_iters)

    # Generate Submission Artifacts
    df_sub, out_parquet, out_csv = generate_submission(P_final, test_ids, working_dir)
    elapsed = time.time() - start_time
    print(f"[Pipeline] Submission generated successfully in {elapsed:.2f}s: {out_parquet} ({len(df_sub)} rows)")

    return df_sub, out_parquet
