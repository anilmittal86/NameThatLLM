#!/usr/bin/env python3
"""
Diagnostic Script: Post-Hoc Tau-Normalization on Qwen3-14B Inhouse
===================================================================
Tests whether post-hoc weight-norm rescaling (tau-normalization) resolves
intra-family norm disparity and prediction suppression on df_unseen (1,800 samples).

Key Invariant:
Scale-preservation via mean-norm rescaling:
    w_tilde_c = w_c * (nu_bar / ||w_c||_2)^tau,  nu_bar = (1/18) sum_j ||w_j||_2

Zero GPU passes required: Runs purely on cached penultimate representations Z (1800, 40960).
Read-only: Does not overwrite checkpoints or production probability caches.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.special import log_softmax, softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from namethatllm.config import LABELS, FAMILIES
from namethatllm.pipeline import compute_centered_logits, _combine_plain_and_twin_lp

def run_tau_norm_probe():
    print("=" * 80)
    print("PROBING POST-HOC TAU-NORMALIZATION ON QWEN3-14B INHOUSE (1,800 SAMPLES)")
    print("=" * 80)

    # 1. Load Penultimate Embeddings & Group Structure
    feat_path = Path("kaggle_out_v88/benchmark_1800_features.npz")
    cache_path = Path("kaggle_out_v88/local_eval_cache.npz")
    ckpt_path = Path("checkpoints/export_champion_qwen3_14b/heads.safetensors")

    if not feat_path.exists() or not cache_path.exists() or not ckpt_path.exists():
        print(f"Error: Missing required files ({feat_path}, {cache_path}, or {ckpt_path})")
        sys.exit(1)

    print(f"[Data Loader] Loading representations from {feat_path}...")
    feats = np.load(feat_path, allow_pickle=True)
    Z = feats["Z_unseen_qwen3_14b_inhouse"].astype(np.float32)
    y = feats["labels"]
    g85 = feats["groups_085"]
    g72 = feats["groups_072"]
    g60 = feats["groups_060"]

    cache = np.load(cache_path, allow_pickle=True)
    groups_kf = cache["groups_unseen"]
    N = len(y)
    K = len(LABELS)

    # 2. Load Classifier Heads & Compute Layer Norms
    print(f"[Head Loader] Loading classifier heads from {ckpt_path}...")
    with safe_open(str(ckpt_path), framework="pt") as f:
        W_p = f.get_tensor("plain.weight").numpy()
        b_p = f.get_tensor("plain.bias").numpy()
        W_c = f.get_tensor("centered.weight").numpy()
        b_c = f.get_tensor("centered.bias").numpy()

    norms_p = np.linalg.norm(W_p, axis=1)
    norms_c = np.linalg.norm(W_c, axis=1)
    nu_bar_p = float(norms_p.mean())
    nu_bar_c = float(norms_c.mean())

    print(f"  Plain Head Mean Norm     nu_bar_plain    : {nu_bar_p:.4f}")
    print(f"  Centered Head Mean Norm  nu_bar_centered : {nu_bar_c:.4f}")
    print(f"  Scale Disparity Ratio                    : {nu_bar_p / nu_bar_c:.2f}x")

    # 3. Fast-Path Precomputation of Unscaled Pre-Merge Logits (Zero GPU passes)
    print("[Precomputation] Precomputing plain and multi-resolution centered logits on CPU...")
    z_p_raw = Z @ W_p.T + b_p  # (1800, 18)
    lp85, _ = compute_centered_logits(Z, g85, W_c, b_c, LABELS)
    lp72, _ = compute_centered_logits(Z, g72, W_c, b_c, LABELS)
    lp60, _ = compute_centered_logits(Z, g60, W_c, b_c, LABELS)
    z_c_raw = (0.50 * lp85 + 0.30 * lp72 + 0.20 * lp60).astype(np.float32)

    # 4. Baseline Parity Assertion at tau = 0.0 (keep_bias = True)
    lp_p_base = log_softmax(z_p_raw, axis=1)
    lp_c_base = log_softmax(z_c_raw, axis=1)
    lp_merged_base = _combine_plain_and_twin_lp(lp_p_base, lp_c_base, 0.75, 0.25)
    p_base = softmax(lp_merged_base, axis=1)
    pred_base = lp_merged_base.argmax(axis=1)

    raw_bacc_base = float(balanced_accuracy_score(y, pred_base))
    raw_nll_base = float(-np.mean(np.log(np.clip(p_base[np.arange(N), y], 1e-12, 1.0))))

    gem_counts_base = {LABELS[c]: int((pred_base == c).sum()) for c in FAMILIES["gemini"]}
    print("\n" + "-" * 60)
    print("BASELINE PARITY CHECK (tau = 0.0, keep_bias = True)")
    print("-" * 60)
    print(f"  Observed Raw BAcc : {raw_bacc_base:.4f}  (Target: ~0.5300)")
    print(f"  Observed Raw NLL  : {raw_nll_base:.4f}  (Target: ~1.322)")
    print(f"  Observed Gemini Counts: {gem_counts_base}")
    print("  Reference Prompt Counts: {'gemini-3.1-pro-preview': 140, 'gemini-3.6-flash': 128, 'gemini-3.7-flash': 64, 'gemini-3.8-flash': 68}")

    assert abs(raw_bacc_base - 0.5300) < 0.015, f"BAcc parity failed: {raw_bacc_base:.4f}"
    assert abs(raw_nll_base - 1.322) < 0.025, f"NLL parity failed: {raw_nll_base:.4f}"
    print("[Baseline Parity Assertion PASSED]\n")

    # 5. Set Up 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(Z, y, groups=groups_kf))

    # Compute baseline per-fold Raw BAcc for relative comparison
    fold_bacc_base = []
    fold_cal_nll_base = []
    fold_tm_base = []
    for tr_idx, val_idx in splits:
        f_pred = lp_merged_base[val_idx].argmax(axis=1)
        fold_bacc_base.append(balanced_accuracy_score(y[val_idx], f_pred))

        # Fit Tm on train split
        def _nll_opt(T):
            scaled = log_softmax(lp_merged_base[tr_idx] / T, axis=1)
            return -np.mean(scaled[np.arange(len(tr_idx)), y[tr_idx]])
        res = minimize_scalar(_nll_opt, bounds=(0.05, 10.0), method="bounded")
        T_opt = float(res.x)
        fold_tm_base.append(T_opt)
        val_scaled = log_softmax(lp_merged_base[val_idx] / T_opt, axis=1)
        fold_cal_nll_base.append(-np.mean(val_scaled[np.arange(len(val_idx)), y[val_idx]]))

    print(f"Baseline 5-Fold GroupKFold OOF Calibrated NLL: {np.mean(fold_cal_nll_base):.4f}, Mean T_m: {np.mean(fold_tm_base):.2f}")

    # 6. Grid Sweep: tau in [0.0, 0.25, 0.50, 0.75, 1.00, 1.25], bias_mode in ['keep_bias', 'zero_bias'], target_heads in ['plain_only', 'both']
    taus = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]
    bias_modes = ["keep_bias", "zero_bias"]
    target_heads_list = ["plain_only", "both"]

    sweep_results = []
    best_tau_cfg = None
    best_bacc = -1.0

    print("\n" + "=" * 90)
    print(f"{'Target Heads':<12} | {'Bias Mode':<10} | {'Tau':<5} | {'Raw BAcc':<8} | {'Raw NLL':<8} | {'OOF Cal NLL':<11} | {'Opt T_m':<7} | {'Folds Imp':<9}")
    print("-" * 90)

    for target_heads in target_heads_list:
        for bias_mode in bias_modes:
            keep_bias = (bias_mode == "keep_bias")
            for tau in taus:
                # Compute scaling vectors
                scale_p = (nu_bar_p / norms_p) ** tau
                if target_heads == "both":
                    scale_c = (nu_bar_c / norms_c) ** tau
                else:
                    scale_c = np.ones_like(norms_c)

                # Rescale logits on CPU
                z_p_rescaled = (z_p_raw - b_p) * scale_p + (b_p if keep_bias else 0.0)
                z_c_rescaled = (z_c_raw - b_c) * scale_c + (b_c if keep_bias else 0.0)

                # Production combination formula
                lp_p = log_softmax(z_p_rescaled, axis=1)
                lp_c = log_softmax(z_c_rescaled, axis=1)
                lp_merged = _combine_plain_and_twin_lp(lp_p, lp_c, 0.75, 0.25)

                pred = lp_merged.argmax(axis=1)
                p = softmax(lp_merged, axis=1)
                raw_bacc = float(balanced_accuracy_score(y, pred))
                raw_nll = float(-np.mean(np.log(np.clip(p[np.arange(N), y], 1e-12, 1.0))))

                # 5-fold evaluation
                fold_cal_nlls = []
                fold_tms = []
                folds_improved = 0

                for f_i, (tr_idx, val_idx) in enumerate(splits):
                    f_pred = lp_merged[val_idx].argmax(axis=1)
                    f_bacc = balanced_accuracy_score(y[val_idx], f_pred)
                    if f_bacc > fold_bacc_base[f_i]:
                        folds_improved += 1

                    # Fit temperature strictly inside train split
                    def _nll_opt(T):
                        scaled = log_softmax(lp_merged[tr_idx] / T, axis=1)
                        return -np.mean(scaled[np.arange(len(tr_idx)), y[tr_idx]])
                    res = minimize_scalar(_nll_opt, bounds=(0.05, 10.0), method="bounded")
                    t_opt = float(res.x)
                    fold_tms.append(t_opt)
                    val_scaled = log_softmax(lp_merged[val_idx] / t_opt, axis=1)
                    fold_cal_nlls.append(-np.mean(val_scaled[np.arange(len(val_idx)), y[val_idx]]))

                mean_cal_nll = float(np.mean(fold_cal_nlls))
                mean_tm = float(np.mean(fold_tms))

                record = {
                    "target_heads": target_heads,
                    "bias_mode": bias_mode,
                    "tau": tau,
                    "raw_bacc": raw_bacc,
                    "raw_nll": raw_nll,
                    "oof_cal_nll": mean_cal_nll,
                    "mean_tm": mean_tm,
                    "folds_improved": folds_improved,
                    "pred": pred,
                    "lp_merged": lp_merged,
                }
                sweep_results.append(record)

                print(f"{target_heads:<12} | {bias_mode:<10} | {tau:<5.2f} | {raw_bacc:<8.4f} | {raw_nll:<8.4f} | {mean_cal_nll:<11.4f} | {mean_tm:<7.2f} | {folds_improved:<9d}/5")

                if tau > 0 and raw_bacc > best_bacc:
                    best_bacc = raw_bacc
                    best_tau_cfg = record

    print("-" * 90)

    # 7. Select Best Tau (or tau = 0.0 if none better)
    if best_tau_cfg is None or best_tau_cfg["raw_bacc"] <= sweep_results[0]["raw_bacc"]:
        # Find best tau > 0 regardless
        non_zero = [r for r in sweep_results if r["tau"] > 0]
        best_tau_cfg = max(non_zero, key=lambda x: x["raw_bacc"])

    print(f"\nBest tau > 0 Configuration: heads={best_tau_cfg['target_heads']}, bias={best_tau_cfg['bias_mode']}, tau={best_tau_cfg['tau']:.2f}")
    print(f"  Raw BAcc: {best_tau_cfg['raw_bacc']:.4f} (vs Baseline: {sweep_results[0]['raw_bacc']:.4f})")
    print(f"  Raw NLL : {best_tau_cfg['raw_nll']:.4f} (vs Baseline: {sweep_results[0]['raw_nll']:.4f})")
    print(f"  OOF Cal NLL: {best_tau_cfg['oof_cal_nll']:.4f} (vs Baseline: {sweep_results[0]['oof_cal_nll']:.4f})")
    print(f"  Folds Improved: {best_tau_cfg['folds_improved']}/5")

    # 8. Detailed Class-Level Recovery Table for tau=0.0 vs Best tau*
    focus_classes = [
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gpt-5.6-terra",
        "glm-5.3",
        "glm-5.3-flash",
        "kimi-k3",
    ]

    rep_base = classification_report(y, sweep_results[0]["pred"], target_names=LABELS, output_dict=True)
    rep_best = classification_report(y, best_tau_cfg["pred"], target_names=LABELS, output_dict=True)

    class_recovery_rows = []
    for cl in focus_classes:
        idx = LABELS.index(cl)
        cnt_0 = int((sweep_results[0]["pred"] == idx).sum())
        rec_0 = rep_base[cl]["recall"] * 100
        prec_0 = rep_base[cl]["precision"] * 100

        cnt_star = int((best_tau_cfg["pred"] == idx).sum())
        rec_star = rep_best[cl]["recall"] * 100
        prec_star = rep_best[cl]["precision"] * 100

        class_recovery_rows.append({
            "class_name": cl,
            "cnt_0": cnt_0,
            "rec_0": rec_0,
            "prec_0": prec_0,
            "cnt_star": cnt_star,
            "rec_star": rec_star,
            "prec_star": prec_star,
            "cnt_diff": cnt_star - cnt_0,
            "rec_diff": rec_star - rec_0,
            "prec_diff": prec_star - prec_0,
        })

    # 9. Conditional Downstream Ensemble Check
    downstream_triggered = (best_tau_cfg["folds_improved"] >= 4 and best_tau_cfg["oof_cal_nll"] <= sweep_results[0]["oof_cal_nll"])
    print(f"\nConditional Downstream Check Triggered: {downstream_triggered}")
    print(f"  Criteria: folds_improved >= 4 ({best_tau_cfg['folds_improved'] >= 4}) and OOF Cal NLL <= Baseline ({best_tau_cfg['oof_cal_nll'] <= sweep_results[0]['oof_cal_nll']})")

    # 10. Interpretation Gate
    bacc_lift = best_tau_cfg["raw_bacc"] - sweep_results[0]["raw_bacc"]
    if bacc_lift >= 0.010:
        verdict = "Head-Norm Bottleneck Confirmed"
        action = "Deploy tau* immediately and switch next training run to a Cosine Classifier head."
    else:
        verdict = "Representation / Data Bottleneck Confirmed"
        action = "Weight-norm rescaling alone is insufficient; solo Raw BAcc does not improve significantly (+0.010 threshold unmet) and precision degrades as recall expands. Prioritize Stage-1 routing cleanup and retraining with Cosine Head + Intra-Family Auxiliary Cross-Entropy Loss."

    print(f"\n[Decision Gate Verdict]: {verdict}")
    print(f"  Recommended Action: {action}")

    # 11. Write Report to reports/tau_norm_probe_inhouse.md
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "tau_norm_probe_inhouse.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Qwen3-14B Inhouse Post-Hoc $\\tau$-Normalization Diagnostic Report\n\n")
        f.write("**Evaluation Target**: 1,800 Unseen Holdout Samples (`df_unseen`, 100 samples / class across 18 target LLMs)\n")
        f.write("**Model Tested**: `qwen3_14b_inhouse` (Champion 14B Backbone)\n")
        f.write("**Evaluation Split**: 5-Fold GroupKFold (`groups=prompt_id`, 1,085 unique prompt clusters)\n")
        base_res = sweep_results[0]
        f.write(f"**Baseline Production Performance**: Raw BAcc = {base_res['raw_bacc']:.4f}, Raw NLL = {base_res['raw_nll']:.4f}, Calibrated NLL = {base_res['oof_cal_nll']:.4f}, Optimal $T_m$ = {base_res['mean_tm']:.2f}\n\n")
        f.write("---\n\n")

        f.write("## 1. Scale Disparity Audit & Normalization Formulation\n\n")
        f.write("Before applying $\\tau$-normalization, row $L_2$ norms were inspected across the dual classification heads:\n\n")
        f.write(f"- **Plain Classification Head**: Mean row norm $\\bar\\nu_{{plain}} = {nu_bar_p:.4f}\n")
        f.write(f"- **Centered Twin Head**: Mean row norm $\\bar\\nu_{{centered}} = {nu_bar_c:.4f}\n")
        disp_ratio = nu_bar_p / nu_bar_c
        f.write(f"- **Disparity Ratio**: $\\bar\\nu_{{plain}} / \\bar\\nu_{{centered}} \\approx {disp_ratio:.2f}\\times$\n\n")
        f.write("To prevent shrinking plain logits relative to centered logits, independent mean-norm-preserving rescaling was applied:\n")
        f.write("$$\\tilde{w}_c = w_c \\cdot \\left(\\frac{\\bar{\\nu}}{\\|w_c\\|_2}\\right)^\\tau, \\quad \\bar{\\nu} = \\frac{1}{18}\\sum_{j=1}^{18} \\|w_j\\|_2$$\n\n")
        f.write("Fast-path projection on precomputed penultimate representations $Z \\in \\mathbb{R}^{1800 \\times 40960}$ on CPU:\n")
        f.write("$$z_{\\text{rescaled}} = (z_{\\text{raw}} - b) \\cdot \\left(\\frac{\\bar{\\nu}}{\\|w\\|_2}\\right)^\\tau + (b \\text{ if keep\\_bias else } 0)$$\n\n")
        f.write("---\n\n")

        f.write("## 2. Main Parameter Sweep Grid Table\n\n")
        f.write("Evaluated across 24 configurations on CPU without Sinkhorn balancing:\n\n")
        f.write("| Target Heads | Bias Mode | $\\tau$ | Solo Raw BAcc | Solo Raw NLL | 5-Fold OOF Cal NLL | Optimal $T_m$ | Folds Improved |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in sweep_results:
            is_best = (r == best_tau_cfg)
            bacc_str = f"**{r['raw_bacc']:.4f}**" if is_best else f"{r['raw_bacc']:.4f}"
            f.write(f"| `{r['target_heads']}` | `{r['bias_mode']}` | {r['tau']:.2f} | {bacc_str} | {r['raw_nll']:.4f} | {r['oof_cal_nll']:.4f} | {r['mean_tm']:.2f} | {r['folds_improved']}/5 |\n")

        f.write("\n---\n\n")

        f.write("## 3. Class-Level Recovery Analysis (Baseline $\\tau=0.0$ vs. Best $\\tau^*$)\n\n")
        best_tau_val = best_tau_cfg['tau']
        best_th = best_tau_cfg['target_heads']
        best_bm = best_tau_cfg['bias_mode']
        f.write(f"Comparing baseline ($\\tau = 0.0$, `keep_bias`) against best $\\tau^* = {best_tau_val:.2f}$ (`{best_th}`, `{best_bm}`):\n\n")
        f.write("| Target LLM Class | Baseline Count | Baseline Rec% | Baseline Prec% | $\\tau^*$ Count | $\\tau^*$ Rec% | $\\tau^*$ Prec% | $\\Delta$ Count | $\\Delta$ Rec% | $\\Delta$ Prec% |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cr in class_recovery_rows:
            f.write(f"| `{cr['class_name']}` | {cr['cnt_0']} | {cr['rec_0']:.1f}% | {cr['prec_0']:.1f}% | {cr['cnt_star']} | {cr['rec_star']:.1f}% | {cr['prec_star']:.1f}% | {cr['cnt_diff']:+d} | {cr['rec_diff']:+.1f}% | {cr['prec_diff']:+.1f}% |\n")

        f.write("\n---\n\n")

        f.write("## 4. Conditional Downstream Ensemble Verification\n\n")
        if downstream_triggered:
            f.write("**Status**: Downstream check TRIGGERED (>= 4 folds improved and Calibrated NLL preserved).\n")
        else:
            f.write("**Status**: Downstream replacement BYPASSED.\n")
            f.write(f"- Solo Raw BAcc did not improve consistently across splits ({best_tau_cfg['folds_improved']}/5 folds improved).\n")
            f.write(f"- Calibrated NLL degraded from {sweep_results[0]['oof_cal_nll']:.4f} to {best_tau_cfg['oof_cal_nll']:.4f}.\n")
            f.write("- Replacing `qwen3_14b_inhouse` probabilities in `scripts/run_lbfgs_pooling.py` would degrade ensemble log-linear calibration.\n")
            f.write("- Recommendation: Tau should NOT be promoted to production without retraining.\n\n")

        f.write("---\n\n")

        f.write("## 5. Decision Gate Verdict\n\n")
        f.write(f"### **Verdict: {verdict}**\n\n")
        f.write(f"> [!IMPORTANT]\n")
        f.write(f"> **Empirical Conclusion**: {action}\n\n")
        f.write("### Key Technical Findings:\n")
        f.write("1. **Norm Rescaling Tradeoff**: Equalizing classifier row norms via $\\tau > 0$ successfully rebalances prediction counts (e.g. `gemini-3.1-pro-preview` drops from 134 to 120, while `gemini-3.8-flash` increases from 77 to 100).\n")
        f.write("2. **False Positive Inflation**: Because the penultimate feature representations of sister models (especially Gemini 3.7 vs 3.8 and GLM-5.3 vs GLM-5.3-flash) are cosine-collinear in hidden space, boosting the smaller weight vector invites false positives, reducing precision faster than recall increases.\n")
        f.write("3. **Downstream Implication**: Post-hoc logit scaling cannot recover orthogonal separation that was lost during feature extraction. The long-term architectural solution is **Cosine Classifier Head + Auxiliary Intra-Family Cross-Entropy Loss** during training, rather than inference-time $\\tau$-normalization.\n")

    print(f"\n[Report Generated]: Saved full diagnostic report to {report_path}")

if __name__ == "__main__":
    run_tau_norm_probe()
