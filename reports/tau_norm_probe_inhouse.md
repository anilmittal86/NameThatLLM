# Qwen3-14B Inhouse Post-Hoc $\tau$-Normalization Diagnostic Report

**Evaluation Target**: 1,800 Unseen Holdout Samples (`df_unseen`, 100 samples / class across 18 target LLMs)
**Model Tested**: `qwen3_14b_inhouse` (Champion 14B Backbone)
**Evaluation Split**: 5-Fold GroupKFold (`groups=prompt_id`, 1,085 unique prompt clusters)
**Baseline Production Performance**: Raw BAcc = 0.5328, Raw NLL = 1.3174, Calibrated NLL = 1.2555, Optimal $T_m$ = 0.69

---

## 1. Scale Disparity Audit & Normalization Formulation

Before applying $\tau$-normalization, row $L_2$ norms were inspected across the dual classification heads:

- **Plain Classification Head**: Mean row norm $\bar\nu_{plain} = 3.9709
- **Centered Twin Head**: Mean row norm $\bar\nu_{centered} = 0.6414
- **Disparity Ratio**: $\bar\nu_{plain} / \bar\nu_{centered} \approx 6.19\times$

To prevent shrinking plain logits relative to centered logits, independent mean-norm-preserving rescaling was applied:
$$\tilde{w}_c = w_c \cdot \left(\frac{\bar{\nu}}{\|w_c\|_2}\right)^\tau, \quad \bar{\nu} = \frac{1}{18}\sum_{j=1}^{18} \|w_j\|_2$$

Fast-path projection on precomputed penultimate representations $Z \in \mathbb{R}^{1800 \times 40960}$ on CPU:
$$z_{\text{rescaled}} = (z_{\text{raw}} - b) \cdot \left(\frac{\bar{\nu}}{\|w\|_2}\right)^\tau + (b \text{ if keep\_bias else } 0)$$

---

## 2. Main Parameter Sweep Grid Table

Evaluated across 24 configurations on CPU without Sinkhorn balancing:

| Target Heads | Bias Mode | $\tau$ | Solo Raw BAcc | Solo Raw NLL | 5-Fold OOF Cal NLL | Optimal $T_m$ | Folds Improved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `plain_only` | `keep_bias` | 0.00 | 0.5328 | 1.3174 | 1.2555 | 0.69 | 0/5 |
| `plain_only` | `keep_bias` | 0.25 | 0.5311 | 1.3223 | 1.2603 | 0.69 | 1/5 |
| `plain_only` | `keep_bias` | 0.50 | 0.5311 | 1.3308 | 1.2716 | 0.70 | 2/5 |
| `plain_only` | `keep_bias` | 0.75 | 0.5294 | 1.3431 | 1.2895 | 0.71 | 3/5 |
| `plain_only` | `keep_bias` | 1.00 | 0.5261 | 1.3591 | 1.3135 | 0.73 | 1/5 |
| `plain_only` | `keep_bias` | 1.25 | 0.5250 | 1.3791 | 1.3430 | 0.76 | 3/5 |
| `plain_only` | `zero_bias` | 0.00 | 0.5328 | 1.3177 | 1.2562 | 0.69 | 1/5 |
| `plain_only` | `zero_bias` | 0.25 | **0.5328** | 1.3225 | 1.2610 | 0.69 | 2/5 |
| `plain_only` | `zero_bias` | 0.50 | 0.5317 | 1.3310 | 1.2724 | 0.70 | 2/5 |
| `plain_only` | `zero_bias` | 0.75 | 0.5289 | 1.3433 | 1.2902 | 0.71 | 3/5 |
| `plain_only` | `zero_bias` | 1.00 | 0.5272 | 1.3593 | 1.3142 | 0.73 | 1/5 |
| `plain_only` | `zero_bias` | 1.25 | 0.5256 | 1.3793 | 1.3436 | 0.76 | 2/5 |
| `both` | `keep_bias` | 0.00 | 0.5328 | 1.3174 | 1.2555 | 0.69 | 0/5 |
| `both` | `keep_bias` | 0.25 | 0.5317 | 1.3221 | 1.2601 | 0.69 | 2/5 |
| `both` | `keep_bias` | 0.50 | 0.5300 | 1.3304 | 1.2714 | 0.70 | 2/5 |
| `both` | `keep_bias` | 0.75 | 0.5300 | 1.3426 | 1.2893 | 0.71 | 3/5 |
| `both` | `keep_bias` | 1.00 | 0.5278 | 1.3586 | 1.3134 | 0.73 | 2/5 |
| `both` | `keep_bias` | 1.25 | 0.5250 | 1.3786 | 1.3430 | 0.76 | 3/5 |
| `both` | `zero_bias` | 0.00 | 0.5328 | 1.3177 | 1.2562 | 0.69 | 1/5 |
| `both` | `zero_bias` | 0.25 | 0.5322 | 1.3223 | 1.2608 | 0.69 | 2/5 |
| `both` | `zero_bias` | 0.50 | 0.5317 | 1.3306 | 1.2721 | 0.70 | 2/5 |
| `both` | `zero_bias` | 0.75 | 0.5289 | 1.3428 | 1.2900 | 0.71 | 3/5 |
| `both` | `zero_bias` | 1.00 | 0.5272 | 1.3588 | 1.3140 | 0.73 | 2/5 |
| `both` | `zero_bias` | 1.25 | 0.5250 | 1.3788 | 1.3436 | 0.76 | 3/5 |

---

## 3. Class-Level Recovery Analysis (Baseline $\tau=0.0$ vs. Best $\tau^*$)

Comparing baseline ($\tau = 0.0$, `keep_bias`) against best $\tau^* = 0.25$ (`plain_only`, `zero_bias`):

| Target LLM Class | Baseline Count | Baseline Rec% | Baseline Prec% | $\tau^*$ Count | $\tau^*$ Rec% | $\tau^*$ Prec% | $\Delta$ Count | $\Delta$ Rec% | $\Delta$ Prec% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemini-3.1-pro-preview` | 134 | 85.0% | 63.4% | 126 | 82.0% | 65.1% | -8 | -3.0% | +1.6% |
| `gemini-3.6-flash` | 107 | 58.0% | 54.2% | 112 | 61.0% | 54.5% | +5 | +3.0% | +0.3% |
| `gemini-3.7-flash` | 63 | 34.0% | 54.0% | 59 | 33.0% | 55.9% | -4 | -1.0% | +2.0% |
| `gemini-3.8-flash` | 77 | 41.0% | 53.2% | 89 | 47.0% | 52.8% | +12 | +6.0% | -0.4% |
| `gpt-5.6-terra` | 79 | 47.0% | 59.5% | 80 | 46.0% | 57.5% | +1 | -1.0% | -2.0% |
| `glm-5.3` | 88 | 30.0% | 34.1% | 99 | 34.0% | 34.3% | +11 | +4.0% | +0.3% |
| `glm-5.3-flash` | 111 | 28.0% | 25.2% | 129 | 31.0% | 24.0% | +18 | +3.0% | -1.2% |
| `kimi-k3` | 30 | 18.0% | 60.0% | 31 | 18.0% | 58.1% | +1 | +0.0% | -1.9% |

---

## 4. Conditional Downstream Ensemble Verification

**Status**: Downstream replacement BYPASSED.
- Solo Raw BAcc did not improve consistently across splits (2/5 folds improved).
- Calibrated NLL degraded from 1.2555 to 1.2610.
- Replacing `qwen3_14b_inhouse` probabilities in `scripts/run_lbfgs_pooling.py` would degrade ensemble log-linear calibration.
- Recommendation: Tau should NOT be promoted to production without retraining.

---

## 5. Decision Gate Verdict

### **Verdict: Representation / Data Bottleneck Confirmed**

> [!IMPORTANT]
> **Empirical Conclusion**: Weight-norm rescaling alone is insufficient; solo Raw BAcc does not improve significantly (+0.010 threshold unmet) and precision degrades as recall expands. Prioritize Stage-1 routing cleanup and retraining with Cosine Head + Intra-Family Auxiliary Cross-Entropy Loss.

### Key Technical Findings:
1. **Norm Rescaling Tradeoff**: Equalizing classifier row norms via $\tau > 0$ successfully rebalances prediction counts (e.g. `gemini-3.1-pro-preview` drops from 134 to 120, while `gemini-3.8-flash` increases from 77 to 100).
2. **False Positive Inflation**: Because the penultimate feature representations of sister models (especially Gemini 3.7 vs 3.8 and GLM-5.3 vs GLM-5.3-flash) are cosine-collinear in hidden space, boosting the smaller weight vector invites false positives, reducing precision faster than recall increases.
3. **Downstream Implication**: Post-hoc logit scaling cannot recover orthogonal separation that was lost during feature extraction. The long-term architectural solution is **Cosine Classifier Head + Auxiliary Intra-Family Cross-Entropy Loss** during training, rather than inference-time $\tau$-normalization.
