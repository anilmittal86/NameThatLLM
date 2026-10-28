# Name That LLM: High-Precision Authorship Attribution Pipeline

[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.612%20(SOTA)-brightgreen)](https://www.kaggle.com/competitions/name-that-llm)
[![Architecture Version](https://img.shields.io/badge/Production%20Version-Version%2040%20(Full--Rank%20wPCA%20%2B%20Spectral%20Bank)-blue)](#)
[![Models](https://img.shields.io/badge/Backbones-Qwen3--14B%20%7C%20Qwen2.5--14B%20%7C%20Qwen3--8B%20%7C%20DeBERTa-orange)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)

This repository contains the end-to-end, competition-grade machine learning solution for the **Kaggle "Name That LLM"** competition. The objective is to identify which of 18 specific Large Language Models authored an unattributed response passage.

---

## Key Milestones & Progression Scorecard

| Version | Architecture / Pipeline Components | Training BAcc | Validation CV | Public LB | Generalization & Overfitting Intuition |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **V1** | Flat Word N-Gram TF-IDF Baseline | **1.000 (100%)** | 0.335 (33.5%) | **0.321** | Extreme Overfitting: Bag-of-words memorizes prompt tokens perfectly but fails on unseen test prompts. |
| **V9** | DeBERTa-v3 Multi-Layer + ANOVA Specialists | **0.892 (89.2%)** | 0.362 (36.2%) | **0.336** | Moderate Overfitting: Hungarian matching inflated dense fold CV (0.400) but degraded on sparse test prompts. |
| **V12** | Qwen3-8B LoRA Base (Attention Sink Unmasked) | **0.993 (99.3%)** | 0.497 (49.7%) | **0.347** | Parity Drop: Token-0 attention sink corrupted live test vectors, destroying performance despite 99.3% train fit. |
| **V15** | Verified Qwen3-8B LoRA + Token-0 Mask (`m[:, 0] = 0`) + Soft Centering | **0.993 (99.3%)** | 0.542 (54.2%) | **0.491** | Breakthrough Parity: Attention sink zeroed; training fidelity restored; generalization gap narrowed to 0.051. |
| **V18** | Multi-Backbone Two-Stage Pipeline Refactor | **1.000 (100%)** | 0.615 (61.5%) | **Build Valid.** | Code Validation: Verified 14B/8B multi-backbone pipeline on 2xT4 GPUs; fixed PyTorch tensor `.detach()` requirement. |
| **V19** | Multi-Backbone Two-Stage 14B/8B LoRAs + wPCA Gaussian Readout + 4-View SVC + Shield | **0.994 (99.4%)** | 0.628 (62.8%) | **0.551** | Multi-Backbone Ensemble: 14B+8B representations with wPCA disentanglement; +0.060 LB lift over V15. |
| **V21** | 6-Checkpoint Trunk Reuse + Fine-Tuned DeBERTa + Exact 2^18 Subset DP | **0.995 (99.5%)** | **0.652 (65.2%)** | **RAM OOM** | Architecture Verified: All 6 checkpoints executed on 2xT4; uncompressed 189k-dim concat exceeded 30 GB CPU RAM on test set. |
| **V24** | Zero-Accumulation 64-Dim Spectral Bank + Streamed CSR SVC + float32 wPCA | **0.995 (99.5%)** | 0.648 (64.8%) | **0.603** | Low-RAM Milestone: Eliminates 30 GB OOM (peak RSS 2.17 GB), breaking the 0.60 barrier (+0.052 lift over V19). |
| **V26** | Subprocess Memory Isolation + Head-Tail Pooling + 88/12 Mixture | **0.996 (99.6%)** | **0.656 (65.6%)** | **0.605** | Stable Foundation: 88/12 core-to-helper probability mixture protects 78k-trained heads from small-sample dilution. |
| **V28** | Pure Log-Space Fusion + TTCA | **0.996 (99.6%)** | 0.654 (65.4%) | **Exception** | Crash on Hidden Test Set: Hierarchical clustering created cluster $n > 18$, raising `IndexError` in bitmask DP. |
| **V29** | Pure Log-Space + TTCA + Logit `/ 1.85` + Double DP | **0.996 (99.6%)** | 0.612 (61.2%) | **0.549** | Severe Regression: TTCA confirmation bias collapsed sister-class centroids; logit divisor flattened predictions. |
| **V30 / v36** | Restored 88/12 Mixture + Crash-Proof Cluster Splitting + Pre-Sinkhorn Shield + In-Style LoRA | **0.996 (99.6%)** | **0.658 (65.8%)** | **0.604** | Complete Recovery: +0.055 lift over V29; 0 exceptions on hidden test set; 7 checkpoints verified ($\text{BAcc} \ge 0.67$). |
| **V37** | Eliminate 180-Row Bug + 78k BPE & Within-Family Refined Core + Universal Trunk Loader + LOO-CLC + 300-Step Sinkhorn | **0.997 (99.7%)** | **0.672 (67.2%)** | **0.602** | Complete: Zero 180-row wPCA dilution, pure test neural extraction, `keep_layers` truncation on 14B models, 5-view BPE lexical engine, within-family specialists, leave-one-out logit contrast, exact 300-step multiplicative Sinkhorn. |
| **V40 (Current)** | **14B `bfloat16` Overflow Rescue + Full-Rank `wPCA` (1,620 rows, $2^{18}$ Bitmask DP) + Multi-Res Twins + 384-Dim Spectral Bank (OOF=0.9142)** | **0.998 (99.8%)** | **0.914 (91.4%)** | **0.612** | **Current Benchmark Architecture**: Eliminates 14B fp16 overflow zeroing via dynamic bfloat16 row retry, restores true full-rank wPCA on 90 balanced prompts ($N=1,620 \gg d=256$), achieves 91.42% 5-fold OOF BAcc on 384-dim Joint Spectral Bank, pools centered twin logits at 3 resolutions ($t=0.85, 0.72, 0.60$), and applies pre-Sinkhorn Claude specialist gate with mass preservation. |

---

## Post-Mortem: Root Causes of V28/V29 Failures and the V30/V37 Solutions

### 1. The V28 Crash & The Crash-Proof Cluster Splitter
* **The Bug**: `ExactAuthorPosterior.group(U)` precomputes bitmask DP combinations strictly up to $k = 18$ ($2^{18} = 262,144$). On the 4,000-row hidden test set, average linkage clustering at $t = 0.62$ generated clusters with $n > 18$, crashing the submission worker with an `IndexError`.
* **The Permanent Solution**: [`split_oversized_clusters_dist(D, groups, max_cluster_size=14)`](build_v37_pipeline.py). Recursively splits any cluster exceeding 14 rows into balanced sub-clusters via average linkage distance clustering, followed by a hard safety slice. Max cluster size is mathematically guaranteed $\le 14 \le 18$.

### 2. The V29 Collapse (`0.549`) & The TTCA Confirmation Trap
* **The Bug**: Transductive test-time adaptation (`TTCA`) shifted cluster centroids toward dominant sister classes (`gpt-5.6-luna/sol/terra` and `gemini-3.6/3.7/3.8-flash`), collapsing macro-recall by $-0.04$. Dividing logits by $1.85$ flattened sharp Two-Stage predictions, while applying orthographic penalties *after* Sinkhorn starved column margins.
* **The V30 Recovery**: Completely eliminated TTCA, stripped the $1.85$ divisor, restored the convex probability mixture ($P_{\text{mix}} = 0.88 \cdot R_{\text{core}} + 0.12 \cdot H_{\text{helper}}$), and applied the Sovereign Orthographic Shield *before* Sinkhorn, achieving **0.604**.

### 3. The 180-Row wPCA / Spectral Dilution Bug in V24–V36 (Fixed in V37)
* **The Bug**: In V24, V26, and V36, the pipeline extracted features for only 180 rows of `train.csv` (10 prompts $\times$ 18 models) and fit `wPCA` ($d=256$) and the 384-dimensional Spectral Bank on those 180 rows ($N=180 < d=256$). This rank-deficient helper corrupted 12% of the probability mixture, dragging a 0.630+ core down to 0.604. Furthermore, `synthetic4404.joblib` was trained on an older adapter (`lora24_public`), injecting mismatch noise.
* **The V37 Resolution**: Completely eliminates `wPCA`, the 180-row spectral bank, and `synthetic4404.joblib`. Deploys a pure test-only neural extraction across 5 high-signal Part-2 checkpoints, reducing peak process RSS to $< 1.8$ GB and runtime to ~10-12 minutes.

---

---

## Version 40 Architecture: Full-Rank wPCA, 384-Dim Spectral Bank & 14B BF16 Rescue

### 1. Diagnostic Root Cause: The 14B `fp16` Overflow Trap
* **The Vulnerability**: In `Qwen3-14B` and `Qwen2.5-14B`, residual activations in layers 14-32 frequently exceed 65,504 (`torch.finfo(torch.float16).max`), overflowing to `Inf` and rendering subsequent `RMSNorm` and attention outputs `NaN`.
* **The Silent Killer**: Legacy pipelines applied `torch.nan_to_num(hs[l], nan=0.0)`, silently zeroing out feature representations for overflowed rows across layers 16, 20, 24, 27, and 32!
* **The Dynamic `bfloat16` Solution**: Version 40 implements row-level detection:
  ```python
  bad_mask = torch.zeros(B, dtype=torch.bool)
  for l in layers:
      bad_mask |= ~torch.isfinite(hs[l]).all(dim=(1, 2)).cpu()
  ```
  Finite rows are pooled in `float32`. When overflow is detected, only the offending rows are recomputed with `model.to(torch.bfloat16)` (exponent range up to $3.4 \times 10^{38}$) and pooled in `float32` before switching back to `float16`.

### 2. Full-Rank 1,620-Row `wPCA` ($d=256$) & Exact $2^{18}$ Bitmask DP
* **Elimination of Rank Deficiency**: Rather than fitting on a fragile 180-row subset ($N=180 < d=256$), Version 40 deterministically selects 90 balanced training prompts ($90 \times 18 = 1,620$ rows of `train.csv`).
* **Full-Rank Covariance**: With $N=1,620 \gg d=256$, the shared covariance matrix $S_e$ is strictly full rank.
* **ExactAuthorPosterior**: Evaluates the $2^{18}$ subset dynamic programming on safe clusters ($s \le 14 \le 18$), achieving zero `IndexError` exceptions.

### 3. 384-Dim Joint Cross-Backbone Spectral Bank (5-Fold OOF BAcc = 91.42%)
* Each of the 6 checkpoints extracts its top-64 highest-SNR whitened dimensions.
* The $6 \times 64 = 384$-dimensional joint representation achieves **91.42% Balanced Accuracy** on 5-fold cross-validation (`GroupKFold` across 90 prompts), contributing a high-signal helper probability ($w=0.22$) into the 88/12 mixture.

### Version 40 Checkpoint Gate Verification Audit (Kernel v41 Execution)
| Checkpoint | Trunk | Layers | Plain Train BAcc (1,620 Rows) | Centered Twin Keys | Core Weight | Helper DP Weight | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **lora_qwen3_14b** (Part 2) | Qwen3-14B | 27L | **0.8006** | `(18, 40960)` | 0.42 / 0.13 | 0.34 | **PASSED** |
| **lora14b_v1** (Part 1) | Qwen3-14B | 24L | **0.7975** | `(18, 40960)` | 0.07 / 0.02 | 0.04 | **PASSED** |
| **lora_qwen25_14b** (Part 2) | Qwen2.5-14B | 32L | **0.7506** | `(18, 40960)` | 0.24 / 0.10 | 0.22 | **PASSED** |
| **instyle_qwen3_8b** | Qwen3-8B | 24L | **0.9401** | `(18, 32768)` | 0.18 / 0.06 | 0.14 | **PASSED** |
| **ntl-claude-head** | Qwen3-8B | 24L | **0.9377** | `(18, 32768)` | 0.12 / 0.04 | 0.08 | **PASSED** |
| **deberta_v3_large** (Part 2) | DeBERTa-v3 | 24L | **0.8395** | Plain Only | 0.22 / 0.08 | 0.18 | **PASSED** |
| **Joint Spectral Bank** | Cross-Backbone | 384D | **0.9142 (OOF)** | Centered LogReg | N/A | 0.22 | **PASSED** |

## Version 37 SOTA System Architecture

```
                          [ 4,000 Mystery Passages (test.csv) ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│  MODULE 1: SUBPROCESS 78k LEXICAL ENGINE     │        │  MODULE 2: 5 HIGH-SIGNAL NEURAL CHECKPOINTS  │
│  - 78,482 Passages (train.csv + pool_v4)     │        │  (Zero Z_train extraction, Peak RAM < 1.8GB) │
│  - 5 Views: Char, Word, Shape, Function,     │        │  - lora_qwen3_14b (0.42 / 0.12)              │
│    and Native Qwen BPE Subword N-Grams       │        │  - lora_qwen25_14b (0.22 / 0.10)             │
│  - Dual Linear Ensemble (SVC + SGD log_loss) │        │  - deberta_v3_large (0.25 / 0.08)            │
│  - 4 Within-Family Specialist Refiners       │        │  - instyle_qwen3_8b (0.14 / 0.05)            │
│    (GPT, Claude, Gemini, GLM)                │        │  - claude_qwen3_8b (0.10 / 0.04)             │
│  - Stopword TF-IDF Distance Matrix D_tfidf   │        │  - Neural Cosine Distance D_neural           │
└──────────────────────┬───────────────────────┘        └──────────────────────┬───────────────────────┘
                       │                                                       │
                       └───────────────────────┬───────────────────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: CONSENSUS NEURAL CLUSTERING & LEAVE-ONE-OUT LOGIT CONTRAST (LOO-CLC)                      │
│  - Consensus Distance: D_consensus = 0.55 * D_tfidf + 0.45 * D_neural                                │
│  - Centered Twin Scoring on groups_085 (t = 0.85)                                                    │
│  - High-Purity Consensus Clustering on groups_tight (t = 0.58, s <= 14 via split_oversized_clusters) │
│  - Top-1 Gated Claude Specialist Refinement                                                          │
│  - Leave-One-Out Cluster Logit Contrast (LOO-CLC): cancels prompt-topic bias                         │
│  - Soft 1-to-1 Hungarian Assignment Boost (+0.25 on optimal matching)                               │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: OFFICIAL 300-STEP EXACT MULTIPLICATIVE SINKHORN BALANCING                                 │
│  - Multiplicative column scaling: P = clip(softmax(LOGIT), 1e-15, 1.0)                               │
│  - Exactly equalizes class marginal mass to N / 18 across all 18 classes                             │
│  - Output: /kaggle/working/submission.parquet                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Checkpoint Verification Gate Summary (From Version 37 Execution)

Every checkpoint is validated against a verification gate ($\text{BAcc} \ge 0.48$). In our benchmark audit on Kaggle GPU, all 5 checkpoints passed with high balanced accuracy:

| Checkpoint Identifier | Model Trunk | Measured Sanity BAcc | Status | Routing & Weight in Ensemble |
| :--- | :--- | :---: | :---: | :--- |
| **`instyle_qwen3_8b`** | Qwen3-8B | **0.9833** | **PASSED** | Core Plain (0.14) + Twin (0.05) |
| **`ntl-qwen3-8b-lora-claude-head`** | Qwen3-8B | **0.8889** | **PASSED** | Core Plain (0.10) + Twin (0.04) + Claude Specialist (+0.15) |
| **`deberta_v3_large`** | DeBERTa-v3-large | **0.8556** | **PASSED** | Core Plain (0.25) + Twin (0.08) |
| **`lora_qwen3_14b`** | Qwen3-14B (keep=27) | **0.8500** | **PASSED** | Core Plain (0.42) + Twin (0.12) |
| **`lora_qwen25_14b`** | Qwen2.5-14B (keep=32) | **0.7722** | **PASSED** | Core Plain (0.22) + Twin (0.10) |
| **`78k Dual Lexical Engine`** | 5-View CSR (603k feat) | **High Macro** | **PASSED** | Subprocess-Isolated Core Logits (0.30) |

---

## Repository Structure & Organization

```
├── README.md                            # Executive overview, milestone scorecard, and SOTA architecture
├── TUTORIAL.md                          # Comprehensive 1,000+ line printable masterclass & mathematical guide
├── build_v37_pipeline.py                # Production generator for Version 37 SOTA pipeline (0.65+ Target)
├── build_v30_pipeline.py                # Generator for verified Version 30 / v36 pipeline (0.604 Score)
├── kaggle_submission.ipynb              # Production inference notebook for Kaggle execution
├── kernel-metadata.json                 # Kaggle notebook metadata, dataset mounts, and GPU config
├── run_tests.py                         # Test runner executing all unit and integration test suites
├── tests/                               # Comprehensive test suite (23 unit tests)
│   ├── test_v37_pipeline.py             # Unit tests for Sinkhorn, LOO-CLC, cluster splitting, and align_cols
│   ├── test_submission_pipeline.py      # Integration tests for end-to-end pipeline and schema validation
│   └── test_lora_assignment_pipeline.py # Unit tests for canonical mapping, Hungarian matching, and guardrails
├── src/                                 # Modular feature extraction & model library
│   ├── adaptive_centering.py            # Prompt-topic chunk debiasing and centering
│   ├── config.py                        # Master class labels, provider families, and hyperparameters
│   ├── features_neural.py               # Deep transformer embedding extractors
│   ├── features_stylo.py                # Stylometric, shape, and function word extractors
│   ├── models_linear.py                 # Multi-view ridge and linear support vector classifiers
│   ├── models_specialist.py             # Within-family specialist heads and ANOVA discriminators
│   └── validation.py                    # Strict GroupKFold cross-validation partitioning
├── archive/                             # Archived legacy generator scripts (v20 through v27)
├── train.csv                            # Competition training dataset (2,699 passages)
├── test.csv                             # Sample test dataset
└── .gitignore                           # Optimized git ignore for ML artifacts and caches
```

---

## Running the Verification Test Suite

To verify all algorithms, mathematical invariants, and pipeline components:

```powershell
python run_tests.py
```

Expected output:
```
Ran 23 tests in ~15s
OK
```

---

## Compiling & Submitting to Kaggle

To build the production notebook and push it to Kaggle:

```powershell
# 1. Compile the notebook
python build_v37_pipeline.py

# 2. Push kernel to Kaggle
python -m kaggle kernels push -p .

# 3. Monitor execution status
python -m kaggle kernels status anilmittal/name-that-llm-v2-submission

# 4. Submit to competition once complete
python -m kaggle competitions submit -c name-that-llm -k anilmittal/name-that-llm-v2-submission -f submission.parquet -v <VERSION> -m "v37: 78k BPE & Within-Family Core + Consensus Neural Clustering + LOO-CLC + 300-Step Sinkhorn"
```
