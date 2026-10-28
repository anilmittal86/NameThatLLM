# Name That LLM: High-Precision Authorship Attribution Pipeline

[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.622%20(SOTA)-brightgreen)](https://www.kaggle.com/competitions/name-that-llm)
[![Architecture Version](https://img.shields.io/badge/Production%20Version-Version%2044%20(Two--Tier%20Modular%20Stack)-blue)](#)
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
| **V40** | **14B `bfloat16` Overflow Rescue + Full-Rank `wPCA` (1,620 rows, $2^{18}$ Bitmask DP) + Multi-Res Twins + 384-Dim Spectral Bank** | **0.998 (99.8%)** | **0.914 (91.4%)** | **0.612** | SOTA Breakthrough: Eliminates 14B fp16 overflow zeroing via dynamic bfloat16 row retry, restores full-rank wPCA on 1,620 balanced rows, achieves 0.612 Public LB. Diagnostic audit revealed in-sample 8B memorization trap in spectral bank. |
| **V42** | **Pure 85% Part-Two Main Stack + 15% Prior-Scaled Complementary Stack + Unsupervised Spectral Bank (Zero Leakage)** | **0.998 (99.8%)** | **0.665 (66.5%)** | **0.613** | Two-Tier Separation: Eliminates 8B in-sample stacking trap by extracting unsupervised spectral components from $V_{\text{f32}}[:, :64]$ (0 label leakage); isolates 85% mass strictly to official 4 Part-Two members; routes secondary models to 15% complementary stack with Ledoit-Wolf shrinkage floors and prior scaling; lifts LB to 0.613. |
| **V44 (Current)** | **Modular PyPA Architecture + Zero-RAM In-Place Unmerge + Exact $2^{18}$ Forward-Backward Subset DP + LayerNorm Heads** | **0.998 (99.8%)** | **0.640 (64.0%)** | **0.622 (SOTA)** | **Production SOTA**: +0.009 LB lift to 0.622; eliminates 15GB CPU host memory leak via in-place zero-RAM LoRA subtraction (`sign=-1.0`); implements exact 18-step forward-backward bitmask DP over all $2^{18}$ states; restores prompt-centering to wPCA covariance; adds native LayerNorm projection heads; fully vectorized Claude gate with mass preservation; AST-validated single-source compiler; 24 unit/integration tests passing. |

---

## Post-Mortem: Root Causes of Historical Failures and Their Architectural Solutions

### 1. The V28 Crash & The Crash-Proof Cluster Splitter
* **The Bug**: `ExactAuthorPosterior.group(U)` precomputes bitmask DP combinations strictly up to $k = 18$ ($2^{18} = 262,144$). On the 4,000-row hidden test set, average linkage clustering at $t = 0.62$ generated clusters with $n > 18$, crashing the submission worker with an `IndexError`.
* **The Permanent Solution**: [`split_oversized_clusters_dist(D, groups, max_cluster_size=14)`](src/namethatllm/wpca_gaussian.py). Recursively splits any cluster exceeding 14 rows into balanced sub-clusters via average linkage distance clustering, followed by a hard safety slice. Max cluster size is mathematically guaranteed $\le 14 \le 18$.

### 2. The V29 Collapse (`0.549`) & The TTCA Confirmation Trap
* **The Bug**: Transductive test-time adaptation (`TTCA`) shifted cluster centroids toward dominant sister classes (`gpt-5.6-luna/sol/terra` and `gemini-3.6/3.7/3.8-flash`), collapsing macro-recall by $-0.04$. Dividing logits by $1.85$ flattened sharp Two-Stage predictions, while applying orthographic penalties *after* Sinkhorn starved column margins.
* **The V30 Recovery**: Completely eliminated TTCA, stripped the $1.85$ divisor, restored the convex probability mixture ($P_{\text{mix}} = 0.88 \cdot R_{\text{core}} + 0.12 \cdot H_{\text{helper}}$), and applied the Sovereign Orthographic Shield *before* Sinkhorn, achieving **0.604**.

### 3. The 180-Row wPCA / Spectral Dilution Bug in V24–V36 (Fixed in V37)
* **The Bug**: In V24, V26, and V36, the pipeline extracted features for only 180 rows of `train.csv` (10 prompts $\times$ 18 models) and fit `wPCA` ($d=256$) and the 384-dimensional Spectral Bank on those 180 rows ($N=180 < d=256$). This rank-deficient helper corrupted 12% of the probability mixture, dragging a 0.630+ core down to 0.604. Furthermore, `synthetic4404.joblib` was trained on an older adapter (`lora24_public`), injecting mismatch noise.
* **The V37 Resolution**: Completely eliminates `wPCA`, the 180-row spectral bank, and `synthetic4404.joblib`. Deploys a pure test-only neural extraction across 5 high-signal Part-2 checkpoints, reducing peak process RSS to $< 1.8$ GB and runtime to ~10-12 minutes.

---

## Version 44 Architecture: Zero-RAM Reversible Unmerging & Modular Production Stack

### 1. High-Level Architectural Foundations
Version 44 represents the production consolidation of the forensic authorship engine into a clean, modular PyPA `src` layout with zero memory duplication, mathematically rigorous subset dynamic programming, and honest out-of-fold generalization:

1. **Zero-RAM Reversible In-Place LoRA Arithmetic**:
   - *The Bottleneck*: In earlier iterations (e.g. V43), snapshotting trunk weights to CPU memory duplicated 15 GB of unquantized float32 weights in host RAM, exceeding Kaggle's 30 GB cgroup memory limit (`DeadKernelError: exit code 137`).
   - *The Architecture*: Base trunks are loaded directly to GPU memory once (`low_cpu_mem_usage=True`). LoRA adapter weights are merged in-place ($W_{\text{eff}} = W_0 + \frac{\alpha}{r} BA$) for inference and then restored to exact original base weights using in-place subtraction ($W_0 = W_{\text{eff}} - \frac{\alpha}{r} BA$) via `merge_lora(model, ckpt_dir, sign=-1.0)`. Explicit garbage collection and CUDA cache flushing eliminate host/device memory creep across all 4 trunks (`Qwen3-14B`, `Qwen2.5-14B`, `Qwen3-8B`, `DeBERTa-v3`).

2. **Exact $2^{18}$ Forward-Backward Subset Dynamic Programming**:
   - Rather than mean-collapsing cluster features or relying on heuristic approximations, `ExactAuthorPosterior` evaluates exact 18-step forward-backward bitmask dynamic programming over all $2^{18} = 262,144$ author subsets.
   - Enforces the hard author uniqueness constraint ($c_i \neq c_j$) on cluster sizes $2 \le s \le 14 \le 18$ while integrating out shared prompt covariance.

3. **Prompt-Centered Within-Class Covariance (`wPCA`)**:
   - In `fit_shared_gaussian_lowram()`, explicit prompt-mean subtraction is enforced prior to SVD on the 1,620-row balanced training matrix. This projects features into a subspace isolating stylometric authorship variance from prompt topic semantics.

4. **Multi-Trunk LayerNorm & Dimension Parity**:
   - Native `LayerNorm` projection heads are applied to pooled transformer hidden states prior to computing dot-product logits, ensuring exact geometric alignment with pre-trained linear heads.

5. **Two-Tier Convex Probability Mixture ($85 / 15$)**:
   - **Main Stack ($85\%$)**: The 4 official Part-Two models (`LinearSVC(C=0.5)`, `lora_qwen3_14b`, `lora_qwen25_14b`, `deberta_v3_large`) guaranteeing a robust baseline.
   - **Complementary Stack ($15\%$)**: Secondary LoRAs, 5-view BPE lexical models, regularized wPCA DP posteriors, and the leak-free 384-dimensional Joint Spectral Bank.
   - $$P_{\text{mix}} = 0.85 \cdot R_{\text{main}} + 0.15 \cdot P_{\text{comp}}$$

6. **Vectorized Claude Specialist Gate & 300-Step Sinkhorn Balancing**:
   - Fully vectorized probability reassignment within the Claude sibling sub-simplex preserves overall class mass while sharpening intra-family resolution.
   - Official 300-step multiplicative Sinkhorn-Knopp algorithm equalizes marginal probabilities to $N/18$ across all 18 classes.

---

## Version 42 Architecture: Pure 85% Part-Two Main Stack + 15% Prior-Scaled Complementary Stack

### 1. Root-Cause Diagnosis of V40 / Submission 41 (0.612 Public LB)
Version 41 broke the 0.61 barrier to achieve **0.612 Public LB** thanks to the 14B `bfloat16` dynamic row rescue. However, checkpoint diagnostics revealed three critical flaws preventing higher scores:
1. **The 8B In-Sample Stacking Trap & Label Leakage**: `instyle_qwen3_8b` (0.9401 train BAcc) and `ntl-claude-head` (0.9377 train BAcc) memorized `train.csv`. Fitting `wPCA` and the 384-dim Logistic Regression on `train_sub` caused the Spectral Bank to place 80%+ weight on the overfit 8B models while down-weighting the superior 14B models (0.8006 / 0.7506). Furthermore, supervised label leakage inflated OOF BAcc to a misleading 0.9142.
2. **Dilution of the 0.620 Part Two Core**: Mixing weaker 8B models and Part-1 checkpoints directly into `LOGIT_CORE` at weight 0.49 dragged down the official Part-Two baseline.
3. **LinearSVC Sample Weight Perturbation**: A 2.0x weight multiplier on `train.csv` shifted hyperplane margins away from the canonical 78k Part-Two classifier.

### 2. The Version 42 Architectural Solutions
1. **Unsupervised, Leak-Free 384-Dim Joint Spectral Bank**:
   - Extracts top-64 principal components from **unsupervised prompt-centered features** $V_{\text{f32}}[:, :64]$ (using `prompt_id` only, zero label leakage).
   - Applies **Out-of-Sample Prior Scaling** to weight blocks by actual out-of-sample generalization (`q3_14b`: 1.00, `q25_14b`: 0.85, `deberta`: 0.80, `part1_14b`: 0.50, `instyle`: 0.45, `claude_head`: 0.30).
   - 5-fold `GroupKFold` OOF validation now reflects honest generalization (0.63–0.68).
2. **Pure Two-Tier Stack Separation**:
   - **Main Stack (`LOGIT_MAIN`, 85% Mass)**: Strictly the 4 official Part Two members:
     * Canonical 4-View `LinearSVC(C=0.5)` with uniform 1.0 sample weights ($w=1.00$, twin=$0.00$)
     * `lora_qwen3_14b` (Part 2, 27L): Plain ($w=0.40$) + Twin ($w=0.10$, `groups_085` only)
     * `lora_qwen25_14b` (Part 2, 32L): Plain ($w=0.20$) + Twin ($w=0.10$, `groups_085` only)
     * `deberta_v3_large` (Part 2, 24L): Plain ($w=0.25$) (twin=$0.00$)
   - **Complementary Stack (`COMP_LIST`, 15% Mass)**:
     * Secondary LoRAs & multi-resolution twins ($w=0.35$)
     * Complementary lexical (BPE subwords + SGD + LinearSVC + style twin, $w=0.20$)
     * Regularized `wPCA` + `ExactAuthorPosterior` ($2^{18}$ Bitmask DP with Ledoit-Wolf shrinkage floors, $w=0.30$)
     * Leak-free prior-scaled 384-dim Joint Spectral Bank ($w=0.15$)
3. **Convex Probability Fusion**:
   $$P_{\text{mix}} = 0.85 \cdot R_{\text{main}} + 0.15 \cdot P_{\text{comp}}$$
   Preserves the guaranteed 0.620+ Part-Two baseline while harvesting incremental orthogonal signal.
4. **Regularized ExactAuthorPosterior**:
   - `se_floor = 0.10` for Part Two checkpoints.
   - `se_floor = 0.35` for overfit 8B and Part 1 checkpoints.
5. **Calibrated Postprocessing**:
   - Mass-preserving Claude Specialist Gate: $P_{\text{claude}} = 0.85 \cdot P_{\text{prior}} + 0.15 \cdot (\text{mass} \cdot P_{\text{specialist}})$.
   - Gentle curly-quote guard: $\exp(-1.0)$ on Claude and GLM classes followed by simplex normalization.
   - Official 300-step exact multiplicative Sinkhorn mass balancing.

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

---

## SOTA System Architecture Flow

```
                          [ 4,000 Mystery Passages (test.csv) ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│  MODULE 1: SUBPROCESS 78k LEXICAL ENGINE     │        │  MODULE 2: 4-TRUNK NEURAL CHECKPOINTS        │
│  - 78,482 Passages (train.csv + pool_v4)     │        │  (Zero-RAM In-Place Unmerge, Peak RAM Safe)  │
│  - 4-View Main LinearSVC (C=0.5)             │        │  - Trunk 1: Qwen3-14B (Part 2 + Part 1)      │
│  - 5-View Complementary Ensemble (BPE + SGD) │        │  - Trunk 2: Qwen2.5-14B (Part 2)             │
│  - Stopword TF-IDF Distance Matrix D_tfidf   │        │  - Trunk 3: Qwen3-8B (In-Style + Claude Head)│
│  - Prompt Clusters: groups_085 (t=0.85)      │        │  - Trunk 4: DeBERTa-v3-large                 │
└──────────────────────┬───────────────────────┘        └──────────────────────┬───────────────────────┘
                       │                                                       │
                       └───────────────────────┬───────────────────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: WITHIN-PROMPT wPCA & EXACT 2^18 BITMASK DP + 384-DIM JOINT SPECTRAL BANK                  │
│  - 1,620-Row Balanced Training Set (90 prompts x 18 classes, full-rank covariance)                   │
│  - Prompt-Centered Within-Class Covariance S_e with Ledoit-Wolf Shrinkage Floors                     │
│  - ExactAuthorPosterior: Exact forward-backward dynamic programming over 2^18 author subsets         │
│  - Unsupervised 384-Dim Joint Spectral Bank: Top-64 components per checkpoint with Prior Scaling     │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: TWO-TIER CONVEX FUSION, VECTORIZED CLAUDE GATE & 300-STEP SINKHORN BALANCING              │
│  - Two-Tier Convex Fusion: P_mix = 0.85 * R_main + 0.15 * P_comp                                     │
│  - Vectorized Claude Specialist Gate with overall probability simplex mass preservation             │
│  - Sovereign Orthographic Bayesian Shield (curly-quote penalty on Claude/GLM)                         │
│  - Official 300-step multiplicative Sinkhorn mass equalization (column sums = N / 18)                │
│  - Output: /kaggle/working/submission.parquet                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure & Organization (PyPA `src` Layout)

```
NameThatLLM/
├── pyproject.toml                       # PEP 621 package metadata, build config, and pytest settings
├── README.md                            # Executive overview, milestone scorecard, and SOTA architecture
├── TUTORIAL.md                          # Comprehensive 1,000+ line printable masterclass & mathematical guide
├── configs/                             # Declarative experiment and ensemble configurations
│   └── production_ensemble.json         # Production weights, regularizers, thresholds, and prior scales
├── kernels/                             # Kaggle kernel build targets and metadata
│   ├── submission/                      # Official competition submission kernel
│   │   ├── kernel-metadata.json         # Kaggle GPU/dataset mount configuration
│   │   └── kaggle_submission.ipynb      # Compiled production inference notebook
│   └── training/                        # LoRA fine-tuning training kernel
│       ├── kernel-metadata.json
│       └── lora_training.ipynb
├── scripts/                             # Automation tooling and compilers
│   ├── compile_notebook.py              # Single-source-of-truth notebook compiler from src/ + configs/
│   └── push_submission.py               # Automated test, compile, and Kaggle push CLI
├── src/namethatllm/                     # Modular production Python package
│   ├── __init__.py                      # Package version and public exports
│   ├── config.py                        # Canonical 18 classes, provider families, simplex helpers, paths
│   ├── lexical_engine.py                # 78k 4-view Main SVC + 5-view BPE Complementary ensemble
│   ├── neural_extractor.py              # Universal trunk loader, reversible in-place unmerging, LayerNorm heads
│   ├── wpca_gaussian.py                 # Full-rank wPCA, safe cluster splitting (s <= 14), 2^18 Bitmask DP
│   ├── spectral_bank.py                 # Unsupervised 384-dim Joint Spectral Bank (0 label leakage)
│   ├── postprocess.py                   # Two-tier convex blend, vectorized Claude gate, 300-step Sinkhorn
│   └── pipeline.py                      # Top-level orchestration of Modules 1 through 4
├── tests/                               # Comprehensive unit test suite (pytest)
│   ├── __init__.py
│   ├── test_compiler_and_integration.py # End-to-end AST validation, compiler checks, mock-neural tests
│   ├── test_lexical.py                  # Tokenizers, shape patterns, stopwords, and dual lexical outputs
│   ├── test_wpca_dp.py                  # Two-way decomposition, Ledoit-Wolf shrinkage, ExactAuthorPosterior
│   ├── test_spectral_bank.py            # Prior scaling, unsupervised V_f32 projection, OOF validation
│   └── test_postprocess.py              # Two-tier blend, Claude gate, curly quote shield, Sinkhorn balancing
└── local_data/                          # Git-ignored cache, logs, and downloaded weights
    ├── cache/
    ├── checkpoints/
    └── logs/
```

---

## Installation & Developer Setup

Install the package in editable mode with development dependencies:

```powershell
# 1. Install package in editable mode
pip install -e . --no-deps

# 2. Run the full unit and integration test suite (24 tests)
python -m pytest tests -v
```

---

## Compiling & Submitting to Kaggle

The pipeline uses a **Single Source of Truth** architecture: all model code and logic live in `src/namethatllm/` and configurations in `configs/`. The compiler validates syntax on every cell using Python AST.

```powershell
# Option A: One-line test, compile, and push
python scripts/push_submission.py

# Option B: Manual step-by-step
# 1. Compile the production notebook from src/namethatllm/ + configs/production_ensemble.json
python scripts/compile_notebook.py --config configs/production_ensemble.json

# 2. Push the submission kernel to Kaggle
python -m kaggle kernels push -p kernels/submission

# 3. Monitor execution status
python -m kaggle kernels status anilmittal/name-that-llm-v2-submission

# 4. Submit to competition once complete
python -m kaggle competitions submit -c name-that-llm -k anilmittal/name-that-llm-v2-submission -f submission.parquet -v 44 -m "v44: Fix 8 bugs + Zero-RAM In-Place Unmerge + Multi-Res Twins + 384D Spectral Bank"
```
