# Name That LLM: High-Precision Authorship Attribution Pipeline

[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.622%20(SOTA)-brightgreen)](https://www.kaggle.com/competitions/name-that-llm)
[![Architecture Version](https://img.shields.io/badge/Production%20Stack-Modular%20PyPA%20Package-blue)](#)
[![Models](https://img.shields.io/badge/Backbones-Qwen3--14B%20%7C%20Qwen2.5--14B%20%7C%20Qwen3--8B%20%7C%20DeBERTa-orange)](#)
[![Tests](https://img.shields.io/badge/Tests-27%20Passed-brightgreen)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)

This repository contains the end-to-end, competition-grade machine learning solution for the **Kaggle "Name That LLM"** competition. The objective is to identify which of 18 specific Large Language Models authored an unattributed response passage.

---

## High-Level Evolutionary Milestones

The pipeline advanced from a simple bag-of-words baseline to the state-of-the-art **0.622 Public LB** through key architectural breakthroughs:

| Milestone | Architecture / Pipeline Components | Validation CV | Public LB | Key Architectural Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **V1: Baseline** | Flat Word N-Gram TF-IDF Baseline | 0.335 | **0.321** | Extreme prompt memorization; bag-of-words failed on unseen test prompts. |
| **V15: Attention Sink Fix** | Verified Qwen3-8B LoRA + Token-0 Mask (`m[:, 0] = 0`) | 0.542 | **0.491** | Zeroing causal attention sinks restored extraction fidelity and closed the CV-LB gap. |
| **V19: Multi-Backbone** | Two-Stage 14B/8B LoRAs + wPCA Gaussian + 4-View SVC | 0.628 | **0.551** | Multi-backbone stylometric disentanglement provided +0.060 lift over single models. |
| **V26: Low-RAM Core** | Subprocess Memory Isolation + 78k Lexical + 88/12 Mixture | 0.656 | **0.605** | Isolated memory footprint; convex mixture protected 78k heads from small-sample dilution. |
| **V42: Two-Tier Stack** | 85% Part-Two Main Stack + 15% Prior-Scaled Complementary Stack | 0.665 | **0.613** | Strict two-tier separation isolated the core Part-Two baseline while harvesting orthogonal signal. |
| **Current Production (SOTA)** | Modular PyPA Stack + Reversible Unmerging + $2^{18}$ Subset DP + Unified Orthographic Shield | 0.640 | **0.622** | **Production SOTA**: Bit-exact weight isolation, exact forward-backward subset DP, full-rank wPCA, and unified discrete/continuous handwriting shield. |

---

## Production System Architecture

The production pipeline is organized into 4 decoupled, memory-isolated modules executing sequentially within Kaggle's 30 GB RAM and 2-hour GPU limits:

```
                          [ 4,000 Mystery Passages (test.csv) ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│  MODULE 1: SUBPROCESS 78k LEXICAL ENGINE     │        │  MODULE 2: 4-TRUNK NEURAL CHECKPOINTS        │
│  - 78,482 Passages (train.csv + pool_v4)     │        │  (Zero-RAM In-Place Unmerge & Snapshot Iso)  │
│  - 4-View Main LinearSVC (C=0.50)            │        │  - Trunk 1: Qwen3-14B (Part 2 + Part 1)      │
│  - 5-View Complementary Stack (BPE + SGD)    │        │  - Trunk 2: Qwen2.5-14B (Part 2)             │
│  - Dedicated 26-Feature Handwriting Model    │        │  - Trunk 3: Qwen3-8B (In-Style + Claude Head)│
│  - Stopword TF-IDF Distance Matrix D_tfidf   │        │  - Trunk 4: DeBERTa-v3-large                 │
│  - Fast N-Gram CSR Matrices (100% Sparse)    │        │  - Dynamic bfloat16 Overflow Rescue          │
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
│  MODULE 4: TWO-TIER CONVEX FUSION, SPECIALIST GATE & UNIFIED POSTPROCESSING                          │
│  - Two-Tier Convex Fusion: P_mix = 0.85 * R_main + 0.15 * P_comp                                     │
│  - Vectorized Claude Specialist Gate with overall probability simplex mass preservation             │
│  - Unified Handwriting & Orthographic Shield (discrete Bayesian rules + continuous density model)    │
│  - Official 300-step multiplicative Sinkhorn mass equalization (column sums = N / 18)                │
│  - Output: /kaggle/working/submission.parquet                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

1. **Module 1: Subprocess Dual-Output Lexical Engine (`src/namethatllm/lexical_engine.py`)**:
   - Operates in memory-isolated subprocess space on 78,482 passages (`train.csv` + `pool_v4.parquet`).
   - **Output 1 (Main Stack)**: Pure 4-view `LinearSVC(C=0.50)` (char 3-5, word 1-3, shape 2-6, function 1-3).
   - **Output 2 (Complementary Stack)**: 5-view BPE subwords + SGD classifier + prompt-centered style twin on cluster graph $D_{\text{tfidf}}$.
   - **Dedicated Handwriting Classifier**: Standardizes 26 continuous rhythm, punctuation, and contraction features, fitting plain `LogisticRegression(C=0.25)` and prompt-centered `LogisticRegression(C=0.15)` to produce calibrated handwriting log-probabilities without dense contamination of the n-gram matrices.

2. **Module 2: Multi-Trunk Neural Extractors (`src/namethatllm/neural_extractor.py`)**:
   - Reuses 4 base model trunks across 6 checkpoint adapters:
     - `Trunk 1`: Qwen3-14B (Checkpoint 1A Part-2 + Checkpoint 1B Part-1)
     - `Trunk 2`: Qwen2.5-14B (Checkpoint 2 Part-2)
     - `Trunk 3`: Qwen3-8B (Checkpoint 3A In-Style + Checkpoint 3B Claude Head)
     - `Trunk 4`: DeBERTa-v3-large sequence classification encoder
   - **Dynamic `bfloat16` Overflow Rescue**: Detects residual activation blowup ($> 65,504$) in deep layers and re-computes only affected rows in `bfloat16`, preventing NaN propagation.
   - **Reversible In-Place Unmerging & Snapshot Isolation**: Merges LoRA adapters directly on GPU memory, restoring exact base weights via snapshot weight caches to prevent cumulative float16 arithmetic drift.

3. **Module 3: Disentangled wPCA, $2^{18}$ Bitmask DP & Spectral Bank (`src/namethatllm/wpca_gaussian.py`, `spectral_bank.py`)**:
   - Selects 90 balanced prompts ($1,620$ rows) to guarantee full-rank within-prompt covariance ($N=1,620 \gg d=256$).
   - `ExactAuthorPosterior`: Evaluates exact 18-step forward-backward bitmask dynamic programming over all $2^{18} = 262,144$ author configurations on safe clusters ($s \le 14 \le 18$), enforcing the hard 1-author-per-cluster constraint.
   - 384-dimensional Joint Spectral Bank extracts top-64 principal components from unsupervised prompt-centered features with out-of-sample prior scaling (zero label leakage).

4. **Module 4: Two-Tier Convex Fusion & Unified Postprocessing (`src/namethatllm/postprocess.py`)**:
   - Convex mixture: $P_{\text{mix}} = 0.85 \cdot R_{\text{main}} + 0.15 \cdot P_{\text{comp}}$.
   - Vectorized Claude Specialist Gate with exact intra-family mass preservation.
   - **Unified Handwriting & Orthographic Shield**: Strips code blocks before evaluating discrete structural indicators (curly quotes, multi em-dashes, straight vs. curly contractions), computing `rule_flags @ W_pen + continuous_hw_weight * log_p_handwriting` in a single vectorized operation.
   - 300-step multiplicative Sinkhorn-Knopp mass balancing equalizing column margins to $N/18$.

---

## Repository Structure (PyPA `src` Layout)

```
NameThatLLM/
├── pyproject.toml                       # Package metadata, build configuration, and pytest settings
├── README.md                            # Executive overview, milestone scorecard, and SOTA architecture
├── TUTORIAL.md                          # Comprehensive printable masterclass & mathematical guide
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
│   ├── lexical_engine.py                # 78k 4-view Main SVC + 5-view BPE Complementary + Handwriting model
│   ├── neural_extractor.py              # Universal trunk loader, reversible in-place unmerging, LayerNorm heads
│   ├── wpca_gaussian.py                 # Full-rank wPCA, safe cluster splitting (s <= 14), 2^18 Bitmask DP
│   ├── spectral_bank.py                 # Unsupervised 384-dim Joint Spectral Bank (0 label leakage)
│   ├── postprocess.py                   # Two-tier convex blend, Claude gate, unified handwriting shield, Sinkhorn
│   └── pipeline.py                      # Top-level orchestration of Modules 1 through 4
├── tests/                               # Comprehensive unit test suite (pytest)
│   ├── test_compiler_and_integration.py # End-to-end AST validation, compiler checks, mock-neural tests
│   ├── test_lexical.py                  # Tokenizers, shape patterns, stopwords, and dual lexical outputs
│   ├── test_wpca_dp.py                  # Two-way decomposition, Ledoit-Wolf shrinkage, ExactAuthorPosterior
│   ├── test_spectral_bank.py            # Prior scaling, unsupervised V_f32 projection, OOF validation
│   └── test_postprocess.py              # Two-tier blend, Claude gate, unified shield, Sinkhorn balancing
└── local_data/                          # Git-ignored cache, logs, and downloaded weights
```

---

## Installation & Developer Setup

Install the package in editable mode with development dependencies:

```powershell
# 1. Install package in editable mode
pip install -e . --no-deps

# 2. Run the full unit and integration test suite (27 tests)
python -m pytest tests -v
```

---

## Compiling & Submitting to Kaggle

The pipeline maintains a **Single Source of Truth**: all modular logic lives in `src/namethatllm/` and parameters in `configs/production_ensemble.json`. The compiler validates Python AST across every block before assembling the submission kernel.

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
python -m kaggle competitions submit -c name-that-llm -k anilmittal/name-that-llm-v2-submission -f submission.parquet -v 45 -m "v45: Unified Handwriting & Orthographic Rules + Shared Trunk Snapshot Isolation"
```
