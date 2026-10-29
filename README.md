# Name That LLM: High-Precision Authorship Attribution Engine

[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.622%20(SOTA)-brightgreen)](https://www.kaggle.com/competitions/name-that-llm)
[![Codebase Architecture](https://img.shields.io/badge/Architecture-Modular%20PyPA%20Package-blue)](#)
[![Models](https://img.shields.io/badge/Backbones-Qwen3--14B%20%7C%20Qwen2.5--14B%20%7C%20Qwen3--8B%20%7C%20DeBERTa-orange)](#)
[![Tests](https://img.shields.io/badge/Unit%20Tests-27%20Passed-brightgreen)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)

A competition-grade, memory-isolated machine learning solution for the **Kaggle "Name That LLM"** competition. The objective is to identify which of 18 specific Large Language Models authored an unattributed response passage.

---

## 1. System Overview

The production system operates as a **two-tier, memory-isolated ensemble** combining 4 foundation model trunks, 6 fine-tuned LoRA/encoder checkpoints, dual lexical engines, within-prompt stylometric covariance analysis, exact subset dynamic programming, and unified Bayesian orthographic postprocessing.

```
                          [ 4,000 Mystery Passages (test.csv) ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│  MODULE 1: SUBPROCESS 78k LEXICAL ENGINE     │        │  MODULE 2: 4-TRUNK NEURAL EXTRACTORS         │
│  - 78,482 Passages (train.csv + pool_v4)     │        │  - Base Trunks: Q3-14B, Q2.5-14B, Q3-8B, Deb │
│  - Output 1: 4-View LinearSVC (C=0.50)       │        │  - Dynamic bfloat16 Overflow Rescue          │
│  - Output 2: 5-View BPE + SGD + Style Twins  │        │  - Reversible In-Place LoRA Arithmetic       │
│  - Dedicated 26-Feature Handwriting Model    │        │  - Snapshot Weight Isolation (Trunks 1 & 3)  │
│  - Pure Sparse Matrices (100% Non-Zero Fast) │        │  - Native LayerNorm Readout Projections      │
│  - Discrete Structural Rule Flags (N, 4)     │        │  - DeBERTa Token-0 Probing & NaN Guard       │
└──────────────────────┬───────────────────────┘        └──────────────────────┬───────────────────────┘
                       │                                                       │
                       └───────────────────────┬───────────────────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: WITHIN-PROMPT wPCA & EXACT 2^18 BITMASK DP + 384-DIM JOINT SPECTRAL BANK                  │
│  - 1,620-Row Balanced Training Set (90 prompts x 18 classes, full-rank covariance)                   │
│  - Prompt-Centered Within-Class Covariance S_e with Ledoit-Wolf Shrinkage Floors                     │
│  - ExactAuthorPosterior: Exact forward-backward dynamic programming over 2^18 author subsets         │
│  - Crash-Proof Cluster Splitting: Guaranteed cluster size 2 <= s <= 14 <= 18                         │
│  - Unsupervised 384-Dim Joint Spectral Bank: Top-64 components per checkpoint with Prior Scaling     │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: TWO-TIER CONVEX FUSION, SPECIALIST GATE & UNIFIED POSTPROCESSING                          │
│  - Two-Tier Convex Fusion: P_mix = 0.85 * R_main + 0.15 * P_comp                                     │
│  - Vectorized Claude Specialist Gate with exact marginal mass preservation                          │
│  - Unified Handwriting & Orthographic Shield:                                                        │
│      * Code-block stripped non-code prose discrete rule subtraction: log P += rule_flags @ W_pen    │
│      * Continuous 26-feature handwriting log-likelihood blend: log P += 0.04 * log_p_handwriting    │
│  - Official 300-step multiplicative Sinkhorn-Knopp column mass balancing (column sums = N / 18)      │
│  - Output: /kaggle/working/submission.parquet                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 18 Target Models Across Families

```
1. Anthropic Claude (5):  claude-fable-5, claude-opus-4-6, claude-opus-4-8,
                          claude-opus-5, claude-sonnet-5
2. OpenAI GPT (5):        gpt-5.5, gpt-5.6-luna, gpt-5.6-sol,
                          gpt-5.6-terra, gpt-6-astra
3. Google Gemini (4):     gemini-3.1-pro-preview, gemini-3.6-flash,
                          gemini-3.7-flash, gemini-3.8-flash
4. Zhipu GLM (2):         glm-5.3, glm-5.3-flash
5. Solo Artists (2):      kimi-k3 (Moonshot), qwen3.8-max (Alibaba)
```

---

## 3. Current Codebase Architecture

The codebase is organized in `src/namethatllm/` as a modular PyPA package with 4 decoupled execution modules:

### Module 1: Subprocess Dual-Output Lexical Engine (`lexical_engine.py`)
- **Memory Isolation**: Executes in a subprocess prior to GPU neural loading, freeing all host RAM before PyTorch initializes CUDA.
- **Output 1 (`log_p_svc_main`, Main Stack)**: Pure 4-view `LinearSVC(C=0.50)` fitted on 78,482 passages (`train.csv` + `pool_v4.parquet`):
  - Char n-grams: (3, 5), 250,000 features, scale 1.0
  - Word n-grams: (1, 3), 180,000 features, scale 0.7
  - Shape character patterns: (2, 6), 60,000 features, scale 0.5 (preserves uppercase/lowercase character blocks)
  - Function words & punctuation syntax: (1, 3), 50,000 features, scale 0.5 (115 canonical function words + contraction suffixes + punctuation tokens)
- **Output 2 (`p_lex_comp`, Complementary Stack)**: 5-view BPE subwords (75k features, sublinear TF) + SGDClassifier + prompt-centered style twin on $D_{\text{tfidf}}$ ($t=0.85$) + 4 within-family specialist refiners.
- **Dedicated Continuous Handwriting Model**: Extracts 26 continuous rhythm, punctuation, and contraction densities, standardized with `StandardScaler` (clipped to $[-5, 5]$). Trains plain `LogisticRegression(C=0.25)` and prompt-centered `LogisticRegression(C=0.15)` to output `log_p_handwriting`.
- **Discrete Rule Extraction**: Strips fenced code blocks (`_RE_CODE_BLOCK`) and returns `rule_flags_test` `(N, 4)`. Keeps n-gram matrices 100% sparse for ~30% faster SVM/SGD training.

### Module 2: Multi-Trunk Neural Extractors (`neural_extractor.py`)
- **Trunk Reuse Across 6 Checkpoints**:
  - `Trunk 1 (Qwen3-14B)`: Checkpoint 1A (Part-2, 27 layers) + Checkpoint 1B (Part-1, 24 layers)
  - `Trunk 2 (Qwen2.5-14B)`: Checkpoint 2 (Part-2, 32 layers)
  - `Trunk 3 (Qwen3-8B)`: Checkpoint 3A (In-Style, 24 layers) + Checkpoint 3B (Claude Head, 24 layers)
  - `Trunk 4 (DeBERTa-v3-large)`: Fine-tuned sequence classification encoder (24 layers)
- **Dynamic `bfloat16` Overflow Rescue**: Detects residual activation blowup ($> 65,504$) in deep layers and recomputes only affected rows in `bfloat16`, eliminating NaN propagation.
- **Reversible LoRA In-Place Arithmetic**: Merges LoRA adapters directly on GPU memory ($W_{\text{eff}} = W_0 + \frac{\alpha}{r} BA$).
- **Bit-Exact Snapshot Weight Isolation**: Caches affected trunk weights on CPU prior to merging and restores them via `restore_weights` on Trunks 1 and 3, eliminating cumulative FP16 subtraction drift.
- **Native LayerNorm Heads**: Evaluates LayerNorm projection modules matching checkpoint heads prior to logit computation.
- **DeBERTa NaN Guard**: Hidden states are clamped with `torch.nan_to_num(hs[l].float(), nan=0.0, posinf=1e4, neginf=-1e4)`.

### Module 3: Stylometric wPCA, $2^{18}$ Bitmask DP & Spectral Bank (`wpca_gaussian.py`, `spectral_bank.py`)
- **Within-Prompt Covariance Analysis (`wPCA`)**: Fits on a balanced 1,620-row training matrix (90 prompts $\times$ 18 classes). Explicit prompt-mean subtraction projects features into an authorial stylometric subspace isolated from prompt semantics.
- **Shrinkage Regularization**: Ledoit-Wolf style covariance shrinkage with per-checkpoint floors (`se_floor`).
- **Crash-Proof Cluster Splitting (`split_oversized_clusters_dist`)**: Splits oversized test clusters via average-linkage distance clustering, mathematically guaranteeing cluster sizes $2 \le s \le 14 \le 18$.
- **`ExactAuthorPosterior`**: Evaluates exact 18-step forward-backward bitmask dynamic programming over all $2^{18} = 262,144$ author configurations on safe clusters, enforcing the hard author uniqueness constraint ($c_i \neq c_j$).
- **384-Dim Joint Spectral Bank**: Extracts top-64 principal components from unsupervised prompt-centered features ($V_{\text{f32}}[:, :64]$) with out-of-sample prior scaling (zero label leakage). Fits dual logistic regression (plain $C=0.04$ + prompt-centered $C=0.03$).

### Module 4: Two-Tier Convex Fusion & Unified Postprocessing (`postprocess.py`)
- **Two-Tier Convex Blend**:
  $$P_{\text{mix}} = 0.85 \cdot R_{\text{main}} + 0.15 \cdot P_{\text{comp}}$$
  Guarantees the robust Part-Two baseline while harvesting incremental orthogonal signal from secondary models.
- **Vectorized Claude Specialist Gate**: Mass-preserving probability refinement within the Claude sibling classes:
  $$P_{\text{claude}} = 0.85 \cdot P_{\text{prior}} + 0.15 \cdot (\text{mass} \cdot P_{\text{specialist}})$$
- **Unified Handwriting & Orthographic Shield**:
  - Builds $(4, 18)$ log-penalty matrix $W_{\text{pen}}$:
    - Rule 1 (curly quotes `‘, ’, “, ”`): $-1.00$ log-penalty on Claude (5) and GLM (2)
    - Rule 2 ($\ge 2$ prose em-dashes `—`): $-0.45$ log-penalty on Gemini (4)
    - Rule 3 ($\ge 2$ straight contractions in non-code prose): $-0.60$ log-penalty on `gpt-5.6-sol`
    - Rule 4 ($\ge 1$ curly contraction in non-code prose): $-0.40$ reinforcement on Claude & GLM
  - Vectorized application: $\log P \mathrel{+}= B_{\text{disc}} W_{\text{pen}} + 0.04 \cdot \log P_{\text{hw}}$
  - Re-normalizes rows to the probability simplex.
- **300-Step Multiplicative Sinkhorn-Knopp Mass Balancing**: Iteratively rescales row and column marginals, equalizing predicted class mass to $N/18$.

---

## 4. Repository Layout (PyPA `src` Layout)

```
NameThatLLM/
├── pyproject.toml                       # Package build metadata, dependencies, and pytest config
├── README.md                            # High-level architecture guide based on current codebase
├── TUTORIAL.md                          # Comprehensive technical masterclass & mathematical guide
├── configs/
│   └── production_ensemble.json         # Declarative weights, floors, prior scales, and thresholds
├── kernels/
│   ├── submission/
│   │   ├── kernel-metadata.json         # Kaggle GPU/dataset mounts configuration
│   │   └── kaggle_submission.ipynb      # Compiled single-file production notebook
│   └── training/
│       ├── kernel-metadata.json
│       └── lora_training.ipynb
├── scripts/
│   ├── compile_notebook.py              # Single-source-of-truth AST compiler from src/ + configs/
│   └── push_submission.py               # Automated test, compile, and push CLI runner
├── src/namethatllm/                     # Decoupled production Python package
│   ├── __init__.py                      # Package exports and version
│   ├── config.py                        # 18 labels, families, simplex helpers, path resolvers
│   ├── lexical_engine.py                # 78k 4-view Main SVC + 5-view BPE + Handwriting classifier
│   ├── neural_extractor.py              # 4-trunk loader, reversible LoRA, snapshot isolation, LayerNorm
│   ├── wpca_gaussian.py                 # Full-rank wPCA, safe cluster splitting, 2^18 Bitmask DP
│   ├── spectral_bank.py                 # Unsupervised 384-dim Joint Spectral Bank (0 label leakage)
│   ├── postprocess.py                   # Two-tier blend, Claude gate, unified handwriting shield, Sinkhorn
│   └── pipeline.py                      # Top-level orchestrator connecting Modules 1 through 4
└── tests/                               # Comprehensive unit and integration test suite
    ├── test_compiler_and_integration.py # AST validation, compiler checks, mock-neural integration
    ├── test_lexical.py                  # Tokenizers, shape patterns, stopwords, dual lexical outputs
    ├── test_wpca_dp.py                  # Covariance shrinkage, two-way decomposition, ExactAuthorPosterior
    ├── test_spectral_bank.py            # Prior scaling, unsupervised V_f32 projection, OOF validation
    └── test_postprocess.py              # Two-tier blend, Claude gate, unified shield, Sinkhorn balancing
```

---

## 5. Developer Quickstart

### 1. Installation
Install the package locally in editable mode:
```powershell
pip install -e . --no-deps
```

### 2. Running Unit & Integration Tests
Run the complete 27-test suite:
```powershell
python -m pytest tests -v
```

### 3. Compiling & Pushing to Kaggle
Compile the notebook directly from `src/namethatllm/` and push to Kaggle:
```powershell
# Automated single-command execution (tests -> compile -> push)
python scripts/push_submission.py

# Or step-by-step
python scripts/compile_notebook.py --config configs/production_ensemble.json
python -m kaggle kernels push -p kernels/submission
python -m kaggle kernels status anilmittal/name-that-llm-v2-submission
```
