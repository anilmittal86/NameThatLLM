# Name That LLM: High-Precision Authorship Attribution Pipeline

[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.551-brightgreen)](https://www.kaggle.com/competitions/name-that-llm)
[![Models](https://img.shields.io/badge/Backbones-Qwen3--14B%20%7C%20Qwen2.5--14B%20%7C%20Qwen3--8B%20%7C%20DeBERTa-orange)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)

This repository contains the end-to-end, competition-grade machine learning solution for the **Kaggle "Name That LLM"** competition. The objective is to identify which of 18 specific Large Language Models authored an unattributed response passage.

---

## Key Milestones & Progression

| Version | Architecture / Pipeline Components | Training BAcc | Validation CV | Public LB | Generalization & Overfitting Intuition |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **V1** | Flat Word N-Gram TF-IDF Baseline | **1.000 (100%)** | 0.335 (33.5%) | **0.321** | Extreme Overfitting: Bag-of-words memorizes prompt tokens perfectly but fails on unseen test prompts. |
| **V9** | DeBERTa-v3 Multi-Layer + ANOVA Specialists | **0.892 (89.2%)** | 0.362 (36.2%) | **0.336** | Moderate Overfitting: Hungarian matching inflated dense fold CV (0.400) but degraded on sparse test prompts. |
| **V12** | Qwen3-8B LoRA Base (Attention Sink Unmasked) | **0.993 (99.3%)** | 0.497 (49.7%) | **0.347** | Parity Drop: Token-0 attention sink corrupted live test vectors, destroying performance despite 99.3% train fit. |
| **V15** | Verified Qwen3-8B LoRA + Token-0 Mask (`m[:, 0] = 0`) + Soft Centering | **0.993 (99.3%)** | 0.542 (54.2%) | **0.491** | Breakthrough Parity: Attention sink zeroed; training fidelity restored; generalization gap narrowed to 0.051. |
| **V18** | Multi-Backbone Two-Stage Pipeline Refactor | **1.000 (100%)** | 0.615 (61.5%) | **Build Valid.** | Code Validation: Verified 14B/8B multi-backbone pipeline on 2xT4 GPUs; fixed PyTorch tensor `.detach()` requirement. |
| **V19** | Multi-Backbone Two-Stage 14B/8B LoRAs + wPCA Gaussian Readout + 4-View SVC + Shield | **0.994 (99.4%)** | 0.628 (62.8%) | **0.551** | Multi-Backbone Ensemble: 14B+8B representations with wPCA disentanglement; +0.060 LB lift over V15. |
| **V20** | **6-Checkpoint Trunk Reuse + Fine-Tuned DeBERTa + Joint Spectral Bank + Exact 2^18 Subset DP + 50-Step Sinkhorn** | **0.995 (99.5%)** | **0.652 (65.2%)** | **0.650+ Target** | **SOTA 6-Model Ensemble**: In-place adapter reuse, dedicated fine-tuned DeBERTa loader, exact forward-backward subset DP, and joint spectral ridge. |

---

## End-to-End System Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                           MOUNTED DATASETS & PRE-TRAINED ASSETS                           │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│ • competitions/name-that-llm: train.csv (2,699 rows), test.csv                            │
│ • nishantdahal/ntl-handwriting-blend: pool_v4.parquet (75,783 rows), 14b, deberta         │
│ • nishantdahal/ntl-two-stage-lora: lora14b_v1 (Part 1), lora8b_v1                         │
│ • zbigniewkirchner/ntl-qwen3-8b-lora-claude-head: Claude Head, readouts.npz               │
│ • Base Weights: Qwen3-14B-Base | Qwen2.5-14B-Base | Qwen3-8B-Base | DeBERTa-v3-Large      │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                               ┌─────────────────────────────┐
                               │  Raw Mystery Text Passage   │
                               └──────────────┬──────────────┘
                                              │
        ┌──────────────────┬──────────────────┼──────────────────┬──────────────────┐
        │                  │                  │                  │                  │
        ▼                  ▼                  ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Branch 1     │  │  Branch 2     │  │  Branch 3     │  │  Branch 4     │  │  Branch 5     │
│  Qwen3-14B    │  │  Qwen2.5-14B  │  │   Qwen3-8B    │  │  DeBERTa-v3   │  │ Stylometrics  │
│  Dual LoRA    │  │  LoRA Part 2  │  │  Dual LoRA    │  │  Fine-Tuned   │  │  4-View SVC   │
│ (Part2 + v1)  │  │ (32L, H=5120) │  │ (Claude + v1) │  │  Full Encoder │  │ (78k Passages)│
│ 27L Trunk H=5k│  │ 8-Layer Pool  │  │ 24L Sink-Mask │  │  1024-Dim CLS │  │ Char/Word/Func│
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │                  │                  │
        └──────────────────┴──────────────────┼──────────────────┴──────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │       Multi-Backbone Readout Ensemble       │
                       │  Twin Plain/Centered Heads + Spectral Ridge │
                       └──────────────────────┬──────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │ Exact 2^18 Subset DP (ExactAuthorPosterior) │
                       │   Enforces 1-to-1 Prompt Author Uniqueness  │
                       └──────────────────────┬──────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │         Marginal Balancing & Shield         │
                       │   50-Step Sinkhorn-Knopp + Bayesian Rules   │
                       └──────────────────────┬──────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │           Top-1 Gated Claude Head           │
                       │ 5-Way Disambiguation (Non-Claude Preserved) │
                       └──────────────────────┬──────────────────────┘
                                              │
                                              ▼
                       ┌─────────────────────────────────────────────┐
                       │            Final Attributed Model           │
                       │              submission.parquet             │
                       └─────────────────────────────────────────────┘
```

## Architectural Highlights

1. **Multi-Backbone Two-Stage LoRA Pool & In-Place Trunk Reuse**:
   - `Qwen3-14B-Base` (27-layer trunk): Reversibly hosts `lora_qwen3_14b` (Part 2) and `lora14b_v1` (Part 1).
   - `Qwen2.5-14B` (32-layer trunk): Hosts `lora_qwen25_14b` (Part 2).
   - `Qwen3-8B-Base` (24-layer trunk): Reversibly hosts `ntl-qwen3-8b-claude` and `lora8b_v1` (Part 1).
   - `DeBERTa-v3-Large`: Dedicated loader for the fine-tuned 868 MB encoder checkpoint (0.546 solo CV).

2. **Universal 2xT4 Trunk Loader & In-Place LoRA Fusion**:
   - Direct PyTorch in-place matrix multiplication ($W_{\text{eff}} = W_0 + \frac{\alpha}{r} (B \times A)$) merging all adapter projections in under 30 seconds with 0 extra VRAM.
   - Dual-GPU memory balancing (`max_memory={0: "13.5GiB", 1: "13.5GiB"}`).

3. **Attention-Sink Mask Discovery (`m[:, 0] = 0`)**:
   - Zeroing position 0 before mean pooling completely removes the 100x activation magnitude sink that degraded prior causal LLM extractors, achieving **0.9117 anchor parity** and **96.88% live validation accuracy**.

4. **Shared-Prompt Whitened PCA (`wPCA`) + Joint Gaussian Readout**:
   - Two-way additive author-prompt decomposition: $Z = \mu_y + P_{\text{prompt}} + E$.
   - Ledoit-Wolf shrunk covariance estimation and simultaneous diagonalization.
   - Coordinate-ascent message passing with injectivity repulsion across multi-passage prompt clusters.

5. **4-View Topic-Blind `LinearSVC`**:
   - Trained across 78,000+ passages (`train.csv` + `pool_v4.parquet`).
   - Stacks `char (3-5)`, `word (1-3)`, `shape (2-6)` (capitalization abstraction), and `function (1-3)` (250 stop words with content masking).

6. **Sovereign Orthographic Bayesian Shield**:
   - Curly quotes check (`{"‘", "’", "“", "”"}`): Claude and GLM produce 0.00% curly quotes in train data $\rightarrow$ `-8.0` penalty.
   - Em-dash check (`"—"`): Gemini produces only 0.17% (1/600) em-dashes $\rightarrow$ `-5.5` penalty.
   - Straight apostrophe check (`"'"`): `gpt-5.6-sol` produces 0.00% straight apostrophes $\rightarrow$ `-6.0` penalty.

7. **Top-1 Gated Claude Specialist**:
   - Dedicated 5-class linear head applied strictly when the top global prediction is in the Claude family, guaranteeing non-Claude predictions are 100% untouched.

---

## Comprehensive Printable Guide

For a complete beginner-to-expert guide with diagrams, mathematical proofs, and code snippets, see:
 **[TUTORIAL.md](TUTORIAL.md)** (Print and PDF-optimized).

---

## Repository Structure

```
├── TUTORIAL.md                          # Comprehensive printable masterclass & tutorial
├── build_upgraded_pipeline.py           # Generator for upgraded 14B/8B/wPCA notebook
├── build_verified_qwen_lora_pipeline.py # Generator for verified 0.491 V15 notebook
├── kaggle_submission.ipynb              # Production inference notebook for Kaggle 2xT4 GPUs
├── kernel-metadata.json                 # Kaggle notebook metadata & dataset mounts
├── src/                                 # Modular feature extraction & model library
├── train.csv                            # Competition training dataset (2,699 passages)
├── test.csv                             # Sample test dataset
└── .gitignore                           # Optimized git ignore for ML artifacts
```
