# Name That LLM: High-Precision Authorship Attribution Pipeline

[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.491%2B-blue)](https://www.kaggle.com/competitions/name-that-llm)
[![Models](https://img.shields.io/badge/Backbones-Qwen3--14B%20%7C%20Qwen2.5--14B%20%7C%20Qwen3--8B%20%7C%20DeBERTa-orange)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)

This repository contains the end-to-end, competition-grade machine learning solution for the **Kaggle "Name That LLM"** competition. The objective is to identify which of 18 specific Large Language Models authored an unattributed response passage.

---

## Key Milestones & Progression

| Version | Architecture / Pipeline Components | Score (Balanced Accuracy) |
| :--- | :--- | :---: |
| **V1** | Flat Word N-Gram TF-IDF Baseline | **0.321** |
| **V9** | DeBERTa-v3 Multi-Layer + ANOVA Specialists | **0.347** |
| **V12** | Qwen3-8B LoRA Base (Attention Sink Unmasked) | **0.347** |
| **V15** | **Verified Qwen3-8B LoRA + Token-0 Mask (`m[:, 0] = 0`) + Soft Centering** | **0.491** |
| **V18** | Multi-Backbone Two-Stage Pipeline Refactor | Build Validation |
| **V19** | **Multi-Backbone Two-Stage 14B/8B LoRAs + wPCA Gaussian Readout + 4-View SVC + Shield** | **0.620+ Target (Ref 57628730)** |

---

## Architectural Highlights

1. **Multi-Backbone Two-Stage LoRA Pool**:
   - `Qwen3-14B-Base` (27-layer truncated trunk, 8-layer extraction).
   - `Qwen2.5-14B` (32-layer truncated trunk, 8-layer extraction).
   - `Qwen3-8B-Base` (24-layer truncated trunk, 8-layer extraction + Claude specialist head).
   - `DeBERTa-v3-Large` (frozen and fine-tuned pooled representations).

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
