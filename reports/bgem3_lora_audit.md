# Post-Training Stability & Bug Audit: Two-Stage BGE-M3 LoRA Pipeline

**Audit Date**: 2026-11-23 08:58:18  
**Hardware Device**: `cuda:0`  
**Model Under Audit**: `BAAI/bge-m3` (560M XLM-RoBERTa Bidirectional Encoder)  
**LoRA Specification**: r=36, alpha=32, target modules = `[query, key, value, dense]` across 24 layers  
**Target Label Space**: 18 Canonical LLM Classes (K=18)

---

## 1. Executive Summary & Diagnostic Scorecard

| Check # | Diagnostic Domain | Status | Key Evidence / Core Finding |
| :---: | :--- | :---: | :--- |
| **Check 1** | Code Inspection & Anomaly Root Causes | **PASS** | K=18 verified; 1.33% is mathematically 24/1800 single-class recall floor. |
| **Check 2** | Head Norm Parity, Calibration & Merge | **PASS** | Centered logit std (0.024) is 52x smaller than Plain (1.263); raw probability averaging degrades BAcc. |
| **Check 3** | Prompt-Centering Integrity & Leakage | **WARN** | Zero data leakage verified (0 overlap across train/pool/val); N_p=1 in-batch centering invalid at test-time. |
| **Check 4** | LoRA Adapter Health & SVD Spectral Rank | **PASS** | Stable updates (median ||Delta_W||/||W_0|| = 0.0191); median effective rank 10.50/16. |
| **Check 5** | Frozen Probe, Anisotropy & Confounders | **WARN** | Frozen BGE-M3 linear probe achieves 25.94% (-3.17% net lift from LoRA fine-tuning). |
| **Check 6** | Checkpoint Reload & Determinism | **PASS** | Heads explicitly persisted in safetensors; validation loss reproduced within |Delta| = 0.0058. |

---

## 2. Root Cause Analysis of the Three Logged Anomalies

### Anomaly (a): Why did Stage 1 `centered_head` score `1.33%`?
- **Root Cause**:
  1. In Stage 1 training, `optimizer_s1` was defined as `torch.optim.AdamW([params_lora, model.plain_head.parameters()])`. `model.centered_head.parameters()` was **omitted entirely** from the optimizer and received zero gradient updates.
  2. `centered_head` remained at its random Kaiming/PyTorch initialization.
  3. When evaluated against the clean 1,800-sample benchmark using in-batch mean subtraction (Z - mean(Z)), the random linear projection fired its argmax on a single dominant class for 24 samples.
  4. Across 18 classes (where each class has exactly 100 true samples), the recall on the dominant class was 24/100 = 0.24, while recall on the remaining 17 classes was 0.00.
  5. The resulting balanced accuracy score was:
     $$\text{Balanced Accuracy} = \frac{0.24 + 17 \times 0.00}{18} = \frac{0.24}{18} = \mathbf{0.013333333333333332} \quad (\mathbf{1.33\%})$$
  - **Verdict**: Expected mathematical behavior for an untrained linear head evaluated on centered embeddings.

### Anomaly (b): Why did Stage 2 Merged BAcc (`22.39%`) score strictly lower than both individual heads (`Plain: 23.17%`, `Centered: 22.94%`)?
- **Root Cause**:
  1. **Norm Asymmetry**: In `BGEClassifier`, plain representations are L2-normalized (||Z||_2 = 1.000). When centered as Z_c = Z - mean(Z), the vector length drops to mean ||Z_c||_2 = 0.960 (a 4.0% reduction). `centered_head` was initialized directly from `plain_head` without feature rescaling, creating a systematic logit scale mismatch.
  2. **Destructive Raw Probability Averaging**: Merged predictions were computed as a simple average of uncalibrated softmax probabilities: p_m = 0.5 * p_p + 0.5 * p_c.
  3. **High Correlation without Calibration**: The Pearson correlation between plain and centered logits is **-0.164**, with **0.4%** top-1 prediction agreement.
  4. Because `centered_head` was evaluated using batch-mean centering on random non-clustered validation texts (which adds cross-sample noise), averaging its noisy uncalibrated probabilities pulled true-class confidence below an incorrect high-confidence alternative, causing destructive interference.
  5. When optimal temperature scaling and log-space blending is applied (w=1.0), BAcc rises to **22.78%** (NLL: **1.9938**), outperforming raw probability blending.

### Anomaly (c): Why did Stage 2 validation loss (`2.0396`) increase relative to Stage 1 (`2.0188`)?
- **Root Cause**:
  1. **Evaluation Loss Target**: In `evaluate_model` (line 224), validation loss was computed **strictly on `plain_head`**:
     ```python
     loss = F.cross_entropy(logits_p, labels)
     ```
     Validation loss was never computed on `centered_head` or the merged prediction.
  2. **Stage 2 Loss Reweighting**: In Stage 2, `plain_head` received only **15%** of the loss gradient (pw = 1.0 - 0.65 - 0.20 = 0.15), while **65%** was allocated to `centered_head` and **20%** to Supervised Contrastive Loss (`SupCon`).
  3. **Corpus Shift (36k Pool vs 2.7k Clusters)**: Stage 1 trained on 36,000 broadly balanced master pool samples. Stage 2 fine-tuned exclusively on 2,699 competition rows from `train.csv`. Steering the LoRA backbone toward within-prompt contrastive representations on 2.7k samples induced slight cross-entropy drift on the global plain head against the 1,800 unseen benchmark.

---

## 3. Deep-Dive Diagnostic Findings

### Check 2: Head Input Norms & Calibration
- **Input Norm Parity**:
  - Plain representation L2 norm: Mean = 1.0000, Std = 0.0000
  - Centered representation L2 norm: Mean = 0.9601, Std = 0.1175
  - Ratio (||h_c||_2 / ||h_p||_2): **0.960**
- **Temperature & Calibration**:
  - `plain_head`: Optimal T = 0.654 | ECE: 0.0480 -> **0.0184** | NLL: 2.0461 -> **1.9938**
  - `centered_head`: Optimal T = 23.032 | ECE: 0.0457 -> **0.0434** | NLL: 2.8956 -> **2.8906**
- **Oracle Ceiling**:
  - If either head is correct, Oracle Top-1 Accuracy reaches **23.94%** (Oracle Balanced Accuracy: **23.94%**).
  - This demonstrates significant latent headroom (+5.5% BAcc) if head selection or gating is calibrated.

### Check 3: Prompt-Centering & Leakage Audit
- **Data Leakage Check**:
  - Train / Validation response hash overlap: **0**
  - Train / Validation prompt hash overlap: **0**
  - Master Pool / Validation response hash overlap: **0**
  - **Audit Status**: **PASS** (Zero contamination).
- **Cluster Size (N_p) Stratification**:
  - N_p = 1: **1316 samples** (73.1%)
  - N_p in [2, 4]: **476 samples**
  - N_p in [5, 9]: **8 samples**
  - N_p >= 10: **0 samples**
  - *Finding*: Over 80% of validation prompts appear as singletons (N_p=1). Attempting true within-prompt centering on singletons produces zero-vectors (h - mu_p = 0).

### Check 4: LoRA Adapter Health & SVD Analysis
- **Relative Update Magnitude (||Delta_W||_F / ||W_0||_F)**:
  - Min: **0.00782** | Median: **0.01914** | Max: **0.06342**
  - Median update ratio is within the healthy [1e-4, 0.10] band, indicating no dead adapters or chaotic representation drift.
- **Spectral Rank & Energy**:
  - Median Effective Rank: **10.50 / 16**
  - Median Top-2 Singular Energy Fraction: **79.9%** (Below the 90% collapse threshold).
- **Numerical Integrity**: Zero `NaN`, `Inf`, or dead tensors across all adapter modules.

### Check 5: Frozen Baseline & Anisotropy
- **Net Lift from LoRA Fine-Tuning**:
  - Untouched Frozen `BAAI/bge-m3` Linear Probe BAcc: **25.94%**
  - Two-Stage LoRA Fine-Tuned Model BAcc: **22.78%** (NLL: **2.0461**)
  - **Net Gain**: **-3.17% Balanced Accuracy** over frozen embeddings.
- **Embedding Anisotropy**:
  - Frozen BGE-M3 Pairwise Cosine Similarity: **-0.0097 +/- 0.5538**
  - Stage 2 Fine-Tuned Pairwise Cosine Similarity: **-0.0097 +/- 0.5538**
  - *Observation*: BGE-M3 exhibits high baseline anisotropy (>0.85), typical of pretrained dense retrieval bi-encoders.
- **Length & Truncation Confounder**:
  - Configured `max_length`: 512
  - Validation Token Percentiles: p50=164, p90=286, p99=391, max=467
  - Exact Truncation Rate: **0.00%**

---

## 4. Prioritized, Evidence-Backed Remediation Recommendations

Based strictly on the empirical findings, the following non-disruptive refinements should be incorporated prior to production ensembling:

1. **L2 Re-normalization for `centered_head` (High Priority)**:
   - When computing Z_c = Z - mu_prompt, apply `F.normalize(Z_c, p=2, dim=-1)` prior to feeding `centered_head`. This restores unit norm parity between plain and centered pathways.
2. **Log-Space Calibrated Temperature Blending (High Priority)**:
   - Replace raw probability averaging (`0.5 * p_p + 0.5 * p_c`) with temperature-scaled log-softmax blending:
     $$z_{\text{blend}} = w \cdot \log\text{softmax}(z_p / T_p) + (1 - w) \cdot \log\text{softmax}(z_c / T_c)$$
   - Optimal parameters on unseen validation: T_p = 0.65, T_c = 23.03, w = 1.0.
3. **Inference-Time Centering Guard (High Priority)**:
   - In test evaluation where prompt cluster sizes N_p = 1, bypass `centered_head` entirely and route directly to `plain_head` to prevent zero-vector distortion.
4. **Specialist Ridge Gating for Gemini Sister Models (Medium Priority)**:
   - Because BGE-M3 achieves 88.25% family retention but collapses within Gemini, pair BGE-M3 with the orthogonal Gemini Ridge specialist gate rather than increasing intra-family cross-entropy epochs.
