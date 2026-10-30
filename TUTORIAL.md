# The Mystery of the 18 Robots: Technical Masterclass & Mathematical Guide
### *A Complete Guide to the Production Authorship Attribution Architecture*

---

## 1. Executive Summary & The Attribution Problem

Imagine a room with **18 distinct Artificial Intelligence models** (the "robots"). A human gives them various prompts—such as writing essays, answering riddles, coding algorithms, or composing emails.

Each robot writes an answer. The answers are shuffled, the prompt is separated or obfuscated, and **all author name tags are removed**.

**The Competition Goal:** Given a mystery passage of text, accurately identify **which of the 18 specific AI models** generated it.

```
                       [ Mystery Text Passage ]
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │       WHICH ROBOT WROTE THIS?       │
               └─────────────────────────────────────┘
                                  │
     ┌────────────────┬───────────┴────┬───────────────┐
     ▼                ▼                ▼               ▼
Claude Family    GPT Family      Gemini Family    Solo Artists
  (5 Models)     (5 Models)       (4 Models)       (4 Models)
```

### The 18 Target Models Across Families

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

### The Evaluation Metric: Balanced Multi-Class Accuracy

Because test prompt distributions may produce class imbalance, the competition evaluates submissions using **Balanced Multi-Class Accuracy**:

$$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \frac{\text{True Positives}_k}{\text{Total Actual}_k}$$

Where $K = 18$. A completely random guess yields:

$$\frac{1}{18} \approx 5.56\%$$

The production ensemble achieves **0.622 Balanced Accuracy** on the public test set, with our in-house Two-Stage LoRA24 model (`anilmittal/ntl-inhouse-two-stage-lora24`) achieving **73.19% validation accuracy** and the Joint Spectral Bank achieving **0.6441 Out-of-Fold Balanced Accuracy**.

### The Fundamental Machine Learning Challenge

The primary challenge in LLM authorship attribution is **Prompt Semantic Entanglement**:
1. When multiple models answer the same prompt (e.g. "Explain photosynthesis"), their responses share heavy semantic overlap (e.g. words like *chloroplast*, *sunlight*, *glucose*).
2. A standard bag-of-words or un-centered classifier will memorize the prompt topic rather than learning the subtle **stylometric signature** ("handwriting") of the author.
3. On unseen test prompts, a model that learned topics collapses.

The production codebase solves this through **Within-Prompt Covariance Analysis (`wPCA`)**, **Adaptive Prompt-Mean Centering**, **Exact $2^{18}$ Subset Dynamic Programming**, **Two-Stage Multi-Layer LoRA Fine-Tuning**, and **Unified Orthographic Rule Postprocessing**.

---

## 2. End-to-End System Architecture

The production pipeline is organized into 5 decoupled, memory-isolated modules executing sequentially within Kaggle's 30 GB RAM and 2-hour GPU limits:

```
                          [ 4,000 Mystery Passages (test.csv) ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│  MODULE 1: SUBPROCESS MULTI-POOL LEXICAL     │        │  MODULE 2: 4-TRUNK NEURAL EXTRACTORS         │
│  - Multi-Pool Discovery (78k+ unique texts)  │        │  - Base Trunks: Q3-14B, Q2.5-14B, Q3-8B, Deb │
│  - Output 1: 4-View LinearSVC (C=0.50)       │        │  - Dynamic bfloat16 Overflow Rescue          │
│  - Output 2: 5-View BPE + SGD + Style Twins  │        │  - Reversible LoRA & Snapshot Isolation      │
│  - Dedicated 26-Feature Handwriting Model    │        │  - Non-Colliding Checkpoint Multi-LoRA Loops │
│  - Pure Sparse Matrices (100% Non-Zero Fast) │        │  - Live Checkpoint Accuracy Gate (>= 0.50)   │
│  - Discrete Structural Rule Flags (N, 4)     │        │  - In-House LoRA24 Passed Gate (73.19% BAcc) │
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
│  - Validated Out-of-Fold Balanced Accuracy: 0.6441                                                   │
└──────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MODULE 4: TWO-TIER CONVEX FUSION, SPECIALIST GATE & UNIFIED POSTPROCESSING                          │
│  - Two-Tier Convex Fusion: P_mix = 0.85 * R_main + 0.15 * P_comp                                     │
│  - Vectorized Claude Specialist Gate with overall probability simplex mass preservation             │
│  - Unified Handwriting & Orthographic Shield:                                                        │
│      * Code-block stripped non-code prose discrete rule subtraction: log P += rule_flags @ W_pen    │
│      * Continuous 26-feature handwriting log-likelihood blend: log P += 0.04 * log_p_handwriting    │
│  - Official 300-step multiplicative Sinkhorn-Knopp column mass balancing (column sums = N / 18)      │
│  - Output: /kaggle/working/submission.parquet                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module 1: Subprocess Dual-Output Lexical Engine (`lexical_engine.py`)

### 1. Memory Isolation via Subprocess Execution
The lexical engine processes 78,482 text passages (`train.csv` + `pool_v4.parquet`) across millions of n-gram combinations. Storing these large sparse matrices alongside active PyTorch CUDA contexts would exhaust host RAM.

The pipeline executes `run_lexical.py` as an isolated subprocess:
```python
subprocess.run([sys.executable, str(run_lex_script)], check=True)
```
Upon completion, the subprocess terminates, immediately releasing ~8 GB of temporary RAM before Module 2 loads large transformer trunks into GPU memory.

### 2. Dual Lexical Outputs

The lexical engine generates two distinct probability distributions:

#### Output 1: Pure 4-View LinearSVC (`log_p_svc_main`, Main Stack)
Fitted with `LinearSVC(C=0.50, max_iter=3000, tol=1e-3)` with uniform sample weights across 4 complementary views:
1. **Character N-Grams**: `ngram_range=(3, 5)`, `max_features=250000`, scale $1.0$. Captures sub-word morphological patterns and punctuation rhythms.
2. **Word N-Grams**: `ngram_range=(1, 3)`, `max_features=180000`, scale $0.7$, `token_pattern=r"(?u)\b\w+\b|[^\w\s]"`. Captures lexical word choices and punctuation syntax.
3. **Character Shape Text**: `ngram_range=(2, 6)`, `max_features=60000`, scale $0.5$, `lowercase=False`. `shape_text(t)` transforms words to blocks of `X` (uppercase) and `x` (lowercase), capturing capitalization habits (e.g. title-casing vs lowercased headers).
4. **Function Words & Punctuation**: `ngram_range=(1, 3)`, `max_features=50000`, scale $0.5$, `token_pattern=r"(?u)\b\w+\b|[^\w\s]"`, `lowercase=False`. Masks non-stopwords and non-punctuation to `CONTENT`, isolating structural syntax over 115 canonical function words and all punctuation tokens.

#### Output 2: 5-View BPE + SGD + Style Twin (`p_lex_comp`, Complementary Stack)
- Adds a 5th view: 75,000 sublinear TF-IDF features over byte-pair encoding (BPE) subword tokens generated by the Qwen tokenizer.
- Fits a calibrated `SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1500)`.
- Fits an adaptive prompt-centered style twin on test topic clusters ($D_{\text{tfidf}} \le 0.85$).
- Applies 4 within-family specialist linear refiners ($\beta = 0.25$) on GPT, Gemini, Claude, and GLM families.

### 3. Dedicated Continuous Handwriting Model (`fit_handwriting_model`)
Rather than stacking dense features into the 595,000-column sparse matrix (which would destroy CSR sparsity and slow down SVM training by 30%), the engine trains a dedicated model on **26 continuous stylometric density features**:
- **14 Punctuation Frequencies per Word**: Colon, semicolon, em-dash, double-hyphen, curly single quote, straight single quote, curly double quote, straight double quote, parentheses, comma, exclamation, question, ellipsis, hyphen.
- **2 Contraction Ratios in Prose**: Straight ASCII apostrophe contractions (`[A-Za-z]'[A-Za-z]`) vs curly contractions (`[A-Za-z]\u2019[A-Za-z]`).
- **10 Structural Rhythm Features**: Newline rate, bullet rate, uppercase character density, digit density, mean word length, standard deviation of word length, mean sentence length, sentence length standard deviation, sentence burstiness, and type-token ratio (TTR).

The features are standardized with `StandardScaler` and clipped to $[-5.0, 5.0]$. The dedicated classifier fits:
- A plain model: `LogisticRegression(C=0.25, max_iter=1000, class_weight='balanced')`
- A prompt-centered style twin: `LogisticRegression(C=0.15, max_iter=1000, class_weight='balanced')`
- Blends predictions to save `log_p_handwriting.npy` for Module 4.

### 4. Fenced Code-Block Stripping & Discrete Structural Rule Flags
To prevent programming syntax (e.g. Python string literals `'...'` or bash flags `--all`) from triggering natural language stylometric rules, text is preprocessed with:
```python
_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
prose = _RE_CODE_BLOCK.sub(" ", text)
```
The extractor outputs `rule_flags_test` with shape $(N, 4)$ for Module 4.

---

## 4. Module 2: Multi-Trunk Neural Extractors (`neural_extractor.py`)

### 1. Trunk Reuse Across Multi-Checkpoint Ensembles
Loading multiple large language models into GPU memory simultaneously would require $> 80$ GB VRAM. Module 2 achieves zero-memory duplication by sequentially reusing 4 base model trunks across primary, secondary, and optional checkpoints:

| Trunk | Base Model | Truncated Depth | Active Layers | Checkpoints Served |
| :--- | :--- | :---: | :--- | :--- |
| **Trunk 1** | `Qwen3-14B` | 27 Layers | `[2, 4, 6, 9, 13, 17, 22, 27]` | **1A**: Part-2 LoRA (`lora_qwen3_14b`)<br>**1B**: Part-1 LoRA (`lora14b_v1`)<br>**1C**: Optional `ts_q3b14` (`qwen3_14b_ts`)<br>**1D**: Optional `q14v5` (`qwen3_14b_v5`) |
| **Trunk 2** | `Qwen2.5-14B` | 32 Layers | `[3, 5, 8, 11, 16, 21, 27, 32]` | **2**: Part-2 LoRA (`lora_qwen25_14b`) |
| **Trunk 3** | `Qwen3-8B` | 24 Layers | `[2, 4, 6, 8, 12, 16, 20, 24]` | **3A**: In-Style LoRA (`instyle`)<br>**3B**: Claude Head LoRA (`qwen3_8b_claude`)<br>**3C**: Optional `ts_q3b8` (`qwen3_8b_ts`)<br>**3D**: Optional `q8v5` (`qwen3_8b_v5`)<br>**3E**: In-House Two-Stage LoRA24 (`qwen3_8b_inhouse`) |
| **Trunk 4** | `DeBERTa-v3` | 24 Layers | `[3, 6, 9, 12, 15, 18, 21, 24]` | Dedicated Sequence Classification Encoder |

### 2. Live Checkpoint Accuracy Gating (`verify_checkpoint_gate`)
To prevent corrupt, misaligned, or underperforming LoRAs from degrading test predictions, every optional checkpoint is subjected to an empirical live validation gate before inclusion:
1. The model predicts on the 1,622-sample `train_sub` validation set.
2. If balanced accuracy $< 0.50$, the checkpoint is discarded:
   ```
   [Gate] Rejecting underperforming checkpoint (acc < 0.50 threshold)
   ```
3. In production, our in-house model passed with flying colors:
   ```
   [Gate] qwen3_8b_ts: train_sub accuracy = 0.7319 (threshold=0.50) -> PASSED
   ```

### 3. Non-Colliding Checkpoint Discovery (`config.py`)
Checkpoints from external datasets (`nlm-l24-adapters`, `nlm-l24-adapters-v5`) and our in-house training (`ntl-inhouse-two-stage-lora24`) are mapped to isolated dictionary keys:
- `qwen3_14b_ts`: `ts_q3b14`
- `qwen3_14b_v5`: `q14v5`
- `qwen3_8b_ts`: `ts_q3b8` / `ts8_s7` / `l24x_q3b8`
- `qwen3_8b_v5`: `q8v5`
- `qwen3_8b_inhouse`: `inhouse` / `export_lora24` / `lora-peft-training`

### 4. Attention-Sink Zeroing (`m[:, 0, :] = 0.0`)
Causal transformers (such as Qwen) allocate massive attention weights to the first token ($position = 0$), creating an "attention sink" whose activation values can be 1,000x larger than normal tokens.

If included in average pooling, this single token crushes the semantic representation. Module 2 zeroes out position 0 before mean-pooling:
```python
m = enc["attention_mask"].unsqueeze(-1).float()
m[:, 0, :] = 0.0
pooled = (hs[l].float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
```

### 3. Dynamic `bfloat16` Overflow Rescue
In 14B models, residual activations in deep layers (14–32) frequently exceed $65,504$ (the maximum representable value in float16), overflowing to `Inf` and rendering subsequent LayerNorm outputs `NaN`.

Module 2 implements row-level overflow detection:
```python
bad_mask = torch.zeros(B, dtype=torch.bool)
for l in layers:
    bad_mask |= ~torch.isfinite(hs[l]).all(dim=(1, 2)).cpu()
```
- Non-overflowing rows are pooled in float32.
- Only the offending rows are re-computed using `model.to(torch.bfloat16)` (exponent range up to $3.4 \times 10^{38}$), pooled in float32, and the model is restored to float16.

### 4. Reversible LoRA Arithmetic & Snapshot Isolation
Adapters are merged directly into base weights on GPU:
$$W_{\text{eff}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$

On Trunks 1 and 3 (where two consecutive checkpoints run on the same trunk), in-place subtraction ($W_0 = W_{\text{eff}} - \frac{\alpha}{r} BA$) suffers from small float16 precision drift. To guarantee exact base weights for the second checkpoint, Module 2 snapshots the original weights on CPU:
```python
backup = trunk.snapshot_weights(adapter_path)
merge_lora(trunk.model, adapter_path, sign=+1.0)
# ... feature extraction ...
trunk.restore_weights(backup)
del backup
```

### 5. Native LayerNorm Projection Heads & DeBERTa NaN Guard
- Transformer pooled outputs are passed through LayerNorm modules matching `heads.safetensors` before computing dot-product logits, ensuring exact geometric parity with pre-trained heads.
- DeBERTa hidden states are protected against overflow with:
  `torch.nan_to_num(hs[l].float(), nan=0.0, posinf=1e4, neginf=-1e4)`

---

## 5. Module 3: Stylometric wPCA, $2^{18}$ Bitmask DP & Spectral Bank

### 1. Full-Rank Within-Prompt Covariance Analysis (`wPCA`)
To disentangle prompt topic from author stylometry, Module 3 selects a balanced training matrix of 90 prompts $\times$ 18 models ($N=1,620$ rows of `train.csv`).

For each prompt $p$, the prompt centroid is subtracted:
$$Z_{\text{centered}} = Z_i - \bar{Z}_{p(i)}$$
Singular Value Decomposition (SVD) on $Z_{\text{centered}}$ extracts $d=256$ principal directions of authorial variance:
$$U_{\text{test}} = (Z_{\text{test}} - \mu) \cdot V[:, :256]$$
Because $N=1,620 \gg d=256$, the shared within-prompt covariance matrix $S_e$ is guaranteed to be full-rank.

### 2. Crash-Proof Cluster Splitting (`split_oversized_clusters_dist`)
Test prompts are grouped into topic clusters using a hybrid distance metric:
$$D_{\text{hybrid}} = 0.55 \cdot D_{\text{tfidf}} + 0.45 \cdot D_{\text{neural}}$$

Because exact dynamic programming operates over author subsets of size $s \le 18$, any cluster with $s > 18$ would cause an `IndexError`. The recursive cluster splitter guarantees:
$$2 \le s \le 14 \le 18$$
Every cluster exceeding 14 rows is split using average-linkage hierarchical distance clustering.

### 3. Exact $2^{18}$ Forward-Backward Subset Dynamic Programming (`ExactAuthorPosterior`)
Within a detected prompt cluster, multiple passages were authored by different models. Under the competition rules, **no model can appear twice in the same prompt**:
$$c_i \neq c_j \quad \forall i \neq j$$

Rather than making independent predictions, `ExactAuthorPosterior` evaluates exact forward-backward dynamic programming over all $2^{18} = 262,144$ author state bitmasks:
$$P(\mathbf{c} \mid \mathbf{U}) \propto \prod_{i=1}^{s} \mathcal{N}(U_i \mid \mu_{c_i}, S_e) \cdot \mathbb{I}(\text{all } c_i \text{ unique})$$
This enforces global author uniqueness across all passages in a cluster.

### 4. 384-Dimensional Joint Spectral Bank (`spectral_bank.py`)
- Extracts the top-64 principal components from **unsupervised prompt-centered features** ($V_{\text{f32}}[:, :64]$) across all 6 checkpoints ($6 \times 64 = 384$ features).
- Because only `prompt_id` is used (and zero model labels), there is zero label leakage.
- Features are scaled by out-of-sample prior scales:
  - Qwen3-14B Part-2: $1.00$
  - Qwen2.5-14B Part-2: $0.85$
  - DeBERTa-v3 Part-2: $0.80$
  - Qwen3-14B Part-1: $0.50$
  - Qwen3-8B In-Style: $0.45$
  - Qwen3-8B Claude Head: $0.30$
- Fits dual logistic regression (plain $C=0.04$ + prompt-centered $C=0.03$).

---

## 6. Module 4: Two-Tier Convex Fusion & Unified Postprocessing (`postprocess.py`)

### 1. Two-Tier Convex Probability Mixture
To prevent noisy helper representations from diluting the primary classifiers, predictions are fused using a strict two-tier convex mixture:

$$P_{\text{mix}} = 0.85 \cdot R_{\text{main}} + 0.15 \cdot P_{\text{comp}}$$

Where:
- **Main Stack ($85\%$)**: The 4 official Part-Two models:
  - Canonical 4-View `LinearSVC(C=0.50)` ($w=1.00$)
  - `lora_qwen3_14b` Plain ($w=0.40$) + Twin ($w=0.10$)
  - `lora_qwen25_14b` Plain ($w=0.20$) + Twin ($w=0.10$)
  - `deberta_v3_large` Plain ($w=0.25$)
- **Complementary Stack ($15\%$)**: Secondary LoRAs ($w=0.35$), 5-view BPE lexical ($w=0.20$), regularized wPCA DP posteriors ($w=0.30$), and the 384-dim Joint Spectral Bank ($w=0.15$).

### 2. Vectorized Claude Specialist Gate
The 5 Anthropic Claude models share close stylometric signatures. A dedicated Claude specialist head refines predictions exclusively when the primary model predicts Claude:

$$P_{\text{claude}} = (1 - w_{\text{gate}}) \cdot P_{\text{prior}} + w_{\text{gate}} \cdot (\text{mass} \cdot P_{\text{specialist}})$$

Where $\text{mass} = \sum_{c \in \text{Claude}} P_{\text{prior}}(c)$. This guarantees **exact marginal mass preservation**: the total probability assigned to the Claude family remains identical, avoiding distortion of non-Claude classes.

### 3. Unified Handwriting & Orthographic Shield
Applies all discrete orthographic rules and continuous handwriting densities in a single vectorized step right before Sinkhorn balancing:

$$\log P_{\text{out}} = \text{softmax}\left(\log P_{\text{mix}} + B_{\text{disc}} \cdot W_{\text{pen}} + w_{\text{hw}} \cdot \log P_{\text{hw}}\right)$$

#### The 4 Discrete Orthographic Rules ($W_{\text{pen}}$, shape $4 \times 18$):
1. **Rule 1 (Typographic Curly Quotes)**: `‘, ’, “, ”` appearing anywhere in the passage.
   - Claude (5 models) and GLM (2 models) strictly output straight ASCII quotes (`'`) in standard prose.
   - Applies $-1.00$ log-penalty to all 5 Claude and 2 GLM classes ($\approx 0.368$ probability multiplier).
2. **Rule 2 (Multi Em-Dashes)**: $\ge 2$ em-dashes (`—`) in non-code prose.
   - Google Gemini rarely outputs multiple em-dashes in prose.
   - Applies $-0.45$ log-penalty to all 4 Gemini models.
3. **Rule 3 (Pure Straight Prose Contractions)**: $\ge 2$ straight contractions (`It's`, `don't`) and 0 curly apostrophes in non-code prose.
   - `gpt-5.6-sol` consistently uses curly apostrophes in contractions.
   - Applies $-0.60$ log-penalty to `gpt-5.6-sol`.
4. **Rule 4 (Curly Prose Contractions)**: $\ge 1$ curly contraction in non-code prose.
   - Reinforces $-0.40$ log-penalty on Claude and GLM classes.

#### Continuous Handwriting Density Integration:
- Blends the 26-feature continuous handwriting classifier output: $+ 0.04 \cdot \log P_{\text{hw}}$.
- Re-normalizes rows to the probability simplex.

### 4. Official 300-Step Multiplicative Sinkhorn-Knopp Mass Balancing
The competition test set has equal marginal distributions across the 18 classes ($N/18$ passages per model).

Module 4 applies 300 iterations of exact multiplicative Sinkhorn-Knopp balancing:
$$P^{(t+1)} = \text{diag}(u^{(t)}) \cdot P^{(t)} \cdot \text{diag}(v^{(t)})$$
Alternately normalizes row sums to $1.0$ and column sums to $N / 18$. This equalizes marginal class predictions without distorting sample-level probability rankings.

---

## 7. Declarative Configuration Architecture

All ensemble weights, regularizers, prior scales, and thresholds are defined declaratively in [`configs/production_ensemble.json`](file:///c:/Projects/NameThatLLM-Anitgravity/configs/production_ensemble.json):

```json
{
  "main_stack": {
    "linear_svc_weight": 1.00,
    "members": {
      "lora_qwen3_14b": { "plain_weight": 0.40, "twin_weight": 0.10, "twin_clustering": "groups_085" },
      "lora_qwen25_14b": { "plain_weight": 0.20, "twin_weight": 0.10, "twin_clustering": "groups_085" },
      "deberta_v3_large": { "plain_weight": 0.25, "twin_weight": 0.00 }
    }
  },
  "complementary_stack": {
    "stream_weights": {
      "lora_and_twins": 0.35, "lexical": 0.20, "dp_posteriors": 0.30, "spectral_bank": 0.15
    },
    "dp_weights": {
      "q3_14b": 0.38, "q25_14b": 0.24, "deberta": 0.20, "instyle": 0.10, "part1_14b": 0.08, "claude_8b": 0.04
    },
    "prior_scales": {
      "qwen3_14b_p2": 1.00, "qwen25_14b_p2": 0.85, "deberta_p2": 0.80,
      "qwen3_14b_p1": 0.50, "instyle": 0.45, "qwen3_8b_claude": 0.30
    }
  },
  "fusion": { "main_ratio": 0.85, "comp_ratio": 0.15 },
  "postprocess": {
    "claude_specialist_weight": 0.15,
    "curly_quote_multiplier": 0.36787944117144233,
    "multi_em_dash_log_penalty": -0.45,
    "straight_contr_log_penalty": -0.60,
    "curly_contr_log_penalty": -0.40,
    "continuous_hw_weight": 0.04,
    "sinkhorn_iterations": 300,
    "max_cluster_size": 14
  }
}
```

---

## 8. Module 5: Two-Stage LoRA Training Engine (`trainer.py` & `lora_training.ipynb`)

Module 5 provides the end-to-end training and artifact generation pipeline used to build our in-house multi-layer LoRA checkpoints (such as `anilmittal/ntl-inhouse-two-stage-lora24`).

### 1. 2x T4 Multi-GPU Memory Balancing & Non-Reentrant Gradient Checkpointing
To train Qwen3-8B across Kaggle's dual T4 GPUs (15 GB VRAM each) without OOM or CUDA context crashes:
- Device placement is constrained explicitly: `max_memory={"0": "9.0GiB", "1": "9.0GiB"}` via HuggingFace `accelerate`.
- PyTorch gradient checkpointing is activated with `use_reentrant=False` on the causal transformer backbone, eliminating peak autograd cache spikes.

### 2. Stage 1: Sequence Classification Pre-Training
- Trained across 75,000 augmented passages (`configs/training_lora.json`).
- Optimizes Cross-Entropy Loss with AdamW ($lr = 10^{-4}, \text{weight\_decay} = 0.01$).
- Accumulates gradients over micro-batches (`batch_size = 4, gradient_accumulation_steps = 4`).

### 3. Stage 2: Prompt-Group Multi-Task Contrastive Fine-Tuning
In Stage 2, samples are grouped into 18-row prompt batches via `PromptGroupBatchSampler`. Each prompt group contains responses to the same topic across all 18 models:
- **Two-Pass Micro-Batched Execution (`train_prompt_group_step`)**:
  - **Pass 1 (`torch.no_grad()`)**: Iterates through the 18 rows in chunks of 4 to compute the exact prompt mean embedding $\mu_{\text{prompt}} = \frac{1}{18} \sum_{i=1}^{18} z_i$.
  - **Pass 2 (Autograd Micro-Batches)**: Evaluates loss over micro-batches of size 2 with dynamic sequence length slicing (`:max_l` across the micro-batch), dynamically centering features:
    $$z_{\text{centered}, i} = z_i - \mu_{\text{prompt}}$$
    Loss is a dual objective with label smoothing ($\epsilon = 0.10$):
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\text{plain}) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\text{centered})$$
  - Accumulates gradients directly into parameters with peak VRAM bounded to only 2 sequences.

### 4. Critical Training Bug Fixes & Protections
1. **Token-0 Padding Bug**: Qwen token ID `0` is `"!"` (exclamation mark), while `pad_token_id = 151643`. In `collate_text_batch`, `attention_mask` is initialized to 1 for all valid tokens, padded with 0, and `input_ids` padded with 151643. Token `0` (`"!"`) is never masked out.
2. **Recursive Final Norm Removal (`_disable_final_norm`)**: When wrapping models with PEFT, `norm` is nested under `base_model.model.norm`. `_disable_final_norm` recursively replaces all final RMSNorm modules with `nn.Identity()`, ensuring hidden states extracted at layer 24 during training match the un-normalized representations extracted during inference.
3. **Closed-Form Dual Ridge Regression Head Calibration (`fit_ridge_heads`)**: Before Stage 2 and immediately prior to exporting artifacts, closed-form $N \times N$ dual Ridge regression ($L_2$-regularized, $\lambda = 1.0$) with label smoothing $0.10$ solves optimal classification weights $W_p$ and $W_c$ directly on extracted training features, preventing overfitting to prompt topics.

---

## 9. Single-Source-of-Truth Compilation & Verification

The repository enforces a **Single Source of Truth** architecture:
- All model logic lives in `src/namethatllm/`.
- All hyperparameter values live in `configs/production_ensemble.json` and `configs/training_lora.json`.
- `scripts/compile_notebook.py` parses module ASTs, strips package imports, and bundles everything into two standalone notebooks:
  1. [`kaggle_submission.ipynb`](file:///c:/Projects/NameThatLLM-Anitgravity/kernels/submission/kaggle_submission.ipynb) (Inference pipeline)
  2. [`lora_training.ipynb`](file:///c:/Projects/NameThatLLM-Anitgravity/kernels/training/lora_training.ipynb) (Training pipeline)

### Running the 37 Unit Tests:
```powershell
python -m pytest tests -v
```

All 37 tests cover every component across 7 test suites:
1. `test_compiler_and_integration.py` (8 tests): AST parity, single-source bundling, multi-key checkpoint discovery, end-to-end smoke pipeline, full mock-neural pipeline, LayerNorm specifications.
2. `test_lexical.py` (5 tests): Shape patterns, function word preservation, punctuation retention, dual lexical outputs, continuous handwriting feature extraction.
3. `test_neural_extractor.py` (4 tests): Reversible LoRA unmerging, bfloat16 rescue, LayerNorm projection.
4. `test_wpca_dp.py` (4 tests): Covariance shrinkage floors, two-way prompt decomposition, safe cluster splitting ($s \le 14$), exact $2^{18}$ forward-backward bitmask DP.
5. `test_spectral_bank.py` (4 tests): Unsupervised $V_{\text{top64}}$ extraction (0 label leakage), prior scaling normalization, joint spectral bank execution.
6. `test_postprocess.py` (6 tests): Two-tier convex blend, mass-preserving Claude specialist gate, discrete orthographic rules, continuous handwriting integration, 300-step Sinkhorn mass equalization, submission schema.
7. `test_trainer.py` (6 tests): Prompt group batch sampler, loss formulation, dataset tokenization, token-0 attention mask preservation, PEFT final norm deactivation, two-pass micro-batched prompt group step.

---

## 10. Mathematical Reference Cheatsheet

```
1. LoRA Weight Fusion (GPU Device):
   W_effective = W_0 + (alpha / r) * (B @ A)

2. Attention-Sink Zeroed Mean Pooling:
   z_pooled = sum_{i=1}^{L-1} (h_i * m_i) / sum_{i=1}^{L-1} m_i    (m_0 = 0)

3. Prompt-Centered Feature Normalization:
   z_{i, centered} = z_i - (1 / |C|) * sum_{j in C} z_j            (for 2 <= |C| <= 18)

4. Snapshot-Isolated Reversible Trunk Weight Restoration:
   backup = trunk.snapshot_weights(adapter_path)
   merge_lora(trunk.model, adapter_path, sign=+1.0)
   trunk.restore_weights(backup)                                    (Bit-exact zero drift)

5. Two-Pass Micro-Batched Prompt Group Step:
   Pass 1 (no_grad):  mu_prompt = (1 / 18) * sum_{j=1}^{18} z_j
   Pass 2 (autograd): z_{c, i} = z_i - mu_prompt;  L = L_plain + 0.5 * L_centered

6. Closed-Form Dual Ridge Regression (N x N Gram Solver):
   W = (Z^T @ Z + lambda * I)^{-1} @ Z^T @ Y_smoothed
     = Z^T @ (Z @ Z^T + lambda * I)^{-1} @ Y_smoothed

7. Two-Tier Convex Mixture (85/15):
   P_mix = 0.85 * R_main + 0.15 * P_comp

8. Vectorized Claude Specialist Gate (Mass-Preserving):
   P_claude = (1 - w_gate) * P_prior + w_gate * (mass * P_specialist)

9. Unified Handwriting & Orthographic Shield:
   log P_out = softmax(log P_mix + B_disc @ W_pen + w_hw * log P_hw)

10. Exact Multiplicative Sinkhorn-Knopp Equalization (300 Steps):
    P^{(t+1)} = diag(u^{(t)}) @ P^{(t)} @ diag(v^{(t)})             (column sums = N / 18)
```
