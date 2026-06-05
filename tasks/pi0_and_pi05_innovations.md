# π₀ and π₀.₅: Comprehensive Innovation Report

> **Purpose**: This report documents every innovation in the π₀ and π₀.₅ papers for guiding engineering source code understanding and reproduction. Generated from thorough reading of the papers, LaTeX source in `arXiv-2504.16054v1/`, and the open-source codebase at [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [π₀: Vision-Language-Action Flow Model](#2-π₀-vision-language-action-flow-model)
3. [π₀.₅: Co-Trained VLA with Open-World Generalization](#3-π₀₅-co-trained-vla-with-open-world-generalization)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Training Methodology](#5-training-methodology)
6. [Codebase Mapping](#6-codebase-mapping)
7. [Key Dependencies and References](#7-key-dependencies-and-references)
8. [Reproduction Guide](#8-reproduction-guide)

---

## 1. Background and Motivation

### 1.1 Problem Statement

Both papers address the core challenge of *open-world generalization* for robotic manipulation. Prior robot learning systems either:
- Work well only in environments closely matching training data (narrow generalization), or
- Achieve broad generalization only for simple primitives (grasping, navigation)

The π₀ and π₀.₅ papers argue that generalizable robotic systems must transfer knowledge from **heterogeneous information sources**, much like humans draw on diverse experience (firsthand, instructed, read, observed).

### 1.2 VLA Paradigm

Both models follow the **Vision-Language-Action (VLA)** paradigm, which casts robot control as a sequence modeling problem:

$$\max_\theta \mathbb{E}_{(\mathbf{a}_{t:t+H}, \mathbf{o}_t, \ell) \sim \mathcal{D}} \log \pi_\theta(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t, \ell)$$

where:
- $\mathbf{o}_t = [\mathbf{I}^1_t, ..., \mathbf{I}^n_t, \mathbf{q}_t]$ = multimodal observation (images + proprioception)
- $\ell$ = natural language task instruction
- $\mathbf{a}_{t:t+H}$ = action chunk (horizon $H$)

---

## 2. π₀: Vision-Language-Action Flow Model

**Paper**: Black et al., "π₀: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024.

### 2.1 Innovation 1: Flow Matching for Continuous Action Generation

**What**: π₀ is the first VLA to use **flow matching** (Lipman et al., 2022) rather than diffusion to represent the action distribution.

**Details**:
- Given action chunk $\mathbf{a}_{t:t+H}$, noise $\omega \sim \mathcal{N}(0, \mathbf{I})$, and flow time $\tau \in [0, 1]$
- Interpolated action: $\mathbf{a}^{\tau, \omega}_{t:t+H} = \tau \mathbf{a}_{t:t+H} + (1-\tau)\omega$
- Model predicts the vector field: $\mathbf{u}_t = \omega - \mathbf{a}_{t:t+H}$
- Loss: $||\omega - \mathbf{a}_{t:t+H} - f^a_\theta(\mathbf{a}^{\tau, \omega}_{t:t+H}, \mathbf{o}_t, \ell)||^2$
- At inference: 10 denoising steps via Euler integration, starting from $\tau=1$ (noise) to $\tau=0$ (action)

**Key detail — non-uniform timestep sampling**: π₀ uses a timestep distribution $p(\tau) = \text{Beta}(\frac{s-\tau}{s}; \alpha=1.5, \beta=1)$ where $s = 0.999$. This emphasizes **low timesteps** (near denoised), departing from standard uniform sampling. This focuses training on the fine-grained denoising regime that matters most. Timesteps above $s$ are excluded since the integration step $\delta > 1-s$.

**Code location**: [pi0.py:217-279](/src/openpi/models/pi0.py#L217-L279) — `sample_actions` method, [pi0.py:197](/src/openpi/models/pi0.py#L197) — timestep sampling in `compute_loss`.

### 2.2 Innovation 2: Action Expert (Mixture of Experts Architecture)

**What**: π₀ introduces an **action expert** — a separate, smaller transformer that processes action tokens, analogous to a Mixture of Experts (MoE) design.

**Architecture**:
- **VLM Backbone** (PaliGemma): 2B parameters, width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256
- **Action Expert**: 300M parameters, width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
- Both share the same number of transformer layers (depth=18), interacting through **self-attention only**
- VLM backbone never attends to action expert tokens; information flows **unidirectionally** from VLM → Action Expert

**How it works**:
- The VLM backbone processes image patches, language tokens, and (in π₀) a continuous state token
- The Action Expert processes noisy action tokens concatenated with timestep embeddings
- Self-attention layers see both expert outputs as a single sequence but use separate weights per token type

**Weight naming convention**: First expert weights have no suffix (e.g., `"attn"`); second expert weights use suffix `"_1"` (e.g., `"attn_1"`). This allows loading pretrained PaliGemma weights seamlessly.

**Code location**: [gemma.py:340-411](/src/openpi/models/gemma.py#L340-L411) — `Module` with multi-config support; [pi0.py:67-101](/src/openpi/models/pi0.py#L67-L101) — `Pi0.__init__`.

### 2.3 Innovation 3: PaliGemma VLM Backbone + SigLIP Vision Encoder

**What**: π₀ initializes from **PaliGemma** (Beyer et al., 2024), a pretrained vision-language model, inheriting web-scale visual and language understanding.

**Vision Encoder**: SigLIP ViT (So400m/14 variant)
- Width: 1152, Depth: 27, Patch size: 14×14
- Pool type: `"none"` (outputs patch-level features, no global pooling)
- Image resolution: 224×224, producing 16×16 = 256 visual tokens per image
- 3 camera views by default: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`

**Code location**: [siglip.py:188-290](/src/openpi/models/siglip.py#L188-L290) — SigLIP `_Module`; [pi0.py:81-90](/src/openpi/models/pi0.py#L81-L90) — vision encoder initialization.

### 2.4 Innovation 4: Proprioceptive State as Continuous Token

**What**: In π₀, the robot's proprioceptive state $\mathbf{q}_t$ (joint angles, gripper positions, etc.) is treated as a **continuous token** fed to the action expert, NOT discretized into text.

**Details**:
- State vector is linearly projected to the action expert embedding dimension
- Added as a single token that precedes the noisy action tokens in the suffix
- Uses `ar_mask = True` for the state token (previous tokens don't attend to it)

**Code location**: [pi0.py:151-153](/src/openpi/models/pi0.py#L151-L153) — `state_proj` and state token insertion in `embed_suffix`.

### 2.5 Innovation 5: Timestep Conditioning via Concatenation (π₀)

**What**: In π₀, the flow matching timestep is injected by:
1. Converting $\tau$ to a sinusoidal positional encoding
2. Repeating it across the action horizon
3. Concatenating with action tokens: `[action_tokens, time_tokens]`
4. Processing through a 2-layer MLP with Swish activation

**Code location**: [pi0.py:97-99, 171-178](/src/openpi/models/pi0.py#L97-L99) and [pi0.py:171-178]. `action_time_mlp_in` and `action_time_mlp_out`.

### 2.6 Innovation 6: Prefix-LM Attention Masking

**What**: π₀ uses a **prefix-lm attention pattern**, not pure causal attention:
- Image patches: bidirectional attention within themselves
- Language tokens: bidirectional attention with images AND between all prompt tokens
- State token: prefix tokens cannot attend to it, it can attend to all prefix
- Action tokens: auto-regressive among themselves, cannot be attended by prefix tokens

**Implementation**: The `make_attn_mask` function (shared by π₀ and π₀.₅) implements this via a cumulative AR mask trick:
- Tokens with same cumulative `mask_ar` value can attend to each other
- Tokens with lower cumulative value can be attended by higher ones

**Code location**: [pi0.py:19-44](/src/openpi/models/pi0.py#L19-L44) — `make_attn_mask`.

---

## 3. π₀.₅: Co-Trained VLA with Open-World Generalization

**Paper**: Physical Intelligence, "π₀.₅: a Vision-Language-Action Model with Open-World Generalization," arXiv:2504.16054, 2025.

The π₀.₅ model builds on π₀ with the following key innovations for open-world generalization:

### 3.1 Innovation 7: Heterogeneous Co-Training Framework

**What**: π₀.₅ trains on 5 distinct data categories, with the model's task being to predict BOTH text AND actions from the same architecture:

$$\pi_\theta(\mathbf{a}_{t:t+H}, \hat{\ell} \mid \mathbf{o}_t, \ell) = \pi_\theta(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t, \hat{\ell})\,\pi_\theta(\hat{\ell} \mid \mathbf{o}_t, \ell)$$

This factorization means the high-level inference captures $\pi_\theta(\hat{\ell} \mid \mathbf{o}_t, \ell)$ and low-level captures $\pi_\theta(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t, \hat{\ell})$, with **both distributions represented by the SAME model**.

**Data Categories**:

| Code | Name | Description | ~% of Pre-training |
|------|------|-------------|-------------------|
| **MM** | Mobile Manipulator | 400 hours, ~100 homes, target platform data. Bimanual mobile manipulators for household tasks. | ~2.4% |
| **ME** | Multi-Environment | Single/bimanual non-mobile arms in diverse real homes. Easier to transport → wider home coverage. | ~majority of robot data |
| **CE** | Cross-Embodiment Lab | Diverse lab tasks (bussing tables, folding shirts, coffee grinding). Multiple robot types + OXE dataset. | large portion |
| **HL** | High-Level Subtask | Manual annotations of subtask labels + bounding boxes. Trains model to jointly predict subtask → actions. | moderate |
| **WD** | Web Data | Image captioning (CapsFusion, COCO), QA (Cambrian-7M, PixMo, VQAv2), object localization with indoor emphasis. | large portion |

**Key insight**: 97.6% of pre-training examples do NOT come from the target mobile manipulator. The model transfers knowledge across embodiments, tasks, and modalities.

### 3.2 Innovation 8: Hierarchical Inference (Chain-of-Thought for Robots)

**What**: π₀.₅ introduces a two-level inference procedure at runtime using the **same unified model**:

1. **High-Level Inference** (~1 Hz): Given observation $\mathbf{o}_t$ and task $\ell$, predict subtask $\hat{\ell}$ (e.g., "pick up the plate")
2. **Low-Level Inference** (50 Hz): Given $\mathbf{o}_t$ and predicted subtask $\hat{\ell}$, predict action chunk $\mathbf{a}_{t:t+H}$ via flow matching

This is analogous to chain-of-thought reasoning (Wei et al., 2022) but the high-level inference runs at a **lower frequency** than low-level action inference.

**Critical design choice**: Unlike prior work that uses separate VLM + policy models (SayCan, Yell At Your Robot, NaVILA, etc.), π₀.₅ uses the **exact same model weights** for both levels. This is a form of **test-time compute** applied to robotics.

**Model output structure**: The output of the transformer $f$ is split into:
- $y^\ell_{1:M}$: text token logits (for sampling $\hat{\ell}$)
- $y^a_{1:H}$: action output tokens from the action expert

where $M + H \le N$ (not all outputs associated with a loss).

### 3.3 Innovation 9: Combined Discrete & Continuous Action Representations

**What**: π₀.₅ combines two action representation paradigms in a single model:

**Phase 1 — Pre-training (Discrete)**:
- Actions encoded as discrete tokens using the FAST tokenizer (Pertsch et al., 2025)
- Trained as standard autoregressive next-token prediction: $\alpha = 0$ in the combined loss
- Benefits: fast, scalable, stable training; excellent language following

**Phase 2 — Post-training (Continuous)**:
- Action expert added with flow matching objective: $\alpha = 10.0$
- 10 denoising steps at inference for efficient real-time control
- Benefits: fine-grained continuous action representation, non-autoregressive decoding

**Combined Loss** (post-training):

$$\mathbb{E}_{\mathcal{D}, \tau, \omega} \left[ H(x_{1:M}, f^\ell_\theta(\mathbf{o}_t, \ell)) + \alpha \left\| \omega - \mathbf{a}_{t:t+H} - f^a_\theta(\mathbf{a}^{\tau, \omega}_{t:t+H}, \mathbf{o}_t, \ell) \right\|^2 \right]$$

**Attention masking between representations**: Discrete (FAST) action tokens and continuous (flow matching) action tokens do NOT attend to each other via the attention mask, preventing information leakage.

### 3.4 Innovation 10: Two-Stage Training Pipeline

**Stage 1 — Pre-training (280k gradient steps)**:
- All 5 data categories (MM, ME, CE, HL, WD)
- Pure autoregressive next-token prediction (no action expert)
- Actions use FAST tokenizer (discrete)
- $\alpha = 0$ (no flow matching)

**Stage 2 — Post-training (80k additional steps)**:
- Data: MM + ME + WD + HL + VI (drops CE to focus on mobile manipulation)
- Action expert initialized from scratch (random weights)
- Joint loss with $\alpha = 10.0$
- Dataset filtered to successful episodes below length threshold
- Only ME/MM robot data used for flow matching loss

**Training details**:
- Image augmentation: RandomCrop(95%), Resize, Rotate(±5°), ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
- Wrist camera images skip spatial augmentations (only color jitter)
- Action normalization to [-1, 1] using 1% and 99% quantiles
- Fixed action dimension to accommodate largest action space; zero-padding for smaller robots

### 3.5 Innovation 11: AdaRMS Norm for Timestep Injection (π₀.₅)

**What**: π₀.₅ replaces π₀'s timestep concatenation MLP with **adaptive RMSNorm** (adaRMS), a more efficient conditioning mechanism.

**Mechanism**:
1. Timestep $\tau$ → sinusoidal positional encoding → **time MLP** (2-layer with Swish) → conditioning vector
2. Each transformer layer's RMSNorm applies the conditioning:
   - `modulation = Dense(3 * width)(cond)`
   - `scale, shift, gate = split(modulation)`
   - `normed = normed * (1 + scale) + shift`
   - Return: `(normed, gate)` where gate modulates residual connection

**Code location**: [gemma.py:113-131](/src/openpi/models/gemma.py#L113-L131) — `RMSNorm` with adaptive conditioning; [pi0.py:93-95](/src/openpi/models/pi0.py#L93-L95) — time MLP for adaRMS; [pi0.py:163-168](/src/openpi/models/pi0.py#L163-L168) — time embedding computation for π₀.₅.

**Difference from π₀**:

| Aspect | π₀ | π₀.₅ |
|--------|-----|-------|
| Timestep injection | Concat with action + 2-layer MLP | adaRMS in every transformer layer |
| State input | Continuous token in action suffix | Discretized into language tokens |
| Prop. representation | `state_proj` Linear → action expert | Part of text token sequence |

Code signal for π₀ vs π₀.₅: The `pi05: bool` flag in [Pi0Config](/src/openpi/models/pi0_config.py#L31) controls these architectural differences.

### 3.6 Innovation 12: Verbal Instruction Data (VI)

**What**: A novel supervision modality where human experts provide **language demonstrations** — selecting appropriate sub-task commands to guide the robot through complex tasks in real-time.

**Details**:
- Created by expert users "teleoperating" the robot with language
- Expert provides sequential high-level commands to a trained low-level policy
- These demonstrations teach the model good high-level planning strategies
- Only ~11% of HL examples but **critical** for performance

**Ablation result**: Removing VI causes significant performance degradation. The `no VI` ablation was significantly weaker than the full model.

### 3.7 Innovation 13: Bounding Box Prediction in HL Data

**What**: For the High-Level subtask prediction data, π₀.₅ also predicts bounding boxes of relevant objects alongside subtask labels.

**Rationale**: Predicting object locations before the subtask helps the model ground its semantic understanding in the visual scene, improving spatial reasoning.

### 3.8 Innovation 14: Data Mixture Design Principles

π₀.₅'s ablation experiments reveal key design principles:

1. **Cross-embodiment transfer is critical**: Removing either ME or CE data causes significant degradation. Removing both is worse.

2. **Web data matters for OOD generalization**: The `no WD` ablation shows significant degradation on out-of-distribution object categories (language following) and high-level inference, but not on in-distribution tasks. Web data provides broad knowledge of physical objects.

3. **Scaling environments improves generalization**: Performance on mock home tasks improves steadily from 3→104 training locations. With 104 locations, the model reaches **similar performance to a model trained directly on the test home**.

4. **High-level data benefits even without explicit HL inference**: The "implicit HL" ablation (HL data in training, no HL at inference) is the second-best model, suggesting the co-training itself teaches implicit planning capabilities.

5. **In-domain HL training > GPT-4 zero-shot**: GPT-4 as high-level policy (prompted with task description + label list) performs worse than the full model, demonstrating the importance of adapting VLMs with robot-specific data.

---

## 4. Architecture Deep Dive

### 4.1 Full Model Architecture

```
Input Images (3 cameras × 224×224×3)
        │
        ▼
┌──────────────────────┐
│  SigLIP ViT (So400m) │  ← Patch embedding (14×14) → 256 tokens/img × 3 = 768 tokens
│  pool_type="none"     │    Output: per-patch features, not pooled
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│              PaliGemma Transformer (2B)                    │
│  width=2048, depth=18, mlp_dim=16384, num_heads=8         │
│  num_kv_heads=1 (MQA), head_dim=256                       │
│                                                           │
│  ┌─────────────┐    ┌─────────────┐                      │
│  │ Expert 0    │    │ Expert 1    │                      │
│  │ (PaliGemma) │◄──►│ (Action)    │  ← Self-attention    │
│  │ 2B params   │    │ 300M params │     only interaction  │
│  └──────┬──────┘    └──────┬──────┘                      │
│         │                  │                              │
│    Text logits        Action tokens                       │
│    (subtask pred)     (flow matching field)               │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Token Types and Processing

| Token Type | Processing | Attention | Used By |
|-----------|-----------|-----------|---------|
| Image patch ($x_i^I$) | SigLIP ViT → projection | Bidirectional within images + prompt | Expert 0 |
| Text ($x_i^w$) | Embedding matrix (257k vocab) | Bidirectional with images + within prompt | Expert 0 |
| Proprioception (π₀.₅) | Discretized → text tokens | Same as text | Expert 0 |
| Proprioception (π₀) | Linear projection → continuous token | Prefix can't attend; attends to all prefix | Expert 1 |
| FAST action tokens | Discrete tokens (FSQ) | Autoregressive; prefix can't attend | Expert 0 |
| Flow matching action tokens ($x_i^a$) | Linear projection | Bidirectional among themselves; prefix can't attend | Expert 1 |

### 4.3 Attention Mask Pattern

```
                    Image  Prompt  FAST_actions  FM_actions
          Image      1       1        1            0
          Prompt     1       1        1            0
          FAST_act   1       1       AR            0
          FM_act     1       1        0            1
```

Where `1` = can attend, `0` = cannot attend, `AR` = autoregressive (causal).

Key: VLM tokens **never** attend to action tokens. FAST and FM action tokens **never** attend to each other.

### 4.4 Action Space

- Action dimension: 32 (padded to accommodate largest robot)
- Action horizon: 50 steps (π₀), 32 steps (π₀-FAST)
- Action representation: target joint/end-effector poses, gripper commands, base velocity, torso lift
- Control frequency: 50 Hz via PD controllers tracking model outputs
- 18-19 DoF depending on robot platform (2×6-DoF arms + 2 grippers + 3 DoF base + 1-2 DoF torso)

### 4.5 Model Variants in Codebase

| ModelType | Class | Tokenizer | Action Decoding | HL Inference |
|-----------|-------|-----------|-----------------|--------------|
| `PI0` | `Pi0` | — (continuous state) | Flow matching | No |
| `PI0_FAST` | `Pi0FAST` | FSQ discrete tokens | Autoregressive | No |
| `PI05` | `Pi0(pi05=True)` | Discretized state | Flow matching | Yes |

---

## 5. Training Methodology

### 5.1 π₀ Training

1. Initialize PaliGemma weights from pretrained VLM checkpoint
2. Initialize action expert weights from scratch
3. Train on large robot dataset (10k+ hours across embodiments) with flow matching objective
4. Image augmentation: random crop, resize, rotation, color jitter
5. Non-uniform timestep sampling (Beta distribution)

### 5.2 π₀.₅ Training (Full Recipe)

**Pre-training (280k steps)**:
- All data sources (MM, ME, CE, HL, WD)
- Actions as FAST discrete tokens
- Standard autoregressive cross-entropy loss
- No action expert (no flow matching)

**Post-training (80k steps)**:
- Data: MM + ME + WD + HL + VI
- Action expert initialized randomly
- Joint loss: cross-entropy (text + FAST tokens) + α · flow matching MSE
- α = 10.0
- Filtered to successful episodes, below length threshold
- Image augmentation preserved

### 5.3 Fine-Tuning (openpi Codebase)

The codebase provides three fine-tuning modes:
- **Full fine-tuning**: All weights trainable, requires >70GB GPU memory (A100/H100)
- **LoRA fine-tuning**: Low-rank adapters on attention + FFN, requires >22.5GB (RTX 4090)
- **PyTorch fine-tuning**: Newer option, supports DDP multi-GPU

Checkpoints provided:
- `pi0_base`: π₀ base model
- `pi0_fast_base`: π₀-FAST base model  
- `pi05_base`: π₀.₅ base model
- Expert fine-tuned models: `pi0_fast_droid`, `pi0_droid`, `pi0_aloha_*`, `pi05_libero`, `pi05_droid`

---

## 6. Codebase Mapping

### 6.1 Core Model Files

| File | Purpose |
|------|---------|
| [src/openpi/models/pi0.py](/src/openpi/models/pi0.py) | π₀ and π₀.₅ JAX implementation (`Pi0` class). Single file covers both via `pi05` flag. |
| [src/openpi/models/pi0_config.py](/src/openpi/models/pi0_config.py) | Configuration dataclass. Key flag: `pi05: bool`. Controls state input mode, adaRMS, max_token_len. |
| [src/openpi/models/pi0_fast.py](/src/openpi/models/pi0_fast.py) | π₀-FAST autoregressive model (`Pi0FAST` class). Uses discrete action tokens. |
| [src/openpi/models/gemma.py](/src/openpi/models/gemma.py) | Gemma transformer with multi-expert support. Implements RMSNorm (regular + adaRMS), Attention with RoPE, MoE-style blocks. |
| [src/openpi/models/siglip.py](/src/openpi/models/siglip.py) | SigLIP ViT vision encoder. So400m/14 variant, pool_type="none". |
| [src/openpi/models/model.py](/src/openpi/models/model.py) | Base classes: `BaseModel`, `BaseModelConfig`, `Observation`, `Actions`. Data format specification. |
| [src/openpi/models/model_test.py](/src/openpi/models/model_test.py) | Model tests including loss and inference shape checks. |
| [src/openpi/models/tokenizer.py](/src/openpi/models/tokenizer.py) | Text tokenizer (PaliGemma sentencepiece-based). |
| [src/openpi/models/utils/fsq_tokenizer.py](/src/openpi/models/utils/fsq_tokenizer.py) | FSQ (Finite Scalar Quantization) action tokenizer. Encoder-decoder with cross-attention. |
| [src/openpi/models/lora.py](/src/openpi/models/lora.py) | LoRA adapters for efficient fine-tuning. |
| [src/openpi/models_pytorch/pi0_pytorch.py](/src/openpi/models_pytorch/pi0_pytorch.py) | PyTorch reimplementation of π₀/π₀.₅. |
| [src/openpi/models_pytorch/gemma_pytorch.py](/src/openpi/models_pytorch/gemma_pytorch.py) | PyTorch Gemma with multi-expert support. |

### 6.2 Training Files

| File | Purpose |
|------|---------|
| [src/openpi/training/config.py](/src/openpi/training/config.py) | Training configurations for all model variants + fine-tuning recipes. |
| [src/openpi/training/data_loader.py](/src/openpi/training/data_loader.py) | LeRobot dataset loader and data pipeline. |
| [src/openpi/training/checkpoints.py](/src/openpi/training/checkpoints.py) | Orbax checkpoint save/restore. |
| [src/openpi/training/weight_loaders.py](/src/openpi/training/weight_loaders.py) | Weight initialization from pretrained checkpoints. |
| [src/openpi/training/optimizer.py](/src/openpi/training/optimizer.py) | Optimizer configuration. |
| [src/openpi/training/sharding.py](/src/openpi/training/sharding.py) | FSDP sharding utilities. |
| [src/openpi/shared/normalize.py](/src/openpi/shared/normalize.py) | Action/state normalization (quantile-based). |

### 6.3 Policy/Deployment Files

| File | Purpose |
|------|---------|
| [src/openpi/policies/policy.py](/src/openpi/policies/policy.py) | Policy base class. |
| [src/openpi/policies/policy_config.py](/src/openpi/policies/policy_config.py) | Policy creation from config + checkpoint. |
| [src/openpi/serving/websocket_policy_server.py](/src/openpi/serving/websocket_policy_server.py) | Remote inference server. |

### 6.4 Key Architectural Differences: π₀ vs π₀.₅ in Code

In [pi0.py](/src/openpi/models/pi0.py), the `pi05` flag controls:

```python
# π₀: continuous state token in action expert suffix
if not self.pi05:
    state_token = self.state_proj(obs.state)[:, None, :]  # → action expert
    # timestep via concat + MLP
    action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
    action_time_tokens = self.action_time_mlp_in(action_time_tokens)

# π₀.₅: state as discrete language tokens, adaRMS norm
if self.pi05:
    # state is part of text tokens (not here)
    # timestep via time MLP → adaRMS conditioning
    time_emb = self.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = self.time_mlp_out(time_emb)
    adarms_cond = time_emb  # passed to each transformer layer
```

---

## 7. Key Dependencies and References

### 7.1 Foundational Methods

| Method | Paper | Role |
|--------|-------|------|
| **Flow Matching** | Lipman et al., "Flow Matching for Generative Modeling," arXiv:2210.02747, 2022 | Action generation framework |
| **FAST Tokenizer** | Pertsch et al., "FAST: Efficient Action Tokenization for Vision-Language-Action Models," RSS 2025 | Discrete action tokenization |
| **PaliGemma** | Beyer et al., "PaliGemma: A Versatile 3B VLM for Transfer," arXiv:2407.07726, 2024 | VLM backbone |
| **SigLIP** | Zhai et al., "Sigmoid Loss for Language Image Pre-Training," ICCV 2023 | Vision encoder |
| **Gemma** | Google DeepMind, 2024 | Transformer architecture with MQA, GeGLU, RMSNorm |
| **RT-2** | Zitkovich et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," CoRL 2023 | VLA paradigm |
| **Octo** | Octo Model Team, RSS 2024 | Open-source generalist robot policy |
| **OpenVLA** | Kim et al., arXiv:2406.09246, 2024 | Open-source VLA |
| **RT-1** | Brohan et al., arXiv:2212.06817, 2022 | Robotics Transformer |
| **PaLM-E** | Driess et al., arXiv:2303.03378, 2023 | Embodied multimodal LM |

### 7.2 Related Robot Learning Methods

| Method | Paper | Relationship |
|--------|-------|-------------|
| **SayCan** | Ahn et al., arXiv:2204.01691, 2022 | Separate high-level (LLM) + low-level policy |
| **RT-H** | Belkhale et al., arXiv:2403.01823, 2024 | Action hierarchies with language |
| **Embodied Chain-of-Thought** | Zawalski et al., CoRL 2024 | Chain-of-thought for robot control |
| **Hi Robot** | Shi et al., arXiv:2502.19417, 2025 | Hierarchical VLA from same group |
| **DROID** | Khazatsky et al., RSS 2024 | Large-scale manipulation dataset |
| **Open X-Embodiment** | Collaboration et al., 2023 | Cross-embodiment dataset consortium |
| **RDT-1B** | Liu et al., arXiv:2410.07864, 2024 | Diffusion-based bimanual VLA |
| **GR00T N1** | Bjorck et al., arXiv:2503.14734, 2025 | Generalist humanoid foundation model |
| **Gemini Robotics** | Team et al., arXiv:2503.20020, 2025 | Google's VLA system |
| **CoT-VLA** | Zhao et al., CVPR 2025 | Visual chain-of-thought for VLAs |

### 7.3 Vision-Language Methods

| Method | Paper | Role |
|--------|-------|------|
| **Chain-of-Thought** | Wei et al., NeurIPS 2022 | Inspiration for hierarchical inference |
| **CLIP** | Radford et al., ICML 2021 | Vision-language pretraining |
| **Segment Anything** | Kirillov et al., 2023 | Visual grounding |
| **LLaMA** | Touvron et al., 2023 | LLM architecture reference |

---

## 8. Reproduction Guide

### 8.1 Key Hyperparameters

| Parameter | π₀ | π₀.₅ |
|-----------|-----|-------|
| Image resolution | 224×224 | 224×224 |
| Action horizon | 50 | 50 |
| Action dim | 32 | 32 |
| Max token len | 48 | 200 |
| Pre-training steps | N/A (single-stage) | 280k |
| Post-training steps | — | 80k |
| Flow matching steps | 10 | 10 |
| Timestep distribution | Beta(1.5, 1), s=0.999 | Same |
| α (flow matching weight) | 1.0 (implicit) | 10.0 |
| Precision | bfloat16 | bfloat16 |
| Vision backbone | SigLIP So400m/14 | Same |
| VLM backbone | PaliGemma 2B | Same |
| Action expert | 300M params | Same |

### 8.2 Data Requirements for Reproduction

To reproduce π₀.₅'s co-training, one would need:

1. **Mobile Manipulator data (MM)**: ~400 hours across ~100 distinct home environments, bimanual mobile manipulators
2. **Multi-Environment data (ME)**: Static arm data in diverse homes (more homes than MM)
3. **Cross-Embodiment data (CE)**: Lab tasks across robot types + OXE open-source dataset
4. **High-Level annotations (HL)**: Per-episode subtask labels + bounding boxes
5. **Web data (WD)**: Image captioning (CapsFusion, COCO), VQA (VQAv2, Cambrian-7M, PixMo), object localization
6. **Verbal Instructions (VI)**: Language demonstrations from expert supervisors

### 8.3 Minimal Fine-Tuning Path

The openpi repo supports fine-tuning base models on custom data:
1. Convert data to LeRobot format
2. Compute normalization statistics: `uv run scripts/compute_norm_stats.py`
3. Train: `uv run scripts/train.py <config_name>`
4. Serve: `uv run scripts/serve_policy.py policy:checkpoint`

### 8.4 Known Limitations (from paper)

- Model makes mistakes on unfamiliar hardware (handles, cabinets)
- Partial observability challenges (arm occluding spills)
- High-level inference can be distracted (repeatedly opening/closing drawers)
- Prompt complexity limited by training data annotations
- Relatively modest context window (no cross-room memory)
- Specific co-training data combination explored; broader data types remain future work

---

*Report generated from: π₀ paper (Black et al., arXiv:2410.24164, 2024), π₀.₅ paper (Physical Intelligence, arXiv:2504.16054, 2025), and the openpi open-source codebase.*
