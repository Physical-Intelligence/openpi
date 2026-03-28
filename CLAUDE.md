# OpenPI - Pi0 / Pi0.5 VLA Model Codebase

> Forked from Physical Intelligence's [openpi](https://github.com/Physical-Intelligence/openpi).
> Paper: *pi0.5: a Vision-Language-Action Model with Open-World Generalization* (arXiv:2504.16054)

## Quick Reference

```bash
# Environment setup (requires Python 3.11+, uv package manager)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# JAX training
uv run scripts/train.py <config_name> --exp_name <name>

# PyTorch training (single GPU)
uv run scripts/train_pytorch.py <config_name> --exp_name <name>

# PyTorch training (multi-GPU)
torchrun --nproc_per_node=N scripts/train_pytorch.py <config_name> --exp_name <name>

# Serve policy (WebSocket server)
uv run scripts/serve_policy.py --env <ENV_NAME> --config <config_name>

# Compute normalization stats
uv run scripts/compute_norm_stats.py --config <config_name>

# Convert JAX checkpoint to PyTorch
uv run examples/convert_jax_model_to_pytorch.py --checkpoint_dir <path> --output_path <path>

# Run tests
uv run pytest
```

## Project Structure

```
openpi/
├── src/openpi/                      # Core library
│   ├── models/                      # JAX/Flax model implementations
│   │   ├── pi0.py                   # Pi0/Pi0.5 model (flow matching VLA)
│   │   ├── pi0_config.py            # Model config (Pi0Config, pi05=True/False)
│   │   ├── pi0_fast.py              # Pi0-FAST (autoregressive action tokenizer variant)
│   │   ├── gemma.py                 # Gemma LLM backbone (big_vision style)
│   │   ├── siglip.py                # SigLIP vision encoder
│   │   ├── model.py                 # BaseModel, Observation, Actions, IMAGE_KEYS
│   │   ├── tokenizer.py             # PaliGemma tokenizer + state discretization
│   │   ├── lora.py                  # LoRA adapter
│   │   └── utils/fsq_tokenizer.py   # FAST action tokenizer (FSQ)
│   │
│   ├── models_pytorch/              # PyTorch model implementations (mirrors JAX)
│   │   ├── pi0_pytorch.py           # PI0Pytorch - main PyTorch model
│   │   ├── gemma_pytorch.py         # PaliGemmaWithExpertModel (HF-based)
│   │   ├── preprocessing_pytorch.py # Image preprocessing
│   │   └── transformers_replace/    # Modified HuggingFace modules (adaRMS support)
│   │
│   ├── policies/                    # Inference wrappers (model + transforms)
│   │   ├── policy.py                # Policy class - main inference interface
│   │   ├── policy_config.py         # create_trained_policy() factory
│   │   ├── aloha_policy.py          # ALOHA robot transforms (AlohaInputs/Outputs)
│   │   ├── droid_policy.py          # DROID robot transforms (DroidInputs/Outputs)
│   │   └── libero_policy.py         # LIBERO benchmark transforms
│   │
│   ├── training/                    # Training infrastructure
│   │   ├── config.py                # TrainConfig, DataConfig, named configs (_CONFIGS)
│   │   ├── data_loader.py           # Dataset/DataLoader protocols + transforms
│   │   ├── checkpoints.py           # Orbax checkpoint management
│   │   ├── optimizer.py             # AdamW/SGD with LR schedules (optax)
│   │   ├── weight_loaders.py        # Checkpoint/PaliGemma weight loading
│   │   ├── sharding.py              # FSDP sharding for JAX
│   │   └── droid_rlds_dataset.py    # DROID RLDS format data loading
│   │
│   ├── serving/
│   │   └── websocket_policy_server.py  # Async WebSocket server for remote inference
│   │
│   ├── shared/                      # Shared utilities
│   │   ├── normalize.py             # NormStats load/save
│   │   ├── download.py              # Asset downloading (GCS)
│   │   └── image_tools.py           # Image processing
│   │
│   └── transforms.py               # Transform pipeline (repack, normalize, tokenize, delta actions, etc.)
│
├── scripts/
│   ├── train.py                     # JAX training entrypoint
│   ├── train_pytorch.py             # PyTorch training entrypoint (DDP support)
│   ├── serve_policy.py              # Policy server entrypoint
│   └── compute_norm_stats.py        # Norm stats computation
│
├── packages/openpi-client/          # Standalone client library (minimal deps)
│   └── src/openpi_client/
│       ├── websocket_client_policy.py  # WebSocket client for remote inference
│       ├── base_policy.py              # BasePolicy interface
│       ├── runtime/                    # Episode loop framework (Runtime, Environment, Agent)
│       └── action_chunk_broker.py      # Action sequence buffering
│
├── examples/
│   ├── aloha_sim/                   # ALOHA simulation example
│   ├── aloha_real/                  # ALOHA real robot example
│   ├── libero/                      # LIBERO benchmark example
│   ├── droid/                       # DROID robot example + data conversion
│   ├── simple_client/               # Minimal client-server example
│   └── convert_jax_model_to_pytorch.py  # JAX→PyTorch weight conversion
│
├── convert.py                       # HDF5 demo data → LeRobot format
├── simple_pytorch_train.py          # Minimal standalone PyTorch MLP training
└── pi0.5vla.pdf                     # Pi0.5 paper
```

## Architecture Overview

### Three Model Variants

| Model | Type | Action Representation | Config Flag |
|-------|------|-----------------------|-------------|
| **Pi0** | Flow matching | Continuous (Euler ODE) | `pi05=False` |
| **Pi0.5** | Flow matching + co-training | Continuous (Euler ODE) | `pi05=True` |
| **Pi0-FAST** | Autoregressive | Discrete tokens (FSQ) | `Pi0FASTConfig` |

All three share the same backbone: **PaliGemma** (SigLIP 400M vision + Gemma 2B LLM) + **Action Expert** (Gemma 300M).

### Pi0 / Pi0.5 Core Architecture

```
Input: images + language prompt + robot state + noise x_t (t=1.0)
         |
         +-- [once] embed_prefix() -> KV Cache
         |     Images -> SigLIP -> visual tokens
         |     Language prompt -> Gemma embedding -> text tokens
         |     (Pi0.5: state discretized into text tokens here)
         |     All prefix tokens use bidirectional attention (prefix-LM)
         |
         +-- while t in [1.0 -> 0.0]:
                embed_suffix(x_t, t) -> action tokens + adaRMS cond
                Gemma([prefix KV cache | suffix]) -> v_t
                x_t = x_t + dt * v_t   (Euler step)
                t += dt                 (dt = -1/num_steps, default 10 steps)
         |
Output: x_0 (denoised actions, shape [B, action_horizon, action_dim])
```

### Pi0 vs Pi0.5 Key Differences

| Aspect | Pi0 | Pi0.5 |
|--------|-----|-------|
| **State input** | Continuous vector in suffix (linear projection) | Discretized to text tokens in prefix (256 bins) |
| **Timestep injection** | Concatenated with action tokens -> MLP | Separate MLP -> adaRMSNorm modulates every layer |
| **Suffix tokens** | state(1) + action(50) = 51 | action(50) only |
| **max_token_len** | 48 | 200 (state becomes text, prefix longer) |
| **Normalization** | Standard RMSNorm | Action Expert uses adaRMSNorm |
| **Training recipe** | Single-stage flow matching | Two-stage: FAST pre-training + flow matching post-training |
| **Co-training data** | Robot action data only | Robot + web data + high-level subtask + verbal instructions |

### Pi0.5 Two-Stage Training (from paper)

**Stage 1 - Pre-training (280k steps):** Standard autoregressive next-token prediction on heterogeneous data:
- Mobile manipulator data (MM, ~400h from ~100 homes)
- Non-mobile robot data (ME, diverse environments)
- Cross-embodiment lab data (CE)
- High-level subtask prediction (HL)
- Web data (WD: captioning, VQA, object localization)
- Actions represented as discrete FAST tokens

**Stage 2 - Post-training (80k steps):** Specializes for mobile manipulation:
- Adds action expert (random init) for flow matching
- Joint training: next-token for text + flow matching for actions (alpha=10.0)
- Uses MM + ME data, web data (WD), high-level labels (HL), verbal instructions (VI)
- At inference: first predicts high-level subtask (text), then low-level actions (flow matching)

### Pi0.5 Hierarchical Inference

At runtime, Pi0.5 performs two-stage inference with the **same model**:
1. **High-level:** Given observation + high-level prompt (e.g., "clean the kitchen"), auto-regressively predict a subtask (e.g., "pick up the plate")
2. **Low-level:** Given observation + subtask as prompt, run flow matching to produce continuous action chunks

### Attention Structure

```
Prefix (images + text + discrete state)        Suffix (action tokens)
+------------------------------------+         +------------------------+
| img_tok ... lang_tok ...           | <------ | act_0  act_1 ... act_49|
| Bidirectional attention (ar=False) |         | Causal attention       |
| PaliGemma 2B processes these       |    X    | Action Expert 300M     |
|                                    |         | + adaRMSNorm(time_emb) |
+------------------------------------+         +------------------------+
      ^ KV Cache (computed once)                       ^ Recomputed each step

Action tokens attend to prefix (via KV cache). Prefix tokens cannot attend to action tokens.
```

### Flow Matching Training

- Noise schedule: `t ~ Beta(1.5, 1)` clipped to [0.001, 0.999]
- Interpolation: `x_t = t * noise + (1-t) * clean_actions`
- Target: velocity field `u_t = noise - clean_actions`
- Loss: MSE between predicted and target velocity field

## Key Code Paths

### JAX Implementation (`src/openpi/models/pi0.py`)

| Method | Purpose |
|--------|---------|
| `embed_prefix()` | Build image + language tokens (computed once per inference) |
| `embed_suffix()` | Build action tokens + timestep conditioning (per denoise step) |
| `compute_loss()` | Training forward pass: flow matching loss |
| `sample_actions()` | Full inference: KV cache + Euler denoising loop |

### PyTorch Implementation (`src/openpi/models_pytorch/pi0_pytorch.py`)

| Method | Purpose |
|--------|---------|
| `embed_prefix()` | Same as JAX version |
| `embed_suffix()` | Same as JAX version |
| `forward()` | Training forward pass |
| `sample_actions()` | Inference with optional staged timing |
| `denoise_step()` | Single Euler denoising step (extracted for profiling) |

### Transform Pipeline (data flows through in this order)

```
Raw robot data
  -> repack_transforms.inputs     (dataset-specific key remapping)
  -> data_transforms.inputs       (robot-specific: AlohaInputs, DroidInputs, etc.)
  -> Normalize                    (z-score or quantile, from NormStats)
  -> model_transforms.inputs      (tokenize prompt, resize images, pad states)
  -> Model.sample_actions()
  -> model_transforms.outputs     (extract FAST actions if applicable)
  -> Unnormalize
  -> data_transforms.outputs      (robot-specific inverse: DroidOutputs, etc.)
  -> repack_transforms.outputs
  -> Final action command
```

## Named Training Configs

Defined in `src/openpi/training/config.py` (_CONFIGS list). Key configs:

**Inference-ready (pre-trained checkpoints):**
- `pi0_aloha`, `pi05_aloha` - ALOHA robot
- `pi0_droid`, `pi05_droid`, `pi0_fast_droid` - DROID robot
- `pi0_aloha_towel`, `pi0_aloha_tupperware` - Task-specific ALOHA

**Fine-tuning templates:**
- `pi0_libero`, `pi05_libero` - LIBERO benchmark
- `pi0_fine_tune_libero`, `pi05_fine_tune_libero` - LoRA fine-tuning
- `pi0_fast_libero` - FAST variant for LIBERO

**Debug:**
- `debug`, `debug_fast`, `debug_pi05` - Small models for testing

**Base checkpoints (GCS):**
- `gs://openpi-assets/checkpoints/pi0_base/params`
- `gs://openpi-assets/checkpoints/pi05_base/params`

## Deployment Options

**1. Direct inference:**
```python
from openpi.policies.policy_config import create_trained_policy
policy = create_trained_policy(config, checkpoint_dir)
actions = policy.infer(observation)
```

**2. Client-server (WebSocket):**
```python
# Server
from openpi.serving import WebsocketPolicyServer
server = WebsocketPolicyServer(policy, port=8000)

# Client (separate machine, minimal deps)
from openpi_client import WebsocketClientPolicy
client = WebsocketClientPolicy(host="server_ip", port=8000)
actions = client.infer(obs)
```

**3. Runtime loop (for robot control):**
```python
from openpi_client.runtime import Runtime
runtime = Runtime(environment, agent, subscribers, max_hz=10)
runtime.run()
```

## Hardware Requirements

- **Inference:** 8GB+ VRAM
- **LoRA fine-tuning:** 22.5GB+ VRAM
- **Full fine-tuning:** 70GB+ VRAM (multi-GPU recommended)
- **Multi-GPU:** JAX uses FSDP; PyTorch uses DDP via torchrun

## Dependencies

- **Primary:** JAX 0.5.3, Flax 0.10.2 (original framework)
- **Secondary:** PyTorch 2.7.1, Transformers 4.53.2 (PyTorch port)
- **Vision:** SigLIP (via PaliGemma)
- **Data:** LeRobot for dataset management, Orbax for checkpoints
- **Tracking:** WandB
- **CLI:** Tyro for config-as-CLI

## Custom Additions in This Fork

- `simple_pytorch_train.py` - Minimal MLP training on HDF5 demo data (no VLA, for quick experiments)
- `convert.py` - HDF5 demonstration data to LeRobot format converter
- `scripts/train_pytorch.py` - Full PyTorch DDP training (added to complement JAX trainer)
- `src/openpi/models_pytorch/` - Complete PyTorch model port
- `examples/convert_jax_model_to_pytorch.py` - JAX-to-PyTorch checkpoint converter

## Caveats and Gotchas

- The `pi05` flag on `Pi0Config` controls whether Pi0 or Pi0.5 codepath is used. Many conditional branches depend on it.
- Pi0.5 `max_token_len=200` vs Pi0 `max_token_len=48` because state is discretized into text.
- `adaRMSNorm` is only used in the Action Expert, not the PaliGemma backbone.
- The PyTorch port in `models_pytorch/` uses modified HuggingFace modules in `transformers_replace/` to support adaRMS. These are not standard HF classes.
- Normalization uses quantile normalization (q01/q99) for Pi0.5 and Pi0-FAST, z-score for Pi0.
- `action_dim=32` is padded to accommodate the largest robot action space; smaller robots zero-pad.
- DROID data can come from either LeRobot format or RLDS format (the latter for large-scale training).
- Image keys are standardized to: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` (defined in `model.py`).
- The `Policy` class handles both JAX and PyTorch models via the `is_pytorch` flag.
