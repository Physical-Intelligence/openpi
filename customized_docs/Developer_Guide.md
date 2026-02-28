# Developer Guide: SpaceCIL + LunarCompose on openpi

This guide covers the codebase architecture, how research code integrates with openpi, and how to extend the system. It assumes you know Python but haven't read openpi internals.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Architecture Overview](#4-architecture-overview)
5. [Data Pipeline](#5-data-pipeline)
6. [Training System](#6-training-system)
7. [Config System](#7-config-system)
8. [Inference and Serving](#8-inference-and-serving)
9. [Testing](#9-testing)
10. [Git Workflow](#10-git-workflow)
11. [Extending the System](#11-extending-the-system)

---

## 1. Project Overview

Two research papers build on top of the [openpi](https://github.com/Physical-Intelligence/openpi) VLA codebase, which provides the pi0 / pi0.5 flow-matching policy implementation in JAX.

**Paper A: SpaceCIL** studies whether a released pi0.5 backbone can be continually specialized as new operational tasks arrive sequentially, without catastrophic forgetting of earlier tasks.

**Paper B: LunarCompose** studies whether factorized task x environment adaptation (separate LoRA adapters for each dimension) enables generalization to unseen task-environment combinations.

Paper B depends on Paper A infrastructure. The build order reflects that dependency: shared infra first, then SpaceCIL core, then LunarCompose extension.

### Platform

- Wheeled base + RM75 7-DoF arm + two-finger gripper
- Primary policy camera: wrist RGB (hand-eye calibrated)
- Action space: 7 absolute joint positions + 1 gripper command = 8D total
- Backbone: pi0.5 via openpi's flow-matching policy path

### Four Operational Tasks

| Task ID | Description |
|---|---|
| `payload` | Transfer payload to goal region |
| `latch` | Actuate a mechanical latch |
| `clean` | Wipe a surface with workspace coverage |
| `connector` | Mate an electrical connector |

### Three Environment Conditions (LunarCompose)

| Env ID | Description |
|---|---|
| `nominal` | Standard lighting and surface conditions |
| `shadow` | Partial shadowing from overhead structures |
| `contamination` | Surface dust/particle contamination |

---

## 2. Repository Structure

```
openpi/                              # Repo root
├── src/openpi/                      # openpi core (DO NOT modify without good reason)
│   ├── models/                      # JAX model definitions
│   │   ├── pi0_config.py            # Pi0Config / Pi0FASTConfig dataclasses
│   │   ├── lora.py                  # LoRA parameter injection (Einsum/FeedForward)
│   │   ├── siglip.py                # SigLIP vision encoder
│   │   └── gemma.py                 # Gemma language model backbone
│   ├── policies/                    # Policy wrappers
│   │   ├── policy.py                # Policy class: .infer(obs) -> actions
│   │   └── policy_config.py         # create_trained_policy()
│   ├── training/                    # Training pipeline
│   │   ├── config.py                # TrainConfig, DataConfig, _CONFIGS list
│   │   ├── data_loader.py           # LeRobot dataset loading
│   │   ├── optimizer.py             # AdamW + schedule
│   │   ├── checkpoints.py           # orbax checkpoint save/load
│   │   └── misc/                    # polaris_config.py, roboarena_config.py
│   ├── transforms.py                # Transform system: Group, RepackTransform, etc.
│   ├── shared/                      # Utilities: download, normalize, nnx_utils
│   └── serving/                     # WebSocket policy server
│
├── src/openpi/research/             # Research code (OUR CODE)
│   ├── __init__.py
│   ├── shared/                      # Shared infrastructure for both papers
│   │   ├── episode_schema.py        # Unified episode schema (obs, action, meta, labels)
│   │   ├── action_transforms.py     # RM75 delta/absolute action transforms
│   │   ├── rm75_policy.py           # RM75Inputs, RM75Outputs, LeRobotRM75DataConfig
│   │   └── scorer_base.py           # Scorer protocol + per-task scorer implementations
│   ├── spacecil/                    # Paper A: SpaceCIL modules
│   │   ├── config.py                # get_spacecil_configs() -> list[TrainConfig]
│   │   ├── task_adapter_bank.py     # TaskAdapterBank: per-task LoRA registry
│   │   ├── router.py                # TaskRouter: language-visual routing MLP
│   │   ├── behavior_distillation.py # CalibrationMemory, TeacherSnapshot, BehaviorDistillation
│   │   ├── continual_harness.py     # ContinualHarness: sequential training orchestrator
│   │   └── metrics.py               # average_success, backward_transfer, forgetting, etc.
│   └── lunarcompose/                # Paper B: LunarCompose modules
│       ├── config.py                # get_lunarcompose_configs() -> list[TrainConfig]
│       ├── env_adapter_bank.py      # EnvAdapterBank: per-environment LoRA registry
│       ├── dual_head_router.py      # DualHeadRouter: separate task + env routing heads
│       ├── missing_corner_harness.py # MissingCornerHarness: compositional eval
│       └── factorization_diagnostics.py # seen_unseen_gap, entanglement analysis
│
├── scripts/                         # Training scripts
│   ├── train.py                     # openpi standard training loop (DO NOT modify)
│   ├── train_spacecil.py            # SpaceCIL continual training script
│   ├── train_lunarcompose.py        # LunarCompose factorized training script
│   ├── serve_policy.py              # Policy server launcher
│   └── compute_norm_stats.py        # Normalization stats precomputation
│
├── projects/                        # Plans and documentation (NOT source code)
│   ├── PLAN.md
│   ├── shared/PLAN.md
│   ├── spacecil/PLAN.md
│   └── lunarcompose/PLAN.md
│
└── customized_docs/                 # Research documentation
    ├── Developer_Guide.md           # This file
    ├── Research_Idea_Blueprint_SpaceCIL_LunarCompose.md
    └── Implementation_Masterplan_SpaceCIL_LunarCompose.md
```

### Module Classification

Every module falls into one of three categories:

| Classification | Meaning | Examples |
|---|---|---|
| **reuse as-is** | Use openpi directly, no changes | `models/lora.py`, `transforms.py`, `training/optimizer.py` |
| **patch lightly** | One-line addition to openpi core | `training/config.py` (`_CONFIGS` splat) |
| **new module** | New code in `src/openpi/research/` | All research modules |

The `training/config.py` patch is intentionally minimal: two lazy-import helper functions and two splat entries at the end of `_CONFIGS`. Everything else is additive.

---

## 3. Environment Setup

### Initial Setup

The environment is managed by [uv](https://docs.astral.sh/uv/). The virtual environment is already created at `.venv/`.

```bash
# Activate the virtualenv
source .venv/bin/activate

# Or run any command directly via uv (no activation needed)
uv run python -c "import openpi; print('ok')"
```

If you need to reinstall dependencies after a `pyproject.toml` change:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### Key Commands

```bash
# Run training (standard openpi config)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
    --exp-name my_experiment

# Run SpaceCIL continual training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector

# Run LunarCompose factorized training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_factorized \
    --rotation 0

# Compute normalization stats (required before first training run on new data)
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_payload

# Serve a trained policy checkpoint for inference
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=spacecil_rm75_payload \
    --policy.dir=checkpoints/spacecil_rm75_payload/my_run/10000

# Run tests (all research code, CPU-only)
JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q

# Run tests (full suite)
JAX_PLATFORMS=cpu uv run pytest src/ scripts/ -x -q
```

### Environment Variables

| Variable | Purpose | Typical Value |
|---|---|---|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX GPU memory allocation | `0.9` |
| `JAX_PLATFORMS` | Force JAX to use a specific backend | `cpu` (for testing) |
| `OPENPI_DATA_HOME` | Override checkpoint download cache | `~/.cache/openpi` |
| `WANDB_DISABLED` | Disable Weights & Biases logging | `true` |

---

## 4. Architecture Overview

### Component Relationships

```
 openpi Core (DO NOT modify unless necessary)
 +--------------------------------------------------+
 |  models/          training/         policies/    |
 |  +----------+    +----------+    +----------+   |
 |  | Pi0Config|    | config.py|    |policy.py |   |
 |  | lora.py  |    | train.py |    |policy_   |   |
 |  | siglip   |    | optim.py |    |config.py |   |
 |  | gemma    |    | checkpts |    +----------+   |
 |  +----------+    +----------+                   |
 |  transforms.py  (Group, RepackTransform, ...)   |
 +--------------------------------------------------+
          ^                   ^
          |  import / reuse   |
 +--------------------------------------------------+
 |  src/openpi/research/                           |
 |                                                  |
 |  shared/                                         |
 |  +------------------+  +---------------------+  |
 |  | episode_schema   |  | action_transforms   |  |
 |  | rm75_policy      |  | scorer_base         |  |
 |  +------------------+  +---------------------+  |
 |                                                  |
 |  spacecil/                lunarcompose/          |
 |  +------------------+    +------------------+   |
 |  | config           |    | config           |   |
 |  | task_adapter_bank|    | env_adapter_bank |   |
 |  | router           |    | dual_head_router |   |
 |  | behavior_distill |    | missing_corner   |   |
 |  | continual_harness|    | factorization    |   |
 |  | metrics          |    | diagnostics      |   |
 |  +------------------+    +------------------+   |
 |                                                  |
 |  scripts/                                        |
 |  train_spacecil.py    train_lunarcompose.py      |
 +--------------------------------------------------+
```

### How Research Code Plugs In

Research code connects to openpi through three narrow integration points:

1. **Config registration** -- `get_spacecil_configs()` and `get_lunarcompose_configs()` are called via lazy-import helpers in `training/config.py` and their results are splatted into `_CONFIGS`. This is the only modification to openpi core.

2. **Data pipeline** -- `LeRobotRM75DataConfig` implements `DataConfigFactory`, the same interface used by `LeRobotLiberoDataConfig`. It returns a populated `DataConfig` with our custom transforms wired in.

3. **LoRA infrastructure** -- `TaskAdapterBank` and `EnvAdapterBank` use openpi's existing `lora.py` parameters and `nnx_utils.PathRegex` filter. They snapshot and inject LoRA weights via `nnx.state(model).filter(...)` and `nnx.update(model, lora_state)`. No changes to `lora.py` needed.

### Import Structure and Circular Import Avoidance

The main circular import risk is:
```
research/spacecil/config.py
  -> rm75_policy.py
     -> openpi.training.config (DataConfigFactory)
        -> _get_spacecil_configs()
           -> spacecil/config.py  <-- cycle!
```

Two mechanisms prevent this:

**Re-entrancy guard** in each `get_*_configs()` function:
```python
_building = False
_cache: list | None = None

def get_spacecil_configs():
    global _building, _cache
    if _cache is not None:
        return list(_cache)
    if _building:
        return []   # nested call during _CONFIGS construction: return empty
    _building = True
    try:
        # ALL imports deferred here inside the function
        from openpi.training.config import TrainConfig
        ...
        _cache = configs
        return list(_cache)
    finally:
        _building = False
```

**Lazy import helpers** in `training/config.py`:
```python
def _get_spacecil_configs():
    import openpi.research.spacecil.config as spacecil_config
    return spacecil_config.get_spacecil_configs()
```

The `_CONFIGS` list calls `_get_spacecil_configs()` (not the module directly), deferring the import until after `config.py` finishes loading.

---

## 5. Data Pipeline

### 5.1 Episode Schema

`src/openpi/research/shared/episode_schema.py` defines the unified episode schema shared across both papers. All collected data is represented as `Episode` objects before being written to LeRobot format.

```
Episode
├── metadata: EpisodeMetadata
│   ├── task_id: str             # "payload", "latch", "clean", "connector"
│   ├── env_id: str              # "nominal", "shadow", "contamination"
│   ├── operator_id: str
│   ├── session_id: str
│   ├── camera_preset_id: str
│   ├── calibration_version: str
│   ├── scene_revision: str
│   └── object_revision: str
├── labels: EpisodeLabels
│   ├── success: bool
│   └── fail_type: str | None
├── steps: list[EpisodeStep]
│   └── EpisodeStep
│       ├── observation: Observation
│       │   ├── wrist_rgb: np.ndarray     # (H, W, 3) uint8 -- required
│       │   ├── joint_position: (7,)      # absolute, radians
│       │   ├── joint_velocity: (7,)
│       │   ├── gripper_position: (1,)    # normalised [0, 1]
│       │   ├── scene_rgb: (H, W, 3) | None  # optional
│       │   └── base_state: (N,) | None      # optional
│       ├── action: Action
│       │   ├── joint_pos: (7,)           # absolute joint angles
│       │   └── gripper_cmd: float        # [0, 1]
│       └── timestamp_s: float
└── prompt: str                            # language instruction
```

`Episode.to_dict()` produces JSON-serializable output. `Episode.from_dict()` reconstructs. Both use `SCHEMA_VERSION = "1.0"` for forward compatibility.

`make_repack_structure()` returns a `{dst_key: src_key}` mapping for use with `RepackTransform`, bridging LeRobot-style flat keys to the model's expected input keys.

### 5.2 Data Flow

```
  LeRobot Dataset on disk
  (HuggingFace format)
         |
         v
  data_loader.py
  (loads episodes as dict batches)
         |
         v
  RepackTransform              <-- LeRobotRM75DataConfig.repack_transforms
  key mapping: identity for RM75
  (dataset keys already match inference keys)
         |
         v
  RM75Inputs (DataTransformFn) <-- data_config.data_transforms.inputs
  - parse wrist image to (H,W,C) uint8
  - build 15D state: joint_pos(7) + joint_vel(7) + gripper(1)
  - map to model image slots (PI0/PI05 or PI0_FAST layout)
  - pass through actions, prompt
         |
         v
  RM75DeltaActions             <-- data_config.data_transforms.inputs (pushed)
  - actions[..., :7] -= state[..., :7]
  - gripper dim stays absolute
         |
         v
  Normalize                    <-- norm_stats.json (per-dim quantile stats)
         |
         v
  model_transforms             <-- tokenization, image preprocessing
         |
         v
  Pi0Config model (JAX, nnx)
  (flow-matching policy)
         |
         v
  model_transforms.outputs
         |
         v
  Unnormalize
         |
         v
  RM75AbsoluteActions          <-- data_config.data_transforms.outputs
  - actions[..., :7] += state[..., :7]  (undo delta)
         |
         v
  RM75Outputs (DataTransformFn)
  - slice to first 8 dims
  - return {"actions": (T, 8)}
```

### 5.3 Action Space

The RM75 action space is 8-dimensional throughout:

```
Index    Meaning                  Transform
[0:7]    Absolute joint position  DELTA during training, ABSOLUTE at inference
[7]      Gripper command [0, 1]   Always absolute (no delta)
```

The delta conversion is handled by `RM75_DELTA_MASK = transforms.make_bool_mask(7, -1)`, which is `True` for the first 7 dims and `False` for dim 7 (gripper). The transform pair is:

- **Training input**: `RM75DeltaActions` converts absolute actions to deltas
- **Inference output**: `RM75AbsoluteActions` adds state back to recover absolute actions

The full conversion chain for collected data is:

```
teleop_to_canonical()      -- validate shape, clip gripper
canonical_to_training()    -- identity + validation (canonical IS training format)
```

And the reverse for inference:

```
training_to_canonical()    -- identity + validation
```

`RM75Inputs` builds the 15D proprioceptive state vector used both as the model's state input and as the reference for delta computation.

### 5.4 Image Slot Mapping

The RM75 has only a wrist camera. Unused image slots are zero-padded. The slot assignment differs by model type:

```
PI0 / PI05:
  base_0_rgb       <- zeros (mask=False)
  left_wrist_0_rgb <- wrist image (mask=True)
  right_wrist_0_rgb <- zeros (mask=False)

PI0_FAST:
  base_0_rgb  <- zeros (mask=True, FAST doesn't mask)
  base_1_rgb  <- zeros (mask=True)
  wrist_0_rgb <- wrist image (mask=True)
```

---

## 6. Training System

### 6.1 openpi Training Pipeline

The standard training loop lives in `scripts/train.py`. Key functions:

```python
# Initialize model, optimizer, EMA, and sharding
train_state, state_sharding = init_train_state(config, rng, mesh, resume=False)

# One JIT-compiled training step
train_info = train_step(train_state, batch)

# Save checkpoint
save_checkpoint(train_state, step, checkpoint_dir)
```

`train_state` contains:
- `model` -- the `nnx.Module` (Pi0 with all parameters)
- `optimizer_state` -- AdamW state
- `ema_state` -- exponential moving average weights (optional; `ema_decay=None` disables it)
- `step` -- global step counter

Checkpoints are written using [orbax](https://github.com/google/orbax) to `checkpoints/<config_name>/<exp_name>/<step>/`.

### 6.2 LoRA Fine-Tuning

openpi's LoRA is implemented in `models/lora.py`. LoRA parameters appear in `Einsum` and `FeedForward` layers as `lora_a` and `lora_b` weight matrices.

**Freeze filter**: Controls which parameters receive gradient updates. The shared LoRA freeze filter used by all research configs is:

```python
_lora_freeze = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter()
```

This freezes the base backbone and leaves only `.*lora.*` parameters trainable.

**LoRA parameter extraction**: The `LORA_FILTER` in `task_adapter_bank.py`:

```python
from openpi.shared import nnx_utils
LORA_FILTER = nnx_utils.PathRegex(".*lora.*")
```

Applied as:

```python
model_state = nnx.state(model)
lora_state = model_state.filter(LORA_FILTER)
pure_dict = lora_state.to_pure_dict()
```

**Adapter swapping**: Injecting a saved adapter back into the model:

```python
# MUST be called outside JIT boundaries
model_state = nnx.state(model)
lora_state = model_state.filter(LORA_FILTER)
lora_state.replace_by_pure_dict(jax_dict)
nnx.update(model, lora_state)
```

Calling `nnx.update` inside a JIT-compiled function would cause recompilation on every call. Always swap adapters in Python (eager) mode, before or after `train_step(...)`.

**Environment adapters** (LunarCompose) use the same mechanism but with a different filter targeting SigLIP layers:

```python
ENV_LORA_FILTER = nnx_utils.PathRegex(".*siglip.*")
```

The composition rule is:
```
effective_weights = base + task_lora_delta + env_lora_delta
```

Both deltas are applied via sequential `nnx.update` calls before the forward pass.

### 6.3 Custom Training Scripts

`scripts/train_spacecil.py` and `scripts/train_lunarcompose.py` follow the same pattern:

1. Import `init_train_state` and `train_step` from `scripts/train.py`
2. Build research components (adapter banks, distillation, harness)
3. Provide a `train_fn` closure to the harness
4. The harness calls `train_fn(task_id)` for each task/cell in sequence
5. Adapter swapping happens between `train_step` calls, outside JIT

The intended wiring for SpaceCIL:

```python
mesh = jax.sharding.Mesh(jax.devices(), ("fsdp",))
rng = jax.random.PRNGKey(args.seed)
train_state, state_sharding = init_train_state(config, rng, mesh, resume=False)

def train_fn(task_id: str):
    # 1. Load data for this task
    # 2. Swap in this task's LoRA adapter (outside JIT)
    adapter_bank.merge_into_model(model, task_id)
    # 3. Run N steps
    for batch in data_loader:
        train_info = train_step(train_state, batch)
    # 4. Return trained model + info
    return model_from_state(train_state), [train_info]

result = harness.run_sequence(train_fn)
```

For LunarCompose, `train_fn(task_id, env_id)` applies both task and env adapters:

```python
def train_fn(task_id: str, env_id: str):
    task_bank.merge_into_model(model, task_id)   # task LoRA
    env_bank.merge_into_model(model, env_id)     # env LoRA (sequential)
    # ... run train_step loop ...
```

### 6.4 Behavior Distillation (SpaceCIL)

Anti-forgetting is implemented in `behavior_distillation.py`:

```
CalibrationMemory    -- stores up to N episodes per task for replay
TeacherSnapshot      -- frozen copy of the model's full nnx.State
BehaviorDistillation -- orchestrates the two, computes combined loss
```

The combined loss is:

```
total_loss = task_loss + alpha * distillation_loss(student_actions, teacher_actions)
```

where `distillation_loss` is MSE (default) or L1. `jax.lax.stop_gradient` is applied to teacher predictions so no gradients flow to the frozen teacher parameters.

After training each task:

```python
distillation.update_teacher(trained_model)     # snapshot current model
distillation.add_calibration_episodes(task_id, episodes)  # store for replay
```

---

## 7. Config System

### 7.1 TrainConfig

`TrainConfig` in `training/config.py` is the central configuration dataclass. Key fields:

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | Globally unique config name |
| `model` | `BaseModelConfig` | Model variant (Pi0Config, Pi0FASTConfig) |
| `data` | `DataConfigFactory` | Data pipeline factory |
| `weight_loader` | `WeightLoader` | Where to load base weights from |
| `freeze_filter` | `FreezeFilter` | Which params to freeze during training |
| `batch_size` | `int` | Training batch size |
| `num_train_steps` | `int` | Total training steps |
| `ema_decay` | `float or None` | EMA decay; `None` disables EMA |
| `exp_name` | `str or None` | Experiment name for checkpoint directory |
| `wandb_enabled` | `bool` | Enable Weights & Biases logging |
| `overwrite` | `bool` | Overwrite existing checkpoint directory |

Retrieving a config by name:

```python
from openpi.training import config as _config
config = _config.get_config("spacecil_rm75_payload")
```

`get_config` raises `ValueError` with a closest-match suggestion if the name isn't found.

### 7.2 Config Registration Pattern

New config sets are registered via the "polaris pattern":

**Step 1**: Write a `get_*_configs()` function in `src/openpi/research/<paper>/config.py`:

```python
_building = False
_cache: list | None = None

def get_spacecil_configs():
    global _building, _cache
    if _cache is not None:
        return list(_cache)
    if _building:
        return []
    _building = True
    try:
        # ALL imports deferred inside this function
        from openpi.training.config import TrainConfig
        ...
        configs = [TrainConfig(name="spacecil_...", ...) for ...]
        _cache = configs
        return list(_cache)
    finally:
        _building = False
```

**Step 2**: Add a lazy-import helper in `src/openpi/training/config.py`:

```python
def _get_spacecil_configs():
    """Lazy import to avoid circular: spacecil/config -> rm75_policy -> training/config."""
    import openpi.research.spacecil.config as spacecil_config
    return spacecil_config.get_spacecil_configs()
```

**Step 3**: Splat into `_CONFIGS` at the end of the list:

```python
_CONFIGS = [
    ...existing configs...,
    *_get_spacecil_configs(),
    *_get_lunarcompose_configs(),
]
```

That's the entire integration. Config names are validated for uniqueness immediately after the list is built:

```python
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
```

### 7.3 Config Namespaces

| Namespace | Count | Pattern |
|---|---|---|
| `spacecil_rm75_*` | 4 | one per task: `payload`, `latch`, `clean`, `connector` |
| `spacecil_debug` | 1 | dummy model, FakeDataConfig, 10 steps |
| `lunarcompose_{task}_{env}` | 12 | all 4x3 task-env cells |
| `lunarcompose_factorized` | 1 | architecture-level config |
| `lunarcompose_monolithic` | 1 | baseline comparison config |
| `lunarcompose_debug` | 1 | dummy model, FakeDataConfig, 10 steps |

All production configs use:
- `pi05=True` (pi0.5 backbone)
- `paligemma_variant="gemma_2b_lora"` (enables LoRA in language path)
- `action_expert_variant="gemma_300m_lora"` (enables LoRA in action expert)
- `weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
- `ema_decay=None` (disabled to reduce memory and checkpoint size)
- `batch_size=32`, `num_train_steps=10_000`

Debug configs use `paligemma_variant="dummy"` and `action_expert_variant="dummy"` so they run in seconds without downloading any weights.

### 7.4 DataConfigFactory Pattern

`LeRobotRM75DataConfig` implements `DataConfigFactory` by subclassing the dynamically resolved base class:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotRM75DataConfig(_RM75DataConfigFactoryBase):
    def create(self, assets_dirs, model_config):
        repack_transform = Group(inputs=[RepackTransform({...})])
        data_transforms = Group(
            inputs=[RM75Inputs(model_type=model_config.model_type)],
            outputs=[RM75Outputs()],
        )
        data_transforms = data_transforms.push(
            inputs=[RM75DeltaActions(mask=RM75_DELTA_MASK)],
            outputs=[RM75AbsoluteActions(mask=RM75_DELTA_MASK)],
        )
        model_transforms = _RM75ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

The dynamic base class resolution avoids an import cycle: `rm75_policy.py` tries to import `DataConfigFactory` from `training/config.py`, but if that import fails (e.g., during early module loading), it falls back to a stub that raises a clear error at runtime.

---

## 8. Inference and Serving

### Creating a Policy

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config = _config.get_config("spacecil_rm75_payload")
policy = policy_config.create_trained_policy(
    config,
    checkpoint_dir="checkpoints/spacecil_rm75_payload/my_run/10000"
)
```

`create_trained_policy` automatically detects whether the checkpoint is JAX (orbax) or PyTorch (looks for `model.safetensors`).

The resulting `Policy` object has a single inference method:

```python
obs = {
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/joint_position": np.zeros(7),
    "observation/joint_velocity": np.zeros(7),
    "observation/gripper_position": np.zeros(1),
    "prompt": "transfer the payload to the goal region",
}
result = policy.infer(obs)
actions = result["actions"]   # shape (action_horizon, 8)
```

### Inference Transform Chain

```
obs dict (raw)
    |
    v
repack_transforms.inputs      # optional, default empty
    |
    v
InjectDefaultPrompt            # injects default prompt if none provided
    |
    v
data_transforms.inputs         # RM75Inputs: image parsing, state building
    |
    v
Normalize                      # per-dim quantile normalization
    |
    v
model_transforms.inputs        # tokenization, image encoding
    |
    v
  Pi0 model forward pass
    |
    v
model_transforms.outputs
    |
    v
Unnormalize
    |
    v
data_transforms.outputs        # RM75Outputs: slice to 8D; RM75AbsoluteActions: undo delta
    |
    v
repack_transforms.outputs
    |
    v
result dict {"actions": (T, 8)}
```

### Policy Server

For remote inference (policy runs on a GPU server, robot queries over the network):

```bash
# Launch server on port 8000 (default)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=spacecil_rm75_payload \
    --policy.dir=checkpoints/spacecil_rm75_payload/my_run/10000
```

The server accepts observations over WebSocket and returns action chunks. See `src/openpi/serving/` for the protocol implementation.

### Adapter Swapping at Inference

For SpaceCIL inference with task routing, swap the adapter before creating the policy or before each inference call:

```python
# Hard routing: argmax task selection
task_idx = router.route_hard(lang_emb, visual_emb, active_mask)
task_id = task_sequence[int(task_idx[0])]

# Swap adapter (outside any JIT scope)
adapter_bank.merge_into_model(model, task_id)

# Then infer
actions = policy.infer(obs)["actions"]
```

For LunarCompose, apply both adapters sequentially:

```python
task_bank.merge_into_model(model, task_id)
env_bank.merge_into_model(model, env_id)
actions = policy.infer(obs)["actions"]
```

---

## 9. Testing

### Test Placement

Tests are co-located with source files, named `<module>_test.py`. This follows the openpi convention:

```
src/openpi/research/shared/
    episode_schema.py
    episode_schema_test.py
    rm75_policy.py
    rm75_policy_test.py
    ...
```

### conftest.py

`src/openpi/conftest.py` sets `JAX_PLATFORMS=cpu` at import time, so all tests run on CPU by default without requiring a GPU. This means test collection doesn't need `JAX_PLATFORMS=cpu` set explicitly, but the environment variable is still useful to set explicitly for clarity.

### Debug Config Pattern

Tests that need a real model use the debug config pattern to avoid downloading weights:

```python
import openpi.models.pi0_config as pi0_config

model_config = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="dummy",
    action_expert_variant="dummy",
)
```

`"dummy"` variants initialize random small matrices rather than loading from a checkpoint, so tests run in milliseconds.

### Running Tests

```bash
# Research code only (fastest, recommended during development)
JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q

# Full suite
JAX_PLATFORMS=cpu uv run pytest src/ scripts/ -x -q

# Specific module
JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/task_adapter_bank_test.py -v

# Run before every push
JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q && echo "All tests passed"
```

The `-x` flag stops on first failure. The `-q` flag reduces output verbosity. Add `-s` if you need `print` output from tests.

### Test Patterns

**Testing a transform**:

```python
from openpi.research.shared.rm75_policy import RM75Inputs, make_rm75_example
from openpi.models.model import ModelType

transform = RM75Inputs(model_type=ModelType.PI0)
result = transform(make_rm75_example())
assert result["state"].shape == (15,)
assert "base_0_rgb" in result["image"]
```

**Testing a scorer**:

```python
from openpi.research.shared.scorer_base import PayloadTransferScorer
from openpi.research.shared.episode_schema import Episode, EpisodeMetadata, EpisodeLabels

scorer = PayloadTransferScorer(goal_region_threshold=0.1)
episode = Episode(metadata=..., labels=..., steps=[...], prompt="...")
result = scorer.score(episode)
assert isinstance(result.success, bool)
assert 0.0 <= result.confidence <= 1.0
```

**Testing adapter bank**:

```python
import flax.nnx as nnx
from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank

# Use a tiny model with LoRA
model = SomeNnxModuleWithLoRA(rngs=nnx.Rngs(0))
bank = TaskAdapterBank()
bank.register_adapter("task_0", model)
bank.freeze_adapter("task_0")
assert bank.is_frozen("task_0")

# Save and reload
bank.save("/tmp/test_bank")
loaded = TaskAdapterBank.load("/tmp/test_bank")
assert loaded.registered_tasks == ["task_0"]
```

---

## 10. Git Workflow

### Remotes

| Remote | URL | Purpose |
|---|---|---|
| `origin` | `git@github.com:Physical-Intelligence/openpi.git` | Upstream openpi -- READ-ONLY |
| `lunarbot` | `git@github.com:DsslRobot/openpi-lunarbot.git` | Our research fork -- push here |

**NEVER push to `origin`.**

### Branches

- `lunarbot-research` -- main working branch, tracked on `lunarbot` remote
- `main` -- tracks upstream openpi, do NOT commit research code here

### Commit Workflow

```bash
# Check state before starting
git status
git stash   # stash if switching context

# After completing a logical unit of work
git add src/openpi/research/spacecil/task_adapter_bank.py
git add src/openpi/research/spacecil/task_adapter_bank_test.py
git commit -m "feat: add TaskAdapterBank with LoRA snapshot and merge_into_model"

# Verify tests pass before pushing
JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q

# Push to our fork only
git push lunarbot lunarbot-research
```

### Commit Message Format

```
feat: add episode schema dataclass with obs/action/meta fields
fix: correct delta action normalization bounds
test: add scorer_base protocol compliance tests
refactor: extract common transform logic to shared/
docs: update PLAN.md with revised build order
chore: update .gitignore for checkpoint dirs
```

One commit per logical unit. Never commit broken code. Never commit large binary files (checkpoints, datasets, model weights).

### Safety Rules

- NEVER push to `origin`
- ALWAYS push to `lunarbot` remote
- NEVER force-push without explicit request
- NEVER commit `.env`, credentials, or API keys
- NEVER commit `assets/`, `checkpoints/`, `data/`, or `wandb/` directories
- Run tests before every push

---

## 11. Extending the System

### 11.1 Adding a New Task

A new operational task (e.g., `"drill"`) requires:

**1. Add a scorer** in `src/openpi/research/shared/scorer_base.py`:

```python
class DrillOperationScorer(Scorer):
    """Checks if drill operation completed successfully."""

    def __init__(self, penetration_threshold: float = 0.3) -> None:
        self.penetration_threshold = penetration_threshold

    def score(self, episode: Episode) -> ScorerResult:
        if len(episode.steps) == 0:
            return ScorerResult(success=False, confidence=0.0, fail_type="no_data")
        # ... your heuristic logic using joint positions / velocities / gripper ...
        return ScorerResult(success=True, confidence=0.8)
```

**2. Collect data** and convert to LeRobot format. The dataset `repo_id` will be `"your_org/spacecil_drill"`.

**3. Add a config** in `src/openpi/research/spacecil/config.py`:

- Add `"drill"` to `_TASKS`:

```python
_TASKS = ("payload", "latch", "clean", "connector", "drill")
```

The list comprehension in `get_spacecil_configs()` generates the new config automatically. Invalidate the cache:

```python
_cache = None   # force rebuild on next import
```

**4. Compute norm stats** for the new dataset:

```bash
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_drill
```

**5. Register the adapter** in your training script and add it to the task sequence.

### 11.2 Adding a New Environment Condition

A new environment condition (e.g., `"glare"`) for LunarCompose:

**1. Add to `_ENVS`** in `src/openpi/research/lunarcompose/config.py`:

```python
_ENVS = ("nominal", "shadow", "contamination", "glare")
```

This generates 4 new per-cell configs (`lunarcompose_{task}_glare` for each task) automatically.

**2. Update `_CANONICAL_SPLITS`** in `missing_corner_harness.py` to include the new environment in the rotation definitions.

**3. Collect data** for each task under the new environment condition.

**4. Compute norm stats** for all new configs.

**5. Register env adapters**:

```python
env_bank.register_env("glare", model)
env_bank.freeze_env("glare")
```

`EnvAdapterBank` uses `.*siglip.*` as the default LoRA target, so environment adapters specifically target the vision encoder. You can override this:

```python
env_bank = EnvAdapterBank(lora_target=".*siglip.*lora.*")
```

### 11.3 Adding a New Scorer

Subclass `Scorer` from `scorer_base.py`:

```python
class MyScorer(Scorer):
    def score(self, episode: Episode) -> ScorerResult:
        # Access episode.steps[i].observation.{wrist_rgb, joint_position, ...}
        # Access episode.steps[i].action.{joint_pos, gripper_cmd}
        ...
        return ScorerResult(
            success=True,
            confidence=0.9,
            fail_type=None,
            details={"my_metric": 0.42},
        )
```

Every scorer must be validated against manual labels on a pilot subset before use in any experiment. The scorer must not become a hidden confounder of the paper's claims.

Wire the scorer into `ContinualHarness` or `MissingCornerHarness`:

```python
harness = ContinualHarness(
    scorers={"payload": PayloadTransferScorer(), "drill": DrillOperationScorer()},
    ...
)
```

### 11.4 Adding a New Config

For a config that doesn't fit the auto-generated pattern:

```python
# Inside get_spacecil_configs(), before setting _cache
configs.append(
    TrainConfig(
        name="spacecil_rm75_drill_highres",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRM75DataConfig(
            repo_id="your_org/spacecil_drill_highres",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_lora_freeze,
        ema_decay=None,
        num_train_steps=15_000,
        batch_size=16,   # smaller batch for high-res images
        wandb_enabled=True,
    )
)
```

Config names must be globally unique across all `_CONFIGS`. The uniqueness check runs at import time and will raise immediately if there's a collision.

### 11.5 Adding a New Adapter Bank Type

If you need a third dimension of adaptation beyond task and environment (e.g., per-operator adaptation), model `TaskAdapterBank` from `task_adapter_bank.py`. The key design decisions:

- Store weights as pure numpy dicts (not live `nnx.State`) to avoid serialization issues
- Provide `register_*(id, model)`, `merge_into_model(model, id)`, `freeze_*(id)`, `save(path)`, `load(path)`
- Use `nnx.state(model).filter(YOUR_FILTER)` for extraction
- Use `nnx.update(model, state)` for injection -- ALWAYS outside JIT

The `EnvAdapterBank` in `env_adapter_bank.py` is a good reference for a bank with a configurable LoRA target and an optional fallback mode.

Adapter composition with multiple banks is additive:

```python
# Apply in any order; deltas accumulate
task_bank.merge_into_model(model, task_id)
env_bank.merge_into_model(model, env_id)
operator_bank.merge_into_model(model, operator_id)
# forward pass sees base + task_delta + env_delta + operator_delta
```

### 11.6 Adding Metrics

All metric functions in `spacecil/metrics.py` are pure functions operating on numpy arrays. The `result_matrix` convention is `R[i][j] = success rate on task j after training task i`.

To add a new metric:

```python
def my_metric(result_matrix: np.ndarray, **kwargs) -> float:
    """Document the matrix convention and return type clearly."""
    ...
    return float(result)
```

Import and call from your training script:

```python
from openpi.research.spacecil import metrics

final_metrics = {
    "average_success": metrics.average_success(result.result_matrix),
    "forgetting": metrics.forgetting(result.result_matrix),
    "backward_transfer": metrics.backward_transfer(result.result_matrix),
    "my_metric": metrics.my_metric(result.result_matrix),
}
```

For LunarCompose, `factorization_diagnostics.py` provides `seen_unseen_gap` and related functions for analyzing task-environment entanglement. These operate on `MissingCornerResult` objects.

---

## Appendix: Build Order Gates

Before running experiments, verify the relevant gate criteria:

| Gate | Requirement |
|---|---|
| **G1** | One task trains and replays correctly; wrist-camera policy works; scorer matches manual labels |
| **G2** | 2+ tasks train sequentially; router beats random; distillation executes stably; adapter checkpoint restore works |
| **G3** | Env metadata enforced without leakage; missing-corner split verified; env adaptation path exists |

G1 must pass before any continual learning experiments. G2 before Paper A mainline claims. G3 before Paper B mainline claims.

---

## Appendix: Key File Index

| File | Purpose |
|---|---|
| `src/openpi/research/shared/episode_schema.py` | Unified episode schema, serialization |
| `src/openpi/research/shared/action_transforms.py` | RM75 8D action space, delta/absolute transforms |
| `src/openpi/research/shared/rm75_policy.py` | RM75Inputs, RM75Outputs, LeRobotRM75DataConfig |
| `src/openpi/research/shared/scorer_base.py` | Scorer protocol, per-task scorer implementations |
| `src/openpi/research/spacecil/config.py` | SpaceCIL TrainConfig registration |
| `src/openpi/research/spacecil/task_adapter_bank.py` | Per-task LoRA registry with save/load |
| `src/openpi/research/spacecil/router.py` | TaskRouter MLP: language+visual -> task softmax |
| `src/openpi/research/spacecil/behavior_distillation.py` | CalibrationMemory, TeacherSnapshot, distillation loss |
| `src/openpi/research/spacecil/continual_harness.py` | Sequential training loop, result matrix |
| `src/openpi/research/spacecil/metrics.py` | average_success, backward_transfer, forgetting |
| `src/openpi/research/lunarcompose/config.py` | LunarCompose TrainConfig registration (15 configs) |
| `src/openpi/research/lunarcompose/env_adapter_bank.py` | Per-environment LoRA registry (SigLIP target) |
| `src/openpi/research/lunarcompose/dual_head_router.py` | Separate task + env routing heads |
| `src/openpi/research/lunarcompose/missing_corner_harness.py` | Compositional train/test split, seen-unseen gap |
| `src/openpi/research/lunarcompose/factorization_diagnostics.py` | Task-env entanglement analysis |
| `src/openpi/training/config.py` | TrainConfig, _CONFIGS list (patched with splat entries) |
| `src/openpi/policies/policy_config.py` | create_trained_policy() |
| `scripts/train_spacecil.py` | SpaceCIL CLI training script |
| `scripts/train_lunarcompose.py` | LunarCompose CLI training script |
