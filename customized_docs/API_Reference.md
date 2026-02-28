# API Reference — SpaceCIL + LunarCompose Research Infrastructure

This document covers every public class, function, and dataclass in the three research packages built on top of openpi. The code lives entirely under `src/openpi/research/` and is importable as `openpi.research.*` after a standard `uv sync`.

**Framework context:** JAX + Flax NNX. Training uses openpi's flow-matching pipeline on a pi0.5 backbone. The robot platform is an RM75 7-DoF arm with a two-finger gripper and a wrist RGB camera. All actions are 8-dimensional: 7 absolute joint positions (radians) + 1 gripper command ([0, 1]).

---

## Table of Contents

1. [Package: `openpi.research.shared`](#package-openpuresearchshared)
   - [Module: `episode_schema`](#module-episode_schema)
   - [Module: `action_transforms`](#module-action_transforms)
   - [Module: `rm75_policy`](#module-rm75_policy)
   - [Module: `scorer_base`](#module-scorer_base)
2. [Package: `openpi.research.spacecil`](#package-openpuresearchspacecil)
   - [Module: `task_adapter_bank`](#module-task_adapter_bank)
   - [Module: `router`](#module-router)
   - [Module: `behavior_distillation`](#module-behavior_distillation)
   - [Module: `continual_harness`](#module-continual_harness)
   - [Module: `metrics`](#module-metrics)
   - [Module: `config`](#module-config-spacecil)
3. [Package: `openpi.research.lunarcompose`](#package-openpuresearchlunarcompose)
   - [Module: `env_adapter_bank`](#module-env_adapter_bank)
   - [Module: `dual_head_router`](#module-dual_head_router)
   - [Module: `missing_corner_harness`](#module-missing_corner_harness)
   - [Module: `factorization_diagnostics`](#module-factorization_diagnostics)
   - [Module: `config`](#module-config-lunarcompose)
4. [Cross-Reference Index](#cross-reference-index)

---

## Package: `openpi.research.shared`

Shared infrastructure used by both SpaceCIL and LunarCompose. Contains the unified episode schema, action transform layer, RM75 platform policy wiring, and task scorer abstractions.

---

### Module: `episode_schema`

**Import:** `from openpi.research.shared.episode_schema import ...`

Defines the frozen dataclass hierarchy for storing and serializing robot episodes. All dataclasses use `frozen=True` (immutable after construction).

---

#### `EpisodeMetadata`

```python
@dataclasses.dataclass(frozen=True)
class EpisodeMetadata:
    task_id: str
    env_id: str
    operator_id: str = ""
    session_id: str = ""
    camera_preset_id: str = ""
    calibration_version: str = ""
    scene_revision: str = ""
    object_revision: str = ""
```

Metadata fields attached to a single episode. All fields are strings.

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Identifies the operational task (e.g. `"payload"`, `"latch"`). Required. |
| `env_id` | `str` | Identifies the environment condition (e.g. `"nominal"`, `"shadow"`). Required. |
| `operator_id` | `str` | Identifier for the human operator. Defaults to empty string. |
| `session_id` | `str` | Recording session identifier. Defaults to empty string. |
| `camera_preset_id` | `str` | Camera mounting / calibration preset. Defaults to empty string. |
| `calibration_version` | `str` | Version tag for the hand-eye calibration file. Defaults to empty string. |
| `scene_revision` | `str` | Scene configuration version (lighting, layout). Defaults to empty string. |
| `object_revision` | `str` | Object configuration version (props, placement). Defaults to empty string. |

**Example:**

```python
meta = EpisodeMetadata(task_id="payload", env_id="nominal", operator_id="op_01")
```

---

#### `EpisodeLabels`

```python
@dataclasses.dataclass(frozen=True)
class EpisodeLabels:
    success: bool
    fail_type: str | None = None
```

Ground-truth outcome labels for a single episode.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the episode was successful. |
| `fail_type` | `str \| None` | Failure category string (e.g. `"drop"`, `"timeout"`). `None` on success. |

**Example:**

```python
labels = EpisodeLabels(success=False, fail_type="drop")
```

---

#### `Observation`

```python
@dataclasses.dataclass(frozen=True)
class Observation:
    wrist_rgb: np.ndarray          # (H, W, 3) uint8
    joint_position: np.ndarray     # (7,) float32 — arm-only joint angles
    joint_velocity: np.ndarray     # (7,) float32
    gripper_position: np.ndarray   # (1,) float32, normalised [0, 1]
    scene_rgb: np.ndarray | None = None   # (H, W, 3) uint8, optional
    base_state: np.ndarray | None = None  # (N,) float32, optional
```

Single-step observation bundle. `wrist_rgb` is the primary policy input and is always required. `scene_rgb` and `base_state` are optional peripherals that may be absent in most RM75 deployments.

| Field | Type | Description |
|---|---|---|
| `wrist_rgb` | `np.ndarray` | `(H, W, 3)` uint8 wrist camera image. Required. |
| `joint_position` | `np.ndarray` | `(7,)` float32 absolute joint angles in radians for the 7-DoF arm. |
| `joint_velocity` | `np.ndarray` | `(7,)` float32 joint angular velocities. |
| `gripper_position` | `np.ndarray` | `(1,)` float32 gripper openness in [0, 1]. |
| `scene_rgb` | `np.ndarray \| None` | `(H, W, 3)` uint8 external scene camera. `None` when not recorded. |
| `base_state` | `np.ndarray \| None` | `(N,)` float32 mobile base state. `None` for fixed-base setups. |

**Example:**

```python
import numpy as np
obs = Observation(
    wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
    joint_position=np.zeros(7, dtype=np.float32),
    joint_velocity=np.zeros(7, dtype=np.float32),
    gripper_position=np.array([0.5], dtype=np.float32),
)
```

---

#### `Action`

```python
@dataclasses.dataclass(frozen=True)
class Action:
    joint_pos: np.ndarray   # (7,) float32 — absolute joint angles (radians)
    gripper_cmd: float      # [0, 1]
```

Single-step action in the canonical absolute joint position + gripper command space.

| Field | Type | Description |
|---|---|---|
| `joint_pos` | `np.ndarray` | `(7,)` float32 target joint angles in radians. |
| `gripper_cmd` | `float` | Gripper command in [0, 1]. 0 = closed, 1 = open. |

##### `Action.to_array() -> np.ndarray`

```python
def to_array(self) -> np.ndarray:
```

Flatten to a `(8,)` training array: `[joint_pos(7), gripper(1)]`.

**Returns:** `np.ndarray` of shape `(8,)`, dtype float32.

##### `Action.from_array(arr) -> Action`

```python
@classmethod
def from_array(cls, arr: np.ndarray) -> Action:
```

Reconstruct an `Action` from a `(8,)` array.

| Parameter | Type | Description |
|---|---|---|
| `arr` | `np.ndarray` | Array of shape `(8,)`. |

**Returns:** A new `Action` instance.

**Raises:** `ValueError` if `arr.shape != (8,)`.

**Example:**

```python
arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
action = Action.from_array(arr)
assert action.to_array().shape == (8,)
```

---

#### `EpisodeStep`

```python
@dataclasses.dataclass(frozen=True)
class EpisodeStep:
    observation: Observation
    action: Action
    timestamp_s: float = 0.0
```

One timestep inside an episode, pairing an observation with its commanded action.

| Field | Type | Description |
|---|---|---|
| `observation` | `Observation` | Sensor readings at this timestep. |
| `action` | `Action` | Action commanded at this timestep. |
| `timestamp_s` | `float` | Wall-clock timestamp in seconds. Defaults to 0.0. |

---

#### `Episode`

```python
@dataclasses.dataclass(frozen=True)
class Episode:
    SCHEMA_VERSION: str  # "1.0" (init=False, not a constructor argument)
    metadata: EpisodeMetadata
    labels: EpisodeLabels
    steps: list[EpisodeStep]
    prompt: str = ""
```

Complete episode container. Serializable to JSON-compatible dicts via `to_dict` / `from_dict`.

| Field | Type | Description |
|---|---|---|
| `SCHEMA_VERSION` | `str` | Fixed at `"1.0"`. Not passed to constructor. |
| `metadata` | `EpisodeMetadata` | Task, environment, and session metadata. |
| `labels` | `EpisodeLabels` | Success/fail ground-truth outcome. |
| `steps` | `list[EpisodeStep]` | Ordered list of timesteps. |
| `prompt` | `str` | Language instruction for this episode. Defaults to empty string. |

##### `Episode.to_dict() -> dict[str, Any]`

```python
def to_dict(self) -> dict[str, Any]:
```

Serialize the episode to a JSON-compatible dict. Arrays are stored as nested Python lists.

**Returns:** `dict` with keys `"schema_version"`, `"metadata"`, `"labels"`, `"prompt"`, `"steps"`.

##### `Episode.from_dict(d) -> Episode`

```python
@classmethod
def from_dict(cls, d: dict[str, Any]) -> Episode:
```

Reconstruct an `Episode` from a dict previously produced by `to_dict`.

| Parameter | Type | Description |
|---|---|---|
| `d` | `dict[str, Any]` | Dict in the schema produced by `to_dict`. |

**Returns:** A new `Episode` instance.

**Example:**

```python
import json
ep = Episode(metadata=meta, labels=labels, steps=steps, prompt="grasp the payload")
serialized = json.dumps(ep.to_dict())
restored = Episode.from_dict(json.loads(serialized))
```

---

#### `make_repack_structure() -> dict[str, str]`

```python
def make_repack_structure() -> dict[str, str]:
```

Return a flat `{dst_key: src_key}` mapping for use with `openpi.transforms.RepackTransform`.

Maps LeRobot-style dataset keys to the inference-like keys expected by downstream `DataTransformFn` classes such as `RM75Inputs`. Both source and destination keys use `/` separators.

**Returns:** `dict[str, str]` with these entries:

| Destination key | Source key |
|---|---|
| `"observation/wrist_image"` | `"observation/wrist_image"` |
| `"observation/scene_image"` | `"observation/scene_image"` |
| `"observation/joint_position"` | `"observation/joint_position"` |
| `"observation/joint_velocity"` | `"observation/joint_velocity"` |
| `"observation/gripper_position"` | `"observation/gripper_position"` |
| `"actions"` | `"actions"` |
| `"prompt"` | `"prompt"` |

**Example:**

```python
from openpi.transforms import RepackTransform
repack = RepackTransform(make_repack_structure())
```

---

### Module: `action_transforms`

**Import:** `from openpi.research.shared.action_transforms import ...`

Authoritative transform path for RM75 actions. Defines constants, functional converters, and `DataTransformFn` subclasses that plug into the openpi data pipeline.

---

#### Constants

```python
ACTION_DIM: int = 8
```
Total action dimensionality: 7 joint positions + 1 gripper command.

```python
JOINT_DIM: int = 7
```
Joint position dimensionality for the 7-DoF arm.

```python
GRIPPER_RANGE: tuple[float, float] = (0.0, 1.0)
```
Valid gripper command range after normalization.

```python
RM75_DELTA_MASK: tuple[bool, ...] = transforms.make_bool_mask(JOINT_DIM, -1)
```
Boolean mask used by `RM75DeltaActions` / `RM75AbsoluteActions`. `True` for the first 7 dims (delta applied to joint positions); `False` for the gripper dim (stays absolute).

---

#### `teleop_to_canonical(teleop_action) -> np.ndarray`

```python
def teleop_to_canonical(teleop_action: np.ndarray) -> np.ndarray:
```

Convert raw teleop controller output to canonical absolute joint position + gripper format.

Currently an identity-like transform: validates shape, converts to float32, and clips the gripper to [0, 1]. If a future teleop controller uses a different representation, this is the single place to adapt.

| Parameter | Type | Description |
|---|---|---|
| `teleop_action` | `np.ndarray` | Shape `(8,)` or `(..., 8)` raw teleop output. |

**Returns:** `np.ndarray` with same shape, dtype float32, gripper clipped to [0, 1].

**Example:**

```python
raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2])  # gripper out-of-range
canonical = teleop_to_canonical(raw)
assert canonical[-1] == 1.0  # clipped
```

---

#### `canonical_to_training(canonical_action) -> np.ndarray`

```python
def canonical_to_training(canonical_action: np.ndarray) -> np.ndarray:
```

Convert canonical format to openpi training format.

The canonical and training formats are identical for RM75 (`[joint_pos(7), gripper(1)]`). This function acts as a safety gate: validates shape and clips the gripper.

| Parameter | Type | Description |
|---|---|---|
| `canonical_action` | `np.ndarray` | Shape `(8,)` or `(..., 8)` canonical action. |

**Returns:** `np.ndarray` in training format, float32, gripper clipped.

---

#### `training_to_canonical(training_action) -> np.ndarray`

```python
def training_to_canonical(training_action: np.ndarray) -> np.ndarray:
```

Convert openpi training format back to canonical format.

Exact inverse of `canonical_to_training` (trivially invertible since the formats are identical up to validation and clipping).

| Parameter | Type | Description |
|---|---|---|
| `training_action` | `np.ndarray` | Shape `(8,)` or `(..., 8)` training action. |

**Returns:** `np.ndarray` canonical action, float32, gripper clipped.

---

#### `RM75DeltaActions`

```python
@dataclasses.dataclass(frozen=True)
class RM75DeltaActions(transforms.DataTransformFn):
    mask: tuple[bool, ...] = RM75_DELTA_MASK
```

Convert absolute joint-space actions to delta actions for RM75. Follows the same pattern as openpi's `transforms.DeltaActions`.

Subtracts current joint state from action joint dims (`mask == True`), leaving the gripper (`dim 7`) as-is.

**Use in:** `DataConfig.data_transforms.inputs`.

| Field | Type | Description |
|---|---|---|
| `mask` | `tuple[bool, ...]` | Which dims receive delta conversion. Defaults to `RM75_DELTA_MASK` (first 7 True, last False). |

##### `__call__(data) -> transforms.DataDict`

```python
def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
```

Apply delta conversion in-place on `data["actions"]` using `data["state"]`.

| Parameter | Type | Description |
|---|---|---|
| `data` | `transforms.DataDict` | Must contain `"state"` and `"actions"` keys. |

**Returns:** Updated `DataDict` with `"actions"` converted to deltas.

**Example:**

```python
transform = RM75DeltaActions()
data = {"state": state_arr, "actions": abs_actions}
delta_data = transform(data)
```

---

#### `RM75AbsoluteActions`

```python
@dataclasses.dataclass(frozen=True)
class RM75AbsoluteActions(transforms.DataTransformFn):
    mask: tuple[bool, ...] = RM75_DELTA_MASK
```

Convert delta actions back to absolute joint-space actions. Inverse of `RM75DeltaActions`.

Adds current joint state back to the masked dims to recover absolute joint positions.

**Use in:** `DataConfig.data_transforms.outputs`.

| Field | Type | Description |
|---|---|---|
| `mask` | `tuple[bool, ...]` | Which dims receive absolute conversion. Defaults to `RM75_DELTA_MASK`. |

##### `__call__(data) -> transforms.DataDict`

```python
def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `transforms.DataDict` | Must contain `"state"` and `"actions"` keys. |

**Returns:** Updated `DataDict` with `"actions"` in absolute joint space.

---

### Module: `rm75_policy`

**Import:** `from openpi.research.shared.rm75_policy import ...`

Follows the `libero_policy.py` / `droid_policy.py` pattern. Wires up the RM75 platform to the openpi training and inference pipelines.

---

#### `make_rm75_example() -> dict`

```python
def make_rm75_example() -> dict:
```

Creates a random input example dict for testing the RM75 policy. Generates plausible shapes and dtypes but uses random values.

**Returns:** `dict` with keys:

| Key | Shape / Type | Description |
|---|---|---|
| `"observation/wrist_image"` | `(224, 224, 3)` uint8 | Random wrist camera image. |
| `"observation/joint_position"` | `(7,)` float64 | Random joint angles. |
| `"observation/joint_velocity"` | `(7,)` float64 | Random joint velocities. |
| `"observation/gripper_position"` | `(1,)` float64 | Random gripper position. |
| `"prompt"` | `str` | Literal `"do something"`. |

**Example:**

```python
example = make_rm75_example()
output = policy.infer(example)
```

---

#### `RM75Inputs`

```python
@dataclasses.dataclass(frozen=True)
class RM75Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType
```

Convert RM75 observations to the model's expected input dictionary.

Builds a 15D state vector from proprioception and maps the wrist camera to the appropriate image slot based on model type. Missing camera slots are zero-padded.

| Field | Type | Description |
|---|---|---|
| `model_type` | `_model.ModelType` | Selects image slot naming and masking strategy. One of `PI0`, `PI05`, `PI0_FAST`. |

**State vector layout:** `[joint_position(7), joint_velocity(7), gripper_position(1)]` = 15D.

**Image slot mapping:**

| Model type | Slot for wrist | Zero-padded slots |
|---|---|---|
| `PI0`, `PI05` | `left_wrist_0_rgb` | `base_0_rgb`, `right_wrist_0_rgb` |
| `PI0_FAST` | `wrist_0_rgb` | `base_0_rgb`, `base_1_rgb` |

##### `__call__(data) -> dict`

```python
def __call__(self, data: dict) -> dict:
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict` | Must contain `"observation/wrist_image"`, `"observation/joint_position"`, `"observation/joint_velocity"`, `"observation/gripper_position"`. Optionally `"actions"` and `"prompt"`. |

**Returns:** `dict` with keys `"state"`, `"image"`, `"image_mask"`, and optionally `"actions"` and `"prompt"`.

**Example:**

```python
transform = RM75Inputs(model_type=ModelType.PI05)
model_input = transform(raw_observation_dict)
```

---

#### `RM75Outputs`

```python
@dataclasses.dataclass(frozen=True)
class RM75Outputs(transforms.DataTransformFn):
```

Convert model output actions back to RM75 format.

Slices the model output to the first 8 dimensions, discarding any padding added by the model backbone.

##### `__call__(data) -> dict`

```python
def __call__(self, data: dict) -> dict:
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict` | Must contain `"actions"` key with shape `(chunk_size, model_action_dim)`. |

**Returns:** `{"actions": data["actions"][:, :8]}` — shape `(chunk_size, 8)`.

---

#### `LeRobotRM75DataConfig`

```python
@dataclasses.dataclass(frozen=True)
class LeRobotRM75DataConfig(_RM75DataConfigFactoryBase):
    repo_id: str
    assets: Any = None
    base_config: Any = None
```

Data config factory for RM75 robot datasets in LeRobot format. Inherits from openpi's `DataConfigFactory`.

Wires up in `create()`:
- `RepackTransform` for identity key mapping (RM75 dataset keys already match inference keys)
- `RM75Inputs` / `RM75Outputs` for observation-to-model and model-to-action transforms
- `RM75DeltaActions` / `RM75AbsoluteActions` for delta conversion during training

##### `create(assets_dirs, model_config) -> DataConfig`

```python
def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> Any:
```

Build and return a fully configured `DataConfig` instance.

| Parameter | Type | Description |
|---|---|---|
| `assets_dirs` | `pathlib.Path` | Directory where model assets are stored. |
| `model_config` | `_model.BaseModelConfig` | Model config, used to determine `model_type` for `RM75Inputs`. |

**Returns:** A `DataConfig` with `repack_transforms`, `data_transforms`, and `model_transforms` populated.

**Example:**

```python
data_cfg = LeRobotRM75DataConfig(
    repo_id="my_org/rm75_payload",
    base_config=DataConfig(prompt_from_task=True),
)
```

---

### Module: `scorer_base`

**Import:** `from openpi.research.shared.scorer_base import ...`

Per-task success scorers. Every scorer validates against manual labels on a pilot subset before use in experiments.

---

#### `ScorerResult`

```python
@dataclasses.dataclass(frozen=True)
class ScorerResult:
    success: bool
    confidence: float
    fail_type: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
```

Result from scoring a single episode.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the episode was scored as successful. |
| `confidence` | `float` | Scorer's confidence in the determination, in [0, 1]. |
| `fail_type` | `str \| None` | Failure category. `None` on success or unknown failures. |
| `details` | `dict[str, Any]` | Scorer-specific diagnostic data (e.g. `{"coverage": 0.72}`). |

---

#### `Scorer`

```python
class Scorer(abc.ABC):
```

Abstract base class for task-specific scorers. Subclasses implement `score()` for each operational task.

##### `score(episode) -> ScorerResult`

```python
@abc.abstractmethod
def score(self, episode: Episode) -> ScorerResult:
```

Score a single episode for task success.

| Parameter | Type | Description |
|---|---|---|
| `episode` | `Episode` | Complete episode to evaluate. |

**Returns:** `ScorerResult` with success determination and confidence.

---

#### `PayloadTransferScorer`

```python
class PayloadTransferScorer(Scorer):
    def __init__(self, goal_region_threshold: float = 0.1) -> None:
```

Checks if payload reached the goal region. Uses gripper closure and joint displacement as heuristic proxies.

**Heuristics:**
- Gripper closed (`gripper_position < 0.5`) → payload was grasped
- Joint displacement `> goal_region_threshold` → arm moved to goal region

| Parameter | Type | Description |
|---|---|---|
| `goal_region_threshold` | `float` | Minimum L2 norm of joint displacement to count as "moved". Default: `0.1`. |

**Failure modes returned:**

| `fail_type` | Condition |
|---|---|
| `"drop"` | Arm moved but gripper open |
| `"timeout"` | Gripper closed but arm didn't move |
| `"other"` | Neither condition met |
| `"no_data"` | Empty episode |

**Confidence levels:** 0.8 (success), 0.7 (drop), 0.6 (timeout), 0.5 (other).

##### `score(episode) -> ScorerResult`

```python
def score(self, episode: Episode) -> ScorerResult:
```

**Example:**

```python
scorer = PayloadTransferScorer(goal_region_threshold=0.15)
result = scorer.score(episode)
if result.success:
    print(f"Task succeeded with confidence {result.confidence}")
```

---

#### `LatchActuationScorer`

```python
class LatchActuationScorer(Scorer):
    def __init__(self, actuation_threshold: float = 0.5) -> None:
```

Checks if the latch reached an actuated state. Uses maximum single-joint displacement as proxy for latch actuation.

| Parameter | Type | Description |
|---|---|---|
| `actuation_threshold` | `float` | Minimum max absolute joint displacement to count as actuated. Default: `0.5`. |

**Failure modes:** `"timeout"` (displacement below threshold), `"no_data"` (empty episode).

**Confidence levels:** 0.85 (success), 0.7 (timeout).

##### `score(episode) -> ScorerResult`

```python
def score(self, episode: Episode) -> ScorerResult:
```

---

#### `SurfaceCleaningScorer`

```python
class SurfaceCleaningScorer(Scorer):
    def __init__(self, coverage_threshold: float = 0.5) -> None:
```

Checks if the surface was cleaned. Estimates workspace coverage by discretizing the joint trajectory into bins and measuring what fraction of bin-space was visited.

| Parameter | Type | Description |
|---|---|---|
| `coverage_threshold` | `float` | Minimum coverage fraction (0-1) to count as cleaned. Default: `0.5`. |

**Coverage calculation:** For each joint, the observed range is divided into 10 bins. Coverage fraction = (unique bins visited) / (total bins across all joints).

**Failure modes:** `"timeout"` (insufficient coverage), `"no_data"` (empty episode).

**Confidence levels:** 0.7 (success), 0.6 (timeout).

`ScorerResult.details` includes `{"coverage": float}` for both success and failure.

##### `score(episode) -> ScorerResult`

```python
def score(self, episode: Episode) -> ScorerResult:
```

---

#### `ConnectorMatingScorer`

```python
class ConnectorMatingScorer(Scorer):
    def __init__(self, stability_window: int = 5, stability_threshold: float = 0.02) -> None:
```

Checks if the connector was mated. Uses gripper closure and final-step stability (low joint velocity) as heuristic proxies.

| Parameter | Type | Description |
|---|---|---|
| `stability_window` | `int` | Number of final steps to examine for stability. Default: `5`. |
| `stability_threshold` | `float` | Maximum mean absolute joint velocity to count as stable. Default: `0.02`. |

**Failure modes:**

| `fail_type` | Condition |
|---|---|
| `"contact"` | Gripper closed but arm still moving |
| `"other"` | Gripper open |
| `"no_data"` | Empty episode |

**Confidence levels:** 0.85 (success), 0.6 (contact), 0.5 (other).

##### `score(episode) -> ScorerResult`

```python
def score(self, episode: Episode) -> ScorerResult:
```

---

## Package: `openpi.research.spacecil`

SpaceCIL (Paper A) modules for continual skill acquisition. Trains task adapters sequentially, routes between them without oracle task IDs, and uses behavior distillation to prevent forgetting.

---

### Module: `task_adapter_bank`

**Import:** `from openpi.research.spacecil.task_adapter_bank import ...`

Per-task PEFT module registry wrapping openpi's LoRA infrastructure.

---

#### `LORA_FILTER`

```python
LORA_FILTER = nnx_utils.PathRegex(".*lora.*")
```

`nnx_utils.PathRegex` filter matching any parameter path containing `"lora"` (e.g. `lora_a`, `lora_b`). Used to extract and inject LoRA weights from live models.

---

#### `TaskAdapterBank`

```python
class TaskAdapterBank:
```

Versioned registry mapping task IDs to LoRA parameter states.

Stores adapter weights as pure dicts (via `nnx.State.to_pure_dict()`) for serialization safety. Adapters can be frozen to prevent accidental re-registration.

> **Note:** `merge_into_model` must be called outside JIT boundaries to avoid JAX recompilation.

**Usage pattern:**

```python
bank = TaskAdapterBank()
bank.register_adapter("task_0", model)   # snapshot LoRA from live model
bank.freeze_adapter("task_0")            # make immutable
bank.merge_into_model(model, "task_0")   # inject weights back (outside JIT)
bank.save("/tmp/bank")
bank = TaskAdapterBank.load("/tmp/bank")
```

##### `TaskAdapterBank.__init__()`

```python
def __init__(self) -> None:
```

Initialize an empty bank with no registered adapters.

---

##### `TaskAdapterBank.register_adapter(task_id, model)`

```python
def register_adapter(self, task_id: str, model: nnx.Module) -> None:
```

Extract LoRA state from a live model and store it under `task_id`.

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | Unique identifier for this task adapter. |
| `model` | `nnx.Module` | Live model whose LoRA parameters will be snapshotted. |

**Raises:** `ValueError` if `task_id` is already frozen.

---

##### `TaskAdapterBank.register_adapter_from_state(task_id, lora_state)`

```python
def register_adapter_from_state(self, task_id: str, lora_state: nnx.State) -> None:
```

Store adapter from an existing `nnx.State` rather than a live model.

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | Unique identifier for this task adapter. |
| `lora_state` | `nnx.State` | Pre-filtered LoRA state to store. |

**Raises:** `ValueError` if `task_id` is already frozen.

---

##### `TaskAdapterBank.get_adapter(task_id) -> dict`

```python
def get_adapter(self, task_id: str) -> dict:
```

Return the stored pure dict for a task adapter.

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | Which adapter to retrieve. |

**Returns:** Pure dict of numpy arrays representing the LoRA weights.

**Raises:** `KeyError` if `task_id` is not registered.

---

##### `TaskAdapterBank.merge_into_model(model, task_id)`

```python
def merge_into_model(self, model: nnx.Module, task_id: str) -> None:
```

Apply stored LoRA weights to a live model via `nnx.update`.

| Parameter | Type | Description |
|---|---|---|
| `model` | `nnx.Module` | The live model to update in-place. |
| `task_id` | `str` | Which adapter's weights to inject. |

**Raises:** `KeyError` if `task_id` is not registered.

> **Warning:** Must be called **outside JIT boundaries** to avoid triggering JAX recompilation.

---

##### `TaskAdapterBank.freeze_adapter(task_id)`

```python
def freeze_adapter(self, task_id: str) -> None:
```

Mark an adapter as immutable, preventing re-registration.

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | Adapter to freeze. |

**Raises:** `KeyError` if `task_id` is not registered.

---

##### `TaskAdapterBank.is_frozen(task_id) -> bool`

```python
def is_frozen(self, task_id: str) -> bool:
```

Check whether an adapter is frozen.

**Returns:** `True` if the adapter is frozen, `False` otherwise.

---

##### `TaskAdapterBank.num_adapters` (property)

```python
@property
def num_adapters(self) -> int:
```

Number of registered adapters.

---

##### `TaskAdapterBank.registered_tasks` (property)

```python
@property
def registered_tasks(self) -> list[str]:
```

Task IDs in registration order.

---

##### `TaskAdapterBank.save(path)`

```python
def save(self, path: str | Path) -> None:
```

Save adapter bank to disk. Creates:
- `path/metadata.json` — frozen set and registration order
- `path/{task_id}/adapter.npz` — flattened LoRA arrays per task

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Directory to write to. Created if it does not exist. |

---

##### `TaskAdapterBank.load(path) -> TaskAdapterBank`

```python
@classmethod
def load(cls, path: str | Path) -> TaskAdapterBank:
```

Load adapter bank from disk.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Directory previously written by `save()`. |

**Returns:** A new `TaskAdapterBank` with all adapters and frozen state restored.

---

### Module: `router`

**Import:** `from openpi.research.spacecil.router import ...`

Language-visual router for task adapter selection without oracle task IDs.

---

#### `make_active_mask(num_active, max_tasks) -> jax.Array`

```python
def make_active_mask(num_active: int, max_tasks: int) -> jax.Array:
```

Create a boolean mask with the first `num_active` entries `True`.

| Parameter | Type | Description |
|---|---|---|
| `num_active` | `int` | Number of currently registered tasks. |
| `max_tasks` | `int` | Total pre-allocated task slots. |

**Returns:** Boolean `jax.Array` of shape `(max_tasks,)`. Entry `i` is `True` iff `i < num_active`.

**Example:**

```python
mask = make_active_mask(num_active=3, max_tasks=16)
# mask[:3] == True, mask[3:] == False
```

---

#### `TaskRouter`

```python
class TaskRouter(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        max_tasks: int = 16,
        num_layers: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
```

Lightweight MLP router mapping concatenated language + visual embeddings to a softmax distribution over task adapters.

**Architecture:** `input -> [Linear -> LayerNorm -> GELU] x num_layers -> Linear -> masked softmax`

The output head is pre-allocated to `max_tasks` slots. Inactive slots are masked to `-1e9` before softmax, so the router can grow (new tasks added) without resizing.

| Parameter | Type | Description |
|---|---|---|
| `input_dim` | `int` | Dimensionality of `concat(lang_embedding, visual_summary)`. |
| `hidden_dim` | `int` | Hidden layer size for each MLP block. Default: `256`. |
| `max_tasks` | `int` | Pre-allocated output slots. Default: `16`. |
| `num_layers` | `int` | Number of `(Linear -> LayerNorm -> GELU)` blocks. Default: `2`. |
| `rngs` | `nnx.Rngs` | Flax NNX random number generators. Keyword-only. |

---

##### `TaskRouter.__call__(lang_embedding, visual_summary, active_mask) -> jax.Array`

```python
def __call__(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    active_mask: jax.Array | None = None,
) -> jax.Array:
```

Forward pass returning softmax routing probabilities.

| Parameter | Type | Description |
|---|---|---|
| `lang_embedding` | `jax.Array` | Shape `(B, lang_dim)` language embedding. |
| `visual_summary` | `jax.Array` | Shape `(B, visual_dim)` visual summary embedding. |
| `active_mask` | `jax.Array \| None` | Boolean mask `(max_tasks,)`. `True` = active. `None` = all active. |

**Returns:** `jax.Array` of shape `(B, max_tasks)` — softmax probabilities.

---

##### `TaskRouter.route_hard(lang_embedding, visual_summary, active_mask) -> jax.Array`

```python
def route_hard(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    active_mask: jax.Array | None = None,
) -> jax.Array:
```

Argmax routing — returns selected task index per batch element.

**Returns:** Integer `jax.Array` of shape `(B,)`.

---

##### `TaskRouter.route_soft(lang_embedding, visual_summary, active_mask) -> jax.Array`

```python
def route_soft(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    active_mask: jax.Array | None = None,
) -> jax.Array:
```

Soft routing — returns the full probability distribution. Alias for `__call__`.

**Returns:** `jax.Array` of shape `(B, max_tasks)`.

**Example:**

```python
router = TaskRouter(input_dim=512, rngs=nnx.Rngs(0))
mask = make_active_mask(num_active=2, max_tasks=16)
probs = router.route_soft(lang_emb, vis_emb, active_mask=mask)
task_idx = router.route_hard(lang_emb, vis_emb, active_mask=mask)
```

---

### Module: `behavior_distillation`

**Import:** `from openpi.research.spacecil.behavior_distillation import ...`

Anti-forgetting via teacher snapshot and calibration memory replay.

---

#### `distillation_loss(student_actions, teacher_actions, loss_type) -> jax.Array`

```python
def distillation_loss(
    student_actions: jax.Array,
    teacher_actions: jax.Array,
    loss_type: str = "mse",
) -> jax.Array:
```

Compute distillation loss between student and teacher action predictions.

`stop_gradient` is automatically applied to `teacher_actions`, so no gradient flows through the teacher.

| Parameter | Type | Description |
|---|---|---|
| `student_actions` | `jax.Array` | Shape `(B, action_horizon, action_dim)` student predictions. |
| `teacher_actions` | `jax.Array` | Shape `(B, action_horizon, action_dim)` teacher predictions. `stop_gradient` applied internally. |
| `loss_type` | `str` | `"mse"` (default) for mean-squared error, `"l1"` for mean absolute error. |

**Returns:** Scalar `jax.Array` loss value.

**Raises:** `ValueError` if `loss_type` is not `"mse"` or `"l1"`.

**Formulas:**
- `"mse"`: `mean((student - stop_gradient(teacher))^2)`
- `"l1"`: `mean(|student - stop_gradient(teacher)|)`

**Example:**

```python
loss = distillation_loss(student_acts, teacher_acts, loss_type="mse")
```

---

#### `CalibrationMemory`

```python
@dataclasses.dataclass
class CalibrationMemory:
    max_episodes_per_task: int = 100
```

Buffer storing calibration episodes from previous tasks for replay during distillation.

Each task contributes up to `max_episodes_per_task` episodes. `sample_batch` draws uniformly across all stored tasks.

| Field | Type | Description |
|---|---|---|
| `max_episodes_per_task` | `int` | Maximum episodes retained per task. Default: `100`. |

##### `CalibrationMemory.add_episodes(task_id, episodes)`

```python
def add_episodes(self, task_id: str, episodes: list[dict]) -> None:
```

Add episodes for a task, truncating to `max_episodes_per_task`.

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | Task identifier. |
| `episodes` | `list[dict]` | List of episode dicts to add. |

---

##### `CalibrationMemory.sample_batch(batch_size, rng) -> list[dict]`

```python
def sample_batch(
    self,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> list[dict]:
```

Sample `batch_size` episodes uniformly across all stored tasks.

| Parameter | Type | Description |
|---|---|---|
| `batch_size` | `int` | Requested number of episodes. |
| `rng` | `np.random.Generator \| None` | Random number generator. Creates a new one if `None`. |

**Returns:** `list[dict]` of sampled episodes. Returns `[]` when no episodes are stored.

---

##### `CalibrationMemory.num_episodes(task_id) -> int`

```python
def num_episodes(self, task_id: str | None = None) -> int:
```

Count episodes for a specific task or total across all tasks.

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str \| None` | If provided, counts for that task only. If `None`, counts all. |

**Returns:** `int` episode count.

---

##### `CalibrationMemory.task_ids` (property)

```python
@property
def task_ids(self) -> list[str]:
```

Return list of task IDs with stored episodes.

---

#### `TeacherSnapshot`

```python
@dataclasses.dataclass
class TeacherSnapshot:
    params: dict | None = None
```

Frozen copy of model parameters for distillation. Stores a pure dict snapshot of the full model state (backbone + LoRA). The teacher is always frozen — no gradient flow through it.

| Field | Type | Description |
|---|---|---|
| `params` | `dict \| None` | Full model state as pure dict. `None` until `snapshot()` is called. |

##### `TeacherSnapshot.snapshot(model)`

```python
def snapshot(self, model: nnx.Module) -> None:
```

Extract and deep-copy full model state from an `nnx.Module`.

| Parameter | Type | Description |
|---|---|---|
| `model` | `nnx.Module` | Live model to snapshot. |

---

##### `TeacherSnapshot.has_snapshot` (property)

```python
@property
def has_snapshot(self) -> bool:
```

Whether a snapshot has been taken.

---

##### `TeacherSnapshot.get_params() -> dict`

```python
def get_params(self) -> dict:
```

Return stored params.

**Raises:** `RuntimeError` if no snapshot has been taken.

---

#### `BehaviorDistillation`

```python
@dataclasses.dataclass
class BehaviorDistillation:
    memory: CalibrationMemory
    teacher: TeacherSnapshot
    distillation_weight: float = 0.5
    loss_type: str = "mse"
```

Orchestrator combining calibration memory, teacher snapshot, and distillation loss.

| Field | Type | Description |
|---|---|---|
| `memory` | `CalibrationMemory` | Calibration episode buffer. |
| `teacher` | `TeacherSnapshot` | Frozen teacher snapshot. |
| `distillation_weight` | `float` | Weight for distillation loss term in combined loss. Default: `0.5`. |
| `loss_type` | `str` | Loss type passed to `distillation_loss`. Default: `"mse"`. |

##### `BehaviorDistillation.compute_total_loss(task_loss, student_actions, teacher_actions)`

```python
def compute_total_loss(
    self,
    task_loss: jax.Array,
    student_actions: jax.Array,
    teacher_actions: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
```

Compute combined task + distillation loss.

When `distillation_weight == 0.0`, returns `task_loss` directly with zero distillation loss.

**Formula:** `total = task_loss + distillation_weight * distillation_loss(student, teacher)`

| Parameter | Type | Description |
|---|---|---|
| `task_loss` | `jax.Array` | Scalar task-specific loss (e.g. flow-matching loss). |
| `student_actions` | `jax.Array` | `(B, horizon, dim)` student predictions. |
| `teacher_actions` | `jax.Array` | `(B, horizon, dim)` teacher predictions. |

**Returns:** `(total_loss, metrics_dict)` where `metrics_dict` has keys `"task_loss"`, `"distill_loss"`, `"total_loss"`.

---

##### `BehaviorDistillation.update_teacher(model)`

```python
def update_teacher(self, model: nnx.Module) -> None:
```

Snapshot the current model as the new teacher.

---

##### `BehaviorDistillation.add_calibration_episodes(task_id, episodes)`

```python
def add_calibration_episodes(self, task_id: str, episodes: list[dict]) -> None:
```

Add calibration episodes for a task to the replay memory.

**Example:**

```python
bd = BehaviorDistillation(
    memory=CalibrationMemory(),
    teacher=TeacherSnapshot(),
    distillation_weight=0.5,
)
bd.update_teacher(model)
bd.add_calibration_episodes("task_0", task0_episodes)
total, metrics = bd.compute_total_loss(task_loss, student_acts, teacher_acts)
```

---

### Module: `continual_harness`

**Import:** `from openpi.research.spacecil.continual_harness import ...`

Sequential task training loop and backward transfer evaluation. Decoupled from JAX internals via a caller-provided `TrainFn`.

---

#### `TrainFn`

```python
TrainFn = Callable[[str], tuple[Any, list[dict]]]
```

Type alias for the training function. Called as `train_fn(task_id)` and returns `(trained_model, training_info_list)`.

The `trained_model` must be an `nnx.Module` compatible with `TaskAdapterBank.register_adapter`. `training_info_list` is a list of per-step info dicts (loss curves, etc.).

---

#### `ContinualResult`

```python
@dataclasses.dataclass(frozen=True)
class ContinualResult:
    task_sequence: list[str]
    result_matrix: np.ndarray   # shape (T, T)
    per_step_info: list[dict]   # one dict per task trained
```

Immutable result container for a completed continual learning sequence.

| Field | Type | Description |
|---|---|---|
| `task_sequence` | `list[str]` | Ordered list of task IDs as trained. |
| `result_matrix` | `np.ndarray` | Shape `(T, T)`. `R[i][j]` = success rate on task `j` after training task `i`. Upper-triangle entries (`j > i`) are `NaN`. |
| `per_step_info` | `list[dict]` | Per-task training info dicts. Each dict has `"task_id"` and `"train_info"` keys. |

---

#### `ContinualHarness`

```python
@dataclasses.dataclass
class ContinualHarness:
    task_sequence: list[str]
    adapter_bank: TaskAdapterBank
    distillation: BehaviorDistillation | None
    scorers: dict[str, Scorer]
    eval_episodes: dict[str, list[Episode]]
    distillation_alpha: float = 0.5
```

Testable orchestrator for sequential task training and evaluation. Delegates actual model training to a caller-provided `train_fn`, keeping the harness decoupled from JAX/openpi internals.

| Field | Type | Description |
|---|---|---|
| `task_sequence` | `list[str]` | Ordered list of task IDs to train. |
| `adapter_bank` | `TaskAdapterBank` | Registry for per-task LoRA adapters. |
| `distillation` | `BehaviorDistillation \| None` | Optional behavior distillation module. |
| `scorers` | `dict[str, Scorer]` | `{task_id: Scorer}` for evaluation. |
| `eval_episodes` | `dict[str, list[Episode]]` | `{task_id: [Episode, ...]}` for evaluation. |
| `distillation_alpha` | `float` | Blending weight stored for downstream scripts. Default: `0.5`. |

##### `ContinualHarness.trained_tasks` (property)

```python
@property
def trained_tasks(self) -> list[str]:
```

Task IDs that have been trained, in order.

---

##### `ContinualHarness.train_task(task_id, train_fn) -> list[dict]`

```python
def train_task(self, task_id: str, train_fn: TrainFn) -> list[dict]:
```

Train a single task using the caller-provided training function.

**Side effects:**
- Calls `train_fn(task_id)` to obtain trained model and info
- Registers the adapter in `adapter_bank`
- Freezes the adapter
- Updates teacher snapshot if distillation is configured
- Appends `task_id` to internal `_trained_tasks` list

| Parameter | Type | Description |
|---|---|---|
| `task_id` | `str` | Which task to train. |
| `train_fn` | `TrainFn` | `callable(task_id) -> (trained_model, training_info_list)`. |

**Returns:** Training info dicts from the training loop.

---

##### `ContinualHarness.evaluate_all_tasks() -> dict[str, float]`

```python
def evaluate_all_tasks(self) -> dict[str, float]:
```

Evaluate current model on all tasks seen so far using scorers on pre-collected episodes.

For each trained task: gets eval episodes, runs scorer on each, computes mean success rate.

> **Note:** This is a mock evaluation using scorers on pre-collected episodes. Live rollouts require a robot or simulation environment and are handled by external scripts.

**Returns:** `{task_id: mean_success_rate}` for all trained tasks.

---

##### `ContinualHarness.run_sequence(train_fn) -> ContinualResult`

```python
def run_sequence(self, train_fn: TrainFn) -> ContinualResult:
```

Run the full continual learning sequence.

For each task in `task_sequence`: trains the task, evaluates all tasks seen so far, and records results in a result matrix.

| Parameter | Type | Description |
|---|---|---|
| `train_fn` | `TrainFn` | Training function passed through to `train_task`. |

**Returns:** `ContinualResult` with the full `(T, T)` result matrix and per-step info.

**Example:**

```python
harness = ContinualHarness(
    task_sequence=["payload", "latch", "clean"],
    adapter_bank=TaskAdapterBank(),
    distillation=None,
    scorers={"payload": PayloadTransferScorer(), ...},
    eval_episodes={"payload": [...], ...},
)
result = harness.run_sequence(train_fn=my_train_fn)
print(result.result_matrix)
```

---

### Module: `metrics`

**Import:** `from openpi.research.spacecil.metrics import ...`

Mission-aware forgetting metrics. All functions are pure (no side effects), deterministic, and operate on numpy arrays.

**Result matrix convention:** `R[i][j]` = success rate on task `j` evaluated after training through task `i`. Shape `(T, T)`. Upper-triangle entries (`j > i`) are `NaN` for tasks not yet seen.

---

#### `average_success(result_matrix) -> float`

```python
def average_success(result_matrix: np.ndarray) -> float:
```

Mean success rate across all tasks after training on the final task.

**Formula:** `mean(R[-1, :])`

| Parameter | Type | Description |
|---|---|---|
| `result_matrix` | `np.ndarray` | Shape `(T, T)`. |

**Returns:** Scalar float.

---

#### `backward_transfer(result_matrix) -> float`

```python
def backward_transfer(result_matrix: np.ndarray) -> float:
```

Average change from just-trained performance to final performance.

Positive = improvement (positive transfer). Negative = degradation (forgetting).

**Formula:** `mean(R[j][j] - R[-1][j] for j in range(T-1))`

| Parameter | Type | Description |
|---|---|---|
| `result_matrix` | `np.ndarray` | Shape `(T, T)`. |

**Returns:** Scalar float. Returns `0.0` for single-task matrix (`T=1`).

---

#### `forgetting(result_matrix) -> float`

```python
def forgetting(result_matrix: np.ndarray) -> float:
```

Average drop from peak performance to final performance. Always non-negative.

**Formula:** `mean(max_k(R[k][j]) - R[-1][j] for j in range(T-1))`

| Parameter | Type | Description |
|---|---|---|
| `result_matrix` | `np.ndarray` | Shape `(T, T)`. |

**Returns:** Scalar float. Returns `0.0` for single-task matrix.

---

#### `operational_forgetting(result_matrix, weights) -> float`

```python
def operational_forgetting(result_matrix: np.ndarray, weights: np.ndarray) -> float:
```

Weighted forgetting with per-task operational importance.

**Formula:** `sum_j(w[j] * (max_k(R[k][j]) - R[-1][j])) / sum_j(w[j]) for j in range(T-1)`

Only the first `T-1` entries of `weights` are used (the last task has no forgetting by convention).

| Parameter | Type | Description |
|---|---|---|
| `result_matrix` | `np.ndarray` | Shape `(T, T)`. |
| `weights` | `np.ndarray` | Shape `(T,)` per-task operational importance. |

**Returns:** Weighted forgetting scalar. Returns `0.0` if `sum(weights[:T-1]) == 0` or `T <= 1`.

**Example:**

```python
# High priority on first two tasks
weights = np.array([2.0, 2.0, 1.0, 1.0])
op_forget = operational_forgetting(result_matrix, weights)
```

---

#### `routing_entropy(routing_probs) -> np.ndarray`

```python
def routing_entropy(routing_probs: np.ndarray) -> np.ndarray:
```

Shannon entropy of routing probability distributions.

**Formula:** `-sum(p * log(p + eps), axis=-1)` with `eps=1e-10` for numerical stability.

| Parameter | Type | Description |
|---|---|---|
| `routing_probs` | `np.ndarray` | Shape `(B, K)` or `(K,)` where `K` = number of tasks. Each row should sum to ~1. |

**Returns:** Entropy per sample — shape `(B,)` or scalar (0-d array). Higher entropy = more uncertain routing.

---

#### `routing_accuracy(predicted_tasks, true_tasks) -> float`

```python
def routing_accuracy(predicted_tasks: np.ndarray, true_tasks: np.ndarray) -> float:
```

Fraction of correctly predicted task assignments.

**Formula:** `mean(predicted == true)`

| Parameter | Type | Description |
|---|---|---|
| `predicted_tasks` | `np.ndarray` | Integer array of predicted task IDs. |
| `true_tasks` | `np.ndarray` | Integer array of ground-truth task IDs. Same shape as `predicted_tasks`. |

**Returns:** Scalar float in [0, 1].

---

### Module: `config` (SpaceCIL) {#module-config-spacecil}

**Import:** `from openpi.research.spacecil.config import get_spacecil_configs`

---

#### `get_spacecil_configs() -> list[TrainConfig]`

```python
def get_spacecil_configs() -> list[TrainConfig]:
```

Return SpaceCIL training configs for registration in openpi's config system.

Produces **5 configs** total:

| Config name | Description |
|---|---|
| `spacecil_rm75_payload` | Payload transfer task, pi0.5 + LoRA, 10k steps |
| `spacecil_rm75_latch` | Latch actuation task, pi0.5 + LoRA, 10k steps |
| `spacecil_rm75_clean` | Surface cleaning task, pi0.5 + LoRA, 10k steps |
| `spacecil_rm75_connector` | Connector mating task, pi0.5 + LoRA, 10k steps |
| `spacecil_debug` | Dummy model + FakeDataConfig for fast unit tests |

All production configs use:
- `Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`
- `LeRobotRM75DataConfig` with `prompt_from_task=True`
- `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
- `batch_size=32`, `ema_decay=None`, `wandb_enabled=True`

**Re-entrancy guard:** A module-level guard prevents circular imports when `training/config.py` builds `_CONFIGS`. Nested calls return `[]` safely. Results are cached after the first successful call.

**Example (registration in `training/config.py`):**

```python
from openpi.research.spacecil import config as spacecil_config
_CONFIGS = [
    ...,
    *spacecil_config.get_spacecil_configs(),
]
```

---

## Package: `openpi.research.lunarcompose`

LunarCompose (Paper B) modules for factorized task × environment adaptation. Depends on SpaceCIL infrastructure.

---

### Module: `env_adapter_bank`

**Import:** `from openpi.research.lunarcompose.env_adapter_bank import ...`

Visual-path environment adapters. Analogous to `TaskAdapterBank` but targets vision encoder layers (SigLIP by default) and supports a prefix fallback mode.

---

#### `EnvAdapterBank`

```python
class EnvAdapterBank:
    def __init__(
        self,
        lora_target: str = ".*siglip.*",
        fallback_prefix_mode: bool = False,
    ) -> None:
```

Versioned registry mapping environment IDs to LoRA parameter states (or prefix strings in fallback mode).

Unlike `TaskAdapterBank` (which uses a fixed `.*lora.*` filter), `EnvAdapterBank` accepts a configurable `lora_target` regex so environment adapters can target the SigLIP vision encoder specifically.

**Composition rule:** `effective_weights = base + task_lora_delta + env_lora_delta`. Both deltas are applied via sequential `nnx.update` calls, outside JIT.

| Parameter | Type | Description |
|---|---|---|
| `lora_target` | `str` | Regex matching parameters to snapshot for environment adapters. Default: `".*siglip.*"`. |
| `fallback_prefix_mode` | `bool` | If `True`, stores prefix strings instead of LoRA weights. Default: `False`. |

**Usage (LoRA mode):**

```python
bank = EnvAdapterBank()
bank.register_env("nominal", model)
bank.freeze_env("nominal")
bank.merge_into_model(model, "nominal")
bank.save("/tmp/env_bank")
bank = EnvAdapterBank.load("/tmp/env_bank")
```

**Usage (prefix fallback mode):**

```python
bank = EnvAdapterBank(fallback_prefix_mode=True)
bank.register_env("nominal", prefix="lunar nominal lighting")
prefix = bank.get_prefix("nominal")
```

---

##### `EnvAdapterBank.lora_target` (property)

```python
@property
def lora_target(self) -> str:
```

The regex pattern used to filter LoRA parameters.

---

##### `EnvAdapterBank.fallback_prefix_mode` (property)

```python
@property
def fallback_prefix_mode(self) -> bool:
```

Whether the bank operates in prefix fallback mode.

---

##### `EnvAdapterBank.num_adapters` (property)

```python
@property
def num_adapters(self) -> int:
```

Number of registered environment adapters (LoRA or prefix count, depending on mode).

---

##### `EnvAdapterBank.registered_envs` (property)

```python
@property
def registered_envs(self) -> list[str]:
```

Environment IDs in registration order.

---

##### `EnvAdapterBank.list_envs() -> list[str]`

```python
def list_envs(self) -> list[str]:
```

Return environment IDs in registration order. Alias for `registered_envs`.

---

##### `EnvAdapterBank.register_env(env_id, model, *, prefix)`

```python
def register_env(
    self,
    env_id: str,
    model: nnx.Module | None = None,
    *,
    prefix: str | None = None,
) -> None:
```

Register an environment adapter.

- **LoRA mode:** extracts matching parameters from `model` and stores them.
- **Prefix mode:** stores the `prefix` string for environment conditioning.

| Parameter | Type | Description |
|---|---|---|
| `env_id` | `str` | Unique identifier for this environment adapter. |
| `model` | `nnx.Module \| None` | Required in LoRA mode. Ignored in prefix fallback mode. |
| `prefix` | `str \| None` | Required in prefix fallback mode. Ignored in LoRA mode. Keyword-only. |

**Raises:**
- `ValueError` if `env_id` is already frozen
- `ValueError` if `model` is `None` in LoRA mode
- `ValueError` if `prefix` is `None` in prefix mode

---

##### `EnvAdapterBank.register_env_from_state(env_id, lora_state)`

```python
def register_env_from_state(self, env_id: str, lora_state: nnx.State) -> None:
```

Store adapter from an existing `nnx.State`.

| Parameter | Type | Description |
|---|---|---|
| `env_id` | `str` | Unique identifier for this environment adapter. |
| `lora_state` | `nnx.State` | Pre-filtered LoRA state containing matching parameters. |

**Raises:** `ValueError` if `env_id` is already frozen.

---

##### `EnvAdapterBank.get_env(env_id) -> dict`

```python
def get_env(self, env_id: str) -> dict:
```

Return the stored pure dict for an environment adapter (LoRA mode only).

**Raises:**
- `KeyError` if `env_id` is not registered
- `RuntimeError` if bank is in prefix fallback mode

---

##### `EnvAdapterBank.get_prefix(env_id) -> str`

```python
def get_prefix(self, env_id: str) -> str:
```

Return the stored prefix string for an environment (prefix mode only).

**Raises:**
- `KeyError` if `env_id` is not registered
- `RuntimeError` if bank is not in prefix fallback mode

---

##### `EnvAdapterBank.merge_into_model(model, env_id)`

```python
def merge_into_model(self, model: nnx.Module, env_id: str) -> None:
```

Apply stored LoRA weights to a live model via `nnx.update`.

| Parameter | Type | Description |
|---|---|---|
| `model` | `nnx.Module` | The live model to update in-place. |
| `env_id` | `str` | Which adapter's weights to inject. |

**Raises:**
- `KeyError` if `env_id` is not registered
- `RuntimeError` if bank is in prefix fallback mode

> **Warning:** Must be called **outside JIT boundaries** to avoid JAX recompilation.

---

##### `EnvAdapterBank.freeze_env(env_id)`

```python
def freeze_env(self, env_id: str) -> None:
```

Mark an adapter as immutable.

**Raises:** `KeyError` if `env_id` is not registered.

---

##### `EnvAdapterBank.is_frozen(env_id) -> bool`

```python
def is_frozen(self, env_id: str) -> bool:
```

Check whether an adapter is frozen.

---

##### `EnvAdapterBank.save(path)`

```python
def save(self, path: str | Path) -> None:
```

Save adapter bank to disk.

**LoRA mode creates:**
- `path/metadata.json` — config, frozen set, registration order
- `path/{env_id}/adapter.npz` — flattened LoRA arrays per env

**Prefix mode creates:**
- `path/metadata.json` — includes `"prefixes"` dict inline

---

##### `EnvAdapterBank.load(path) -> EnvAdapterBank`

```python
@classmethod
def load(cls, path: str | Path) -> EnvAdapterBank:
```

Load adapter bank from disk. Automatically detects LoRA vs. prefix mode from `metadata.json`.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Directory previously written by `save()`. |

**Returns:** A new `EnvAdapterBank` with all adapters and frozen state restored.

---

### Module: `dual_head_router`

**Import:** `from openpi.research.lunarcompose.dual_head_router import ...`

Two-headed router producing independent task and environment routing distributions.

---

#### `DualHeadRouter`

```python
class DualHeadRouter(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        max_tasks: int = 16,
        max_envs: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
```

Two-headed router with independent task and environment routing heads. Both heads receive the same concatenated `(lang_embedding, visual_summary)` input but maintain completely separate parameters.

| Parameter | Type | Description |
|---|---|---|
| `input_dim` | `int` | Dimensionality of `concat(lang_embedding, visual_summary)`. |
| `max_tasks` | `int` | Pre-allocated task-adapter bank capacity. Default: `16`. |
| `max_envs` | `int` | Pre-allocated environment-adapter bank capacity. Default: `8`. |
| `hidden_dim` | `int` | Hidden size for each MLP block inside both heads. Default: `256`. |
| `num_layers` | `int` | Number of `(Linear -> LayerNorm -> GELU)` blocks per head. Default: `2`. |
| `rngs` | `nnx.Rngs` | Flax NNX random number generators. Keyword-only. |

**Architecture per head:** `concat(lang, vis) -> [Linear -> LayerNorm -> GELU] x num_layers -> Linear -> masked softmax`

Heads are independent `TaskRouter` instances stored as `self.task_head` and `self.env_head`.

---

##### `DualHeadRouter.__call__(lang_embedding, visual_summary, task_mask, env_mask)`

```python
def __call__(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    task_mask: jax.Array | None = None,
    env_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
```

Forward pass returning softmax routing probabilities for both heads.

| Parameter | Type | Description |
|---|---|---|
| `lang_embedding` | `jax.Array` | Shape `(B, lang_dim)`. |
| `visual_summary` | `jax.Array` | Shape `(B, vis_dim)`. |
| `task_mask` | `jax.Array \| None` | Boolean `(max_tasks,)`. `None` = all active. |
| `env_mask` | `jax.Array \| None` | Boolean `(max_envs,)`. `None` = all active. |

**Returns:** `(task_probs, env_probs)` where `task_probs` has shape `(B, max_tasks)` and `env_probs` has shape `(B, max_envs)`.

---

##### `DualHeadRouter.route(lang_embedding, visual_summary, task_ids, env_ids, task_mask, env_mask)`

```python
def route(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    task_ids: list[str],
    env_ids: list[str],
    task_mask: jax.Array | None = None,
    env_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
```

Argmax routing for both heads.

`task_ids` and `env_ids` are accepted for interface compatibility (e.g. logging) but are not used for routing computation.

**Returns:** `(task_indices, env_indices)` — integer arrays of shape `(B,)`.

---

##### `DualHeadRouter.route_soft(lang_embedding, visual_summary, task_mask, env_mask)`

```python
def route_soft(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    task_mask: jax.Array | None = None,
    env_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
```

Alias for `__call__` — returns soft routing probability distributions.

**Returns:** `(task_probs, env_probs)`.

---

##### `DualHeadRouter.counterfactual_task(lang_embedding, visual_summary, task_mask)`

```python
def counterfactual_task(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    task_mask: jax.Array | None = None,
) -> jax.Array:
```

Route the task head only, ignoring the environment head. Useful for ablation studies.

**Returns:** Task routing probabilities of shape `(B, max_tasks)`.

---

##### `DualHeadRouter.counterfactual_env(lang_embedding, visual_summary, env_mask)`

```python
def counterfactual_env(
    self,
    lang_embedding: jax.Array,
    visual_summary: jax.Array,
    env_mask: jax.Array | None = None,
) -> jax.Array:
```

Route the environment head only, ignoring the task head. Useful for ablation studies.

**Returns:** Environment routing probabilities of shape `(B, max_envs)`.

---

##### `DualHeadRouter.mutual_information_estimate(lang_embeddings, visual_summaries, task_mask, env_mask)`

```python
def mutual_information_estimate(
    self,
    lang_embeddings: jax.Array,
    visual_summaries: jax.Array,
    task_mask: jax.Array | None = None,
    env_mask: jax.Array | None = None,
) -> jax.Array:
```

Estimate mutual information between task and environment routing distributions over a batch.

**Formula:**

```
P(t, e) = (1/B) * sum_b [ p_task_b(t) * p_env_b(e) ]
MI = sum_{t,e} P(t,e) * log( P(t,e) / (P(t)*P(e) + eps) + eps )
```

Lower MI = more factorized (independent) routing. Zero = fully independent.

| Parameter | Type | Description |
|---|---|---|
| `lang_embeddings` | `jax.Array` | `(B, lang_dim)` batch of language embeddings. |
| `visual_summaries` | `jax.Array` | `(B, vis_dim)` batch of visual summaries. |
| `task_mask` | `jax.Array \| None` | Optional `(max_tasks,)` boolean mask. |
| `env_mask` | `jax.Array \| None` | Optional `(max_envs,)` boolean mask. |

**Returns:** Scalar `jax.Array` — estimated mutual information in nats.

**Example:**

```python
router = DualHeadRouter(input_dim=512, max_tasks=4, max_envs=3, rngs=nnx.Rngs(0))
task_probs, env_probs = router(lang_emb, vis_emb)
mi = router.mutual_information_estimate(lang_embs_batch, vis_embs_batch)
task_idx, env_idx = router.route(lang_emb, vis_emb, task_ids=[], env_ids=[])
```

---

### Module: `missing_corner_harness`

**Import:** `from openpi.research.lunarcompose.missing_corner_harness import ...`

Scientific centerpiece of Paper B. Manages a 2D grid of task × environment cells, partitions them into seen (train) and unseen (test) subsets, and evaluates compositional generalization.

**Grid definition:** 4 tasks (`payload`, `latch`, `clean`, `connector`) × 3 environments (`nominal`, `shadow`, `contamination`) = 12 cells. Three canonical split rotations are hardcoded for reproducibility.

---

#### `MissingCornerResult`

```python
@dataclasses.dataclass(frozen=True)
class MissingCornerResult:
    per_cell_success: dict[tuple[str, str], float]
    seen_mean: float
    unseen_mean: float
    seen_unseen_gap: float
    per_task_breakdown: dict[str, float]
    per_env_breakdown: dict[str, float]
    rotation_id: int
    timestamp: str
```

Result of missing-corner compositional evaluation.

| Field | Type | Description |
|---|---|---|
| `per_cell_success` | `dict[tuple[str, str], float]` | `(task_id, env_id) -> success_rate` for all evaluated cells. |
| `seen_mean` | `float` | Mean success rate over training (seen) cells. |
| `unseen_mean` | `float` | Mean success rate over test (unseen) cells. |
| `seen_unseen_gap` | `float` | `seen_mean - unseen_mean`. Primary generalization metric. |
| `per_task_breakdown` | `dict[str, float]` | `task_id -> mean success across all envs`. |
| `per_env_breakdown` | `dict[str, float]` | `env_id -> mean success across all tasks`. |
| `rotation_id` | `int` | Which split rotation (0, 1, or 2). |
| `timestamp` | `str` | ISO format UTC timestamp of evaluation. |

---

#### `MissingCornerHarness`

```python
@dataclasses.dataclass
class MissingCornerHarness:
    task_ids: list[str]
    env_ids: list[str]
    scorers: dict[str, Scorer]
    eval_episodes: dict[tuple[str, str], list[Episode]]
    train_cells: set[tuple[str, str]] = ...
    test_cells: set[tuple[str, str]] = ...
```

Manages train/test splits over a task × environment grid.

| Field | Type | Description |
|---|---|---|
| `task_ids` | `list[str]` | Ordered list of task IDs. |
| `env_ids` | `list[str]` | Ordered list of environment IDs. |
| `scorers` | `dict[str, Scorer]` | `{task_id: Scorer}` for evaluation. |
| `eval_episodes` | `dict[tuple[str, str], list[Episode]]` | `{(task_id, env_id): [Episode, ...]}`. |
| `train_cells` | `set[tuple[str, str]]` | Cells used for training. Set by `generate_split`. |
| `test_cells` | `set[tuple[str, str]]` | Cells held out for evaluation. Set by `generate_split`. |

**Canonical splits (hardcoded for reproducibility):**

| Rotation | Test cells (held out) |
|---|---|
| 0 | `(payload, contamination)`, `(latch, shadow)`, `(clean, nominal)`, `(connector, shadow)` |
| 1 | `(payload, nominal)`, `(latch, contamination)`, `(clean, shadow)`, `(connector, contamination)` |
| 2 | `(payload, shadow)`, `(latch, nominal)`, `(clean, contamination)`, `(connector, nominal)` |

---

##### `MissingCornerHarness.all_cells` (property)

```python
@property
def all_cells(self) -> set[tuple[str, str]]:
```

All possible `(task_id, env_id)` cells in the grid.

---

##### `MissingCornerHarness.generate_split(rotation) -> tuple[set, set]`

```python
def generate_split(self, rotation: int = 0) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
```

Generate train/test split for the given rotation. Sets `self.train_cells` and `self.test_cells`.

| Parameter | Type | Description |
|---|---|---|
| `rotation` | `int` | Split rotation index. Must be 0, 1, or 2. Default: `0`. |

**Returns:** `(train_cells, test_cells)` as sets of `(task_id, env_id)` tuples.

**Raises:**
- `ValueError` if `rotation` not in `{0, 1, 2}`
- `ValueError` if the generated split fails validation

**Side effects:** Calls `validate_split()` internally.

---

##### `MissingCornerHarness.validate_split() -> bool`

```python
def validate_split(self) -> bool:
```

Validate the current train/test split against four constraints:

1. Every `task_id` appears in at least one training cell.
2. Every `env_id` appears in at least one training cell.
3. No overlap between `train_cells` and `test_cells`.
4. `train_cells ∪ test_cells` covers all cells (complete partition).

**Returns:** `True` if all constraints pass.

**Raises:** `ValueError` with a specific message if any constraint fails.

---

##### `MissingCornerHarness.seen_cells() -> set[tuple[str, str]]`

```python
def seen_cells(self) -> set[tuple[str, str]]:
```

Alias for `train_cells`.

---

##### `MissingCornerHarness.unseen_cells() -> set[tuple[str, str]]`

```python
def unseen_cells(self) -> set[tuple[str, str]]:
```

Alias for `test_cells`.

---

##### `MissingCornerHarness.evaluate_all_cells() -> MissingCornerResult`

```python
def evaluate_all_cells(self) -> MissingCornerResult:
```

Evaluate current model on all cells in the grid. For each cell: gets episodes, runs scorer, computes mean success rate.

Cells with no episodes default to `0.0`. Cells with no scorer log a warning and default to `0.0`.

**Returns:** `MissingCornerResult` with per-cell scores and aggregate metrics.

**Raises:** `RuntimeError` if `generate_split()` has not been called yet.

**Example:**

```python
harness = MissingCornerHarness(
    task_ids=["payload", "latch", "clean", "connector"],
    env_ids=["nominal", "shadow", "contamination"],
    scorers={"payload": PayloadTransferScorer(), ...},
    eval_episodes={("payload", "nominal"): [...], ...},
)
train_cells, test_cells = harness.generate_split(rotation=0)
result = harness.evaluate_all_cells()
print(f"Seen: {result.seen_mean:.3f}, Unseen: {result.unseen_mean:.3f}, Gap: {result.seen_unseen_gap:.3f}")
```

---

### Module: `factorization_diagnostics`

**Import:** `from openpi.research.lunarcompose.factorization_diagnostics import ...`

Five diagnostic functions assessing whether the task-environment factorization assumption holds. All are pure functions (no side effects).

**Interpretation guide:**

| Metric | Factorization holds | Factorization fails |
|---|---|---|
| `seen_unseen_gap` | < 0.1 | > 0.3 |
| `routing_interaction_analysis` (MI) | ≈ 0 | Large positive |
| `task_env_entanglement` (cosine sim) | ≈ 0 | High absolute value |
| `counterfactual_swap_test` mean delta | ≈ 0 | Large positive or negative |

---

#### `seen_unseen_gap(results) -> float`

```python
def seen_unseen_gap(results: Any) -> float:
```

Primary factorization metric: gap between seen and unseen success rates.

| Parameter | Type | Description |
|---|---|---|
| `results` | `Any` | Object with `seen_mean: float` and `unseen_mean: float` attributes. Typically a `MissingCornerResult`. |

**Returns:** `results.seen_mean - results.unseen_mean`. Positive = unseen combinations harder.

---

#### `cross_condition_breakdown(results) -> dict[str, dict[str, float]]`

```python
def cross_condition_breakdown(results: Any) -> dict[str, dict[str, float]]:
```

Decompose results by task and environment to find performance bottlenecks.

| Parameter | Type | Description |
|---|---|---|
| `results` | `Any` | Object with `per_task_breakdown: dict[str, float]` and `per_env_breakdown: dict[str, float]`. |

**Returns:** `{"per_task": results.per_task_breakdown, "per_env": results.per_env_breakdown}`.

---

#### `routing_interaction_analysis(task_probs, env_probs) -> float`

```python
def routing_interaction_analysis(
    task_probs: jax.Array,
    env_probs: jax.Array,
) -> float:
```

Estimate mutual information between task and environment routing distributions.

Operates on raw probability arrays — testable without constructing a full `DualHeadRouter`.

**Formula:**
```
joint(t,e) = mean_b [ task_probs_b(t) * env_probs_b(e) ]
MI = sum_{t,e} joint(t,e) * log( joint(t,e) / (p_task(t) * p_env(e) + 1e-10) + 1e-10 )
```

| Parameter | Type | Description |
|---|---|---|
| `task_probs` | `jax.Array` | Shape `(B, num_tasks)` softmax routing probabilities. |
| `env_probs` | `jax.Array` | Shape `(B, num_envs)` softmax routing probabilities. |

**Returns:** Non-negative float scalar — estimated MI in nats. Zero = fully independent routing.

---

#### `task_env_entanglement(task_params, env_params) -> float`

```python
def task_env_entanglement(
    task_params: dict,
    env_params: dict,
) -> float:
```

Cosine similarity between flattened task and environment adapter parameters.

Near zero = orthogonal adapter sets (ideal for factorization). High absolute value = entanglement.

If the two parameter dicts have different total sizes, the shorter vector is zero-padded.

| Parameter | Type | Description |
|---|---|---|
| `task_params` | `dict` | Nested dict of numpy-compatible arrays (e.g. from `task_bank.get_adapter(task_id)`). |
| `env_params` | `dict` | Nested dict of numpy-compatible arrays (e.g. from `env_bank.get_env(env_id)`). |

**Returns:** Cosine similarity in `[-1, 1]`.

**Example:**

```python
entanglement = task_env_entanglement(
    task_bank.get_adapter("payload"),
    env_bank.get_env("nominal"),
)
```

---

#### `counterfactual_swap_test(original_scores, swapped_scores) -> dict[str, float]`

```python
def counterfactual_swap_test(
    original_scores: dict[tuple[str, str], float],
    swapped_scores: dict[tuple[str, str], float],
) -> dict[str, float]:
```

Compare per-cell scores before and after swapping one adaptation factor.

If factorization holds, swapping (for example) the env adapter while keeping the task adapter should not degrade performance. Deltas are `swapped - original`. `max_delta` and `min_delta` are over *absolute* deltas.

| Parameter | Type | Description |
|---|---|---|
| `original_scores` | `dict[tuple[str, str], float]` | `{(task_id, env_id): success_rate}` before swap. |
| `swapped_scores` | `dict[tuple[str, str], float]` | `{(task_id, env_id): success_rate}` after one factor was replaced. |

**Returns:** `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `"mean_delta"` | `float` | Mean of `(swapped - original)` over common cells. |
| `"max_delta"` | `float` | Max of absolute deltas. |
| `"min_delta"` | `float` | Min of absolute deltas. |
| `"num_cells"` | `float` | Number of common cells evaluated. |

When no common cells exist, all values are `0.0`.

---

### Module: `config` (LunarCompose) {#module-config-lunarcompose}

**Import:** `from openpi.research.lunarcompose.config import get_lunarcompose_configs`

---

#### `get_lunarcompose_configs() -> list[TrainConfig]`

```python
def get_lunarcompose_configs() -> list[TrainConfig]:
```

Return LunarCompose training configs for registration in openpi's config system.

Produces **15 configs** total:

**12 per-cell configs** (4 tasks × 3 environments):

| Config name | Task | Environment |
|---|---|---|
| `lunarcompose_payload_nominal` | payload | nominal |
| `lunarcompose_payload_shadow` | payload | shadow |
| `lunarcompose_payload_contamination` | payload | contamination |
| `lunarcompose_latch_nominal` | latch | nominal |
| `lunarcompose_latch_shadow` | latch | shadow |
| `lunarcompose_latch_contamination` | latch | contamination |
| `lunarcompose_clean_nominal` | clean | nominal |
| `lunarcompose_clean_shadow` | clean | shadow |
| `lunarcompose_clean_contamination` | clean | contamination |
| `lunarcompose_connector_nominal` | connector | nominal |
| `lunarcompose_connector_shadow` | connector | shadow |
| `lunarcompose_connector_contamination` | connector | contamination |

**2 architecture-level configs:**

| Config name | Description |
|---|---|
| `lunarcompose_factorized` | Full factorized training, 20k steps |
| `lunarcompose_monolithic` | Monolithic baseline, 20k steps |

**1 debug config:**

| Config name | Description |
|---|---|
| `lunarcompose_debug` | Dummy model + FakeDataConfig, 10 steps, no W&B |

All production configs use:
- `Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`
- `LeRobotRM75DataConfig` with `prompt_from_task=True`
- `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
- `batch_size=32`, `ema_decay=None`, `wandb_enabled=True`

The same re-entrancy guard as `get_spacecil_configs` prevents circular import issues during `_CONFIGS` construction.

**Example (registration in `training/config.py`):**

```python
from openpi.research.lunarcompose import config as lunarcompose_config
_CONFIGS = [
    ...,
    *spacecil_config.get_spacecil_configs(),
    *lunarcompose_config.get_lunarcompose_configs(),
]
```

---

## Cross-Reference Index

Alphabetical index of all public classes and functions.

| Name | Module | Kind |
|---|---|---|
| `Action` | `shared.episode_schema` | dataclass |
| `Action.from_array` | `shared.episode_schema` | classmethod |
| `Action.to_array` | `shared.episode_schema` | method |
| `ACTION_DIM` | `shared.action_transforms` | constant |
| `average_success` | `spacecil.metrics` | function |
| `backward_transfer` | `spacecil.metrics` | function |
| `BehaviorDistillation` | `spacecil.behavior_distillation` | dataclass |
| `BehaviorDistillation.add_calibration_episodes` | `spacecil.behavior_distillation` | method |
| `BehaviorDistillation.compute_total_loss` | `spacecil.behavior_distillation` | method |
| `BehaviorDistillation.update_teacher` | `spacecil.behavior_distillation` | method |
| `CalibrationMemory` | `spacecil.behavior_distillation` | dataclass |
| `CalibrationMemory.add_episodes` | `spacecil.behavior_distillation` | method |
| `CalibrationMemory.num_episodes` | `spacecil.behavior_distillation` | method |
| `CalibrationMemory.sample_batch` | `spacecil.behavior_distillation` | method |
| `CalibrationMemory.task_ids` | `spacecil.behavior_distillation` | property |
| `canonical_to_training` | `shared.action_transforms` | function |
| `ConnectorMatingScorer` | `shared.scorer_base` | class |
| `ConnectorMatingScorer.score` | `shared.scorer_base` | method |
| `ContinualHarness` | `spacecil.continual_harness` | dataclass |
| `ContinualHarness.evaluate_all_tasks` | `spacecil.continual_harness` | method |
| `ContinualHarness.run_sequence` | `spacecil.continual_harness` | method |
| `ContinualHarness.train_task` | `spacecil.continual_harness` | method |
| `ContinualHarness.trained_tasks` | `spacecil.continual_harness` | property |
| `ContinualResult` | `spacecil.continual_harness` | dataclass |
| `cross_condition_breakdown` | `lunarcompose.factorization_diagnostics` | function |
| `counterfactual_swap_test` | `lunarcompose.factorization_diagnostics` | function |
| `distillation_loss` | `spacecil.behavior_distillation` | function |
| `DualHeadRouter` | `lunarcompose.dual_head_router` | class (nnx.Module) |
| `DualHeadRouter.counterfactual_env` | `lunarcompose.dual_head_router` | method |
| `DualHeadRouter.counterfactual_task` | `lunarcompose.dual_head_router` | method |
| `DualHeadRouter.mutual_information_estimate` | `lunarcompose.dual_head_router` | method |
| `DualHeadRouter.route` | `lunarcompose.dual_head_router` | method |
| `DualHeadRouter.route_soft` | `lunarcompose.dual_head_router` | method |
| `EnvAdapterBank` | `lunarcompose.env_adapter_bank` | class |
| `EnvAdapterBank.freeze_env` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.get_env` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.get_prefix` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.is_frozen` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.list_envs` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.load` | `lunarcompose.env_adapter_bank` | classmethod |
| `EnvAdapterBank.merge_into_model` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.num_adapters` | `lunarcompose.env_adapter_bank` | property |
| `EnvAdapterBank.register_env` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.register_env_from_state` | `lunarcompose.env_adapter_bank` | method |
| `EnvAdapterBank.registered_envs` | `lunarcompose.env_adapter_bank` | property |
| `EnvAdapterBank.save` | `lunarcompose.env_adapter_bank` | method |
| `Episode` | `shared.episode_schema` | dataclass |
| `Episode.from_dict` | `shared.episode_schema` | classmethod |
| `Episode.to_dict` | `shared.episode_schema` | method |
| `EpisodeLabels` | `shared.episode_schema` | dataclass |
| `EpisodeMetadata` | `shared.episode_schema` | dataclass |
| `EpisodeStep` | `shared.episode_schema` | dataclass |
| `forgetting` | `spacecil.metrics` | function |
| `get_lunarcompose_configs` | `lunarcompose.config` | function |
| `get_spacecil_configs` | `spacecil.config` | function |
| `GRIPPER_RANGE` | `shared.action_transforms` | constant |
| `JOINT_DIM` | `shared.action_transforms` | constant |
| `LatchActuationScorer` | `shared.scorer_base` | class |
| `LatchActuationScorer.score` | `shared.scorer_base` | method |
| `LeRobotRM75DataConfig` | `shared.rm75_policy` | dataclass |
| `LeRobotRM75DataConfig.create` | `shared.rm75_policy` | method |
| `LORA_FILTER` | `spacecil.task_adapter_bank` | constant |
| `make_active_mask` | `spacecil.router` | function |
| `make_repack_structure` | `shared.episode_schema` | function |
| `make_rm75_example` | `shared.rm75_policy` | function |
| `MissingCornerHarness` | `lunarcompose.missing_corner_harness` | dataclass |
| `MissingCornerHarness.all_cells` | `lunarcompose.missing_corner_harness` | property |
| `MissingCornerHarness.evaluate_all_cells` | `lunarcompose.missing_corner_harness` | method |
| `MissingCornerHarness.generate_split` | `lunarcompose.missing_corner_harness` | method |
| `MissingCornerHarness.seen_cells` | `lunarcompose.missing_corner_harness` | method |
| `MissingCornerHarness.unseen_cells` | `lunarcompose.missing_corner_harness` | method |
| `MissingCornerHarness.validate_split` | `lunarcompose.missing_corner_harness` | method |
| `MissingCornerResult` | `lunarcompose.missing_corner_harness` | dataclass |
| `Observation` | `shared.episode_schema` | dataclass |
| `operational_forgetting` | `spacecil.metrics` | function |
| `PayloadTransferScorer` | `shared.scorer_base` | class |
| `PayloadTransferScorer.score` | `shared.scorer_base` | method |
| `RM75AbsoluteActions` | `shared.action_transforms` | dataclass |
| `RM75DeltaActions` | `shared.action_transforms` | dataclass |
| `RM75_DELTA_MASK` | `shared.action_transforms` | constant |
| `RM75Inputs` | `shared.rm75_policy` | dataclass |
| `RM75Outputs` | `shared.rm75_policy` | dataclass |
| `routing_accuracy` | `spacecil.metrics` | function |
| `routing_entropy` | `spacecil.metrics` | function |
| `routing_interaction_analysis` | `lunarcompose.factorization_diagnostics` | function |
| `Scorer` | `shared.scorer_base` | abstract class |
| `Scorer.score` | `shared.scorer_base` | abstract method |
| `ScorerResult` | `shared.scorer_base` | dataclass |
| `seen_unseen_gap` | `lunarcompose.factorization_diagnostics` | function |
| `SurfaceCleaningScorer` | `shared.scorer_base` | class |
| `SurfaceCleaningScorer.score` | `shared.scorer_base` | method |
| `task_env_entanglement` | `lunarcompose.factorization_diagnostics` | function |
| `TaskAdapterBank` | `spacecil.task_adapter_bank` | class |
| `TaskAdapterBank.freeze_adapter` | `spacecil.task_adapter_bank` | method |
| `TaskAdapterBank.get_adapter` | `spacecil.task_adapter_bank` | method |
| `TaskAdapterBank.is_frozen` | `spacecil.task_adapter_bank` | method |
| `TaskAdapterBank.load` | `spacecil.task_adapter_bank` | classmethod |
| `TaskAdapterBank.merge_into_model` | `spacecil.task_adapter_bank` | method |
| `TaskAdapterBank.num_adapters` | `spacecil.task_adapter_bank` | property |
| `TaskAdapterBank.register_adapter` | `spacecil.task_adapter_bank` | method |
| `TaskAdapterBank.register_adapter_from_state` | `spacecil.task_adapter_bank` | method |
| `TaskAdapterBank.registered_tasks` | `spacecil.task_adapter_bank` | property |
| `TaskAdapterBank.save` | `spacecil.task_adapter_bank` | method |
| `TaskRouter` | `spacecil.router` | class (nnx.Module) |
| `TaskRouter.route_hard` | `spacecil.router` | method |
| `TaskRouter.route_soft` | `spacecil.router` | method |
| `TeacherSnapshot` | `spacecil.behavior_distillation` | dataclass |
| `TeacherSnapshot.get_params` | `spacecil.behavior_distillation` | method |
| `TeacherSnapshot.has_snapshot` | `spacecil.behavior_distillation` | property |
| `TeacherSnapshot.snapshot` | `spacecil.behavior_distillation` | method |
| `teleop_to_canonical` | `shared.action_transforms` | function |
| `TrainFn` | `spacecil.continual_harness` | type alias |
| `training_to_canonical` | `shared.action_transforms` | function |
