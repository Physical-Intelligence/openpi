# Shared Infrastructure Plan

Phase A work. Both papers depend on every module here. Nothing in SpaceCIL or LunarCompose can start until this phase is complete and G1 is passed.

---

## 1. Overview

The shared infrastructure provides four capabilities that both papers share:

1. A frozen canonical episode schema that both papers read and write
2. One authoritative action transform path from teleop to training format and back
3. An RM75 platform policy configuration that plugs into openpi's training pipeline
4. A scorer base interface with per-task implementations for automated evaluation

These modules are leaf-level infrastructure. They must be stable before any training experiments. If they change after training begins, all downstream results are suspect.

**Target location:** `src/openpi/research/shared/`

**Run tests with:** `uv run pytest src/openpi/research/shared/ -x -q`

---

## 2. Module: `episode_schema.py`

### Purpose

Defines the one canonical episode format for both papers. There must be no alternative schema elsewhere in the codebase. Every loader, transform, scorer, and harness reads from this format.

### Fields

The schema is organized into five namespaces:

**Observations (`obs.*`)**
- `obs.wrist_rgb` — primary policy camera frame, hand-eye calibrated wrist RGB
- `obs.scene_rgb` — optional; used for scoring, replay, and debug only
- `obs.q` — joint positions
- `obs.dq` — joint velocities
- `obs.gripper` — gripper state
- `obs.base_state` — base position/heading, if logged

**Actions (`action.*`)**
- `action.joint_pos` — 7D absolute joint position (radians, 7-DoF arm)
- `action.gripper_cmd` — 1D gripper command

**Language (`lang.*`)**
- `lang.instruction` — natural language task instruction string

**Labels (`label.*`)**
- `label.success` — bool
- `label.fail_type` — optional string, one of a fixed enum (contact, timeout, drop, other)

**Metadata (`meta.*`)**
- `meta.task_id` — string identifier for task
- `meta.env_id` — string identifier for environment condition (used heavily by LunarCompose)
- `meta.operator_id` — who collected the demo
- `meta.session_id` — session-level identifier for grouping episodes
- `meta.camera_preset_id` — camera configuration at collection time
- `meta.calibration_version` — hand-eye calibration version hash
- `meta.scene_revision` — scene physical setup version
- `meta.object_revision` — object set version

### Implementation

Python frozen dataclasses. The dataclass hierarchy mirrors the namespace structure above: `EpisodeSchema` contains `ObsFields`, `ActionFields`, `LangFields`, `LabelFields`, `MetaFields`.

The schema must be frozen and versioned. A version string is embedded in `MetaFields`. Any breaking field change requires a new version, not silent modification.

Integration with openpi's training pipeline: a `RepackTransform` reads from the canonical schema fields and maps them to the key names openpi's model expects. The canonical schema is never passed directly to the model.

### Tests (`episode_schema_test.py`)

- Schema instantiation with all required fields
- Frozen validation: assert that mutation raises an error
- Serialization round-trip: schema to dict and back
- Missing required field: assert that creation raises
- Optional field handling: `obs.scene_rgb` and `obs.base_state` absent by default

---

## 3. Module: `action_transforms.py`

### Purpose

One authoritative path for action representation across the entire program. Three directions must work correctly:

1. `teleop_to_canonical()` — controller output to canonical absolute joint position + gripper
2. `canonical_to_training()` — canonical form to the representation openpi's model expects
3. `training_to_canonical()` — inverse of the above, for debugging and policy output interpretation

There must be no second converter anywhere. Any discrepancy between training-time and inference-time action representations is a fatal error for the experiment.

### Implementation

Follows openpi's `DataTransformFn` interface, the same pattern used by `DeltaActions` in `openpi/transforms.py`. The transforms are stateless functions plus a thin class wrapper that makes them composable with openpi's `DataConfig.data_transforms` list.

Dimensional contract:
- `action.joint_pos`: shape `(7,)` — absolute joint angles in radians for the 7-DoF arm
- `action.gripper_cmd`: shape `(1,)` — normalized in `[0, 1]`

The convention freeze is a hard gate. Write it into a docstring constant at the top of the file once decided. Do not leave it unspecified.

### Integration

Plugged into `DataConfig.data_transforms` alongside openpi's existing transforms. The `RM75Inputs` class in `rm75_policy.py` calls these transforms as part of the observation and action mapping pipeline.

### Tests (`action_transforms_test.py`)

- Round-trip: `canonical_to_training` then `training_to_canonical` recovers input within float tolerance
- Known-value: a specific teleop delta maps to the expected canonical delta
- Dimension validation: wrong-shape input raises immediately
- Batch dimension: transforms handle `(B, T, D)` shapes correctly
- Gripper clipping: values outside the expected range are handled gracefully

---

## 4. Module: `rm75_policy.py`

### Purpose

RM75 platform policy configuration, following `libero_policy.py` exactly in structure. This is the adapter between the RM75 robot's observation and action space and the openpi model's expected inputs and outputs. It is used identically during training (as part of `DataConfig`) and inference (as part of policy serving).

### Key Classes

**`RM75Inputs(DataTransformFn)`**

Maps RM75 observations to openpi model input slots:

- `obs.wrist_rgb` maps to the model's primary image input slot
- `obs.q`, `obs.dq`, `obs.gripper` are concatenated into the model's state vector
- `lang.instruction` maps to the language token slot
- The image is resized and normalized to match SigLIP input expectations

**`RM75Outputs(DataTransformFn)`**

Maps openpi model action outputs back to RM75 action space:

- Model action output is split into `action.joint_pos` (7D) and `action.gripper_cmd` (1D)
- Applies denormalization using precomputed norm stats

**`LeRobotRM75DataConfig(DataConfigFactory)`**

Returned by the factory function and used in `TrainConfig.data`. Specifies:

- Dataset repo ID
- Image keys and camera slots
- Action keys
- State keys
- The sequence of `data_transforms` to apply (including `RM75Inputs`)
- Norm stats path

### Integration

`LeRobotRM75DataConfig` is returned by `rm75_policy.py` and referenced in `spacecil/config.py` and `lunarcompose/config.py` as the `data` field of each `TrainConfig`. It is also imported by the training scripts for norm stats computation.

### Tests (`rm75_policy_test.py`)

- `RM75Inputs`: input dict with correct field names produces output with correct model key names
- `RM75Inputs`: image shape after transform matches SigLIP expectations
- `RM75Outputs`: model output with correct shape produces split joint_pos and gripper_cmd
- `LeRobotRM75DataConfig`: instantiation with a debug config completes without error
- Dimension contract: action output dimension is exactly 8 (7 + 1)

---

## 5. Module: `scorer_base.py`

### Purpose

Automated evaluation for all tasks. The scorer is not a convenience tool; it is the primary evaluation signal for both papers. If the scorer is wrong, the paper results are wrong.

### Design

**`Scorer` (abstract base class)**

```
score(episode: EpisodeSchema) -> ScorerResult
```

Every per-task scorer must implement this interface. No scorer may bypass `EpisodeSchema` or read raw sensor data directly.

**`ScorerResult` (frozen dataclass)**

- `success: bool` — primary outcome
- `confidence: float` — scorer's self-reported confidence in `[0, 1]`
- `fail_type: Optional[str]` — one of the fixed enum in `label.fail_type`
- `details: dict` — scorer-specific diagnostic data for debugging

### Per-Task Scorers

**`PayloadTransferScorer`**
Checks whether a payload object has been moved to the designated goal region. Uses wrist camera or scene camera depending on configuration. Primary signal: object pose estimate or presence in goal zone.

**`LatchActuationScorer`**
Checks whether a latch or lever has reached its actuated state. Primary signal: joint state reading or visual state of the latch indicator.

**`SurfaceCleaningScorer`**
Checks whether a surface has been cleaned. This is the most fragile scorer because pixel-difference signals are sensitive to lighting. The scorer must either use a lighting-controlled scorer camera isolated from the policy view, or use marker-based / segmentation-based signals robust to the tested environment conditions. Do not ship a pixel-difference scorer that reads from the policy camera view under environment perturbation.

**`ConnectorMatingScorer`**
Checks whether a connector has been successfully mated. Primary signal: visual confirmation of connector engagement or a binary electrical/mechanical contact signal if available.

### Scorer Validation Protocol

Every scorer must be validated against manual labels before it counts as production-ready:

- Run scorer on a pilot subset of at least 5 episodes with known manual labels
- Record precision and recall relative to manual labels
- Define a minimum agreement threshold; scorer revision is triggered if it falls below
- Document the validation results; these go into the experiment artifact record

A scorer that has not passed validation must not be used in any experiment that contributes to paper results.

### Tests (`scorer_base_test.py`)

- Mock episode with known outcome: `PayloadTransferScorer` returns correct `success=True`
- Mock episode with known failure: scorer returns `success=False` with expected `fail_type`
- Edge cases: scorer handles missing optional fields without crashing
- Confidence range: all scorer outputs have `confidence` in `[0, 1]`
- Scorer agreement: given a set of 5 mock episodes with known labels, scorer agreement exceeds threshold

---

## 6. Dependencies and Integration Points

```
episode_schema.py
    depends on: nothing (leaf module)
    consumed by: action_transforms, rm75_policy, scorer_base, all harnesses

action_transforms.py
    depends on: openpi.transforms (DeltaActions pattern for interface)
    consumed by: rm75_policy (RM75Inputs/RM75Outputs)

rm75_policy.py
    depends on: action_transforms.py, episode_schema.py, openpi.policies (libero pattern)
    consumed by: spacecil/config.py, lunarcompose/config.py, training scripts

scorer_base.py
    depends on: episode_schema.py
    consumed by: continual_harness, missing_corner_harness, G1/G2/G3 gate validation
```

No circular dependencies. The dependency graph is a strict DAG.

---

## 7. Acceptance Criteria

Phase A is complete when all of these pass:

1. `uv run pytest src/openpi/research/shared/ -x -q` exits green
2. `LeRobotRM75DataConfig` instantiates successfully with a debug config (no GPU required)
3. Action transforms are invertible: `canonical_to_training(training_to_canonical(x)) == x` within float tolerance
4. All four per-task scorers agree with manual labels on at least 5 mock episodes each
5. Episode schema serializes and deserializes without data loss

Only after all five are satisfied does G1 evaluation begin. G1 evaluation adds end-to-end training and replay of one task.

---

## 8. What Not to Do

- Do not create a second episode schema for LunarCompose or for any subset of fields.
- Do not write action converters in the scorer code or the harness code. All action conversion goes through `action_transforms.py`.
- Do not use the policy camera view for the `SurfaceCleaningScorer` without a lighting-controlled setup.
- Do not mark a scorer as production-ready until its validation results are documented.
- Do not change field names in `episode_schema.py` after training begins without incrementing the schema version and auditing all consumers.
