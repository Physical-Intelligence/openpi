# SO101 Stacking Rings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add training support for the SO101 robot on the stacking rings task, using pi0.5 with delta actions.

**Architecture:** A lightweight policy module (`so101_policy.py`) handles the 6D joint-space input/output transforms. A data config factory (`LeRobotSO101DataConfig`) in `config.py` wires up repack transforms, delta action conversion, and per-timestep normalization. No canonical dimension lifting -- native 6D throughout.

**Tech Stack:** JAX/Flax (model), NumPy (transforms), LeRobot v3 dataset format, pi0.5 (thinking + flow matching)

---

## Context

### Dataset: `lorenzouttini/so101_stacking_rings`

- Robot: SO101 (so_follower), 5-DOF arm + gripper = **6D joint-space**
- 101 episodes, 34,401 frames, 30 FPS
- Features:
  - `action`: float32, shape [6] — shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
  - `observation.state`: float32, shape [6] — same joints
  - `observation.images.front`: video, 480×640×3
  - `observation.images.wrist`: video, 480×640×3
- Single task (task_index always 0), no per-episode prompt variation
- Actions are absolute joint targets (same magnitude/scale as state)

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Native 6D (no 17D lifting) | No EEF pose data exists; lifting adds meaningless zeros |
| Delta mask `[T,T,T,T,T,F]` | 5 joints as delta, gripper stays absolute (open/close position) |
| Per-timestep action norm | Enabled automatically when using delta actions — future timesteps have larger deltas |
| Quantile normalization | Default for pi0.5; robust to outliers |
| `action_horizon=10` | ~0.33s lookahead at 30fps; good starting point |
| Fixed prompt "stack the rings" | Single task dataset; no variation needed |

### Reference Files

- Similar policy module: `src/openpi/policies/libero_policy.py` (single-arm, simple transforms)
- Similar data config: `LeRobotLiberoDataConfig` in `src/openpi/training/config.py:367`
- Delta action machinery: `src/openpi/transforms.py:285` (`DeltaActions` / `AbsoluteActions`)
- Norm stat script: `scripts/compute_norm_stats.py`
- Per-timestep norm script: `scripts/compute_norm_stats_per_timestep.py`

---

## Task 1: Create SO101 Policy Module

**Files:**
- Create: `src/openpi/policies/so101_policy.py`

**Step 1: Create the policy module**

```python
"""Data transforms for SO101 single-arm robot (LeRobot v3 format).

SO101 is a 5-DOF arm + gripper = 6D joint-space:
  - observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] (6D)
  - action: same 6D joint positions
  - observation.images.front, observation.images.wrist
"""

import dataclasses

import numpy as np

from openpi import transforms


_SO101_ACTION_DIM = 6


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        import einops
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_key(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing keys: {keys}")


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Transforms SO101 dataset observations into model input format.

    Handles both training (LeRobot keys with dots/slashes) and inference
    (keys as sent by the robot driver).
    """

    default_prompt: str = "stack the rings"

    def __call__(self, data: dict) -> dict:
        front = _parse_image(
            _get_key(data, "observation.images.front", "observation/images/front", "image", "images.front")
        )
        try:
            wrist = _parse_image(
                _get_key(data, "observation.images.wrist", "observation/images/wrist", "wrist_image", "images.wrist")
            )
        except KeyError:
            wrist = np.zeros_like(front)

        state = np.asarray(
            _get_key(data, "observation.state", "observation/state", "state"),
            dtype=np.float32,
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        elif "task" in data:
            task = data["task"]
            inputs["prompt"] = task.decode("utf-8") if isinstance(task, bytes) else str(task)
        else:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Slices model output back to SO101's native 6D action space."""

    action_dim: int = _SO101_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, :self.action_dim].astype(np.float32)}
```

**Step 2: Verify the file is importable**

Run: `cd /Users/ps/Desktop/code/openpi && uv run python -c "from openpi.policies import so101_policy; print('OK')"`
Expected: `OK`

---

## Task 2: Add Data Config Factory and TrainConfig Entry

**Files:**
- Modify: `src/openpi/training/config.py`
  - Add import for `so101_policy` (near line 28)
  - Add `LeRobotSO101DataConfig` class (after `LeRobotDROIDDataConfig`, around line 913)
  - Add `TrainConfig` entry to `_CONFIGS` list

**Step 1: Add the import**

At `src/openpi/training/config.py`, near line 28 (with the other policy imports), add:

```python
import openpi.policies.so101_policy as so101_policy
```

**Step 2: Add the data config factory**

Insert after the `LeRobotDROIDDataConfig` class (around line 913), before the `LeRobotLiberoSubtaskDataConfig`:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """Data config for SO101 single-arm robot (6D joint-space).

    Supports optional delta action conversion (5 joints delta, gripper absolute).
    """

    default_prompt: str | None = "stack the rings"
    use_delta_actions: bool = False

    # Repack transforms: map LeRobot v3 keys to canonical internal keys.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.front": "observation.images.front",
                        "observation.images.wrist": "observation.images.wrist",
                        "observation.state": "observation.state",
                        "action": "actions",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[so101_policy.SO101Inputs(default_prompt=self.default_prompt or "stack the rings")],
            outputs=[so101_policy.SO101Outputs()],
        )

        if self.use_delta_actions:
            # 5 joints as delta, gripper (dim 5) stays absolute.
            delta_action_mask = _transforms.make_bool_mask(5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.use_delta_actions and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True

        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )
```

**Step 3: Add the TrainConfig entry**

Insert into the `_CONFIGS` list (e.g., after the ALOHA sim config around line 1664, or in its own section):

```python
    #
    # SO101 stacking rings config.
    #
    TrainConfig(
        name="pi05_so101_stacking_rings",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
        data=LeRobotSO101DataConfig(
            repo_id="lorenzouttini/so101_stacking_rings",
            default_prompt="stack the rings",
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50_000,
        save_interval=5000,
        batch_size=32,
        ema_decay=0.999,
        wandb_enabled=True,
    ),
```

**Step 4: Verify config loads**

Run: `cd /Users/ps/Desktop/code/openpi && uv run python -c "from openpi.training.config import get_config; c = get_config('pi05_so101_stacking_rings'); print(f'Config: {c.name}, model_type={c.model.model_type}, action_horizon={c.model.action_horizon}')"`
Expected: `Config: pi05_so101_stacking_rings, model_type=ModelType.PI05, action_horizon=10`

---

## Task 3: Compute Normalization Stats

**Files:** None (runtime step, generates assets)

**Step 1: Run global norm stats**

```bash
cd /Users/ps/Desktop/code/openpi
uv run python scripts/compute_norm_stats.py --config-name pi05_so101_stacking_rings
```

Expected: Writes stats to `assets/pi05_so101_stacking_rings/` (or whatever `config.assets_dirs` resolves to). You should see 6 dimensions for both `state` and `actions` with reasonable mean/std/quantile values.

**Step 2: Run per-timestep action norm stats**

```bash
uv run python scripts/compute_norm_stats_per_timestep.py --config-name pi05_so101_stacking_rings
```

Expected: Writes per-timestep action stats alongside the global stats. You should see stats for 10 timesteps × 6 action dims.

**Step 3: Sanity check the stats**

Verify the output makes sense:
- State means should be in the range of the raw joint values (~[-100, +100] for most joints, ~1 for gripper)
- Delta action means should be near 0 (since deltas average out over a dataset)
- Delta action std should be small for timestep 0 and grow for later timesteps

---

## Task 4: Run Training (Smoke Test)

**Files:** None (runtime verification)

**Step 1: Verify training starts successfully**

```bash
cd /Users/ps/Desktop/code/openpi
uv run python scripts/train.py \
    --config pi05_so101_stacking_rings \
    --exp_name smoke_test \
    --num_train_steps 50 \
    --batch_size 4 \
    --wandb_enabled false \
    --overwrite true
```

Expected: Training starts, loss decreases or is stable, no crashes. Check:
- Data loading works (no key errors)
- Delta action conversion works (no shape mismatches)
- Normalization applies correctly (loss values are reasonable, e.g., ~0.1-1.0 range)
- No warnings about tokens being chopped

**Step 2: If smoke test passes, launch full training**

```bash
uv run python scripts/train.py \
    --config pi05_so101_stacking_rings \
    --exp_name first_run
```

---

## Summary of Created/Modified Files

| File | Action |
|------|--------|
| `src/openpi/policies/so101_policy.py` | **Create** — 6D input/output transforms |
| `src/openpi/training/config.py` | **Modify** — add import, `LeRobotSO101DataConfig`, `TrainConfig` entry |
| `assets/pi05_so101_stacking_rings/` | **Generated** — norm stats (runtime) |

## Hyperparameter Notes for Tuning

If the initial run doesn't converge well:
- **Overfitting** (101 episodes is small): try LoRA (`paligemma_variant="gemma_2b_lora"`), reduce `peak_lr` to 1e-5, or add `freeze_filter` for the vision backbone
- **Underfitting**: increase `peak_lr` to 5e-5
- **Action frequency mismatch**: if the base model was pretrained at a different frequency, try `action_horizon=5` or `action_horizon=15`
- **Gripper issues**: if gripper actions are noisy, consider excluding gripper from delta (already done) or using a binary threshold
