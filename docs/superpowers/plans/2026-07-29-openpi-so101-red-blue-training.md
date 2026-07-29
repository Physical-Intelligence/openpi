# OpenPI SO-101 Red/Blue Joint Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested OpenPI π0 LoRA configuration that jointly trains on the local red-cube and blue-cube SO-101 datasets, converts the first five joints to delta actions, preserves the absolute gripper action, and saves every 2,500 steps.

**Architecture:** Extend `DataConfig` with an optional ordered list of LeRobot repository IDs while preserving the existing single-repository path. Use LeRobot's pinned `MultiLeRobotDataset`, derive prompts from each returned sample's `task` string, and add a focused SO-101 policy transform that maps front/wrist cameras into π0's image slots. Store combined normalization statistics under one explicit asset ID and expose a separate normalization batch-size override so training can remain at batch size 1.

**Tech Stack:** Python 3.11, dataclasses, NumPy, JAX, PyTorch DataLoader, LeRobot v2.0, pytest, Docker Desktop with NVIDIA GPU.

---

### Task 0: Prepare the Persistent Test Container

**Files:**
- No repository files change.

- [ ] **Step 1: Start the container with source and datasets mounted**

Run:

```powershell
docker run -d --name openpi-dev --gpus all `
  -e PYTHONPATH=/app/src `
  -e HF_LEROBOT_HOME=/lerobot_data `
  -e GIT_LFS_SKIP_SMUDGE=1 `
  -v "E:\openpi:/app" `
  -v "F:\lerobot_data:/lerobot_data" `
  -w /app `
  --entrypoint /bin/bash `
  openpi_server:latest -lc "sleep infinity"
```

Expected: Docker returns a container ID and `docker inspect openpi-dev` reports `running`.

- [ ] **Step 2: Install the test runner in the container layer**

Run:

```powershell
docker exec openpi-dev uv pip install --python /.venv/bin/python pytest
docker exec openpi-dev /.venv/bin/pytest --version
```

Expected: pytest reports its installed version without changing the repository lockfile.

### Task 1: Preserve Per-Sample Task Text as the Prompt

**Files:**
- Modify: `src/openpi/transforms.py`
- Modify: `src/openpi/transforms_test.py`

- [ ] **Step 1: Write the failing transform tests**

Add these tests to `src/openpi/transforms_test.py`:

```python
def test_extract_prompt_from_task_string():
    transform = _transforms.PromptFromLeRobotTaskString()

    data = transform({"task": "Put the red cube into the box."})

    assert data["prompt"] == "Put the red cube into the box."


def test_extract_prompt_from_task_string_requires_task():
    transform = _transforms.PromptFromLeRobotTaskString()

    with pytest.raises(ValueError, match='Cannot extract prompt without "task"'):
        transform({"task_index": 0})
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/transforms_test.py -k task_string
```

Expected: collection or test failure because `PromptFromLeRobotTaskString` does not exist.

- [ ] **Step 3: Add the minimal prompt transform**

Add this class immediately after `PromptFromLeRobotTask` in `src/openpi/transforms.py`:

```python
@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTaskString(DataTransformFn):
    """Copies the task string already resolved by LeRobot into the prompt field."""

    def __call__(self, data: DataDict) -> DataDict:
        if "task" not in data:
            raise ValueError('Cannot extract prompt without "task"')

        prompt = data["task"]
        if not isinstance(prompt, str):
            prompt = np.asarray(prompt).item()
        if not isinstance(prompt, str):
            raise ValueError(f'Expected "task" to be a string, got {type(prompt).__name__}')

        return {**data, "prompt": prompt}
```

- [ ] **Step 4: Run the focused and existing transform tests**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/transforms_test.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/openpi/transforms.py src/openpi/transforms_test.py
git commit -m "feat: preserve LeRobot task prompts"
```

### Task 2: Add SO-101 Model Input and Output Transforms

**Files:**
- Create: `src/openpi/policies/so101_policy.py`
- Create: `src/openpi/policies/so101_policy_test.py`

- [ ] **Step 1: Write failing SO-101 policy tests**

Create `src/openpi/policies/so101_policy_test.py`:

```python
import numpy as np
import pytest

from openpi.policies import so101_policy


def make_so101_example() -> dict:
    return {
        "images": {
            "front": np.full((3, 8, 8), 1, dtype=np.uint8),
            "wrist": np.full((3, 8, 8), 2, dtype=np.uint8),
        },
        "state": np.arange(6, dtype=np.float32),
        "actions": np.arange(18, dtype=np.float32).reshape(3, 6),
        "prompt": "Put the red cube into the box.",
    }


def test_so101_inputs_map_cameras_and_preserve_robot_values():
    output = so101_policy.SO101Inputs()(make_so101_example())

    np.testing.assert_array_equal(output["image"]["base_0_rgb"], np.full((3, 8, 8), 1, dtype=np.uint8))
    np.testing.assert_array_equal(output["image"]["left_wrist_0_rgb"], np.full((3, 8, 8), 2, dtype=np.uint8))
    np.testing.assert_array_equal(output["image"]["right_wrist_0_rgb"], np.zeros((3, 8, 8), dtype=np.uint8))
    assert output["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    np.testing.assert_array_equal(output["state"], np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(output["actions"], np.arange(18, dtype=np.float32).reshape(3, 6))
    assert output["prompt"] == "Put the red cube into the box."


def test_so101_inputs_require_front_camera():
    data = make_so101_example()
    del data["images"]["front"]

    with pytest.raises(ValueError, match='SO-101 input requires the "front" camera'):
        so101_policy.SO101Inputs()(data)


def test_so101_outputs_return_six_action_dimensions():
    actions = np.arange(24, dtype=np.float32).reshape(2, 12)

    output = so101_policy.SO101Outputs()({"actions": actions})

    np.testing.assert_array_equal(output["actions"], actions[:, :6])
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/policies/so101_policy_test.py
```

Expected: import failure because `so101_policy.py` does not exist.

- [ ] **Step 3: Implement the focused SO-101 transforms**

Create `src/openpi/policies/so101_policy.py`:

```python
import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Maps SO-101 state, actions, and front/wrist images into π0 inputs."""

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        extra_cameras = set(in_images) - {"front", "wrist"}
        if extra_cameras:
            raise ValueError(f"Unexpected SO-101 cameras: {tuple(sorted(extra_cameras))}")
        if "front" not in in_images:
            raise ValueError('SO-101 input requires the "front" camera')

        front = np.asarray(in_images["front"])
        wrist = np.asarray(in_images["wrist"]) if "wrist" in in_images else np.zeros_like(front)
        output = {
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if "wrist" in in_images else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": np.asarray(data["state"]),
        }
        if "actions" in data:
            output["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Returns the six physical SO-101 action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :6]}
```

- [ ] **Step 4: Run the SO-101 policy tests**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/policies/so101_policy_test.py
```

Expected: all three tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/openpi/policies/so101_policy.py src/openpi/policies/so101_policy_test.py
git commit -m "feat: add SO-101 policy transforms"
```

### Task 3: Load Multiple LeRobot Repositories Without Mixing Prompts

**Files:**
- Modify: `src/openpi/training/config.py`
- Modify: `src/openpi/training/data_loader.py`
- Modify: `src/openpi/training/data_loader_test.py`

- [ ] **Step 1: Write failing multi-repository loader tests**

Add these tests and supporting imports to `src/openpi/training/data_loader_test.py`:

```python
import numpy as np
import pytest


class _DatasetStub:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class _MetadataStub:
    def __init__(self, repo_id):
        self.repo_id = repo_id
        self.fps = 30
        self.tasks = {0: f"task for {repo_id}"}


def test_create_torch_dataset_combines_repositories_and_uses_sample_task(monkeypatch):
    captured = {}

    def make_multi_dataset(repo_ids, *, delta_timestamps):
        captured["repo_ids"] = repo_ids
        captured["delta_timestamps"] = delta_timestamps
        return _DatasetStub(
            [
                {"task": "Put the red cube into the box."},
                {"task": "Put the blue cube into the box."},
            ]
        )

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _MetadataStub)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "MultiLeRobotDataset", make_multi_dataset)
    data_config = _config.DataConfig(
        repo_ids=("YukiiLiu/red", "YukiiLiu/blue"),
        prompt_from_task=True,
        action_sequence_keys=("action",),
    )

    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=3,
        model_config=pi0_config.Pi0Config(),
    )

    assert captured["repo_ids"] == ["YukiiLiu/red", "YukiiLiu/blue"]
    assert captured["delta_timestamps"] == {"action": [0.0, 1 / 30, 2 / 30]}
    assert dataset[0]["prompt"] == "Put the red cube into the box."
    assert dataset[1]["prompt"] == "Put the blue cube into the box."


def test_create_torch_dataset_rejects_mixed_fps(monkeypatch):
    class MixedMetadata(_MetadataStub):
        def __init__(self, repo_id):
            super().__init__(repo_id)
            self.fps = 30 if repo_id.endswith("red") else 20

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", MixedMetadata)
    data_config = _config.DataConfig(repo_ids=("YukiiLiu/red", "YukiiLiu/blue"))

    with pytest.raises(ValueError, match="same FPS"):
        _data_loader.create_torch_dataset(data_config, 3, pi0_config.Pi0Config())
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/training/data_loader_test.py -k 'combines_repositories or mixed_fps'
```

Expected: failure because `DataConfig` has no `repo_ids` field.

- [ ] **Step 3: Extend configuration with an ordered repository list**

Add this field after `repo_id` in both `DataConfig` and `DataConfigFactory` in `src/openpi/training/config.py`:

```python
# Ordered LeRobot repo IDs. Mutually exclusive with repo_id.
repo_ids: Sequence[str] = ()
```

Replace `DataConfigFactory.create_base_config` with:

```python
def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
    repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
    repo_ids = tuple(self.repo_ids)
    if repo_id is not None and repo_ids:
        raise ValueError("repo_id and repo_ids are mutually exclusive")
    asset_id = self.assets.asset_id or repo_id
    return dataclasses.replace(
        self.base_config or DataConfig(),
        repo_id=repo_id,
        repo_ids=repo_ids,
        asset_id=asset_id,
        norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        use_quantile_norm=model_config.model_type != ModelType.PI0,
    )
```

- [ ] **Step 4: Implement single- and multi-repository dataset construction**

Replace `create_torch_dataset` in `src/openpi/training/data_loader.py` with:

```python
def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    if data_config.repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    repo_ids = tuple(data_config.repo_ids)
    if data_config.repo_id is not None:
        if repo_ids:
            raise ValueError("repo_id and repo_ids are mutually exclusive")
        repo_ids = (data_config.repo_id,)
    if not repo_ids:
        raise ValueError("Repo ID is not set. Cannot create dataset.")

    metadata = [lerobot_dataset.LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
    fps_values = {item.fps for item in metadata}
    if len(fps_values) != 1:
        raise ValueError(f"All LeRobot repositories must use the same FPS, got {sorted(fps_values)}")
    fps = metadata[0].fps
    delta_timestamps = {
        key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }

    if len(repo_ids) == 1:
        dataset = lerobot_dataset.LeRobotDataset(repo_ids[0], delta_timestamps=delta_timestamps)
    else:
        dataset = lerobot_dataset.MultiLeRobotDataset(list(repo_ids), delta_timestamps=delta_timestamps)

    if data_config.prompt_from_task:
        prompt_transform = (
            _transforms.PromptFromLeRobotTask(metadata[0].tasks)
            if len(repo_ids) == 1
            else _transforms.PromptFromLeRobotTaskString()
        )
        dataset = TransformedDataset(dataset, [prompt_transform])

    return dataset
```

- [ ] **Step 5: Run loader tests**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/training/data_loader_test.py -k 'not real_dataset'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/openpi/training/config.py src/openpi/training/data_loader.py src/openpi/training/data_loader_test.py
git commit -m "feat: load multiple LeRobot datasets"
```

### Task 4: Register the SO-101 LoRA Training Configuration

**Files:**
- Modify: `src/openpi/training/config.py`
- Create: `src/openpi/training/config_test.py`

- [ ] **Step 1: Write failing configuration tests**

Create `src/openpi/training/config_test.py`:

```python
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi import transforms


def test_so101_red_blue_lora_config():
    config = _config.get_config("pi0_so101_red_blue_lora")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.batch_size == 1
    assert config.num_train_steps == 20_000
    assert config.save_interval == 2_500
    assert config.keep_period == 2_500
    assert config.wandb_enabled is False
    assert config.ema_decay is None
    assert data_config.repo_id is None
    assert data_config.repo_ids == (
        "YukiiLiu/so101_red_cube_box_formal_clean_v20_100src",
        "YukiiLiu/so101_blue_cube_box_formal_clean_v20_100src",
    )
    assert data_config.asset_id == "so101_red_blue"
    assert data_config.prompt_from_task is True


def test_so101_delta_mask_preserves_gripper():
    config = _config.get_config("pi0_so101_red_blue_lora")
    data_config = config.data.create(config.assets_dirs, config.model)
    delta_transform = next(
        item for item in data_config.data_transforms.inputs if isinstance(item, transforms.DeltaActions)
    )
    state = np.arange(6, dtype=np.float32)
    actions = np.stack([state + 1, state + 2])

    output = delta_transform({"state": state, "actions": actions.copy()})

    np.testing.assert_array_equal(output["actions"][:, :5], np.ones((2, 5)) * np.array([[1], [2]]))
    np.testing.assert_array_equal(output["actions"][:, 5], actions[:, 5])
```

- [ ] **Step 2: Run the configuration tests and verify RED**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/training/config_test.py
```

Expected: failure because `pi0_so101_red_blue_lora` is not registered.

- [ ] **Step 3: Add `LeRobotSO101DataConfig`**

Import the new policy in `src/openpi/training/config.py`:

```python
import openpi.policies.so101_policy as so101_policy
```

Add this factory after `LeRobotAlohaDataConfig`:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "front": "observation.images.front",
                            "wrist": "observation.images.wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        delta_action_mask = (True, True, True, True, True, False)
        data_transforms = _transforms.Group(
            inputs=[
                so101_policy.SO101Inputs(),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                so101_policy.SO101Outputs(),
            ],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )
```

- [ ] **Step 4: Register the joint-training config**

Add this entry before the debugging configs in `_CONFIGS`:

```python
TrainConfig(
    name="pi0_so101_red_blue_lora",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotSO101DataConfig(
        repo_ids=(
            "YukiiLiu/so101_red_cube_box_formal_clean_v20_100src",
            "YukiiLiu/so101_blue_cube_box_formal_clean_v20_100src",
        ),
        assets=AssetsConfig(asset_id="so101_red_blue"),
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    batch_size=1,
    num_workers=0,
    num_train_steps=20_000,
    save_interval=2_500,
    keep_period=2_500,
    wandb_enabled=False,
)
```

- [ ] **Step 5: Run the configuration tests**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q src/openpi/training/config_test.py
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/openpi/training/config.py src/openpi/training/config_test.py
git commit -m "feat: configure SO-101 joint LoRA training"
```

### Task 5: Write Combined Normalization Statistics to the Asset ID

**Files:**
- Modify: `scripts/compute_norm_stats.py`
- Create: `scripts/compute_norm_stats_test.py`

- [ ] **Step 1: Write failing normalization helper tests**

Create `scripts/compute_norm_stats_test.py`:

```python
import pathlib
from types import SimpleNamespace

import pytest

from openpi.training import config as _config
from scripts import compute_norm_stats


def test_norm_stats_output_uses_asset_id():
    train_config = SimpleNamespace(assets_dirs=pathlib.Path("/tmp/assets"))
    data_config = _config.DataConfig(
        repo_ids=("YukiiLiu/red", "YukiiLiu/blue"),
        asset_id="so101_red_blue",
    )

    output = compute_norm_stats.get_output_path(train_config, data_config)

    assert output == pathlib.Path("/tmp/assets/so101_red_blue")


def test_norm_stats_output_requires_identifier():
    train_config = SimpleNamespace(assets_dirs=pathlib.Path("/tmp/assets"))

    with pytest.raises(ValueError, match="asset_id or repo_id"):
        compute_norm_stats.get_output_path(train_config, _config.DataConfig())
```

- [ ] **Step 2: Run the helper tests and verify RED**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q scripts/compute_norm_stats_test.py
```

Expected: failure because `get_output_path` does not exist.

- [ ] **Step 3: Add the output-path helper and normalization batch override**

Add this helper above `main` in `scripts/compute_norm_stats.py`:

```python
def get_output_path(config: _config.TrainConfig, data_config: _config.DataConfig):
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must have an asset_id or repo_id")
    return config.assets_dirs / asset_id
```

Change the `main` signature and torch loader call:

```python
def main(config_name: str, max_frames: int | None = None, batch_size: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats_batch_size = batch_size or config.batch_size
```

```python
data_loader, num_batches = create_torch_dataloader(
    data_config,
    config.model.action_horizon,
    stats_batch_size,
    config.model,
    config.num_workers,
    max_frames,
)
```

Replace the output path assignment with:

```python
output_path = get_output_path(config, data_config)
```

- [ ] **Step 4: Run normalization and related unit tests**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q scripts/compute_norm_stats_test.py src/openpi/training/config_test.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/compute_norm_stats.py scripts/compute_norm_stats_test.py
git commit -m "feat: support combined normalization assets"
```

### Task 6: Validate Real Data, Compute Statistics, and Start Training

**Files:**
- Modify: `agent_memory/progress.md`
- Modify if a verified runtime issue requires it: only the smallest source or configuration file covered by a new failing test

- [ ] **Step 1: Verify the persistent validation container**

Run:

```powershell
docker inspect --format "{{.State.Status}}" openpi-dev
docker exec openpi-dev /.venv/bin/python -c "import jax; print(jax.devices())"
```

Expected: the container status is `running` and JAX reports `[CudaDevice(id=0)]`.

- [ ] **Step 2: Run the relevant unit suite and formatter checks**

Run:

```powershell
docker exec openpi-dev /.venv/bin/pytest -q `
  src/openpi/transforms_test.py `
  src/openpi/policies/so101_policy_test.py `
  src/openpi/training/data_loader_test.py `
  src/openpi/training/config_test.py `
  scripts/compute_norm_stats_test.py
docker exec openpi-dev /bin/uvx ruff check `
  src/openpi/transforms.py `
  src/openpi/policies/so101_policy.py `
  src/openpi/training/config.py `
  src/openpi/training/data_loader.py `
  scripts/compute_norm_stats.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 3: Run a two-repository real-data smoke test**

Run:

```powershell
docker exec openpi-dev /.venv/bin/python -c "from openpi.training import config, data_loader; c=config.get_config('pi0_so101_red_blue_lora'); d=c.data.create(c.assets_dirs,c.model); ds=data_loader.create_torch_dataset(d,c.model.action_horizon,c.model); print(type(ds).__name__,len(ds),ds[0]['task'],ds[19693]['task'])"
```

Expected: length `40948`; the first prompt is the red-cube task and the first blue-dataset sample is the blue-cube task.

- [ ] **Step 4: Compute a small normalization smoke sample**

Run:

```powershell
docker exec openpi-dev /.venv/bin/python scripts/compute_norm_stats.py `
  --config-name pi0_so101_red_blue_lora `
  --max-frames 64 `
  --batch-size 8
```

Expected: finite state/action statistics are written to `assets/pi0_so101_red_blue_lora/so101_red_blue/norm_stats.json`.

- [ ] **Step 5: Compute full combined normalization statistics**

Run:

```powershell
docker exec openpi-dev /.venv/bin/python scripts/compute_norm_stats.py `
  --config-name pi0_so101_red_blue_lora `
  --batch-size 16
```

Expected: the progress bar reaches all combined frames and the final statistics contain finite values for six-dimensional state and action data.

- [ ] **Step 6: Start a one-step training smoke run**

Run:

```powershell
docker exec openpi-dev /.venv/bin/python scripts/train.py `
  pi0_so101_red_blue_lora `
  --exp-name so101_red_blue_smoke `
  --num-train-steps 1 `
  --overwrite
```

Expected: JAX initializes on `CudaDevice(id=0)`, the base checkpoint loads, and step 0 reports finite loss and gradient metrics.

- [ ] **Step 7: Start the full training run**

Run:

```powershell
docker exec openpi-dev /.venv/bin/python scripts/train.py `
  pi0_so101_red_blue_lora `
  --exp-name so101_red_blue_lora `
  --overwrite
```

Expected: training continues toward 20,000 steps and writes checkpoints every 2,500 steps under `checkpoints/pi0_so101_red_blue_lora/so101_red_blue_lora`.

- [ ] **Step 8: Update task memory with measured results**

Record the dataset length, normalization output path, first-step loss, peak GPU memory, checkpoint path, and any OOM or compatibility failure in `agent_memory/progress.md` and `agent_memory/bugs.md`.
