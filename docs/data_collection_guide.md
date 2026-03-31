# Data Collection Guide

## Overview

This data collection path records one HDF5 file per episode during remote inference.

Each file contains:

- Episode metadata: experiment name, task name, episode id, success flag, timestamp
- Per-step embeddings:
  - `vision_0`, `vision_1`, `vision_2`
  - `prompt_emb`
  - `robot_state`
  - `noise_action_*`
  - `clean_action`

The collector is designed as an outer wrapper around the normal policy path:

- `--collect` off: no collection wrapper, no behavior change
- `--collect` on but no active episode: pure delegation, no hooks
- active episode: temporary forward hooks are attached during each `infer()` call and removed immediately afterward

## Output Location

By default, collected files are written under the current working directory:

```bash
./data/<experiment_name>/episode_<episode_id>_<timestamp>.h5
```

For LIBERO, `experiment_name` is usually the task suite name, for example:

```bash
./data/libero_spatial/episode_0007_20260331_035410_446588.h5
```

You can override the root directory with `--collect_dir`.

## Server Command

Start the policy server with collection enabled:

```bash
uv run scripts/serve_policy.py --collect --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Useful flags:

- `--collect`: enables per-episode HDF5 recording
- `--collect_dir ./data`: optional custom output root
- `--env LIBERO`: selects the LIBERO policy setup
- `policy:checkpoint`: load a specific checkpoint instead of the environment default
- `--policy.config pi05_libero`: choose the train/inference config
- `--policy.dir ...`: point to the PyTorch checkpoint directory

If you also enable staged cache timing:

```bash
uv run scripts/serve_policy.py --cache --collect --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

## What Gets Saved

Each HDF5 file stores one episode.

Top-level attributes:

- `experiment_name`
- `task`
- `episode_id`
- `num_steps`
- `timestamp`
- `success`

Top-level groups:

- `step_0000`
- `step_0001`
- ...
- `step_NNNN`

Each `step_xxxx` group contains:

- `vision_0`: `float16[256, 2048]`
- `vision_1`: `float16[256, 2048]`
- `vision_2`: `float16[256, 2048]`
- `prompt_emb`: `float16[num_lang_tokens, 2048]`
- `robot_state`: `float32[32]`
- `noise_action_1 ... noise_action_9`: `float32[action_horizon, action_dim]`
- `clean_action`: `float32[action_horizon, action_dim]`

For the released `pi05_libero` checkpoint, this usually means:

- `action_horizon = 10`
- `action_dim = 32`

So the action tensors are typically:

- `noise_action_*`: `(10, 32)`
- `clean_action`: `(10, 32)`

## Minimal Simulator Changes

Your simulator client must send two lifecycle control messages:

1. episode start
2. episode end

The two pieces of metadata that matter most are:

- task name
- episode id

### Task Name

Use a stable human-readable task string. For LIBERO this is usually `task_description`.

Example:

```python
task=str(task_description)
```

This value is stored in the HDF5 file attribute:

```text
attrs["task"]
```

### Episode ID

Use a globally increasing episode counter, not a task-local counter that resets to zero for every task.

Good:

```python
global_episode_id = 0
...
client.episode_start(..., episode_id=global_episode_id)
...
global_episode_id += 1
```

Bad:

```python
client.episode_start(..., episode_id=episode_idx)
```

If `episode_idx` resets inside each task loop, your saved files will repeatedly show `episode_id=0`, `1`, etc. across different tasks, which makes downstream analysis harder.

### Example Client-Side Calls

At the start of each episode:

```python
client.episode_start(
    experiment=args.task_suite_name,
    task=str(task_description),
    episode_id=global_episode_id,
)
```

At the end of each episode:

```python
client.episode_end(success=done)
global_episode_id += 1
```

## Recommended LIBERO Pattern

Inside `examples/libero/main.py`, the intended pattern is:

1. Create one websocket client
2. Maintain one global episode counter across the full evaluation run
3. Before each rollout:
   - send `episode_start(...)`
4. After each rollout:
   - send `episode_end(success=done)`
   - increment the global episode counter

Suggested values:

- `experiment=args.task_suite_name`
- `task=str(task_description)`
- `episode_id=global_episode_id`

## Concise Data Inspection

List saved files:

```bash
find data -name '*.h5'
```

Inspect the newest file:

```bash
python - <<'PY'
import glob
import h5py
paths = sorted(glob.glob('data/libero_spatial/*.h5'))
print("latest:", paths[-1])
with h5py.File(paths[-1], 'r') as f:
    print("attrs:", dict(f.attrs))
    print("groups:", list(f.keys())[:5])
    step0 = f['step_0000']
    for key in step0.keys():
        print(key, step0[key].shape, step0[key].dtype)
PY
```

## Notes YOU MAST READ IT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

- After you enable `--collect`, the first real simulator inference may trigger model compilation. This is especially noticeable when `--cache` is also enabled, but the first compiled inference can also happen on the normal PyTorch path.
- During that first compile, the server may look completely stuck for a long time and may not print new progress messages. This is normal. Do not assume it crashed immediately.
- In practice, the most common symptom is: the simulator sends the first episode, and then the server appears to hang before returning the first action chunk.
- The correct response is usually to wait patiently.
- If you want a quick reality check, open another terminal and run `top` or `htop`. If the Python process is still using a lot of CPU, it is often still compiling rather than deadlocked.
- The collector writes one file per episode, not one file per task suite.
- Disconnect during an episode will flush partial data on connection close.
- File names include timestamps, so repeated runs do not overwrite earlier files.
