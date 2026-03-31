# Data Collection Verification

## 1. Verify control-message ack when `--collect` is off

Start the server without collection:

```bash
uv run scripts/serve_policy.py --cache --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

In another terminal, send only episode lifecycle control messages:

```bash
PYTHONPATH=packages/openpi-client/src python - <<'PY'
from openpi_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy("155.98.36.47", 9000)
print(client.episode_start("smoke_test", "libero_spatial", 0))
print(client.episode_end(False))
PY
```

Expected result:

- The client does not hang.
- It prints `{'__ack__': 'episode_start'}` and `{'__ack__': 'episode_end'}`.
- No HDF5 file is created.

## 2. Verify HDF5 writing when `--collect` is on

Start the server with collection enabled:

```bash
uv run scripts/serve_policy.py --cache --collect --collect_dir /tmp/libero_collect --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Run a minimal LIBERO rollout:

```bash
MUJOCO_GL=egl python examples/libero/main.py --args.host 155.98.36.47 --args.port 9000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1
```

If you want the render window:

```bash
MUJOCO_GL=egl python examples/libero/main.py --args.host 155.98.36.47 --args.port 9000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.display
```

Expected result:

- The rollout completes normally.
- At least one `.h5` file appears under `/tmp/libero_collect/libero_spatial/`.

## 3. Inspect generated HDF5 files

List generated files:

```bash
find /tmp/libero_collect -name '*.h5'
```

If `h5ls` is installed:

```bash
h5ls -r /tmp/libero_collect/libero_spatial/*.h5 | head -200
```

If not, inspect with Python:

```bash
python - <<'PY'
import glob
import h5py
paths = sorted(glob.glob('/tmp/libero_collect/libero_spatial/*.h5'))
print("latest file:", paths[-1])
with h5py.File(paths[-1], 'r') as f:
    print("attrs:", dict(f.attrs))
    print("top-level groups:", list(f.keys())[:5])
    step0 = f['step_0000']
    print("step_0000 datasets:", list(step0.keys()))
    for k in step0.keys():
        print(k, step0[k].shape, step0[k].dtype)
PY
```

Expected result:

- File attrs include `experiment_name`, `task`, `episode_id`, `num_steps`, and `success`.
- `step_0000` exists.
- Each step contains datasets such as `vision_0`, `vision_1`, `vision_2`, `prompt_emb`, `robot_state`, `noise_action_*`, and `clean_action`.

## 4. Verify partial flush on disconnect

Keep the server running with `--collect` enabled, then start a rollout:

```bash
MUJOCO_GL=egl python examples/libero/main.py --args.host 155.98.36.47 --args.port 9000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1
```

While the episode is still running, interrupt the client with `Ctrl-C`.

Then check whether a partial HDF5 file was written:

```bash
find /tmp/libero_collect -name '*.h5'
```

Inspect the newest file:

```bash
ls -lt /tmp/libero_collect/libero_spatial/*.h5 | head
```

Optionally inspect attrs:

```bash
python - <<'PY'
import glob
import h5py
paths = sorted(glob.glob('/tmp/libero_collect/libero_spatial/*.h5'))
print("latest file:", paths[-1])
with h5py.File(paths[-1], 'r') as f:
    print(dict(f.attrs))
PY
```

Expected result:

- A partial `.h5` file still exists.
- The latest file should have `success=False`.

## 5. Minimal regression check

If you only want a fast smoke test, run these three commands:

```bash
PYTHONPATH=packages/openpi-client/src python - <<'PY'
from openpi_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy("155.98.36.47", 9000)
print(client.episode_start("smoke_test", "libero_spatial", 0))
print(client.episode_end(False))
PY
```

```bash
MUJOCO_GL=egl python examples/libero/main.py --args.host 155.98.36.47 --args.port 9000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1
```

```bash
find /tmp/libero_collect -name '*.h5'
```

Expected result:

- Control messages return ack.
- A rollout still works normally.
- At least one HDF5 file is produced.
