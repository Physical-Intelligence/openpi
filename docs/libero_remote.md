# LIBERO Remote Inference Guide

The simulator runs on the local WSL2 machine, while model inference runs on a remote GPU server, communicating via WebSocket.

```
[Local WSL2]                        [GPU Server]
  LIBERO simulator    <--websocket-->   π0.5 model inference
  main.py                              serve_policy.py
  port 9000 (client)                   port 9000 (server)
```

Video recording strategy: A background thread captures frames at a fixed rate (30fps). **During inference wait periods, the last frame is repeated**, so inference latency is faithfully reflected in the recorded video.

---

## 1. Code Modifications

The original `examples/libero/main.py` only saves frames after `env.step()`, meaning inference latency is not captured in the video and there is no real-time rendering window. The following modifications are needed.

Replace `examples/libero/main.py` with the following:

```python
import collections
import dataclasses
import logging
import math
import pathlib
import threading
import time

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RECORD_FPS = 30  # Recording frame rate, independent of simulation step rate


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    display: bool = False        # Whether to show a real-time rendering window (requires WSLg / X11)
    video_out_path: str = "data/libero/videos"
    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan = collections.deque()

            # ── Recording state ──────────────────────────────────
            current_frame = [None]       # Shared frame read by background thread (list as mutable container)
            recorded_frames = []
            stop_event = threading.Event()

            def recorder_thread():
                while not stop_event.is_set():
                    t0 = time.perf_counter()
                    frame = current_frame[0]
                    if frame is not None:
                        recorded_frames.append(frame.copy())
                        if args.display:
                            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imshow("LIBERO", bgr)
                            cv2.waitKey(1)
                    elapsed = time.perf_counter() - t0
                    time.sleep(max(0.0, 1.0 / RECORD_FPS - elapsed))

            recorder = threading.Thread(target=recorder_thread, daemon=True)
            recorder.start()
            # ─────────────────────────────────────────────────────

            t = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        current_frame[0] = _get_display_frame(obs)
                        t += 1
                        continue

                    img = _preprocess_img(obs["agentview_image"], args.resize_size)
                    wrist_img = _preprocess_img(obs["robot0_eye_in_hand_image"], args.resize_size)
                    current_frame[0] = _get_display_frame(obs)  # Update display frame

                    if not action_plan:
                        # ── Background thread keeps repeating last frame during inference ──
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])
                        # ─────────────────────────────────────────────

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    current_frame[0] = _get_display_frame(obs)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # ── Stop recording and save video ────────────────────
            stop_event.set()
            recorder.join()
            if args.display:
                cv2.destroyAllWindows()

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4"
            if recorded_frames:
                imageio.mimwrite(str(video_path), recorded_frames, fps=RECORD_FPS)
                logging.info(f"Video saved: {video_path} ({len(recorded_frames)} frames)")
            # ─────────────────────────────────────────────────────

            task_episodes += 1
            total_episodes += 1
            logging.info(f"Success: {done}")
            logging.info(f"# episodes: {total_episodes}, # successes: {total_successes} "
                         f"({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Task success rate: {float(task_successes) / float(task_episodes):.2%}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.2%}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _preprocess_img(raw_img, resize_size):
    """Flip + resize to match training preprocessing."""
    img = np.ascontiguousarray(raw_img[::-1, ::-1])
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )


def _get_display_frame(obs):
    """Get the agentview image (flipped) for display/recording at original resolution."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
```

---

## 2. Server Side (GPU Machine)

Server-side setup is identical to aloha_sim.

### Installation

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

curl -LsSf https://astral.sh/uv/install.sh | sh

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Start the Policy Server

```bash
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

- Uses the `pi05_libero` config and explicitly loads the converted PyTorch checkpoint at `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
- Listens on `0.0.0.0:8000` by default — make sure port 8000 is open in the firewall

---

## 3. Local WSL2 (Simulator)

### Install System Dependencies

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libglfw3 libglfw3-dev
```

For the real-time rendering window, also install:

```bash
sudo apt-get install -y libopencv-dev python3-opencv
```

### Create a Conda Environment

> **Python 3.8 is required.** LIBERO's dependency `numba==0.53.1` does not support Python 3.9+. You cannot reuse the aloha_sim conda environment.

```bash
conda create -n libero_sim python=3.8 -y
conda activate libero_sim
```

### Install Python Dependencies

```bash
cd /path/to/openpi

# Install dependencies (note: PyTorch requires the CUDA 11.3 index)
pip install -r examples/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install libero from PyPI (no git submodule needed)
pip install libero

pip install -e packages/openpi-client
```

---

## 4. Running

> **Note:** This script uses `tyro` nested argument parsing. All runtime arguments must be prefixed with `--args.`, e.g., `--args.host`, `--args.port`, `--args.display`. Using `--host` directly will cause an `Unrecognized options` error.

### Option A: Save Video Only (No Rendering Window, Most Stable)

```bash
cd /path/to/openpi
conda activate libero_sim
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

MUJOCO_GL=egl python examples/libero/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000 \
    --args.task-suite-name libero_spatial
```

Videos are saved to: `data/libero/videos/rollout_<task>_<success|failure>.mp4`

### Option B: Real-Time Rendering Window + Video (Requires WSLg, Windows 11 Only)

```bash
MUJOCO_GL=egl python examples/libero/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000 \
    --args.task-suite-name libero_spatial \
    --args.display
```

> **Check if WSLg is available:** Run `echo $DISPLAY`. If there is output (e.g., `:0`), WSLg is available. Windows 10 does not support WSLg — use Option A instead.

### Common Parameters

| Parameter | Default | Description |
|---|---|---|
| `--host` | `155.98.36.47` | Server IP |
| `--port` | `9000` | Server port |
| `--task-suite-name` | `libero_spatial` | Task suite: `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` / `libero_90` |
| `--replan-steps` | `5` | Re-infer every N steps |
| `--num-trials-per-task` | `50` | Number of episodes per task |
| `--display` | `False` | Show real-time rendering window |
| `--video-out-path` | `data/libero/videos` | Video output directory |

---

## 5. Verify Connection

Before starting the simulator, confirm the server is reachable:

```bash
nc -zv 155.98.36.47 9000
```

Start the local simulator only after the server log shows `Listening on port 8000`.

---

## 6. Video Recording Details

| Original Behavior | Modified Behavior |
|---|---|
| Frames saved only after `env.step()` | Background thread records continuously at **30fps** |
| Inference latency not captured in video | Last frame repeated during inference — latency fully reflected |
| No real-time rendering | `--display` flag opens a cv2 window |

The recording frame rate can be changed via `RECORD_FPS = 30` at the top of `main.py`.
