# LIBERO 远程推理指南

模拟器运行在本机 WSL2，模型推理运行在远端 GPU 服务器，通过 WebSocket 通信。

```
[本机 WSL2]                        [GPU 服务器]
  LIBERO 模拟器    <--websocket-->   π0.5 模型推理
  main.py                            serve_policy.py
  port 9000 (client)                 port 9000 (server)
```

视频录制策略：使用后台线程以固定帧率（30fps）持续捕帧，**推理等待期间重复最后一帧**，从而将推理延迟真实录入视频。

---

## 一、代码修改

原始 `examples/libero/main.py` 只在 `env.step()` 后保存帧，推理延迟不会录入视频，也没有实时渲染窗口。需要做以下修改。

将 `examples/libero/main.py` 替换为如下内容：

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
RECORD_FPS = 30  # 录制帧率，独立于仿真步长


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    display: bool = False        # 是否弹出实时渲染窗口（需要 WSLg / X11）
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

            # ── 录制状态 ──────────────────────────────────────────
            current_frame = [None]       # 后台线程读取的共享帧（list 作可变容器）
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
                    current_frame[0] = _get_display_frame(obs)  # 更新显示帧

                    if not action_plan:
                        # ── 推理期间后台线程持续重复最后一帧 ──────────
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

            # ── 停止录制并保存视频 ────────────────────────────────
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
    """翻转 + resize，与训练预处理对齐。"""
    img = np.ascontiguousarray(raw_img[::-1, ::-1])
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )


def _get_display_frame(obs):
    """取 agentview 图像翻转后用于显示/录制（保持原始分辨率）。"""
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

## 二、服务器端（GPU 机器）

服务器端配置与 aloha_sim 完全相同。

### 安装

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

curl -LsSf https://astral.sh/uv/install.sh | sh

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 启动 Policy Server

```bash
uv run scripts/serve_policy.py --env LIBERO
```

- 使用 `pi05_libero` config（`src/openpi/training/config.py` 中已定义）
- 权重自动下载到 `~/.cache/openpi`
- 默认监听 `0.0.0.0:8000`，确保防火墙放开 8000 端口

---

## 三、本地 WSL2（模拟器）

### 安装系统依赖

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libglfw3 libglfw3-dev
```

如需实时渲染窗口还需要：

```bash
sudo apt-get install -y libopencv-dev python3-opencv
```

### 创建 conda 环境

> **必须用 Python 3.8**，LIBERO 的依赖 `numba==0.53.1` 不支持 Python 3.9+，无法复用 aloha_sim 的 conda 环境。

```bash
conda create -n libero_sim python=3.8 -y
conda activate libero_sim
```

### 安装 Python 依赖

```bash
cd /path/to/openpi

# 安装依赖（注意 PyTorch 需要 CUDA 11.3 的源）
pip install -r examples/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 从 PyPI 安装 libero（无需 git submodule）
pip install libero

pip install -e packages/openpi-client
```

---

## 四、运行

> **注意**：该脚本使用 `tyro` 嵌套参数解析，所有运行参数必须加 `--args.` 前缀，例如 `--args.host`、`--args.port`、`--args.display`。直接写 `--host` 会报 `Unrecognized options` 错误。

### 方案 A：只保存视频（无渲染窗口，最稳定）

```bash
cd /path/to/openpi
conda activate libero_sim
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

MUJOCO_GL=egl python examples/libero/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000 \
    --args.task-suite-name libero_spatial
```

视频保存到：`data/libero/videos/rollout_<task>_<success|failure>.mp4`

### 方案 B：实时渲染窗口 + 保存视频（需要 WSLg，仅 Windows 11）

```bash
MUJOCO_GL=egl python examples/libero/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000 \
    --args.task-suite-name libero_spatial \
    --args.display
```

> **检查 WSLg 是否可用**：运行 `echo $DISPLAY`，有输出（如 `:0`）则可用。Windows 10 不支持 WSLg，请使用方案 A。

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `155.98.36.47` | 服务器 IP |
| `--port` | `9000` | 服务器端口 |
| `--task-suite-name` | `libero_spatial` | 任务套件，可选 `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` / `libero_90` |
| `--replan-steps` | `5` | 每执行多少步重新推理一次 |
| `--num-trials-per-task` | `50` | 每个任务的 episode 数 |
| `--display` | `False` | 是否弹出实时渲染窗口 |
| `--video-out-path` | `data/libero/videos` | 视频保存目录 |

---

## 五、验证连接

启动模拟器前先确认服务器可达：

```bash
nc -zv 155.98.36.47 9000

```

服务器日志出现 `Listening on port 8000` 后再启动本地模拟器。

---

## 六、视频录制说明

| 原始行为 | 修改后行为 |
|---|---|
| 只在 `env.step()` 后保存帧 | 后台线程以 **30fps** 持续录制 |
| 推理延迟不录入视频 | 推理期间重复最后一帧，延迟完整体现 |
| 无实时渲染 | 支持 `--display` 弹出 cv2 窗口 |

录制帧率在 `main.py` 顶部的 `RECORD_FPS = 30` 修改。
