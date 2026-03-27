# Aloha Sim 远程推理指南

模拟器运行在本机 WSL2，模型推理运行在远端 GPU 服务器，通过 WebSocket 通信。

```
[本机 WSL2]                        [GPU 服务器]
  aloha_sim 模拟器   <--websocket-->  π0/π0.5 模型推理
  main.py                            serve_policy.py
  port 9000 (client)                 port 9000 (server)
```

---

## 一、代码修改（支持实时渲染窗口）

默认代码中 `--display` 参数未接入渲染逻辑，需修改以下两个文件。

### `examples/aloha_sim/env.py`

将 `__init__` 和 `apply_action` 修改如下：

```python
def __init__(self, task: str, obs_type: str = "pixels_agent_pos", seed: int = 0, display: bool = False) -> None:
    np.random.seed(seed)
    self._rng = np.random.default_rng(seed)

    render_mode = "human" if display else None
    self._gym = gymnasium.make(task, obs_type=obs_type, render_mode=render_mode)
    self._display = display

    self._last_obs = None
    self._done = True
    self._episode_reward = 0.0

def apply_action(self, action: dict) -> None:
    gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])
    self._last_obs = self._convert_observation(gym_obs)
    self._done = terminated or truncated
    self._episode_reward = max(self._episode_reward, reward)
    if self._display:
        self._gym.render()
```

### `examples/aloha_sim/main.py`

将环境初始化改为：

```python
environment=_env.AlohaSimEnvironment(
    task=args.task,
    seed=args.seed,
    display=args.display,
),
```

---

## 二、服务器端（GPU 机器）

### 安装

```bash
# 克隆仓库
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 启动 Policy Server

```bash
uv run scripts/serve_policy.py --env ALOHA_SIM
```

- 使用 π0.5 base 模型（`gs://openpi-assets/checkpoints/pi05_base`），权重自动下载到 `~/.cache/openpi`
- 默认监听 `0.0.0.0:8000`，确保防火墙放开 8000 端口

> **注意**：仓库默认的 `ALOHA_SIM` 环境使用的是 π0 模型。本指南已在以下两处做了修改以使用 π0.5：
> - `src/openpi/training/config.py`：新增 `pi05_aloha_sim` config
> - `scripts/serve_policy.py`：将 `ALOHA_SIM` 默认 checkpoint 改为 `pi05_base`

---

## 三、本地 WSL2（模拟器）

### 安装系统依赖

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libglfw3 libglfw3-dev
```

### 创建 conda 环境

```bash
conda create -n aloha_sim python=3.10 -y
conda activate aloha_sim
```

### 安装 Python 依赖

```bash
pip install gym-aloha==0.1.1 imageio imageio-ffmpeg matplotlib msgpack \
    "numpy>=1.22.4,<2.0.0" typing-extensions tyro websockets \
    mujoco==2.3.7 gymnasium==1.0.0 pyopengl glfw
```

### 安装 openpi-client

```bash
# 进入 openpi 仓库目录
pip install -e packages/openpi-client
```

---

## 四、运行

> **注意**：该脚本使用 `tyro` 嵌套参数解析，所有运行参数必须加 `--args.` 前缀，例如 `--args.host`、`--args.port`、`--args.display`。直接写 `--host` 会报 `Unrecognized options` 错误。

### 方案 A：保存视频文件（推荐，最稳定）

不需要窗口，视频自动保存为 mp4。

```bash
cd /path/to/openpi
MUJOCO_GL=egl python examples/aloha_sim/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000
```

视频保存路径：`data/aloha_sim/videos/out_0.mp4`

### 方案 B：实时渲染窗口（需要 WSLg，仅 Windows 11）

```bash
cd /path/to/openpi
MUJOCO_GL=glfw python examples/aloha_sim/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000 \
    --args.display
```

> **检查 WSLg 是否可用**：在 WSL2 中运行 `echo $DISPLAY`，有输出（如 `:0`）则可用。Windows 10 不支持 WSLg，请使用方案 A。

---

## 五、验证连接

启动模拟器前，先确认服务器端口可达：

```bash
nc -zv 155.98.36.47 9000
```

服务器日志中看到 `Listening on port 8000` 后再启动本地模拟器。
