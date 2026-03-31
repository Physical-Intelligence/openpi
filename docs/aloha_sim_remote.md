# Aloha Sim Remote Inference Guide

The simulator runs on the local WSL2 machine, while model inference runs on a remote GPU server, communicating via WebSocket.

```
[Local WSL2]                        [GPU Server]
  aloha_sim simulator  <--websocket-->  π0/π0.5 model inference
  main.py                              serve_policy.py
  port 9000 (client)                   port 9000 (server)
```

---

## 1. Code Modifications (Enable Real-Time Rendering Window)

The default code does not wire the `--display` flag into the rendering logic. The following two files need to be modified.

### `examples/aloha_sim/env.py`

Modify `__init__` and `apply_action` as follows:

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

Change the environment initialization to:

```python
environment=_env.AlohaSimEnvironment(
    task=args.task,
    seed=args.seed,
    display=args.display,
),
```

---

## 2. Server Side (GPU Machine)

### Installation

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Start the Policy Server

```bash
uv run scripts/serve_policy.py \
    --env ALOHA_SIM \
    policy:checkpoint \
    --policy.config pi05_aloha_sim \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"
```

- Uses the `pi05_aloha_sim` config with the converted PyTorch π0.5 base checkpoint at `~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch`
- Listens on `0.0.0.0:8000` by default — make sure port 8000 is open in the firewall

> **Note:** The upstream repository's `ALOHA_SIM` environment defaults to the π0 model. This guide has been modified in the following two places to use π0.5 instead:
> - `src/openpi/training/config.py`: added a `pi05_aloha_sim` config
> - `scripts/serve_policy.py`: changed the `ALOHA_SIM` default checkpoint to `pi05_base`

---

## 3. Local WSL2 (Simulator)

### Install System Dependencies

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libglfw3 libglfw3-dev
```

### Create a Conda Environment

```bash
conda create -n aloha_sim python=3.10 -y
conda activate aloha_sim
```

### Install Python Dependencies

```bash
pip install gym-aloha==0.1.1 imageio imageio-ffmpeg matplotlib msgpack \
    "numpy>=1.22.4,<2.0.0" typing-extensions tyro websockets \
    mujoco==2.3.7 gymnasium==1.0.0 pyopengl glfw
```

### Install openpi-client

```bash
# Navigate to the openpi repository directory
pip install -e packages/openpi-client
```

---

## 4. Running

> **Note:** This script uses `tyro` nested argument parsing. All runtime arguments must be prefixed with `--args.`, e.g., `--args.host`, `--args.port`, `--args.display`. Using `--host` directly will cause an `Unrecognized options` error.

### Option A: Save Video Files (Recommended, Most Stable)

No window required — videos are automatically saved as mp4.

```bash
cd /path/to/openpi
MUJOCO_GL=egl python examples/aloha_sim/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000
```

Video output path: `data/aloha_sim/videos/out_0.mp4`

### Option B: Real-Time Rendering Window (Requires WSLg, Windows 11 Only)

```bash
cd /path/to/openpi
MUJOCO_GL=glfw python examples/aloha_sim/main.py \
    --args.host 155.98.36.47 \
    --args.port 9000 \
    --args.display
```

> **Check if WSLg is available:** Run `echo $DISPLAY` in WSL2. If there is output (e.g., `:0`), WSLg is available. Windows 10 does not support WSLg — use Option A instead.

---

## 5. Verify Connection

Before starting the simulator, confirm the server port is reachable:

```bash
nc -zv 155.98.36.47 9000
```

Start the local simulator only after the server log shows `Listening on port 8000`.
