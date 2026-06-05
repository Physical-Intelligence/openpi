# Isaac Sim Environment Deployment & Inference Integration Plan

> **Goal**: Install Isaac Sim 4.5+ on Ubuntu 24.04, build a Franka Panda robot simulation scene, and bridge to the openpi π₀.₅ policy server via WebSocket for a complete sim-to-action inference loop.

---

## 0. Architecture Overview

```
┌──────────────────────────────┐      WebSocket (msgpack-numpy)       ┌──────────────────────────────┐
│     Isaac Sim Environment     │  ───────────observations───────────> │      openpi .venv/            │
│                              │                                      │                              │
│  • USD Scene (Franka Panda)  │  <────────────actions──────────────── │  • Policy server              │
│  • Camera rendering → obs    │                                      │  • π₀.₅ inference (JAX+GPU)   │
│  • Joint controllers → exec  │                                      │  • pi05_droid checkpoint      │
│                              │                                      │                              │
│  Python: isaacsim (pip)      │                                      │  Python: .venv/ (existing)    │
│  Deps: PyTorch, numpy, etc.  │                                      │  Deps: JAX 0.5.3, flax, etc.  │
└──────────────────────────────┘                                      └──────────────────────────────┘
```

**Core design decision**: Two completely isolated Python environments communicating via WebSocket. Isaac Sim doesn't need JAX; openpi doesn't need sim libraries. This protocol is already verified (see inference reproduction plan, Phase 3).

---

## Phase 1: Isaac Sim Installation & Environment Setup

### 1.1 Installation Method

Using **pip-based Isaac Lab** (built on Isaac Sim 4.5+):
- No Omniverse Launcher GUI required for core components
- Headless-friendly for server/remote environments
- Transparent pip dependency management, easier to co-exist with existing project

### 1.2 Create Isolated Virtual Environment

```bash
# Option A (recommended): dedicated venv under the project
python3.11 -m venv .venv-isaac

# Option B: conda (if already installed)
conda create -n isaacsim python=3.11
```

**Python version note**: Isaac Sim 4.5+ requires Python 3.10 or 3.11. Confirm the exact Python version requirement for the specific Isaac Sim 4.5 release.

### 1.3 Install Isaac Sim pip Packages

```bash
source .venv-isaac/bin/activate

# If using a mirror (recommended for users in China):
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Install Isaac Sim (adjust package names per latest official docs)
pip install isaacsim-rl isaacsim-core  # Isaac Lab + core simulation

# Or full Isaac Sim (all extensions)
pip install isaacsim
```

**Caveats**:
- First run triggers automatic download of NVIDIA Omniverse Kit runtime (several GB) — ensure sufficient disk space
- /home/srh/ currently has ~234 GB free, expected to be sufficient
- May need to configure `NVIDIA_ISAAC_SIM_PATH` and related environment variables

### 1.4 Verify GPU Rendering

```bash
source .venv-isaac/bin/activate

# Start headless mode and verify GPU access
python -c "
from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': True})
import omni.usd
print('Isaac Sim GPU rendering ready')

import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
simulation_app.close()
"
```

**Expected result**: PyTorch reports CUDA available, device name "NVIDIA GeForce RTX 5060 Ti".

### 1.5 Environment Variables & Launch Script

Create `scripts/isaac_env.sh` for unified environment management:

```bash
# Activate openpi environment (policy server)
alias openpi-env='source /home/srh/VLA/openpi/.venv/bin/activate'

# Activate Isaac Sim environment
alias isaac-env='source /home/srh/VLA/openpi/.venv-isaac/bin/activate'
```

---

## Phase 2: Franka Panda Base Scene

### 2.1 Create USD Scene

Create the scene programmatically via Python script:

```python
"""scripts/isaac_scenes/create_franka_scene.py — Create base Franka Panda tabletop scene."""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
import numpy as np

# Initialize world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Load Franka Panda
# Isaac Sim built-in Franka asset path
franka_usd = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
franka = world.scene.add(Robot(
    prim_path="/World/Franka",
    name="franka",
    usd_path=franka_usd,
    position=np.array([0.0, 0.0, 0.0]),
))

# Add table
from omni.isaac.core.objects import DynamicCuboid
table = world.scene.add(DynamicCuboid(
    prim_path="/World/Table",
    name="table",
    position=np.array([0.5, 0.0, 0.0]),
    scale=np.array([0.5, 0.5, 0.05]),
    color=np.array([0.5, 0.3, 0.2]),
))

# Initialize physics
world.reset()

# Save USD scene
from omni.usd import get_context
get_context().save_as_stage("/home/srh/VLA/openpi/scripts/isaac_scenes/franka_tabletop.usd")
print("Scene saved.")

simulation_app.close()
```

### 2.2 Camera Configuration

DROID format requires two camera views:

| Camera | Key | Position | Resolution |
|--------|-----|----------|------------|
| Exterior camera 1 (left) | `observation/exterior_image_1_left` | In front/side of robot | 224×224×3 |
| Wrist camera (left) | `observation/wrist_image_left` | Franka end-effector | 224×224×3 |

```python
# Add exterior camera
exterior_camera = world.scene.add(Camera(
    prim_path="/World/ExteriorCamera",
    name="exterior_camera",
    position=np.array([0.8, 0.3, 0.6]),
    target=np.array([0.3, 0.0, 0.2]),
    resolution=(224, 224),
))

# Add wrist camera (attached to Franka end-effector)
wrist_camera = world.scene.add(Camera(
    prim_path="/World/Franka/panda_hand/wrist_camera",
    name="wrist_camera",
    position=np.array([0.0, 0.05, -0.05]),
    resolution=(224, 224),
))
```

### 2.3 Add Interactive Objects

Place manipulable objects on the table for pick-and-place tasks:

```python
objects = []
for name, pos, color in [
    ("red_cube", (0.4, 0.1, 0.35), (1.0, 0.0, 0.0)),
    ("blue_cube", (0.5, -0.1, 0.35), (0.0, 0.0, 1.0)),
    ("green_cube", (0.45, 0.0, 0.35), (0.0, 1.0, 0.0)),
]:
    obj = world.scene.add(DynamicCuboid(
        prim_path=f"/World/Objects/{name}",
        name=name,
        position=np.array(pos),
        scale=np.array([0.03, 0.03, 0.03]),
        color=np.array(color),
    ))
    objects.append(obj)
```

---

## Phase 3: Observation Pipeline — Isaac Sim → DROID Format

### 3.1 Observation Collector Class

Create `scripts/isaac_scenes/droid_observation.py`:

```python
"""DROID-format observation collector — extracts observations from Isaac Sim scene."""
import numpy as np
from typing import Dict


class DroidObservationCollector:
    """Collect DROID-format observations from the simulation scene.

    Output format matches pi05_droid policy expectations exactly:
    {
        "observation/exterior_image_1_left": np.ndarray(224, 224, 3),  # uint8
        "observation/wrist_image_left":      np.ndarray(224, 224, 3),  # uint8
        "observation/joint_position":        np.ndarray(7,),           # float64
        "observation/gripper_position":      np.ndarray(1,),           # float64
        "prompt": str,                                                # task instruction
    }
    """

    def __init__(self, exterior_camera, wrist_camera, franka_articulation):
        self._exterior = exterior_camera
        self._wrist = wrist_camera
        self._franka = franka_articulation

    def get_observation(self, prompt: str = "") -> Dict:
        # Get joint positions (Franka has 7 joints)
        joint_positions = self._franka.get_joint_positions()  # (7,)

        # Get gripper position
        gripper_pos = self._franka.get_gripper_position()  # scalar

        # Render camera images
        exterior_img = self._capture_rgb(self._exterior)  # (224, 224, 3) uint8
        wrist_img = self._capture_rgb(self._wrist)        # (224, 224, 3) uint8

        return {
            "observation/exterior_image_1_left": exterior_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_positions.astype(np.float64),
            "observation/gripper_position": np.array([gripper_pos], dtype=np.float64),
            "prompt": prompt,
        }

    def _capture_rgb(self, camera) -> np.ndarray:
        """Trigger render and get RGBA → RGB uint8 image."""
        # Isaac Sim camera rendering pipeline
        # Exact API depends on Isaac Sim 4.5 Camera interface
        ...
```

### 3.2 Image Format Validation

Ensure output images match real DROID data format exactly:

| Property | Requirement |
|----------|-------------|
| Resolution | 224 × 224 |
| Channels | RGB (3 channels) |
| Data type | uint8 |
| Value range | [0, 255] |
| Color space | sRGB (consistent with DROID dataset) |
| Orientation | Standard (adjust if DROID expects flipped images) |

---

## Phase 4: WebSocket Bridge — Isaac Sim ↔ Policy Server

### 4.1 Policy Client Wrapper

Create `scripts/isaac_scenes/policy_bridge.py`:

```python
"""Policy bridge — Isaac Sim ↔ Policy Server WebSocket communication."""
import sys
sys.path.insert(0, "/home/srh/VLA/openpi/packages/openpi-client/src")

import numpy as np
from openpi_client import websocket_client_policy
from openpi_client.action_chunk_broker import ActionChunkBroker


class PolicyBridge:
    """Manages WebSocket connection to policy server, wraps obs → action calls."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port
        )
        metadata = self._policy.get_server_metadata()
        action_horizon = metadata.get("action_horizon", 15)
        self._broker = ActionChunkBroker(action_horizon=action_horizon)

    def infer(self, obs: dict) -> np.ndarray:
        """Get a single action.

        Args:
            obs: DROID-format observation dict
        Returns:
            action: (8,) numpy array [joint_vel(7), gripper(1)]
        """
        result = self._policy.infer(obs)
        action_chunk = result["actions"]  # (15, 8)
        return self._broker.get_action(action_chunk)

    def close(self):
        self._policy.close()
```

### 4.2 Launch Sequence

```bash
# Terminal 1: Start openpi policy server
source .venv/bin/activate
python scripts/serve_policy.py --env droid --port 8000

# Terminal 2: Run Isaac Sim main loop
source .venv-isaac/bin/activate
python scripts/isaac_scenes/main_loop.py
```

**Launch checklist**:
- [ ] Policy server prints "Creating server..." ready log
- [ ] Isaac Sim main loop prints "Connected to policy server at ws://localhost:8000"
- [ ] First inference completes (including JIT compilation warmup)

---

## Phase 5: Action Execution — Policy Output → Franka Joint Control

### 5.1 Action Decoding

The policy server returns a `(15, 8)` action array. Decoding:

```python
def decode_action(raw_action: np.ndarray) -> dict:
    """Decode openpi raw action into Franka control commands.

    Args:
        raw_action: (8,) numpy array
            - raw_action[:7]: normalized joint velocities (or position deltas)
            - raw_action[7]: gripper command (0=close, 1=open)
    Returns:
        {
            "joint_velocities": np.ndarray(7,),  # joint velocities (rad/s)
            "gripper_command": float,             # 0.0 (close) or 1.0 (open)
        }
    """
    # Velocity scaling (determine scale factor from pi05_droid normalization range)
    velocity_scale = 1.0  # TBD: adjust based on actual normalization range
    joint_velocities = raw_action[:7] * velocity_scale

    # Gripper binarization
    gripper_command = 1.0 if raw_action[7] > 0.5 else 0.0

    return {
        "joint_velocities": joint_velocities,
        "gripper_command": gripper_command,
    }
```

### 5.2 Franka Controller

```python
class FrankaController:
    """Joint velocity controller for Franka Panda in Isaac Sim."""

    def __init__(self, franka_articulation):
        self._franka = franka_articulation
        self._dof = 7

    def apply_action(self, decoded: dict):
        """Apply action to the simulated robot."""
        # Set joint velocity targets
        self._franka.set_joint_velocity_targets(decoded["joint_velocities"])

        # Set gripper
        self._franka.set_gripper_position(decoded["gripper_command"])
```

### 5.3 Control Frequency

DROID's original control frequency is 15 Hz. Simulation control loop:

```python
import time

SIMULATION_DT = 1.0 / 60.0    # Physics step 60 Hz
POLICY_DT = 1.0 / 15.0        # Policy control frequency 15 Hz

while simulation_app.is_running():
    t = time.time()

    # Physics stepping
    world.step(render=True)

    # Query policy every POLICY_DT seconds
    if t - last_policy_time >= POLICY_DT:
        obs = collector.get_observation(prompt="pick up the red cube")
        action = bridge.infer(obs)
        decoded = decode_action(action)
        controller.apply_action(decoded)
        last_policy_time = t
```

---

## Phase 6: End-to-End Inference Main Loop

### 6.1 Main Loop Script

Create `scripts/isaac_scenes/main_loop.py`:

```python
"""End-to-end inference main loop — Isaac Sim simulation + π₀.₅ live control.

Usage:
    # Headless (default)
    python scripts/isaac_scenes/main_loop.py

    # GUI mode (visual debugging)
    python scripts/isaac_scenes/main_loop.py --gui

    # Recording mode (save obs/action pairs)
    python scripts/isaac_scenes/main_loop.py --record

Environment variables:
    ISAAC_HEADLESS=0      Enable GUI
    POLICY_HOST=localhost  Policy server address
    POLICY_PORT=8000       Policy server port
"""
import argparse
import time
import numpy as np

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Enable GUI rendering")
    parser.add_argument("--record", action="store_true", help="Record obs/action data")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--prompt", type=str, default="", help="Task instruction")
    args = parser.parse_args()

    # Initialize
    simulation_app = SimulationApp({"headless": not args.gui})

    # TODO: Create scene, load Franka, configure cameras, connect to policy server

    # Main loop
    for step in range(args.num_steps):
        t0 = time.time()

        # 1. Collect observation
        obs = collector.get_observation(prompt=args.prompt)

        # 2. Query policy
        action = bridge.infer(obs)

        # 3. Execute action
        controller.apply_action(decode_action(action))

        # 4. Physics step
        world.step(render=True)

        elapsed = (time.time() - t0) * 1000
        if step % 10 == 0:
            print(f"Step {step:3d}: {elapsed:.1f}ms total")

    print(f"\nAverage inference latency: {np.mean(latencies):.1f}ms")
    print(f"Average loop time: {np.mean(loop_times):.1f}ms")

    simulation_app.close()


if __name__ == "__main__":
    main()
```

### 6.2 Verification Checklist

After end-to-end testing, confirm:

- [ ] Isaac Sim starts successfully (headless mode), no CUDA/rendering errors
- [ ] Franka Panda model loads correctly, joints are controllable
- [ ] Both cameras render correctly, image resolution 224×224
- [ ] Observation dict format matches `DroidInputs` exactly
- [ ] WebSocket connection succeeds, client receives server metadata
- [ ] First inference returns valid (15, 8) action tensor
- [ ] Action values are in reasonable range (no NaN, no Inf)
- [ ] Franka joints move in response to policy output
- [ ] Single loop time < ~200ms (rendering + inference + physics)
- [ ] GUI mode visualizes correctly (manual verification)

---

## Phase 7: Debugging & Performance Analysis

### 7.1 Latency Breakdown

```
Total loop time = camera render + network(obs) + server inference + network(action) + physics step
```

Expected latency per stage:

| Stage | Expected | Notes |
|-------|----------|-------|
| Camera rendering | ~10ms | GPU render 224×224 |
| Network (round-trip) | <5ms | Local loopback |
| Server inference | <100ms | After JIT compilation |
| Physics step | ~5ms | 60 Hz fixed step |
| **Total** | **~120ms** | Supports ~8 Hz real-time control |

### 7.2 Debugging Notes

1. **GPU Memory**: Isaac Sim rendering + JAX inference — each on its own GPU, or sharing?
   - RTX 5060 Ti (16 GB): Isaac Sim ~2-4 GB for rendering, JAX model ~6-10 GB
   - Ideal: Isaac Sim on GPU 0 (rendering), JAX on GPU 1 (inference)
2. **nvidia-smi monitoring**: Watch memory usage and utilization on both GPUs
3. **Isaac Sim logs**: Kit logs under `~/.nvidia-omniverse/logs/`

---

## Dependency & Compatibility Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Isaac Sim 4.5 pip package Python version requirement | May be incompatible with system Python | Use dedicated venv, confirm Python version before install |
| CUDA 13.0 compatibility | Isaac Sim may require a specific CUDA version | Check Isaac Sim 4.5 CUDA requirements; driver is backward-compatible |
| Slow downloads (China) | NVIDIA Omniverse runtime first-time download | Configure proxy or find local mirror sources |
| Dual GPU allocation | Isaac Sim and openpi competing for the same GPU | Isolate via CUDA_VISIBLE_DEVICES |
| RTX 5060 Ti too new | Isaac Sim may not have been tested on this GPU | Verify rendering works before deep development |

---

## Final File Structure

After completion, the project file structure will be:

```
scripts/
├── isaac_env.sh                  # Environment variables & aliases
├── isaac_scenes/
│   ├── create_franka_scene.py    # Scene generation script
│   ├── franka_tabletop.usd       # Saved USD scene (generated programmatically)
│   ├── droid_observation.py      # DROID-format observation collector
│   ├── policy_bridge.py          # WebSocket policy bridge
│   └── main_loop.py              # End-to-end inference main loop
├── serve_policy.py               # (existing) Policy server entry point
└── verify_inference.py           # (existing) Inference smoke test

.venv/                             # (existing) openpi Python environment
.venv-isaac/                       # (new) Isaac Sim Python environment
```

---

## Time Estimate

| Phase | Content | Estimated Time |
|-------|---------|----------------|
| 1 | Isaac Sim pip install + GPU verification | 1-2 hours (incl. downloads) |
| 2 | Franka scene setup + camera config | 1 hour |
| 3 | DROID observation pipeline | 1 hour |
| 4 | WebSocket bridge | 0.5 hours (existing infrastructure) |
| 5 | Action decoding + Franka control | 1 hour |
| 6 | End-to-end main loop + testing | 1 hour |
| **Total** | | **5-7 hours** (incl. debugging) |

---

## Future Extension Directions

- [ ] Multi-object pick-and-place task environment
- [ ] Domain randomization (lighting, textures, object poses)
- [ ] Multi-view cameras (depth maps, segmentation masks)
- [ ] Parallel simulation (multiple Franka scenes running concurrently)
- [ ] Isaac Sim → real robot sim2real transfer
- [ ] π₀.₅ autoregressive mode inference (for long-horizon tasks)