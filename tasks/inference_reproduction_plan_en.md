# π₀.₅ Lightweight Inference Reproduction Plan

> **Goals**: (1) Verify hardware supports π₀.₅ inference; (2) Step through model execution with VSCode debugger; (3) Validate server-client architecture for future Isaac Sim integration.

---

## Prerequisites Check

| Item | Status |
|------|--------|
| GPUs | 2× RTX 5060 Ti (16 GB each) |
| JAX | 0.5.3 with CUDA (both GPUs visible) |
| Python venv | `.venv/` under workspace root |
| Checkpoint | `pi05_droid` cached at `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/` |
| VSCode | debugpy available, basic launch.json exists |

---

## Phase 1: Local Inference Smoke Test

**Objective**: Confirm the model loads and produces valid action outputs without any server infrastructure.

### Step 1.1 — Create the test script

Create `scripts/verify_inference.py`:

```python
"""Minimal π₀.₅ inference verification — no server, no robot."""
import logging
import time

import jax
import numpy as np

from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO)

def main():
    # Hardware check
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    assert len(devices) > 0 and "cuda" in str(devices[0]).lower(), \
        f"Expected CUDA device, got: {devices}"

    # Load pi05_droid from local cache
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "gs://openpi-assets/checkpoints/pi05_droid"
    print(f"Loading config: pi05_droid (action_dim={config.model.action_dim}, "
          f"action_horizon={config.model.action_horizon})")

    t0 = time.time()
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # First inference (includes JIT compilation)
    example = droid_policy.make_droid_example()
    t0 = time.time()
    result = policy.infer(example)
    jit_time = time.time() - t0
    print(f"First inference (w/ JIT): {jit_time:.1f}s")

    # Second inference (post-JIT, measures true speed)
    example2 = droid_policy.make_droid_example()
    t0 = time.time()
    result = policy.infer(example2)
    infer_time = (time.time() - t0) * 1000
    print(f"Second inference (post-JIT): {infer_time:.1f}ms")

    # Validate output
    expected_shape = (config.model.action_horizon, 8)  # DROID: 15×8
    assert result["actions"].shape == expected_shape, \
        f"Expected {expected_shape}, got {result['actions'].shape}"
    print(f"Actions shape: {result['actions'].shape} ✓")
    print(f"Action value range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

    print("\n=== Inference verification PASSED ===")
    print(f"Model: π₀.₅ (pi05_droid)")
    print(f"Inference latency: {infer_time:.1f}ms")
    print(f"GPU memory: check with nvidia-smi")

if __name__ == "__main__":
    main()
```

### Step 1.2 — Run from terminal

```bash
source .venv/bin/activate
python scripts/verify_inference.py
```

**Expected output**:
- JAX reports 2 CUDA devices
- First inference: 5-30s (JIT compile)
- Second inference: < 500ms
- Actions shape: `(15, 8)` — confirmed

### Step 1.3 — Check GPU memory

```bash
nvidia-smi
```

Verify VRAM usage is well within 16 GB (expect ~6-10 GB for pi05_droid).

---

## Phase 2: VSCode Step-Through Debugging

**Objective**: Step into key model functions to understand execution flow.

### Step 2.1 — Configure VSCode launch.json

Replace `.vscode/launch.json` with:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Verify Inference (debug)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/verify_inference.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${workspaceFolder}/packages/openpi-client/src"
            },
            "justMyCode": false
        },
        {
            "name": "Serve Policy (debug)",
            "type": "debugpy",
            "request": "launch",
            "module": "openpi.serving.websocket_policy_server",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${workspaceFolder}/packages/openpi-client/src"
            },
            "justMyCode": false,
            "args": []
        },
        {
            "name": "Simple Client (debug)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/examples/simple_client/main.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${workspaceFolder}/packages/openpi-client/src"
            },
            "justMyCode": false,
            "args": ["--env", "droid", "--num_steps", "5"]
        }
    ]
}
```

### Step 2.2 — Set breakpoints for key execution path

Recommended breakpoints (in order of execution):

| # | File | Line/Function | What to observe |
|---|------|---------------|-----------------|
| 1 | `src/openpi/policies/policy.py` | `Policy.infer()` → first line inside | Entry point: raw observation dict received |
| 2 | `src/openpi/models/pi0.py` | `Pi0.sample_actions()` → first line | Model input: `Observation` object |
| 3 | `src/openpi/models/pi0.py` | Inside `Pi0.embed_inputs()` | How images + text → token embeddings |
| 4 | `src/openpi/models/pi0.py` | Inside `Pi0.embed_suffix()` | How action noise + time → action expert tokens |
| 5 | `src/openpi/models/gemma.py` | Attention calculation in transformer block | Self-attention between VLM + action expert |
| 6 | `src/openpi/models/pi0.py` | Flow matching denoising loop | 10-step Euler integration |
| 7 | `src/openpi/policies/droid_policy.py` | `DroidOutputs.__call__()` | How model output → DROID action format |

### Step 2.3 — Debug session walkthrough

1. Open `scripts/verify_inference.py` in VSCode
2. Set a breakpoint at `result = policy.infer(example)` (after the first inference)
3. Press F5 → select "Verify Inference (debug)"
4. Wait for JIT warmup (first `infer` call completes)
5. When breakpoint hits, **Step Into** (F11) the second `policy.infer()` call
6. Follow the execution through: `Policy.infer()` → `Pi0.sample_actions()` → flow matching loop
7. After `sample_actions` returns, inspect `result["actions"]` in the VARIABLES pane

---

## Phase 3: Server-Client Architecture

**Objective**: Run server and client in separate processes, validate WebSocket communication.

### Step 3.1 — Start the policy server

Terminal 1:
```bash
source .venv/bin/activate
python scripts/serve_policy.py --env droid --port 8000
```

**Wait for**: `"Creating server (host: ..., ip: ...)"` — the server is ready.

### Step 3.2 — Run the simple client

Terminal 2:
```bash
source .venv/bin/activate
python examples/simple_client/main.py --env droid --port 8000 --num_steps 10
```

**Expected output**:
```
Server metadata: ...
Running policy: 100% |████████████| 10/10
Timing Statistics
┌─────────────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ Metric           │ Mean  │ Std   │ P25   │ P50   │ P75   │ P90   │ P95   │ P99   │
├─────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ client_infer_ms  │ 12.3  │ 2.1   │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │
│ server_infer_ms  │ 8.5   │ 1.2   │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │
└─────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

### Step 3.3 — Verify the protocol

Run with recording enabled to inspect actual messages:

Terminal 1:
```bash
source .venv/bin/activate
python scripts/serve_policy.py --env droid --port 8000 --record
```

This creates `policy_records/` with `.npy` files of every inference input/output pair.

### Step 3.4 — Understand the message format

The WebSocket protocol uses **msgpack-numpy** encoding:

```python
# What the client sends:
{
    "observation/exterior_image_1_left": np.ndarray(224, 224, 3),  # uint8
    "observation/wrist_image_left":      np.ndarray(224, 224, 3),  # uint8
    "observation/joint_position":        np.ndarray(7,),           # float64
    "observation/gripper_position":      np.ndarray(1,),           # float64
    "prompt": "do something",
}

# What the server returns:
{
    "actions":         np.ndarray(15, 8),  # float32, normalized
    "policy_timing":   {"infer_ms": 8.5},
    "server_timing":   {...},
}
```

This is the exact format Isaac Sim will need to produce/consume.

---

## Phase 4: Isaac Sim Integration Prep

### Step 4.1 — Define Isaac Sim observation interface

The Isaac Sim environment must provide observations matching `DroidInputs` format:

```python
observation = {
    "observation/exterior_image_1_left": camera_rgb,    # (224, 224, 3) uint8
    "observation/wrist_image_left":      wrist_camera_rgb,  # (224, 224, 3) uint8
    "observation/joint_position":        joint_angles,   # (7,) float
    "observation/gripper_position":      gripper_pos,    # (1,) float
    "prompt": "pick up the red cube",                    # str
}
```

### Step 4.2 — Define action application interface

Actions returned by the server are `(15, 8)` normalized arrays. The Isaac Sim controller must:

1. Extract one timestep: `action = actions[step_index]` (or use `ActionChunkBroker`)
2. Denormalize / scale to robot joint ranges
3. Apply as joint velocity targets: `robot.set_joint_velocity_targets(action[:7])`
4. Apply gripper command: `gripper.set_position(1.0 if action[7] > 0.5 else 0.0)`

### Step 4.3 — Client integration template

```python
"""Template for Isaac Sim → π₀.₅ policy server integration."""
from openpi_client import websocket_client_policy

# Connect to server (can be on same machine or remote)
policy = websocket_client_policy.WebsocketClientPolicy(
    host="localhost",  # or IP of GPU server
    port=8000,
)

def get_action(obs_dict: dict) -> np.ndarray:
    """Get action from π₀.₅ policy server.
    Args:
        obs_dict: observation dict in DROID format
    Returns:
        action: (8,) numpy array [joint_velocities(7), gripper(1)]
    """
    result = policy.infer(obs_dict)
    # Take first action from the predicted chunk
    return result["actions"][0]  # (8,)
```

---

## Expected Timeline

| Phase | Step | Time |
|-------|------|------|
| 1 | Local inference smoke test | 5 min |
| 2 | VSCode debugging setup + walkthrough | 15 min |
| 3 | Server-client verification | 10 min |
| 4 | Isaac Sim integration prep | N/A (future) |

**Total**: ~30 minutes to full verification.

---

## Verification Checklist

- [ ] `verify_inference.py` runs without error, produces `(15, 8)` actions
- [ ] Post-JIT inference latency < 500ms
- [ ] GPU VRAM usage < 16 GB during inference
- [ ] VSCode debugger steps into `Policy.infer()` successfully
- [ ] VSCode debugger steps into `Pi0.sample_actions()` successfully
- [ ] VSCode debugger shows action tensor values in VARIABLES pane
- [ ] Policy server starts without error
- [ ] Simple client connects and receives valid actions
- [ ] Timing stats show server_infer_ms < 100ms
- [ ] `--record` mode produces policy_records/ files