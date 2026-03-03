# RM75 Data Collection & pi0.5 Fine-Tuning Guide

This is a hands-on guide for collecting demonstration data on the RM75 7-DoF arm platform and fine-tuning a pi0.5 VLA model using the openpi pipeline. No research modules (SpaceCIL/LunarCompose) are involved — this is vanilla openpi fine-tuning.

---

## 1. Platform Specification

| Property | Value |
|---|---|
| Arm | RM75, 7 DoF |
| Gripper | Two-finger, 1 DoF |
| Wrist Camera | Wrist-mounted RGB (hand-eye calibrated) |
| Scene Camera | Fixed scene/base RGB camera |
| Action space | Absolute Joint Position (7D) + Gripper (1D) = **8D total** |
| State space | Joint Position (7D) + Gripper (1D) = **8D total** |
| State (recorded but not in model input) | Joint Velocity (7D) — stored in dataset for future use |

> **Two cameras are used**: wrist-mounted RGB and a fixed scene/base RGB camera. The wrist image maps to `left_wrist_0_rgb` and the scene image maps to `base_0_rgb` in the model's internal representation. The third image slot (`right_wrist_0_rgb`) is zero-padded and masked out.

---

## 2. Exactly What Data To Collect

For each demonstration episode, your data collection code must record **synchronized, per-timestep** snapshots of the following fields. All fields must have the same number of timesteps N within a single episode.

### 2.1 Observations (what the robot sees/feels at timestep t)

| Field | Shape | Dtype | Unit / Range | Description |
|---|---|---|---|---|
| `joint_position` | `(7,)` | `float32` | **radians** | Current absolute joint angles of all 7 joints. Read directly from joint encoders. Joint order must be consistent across all episodes (typically joint 1 → joint 7 from base to wrist). |
| `joint_velocity` | `(7,)` | `float32` | **radians/second** | Current joint angular velocities. Read from joint encoders or differentiated from position. Same joint ordering as `joint_position`. |
| `gripper_position` | `(1,)` | `float32` | **normalized [0.0, 1.0]** | Current gripper opening. `0.0` = fully closed, `1.0` = fully open. Must be a length-1 array, NOT a scalar. |
| `wrist_image` | `(H, W, 3)` | `uint8` | **pixel values [0, 255]** | RGB image from the wrist-mounted camera. Channel order is **RGB** (not BGR). Any resolution ≥224×224 is acceptable (resized to 224×224 during training by `ResizeImages(224, 224)`). |
| `scene_image` | `(H, W, 3)` | `uint8` | **pixel values [0, 255]** | RGB image from the fixed scene/base camera. Provides a wider view of the workspace. Channel order is **RGB**. Any resolution ≥224×224 is acceptable (resized to 224×224 during training). |

### 2.2 Actions (what the robot should do at timestep t)

| Field | Shape | Dtype | Unit / Range | Description |
|---|---|---|---|---|
| `action_joint` | `(7,)` | `float32` | **radians** | **Target** absolute joint position at this timestep. This is the command sent to the joint position controller — i.e., "where should each joint be next." Same joint ordering as `joint_position`. |
| `action_gripper` | `(1,)` | `float32` | **normalized [0.0, 1.0]** | **Target** gripper position. `0.0` = close, `1.0` = open. Must be a length-1 array. |

### 2.3 Critical Details

**Units — do not mix these up:**

- Joint positions and actions are in **radians**, NOT degrees. If your robot SDK returns degrees, convert: `rad = deg * (math.pi / 180)`.
- Gripper is **normalized to [0, 1]**. If your gripper SDK returns raw encoder ticks or millimeters, you must normalize: `normalized = (raw - raw_min) / (raw_max - raw_min)`.

**Actions are absolute targets, NOT deltas:**

- `action_joint[t]` is the absolute joint position the robot should move to at timestep t — NOT "how much to move from the current position."
- The openpi pipeline internally converts absolute → delta for training and delta → absolute for inference. You just record absolute targets.

**How to obtain action labels during teleoperation:**

- If using **leader-follower teleop**: the action at timestep t is typically the follower's joint position at timestep t+1 (one step lookahead), OR the leader's current commanded position. Choose one approach and be consistent.
- If using **kinesthetic teaching** (hand-guiding): the action at timestep t is the recorded joint position at timestep t+1.
- If using **VR/spacemouse teleop with IK**: record the IK-solved absolute joint angles as the action target.

**Image format:**

- RGB channel order (not BGR — if using OpenCV, convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`).
- `uint8` dtype with values in [0, 255]. Do not store as float.
- Any resolution is acceptable (will be resized to 224×224 during training), but avoid extremely low resolution (<128×128).
- **Resolution**: Any resolution ≥224×224 works. The training pipeline automatically resizes all images to 224×224 via `ResizeImages(224, 224)` + `resize_with_pad`. Recommended native resolution: **640×480** or **480×480** for good quality after downscale.

**Synchronization:**

- All fields at timestep t must correspond to the same physical moment. Use a single timestamp per frame.
- Recommended recording frequency: **10–50 Hz** (30 Hz is a good default). The pipeline supports any fps — you just need to specify it during data conversion.

**Language prompt:**

- Provide a clear, specific natural language instruction describing the task (e.g., `"pick up the cloth and wipe the solar panel"`).
- Keep prompts under ~50 words. The pi0.5 tokenizer (PaliGemma) supports up to 200 tokens, but shorter prompts work better.
- Be consistent: use the same prompt wording for all episodes of the same task.
- The prompt is passed as the `--task-label` argument during data conversion and becomes the `prompt` field in the training data.
- During inference, send the same prompt string in the observation dict.

---

## 3. HDF5 File Format

Store each episode as a separate HDF5 file. The existing conversion script (`scripts/convert_rm75_data_to_lerobot.py`) expects this layout.

### 3.1 File structure

```
episode_001.hdf5
├── joint_position      (N, 7)    float32    # observation
├── joint_velocity      (N, 7)    float32    # observation
├── gripper_position    (N, 1)    float32    # observation
├── wrist_image         (N, H, W, 3)  uint8  # observation
├── scene_image         (N, H, W, 3)  uint8  # observation
├── action_joint        (N, 7)    float32    # action
└── action_gripper      (N, 1)    float32    # action
```

Where N = number of timesteps in this episode.

### 3.2 Naming convention

- Place all episode files in a single flat directory.
- File extension: `.hdf5` or `.h5` (both are auto-detected).
- Naming is arbitrary (e.g., `episode_001.hdf5`, `demo_2024_001.h5`). Files are sorted by name during conversion.

### 3.3 Custom HDF5 key names

If your recording code uses different key names inside the HDF5 file (e.g., `images/wrist_rgb` instead of `wrist_image`), you do NOT need to change your code. The conversion script supports a `--key-map` argument to remap keys:

```bash
--key-map '{"wrist_image": "images/wrist_rgb", "action_joint": "cmd/joint_pos"}'
```

Only specify keys you want to override; the rest use defaults.

### 3.4 Minimal Python example for writing one episode

```python
import h5py
import numpy as np

def save_episode(filepath, episode_data):
    """
    episode_data: dict with keys:
        joint_position:   np.ndarray (N, 7)  float32  radians
        joint_velocity:   np.ndarray (N, 7)  float32  rad/s
        gripper_position: np.ndarray (N, 1)  float32  [0, 1]
        wrist_image:      np.ndarray (N, H, W, 3) uint8
        scene_image:      np.ndarray (N, H, W, 3) uint8
        action_joint:     np.ndarray (N, 7)  float32  radians
        action_gripper:   np.ndarray (N, 1)  float32  [0, 1]
    """
    with h5py.File(filepath, "w") as f:
        for key, value in episode_data.items():
            f.create_dataset(key, data=value)

# Example: writing a 100-step episode
N = 100
episode = {
    "joint_position":   np.random.randn(N, 7).astype(np.float32),
    "joint_velocity":   np.random.randn(N, 7).astype(np.float32),
    "gripper_position": np.random.rand(N, 1).astype(np.float32),
    "wrist_image":      np.random.randint(0, 256, (N, 480, 640, 3), dtype=np.uint8),
    "scene_image":      np.random.randint(0, 256, (N, 480, 640, 3), dtype=np.uint8),
    "action_joint":     np.random.randn(N, 7).astype(np.float32),
    "action_gripper":   np.random.rand(N, 1).astype(np.float32),
}
save_episode("data/raw/episode_001.hdf5", episode)
```

### 3.5 Sanity checklist before conversion

Run through this checklist on your first recorded episode before converting the full dataset:

- [ ] `joint_position` values are in radians (typical range: [-π, π] or [-2π, 2π])
- [ ] `joint_velocity` values are in rad/s (not deg/s)
- [ ] `gripper_position` values are in [0, 1] (not raw encoder ticks)
- [ ] `wrist_image` is uint8, shape (N, H, W, 3), RGB channel order
- [ ] `scene_image` is uint8, shape (N, H, W, 3), RGB channel order, ≥224×224
- [ ] `action_joint` is absolute joint target (not delta), in radians
- [ ] `action_gripper` is in [0, 1]
- [ ] All arrays have the same N (timestep count)
- [ ] Multiple episodes have consistent joint ordering

---

## 4. Data Conversion: HDF5 → LeRobot

The conversion script already exists. No code to write here.

### 4.1 Dry run (inspect schema without writing)

```bash
uv run scripts/convert_rm75_data_to_lerobot.py \
    --raw-dir /path/to/your/hdf5/episodes \
    --repo-id myorg/rm75_wipe_solar \
    --task-label "pick up the cloth and wipe the solar panel" \
    --dry-run
```

This prints the detected schema from your first episode file and verifies shapes/dtypes. Always run this first.

### 4.2 Full conversion

```bash
uv run scripts/convert_rm75_data_to_lerobot.py \
    --raw-dir /path/to/your/hdf5/episodes \
    --repo-id myorg/rm75_wipe_solar \
    --task-label "pick up the cloth and wipe the solar panel" \
    --fps 30
```

- `--repo-id`: An identifier for the resulting LeRobot dataset. Use whatever name you like (e.g., `myorg/rm75_wipe_solar`). It determines the output directory under `$HF_LEROBOT_HOME` (typically `~/.cache/huggingface/lerobot/`).
- `--task-label`: The natural language instruction describing the task. This becomes the `prompt` used during training. Choose a clear, specific instruction.
- `--fps`: The recording frequency in Hz (must match your actual recording rate).

The script will:
1. Read all `.hdf5` / `.h5` files from `--raw-dir`
2. Concatenate `action_joint(7)` + `action_gripper(1)` → `actions(8)` per frame
3. Write a LeRobot dataset with dot-notation keys (LeRobot v0.1.0 requirement)
4. Print a verification summary after completion

### 4.3 What the conversion produces

```
~/.cache/huggingface/lerobot/myorg/rm75_wipe_solar/
├── data/                    # Parquet files with numeric data
├── meta/                    # Dataset metadata, episode info
└── images/                  # Stored images (PNG)
```

The LeRobot dataset features:

| LeRobot Key | Source | Shape |
|---|---|---|
| `observation.wrist_image` | `wrist_image` | (H, W, 3) image |
| `observation.scene_image` | `scene_image` | (H, W, 3) image |
| `observation.joint_position` | `joint_position` | (7,) float32 |
| `observation.joint_velocity` | `joint_velocity` | (7,) float32 |
| `observation.gripper_position` | `gripper_position` | (1,) float32 |
| `actions` | `action_joint` ⊕ `action_gripper` | (8,) float32 |

---

## 5. Training Config

A training config named `pi05_rm75_wipe` needs to be registered so that the openpi training pipeline knows how to process your data. This config will be created in the repo (see section 5.2 for exact location and code).

### 5.1 What the config does

The config wires together:
- **Model**: pi0.5 base with LoRA adapters (for parameter-efficient fine-tuning)
- **Data pipeline**: `LeRobotRM75DataConfig` — handles key remapping (dot→slash), image parsing, state vector assembly (8D), and absolute→delta action conversion
- **Weight loader**: Downloads the pi0.5 base checkpoint from Google Cloud
- **Freeze filter**: Freezes everything except LoRA parameters

### 5.2 Config location

The config will be added to `src/openpi/training/config.py` in the `_CONFIGS` list, following the exact pattern of `pi05_libero`.

### 5.3 Training commands

**Step 1: Compute normalization statistics** (required before first training)

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_rm75_wipe
```

This reads your entire LeRobot dataset, computes per-feature mean/std/quantile statistics, and saves them under `./assets/pi05_rm75_wipe/`. Training will fail without this step.

**Step 2: Launch training**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_rm75_wipe \
    --exp-name=wipe_solar_v1 \
    --overwrite
```

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: Lets JAX use 90% GPU memory (vs default 75%). Critical for fitting pi0.5 in memory.
- `--exp-name`: Names your experiment run. Checkpoints save to `checkpoints/pi05_rm75_wipe/wipe_solar_v1/`.
- `--overwrite`: Allows overwriting if rerunning. Remove this flag to prevent accidental overwrites.

**GPU requirements for LoRA fine-tuning**: ≥ 22.5 GB VRAM (RTX 4090 or better).

**Step 3: Monitor training**

- Console output logs loss every 100 steps by default.
- If `wandb_enabled=True` in config (default), training logs to Weights & Biases.
- Checkpoints are saved every 1000 steps.
- Typical fine-tuning: 5,000–30,000 steps depending on dataset size and task complexity.

---

## 6. Inference (Deploying the Trained Model)

### 6.1 Start the policy server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_rm75_wipe \
    --policy.dir=checkpoints/pi05_rm75_wipe/wipe_solar_v1/10000
```

Replace `10000` with the checkpoint step you want to serve. This starts a WebSocket server on port 8000.

### 6.2 Query from your robot code

Install the lightweight client on your robot machine:

```bash
cd packages/openpi-client && pip install -e .
```

Then in your robot control loop:

```python
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Connect to policy server (can be on a different machine)
client = websocket_client_policy.WebsocketClientPolicy(
    host="POLICY_SERVER_IP",  # e.g., "192.168.1.100" or "localhost"
    port=8000,
)

for step in range(num_steps):
    # Read sensors
    joint_pos = robot.get_joint_positions()       # (7,) float32, radians
    joint_vel = robot.get_joint_velocities()       # (7,) float32, rad/s
    gripper_pos = robot.get_gripper_position()     # (1,) float32, [0,1]
    wrist_img = camera.get_rgb_image()             # (H, W, 3) uint8, RGB
    scene_img = scene_camera.get_rgb_image()       # (H, W, 3) uint8, RGB

    # Resize images to 224x224 (matching training)
    wrist_img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )
    scene_img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(scene_img, 224, 224)
    )

    # Build observation dict — keys must match RM75Inputs expectations
    observation = {
        "observation/wrist_image": wrist_img_resized,
        "observation/scene_image": scene_img_resized,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": "pick up the cloth and wipe the solar panel",
    }

    # Get action chunk: shape (action_horizon, 8)
    # Each row: [joint_target_1..7, gripper_target]
    result = client.infer(observation)
    action_chunk = result["actions"]  # (action_horizon, 8) — absolute joint targets

    # Execute first action (or implement action chunking strategy)
    joint_target = action_chunk[0, :7]    # (7,) radians — send to joint position controller
    gripper_target = action_chunk[0, 7]   # scalar [0,1] — send to gripper controller

    robot.move_to_joint_positions(joint_target)
    robot.set_gripper(gripper_target)
```
    # Read sensors
    joint_pos = robot.get_joint_positions()       # (7,) float32, radians
    joint_vel = robot.get_joint_velocities()       # (7,) float32, rad/s
    gripper_pos = robot.get_gripper_position()     # (1,) float32, [0,1]
    wrist_img = camera.get_rgb_image()             # (H, W, 3) uint8, RGB
    scene_img = scene_camera.get_rgb_image()       # (H, W, 3) uint8, RGB

    # Resize image to 224x224 (matching training)
    # Resize images to 224x224 (matching training)
    wrist_img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )
    scene_img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(scene_img, 224, 224)
    )

    # Build observation dict — keys must match RM75Inputs expectations
    observation = {
        "observation/wrist_image": wrist_img_resized,
        "observation/scene_image": scene_img_resized,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": "pick up the cloth and wipe the solar panel",
    }
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )
    scene_img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(scene_img, 224, 224)
    )
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )

    # Build observation dict — keys must match RM75Inputs expectations
    observation = {
        "observation/wrist_image": wrist_img_resized,
        "observation/scene_image": scene_img_resized,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": "pick up the cloth and wipe the solar panel",
    }
        "observation/wrist_image": wrist_img_resized,
        "observation/joint_position": joint_pos,
        "observation/joint_velocity": joint_vel,
        "observation/gripper_position": gripper_pos,
        "prompt": "pick up the cloth and wipe the solar panel",
    }

    # Get action chunk: shape (action_horizon, 8)
    # Each row: [joint_target_1..7, gripper_target]
    result = client.infer(observation)
    action_chunk = result["actions"]  # (action_horizon, 8) — absolute joint targets

    # Execute first action (or implement action chunking strategy)
    joint_target = action_chunk[0, :7]    # (7,) radians — send to joint position controller
    gripper_target = action_chunk[0, 7]   # scalar [0,1] — send to gripper controller

    robot.move_to_joint_positions(joint_target)
    robot.set_gripper(gripper_target)
```

**Key points for inference:**
- The observation dict keys use **slash notation** (`observation/wrist_image`), matching the policy inputs spec.
- State values are sent **unnormalized** — the server handles normalization.
- Returned actions are **absolute joint targets** (the server internally converts delta → absolute).
- `action_chunk` shape is `(action_horizon, 8)`. You typically execute the first row and re-query the policy at the next timestep, OR execute multiple steps open-loop from the chunk.

---

## 7. End-to-End Checklist

```
[ ] 1. Write data collection code (your teleop + recording)
        - Record: joint_position, joint_velocity, gripper_position, wrist_image, scene_image
        - Record: action_joint, action_gripper
        - Save as per-episode HDF5 files (Section 3)

[ ] 2. Collect demonstrations
        - Aim for 50–200 episodes for a single-task fine-tune
        - Vary initial conditions (object placement, lighting)

[ ] 3. Dry-run conversion to verify schema
        uv run scripts/convert_rm75_data_to_lerobot.py \
            --raw-dir <DIR> --repo-id myorg/rm75_wipe_solar \
            --task-label "pick up the cloth and wipe the solar panel" --dry-run

[ ] 4. Full conversion
        uv run scripts/convert_rm75_data_to_lerobot.py \
            --raw-dir <DIR> --repo-id myorg/rm75_wipe_solar \
            --task-label "pick up the cloth and wipe the solar panel" --fps 30

[ ] 5. Register training config (Section 5.2 — will be done in this repo)

[ ] 6. Compute norm stats
        uv run scripts/compute_norm_stats.py --config-name pi05_rm75_wipe

[ ] 7. Train
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run scripts/train.py pi05_rm75_wipe --exp-name=wipe_solar_v1

[ ] 8. Serve & test
        uv run scripts/serve_policy.py policy:checkpoint \
            --policy.config=pi05_rm75_wipe \
            --policy.dir=checkpoints/pi05_rm75_wipe/wipe_solar_v1/<STEP>
```

---

## 8. FAQ & Troubleshooting

**Q: Do I need a scene/base camera?**
A: Yes, a scene/base camera IS required. The RM75 policy maps the scene image to `base_0_rgb` (the model's primary scene view slot). Studies show removing the scene camera can degrade performance significantly (up to 50% success rate drop). Mount a fixed RGB camera with a wide view of the workspace.

**Q: What if my actions are delta (velocity commands)?**
A: The pipeline expects **absolute joint position targets**. If your teleop produces velocity commands, you need to integrate them into absolute positions before recording. Alternatively, record the actual joint positions as the action targets (i.e., `action_joint[t] = joint_position[t+1]`).

**Q: What if I have different joint ordering?**
A: Pick a consistent ordering (e.g., base-to-wrist: joint1, joint2, ..., joint7) and stick with it across all episodes. The model learns from consistency — it doesn't know your joint names, only the order.

**Q: How many demonstrations do I need?**
A: For a single-task fine-tune (one task instruction), typically:
- 20–50 episodes: may work for simple tasks with low variability
- 50–200 episodes: recommended starting point
- 200+ episodes: for tasks with high variability (diverse objects, placements, lighting)

Start with 50 episodes, train, evaluate, then collect more if needed.

**Q: What if compute_norm_stats fails with "repo_id not found"?**
A: The `--repo-id` in the conversion script must match the `repo_id` in the training config. Check that the LeRobot dataset exists at `~/.cache/huggingface/lerobot/<repo_id>/`.

**Q: Training runs out of GPU memory?**
A: Options in order of preference:
1. Ensure `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is set
2. Reduce `batch_size` in the config (e.g., from 32 to 16)
3. Use `--fsdp-devices N` for multi-GPU sharding
4. Use a GPU with more VRAM (A100 80GB recommended for full fine-tune)

**Q: Can I do full fine-tuning instead of LoRA?**
A: Yes. Use the `pi05_rm75_wipe_full` config variant (if created), which removes the LoRA constraint and freeze filter. Requires ≥70 GB VRAM (A100/H100).

**Q: My gripper has a different range (e.g., 0–255 encoder ticks)?**
A: Normalize to [0, 1] in your recording code: `normalized = (raw - raw_min) / (raw_max - raw_min)`. The pipeline clips gripper values to [0, 1] as a safety measure.

**Q: What resolution should I record images at?**
A: Any resolution ≥224×224 works. The training pipeline automatically resizes to 224×224 via `ResizeImages(224, 224)`. Recommended: 640×480 or 480×480 native. Wrist and scene cameras can have different resolutions.

**Q: What should the language prompt look like?**
A: Use a clear, specific natural language instruction (e.g., "pick up the cloth and wipe the solar panel"). Keep it under ~50 words. The pi0.5 PaliGemma tokenizer has a 200-token limit. Use the same prompt for all episodes of the same task. The prompt is set via `--task-label` during conversion and sent as `"prompt"` during inference.

**Q: Is joint velocity used by the model?**
A: No. Pi0.5 does NOT consume state as a model input (unlike pi0, it has no `state_proj` layer). The 8D state vector (joint_position + gripper_position) is only used internally for absolute↔delta action conversion. Joint velocity is recorded in the dataset for potential future use but is ignored during training and inference.
