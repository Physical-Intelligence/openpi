# Experiment Guide: SpaceCIL (Paper A)
## Continual Skill Acquisition on a Real Mobile Manipulation Platform

---

## 1. Overview

### 1.1 Scientific Question

SpaceCIL asks: **can a released pi0.5-class VLA backbone be continually specialized on a real mobile manipulation platform as new operational tasks arrive sequentially, without catastrophically forgetting what it already knows?**

The motivation is operational: lunar surface systems accumulate new task requirements over time. Payload handling comes first, then interface actuation, then cleaning routines, then connector operations. Re-training a full policy from scratch after each new task is slow, risky, and hard to validate. SpaceCIL studies whether parameter-efficient adaptation modules, a learned router, and a small anti-forgetting objective can make continual specialization tractable.

### 1.2 Hypotheses

| ID | Statement |
|---|---|
| **H1** | Task-specialized PEFT modules outperform sequential full fine-tuning and a single shared PEFT module in the continual setting. |
| **H2** | A lightweight language-visual router is sufficient to select task-specialized capability without test-time oracle task identity. |
| **H3** | A small anti-forgetting objective based on prior-task calibration trajectories significantly reduces mission-relevant forgetting. |
| **H4** | An operationally weighted forgetting metric reveals degradation patterns not exposed by uniform continual-learning metrics alone. |

### 1.3 Claim Boundary

When writing the paper, use this exact phrasing:

> "To our knowledge, this is the first real-robot study of continual specialization of a released pi0.5-class VLA on a wheeled mobile manipulation platform under an operationally ordered lunar-analog task stream."

Do not claim unqualified firstness on "continual VLA" as a whole. The scoped claim is the defensible one.

### 1.4 What This Guide Covers

This guide is the complete operational recipe for producing publishable SpaceCIL results:

1. Platform setup and calibration
2. Task suite construction and demo collection
3. Scorer implementation and validation
4. Single-task and continual training
5. Baseline experiment execution
6. Metric computation and result matrix assembly
7. Ablation studies
8. Artifact policy and paper-writing notes

Estimated total timeline from a fresh platform: **8-12 weeks** for a focused team of two researchers with access to the robot 4-5 days per week. The breakdown is roughly: 2 weeks for platform + scorer validation (Gate G1), 4-6 weeks for data collection and training, 2-4 weeks for baselines and ablations.

---

## 2. Prerequisites

### 2.1 Hardware

- RM75 7-DoF arm, mounted on wheeled base
- Two-finger gripper (attached to arm flange)
- **Primary policy camera:** wrist RGB, hand-eye calibrated, attached at the wrist
- **Optional scene camera(s):** fixed-mount overhead or angled, for scoring, replay audit, and debugging. The scene camera is NOT a policy input.
- Emergency stop button within supervisor reach at all times

### 2.2 Software

```bash
# Activate the uv virtual environment
source .venv/bin/activate

# Or run via uv directly
uv run python -c "import openpi; print('ok')"

# Verify JAX sees the GPU
uv run python -c "import jax; print(jax.devices())"

# Verify research package is importable
uv run python -c "from openpi.research.spacecil import config; print('spacecil ok')"
```

Required: Python 3.11+, JAX with CUDA backend (NVIDIA GPU, at least 22.5 GB VRAM for LoRA fine-tuning), uv package manager.

### 2.3 Gate G1 Checks (Must Pass Before Any Continual Experiments)

Before starting any multi-task continual learning work, verify all three of:

1. **One task trains and replays correctly.** Run the single-task training command for `spacecil_rm75_payload` and confirm the checkpoint loads and generates reasonable wrist-camera-conditioned actions.
2. **Wrist-camera policy path works.** The policy must run inference using only `observation/wrist_image`, joint states, and a language prompt. No scene camera inputs during policy execution.
3. **Scorer agrees with manual labels.** For the first task, compare scorer outputs against 20-30 manually labeled episodes. Precision and recall must both exceed 0.80 before continuing.

If any of these fail, stop. Fix the failure. Gate G1 is not optional.

---

## 3. Platform Setup

### 3.1 Robot Configuration

The RM75 arm has 7 revolute joints. The action space throughout SpaceCIL is:

```
action = [q1, q2, q3, q4, q5, q6, q7, gripper]   # 8D
```

where `q1..q7` are **absolute joint angles in radians** and `gripper` is a normalized scalar in `[0, 1]` (0 = open, 1 = closed). This is not delta-action control. Every policy action is a full target configuration, not a displacement from the current pose.

The wheeled base is available operationally but is **not in-policy** for SpaceCIL mainline. The policy does not command the base. The base is driven to a fixed docking position before each episode, and that position is treated as constant across a session.

### 3.2 Camera Calibration

**Hand-eye calibration** (wrist camera extrinsics relative to end-effector flange):

1. Attach an AprilTag or ChArUco board at a fixed scene location.
2. Move the arm to at least 20 diverse joint configurations that keep the tag visible.
3. Record end-effector pose (from forward kinematics) and tag pose in camera frame at each configuration.
4. Solve for the hand-eye transform using a standard AX=XB or AX=YB solver.
5. Verify by projecting known scene points into the camera image. Reprojection error must be below 3 pixels RMS.
6. Store the calibration as `calibration_version` in episode metadata.

**Exposure lock:** Set the wrist camera to manual exposure after the room lights stabilize. Lock shutter speed and gain. Record the exposure settings alongside the calibration version. Do not use auto-exposure during data collection; illumination shifts will corrupt normalization.

**Scene camera pose repeatability:** If using a fixed overhead scene camera for scoring, mount it on a rigid tripod or ceiling bracket. Verify that removing and re-attaching the camera introduces less than 5mm positional drift at the working surface. Check this at the start of every session.

**Recertification rules:**

- Re-run hand-eye calibration if the wrist camera is physically removed or remounted.
- Re-run after any arm crash that involved end-effector contact forces above 50 N.
- Re-check reprojection error at the start of every data collection session. If it exceeds 5 pixels RMS, recertify before collecting.
- Any session collected with an invalid calibration version is hardware-invalid and must be discarded.

### 3.3 Safety Configuration

Freeze these limits in the robot controller before starting any data collection or policy evaluation:

| Parameter | Limit | Notes |
|---|---|---|
| Joint velocity | 0.5 rad/s (max) | Halve for wrist joints (J5-J7) |
| End-effector speed | 0.3 m/s linear | Track via FK at 100 Hz |
| Gripper closure force | 15 N max | Sufficient for all four tasks |
| Insertion timeout | 5 seconds | Abort if connector not mated within 5s |
| Collision stop threshold | 20 N external force | Triggers immediate halt |
| Torque anomaly threshold | 3x nominal torque on any joint | Indicates unexpected contact |
| Supervisor authority | Emergency stop overrides all commands | No software interlock can override |

If a trial enters an unsafe state: stop immediately, log the safety trigger, label the episode `safety-aborted`, and do NOT silently retry. Every safety event must appear in the session log.

---

## 4. Task Suite Design

The four SpaceCIL tasks are **lunar-analog proxy tasks** that are operationally motivated but not one-to-one replicas of lunar mission procedures. They are chosen because they are semantically adjacent to terrestrial manipulation primitives already represented in modern robot datasets, which makes Earth-trained priors applicable.

### 4.1 Task 1: Payload Transfer / Unloading

**Operational analog:** Moving surface cargo between storage zones and deployment sites.

**Terrestrial primitive:** Pick and place.

**Config name:** `spacecil_rm75_payload`

**Object:** Rectangular payload block, approximately 20x10x5 cm, with flat grasping surfaces. Material should provide moderate friction (rubber-coated wood or dense foam work well). Avoid glossy surfaces that create specular highlights in the wrist image.

**Goal state:** Payload resting inside the target zone (marked bin or colored floor patch), confirmed by top-down occupancy check.

**Setup:**
- Define a fixed start region (approximately 30x30 cm workspace patch) where the payload is placed before each episode.
- Define a separate target zone (approximately 25x25 cm) that does not overlap the start region.
- Randomize the payload orientation within ±30 degrees and position within the start region for each reset.
- Record the exact start pose in episode metadata under `object_revision`.

**Scorer:** `PayloadTransferScorer` from `src/openpi/research/shared/scorer_base.py`.

```python
from openpi.research.shared.scorer_base import PayloadTransferScorer
scorer = PayloadTransferScorer(goal_region_threshold=0.1)
result = scorer.score(episode)
# result.success: True/False
# result.confidence: float
# result.fail_type: "drop" | "timeout" | "other" | None
```

The scorer uses gripper state and joint displacement as proxies. Validate it against overhead camera manual labels (see Section 6). If agreement is below 0.80, augment with a physical contact sensor in the target zone (a simple pressure pad works).

**Reset procedure:**
1. Policy trial ends (success or failure).
2. Operator picks up the payload and places it at a new randomized start pose.
3. Log the new start pose as `object_revision`.
4. Arm returns to home configuration.
5. New trial begins.

**Failure modes:** Dropped payload, missed grasp, payload placed in wrong zone, payload falls off table edge.

**Demo collection strategy:** Teleoperate from the wrist-camera view. Grasp, carry, then release over the target zone. Aim for 50-100 accepted demos. Vary the grasp angle and trajectory to get policy coverage of the start-pose distribution.

---

### 4.2 Task 2: Latch or Lever Actuation

**Operational analog:** Actuating mechanical interfaces on habitat modules, equipment panels, or instrument racks.

**Terrestrial primitive:** Lever push or toggle pull.

**Config name:** `spacecil_rm75_latch`

**Object:** Spring-loaded lever with approximately 5 cm travel and two stable states (on/off). Mount the lever fixture at a fixed scene location within arm reach. The lever should require moderate force (3-8 N) to actuate so that weak or glancing contacts register as failures.

**Goal state:** Lever in the target position (typically from "off" to "on"), confirmed by a hall-effect sensor or a high-contrast visual marker that changes state.

**Setup:**
- Fix the lever at a known pose relative to the robot base. Use a jig or mount point that resets repeatably.
- Acceptable lever-pose variation: ±2 mm position, ±2 degrees orientation (within the reset jig).
- If using a visual marker, ensure the scene/scorer camera has a clear view of the marker and the lighting is consistent.

**Scorer:** `LatchActuationScorer` from `src/openpi/research/shared/scorer_base.py`.

```python
from openpi.research.shared.scorer_base import LatchActuationScorer
scorer = LatchActuationScorer(actuation_threshold=0.5)
result = scorer.score(episode)
```

The scorer uses maximum single-joint displacement as a proxy. For high confidence, wire the lever's hall-effect sensor output to the data pipeline and add it as a `details` field in the scorer result. Binary sensor output is more reliable than kinematic proxy.

**Reset procedure:**
1. Operator returns the lever to the start state manually (no arm intervention needed).
2. Verify lever sensor reads the start state.
3. New trial begins.

**Failure modes:** Missed lever entirely, insufficient contact force (glancing), lever overtravel that snaps the spring or damages the fixture.

**Demo collection strategy:** Approach from the direction that the wrist camera provides a clear view of the lever. Slower, more controlled contact is preferable to fast swipes. Collect demos with varied approach angles within ±15 degrees.

---

### 4.3 Task 3: Surface Cleaning / Wiping

**Operational analog:** Dust mitigation on solar panels, sensor windows, or habitat surfaces.

**Terrestrial primitive:** Wipe or scrub.

**Config name:** `spacecil_rm75_clean`

**Object:** Flat 20x20 cm contaminated surface panel mounted at a fixed scene pose. Attach a soft wiper (microfiber cloth or foam pad) to the gripper. The gripper holds the wiper closed during the task; no grasping motion is needed.

**Goal state:** At least 80% of the surface area covered by the wiper path.

**Setup:**
- Apply a uniform layer of washable powder (baby powder or chalk dust works well) or use a dry-erase marker grid as a coverage ground truth.
- The scorer camera must be positioned overhead with a clear view of the entire panel surface.
- **Critical:** The scorer camera must be lighting-controlled and physically isolated from any lighting perturbation applied to the policy view (wrist camera). A pixel-difference scorer that sees the same illumination variation as the policy camera is a confound and will corrupt evaluation. If you cannot isolate the scorer camera view, use a marker-based or segmentation-based coverage signal instead of raw pixel difference.

**Scorer:** `SurfaceCleaningScorer` from `src/openpi/research/shared/scorer_base.py`.

```python
from openpi.research.shared.scorer_base import SurfaceCleaningScorer
scorer = SurfaceCleaningScorer(coverage_threshold=0.5)
result = scorer.score(episode)
# result.details["coverage"]: float in [0, 1]
```

The current implementation estimates coverage by discretizing joint trajectories into bins. For production experiments, replace or augment this with the overhead pixel-difference scorer on the scorer camera. The scorer camera signal is the ground truth; the kinematic proxy is the fallback.

**Scorer note from the masterplan:** If the overhead scorer camera is not lighting-controlled, the wipe task evaluation is a confound and must not be used for primary claims. Fix the scorer before collecting eval episodes.

**Reset procedure:**
1. Operator re-applies powder or re-draws the marker grid.
2. Verify the surface visually (uniform coating before each episode).
3. Return arm to home configuration.

**Failure modes:** Wiper lifts off surface partway through, incomplete coverage (missed edges or corners), wiper slips off the surface panel.

**Demo collection strategy:** Teleoperate a systematic wipe pattern: left-to-right passes, each shifted slightly downward. Collect demos with at least 3 different wipe patterns so the policy learns coverage rather than a single trajectory.

---

### 4.4 Task 4: Connector Mating / Insertion

**Operational analog:** Utility connector mating for power lines, data cables, or fluid transfer.

**Terrestrial primitive:** Peg-in-hole insertion.

**Operational weight: 1.0 (highest in the task suite).** This is the highest-stakes contact task. Errors can damage the connector or the arm. Every safety protocol is active during this task.

**Config name:** `spacecil_rm75_connector`

**Object:** Circular or rectangular connector with approximately 1-2 cm insertion depth and 3-5 mm alignment tolerance. Mount the receptacle at a fixed, rigid scene pose. The connector (plug) is held by the gripper. Avoid compliant mounting for the receptacle; rigid mounting is required so the task difficulty comes from policy alignment, not scene variability.

**Goal state:** Full insertion confirmed by a tactile or electrical contact sensor (preferred). If no sensor is available, use a visual insertion depth classifier on the scorer camera.

**Setup:**
- Fix the receptacle at a known pose with sub-millimeter repeatability. Use a precision jig.
- Acceptable pose variation at the receptacle: ±1 mm position, ±1 degree orientation.
- The connector (plug) is placed in the gripper at a fixed grasp pose at the start of each episode.
- Log the grasp pose and receptacle pose as `object_revision`.

**Scorer:** `ConnectorMatingScorer` from `src/openpi/research/shared/scorer_base.py`.

```python
from openpi.research.shared.scorer_base import ConnectorMatingScorer
scorer = ConnectorMatingScorer(stability_window=5, stability_threshold=0.02)
result = scorer.score(episode)
```

The scorer checks gripper closure and final-state velocity stability. For production, add the contact sensor signal as the primary indicator and keep the kinematic proxy as a secondary check.

**Reset procedure:**
1. Operator disconnects the connector from the receptacle.
2. Gripper releases and arm returns to a pre-grasp position.
3. Operator places the plug at the start grasp pose.
4. New trial begins.

**Failure modes:** Misalignment at approach (connector bends or skips), partial insertion (connector partially engaged but not fully seated), connector drop.

**Demo collection strategy:** This task requires slow, precise teleoperation. Collect demos at a deliberate pace. For each demo, ensure full insertion before releasing. Collect at least 60-80 accepted demos given the precision required.

---

### 4.5 Task Ordering for Continual Learning

The recommended task stream order is:

```
payload → latch → clean → connector
```

This order is motivated operationally (simpler logistics tasks before precision contact tasks) and also provides a useful structure for the continual learning study: forgetting on the simpler tasks (payload, latch) after learning the harder tasks (connector) reveals whether PEFT adaptation generalizes or interferes.

The order matters for the paper's continual learning framing. Do not randomize it during the main experiments. Rotated orders can be an ablation.

---

## 5. Data Collection

### 5.1 Episode Requirements

Every collected episode must carry the following fields, exactly matching the `Episode` schema in `src/openpi/research/shared/episode_schema.py`:

**Observation (per timestep):**
```python
obs.wrist_rgb        # (H, W, 3) uint8 — primary policy camera, required
obs.joint_position   # (7,) float — 7-DoF absolute joint angles
obs.joint_velocity   # (7,) float — joint velocities
obs.gripper_position # (1,) float — gripper state in [0, 1]
obs.scene_rgb        # (H, W, 3) uint8 — optional, scorer/audit only
obs.base_state       # optional, not used in mainline policy
```

**Action (per timestep):**
```python
action.joint_pos     # (7,) float — absolute target joint angles
action.gripper_cmd   # float in [0, 1]
```

The flattened action array fed to the model is 8D: `[joint_pos(7), gripper_cmd(1)]`.

**Metadata (per episode):**
```python
meta.task_id              # e.g. "payload", "latch", "clean", "connector"
meta.env_id               # e.g. "E0" for nominal lab condition
meta.operator_id          # identifies the human demonstrator
meta.session_id           # unique session identifier (date + session number)
meta.camera_preset_id     # exposure settings version
meta.calibration_version  # hand-eye calibration version
meta.scene_revision       # scene layout version
meta.object_revision      # specific object start pose for this episode
```

**Labels (per episode):**
```python
label.success     # bool — did the task succeed?
label.fail_type   # "drop" | "timeout" | "contact" | "other" | None
```

### 5.2 Episode Classification

Every collected episode must be classified as exactly one of:

| Class | Meaning |
|---|---|
| **accepted** | Clean demo meeting all quality criteria |
| **rejected** | Operator-rejected for quality reasons (see criteria below) |
| **safety-aborted** | A safety stop triggered during the episode |
| **hardware-invalid** | Sensor failure, frame drop, or calibration mismatch |

**Criteria for acceptance (ALL must hold):**

- No safety stop triggered during the episode
- No major frame drop in the wrist camera stream (less than 5% of frames dropped)
- No gross initial-state mismatch (object within the declared start range)
- No human hand remaining in the wrist camera view during the core manipulation window
- No ambiguous success label (scorer and operator agree)
- No unintended shortcut (e.g., knocking the payload into the goal zone without grasping)

Rejected, safety-aborted, and hardware-invalid episodes must be kept on disk with their classification. Do not silently delete failures; they are part of the session record.

### 5.3 Collection Protocol

**Teleoperation setup:**
- Display the wrist camera feed on a monitor at the teleoperation station.
- Drive the arm using the wrist-camera view, not an external camera. This conditions the operator to collect data that matches the policy's visual perspective.
- Record the teleoperation joint commands as the episode actions.
- Log at 10-30 Hz depending on task dynamics. Connector mating benefits from higher frequency (30 Hz); payload transfer can be collected at 10-15 Hz.

**Volume targets:**

| Task | Target accepted demos | Notes |
|---|---|---|
| payload | 50-100 | Lower end if pose randomization is small |
| latch | 50-80 | Deterministic reset makes fewer demos adequate |
| clean | 60-100 | Varied wipe patterns required |
| connector | 60-100 | Higher precision: more demos help |
| **Total** | **220-380** | Across all 4 tasks |

**Session doctrine:** Every collection session must log:
- `operator_id` — who is demonstrating
- Date and time
- `calibration_version` — result of start-of-session reprojection check
- `camera_preset_id` — current exposure settings
- `scene_revision` — current scene layout (furniture, lighting, objects)
- `object_revision` — initial object placement range for this session
- Environment condition (E0 nominal for SpaceCIL mainline)
- Any notable anomalies (connector slipped, powder clumped, gripper stuck)

**Operator doctrine:**
- SpaceCIL mainline: single operator per task, or two operators with mixed sessions. Record which.
- If using multiple operators, note this in the paper as a positive (more diverse demonstrations) or control for it in analysis.
- Do not mix demo styles within a session (e.g., some slow precise demos and some fast sloppy ones). Keep style consistent within a session.

### 5.4 Data Format and Conversion

Store episodes in LeRobot v2.0 dataset format. The key mapping from your collection pipeline to the LeRobot dataset is:

```python
# LeRobot dataset key -> episode schema field
"observation/images/wrist"      -> obs.wrist_rgb
"observation/state"             -> [obs.joint_position (7), obs.gripper_position (1)]
"action"                        -> [action.joint_pos (7), action.gripper_cmd (1)]
```

Dataset directory structure:
```
data/
  spacecil_payload/
    data/
      chunk-000/
        episode_000000.parquet
        ...
    videos/
      chunk-000/
        observation.images.wrist/
          episode_000000.mp4
          ...
    meta/
      info.json
      stats.json
      episodes.jsonl
  spacecil_latch/
    ...
  spacecil_clean/
    ...
  spacecil_connector/
    ...
```

The config `repo_id` fields in `spacecil_config.py` currently use placeholders (`placeholder/spacecil_<task>`). Update them to point to your local dataset paths before training:

```python
# In src/openpi/research/spacecil/config.py, the LeRobotRM75DataConfig repo_id
# should point to your actual dataset location, e.g.:
repo_id="local/spacecil_payload"   # or a HuggingFace Hub path
```

### 5.5 Calibration Dataset (Anti-Forgetting Memory)

Separate from the main training set, collect a small calibration set per task. These episodes are held out from training and used only as the behavior distillation memory buffer.

- 10-20 accepted episodes per task
- Collected under the same conditions as training data
- Must be independent of the evaluation set (different object poses, different session)
- Stored alongside the main dataset but flagged with `split: calibration` in metadata

The calibration set is the "memory" that lets the distillation module remember previous tasks. Its quality matters more than its size.

---

## 6. Scorer Validation

Scorer validation is mandatory before any evaluation. A miscalibrated scorer is a confound that can invalidate the entire experimental result.

### 6.1 Per-Task Scorer Setup

| Task | Scorer class | Key parameter | Default value |
|---|---|---|---|
| payload | `PayloadTransferScorer` | `goal_region_threshold` | 0.1 |
| latch | `LatchActuationScorer` | `actuation_threshold` | 0.5 |
| clean | `SurfaceCleaningScorer` | `coverage_threshold` | 0.5 |
| connector | `ConnectorMatingScorer` | `stability_window`, `stability_threshold` | 5, 0.02 |

All scorers live in `src/openpi/research/shared/scorer_base.py`. Import and instantiate them:

```python
from openpi.research.shared.scorer_base import (
    PayloadTransferScorer,
    LatchActuationScorer,
    SurfaceCleaningScorer,
    ConnectorMatingScorer,
)
```

For the connector task, augment `ConnectorMatingScorer` with your hardware contact sensor if available. The kinematic proxy is a fallback, not the preferred signal.

### 6.2 Validation Protocol

For each task, before collecting the full training set:

1. **Collect a pilot subset:** 20-30 episodes collected with varied outcomes (both successes and failures). Include intentional failure modes: dropped payloads, missed levers, incomplete wipes, misaligned connectors.

2. **Manually label each episode:** Operator watches each replay (wrist camera + scene camera if available) and assigns a binary success label independently of the scorer. Record the label and any notes.

3. **Run the scorer on the same episodes:** Compare scorer output against manual labels.

4. **Compute agreement:**

```python
# Assuming labels: list of bool (manual), scorer_results: list of ScorerResult
from sklearn.metrics import precision_score, recall_score

manual = [True, False, True, ...]  # operator labels
auto   = [r.success for r in scorer_results]

p = precision_score(manual, auto)
r = recall_score(manual, auto)
print(f"Precision: {p:.3f}, Recall: {r:.3f}")
```

5. **Accept or revise:** Both precision and recall must exceed 0.80. If either is below this threshold, revise the scorer threshold or add a supplementary sensing signal. Document the revision.

**When to revise the scorer:**

- Precision below 0.80: scorer is calling failures as successes. Increase the threshold parameter.
- Recall below 0.80: scorer is missing real successes. Decrease the threshold parameter or add a secondary signal.
- Agreement is high but confidence is uniformly low: the kinematic proxy is noisy; add the hardware sensor.

Record precision and recall in the paper supplementary material for each task.

---

## 7. Training

### 7.1 Verify the Environment

```bash
# Confirm GPU is visible and JAX can use it
uv run python -c "import jax; print('devices:', jax.devices())"

# Confirm the spacecil config loads
uv run python -c "from openpi.training.config import get_config; c = get_config('spacecil_debug'); print('ok:', c.name)"

# Quick smoke test with the debug config
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py spacecil_debug --overwrite
```

The debug config uses `paligemma_variant="dummy"` and `FakeDataConfig`, so it runs in seconds without downloading checkpoints or loading real data. All code paths are exercised.

### 7.2 Compute Normalization Statistics

Before training any task config, compute normalization stats from the collected dataset. Run this once per task, after the dataset is finalized:

```bash
# Payload task
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_payload

# Latch task
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_latch

# Clean task
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_clean

# Connector task
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_connector
```

These write `norm_stats.json` files that the training pipeline reads at startup. If you add more data to a task, recompute its norm stats.

**Watch for degenerate stats:** Check the `q01`, `q99`, and `std` values in `norm_stats.json`. Joint dimensions that are rarely used can have very small ranges and explode after normalization. Manually clamp any dimension with `std < 0.01` to a safe minimum (e.g., set `std = 0.1`).

### 7.3 Single-Task Training (Gate G1 Verification)

Start with payload transfer. This is the simplest task and the right place to debug the full training pipeline.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py spacecil_rm75_payload \
    --exp-name=g1_payload_verify \
    --overwrite
```

Training runs for `num_train_steps=10_000` by default (from the config). Checkpoints are saved to `checkpoints/spacecil_rm75_payload/g1_payload_verify/`.

**Gate G1 verification steps after training completes:**

1. Load the checkpoint and run inference on a held-out wrist-camera sequence:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=spacecil_rm75_payload \
    --policy.dir=checkpoints/spacecil_rm75_payload/g1_payload_verify/10000
```

2. Execute 10-20 trials on the physical robot. Score each trial.
3. Compare scorer outputs against manual observation. Both must agree at rate above 0.80.
4. If success rate is below 0.50 on the first task: check normalization stats, review demo quality, inspect wrist camera exposure consistency. Do not proceed to multi-task training with a broken single-task baseline.

Repeat for latch, clean, and connector as time allows. You must pass G1 on at least the payload task before starting Section 7.4.

### 7.4 Continual Training with SpaceCIL

With Gate G1 verified, run the full sequential continual training loop:

```bash
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --distillation-alpha 0.5 \
    --checkpoint-dir checkpoints \
    --exp-name spacecil_main_run
```

**What happens internally (`ContinualHarness.run_sequence`):**

```
For each task in [payload, latch, clean, connector]:
  1. Load the task's data config and dataset
  2. Activate a new LoRA adapter for this task
  3. Train: run train_step for num_steps_per_task iterations
     (gradient includes L_task + lambda * L_distill on calibration memory)
  4. Register the trained adapter in TaskAdapterBank, freeze it
  5. Update the teacher snapshot (frozen copy of the full policy state)
  6. Evaluate all tasks seen so far → record a row of the result matrix
```

After the sequence completes, the adapter bank contains four frozen adapters and the result matrix `R[i][j]` is fully populated.

**Estimated training time per task:** Approximately 2-4 hours per 10,000 steps on an A100 (80GB) or RTX 4090. The full four-task sequence takes approximately 8-16 hours of GPU time.

### 7.5 Hyperparameter Choices

| Parameter | Default | Range for ablations | Notes |
|---|---|---|---|
| LoRA rank | 16 (from pi0.5 defaults) | 8, 16, 32 | Higher rank: more capacity, more parameters |
| Training steps per task | 10,000 | 5,000-20,000 | Tune based on loss convergence |
| Distillation weight (lambda) | 0.5 | 0.1, 0.5, 1.0 | See Section 11 for ablation |
| Calibration memory size | 100 episodes/task | 50, 100, 200 | See Section 11 for ablation |
| Batch size | 32 | 16-64 | Reduce if GPU OOM |
| Learning rate | openpi default | — | Do not change without justification |
| EMA decay | None (disabled) | — | Disabled in spacecil configs for speed |

If training diverges (loss spikes or NaN): first check the norm stats for degenerate dimensions, then reduce the learning rate by 2x, then reduce lambda.

---

## 8. Baseline Experiments

All baselines use the same datasets, same task sequence, same evaluation protocol, and same scorer. The only difference is the training configuration.

### 8.1 Sequential Full Fine-Tuning

**What it tests:** The standard catastrophic forgetting baseline. The full model (backbone + all parameters) is updated on each new task. No adapter bank, no distillation.

**Config pattern:** `spacecil_rm75_<task>_fulltune` (no `freeze_filter`, all parameters trainable).

**Command:**
```bash
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --no-distillation \
    --exp-name baseline_fulltune
```

**Expected behavior:** High performance on the most recently trained task (connector), severe forgetting on earlier tasks (payload, latch). This is the baseline that H1 claims to beat.

**Note:** Full fine-tuning requires more GPU memory than LoRA (approximately 70+ GB). If you only have a 24GB GPU, use gradient checkpointing or reduce batch size significantly.

### 8.2 Shared Multi-Task PEFT

**What it tests:** A single LoRA module shared across all tasks, without a per-task adapter bank. The shared LoRA is trained sequentially on each task's data.

**Config pattern:** `spacecil_rm75_shared_lora` (single LoRA, no bank, no distillation).

**Command:**
```bash
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --no-distillation \
    --exp-name baseline_shared_lora
```

**Expected behavior:** Better than full fine-tuning on forgetting (PEFT regularizes implicitly), but worse than SpaceCIL with adapter bank because one LoRA module cannot specialize independently for four tasks.

### 8.3 Per-Task PEFT with Oracle Task ID

**What it tests:** The performance upper bound. Adapter bank is used, but at inference the oracle (ground truth) task ID is provided instead of running the router.

**Config pattern:** `spacecil_rm75_<task>_oracle` (adapter bank enabled, router replaced by ground-truth label).

**Command:**
```bash
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --no-distillation \
    --exp-name baseline_oracle
```

**Expected behavior:** Highest success rates across all tasks (no routing errors). The gap between SpaceCIL (with router) and oracle reveals the router's cost.

### 8.4 SpaceCIL without Distillation

**What it tests:** Whether the adapter bank alone provides enough isolation to prevent forgetting, without the distillation anti-forgetting term.

**Config pattern:** `spacecil_rm75_<task>_nodistill` (lambda=0 in `BehaviorDistillation`).

**Command:**
```bash
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --no-distillation \
    --exp-name ablation_nodistill
```

**Expected behavior:** Adapter bank provides structural isolation (old adapters are frozen), so forgetting is lower than sequential full FT. But the shared backbone and router can still drift. H3 claims distillation further reduces forgetting beyond what the bank alone achieves.

### 8.5 SpaceCIL with Random Routing

**What it tests:** Whether the router matters, or whether any adapter (chosen randomly) would work equally well for a given task.

**Config pattern:** `spacecil_rm75_<task>_randrout` (router replaced by uniform random adapter selection).

**Command:**
```bash
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --distillation-alpha 0.5 \
    --exp-name ablation_randrout
# Add --random-routing flag when that flag is implemented in train_spacecil.py
```

**Expected behavior:** Performance drops significantly below oracle and SpaceCIL, confirming that routing provides real value beyond structural isolation alone.

### 8.6 Summary Table

| Baseline | Key variable | Config flag | Tests |
|---|---|---|---|
| Sequential full FT | All params trainable, no bank | `--no-distillation` + full-tune config | Catastrophic forgetting baseline |
| Shared multi-task PEFT | Single shared LoRA | `--no-distillation` + shared-lora config | Single-module PEFT limitation |
| Per-task PEFT (oracle) | Bank + oracle task ID | `--oracle-routing` | Performance upper bound |
| SpaceCIL no distillation | Bank + router, no distillation | `--no-distillation` | Bank-only isolation |
| SpaceCIL random routing | Bank + random routing | `--random-routing` | Router necessity |
| **SpaceCIL (full)** | **Bank + router + distillation** | **defaults** | **Main system** |

---

## 9. Evaluation and Metrics

### 9.1 The Result Matrix

All evaluation is organized around the result matrix `R[i][j]`:

- `i` = task index after which evaluation is run (0-indexed, 0=after payload training, 3=after connector training)
- `j` = task being evaluated
- `R[i][j]` = success rate on task `j` after training task `i`

For a 4-task sequence (payload=0, latch=1, clean=2, connector=3), the full matrix looks like:

```
         payload  latch  clean  connector
after T0:  R00     ---    ---     ---
after T1:  R10     R11    ---     ---
after T2:  R20     R21    R22     ---
after T3:  R30     R31    R32     R33
```

Entries marked `---` are not evaluated (tasks not yet trained). `R[i][j]` is undefined when `j > i`.

**Evaluation protocol for a single cell `R[i][j]`:**
1. Load the adapter for task `j` from the bank.
2. Run the router to confirm it correctly selects task `j` on task-j inputs.
3. Execute 20 trials on the physical robot on task `j`.
4. Score each trial with the task-j scorer.
5. `R[i][j] = num_successes / 20`

In total: 10 matrix cells × 20 trials each = 200 physical robot trials per full experiment run. Budget accordingly.

### 9.2 Primary Metrics

All metric functions are in `src/openpi/research/spacecil/metrics.py`.

**Average success (final performance):**
```python
from openpi.research.spacecil.metrics import average_success
avg = average_success(result.result_matrix)
# = mean(R[-1, :]) across all tasks
```

**Backward transfer:**
```python
from openpi.research.spacecil.metrics import backward_transfer
bt = backward_transfer(result.result_matrix)
# = mean(R[j][j] - R[-1][j]) for j < T
# Negative: forgetting occurred. Positive: later training helped earlier tasks (rare).
```

**Forgetting:**
```python
from openpi.research.spacecil.metrics import forgetting
f = forgetting(result.result_matrix)
# = mean(max_k R[k][j] - R[-1][j]) for j < T
# Always >= 0. How much each task dropped from its peak.
```

**Operational forgetting (mission-aware):**
```python
from openpi.research.spacecil.metrics import operational_forgetting

# Operational weights per task (higher = more mission-critical)
weights = {
    "connector": 1.0,
    "latch":     0.8,
    "payload":   0.6,
    "clean":     0.5,
}

op_f = operational_forgetting(result, weights)
# = weighted mean forgetting, with higher-stakes tasks counting more
```

### 9.3 Router Diagnostics

These are diagnostic signals, not primary claims. Report them in supplementary or diagnostics sections.

```python
from openpi.research.spacecil.metrics import routing_entropy, routing_accuracy

# routing_entropy: should decrease as the router becomes more confident
# High entropy at the end of training indicates the router is uncertain
entropy = routing_entropy(routing_probs)  # routing_probs: [batch, num_tasks] softmax

# routing_accuracy: fraction of episodes where router's argmax matches the true task
acc = routing_accuracy(predicted_task, true_task)  # both: list or array of task indices
```

Track routing entropy over the training sequence: it should start near `log(num_registered_tasks)` and decrease as training progresses on each task. If it stays high, the router is not learning discriminative features.

### 9.4 Operational Weights

| Task | Weight | Rationale |
|---|---|---|
| connector_mating | 1.0 | Highest-stakes contact task; mission-critical |
| latch_actuation | 0.8 | Interface actuation; failure blocks downstream operations |
| payload_transfer | 0.6 | Logistics; failure delays but does not block |
| surface_cleaning | 0.5 | Maintenance; delayed cleaning is rarely catastrophic |

These weights are defaults from `projects/spacecil/PLAN.md`. They are not hardcoded in the metric function and must be passed explicitly. Change them only with explicit operational justification, and report any change in the paper.

---

## 10. Expected Results and Interpretation

### 10.1 H1 Verification (Task-Specialized PEFT vs. Alternatives)

**Expected result:**
- SpaceCIL (full) > Per-task PEFT (oracle) ≈ upper bound
- SpaceCIL (full) >> Sequential full FT on forgetting
- SpaceCIL (full) > Shared multi-task PEFT on both forgetting and final performance

**Key comparison:** Average success and forgetting across all 4 tasks after the full sequence. Plot the result matrix as a heatmap (rows = after which task, columns = task evaluated). SpaceCIL should show a matrix that stays bright along the diagonal and does not fade dramatically in earlier-task columns.

**If H1 fails:** Investigate whether (a) the backbone is drifting despite LoRA freezing (check router and shared trunk gradients), or (b) the tasks are so similar that per-task specialization provides no advantage over a shared module.

### 10.2 H2 Verification (Router vs. Oracle)

**Expected result:** Router accuracy above 0.80 for all tasks. Gap between SpaceCIL-full and oracle is small (less than 10 percentage points in average success).

**Key comparison:** SpaceCIL-full vs. SpaceCIL-oracle in average success. Also plot routing accuracy per task across training.

**If H2 fails:** The router may be confusing visually similar tasks (e.g., latch and connector both involve close-up contact). Consider adding a language-only routing mode as a fallback. If routing accuracy is above 0.80 but performance gap to oracle is large, the bottleneck is adapter quality, not routing.

### 10.3 H3 Verification (Distillation Reduces Forgetting)

**Expected result:** SpaceCIL-full (with distillation) shows lower forgetting than SpaceCIL-nodistill. The gap should be most visible on the connector task (highest weight) and payload (earliest, most exposed to forgetting).

**Key comparison:** Forgetting and operational forgetting for SpaceCIL-full vs. SpaceCIL-nodistill.

**If H3 fails or is marginal:** Check whether (a) the calibration memory size is too small, (b) the distillation weight lambda is miscalibrated, or (c) the adapter bank's structural isolation is already sufficient that distillation adds little. All three are informative scientific findings.

### 10.4 H4 Verification (Operational vs. Uniform Forgetting)

**Expected result:** Operational forgetting reveals a worse picture than uniform forgetting for at least one baseline, because the connector task (weight 1.0) degrades more than the cleaning task (weight 0.5) in some conditions.

**Key comparison:** Report both `forgetting` (uniform) and `operational_forgetting` (weighted) for each baseline. A table showing these side by side, with a row that shows diverging rankings, is the H4 evidence.

**Example of a strong H4 finding:** Sequential full FT shows moderate uniform forgetting but high operational forgetting because it specifically destroys connector performance (the hardest task, trained first, most overwritten). SpaceCIL-full shows low operational forgetting because it protects the high-weight tasks via both adapter isolation and distillation.

### 10.5 What to Do If Results Don't Match Expectations

| Failure | Root cause class | Immediate action | Claim downgrade |
|---|---|---|---|
| Router collapses on old tasks | Optimization interference | Freeze trunk, inspect router gradient | Weaken routing claim; strengthen bank claim |
| Distillation is unstable | Loss weighting mismatch | Reduce lambda to 0.1; check calibration memory quality | Weaker anti-forgetting claim |
| Sequential FT doesn't forget much | Task similarity too high | Add a harder task or increase sequence length | Rephrase as partial forgetting study |
| Router doesn't beat random | Routing features too weak | Add state embedding to router; increase router capacity | Remove H2; keep H1 and H3 |
| Scorer is confounded | Evaluation confound | Fix scorer immediately before any further collection | Do not report results with bad scorer |

---

## 11. Ablation Studies

Run ablations after the main experiment is complete and G2 is satisfied. Each ablation varies one factor while holding all others fixed.

### 11.1 Distillation Weight (Lambda)

```bash
# lambda = 0.1
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --distillation-alpha 0.1 \
    --exp-name ablation_lambda01

# lambda = 0.5 (default, already run in main experiment)

# lambda = 1.0
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --distillation-alpha 1.0 \
    --exp-name ablation_lambda10
```

Report forgetting vs. lambda as a line plot. Expected: U-shaped or monotonically decreasing forgetting as lambda increases, until lambda becomes so large that the distillation term overwhelms the task loss and forward transfer suffers.

### 11.2 Calibration Memory Size

Change the memory capacity in the `BehaviorDistillation` configuration before running:

- 50 episodes/task
- 100 episodes/task (default)
- 200 episodes/task

Collect calibration sets of each size before running these ablations. Report forgetting vs. memory size. The expected result is diminishing returns above 100 episodes/task.

### 11.3 Distillation Space (Action vs. Latent)

The `BehaviorDistillation` module supports two loss variants:

- **Action-space distillation (default):** MSE between student and teacher action predictions on calibration episodes.
- **Latent-space distillation:** KL divergence between student and teacher intermediate flow-matching representations.

```bash
# Latent-space distillation ablation
# (Requires setting distillation_space="latent" in BehaviorDistillation config)
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --distillation-alpha 0.5 \
    --exp-name ablation_latent_distill
```

Report forgetting for both variants. If latent-space distillation is more stable and provides lower forgetting, mention it as a finding. If it causes training instability (loss NaN or divergence), fall back to action-space and report the failure in the ablation section.

### 11.4 LoRA Rank Sensitivity

Run the main experiment at LoRA ranks 8, 16, and 32. This tests whether higher-capacity adapters provide meaningfully better per-task performance at the cost of larger adapter bank size.

Report: per-task peak performance (`R[j][j]`) vs. LoRA rank, and total parameter count of the adapter bank vs. LoRA rank.

### 11.5 Training Steps Per Task

Run at 5,000 and 20,000 steps per task (in addition to the default 10,000):

```bash
# 5,000 steps
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 5000 \
    --distillation-alpha 0.5 \
    --exp-name ablation_5k_steps

# 20,000 steps
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 20000 \
    --distillation-alpha 0.5 \
    --exp-name ablation_20k_steps
```

This reveals whether 10,000 steps is at the convergence plateau or whether more training would substantially improve results.

---

## 12. Artifact Policy

Every experiment run must save the following artifacts. This is mandatory for paper writing and rebuttal.

### 12.1 Required Artifacts Per Run

| Artifact | Description | Location |
|---|---|---|
| Config hash | SHA256 of the serialized TrainConfig | `run_metadata.json` |
| Code commit | Git commit hash of the codebase | `run_metadata.json` |
| Adapter versions | Per-task adapter checkpoint paths and training step | `adapter_bank/metadata.json` |
| Calibration version | Hand-eye calibration version at training time | `run_metadata.json` |
| Scene revision | Scene layout at evaluation time | `eval_metadata.json` |
| Initial-state seeds | Object placement seeds for each eval trial | `eval_trials.jsonl` |
| Scorer outputs | Raw `ScorerResult` objects for every trial | `scorer_outputs.jsonl` |
| Fail types | Fail type labels for all non-success trials | `scorer_outputs.jsonl` |
| Split membership | Whether each trial is canonical or rotated | `eval_trials.jsonl` |

### 12.2 Checkpoint Layout

```
checkpoints/
  spacecil_main_run/
    run_metadata.json          # config hash, commit, calibration version
    adapter_bank/
      payload/
        state/                 # orbax checkpoint directory
        adapter_metadata.json  # training steps, freeze status
      latch/
        state/
        adapter_metadata.json
      clean/
        state/
        adapter_metadata.json
      connector/
        state/
        adapter_metadata.json
      bank_metadata.json       # registration order, frozen set
    eval/
      after_payload/
        eval_trials.jsonl
        scorer_outputs.jsonl
      after_latch/
        ...
      after_clean/
        ...
      after_connector/
        ...
    result_matrix.npy          # shape: [4, 4]
    metrics.json               # all computed metrics
```

### 12.3 Weights and Biases Logging

SpaceCIL configs have `wandb_enabled=True`. Each training run logs:
- Training loss per step
- Router loss per step
- Distillation loss per step
- Per-task evaluation success rates (logged at evaluation checkpoints)
- Routing entropy over training

Create a W&B project named `spacecil` and tag each run with its experiment type (`main`, `baseline_fulltune`, `ablation_lambda`, etc.).

### 12.4 Video Archiving

For every evaluation episode in the canonical result matrix, save:
- Wrist camera video (from the data pipeline)
- Optional: scene camera video (for manual audit)
- Scorer result alongside video for annotation

Store references to video paths in `eval_trials.jsonl`, not the raw video data.

---

## 13. Paper Writing Notes

### 13.1 Key Figures

**Figure 1: Result matrix heatmap.** A 4x4 grid where each cell shows `R[i][j]`. Color-code by success rate (green = high, red = low). Produce one heatmap per method (SpaceCIL, sequential FT, oracle, shared PEFT). A 2x2 arrangement of four heatmaps makes H1 visually immediate.

**Figure 2: Forgetting curves over the task sequence.** For each task j, plot `R[i][j]` as i goes from j to T-1 (how much that task degrades over subsequent training). Show SpaceCIL vs. sequential FT on the same axes. Include a shaded band for evaluation variance (run 3 seeds if resources allow).

**Figure 3: Routing entropy over training.** Plot routing entropy as the task sequence progresses. Should decrease as each new task is registered and the router specializes. Compare SpaceCIL (learned router) against random routing entropy as a reference line.

**Figure 4: Operational vs. uniform forgetting comparison.** A bar chart or table showing forgetting and operational_forgetting for each method. This is the H4 figure. It should show at least one method where the two metrics give meaningfully different pictures.

### 13.2 Table Structure

**Main results table:**

| Method | Avg. Success ↑ | Forgetting ↓ | Op. Forgetting ↓ | Params |
|---|---|---|---|---|
| Sequential full FT | | | | Full |
| Shared multi-task PEFT | | | | 1x LoRA |
| SpaceCIL (no distill) | | | | 4x LoRA |
| SpaceCIL (random route) | | | | 4x LoRA |
| SpaceCIL (ours) | | | | 4x LoRA |
| Per-task PEFT (oracle) | | | | 4x LoRA |

**Ablation table:**

| Variant | Avg. Success | Forgetting | Op. Forgetting |
|---|---|---|---|
| lambda=0.1 | | | |
| lambda=0.5 (default) | | | |
| lambda=1.0 | | | |
| memory=50 | | | |
| memory=100 (default) | | | |
| memory=200 | | | |
| action-space distill (default) | | | |
| latent-space distill | | | |

### 13.3 Claim Phrasing Reminders

Use these phrasings:

- "real-robot continual specialization"
- "operationally ordered task stream"
- "released pi0.5-class VLA backbone"
- "mission-aware forgetting"
- "parameter-efficient task expansion"

Avoid these phrasings:

- "solves catastrophic forgetting"
- "first continual VLA"
- "first lifelong mobile manipulation"
- "proves that continual learning works for VLA systems"

The paper's contribution is a study with real-robot results, an operationally motivated metric, and evidence for four specific hypotheses. It is not a claim that the problem is solved.

### 13.4 Section on Scorer Validity

Include a short appendix section that reports precision and recall for each task scorer against manual labels. This is the audit trail that makes the evaluation credible. Without it, reviewers will question whether high success rates reflect real task completion or scorer artifacts.

### 13.5 Related Work Positioning

Organize related work into four groups (from the Blueprint):

1. VLA and generalist robot policies (pi0, pi0.5, OpenVLA, RoboVLMs)
2. Continual / lifelong robot learning (LIBERO, SPECI, Wu et al., Xu and Nie)
3. Compositional adaptation and modular PEFT (Task Arithmetic, CompoSuite, PEFT composition)
4. Space / lunar operational motivation (NASA connector programs, dust mitigation roadmap, low-angle lighting study)

The paper is positioned as applying continual learning to VLA specialization on a space-motivated platform, not as a general continual learning methods paper.

---

## Appendix A: Quick Command Reference

```bash
# Environment check
source .venv/bin/activate
uv run python -c "import jax; print(jax.devices())"

# Debug smoke test
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py spacecil_debug --overwrite

# Norm stats (run once per task after dataset is finalized)
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_payload
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_latch
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_clean
uv run scripts/compute_norm_stats.py --config-name spacecil_rm75_connector

# Single-task training (Gate G1)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py spacecil_rm75_payload \
    --exp-name=g1_verify --overwrite

# Serve policy for evaluation
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=spacecil_rm75_payload \
    --policy.dir=checkpoints/spacecil_rm75_payload/g1_verify/10000

# Continual training (main experiment)
uv run scripts/train_spacecil.py \
    --config spacecil_rm75_payload \
    --task-sequence payload latch clean connector \
    --num-steps-per-task 10000 \
    --distillation-alpha 0.5 \
    --exp-name spacecil_main_run

# Run tests before any commit
uv run pytest src/openpi/research/ -x -q
```

---

## Appendix B: Gate Checklist

### Gate G1 (Required before continual experiments)

- [ ] Hand-eye calibration verified (reprojection error < 3px RMS)
- [ ] Wrist camera exposure locked
- [ ] `spacecil_debug` config instantiates and trains for 10 steps without error
- [ ] Norm stats computed for `spacecil_rm75_payload`
- [ ] Single-task training completes for payload task (10,000 steps)
- [ ] Policy serves inference on physical robot
- [ ] Scorer validated on 20-30 manual-labeled pilot episodes (precision > 0.80, recall > 0.80)
- [ ] At least 10 physical robot trials completed; scorer and operator agree on all 10

### Gate G2 (Required before claiming Paper A mainline)

- [ ] At least 2 tasks train sequentially without training divergence
- [ ] Router accuracy exceeds random (above 0.50 for 2 tasks, above 0.40 for 4 tasks)
- [ ] Behavior distillation runs without NaN loss for all lambda values
- [ ] Adapter bank saves and restores correctly (checkpoint round-trip verified)
- [ ] Result matrix `R[i][j]` is fully populated for at least 2 tasks
- [ ] Metrics (`average_success`, `forgetting`, `operational_forgetting`) compute correctly
- [ ] All baselines have been run at least once

---

## Appendix C: Failure Mode Reference

| Failure | Affected claim | Immediate action | Downgrade path |
|---|---|---|---|
| Router old-task collapse after connector training | H2 | Freeze backbone trunk; inspect router gradient scale | Weaken routing claim to "adequate" instead of "sufficient" |
| Distillation loss diverges (NaN) | H3 | Reduce lambda to 0.1; check calibration memory episode quality | Weaker anti-forgetting claim; report instability as finding |
| Sequential FT shows minimal forgetting | H1 | Check task similarity; add a harder or more different task | Reframe as studying partial forgetting; still valid |
| Scorer precision < 0.80 | All | Stop evaluation. Fix scorer before any further trials | Do not report results with unvalidated scorer |
| Calibration drift mid-experiment | All | Recertify hand-eye; discard sessions after drift detected | Flag affected sessions in supplementary |
| Payload and latch tasks are too similar | H1, H4 | Replace one task with a more distinct manipulation primitive | Adjust task suite; recollect demos |
| Router cannot distinguish clean from connector | H2 | Add visual contrast (different scene background per task); retrain router | Limit routing claim to tasks with sufficient visual contrast |
