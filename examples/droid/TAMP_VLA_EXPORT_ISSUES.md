# Issues in the tamp-vla LeRobot export (`_lerobot_raw.json` / `le-robot/`)

Context: while combining the `toys-no-collision/success` trajectories with DROID-100 for openpi
fine-tuning, I found that the per-trajectory LeRobot export (the `_lerobot_raw.json` file and the
`le-robot/` v3.0 sub-dataset) is **not usable as-is** for a velocity-action policy. The numbers below
are measured across all 20 episodes in `toys-no-collision/success`.

The good news: the **raw source data is correct**. Every problem here is in the *export/conversion*
step, not in the planner. The ground truth lives in each trajectory's `tiptop_plan.json` (dense 50 Hz
`positions` + `velocities` + gripper `open`/`close` events). The fixes below all amount to "export
from the dense plan instead of from sparse keyframes."

For reference, my corrected re-export lives in
`examples/droid/convert_combined_droid_toys_to_lerobot.py` (`_iter_toys_episodes` / `_dense_plan`).

---

## Summary

| # | Issue | Severity | Root cause |
|---|-------|----------|------------|
| 1 | Trajectory subsampled to ~20 keyframes (from ~3,500 dense points) | **Critical** | Export samples too few points |
| 2 | `action` velocities physically impossible (max **72 rad/s**, 42% over joint limits) | **Critical** | `action = diff(keyframe positions) × 15`, computed on the sparse keyframes |
| 3 | `gripper_position` / action gripper is a constant `1.0` | **Critical** | Gripper open/close events never read from the plan |
| 4 | Timestamps are synthetic (`frame_index / 15`), not real execution time | High | Fixed 15 fps assumed; real episodes are 63–80 s |
| 5 | Only 2 cameras; no `exterior_2` | Low (note) | Hardware reality, but breaks DROID 3-cam schema |
| 6 | `le-robot/` is codebase_version **v3.0** | Low (note) | Incompatible with openpi's pinned LeRobot v2.1 |

Reference scale (DROID-100, the distribution we want to match):
- `joint_velocity`: mean |·| ≈ **0.13 rad/s**, max ≈ **1.0 rad/s**, **0%** over joint limits.
- `gripper_position`: continuous **0 = open → 1 = closed**, episodes start open (~0).

---

## Issue 1 — Severe temporal subsampling (~20 keyframes for a ~70 s motion)

**What I see.** `_lerobot_raw.json` and `le-robot/` store **~20 frames per episode**. The dense plan in
`tiptop_plan.json` has **~3,500 points per episode** (sum of all `trajectory` steps' rows), spanning
**63–80 s** of motion (mean 71.5 s) at `dt = 0.02` (50 Hz).

```
dense plan total points across 20 eps : 71,513   (~3,576 / episode)
le-robot frames across 20 eps         :    395   (~20 / episode)   -> ~99% of the trajectory discarded
```

**Impact.** 20 frames cannot represent a 70 s manipulation. Downstream this is unusable for a
high-rate control policy regardless of the action fix below.

**Fix.** Export every dense-plan point, resampled to the target control rate (we use DROID's 15 Hz →
~1,000 frames/episode). See Issue 2 for the resampling.

---

## Issue 2 — `action` joint velocities are physically impossible

**What I see.** The exported `action[:, :7]` equals `diff(keyframe_positions) × 15` *exactly*
(ratio = 1.000, per-joint correlation = 1.000 with `diff × fps`). Because it differences the **sparse
keyframes** (which are seconds apart in reality) and multiplies by a nominal 15 fps, the result is
1–2 orders of magnitude too large:

```
                              joint:  [ j1    j2    j3    j4    j5    j6    j7  ]
exported action |mean abs| (rad/s)  : [1.999 2.588 1.952 3.285 2.110 3.429 7.582]
exported action |max  abs| (rad/s)  : [38.2  17.24 31.73 22.21 36.5  27.25 71.92]   <-- impossible
DROID joint_velocity |mean abs|     : [0.096 0.205 0.083 0.180 0.131 0.163 0.158]
FR3 velocity limit          (rad/s) : [2.62  2.62  2.62  2.62  5.26  4.18  5.26 ]

steps exceeding FR3 limit on >=1 joint : 42%   (first step of each episode: 95%)
```

(FR3 max joint velocities used above are the datasheet limits for the Franka Research 3.)

**Impact.** Training a velocity-action policy on this teaches it to command velocities the robot
physically cannot execute, and it wildly inflates action normalization stats (the combined-dataset
`actions` std blew up ~5× even though the toys data is a small fraction of frames).

**Root cause.** Two compounding errors: differencing sparse keyframes (Issue 1) *and* assuming the
keyframes are 1/15 s apart when they are really seconds apart.

**Fix.** Don't recompute velocity from sparse positions at all — the plan already stores the **correct
instantaneous velocities** at 50 Hz. I verified they are self-consistent with the positions:

```
median( diff(plan_positions) / (plan_velocities × 0.02) ) = 0.999   -> velocities are true 50 Hz values
plan velocities |mean abs| : [0.041 0.054 0.040 0.063 0.046 0.070 0.148]  (matches DROID ~0.13)
plan velocities |max  abs| : ~1.0,  0% over FR3 limits
```

Export = resample the dense plan to 15 Hz and use the plan's own `velocities` as the action (or, if you
must recompute, difference the *dense* 15 Hz positions, not the keyframes).

---

## Issue 3 — Gripper channel is a constant `1.0`

**What I see.** In every episode, `_lerobot_raw.json` `gripper_position` and `action[:, 7]` are **all
`1.0`** — the gripper state is never recorded. Meanwhile `tiptop_plan.json` contains explicit gripper
events:

```json
{"type": "gripper", "label": "Pick(brown_toy, ...)",  "action": "close"}
{"type": "gripper", "label": "Place(brown_toy, ...)", "action": "open"}
... (6 events / episode: close, open, close, open, close, open)
```

**Impact.** The policy gets no gripper supervision at all — it can never learn to grasp/release. (`1.0`
also happens to mean "always closed" under the DROID convention, which is doubly wrong since the arm
starts open.) Note this corrupts **two** channels, not one: the gripper position is the **8th element of
the observation `state`** that the policy conditions on (openpi's `DroidInputs` builds
`state = concat(joint_position[7], gripper_position[1])`), *and* the 8th element of the `action` it
predicts (`action = concat(joint_velocity[7], gripper_position[1])`). A constant `1.0` therefore breaks
both what the model sees and what it's trained to output.

**Fix.** Since the gripper isn't stored per-frame upstream, **I reconstructed the gripper channel from
the plan's `open`/`close` events** (in `_dense_plan` in
`convert_combined_droid_toys_to_lerobot.py`): start **open (0.0)**, set **closed (1.0)** at each
`close`, back to **0.0** at each `open`, holding the value across the intervening trajectory steps. This
reconstructed signal then feeds **both** `state[7]` (observation) and `action[7]` (command). Use the
DROID convention **0 = open, 1 = closed** (verified against DROID-100: range [0, 0.99], episodes start
~0). After the fix each toys episode shows the expected 6 gripper toggles.

(This reconstruction is a workaround in the openpi-side converter. The real fix is upstream: record the
actual commanded/measured gripper state per frame during execution, rather than emitting a constant.)

---

## Issue 4 — Synthetic timestamps / fake frame rate

**What I see.** `le-robot` frame timestamps are exactly `frame_index / 15` → `0.0, 0.067, 0.133, …,
1.267 s`, implying a 1.3 s episode. The real motion is **63–80 s**. The 15 fps is assumed, not measured.

**Impact.** This is the assumption that makes Issue 2's `× 15` produce garbage, and it makes the
dataset's notion of time meaningless. The raw camera videos (`external_cam.mp4`, `hand_cam.mp4`) are the
*full* recordings (~2,400 frames, 720×1280) and were only used to pull ~20 thumbnails.

**Fix.** Derive time from the plan: `t = cumulative_points × 0.02`. When resampling video onto the
control timeline, note the raw cameras run at a consistent **~34 fps** across episodes
(`video_frames / (plan_points × 0.02)` has <2% spread), so a time-proportional resample is reasonable —
though see the caveat below.

**Caveat for a proper fix.** The plan timeline does **not** model gripper-actuation pauses (the
`gripper` steps are instantaneous, `dt = None`), but the real robot pauses ~0.5–1 s while the gripper
moves, and the video *does* capture those pauses. A time-proportional video resample therefore drifts by
a few % mid-episode. For frame-accurate image alignment, export real per-frame timestamps from the ZED
`.svo2` recordings (they carry hardware timestamps) and align images to plan time by timestamp rather
than by proportion.

---

## Issue 5 — Only two cameras (note, not a bug)

The rig records `exterior_1_left` + `wrist_left`; DROID's schema expects three
(`exterior_1`, `exterior_2`, `wrist`). Not a data-collection bug, but flagging it: for DROID-schema
compatibility something has to fill `exterior_2`. In my converter I duplicate `exterior_1` into
`exterior_2` (pi05's `DroidInputs` masks that slot anyway), but if you add a second exterior camera
upstream it would be used directly.

---

## Issue 6 — LeRobot v3.0 vs openpi's v2.1 (note, not a bug)

`le-robot/meta/info.json` reports `codebase_version: v3.0` (uses `meta/tasks.parquet`,
`meta/episodes/`, `data/chunk-000/file-000.parquet`). openpi pins LeRobot **v2.1**
(`meta/tasks.jsonl`, `meta/episodes.jsonl`, `data/chunk-000/episode_000000.parquet`), so the exported
`le-robot/` cannot be loaded by openpi directly. This is why my converter reads `tiptop_plan.json` +
the raw mp4s rather than the `le-robot/` sub-dataset. If the export targets openpi, pin the matching
LeRobot version (or write the v2.1 layout).

---

## Recommended correct export (per episode)

1. Flatten `tiptop_plan.json` `steps` in order into dense 50 Hz arrays:
   - `positions[M,7]`, `velocities[M,7]` from every `trajectory` step (`dt = 0.02`).
   - `gripper[M]` from `gripper` events: start `0.0`, `close → 1.0`, `open → 0.0`, held across rows.
2. Resample 50 Hz → target rate (15 Hz for DROID): `N = round(M × 0.02 × 15)`, nearest-index sampling.
3. Per frame: `joint_position = positions`, `gripper_position = gripper`,
   `action = concat(velocities[7], gripper[1])` — **velocities from the plan, never `diff × fps`**.
4. Images: resize raw `external_cam.mp4` / `hand_cam.mp4` to 180×320 and align to the control timeline
   by **`.svo2` timestamp** if available, else time-proportional resample (accepting the gripper-pause
   drift in Issue 4).
5. Use real timestamps (`frame_index / target_fps` is only valid once the data is genuinely at
   `target_fps`).

Sanity checks to assert in the exporter (all pass on my re-export, all fail on the current one):
- `max |action[:, :7]|` ≤ FR3 joint-velocity limits (no impossible velocities).
- `gripper_position` takes both 0 and 1, with the expected number of toggles.
- `n_frames ≈ plan_duration × target_fps` (not a fixed ~20).
