# Experiment Guide: LunarCompose (Paper B)
## Factorized Task x Environment Adaptation for Lunar-Analog Compositional Generalization

---

## 1. Overview

### 1.1 Scientific Question

LunarCompose asks whether adaptation should be **factorized along two independent axes — task and environment** — when transferring an Earth-trained VLA into lunar-analog deployment conditions. If task semantics and environment-specific visual corrections are tangled into a single monolithic module, reuse across condition changes becomes brittle. Factorized adaptation offers a principled alternative: learn task capability once, then compose it with environment correction independently.

The paper tests this on a real mobile manipulation platform (wheeled base + RM75 7-DoF arm + two-finger gripper). The primary policy camera is a wrist RGB image. Action space: Absolute Joint Position (7 DoF) + Gripper (1 DoF) = 8D total. Backbone: released pi0.5 via the openpi flow-matching policy path.

### 1.2 Hypotheses

| Label | Hypothesis |
|-------|-----------|
| H1 | Factorized task + environment adaptation improves unseen-combination performance over non-factorized baselines |
| H2 | Explicit task/environment routing yields more interpretable and controllable specialization |
| H3 | A small lunar-analog visual condition taxonomy is sufficient to expose environment-sensitive failure modes |
| H4 | The missing-corner evaluation protocol can determine whether the factorization assumption actually holds |

### 1.3 Claim Boundary

The safe claim form for this paper:

> "We study factorized adaptation under lunar-analog environment shifts and evaluate compositional generalization to unseen task-environment combinations on a real mobile manipulation platform."

Do **not** claim strict orthogonal disentanglement between task and environment factors. Do not claim it "solves lunar domain adaptation" or "proves task-environment independence." The factorization assumption is explicitly testable and explicitly allowed to fail — an honest falsification is a valid scientific finding.

### 1.4 Dependency on Paper A

Paper B is not independent. It reuses the SpaceCIL infrastructure directly. The following must exist and be stable before any LunarCompose experiments begin:

- One task trains and replays correctly on the wrist-camera policy path (Gate G1).
- At least two tasks train sequentially with a working `TaskAdapterBank` (Gate G2).
- The router, distillation path, and continual harness are stable enough for handoff.

Build order: **Phase A (shared infra) → Phase B (SpaceCIL core) → Phase C (LunarCompose extension).**

### 1.5 What This Guide Covers

This guide walks through every step for Paper B experiments: environment taxonomy setup, data collection, missing-corner splits, factorized training, baseline experiments, evaluation, diagnostics, and artifact management. It does not cover Paper A (SpaceCIL) experiments — those are in the SpaceCIL experiment guide.

---

## 2. Prerequisites

### 2.1 Infrastructure Requirements

Before starting Paper B:

- Paper A (SpaceCIL) Phase B is complete. Gate G2 has passed.
- `TaskAdapterBank` saves and restores reliably.
- `TaskRouter` beats random routing on at least two tasks.
- The behavior distillation path executes stably.
- The wrist-camera policy path produces sensible actions on at least the payload transfer and latch tasks.
- All Paper A tests pass: `uv run pytest src/openpi/research/ -x -q`

### 2.2 Hardware

Same hardware as SpaceCIL plus environment control equipment:

| Equipment | Purpose |
|-----------|---------|
| Wheeled base + RM75 arm + gripper | Robot platform (same as SpaceCIL) |
| Wrist RGB camera | Primary policy camera (same as SpaceCIL) |
| Scorer camera (overhead) | Decoupled scorer view for E1 and E2 |
| Collimated LED panel (800-1200 lux) | E1 hard-shadow illumination |
| Fine gray powder or matte spray tiles | E2 surface contamination |
| 2-4 cm rock or rubble props | E2 partial occlusion |
| Lux meter | Environment condition verification |
| Color temperature meter | Lighting calibration |

### 2.3 Estimated Timeline

| Phase | Duration |
|-------|---------|
| Environment setup and validation | 1-2 weeks |
| Data collection (12 cells, 50-100 eps each) | 3-5 weeks |
| Normalization stats and config verification | 1-2 days |
| Factorized training (all cells, 3 rotations) | 2-4 weeks |
| Baseline training | 2-3 weeks |
| Evaluation and diagnostics | 1-2 weeks |
| Analysis and write-up prep | 1 week |

Total estimated: 10-18 weeks from Gate G2 to paper submission.

---

## 3. Environment Taxonomy Setup

Paper B introduces three lunar-analog visual conditions. These are not claims of full lunar environmental fidelity — they are "lunar-analog visual factors" that expose environment-sensitive failure modes in a controlled, reproducible way.

Every environment condition must be:
1. Defined with enough physical precision to be reproduced across sessions.
2. Validated against the allowed variation window before data collection begins.
3. Logged via `meta.env_id` and `meta.camera_preset_id` in every episode.

### 3.1 E0 — Nominal Laboratory Condition

**Rationale.** E0 is the stable reference condition. All tasks must be demonstrated and validated here first, before any environment variation is introduced. It also serves as the baseline for scorer validation.

**Physical setup:**
- Workspace surface: matte gray or neutral beige, no reflective patches.
- No occlusions or debris on the workspace.
- Standard indoor diffuse lighting: 500-700 lux at object surface.
- Color temperature: 4000-5000K (neutral white).
- Light direction: overhead diffuse panel or multiple ambient sources. No hard shadows.
- Shadow depth: no shadow darker than 30% of ambient (shadow/ambient luminance ratio > 0.7).
- Wrist camera: locked at nominal preset (`camera_preset_id = "nominal"`).
- Scorer camera: same nominal preset as policy camera.

**Allowed variation window.** Small object position jitter (±3 cm), robot initial joint configuration within nominal range.

**Scorer decoupling.** Not required in E0. Perturbation is minimal, so scorer and policy camera see effectively the same scene. This is where you run initial scorer validation against manual labels.

**Setup verification checklist:**
- [ ] Lux meter reads 500-700 lux at object surface.
- [ ] Color temperature meter reads 4000-5000K.
- [ ] No shadow darker than 30% ambient in any part of the workspace.
- [ ] Camera preset ID logged as `"nominal"` in episode metadata.
- [ ] Scorer agreement with manual labels: log precision/recall on a pilot subset before proceeding.

---

### 3.2 E1 — Hard-Shadow / Low-Angle Illumination

**Rationale.** E1 simulates the lunar south-pole illumination environment, where the sun angle sits 1.5-3 degrees above the horizon, producing long and sharp shadows that frequently obscure task-relevant object features. This condition directly motivates the paper's space-relevance narrative (see NASA lunar south-pole lighting study, ntrs.nasa.gov/citations/20240011393).

**Physical setup:**
- Remove or dim all existing overhead ambient lighting (reduce to < 50 lux ambient).
- Primary light source: single collimated LED panel or directional spot simulating low sun angle.
- Incident angle: 10-20 degrees above horizontal. The canonical value is **15 degrees**.
- Panel distance: 1.5-2.5 m from workspace center (to approximate parallel ray geometry).
- Primary source intensity: 800-1200 lux at the directly illuminated side of objects.
- Shadow luminance: < 100 lux (shadow/ambient ratio < 0.1 in shadowed regions).
- Color temperature: 5500-6500K (cool white, simulating sunlight).

**Camera preset:** `e1_shadow`. Fixed aperture and fixed exposure time. Do not auto-expose during a trial — the shadow contrast is the target visual condition, not something to compensate away.

**Allowed variation window.** Shadow direction allowed ±5 degrees rotation around vertical axis across sessions. Panel distance fixed within ±5 cm per session.

**Scorer decoupling — critical.** The scorer camera must be placed to avoid the shadow pattern cast by the robot arm and gripper. Use a controlled overhead view with separate diffuse lighting (500-700 lux, same as E0). The scoring signal must not degrade under E1 conditions even when the policy-view image is dominated by shadow.

Verify scorer performance on E1 episodes before any E1 training or evaluation begins. If scorer performance drops below the E0 baseline by more than the accepted tolerance, stop and redesign the scorer view.

**Failure modes to watch:**
- Gripper occluded by its own shadow.
- Object features (latch handle, connector port) disappearing into shadow.
- Sharp shadow edges creating optical-flow-like artifacts that distract the policy.
- Scorer camera picking up shadow variation from the robot arm.

**Setup verification checklist:**
- [ ] Lux meter reads < 50 lux in shadowed regions.
- [ ] Lux meter reads 800-1200 lux on directly illuminated object face.
- [ ] Color temperature reads 5500-6500K.
- [ ] Camera preset ID logged as `"e1_shadow"` in all episodes.
- [ ] Scorer camera verified to be outside shadow cast of robot arm.
- [ ] Scorer validated on 10+ E1 pilot episodes against manual labels.

---

### 3.3 E2 — Partial Contamination / Occlusion / Appearance Perturbation

**Rationale.** E2 simulates lunar dust accumulation, surface debris, and partial occlusion of task-relevant objects. The appearance of objects and workspace surfaces is perturbed without changing the underlying task geometry. Illumination is the same as E0 — E2 isolates texture and appearance change from illumination change, so task-specific failures in E2 cannot be attributed to shadow effects.

**Physical setup:**
- Workspace surface: apply a controlled simulated dust layer (fine gray powder or matte spray coating on removable surface tiles that can be reset between sessions).
- Object contamination: light powder coating on object surfaces. Changes texture and reflectance without affecting geometry.
- Partial occlusion: fixed small props (2-4 cm rocks or rubble-like objects) at defined positions within the camera field of view but outside the direct manipulation zone.
- Lighting: same as E0 (nominal diffuse, 500-700 lux). Isolating texture change from illumination change is the key design intent.

**Dust density levels.** Two canonical levels: light contamination (barely visible powder layer) and heavy contamination (clearly visible gray coating). Log which level is in use per session.

**Camera preset:** `e2_contamination`. Same as the E0 nominal preset, to isolate texture/appearance change from any exposure change.

**Allowed variation window.** Occlusion props at fixed positions per session (write them down). Object powder coating reapplied every N sessions per maintenance protocol. Dust density must match the logged level for that session.

**Scorer decoupling — critical.** If the scorer uses pixel-difference signals (relevant for the cleaning task), the scoring region must exclude contaminated surface tiles. Validate that the scorer cannot be confused by the gray powder — either by placing the scoring region over clean tiles, or by using a marker/segmentation signal robust to surface texture. Scorer camera must be verified not to rely on surface color or texture in any region affected by the contamination.

Run scorer validation on E2 episodes before E2 training begins. This validation is a blocking requirement — do not proceed until fixed.

**Failure modes to watch:**
- Policy distracted by occlusion props (arm swings to avoid them unnecessarily).
- Texture-matching failures when object appearance changes (connector not recognized under powder).
- Cleaning task scorer confounded by dust on the scoring surface tile.
- Object powder coating too thick and changing geometry.

**Setup verification checklist:**
- [ ] Dust density level (light/heavy) logged per session.
- [ ] Occlusion prop positions recorded (reproducible across sessions).
- [ ] Lighting reads 500-700 lux (same as E0).
- [ ] Camera preset ID logged as `"e2_contamination"` in all episodes.
- [ ] Scorer camera verified to exclude contaminated surface regions.
- [ ] Scorer validated on 10+ E2 pilot episodes against manual labels.
- [ ] Cleaning task scorer explicitly validated not to use contaminated surface pixels.

---

### 3.4 Cross-Condition Rules

These rules apply across all three conditions without exception:

1. `meta.env_id` must be logged in every episode — either `"nominal"`, `"shadow"`, or `"contamination"`.
2. `meta.camera_preset_id` must match the declared environment for that session.
3. No condition mixing within a session (except the domain-randomization baseline, which explicitly requires it).
4. Scorer camera is always decoupled from the policy camera for E1 and E2.
5. Before starting any environment, run the setup verification checklist and log the result in session metadata.
6. Before switching conditions between sessions, reset the workspace completely. Powder residue from E2 must not carry over into E0 or E1 sessions.

---

## 4. Data Collection

### 4.1 Collection Grid

The experiment requires data across the full 4 tasks × 3 environments = **12 cells**.

| | nominal (E0) | shadow (E1) | contamination (E2) |
|-|---|---|---|
| **payload** | lunarcompose_payload_nominal | lunarcompose_payload_shadow | lunarcompose_payload_contamination |
| **latch** | lunarcompose_latch_nominal | lunarcompose_latch_shadow | lunarcompose_latch_contamination |
| **clean** | lunarcompose_clean_nominal | lunarcompose_clean_shadow | lunarcompose_clean_contamination |
| **connector** | lunarcompose_connector_nominal | lunarcompose_connector_shadow | lunarcompose_connector_contamination |

Target: **50-100 episodes per cell**, for 600-1200 total demonstrations. Collect conservatively and check scorer agreement before moving to the next cell. It's better to have 70 clean accepted demos than 120 with 40 ambiguous.

### 4.2 Episode Schema

Each episode uses the shared schema from `src/openpi/research/shared/episode_schema.py`. For Paper B, the critical addition over SpaceCIL is the `meta.env_id` field — this must be populated in every collected episode.

Required metadata fields for Paper B episodes:

| Field | Value |
|-------|-------|
| `meta.task_id` | `"payload"`, `"latch"`, `"clean"`, or `"connector"` |
| `meta.env_id` | `"nominal"`, `"shadow"`, or `"contamination"` |
| `meta.camera_preset_id` | Must match env: `"nominal"`, `"e1_shadow"`, or `"e2_contamination"` |
| `meta.session_id` | Unique session identifier |
| `meta.operator_id` | Operator identifier |
| `meta.calibration_version` | Current calibration version string |
| `meta.scene_revision` | Scene setup revision |
| `meta.object_revision` | Object revision (important for E2 — note powder coating state) |
| `label.success` | Boolean success label |
| `label.fail_type` | Failure type if not successful |

### 4.3 Session Protocol

Each collection session must be **single-environment**. Do not mix E0, E1, and E2 within the same session (the domain-randomization baseline is handled separately by loading data from multiple session-level datasets).

Per-session procedure:

1. Verify environment setup using the checklist for that condition.
2. Record setup metrics (lux, color temp, prop positions, dust level).
3. Lock camera preset ID.
4. Run 5-10 warm-up demos. Verify scorer is working before counting any demos toward the dataset.
5. Collect demos. Each episode: verify initial state, run demonstration, score immediately, classify as accepted/rejected/safety-aborted/hardware-invalid.
6. Log all metadata fields.
7. At the end of session, run scorer on all accepted demos and compare to manual labels on a random 10% sample.

Rejection criteria (do not count these toward the target episode count):
- Safety stop during demonstration.
- Major frame drop in wrist camera.
- Gross initial-state mismatch.
- Human hand visible in policy camera view during core execution window.
- Ambiguous success label.
- Unintended shortcut that bypasses the intended manipulation primitive.

### 4.4 Data Organization

Each cell maps to a LeRobot dataset with a placeholder repo ID:

```
placeholder/lunarcompose_payload_nominal
placeholder/lunarcompose_payload_shadow
placeholder/lunarcompose_payload_contamination
placeholder/lunarcompose_latch_nominal
placeholder/lunarcompose_latch_shadow
placeholder/lunarcompose_latch_contamination
placeholder/lunarcompose_clean_nominal
placeholder/lunarcompose_clean_shadow
placeholder/lunarcompose_clean_contamination
placeholder/lunarcompose_connector_nominal
placeholder/lunarcompose_connector_shadow
placeholder/lunarcompose_connector_contamination
```

Each dataset corresponds to one config name (see Section 6.1). Do not merge datasets across cells at collection time — keep them separate so the missing-corner harness can load exactly the cells it needs.

### 4.5 Calibration and Drift Rules

Before every data collection session:

- Run hand-eye calibration check. If calibration drift exceeds threshold, recertify and log `meta.calibration_version` accordingly.
- Lock wrist camera exposure. Do not change exposure between the first and last demo of a session.
- For E1: verify shadow angle with lux meter before starting.
- For E2: verify dust density matches the logged level. Reapply powder if needed.

If calibration drift invalidates a session, label all demos from that session as `hardware-invalid` and do not include them in any training dataset.

---

## 5. Missing-Corner Protocol

The missing-corner evaluation protocol is the scientific centerpiece of Paper B. Training happens on a subset of the 12 cells. Evaluation happens on all 12 — both seen (train) and unseen (test) cells. The gap between seen and unseen performance determines whether factorized adaptation actually generalizes.

### 5.1 The Grid (Full 4 x 3)

```
               nominal    shadow    contamination
  payload       [P,N]     [P,S]       [P,C]
  latch         [L,N]     [L,S]       [L,C]
  clean         [C,N]     [C,S]       [C,C]
  connector     [K,N]     [K,S]       [K,C]
```

12 cells total. Every experiment (factorized + all baselines) uses the same grid.

### 5.2 Canonical Splits (3 Rotations)

The splits are hardcoded in `src/openpi/research/lunarcompose/missing_corner_harness.py` and must not be modified after data collection begins. Each rotation holds out 4 cells (unseen) and trains on 8 (seen).

**Rotation 0 (canonical):**
```
Train cells (8):
  (payload, nominal),     (payload, shadow)
  (latch, nominal),       (latch, contamination)
  (clean, shadow),        (clean, contamination)
  (connector, nominal),   (connector, contamination)

Test cells (4):
  (payload, contamination)
  (latch, shadow)
  (clean, nominal)
  (connector, shadow)
```

**Rotation 1:**
```
Test cells (4):
  (payload, nominal)
  (latch, contamination)
  (clean, shadow)
  (connector, contamination)
Train = remaining 8 cells
```

**Rotation 2:**
```
Test cells (4):
  (payload, shadow)
  (latch, nominal)
  (clean, contamination)
  (connector, nominal)
Train = remaining 8 cells
```

All three rotations satisfy the same four split constraints:
1. Every task appears in at least one training cell.
2. Every environment appears in at least one training cell.
3. No overlap between train and test cells.
4. `train ∪ test = all 12 cells` (complete partition).

### 5.3 Split Validation

Always validate splits programmatically before any training run begins. This catches data hygiene errors early:

```python
from openpi.research.lunarcompose.missing_corner_harness import MissingCornerHarness

harness = MissingCornerHarness(
    task_ids=["payload", "latch", "clean", "connector"],
    env_ids=["nominal", "shadow", "contamination"],
    scorers={},        # populated when evaluating
    eval_episodes={},  # populated when evaluating
)

# Validate all three rotations before starting data collection
for rotation in [0, 1, 2]:
    train_cells, test_cells = harness.generate_split(rotation=rotation)
    harness.validate_split()  # raises ValueError on any constraint violation
    print(f"Rotation {rotation}: {len(train_cells)} train, {len(test_cells)} test cells — OK")
```

`validate_split()` raises `ValueError` loudly if any constraint is violated. Do not silence these errors. If a constraint fails, rebuild the split and revalidate before proceeding.

### 5.4 No Scene-Instance Leakage

The hardest leakage to catch: if the same physical object setup (same object pose seed, same workspace layout) appears in both a train cell and a test cell, the unseen-combination result is invalid.

Rules:
- Scene-instance IDs (committed JSON files with initial-state seeds) must be unique per cell.
- The same initial-state seed file must not appear in both a train-cell evaluation set and a test-cell evaluation set.
- `meta.scene_revision` and `meta.object_revision` must be logged for every episode to make this auditable.

Before running evaluation, cross-check that no seed file appears in both `train_cells` and `test_cells` evaluation datasets. If leakage is found, rebuild the affected evaluation sets and invalidate any previous runs that used the leaked seeds.

---

## 6. Training

### 6.1 Normalization Stats (per cell)

Normalization stats must be computed separately for each of the 12 task-env cells. Run these before the first training pass for any cell:

```bash
# E0 nominal
uv run scripts/compute_norm_stats.py --config-name lunarcompose_payload_nominal
uv run scripts/compute_norm_stats.py --config-name lunarcompose_latch_nominal
uv run scripts/compute_norm_stats.py --config-name lunarcompose_clean_nominal
uv run scripts/compute_norm_stats.py --config-name lunarcompose_connector_nominal

# E1 shadow
uv run scripts/compute_norm_stats.py --config-name lunarcompose_payload_shadow
uv run scripts/compute_norm_stats.py --config-name lunarcompose_latch_shadow
uv run scripts/compute_norm_stats.py --config-name lunarcompose_clean_shadow
uv run scripts/compute_norm_stats.py --config-name lunarcompose_connector_shadow

# E2 contamination
uv run scripts/compute_norm_stats.py --config-name lunarcompose_payload_contamination
uv run scripts/compute_norm_stats.py --config-name lunarcompose_latch_contamination
uv run scripts/compute_norm_stats.py --config-name lunarcompose_clean_contamination
uv run scripts/compute_norm_stats.py --config-name lunarcompose_connector_contamination
```

After computing stats, inspect `norm_stats.json` for each cell. Check `q01`, `q99`, and `std` values. Dimensions with very small `std` (rarely-used joints in certain conditions) can cause normalization instability. If any dimension has `std < 0.01`, manually examine those demonstrations and decide whether to adjust.

### 6.2 Verifying the Debug Config

Before any GPU training, verify the debug config instantiates cleanly:

```bash
uv run scripts/train_lunarcompose.py --config lunarcompose_debug --rotation 0 --num-steps-per-cell 10
```

This uses `paligemma_variant="dummy"` and `FakeDataConfig`, so it runs without a GPU and without real data. It exercises the harness setup, split generation, and import chain. If this fails, debug the import and config paths before touching real data.

### 6.3 Factorized Training

The main factorized training loop trains both task and environment adapters for each cell in the training split.

**Training entry point:**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_factorized \
    --rotation 0 \
    --num-steps-per-cell 10000 \
    --checkpoint-dir checkpoints \
    --exp-name lunarcompose_factorized_rot0
```

Repeat for rotations 1 and 2:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_factorized \
    --rotation 1 \
    --num-steps-per-cell 10000 \
    --exp-name lunarcompose_factorized_rot1

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_factorized \
    --rotation 2 \
    --num-steps-per-cell 10000 \
    --exp-name lunarcompose_factorized_rot2
```

**What happens inside the training loop (per cell):**

For each `(task_id, env_id)` in the training split:
1. Load the cell-specific data config: `lunarcompose_{task_id}_{env_id}`.
2. Apply the task adapter for this `task_id` from `TaskAdapterBank` via `task_bank.merge_into_model(model, task_id)`.
3. Apply the env adapter for this `env_id` from `EnvAdapterBank` via `env_bank.merge_into_model(model, env_id)`.
4. Run `num_steps_per_cell` gradient steps using openpi's JIT-compiled `train_step`.
5. Register the updated adapters back into their respective banks.

**Adapter composition:** Both adapters are applied sequentially. Task adapter is applied first, env adapter second:

```python
# Effective weights = base + task_lora_delta + env_lora_delta
task_bank.merge_into_model(model, task_id)
env_bank.merge_into_model(model, env_id)
```

This works cleanly when task and env adapters target disjoint parameter groups (the preferred path). Task adapters target `action_expert` and language model LoRA layers. Env adapters target SigLIP vision encoder layers.

**Adapter swapping happens outside JIT boundaries** to avoid recompilation. The JIT-compiled `train_step` sees a model with parameters already loaded; it does not see the bank-swapping logic.

### 6.4 Layer Targeting Strategy

**Preferred path (visual-path LoRA):**

- Task adapters target action expert and language model layers: `PathRegex(".*action_expert.*|.*lm.*")`
- Env adapters target the vision encoder: `PathRegex(".*siglip.*")`
- Since they target different sub-graphs, the composition is additive and non-conflicting. Order of application does not matter.

**Fallback path (env-prefix conditioning):**

If visual-path LoRA injection into SigLIP is technically blocked (e.g., the visual encoder is frozen in a way that prevents LoRA injection), fall back to environment-prefix conditioning:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_factorized \
    --no-env-adapters \
    --rotation 0 \
    --exp-name lunarcompose_prefix_fallback_rot0
```

In fallback mode, `EnvAdapterBank(fallback_prefix_mode=True)` prepends an environment token to the language instruction instead of modifying visual-path weights. This still tests the factorization hypothesis but weakens the vision-path separation claim. Document immediately in experiment logs and downgrade the vision-path claim in the paper. Do not silently run in fallback mode and report as if visual-path adaptation worked.

### 6.5 Checkpoint Layout

Each training run produces:

```
checkpoints/
  lunarcompose_factorized_rot0/
    task_bank/
      payload/
      latch/
      clean/
      connector/
    env_bank/
      nominal/
      shadow/
      contamination/
    dual_router/
    config_hash.txt
    rotation_id.txt
    code_commit.txt
```

The `task_bank/` and `env_bank/` directories follow the same `nnx.State` dict serialization format as the SpaceCIL adapter checkpoints. Restore with `TaskAdapterBank.load(path)` and `EnvAdapterBank.load(path)`.

---

## 7. Baseline Experiments

All baselines use the same environment taxonomy (E0, E1, E2) and the same missing-corner split (same three rotations) as the factorized system. Evaluation runs across all 12 cells.

### 7.1 Monolithic (Single LoRA per Cell)

One LoRA module per task-env combination. No factorization. Each cell trains an entirely separate adapter.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_monolithic \
    --rotation 0 \
    --num-steps-per-cell 10000 \
    --exp-name lunarcompose_monolithic_rot0
```

Repeat for rotations 1 and 2. The monolithic baseline cannot generalize to unseen cells by design (no shared structure to compose). It sets an upper bound on seen-cell performance and a pessimistic lower bound on unseen-cell performance.

### 7.2 Domain-Randomized (No Env Adapter)

Uses the same `TaskAdapterBank` as SpaceCIL but mixes all three environment conditions in each task's training data without an explicit env adapter. Tests whether simple data augmentation across environments is sufficient.

Training: load each task's data from all three environment cells, mix, and train a single task adapter per task. No `EnvAdapterBank`.

```bash
# Training uses the task-only configs with mixed env data
# (wire this up manually in the training script as a --no-env-adapters variant)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_factorized \
    --no-env-adapters \
    --rotation 0 \
    --exp-name lunarcompose_domain_rand_rot0
```

### 7.3 Task Arithmetic

Compute task and environment adapters independently, then add their weight deltas at a fixed mixing coefficient rather than through learned composition. Follows Ilharco et al. (2022) and Zhang et al. (2023).

```
theta_final = theta_base + lambda_t * delta_task + lambda_e * delta_env
```

For task arithmetic:
1. Train task adapters independently (same as SpaceCIL).
2. Train env adapters independently on E0-only baselines per task.
3. At inference, add the deltas with `lambda_t = 1.0, lambda_e = 1.0` (default) or sweep both lambdas.

This baseline tests whether the factorized structure holds even without learning the composition jointly — pure arithmetic combination of independently trained modules.

### 7.4 Parameter-Matched PEFT

Same total parameter count as the factorized system, but a single LoRA module (no factorization). The single LoRA rank is scaled up so that `rank_single = rank_task + rank_env`.

This controls for the confound that factorized might win simply because it has more parameters than monolithic. If parameter-matched still underperforms factorized on unseen cells, the factorization structure itself is doing the work, not parameter count.

```bash
# Use lunarcompose_monolithic with a higher LoRA rank
# Adjust rank in config to match factorized total param count
# then run:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_lunarcompose.py \
    --config lunarcompose_monolithic \
    --rotation 0 \
    --exp-name lunarcompose_param_matched_rot0
```

Document the exact rank values used for both factorized and parameter-matched in the paper's implementation section.

### 7.5 Full Adaptation

Full fine-tuning of all parameters per task-env cell. No LoRA — all weights unfrozen. This is the unconstrained upper bound: the most parameters, the most expressiveness, and likely the worst generalization to unseen cells (overfit to seen conditions).

Run this only if compute budget permits. It is not required for the main claims but is useful for bounding the performance ceiling.

### 7.6 Baseline Summary Table

| Baseline | Config name | Env adapter | Task adapter | Notes |
|---------|------------|------------|------------|-------|
| Factorized (main) | `lunarcompose_factorized` | Yes (visual-path LoRA) | Yes (SpaceCIL bank) | Main system |
| Monolithic | `lunarcompose_monolithic` | No (merged) | No (merged) | Single LoRA per cell |
| Domain-randomized | `lunarcompose_factorized --no-env-adapters` | No | Yes | Mix envs, no env adapter |
| Task arithmetic | manual composition | Additive delta | Additive delta | Fixed lambda at inference |
| Parameter-matched | `lunarcompose_monolithic` (high rank) | No | No | Matched param count |
| Full adaptation | custom | No | No | All weights unfrozen |

---

## 8. Evaluation

### 8.1 Pre-Evaluation Setup

Before running any evaluation:

1. Run `harness.validate_split()` to confirm the split is intact.
2. Verify scorer cameras are properly mounted for E1 and E2 evaluation.
3. Confirm scorer presets are locked (no auto-exposure during trials).
4. Load initial-state seed files from the repo for each cell.
5. Verify no seed files are shared between train and test cell evaluation sets (leakage check).

### 8.2 Per-Cell Evaluation

Evaluation runs on all 12 cells (both seen and unseen). For each cell, use the evaluation episode set drawn from the committed initial-state seeds.

```python
from openpi.research.lunarcompose.missing_corner_harness import MissingCornerHarness

harness = MissingCornerHarness(
    task_ids=["payload", "latch", "clean", "connector"],
    env_ids=["nominal", "shadow", "contamination"],
    scorers={
        "payload": payload_scorer,
        "latch": latch_scorer,
        "clean": clean_scorer,
        "connector": connector_scorer,
    },
    eval_episodes=eval_episodes_dict,  # {(task_id, env_id): [Episode, ...]}
)

train_cells, test_cells = harness.generate_split(rotation=0)
result = harness.evaluate_all_cells()
```

Evaluation episodes per cell: 15-25 episodes per cell is the target. Use the same initial-state seeds across all compared systems (factorized vs. baselines) for a valid comparison.

### 8.3 Reading the Results

```python
print(f"Seen cells mean:   {result.seen_mean:.3f}")
print(f"Unseen cells mean: {result.unseen_mean:.3f}")
print(f"Seen-unseen gap:   {result.seen_unseen_gap:.3f}")

for task_id, score in result.per_task_breakdown.items():
    print(f"  {task_id}: {score:.3f}")

for env_id, score in result.per_env_breakdown.items():
    print(f"  {env_id}: {score:.3f}")
```

The `seen_unseen_gap` is the primary metric. A small gap (< 0.1) is consistent with successful factorization. A large gap (> 0.3) is strong evidence against it.

### 8.4 Factorization Diagnostics

Run all diagnostics after evaluating all cells. These are not optional checks — they are core scientific content.

```python
from openpi.research.lunarcompose.factorization_diagnostics import (
    seen_unseen_gap,
    cross_condition_breakdown,
    routing_interaction_analysis,
    task_env_entanglement,
    counterfactual_swap_test,
)

# Primary metric
gap = seen_unseen_gap(result)  # result.seen_mean - result.unseen_mean

# Which envs/tasks are the bottleneck?
breakdown = cross_condition_breakdown(result)

# Are the routing heads independent? (H2 test)
mi_score = routing_interaction_analysis(dual_router, held_out_batch)
# High MI = heads not routing independently = warning for H2

# Are adapter parameters entangled? (parameter-space H2 test)
entanglement = task_env_entanglement(task_bank, env_bank)
# Near zero = orthogonal (good); high = entangled (bad)

# Strongest test: swap one factor, measure impact
swap_result = counterfactual_swap_test(model, task_bank, env_bank, swap_spec)
# If swapping task routing changes env-sensitive behavior, factorization is broken
```

Diagnostic interpretation guide:

| Diagnostic | Good | Warning | Failure |
|-----------|------|---------|---------|
| `seen_unseen_gap` | < 0.1 | 0.1-0.3 | > 0.3 |
| `routing_interaction_analysis` (MI) | Near 0 | 0.1-0.3 | > 0.3 |
| `task_env_entanglement` | Near 0 | 0.1-0.3 | > 0.3 |
| `counterfactual_swap_test` | Task swap changes task perf only | Task swap leaks into env perf | Cross-factor contamination |

### 8.5 Multiple Rotations

Run the full evaluation pipeline for all three rotations. The main result is the mean and standard deviation across rotations.

```python
results = {}
for rotation in [0, 1, 2]:
    train_cells, test_cells = harness.generate_split(rotation=rotation)
    results[rotation] = harness.evaluate_all_cells()

# Aggregate
import statistics
gaps = [results[r].seen_unseen_gap for r in [0, 1, 2]]
seen_means = [results[r].seen_mean for r in [0, 1, 2]]
unseen_means = [results[r].unseen_mean for r in [0, 1, 2]]

print(f"Seen:   {statistics.mean(seen_means):.3f} ± {statistics.stdev(seen_means):.3f}")
print(f"Unseen: {statistics.mean(unseen_means):.3f} ± {statistics.stdev(unseen_means):.3f}")
print(f"Gap:    {statistics.mean(gaps):.3f} ± {statistics.stdev(gaps):.3f}")
```

Report all three rotation results in the paper, plus mean and std. Do not cherry-pick the best rotation.

---

## 9. Scorer Decoupling Verification

This is a blocking requirement for E1 and E2. The scorer must not become the hidden confounder of the paper.

### 9.1 Why This Matters

If the scorer camera is affected by the E1 shadow pattern or by E2 surface contamination, the scorer itself may fail — making it appear that the policy is failing when the evaluation signal is broken. This would invalidate all E1 and E2 results.

### 9.2 Validation Protocol

For each scorer and each environment condition:

1. Collect 20+ pilot episodes in the target environment with both the policy camera and scorer camera active.
2. Score each episode using the automated scorer.
3. Have a human operator independently label the same episodes from the scorer camera feed.
4. Compute agreement (precision and recall against human labels).
5. Accept threshold: agreement must be within ±5% of the E0 baseline agreement.

If scorer agreement drops more than 5 percentage points from E0 baseline:
- Stop all E1 or E2 experiments using that scorer.
- Redesign the scorer view or scoring signal.
- Do not report E1 or E2 results until fixed.

### 9.3 Cleaning Task Scorer Special Case

The cleaning task scorer is the highest-risk scorer for E2. If it uses pixel-difference or surface-color signals, the gray powder on the workspace will corrupt the baseline measurement. Verify explicitly:

- Scoring region must be outside all contaminated tiles.
- Or the scorer must use marker-based or segmentation-based signals that are robust to surface appearance change.
- Run a dedicated contamination-only test: set up E2 conditions with a clean (un-wiped) scoring region and verify the scorer returns the expected "no clean" score.

### 9.4 E1 Scorer Camera Mounting

The scorer camera for E1 must be positioned so that:
- The robot arm and gripper do not cast shadows into the scorer's field of view.
- The scorer has its own separate diffuse light source.
- The E1 collimated LED panel does not illuminate the scorer's target region directly.

Verify by running the E1 LED at full intensity and checking that the scorer camera image does not change compared to E0. If it changes, remount or add a baffle.

---

## 10. Expected Results and Interpretation

### 10.1 H1: Factorized vs. Monolithic on Unseen Cells

**Success condition.** Factorized system shows smaller seen-unseen gap than monolithic on average across rotations. Ideally: factorized gap < 0.15, monolithic gap > 0.3.

**If H1 is confirmed:** The paper can claim factorized adaptation improves compositional generalization to unseen task-environment combinations.

**If H1 is partially confirmed:** Factorized outperforms on some rotations but not others. Report disaggregated results. Investigate which tasks or environments are driving failure.

**If H1 is not confirmed:** The monolithic and factorized systems perform similarly on unseen cells. This is a valid finding — report honestly. The factorization structure may not generalize better in this task-environment regime.

### 10.2 H2: Routing Independence

**Success condition.** Mutual information between task routing weights and env routing weights is low (< 0.1) over a held-out batch. Counterfactual swap test shows that swapping task routing changes task-sensitive behavior without significantly changing env-sensitive behavior.

**If H2 is confirmed:** The paper can claim the dual-head router produces interpretable and controllable specialization.

**If H2 fails (high MI):** The heads are entangled. Report the MI score. Investigate whether the training signal for the two heads is too correlated (e.g., some tasks only appear in one environment during training, making task and env signals linearly dependent).

### 10.3 H3: Visual Condition Taxonomy

**Success condition.** Per-env breakdown shows E1 and E2 degrade performance relative to E0 for at least one baseline. The environment conditions are not too easy (baseline succeeds in all three) and not too hard (all systems fail in all three).

**Target regime.** A policy that only saw E0 during training should score meaningfully lower in E1 and E2 than in E0 (e.g., 30+ percentage points lower). If no gap exists, the conditions are too similar to be meaningful. If the gap is 100%, the conditions are too hard to study adaptation at all.

**If H3 is not confirmed:** The E1 or E2 conditions may not be sufficiently perturbing. Increase shadow contrast (lower ambient to < 20 lux) or increase dust density. Document the change and recollect.

### 10.4 H4: Missing-Corner Protocol Detectability

**Success condition.** The seen-unseen gap differs meaningfully between the factorized system and at least one baseline across all three rotations. If the gap is consistent across rotations, the protocol is detecting a real signal, not rotation-specific noise.

**If H4 fails:** Rotation variance is so high that no consistent conclusion can be drawn. This may indicate the evaluation episode count per cell is too low. Increase to 25+ episodes per cell.

### 10.5 Falsifiability and Honest Reporting

If **both** of the following hold simultaneously, the factorization assumption is falsified:

1. `seen_unseen_gap` > 0.3 across **all three rotations** for the factorized system.
2. `counterfactual_swap_test` shows swapping one factor degrades performance in the domain of the other factor.

This is a valid scientific result. The paper must report it honestly. The claim boundary explicitly allows for this outcome — "We study factorized adaptation and evaluate whether the factorization assumption holds" is still a true and meaningful claim even if the answer is "it does not hold in this setting."

Do not bury a falsified factorization result in an appendix. Report it in the main paper body and frame it as a finding about the limits of factorized adaptation in this regime.

---

## 11. Artifact Policy

Every experiment run must save the following before results are reported:

### 11.1 Per-Run Artifacts

| Artifact | Where to save |
|---------|-------------|
| Config hash (SHA-256 of config JSON) | `{checkpoint_dir}/{exp_name}/config_hash.txt` |
| Code commit (git SHA) | `{checkpoint_dir}/{exp_name}/code_commit.txt` |
| Rotation ID | `{checkpoint_dir}/{exp_name}/rotation_id.txt` |
| Task adapter versions | `{checkpoint_dir}/{exp_name}/task_bank/` |
| Env adapter versions | `{checkpoint_dir}/{exp_name}/env_bank/` |
| Per-cell success scores (JSON) | `{checkpoint_dir}/{exp_name}/per_cell_scores.json` |
| `MissingCornerResult` (serialized) | `{checkpoint_dir}/{exp_name}/missing_corner_result.json` |
| Diagnostic outputs | `{checkpoint_dir}/{exp_name}/diagnostics.json` |
| Scorer outputs per cell | `{checkpoint_dir}/{exp_name}/scorer_outputs/` |
| Raw video references | `{checkpoint_dir}/{exp_name}/video_refs.txt` |
| Fail type log | `{checkpoint_dir}/{exp_name}/fail_types.json` |
| Split membership (train/test cell log) | `{checkpoint_dir}/{exp_name}/split.json` |

### 11.2 Required Before Any Push to the Paper

Before any result makes it into the paper draft:

- [ ] All three rotations completed for the factorized system.
- [ ] All three rotations completed for each baseline.
- [ ] Scorer validation logs present for E1 and E2.
- [ ] No scene-instance leakage detected in any rotation.
- [ ] All `MissingCornerResult` objects have matching `rotation_id` and `timestamp`.
- [ ] `harness.validate_split()` passing for all rotations used in the paper.

---

## 12. Running Tests Before Commits

Before committing or pushing any LunarCompose code:

```bash
# Run all LunarCompose module tests
uv run pytest src/openpi/research/lunarcompose/ -x -q

# Run all research tests (SpaceCIL + LunarCompose + shared)
uv run pytest src/openpi/research/ -x -q

# Run the full test suite
uv run pytest src/ scripts/ -x -q
```

All tests must pass before committing. Do not push broken code.

### 12.1 Gate G3 Checklist

The LunarCompose extension is ready for mainline experiments when all of these pass:

**Module-level:**
- [ ] `uv run pytest src/openpi/research/lunarcompose/ -x -q` passes.
- [ ] `lunarcompose_debug` config instantiates without GPU.
- [ ] `EnvAdapterBank` creates, registers, retrieves, and merges env adapters correctly.
- [ ] `DualHeadRouter` produces two independent softmax distributions on dummy input.
- [ ] `MissingCornerHarness.generate_split(rotation=0)` produces a valid split (no leakage, full coverage).
- [ ] `factorization_diagnostics.seen_unseen_gap` computes correctly on synthetic `MissingCornerResult`.

**Environment-level (real robot):**
- [ ] E0 scorer validated against manual labels (precision/recall logged).
- [ ] E1 shadow setup reproduces within allowed variation window across two sessions.
- [ ] E2 contamination setup is resettable per the maintenance protocol.
- [ ] Scorer camera confirmed decoupled for E1 and E2.
- [ ] Scorer does not degrade more than 5pp under E1 or E2 conditions.

**Data pipeline:**
- [ ] `meta.env_id` is logged in all collected episodes.
- [ ] `meta.camera_preset_id` matches declared env for all episodes.
- [ ] Missing-corner split tooling verified on real collected data (no scene-instance leakage).

---

## 13. Paper Writing Notes

### 13.1 Key Figures

**Figure 1 (primary result): 4×3 heatmap.** One heatmap per system (factorized, monolithic, domain-randomized). Rows = tasks, columns = environments. Color = success rate. Test cells marked with a border or asterisk. Averaged across rotations.

**Figure 2: Seen vs. unseen bar chart.** Side-by-side bars for seen-cell mean and unseen-cell mean, grouped by system. Rotation 0 as the main bar, rotations 1 and 2 as error bars (std).

**Figure 3: Diagnostic plots.** Three subplots: (a) MI between routing heads across training, (b) task-env entanglement score per run, (c) counterfactual swap test delta.

**Figure 4: Per-environment breakdown.** Bar chart showing per-env mean success for factorized vs. monolithic. Shows whether E1 or E2 is the harder condition.

### 13.2 Table Structure

**Main results table:**

| System | Seen | Unseen | Gap | Rotation 0 | Rotation 1 | Rotation 2 |
|--------|------|--------|-----|-----------|-----------|-----------|
| Factorized (ours) | | | | | | |
| Monolithic | | | | | | |
| Domain-randomized | | | | | | |
| Task arithmetic | | | | | | |
| Parameter-matched | | | | | | |

Values = mean success rate (±std over 3 rotations).

**Diagnostic table:**

| System | MI score | Entanglement | Counterfactual delta |
|--------|---------|-------------|---------------------|
| Factorized | | | |

### 13.3 Claim Phrasing Reminders

Use these formulations:

- "factorized adaptation under lunar-analog environment shifts"
- "unseen task-environment combination generalization"
- "missing-corner real-robot protocol"
- "compositional generalization evaluated on a real mobile manipulation platform"

Avoid these formulations:

- "strict orthogonal disentanglement"
- "solves lunar domain adaptation"
- "proves task-environment independence"
- "first factorized VLA system"

### 13.4 Related Work Positioning

Organize related work into four groups:

1. VLA and generalist policies (pi0, pi0.5, OpenVLA, RoboVLMs)
2. Continual and compositional robot learning (LIBERO, CompoSuite, lifelong manipulation papers)
3. Compositional and modular PEFT (Task Arithmetic, Composing PEFT Modules with Arithmetic)
4. Space and lunar operational context (NASA lighting study, NASA Dust Mitigation Roadmap, ESA lunar south pole)

The paper is positioned at the intersection of (2), (3), and (4) — not as a pure continual learning paper and not as a pure domain adaptation paper.

### 13.5 Handling a Falsified Factorization Result

If the factorization assumption is falsified (Section 10.5), the paper's abstract and introduction must be revised to frame the contribution honestly:

- Lead with the protocol contribution: "We introduce a missing-corner evaluation protocol for real-robot compositional generalization under lunar-analog visual conditions."
- Report the factorization result as the finding: "Under our protocol, we find that factorized adaptation does not yield compositional generalization in this task-environment regime, and we characterize the failure mode using routing entanglement diagnostics."
- Keep all diagnostic results. A falsification finding is most credible when the diagnostics show clearly *why* it failed.

This is not a weaker paper. It is a more honest one, and it sets a clearer challenge for future work.

---

## Appendix A: Config Reference

### A.1 Per-Cell Configs (12 total)

All follow the pattern `lunarcompose_{task}_{env}`. Each uses:
- `model`: pi0.5, `gemma_2b_lora` + `gemma_300m_lora`
- `weight_loader`: pi0.5 base checkpoint
- `freeze_filter`: LoRA freeze filter (action expert + language model path)
- `num_train_steps`: 10,000 per cell
- `batch_size`: 32

| Config name | Task | Environment |
|-------------|------|------------|
| `lunarcompose_payload_nominal` | payload transfer | E0 |
| `lunarcompose_payload_shadow` | payload transfer | E1 |
| `lunarcompose_payload_contamination` | payload transfer | E2 |
| `lunarcompose_latch_nominal` | latch actuation | E0 |
| `lunarcompose_latch_shadow` | latch actuation | E1 |
| `lunarcompose_latch_contamination` | latch actuation | E2 |
| `lunarcompose_clean_nominal` | surface cleaning | E0 |
| `lunarcompose_clean_shadow` | surface cleaning | E1 |
| `lunarcompose_clean_contamination` | surface cleaning | E2 |
| `lunarcompose_connector_nominal` | connector mating | E0 |
| `lunarcompose_connector_shadow` | connector mating | E1 |
| `lunarcompose_connector_contamination` | connector mating | E2 |

### A.2 Architecture-Level Configs

| Config name | Purpose | Steps |
|-------------|---------|-------|
| `lunarcompose_factorized` | Main factorized training | 20,000 |
| `lunarcompose_monolithic` | Monolithic baseline | 20,000 |
| `lunarcompose_debug` | Smoke-test, no GPU, fake data | 10 |

### A.3 Training Script CLI Reference

```
uv run scripts/train_lunarcompose.py [OPTIONS]

Options:
  --config TEXT                  openpi config name (required)
  --rotation [0|1|2]             Missing-corner split rotation (default: 0)
  --num-steps-per-cell INT       Training steps per cell (default: 10000)
  --no-env-adapters              Disable env adapters (fallback/domain-rand mode)
  --checkpoint-dir TEXT          Checkpoint directory (default: checkpoints)
  --seed INT                     Random seed (default: 42)
  --exp-name TEXT                Experiment name (default: lunarcompose_run)
```

---

## Appendix B: Environment Equipment Reference

| Parameter | E0 | E1 | E2 |
|-----------|----|----|-----|
| Lux at object | 500-700 | 800-1200 (direct), < 100 (shadow) | 500-700 |
| Color temp | 4000-5000K | 5500-6500K | 4000-5000K |
| Shadow depth | > 0.7 ratio | < 0.1 ratio | > 0.7 ratio |
| Camera preset | `nominal` | `e1_shadow` | `e2_contamination` |
| Scorer decoupled | No | Yes | Yes |
| Surface modification | None | None | Powder/rubble |

---

## Appendix C: Failure Mode Reference

| Failure | Root cause | Immediate action | Claim impact |
|---------|-----------|-----------------|-------------|
| Visual-path env adapter blocked | SigLIP LoRA injection unavailable | Switch to env-prefix fallback immediately, document | Weaken "vision-path separation" claim; factorization experiment still valid |
| Env scorer confounded by E1 lighting | Evaluation confound | Decouple scorer camera, redesign scoring signal; **do not proceed until fixed** | Cannot report E1 results until resolved |
| Env scorer confounded by E2 contamination | Evaluation confound | Exclude contaminated tiles from scoring region or use marker-based scorer | Cannot report E2 results until resolved |
| Task-env leakage in missing-corner split | Data hygiene error | Rebuild split, revalidate, invalidate affected runs | No paper impact if caught before results are reported |
| Dual-head router entangled | Optimization / correlated training signal | Separate head training signals, add independence penalty | Weakens H2 (routing controllability); H1 unaffected |
| Seen-unseen gap too large to claim success | Factorization assumption failed | Report honestly as finding; do not suppress | Reframe contribution as "factorization falsified in this setting" |
| Calibration drift mid-session | Hardware | Recertify, label all demos from that session as hardware-invalid | No impact if caught early |
| Domain-rand baseline too strong | Factorization unnecessary here | Report this as a genuine finding; discuss implications for when factorization helps | Weakens H1 unless gap is still present in some rotation |
