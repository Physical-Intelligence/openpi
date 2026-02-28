# LunarCompose — Implementation Plan
## Paper B: Factorized Task × Environment Adaptation for Lunar-Analog Compositional Generalization

---

## 1. Paper Summary

LunarCompose asks a focused question: when you transfer an Earth-trained VLA into a robot operating under lunar-analog conditions, should the adaptation be factorized along two axes (task and environment) or collapsed into a single module?

The paper tests this on a real mobile manipulation platform, a wheeled base with an RM75 7-DoF arm and two-finger gripper, where the primary policy camera is a wrist RGB image. Backbone: released pi0.5 via the openpi flow-matching policy path. The evaluation protocol trains on a subset of task-environment combinations and measures whether factorized adaptation generalizes to unseen combinations better than non-factorized baselines.

**Why this matters for space robotics.** Lunar robots won't see a single frozen condition. Illumination, shadow geometry, surface contamination, and visual contrast all shift across locations and mission phases. If adaptation entangles task semantics and environment-specific corrections into one monolithic module, reuse across condition changes becomes brittle. Factorized adaptation offers a principled alternative: learn task capability once, then compose with environment correction independently.

**What the paper is not claiming.** This is not a proof of strict orthogonal disentanglement between task and environment factors. The factorization assumption is explicitly testable and explicitly allowed to fail. The claim boundary is:

> "We study factorized adaptation under lunar-analog environment shifts and evaluate compositional generalization to unseen task-environment combinations on a real mobile manipulation platform."

### 1.1 Hypotheses

| Label | Hypothesis |
|-------|------------|
| H1 | Factorized task+env adaptation improves unseen-combination performance over non-factorized baselines |
| H2 | Explicit task/environment routing yields more interpretable and controllable specialization |
| H3 | A small lunar-analog visual condition taxonomy suffices to expose environment-sensitive failure modes |
| H4 | The missing-corner evaluation protocol can determine whether the factorization assumption actually holds |

### 1.2 Core Contributions

1. A real-robot factorized adaptation benchmark for lunar-analog visual conditions.
2. A task × environment compositional generalization protocol using unseen-combination evaluation.
3. A factorized adaptation framework with separate task and environment specialization modules.
4. Diagnostics that can falsify the factorization assumption honestly.

### 1.3 Dependency on Paper A

Paper B is not independent. It extends the Paper A (SpaceCIL) infrastructure and must not be started before:

- One task trains and replays correctly on the wrist-camera policy path (Gate G1).
- At least two tasks train sequentially with a working task adapter bank (Gate G2).
- The router, distillation, and continual harness are stable enough to hand off.

Build order: **Phase A (shared infra) → Phase B (SpaceCIL core) → Phase C (LunarCompose extension).**

---

## 2. Module: `config.py`

**File:** `src/openpi/research/lunarcompose/config.py`
**Classification:** new module
**Pattern:** polaris_config.py, identical registration pattern

### 2.1 Purpose

Exports `get_lunarcompose_configs() → list[TrainConfig]`. This list is splat-registered into `src/openpi/training/config.py` with one line:

```python
*lunarcompose_config.get_lunarcompose_configs(),
```

All config names must be globally unique and namespace-prefixed with `lunarcompose_`.

### 2.2 Config Taxonomy

**Per-cell configs** (one per task-environment combination):

| Config name | Task | Environment |
|---|---|---|
| `lunarcompose_payload_nominal` | payload transfer | E0 nominal |
| `lunarcompose_payload_shadow` | payload transfer | E1 hard-shadow |
| `lunarcompose_payload_contamination` | payload transfer | E2 contamination |
| `lunarcompose_latch_nominal` | latch actuation | E0 nominal |
| `lunarcompose_latch_shadow` | latch actuation | E1 hard-shadow |
| `lunarcompose_latch_contamination` | latch actuation | E2 contamination |
| `lunarcompose_clean_nominal` | surface cleaning | E0 nominal |
| `lunarcompose_clean_shadow` | surface cleaning | E1 hard-shadow |
| `lunarcompose_clean_contamination` | surface cleaning | E2 contamination |
| `lunarcompose_connector_nominal` | connector mating | E0 nominal |
| `lunarcompose_connector_shadow` | connector mating | E1 hard-shadow |
| `lunarcompose_connector_contamination` | connector mating | E2 contamination |

**Architecture-level configs:**

| Config name | Purpose |
|---|---|
| `lunarcompose_factorized` | Main factorized training config, task+env adapter banks active |
| `lunarcompose_monolithic` | Baseline: single LoRA module per task-env cell, no factorization |
| `lunarcompose_debug` | Fast smoke-test using `paligemma_variant="dummy"` |

### 2.3 Config Fields to Set

Each config specifies:
- `model_variant`: pi05 flow-matching head
- `peft_config`: LoRA rank and target modules (task-path vs env-path)
- `data_config`: which task-env cells are included in this training run
- `env_metadata_mode`: `"factorized"` or `"monolithic"` or `"prefix_only"`
- `weight_loader`: checkpoint from SpaceCIL task adapter bank where relevant
- `lora_rank`, `lora_alpha`: consistent with SpaceCIL adapter bank configuration

### 2.4 Integration Checklist

- [ ] `get_lunarcompose_configs()` returns a non-empty list
- [ ] All config names are unique across the full `_CONFIGS` list
- [ ] Debug config instantiates without GPU (uses dummy model)
- [ ] Factorized config and monolithic config differ only in `env_metadata_mode` and PEFT structure

---

## 3. Module: `env_adapter_bank.py`

**File:** `src/openpi/research/lunarcompose/env_adapter_bank.py`
**Classification:** new module
**Depends on:** `task_adapter_bank.py` (Paper A), openpi LoRA (`models/lora.py`)

### 3.1 Purpose

Maintains a registry of environment-specialized PEFT modules, one per environment condition. Composable with `TaskAdapterBank` so that both can be applied simultaneously to the same model.

### 3.2 Key Class: `EnvAdapterBank`

```
EnvAdapterBank
  adapters: dict[str, nnx.State]   # env_id -> LoRA parameter state dict
  lora_target: str                  # which layer group env adapters target
  register_env(env_id, lora_state)  # add or overwrite env adapter
  get_env(env_id) -> nnx.State      # retrieve env adapter state
  merge_into_model(model, env_id)   # apply env adapter via nnx.update(model, state)
  list_envs() -> list[str]          # registered env IDs
  save(path)                        # checkpoint env adapters
  load(path) -> EnvAdapterBank      # restore from checkpoint
```

### 3.3 Layer Targeting Strategy

The key design goal is orthogonality between task and env adapters:

**Preferred path (visual-path LoRA):**
- Env adapters target the vision encoder (SigLIP layers).
- Task adapters (from SpaceCIL) target the action expert and language model LoRA layers.
- Since they target different sub-graphs, applying both is additive and non-conflicting.
- Filter string example: `".*siglip.*"` for env adapters, `".*action_expert.*|.*lm.*"` for task adapters.

**Fallback path (additive LoRA on shared layers):**
- If visual-path LoRA is technically blocked (e.g., the SigLIP visual encoder is frozen in a way that prevents LoRA injection), fall back to environment-prefix conditioning: prepend an environment token to the language instruction.
- Fallback must be activated and documented explicitly, not silently.
- Fallback weakens the vision-path separation claim but does not invalidate the factorization experiment.

**Additive composition rule:**
```
effective_weights = base_weights + task_lora_delta + env_lora_delta
```
Both deltas are computed independently from separate LoRA parameter sets. `nnx.update` is called twice in sequence, once with the task state, once with the env state.

### 3.4 Compatibility Contract

`EnvAdapterBank` must not break `TaskAdapterBank`. Specifically:
- Calling `task_bank.merge_into_model(model, task_id)` followed by `env_bank.merge_into_model(model, env_id)` must produce a model with both adapter sets applied.
- Order matters if adapters share layers. If visual-path targeting is clean, order is irrelevant.
- Both banks use the same checkpoint serialization format (`nnx.State` dict) for consistency.

### 3.5 Tests (`env_adapter_bank_test.py`)

| Test | What it checks |
|---|---|
| `test_register_and_retrieve` | Register a dummy env adapter, retrieve it, verify state matches |
| `test_merge_into_model` | Apply env adapter to a dummy model, verify parameter values changed |
| `test_composition_with_task_adapter` | Apply task adapter then env adapter, verify both are applied |
| `test_checkpoint_roundtrip` | Save and load env bank, verify states are identical |
| `test_visual_path_targeting` | Verify `lora_target` filter matches only SigLIP layers |
| `test_fallback_prefix_mode` | If visual-path is disabled, prefix-mode env conditioning works |

---

## 4. Module: `dual_head_router.py`

**File:** `src/openpi/research/lunarcompose/dual_head_router.py`
**Classification:** new module
**Depends on:** `router.py` (SpaceCIL), same language+visual input pipeline

### 4.1 Purpose

Extends SpaceCIL's `TaskRouter` with a second routing head dedicated to environment selection. The two heads share input features but produce independent routing distributions over separate adapter banks.

### 4.2 Key Class: `DualHeadRouter`

```
DualHeadRouter
  task_head: MLP     # routes over task adapter bank
  env_head: MLP      # routes over env adapter bank
  feature_dim: int   # shared input feature dimensionality

  forward(lang_emb, visual_summary) -> (task_weights, env_weights)
    # task_weights: softmax over task_ids, shape [num_tasks]
    # env_weights: softmax over env_ids, shape [num_envs]

  route(lang_emb, visual_summary) -> (task_id, env_id)
    # argmax of each head's output

  counterfactual_task(lang_emb, visual_summary, fixed_env_id) -> task_id
    # route task while holding env routing fixed

  counterfactual_env(lang_emb, visual_summary, fixed_task_id) -> env_id
    # route env while holding task routing fixed

  mutual_information_estimate(batch) -> float
    # diagnostic: MI between task_weights and env_weights over a batch
```

### 4.3 Input Features

Same as SpaceCIL router:
- `lang_emb`: language embedding from the frozen language encoder, summarized to a fixed-dim vector.
- `visual_summary`: pooled visual features from the vision encoder.
- Optionally concatenated: compact proprioceptive state summary.

The two heads share the same input but do not share weights with each other.

### 4.4 Independence Design

The two heads should produce routing decisions that are as independent as possible:

- **Training signal:** task head is trained on task-classification accuracy; env head on env-classification accuracy. They do not jointly optimize a combined routing loss.
- **Diagnostic hook:** `mutual_information_estimate` measures empirical MI between `task_weights` and `env_weights` over a held-out batch. High MI is a warning sign that the heads are entangled.
- **Counterfactual support:** Swap one head's output while freezing the other's. If the model's behavior changes appropriately (task changes while env held, or vice versa), factorization is working as intended.

### 4.5 Relationship to SpaceCIL Router

`DualHeadRouter` wraps or subclasses SpaceCIL's `TaskRouter`:
- `task_head` is the original SpaceCIL router head (reused weights or re-initialized).
- `env_head` is a new MLP of the same architecture.
- Forward pass calls both heads in parallel on the same feature input.

### 4.6 Tests (`dual_head_router_test.py`)

| Test | What it checks |
|---|---|
| `test_forward_shapes` | Output shapes match (num_tasks,) and (num_envs,) |
| `test_softmax_validity` | Both outputs are valid probability distributions (sum to 1, non-negative) |
| `test_independence_diagnostic` | MI estimate returns a float, runs without error |
| `test_counterfactual_task_swap` | Fixing env routing while changing lang_emb changes task routing |
| `test_counterfactual_env_swap` | Fixing task routing while changing visual_summary changes env routing |
| `test_route_returns_ids` | `route()` returns valid task_id and env_id strings |

---

## 5. Module: `missing_corner_harness.py`

**File:** `src/openpi/research/lunarcompose/missing_corner_harness.py`
**Classification:** new module
**Depends on:** `EnvAdapterBank`, `TaskAdapterBank`, `scorer_base.py`

### 5.1 Purpose

Manages the train/test split over the task × environment grid, runs evaluation on all cells (seen and unseen), and returns structured results. This is the scientific centerpiece of Paper B.

### 5.2 Key Class: `MissingCornerHarness`

```
MissingCornerHarness
  task_ids: list[str]       # e.g. ["payload", "latch", "clean", "connector"]
  env_ids: list[str]        # e.g. ["nominal", "shadow", "contamination"]
  train_cells: set[tuple[str, str]]   # (task_id, env_id) pairs used in training
  test_cells: set[tuple[str, str]]    # held-out pairs for evaluation

  generate_split(rotation: int) -> (train_cells, test_cells)
    # rotation=0 is canonical, rotation=1,2 are rotated splits

  validate_split() -> bool
    # checks coverage and leakage constraints

  run_evaluation(model, task_bank, env_bank, dual_router) -> MissingCornerResult
    # evaluate all cells, return per-cell success rates

  seen_cells() -> set[tuple[str, str]]
    # alias for train_cells (for readability in analysis)

  unseen_cells() -> set[tuple[str, str]]
    # alias for test_cells
```

### 5.3 `MissingCornerResult` Data Structure

```
MissingCornerResult
  per_cell_success: dict[tuple[str, str], float]   # (task, env) -> success rate
  seen_mean: float                                  # mean over train_cells
  unseen_mean: float                                # mean over test_cells
  seen_unseen_gap: float                            # seen_mean - unseen_mean
  per_task_breakdown: dict[str, float]              # mean across envs per task
  per_env_breakdown: dict[str, float]               # mean across tasks per env
  rotation_id: int                                  # which split rotation
  timestamp: str
```

### 5.4 Split Constraints

Every valid split must satisfy:
1. Every task appears in at least one training cell.
2. Every environment appears in at least one training cell.
3. Test cells are strictly unseen combinations (no overlap with training cells).
4. No scene-instance ID leakage: same physical setup cannot appear in both train and test.

Violation of any constraint causes `validate_split()` to return `False` and raises a loud error before any training or evaluation runs.

### 5.5 Canonical Split Example (4 tasks × 3 envs = 12 cells)

**Rotation A (canonical):**
```
Train cells (8):
  (payload, nominal), (payload, shadow)
  (latch, nominal), (latch, contamination)
  (clean, shadow), (clean, contamination)
  (connector, nominal), (connector, contamination)

Test cells (4):
  (payload, contamination)
  (latch, shadow)
  (clean, nominal)
  (connector, shadow)
```

**Rotation B:**
```
Test cells (4):
  (payload, nominal)
  (latch, contamination)
  (clean, shadow)
  (connector, contamination)
```

**Rotation C:**
```
Test cells (4):
  (payload, shadow)
  (latch, nominal)
  (clean, contamination)
  (connector, nominal)
```

Each rotation must satisfy all four split constraints above.

### 5.6 Initial-State Reproducibility

- Each evaluation cell uses a fixed set of initial-state seeds (JSON files).
- Seeds are committed to the repo so that runs are replayable.
- Rotation ID and split version are logged in every experiment artifact.

### 5.7 Tests (`missing_corner_harness_test.py`)

| Test | What it checks |
|---|---|
| `test_split_coverage` | Every task and env appears in train_cells for each rotation |
| `test_no_leakage` | train_cells and test_cells are disjoint for all rotations |
| `test_constraint_violation_raises` | Invalid splits (uncovered task/env) raise loudly |
| `test_result_structure` | `MissingCornerResult` has all required fields, correct types |
| `test_mock_evaluation` | With a mock scorer returning fixed values, result math is correct |
| `test_seen_unseen_gap` | Gap computed correctly from known per-cell values |

---

## 6. Module: `factorization_diagnostics.py`

**File:** `src/openpi/research/lunarcompose/factorization_diagnostics.py`
**Classification:** new module
**Depends on:** `MissingCornerResult`, `DualHeadRouter`, `EnvAdapterBank`, `TaskAdapterBank`

### 6.1 Purpose

Provides diagnostic functions to assess whether the factorization assumption holds. These are not auxiliary checks, they are core scientific content. The paper must include these results and report honestly if factorization fails.

### 6.2 Diagnostic Functions

**`seen_unseen_gap(results: MissingCornerResult) -> float`**
- Returns `results.seen_mean - results.unseen_mean`.
- A large gap means the model did not generalize to unseen combinations.
- Interpretation guide: gap > 0.3 is strong evidence against factorization; gap < 0.1 is consistent with successful factorization.

**`cross_condition_breakdown(results: MissingCornerResult) -> dict`**
- Returns per-env and per-task performance breakdowns.
- Reveals which environments or tasks are the bottleneck.
- Useful for diagnosing whether one particular condition (e.g. E1 shadow) dominates failure.

**`routing_interaction_analysis(dual_router: DualHeadRouter, batch) -> float`**
- Estimates mutual information between task routing weights and env routing weights over a batch.
- High MI indicates the heads are not routing independently.
- This is a direct test of H2.

**`task_env_entanglement(task_bank: TaskAdapterBank, env_bank: EnvAdapterBank) -> float`**
- Measures parameter-space overlap or interference between task and env adapters.
- One approach: cosine similarity between flattened task LoRA parameters and flattened env LoRA parameters in shared-layer configurations.
- Zero is ideal (perfectly orthogonal); high values indicate the adapters are not factorized at the parameter level.

**`counterfactual_swap_test(model, task_bank, env_bank, swap_spec) -> dict`**
- Given a `swap_spec` that specifies which factor to swap while holding the other fixed, runs evaluation under both original and swapped routing.
- Returns performance delta: does swapping task routing (while fixing env) change task behavior without changing env-sensitive behavior?
- This is the strongest test of controllability (H2).

### 6.3 Falsifiability Protocol

If the following conditions both hold, the factorization assumption is falsified and the paper must report this clearly:

1. `seen_unseen_gap` is large (> 0.3) across all rotations.
2. `counterfactual_swap_test` shows that swapping one factor degrades performance in the domain of the other factor.

**Reporting rule:** Do not bury this result. A falsified factorization assumption is a valid scientific finding. The claim boundary allows for this outcome.

### 6.4 Tests (`factorization_diagnostics_test.py`)

| Test | What it checks |
|---|---|
| `test_seen_unseen_gap_known_values` | With synthetic results where seen=0.8, unseen=0.4, gap=0.4 |
| `test_cross_condition_breakdown_shape` | Returns dict with correct keys per task and env |
| `test_routing_interaction_scalar` | MI estimate returns a single float, non-negative |
| `test_entanglement_zero_for_orthogonal` | Random orthogonal adapters produce near-zero entanglement |
| `test_entanglement_high_for_identical` | Identical adapter states produce high entanglement |
| `test_counterfactual_swap_structure` | Returns dict with required keys, no crash |

---

## 7. Baseline Configuration

All baselines use the same environment taxonomy and missing-corner split as the main factorized system.

| Baseline | Description | Config name |
|---|---|---|
| Full adaptation | Fine-tune all parameters per task-env cell, no factorization | `lunarcompose_full_ft` |
| Single merged PEFT | One shared LoRA for all task-env combinations | `lunarcompose_monolithic` |
| Domain-randomized | Train on all envs mixed per task, no env adapter | `lunarcompose_domain_rand` |
| Task arithmetic | Compose task and env adapters via arithmetic (Ilharco et al.) | `lunarcompose_task_arith` |
| Parameter-matched PEFT | Same total parameter count as factorized, but single module | `lunarcompose_param_matched` |

**Domain-randomized baseline notes:** Uses the same task adapter bank as SpaceCIL but mixes all three environment conditions in each task's training data without an explicit env adapter. This tests whether factorization is necessary or if simple data augmentation across environments suffices.

**Task arithmetic baseline notes:** Follows Ilharco et al. (2022) and Zhang et al. (2023). Compute task adapter and env adapter independently, then add their weight deltas: `theta_final = theta_base + lambda_t * delta_task + lambda_e * delta_env`. Unlike the main system, the arithmetic coefficients are fixed at inference time rather than learned.

**Parameter-matched baseline design:** Match the total number of trainable parameters between factorized (task_lora + env_lora) and the merged baseline (one_lora). The merged baseline uses a single LoRA with rank scaled up to match. This controls for parameter count as a confound.

---

## 8. Environment Taxonomy

Three lunar-analog visual conditions. Each is defined with enough physical precision to be reproduced across sessions.

### E0 — Nominal Laboratory Condition

**Rationale:** Controlled indoor lighting, clean workspace surfaces. Serves as the held-stable reference condition. All tasks must first be demonstrated and validated in E0.

**Physical setup:**
- Standard indoor diffuse lighting: 500-700 lux at object surface.
- Workspace surface: matte gray or neutral beige with no reflective patches.
- No occlusions or debris on the workspace.
- Wrist camera exposure: locked at nominal preset (see `meta.camera_preset_id`).
- Scene camera: same nominal preset.

**Lighting parameters:**
- Color temperature: 4000-5000K (neutral white).
- Direction: overhead diffuse panel or multiple ambient sources to minimize hard shadows.
- Shadow depth: no shadow darker than 30% of ambient (shadow/ambient luminance ratio > 0.7).

**Allowed variation window:** Small object position jitter (±3 cm), robot initial joint configuration within nominal range.

**Scorer decoupling note:** In E0, scorer camera is not decoupled from policy view. Perturbation in E0 is minimal, so scorer and policy camera see effectively the same scene. This is the baseline for scorer validation.

---

### E1 — Hard-Shadow / Low-Angle Illumination Condition

**Rationale:** Simulates lunar south-pole illumination, where the sun angle is 1.5-3 degrees above the horizon, producing long, sharp shadows that can obscure task-relevant object features.

**Physical setup:**
- Primary light source: single collimated LED panel or directional spot (simulating low sun angle).
- Incident angle: 10-20 degrees above horizontal (adjustable; 15 degrees is canonical).
- Panel distance: 1.5-2.5 m from workspace to simulate parallel light rays.
- Secondary illumination: none or very dim ambient (< 50 lux) to preserve high shadow contrast.

**Lighting parameters:**
- Primary source intensity: 800-1200 lux at object surface (direct side).
- Shadow luminance: < 100 lux (shadow/ambient ratio < 0.1 in shadowed regions).
- Color temperature: 5500-6500K (cool white, simulating sunlight color temperature).

**Camera preset:** `e1_shadow` preset. Fixed aperture and exposure time. Do not auto-expose during a trial.

**Allowed variation window:** Shadow direction allowed ±5 degrees rotation around vertical axis across sessions. Panel distance fixed ±5 cm.

**Scorer decoupling:** The scorer camera must be placed to avoid the shadow pattern cast by the robot arm and gripper. Scorer uses a controlled overhead view with separate diffuse lighting. The scoring signal must not degrade under E1 conditions even when the policy-view image does.

**Failure modes to watch:** Gripper occlusion by its own shadow, object features disappearing into shadow, sharp shadow edges causing optical-flow-like artifacts.

---

### E2 — Partial Contamination / Occlusion / Appearance Perturbation

**Rationale:** Simulates lunar dust accumulation, surface debris, and partial occlusion of task-relevant objects. Appearance of objects and workspace surfaces is perturbed without changing the underlying task geometry.

**Physical setup:**
- Workspace surface: apply controlled simulated dust layer (fine gray powder or matte spray coating on removable surface tiles).
- Object contamination: light powder coating on object surfaces that changes texture and reflectance without affecting geometry.
- Partial occlusion: fixed small props (2-4 cm rocks or rubble-like objects) placed at defined positions outside the direct manipulation zone but within the camera field.
- Lighting: same as E0 (nominal diffuse), so that visual perturbation is exclusively from surface/texture changes, not illumination.

**Allowed variation window:** Dust density: two canonical levels (light contamination, heavy contamination). Occlusion props: fixed positions per session (reproducible setup). Object powder coating: reapplied every N sessions per maintenance protocol.

**Camera preset:** `e2_contamination` preset. Same as E0 nominal preset to isolate texture/appearance change from exposure change.

**Scorer decoupling:** Scorer camera must be verified not to rely on surface color or texture in regions affected by contamination. If the scorer uses pixel-difference signals (e.g. for cleaning task), the scoring region must be explicitly excluded from contaminated surface tiles. Scorer validation required before E2 experiments begin.

**Failure modes to watch:** Policy distracted by occlusion props, texture-matching failures when object appearance changes, cleaning task scorer confounded by dust on scorer surface.

---

### Cross-Condition Notes

- Every data collection session must log `meta.env_id` in the episode schema (`E0`, `E1`, or `E2`).
- `meta.camera_preset_id` must match the canonical preset for the declared env.
- Mixing conditions within a session is allowed only for domain-randomization baselines; factorized training requires clean per-session env labeling.
- Scorer camera is always decoupled from the policy camera for E1 and E2.

---

## 9. Dependencies

### 9.1 From Paper A Infrastructure (Phase B)

These must be complete and stable before LunarCompose work begins:

| Module | What LunarCompose uses |
|---|---|
| `task_adapter_bank.py` | `EnvAdapterBank` follows the same interface and checkpoint format |
| `router.py` (`TaskRouter`) | `DualHeadRouter` extends this with a second head |
| `continual_harness.py` | `MissingCornerHarness` is a parallel evaluation harness, not a fork |
| `scorer_base.py` | Same scorer interface; env-specific scorer calibration added |

### 9.2 From Shared Infrastructure (Phase A)

| Module | What LunarCompose uses |
|---|---|
| `episode_schema.py` | `meta.env_id` field — must already be in schema |
| `action_transforms.py` | Same action convention, no change needed |
| `rm75_policy.py` | Same `RM75Inputs` / `RM75Outputs`, no change needed |

### 9.3 From openpi Core

| Module | Usage |
|---|---|
| `models/lora.py` | LoRA injection for both task and env adapters |
| `training/config.py` | Config registration via `_CONFIGS` splat |
| `models/pi05.py`, `models/siglip.py` | Backbone and vision encoder (target of env-path LoRA) |
| `transforms.py` | Data transform system, unchanged |

### 9.4 New in LunarCompose Only

| Module | Depends on |
|---|---|
| `env_adapter_bank.py` | `task_adapter_bank.py`, `models/lora.py` |
| `dual_head_router.py` | `router.py`, shared feature extraction |
| `missing_corner_harness.py` | `env_adapter_bank.py`, `task_adapter_bank.py`, `scorer_base.py` |
| `factorization_diagnostics.py` | `MissingCornerResult`, `DualHeadRouter`, both adapter banks |
| `config.py` | All of the above |

---

## 10. Failure Mode Table

This table is maintained in sync with the master Implementation Masterplan. LunarCompose-specific rows:

| Failure | Root cause class | Immediate action | Claim downgrade |
|---|---|---|---|
| Visual-path env adapter blocked | Engineering dependency on SigLIP LoRA injection | Switch to env-prefix fallback immediately, document clearly | Weaken "vision-path separation" claim; factorization experiment still valid |
| Env scorer confounded by lighting (E1) | Evaluation confound | Decouple scorer camera, redesign scoring signal | Do not proceed until fixed; no partial workaround accepted |
| Task-env leakage in missing-corner split | Data hygiene | Rebuild split and revalidate; invalidate affected runs | None if caught early |
| Dual-head router entangled | Optimization / interference | Separate head training signals, increase independence penalty | Weaken H2 (routing controllability) but not H1 |
| Seen/unseen gap too large to report as success | Factorization assumption failed | Report honestly as finding; do not suppress | Reframe as "factorization assumption is falsified in this setting" |

---

## 11. Acceptance Criteria

The LunarCompose extension is ready for mainline experiments when all of the following pass:

**Module-level (Gate G3):**
- [ ] All test files pass: `uv run pytest src/openpi/research/lunarcompose/ -x -q`
- [ ] `lunarcompose_debug` config instantiates without GPU
- [ ] `EnvAdapterBank` creates, registers, retrieves, and merges env adapters
- [ ] `DualHeadRouter` produces two independent softmax distributions on dummy input
- [ ] `MissingCornerHarness.generate_split(rotation=0)` produces a valid split (no leakage, full coverage)
- [ ] `factorization_diagnostics.seen_unseen_gap` computes correctly on synthetic `MissingCornerResult`

**Environment-level (real robot, pre-experiment):**
- [ ] E0 scorer validated against manual labels (precision/recall logged)
- [ ] E1 shadow setup reproduces within allowed variation window across sessions
- [ ] E2 contamination setup is resettable per maintenance protocol
- [ ] Scorer camera decoupled for E1 and E2 (scorer does not degrade under env perturbation)

**Data pipeline:**
- [ ] `meta.env_id` is logged in all collected episodes
- [ ] `meta.camera_preset_id` matches env for all episodes
- [ ] Missing-corner split tooling verified on real collected data (no scene-instance leakage)

---

## 12. Reference Docs

- `customized_docs/Research_Idea_Blueprint_SpaceCIL_LunarCompose.md` — Section 4 (Paper B scientific claims, hypotheses, claim boundary)
- `customized_docs/Implementation_Masterplan_SpaceCIL_LunarCompose.md` — Sections 4.6, 4.7, 4.8, 5.2, 5.3, 9.3, 11
- `projects/shared/PLAN.md` — Shared infrastructure (episode schema, action transforms, RM75 policy)
- `projects/spacecil/PLAN.md` — Paper A infrastructure (TaskAdapterBank, TaskRouter, ContinualHarness, scorer base)
- `src/openpi/research/lunarcompose/` — Source code (to be implemented per this plan)
