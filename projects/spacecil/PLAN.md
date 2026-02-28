# SpaceCIL — Implementation Plan
## Paper A: Continual Skill Acquisition on a Real Mobile Manipulation Platform

---

## 1. Paper Summary

SpaceCIL studies whether a released pi0.5 VLA backbone can be **continually specialized** on a real mobile manipulation platform as new operational tasks arrive sequentially. The platform is a wheeled base with an RM75 7-DoF arm and two-finger gripper. The primary policy camera is a wrist RGB (hand-eye calibrated). The action space is Delta EE + gripper.

The paper focuses on three core problems:

1. **Parameter-efficient task expansion.** Each new task gets its own PEFT module rather than retraining the full policy. The backbone stays frozen; only task-specific LoRA weights grow.
2. **Router-based task selection.** At inference, no oracle task ID is available. A lightweight language-visual router must select the right adapter from the bank.
3. **Mission-aware forgetting analysis.** Standard CL metrics treat all tasks equally. Operationally weighted forgetting reveals whether the tasks that matter most are the ones degrading.

### Hypotheses

- **H1**: Task-specialized PEFT modules outperform sequential full fine-tuning and a single shared PEFT module in the continual setting.
- **H2**: A lightweight language-visual router is sufficient to select task-specialized capability without test-time oracle task identity.
- **H3**: A small anti-forgetting objective based on prior-task calibration trajectories significantly reduces mission-relevant forgetting.
- **H4**: An operationally weighted forgetting metric reveals degradation patterns not exposed by uniform CL metrics alone.

### Claim boundary

> "To our knowledge, this is the first real-robot study of continual specialization of a released pi0.5-class VLA on a wheeled mobile manipulation platform under an operationally ordered lunar-analog task stream."

The paper does **not** claim unqualified firstness on "continual VLA" as a whole.

---

## 2. Module: `config.py`

**File:** `src/openpi/research/spacecil/config.py`

**Classification:** new module

### Purpose

Expose `get_spacecil_configs() -> list[TrainConfig]` following the polaris_config.py pattern. This function returns all SpaceCIL training configs, which are then splat into `_CONFIGS` in `src/openpi/training/config.py` with a single line.

### Config naming convention

All config names are globally unique and prefixed `spacecil_`:

| Config name | Task |
|---|---|
| `spacecil_rm75_payload` | Payload transfer / unloading |
| `spacecil_rm75_latch` | Latch or lever actuation |
| `spacecil_rm75_clean` | Surface cleaning / wiping |
| `spacecil_rm75_connector` | Connector mating / insertion |
| `spacecil_debug` | Fast test config (dummy model) |

### Per-config design

Each `spacecil_rm75_*` config specifies:

- **Model:** pi0.5 with LoRA adaptation (via `models/lora.py`). The backbone is frozen; only LoRA params are trainable.
- **Data config:** `LeRobotRM75DataConfig` pointing to the task-specific LeRobot dataset.
- **freeze_filter:** Freeze all non-LoRA parameters. This is standard openpi LoRA practice.
- **Weight loader:** Points to the released pi0.5 base checkpoint (`gs://openpi-assets/checkpoints/pi05_base`).

The `spacecil_debug` config uses:

- `paligemma_variant="dummy"` and a minimal model spec for fast instantiation.
- Tiny batch sizes and short horizon.
- Enough structure to exercise all code paths without downloading real checkpoints.

### Integration

Add one line at the end of `_CONFIGS` in `src/openpi/training/config.py`:

```python
*spacecil_config.get_spacecil_configs(),
```

That is the **only** modification to openpi core for config registration.

### Acceptance criteria

- `get_config("spacecil_debug")` instantiates without errors.
- All `spacecil_rm75_*` configs have unique names and correct model/data specs.
- Splat line integrates without touching existing config entries.

---

## 3. Module: `task_adapter_bank.py`

**File:** `src/openpi/research/spacecil/task_adapter_bank.py`

**Classification:** new module

### Purpose

The `TaskAdapterBank` is a versioned registry mapping task IDs to their LoRA parameter states. It handles:

- Registration of new task adapters after training completes.
- Application of a stored adapter to the live model before inference or evaluation.
- Freezing old adapters so they can't be modified.
- Checkpointing and restoring the full bank.

### Key class: `TaskAdapterBank`

```python
# Pseudocode sketch — not runnable
class TaskAdapterBank:
    adapters: dict[str, nnx.State]   # task_id -> LoRA param state
    frozen: set[str]                  # task_ids that are frozen

    def register_adapter(task_id: str, lora_state: nnx.State) -> None
    def get_adapter(task_id: str) -> nnx.State
    def merge_into_model(model, task_id: str) -> None   # nnx.update(model, lora_state)
    def freeze_adapter(task_id: str) -> None
    def save(path: str) -> None
    def load(path: str) -> None
```

### Design details

**Adapter representation.** Each adapter is an `nnx.State` filtered to parameters matching `.*lora.*`. This is consistent with how openpi's `models/lora.py` separates LoRA weights from frozen backbone weights.

**Frozen-old / train-new semantics.** When task N is complete:

1. `freeze_adapter(task_N_id)` marks it read-only in the bank.
2. `TrainConfig.freeze_filter` already freezes non-LoRA params during training.
3. Only the new task's LoRA params are part of the active optimizer state.

This means the adapter bank is the source of truth for old task weights, and the active model holds only the current task's LoRA.

**Applying an adapter.** `merge_into_model` calls `nnx.update(model, lora_state)`, which overwrites the model's LoRA params with the stored state. This happens outside JIT boundaries to avoid recompilation.

**Checkpointing.** Use orbax `PyTreeCheckpointer`. One subdirectory per task adapter:

```
checkpoints/adapter_bank/
  payload/
    state/
  latch/
    state/
  ...
```

Metadata (frozen set, registration order) is saved alongside as a JSON sidecar.

### Tests (`task_adapter_bank_test.py`)

- Create a dummy LoRA state and register it.
- Retrieve by task_id and verify it matches.
- Freeze a task, attempt write, expect error.
- Save bank to temp dir, reload, verify round-trip equality.
- Merge adapter into a small dummy model, verify param values change.

### Acceptance criteria

- Register / retrieve round-trips correctly.
- Frozen adapters cannot be overwritten.
- Save / load round-trips without shape or value errors.
- `merge_into_model` updates the model's LoRA params correctly.

---

## 4. Module: `router.py`

**File:** `src/openpi/research/spacecil/router.py`

**Classification:** new module

### Purpose

The `TaskRouter` selects which task adapter to use at inference time without receiving an oracle task ID. It runs on features already computed by the backbone, adding negligible compute overhead.

### Key class: `TaskRouter`

```python
# Pseudocode sketch
class TaskRouter(nnx.Module):
    mlp: small MLP (2-3 layers, ~256 hidden units)
    output_head: Linear(hidden_dim, num_tasks)

    def __call__(
        lang_embedding: jax.Array,    # from backbone text encoder
        visual_summary: jax.Array,    # from backbone vision encoder (pooled)
    ) -> jax.Array:                   # softmax routing weights over task adapters
```

**Input.** Language embedding and visual summary are extracted from the backbone's intermediate representations (not re-encoded; the backbone runs once and features are reused).

**Output.** A softmax distribution over registered adapters. At inference, either argmax (hard routing) or the full distribution (soft routing) is used.

**Architecture.** A small MLP on the concatenation `[lang_embedding, visual_summary]`. Two or three hidden layers with LayerNorm and GELU activation. The output head is a linear layer to `num_tasks`.

### Growing adapter bank

The router's output head must expand when a new task is registered. Design options:

- **Option A (preferred for simplicity):** Allocate the output head for `max_tasks` at init time. Inactive logits are masked out during routing.
- **Option B:** Dynamically replace the output head. Requires re-initializing and transferring weights, which complicates JIT compilation.

Option A is simpler and compatible with JIT. The mask is stored alongside the bank and updated when a new task is registered.

### Training

The router is trained jointly with the current task's LoRA adapter. During task N training:

- The router receives a supervision signal: route to task N.
- Old task routing knowledge is preserved because the router loss only pushes toward task N on task N data; old task data (from calibration memory) can provide complementary supervision if desired.

### Inference modes

- **Hard routing (argmax):** Pick the adapter with highest routing probability.
- **Soft routing:** Interpolate adapter outputs weighted by routing probabilities. Only feasible if adapter outputs can be cheaply combined (e.g., action prediction).

### Tests (`router_test.py`)

- Forward pass with fixed lang + visual inputs; output shape matches `[batch, num_tasks]`.
- Routing probabilities sum to 1.
- With a 1-task bank, argmax always returns task 0.
- With 3-task bank, routing entropy is between 0 and log(3).
- Masked logits for unregistered tasks do not influence argmax.

### Acceptance criteria

- Forward pass shape is correct for 1, 2, and 4 registered tasks.
- Routing probabilities are valid distributions.
- Masking works correctly when fewer tasks are registered than `max_tasks`.

---

## 5. Module: `behavior_distillation.py`

**File:** `src/openpi/research/spacecil/behavior_distillation.py`

**Classification:** new module

### Purpose

Anti-forgetting via a teacher-snapshot + calibration-memory approach. When training on task N, a frozen copy of the policy (with all previously trained adapters) acts as a teacher. A small buffer of calibration episodes from previous tasks provides data for the distillation loss.

### Key class: `BehaviorDistillation`

```python
# Pseudocode sketch
class BehaviorDistillation:
    teacher_model: frozen policy snapshot   # state before current task training
    calibration_memory: EpisodeBuffer       # episodes from tasks 1..N-1
    distillation_weight: float              # lambda

    def distillation_loss(
        student_output,     # model output on calibration batch
        teacher_output,     # teacher output on same batch (no grad)
    ) -> jax.Array:         # scalar loss

    def sample_memory(batch_size: int) -> Batch
    def update_teacher(model) -> None        # snapshot current model state
    def add_calibration_episodes(episodes) -> None
```

**Teacher.** The teacher is created by copying the model's full parameter state (backbone + all registered LoRA adapters) at the end of task N-1 training. It is permanently frozen and lives on CPU or a secondary device if memory is tight.

**Calibration memory.** A small episode buffer, one per previous task. Capacity options:

| Memory size | Episodes per task |
|---|---|
| Small | 50 |
| Medium | 100 |
| Large | 200 |

Episodes are sampled uniformly across tasks (or weighted by operational importance).

**Distillation loss.** Two variants:

- **Action-space distillation:** MSE between student and teacher action predictions. Simple and interpretable. Suitable as default.
- **Latent-space distillation:** KL divergence between student and teacher latent distributions (flow matching intermediate representations). Richer signal but more fragile.

Action-space distillation is the default; latent-space is an ablation.

**Total training loss:**

```
L_total = L_task + lambda * L_distill
```

where `L_task` is the standard flow-matching loss on the current task data, and `L_distill` is computed on the calibration memory batch.

### Planned ablations

| Variable | Values |
|---|---|
| Memory size | 50, 100, 200 episodes/task |
| Lambda | 0.1, 0.5, 1.0 |
| Distillation space | action-space, latent-space |

### Tests (`behavior_distillation_test.py`)

- `distillation_loss` returns a scalar with correct shape.
- Teacher params do not change after gradient updates on student.
- Memory sampling returns correct batch size.
- With lambda=0, total loss equals task loss exactly.
- Adding calibration episodes increases memory size.

### Acceptance criteria

- Loss computation is numerically stable across lambda values.
- Teacher is verifiably frozen (no gradient flow).
- Memory round-trips through add/sample correctly.

---

## 6. Module: `continual_harness.py`

**File:** `src/openpi/research/spacecil/continual_harness.py`

**Classification:** new module

### Purpose

The `ContinualHarness` orchestrates the full sequential task training loop and backward transfer evaluation. It is the top-level entry point for running a SpaceCIL experiment.

### Key class: `ContinualHarness`

```python
# Pseudocode sketch
class ContinualHarness:
    task_sequence: list[str]                  # ordered task IDs
    adapter_bank: TaskAdapterBank
    router: TaskRouter
    distillation: BehaviorDistillation
    scorer: dict[str, ScorerBase]

    def train_task(task_id: str) -> None
    def evaluate_all_tasks() -> dict[str, float]
    def run_sequence() -> ContinualResult
```

### `run_sequence` logic

```
for task_id in task_sequence:
    1. Load task data config
    2. Activate new LoRA adapter for task_id
    3. Train adapter (+ router, + distillation if applicable)
    4. Register adapter in bank, freeze it
    5. Update teacher snapshot
    6. evaluate_all_tasks() -> record result matrix
```

### `ContinualResult` dataclass

```python
@dataclass
class ContinualResult:
    task_sequence: list[str]
    # R[i][j] = success rate on task j after training task i
    result_matrix: np.ndarray               # shape: [num_tasks, num_tasks]
    backward_transfer_matrix: np.ndarray    # shape: [num_tasks, num_tasks]
    forgetting_matrix: np.ndarray           # shape: [num_tasks, num_tasks]
    final_success_rates: dict[str, float]   # success after full sequence
```

### Integration with existing modules

The harness imports and composes:

- `TaskAdapterBank` for adapter lifecycle.
- `TaskRouter` for routing setup and training.
- `BehaviorDistillation` for anti-forgetting.
- `ScorerBase` subclasses from `scorer_base.py` for task success evaluation.
- openpi's `init_train_state()` and `train_step()` from `scripts/train.py` for the inner gradient loop.

Adapter swapping happens **outside JIT boundaries** to avoid recompilation. The JIT-compiled `train_step` function sees a model with already-swapped LoRA weights.

### Tests (`continual_harness_test.py`)

- Mock 2-task sequence with toy data; `run_sequence` completes without errors.
- `result_matrix` shape is `[num_tasks, num_tasks]`.
- `final_success_rates` contains all task IDs.
- Backward transfer is computed correctly on a known result matrix.
- Distillation is called the correct number of times per training step.

### Acceptance criteria

- 2-task mock sequence runs end-to-end without errors.
- Result matrix has correct shape and value structure.
- Backward transfer calculation matches hand-computed values on known inputs.

---

## 7. Module: `metrics.py`

**File:** `src/openpi/research/spacecil/metrics.py`

**Classification:** new module

### Purpose

Compute CL metrics and SpaceCIL-specific mission-aware metrics from a `ContinualResult`. All functions are pure (no side effects) and can be tested with synthetic data.

### Metric definitions

Let `R[i][j]` be the success rate on task `j` evaluated after training task `i` (from `ContinualResult.result_matrix`).

**Standard CL metrics:**

| Function | Formula | Notes |
|---|---|---|
| `average_success(result)` | `mean(R[-1, :])` | Mean success on last evaluation |
| `backward_transfer(result)` | `mean(R[j][j] - R[-1][j] for j < T)` | How much old task perf changes after later training |
| `forgetting(result)` | `mean(max_k R[k][j] - R[-1][j] for j < T)` | Drop from peak performance |

**Mission-aware metric:**

```
operational_forgetting(result, weights) =
    sum_j( weights[j] * (max_k R[k][j] - R[-1][j]) ) / sum_j(weights[j])
```

Operational weights are configurable per task. Example defaults:

| Task | Weight |
|---|---|
| connector_mating | 1.0 |
| latch_actuation | 0.8 |
| payload_transfer | 0.6 |
| surface_cleaning | 0.5 |

Weights are passed explicitly; they are not hardcoded in the function. The caller (usually `ContinualHarness`) provides them from config.

**Router diagnostics:**

| Function | Definition |
|---|---|
| `routing_entropy(routing_probs)` | `-sum_k p_k * log(p_k)` |
| `routing_accuracy(predicted_task, true_task)` | `mean(predicted == true)` |

High routing entropy means the router is uncertain. Low routing accuracy means it's picking the wrong adapter. Both are diagnostic signals, not primary claims.

### Tests (`metrics_test.py`)

- Perfect performance matrix (`R[i][j] = 1.0 for all i >= j`): forgetting = 0, backward transfer = 0.
- Zero final performance: forgetting equals max historical performance.
- Known 2x2 result matrix: verify each metric by hand.
- Weights that are all equal: `operational_forgetting` equals `forgetting`.
- Routing entropy of uniform distribution equals `log(num_tasks)`.
- Routing entropy of one-hot distribution equals 0.

### Acceptance criteria

- All metrics produce correct values on synthetic data.
- Weight normalization is handled correctly (zero-weight edge case).
- Entropy and accuracy functions handle batched inputs.

---

## 8. Baseline Configuration

Each baseline is a config variant in `config.py`, implemented by adjusting the training setup rather than writing new modules where possible.

| Baseline | Config name | Implementation |
|---|---|---|
| Sequential full fine-tuning | `spacecil_rm75_<task>_fulltune` | No freeze_filter; full param update |
| Shared multi-task PEFT | `spacecil_rm75_shared_lora` | Single LoRA for all tasks; no adapter bank |
| Per-task PEFT with oracle task ID | `spacecil_rm75_<task>_oracle` | Adapter bank but router replaced by ground-truth task label |
| SpaceCIL without distillation | `spacecil_rm75_<task>_nodistill` | lambda=0 in BehaviorDistillation |
| SpaceCIL with random routing | `spacecil_rm75_<task>_randrout` | Router replaced by uniform random adapter selection |

The oracle baseline is the performance upper bound. The random routing ablation tests whether routing matters or whether adapter-per-task alone is sufficient.

---

## 9. Task Suite (Benchmark)

Four lunar-analog proxy tasks. Each is operationally relevant to lunar infrastructure and semantically adjacent to terrestrial manipulation primitives.

### Task 1: Payload Transfer / Unloading

**Operational analog:** Moving surface cargo between storage and deployment sites.

**Terrestrial primitive:** Pick and place.

| Spec field | Detail |
|---|---|
| Object geometry | Rectangular payload block, ~20x10x5 cm, flat grasp surfaces |
| Goal state | Payload resting in target bin or marked zone, contact confirmed |
| Scorer | Top-down binary occupancy check in target zone |
| Reset procedure | Operator places payload at randomized start position within defined range |
| Demo collection | Wrist-camera teleoperation; grasp then carry then release |
| Failure modes | Dropped payload, missed grasp, incorrect placement zone |

### Task 2: Latch or Lever Actuation

**Operational analog:** Actuating mechanical interfaces on habitat modules, panels, or equipment racks.

**Terrestrial primitive:** Lever push / toggle pull.

| Spec field | Detail |
|---|---|
| Object geometry | Spring-loaded lever with ~5 cm travel, 2-position state |
| Goal state | Lever in target position (on/off), confirmed by switch sensor or visual |
| Scorer | Binary state check via hall-effect sensor or color marker |
| Reset procedure | Operator returns lever to start state; no arm intervention needed |
| Demo collection | Wrist-camera teleoperation; approach, contact, actuate |
| Failure modes | Missed lever, insufficient force, overtravel |

### Task 3: Surface Cleaning / Wiping

**Operational analog:** Dust mitigation on sensor surfaces, solar panels, or habitat windows.

**Terrestrial primitive:** Wipe or scrub.

| Spec field | Detail |
|---|---|
| Object geometry | Flat 20x20 cm contaminated surface panel; wiper attached to gripper |
| Goal state | >80% surface area covered by wiper path |
| Scorer | Pixel-difference scorer on scorer camera (lighting-controlled, decoupled from policy view) |
| Reset procedure | Operator applies powder or marker to surface |
| Demo collection | Wrist-camera teleoperation; contact surface, execute wipe pattern |
| Failure modes | Lifted wiper, incomplete coverage, missed surface |

**Scorer note:** The scorer camera must be lighting-controlled and isolated from any environment perturbation applied to the policy camera. A pixel-difference scorer that sees the same perturbation as the policy view is a confound and must be rejected.

### Task 4: Connector Mating / Insertion

**Operational analog:** Utility connector mating for power, data, or fluid lines.

**Terrestrial primitive:** Peg-in-hole insertion.

**Operational weight: 1.0 (highest).** Connector mating is the highest-stakes task in the sequence due to contact sensitivity and mission criticality.

| Spec field | Detail |
|---|---|
| Object geometry | Circular or rectangular connector, ~1-2 cm insertion depth, 3-5 mm tolerance |
| Goal state | Full insertion confirmed by tactile sensor or visual depth cue |
| Scorer | Binary contact/insertion sensor (preferred) or visual insertion depth classifier |
| Reset procedure | Operator disconnects and repositions connector to start pose |
| Demo collection | Wrist-camera teleoperation; approach, align, insert |
| Failure modes | Misalignment, partial insertion, connector drop |

---

## 10. Dependencies

### Shared infrastructure (must be stable before SpaceCIL core)

| Module | File | Gate |
|---|---|---|
| Episode schema | `shared/episode_schema.py` | G1 |
| Action transforms | `shared/action_transforms.py` | G1 |
| RM75 policy | `shared/rm75_policy.py` | G1 |
| Scorer base | `shared/scorer_base.py` | G1 |

SpaceCIL modules must not be started until Gate G1 is verified: one task trains and replays correctly, the wrist-camera policy path works, and the scorer agrees with manual labels on a pilot subset.

### openpi core (reuse as-is)

| Module | Usage |
|---|---|
| `models/lora.py` | LoRA parameter definitions and freeze_filter |
| `training/config.py` | TrainConfig base class; _CONFIGS registry |
| `transforms.py` | RepackTransform, DeltaActions, DataTransformFn |
| `policies/policy_config.py` | Policy wrapper interface |
| `scripts/train.py` | `init_train_state()`, `train_step()` |

### Gate G2 (SpaceCIL readiness)

Before claiming Paper A mainline results:

- At least 2 tasks train sequentially.
- Router beats random and a weak routing baseline.
- Behavior distillation executes stably without loss divergence.
- Adapter registry and checkpoint restore are reliable across runs.

---

## 11. Acceptance Criteria

The SpaceCIL implementation is considered complete when all of the following hold:

| Criterion | Verification |
|---|---|
| All module tests pass | `uv run pytest src/openpi/research/spacecil/ -x -q` |
| Debug config instantiates | `get_config("spacecil_debug")` runs without error |
| 2-task mock sequence runs | `ContinualHarness` with mock data completes end-to-end |
| Metrics correct on synthetic data | All `metrics_test.py` cases pass with known-value checks |
| Adapter bank save/load round-trips | Save to temp dir, reload, verify param equality |
| Router shape and distribution tests pass | Output shape, softmax sum, masking all verified |
| Distillation loss stable | Non-NaN loss with both action-space and latent-space variants |
| Baseline configs instantiate | Each baseline config name resolves without error |

---

## 12. Build Order

This plan follows Phase B of the master build order. Phase A (shared infra) must be complete before starting here.

```
Phase A [SHARED INFRA — prerequisite]:
  episode_schema.py
  action_transforms.py
  rm75_policy.py
  scorer_base.py
  -> Gate G1

Phase B [SPACECIL CORE — this plan]:
  config.py
  task_adapter_bank.py
  router.py
  behavior_distillation.py
  continual_harness.py
  metrics.py
  scripts/train_spacecil.py
  -> Gate G2

Phase C [LUNARCOMPOSE — next plan]:
  env_adapter_bank.py
  dual_head_router.py
  missing_corner_harness.py
  factorization_diagnostics.py
  -> Gate G3
```

Within Phase B, the recommended implementation order is:

1. `config.py` — unblocks all other modules for instantiation testing.
2. `task_adapter_bank.py` — core data structure; everything depends on it.
3. `metrics.py` — pure functions; can be written and tested independently.
4. `behavior_distillation.py` — no dependency on router; can be developed in parallel with step 3.
5. `router.py` — depends on the bank structure for output head sizing.
6. `continual_harness.py` — integrates all of the above; written last.
7. `scripts/train_spacecil.py` — wraps the harness for command-line execution.

---

## 13. Anti-Patterns to Avoid

- Do **not** rewrite openpi core training logic. Import `init_train_state` and `train_step` directly.
- Do **not** swap adapters inside JIT-compiled functions. Adapter swapping happens outside JIT.
- Do **not** maintain two LoRA state representations. One `nnx.State` filtered to `.*lora.*` is the single source of truth.
- Do **not** hardcode operational weights inside `metrics.py`. Weights are always passed by the caller.
- Do **not** use the scorer camera view for policy training data, or the policy camera view for scorer computation.
- Do **not** start LunarCompose (Phase C) modules until Gate G2 is verified.
- Do **not** unify the continual harness and the missing-corner harness into one class prematurely.
