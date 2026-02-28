# Master Plan: SpaceCIL + LunarCompose

Two-paper research program built on the openpi VLA codebase. This document is the single source of truth for implementation strategy, build order, module registry, and failure handling.

---

## 1. Program Overview

### The Two Papers

**Paper A: SpaceCIL** studies whether a released pi0.5 VLA backbone can be continually specialized as new operational tasks arrive sequentially on a real mobile manipulation platform. The core questions: can task-specialized PEFT modules prevent forgetting? Can a lightweight router select the right adapter without knowing the task ID at test time?

**Paper B: LunarCompose** extends the same infrastructure to study factorized adaptation along two independent axes: task and environment. It tests whether factorized adaptation improves generalization to unseen task-environment combinations, using a missing-corner evaluation protocol on a real robot.

These papers are not two weak slices of the same result. They address orthogonal scientific questions: SpaceCIL is about task-axis capability expansion over time; LunarCompose is about environment-axis generalization across deployment conditions.

### Platform

- Wheeled base + RM75 7-DoF arm + two-finger gripper
- Primary policy camera: wrist RGB (hand-eye calibrated)
- Optional scene cameras for scoring, replay, and audit
- Main policy action space: Delta EE + gripper (6D + 1D)
- Base motion is available but the mainline policy is arm-centric unless explicitly extended

### Backbone

Released pi0.5 via the openpi flow-matching policy path. We are not building a VLA from scratch. The correct approach is: reuse first, patch minimally, extend only when needed, and downgrade claims immediately if an extension is blocked.

---

## 2. Architecture Summary

### Code Placement

Research code lives inside the openpi package at `src/openpi/research/`. This gives us standard Python imports (`from openpi.research.spacecil import task_adapter_bank`), automatic inclusion in `uv sync`, and `conftest.py` coverage for JAX CPU fallback on test runs. No `sys.path` hacks.

Plans and documentation live in `projects/`. Source code never goes there.

### Config Integration

New training configs register via the polaris_config.py pattern: one-line splats added at the end of `_CONFIGS` in `src/openpi/training/config.py`:

```python
*spacecil_config.get_spacecil_configs(),
*lunarcompose_config.get_lunarcompose_configs(),
```

Config names must be globally unique. Namespace: `spacecil_*` and `lunarcompose_*`.

### PEFT / LoRA

Task adapter bank wraps openpi's existing LoRA infrastructure in `models/lora.py`. Each task adapter is a separate LoRA parameter set (filtered as `nnx.State` on `.*lora.*`). The bank is a dictionary mapping `task_id` to `lora_state_dict`. Swapping uses `nnx.update(model, lora_state)` before forward passes. `freeze_filter` handles frozen-old / train-new semantics.

### Training Scripts

Custom training scripts import openpi's `init_train_state()` and `train_step()` rather than forking them. The scripts add custom outer loops for task sequencing, distillation, and router training. JIT-compiled `train_step` handles gradient computation. Adapter swapping happens outside JIT boundaries to avoid recompilation.

### Data Pipeline

The pipeline works inside openpi's existing transform system:

- Episode schema is implemented as `RepackTransform`
- Action transforms follow the `DataTransformFn` pattern (like `DeltaActions`)
- RM75 policy follows the `RM75Inputs` + `RM75Outputs` + `LeRobotRM75DataConfig` pattern (like `libero_policy.py`)

### Tests

Tests are co-located as `*_test.py` files next to their source. The `conftest.py` at `src/openpi/conftest.py` handles JAX CPU fallback. Use the `debug` config pattern with `paligemma_variant="dummy"` for fast test models. Test paths: `["src", "scripts", "packages"]`.

---

## 3. Build Phases

### Phase A: Shared Infrastructure

Must be completed before any continual-learning experiments. No training runs happen until G1 is passed.

- Freeze the unified episode schema
- Freeze the Delta EE + gripper action convention
- Implement action transforms
- Implement RM75 policy config
- Implement scorer base and per-task scorers
- Validate wrist-camera semantics end-to-end
- Complete manual-vs-auto scorer audit

### Phase B: SpaceCIL Core

Builds on Phase A. Requires G1 before starting.

- Task adapter bank with per-task LoRA sets
- Language-visual router (no oracle task ID)
- Behavior distillation anti-forgetting
- Continual evaluation harness
- Baseline pipeline (sequential fine-tuning, shared PEFT, oracle routing)
- Checkpoint registry and adapter restore
- Plotting and reporting

### Phase C: LunarCompose Extension

Builds on Phase B. Requires G2 before starting.

- Enforce environment metadata from collection onward
- Environment adapter bank (visual path preferred, env-prefix fallback)
- Dual-head router (separate task and environment heads)
- Missing-corner compositional evaluation harness
- Factorization diagnostics
- Activate fallback immediately if visual-path env adaptation is blocked

---

## 4. Gate Criteria

### G1: One-task sanity

**Required before:** any continual-learning experiments

- One task trains and replays correctly
- Wrist-camera policy path works end-to-end
- Scorer agrees with manual labels on a pilot subset

### G2: SpaceCIL readiness

**Required before:** Paper A mainline claims

- At least two tasks train sequentially without failure
- Router beats random routing and a weak baseline
- Behavior distillation path executes stably
- Adapter registry and checkpoint restore are reliable

### G3: LunarCompose readiness

**Required before:** Paper B mainline claims

- Environment metadata are enforced without leakage into the training split
- Missing-corner split tooling is verified
- An environment adaptation path exists (visual or fallback)
- If visual-path env adaptation is blocked, fallback is activated immediately

---

## 5. Module Registry

| Module | Location | Classification | Phase | Dependencies |
|---|---|---|---|---|
| Episode schema | `research/shared/episode_schema.py` | new | A | none |
| Action transforms | `research/shared/action_transforms.py` | new | A | openpi transforms |
| RM75 policy | `research/shared/rm75_policy.py` | new | A | openpi policies / libero pattern |
| Scorer base | `research/shared/scorer_base.py` | new | A | none |
| Task adapter bank | `research/spacecil/task_adapter_bank.py` | new | B | openpi LoRA |
| Router | `research/spacecil/router.py` | new | B | task adapter bank |
| Behavior distillation | `research/spacecil/behavior_distillation.py` | new | B | task adapter bank |
| Continual harness | `research/spacecil/continual_harness.py` | new | B | all SpaceCIL modules |
| SpaceCIL metrics | `research/spacecil/metrics.py` | new | B | none |
| SpaceCIL config | `research/spacecil/config.py` | new | B | RM75 policy |
| Env adapter bank | `research/lunarcompose/env_adapter_bank.py` | new | C | task adapter bank |
| Dual-head router | `research/lunarcompose/dual_head_router.py` | new | C | router |
| Missing-corner harness | `research/lunarcompose/missing_corner_harness.py` | new | C | continual harness |
| Factorization diagnostics | `research/lunarcompose/factorization_diagnostics.py` | new | C | dual-head router |
| LunarCompose config | `research/lunarcompose/config.py` | new | C | RM75 policy |
| openpi config.py | `src/openpi/training/config.py` | patch lightly | B/C | existing _CONFIGS list |
| SpaceCIL training script | `scripts/train_spacecil.py` | new | B | openpi train functions |
| LunarCompose training script | `scripts/train_lunarcompose.py` | new | C | openpi train functions |

**Classification key:**
- **reuse as-is**: use openpi module directly, no changes
- **patch lightly**: minimal modification (e.g., adding a config splat line)
- **new**: new code in `src/openpi/research/`

---

## 6. Failure Mode Table

| Failure | Affected paper | Root cause class | Immediate action | Claim downgrade |
|---|---|---|---|---|
| Router old-task collapse | A | optimization / interference | freeze trunk, inspect router expansion | none or weaker routing claim |
| Behavior distillation unstable | A | loss weighting / decoding mismatch | reduce lambda, inspect target format | weaker anti-forgetting claim |
| Visual env adapter blocked | B | engineering dependency | switch to env-prefix fallback | weaken vision-path separation claim |
| Env scorer confounded by lighting | B | evaluation confound | decouple scorer view or redesign signal | do not proceed until fixed |
| Task-env leakage in split | B | data hygiene | rebuild split | invalidate affected run |

The most critical rule: never suppress or defer a failure. Downgrade the claim immediately and document the reason. Every experiment run must save the config hash, adapter versions, code commit, scene revision, calibration version, initial-state seed, scorer outputs, raw video references, fail type, and which canonical split it belongs to.

---

## 7. Directory Map

```
src/openpi/research/              # All research source code
    shared/                       # Phase A: shared infrastructure
    spacecil/                     # Phase B: SpaceCIL modules
    lunarcompose/                 # Phase C: LunarCompose modules

projects/                         # Documentation only (no source code)
    PLAN.md                       # This file
    shared/PLAN.md                # Shared infrastructure detail
    spacecil/PLAN.md              # SpaceCIL detail
    lunarcompose/PLAN.md          # LunarCompose detail

scripts/
    train_spacecil.py             # Phase B custom training script
    train_lunarcompose.py         # Phase C custom training script
```

---

## 8. Anti-Patterns

These apply to all coding agents and sessions:

- Do not rewrite openpi core from scratch. Reuse first, patch minimally.
- Do not ask an agent to "implement SpaceCIL" as a single task. Always decompose into verifiable modules.
- Do not start Phase C before Phase A infrastructure is stable.
- Do not maintain duplicate schemas for different papers.
- Do not maintain two inconsistent action converters.
- Do not suppress type errors.
- Do not modify the `_CONFIGS` list beyond adding splat entries for new config modules.
- Do not let the visual-path env adapter block Paper B without activating the fallback.
