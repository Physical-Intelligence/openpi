# Research Architecture: SpaceCIL + LunarCompose
## Mission-Driven Reuse of Earth-Trained Embodied Intelligence for Lunar-Analog Mobile Manipulation

---

## 1. Program Thesis

### 1.1 Central Thesis

Recent progress in embodied AI has produced increasingly capable Earth-trained robot foundation models. The scientific opportunity for space robotics is not to build another policy from scratch for each mission, but to develop principled mechanisms that let us **reuse, specialize, and expand** Earth-developed embodied intelligence as space missions evolve.

This program answers one overarching question (Blueprint Section 1.3):

> How can a released Earth-trained VLA backbone remain useful as tasks and deployment conditions move toward more out-of-distribution, space-relevant operating regimes?

### 1.2 Why Two Papers, Not One

The answer requires attacking two orthogonal adaptation problems. Collapsing them into one paper would conflate the evaluation protocol and undermine the falsifiability of each claim independently.

**Paper A (SpaceCIL)** studies the **task axis**: can a released VLA continually acquire new task capability without forgetting old ones? The independent variable is the task stream. Environment conditions are held nominal.

**Paper B (LunarCompose)** studies the **environment axis**: can a VLA separate task semantics from environment-specific corrections, and generalize to unseen task-environment combinations? The independent variable is the environment condition. Tasks remain from the SpaceCIL suite.

### 1.3 Two Orthogonal Adaptation Axes

```
  Two-Axis Adaptation Framework
  ==============================

  Environment axis (LunarCompose)
  ^
  |   E2: Contamination  [ ]  [ ]  [ ]  [ ]
  |   E1: Hard-Shadow    [ ]  [ ]  [ ]  [ ]
  |   E0: Nominal        [ ]  [ ]  [ ]  [ ]
  |                       |    |    |    |
  +------------------------+----+----+----+---> Task axis (SpaceCIL)
                          T1   T2   T3   T4
                        (pay) (lat)(cln)(con)

  SpaceCIL: fix E0, grow T1 -> T2 -> T3 -> T4 sequentially
  LunarCompose: fixed T1..T4, factorize over E0, E1, E2
  The two axes are orthogonal; the two papers are not subsets of one another.
```

### 1.4 Shared Program Thesis

Together the two papers support (Blueprint Section 2.2):

> Space robotics should be formulated as **capability reuse and expansion from Earth-trained embodied intelligence**, rather than repeated from-scratch policy construction.

This is a program-level framing, not a paper-level claim. Neither paper alone carries this weight; both are required.

---

## 2. Platform and Backbone

### 2.1 Robot Platform

```
  RM75 Mobile Manipulation System
  =================================

      ┌─────────────────────────────────┐
      │          WRIST CAMERA           │  <-- primary policy camera
      │          (RGB, hand-eye         │
      │           calibrated)           │
      └────────────────┬────────────────┘
                       │
      ┌────────────────┴────────────────┐
      │        TWO-FINGER GRIPPER       │
      └────────────────┬────────────────┘
                       │
      ┌────────────────┴────────────────┐
      │     RM75 ARM (7-DoF)            │
      │     Joint 1 .. Joint 7          │
      │     Absolute joint position     │
      │     action space (7D)           │
      └────────────────┬────────────────┘
                       │
      ┌────────────────┴────────────────┐
      │         WHEELED BASE            │
      │    (navigation / positioning)   │
      └─────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │ Platform Specifications                                    │
  ├─────────────────────┬──────────────────────────────────────┤
  │ Arm                 │ RM75, 7 Degrees of Freedom            │
  │ Gripper             │ Two-finger, 1D binary/continuous open │
  │ Action space        │ 8D: 7 joint positions + 1 gripper     │
  │ Action convention   │ Absolute joint position (not delta)   │
  │ Primary camera      │ Wrist RGB (hand-eye calibrated)       │
  │ Base                │ Wheeled, used for positioning only     │
  └─────────────────────┴──────────────────────────────────────┘
```

The wrist camera is the sole observation input to the policy during both training and inference. A separate scorer camera is used only for task success evaluation and is always decoupled from the policy view in E1 and E2 conditions (see Section 5).

### 2.2 VLA Backbone: pi0.5

pi0.5 (Black et al. 2025) is a flow-matching vision-language-action model pretrained on 10k+ hours of robot data. It uses three stacked components. The flow-matching head generates actions iteratively by denoising a random noise vector conditioned on vision and language features.

```
  pi0.5 Architecture with LoRA Injection Points
  ===============================================

  wrist_rgb ─────────────────┐
                              v
                   ┌──────────────────────────┐
                   │   SigLIP Vision Encoder   │ <-- Env LoRA injected here (LunarCompose)
                   │   (patch embeddings +      │     targets: ".*siglip.*"
                   │    self-attention blocks)  │
                   └──────────────┬────────────┘
                                  │ visual features
  language instruction ──────┐    │
                              v    v
                   ┌──────────────────────────┐
                   │  Gemma 2B Language Model  │ <-- Task LoRA injected here (SpaceCIL)
                   │   (cross-attn + self-attn)│     targets: ".*lm.*"
                   │   conditions on text +    │
                   │   visual features         │
                   └──────────────┬────────────┘
                                  │ context embeddings
                                  v
                   ┌──────────────────────────┐
                   │  Gemma 300M Action Expert │ <-- Task LoRA injected here (SpaceCIL)
                   │   (generates action       │     targets: ".*action_expert.*"
                   │    tokens from context)   │
                   └──────────────┬────────────┘
                                  │ action tokens
                                  v
                   ┌──────────────────────────┐
                   │  Flow Matching Head       │
                   │   (denoising: noise ->    │
                   │    action trajectory)     │
                   └──────────────┬────────────┘
                                  │
                                  v
                          8D action chunk
                    (7 joint positions + 1 gripper)

  LoRA parameter structure:
    lora_a: (d_model, rank)   -- down-projection
    lora_b: (rank, d_model)   -- up-projection
    Filter: PathRegex(".*lora.*")
    Swap via: nnx.update(model, lora_state)  -- outside JIT
```

LoRA weights are extracted and swapped using Flax NNX state management. The `nnx.update(model, lora_state)` call must happen **outside JIT boundaries** to avoid triggering recompilation, which would be prohibitively expensive.

---

## 3. Paper A: SpaceCIL Architecture

### 3.1 Problem Formulation

SpaceCIL studies whether a released VLA backbone can be **continually specialized** as new operational tasks arrive sequentially (Blueprint Section 3.2):

1. New task performance must improve quickly from a small demonstration set.
2. Previously acquired tasks must degrade minimally as new tasks are added.
3. Inference must not require a test-time oracle task ID.
4. Adaptation must remain parameter-efficient and operationally diagnosable.

The task stream is ordered by operational logic: payload transfer first (logistics), then latch actuation (mechanical interfaces), then surface cleaning (maintenance), then connector mating (utility connections). This ordering is not arbitrary; it mirrors the expected sequence in which lunar surface infrastructure capabilities are brought online.

### 3.2 Full System Architecture

```
  SpaceCIL Full System Architecture
  ====================================

                    ┌──────────────────────────────────────────┐
                    │          pi0.5 Backbone (FROZEN)          │
                    │                                           │
  wrist_rgb ───────>│  ┌───────────┐      ┌──────────────────┐ │
                    │  │  SigLIP   │─────>│  Gemma 2B LM     │ │
                    │  │ (vision)  │      │  + Gemma 300M    │ │
  instruction ────> │  └───────────┘      │  Action Expert   │ │
                    │                     └────────┬─────────┘ │
                    │                    lang_emb  │ vis_feat  │
                    └────────────────────────┬─────┼───────────┘
                                             │     │
                    ┌────────────────────────┼─────┼───────────┐
                    │  Task Adapter Bank      │     │           │
                    │                        │ (features        │
                    │  ┌─────────┐ ┌───────┐ │  reused)        │
                    │  │  LoRA   │ │  LoRA │ │                 │
                    │  │ payload │ │ latch │ │  ...            │
                    │  └────┬────┘ └───┬───┘ │                 │
                    │       │ (frozen) │(frzn)│                 │
                    │  ┌─────────┐                              │
                    │  │  LoRA   │  <- active task: trains      │
                    │  │connector│     current task only        │
                    │  └────┬────┘                              │
                    └───────┼───────────────────────────────────┘
                            │ nnx.update(model, lora_state)
                            │ (outside JIT)
                            v
                    ┌───────────────────────────────────────────┐
                    │            TaskRouter                      │
                    │                                           │
                    │  concat(lang_emb, vis_feat)               │
                    │  -> [Linear -> LayerNorm -> GELU] x2      │
                    │  -> Linear(hidden=256, out=max_tasks=16)  │
                    │  -> mask inactive slots to -1e9           │
                    │  -> softmax over active tasks             │
                    │  -> argmax or weighted combination        │
                    └───────────────────────────────────────────┘
                            │
                            v
                    selected adapter index
```

### 3.3 Task Adapter Bank Design

`TaskAdapterBank` is a versioned registry mapping task IDs to their LoRA parameter states. Each adapter is stored as a pure dict of numpy arrays, compatible with orbax checkpointing.

**Frozen-old / train-new semantics.** After training task N:
1. `freeze_adapter(task_N_id)` marks the adapter read-only in the bank.
2. The active model's LoRA params are saved into the bank via `register_adapter(task_id, model)`.
3. Only the new task's LoRA params enter the optimizer state for the next task.

**Applying an adapter.** `merge_into_model(model, task_id)` calls `nnx.update(model, lora_state)`, overwriting the model's LoRA params with the stored state. This happens outside JIT boundaries.

**Checkpoint structure:**
```
  checkpoints/adapter_bank/
  ├── metadata.json          -- frozen set, registration order
  ├── payload/
  │   └── adapter.npz        -- flattened LoRA arrays
  ├── latch/
  │   └── adapter.npz
  ├── clean/
  │   └── adapter.npz
  └── connector/
      └── adapter.npz
```

The `metadata.json` file records which adapters are frozen and in what order tasks were registered. This is the source of truth for reconstructing the bank after a checkpoint restore.

### 3.4 Language-Visual Router

The `TaskRouter` is a small MLP that selects which adapter to use at inference time without an oracle task ID. It runs on features already computed by the frozen backbone, adding negligible compute.

**Architecture** (from `router.py`):

```
  TaskRouter Forward Pass
  ========================

  lang_embedding  (B, lang_dim)   ─────┐
                                        ├──> concat (B, lang_dim + vis_dim)
  visual_summary  (B, vis_dim)   ──────┘
                                        |
                              ┌─────────v─────────┐
                              │  Linear(in, 256)   │
                              │  LayerNorm(256)     │  <- block 1
                              │  GELU               │
                              └─────────┬─────────┘
                                        |
                              ┌─────────v─────────┐
                              │  Linear(256, 256)  │
                              │  LayerNorm(256)     │  <- block 2
                              │  GELU               │
                              └─────────┬─────────┘
                                        |
                              ┌─────────v─────────┐
                              │  Linear(256, 16)   │  <- max_tasks=16 slots
                              │  mask inactive     │     pre-allocated at init
                              │  softmax           │
                              └─────────┬─────────┘
                                        |
                                  (B, max_tasks)
                              routing probabilities
```

**Growing adapter bank.** The output head is pre-allocated for `max_tasks=16` slots at initialization. Inactive task slots receive a logit of `-1e9` before softmax, collapsing their contribution to ~0. The active mask is a boolean array updated when a new task registers. This avoids the JIT-recompilation cost of dynamically resizing the head.

**Inference modes:**
- Hard routing: `argmax(routing_probs)` selects one adapter deterministically.
- Soft routing: the full distribution is used to interpolate adapter outputs (feasible when adapter outputs are in the same action space).

**Training signal.** During task N training, the router receives supervision to route toward task N on task N data. Calibration memory batches from previous tasks can provide complementary signal to maintain old routing knowledge.

### 3.5 Behavior Distillation (Anti-Forgetting)

Catastrophic forgetting is mitigated by a teacher-snapshot approach rather than by storing the full previous dataset.

```
  Behavior Distillation Teacher-Student Flow
  ===========================================

  TEACHER (frozen snapshot of policy before task N training)
  ┌────────────────────────────────────────────────────────┐
  │  pi0.5 backbone + all adapters 1..(N-1) applied       │
  │  Parameters: NO GRADIENT FLOW                          │
  │  Lives on CPU if GPU memory is tight                   │
  └──────────────────────────┬─────────────────────────────┘
                             │ teacher_actions (no grad)
  calibration                │
  memory ──────────────────> SAME BATCH
  (episodes from             │
   tasks 1..N-1)             v
                   ┌─────────────────────┐
                   │  MSE(student, teach) │ = L_distill
                   └─────────────────────┘
                             +
  task N data ──────────────> STUDENT (current model)
                             │ L_task = flow matching loss
                             v
                   ┌─────────────────────┐
                   │  L_total =          │
                   │  L_task             │
                   │  + lambda * L_distill│
                   └─────────────────────┘
                             |
                             v
                     gradient update
                  (only current task LoRA)
```

**Calibration memory.** A small episode buffer, one per previous task. Default capacity: 50-100 episodes per task. Episodes are sampled uniformly across all previous tasks, or weighted by operational importance if desired.

**Distillation variants:**
- **Action-space distillation (default):** `L_distill = MSE(student_actions, teacher_actions)`. Simple, interpretable, minimal engineering overhead.
- **Latent-space distillation (ablation):** KL divergence between student and teacher intermediate flow-matching representations. Richer signal but more fragile numerically.

**Lambda ablation plan:**

| Lambda | Distillation strength | Expected behavior |
|--------|----------------------|-------------------|
| 0.0    | None (ablation)      | Baseline without anti-forgetting |
| 0.1    | Light                | Minimal regularization |
| 0.5    | Medium               | Recommended default |
| 1.0    | Strong               | May suppress new task learning |

### 3.6 Continual Training Protocol

```
  SpaceCIL Sequential Training Protocol
  =======================================

  Task sequence: payload -> latch -> clean -> connector

  ┌───────────────────────────────────────────────────────────┐
  │ FOR each task_i in [payload, latch, clean, connector]:    │
  │                                                           │
  │   Step 1: Initialize new LoRA adapter for task_i         │
  │           (random init, rank=8, target: lm + action_exp) │
  │                                                           │
  │   Step 2: Train adapter + router on task_i data          │
  │           Inner loop: standard flow-matching gradient     │
  │           L = L_flow_matching(task_i_data)               │
  │             + lambda * L_distill(calibration_memory)     │
  │           Router supervised toward task_i index          │
  │                                                           │
  │   Step 3: Register adapter in bank, freeze it            │
  │           bank.register_adapter(task_i, model)           │
  │           bank.freeze_adapter(task_i)                    │
  │                                                           │
  │   Step 4: Snapshot model as new teacher                  │
  │           distillation.update_teacher(model)             │
  │                                                           │
  │   Step 5: Add calibration episodes from task_i           │
  │           distillation.add_calibration_episodes(...)     │
  │                                                           │
  │   Step 6: Evaluate ALL tasks                             │
  │           for task_j in registered_tasks:                │
  │             success[task_i][task_j] = eval(task_j)      │
  │           -> fills row i of result_matrix R              │
  │                                                           │
  └───────────────────────────────────────────────────────────┘

  Result matrix R[i][j]:
    Row i: which task was just trained
    Col j: which task is being evaluated
    R[i][j] = success rate on task_j after training through task_i
```

The adapter swap (`nnx.update`) happens between training steps, outside the JIT-compiled `train_step` function. The JIT boundary sees only a model with already-loaded adapter weights.

### 3.7 Metrics Architecture

All metrics are computed from the result matrix `R[i][j]` where `R[i][j]` is the success rate on task `j` evaluated after training through task `i`.

**Standard continual learning metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `average_success` | `mean(R[-1, :])` | Mean over all tasks after full sequence |
| `backward_transfer` | `mean(R[j][j] - R[-1][j] for j < T)` | How performance changes after continued training |
| `forgetting` | `mean(max_k R[k][j] - R[-1][j] for j < T)` | Drop from each task's peak performance |

**Mission-aware metric (SpaceCIL contribution):**

```
  operational_forgetting(R, weights) =
    sum_j( weights[j] * (max_k R[k][j] - R[-1][j]) )
    ─────────────────────────────────────────────────
                    sum_j(weights[j])
```

Default operational weights by task importance:

| Task | Weight | Justification |
|------|--------|---------------|
| connector_mating | 1.0 | Highest mission criticality, contact precision required |
| latch_actuation | 0.8 | Mechanical interface, safety-relevant |
| payload_transfer | 0.6 | Logistics, recoverable failure |
| surface_cleaning | 0.5 | Maintenance, lowest time-criticality |

Weights are always passed by the caller; they are not hardcoded in the metric functions.

**Router diagnostics (informative, not primary claims):**
- `routing_entropy`: `-sum_k p_k * log(p_k)`. High entropy signals router uncertainty.
- `routing_accuracy`: fraction of trials where `argmax(router)` matches the true task.

---

## 4. Paper B: LunarCompose Architecture

### 4.1 Problem Formulation

LunarCompose studies whether adaptation should be **factorized along two axes** when transferring Earth-trained VLA priors into lunar-analog conditions (Blueprint Section 4.2):

- Train on a subset of task-environment cell combinations.
- Ensure every task and every environment appears in training.
- Evaluate on unseen task-environment combinations.
- Test whether factorized adaptation generalizes better than non-factorized baselines.

The paper explicitly allows the factorization assumption to fail. A large seen-unseen gap is a valid finding, not a failure to report.

### 4.2 Full System Architecture

```
  LunarCompose Full System Architecture
  =======================================

                    ┌──────────────────────────────────────────────────┐
                    │              pi0.5 Backbone (FROZEN)              │
                    │                                                   │
  wrist_rgb ───────>│  ┌────────────────────┐   ┌──────────────────┐  │
                    │  │   SigLIP (vision)   │   │  Gemma 2B LM    │  │
                    │  │   + Env LoRA        │──>│  + Task LoRA    │  │
                    │  │   (siglip layers)   │   │  (lm layers)    │  │
  instruction ────> │  └─────────┬───────────┘   └──────────┬───────┘  │
                    │            │ visual_feat               │           │
                    │            │                ┌──────────┴───────┐  │
                    │            │                │  Gemma 300M      │  │
                    │            │                │  Action Expert   │  │
                    │            │                │  + Task LoRA     │  │
                    │            │                └──────────┬───────┘  │
                    └────────────┼────────────────────┬──────┼──────────┘
                                 │                    │      │
                    lang_emb ────┘          vis_feat──┘      │
                                                             │
              ┌──────────────────────┐    ┌─────────────────────────────┐
              │   Env Adapter Bank   │    │     Task Adapter Bank       │
              │   (SigLIP LoRA)      │    │     (LM + Action LoRA)      │
              │                      │    │                             │
              │  ┌──────┐ ┌──────┐   │    │  ┌──────┐ ┌──────┐        │
              │  │  E0  │ │  E1  │   │    │  │  T1  │ │  T2  │        │
              │  │nomin.│ │shadw.│...│    │  │payld.│ │latch │ ...    │
              │  └──┬───┘ └──┬───┘   │    │  └──┬───┘ └──┬───┘        │
              └─────┼────────┼───────┘    └─────┼────────┼────────────┘
                    │        │                  │        │
              ┌─────┴────────┴──────────────────┴────────┴────────────┐
              │                   DualHeadRouter                       │
              │                                                        │
              │   concat(lang_emb, visual_summary)                    │
              │                   |                                    │
              │         ┌─────────┴─────────┐                         │
              │         |                   |                          │
              │  ┌──────v──────┐   ┌────────v──────┐                  │
              │  │  task_head  │   │   env_head    │                  │
              │  │  MLP x2     │   │   MLP x2      │                  │
              │  │  -> softmax │   │   -> softmax  │                  │
              │  │  over tasks │   │   over envs   │                  │
              │  └──────┬──────┘   └────────┬──────┘                  │
              └─────────┼───────────────────┼───────────────────────  ┘
                        │                   │
                task_idx (argmax)       env_idx (argmax)
```

### 4.3 Environment Adapter Bank

`EnvAdapterBank` mirrors `TaskAdapterBank` but targets different model layers. The key design goal is orthogonality: task and environment adapters should not interfere with each other at the parameter level.

```python
# Preferred filter (visual-path LoRA):
EnvAdapterBank(lora_target=".*siglip.*")

# Fallback (if SigLIP LoRA injection is blocked):
EnvAdapterBank(fallback_prefix_mode=True)
```

**Additive composition rule:**
```
  effective_weights = base_weights
                    + task_lora_delta      (from TaskAdapterBank)
                    + env_lora_delta       (from EnvAdapterBank)
```

Both deltas are applied via sequential `nnx.update` calls. Task adapter first, then environment adapter. If visual-path targeting is clean (no shared layers), the order is irrelevant.

**Fallback mode.** If SigLIP LoRA injection is technically blocked, the bank stores environment-prefix conditioning strings instead of LoRA weights. The prefix is prepended to the language instruction at inference time. This fallback weakens the vision-path separation claim but does not invalidate the factorization experiment; factorization can still be tested via the dual-head router and missing-corner protocol.

### 4.4 Dual-Head Router

`DualHeadRouter` wraps two independent `TaskRouter` instances. They share the same input features but maintain completely independent parameters and independent training signals.

```python
# Architecture (from dual_head_router.py):
class DualHeadRouter(nnx.Module):
    task_head: TaskRouter   # max_tasks=16, hidden_dim=256, num_layers=2
    env_head: TaskRouter    # max_envs=8,   hidden_dim=256, num_layers=2

    def __call__(lang_embedding, visual_summary, task_mask, env_mask):
        task_probs = task_head(lang_embedding, visual_summary, task_mask)
        env_probs = env_head(lang_embedding, visual_summary, env_mask)
        return task_probs, env_probs   # shapes: (B, max_tasks), (B, max_envs)
```

**Independence design.** The two heads are trained with separate supervision:
- Task head: supervised on task classification accuracy (route to correct task).
- Environment head: supervised on environment classification accuracy (route to correct env).
- They do not share a joint routing loss.

**Counterfactual probes.** The `DualHeadRouter` supports two counterfactual queries:
- `counterfactual_task(lang, vis)`: run task head in isolation, ignoring env head.
- `counterfactual_env(lang, vis)`: run env head in isolation, ignoring task head.

These are used in the factorization diagnostics (Section 4.7) to test whether swapping one factor while holding the other fixed affects the right things.

**Mutual information diagnostic.** `mutual_information_estimate(batch)` computes the empirical MI between `task_probs` and `env_probs` over a batch:
```
  P(t, e) = (1/B) * sum_b  task_probs_b[t] * env_probs_b[e]
  MI = sum_{t,e} P(t,e) * log( P(t,e) / (P(t)*P(e) + eps) + eps )
```
Lower MI indicates more factorized (independent) routing. This is a direct test of Hypothesis H2.

### 4.5 Layer Targeting Strategy

The spatial separation of task and environment LoRA injection is the central engineering design decision for LunarCompose.

```
  pi0.5 Layer Targeting Strategy
  ================================

  ┌──────────────────────────────────────────────────────────┐
  │  SigLIP Vision Encoder                                   │
  │                                                          │
  │    patch embedding layer                                 │
  │    self-attention block 1  <── Env LoRA targets HERE     │
  │    self-attention block 2  <── (filter: ".*siglip.*")    │
  │    ...                                                   │
  │    visual feature output                                 │
  │                                                          │
  │  RATIONALE: environment perturbations (lighting,         │
  │  shadow, contamination) manifest primarily as            │
  │  visual distribution shifts. The vision encoder is       │
  │  the right place to absorb them.                         │
  ├──────────────────────────────────────────────────────────┤
  │  Gemma 2B Language Model                                 │
  │                                                          │
  │    text token embeddings                                 │
  │    cross-attention (vision + language)  <── Task LoRA    │
  │    self-attention layers                <── targets HERE  │
  │    MLP blocks                           <── (filter:      │
  │    ...                                      ".*lm.*")    │
  │                                                          │
  │  RATIONALE: task semantics (what action to perform,      │
  │  what object to interact with) are encoded in the        │
  │  language-visual fusion path.                            │
  ├──────────────────────────────────────────────────────────┤
  │  Gemma 300M Action Expert                                │
  │                                                          │
  │    action token generation     <── Task LoRA targets HERE│
  │    cross-attention to context  <── (filter:              │
  │    flow matching head              ".*action_expert.*")  │
  │                                                          │
  │  RATIONALE: task-specific motor patterns (trajectories,  │
  │  grasp strategies, force profiles) are encoded here.     │
  └──────────────────────────────────────────────────────────┘

  KEY PROPERTY: Task LoRA and Env LoRA target different
  sub-graphs. Applying both is additive and non-conflicting.
  If SigLIP LoRA is blocked, env conditioning falls back to
  language prefix (documented explicitly, not silently).
```

### 4.6 Missing-Corner Evaluation Protocol

The missing-corner protocol is the scientific centerpiece of Paper B (Blueprint Section 4.7). Its purpose is not to create a hard benchmark, but to test whether **factorized reuse actually exists**.

**Grid structure (4 tasks x 3 environments = 12 cells):**

```
  Missing-Corner Grid (Rotation 0 -- Canonical)
  ===============================================

  Environment   │  payload   latch    clean   connector
  ──────────────┼──────────────────────────────────────
  E0 nominal    │  [TRAIN]  [TRAIN]  [TEST ]  [TRAIN]
  E1 shadow     │  [TRAIN]  [TEST ]  [TRAIN]  [TEST ]
  E2 contam.    │  [TEST ]  [TRAIN]  [TRAIN]  [TRAIN]

  Train cells (8): every task and every env appears at least once.
  Test  cells (4): (payload,contam.), (latch,shadow),
                   (clean,nominal), (connector,shadow)

  Rotation 1 (B):
  Test cells: (payload,nominal), (latch,contam.),
              (clean,shadow), (connector,contam.)

  Rotation 2 (C):
  Test cells: (payload,shadow), (latch,nominal),
              (clean,contam.), (connector,nominal)

  Split validation (all rotations must pass):
    [1] Every task appears in at least one train cell.
    [2] Every environment appears in at least one train cell.
    [3] Train and test cells are strictly disjoint.
    [4] Train union test covers all 12 cells exactly.
  Violation of any constraint raises before any training runs.
```

**Why this design works.** Because every task and every environment appears in training, a model that truly separates task and environment representations should generalize to unseen combinations. If it fails, the factorization assumption is falsified, not the experiment.

Three rotations ensure that no single task or environment carries disproportionate weight in the seen-unseen gap estimate.

### 4.7 Factorization Diagnostics

These are core scientific content, not optional post-hoc checks (Blueprint Section 4.9).

**`seen_unseen_gap`:** `results.seen_mean - results.unseen_mean`
- Gap > 0.3 is strong evidence against factorization.
- Gap < 0.1 is consistent with successful factorization.

**`cross_condition_breakdown`:** per-env and per-task breakdowns from `MissingCornerResult`. Reveals which environment or task is the bottleneck.

**`routing_interaction_analysis`:** estimates mutual information between task and env routing distributions. High MI warns that the heads are entangled.

**`task_env_entanglement`:** cosine similarity between flattened task LoRA parameters and env LoRA parameters in shared-layer configurations. Zero is ideal; high values indicate parameter-level entanglement.

**`counterfactual_swap_test`:** swap one factor's routing while holding the other fixed, then measure performance change. If swapping the task routing (while env routing is fixed) changes only task-relevant performance metrics, factorization is confirmed.

**Falsification criteria.** If both of these hold simultaneously, the factorization assumption is falsified:
1. `seen_unseen_gap > 0.3` across all three rotations.
2. `counterfactual_swap_test` shows that swapping one factor degrades performance in the domain of the other factor.

This must be reported clearly. A falsified factorization is a valid scientific finding.

---

## 5. Environment Taxonomy

### 5.1 E0: Nominal Laboratory Condition

```
  E0 Nominal Setup (Top View)
  ============================

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │   [Overhead diffuse panel]                      │
  │        | | | | | | |                            │
  │        v v v v v v v                            │
  │                                                 │
  │   ─────────────────────────────── workspace     │
  │       [OBJECT]    [target zone]                 │
  │                                                 │
  │   Robot base at bottom, arm reaching in         │
  │                                                 │
  └─────────────────────────────────────────────────┘

  Lighting: 500-700 lux, 4000-5000K neutral white,
            overhead diffuse (shadow/ambient ratio > 0.7)
  Surface: matte gray/beige, no reflective patches
  Variation: object position ±3 cm jitter
  Scorer: camera not decoupled (perturbation minimal)
```

E0 is the reference condition. All tasks are first demonstrated and validated in E0 before moving to perturbed conditions. The scorer camera sees effectively the same scene as the policy camera.

### 5.2 E1: Hard-Shadow / Low-Angle Illumination Condition

```
  E1 Shadow Setup (Side View)
  ============================

          [directional LED panel, 1.5-2.5m away]
           \  \  \  \  \
            \  \  \  \  \   <- 10-20 degrees above horizontal
             \  \  \  \  \    (15 degrees canonical)
              \  \  \  \  \
  ─────────────────────────────── workspace surface

  [OBJECT]  |||||||||||||||||||||   <- hard shadow region
   lit side   shadow boundary  shadow extends across workspace

  Primary source: 800-1200 lux on lit side
  Shadow region: < 100 lux (ratio < 0.1)
  Color temperature: 5500-6500K (cool, simulates sunlight)
  Ambient: < 50 lux secondary fill

  Scorer camera: overhead, separate diffuse lighting,
                 placed to avoid arm/gripper shadow.
                 Scorer must not degrade under E1.
```

E1 simulates the illumination geometry of the lunar south pole (sun angle 1.5-3 degrees above horizon). Sharp shadow boundaries can obscure task-relevant object features from the policy camera. The scorer camera is always decoupled for E1.

### 5.3 E2: Partial Contamination / Occlusion Condition

```
  E2 Contamination Setup (Top View)
  ===================================

  ┌─────────────────────────────────────────────────┐
  │  [overhead diffuse -- same as E0]               │
  │                                                 │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  <- dust layer  │
  │  ░░░░░[OBJECT]░░░░░░░░░░░░░░░░     on surface  │
  │  ░░░░ (powder coated)░░░░░░░░░░                │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                │
  │       [rock] [rubble]  <- occlusion props       │
  │        2-4cm, fixed positions                   │
  │        within camera FOV, outside manip zone    │
  └─────────────────────────────────────────────────┘

  Lighting: E0 nominal (isolates texture change from lighting)
  Surface: simulated dust (fine gray powder or matte coating)
  Objects: light powder coating (changes texture, not geometry)
  Occlusion props: fixed positions per session, reproducible
  Scorer: scoring region explicitly excluded from dust tiles.
          Pixel-difference scorer verified against dust surface.
```

E2 simulates lunar dust accumulation and workspace debris. The lighting is identical to E0, so the visual perturbation comes exclusively from surface texture and appearance changes, not illumination shifts. The scorer camera must be verified to not be confounded by dust.

---

## 6. Shared Infrastructure

### 6.1 Episode Schema

A unified episode dataclass is shared by both papers. Both the continual harness (SpaceCIL) and the missing-corner harness (LunarCompose) consume the same episode format.

```
  Episode Schema
  ===============

  Episode
  ├── obs: Observation
  │   ├── wrist_rgb: np.ndarray  (H, W, 3) uint8
  │   └── joint_positions: np.ndarray  (7,) float32
  │
  ├── action: Action
  │   └── joint_positions: np.ndarray  (7,) float32
  │       gripper: float32
  │       (8D total: 7 joints + 1 gripper)
  │
  └── meta: EpisodeMeta
      ├── task_id: str           e.g. "payload", "latch"
      ├── env_id: str            e.g. "nominal", "shadow"   <- LunarCompose key field
      ├── camera_preset_id: str  must match env_id canonically
      ├── episode_id: str        unique, for leakage checking
      └── split: str             "train" | "val" | "test"
```

The `meta.env_id` field is mandatory for LunarCompose. If it is absent from any episode, the missing-corner split validation will reject that episode before training begins.

### 6.2 Action Space

The action space is 8D: 7 absolute joint positions plus 1 gripper command. The transform chain converts raw demonstration data into model-ready tensors.

```
  Action Transform Chain
  =======================

  Raw demonstration (joint angles, radians)
          |
          v
  AbsoluteJointPositionTransform
  (validates dimensions, clips to joint limits)
          |
          v
  NormalizationTransform
  (normalize to [-1, 1] per dimension using norm_stats.json)
          |
          v
  RepackTransform
  (pack into model input format expected by RM75Inputs/Outputs)
          |
          v
  Model input: {"actions": (T, 8), "observation": {...}}
```

Computing norm stats (`uv run scripts/compute_norm_stats.py`) is required before any training run. The stats capture the per-dimension mean and standard deviation (or quantile bounds) of joint angles across the demonstration dataset.

### 6.3 Scorer Architecture

Scorers are decoupled from the policy: they use a separate camera with independent lighting to avoid confounding environment perturbations with task success measurement.

```
  Scorer Class Hierarchy
  =======================

  Scorer (ABC)
  ├── score(episode: Episode) -> ScorerResult
  │     ScorerResult.success: bool
  │     ScorerResult.confidence: float [0, 1]
  └── (abstract)

  PayloadTransferScorer
      Top-down binary occupancy check in target zone.
      Camera: overhead, decoupled from E1/E2 perturbation.

  LatchActuationScorer
      Binary state check via hall-effect sensor or color marker.
      Does not rely on policy camera.

  SurfaceCleaningScorer
      Pixel-difference scorer on scorer camera (lighting-controlled).
      Scoring region explicitly excluded from dust-affected tiles in E2.
      CRITICAL: scorer camera must not degrade under E1 or E2 conditions.

  ConnectorMatingScorer
      Binary contact/insertion sensor (preferred) or visual depth classifier.
      Highest operational weight (1.0) in mission-aware metrics.
```

Scorer decoupling requirement (Blueprint Section 3.7): a scorer camera that sees the same perturbation as the policy camera is a confound and must be rejected. All four scorers are validated against manual labels on a pilot subset before experiments begin.

---

## 7. Evaluation Framework

### 7.1 SpaceCIL Evaluation

The primary evaluation artifact is the result matrix `R[i][j]`.

```
  SpaceCIL Result Matrix Structure
  ==================================

  After training task 1 (payload):
    R[0][0] = success rate on payload  (current task)

  After training task 2 (latch):
    R[1][0] = success rate on payload  (backward transfer check)
    R[1][1] = success rate on latch    (current task)

  After training task 3 (clean):
    R[2][0] = success rate on payload  <- forgetting visible here
    R[2][1] = success rate on latch
    R[2][2] = success rate on clean

  After training task 4 (connector):
    R[3][0] = success rate on payload
    R[3][1] = success rate on latch
    R[3][2] = success rate on clean
    R[3][3] = success rate on connector

  Full matrix (4x4, upper triangle undefined):

         payload  latch  clean  connector
  T1: [  R[0,0]    -      -       -    ]
  T2: [  R[1,0]  R[1,1]   -       -    ]
  T3: [  R[2,0]  R[2,1] R[2,2]    -    ]
  T4: [  R[3,0]  R[3,1] R[3,2]  R[3,3] ]

  Metrics derived from R:
    average_success     = mean(R[3, :])
    forgetting          = mean(max_k R[k,j] - R[3,j] for j < 3)
    operational_forgetting = weighted forgetting (task weights above)
```

### 7.2 LunarCompose Evaluation

The primary evaluation artifact is the `MissingCornerResult` per rotation.

```
  LunarCompose Evaluation Flow
  ==============================

  For each rotation r in {0, 1, 2}:

    1. Generate split: train_cells, test_cells = harness.generate_split(r)
    2. Validate split: constraint check (raises if invalid)
    3. Train factorized model on train_cells
    4. Evaluate all 12 cells:
       per_cell_success[(task_id, env_id)] = mean success over eval episodes
    5. Compute aggregates:
       seen_mean   = mean over train_cells
       unseen_mean = mean over test_cells
       gap         = seen_mean - unseen_mean

  Cross-rotation aggregation:
    mean_gap   = mean(gap for r in {0,1,2})
    std_gap    = std(gap for r in {0,1,2})

  Comparison against baselines:
    delta_gap = factorized_gap - baseline_gap  (lower is better)

  Reporting:
    If mean_gap > 0.3: factorization assumption weakly held or failed.
    If mean_gap < 0.1: consistent with successful factorization.
    Always report all three rotations, not just the best one.
```

---

## 8. Baseline Taxonomy

### 8.1 SpaceCIL Baselines

| Baseline | Config suffix | What it controls for |
|----------|---------------|----------------------|
| Sequential full fine-tuning | `_fulltune` | Upper bound on plasticity; lower bound on stability |
| Shared multi-task PEFT | `_shared_lora` | Whether per-task specialization matters at all |
| Per-task PEFT with oracle task ID | `_oracle` | Performance ceiling with perfect routing |
| SpaceCIL without distillation | `_nodistill` | Whether anti-forgetting component contributes |
| SpaceCIL with random routing | `_randrout` | Whether routing matters or adapter-per-task is enough |

The oracle baseline is the performance ceiling. The random routing ablation isolates the router's contribution from the adapter bank's contribution. A system with per-task adapters but random routing still benefits from specialization via lucky routing.

### 8.2 LunarCompose Baselines

| Baseline | Config name | What it controls for |
|----------|-------------|----------------------|
| Full adaptation (per-cell fine-tune) | `lunarcompose_full_ft` | Oracle ceiling; no generalization expected |
| Single merged PEFT (one LoRA all cells) | `lunarcompose_monolithic` | Whether any factorization is necessary |
| Domain-randomized (no env adapter) | `lunarcompose_domain_rand` | Whether augmentation alone suffices vs explicit adaptation |
| Task arithmetic (Ilharco et al.) | `lunarcompose_task_arith` | Fixed-coefficient composition vs learned routing |
| Parameter-matched PEFT (same N params) | `lunarcompose_param_matched` | Controls parameter count as a confound |

The domain-randomized baseline is the most practically relevant competitor: it uses the same data but mixes all three environments per task without explicit env adaptation. If it matches factorized performance, the env adapter is not needed.

---

## 9. Expected Results Summary

### 9.1 SpaceCIL Expected Results

**H1** (task-specialized PEFT beats alternatives): Task-per-adapter setup should outperform sequential full fine-tuning (catastrophic forgetting) and shared PEFT (insufficient capacity or transfer conflicts). The oracle adapter baseline sets the ceiling.

**H2** (router sufficient without oracle): The language-visual router should achieve routing accuracy substantially above random (baseline: `1/N_tasks`), reducing the gap between no-oracle and oracle conditions. If routing accuracy is near random, the system degrades to essentially random adapter selection.

**H3** (distillation reduces forgetting): `operational_forgetting` with distillation should be measurably lower than the `_nodistill` ablation. The effect should be most visible on high-weight tasks (connector, latch) since those are penalized most heavily by the metric.

**H4** (mission-aware metric reveals patterns): `operational_forgetting` should diverge from uniform `forgetting` at least once across the task sequence, revealing that the tasks that matter most are not necessarily the ones that degrade most.

### 9.2 LunarCompose Expected Results

**H1** (factorized beats non-factorized on unseen cells): The factorized system should achieve a smaller seen-unseen gap than `lunarcompose_monolithic` and `lunarcompose_param_matched`. The domain-randomized baseline is the hardest competitor and may match factorized on easy conditions.

**H2** (routing interpretability): The MI estimate between task and env routing heads should be measurably lower than a single-head ablation where one head routes both factors. Counterfactual swap tests should confirm that swapping task routing changes task-relevant metrics without degrading env-sensitive metrics.

**H3** (small taxonomy sufficient): The three-condition taxonomy (E0, E1, E2) should expose meaningfully different failure modes across conditions. Per-env breakdowns should show that E1 and E2 have lower unseen success rates than E0 across most baselines.

**H4** (missing-corner falsifiability): The protocol should be able to reject factorization if `gap > 0.3` consistently across rotations. If factorization holds, all three rotations should agree.

### 9.3 Falsifiability

Both papers allow negative results. This is intentional:

- If SpaceCIL finds that the router cannot beat random, the paper reports this as a finding about the limits of language-visual routing for task disambiguation on this platform, and proposes what additional signal would be needed.
- If LunarCompose finds `mean_gap > 0.3` across all rotations and counterfactual swaps show cross-factor interference, the paper reports that the factorization assumption is falsified in this setting, and analyzes which task-environment combinations are hardest to separate.

Neither outcome invalidates the scientific value of the study. The contribution is the protocol and the evidence, not the confirmation of a predetermined hypothesis.

---

## 10. Program Thesis

The two papers together support:

> Space robotics should be formulated as **capability reuse and expansion from Earth-trained embodied intelligence**, rather than repeated from-scratch policy construction.

This thesis rests on three pillars, each requiring both papers:

1. **Reuse is viable.** Earth-trained VLA priors (pi0.5) transfer to lunar-analog proxy tasks with minimal data. Neither paper would be possible if the baseline policy had zero capability on these tasks.

2. **Expansion is controlled.** SpaceCIL shows that new tasks can be added without destroying old capability, if parameter-efficient expansion is done carefully. This is the "capability expansion" half.

3. **Adaptation is composable.** LunarCompose shows that environment-specific corrections can be separated from task semantics and composed independently. This is the "reuse" half.

This is a program-level framing for future work. Neither SpaceCIL alone nor LunarCompose alone makes this claim. Both are required, and together they form the scientific basis for treating space manipulation as an Earth-to-Space capability transfer problem rather than a domain-specific construction problem.

---

## 11. References

Full reference lists are maintained in the Blueprint and are not duplicated here. See:

- `customized_docs/Research_Idea_Blueprint_SpaceCIL_LunarCompose.md` -- Sections 9.1 through 9.5 for the complete verified reference list, organized into:
  - VLA and generalist policy references (Black et al. 2025, Kim et al. 2024, etc.)
  - Continual / lifelong robot learning references (LIBERO, SPECI, etc.)
  - Data and transfer ecosystem references (Open X-Embodiment, DROID, etc.)
  - Compositionality and modularity references (CompoSuite, Task Arithmetic, etc.)
  - Official mission and operations references (NASA south-pole lighting, lunar dust roadmap, ESA lunar south pole)

Key papers by section:

**Backbone**: Black et al. (2025) *pi0.5: a Vision-Language-Action Model with Open-World Generalization*. ArXiv 2504.16054.

**Continual learning**: Liu et al. (2023) *LIBERO*; Lei et al. (2025) *Dynamic Mixture of Progressive Parameter-Efficient Expert Library*; Wu et al. (2025) *Continually Evolving Skill Knowledge in VLA*.

**Composition / PEFT**: Ilharco et al. (2022) *Editing Models with Task Arithmetic*; Zhang et al. (2023) *Composing Parameter-Efficient Modules with Arithmetic Operations*; Mendez et al. (2022) *CompoSuite*.

**Mission context**: NASA lunar south-pole lighting study (ntrs.nasa.gov/citations/20240011393); NASA Lunar Dust Mitigation Roadmap (2024); ESA lunar south pole operational context.
