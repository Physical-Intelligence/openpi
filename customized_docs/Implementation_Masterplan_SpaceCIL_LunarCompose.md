# Internal Implementation Masterplan
## SpaceCIL + LunarCompose
### Released pi0.5 Backbone Adaptation on the RM75 Mobile Manipulation Platform

**Document role.** This is the internal-only implementation masterplan for the two-paper program:
- **Paper A: SpaceCIL**
- **Paper B: LunarCompose**

This document exists to guide:
- repo-level implementation,
- coding-agent task decomposition,
- infrastructure build-out,
- data collection and calibration,
- safety constraints,
- experiment orchestration,
- fallback decisions.

This document is intentionally **operationally honest**. It should explicitly state which components are reused, which are patched, which are new, what is speculative, and what must happen if a technical dependency fails.

---

## 1. System reality

### 1.1 Backbone reality
- We are **not** building a VLA from scratch.
- We are instantiating the project on a **released pi0.5 backbone** through the public **openpi** stack.
- Public pi0.5 support in openpi currently centers on the **flow-matching policy path**.
- Therefore the implementation strategy should stay as close as possible to the released path for **Paper A**, and only introduce additional extensions where necessary for **Paper B**.

### 1.2 Platform reality
- Platform: wheeled base + RM75 7-DoF arm + two-finger gripper
- Primary policy camera: wrist RGB, hand-eye calibrated
- Optional scene camera(s): scoring, replay, debugging, audit
- Main policy action space: **Absolute Joint Position (7 DoF) + Gripper (1 DoF) = 8D total**
- Base motion is available operationally, but the mainline policy is **arm-centric** unless base control is explicitly brought into-policy later

### 1.3 Internal truth policy
Internally we must never pretend that:
- openpi support already includes every module we want,
- visual-path environment adapters are already available,
- or the public codebase natively supports our full factorized adaptation design.

The correct internal attitude is:
- **reuse first**,
- **patch minimally**,
- **extend only when needed**,
- **downgrade claims immediately if an extension is blocked**.

---

## 2. Two-paper implementation split

### 2.1 Paper A implementation goal
Build a robust continual specialization stack:
- task adapter bank,
- language-visual router,
- behavior distillation anti-forgetting,
- continual evaluation harness,
- real-robot benchmark tasks,
- baseline orchestration.

### 2.2 Paper B implementation goal
Extend the same infra with:
- environment metadata enforcement,
- environment adapter bank,
- dual-head routing,
- missing-corner compositional evaluation harness,
- factorization diagnostics,
- fallback path if visual-path environment adaptation fails.

### 2.3 Development rule
Do **not** start by building Paper B first. Paper B depends on:
- a stable one-task training path,
- a validated wrist-camera policy path,
- a reliable scorer stack,
- and a working Paper A continual harness.

---

## 3. Repo strategy

### 3.1 Primary substrate
Use **openpi** as the canonical starting point because it already provides:
- released pi0.5 model plumbing,
- LeRobot data compatibility,
- training / evaluation scaffolding,
- policy-serving assumptions close to our target use.

### 3.2 Repo-use policy
For every module, classify it as one of:
- **reuse as-is**
- **patch lightly**
- **new module**

This table should be maintained explicitly and shown to every coding agent before they start modifying code.

### 3.3 Anti-chaos rule
No coding agent should rewrite the full stack from scratch unless a minimal patch path has already failed and that failure is documented.

---

## 4. Core infra modules

### 4.1 Unified episode schema
Freeze one episode schema for both papers:
- `obs.wrist_rgb`
- `obs.scene_rgb` (optional or scorer-only)
- `obs.q`
- `obs.dq`
- `obs.gripper`
- `obs.base_state` (if logged)
- `action.joint_pos`
- `action.gripper_cmd`
- `lang.instruction`
- `label.success`
- `label.fail_type`
- `meta.task_id`
- `meta.env_id`
- `meta.operator_id`
- `meta.session_id`
- `meta.camera_preset_id`
- `meta.calibration_version`
- `meta.scene_revision`
- `meta.object_revision`

No duplicate schemas for different papers.

### 4.2 Action transform layer
Implement one authoritative path from:
- teleop / controller output
- to canonical absolute joint position + gripper representation
- to robot execution
- and to the representation expected by the released training path.

Never maintain two inconsistent converters.

### 4.3 Task adapter bank
Paper A mainline module:
- one task-specialized PEFT module per task,
- versioned registry,
- frozen-old / train-new semantics,
- restore-safe naming and checkpoint layout.

### 4.4 Language-visual router
Paper A mainline module:
- input = language embedding + visual summary (+ optional compact state summary),
- output = task adapter routing weights,
- train = joint with the current task adapter,
- inference = no oracle task ID.

### 4.5 Behavior distillation
Paper A anti-forgetting module:
- teacher = previous policy snapshot,
- memory = small calibration set from earlier tasks,
- loss = action-space or latent-space imitation term,
- ablations on memory size and weighting.

### 4.6 Environment adapter bank
Paper B extension:
- preferred path = visual-path environment adapters,
- safe fallback = environment-prefix or non-visual modulation,
- registry must remain compatible with the task adapter bank.

This is the biggest engineering dependency in the project and must be marked as such.

### 4.7 Dual-head router
Paper B extension:
- one head for task routing,
- one head for environment routing,
- hooks for independence diagnostics,
- support for counterfactual task-env swap tests.

### 4.8 Evaluation harnesses
Build two separate harnesses:
- **continual harness** for SpaceCIL,
- **missing-corner compositional harness** for LunarCompose.

Do not prematurely unify them into a single giant evaluation framework.

---

## 5. Benchmark implementation doctrine

### 5.1 Task suite v1
For each task, define internally:
- exact object geometry,
- object material / friction assumptions,
- goal-state specification,
- permissible initial pose range,
- scorer implementation,
- reset procedure,
- failure modes,
- contact-risk notes,
- operator demo policy,
- safety override rules.

### 5.2 Environment suite v1
For each environment condition, define:
- physical setup,
- lighting setup,
- camera preset,
- allowed variation window,
- whether scorer camera is decoupled from policy view,
- whether the perturbation affects only policy view or both policy and scorer views.

### 5.3 Missing-corner protocol
Hard-code:
- canonical train/test split,
- rotation B,
- rotation C,
- replayable initial-state JSONs,
- no train/test leakage by scene-instance ID.

### 5.4 Pretraining-affinity note
Internally keep a short mapping from each benchmark task to nearby terrestrial manipulation primitives (e.g. pick-place, lever actuation, tool use, insertion). This is useful both for scientific interpretation and for diagnosing whether disappointing results come from true OOD difficulty or poor benchmark construction.

---

## 6. Data collection doctrine

### 6.1 Accepted-demo protocol
Every episode must be classified as one of:
- **accepted**
- **rejected**
- **safety-aborted**
- **hardware-invalid**

Accepted demos must satisfy:
- no safety stop during demonstration,
- no major frame drop in required cameras,
- no gross initial-state mismatch,
- no human hand remaining in-policy-view during the core execution window,
- no ambiguous success label,
- no unintended shortcut that bypasses the intended manipulation primitive.

### 6.2 Session doctrine
Every session must log:
- operator ID,
- date/time,
- robot calibration version,
- hand-eye calibration check result,
- camera preset,
- scene revision,
- object revision,
- environment condition,
- notable anomalies.

### 6.3 Calibration doctrine
Define recertification rules for:
- hand-eye calibration,
- wrist-camera exposure lock,
- scorer-camera exposure lock,
- scene-camera pose repeatability,
- drift threshold for invalidating a session.

### 6.4 Operator doctrine
Record whether:
- multiple operators are used,
- operators are mixed across train/val/test,
- operator identity is included in metadata,
- demo style changes across sessions.

---

## 7. Safety doctrine

### 7.1 Real-robot control contract
Freeze internal limits for:
- joint speed,
- end-effector speed,
- gripper force / closure,
- insertion timeout,
- collision stop threshold,
- torque anomaly threshold,
- supervisor emergency stop authority.

### 7.2 Evaluation doctrine
If a trial enters an unsafe state:
- stop the trial,
- log the safety trigger,
- label the fail type,
- do **not** silently retry the same trial without logging.

---

## 8. Scorer doctrine

### 8.1 Scorer design principle
The scorer must not become the hidden confounder of the paper.

### 8.2 Wipe task scorer
If a wipe / cleaning scorer is based on pixel difference, then one of the following must hold:
- scorer camera is lighting-controlled and isolated from policy-view perturbation, or
- the scorer uses markers / segmentation signals robust to the tested environment conditions.

### 8.3 Scorer validation
For every scorer:
- compare against manual labels on a pilot subset,
- record precision / recall or agreement,
- define what level of disagreement triggers scorer revision.

---

## 9. Build order

### 9.1 Phase A — before SpaceCIL training
- freeze schema,
- freeze absolute joint position + gripper action convention,
- implement one-task end-to-end loop,
- validate wrist-camera semantics,
- validate camera-slot hypothesis,
- validate state-input path,
- finish scorer prototypes,
- finish manual-vs-auto scorer audit.

### 9.2 Phase B — SpaceCIL core
- task adapter bank,
- router,
- behavior distillation,
- continual harness,
- baseline pipeline,
- restore / checkpoint registry,
- plotting and reporting.

### 9.3 Phase C — LunarCompose extension
- enforce env metadata from collection onward,
- build environment adapter path,
- add dual-head router,
- add missing-corner harness,
- add factorization diagnostics,
- run fallback if visual-path env adaptation is blocked.

---

## 10. Pilot gates

### 10.1 Gate G1 — one-task sanity
Before any continual-learning experiments:
- one task must train and replay correctly,
- wrist-camera policy path must work,
- scorer must agree with manual labels on a pilot subset.

### 10.2 Gate G2 — SpaceCIL readiness
Before claiming Paper A mainline:
- at least two tasks must train sequentially,
- router must beat random and a weak routing baseline,
- behavior-distillation path must execute stably,
- adapter registry and checkpoint restore must be reliable.

### 10.3 Gate G3 — LunarCompose readiness
Before claiming Paper B mainline:
- env metadata are enforced without leakage,
- missing-corner split tooling is verified,
- env adaptation path exists,
- and if visual-path env adaptation is blocked, the fallback is activated immediately.

---

## 11. Failure-mode table

Maintain an internal table of the form:

| Failure | Affected paper | Root cause class | Immediate action | Claim downgrade |
|---|---|---|---|---|
| Router old-task collapse | A | optimization / interference | freeze trunk, inspect router expansion | none or weaker routing claim |
| Behavior distillation unstable | A | loss weighting / decoding mismatch | reduce lambda, inspect target format | weaker anti-forgetting claim |
| Visual env adapter blocked | B | engineering dependency | switch to env-prefix fallback | weaken vision-path separation claim |
| Env scorer confounded by lighting | B | evaluation confound | decouple scorer view or redesign signal | do not proceed until fixed |
| Task-env leakage in split | B | data hygiene | rebuild split | invalidate affected run |

---

## 12. Coding-agent interface

### 12.1 Agent-task format
Every coding-agent task should specify:
- target module,
- input assumptions,
- output artifact,
- tests to pass,
- what must not be changed,
- fallback if blocked.

### 12.2 Suggested agent workstreams
Split coding agents into focused streams:
- schema / loaders,
- controller / action transform,
- task adapter bank,
- router features,
- behavior distillation,
- scorer stack,
- continual harness,
- environment harness,
- diagnostics / plotting,
- baseline orchestration.

### 12.3 Guardrail
Never ask an agent to “implement SpaceCIL” or “implement LunarCompose” as one task. Always decompose into verifiable modules.

---

## 13. Experiment artifact policy

Every experiment run must save:
- config hash,
- adapter versions,
- code commit,
- scene revision,
- calibration version,
- initial-state seed,
- scorer outputs,
- raw video references,
- fail type,
- whether the run belongs to canonical split or rotated split.

This is mandatory for later paper writing and rebuttal.

---

## 14. Mapping between the two documents

- **Research Idea Blueprint** answers: *what is the scientific problem, claim structure, and evaluation logic?*
- **Implementation Masterplan** answers: *how do we actually build, test, and run the system?*

These documents should only cross-reference by concept names:
- SpaceCIL
- LunarCompose
- task stream
- environment taxonomy
- missing-corner protocol
- operationally weighted forgetting

The idea document must never depend on this implementation document for scientific coherence.

---

## 15. Verified engineering and literature references

### 15.1 Backbone and stack
1. **Physical Intelligence / openpi repository** (public stack for released pi0.5 usage).  
   https://github.com/Physical-Intelligence/openpi
2. **Black et al. (2025)**, *pi0.5: a Vision-Language-Action Model with Open-World Generalization*.  
   ArXiv: https://arxiv.org/abs/2504.16054
3. **Black et al. (2024)**, *pi0: A Vision-Language-Action Flow Model for General Robot Control*.  
   ArXiv: https://arxiv.org/abs/2410.24164
4. **LeRobot dataset format documentation** (for implementation alignment).  
   https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3

### 15.2 Continual and compositional learning neighbors
5. **Liu et al. (2023)**, *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*.  
   ArXiv: https://arxiv.org/abs/2306.03310
6. **Lei et al. (2025)**, *Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning*.  
   ArXiv: https://arxiv.org/abs/2506.05985
7. **Xu and Nie (2025)**, *SPECI: Skill Prompts based Hierarchical Continual Imitation Learning for Robot Manipulation*.  
   ArXiv: https://arxiv.org/abs/2504.15561
8. **Wu et al. (2025)**, *Continually Evolving Skill Knowledge in Vision Language Action Model*.  
   ArXiv: https://arxiv.org/abs/2511.18085
9. **Task-agnostic adaptation (2024)**, *Task-agnostic Lifelong Robot Learning with Retrieval-based Weighted Local Adaptation*.  
   ArXiv: https://arxiv.org/abs/2410.02995
10. **Primitive Prompt Learning (2025)**, *Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation*.  
    ArXiv: https://arxiv.org/abs/2504.00420
11. **Mendez et al. (2022)**, *CompoSuite: A Compositional Reinforcement Learning Benchmark*.  
    ArXiv: https://arxiv.org/abs/2207.04136
12. **Ilharco et al. (2022)**, *Editing Models with Task Arithmetic*.  
    ArXiv: https://arxiv.org/abs/2212.04089
13. **Zhang et al. (2023)**, *Composing Parameter-Efficient Modules with Arithmetic Operations*.  
    ArXiv: https://arxiv.org/abs/2306.14870

### 15.3 Data and teleoperation ecosystem
14. **Open X-Embodiment (2023)**, *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*.  
    ArXiv: https://arxiv.org/abs/2310.08864
15. **Khazatsky et al. (2024)**, *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*.  
    ArXiv: https://arxiv.org/abs/2403.12945
16. **Mobile ALOHA (2024)**, *Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation*.  
    ArXiv: https://arxiv.org/abs/2401.02117
17. **ALOHA / ACT (2023)**, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.  
    ArXiv: https://arxiv.org/abs/2304.13705

### 15.4 Mission and environment references
18. **NASA lunar south-pole lighting study**.  
    https://ntrs.nasa.gov/citations/20240011393
19. **NASA Lunar Dust Mitigation Roadmap (2024)**.  
    https://ntrs.nasa.gov/api/citations/20240013978/downloads/NASA%20Lunar%20Dust%20Mitigation%20Roadmap%20Fall%202024.pdf
20. **NASA Autonomous Utility Connector for Lunar Surface Systems**.  
    https://techport.nasa.gov/projects/8720
21. **NASA lunar dust-tolerant connector reference**.  
    https://ntrs.nasa.gov/citations/20100005258
22. **ESA lunar south pole operational context**.  
    https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/Lunar_Lander/Exploring_the_lunar_South_Pole
