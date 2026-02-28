# Research Idea Blueprint
## SpaceCIL + LunarCompose
### Mission-Driven Reuse of Earth-Trained Embodied Intelligence for Lunar-Analog Mobile Manipulation

**Document role.** This is the paper-facing research blueprint for two articles:
- **Paper A: SpaceCIL**
- **Paper B: LunarCompose**

This document defines the **scientific problem**, **claims**, **evaluation logic**, and **space-relevant narrative**. It intentionally excludes repository names, training scripts, code hooks, configuration files, and implementation-specific engineering details.

---

## 1. Mission framing

### 1.1 The central thesis
Recent progress in embodied AI has produced increasingly capable **Earth-trained** robot foundation policies, especially vision-language-action (VLA) models. The scientific opportunity for space robotics is not merely to build another policy from scratch, but to develop principled mechanisms that let us **reuse, specialize, and expand** Earth-developed embodied intelligence as space missions evolve.

### 1.2 Why this matters for space robotics
Lunar surface systems will not operate under a fixed task set or a static deployment condition. The operational burden grows over time: new interfaces, servicing procedures, transfer tasks, cleaning and maintenance routines, and connector operations emerge as infrastructure expands. At the same time, lunar deployment introduces visual and operational conditions that are weakly represented in terrestrial robot datasets, such as **low-angle hard shadows**, **low-texture appearance**, and **surface contamination or partial occlusion**.

### 1.3 Program-level scientific question
> How can a released Earth-trained VLA backbone remain useful as tasks and deployment conditions move toward more out-of-distribution, space-relevant operating regimes?

We answer this through two orthogonal scientific questions:
- **Task-axis capability expansion** under sequential task emergence (**SpaceCIL**)
- **Environment-axis capability reuse and composition** under lunar-analog condition shifts (**LunarCompose**)

---

## 2. Program structure

### 2.1 Why this is a two-paper program
These are not two weak slices of the same result. They address **different axes of adaptation**:
- **Paper A (SpaceCIL)** studies whether a released VLA can acquire **new task capability** over time without catastrophic forgetting.
- **Paper B (LunarCompose)** studies whether a released VLA can separate **task semantics** from **environment adaptation** and generalize to unseen task-environment combinations.

### 2.2 Shared program thesis
Together, the two papers support a broader thesis:

> Space robotics should be formulated as **capability reuse and expansion from Earth-trained embodied intelligence**, rather than repeated from-scratch policy construction.

---

## 3. Paper A — SpaceCIL
## Continual Skill Acquisition on a Real Mobile Manipulation Platform for Expanding Lunar Operations

### 3.1 One-paragraph idea
SpaceCIL studies whether a released VLA backbone can be **continually specialized** on a real mobile manipulation platform as new operational tasks arrive sequentially. The paper focuses on **parameter-efficient task expansion**, **router-based task selection without test-time oracle task identity**, and **mission-aware forgetting analysis**.

### 3.2 Problem statement
Given a pretrained visuomotor policy and a sequence of task datasets arriving over time, learn task-specialized adaptation modules such that:
1. new task performance improves quickly,
2. previously acquired tasks degrade minimally,
3. inference does not require test-time oracle task ID,
4. adaptation remains parameter-efficient and operationally diagnosable.

### 3.3 Scientific motivation
In realistic lunar or space-infrastructure operations, the task stream is not known once and for all. New payload interfaces, maintenance actions, connector manipulations, and cleaning procedures emerge over the operational life of the system. Re-training the full policy after each update is costly, risky, and hard to validate.

### 3.4 Hypotheses
- **H1**: Task-specialized PEFT modules outperform sequential full fine-tuning and a single shared PEFT module in the continual setting.
- **H2**: A lightweight language-visual router is sufficient to select task-specialized capability without test-time oracle task identity.
- **H3**: A small anti-forgetting objective based on prior-task calibration trajectories significantly reduces mission-relevant forgetting.
- **H4**: An operationally weighted forgetting metric reveals degradation patterns that are not adequately exposed by uniform continual-learning metrics alone.

### 3.5 Core contributions
1. A real-robot continual specialization benchmark on a mobile manipulation platform under an **operationally ordered lunar-analog task stream**.
2. A task-bank adaptation framework for continual VLA specialization.
3. A mission-aware forgetting metric and evaluation protocol.
4. A study of routing, anti-forgetting, and parameter-efficiency trade-offs.

### 3.6 What SpaceCIL is not
- not a claim of fully open-ended robot autonomy,
- not a world-model or world-action-model paper,
- not a generic continual-learning benchmark for all robots,
- not necessarily a joint base-plus-arm policy paper unless base control is explicitly in-policy.

### 3.7 Benchmark structure
The task suite should contain a compact set of **controlled lunar-analog proxy tasks** that are both:
- operationally relevant to lunar infrastructure and surface servicing, and
- semantically adjacent to terrestrial manipulation primitives already represented in modern robot data.

Recommended task families:
- **payload transfer / unloading**
- **latch or lever actuation**
- **surface cleaning / wiping**
- **connector mating / insertion**

These tasks should be described as **proxy operational motifs**, not as one-to-one replicas of lunar mission procedures.

### 3.8 Metrics
Primary metrics:
- task success rate,
- average success over the sequence,
- backward transfer / forgetting,
- operationally weighted forgetting,
- parameter growth vs performance,
- routing quality / routing entropy as diagnostic signals.

### 3.9 Baseline classes
The paper should compare against:
- sequential full fine-tuning,
- shared multi-task PEFT,
- per-task specialization with oracle task identity,
- continual specialization without anti-forgetting,
- routing ablations.

### 3.10 Claim boundary
A safe scoped claim form is:

> “To our knowledge, this is the first real-robot study of continual specialization of a released pi0.5-class VLA on a wheeled mobile manipulation platform under an operationally ordered lunar-analog task stream.”

The paper should **not** claim unqualified firstness on “continual VLA” as a whole.

---

## 4. Paper B — LunarCompose
## Factorized Task × Environment Adaptation for Lunar-Analog Compositional Generalization

### 4.1 One-paragraph idea
LunarCompose studies whether adaptation should be **factorized along two axes—task and environment**—when transferring Earth-trained VLA priors into lunar-analog deployment conditions. The paper tests whether this factorization improves generalization to **unseen task-environment combinations** on a real mobile manipulation platform.

### 4.2 Problem statement
Given a set of tasks and a set of environment conditions, train only on a subset of task-environment combinations and evaluate whether a factorized adaptation design generalizes to unseen combinations better than non-factorized baselines.

### 4.3 Scientific motivation
Space robots will not experience a single frozen condition. Illumination, contrast, shadow structure, contamination, and partial occlusion vary across locations and operational phases. If adaptation entangles task semantics and environment-specific corrections into one monolithic module, capability reuse becomes inefficient and brittle.

### 4.4 Hypotheses
- **H1**: Factorized task and environment adaptation improves unseen-combination performance over non-factorized baselines.
- **H2**: Explicit task/environment routing yields more interpretable and controllable specialization than monolithic adaptation.
- **H3**: A small but operationally meaningful taxonomy of lunar-analog visual conditions is sufficient to expose environment-sensitive failure modes.
- **H4**: A missing-corner evaluation protocol is sufficient to determine whether the factorization assumption actually holds.

### 4.5 Core contributions
1. A real-robot factorized adaptation benchmark for lunar-analog visual conditions.
2. A task × environment compositional generalization protocol using unseen combination evaluation.
3. A factorized adaptation framework with separate task and environment specialization.
4. Diagnostics for determining whether the factorization assumption holds or fails.

### 4.6 Environment taxonomy
The paper should define a compact and operationally meaningful environment taxonomy, for example:
- **E0 — nominal laboratory condition**
- **E1 — hard-shadow / low-angle illumination condition**
- **E2 — partial contamination / occlusion / appearance perturbation condition**

These should be presented as **lunar-analog visual factors**, not as claims of full lunar environmental fidelity.

### 4.7 Missing-corner protocol
This should remain the scientific centerpiece of the paper:
- train on a subset of task-environment cells,
- ensure every task and every environment appears in training,
- evaluate on unseen task-environment combinations,
- repeat with multiple rotated splits.

The point of this protocol is not just to create a hard benchmark, but to test whether **factorized reuse** actually exists.

### 4.8 Baseline classes
The paper should compare against:
- full adaptation,
- single merged adaptation,
- domain-randomized non-factorized adaptation,
- adapter-composition baselines,
- parameter-matched merged PEFT baselines.

### 4.9 Diagnostics and falsifiability
The paper must explicitly allow the factorization hypothesis to fail. It should include:
- seen vs unseen gap analysis,
- cross-condition breakdowns,
- routing interaction analysis,
- task-env entanglement diagnostics.

### 4.10 Claim boundary
A safe claim form is:

> “We study factorized adaptation under lunar-analog environment shifts and evaluate compositional generalization to unseen task-environment combinations on a real mobile manipulation platform.”

The paper should **not** claim that it proves strict orthogonal disentanglement between task and environment factors.

---

## 5. Shared benchmark narrative

### 5.1 Why these tasks are space-relevant
The selected tasks should be tied to operational categories such as:
- logistics and offloading,
- mechanical securing and interface actuation,
- surface maintenance and dust mitigation,
- utility or instrument connection.

### 5.2 Why these tasks remain compatible with Earth-trained priors
A key part of the narrative is that the proposed tasks are **not motorically alien**. They are semantically adjacent to terrestrial manipulation primitives that already appear in modern robot datasets. The novelty lies in:
- mission framing,
- continual task emergence,
- and deployment under lunar-analog condition shifts.

This justifies the broader Earth-to-Space reuse thesis.

---

## 6. Related-work structure

The paper-facing writeup should organize the literature into four groups:
1. **VLA and generalist robot policies**
2. **Continual / lifelong robot learning**
3. **Compositional adaptation and modular PEFT**
4. **Space / lunar operational motivation**

Do **not** mention repository lineage, training stack provenance, code reuse, file names, hooks, or implementation substrate in this document.

---

## 7. Claim discipline

### 7.1 Safe phrasing for SpaceCIL
Prefer:
- “real-robot continual specialization”
- “operationally ordered task stream”
- “released pi0.5-class VLA backbone”
- “mobile manipulation platform”

Avoid:
- “first continual VLA”
- “first lifelong mobile manipulation”
- “solves catastrophic forgetting in VLA systems”

### 7.2 Safe phrasing for LunarCompose
Prefer:
- “factorized adaptation under lunar-analog shifts”
- “unseen task-environment composition”
- “missing-corner real-robot protocol”

Avoid:
- “strict disentanglement”
- “solves lunar domain adaptation”
- “proves task-environment independence”

---

## 8. Programmatic significance

Taken together, the two papers support the following larger thesis:

> Space robotics should be approached as a problem of **capability reuse and expansion from Earth-trained embodied intelligence**, rather than repeated from-scratch policy construction.

This is the correct bridge to a future dissertation-level narrative, but it should remain a **program-level framing**, not a third paper-level claim.

---

## 9. Verified reference list

### 9.1 VLA and generalist policy references
1. **Black et al. (2025)**, *pi0.5: a Vision-Language-Action Model with Open-World Generalization*.  
   ArXiv: https://arxiv.org/abs/2504.16054
2. **Black et al. (2024)**, *pi0: A Vision-Language-Action Flow Model for General Robot Control*.  
   ArXiv: https://arxiv.org/abs/2410.24164
3. **Kim et al. (2024)**, *OpenVLA: An Open-Source Vision-Language-Action Model*.  
   ArXiv: https://arxiv.org/abs/2406.09246
4. **Finetuning study / OpenVLA-OFT (2025)**, *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*.  
   ArXiv: https://arxiv.org/abs/2502.19645
5. **RoboVLMs (2024)**, *What Matters in Building Vision-Language-Action Models for Generalist Robots*.  
   ArXiv: https://arxiv.org/abs/2412.14058
6. **FTM / FLA paper (2025)**, *VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling*.  
   ArXiv: https://arxiv.org/abs/2512.02902

### 9.2 Continual / lifelong robot learning references
7. **Liu et al. (2023)**, *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*.  
   ArXiv: https://arxiv.org/abs/2306.03310
8. **Lei et al. (2025)**, *Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning*.  
   ArXiv: https://arxiv.org/abs/2506.05985
9. **Xu and Nie (2025)**, *SPECI: Skill Prompts based Hierarchical Continual Imitation Learning for Robot Manipulation*.  
   ArXiv: https://arxiv.org/abs/2504.15561
10. **Wu et al. (2025)**, *Continually Evolving Skill Knowledge in Vision Language Action Model*.  
    ArXiv: https://arxiv.org/abs/2511.18085
11. **Task-agnostic lifelong adaptation (2024)**, *Task-agnostic Lifelong Robot Learning with Retrieval-based Weighted Local Adaptation*.  
    ArXiv: https://arxiv.org/abs/2410.02995
12. **Primitive Prompt Learning (2025)**, *Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation*.  
    ArXiv: https://arxiv.org/abs/2504.00420

### 9.3 Data and transfer ecosystem references
13. **Open X-Embodiment (2023)**, *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*.  
    ArXiv: https://arxiv.org/abs/2310.08864
14. **Khazatsky et al. (2024)**, *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*.  
    ArXiv: https://arxiv.org/abs/2403.12945
15. **Mobile ALOHA (2024)**, *Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation*.  
    ArXiv: https://arxiv.org/abs/2401.02117
16. **ALOHA / ACT (2023)**, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.  
    ArXiv: https://arxiv.org/abs/2304.13705

### 9.4 Compositionality and modularity references
17. **Mendez et al. (2022)**, *CompoSuite: A Compositional Reinforcement Learning Benchmark*.  
    ArXiv: https://arxiv.org/abs/2207.04136
18. **Ilharco et al. (2022)**, *Editing Models with Task Arithmetic*.  
    ArXiv: https://arxiv.org/abs/2212.04089
19. **Zhang et al. (2023)**, *Composing Parameter-Efficient Modules with Arithmetic Operations*.  
    ArXiv: https://arxiv.org/abs/2306.14870

### 9.5 Official mission and operations references (non-ArXiv)
20. **NASA lunar south-pole lighting study**: low sun angle and extreme shadows.  
    https://ntrs.nasa.gov/citations/20240011393
21. **NASA Lunar Dust Mitigation Roadmap (2024)**.  
    https://ntrs.nasa.gov/api/citations/20240013978/downloads/NASA%20Lunar%20Dust%20Mitigation%20Roadmap%20Fall%202024.pdf
22. **NASA Autonomous Utility Connector for Lunar Surface Systems**.  
    https://techport.nasa.gov/projects/8720
23. **ESA lunar south pole operational context**.  
    https://www.esa.int/Science_Exploration/Human_and_Robotic_Exploration/Lunar_Lander/Exploring_the_lunar_South_Pole
