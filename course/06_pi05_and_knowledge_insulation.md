# Module 06 — π₀.₅ Specifics & Knowledge Insulation

You now understand π₀ completely. π₀.₅ is π₀ plus a small set of deliberate changes aimed
at **open-world generalization** — performing tasks in homes it has never seen. This
module isolates exactly what changes, in code and in concept, and explains the training
recipe (knowledge insulation) that the paper introduces. Primary source:
[`2504.16054v1.pdf`](2504.16054v1.pdf).

## Learning objectives

- Enumerate the three code-level differences `pi05=True` toggles, and the reasoning.
- Implement the discrete-state tokenizer and adaRMS time conditioning.
- Explain co-training on heterogeneous data (robot + web + high-level subtasks).
- Explain **knowledge insulation**: what is insulated from what, and why it both speeds
  training and preserves language-following.
- Know precisely what `openpi` implements vs. what the paper describes.

---

## 1. The three code-level deltas (`pi05=True`)

From [`pi0_config.py:28`](../src/openpi/models/pi0_config.py) and the branches in
`pi0.py`:

### (a) State as discrete language tokens, not a continuous suffix token
- **π₀**: continuous state token in the suffix via `state_proj`
  ([`pi0.py:151`](../src/openpi/models/pi0.py)).
- **π₀.₅**: state is **discretized and written into the prompt string**, then tokenized
  as ordinary language ([`tokenizer.py:22`](../src/openpi/models/tokenizer.py)):
  ```python
  discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1   # 256 bins
  full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
  tokens = tokenizer.encode(full_prompt, add_bos=True)
  ```
  This is why π₀.₅ uses **quantile normalization** (state must be in `[-1,1]` before
  binning, Module 01) and a larger `max_token_len` (200 vs 48,
  [`pi0_config.py:38`](../src/openpi/models/pi0_config.py)).

  *Why?* Folding state into the shared, pretrained language stream lets the powerful VLM
  expert reason jointly over task + state + vision with its full capacity, rather than
  bottlenecking proprioception through a single learned-from-scratch token. It also
  unifies the input format with the discrete action head used during pretraining (§3).

### (b) adaRMS time conditioning instead of concat-MLP
- **π₀**: timestep concatenated with action features, MLP-mixed
  ([`pi0.py:170`](../src/openpi/models/pi0.py)).
- **π₀.₅**: timestep → small MLP → `adarms_cond`, injected as scale/shift/gate in **every
  RMSNorm of the action expert** ([`pi0.py:162`](../src/openpi/models/pi0.py),
  [`gemma.py:127`](../src/openpi/models/gemma.py)). This is exactly DiT/FiLM-style
  conditioning and is enabled by `adarms=config.pi05` with `use_adarms=[False, True]`
  ([`pi0.py:77`](../src/openpi/models/pi0.py),
  [`pi0.py:80`](../src/openpi/models/pi0.py)) — only the action expert (expert 1) gets
  adaRMS; the VLM never does.

  *Why?* Modulating the normalization throughout the network is a stronger, more uniform
  way to tell every layer "how noisy is this input," compared to mixing time into the
  input features once. It's the conditioning mechanism that's proven best for
  flow/diffusion transformers.

### (c) Token budget & defaults
`max_token_len` 200, `discrete_state_input=True`, `model_type=PI05`
([`pi0_config.py:37`](../src/openpi/models/pi0_config.py)). The `ModelTransformFactory`
PI05 branch passes `discrete_state_input` to the tokenizer
([`config.py:127`](../src/openpi/training/config.py)).

> Everything else — SigLIP, the two-expert transformer, the flow-matching loss, the
> Euler sampler, the KV cache — is **identical** to π₀. That's the payoff of reading π₀
> first: π₀.₅ is a focused diff.

---

## 2. Co-training on heterogeneous data (paper §3, the generalization story)

π₀.₅'s open-world ability comes mostly from **data**, not architecture. It co-trains a
single model on a mixture:

- **Mobile manipulation** robot demos (the target embodiment).
- **Other robot data** (diverse embodiments / static manipulators).
- **Web / multimodal data**: image captioning, VQA, object detection/grounding,
  high-level "what subtask should I do next" prediction.

Crucially, π₀.₅ predicts at **two levels**: a high-level *semantic subtask* ("pick up the
plate") and the low-level *action chunk* — both from the same model. The language
prediction of subtasks is what keeps the VLM's semantic knowledge engaged and
transferable to novel scenes. The figure on page 1 of the PDF ("Close the microwave" →
subtask commands → robot actions) is the one-image summary.

The takeaway for *you as a builder*: a VLA's generalization budget is dominated by the
breadth of co-training data and the preservation of VLM semantics — not by clever action
heads. Architecture buys efficiency; data buys generalization.

---

## 3. Knowledge insulation (the KI paper — the training recipe)

This is the conceptually richest idea, and the part `openpi`'s released training only
partially implements. The problem it solves:

> When you attach a **flow-matching action expert** to a pretrained VLM and train
> end-to-end, gradients from the noisy continuous action loss flow back into the VLM and
> **degrade its pretrained representations and language-following** — and flow-matching
> gradients are also slow/noisy to train the VLM with.

**Knowledge insulation** (KI) resolves this with two coupled ideas:

1. **Train the VLM backbone with a *discrete* action objective, not the flow loss.**
   Actions are *also* tokenized (FAST-style discrete tokens) and the VLM is trained with
   a standard next-token cross-entropy on them — the same kind of signal it was
   pretrained with. This is a clean, well-conditioned gradient that *adapts* the VLM to
   actions while keeping language intact. (See the discrete prompt format
   `Task: ... State: ... Action:` shared by the tokenizers,
   [`tokenizer.py:28`](../src/openpi/models/tokenizer.py),
   [`tokenizer.py:74`](../src/openpi/models/tokenizer.py).)

2. **Insulate the VLM from the flow expert's gradients.** The flow-matching action
   expert still trains (it's what you use at inference for smooth continuous control),
   but its gradients are **stopped from flowing into the VLM backbone** — the action
   expert attends to the VLM's representations (forward pass) but does not get to rewrite
   them (backward pass). The two-expert architecture (Module 03) is precisely what makes
   this surgical: separate weights, shared attention.

The result: the VLM keeps its web-scale knowledge and language-following, gets adapted to
the action domain via a stable discrete loss, and the flow expert delivers fast smooth
continuous actions at inference — best of both. KI also **speeds up training** because
the backbone isn't being dragged by the high-variance flow gradient.

### What `openpi` actually ships
Per this repo's README: *"in this repository, we currently only support the flow matching
head for both π₀.₅ training and inference."* So:
- ✅ Implemented: the flow-matching expert, adaRMS, discrete **state** tokens, the π₀.₅
  inference path. The released `pi05_base` checkpoint **was** trained with the full KI
  recipe.
- ⚠️ Not in the open training loop: the simultaneous discrete **action** AR loss + the
  gradient-insulation `stop_gradient` plumbing. Fine-tuning here uses the flow loss only
  (often with the VLM partially frozen / LoRA, which is itself a crude form of
  insulation — Module 07).

> When you build your own, you can choose your insulation strength: full freeze of the
> VLM, LoRA on the VLM, or the proper KI dual-objective. The lab implements the discrete
> state + adaRMS pieces and discusses where a `stop_gradient` would go to approximate KI.

---

## 4. Putting the diff in a table

| Aspect | π₀ | π₀.₅ |
|--------|----|----|
| State input | continuous suffix token (`state_proj`) | discretized into prompt language tokens |
| Time conditioning | concat + MLP into action features | adaRMS scale/shift/gate in action expert |
| Normalization | z-score | quantile (→ `[-1,1]` for binning) |
| `max_token_len` | 48 | 200 |
| Backbone training signal | flow loss end-to-end | (KI) discrete action CE on VLM + insulated flow expert |
| Data | robot demos | + web/VQA/grounding + high-level subtask prediction |
| Goal | general robot control | **open-world** generalization |

---

## Self-check

1. Show the exact transformation from a normalized state vector to prompt tokens in
   π₀.₅. Why must the state be in `[-1,1]` first?
2. Which expert gets `adarms_cond`, and where inside the block does it act?
3. State the two halves of knowledge insulation and the problem each solves.
4. Why does the two-expert architecture make gradient insulation easy/surgical?
5. What does `openpi` implement of the full π₀.₅ recipe, and what does it omit?
6. Is π₀.₅'s open-world generalization primarily an architecture win or a data win?

## Lab 06

Open [`labs/lab06_pi05.py`](labs/lab06_pi05.py). You will:
1. Implement `tokenize(prompt, state)` (the π₀.₅ discrete-state path) and verify token
   ids match `PaligemmaTokenizer.tokenize` for the same inputs.
2. Implement adaRMS (`RMSNorm` with a conditioning vector → scale/shift/gate) and confirm
   that with `cond=None` it reduces exactly to standard RMSNorm.
3. Build `Pi0Config(pi05=True)` and `pi05=False` toy models and diff their suffix:
   confirm the π₀.₅ suffix has H tokens with non-None `adarms_cond`, while π₀ has H+1
   tokens (state) with `adarms_cond=None`.
4. (Discussion + stub) Mark in your `compute_loss` where a `stop_gradient` on the prefix
   representations would approximate knowledge insulation, and explain the consequence.

Next: [Module 07 — Training at scale](07_training_at_scale.md).
