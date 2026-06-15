# Building π₀.₅ From Scratch — A Rigorous VLA Course

A self-paced course that takes you from "I understand JAX, transformers, and flow
policies" to "I can implement a π₀.₅-class vision-language-action (VLA) model from
scratch and explain every design decision." The course is built **around the real
`openpi` implementation in this repository** — every concept is anchored to specific
files and line numbers so the theory and the code never drift apart.

The reference model is **π₀.₅** (`Pi0Config(pi05=True)`), Physical Intelligence's
flow-matching VLA with open-world generalization. We use π₀ as the pedagogical
stepping stone, since π₀.₅ is best understood as a small set of deltas on top of π₀.

## What you will be able to do at the end

- Explain, derive, and implement each component of a VLA: vision encoder → multimodal
  token sequence → mixture-of-experts transformer → flow-matching action head.
- Implement the π₀.₅ training objective (conditional flow matching) and its inference
  procedure (ODE integration with a KV cache) yourself, in JAX/Flax.
- Articulate exactly how π₀.₅ differs from π₀ and π₀-FAST, and *why* (adaRMS time
  conditioning, discrete state tokens, knowledge insulation).
- Assemble a minimal but faithful π₀.₅ in a single small package and train it on a toy
  dataset.

## How this course is structured

Per the agreed design, each module has three parts:

1. **Lecture notes** — brief refreshers on prerequisites you already know, then deep
   VLA-specific theory. Math is rigorous but assumes comfort with flow matching,
   attention, and JAX.
2. **Code walkthrough** — a guided read of the corresponding `openpi` source, with
   exact `file:line` references.
3. **Build lab** — a stub file in [`labs/`](labs/) you complete yourself. Each lab
   tells you precisely what to implement and how to check it against the openpi
   reference (the "answer key" is the repo itself). You run training/experiments on
   your own hardware.

> Convention used throughout: code references look like
> [`pi0.py:189`](../src/openpi/models/pi0.py). Paper references look like
> *(π₀.₅ paper §3.1)* and point at the PDFs in this folder.

> **Teaching this course?** If you're an AI agent or human TA helping someone through it,
> read [`INSTRUCTOR.md`](INSTRUCTOR.md) first — it has the pedagogy, per-module talking
> points, misconceptions to pre-empt, and a grading rubric with verified answer-key facts.

## Primary sources (in this folder)

- [`2410.24164v1.pdf`](2410.24164v1.pdf) — **π₀: A Vision-Language-Action Flow Model
  for General Robot Control** (Black et al., 2024). The architecture bible.
- [`2504.16054v1.pdf`](2504.16054v1.pdf) — **π₀.₅: a Vision-Language-Action Model with
  Open-World Generalization** (Black et al., 2025). Co-training, discrete-state, and
  knowledge insulation.

See [`reading_guide.md`](reading_guide.md) for a paper-section → code-file map.

## Module roadmap

| #  | Module | Core question | Key openpi files |
|----|--------|---------------|------------------|
| 00 | [Orientation](00_orientation.md) | What *is* a VLA, and what is the shape of π₀.₅? | `models/`, `README.md` |
| 01 | [Data & representations](01_data_and_representations.md) | How does a robot episode become model tensors? | `model.py`, `transforms.py`, `training/config.py` |
| 02 | [Vision-language backbone](02_vision_language_backbone.md) | How do images + text become a token sequence? | `siglip.py`, `gemma.py` (Embedder), `pi0.py::embed_prefix` |
| 03 | [The MoE transformer](03_transformer_moe_experts.md) | How do two experts share one attention stack? | `gemma.py` |
| 04 | [Flow-matching action head](04_flow_matching_action_head.md) | How are continuous actions modeled & trained? | `pi0.py::embed_suffix`, `compute_loss` |
| 05 | [Inference: ODE + KV cache](05_inference_flow_and_kvcache.md) | How do we sample an action chunk fast? | `pi0.py::sample_actions` |
| 06 | [π₀.₅ & knowledge insulation](06_pi05_and_knowledge_insulation.md) | What makes π₀.₅ ≠ π₀, and why? | `pi0_config.py`, `tokenizer.py`, `gemma.py::RMSNorm` |
| 07 | [Training at scale](07_training_at_scale.md) | Optimizer, EMA, freezing/LoRA, sharding, configs. | `scripts/train.py`, `training/` |
| 08 | [Capstone: minimal π₀.₅](08_capstone_minimal_pi05.md) | Put it all together from scratch. | `labs/` |

## Suggested pace

Roughly one module per focused study session. Do not skip the labs — the entire point
of the course is that you can reproduce the model, not just read about it. A realistic
schedule is 2–4 weeks part-time. Module 08 (capstone) is itself ~a week.

## Working environment

- The labs target small, CPU/single-GPU-friendly configs (e.g. the `dummy` Gemma
  variant in [`gemma.py:60`](../src/openpi/models/gemma.py)) so you can iterate fast.
- For real fine-tuning runs, use your Babel cluster setup and record exact commands in
  `COMMANDS.md` at the repo root (your established convention).
- Verify your lab solutions against openpi by importing the real classes and comparing
  shapes/outputs on `fake_obs()` / `fake_act()`.

Start with [Module 00](00_orientation.md).
