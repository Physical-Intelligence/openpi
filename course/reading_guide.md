# Reading Guide — Papers ↔ Code ↔ Modules

A cross-reference so you can move fluidly between the two PDFs in this folder, the
`openpi` source, and the course modules. Read each paper *after* the corresponding
module — you'll extract far more once the code is in your head.

## The two papers

- **π₀** — [`2410.24164v1.pdf`](2410.24164v1.pdf), *"π₀: A Vision-Language-Action Flow
  Model for General Robot Control"* (Black et al., Oct 2024).
- **π₀.₅** — [`2504.16054v1.pdf`](2504.16054v1.pdf), *"π₀.₅: a Vision-Language-Action
  Model with Open-World Generalization"* (Black et al., Apr 2025).

The **knowledge insulation** paper (referenced by the repo README as
`physicalintelligence.company/research/knowledge_insulation`) is the training-recipe
companion to π₀.₅; its ideas are summarized in Module 06.

## Concept → where to read it

| Concept | Paper §(approx.) | openpi code | Module |
|---------|------------------|-------------|--------|
| VLA problem setup, action chunking | π₀ §1–2 | `model.py` (Observation/Actions) | 00, 01 |
| PaliGemma / SigLIP backbone | π₀ §3 | `siglip.py`, `gemma.py::Embedder` | 02 |
| Multimodal prefix construction | π₀ §3 | `pi0.py::embed_prefix` | 02 |
| Two-expert ("mixture") transformer | π₀ §3 (action expert) | `gemma.py` (multi-config attention) | 03 |
| Prefix-LM / blockwise attention | π₀ §3 | `pi0.py::make_attn_mask` | 03 |
| Flow-matching action objective | π₀ §3 (training) | `pi0.py::compute_loss` | 04 |
| Time conditioning (concat-MLP) | π₀ §3 | `pi0.py::embed_suffix` (pi05=False) | 04 |
| Flow inference / Euler integration | π₀ §3 | `pi0.py::sample_actions` | 05 |
| KV cache for fast control | (impl detail) | `pi0.py::sample_actions`, `gemma.py` KVCache | 05 |
| Discrete state tokens | π₀.₅ §3 | `tokenizer.py::PaligemmaTokenizer` | 06 |
| adaRMS time conditioning | π₀.₅ / KI | `gemma.py::RMSNorm`, `pi0.py::embed_suffix` (pi05=True) | 06 |
| Co-training on heterogeneous data | π₀.₅ §3–4 | (data configs) `training/config.py` | 06 |
| High-level subtask + low-level action | π₀.₅ §3 | (not in open training loop) | 06 |
| Knowledge insulation (dual objective) | KI paper | (partial in repo; flow head only) | 06 |
| Optimizer / schedule / EMA | (impl) | `optimizer.py`, `scripts/train.py` | 07 |
| LoRA / freezing | (impl) | `lora.py`, `pi0_config.py::get_freeze_filter` | 07 |
| FSDP sharding | (impl) | `training/sharding.py` | 07 |
| Autoregressive FAST alternative | FAST paper | `pi0_fast.py`, `tokenizer.py::FASTTokenizer` | 04 (contrast), 08 |

## Reading order

1. Module 00 → skim π₀ §1–2 (motivation, problem).
2. Modules 01–05 → read π₀ §3 in full; you'll recognize every component.
3. Module 06 → read π₀.₅ §1–4 + the knowledge-insulation blog/paper.
4. Module 07–08 → back to the code; the papers are now reference, not mystery.

## Convention reminders that trip people up

- **Time direction**: in `openpi` code, `t=1` is noise and `t=0` is data — *opposite*
  of the π₀ paper's text. (`pi0.py:226` says so explicitly.)
- **`pi05` is a flag on one class** (`Pi0`), not a separate model file.
- **openpi ships the flow head only** for π₀.₅ training/inference, even though the full
  paper recipe also uses a discrete action loss + gradient insulation.
- **The model's contract ends at normalized actions**; the transform shell handles
  physical units, keys, and robot-specific adaptation.
