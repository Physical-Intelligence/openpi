# Labs

Each `labXX_*.py` corresponds to a course module. They contain `TODO(you)` stubs and
self-checking `check_*()` functions. The "answer key" is the real `openpi` source — the
checks import the real classes and compare your output to theirs where feasible.

## Running

From the repo root, with the project env active:

```bash
uv run python course/labs/lab01_data.py
uv run python course/labs/lab03_transformer.py
# ...etc
```

A lab "passes" when its `check_*()` functions print `OK` (or, for the discussion/stretch
parts, when you've implemented and tested the described behavior yourself).

## Order

Do them in module order — later labs reuse components you built earlier:

| Lab | Module | Builds |
|-----|--------|--------|
| `lab00_orientation.py` | 00 | (exploration only) |
| `lab01_data.py` | 01 | quantile norm, fake episode → Observation |
| `lab02_prefix.py` | 02 | mini-SigLIP, `embed_prefix` |
| `lab03_transformer.py` | 03 | `make_attn_mask`, two-expert block |
| `lab04_flow_head.py` | 04 | `posemb_sincos`, flow targets, `embed_suffix`, loss |
| `lab05_sampling.py` | 05 | Euler sampler, KV cache |
| `lab06_pi05.py` | 06 | discrete state tokens, adaRMS |
| `lab07_training.py` | 07 | optimizer, `train_step`, EMA, freezing |

## Capstone

[`minipi05/`](minipi05/) is the skeleton for the Module 08 capstone. It is intentionally
mostly empty — you assemble it from your lab solutions. Run its test with:

```bash
uv run python course/labs/minipi05/test_against_openpi.py
```

## Notes

- Several labs cross-check against `openpi` functions that need **no** network
  (`make_attn_mask`, `posemb_sincos`, quantile norm, `np.digitize`). Anything touching the
  PaliGemma tokenizer or real checkpoints needs `gs://` access; those checks degrade
  gracefully / are marked optional.
- Keep models tiny (the `dummy` Gemma variant, downsampled images) so everything runs on
  CPU in seconds.
