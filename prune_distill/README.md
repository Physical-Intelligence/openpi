# Prefix Distillation

This directory contains a low-memory distillation path that trains `gemma_prune` from the frozen `gemma_2b` prefix branch used by the LIBERO `pi05` checkpoint.

The runner keeps memory down by:
- freezing SigLIP
- freezing the teacher Gemma-2B
- dropping the action branch from the distillation graph
- mixing real LIBERO batches with fake random batches

The loss is a weighted sum of:
- hidden-state MSE on all valid prefix tokens
- cosine distance on the same tokens

The student is warm-started from the teacher by:
- copying the full token embedder and final norm
- copying the first 14 attention and norm layers
- slicing the teacher MLP weights down to the pruned hidden size

Run it from the repo root:

```bash
.venv/bin/python prune_distill/train_prefix_distill.py --exp-name gemma_prune_prefix --overwrite
```

TensorBoard logs are written to `checkpoints/prune_distill/<exp_name>/tensorboard`.

```bash
.venv/bin/python -m tensorboard.main --logdir checkpoints/prune_distill
```

Useful flags:

```bash
.venv/bin/python prune_distill/train_prefix_distill.py \
  --exp-name gemma_prune_prefix \
  --batch-size 8 \
  --num-train-steps 10000 \
  --real-batch-prob 0.8 \
  --log-interval 20 \
  --save-interval 500 \
  --overwrite
```

Resume from the latest saved `step_*` checkpoint:

```bash
.venv/bin/python prune_distill/train_prefix_distill.py \
  --exp-name gemma_prune_prefix \
  --num-train-steps 10000 \
  --resume
```
