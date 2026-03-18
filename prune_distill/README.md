# Prefix Distillation

This directory contains a low-memory distillation path that trains `gemma_prune` from the frozen `gemma_2b` prefix branch used by the LIBERO `pi05` checkpoint.

The runner keeps memory down by:
- freezing SigLIP
- freezing the teacher Gemma-2B
- dropping the action branch from the distillation graph
- training only the pruned student on a local LeRobot-format dataset at `/root/flatten_fold_v2`
- reusing the LIBERO normalization stats from `/root/pi_train/pi05_libero/assets`
- limiting the loaded training subset by default to about `50_000` examples

The loss is a weighted sum of:
- hidden-state MSE on all valid prefix tokens
- cosine distance on the same tokens

The student is warm-started from the teacher by:
- copying the full token embedder and final norm
- copying the first 14 attention and norm layers
- slicing the teacher MLP weights down to the pruned hidden size

Run it from the repo root:

```bash
.venv/bin/python prune_distill/train_prefix_distill.py \
  --exp-name gemma_prune_prefix \
  --dataset-path /root/flatten_fold_v2 \
  --max-examples 50000 \
  --overwrite
```

TensorBoard logs are written to `checkpoints/prune_distill/<exp_name>/tensorboard`.

```bash
.venv/bin/python -m tensorboard.main --logdir checkpoints/prune_distill
```

Useful flags:

```bash
.venv/bin/python prune_distill/train_prefix_distill.py \
  --exp-name gemma_prune_prefix \
  --dataset-path /root/flatten_fold_v2 \
  --batch-size 8 \
  --max-examples 50000 \
  --num-train-steps 10000 \
  --log-interval 20 \
  --save-interval 500 \
  --overwrite
```

If you still hit disk pressure during dataset materialization, lower the subset further:

```bash
.venv/bin/python prune_distill/train_prefix_distill.py \
  --dataset-path /root/flatten_fold_v2 \
  --max-examples 10000
```

Resume from the latest saved `step_*` checkpoint:

```bash
.venv/bin/python prune_distill/train_prefix_distill.py \
  --exp-name gemma_prune_prefix \
  --dataset-path /root/flatten_fold_v2 \
  --num-train-steps 10000 \
  --resume
```

## PI0.5 Sensitivity Analysis

Use the sensitivity analyzer to score which `pi05` student tensors are most fragile for quantization, pruning, and distillation drift.

It will:
- rank student layers by distillation gradient / Taylor sensitivity
- run fake-quant perturbations on the top-ranked tensors
- run magnitude-prune perturbations on the same tensors
- save `summary.json`, `candidate_scores.csv`, and `family_summary.csv`

Example:

```bash
.venv/bin/python prune_distill/analyze_pi05_sensitivity.py \
  --exp-name pi05_sensitivity \
  --dataset-path /liujinxin/ZZF/openpi/datasets/piper/flatten_fold_v2 \
  --student-checkpoint /root/openpi-wr/checkpoints/prune_distill/gemma_prune_prefix/step_0072001/student \
  --max-examples 2048 \
  --max-batches 4 \
  --eval-top-k 24 \
  --quant-bits 8 4 \
  --prune-ratios 0.1 0.3 0.5 \
  --overwrite
```

Results are written to `checkpoints/prune_distill/sensitivity/<exp_name>`.

## PI0.5 Benchmark

Use the benchmark runner to evaluate:
- the original full `pi05` checkpoint on offline datasets
- the distilled pruned prefix checkpoint on offline teacher-agreement metrics
- the original full `pi05` checkpoint on real LIBERO rollout success

The current distilled student checkpoint is only a pruned prefix model, so it does not support LIBERO action rollouts yet.

Offline benchmark on a custom dataset:

```bash
.venv/bin/python prune_distill/benchmark_pi05_models.py \
  --exp-name pi05_dataset_benchmark \
  --origin-checkpoint-dir /root/pi_train/pi05_libero \
  --pruned-student-checkpoint /root/openpi-wr/checkpoints/prune_distill/gemma_prune_prefix/step_0072001/student \
  --dataset-path /liujinxin/ZZF/openpi/datasets/piper/flatten_fold_v2 \
  --max-examples 50000 \
  --max-eval-examples 256 \
  --overwrite
```

LIBERO rollout success for the original full `pi05` checkpoint:

```bash
.venv/bin/python prune_distill/benchmark_pi05_models.py \
  --exp-name pi05_libero_rollout \
  --origin-checkpoint-dir /root/pi_train/pi05_libero \
  --no-run-dataset-benchmark \
  --run-libero-rollout \
  --libero-task-suite-names libero_spatial libero_object libero_goal libero_10 \
  --libero-num-trials-per-task 10 \
  --overwrite
```

Benchmark outputs are written to `checkpoints/prune_distill/benchmark/<exp_name>`.

If the pruned prefix benchmark hits JAX GPU OOM during model init, rerun it on CPU:

```bash
JAX_PLATFORMS=cpu .venv/bin/python prune_distill/benchmark_pi05_models.py \
  --exp-name pi05_dataset_benchmark_cpu \
  --origin-checkpoint-dir None \
  --pruned-student-checkpoint /root/openpi-wr/checkpoints/prune_distill/gemma_prune_prefix/step_0072001/student \
  --dataset-path /liujinxin/ZZF/openpi/datasets/piper/flatten_fold_v2 \
  --max-eval-examples 64 \
  --overwrite
```
