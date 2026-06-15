# Cluster runbook — continual YAM finetune (for the on-cluster agent)

You are an agent running on the **CMU Babel** cluster in the `openpi` repo. Your job:
**run the verification tests in order, report results, and—only if they pass—launch the final
continual-training batch job.** Do not skip the tests. Report back after each phase.

## What this change does

`examples/babel/train_foldclo_loravlm.sbatch` was rewritten to **continually finetune** one
pi05 LoRA-VLM model across every dataset in `examples/babel/converted_yam_data.txt`, in order.
Per dataset it: waits until the dataset is on HF (`huggingface.co/leokswang`), computes its
norm stats (downloading it to `/scratch/$USER/lerobot`), trains 2000 more steps continuing the
same model, then deletes the local copy. It emails (`--mail-type`) on END/FAIL/TIME_LIMIT.

Code changes that back it:
- `scripts/compute_norm_stats.py` — new `--repo-id` override (compute stats for any dataset).
- `src/openpi/training/config.py` — new `TrainConfig.max_to_keep` field; the
  `pi05_BiYAMMolmoAct2_loravlm` config now uses `max_to_keep=3`, `keep_period=None`
  (keep only the 3 most-recent checkpoints).
- `src/openpi/training/checkpoints.py` + `scripts/train.py` — thread `max_to_keep` through.

## Preconditions (check first, report if any fail)

1. You are on Babel and can `sbatch`. `cd` to the repo root (where `examples/` lives).
2. Env: `export HF_TOKEN=hf_...` (account `leokswang`) and `export WANDB_API_KEY=...`
   (or `export WANDB_MODE=offline`). Confirm `uv` is available
   (`source ~/.local/bin/env` if needed).
3. `$USER` should be your Babel id (e.g. `leokingw`). `/scratch/$USER` may not exist yet —
   that's expected; the sbatch and tests create it with `mkdir -p`.
4. Confirm the base weights referenced by the config's `weight_loader` exist and are readable:
   `/data/user_data/leokingw/openpi_checkpoints/pi05_foldclo_babel/foldclo_prompts_4gpu/4999/params`
   (stat the full path to trigger the AutoFS mount). If missing, **stop and report** — the
   first dataset's `--overwrite` run needs it.

## Phase 1 — login-node tests (fast, no GPU)

Run from the repo root:

```bash
bash examples/babel/tests/01_check_code.sh        # code + config wiring (no network)
bash examples/babel/tests/02_dataset_exists.sh    # HF existence gate (network)
bash examples/babel/tests/04_not_found_gate.sh    # wait-and-retry gate keeps polling (network)
```

Expected: each prints `TEST n PASS`. **Report the final PASS/FAIL line of each.** If any fail,
stop and paste the output.

## Phase 2 — norm-stats + download test (CPU, network, ~one dataset of /scratch)

```bash
bash examples/babel/tests/03_norm_stats_override.sh
```

This downloads `leokswang/18012026-block-13_lerobot_format` to `/scratch/$USER/lerobot` and
writes `assets/pi05_BiYAMMolmoAct2_loravlm/leokswang/18012026-block-13_lerobot_format/norm_stats.json`.
Expected: `TEST 3 PASS`. Afterward reclaim space:
`rm -rf /scratch/$USER/lerobot/leokswang/18012026-block-13_lerobot_format`.
**Report PASS/FAIL.**

## Phase 3 — GPU smoke run of the real sbatch (1 GPU, ~20–40 min)

Submit a 2-dataset, 20-step-each run of the **actual** training script (via env overrides):

```bash
bash examples/babel/tests/05_smoke_submit.sh      # prints a <jobid>
squeue --me
tail -f yam_smoke-<jobid>.out                      # watch until it ends
```

When it finishes, check the result (run where `/data/user_data/$USER` is visible):

```bash
bash examples/babel/tests/06_check_smoke.sh <jobid>
```

Expected `SMOKE CHECK PASS`, meaning: dataset 1 trained with `--overwrite` (steps→20),
dataset 2 **resumed** and continued (steps→40), ≤3 checkpoints kept, and `/scratch` was
cleaned between datasets. Also confirm you received the **SLURM email** for the smoke job
(check spam). **Report:** the jobid, the `SMOKE CHECK` line, and whether the email arrived.

If the smoke run OOMs or the partition won't grant the GPU, retry with overrides, e.g.
`PARTITION=general GRES=gpu:1 TIME=00:40:00 EXTRA_TRAIN_ARGS="--batch-size=4" bash examples/babel/tests/05_smoke_submit.sh`.

## Checkpoint: report before launching

Summarize Phases 1–3 (each test's PASS/FAIL + the smoke jobid + email yes/no). **Only proceed
to Phase 4 if everything passed.** If anything failed, stop and report details.

## Phase 4 — launch the final continual run

This trains through **all** datasets in `examples/babel/converted_yam_data.txt` in order,
2000 steps each, waiting when it catches up to the upload frontier. It requests 4 GPUs for up
to 2 days and emails on END/FAIL/TIME_LIMIT.

```bash
sbatch examples/babel/train_foldclo_loravlm.sbatch
squeue --me
```

Report the **jobid** and the path it logs to (`yam_continual-<jobid>.out`). Things to note in
the log: `[data]`/`[train]`/`[clean]` lines per dataset, `[wait]` lines if it reaches the
upload frontier, and checkpoints under
`/data/user_data/$USER/openpi_checkpoints/pi05_BiYAMMolmoAct2_loravlm/yam_continual_4gpu/`.

### Resuming later
The run uses a single checkpoint dir and `--resume` after the first dataset, so if the 2-day
limit is hit (TIME_LIMIT email), just `sbatch examples/babel/train_foldclo_loravlm.sbatch`
again — it picks up from the latest checkpoint. (It re-walks the list from the top, but
already-trained datasets are cheap to re-touch; if you want to skip ahead, trim the top of
`converted_yam_data.txt` to the next untrained dataset before resubmitting.)

## Notes / gotchas to watch
- **Wandb**: set `WANDB_API_KEY` or `WANDB_MODE=offline`, else training may block on a prompt.
- **Scratch quota**: datasets are deleted after each one trains, so peak is ~one dataset; if a
  run dies mid-dataset, stale dirs may remain under `/scratch/$USER/lerobot` — clean manually.
- **Email**: native SLURM mail to `leokswang@gmail.com` only fires on job END/FAIL/TIME_LIMIT,
  not at the instant a dataset is missing (the job waits and polls instead).
