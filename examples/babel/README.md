# Fine-tuning the fold-cloth model on the Babel cluster

This guide walks you through running the fold-cloth fine-tuning job on **Babel**,
CMU's shared high-performance computing (HPC) cluster
(<https://wiki.babel.cs.cmu.edu/index.php/BABEL>). It is written so that someone
who has never used a compute cluster before can follow along. Jargon is explained
the first time it appears.

This run differs from the earlier fold-cloth run in one way that matters for the
model: **the task instructions (the text describing what to do, e.g. "fold the
cloth") are now fed to the model during training.** Everything needed for that is
already baked into the training config `pi05_foldclo_babel`, so you don't have to
edit any code.

It also uses **4 GPUs** with a **batch size of 64**. Both of those are already set
in the config too.

---

## Vocabulary (read this first if you're new to clusters)

- **Cluster** — a large collection of computers ("nodes") in a data center that you
  share with other researchers. You don't sit at it; you log in over the network and
  ask it to run your programs.
- **Login node** — the computer you land on when you connect. It is for editing files
  and submitting jobs. **It has no powerful GPUs and you must not run training on it.**
- **Compute node** — a powerful machine (with GPUs) that actually runs your job. You
  never log into it directly; the scheduler assigns one to you.
- **GPU (Graphics Processing Unit)** — the specialized chip that does the heavy math
  for training neural networks. Each GPU has its own memory ("VRAM"), separate from
  the machine's regular RAM. Our model is big, so we spread it over 4 GPUs.
- **Slurm** — the "scheduler": the program that takes everyone's job requests, lines
  them up in a queue, and decides which job runs on which compute node and when. You
  talk to it with commands like `sbatch`, `squeue`, and `scancel`.
- **Job** — one submitted unit of work (here: our training run). It waits in a queue,
  then runs when resources free up.
- **Partition** — a named pool of compute nodes with its own rules (time limits, max
  GPUs). Think of it as a "queue" you pick based on how long/big your job is.
- **Batch job** — a job you submit and walk away from; it runs unattended and writes
  its output to a log file. (The opposite is an "interactive" session where you type
  commands live.)
- **Batch size** — how many training examples the model looks at together before it
  updates itself once. Bigger = more stable/faster per step but needs more GPU memory.
- **FSDP (Fully-Sharded Data Parallelism)** — a technique for splitting one copy of a
  large model across several GPUs so it fits in memory. More on this below.
- **Norm stats (normalization statistics)** — the average and spread of the numbers in
  your dataset. The model trains better when inputs are rescaled to a standard range,
  so we compute these once up front.

---

## How the 4 GPUs are used (parallelism, explained)

There are two common ways to use multiple GPUs:

1. **Data parallelism** — put a *full copy* of the model on each GPU and give each GPU
   a different slice of the batch. Fast, but every GPU must hold the whole model. Our
   model (pi05, ~3 billion parameters) in full fine-tuning needs **~70 GB** by itself,
   which won't fit on most cards.

2. **Model sharding / FSDP** — keep *one* copy of the model but cut each big weight
   matrix into pieces and store one piece on each GPU. Each GPU now holds only ~1/4 of
   the model, so memory per GPU drops a lot. GPUs talk to each other to reassemble
   pieces when needed. This is what lets full fine-tuning fit on 4 smaller GPUs.

This repo controls that with one config field, `fsdp_devices`. We set
`fsdp_devices=4`, which tells it to shard the single model across all 4 GPUs (pure
model sharding, no data-parallel replicas). The batch size of 64 is then divided
across the 4 GPUs → **16 examples per GPU**. (The training script requires that the
batch size divide evenly by the number of GPUs; 64 ÷ 4 = 16, so we're fine.)

All of this is already set in `src/openpi/training/config.py` under the config named
**`pi05_foldclo_babel`** — you do not need to change it. For reference:

```python
TrainConfig(
    name="pi05_foldclo_babel",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotBiYAMDataConfig(
        repo_id="leokswang/18122025-foldclo-01_lerobot_format",
        base_config=DataConfig(prompt_from_task=True),   # <-- task prompts included
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=64,        # <-- batch size 64
    fsdp_devices=4,       # <-- shard the model across 4 GPUs
),
```

> **Important:** `fsdp_devices=4` only works if Slurm actually gives the job 4 GPUs.
> The `--gres=gpu:4` line in the submit script (below) is what does that. The two
> numbers must match.

---

## Step 0 — Log in and get the code onto Babel

From your laptop's terminal:

```bash
ssh <your_andrew_id>@login.babel.cs.cmu.edu
```

Use your **Andrew ID** and password (CMU's SCS credentials do **not** work here).

Babel gives every user two important folders:

| Path | Size | Where it's visible | Use it for |
|---|---|---|---|
| `/home/<your_andrew_id>` | 100 GB | every node (incl. login) | code, the uv venv, small files |
| `/data/user_data/<your_andrew_id>` | 500 GB | **compute nodes only, during a job** | datasets, model checkpoints, caches |

> **Important — `/data` does not exist on the login node.** It is an "on-demand"
> (AutoFS) mount that only appears on a compute node *while your job is running*. If
> you `ls /data` on the login node you'll get "No such file or directory" — that is
> normal, not a bug. So: do all the **setup** below in `/home`, and let the **job
> script** (which runs on a compute node) handle everything under `/data/user_data`.
> The submit script stats the full path to trigger the mount and writes all big files
> (datasets, base-model cache, checkpoints) there automatically.

Clone the repo (with submodules) into your home directory:

```bash
cd ~
git clone --recurse-submodules <your_repo_url> openpi
cd openpi
```

If you already have it but forgot `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

> **Note on AutoFS:** `/data/user_data/<you>` is mounted "on demand" **and only on
> compute nodes**. On the login node it isn't there at all. Even on a compute node you
> may need to type the *full* path (`ls -ld /data/user_data/<your_andrew_id>`) to
> trigger the mount. See the verification step below.

### Verify your `/data` access on a compute node (recommended, ~1 min)

Before submitting the long job, confirm your `user_data` is provisioned. Grab a short
interactive shell on the `debug` partition (the only partition that allows interactive
sessions):

```bash
srun --partition=debug --time=00:05:00 --pty bash
# now you're on a compute node:
ls -ld /data/user_data/$USER    # full path triggers the AutoFS mount
exit
```

If that prints a directory, you're good. If it still says "No such file or directory"
*on the compute node*, your space isn't provisioned yet — ask in the `#babel-babble`
Slack channel (tag `@help-babel`).

---

## Step 1 — Set up the Python environment (uv)

This repo uses [`uv`](https://docs.astral.sh/uv/) to manage Python packages. Babel
does not ship `uv` by default. Check for a module first:

```bash
module avail uv 2>&1 | grep -i uv
```

If one exists, `module load uv` (and add that same `module load` line to the top of
`train_foldclo.sbatch`, since compute nodes start with a clean module environment). If
not, install uv into your home directory — it's a single static binary, no admin
needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env     # this shell; new logins get it from ~/.bashrc
uv --version
```

uv will fetch the correct Python (3.11) itself, so you don't need a Python module.

Now set up the environment **on the login node** (it has internet). Keep everything in
`/home` — remember `/data` isn't mounted here:

```bash
export UV_CACHE_DIR=$HOME/uv_cache
mkdir -p "$UV_CACHE_DIR"

cd ~/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`uv sync` downloads and installs the exact set of packages this project needs into a
local `.venv` folder. The `GIT_LFS_SKIP_SMUDGE=1` part skips downloading large model
files that we don't need at install time.

---

## Step 2 — Credentials (tokens)

Two services need a login token:

- **Hugging Face** — hosts the dataset (`leokswang/18122025-foldclo-01_lerobot_format`).
  Get your token from <https://huggingface.co/settings/tokens>.
- **Weights & Biases (W&B)** — the website that draws live training graphs (loss going
  down, etc.). Get your key from <https://wandb.ai/authorize>. Optional — you can turn
  it off (see below).

The easiest approach is to log in once on the login node so the tokens are cached:

```bash
uv run huggingface-cli login        # paste your HF token
uv run wandb login                  # paste your W&B key  (skip if not using W&B)
```

The base model checkpoint is downloaded automatically from Google Cloud Storage
(`gs://openpi-assets/...`); that bucket is public, so no Google login is needed.

If you'd rather not use W&B at all, add `--no-wandb-enabled` to the `train.py` line in
the submit script, or run `export WANDB_MODE=offline` before submitting.

---

## Step 3 — Submit the training job

The submit script is `examples/babel/train_foldclo.sbatch`. From the repo root:

```bash
# make the tokens available to the job (only needed if you skipped the `login` step above)
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=xxx

sbatch examples/babel/train_foldclo.sbatch
```

`sbatch` hands the script to Slurm, which puts it in the queue and prints something
like `Submitted batch job 1234567`. That number is your **job ID**.

The script (already written for you) asks Slurm for: the `general` partition, **4
GPUs on one machine**, 32 CPUs, 128 GB of system RAM, and up to 2 days. Then on the
assigned compute node it:

1. Points the caches at `/data/user_data/<you>` so `/home` doesn't fill up.
2. Computes normalization statistics (`compute_norm_stats.py`) — once, ~minutes.
3. Runs training (`train.py pi05_foldclo_babel`) for 5,000 steps, saving checkpoints
   under `/data/user_data/<you>/openpi_checkpoints/`.

> **Why 5,000 steps?** The dataset is small (10 episodes / ~37k frames). At batch 64
> that's ~8.6 passes over the data ("epochs"). Training far longer mostly memorizes
> these 10 demos. We deliberately keep it short and pick the best checkpoint by robot
> evaluation (see "Choosing the best checkpoint" below) rather than trusting the last
> step.

### Why the `general` partition?

| Partition | Max time | Max GPUs | Notes |
|---|---|---|---|
| `debug`   | 12 h | 2  | for quick tests only — too few GPUs for us |
| `general` | 2 days | 8 | **our choice**: enough GPUs, long enough, batch-submit only |
| `preempt` | 31 days | 24 | longest, but your job can be **interrupted** ("preempted") by higher-priority jobs |

We need 4 GPUs, so `general` is the natural fit. At 5,000 steps this run finishes well
within the 2-day limit. If you later raise `num_train_steps` and it won't fit in 2
days, see "Resuming" below, or switch to `preempt` (change `--partition=general` to
`--partition=preempt` and raise `--time`) — just know preempt jobs can be paused and
restarted by the scheduler.

---

## Step 4 — Watch the job

```bash
squeue --me                       # see your jobs: PD = pending (waiting), R = running
tail -f foldclo-<job_id>.out      # live view of the log file (Ctrl+C to stop watching)
scancel <job_id>                  # cancel a job if you need to
```

While `R` (running), you can also peek at GPU usage. Find the node name in
`squeue --me` (the `NODELIST` column), then:

```bash
ssh <node_name> nvidia-smi
```

You should see 4 GPUs in use. If W&B is enabled, the log prints a URL where you can
watch the loss curve in your browser.

---

## Step 5 — Where the results land

Checkpoints (the saved model snapshots) are written to:

```
/data/user_data/<your_andrew_id>/openpi_checkpoints/pi05_foldclo_babel/foldclo_prompts_4gpu/<step>/
```

A checkpoint is saved every 1,000 steps. With this config you'll keep snapshots at
steps **1000, 2000, 3000, 4000, and 4999** (the final step).

To serve a given checkpoint for inference, point the policy server at its folder:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_foldclo_babel \
    --policy.dir=/data/user_data/$USER/openpi_checkpoints/pi05_foldclo_babel/foldclo_prompts_4gpu/4000
```

### Choosing the best checkpoint

**The trainer does not automatically save a "best" checkpoint — there is no validation
metric.** It saves by step number only: normally it keeps just the *latest* snapshot,
but because we set `keep_period=1000` (equal to `save_interval`), every 1,000-step
snapshot is kept permanently so you can compare them.

Lowest training loss is *not* a reliable signal on a dataset this small (the model can
fit the 10 demos while getting worse on the real task). So the recommended workflow is:

1. Let the run finish (or stop it early once loss plateaus — watch the W&B curve).
2. Serve each kept checkpoint (1000, 2000, … 4999) and run it on the robot / in eval.
3. Keep whichever scores best; that's your model. Don't assume it's the last one.

---

## Resuming if the job is cut short

If a job hits the time limit (or a preempt job is interrupted) before the final step,
restart from the last saved checkpoint by replacing `--overwrite` with `--resume` in
the submit script's `train.py` line, then `sbatch` it again with the **same**
`--exp-name`. It will pick up where it left off.

---

## Common problems

- **`sbatch: error: ... Invalid partition`** — you're on a partition you can't use, or
  asked for more than it allows. Check `scontrol show part general`.
- **Job stuck in `PD` (pending) for a long time** — the cluster is busy and 4 GPUs
  aren't free yet. `squeue --me --start` estimates when it'll start. Be patient or try
  `preempt`.
- **Out-of-memory (OOM) on the GPU** — the per-GPU memory was exceeded. First confirm
  the job really got 4 GPUs (`nvidia-smi` on the node). If the GPUs Babel gave you are
  smaller than expected, lower `--batch-size` (must stay divisible by 4, so try 32 or
  16) by adding e.g. `--batch-size=32` to the `train.py` line.
- **`Batch size N must be divisible by the number of devices D`** — your batch size
  isn't a multiple of the GPU count. Keep batch size a multiple of 4.
- **Hugging Face 401 / permission errors** — your `HF_TOKEN` isn't set or lacks access
  to the dataset. Re-run `huggingface-cli login`.
- **`/home` full** — you let caches/checkpoints land in `/home`. Make sure
  `HF_HOME`, `OPENPI_DATA_HOME`, `UV_CACHE_DIR`, and `--checkpoint-base-dir` all point
  under `/data/user_data/$USER` (the submit script does this for you).

---

## Quick reference (TL;DR)

```bash
# one time, on the login node
ssh <andrew_id>@login.babel.cs.cmu.edu
cd ~ && git clone --recurse-submodules <repo_url> openpi && cd openpi
export UV_CACHE_DIR=/data/user_data/$USER/uv_cache && mkdir -p "$UV_CACHE_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run huggingface-cli login
uv run wandb login        # optional

# every time you want to train
cd ~/openpi
sbatch examples/babel/train_foldclo.sbatch
squeue --me
tail -f foldclo-<job_id>.out
```
