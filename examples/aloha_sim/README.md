# Run Aloha Sim

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM
```

## Recording and interpreting π₀-FAST action tokens

This workflow runs the `pi0_fast_base` checkpoint through ALOHA sim and records, for
every policy inference, the **raw discrete action tokens** the model generates — so you
can inspect what π₀-FAST actually emits before (and after) FAST decoding.

### What it does

`pi0_fast` does not output continuous actions directly. It autoregressively generates a
sequence of PaliGemma vocabulary tokens; the action-carrying ones live near the top of
the vocabulary and are mapped back to **FAST token IDs** via:

```
fast_token_id = 257152 - 1 - 128 - paligemma_token_id   (= 257023 - paligemma_token_id)
```

That FAST token sequence is then decoded (inverse-BPE + inverse-DCT) into a continuous
action chunk of shape `[action_horizon, action_dim]` (here `[32, 32]`); for ALOHA only the
first 14 dims are robot-relevant.

When logging is enabled, the `ExtractFASTActions` output transform
([src/openpi/transforms.py](../../src/openpi/transforms.py)) writes one JSON object per
inference step to a JSONL file, containing the raw PaliGemma token IDs, the converted FAST
token IDs, `decode_ok`/`decode_error`, the decoded action shape, and a `decoded_all_zero`
flag. Decode failures are caught so the policy server stays alive — useful because
`pi0_fast_base` is *not* fine-tuned for ALOHA sim and routinely emits FAST sequences that
fail to decode (they appear as `decode_ok: false`, `decoded_all_zero: true`).

> **Prerequisite:** this uses a custom training config named `pi0_fast_aloha_sim` (added to
> [src/openpi/training/config.py](../../src/openpi/training/config.py)) that pairs the
> `pi0_fast_base` checkpoint with the ALOHA data/norm-stats config. The
> `OPENPI_FAST_TOKEN_LOG` environment variable is forwarded into the `openpi_server`
> container by [compose.yml](compose.yml); the path is relative to the container, where
> `/app` is the repo root, so it lands in `data/` on the host alongside the episode video.
>
> These changes are not in upstream OpenPI. They live in our fork
> [asu-kim/openpi](https://github.com/asu-kim/openpi), on the
> [`auth`](https://github.com/asu-kim/openpi/tree/auth) branch — check it out to get the
> `pi0_fast_aloha_sim` config, the `ExtractFASTActions` token logging, the `compose.yml`
> env-var forwarding, and `interpret_fast_tokens.py`.

### 1. Run the sim and record tokens

```bash
# Point the server at the FAST config + checkpoint, and choose where to write the token log.
# /app inside the container == the repo root on the host, so this file appears in ./data.
export SERVER_ARGS="policy:checkpoint --policy.config=pi0_fast_aloha_sim --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_base"
export OPENPI_FAST_TOKEN_LOG=/app/data/aloha_sim/token_logs/pi0_fast_tokens.jsonl

# Start fresh — the log is opened in append mode, so remove any previous run's file.
rm -f data/aloha_sim/token_logs/pi0_fast_tokens.jsonl

# Build and run. The server prints "[OPENPI_FAST_TOKEN_LOG] active -> ..." on the first
# decode, confirming the logger is wired up. The episode runs to completion even when
# FAST decoding fails, and the video is saved to data/aloha_sim/videos/out_*.mp4.
docker compose -f examples/aloha_sim/compose.yml up --build
```

### 2. Quick sanity check on the log

```bash
# How many steps were logged.
wc -l data/aloha_sim/token_logs/pi0_fast_tokens.jsonl

# Count genuine decode successes vs. failures.
grep -o '"decode_ok": [a-z]*' data/aloha_sim/token_logs/pi0_fast_tokens.jsonl | sort | uniq -c

# Count steps whose decoded actions were an all-zero fallback (i.e. FAST decode failed
# internally and returned zeros of the correct shape).
grep -o '"decoded_all_zero": [a-z]*' data/aloha_sim/token_logs/pi0_fast_tokens.jsonl | sort | uniq -c
```

### 3. Produce human-readable summaries

[interpret_fast_tokens.py](../../interpret_fast_tokens.py) reads the JSONL and emits
per-record and aggregate summaries plus optional CSV/plots.

```bash
uv run python interpret_fast_tokens.py \
  --input data/aloha_sim/token_logs/pi0_fast_tokens.jsonl \
  --output-dir interpreted_tokens --csv --plots --print-records 5
```

What the flags do:

- `--input` — path to the JSONL token log to read.
- `--output-dir` — where to write outputs (default `interpreted_tokens/`).
- `--csv` — also write `records.csv` (one row per inference step).
- `--plots` — also write PNGs (decode-success-over-time, FAST-token-count per record,
  ALOHA action L2 norm over time, and per-record ALOHA action heatmaps). Requires
  matplotlib; no seaborn.
- `--print-records N` — print compact one-line summaries for the first `N` records.
- `--max-records N` — (optional) only process the first `N` records.

Outputs written to `--output-dir`:

- `summary.json` — aggregate stats (decode success rate, avg FAST token count, most common
  FAST tokens, failed/all-zero record indices, overall top-moving ALOHA dims).
- `fast_tokens.txt` — the filtered FAST action-token IDs, one line per record.
- `records.csv` — per-record table (token counts, decode flags, action/ALOHA stats).
- `*.png` — the plots, when `--plots` is passed.

The script also prints the aggregate summary to stdout. Note: ALOHA motion statistics only
appear if the log records the decoded continuous actions; the default record stores
`decoded_actions_shape` but not the values. To populate the action-level stats, add the
arrays to the record in `ExtractFASTActions`:

```python
record["decoded_actions"] = actions.tolist()           # full [32, 32] chunk
record["aloha_actions_14d"] = actions[:, :14].tolist()  # robot-relevant dims
```

then re-run steps 1–3.
