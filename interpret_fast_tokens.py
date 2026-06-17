#!/usr/bin/env python3
"""interpret_fast_tokens.py

Human-readable summaries for a JSONL log produced while running ``pi0_fast_base``
through ``examples/aloha_sim`` (one policy-inference record per line).

Conceptual interpretation (read this before trusting any number below)
----------------------------------------------------------------------
* ``raw_paligemma_token_ids`` are the *raw* tokens emitted by the model. They are
  a mix of text/formatting tokens (e.g. "Action:", " ", "|"), an EOS token
  (usually ``1``), padding (usually ``0``), and the actual FAST action tokens.
* The action tokens live near the *end* of the PaliGemma vocabulary and map back
  to FAST IDs via ``fast_id = 257023 - paligemma_id`` (= 257152 - 1 - 128 - id).
  We only treat a token as a FAST action token when the resulting id is a small
  non-negative number (``0 <= fast_id < 10000``); applying the formula to padding
  (``0``), EOS (``1``), or ordinary text tokens yields meaningless values, so we
  filter those out.
* A single FAST token is NOT "move joint X". FAST tokens are *compressed
  trajectory* tokens: the whole sequence decodes (inverse-BPE + inverse-DCT) into
  a continuous action chunk, normally shape ``[action_horizon, action_dim]`` =
  ``[32, 32]``. The decoded *continuous* values are what actually drive the robot.
* For ALOHA, only the first 14 dimensions of the decoded chunk are
  robot-relevant (``actions[:, :14]``): 2 arms x (6 joints + 1 gripper).

This tool therefore reports both the token-level view (counts, filtered FAST IDs)
and, when continuous actions are present in the log, the action-level view (per-
dimension statistics and which ALOHA dims move the most).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EOS_TOKEN = 1            # PaliGemma end-of-sequence
PAD_TOKEN = 0            # PaliGemma padding
FAST_OFFSET = 257023     # 257152 - 1 - 128; fast_id = FAST_OFFSET - paligemma_id
FAST_ID_MAX = 10000      # candidate FAST action token if 0 <= fast_id < FAST_ID_MAX
ALOHA_DIMS = 14          # first 14 decoded dims are robot-relevant for ALOHA
NEAR_ZERO_NORM = 1e-6    # a timestep with L2 norm below this counts as "near zero"


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------
def trim_raw_tokens(tokens: list[int]) -> list[int]:
    """Drop padding (0) and stop at the first EOS (1).

    Returns the "live" portion of the generated sequence: everything up to (but
    not including) the first EOS token, with any padding tokens removed.
    """
    trimmed: list[int] = []
    for t in tokens:
        if t == EOS_TOKEN:
            break
        if t == PAD_TOKEN:
            continue
        trimmed.append(int(t))
    return trimmed


def extract_fast_action_tokens(trimmed_tokens: list[int]) -> list[int]:
    """Convert the trimmed PaliGemma tokens to FAST ids, keeping only plausible
    action tokens (``0 <= fast_id < FAST_ID_MAX``).

    Text/formatting tokens that survived trimming convert to large/negative
    values and are filtered out here, so the result is the genuine FAST action
    token subsequence.
    """
    fast_ids: list[int] = []
    for t in trimmed_tokens:
        fast_id = FAST_OFFSET - int(t)
        if 0 <= fast_id < FAST_ID_MAX:
            fast_ids.append(fast_id)
    return fast_ids


# ---------------------------------------------------------------------------
# Action-level helpers
# ---------------------------------------------------------------------------
def _to_array(value) -> np.ndarray | None:
    """Best-effort conversion of a nested list to a 2D float array."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def action_stats(arr: np.ndarray) -> dict:
    """Summary statistics for a [T, D] continuous action chunk."""
    norms = np.linalg.norm(arr, axis=1)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "per_dim_min": arr.min(axis=0).tolist(),
        "per_dim_max": arr.max(axis=0).tolist(),
        "per_dim_mean": arr.mean(axis=0).tolist(),
        "per_dim_std": arr.std(axis=0).tolist(),
        "l2_norm_per_timestep": norms.tolist(),
        "num_near_zero_timesteps": int(np.sum(norms < NEAR_ZERO_NORM)),
    }


def top_dims(arr: np.ndarray, k: int = 3) -> tuple[list[int], list[int]]:
    """Return (top dims by per-dim std, top dims by per-dim max-abs)."""
    per_dim_std = arr.std(axis=0)
    per_dim_absmax = np.abs(arr).max(axis=0)
    by_std = np.argsort(per_dim_std)[::-1][:k].tolist()
    by_abs = np.argsort(per_dim_absmax)[::-1][:k].tolist()
    return [int(d) for d in by_std], [int(d) for d in by_abs]


def get_aloha_array(record: dict, decoded: np.ndarray | None) -> np.ndarray | None:
    """Prefer an explicit ``aloha_actions_14d`` field; otherwise slice the first
    14 dims off the decoded chunk."""
    aloha = _to_array(record.get("aloha_actions_14d"))
    if aloha is not None:
        return aloha[:, :ALOHA_DIMS]
    if decoded is not None and decoded.shape[1] >= 1:
        return decoded[:, :ALOHA_DIMS]
    return None


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def process_record(index: int, record: dict) -> dict:
    """Turn one raw JSONL record into a flat, summarized dict."""
    raw_tokens = record.get("raw_paligemma_token_ids") or []
    if not isinstance(raw_tokens, list):
        raw_tokens = []

    trimmed = trim_raw_tokens(raw_tokens)
    fast_tokens = extract_fast_action_tokens(trimmed)

    # Timestamp parsing (graceful).
    ts = record.get("time_unix")
    iso = None
    if isinstance(ts, (int, float)):
        try:
            iso = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            iso = None

    out: dict = {
        "record_index": index,
        "time_unix": ts if isinstance(ts, (int, float)) else None,
        "time_iso": iso,
        "decode_ok": bool(record.get("decode_ok", False)),
        "decode_error": record.get("decode_error"),
        "decoded_actions_shape": record.get("decoded_actions_shape"),
        "decoded_all_zero": record.get("decoded_all_zero"),
        "raw_token_count": len(raw_tokens),
        "trimmed_token_count": len(trimmed),
        "fast_action_token_count": len(fast_tokens),
        "fast_action_tokens": fast_tokens,
        # Action-level fields, filled in below if continuous actions are present.
        "has_decoded_actions": False,
        "action_shape": None,
        "action_min": None, "action_max": None, "action_mean": None, "action_std": None,
        "aloha_min": None, "aloha_max": None, "aloha_mean": None, "aloha_std": None,
        "num_near_zero_timesteps": None,
        "top_aloha_dims_by_std": None,
        "top_aloha_dims_by_abs": None,
        "_aloha_array": None,  # kept in-memory for aggregate plots/stats; not serialized
    }

    decoded = _to_array(record.get("decoded_actions"))
    if decoded is not None:
        out["has_decoded_actions"] = True
        a_stats = action_stats(decoded)
        out["action_shape"] = a_stats["shape"]
        out["action_min"] = a_stats["min"]
        out["action_max"] = a_stats["max"]
        out["action_mean"] = a_stats["mean"]
        out["action_std"] = a_stats["std"]
        out["full_action_stats"] = a_stats

    aloha = get_aloha_array(record, decoded)
    if aloha is not None:
        out["_aloha_array"] = aloha
        al_stats = action_stats(aloha)
        out["aloha_min"] = al_stats["min"]
        out["aloha_max"] = al_stats["max"]
        out["aloha_mean"] = al_stats["mean"]
        out["aloha_std"] = al_stats["std"]
        out["num_near_zero_timesteps"] = al_stats["num_near_zero_timesteps"]
        by_std, by_abs = top_dims(aloha, k=3)
        out["top_aloha_dims_by_std"] = by_std
        out["top_aloha_dims_by_abs"] = by_abs

    return out


def compact_summary_line(rec: dict) -> str:
    """One-line human-readable summary, e.g.:
    'Record 12: valid decode, 43 FAST tokens, action shape 32x32, ALOHA dims 3, 7, 12 have largest motion.'
    """
    status = "valid decode" if rec["decode_ok"] else "FAILED decode"
    parts = [f"Record {rec['record_index']}: {status}",
             f"{rec['fast_action_token_count']} FAST tokens"]
    if rec["action_shape"]:
        parts.append("action shape " + "x".join(str(d) for d in rec["action_shape"]))
    else:
        parts.append("no decoded actions")
    if rec["top_aloha_dims_by_std"]:
        dims = ", ".join(str(d) for d in rec["top_aloha_dims_by_std"])
        parts.append(f"ALOHA dims {dims} have largest motion")
    if rec["decoded_all_zero"]:
        parts.append("(all-zero output)")
    return ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------
def load_records(path: str, max_records: int | None):
    """Yield (line_number, parsed_dict). Skip malformed lines with a warning."""
    with open(path, "r", encoding="utf-8") as f:
        emitted = 0
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed JSONL line {line_no}: {e}",
                      file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                print(f"WARNING: skipping line {line_no}: not a JSON object",
                      file=sys.stderr)
                continue
            yield obj
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_summary_json(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_fast_tokens_txt(path: str, processed: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in processed:
            ids = " ".join(str(t) for t in rec["fast_action_tokens"])
            f.write(f"record {rec['record_index']}: {ids}\n")


def write_records_csv(path: str, processed: list[dict]) -> None:
    """Write one row per record. Uses pandas if available, else the csv module."""
    columns = [
        "record_index", "time_unix", "decode_ok", "raw_token_count",
        "trimmed_token_count", "fast_action_token_count", "decoded_all_zero",
        "action_shape", "action_min", "action_max", "action_mean", "action_std",
        "aloha_min", "aloha_max", "aloha_mean", "aloha_std",
        "top_aloha_dims_by_std", "top_aloha_dims_by_abs",
    ]

    def row_of(rec: dict) -> dict:
        return {
            "record_index": rec["record_index"],
            "time_unix": rec["time_unix"],
            "decode_ok": rec["decode_ok"],
            "raw_token_count": rec["raw_token_count"],
            "trimmed_token_count": rec["trimmed_token_count"],
            "fast_action_token_count": rec["fast_action_token_count"],
            "decoded_all_zero": rec["decoded_all_zero"],
            "action_shape": "x".join(str(d) for d in rec["action_shape"]) if rec["action_shape"] else "",
            "action_min": rec["action_min"], "action_max": rec["action_max"],
            "action_mean": rec["action_mean"], "action_std": rec["action_std"],
            "aloha_min": rec["aloha_min"], "aloha_max": rec["aloha_max"],
            "aloha_mean": rec["aloha_mean"], "aloha_std": rec["aloha_std"],
            "top_aloha_dims_by_std": "|".join(str(d) for d in rec["top_aloha_dims_by_std"]) if rec["top_aloha_dims_by_std"] else "",
            "top_aloha_dims_by_abs": "|".join(str(d) for d in rec["top_aloha_dims_by_abs"]) if rec["top_aloha_dims_by_abs"] else "",
        }

    rows = [row_of(r) for r in processed]
    try:
        import pandas as pd  # noqa: PLC0415 (lazy import; optional dependency)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    except ImportError:
        import csv  # stdlib fallback
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting (optional; matplotlib only, no seaborn)
# ---------------------------------------------------------------------------
def make_plots(processed: list[dict], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / file output
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available; skipping --plots", file=sys.stderr)
        return

    idx = [r["record_index"] for r in processed]

    # 1. Decode success over time.
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(idx, [1 if r["decode_ok"] else 0 for r in processed], where="mid")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fail", "ok"])
    ax.set_xlabel("record index"); ax.set_title("Decode success over time")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "decode_success.png"), dpi=120)
    plt.close(fig)

    # 2. FAST tokens per record.
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(idx, [r["fast_action_token_count"] for r in processed])
    ax.set_xlabel("record index"); ax.set_ylabel("# FAST action tokens")
    ax.set_title("FAST action token count per record")
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "fast_token_counts.png"), dpi=120)
    plt.close(fig)

    # 3. ALOHA action L2 norm over time (mean per-timestep norm per record).
    aloha_recs = [r for r in processed if r["_aloha_array"] is not None]
    if aloha_recs:
        fig, ax = plt.subplots(figsize=(10, 3))
        mean_norms = [float(np.linalg.norm(r["_aloha_array"], axis=1).mean()) for r in aloha_recs]
        ax.plot([r["record_index"] for r in aloha_recs], mean_norms, marker="o", ms=3)
        ax.set_xlabel("record index"); ax.set_ylabel("mean L2 norm (ALOHA dims)")
        ax.set_title("ALOHA action L2 norm over time")
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, "aloha_l2_norm.png"), dpi=120)
        plt.close(fig)

        # 4. Heatmaps of ALOHA actions for the first few records that have them.
        for r in aloha_recs[:6]:
            arr = r["_aloha_array"]
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(arr, aspect="auto", cmap="viridis")
            ax.set_xlabel("ALOHA action dim"); ax.set_ylabel("timestep")
            ax.set_title(f"ALOHA actions - record {r['record_index']}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"aloha_heatmap_record_{r['record_index']}.png"), dpi=120)
            plt.close(fig)
    else:
        print("NOTE: no decoded ALOHA actions found; skipping action plots.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(processed: list[dict]) -> dict:
    total = len(processed)
    n_ok = sum(1 for r in processed if r["decode_ok"])
    failed = [r["record_index"] for r in processed if not r["decode_ok"]]
    all_zero = [r["record_index"] for r in processed if r["decoded_all_zero"]]
    fast_counts = [r["fast_action_token_count"] for r in processed]

    # Most common FAST tokens across all records.
    token_counter: collections.Counter = collections.Counter()
    for r in processed:
        token_counter.update(r["fast_action_tokens"])

    # Overall top-moving ALOHA dims: std per dim over all stacked timesteps.
    aloha_arrays = [r["_aloha_array"] for r in processed if r["_aloha_array"] is not None]
    overall_top_std: list[int] = []
    overall_top_abs: list[int] = []
    overall_per_dim_std: list[float] = []
    if aloha_arrays:
        stacked = np.concatenate(aloha_arrays, axis=0)  # [sum_T, <=14]
        overall_per_dim_std = stacked.std(axis=0).tolist()
        by_std, by_abs = top_dims(stacked, k=5)
        overall_top_std, overall_top_abs = by_std, by_abs

    return {
        "total_records": total,
        "decode_success_count": n_ok,
        "decode_success_rate": (n_ok / total) if total else 0.0,
        "avg_fast_token_count": (float(np.mean(fast_counts)) if fast_counts else 0.0),
        "median_fast_token_count": (float(np.median(fast_counts)) if fast_counts else 0.0),
        "most_common_fast_tokens": [
            {"fast_token_id": int(tok), "count": int(c)} for tok, c in token_counter.most_common(20)
        ],
        "failed_decode_records": failed,
        "all_zero_records": all_zero,
        "records_with_decoded_actions": len(aloha_arrays),
        "overall_top_aloha_dims_by_std": overall_top_std,
        "overall_top_aloha_dims_by_abs": overall_top_abs,
        "overall_aloha_per_dim_std": overall_per_dim_std,
    }


def print_aggregate(summary: dict) -> None:
    print("=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"Total records:            {summary['total_records']}")
    print(f"Decode success:           {summary['decode_success_count']}"
          f"/{summary['total_records']} "
          f"({summary['decode_success_rate'] * 100:.1f}%)")
    print(f"Avg FAST token count:     {summary['avg_fast_token_count']:.1f} "
          f"(median {summary['median_fast_token_count']:.1f})")
    print(f"Records w/ decoded acts:  {summary['records_with_decoded_actions']}")

    print("\nMost common FAST action tokens (id: count):")
    if summary["most_common_fast_tokens"]:
        for item in summary["most_common_fast_tokens"][:10]:
            print(f"  {item['fast_token_id']:>6}: {item['count']}")
    else:
        print("  (none)")

    failed = summary["failed_decode_records"]
    print(f"\nFailed-decode records ({len(failed)}): "
          f"{failed if len(failed) <= 40 else str(failed[:40]) + ' ...'}")
    zeros = summary["all_zero_records"]
    print(f"All-zero decoded records ({len(zeros)}): "
          f"{zeros if len(zeros) <= 40 else str(zeros[:40]) + ' ...'}")

    if summary["overall_top_aloha_dims_by_std"]:
        print(f"\nOverall top-moving ALOHA dims (by std): "
              f"{summary['overall_top_aloha_dims_by_std']}")
        print(f"Overall top-moving ALOHA dims (by |max|): "
              f"{summary['overall_top_aloha_dims_by_abs']}")
    else:
        print("\nNo decoded continuous actions in log -> no ALOHA motion stats.")
        print("(Expected for un-finetuned pi0_fast_base: decoding fails and the log")
        print(" may omit 'decoded_actions'. Token-level fields above are still valid.)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interpret a pi0_fast token-log JSONL into human-readable summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to pi0_fast_tokens.jsonl")
    parser.add_argument("--output-dir", default="interpreted_tokens",
                        help="Directory for summary.json / fast_tokens.txt / records.csv / plots")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Only process the first N records")
    parser.add_argument("--csv", action="store_true", help="Write records.csv")
    parser.add_argument("--plots", action="store_true", help="Write PNG plots (needs matplotlib)")
    parser.add_argument("--print-records", type=int, default=0,
                        help="Print compact per-record summaries for the first N records")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    processed: list[dict] = []
    for i, record in enumerate(load_records(args.input, args.max_records)):
        processed.append(process_record(i, record))

    if not processed:
        print("ERROR: no valid records found in input.", file=sys.stderr)
        return 1

    # Per-record printing.
    if args.print_records > 0:
        print("-" * 70)
        print(f"PER-RECORD SUMMARIES (first {min(args.print_records, len(processed))})")
        print("-" * 70)
        for rec in processed[:args.print_records]:
            print(compact_summary_line(rec))
            if rec["decode_error"]:
                print(f"    decode_error: {rec['decode_error']}")
        print()

    # Aggregate + outputs.
    summary = aggregate(processed)

    summary_path = os.path.join(args.output_dir, "summary.json")
    write_summary_json(summary_path, summary)

    fast_tokens_path = os.path.join(args.output_dir, "fast_tokens.txt")
    write_fast_tokens_txt(fast_tokens_path, processed)

    written = [summary_path, fast_tokens_path]

    if args.csv:
        csv_path = os.path.join(args.output_dir, "records.csv")
        write_records_csv(csv_path, processed)
        written.append(csv_path)

    if args.plots:
        make_plots(processed, args.output_dir)
        written.append(os.path.join(args.output_dir, "*.png"))

    print_aggregate(summary)
    print("\nWrote:")
    for p in written:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
