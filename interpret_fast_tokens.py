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
import math
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

# ALOHA decoded-action dim -> robot part. Verified against
# src/openpi/policies/aloha_policy.py: state/action layout is
# [left_arm_joints(6), left_gripper, right_arm_joints(6), right_gripper].
ALOHA_JOINT_NAMES = [
    "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4", "L_arm_j5", "L_arm_j6", "L_gripper",
    "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4", "R_arm_j5", "R_arm_j6", "R_gripper",
]

# Token-only motion proxy thresholds (on normalized token entropy, 0..1).
STILL_PROXY_T = 0.35     # below this -> likely "still" (few, repeated tokens)
ACTIVE_PROXY_T = 0.60    # above this -> likely "active" (diverse tokens)

# Per-joint movement: a dim whose peak-to-peak range over the chunk exceeds this
# (in decoded action units) is considered "moving". Tunable via --joint-move-threshold.
DEFAULT_JOINT_MOVE_THRESHOLD = 0.01


def joint_name(dim: int) -> str:
    """Human name for a decoded ALOHA action dim."""
    return ALOHA_JOINT_NAMES[dim] if dim < len(ALOHA_JOINT_NAMES) else f"dim{dim}"


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
# Analysis 1: token-only motion proxy (decode-free; works even when decode fails)
# ---------------------------------------------------------------------------
def token_motion_proxy(fast_tokens: list[int]) -> dict:
    """Coarse, decode-free estimate of how much the chunk *moves*, from the FAST
    token stream alone.

    Rationale: FAST is a frequency-domain code. A near-constant ("still")
    trajectory has energy only in the DC coefficient, quantizes to long runs of
    zeros, and BPE compresses those into few, highly repeated tokens. A moving
    trajectory excites more frequencies -> a longer, more diverse token stream.
    So token *diversity* tracks whole-chunk motion magnitude.

    IMPORTANT: this is a whole-chunk proxy, NOT per-joint and NOT exact. It only
    ranks records by likely activity. Use ``joint_motion_analysis`` (needs decoded
    actions) for anything quantitative or per-joint.
    """
    n = len(fast_tokens)
    base = {
        "token_count": n,
        "unique_tokens": 0,
        "unique_ratio": 0.0,
        "entropy_bits": 0.0,
        "normalized_entropy": 0.0,
        "dominant_token": None,
        "dominant_fraction": 0.0,
        "motion_proxy": 0.0,
        "motion_label": "unknown",
    }
    if n == 0:
        return base

    counts = collections.Counter(fast_tokens)
    unique = len(counts)
    probs = np.array(list(counts.values()), dtype=np.float64) / n
    entropy = float(-(probs * np.log2(probs)).sum())
    # Normalize by the max entropy achievable for this length (all-distinct = log2(n)).
    norm_entropy = entropy / math.log2(n) if n > 1 else 0.0
    dom_token, dom_count = counts.most_common(1)[0]

    label = ("still" if norm_entropy < STILL_PROXY_T
             else "active" if norm_entropy > ACTIVE_PROXY_T
             else "low")

    base.update(
        unique_tokens=unique,
        unique_ratio=unique / n,
        entropy_bits=entropy,
        normalized_entropy=norm_entropy,
        dominant_token=int(dom_token),
        dominant_fraction=dom_count / n,
        motion_proxy=norm_entropy,
        motion_label=label,
    )
    return base


# ---------------------------------------------------------------------------
# Analysis 2: per-joint movement (needs decoded continuous actions in the log)
# ---------------------------------------------------------------------------
def joint_motion_analysis(
    aloha: np.ndarray,
    state: np.ndarray | None = None,
    move_threshold: float = DEFAULT_JOINT_MOVE_THRESHOLD,
) -> dict:
    """Per-joint movement from the DECODED continuous chunk ``[T, <=14]``.

    ALOHA actions here are ABSOLUTE joint positions (the sim config sets
    ``use_delta_joint_actions=False``), so motion is measured as variation across
    the T timesteps of the chunk:

    * ``per_dim_std`` / ``peak_to_peak`` -> how much each joint varies; a dim is
      flagged ``moving`` when its peak-to-peak range exceeds ``move_threshold``.
    * ``step_delta`` (L2 norm of consecutive-timestep differences) -> per-timestep
      motion; a timestep counts as moving when its step delta exceeds
      ``move_threshold``.
    * If ``state`` (the current pose) is provided, ``dist_from_state_per_timestep``
      tells you how far the commanded chunk departs from where the robot is now.
    """
    T, D = aloha.shape
    per_dim_std = aloha.std(axis=0)
    per_dim_ptp = aloha.max(axis=0) - aloha.min(axis=0)
    moving_mask = per_dim_ptp > move_threshold

    joints = [
        {
            "dim": d,
            "name": joint_name(d),
            "std": float(per_dim_std[d]),
            "peak_to_peak": float(per_dim_ptp[d]),
            "moving": bool(moving_mask[d]),
        }
        for d in range(D)
    ]
    moving_joints = [j["name"] for j in joints if j["moving"]]
    still_joints = [j["name"] for j in joints if not j["moving"]]

    # Per-timestep step-to-step motion.
    if T > 1:
        step_deltas = np.linalg.norm(np.diff(aloha, axis=0), axis=1)
    else:
        step_deltas = np.zeros(0)
    timestep_moving = step_deltas > move_threshold

    # Distance from the current pose, if state is available.
    dist_from_state = None
    if state is not None and state.shape[-1] >= D:
        dist_from_state = np.linalg.norm(aloha - state[:D], axis=1).tolist()

    # Rank joints by peak-to-peak range (most-moving first).
    order = np.argsort(per_dim_ptp)[::-1]
    top_moving = [
        {"name": joint_name(int(d)), "dim": int(d), "peak_to_peak": float(per_dim_ptp[d])}
        for d in order[:5]
    ]

    return {
        "shape": [int(T), int(D)],
        "move_threshold": move_threshold,
        "joints": joints,
        "moving_joints": moving_joints,
        "still_joints": still_joints,
        "num_moving_joints": len(moving_joints),
        "chunk_is_static": len(moving_joints) == 0,
        "num_moving_timesteps": int(timestep_moving.sum()),
        "num_timesteps": int(T),
        "step_delta_mean": float(step_deltas.mean()) if step_deltas.size else 0.0,
        "step_delta_max": float(step_deltas.max()) if step_deltas.size else 0.0,
        "dist_from_state_per_timestep": dist_from_state,
        "top_moving_joints": top_moving,
    }


# ---------------------------------------------------------------------------
# Per-timestep view: map each timestep of a decoded chunk to its action values
# and the joints that are "relevant" (changed) at that timestep.
# ---------------------------------------------------------------------------
def per_timestep_rows(aloha: np.ndarray, move_threshold: float,
                      state: np.ndarray | None = None) -> list[dict]:
    """One row per timestep of a decoded ``[T, <=14]`` chunk.

    Each row carries the named joint values at that timestep, the step delta from
    the previous timestep, the list of joints whose value changed by more than
    ``move_threshold`` ("relevant_joints"), and — if ``state`` is given — the
    distance from the current pose.
    """
    T, D = aloha.shape
    rows: list[dict] = []
    for t in range(T):
        row: dict = {"timestep": t}
        for d in range(D):
            row[joint_name(d)] = float(aloha[t, d])
        if t == 0:
            row["step_delta"] = 0.0
            changed: list[str] = []
        else:
            diff = np.abs(aloha[t] - aloha[t - 1])
            row["step_delta"] = float(np.linalg.norm(aloha[t] - aloha[t - 1]))
            changed = [joint_name(d) for d in range(D) if diff[d] > move_threshold]
        row["num_relevant_joints"] = len(changed)
        row["relevant_joints"] = "|".join(changed)
        if state is not None and state.shape[-1] >= D:
            row["dist_from_state"] = float(np.linalg.norm(aloha[t] - state[:D]))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def process_record(index: int, record: dict,
                   move_threshold: float = DEFAULT_JOINT_MOVE_THRESHOLD) -> dict:
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
        "_state_array": None,  # current pose, in-memory only
        # Analysis 1 (decode-free) is always available; analysis 2 needs decoded actions.
        "token_motion": token_motion_proxy(fast_tokens),
        "joint_motion": None,
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

        # Per-joint movement (analysis 2). Use the current pose if the log has it.
        state_val = record.get("state")
        state_arr = None
        if state_val is not None:
            try:
                state_arr = np.asarray(state_val, dtype=np.float64).reshape(-1)
            except (ValueError, TypeError):
                state_arr = None
        out["_state_array"] = state_arr
        out["joint_motion"] = joint_motion_analysis(aloha, state_arr, move_threshold)

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
# Analysis 1 output: token-only motion proxy
# ---------------------------------------------------------------------------
def write_token_motion_csv(path: str, processed: list[dict]) -> None:
    columns = [
        "record_index", "decode_ok", "token_count", "unique_tokens", "unique_ratio",
        "entropy_bits", "normalized_entropy", "dominant_token", "dominant_fraction",
        "motion_proxy", "motion_label",
    ]

    def row_of(rec: dict) -> dict:
        tm = rec["token_motion"]
        return {
            "record_index": rec["record_index"],
            "decode_ok": rec["decode_ok"],
            "token_count": tm["token_count"],
            "unique_tokens": tm["unique_tokens"],
            "unique_ratio": round(tm["unique_ratio"], 4),
            "entropy_bits": round(tm["entropy_bits"], 4),
            "normalized_entropy": round(tm["normalized_entropy"], 4),
            "dominant_token": tm["dominant_token"],
            "dominant_fraction": round(tm["dominant_fraction"], 4),
            "motion_proxy": round(tm["motion_proxy"], 4),
            "motion_label": tm["motion_label"],
        }

    rows = [row_of(r) for r in processed]
    try:
        import pandas as pd  # noqa: PLC0415
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    except ImportError:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def print_token_motion(processed: list[dict], n_print: int) -> None:
    print("=" * 70)
    print("ANALYSIS 1: TOKEN-ONLY MOTION PROXY (decode-free)")
    print("=" * 70)
    print("Proxy = normalized token entropy in [0,1]. Higher -> more diverse tokens")
    print(f"-> more likely motion.  still < {STILL_PROXY_T}  <= low <= {ACTIVE_PROXY_T} < active")
    print("NOTE: whole-chunk estimate only; NOT per-joint, NOT exact.\n")

    shown = min(n_print, len(processed)) if n_print > 0 else 0
    for rec in processed[:shown]:
        tm = rec["token_motion"]
        flag = "" if rec["decode_ok"] else "  [decode failed]"
        print(f"Record {rec['record_index']:>3}: {tm['motion_label']:<6} "
              f"proxy={tm['motion_proxy']:.2f}  "
              f"unique={tm['unique_tokens']:>3}/{tm['token_count']:<3} "
              f"({tm['unique_ratio']:.2f})  "
              f"dominant={tm['dominant_token']}@{tm['dominant_fraction']*100:.0f}%{flag}")
    if shown:
        print()

    labels = collections.Counter(r["token_motion"]["motion_label"] for r in processed)
    proxies = [r["token_motion"]["motion_proxy"] for r in processed if r["token_motion"]["token_count"]]
    print(f"Records by motion label: {dict(labels)}")
    if proxies:
        print(f"Mean motion proxy: {float(np.mean(proxies)):.3f}  "
              f"(min {min(proxies):.2f}, max {max(proxies):.2f})")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Analysis 2 output: per-joint movement
# ---------------------------------------------------------------------------
def write_joint_motion_json(path: str, processed: list[dict]) -> None:
    payload = [
        {"record_index": r["record_index"], "decode_ok": r["decode_ok"], **r["joint_motion"]}
        for r in processed if r["joint_motion"] is not None
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_joint_motion_csv(path: str, processed: list[dict]) -> None:
    columns = [
        "record_index", "decode_ok", "num_timesteps", "chunk_is_static",
        "num_moving_joints", "moving_joints", "num_moving_timesteps",
        "step_delta_mean", "step_delta_max", "top_moving_joints",
    ]

    def row_of(rec: dict) -> dict:
        jm = rec["joint_motion"]
        return {
            "record_index": rec["record_index"],
            "decode_ok": rec["decode_ok"],
            "num_timesteps": jm["num_timesteps"],
            "chunk_is_static": jm["chunk_is_static"],
            "num_moving_joints": jm["num_moving_joints"],
            "moving_joints": "|".join(jm["moving_joints"]),
            "num_moving_timesteps": jm["num_moving_timesteps"],
            "step_delta_mean": round(jm["step_delta_mean"], 5),
            "step_delta_max": round(jm["step_delta_max"], 5),
            "top_moving_joints": "|".join(j["name"] for j in jm["top_moving_joints"]),
        }

    rows = [row_of(r) for r in processed if r["joint_motion"] is not None]
    try:
        import pandas as pd  # noqa: PLC0415
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    except ImportError:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def print_joint_motion(processed: list[dict], n_print: int) -> None:
    print("=" * 70)
    print("ANALYSIS 2: PER-JOINT MOVEMENT (from decoded continuous actions)")
    print("=" * 70)

    recs = [r for r in processed if r["joint_motion"] is not None]
    if not recs:
        print("No decoded continuous actions in the log, so per-joint movement")
        print("cannot be computed. Add these to the ExtractFASTActions record and")
        print("re-run the sim:")
        print('    record["decoded_actions"]   = actions.tolist()')
        print('    record["aloha_actions_14d"] = actions[:, :14].tolist()')
        print('    record["state"]             = data["state"].tolist()  # optional, for dist-from-pose')
        print("=" * 70)
        return

    thr = recs[0]["joint_motion"]["move_threshold"]
    print(f"A joint is 'moving' if its peak-to-peak range over the chunk > {thr} (action units).")
    print("ALOHA dims: 0-5 L_arm_j1..6, 6 L_gripper, 7-12 R_arm_j1..6, 13 R_gripper.\n")

    shown = min(n_print, len(recs)) if n_print > 0 else 0
    for rec in recs[:shown]:
        jm = rec["joint_motion"]
        if jm["chunk_is_static"]:
            desc = "STATIC (holds pose, no joint exceeds threshold)"
        else:
            tops = ", ".join(f"{j['name']}({j['peak_to_peak']:.3f})" for j in jm["top_moving_joints"]
                             if j["peak_to_peak"] > thr)
            desc = f"{jm['num_moving_joints']} joints moving; top: {tops}"
        print(f"Record {rec['record_index']:>3}: {desc}")
        print(f"           moving timesteps {jm['num_moving_timesteps']}/{jm['num_timesteps']-1}, "
              f"step delta mean {jm['step_delta_mean']:.4f} max {jm['step_delta_max']:.4f}")
    if shown:
        print()

    # Aggregate: how often each joint moves across all chunks.
    move_freq: collections.Counter = collections.Counter()
    for r in recs:
        move_freq.update(r["joint_motion"]["moving_joints"])
    n_static = sum(1 for r in recs if r["joint_motion"]["chunk_is_static"])
    print(f"Chunks with decoded actions: {len(recs)}  (static: {n_static})")
    if move_freq:
        print("Most-frequently-moving joints (joint: # chunks):")
        for name, c in move_freq.most_common(10):
            print(f"  {name:<10}: {c}")
    else:
        print("No joint exceeded the movement threshold in any chunk.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Per-timestep output: one CSV per record mapping timestep -> action + relevant joints
# ---------------------------------------------------------------------------
def write_per_timestep(out_subdir: str, processed: list[dict],
                       move_threshold: float, n_print: int) -> int:
    """Write one CSV per decoded record (timestep -> joint values + relevant joints),
    and print a compact preview. Returns the number of files written."""
    recs = [r for r in processed if r["_aloha_array"] is not None]

    print("=" * 70)
    print("PER-TIMESTEP ACTION MAPPING (one file per decoded chunk)")
    print("=" * 70)
    if not recs:
        print("No decoded continuous actions in the log, so per-timestep tables")
        print("cannot be built. Add decoded_actions/aloha_actions_14d to the")
        print("ExtractFASTActions record and re-run the sim. See Analysis 2 above.")
        print("=" * 70)
        return 0

    os.makedirs(out_subdir, exist_ok=True)

    try:
        import pandas as pd  # noqa: PLC0415
        have_pandas = True
    except ImportError:
        import csv
        have_pandas = False

    written = 0
    for rec in recs:
        rows = per_timestep_rows(rec["_aloha_array"], move_threshold, rec["_state_array"])
        path = os.path.join(out_subdir, f"record_{rec['record_index']}.csv")
        if have_pandas:
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        written += 1

    print(f"Wrote {written} per-timestep CSV file(s) to {out_subdir}/")
    print(f"A timestep's 'relevant_joints' = joints whose value changed by > {move_threshold} "
          "vs the previous timestep.\n")

    # Compact preview: for the first decoded record, show which joints are relevant
    # at each of the first few timesteps.
    preview = recs[0]
    rows = per_timestep_rows(preview["_aloha_array"], move_threshold, preview["_state_array"])
    print(f"Preview of record {preview['record_index']} (first {min(n_print, len(rows))} timesteps):")
    for row in rows[:n_print]:
        rel = row["relevant_joints"] or "(none)"
        print(f"  t={row['timestep']:>2}  step_delta={row['step_delta']:.4f}  relevant: {rel}")
    print("=" * 70)
    return written


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
    parser.add_argument("--token-motion", action="store_true",
                        help="Analysis 1: decode-free token motion proxy (works on any log); "
                             "prints a section and writes token_motion.csv")
    parser.add_argument("--joint-motion", action="store_true",
                        help="Analysis 2: per-joint movement from decoded actions (needs "
                             "decoded_actions/aloha_actions_14d in the log); prints a section "
                             "and writes joint_motion.json + joint_motion.csv")
    parser.add_argument("--joint-move-threshold", type=float, default=DEFAULT_JOINT_MOVE_THRESHOLD,
                        help="Per-dim peak-to-peak range above which a joint counts as 'moving'")
    parser.add_argument("--per-timestep", action="store_true",
                        help="Write one CSV per decoded chunk mapping each timestep to its joint "
                             "values and the joints that changed (needs decoded actions in the log)")
    parser.add_argument("--motion-print", type=int, default=10,
                        help="How many per-record lines to print in the motion sections")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    processed: list[dict] = []
    for i, record in enumerate(load_records(args.input, args.max_records)):
        processed.append(process_record(i, record, move_threshold=args.joint_move_threshold))

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

    # Optional analyses (each independent; run either, both, or neither).
    if args.token_motion:
        tm_path = os.path.join(args.output_dir, "token_motion.csv")
        write_token_motion_csv(tm_path, processed)
        written.append(tm_path)
    if args.joint_motion:
        jm_json = os.path.join(args.output_dir, "joint_motion.json")
        jm_csv = os.path.join(args.output_dir, "joint_motion.csv")
        write_joint_motion_json(jm_json, processed)
        write_joint_motion_csv(jm_csv, processed)
        written.extend([jm_json, jm_csv])

    print_aggregate(summary)
    if args.token_motion:
        print()
        print_token_motion(processed, args.motion_print)
    if args.joint_motion:
        print()
        print_joint_motion(processed, args.motion_print)
    if args.per_timestep:
        print()
        pt_dir = os.path.join(args.output_dir, "per_timestep")
        n = write_per_timestep(pt_dir, processed, args.joint_move_threshold, args.motion_print)
        if n:
            written.append(os.path.join(pt_dir, "record_*.csv"))

    print("\nWrote:")
    for p in written:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
