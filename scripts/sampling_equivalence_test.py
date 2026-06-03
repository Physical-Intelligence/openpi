"""Deterministic sampling-equivalence harness for the TraceVLA family.

This is the regression gate for folding the sampling entrypoints
(``sample_trace`` / ``sample_actions``+``sample_actions_and_completion`` /
``predict_completion``) out of the per-variant model classes into
``TraceVLABase``. It drives the SAME public inference API that
``inference_example.py`` uses at deployment — a ``TraceVLAPolicy`` built by
``create_trained_trace_vla_policy`` — so the test exercises the real code path
rather than poking the model internals.

For each of the 7 LIBERO skills it builds a fixed synthetic observation and
calls:

  * ``policy.sample_trace(obs)``      → (N, 2) normalized trace   (planning forward)
  * ``policy.predict_completion(obs)`` → [0, 1] progress scalar   (completion only)
  * ``policy.infer(obs)``             → {"actions", "progress"}   (action + completion)

Determinism: the policy is constructed with its default rng (``jax.random.key(0)``)
and the methods are called in a fixed order, so the rng stream is reproducible
run-to-run. The sampling-code merge changes only the model method bodies (not the
policy's rng handling), so identical outputs across a before/after capture prove
the merge preserved behavior. Nothing here touches LIBERO, OpenRouter, or any LLM:
the plan and semantic-target points Gemini supplies at deployment are replaced
with fixed synthetic values, since only their byte-for-byte stability across the
two runs matters for an equivalence check.

Usage:
    # Capture a baseline for one config from its smoke checkpoint:
    python scripts/sampling_equivalence_test.py trace_vla_lora \\
        --out /tmp/eq_trace_vla_lora.npz

    # After the refactor, re-run and compare:
    python scripts/sampling_equivalence_test.py trace_vla_lora \\
        --out /tmp/eq_trace_vla_lora_post.npz
    python scripts/sampling_equivalence_test.py --compare \\
        /tmp/eq_trace_vla_lora.npz /tmp/eq_trace_vla_lora_post.npz
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib

# JAX memory cap must precede `import jax` (pulled in transitively below).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import numpy as np

from openpi.models import trace_utils as _trace_utils
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


# ---------------------------------------------------------------------------
# The 7 skills (and one representative parameterization each). Each maps to a
# hard-routed expert id via trace_utils.skill_to_expert_id; together they cover
# all 5 experts so every routing path is exercised.
# ---------------------------------------------------------------------------
SKILLS: list[str] = [
    "PICKUP_FROM(black bowl, table)",
    "PLACE_ON(black bowl, plate)",
    "PLACE_IN(black bowl, top drawer)",
    "OPEN(top drawer of the cabinet)",
    "CLOSE(top drawer of the cabinet)",
    "TURN_ON(stove)",
    "TURN_OFF(stove)",
]
PLAN_TEXT = " ".join(f"{i + 1}. {s}" for i, s in enumerate(SKILLS))
TASK_PROMPT = "put the black bowl in the top drawer of the cabinet and close it"

IMAGE_RES = 224
STATE_DIM = 8  # libero: eef_pos(3) + axisangle(3) + gripper_qpos(2); transform pads to model dim.


def _skill_bare(skill_text: str) -> str:
    return skill_text.split("(", 1)[0].strip().upper()


# ---------------------------------------------------------------------------
# Synthetic observation construction (deterministic per skill index).
# ---------------------------------------------------------------------------

def _synthetic_image(seed: int) -> np.ndarray:
    """Deterministic uint8 HxWx3 image — content is arbitrary but reproducible."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(IMAGE_RES, IMAGE_RES, 3), dtype=np.uint8)


def _make_obs_dict(skill_idx: int, skill_text: str, *, with_overlay: bool) -> dict:
    """Build the obs dict the TraceVLA policy expects (mirrors inference_example's
    make_planning_obs / make_execution_obs key set), with fully synthetic, fixed values.

    ``with_overlay=False`` is the planning-mode obs fed to ``sample_trace``;
    ``with_overlay=True`` is the execution-mode obs fed to ``predict_completion`` / ``infer``.
    """
    base_seed = 1000 + skill_idx
    rng = np.random.default_rng(base_seed)
    obs = {
        "observation/image": _synthetic_image(base_seed),
        "observation/wrist_image": _synthetic_image(base_seed + 500),
        "observation/state": rng.standard_normal(STATE_DIM).astype(np.float32),
        "atomic_token": float(_trace_utils.skill_to_expert_id(skill_text)),
        # Fixed-but-distinct semantic target / EE pixels in normalized [0, 1].
        "semantic_target_xy": np.array([0.40 + 0.02 * skill_idx, 0.55], dtype=np.float32),
        "current_ee_xy": np.array([0.50, 0.45 - 0.02 * skill_idx], dtype=np.float32),
        "skill_text": skill_text,
        "skill_name": _skill_bare(skill_text),
        "plan_text": PLAN_TEXT,
        "skill_step_num": int(skill_idx + 1),
        "prompt": TASK_PROMPT,
        "has_trace": True,
        "has_overlay": False,
        "progress": 0.0,
    }
    if with_overlay:
        obs["observation/overlay_image"] = _synthetic_image(base_seed + 900)
        obs["has_overlay"] = True
    return obs


# ---------------------------------------------------------------------------
# Checkpoint resolution.
# ---------------------------------------------------------------------------

def _resolve_step_dir(path: pathlib.Path) -> pathlib.Path:
    """Return ``path`` if it directly holds ``params/``, else its highest-numbered step subdir."""
    path = path.expanduser().resolve()
    if (path / "params").is_dir():
        return path
    steps = [c for c in path.iterdir() if c.is_dir() and c.name.isdigit() and (c / "params").is_dir()]
    if not steps:
        raise FileNotFoundError(f"No checkpoint step dir with params/ under {path}")
    return max(steps, key=lambda p: int(p.name))


def _default_checkpoint_dir(config_name: str) -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    return repo_root / "checkpoints" / config_name / "smoke"


# ---------------------------------------------------------------------------
# Hashing / comparison.
# ---------------------------------------------------------------------------

def _hash_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _compare(path_a: str, path_b: str, tol: float = 2e-2) -> int:
    """Compare two capture files. Reports per-key max|Δ| and a worst-case summary.

    Verdict is tolerance-based: GPU bf16 + XLA make the multi-step samplers only
    reproducible to ~1e-2 across separate runs/compilations, so exact-bit equality is
    too strict. ``IDENTICAL`` = bit-for-bit; ``PASS`` = within ``tol``; ``FAIL`` = above.
    """
    a = np.load(path_a)
    b = np.load(path_b)
    keys = sorted(set(a.files) | set(b.files))
    worst = 0.0
    n_fail = 0
    print(f"{'key':40s} {'max|Δ|':>12s}  status")
    for k in keys:
        if k not in a.files or k not in b.files:
            print(f"{k:40s} {'-':>12s}  MISSING in {'A' if k not in a.files else 'B'}")
            n_fail += 1
            continue
        xa, xb = np.asarray(a[k], np.float32), np.asarray(b[k], np.float32)
        if xa.shape != xb.shape:
            print(f"{k:40s} {'-':>12s}  SHAPE {xa.shape} vs {xb.shape}")
            n_fail += 1
            continue
        d = float(np.max(np.abs(xa - xb))) if xa.size else 0.0
        worst = max(worst, d)
        if _hash_array(xa) == _hash_array(xb):
            status = "IDENTICAL"
        elif d <= tol:
            status = "PASS"
        else:
            status = "FAIL"
            n_fail += 1
        print(f"{k:40s} {d:12.3e}  {status}")
    ok = n_fail == 0
    print(f"\n==> worst max|Δ| = {worst:.3e}  (tol={tol:.1e})  ->  "
          + ("PASS (within tolerance)" if ok else f"FAIL ({n_fail} keys)"))
    return 0 if ok else 1


def _add_compare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tol", type=float, default=2e-2,
                   help="Max |Δ| tolerance for the --compare verdict (default 2e-2).")


# ---------------------------------------------------------------------------
# Main capture path — drives the public policy inference API.
# ---------------------------------------------------------------------------

def _capture(args: argparse.Namespace) -> int:
    train_config = _config.get_config(args.config_name)
    ckpt_dir = pathlib.Path(args.checkpoint_dir) if args.checkpoint_dir else _default_checkpoint_dir(args.config_name)
    step_dir = _resolve_step_dir(ckpt_dir)
    print(f"[eq] config={args.config_name} checkpoint={step_dir}", flush=True)

    # norm_stats=None -> loaded from <step_dir>/assets (smoke checkpoints write assets).
    # The policy's rng defaults to jax.random.key(0); with a fixed call order below
    # the whole run is reproducible.
    policy = _policy_config.create_trained_trace_vla_policy(train_config, step_dir)

    skills = SKILLS if not args.max_skills else SKILLS[: args.max_skills]
    out: dict[str, np.ndarray] = {}
    print(f"\n{'skill':16s} {'expert':>6s}  trace / actions / progress hashes", flush=True)
    for i, skill_text in enumerate(skills):
        bare = _skill_bare(skill_text)
        plan_obs = _make_obs_dict(i, skill_text, with_overlay=False)
        exec_obs = _make_obs_dict(i, skill_text, with_overlay=True)

        # Public inference entrypoints, same as inference_example.py.
        trace = np.asarray(policy.sample_trace(plan_obs, num_steps=args.num_steps))      # (N, 2)
        progress_only = np.asarray(policy.predict_completion(exec_obs))                  # scalar
        result = policy.infer(exec_obs)                                                  # action + completion
        actions = np.asarray(result["actions"])                                          # (ah, ad)
        progress = np.asarray(result["progress"])                                        # scalar

        # Cast to float32 on save: model outputs can be bfloat16 (progress) or float64
        # (unnormalized actions); bfloat16 does not round-trip through np.savez/np.load.
        key = f"{bare}#{i}"
        out[f"{key}/trace"] = np.asarray(trace, dtype=np.float32)
        out[f"{key}/actions"] = np.asarray(actions, dtype=np.float32)
        out[f"{key}/progress"] = np.asarray(progress, dtype=np.float32)
        out[f"{key}/progress_only"] = np.asarray(progress_only, dtype=np.float32)
        print(
            f"{bare:16s} {_trace_utils.skill_to_expert_id(skill_text):>6d}  "
            f"trace={_hash_array(trace)} act={_hash_array(actions)} "
            f"prog={_hash_array(progress)} prog0={_hash_array(progress_only)}",
            flush=True,
        )

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    combined = hashlib.sha256("".join(_hash_array(out[k]) for k in sorted(out)).encode()).hexdigest()[:16]
    print(f"\n[eq] wrote {args.out} ({len(out)} arrays)  combined-hash={combined}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="TraceVLA sampling-equivalence harness.")
    p.add_argument("config_name", nargs="?", help="TrainConfig name (e.g. trace_vla_lora).")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Checkpoint dir (or a parent holding step subdirs). "
                        "Default: checkpoints/<config>/smoke.")
    p.add_argument("--out", default=None, help="Output .npz path (default /tmp/eq_<config>.npz).")
    p.add_argument("--num-steps", type=int, default=10, help="Euler steps for action/trace denoising.")
    p.add_argument("--max-skills", type=int, default=0,
                   help="Cap the number of skills exercised (0 = all 7). Useful for the slower CPU check.")
    p.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"),
                   help="Compare two capture files and exit (no model load).")
    _add_compare_args(p)
    args = p.parse_args()

    if args.compare:
        return _compare(args.compare[0], args.compare[1], tol=args.tol)
    if not args.config_name:
        p.error("config_name is required unless --compare is used.")
    if not args.out:
        args.out = f"/tmp/eq_{args.config_name}.npz"
    return _capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
