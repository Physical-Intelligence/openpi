#!/usr/bin/env python3
"""Fail-closed qualification gate for a G1 Coke RGB-D training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-reward-improvement", type=float, default=0.01)
    parser.add_argument("--minimum-best-step", type=int, default=500)
    parser.add_argument("--maximum-action-mae-rad", type=float, default=0.35)
    parser.add_argument("--maximum-chunk-step-rad", type=float, default=0.45)
    args = parser.parse_args()

    run_state = json.loads((args.run_dir / "run_state.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((args.run_dir / "reward_checkpoints.json").read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    records = [
        record
        for record in checkpoint_manifest.get("records", [])
        if (args.run_dir / str(record.get("step")) / "model.safetensors").is_file()
    ]
    if not records:
        raise RuntimeError("qualification requires at least one complete reward checkpoint")
    best = max(records, key=lambda record: float(record["reward"]))
    best_dir = args.run_dir / str(best["step"])
    baseline = float(run_state.get("baseline_checkpoint_reward", math.nan))
    best_reward = float(best["reward"])
    action_mae = float(evaluation.get("action_mae_rad_mean", math.inf))
    chunk_step = float(evaluation.get("predicted_chunk_step_rad_max", math.inf))
    reasons: list[str] = []
    if run_state.get("state") != "TRAINING_STEPS_COMPLETE":
        reasons.append("training_steps_incomplete")
    if not run_state.get("checkpoint_eval_uses_heldout_data"):
        reasons.append("checkpoint_evaluation_not_heldout")
    if run_state.get("checkpoint_metric") != "heldout_offline_imitation_reward":
        reasons.append("wrong_checkpoint_metric")
    if not math.isfinite(baseline) or not math.isfinite(best_reward):
        reasons.append("nonfinite_reward")
    elif best_reward - baseline < args.minimum_reward_improvement:
        reasons.append("insufficient_heldout_reward_improvement")
    if int(best["step"]) < args.minimum_best_step:
        reasons.append("best_checkpoint_too_early")
    if evaluation.get("status") != "passed":
        reasons.append("policy_evaluation_failed")
    if Path(str(evaluation.get("checkpoint", ""))).resolve() != best_dir.resolve():
        reasons.append("evaluation_checkpoint_is_not_best")
    if int(evaluation.get("hard_limit_violations", -1)) != 0:
        reasons.append("predicted_arm_hard_limit_violation")
    if not math.isfinite(action_mae) or action_mae > args.maximum_action_mae_rad:
        reasons.append("heldout_action_mae_too_high")
    if not math.isfinite(chunk_step) or chunk_step > args.maximum_chunk_step_rad:
        reasons.append("predicted_chunk_step_too_large")

    report = {
        "status": "qualified" if not reasons else "rejected",
        "kind": "g1_coke_rgbd_qualification_v1",
        "created_epoch": int(time.time()),
        "run_dir": str(args.run_dir),
        "checkpoint_eval_repo_id": run_state.get("checkpoint_eval_repo_id"),
        "baseline_heldout_reward": baseline,
        "best_heldout_reward": best_reward,
        "heldout_reward_improvement": best_reward - baseline,
        "best_checkpoint_step": int(best["step"]),
        "best_checkpoint": str(best_dir),
        "best_checkpoint_model_sha256": _sha256(best_dir / "model.safetensors"),
        "evaluation_report": str(args.evaluation_report),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "action_mae_rad_mean": action_mae,
        "predicted_chunk_step_rad_max": chunk_step,
        "rejection_reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
