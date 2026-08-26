#!/usr/bin/env python3
"""Recover only the known 4k wrapper-state qualification mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate(run_dir: Path, session_path: Path, qualification_path: Path) -> tuple[dict, dict, dict]:
    run_state_path = run_dir / "run_state.json"
    session = _load(session_path)
    run_state = _load(run_state_path)
    qualification = _load(qualification_path)

    if session.get("state") != "SESSION_FAILED" or int(session.get("exit_code", -1)) != 2:
        raise ValueError("session is not the known qualification rejection")
    if int(session.get("deadline_epoch", 0)) <= int(time.time()) + 900:
        raise ValueError("original supervised session deadline has expired or is too close")
    if run_state.get("state") != "TRAINING_COMPLETE" or int(run_state.get("global_step", -1)) != 4000:
        raise ValueError("4k training did not complete with the legacy wrapper state")
    if not run_state.get("checkpoint_eval_uses_heldout_data"):
        raise ValueError("checkpoint evaluation was not held out")
    if run_state.get("checkpoint_metric") != "heldout_offline_imitation_reward":
        raise ValueError("unexpected checkpoint metric")
    if qualification.get("status") != "rejected" or qualification.get("rejection_reasons") != [
        "training_steps_incomplete"
    ]:
        raise ValueError("qualification did not fail solely on the known state mismatch")
    if not math.isfinite(float(qualification.get("heldout_reward_improvement", math.nan))):
        raise ValueError("qualification improvement is non-finite")

    best_checkpoint = Path(str(qualification.get("best_checkpoint", "")))
    model_path = best_checkpoint / "model.safetensors"
    if not model_path.is_file() or _sha256(model_path) != qualification.get("best_checkpoint_model_sha256"):
        raise ValueError("best checkpoint identity does not match qualification evidence")
    evaluation_path = Path(str(qualification.get("evaluation_report", "")))
    if not evaluation_path.is_file() or _sha256(evaluation_path) != qualification.get("evaluation_report_sha256"):
        raise ValueError("held-out evaluation identity does not match qualification evidence")
    evaluation = _load(evaluation_path)
    if evaluation.get("status") != "passed" or int(evaluation.get("hard_limit_violations", -1)) != 0:
        raise ValueError("held-out policy evaluation did not pass")
    if Path(str(evaluation.get("checkpoint", ""))).resolve() != best_checkpoint.resolve():
        raise ValueError("held-out evaluation did not use the best checkpoint")
    return session, run_state, qualification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session-state", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    session, run_state, qualification = _validate(
        args.run_dir,
        args.session_state,
        args.qualification_report,
    )
    if args.apply:
        recovered_epoch = int(time.time())
        run_state.update(
            state="TRAINING_STEPS_COMPLETE",
            qualification_recovery_epoch=recovered_epoch,
            qualification_recovery_source="legacy_4k_wrapper_state_mismatch",
        )
        session.update(
            state="QUALIFICATION_RECOVERY_RUNNING",
            qualification_recovery_epoch=recovered_epoch,
            qualification_recovery_source="legacy_4k_wrapper_state_mismatch",
            prior_failure={"failed_epoch": session.get("failed_epoch"), "exit_code": session.get("exit_code")},
        )
        session.pop("failed_epoch", None)
        session.pop("exit_code", None)
        _atomic_write(args.run_dir / "run_state.json", run_state)
        _atomic_write(args.session_state, session)
    print(
        json.dumps(
            {
                "status": "recoverable" if args.check else "recovered",
                "best_checkpoint": qualification["best_checkpoint"],
                "deadline_epoch": session["deadline_epoch"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
