from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPT = Path(__file__).with_name("recover_g1_coke_rgbd_qualification.py")


def _fixture(tmp_path: Path, *, rejection_reasons: list[str] | None = None) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "1400"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"model")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps({"status": "passed", "hard_limit_violations": 0, "checkpoint": str(checkpoint)})
    )
    run_state = {
        "state": "TRAINING_COMPLETE",
        "global_step": 4000,
        "checkpoint_eval_uses_heldout_data": True,
        "checkpoint_metric": "heldout_offline_imitation_reward",
    }
    (run_dir / "run_state.json").write_text(json.dumps(run_state))
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "state": "SESSION_FAILED",
                "exit_code": 2,
                "failed_epoch": int(time.time()),
                "started_epoch": int(time.time()) - 100,
                "deadline_epoch": int(time.time()) + 3600,
            }
        )
    )
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "status": "rejected",
                "rejection_reasons": rejection_reasons or ["training_steps_incomplete"],
                "heldout_reward_improvement": 0.12,
                "best_checkpoint": str(checkpoint),
                "best_checkpoint_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "evaluation_report": str(evaluation),
                "evaluation_report_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
            }
        )
    )
    return run_dir, session, qualification


def _run(run_dir: Path, session: Path, qualification: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--session-state",
            str(session),
            "--qualification-report",
            str(qualification),
            mode,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_recovers_only_known_wrapper_state_mismatch(tmp_path: Path) -> None:
    run_dir, session, qualification = _fixture(tmp_path)

    check = _run(run_dir, session, qualification, "--check")
    apply = _run(run_dir, session, qualification, "--apply")

    assert check.returncode == 0, check.stderr
    assert apply.returncode == 0, apply.stderr
    assert json.loads((run_dir / "run_state.json").read_text())["state"] == "TRAINING_STEPS_COMPLETE"
    assert json.loads(session.read_text())["state"] == "QUALIFICATION_RECOVERY_RUNNING"


def test_refuses_any_additional_qualification_failure(tmp_path: Path) -> None:
    run_dir, session, qualification = _fixture(
        tmp_path,
        rejection_reasons=["training_steps_incomplete", "heldout_action_mae_too_high"],
    )

    result = _run(run_dir, session, qualification, "--check")

    assert result.returncode != 0
    assert json.loads((run_dir / "run_state.json").read_text())["state"] == "TRAINING_COMPLETE"
