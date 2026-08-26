from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).with_name("qualify_g1_coke_rgbd_run.py")


def _fixture(tmp_path: Path, *, best_reward: float = 0.80) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "800"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "state": "TRAINING_STEPS_COMPLETE",
                "checkpoint_eval_uses_heldout_data": True,
                "checkpoint_metric": "heldout_offline_imitation_reward",
                "checkpoint_eval_repo_id": "local/eval",
                "baseline_checkpoint_reward": 0.72,
            }
        )
    )
    (run_dir / "reward_checkpoints.json").write_text(
        json.dumps({"records": [{"step": 800, "reward": best_reward}]})
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "status": "passed",
                "checkpoint": str(checkpoint),
                "hard_limit_violations": 0,
                "action_mae_rad_mean": 0.10,
                "predicted_chunk_step_rad_max": 0.20,
            }
        )
    )
    return run_dir, evaluation, tmp_path / "qualification.json"


def _run(run_dir: Path, evaluation: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--evaluation-report",
            str(evaluation),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_qualification_accepts_improving_heldout_checkpoint(tmp_path: Path) -> None:
    run_dir, evaluation, output = _fixture(tmp_path)

    result = _run(run_dir, evaluation, output)

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "qualified"


def test_qualification_rejects_nonimproving_checkpoint(tmp_path: Path) -> None:
    run_dir, evaluation, output = _fixture(tmp_path, best_reward=0.725)

    result = _run(run_dir, evaluation, output)

    assert result.returncode == 2
    report = json.loads(output.read_text())
    assert report["status"] == "rejected"
    assert "insufficient_heldout_reward_improvement" in report["rejection_reasons"]
