import argparse
from copy import deepcopy
from pathlib import Path

import h5py
import pytest

from examples.ur5_twinwrist.config_loader import config_hash
from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.robot_runtime import MOTION_CONFIRMATION
from examples.ur5_twinwrist.teleop_collect import _override_config
from examples.ur5_twinwrist.teleop_collect import _run_real_collection
from examples.ur5_twinwrist.teleop_collect import collection_plan
from examples.ur5_twinwrist.teleop_collect import main
from examples.ur5_twinwrist.teleop_collect import record_fake_episodes


def test_fake_collection_produces_valid_atomic_episode(tmp_path: Path) -> None:
    project = load_project_config()
    before = config_hash(project)
    result = record_fake_episodes(project, episodes=1, frames=10, output=tmp_path)
    assert result == [{"path": str(tmp_path / "episode_0000.hdf5"), "frames": 10, "validation_errors": []}]
    assert not list(tmp_path.glob("*.tmp.hdf5"))
    with h5py.File(result[0]["path"], "r") as episode:
        assert episode["observations/qpos"].shape == (10, 9)
        assert episode["action"].shape == (10, 9)
        assert episode.attrs["robot_config_hash"] == before
    assert config_hash(project) == before


def test_default_cli_is_dry_and_does_not_require_legacy(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "examples.ur5_twinwrist.teleop_collect.inspect_station",
        lambda *_args, **_kwargs: {"ready": False, "motion_started": False, "checks": []},
    )
    assert main(["--skip-camera-enumeration"]) == 0
    output = capsys.readouterr().out
    assert "dry-run" in output
    assert "legacy-root" not in output


def test_collection_plan_lists_local_controller_locations(monkeypatch) -> None:
    monkeypatch.setattr(
        "examples.ur5_twinwrist.teleop_collect.inspect_station",
        lambda *_args, **_kwargs: {"ready": False},
    )
    plan = collection_plan(load_project_config(), Path("examples/ur5_twinwrist/config"), enumerate_cameras=False)
    assert plan["controllers"]["ur5"].endswith("controller/ur5.py")
    assert plan["controllers"]["cameras"].endswith("cameras/")
    assert plan["action"]["dimension"] == 9


def _real_ready_project() -> dict:
    project = deepcopy(load_project_config())
    project["safety"]["ur5"]["joint_min_rad"] = [-6.0] * 6
    project["safety"]["ur5"]["joint_max_rad"] = [6.0] * 6
    return project


def _real_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "enable_motion": True,
        "confirm": MOTION_CONFIRMATION,
        "config_dir": Path("examples/ur5_twinwrist/config"),
        "skip_camera_enumeration": False,
        "episodes": 1,
        "output": tmp_path,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_failed_preflight_does_not_instantiate_hardware(tmp_path: Path) -> None:
    constructed: list[str] = []

    def forbidden(_config):
        constructed.append("called")
        raise AssertionError("factory must not run")

    with pytest.raises(RuntimeError, match="预检未通过"):
        _run_real_collection(
            _real_ready_project(),
            _real_args(tmp_path),
            preflight_fn=lambda *_args, **_kwargs: {"ready": False, "checks": []},
            mouse_factory=forbidden,
            hardware_factory=forbidden,
        )
    assert constructed == []


def test_fixed_confirmation_is_checked_before_preflight_or_factories(tmp_path: Path) -> None:
    touched: list[str] = []
    with pytest.raises(PermissionError, match="--confirm"):
        _run_real_collection(
            _real_ready_project(),
            _real_args(tmp_path, confirm="wrong"),
            preflight_fn=lambda *_args, **_kwargs: touched.append("preflight") or {"ready": True},
            mouse_factory=lambda _config: touched.append("mouse"),
            hardware_factory=lambda _config: touched.append("hardware"),
        )
    assert touched == []


def test_all_gates_pass_before_factories_and_output_is_not_hashed(tmp_path: Path) -> None:
    project = _real_ready_project()
    baseline_hash = config_hash(project)
    order: list[str] = []

    class FakeSession:
        def __init__(self, _project, *, episodes, spacemouse, hardware, output):
            order.append("session")
            assert episodes == 1
            assert spacemouse == "mouse"
            assert hardware == "hardware"
            assert output == tmp_path

        def run(self):
            order.append("run")
            return {"successful_episodes": 1, "requested_episodes": 1}

    result = _run_real_collection(
        project,
        _real_args(tmp_path),
        preflight_fn=lambda *_args, **_kwargs: order.append("preflight") or {"ready": True, "checks": []},
        mouse_factory=lambda _config: order.append("mouse") or "mouse",
        hardware_factory=lambda _config: order.append("hardware") or "hardware",
        session_factory=FakeSession,
    )
    assert result == 0
    assert order == ["preflight", "mouse", "hardware", "session", "run"]
    assert config_hash(project) == baseline_hash


def test_output_override_is_not_part_of_effective_robot_config_hash(tmp_path: Path) -> None:
    project = load_project_config()
    args = argparse.Namespace(gripper_backend=None, output=tmp_path)
    effective = _override_config(project, args)
    assert effective["collection"]["storage"]["raw_root"] == project["collection"]["storage"]["raw_root"]
    assert config_hash(effective) == config_hash(project)
