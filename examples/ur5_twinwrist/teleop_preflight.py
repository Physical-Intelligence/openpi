"""Read-only preflight for the legacy TASK2 teleoperation station."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(command: list[str], *, cwd: Path, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=str(exc.stdout or ""),
            stderr=f"timeout after {timeout:.1f}s",
        )


def inspect_station(
    legacy_root: str | Path,
    legacy_python: str | Path,
    gripper_backend: str | None = None,
) -> dict:
    root = Path(legacy_root).expanduser().resolve()
    python = Path(legacy_python).expanduser().absolute()
    hardware_path = root / "configs/hardware.yaml"
    task_path = root / "configs/tasks/task2_continuous.yaml"
    strategy_path = root / "configs/strategies/ur5e_wrist_gripper_9dof_collection.yaml"
    hardware, task, strategy = map(_load, (hardware_path, task_path, strategy_path))
    task_home = (task_path.parent / str(task["start_pose_ref"])).resolve()
    control_profile = (task_path.parent / str(task["control_profile_ref"])).resolve()
    pose, control = _load(task_home), _load(control_profile)
    schema_path = (root / str(strategy["dataset"]["input_schema"])).resolve()
    schema = _load(schema_path)
    wrist_config = Path(str(hardware.get("wrist_sensor", {}).get("config", ""))).expanduser()
    wrist_runtime = _load(wrist_config) if wrist_config.is_file() else {}
    checks: list[dict[str, object]] = []

    def check(name: str, passed: object, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("hardware_configured", hardware.get("configured") is True, str(hardware_path))
    check("legacy_python", python.is_file(), str(python))
    check(
        "continuous_strategy",
        strategy.get("dataset", {}).get("state_schema") == "task2_9dof_continuous",
        str(strategy_path),
    )
    check("task_home_pose", task_home.is_file() and pose.get("configured") is True, str(task_home))
    check("control_profile", control_profile.is_file(), str(control_profile))
    check("input_schema", schema_path.is_file(), str(schema_path))
    check("wrist_runtime_config", wrist_config.is_file(), str(wrist_config))
    ur5 = hardware.get("ur5", {})
    check("ur_host", bool(str(ur5.get("host", "")).strip()), "hardware.ur5.host")
    ur_driver_python = Path(str(ur5.get("driver_python", ""))).expanduser()
    check("ur_driver_python", ur_driver_python.is_file(), str(ur_driver_python))
    cameras = hardware.get("cameras", {}).get("devices", [])
    serials = [str(item.get("serial", "")) for item in cameras]
    check("three_unique_cameras", len(serials) == 3 and len(set(serials)) == 3 and all(serials), repr(serials))
    selected = (
        {"hiwonder": "hiwonder", "feetech": "feetech_sts3215"}[gripper_backend]
        if gripper_backend is not None
        else hardware.get("gripper", {}).get("driver")
    )
    selected_config = hardware.get("gripper", {}).get("adapters", {}).get(selected, {})
    serial_paths = {
        "gripper": str(selected_config.get("port", "")),
        "wrist_master": str(hardware.get("wrist_sensor", {}).get("teleop_port", "")),
        "wrist_openrb": str(hardware.get("wrist_sensor", {}).get("openrb_port", "")),
    }
    for name, value in serial_paths.items():
        check(f"{name}_uses_by_id", value.startswith("/dev/serial/by-id/"), value)
        check(f"{name}_present", Path(value).exists(), value)
    imports = subprocess.run(
        [str(python), "-c", "import numpy, yaml, serial, pyrealsense2, lerobot, slai_mi"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    check("legacy_imports", imports.returncode == 0, imports.stderr.strip() or "ok")
    service = _command(["systemctl", "--user", "is-active", "slai-wrist-collection.service"], cwd=root)
    check(
        "collection_service_inactive",
        service.stdout.strip() != "active",
        service.stdout.strip() or service.stderr.strip() or "unknown",
    )
    spacenavd = _command(["systemctl", "is-active", "spacenavd"], cwd=root)
    check(
        "spacenavd_active",
        spacenavd.stdout.strip() == "active",
        spacenavd.stdout.strip() or spacenavd.stderr.strip() or "unknown",
    )
    collectors = _command(["pgrep", "-fa", r"slai_mi\.apps\.collect_real.*--execute-real"], cwd=root)
    check(
        "no_other_real_collector",
        collectors.returncode == 1,
        collectors.stdout.strip() or "none",
    )
    camera_probe = _command(
        [
            str(python),
            "-c",
            (
                "import json, pyrealsense2 as rs; "
                "print(json.dumps([d.get_info(rs.camera_info.serial_number) "
                "for d in rs.context().query_devices()]))"
            ),
        ],
        cwd=root,
        timeout=8.0,
    )
    try:
        detected_camera_serials = json.loads(camera_probe.stdout) if camera_probe.returncode == 0 else []
    except json.JSONDecodeError:
        detected_camera_serials = []
    check(
        "configured_cameras_present",
        set(serials) <= set(detected_camera_serials),
        repr(detected_camera_serials),
    )
    git_commit = _command(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip() or "unknown"
    git_status = _command(["git", "status", "--short"], cwd=root).stdout.splitlines()
    service_workdir = _command(
        [
            "systemctl",
            "--user",
            "show",
            "slai-wrist-collection.service",
            "-p",
            "WorkingDirectory",
            "--value",
        ],
        cwd=root,
    ).stdout.strip()
    config_paths = (hardware_path, task_path, strategy_path, task_home, control_profile, schema_path, wrist_config)
    relative_config_paths = [str(path.relative_to(root)) for path in config_paths if path.is_relative_to(root)]
    relevant_status = _command(["git", "status", "--short", "--", *relative_config_paths], cwd=root).stdout.splitlines()
    return {
        "backend": "jax",
        "motion_enabled": False,
        "paths": {
            "legacy_root": str(root),
            "hardware": str(hardware_path),
            "task": str(task_path),
            "strategy": str(strategy_path),
            "task_home": str(task_home),
            "control_profile": str(control_profile),
            "input_schema": str(schema_path),
            "wrist_runtime": str(wrist_config),
        },
        "legacy_git_commit": git_commit,
        "legacy_git_dirty": bool(git_status),
        "legacy_git_status_count": len(git_status),
        "legacy_git_relevant_status": relevant_status,
        "config_sha256": {str(path): _sha256(path) for path in config_paths if path.is_file()},
        "task_home": {
            "ur5_joint_rad": (pose.get("joint_positions") or [])[:6],
            "wrist_fe_ru_rad": (pose.get("joint_positions") or [])[6:8],
            "gripper_normalized": (pose.get("joint_positions") or [])[8:9],
            "is_mechanical_zero": False,
        },
        "selected_gripper": selected,
        "selected_gripper_parameters": selected_config,
        "camera_roles": cameras,
        "detected_camera_serials": detected_camera_serials,
        "capture_parameters": schema.get("capture", {}),
        "serial_paths": serial_paths,
        "ur5_parameters": hardware.get("ur5", {}),
        "motion_parameters": control.get("motion", {}),
        "synchronization_parameters": schema.get("synchronization", {}),
        "wrist_runtime_parameters": {
            key: wrist_runtime.get(key, {})
            for key in ("stream", "master_mapping", "target_limiter", "controller", "settling")
        },
        "action_contract": {
            "state": "UR actual_q[6] rad + wrist actual FE/RU[2] rad + gripper actual[1] normalized",
            "action": "accepted speedL twist[6] (m/s,rad/s) + wrist absolute target[2] rad + gripper absolute target[1]",
            "raw_spacemouse_is_training_action": False,
        },
        "service_working_directory": service_workdir,
        "bindings_note": "controls YAML bindings are documentation only; runtime keys are hard-coded",
        "checks": checks,
        "ready": all(bool(item["passed"]) for item in checks),
    }


def main() -> int:
    default = Path("/home/user/shiyi/slai-manipulation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", default=str(default))
    parser.add_argument("--legacy-python", default=str(default / ".venv-lerobot-v3/bin/python"))
    parser.add_argument("--gripper-backend", choices=("hiwonder", "feetech"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = inspect_station(args.legacy_root, args.legacy_python, args.gripper_backend)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 2 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
