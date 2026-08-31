# ruff: noqa: E402, RUF001, RUF002, RUF003
"""当前项目的只读真机预检：解析配置、枚举设备，但绝不发送运动。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from importlib import metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.ur5_twinwrist.config_loader import CONFIG_FILES
from examples.ur5_twinwrist.config_loader import DEFAULT_CONFIG_DIR
from examples.ur5_twinwrist.config_loader import config_hash
from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.config_loader import validate_project_config


def _git(command: list[str], root: Path) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr.strip()}"


def _service_active(name: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["systemctl", "is-active", name],
        check=False,
        capture_output=True,
        text=True,
    )
    detail = (result.stdout or result.stderr).strip()
    return result.returncode == 0 and detail == "active", detail


def _camera_serials() -> tuple[list[str], str | None]:
    try:
        import pyrealsense2 as rs

        devices = rs.context().query_devices()
        serials = sorted(str(device.get_info(rs.camera_info.serial_number)) for device in devices)
        return serials, None
    except BaseException as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _tcp_reachable(host: str, port: int = 30004, timeout_s: float = 0.20) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True, f"{host}:{port} 可连接"
    except OSError as exc:
        return False, f"{host}:{port} 不可连接: {exc}"


def _active_collectors() -> list[str]:
    # 旧网页前端即使尚未启动 UR 控制，也可能常驻一个 SpaceMouse worker；
    # 新 collector 与它同时读取 spacenavd 会造成按键/轴状态竞争，因此同样
    # 视为互斥占用。这里只读进程表，不会停止任何用户进程。
    conflict_pattern = (
        "teleop_collect|collect_real|robot_runtime|"
        "collection_frontend|slai_mi\\.devices\\.spacemouse\\.workers"
    )
    conflict_markers = (
        "teleop_collect",
        "collect_real",
        "robot_runtime",
        "collection_frontend",
        "slai_mi.devices.spacemouse.workers",
    )
    result = subprocess.run(
        ["pgrep", "-af", conflict_pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    current_pid = str(os.getpid())
    return [
        line
        for line in result.stdout.splitlines()
        if line.split(maxsplit=1)[0] != current_pid
        and "teleop_preflight" not in line
        and "pgrep -af" not in line
        and any(marker in line for marker in conflict_markers)
    ]


def inspect_station(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    gripper_backend: str | None = None,
    *,
    enumerate_cameras: bool = True,
) -> dict[str, Any]:
    """返回机器可读预检报告; 本函数不会打开串口、相机流或 RTDE 控制。"""

    root = Path(__file__).resolve().parents[2]
    config_root = Path(config_dir).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, detail: Any, *, passed: bool, required: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "required": required, "detail": detail})

    try:
        config = load_project_config(config_root)
    except BaseException as exc:
        return {
            "ready": False,
            "motion_started": False,
            "config_dir": str(config_root),
            "checks": [
                {
                    "name": "project_config",
                    "passed": False,
                    "required": True,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    if gripper_backend is not None:
        config = deepcopy(config)
        config["hardware"]["gripper"]["backend"] = gripper_backend
        validate_project_config(config)
    check("project_config", "五份 YAML 结构与字段一致", passed=True)

    try:
        validate_project_config(config, require_real_ready=True)
        real_ready_detail = "PolyScope 关节软限位已配置"
        real_ready = True
    except ValueError as exc:
        real_ready_detail = str(exc)
        real_ready = False
    check("calibrated_safety_limits", real_ready_detail, passed=real_ready)

    versions: dict[str, str] = {}
    for package in ("pyserial", "pyrealsense2", "ur-rtde"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "未安装"
    check(
        "hardware_python_dependencies",
        versions,
        passed=versions == {"pyserial": "3.5", "pyrealsense2": "2.56.5.9235", "ur-rtde": "1.6.3"},
    )

    spacemouse_service, service_detail = _service_active(config["hardware"]["spacemouse"]["daemon_service"])
    check("spacenavd_active", service_detail, passed=spacemouse_service)

    gripper = config["hardware"]["gripper"]
    selected_backend = str(gripper["backend"])
    selected_gripper = gripper["adapters"][selected_backend]
    serial_paths = {
        "wrist_master": str(config["hardware"]["wrist"]["master_port"]),
        "wrist_controller": str(config["hardware"]["wrist"]["controller_port"]),
        "gripper": str(selected_gripper["port"]),
    }
    for name, value in serial_paths.items():
        check(f"{name}_present", value, passed=Path(value).exists())

    expected_camera_serials = {
        str(item["role"]): str(item["serial"]) for item in config["hardware"]["cameras"]["devices"]
    }
    if enumerate_cameras:
        detected_camera_serials, camera_error = _camera_serials()
        camera_ok = set(expected_camera_serials.values()) <= set(detected_camera_serials)
        check(
            "configured_cameras_present",
            {"expected": expected_camera_serials, "detected": detected_camera_serials, "error": camera_error},
            passed=camera_ok,
        )
    else:
        detected_camera_serials = []
        camera_error = "按调用方要求跳过枚举"
        check("configured_cameras_present", camera_error, passed=True, required=False)

    ur_host = str(config["hardware"]["ur5"]["host"])
    ur_reachable, ur_detail = _tcp_reachable(ur_host)
    check("ur_rtde_reachable", ur_detail, passed=ur_reachable)

    collectors = _active_collectors()
    check("no_other_collector", collectors or "未发现其他采集/推理进程", passed=not collectors)

    config_files = {name: str(config_root / name) for name in CONFIG_FILES}
    status = _git(["status", "--short"], root).splitlines()
    required_checks = [item for item in checks if item["required"]]
    return {
        "ready": all(item["passed"] for item in required_checks),
        "motion_started": False,
        "backend": "jax",
        "project_root": str(root),
        "git_commit": _git(["rev-parse", "HEAD"], root),
        "git_status_count": len(status),
        "git_status": status,
        "config_dir": str(config_root),
        "config_files": config_files,
        "config_sha256": config_hash(config),
        "selected_gripper": selected_backend,
        "serial_paths": serial_paths,
        "expected_camera_serials": expected_camera_serials,
        "detected_camera_serials": detected_camera_serials,
        "camera_enumeration_error": camera_error,
        "ur_host": ur_host,
        "capture_parameters": {
            "camera_fps": config["collection"]["capture"]["camera_fps"],
            "record_hz": config["collection"]["capture"]["record_hz"],
            "max_camera_skew_ms": config["safety"]["timing"]["max_camera_skew_ms"],
        },
        "task_home_rad": config["poses"]["ur5"]["task_home_rad"],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--gripper-backend", choices=("hiwonder", "feetech"))
    parser.add_argument("--skip-camera-enumeration", action="store_true")
    args = parser.parse_args()
    report = inspect_station(
        args.config_dir,
        args.gripper_backend,
        enumerate_cameras=not args.skip_camera_enumeration,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
