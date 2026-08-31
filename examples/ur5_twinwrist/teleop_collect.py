# ruff: noqa: RUF001, RUF002, RUF003
"""项目内 SpaceMouse 遥操作与原始 HDF5 数采入口。

默认只打印计划和只读预检，不打开 RTDE、串口或相机流。``--fake`` 可完成
无硬件原子 HDF5 验收。真实运动必须同时通过 YAML 安全参数、现场预检、
``--enable-motion`` 和固定确认短语四道门禁。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    # Direct script execution puts only examples/ur5_twinwrist on sys.path.
    # Add the repository root so package imports remain identical to ``-m``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from .config_loader import CONFIG_FILES
    from .config_loader import DEFAULT_CONFIG_DIR
    from .config_loader import config_hash
    from .config_loader import load_project_config
    from .config_loader import validate_project_config
    from .controller.spacemouse import SpaceMouse
    from .controller.spacemouse import SpaceMouseConfig
    from .episode_recorder import EpisodeRecorder
    from .robot_runtime import MOTION_CONFIRMATION
    from .robot_runtime import FakeHardware
    from .robot_runtime import safety_filter
    from .teleop_hardware import TeleopHardwareWorker
    from .teleop_preflight import inspect_station
    from .teleop_session import TeleopCollectionSession
    from .validate_dataset import validate_episode
except ImportError:  # 兼容 ``uv run examples/.../teleop_collect.py``
    from examples.ur5_twinwrist.config_loader import CONFIG_FILES
    from examples.ur5_twinwrist.config_loader import DEFAULT_CONFIG_DIR
    from examples.ur5_twinwrist.config_loader import config_hash
    from examples.ur5_twinwrist.config_loader import load_project_config
    from examples.ur5_twinwrist.config_loader import validate_project_config
    from examples.ur5_twinwrist.controller.spacemouse import SpaceMouse
    from examples.ur5_twinwrist.controller.spacemouse import SpaceMouseConfig
    from examples.ur5_twinwrist.episode_recorder import EpisodeRecorder
    from examples.ur5_twinwrist.robot_runtime import MOTION_CONFIRMATION
    from examples.ur5_twinwrist.robot_runtime import FakeHardware
    from examples.ur5_twinwrist.robot_runtime import safety_filter
    from examples.ur5_twinwrist.teleop_hardware import TeleopHardwareWorker
    from examples.ur5_twinwrist.teleop_preflight import inspect_station
    from examples.ur5_twinwrist.teleop_session import TeleopCollectionSession
    from examples.ur5_twinwrist.validate_dataset import validate_episode


def _override_config(project: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = deepcopy(project)
    if args.gripper_backend:
        result["hardware"]["gripper"]["backend"] = args.gripper_backend
    return result


def _bindings(project: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "label": item["label"],
            "code": int(item["code"]),
            "action": item["action"],
        }
        for name, item in project["teleop"]["buttons"].items()
    }


def collection_plan(
    project: dict[str, Any],
    config_dir: Path,
    *,
    enumerate_cameras: bool,
    output: Path | None = None,
) -> dict[str, Any]:
    """生成不会打开真实设备的可读计划。"""

    preflight = inspect_station(
        config_dir,
        str(project["hardware"]["gripper"]["backend"]),
        enumerate_cameras=enumerate_cameras,
    )
    return {
        "mode": "dry-run（未打开 RTDE/串口/相机流，未发送运动）",
        "backend": "jax",
        "config_dir": str(config_dir),
        "config_files": [str(config_dir / name) for name in CONFIG_FILES],
        "config_sha256": config_hash(project),
        "controllers": {
            "ur5": "examples/ur5_twinwrist/controller/ur5.py",
            "spacemouse": "examples/ur5_twinwrist/controller/spacemouse.py",
            "wrist": "examples/ur5_twinwrist/controller/wrist.py",
            "gripper": "examples/ur5_twinwrist/controller/gripper.py",
            "cameras": "examples/ur5_twinwrist/cameras/",
            "hardware_worker": "examples/ur5_twinwrist/teleop_hardware.py",
            "episode_recorder": "examples/ur5_twinwrist/episode_recorder.py",
        },
        "output": str(
            (output or Path(project["collection"]["storage"]["raw_root"])).expanduser()
        ),
        "task": project["collection"]["task"],
        "rates_hz": {
            "ur5_control": project["hardware"]["ur5"]["control_hz"],
            "camera_capture": project["collection"]["capture"]["camera_fps"],
            "hdf5_record": project["collection"]["capture"]["record_hz"],
        },
        "action": project["collection"]["action"],
        "task_home_rad": project["poses"]["ur5"]["task_home_rad"],
        "wrist_zero_method": project["poses"]["wrist"]["zero_method"],
        "gripper_zero": project["poses"]["gripper"],
        "spacemouse_axis_mapping": project["teleop"]["axes"]["mapping"],
        "spacemouse_buttons": _bindings(project),
        "preflight": preflight,
    }


def _synthetic_observation(observation: dict[str, Any], timestamp_ns: int, sequence: int) -> dict[str, Any]:
    result = dict(observation)
    result["timestamp_ns"] = timestamp_ns
    result["monotonic_ns"] = timestamp_ns
    result["camera_timestamps_ns"] = dict.fromkeys(("front", "side", "top"), timestamp_ns)
    result["camera_host_timestamps_ns"] = dict.fromkeys(("front", "side", "top"), timestamp_ns)
    result["camera_sequences"] = dict.fromkeys(("front", "side", "top"), sequence)
    return result


def record_fake_episodes(
    project: dict[str, Any],
    *,
    episodes: int,
    frames: int,
    output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """快速生成可验证的模拟 Episode；不 sleep、不连接任何硬件。"""

    output = Path(output or project["collection"]["storage"]["raw_root"]).expanduser().resolve()
    record_hz = float(project["collection"]["capture"]["record_hz"])
    period_ns = round(1e9 / record_hz)
    hardware = FakeHardware(
        servo_zero_raw=tuple(int(value) for value in project["poses"]["wrist"]["servo_zero_raw"])
    )
    hardware.connect()
    results: list[dict[str, Any]] = []
    try:
        with EpisodeRecorder(output, project) as recorder:
            base_ns = time.monotonic_ns()
            sequence = 0
            for episode_index in range(episodes):
                first = _synthetic_observation(hardware.get_observation(), base_ns, sequence)
                recorder.start(first)
                for frame_index in range(frames):
                    sequence += 1
                    timestamp_ns = base_ns + (episode_index * frames + frame_index) * period_ns
                    observation = _synthetic_observation(hardware.get_observation(), timestamp_ns, sequence)
                    candidate = np.zeros(9, dtype=np.float32)
                    # 小幅变化只用于证明 state/action 的 9D 数据路径工作；它不会离开 FakeHardware。
                    candidate[6] = 2.0 * np.sin(frame_index / 3.0)
                    candidate[7] = 2.0 * np.cos(frame_index / 3.0)
                    candidate[8] = float(frame_index >= frames // 2)
                    action = safety_filter(candidate, observation, project, now_ns=timestamp_ns)
                    applied = hardware.send_action(action)
                    recorder.append_if_due(observation, applied)
                path = recorder.save()
                errors = validate_episode(
                    path,
                    max_camera_skew_ms=float(project["safety"]["timing"]["max_camera_skew_ms"]),
                    expected_image_shape=tuple(
                        int(value) for value in project["collection"]["capture"]["image_shape"]
                    ),
                    expected_wrist_servo_zero_raw=tuple(
                        int(value) for value in project["poses"]["wrist"]["servo_zero_raw"]
                    ),
                )
                results.append(
                    {
                        "path": str(path),
                        "frames": frames,
                        "validation_errors": errors,
                    }
                )
    finally:
        hardware.stop()
        hardware.close()
    return results


def _run_real_collection(
    project: dict[str, Any],
    args: argparse.Namespace,
    *,
    preflight_fn: Any | None = None,
    mouse_factory: Any | None = None,
    hardware_factory: Any | None = None,
    session_factory: Any | None = None,
) -> int:
    """在全部门禁通过后才构造 SpaceMouse、硬件 worker 与数采会话。"""

    if not bool(args.enable_motion):
        raise PermissionError("真实数采必须显式提供 --enable-motion")
    if args.confirm != MOTION_CONFIRMATION:
        raise PermissionError(f"真实数采必须提供 --confirm {MOTION_CONFIRMATION}")
    validate_project_config(project, require_real_ready=True)
    inspect = inspect_station if preflight_fn is None else preflight_fn
    report = inspect(
        args.config_dir,
        str(project["hardware"]["gripper"]["backend"]),
        enumerate_cameras=not bool(args.skip_camera_enumeration),
    )
    if not bool(report.get("ready", False)):
        failed = [
            item.get("name", "unknown")
            for item in report.get("checks", [])
            if item.get("required", True) and not item.get("passed", False)
        ]
        raise RuntimeError(f"真机预检未通过，未实例化任何硬件：{failed or 'ready=false'}")

    make_mouse = mouse_factory or (lambda config: SpaceMouse(SpaceMouseConfig.from_mapping(config)))
    make_hardware = hardware_factory or (
        lambda config: TeleopHardwareWorker(config, enable_motion=True)
    )
    make_session = session_factory or TeleopCollectionSession
    # 上面四道门禁之后，才允许执行以下构造；connect 由 session.run 负责。
    mouse = make_mouse(project)
    hardware = make_hardware(project)
    output = Path(args.output or project["collection"]["storage"]["raw_root"]).expanduser()
    session = make_session(
        project,
        episodes=int(args.episodes),
        spacemouse=mouse,
        hardware=hardware,
        output=output,
    )
    print("[门禁] YAML、固定确认短语与真机预检全部通过，开始本地遥操作数采")
    result = session.run()
    print(json.dumps({"ok": True, "mode": "real", **result}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fake", action="store_true", help="自动录制 FakeHardware Episode")
    parser.add_argument("--fake-frames", type=int, default=12)
    parser.add_argument("--gripper-backend", choices=("hiwonder", "feetech"))
    parser.add_argument("--skip-camera-enumeration", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes 必须大于 0")
    if args.fake_frames < 2:
        parser.error("--fake-frames 必须至少为 2")
    if args.enable_motion and args.fake:
        parser.error("--fake 与 --enable-motion 不能同时使用")
    if args.enable_motion and args.skip_camera_enumeration:
        parser.error("真实运动不允许 --skip-camera-enumeration")
    if args.enable_motion and args.confirm != MOTION_CONFIRMATION:
        parser.error(f"真实运动还必须提供 --confirm {MOTION_CONFIRMATION}")

    try:
        project = load_project_config(args.config_dir, require_real_ready=args.enable_motion)
        project = _override_config(project, args)
        if args.fake:
            episodes = record_fake_episodes(
                project,
                episodes=args.episodes,
                frames=args.fake_frames,
                output=args.output,
            )
            print(
                json.dumps(
                    {
                        "mode": "fake",
                        "motion_started": False,
                        "config_sha256": config_hash(project),
                        "episodes": episodes,
                        "ok": all(not item["validation_errors"] for item in episodes),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0 if all(not item["validation_errors"] for item in episodes) else 2
        if not args.enable_motion:
            print(
                json.dumps(
                    collection_plan(
                        project,
                        args.config_dir.expanduser().resolve(),
                        enumerate_cameras=not args.skip_camera_enumeration,
                        output=args.output,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        return _run_real_collection(project, args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "motion_started": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
