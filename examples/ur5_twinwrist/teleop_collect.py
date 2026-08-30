"""Safe launcher for legacy teleoperation with the new raw-HDF5 sink."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CONFIRMATION = "I_UNDERSTAND_REAL_ROBOT_MOTION"


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    # Do not resolve this symlink: resolving a venv's ``bin/python`` selects the
    # base interpreter and silently drops the venv site-packages.
    legacy_python = Path(args.legacy_python).expanduser().absolute()
    if not legacy_python.is_file():
        raise FileNotFoundError(f"legacy Python not found: {legacy_python}")
    command = [
        str(legacy_python),
        "-m",
        "slai_mi.apps.collect_real",
        "--hardware-config",
        str(Path(args.hardware_config).expanduser().resolve()),
        "--dataset-config",
        str(Path(args.dataset_config).expanduser().resolve()),
        "--strategy",
        args.strategy,
        "--task",
        str(Path(args.task).expanduser().resolve()),
        "--adapter-plugin",
        "examples.ur5_twinwrist.legacy_collection_adapter:make_dependencies",
    ]
    if args.continuous:
        command.append("--continuous")
    else:
        command.extend(("--episodes", str(args.episodes)))
    if args.home_preset:
        command.extend(("--home-preset", args.home_preset))
    if args.no_dashboard:
        command.append("--no-dashboard")
    if args.no_open_dashboard:
        command.append("--no-open-dashboard")
    if args.enable_motion:
        if args.confirm != CONFIRMATION:
            raise PermissionError(
                f"motion denied: pass --enable-motion --confirm {CONFIRMATION} after the physical safety check"
            )
        command.extend(("--execute-real", "--confirm", CONFIRMATION))
    openpi_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["UR5_TWINWRIST_RAW_ROOT"] = str(Path(args.output).expanduser().resolve())
    environment["UR5_TWINWRIST_OPENPI_ROOT"] = str(openpi_root)
    environment["UR5_TWINWRIST_LEGACY_ROOT"] = str(legacy_root)
    environment["UR5_TWINWRIST_RECORD_HZ"] = str(args.record_hz)
    if args.gripper_backend:
        environment["UR5_TWINWRIST_GRIPPER_BACKEND"] = args.gripper_backend
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(openpi_root), str(legacy_root / "src"), environment.get("PYTHONPATH", ""))
    )
    return command, environment


def main(argv: list[str] | None = None) -> int:
    default_legacy = Path("/home/user/shiyi/slai-manipulation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", default=str(default_legacy))
    parser.add_argument("--legacy-python", default=str(default_legacy / ".venv-lerobot-v3/bin/python"))
    parser.add_argument("--hardware-config", default=str(default_legacy / "configs/hardware.yaml"))
    parser.add_argument("--dataset-config", default=str(default_legacy / "configs/dataset.yaml"))
    parser.add_argument("--task", default=str(default_legacy / "configs/tasks/task2_continuous.yaml"))
    parser.add_argument("--strategy", default="ur5e_wrist_gripper_9dof_collection")
    parser.add_argument("--output", default="data/raw/ur5_twinwrist")
    parser.add_argument(
        "--record-hz",
        type=float,
        default=10.0,
        help="raw HDF5 sample rate; cameras remain at the locked legacy capture rate",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--home-preset", choices=("parent", "last-point"))
    parser.add_argument("--gripper-backend", choices=("hiwonder", "feetech"))
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-open-dashboard", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not 0.0 < args.record_hz <= 30.0:
        parser.error("--record-hz must be in (0, 30]")
    try:
        command, environment = build_command(args)
    except (FileNotFoundError, PermissionError) as exc:
        parser.error(str(exc))
    if args.print_command:
        print(" ".join(command))
        return 0
    try:
        from examples.ur5_twinwrist.teleop_preflight import inspect_station
    except ModuleNotFoundError:
        from teleop_preflight import inspect_station

    preflight = inspect_station(args.legacy_root, args.legacy_python, args.gripper_backend)
    if args.enable_motion:
        failures = [item for item in preflight["checks"] if not item["passed"]]
        if failures:
            print(
                json.dumps(
                    {
                        "error": "real motion preflight failed",
                        "failed_checks": failures,
                        "motion_started": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 2
        return subprocess.run(command, env=environment, cwd=args.legacy_root, check=False).returncode

    # Legacy dry-run intentionally does not load adapter plugins.  Capture its
    # validated hardware/strategy plan, then report the dataset sink that this
    # launcher will actually install once the explicit motion gate is passed.
    result = subprocess.run(
        command,
        env=environment,
        cwd=args.legacy_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    try:
        plan = json.loads(result.stdout)
    except json.JSONDecodeError:
        sys.stdout.write(result.stdout)
        return result.returncode
    plan.update(
        {
            "mode": "dry-run (no hardware opened, no motion)",
            "dataset_format": "raw_hdf5_atomic_v1",
            "dataset_root": environment["UR5_TWINWRIST_RAW_ROOT"],
            "capture_fps": preflight["capture_parameters"]["fps"],
            "record_hz": args.record_hz,
            "adapter_plugin": "examples.ur5_twinwrist.legacy_collection_adapter:make_dependencies",
            "gripper_driver": preflight["selected_gripper"],
            "safety_gates": [
                "single-owner cached gripper serial worker",
                "coordinated HOME always zeros SpaceMouse cap",
                "completed 125 Hz command-cycle receipt",
                "speedJ forbidden inside accepted episodes",
                "Fit seals before HOME and emergency saves are rejected",
                "NaN/Inf, stale source, invalid source, camera skew/duplicate fail closed",
                "possible UR5 floor-clamp mismatch rejects the episode",
            ],
            "preflight_ready": preflight["ready"],
            "preflight_failed_checks": [item for item in preflight["checks"] if not item["passed"]],
        }
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
