# ruff: noqa: E402, RUF001, RUF002, RUF003
"""只读取 SpaceMouse 并打印 YAML 映射结果；绝不连接机器人。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.ur5_twinwrist.config_loader import DEFAULT_CONFIG_DIR
from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouse
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouseConfig


def plan(project: dict) -> dict:
    return {
        "mode": "plan（没有打开 SpaceMouse）",
        "robot_connected": False,
        "motion_possible": False,
        "raw_order": project["teleop"]["axes"]["raw_order"],
        "output_order": project["teleop"]["axes"]["output_order"],
        "mapping": project["teleop"]["axes"]["mapping"],
        "buttons": project["teleop"]["buttons"],
        "hint": "加 --read-input 后只连接 spacenavd；仍不会连接 UR、手腕、夹爪或相机",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--read-input", action="store_true")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--print-hz", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.duration_s <= 0.0 or args.print_hz <= 0.0:
        parser.error("duration-s 和 print-hz 必须大于 0")
    project = load_project_config(args.config_dir)
    if not args.read_input:
        print(json.dumps(plan(project), ensure_ascii=False, indent=2))
        return 0

    mouse = SpaceMouse(SpaceMouseConfig.from_mapping(project))
    mouse.connect()
    deadline = time.monotonic() + args.duration_s
    period = 1.0 / args.print_hz
    try:
        while time.monotonic() < deadline:
            sample = mouse.read()
            pressed = [name for name, value in sample.named_buttons.items() if value]
            print(
                json.dumps(
                    {
                        "monotonic_ns": sample.monotonic_ns,
                        "mapped_motion": [round(float(value), 5) for value in sample.motion],
                        "output_order": project["teleop"]["axes"]["output_order"],
                        "pressed": pressed,
                        "stale": sample.stale,
                        "healthy": sample.healthy,
                        "robot_connected": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(period)
    finally:
        mouse.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
