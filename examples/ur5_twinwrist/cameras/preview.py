# ruff: noqa: RUF001, RUF002, RUF003
"""三路 D435 角色与画面检查工具。

默认只打印 YAML 中的 role→serial 计划，不加载 ``pyrealsense2``，也不打开
相机。只有显式传入 ``--capture`` 才会打开三路 RGB 流并保存快照；本工具
从不连接 UR、手腕或夹爪，也不会发送任何运动命令。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.ur5_twinwrist.cameras.realsense import ThreeCameraCapture
from examples.ur5_twinwrist.config_loader import DEFAULT_CONFIG_DIR
from examples.ur5_twinwrist.config_loader import config_hash
from examples.ur5_twinwrist.config_loader import load_project_config

ROLES = ("front", "side", "top")
CaptureFactory = Callable[[Mapping[str, Any]], Any]


def camera_runtime_config(project: Mapping[str, Any]) -> dict[str, Any]:
    """从统一项目配置生成相机运行配置，不修改输入 mapping。"""

    cameras = dict(project["hardware"]["cameras"])
    timing = project["safety"]["timing"]
    cameras["max_camera_skew_ms"] = float(timing["max_camera_skew_ms"])
    cameras["max_frame_age_ms"] = float(timing["camera_timeout_ms"])
    cameras["read_timeout_s"] = float(timing["camera_timeout_ms"]) / 1000.0
    return cameras


def make_mosaic(images: Mapping[str, np.ndarray]) -> np.ndarray:
    """把三张 RGB HWC 图像按 front/side/top 顺序拼接并标注。"""

    if set(images) != set(ROLES):
        raise ValueError("images 必须恰好包含 front/side/top")
    views: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for role in ROLES:
        image = np.asarray(images[role])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{role} 必须是 uint8 HxWx3 RGB 图像")
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError("三路图像 shape 必须一致")
        view = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        cv2.putText(view, role, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        views.append(view)
    return np.concatenate(views, axis=1)


def capture_preview(
    project: Mapping[str, Any],
    output_dir: str | Path,
    *,
    warmup_frames: int = 3,
    capture_factory: CaptureFactory = ThreeCameraCapture.from_mapping,
) -> dict[str, Any]:
    """打开相机、丢弃少量暖机帧并保存最后一组三路快照。"""

    if warmup_frames < 1:
        raise ValueError("warmup_frames 必须大于 0")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    capture = capture_factory(camera_runtime_config(project))
    frames = None
    try:
        capture.connect()
        for _ in range(warmup_frames):
            frames = capture.read()
    finally:
        capture.close()
    if frames is None or set(frames) != set(ROLES):
        raise RuntimeError("相机没有返回完整的 front/side/top 帧")

    images = {role: np.asarray(frames[role].color) for role in ROLES}
    files: dict[str, str] = {}
    for role in ROLES:
        path = destination / f"{role}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(images[role], cv2.COLOR_RGB2BGR)):
            raise OSError(f"无法写入 {path}")
        files[role] = str(path)
    mosaic_path = destination / "front_side_top.png"
    if not cv2.imwrite(str(mosaic_path), make_mosaic(images)):
        raise OSError(f"无法写入 {mosaic_path}")
    return {
        "ok": True,
        "motion_started": False,
        "files": {**files, "mosaic": str(mosaic_path)},
        "frames": {
            role: {
                "serial": str(frames[role].serial),
                "sequence": int(frames[role].sequence),
                "device_timestamp_ms": float(frames[role].device_timestamp_ms),
                "host_timestamp_ns": int(frames[role].host_timestamp_ns),
                "shape": list(images[role].shape),
                "dtype": str(images[role].dtype),
            }
            for role in ROLES
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--capture", action="store_true", help="显式打开三路只读相机流并保存快照")
    parser.add_argument("--output", type=Path, default=Path("data/camera_preview"))
    parser.add_argument("--warmup-frames", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        project = load_project_config(args.config_dir)
        if not args.capture:
            print(
                json.dumps(
                    {
                        "mode": "plan-only（未打开相机）",
                        "motion_started": False,
                        "config_sha256": config_hash(project),
                        "role_to_serial": {
                            item["role"]: item["serial"]
                            for item in project["hardware"]["cameras"]["devices"]
                        },
                        "capture_command": (
                            "uv run python -m examples.ur5_twinwrist.cameras.preview "
                            "--capture --output data/camera_preview"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        report = capture_preview(project, args.output, warmup_frames=args.warmup_frames)
        report["config_sha256"] = config_hash(project)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
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

