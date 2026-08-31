from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from examples.ur5_twinwrist.cameras.preview import capture_preview
from examples.ur5_twinwrist.cameras.preview import main
from examples.ur5_twinwrist.cameras.preview import make_mosaic
from examples.ur5_twinwrist.config_loader import load_project_config


class _FakeCapture:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def read(self) -> dict[str, Any]:
        assert self.connected
        return {
            role: SimpleNamespace(
                role=role,
                serial=f"{role}-serial",
                sequence=index + 1,
                device_timestamp_ms=100.0 + index,
                host_timestamp_ns=1_000_000 + index,
                color=np.full((8, 10, 3), 20 + index, dtype=np.uint8),
            )
            for index, role in enumerate(("front", "side", "top"))
        }

    def close(self) -> None:
        self.closed = True


def test_default_cli_only_prints_plan(capsys: Any) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "plan-only" in output
    assert "254522076307" in output


def test_make_mosaic_has_expected_shape() -> None:
    images = {role: np.zeros((8, 10, 3), np.uint8) for role in ("front", "side", "top")}
    assert make_mosaic(images).shape == (8, 30, 3)


def test_capture_preview_writes_three_views_and_mosaic(tmp_path: Path) -> None:
    instances: list[_FakeCapture] = []

    def factory(config: dict[str, Any]) -> _FakeCapture:
        instance = _FakeCapture(config)
        instances.append(instance)
        return instance

    report = capture_preview(load_project_config(), tmp_path, warmup_frames=2, capture_factory=factory)
    assert report["ok"]
    assert report["motion_started"] is False
    assert instances[0].closed
    assert set(report["files"]) == {"front", "side", "top", "mosaic"}
    assert cv2.imread(report["files"]["mosaic"]).shape == (8, 30, 3)
