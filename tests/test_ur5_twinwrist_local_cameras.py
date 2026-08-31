# ruff: noqa: RUF003
from __future__ import annotations

from collections import deque
import importlib
import threading
import time

import numpy as np
import pytest

from examples.ur5_twinwrist.cameras import CameraConfig
from examples.ur5_twinwrist.cameras import CameraFrame
from examples.ur5_twinwrist.cameras import CameraRigConfig
from examples.ur5_twinwrist.cameras import FrameSynchronizer
from examples.ur5_twinwrist.cameras import ProviderFrame
from examples.ur5_twinwrist.cameras import RealSenseFrameProvider
from examples.ur5_twinwrist.cameras import ThreeCameraCapture


def _rgb(value: int = 1, *, width: int = 4, height: int = 3) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _camera_frame(role: str, sequence: int, host_timestamp_ns: int) -> CameraFrame:
    return CameraFrame(
        role=role,
        serial=f"serial-{role}",
        sequence=sequence,
        device_timestamp_ms=float(sequence),
        host_timestamp_ns=host_timestamp_ns,
        color=_rgb(sequence),
    )


class _FakeProvider:
    def __init__(
        self,
        config: CameraConfig,
        frames: list[ProviderFrame],
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.config = config
        self.frames = deque(frames)
        self.failure = failure
        self.connected_on: str | None = None
        self.read_on: set[str] = set()
        self.closed_on: str | None = None
        self.closed = threading.Event()

    def connect(self) -> None:
        self.connected_on = threading.current_thread().name
        if self.failure is not None:
            raise self.failure

    def read(self, _timeout_s: float) -> ProviderFrame:
        self.read_on.add(threading.current_thread().name)
        if self.frames:
            return self.frames.popleft()
        time.sleep(0.001)
        raise TimeoutError("fake provider has no frame yet")

    def close(self) -> None:
        self.closed_on = threading.current_thread().name
        self.closed.set()


def _rig_config(**overrides: object) -> CameraRigConfig:
    values: dict[str, object] = {
        "cameras": (
            CameraConfig("front", "101", 4, 3, 30),
            CameraConfig("side", "202", 4, 3, 30),
            CameraConfig("top", "303", 4, 3, 30),
        ),
        "max_camera_skew_ms": 50.0,
        "max_frame_age_ms": 500.0,
        "read_timeout_s": 0.2,
        "connect_timeout_s": 0.2,
        "provider_wait_timeout_s": 0.01,
        "stop_timeout_s": 0.2,
    }
    values.update(overrides)
    return CameraRigConfig(**values)


def test_config_from_yaml_mapping_binds_three_unique_roles_and_serials() -> None:
    config = CameraRigConfig.from_mapping(
        {
            "devices": {
                "front": {"serial": "101", "width": 4, "height": 3},
                "side": {"serial": "202", "width": 4, "height": 3},
                "top": {"serial": "303", "width": 4, "height": 3},
            },
            "max_camera_skew_ms": 12.5,
        }
    )
    assert {camera.role: camera.serial for camera in config.cameras} == {
        "front": "101",
        "side": "202",
        "top": "303",
    }
    assert config.max_camera_skew_ms == 12.5

    with pytest.raises(ValueError, match="serials must be unique"):
        CameraRigConfig(
            cameras=(
                CameraConfig("front", "101"),
                CameraConfig("side", "101"),
                CameraConfig("top", "303"),
            )
        )


def test_top_level_image_settings_are_applied_to_every_camera() -> None:
    config = CameraRigConfig.from_mapping(
        {
            "width": 320,
            "height": 240,
            "fps": 15,
            "enable_depth": True,
            "devices": [
                {"role": "front", "serial": "a"},
                {"role": "side", "serial": "b"},
                {"role": "top", "serial": "c", "fps": 30},
            ],
        }
    )
    assert [(item.width, item.height, item.fps, item.enable_depth) for item in config.cameras] == [
        (320, 240, 15, True),
        (320, 240, 15, True),
        (320, 240, 30, True),
    ]


def test_import_and_construction_do_not_import_pyrealsense_or_open_device(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    real_import_module = importlib.import_module

    def tracked_import(name: str, package: str | None = None):
        imported.append(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked_import)
    provider = RealSenseFrameProvider(CameraConfig("front", "101"))
    capture = ThreeCameraCapture(_rig_config())
    assert provider is not None
    assert capture.role_to_serial["front"] == "101"
    assert "pyrealsense2" not in imported


def test_synchronizer_selects_latest_feasible_triplet_and_consumes_it_once() -> None:
    now = [1_100_000_000]
    sync = FrameSynchronizer(
        ("front", "side", "top"),
        max_skew_ms=20,
        max_frame_age_ms=500,
        clock_ns=lambda: now[0],
    )
    sync.add(_camera_frame("front", 1, 1_000_000_000))
    sync.add(_camera_frame("side", 1, 1_008_000_000))
    sync.add(_camera_frame("top", 1, 1_035_000_000))
    sync.add(_camera_frame("front", 2, 1_038_000_000))
    sync.add(_camera_frame("side", 2, 1_033_000_000))

    frames = sync.read(0.01)
    assert {role: frame.sequence for role, frame in frames.items()} == {"front": 2, "side": 2, "top": 1}
    assert (
        max(frame.host_timestamp_ns for frame in frames.values())
        - min(frame.host_timestamp_ns for frame in frames.values())
        <= 20_000_000
    )
    with pytest.raises(TimeoutError):
        sync.read(0.001)


def test_synchronizer_rejects_skew_and_stale_frames_with_diagnostics() -> None:
    now = [2_000_000_000]
    sync = FrameSynchronizer(
        ("front", "side", "top"),
        max_skew_ms=10,
        max_frame_age_ms=50,
        clock_ns=lambda: now[0],
    )
    sync.add(_camera_frame("front", 1, 1_900_000_000))
    sync.add(_camera_frame("side", 1, 1_905_000_000))
    sync.add(_camera_frame("top", 1, 1_930_000_000))

    # 人工推进 clock，避免使用真实 sleep；wait 由另一线程推进单调时钟。
    timer = threading.Timer(0.005, lambda: now.__setitem__(0, 2_100_000_000))
    timer.start()
    with pytest.raises(TimeoutError) as failure:
        sync.read(0.05)
    timer.join()
    message = str(failure.value)
    assert "latest_skew_ms=30.0" in message
    assert all(role in message for role in ("front", "side", "top"))


def test_three_camera_capture_uses_one_owner_thread_per_serial_and_returns_frames() -> None:
    providers: dict[str, _FakeProvider] = {}

    def factory(config: CameraConfig) -> _FakeProvider:
        provider = _FakeProvider(
            config,
            [ProviderFrame(1, 123.0, _rgb(ord(config.role[0])))],
        )
        providers[config.role] = provider
        return provider

    capture = ThreeCameraCapture(_rig_config(), provider_factory=factory)
    capture.connect()
    frames = capture.read()
    capture.close()

    assert set(frames) == {"front", "side", "top"}
    assert {role: frame.serial for role, frame in frames.items()} == {
        "front": "101",
        "side": "202",
        "top": "303",
    }
    assert all(frame.host_timestamp_ns > 0 for frame in frames.values())
    assert all(provider.connected_on == f"ur5-twinwrist-camera-{role}" for role, provider in providers.items())
    assert all(provider.read_on == {provider.connected_on} for provider in providers.values())
    assert all(provider.closed_on == provider.connected_on for provider in providers.values())
    assert all(provider.closed.is_set() for provider in providers.values())


def test_connect_and_read_fail_closed_on_provider_errors() -> None:
    def connect_failure_factory(config: CameraConfig) -> _FakeProvider:
        error = OSError("USB disconnected") if config.role == "side" else None
        return _FakeProvider(config, [], failure=error)

    capture = ThreeCameraCapture(_rig_config(), provider_factory=connect_failure_factory)
    with pytest.raises(RuntimeError, match="side.*USB disconnected"):
        capture.connect()

    providers: dict[str, _FakeProvider] = {}

    def duplicate_factory(config: CameraConfig) -> _FakeProvider:
        provider = _FakeProvider(
            config,
            [ProviderFrame(1, 1.0, _rgb())],
        )
        providers[config.role] = provider
        return provider

    capture = ThreeCameraCapture(_rig_config(), provider_factory=duplicate_factory)
    capture.connect()
    capture.read()
    for provider in providers.values():
        provider.frames.append(ProviderFrame(1, 2.0, _rgb()))
    with pytest.raises(RuntimeError, match="non-increasing sequence"):
        capture.read()
    capture.close()
    assert all(provider.closed.wait(0.1) for provider in providers.values())
