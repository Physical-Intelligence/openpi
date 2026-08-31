# ruff: noqa: RUF002, RUF003
"""三相机采集层的数据结构；导入本模块不会访问任何硬件。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """一台 RealSense 的固定配置。

    ``role`` 是项目内稳定名称，例如 ``front``、``side``、``top``；设备
    始终由 ``serial`` 绑定，绝不依赖 USB 枚举顺序。
    """

    role: str
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_depth: bool = False

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("camera role must not be empty")
        if not self.serial.strip():
            raise ValueError(f"camera {self.role!r} serial must not be empty")
        if min(self.width, self.height, self.fps) <= 0:
            raise ValueError(f"camera {self.role!r} width, height and fps must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, role: str | None = None) -> CameraConfig:
        """从 YAML/TOML 解析后的 mapping 建立配置。"""

        resolved_role = role if role is not None else str(values.get("role", ""))
        return cls(
            role=resolved_role,
            serial=str(values.get("serial", "")),
            width=int(values.get("width", 640)),
            height=int(values.get("height", 480)),
            fps=int(values.get("fps", 30)),
            enable_depth=bool(values.get("enable_depth", False)),
        )


@dataclass(frozen=True, slots=True)
class CameraRigConfig:
    """三台相机和同步门禁的配置。"""

    cameras: tuple[CameraConfig, ...]
    max_camera_skew_ms: float = 50.0
    max_frame_age_ms: float = 250.0
    read_timeout_s: float = 1.0
    connect_timeout_s: float = 5.0
    provider_wait_timeout_s: float = 0.25
    stop_timeout_s: float = 3.0
    queue_size: int = 8

    def __post_init__(self) -> None:
        if len(self.cameras) != 3:
            raise ValueError(f"exactly three cameras are required, got {len(self.cameras)}")
        roles = [camera.role for camera in self.cameras]
        serials = [camera.serial for camera in self.cameras]
        if len(set(roles)) != len(roles):
            raise ValueError("camera roles must be unique")
        if len(set(serials)) != len(serials):
            raise ValueError("camera serials must be unique")
        for name, value in (
            ("max_camera_skew_ms", self.max_camera_skew_ms),
            ("max_frame_age_ms", self.max_frame_age_ms),
            ("read_timeout_s", self.read_timeout_s),
            ("connect_timeout_s", self.connect_timeout_s),
            ("provider_wait_timeout_s", self.provider_wait_timeout_s),
            ("stop_timeout_s", self.stop_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.queue_size < 2:
            raise ValueError("queue_size must be at least 2")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CameraRigConfig:
        """读取适合 YAML 的结构。

        ``devices`` 同时支持列表和以 role 为键的 mapping：

        .. code-block:: yaml

           devices:
             - {role: front, serial: "123"}
             - {role: side, serial: "456"}
             - {role: top, serial: "789"}
        """

        devices = values.get("devices")
        # width/height/fps 是三相机统一的 YAML 参数。早期实现只把每个
        # devices 项传给 CameraConfig，导致修改顶层值时运行仍悄悄使用
        # dataclass 默认值。先合并公共参数，再允许单设备显式覆盖。
        shared = {
            key: values[key]
            for key in ("width", "height", "fps", "enable_depth")
            if key in values
        }
        cameras: tuple[CameraConfig, ...]
        if isinstance(devices, Mapping):
            cameras = tuple(
                CameraConfig.from_mapping(
                    {**shared, **_require_mapping(config, f"devices.{role}")},
                    role=str(role),
                )
                for role, config in devices.items()
            )
        elif isinstance(devices, Sequence) and not isinstance(devices, str | bytes):
            cameras = tuple(
                CameraConfig.from_mapping(
                    {**shared, **_require_mapping(config, f"devices[{index}]")}
                )
                for index, config in enumerate(devices)
            )
        else:
            raise ValueError("cameras config requires a devices list or mapping")

        return cls(
            cameras=cameras,
            max_camera_skew_ms=float(values.get("max_camera_skew_ms", 50.0)),
            max_frame_age_ms=float(values.get("max_frame_age_ms", 250.0)),
            read_timeout_s=float(values.get("read_timeout_s", 1.0)),
            connect_timeout_s=float(values.get("connect_timeout_s", 5.0)),
            provider_wait_timeout_s=float(values.get("provider_wait_timeout_s", 0.25)),
            stop_timeout_s=float(values.get("stop_timeout_s", 3.0)),
            queue_size=int(values.get("queue_size", 8)),
        )


@dataclass(frozen=True, slots=True)
class ProviderFrame:
    """底层 provider 产生的单帧；host 时间由采集线程在收到帧时补上。"""

    sequence: int
    device_timestamp_ms: float
    color: np.ndarray
    depth: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """带设备身份和双时间轴的不可变相机帧。"""

    role: str
    serial: str
    sequence: int
    device_timestamp_ms: float
    host_timestamp_ns: int
    color: np.ndarray
    depth: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.role or not self.serial or self.sequence < 0:
            raise ValueError("invalid camera frame identity")
        if not math.isfinite(self.device_timestamp_ms) or self.device_timestamp_ms < 0.0:
            raise ValueError("device timestamp must be finite and non-negative")
        if self.host_timestamp_ns < 0:
            raise ValueError("host timestamp must be non-negative")
        if self.color.dtype != np.uint8 or self.color.ndim != 3 or self.color.shape[-1] != 3:
            raise ValueError("color frame must be HWC uint8 RGB")

    @property
    def device_timestamp_ns(self) -> int:
        """RealSense 毫秒设备时间戳的纳秒整数表示。"""

        return round(self.device_timestamp_ms * 1_000_000.0)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value
