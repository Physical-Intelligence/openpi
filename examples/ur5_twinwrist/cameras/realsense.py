# ruff: noqa: RUF002, RUF003
"""项目内三台 Intel RealSense D435 的线程化采集实现。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
import importlib
import threading
import time
from typing import Any, Protocol

import numpy as np

from examples.ur5_twinwrist.cameras.models import CameraConfig
from examples.ur5_twinwrist.cameras.models import CameraFrame
from examples.ur5_twinwrist.cameras.models import CameraRigConfig
from examples.ur5_twinwrist.cameras.models import ProviderFrame
from examples.ur5_twinwrist.cameras.synchronizer import FrameSynchronizer


class FrameProvider(Protocol):
    """单相机线程拥有的最小 provider 接口。"""

    def connect(self) -> None: ...

    def read(self, timeout_s: float) -> ProviderFrame: ...

    def close(self) -> None: ...


ProviderFactory = Callable[[CameraConfig], FrameProvider]


class RealSenseFrameProvider:
    """一台 D435 的 pyrealsense2 provider。

    构造函数和模块导入均不加载 ``pyrealsense2``、不枚举 USB、也不打开
    相机。只有所属采集线程调用 ``connect`` 时才进行这些操作。
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._pipeline = None

    def connect(self) -> None:
        if self._pipeline is not None:
            raise RuntimeError(f"camera {self._config.role!r} is already connected")
        try:
            rs = importlib.import_module("pyrealsense2")
        except ImportError as exc:
            raise RuntimeError("RealSense capture requires pyrealsense2 in the hardware environment") from exc

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self._config.serial)
        rs_config.enable_stream(
            rs.stream.color,
            self._config.width,
            self._config.height,
            rs.format.rgb8,
            self._config.fps,
        )
        if self._config.enable_depth:
            rs_config.enable_stream(
                rs.stream.depth,
                self._config.width,
                self._config.height,
                rs.format.z16,
                self._config.fps,
            )
        try:
            pipeline.start(rs_config)
        except Exception:
            with suppress(RuntimeError):
                pipeline.stop()
            raise
        self._pipeline = pipeline

    def read(self, timeout_s: float) -> ProviderFrame:
        if self._pipeline is None:
            raise RuntimeError("RealSense provider is not connected")
        try:
            frames = self._pipeline.wait_for_frames(max(1, round(timeout_s * 1000.0)))
        except RuntimeError as exc:
            # librealsense 用 RuntimeError 表示 wait_for_frames 超时；这不是永久硬件错误。
            if "frame" in str(exc).lower() and ("arrive" in str(exc).lower() or "timeout" in str(exc).lower()):
                raise TimeoutError(str(exc)) from exc
            raise

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise TimeoutError("RealSense frameset did not contain a color frame")
        depth_frame = frames.get_depth_frame() if self._config.enable_depth else None
        color = np.ascontiguousarray(np.asanyarray(color_frame.get_data())).copy()
        depth = None if depth_frame is None else np.ascontiguousarray(np.asanyarray(depth_frame.get_data())).copy()
        return ProviderFrame(
            sequence=int(color_frame.get_frame_number()),
            device_timestamp_ms=float(color_frame.get_timestamp()),
            color=color,
            depth=depth,
        )

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            with suppress(RuntimeError):
                pipeline.stop()


class ThreeCameraCapture:
    """三台相机的统一 ``connect/read/stop/close`` 接口。

    每台相机拥有一个独立线程和 provider。主线程只读取同步后的缓存，既
    不调用 librealsense，也不会按 USB 枚举顺序猜测相机角色。
    """

    def __init__(
        self,
        config: CameraRigConfig,
        *,
        provider_factory: ProviderFactory = RealSenseFrameProvider,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._provider_factory = provider_factory
        self._clock_ns = clock_ns
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._ready = {camera.role: threading.Event() for camera in config.cameras}
        self._synchronizer: FrameSynchronizer | None = None
        self._running = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        provider_factory: ProviderFactory = RealSenseFrameProvider,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> ThreeCameraCapture:
        """从 YAML/TOML 解析结果构建采集器，但仍不打开设备。"""

        return cls(CameraRigConfig.from_mapping(values), provider_factory=provider_factory, clock_ns=clock_ns)

    def __enter__(self) -> ThreeCameraCapture:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def role_to_serial(self) -> dict[str, str]:
        """返回配置中的稳定 role→serial 绑定。"""

        return {camera.role: camera.serial for camera in self.config.cameras}

    def connect(self) -> None:
        """启动三个采集线程并等待每台相机成功打开。"""

        with self._lifecycle_lock:
            if self._running:
                raise RuntimeError("camera capture is already connected")
            self._stop_event.clear()
            self._ready = {camera.role: threading.Event() for camera in self.config.cameras}
            self._synchronizer = FrameSynchronizer(
                tuple(camera.role for camera in self.config.cameras),
                queue_size=self.config.queue_size,
                max_skew_ms=self.config.max_camera_skew_ms,
                max_frame_age_ms=self.config.max_frame_age_ms,
                clock_ns=self._clock_ns,
            )
            self._threads = {
                camera.role: threading.Thread(
                    target=self._capture_loop,
                    args=(camera, self._synchronizer),
                    name=f"ur5-twinwrist-camera-{camera.role}",
                    daemon=True,
                )
                for camera in self.config.cameras
            }
            self._running = True
            for thread in self._threads.values():
                thread.start()

        deadline_s = time.monotonic() + self.config.connect_timeout_s
        try:
            while not all(event.is_set() for event in self._ready.values()):
                synchronizer = self._require_synchronizer()
                synchronizer.raise_if_failed()  # 后台 connect 错误应立刻返回，而不是等满超时。
                if time.monotonic() >= deadline_s:
                    waiting = [role for role, event in self._ready.items() if not event.is_set()]
                    raise TimeoutError(f"timed out connecting cameras: {waiting}")
                time.sleep(0.005)
            self._require_synchronizer().raise_if_failed()
        except BaseException:
            # 清理错误不能覆盖最先发生的相机 connect/timeout 错误。
            with suppress(Exception):
                self.stop()
            raise

    def read(self, timeout_s: float | None = None) -> dict[str, CameraFrame]:
        """返回一组三路同步帧；skew、帧龄或等待超限均不会返回数据。"""

        with self._lifecycle_lock:
            if not self._running:
                raise RuntimeError("camera capture is not connected")
        return self._require_synchronizer().read(self.config.read_timeout_s if timeout_s is None else timeout_s)

    def stop(self) -> None:
        """通知线程停止，并等待 provider 在所属线程中关闭。"""

        with self._lifecycle_lock:
            if not self._running and not self._threads:
                return
            self._stop_event.set()
            synchronizer = self._synchronizer
            threads = dict(self._threads)
        if synchronizer is not None:
            synchronizer.close()
        deadline = time.monotonic() + self.config.stop_timeout_s
        for thread in threads.values():
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [role for role, thread in threads.items() if thread.is_alive()]
        with self._lifecycle_lock:
            if not alive:
                self._threads.clear()
                self._running = False
        if alive:
            raise TimeoutError(f"camera threads did not stop: {alive}")

    def close(self) -> None:
        """``stop`` 的幂等别名，便于统一硬件生命周期管理。"""

        self.stop()

    def _capture_loop(self, camera: CameraConfig, synchronizer: FrameSynchronizer) -> None:
        provider: FrameProvider | None = None
        last_sequence: int | None = None
        try:
            provider = self._provider_factory(camera)
            provider.connect()
            self._ready[camera.role].set()
            while not self._stop_event.is_set():
                try:
                    raw = provider.read(self.config.provider_wait_timeout_s)
                except TimeoutError:
                    continue
                if last_sequence is not None and raw.sequence <= last_sequence:
                    raise RuntimeError(
                        f"non-increasing sequence for {camera.role!r}: {raw.sequence} <= {last_sequence}"
                    )
                last_sequence = raw.sequence
                self._validate_raw_frame(camera, raw)
                synchronizer.add(
                    CameraFrame(
                        role=camera.role,
                        serial=camera.serial,
                        sequence=raw.sequence,
                        device_timestamp_ms=raw.device_timestamp_ms,
                        host_timestamp_ns=self._clock_ns(),
                        color=np.ascontiguousarray(raw.color).copy(),
                        depth=None if raw.depth is None else np.ascontiguousarray(raw.depth).copy(),
                    )
                )
        except Exception as exc:  # 硬件异常必须传给 read，不能静默吞掉。
            if not self._stop_event.is_set():
                synchronizer.report_failure(camera.role, exc)
        finally:
            self._ready[camera.role].set()
            if provider is not None:
                with suppress(Exception):
                    provider.close()

    @staticmethod
    def _validate_raw_frame(camera: CameraConfig, frame: ProviderFrame) -> None:
        expected_color = (camera.height, camera.width, 3)
        if frame.sequence < 0 or not np.isfinite(frame.device_timestamp_ms):
            raise ValueError(f"camera {camera.role!r} returned invalid identity/timestamp")
        if frame.color.dtype != np.uint8 or frame.color.shape != expected_color:
            raise ValueError(
                f"camera {camera.role!r} color must be uint8 {expected_color}, got {frame.color.dtype} {frame.color.shape}"
            )
        if camera.enable_depth and (frame.depth is None or frame.depth.shape != (camera.height, camera.width)):
            raise ValueError(f"camera {camera.role!r} depth frame is missing or has the wrong shape")

    def _require_synchronizer(self) -> FrameSynchronizer:
        synchronizer = self._synchronizer
        if synchronizer is None:
            raise RuntimeError("camera capture is not connected")
        return synchronizer
