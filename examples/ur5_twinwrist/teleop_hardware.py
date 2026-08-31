# ruff: noqa: RUF001, RUF002, RUF003
"""项目内真机遥操作硬件 worker。

本模块把 UR RTDE、ESP32 主腕、OpenRB 从腕和夹爪的全部 I/O 固定在同一
个 actuator owner 线程中。SpaceMouse、UI 和 HDF5 录制线程只能调用
``update_command`` 更新命令缓存，或调用 ``get_observation`` 读取不可变
状态缓存；它们不会直接访问 RTDE 或串口。

三台 RealSense 仍由 :class:`ThreeCameraCapture` 自己的三个 provider 线程
持有。录制线程调用 ``get_observation`` 时只从同步器取一组三路完整帧。

构造和导入均无硬件副作用。``connect`` 会让双轴从腕执行机械 HOME，因而
本类在 ``enable_motion=False`` 时明确拒绝连接。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import math
import threading
import time
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from examples.ur5_twinwrist.cameras.realsense import ThreeCameraCapture
from examples.ur5_twinwrist.controller.gripper import GripperInterface
from examples.ur5_twinwrist.controller.gripper import GripperState
from examples.ur5_twinwrist.controller.gripper import create_gripper
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouseSample
from examples.ur5_twinwrist.controller.ur5 import UR5Config
from examples.ur5_twinwrist.controller.ur5 import UR5Controller
from examples.ur5_twinwrist.controller.ur5 import UR5State
from examples.ur5_twinwrist.controller.wrist import WristMasterSlaveController
from examples.ur5_twinwrist.controller.wrist import WristMasterSlaveState


class TeleopHardwareError(RuntimeError):
    """硬件 worker 已经 fail-closed 或生命周期调用非法。"""


@dataclass(frozen=True, slots=True)
class TeleopWorkerConfig:
    """只包含 worker 调度与 HOME 所需参数。"""

    ur_hz: float
    wrist_hz: float
    gripper_hz: float
    max_command_age_s: float
    max_state_age_s: float
    camera_timeout_s: float
    task_home_rad: tuple[float, ...]
    home_speed_rad_s: float
    home_proportional_gain: float
    home_tolerance_rad: float
    home_stable_s: float
    joint6_jog_speed_rad_s: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> TeleopWorkerConfig:
        hardware = _mapping(config.get("hardware"), "hardware")
        safety = _mapping(config.get("safety"), "safety")
        poses = _mapping(config.get("poses"), "poses")
        ur_hw = _mapping(hardware.get("ur5"), "hardware.ur5")
        wrist_hw = _mapping(hardware.get("wrist"), "hardware.wrist")
        gripper_hw = _mapping(hardware.get("gripper"), "hardware.gripper")
        timing = _mapping(safety.get("timing"), "safety.timing")
        ur_safety = _mapping(safety.get("ur5"), "safety.ur5")
        ur_poses = _mapping(poses.get("ur5"), "poses.ur5")
        return cls(
            ur_hz=_positive(ur_hw.get("control_hz", 125.0), "hardware.ur5.control_hz"),
            wrist_hz=_positive(wrist_hw.get("command_hz", 50.0), "hardware.wrist.command_hz"),
            gripper_hz=_positive(gripper_hw.get("command_hz", 30.0), "hardware.gripper.command_hz"),
            max_command_age_s=(
                _positive(timing.get("max_command_age_ms", 250.0), "safety.timing.max_command_age_ms")
                / 1000.0
            ),
            max_state_age_s=(
                _positive(timing.get("max_state_age_ms", 100.0), "safety.timing.max_state_age_ms")
                / 1000.0
            ),
            camera_timeout_s=(
                _positive(timing.get("camera_timeout_ms", 500.0), "safety.timing.camera_timeout_ms")
                / 1000.0
            ),
            task_home_rad=_finite_tuple(ur_poses.get("task_home_rad"), 6, "poses.ur5.task_home_rad"),
            home_speed_rad_s=_positive(ur_poses.get("home_speed_rad_s", 0.5), "poses.ur5.home_speed_rad_s"),
            home_proportional_gain=_positive(
                ur_poses.get("home_proportional_gain", 1.5),
                "poses.ur5.home_proportional_gain",
            ),
            home_tolerance_rad=_positive(
                ur_poses.get("home_tolerance_rad", 0.01),
                "poses.ur5.home_tolerance_rad",
            ),
            home_stable_s=_positive(ur_poses.get("home_stable_s", 0.3), "poses.ur5.home_stable_s"),
            joint6_jog_speed_rad_s=_positive(
                ur_safety.get("joint6_jog_speed_rad_s", 0.2),
                "safety.ur5.joint6_jog_speed_rad_s",
            ),
        )


@dataclass(frozen=True, slots=True)
class TeleopCommand:
    """非 I/O 线程写入的最新遥操作命令。"""

    ur_speed_l: tuple[float, ...]
    wrist_velocity_deg_s: tuple[float, float] | None
    gripper_target: float
    issued_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    """三个执行器各自最近一次实际下发目标组成的 9 维 action。

    前六维是 ``UR5Controller.send`` 实际返回的 base-frame ``speedL``，七、
    八维是 ``WristMasterSlaveController.step`` 实际下发的 J1/J2 YAML 零位
    相对 raw target，第九维是夹爪实际下发的归一化绝对目标。
    """

    action: tuple[float, ...]
    host_monotonic_ns: int
    sequence: int
    ur_sent_monotonic_ns: int
    wrist_sent_monotonic_ns: int
    gripper_sent_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class MaintenanceStatus:
    """供遥操作线程等待 HOME/J6/resume 完成的纯缓存状态。"""

    home_requested: bool
    home_in_progress: bool
    wrist_home_pending: bool
    joint6_direction: int
    resume_master_requested: bool

    @property
    def active(self) -> bool:
        return bool(
            self.home_requested
            or self.home_in_progress
            or self.wrist_home_pending
            or self.joint6_direction != 0
            or self.resume_master_requested
        )


class _UR(Protocol):
    def connect(self) -> None: ...
    def read(self) -> UR5State: ...
    def send(self, twist: Sequence[float], *, state_timestamp_ns: int | None = None) -> np.ndarray: ...
    def jog_joint6(
        self,
        direction: int,
        *,
        speed_rad_s: float,
        state_timestamp_ns: int | None = None,
    ) -> Any: ...
    def task_home_step(
        self,
        target_joints_rad: Sequence[float],
        *,
        max_speed_rad_s: float,
        tolerance_rad: float,
        proportional_gain: float = 1.5,
        state_timestamp_ns: int | None = None,
    ) -> Any: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def healthy(self) -> bool: ...


class _Wrist(Protocol):
    def connect(self, *, enable_motion: bool = False) -> WristMasterSlaveState: ...
    def resume_master(self) -> Any: ...
    def step(self) -> WristMasterSlaveState: ...
    def set_spacemouse_velocity(self, j1_deg_s: float, j2_deg_s: float) -> None: ...
    def clear_spacemouse_velocity(self) -> None: ...
    def home(self, *, enable_motion: bool = False) -> WristMasterSlaveState: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def healthy(self) -> bool: ...


URFactory = Callable[[Mapping[str, Any], bool], _UR]
WristFactory = Callable[[Mapping[str, Any]], _Wrist]
GripperFactory = Callable[[Mapping[str, Any]], GripperInterface]
CameraFactory = Callable[[Mapping[str, Any]], Any]


class TeleopHardwareWorker:
    """单 actuator-owner worker；公共方法除 ``get_observation`` 外不做 I/O。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        enable_motion: bool = False,
        ur_factory: URFactory | None = None,
        wrist_factory: WristFactory | None = None,
        gripper_factory: GripperFactory | None = None,
        camera_factory: CameraFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._config = config
        self.config = TeleopWorkerConfig.from_mapping(config)
        self.enable_motion = bool(enable_motion)
        self._ur_factory = ur_factory or _default_ur_factory
        self._wrist_factory = wrist_factory or _default_wrist_factory
        self._gripper_factory = gripper_factory or _default_gripper_factory
        self._camera_factory = camera_factory or _default_camera_factory
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        hardware = _mapping(config.get("hardware"), "hardware")
        gripper_config = _mapping(hardware.get("gripper"), "hardware.gripper")
        self._gripper_state_source = str(gripper_config.get("state_source", "measured"))

        self._lock = threading.Lock()
        now_ns = int(self._monotonic_ns())
        self._command = TeleopCommand((0.0,) * 6, None, 0.0, now_ns)
        self._episode_active = False
        self._home_requested = False
        self._home_in_progress = False
        self._wrist_home_pending = False
        self._home_reached_since_ns: int | None = None
        self._joint6_direction = 0
        self._resume_master_requested = False
        self._fatal_command_error: BaseException | None = None
        self._failure: BaseException | None = None
        self._latest_ur: UR5State | None = None
        self._latest_wrist: WristMasterSlaveState | None = None
        self._latest_gripper: GripperState | None = None
        self._latest_receipt: ActionReceipt | None = None
        self._receipt_sequence = 0

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._finished_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cameras: Any | None = None

    @property
    def connected(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive() and self._ready_event.is_set())

    @property
    def episode_active(self) -> bool:
        with self._lock:
            return self._episode_active

    def connect(self, *, timeout_s: float = 60.0) -> None:
        """启动 owner 线程并等待设备就绪；没有运动许可时直接拒绝。

        主从腕的首次连接包含机械 HOME，因此这里不能提供看似“只读”的
        半连接模式。只读硬件预检应使用 ``teleop_preflight.py``。
        """

        if not self.enable_motion:
            raise PermissionError("主从腕 connect 会运动，必须显式 enable_motion=True")
        _require_joint_limits_for_motion(self._config)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s 必须是正有限数")
        if self._thread is not None:
            raise RuntimeError("硬件 worker 已启动，实例不能重复 connect")
        self._stop_event.clear()
        self._ready_event.clear()
        self._finished_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ur5-twinwrist-actuator-owner",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout_s):
            self._stop_event.set()
            self._thread.join(min(timeout_s, 5.0))
            raise TimeoutError("等待遥操作硬件 worker 连接超时")
        self.raise_if_failed()

    def update_command(
        self,
        ur_speed_l: Sequence[float],
        *,
        wrist_velocity_deg_s: Sequence[float] | None,
        gripper_target: float,
        issued_monotonic_ns: int | None = None,
    ) -> TeleopCommand:
        """只更新缓存；本方法不访问 RTDE、串口或相机。"""

        try:
            twist = _finite_tuple(ur_speed_l, 6, "ur_speed_l")
            wrist = (
                None
                if wrist_velocity_deg_s is None
                else _finite_tuple(wrist_velocity_deg_s, 2, "wrist_velocity_deg_s")
            )
            grip = float(gripper_target)
            if not math.isfinite(grip) or not 0.0 <= grip <= 1.0:
                raise ValueError("gripper_target 必须是 [0, 1] 内的有限数")
            issued = int(self._monotonic_ns() if issued_monotonic_ns is None else issued_monotonic_ns)
            now = int(self._monotonic_ns())
            age_ns = now - issued
            if age_ns < 0 or age_ns > int(self.config.max_command_age_s * 1e9):
                raise TimeoutError("提交的遥操作命令时间戳过期或来自未来")
            command = TeleopCommand(twist, wrist, grip, issued)
        except BaseException as exc:
            self._request_fail_closed(exc)
            raise
        with self._lock:
            if self._failure is not None:
                raise TeleopHardwareError(f"硬件 worker 已 fail-closed：{self._failure}") from self._failure
            self._command = command
        return command

    def set_episode_active(self, *, active: bool) -> None:
        """更新 episode 门禁；录制期间禁止 HOME 与 J6 ``speedJ``。"""

        value = bool(active)
        with self._lock:
            if value and (
                self._home_requested
                or self._home_in_progress
                or self._wrist_home_pending
                or self._joint6_direction != 0
            ):
                raise RuntimeError("HOME/J6 维护动作尚未结束，禁止开始 episode")
            self._episode_active = value

    def request_home(self) -> None:
        """请求从腕机械 HOME 和 UR task-home；只入队，不在调用线程做 I/O。"""

        with self._lock:
            if self._episode_active:
                raise PermissionError("episode 录制期间禁止 HOME/speedJ")
            if self._joint6_direction != 0:
                raise RuntimeError("J6 jog 未释放，禁止 HOME")
            self._home_requested = True
            self._wrist_home_pending = True
            self._home_reached_since_ns = None

    def request_wrist_master_resume(self) -> None:
        """HOME/SpaceMouse 覆盖后请求重新以主腕当前姿态建立跟随零点。"""

        with self._lock:
            if self._episode_active:
                raise PermissionError("episode 录制期间禁止重新绑定主腕零点")
            self._resume_master_requested = True

    def set_joint6_jog(self, direction: int) -> None:
        """设置维护用 J6 ``speedJ``；``-1/0/+1``，录制期间只能为零。"""

        if isinstance(direction, bool) or direction not in {-1, 0, 1}:
            raise ValueError("J6 direction 必须是 -1、0 或 +1")
        with self._lock:
            if direction != 0 and self._episode_active:
                raise PermissionError("episode 录制期间禁止 J6 speedJ")
            if direction != 0 and (self._home_requested or self._home_in_progress):
                raise RuntimeError("HOME 进行中，禁止 J6 jog")
            self._joint6_direction = direction

    def latest_action_receipt(self) -> ActionReceipt:
        """仅读取最近实际下发的 9 维 action 缓存。"""

        self.raise_if_failed()
        with self._lock:
            receipt = self._latest_receipt
        if receipt is None:
            raise RuntimeError("尚无完整 action receipt")
        return receipt

    def maintenance_status(self) -> MaintenanceStatus:
        """只读缓存；不会访问 RTDE 或串口。"""

        self.raise_if_failed()
        with self._lock:
            return MaintenanceStatus(
                home_requested=self._home_requested,
                home_in_progress=self._home_in_progress,
                wrist_home_pending=self._wrist_home_pending,
                joint6_direction=self._joint6_direction,
                resume_master_requested=self._resume_master_requested,
            )

    def get_observation(self, *, camera_timeout_s: float | None = None) -> dict[str, Any]:
        """返回可直接传给 :class:`EpisodeRecorder` 的完整 mapping。

        录制调用必须使用 ``observation["action_receipt"].action`` 作为
        ``append_if_due`` 的第二个参数。禁止重新从 SpaceMouse 原始轴或
        ``build_teleop_action`` 拼接训练 action。
        """

        self.raise_if_failed()
        cameras = self._cameras
        if cameras is None or not self.connected:
            raise RuntimeError("硬件 worker 尚未连接")
        timeout = self.config.camera_timeout_s if camera_timeout_s is None else float(camera_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("camera_timeout_s 必须是正有限数")
        try:
            frames = cameras.read(timeout_s=timeout)
            if set(frames) != {"front", "side", "top"}:
                raise TeleopHardwareError(f"相机同步器未返回完整三帧：{sorted(frames)}")
            with self._lock:
                ur = self._latest_ur
                wrist = self._latest_wrist
                gripper = self._latest_gripper
                receipt = self._latest_receipt
                failure = self._failure
            if failure is not None:
                raise TeleopHardwareError(f"硬件 worker 已 fail-closed：{failure}") from failure
            if ur is None or wrist is None or gripper is None or receipt is None:
                raise RuntimeError("执行器缓存尚未产生完整状态")
            qpos = np.asarray(
                (*ur.qpos_rad, *wrist.actual_relative_raw, float(gripper.position)),
                dtype=np.float32,
            )
            if qpos.shape != (9,) or not np.isfinite(qpos).all():
                raise ValueError("执行器 observation 含非法 qpos")
            now_ns = int(self._monotonic_ns())
            ages_s = {
                "ur5": (now_ns - int(ur.monotonic_ns)) / 1e9,
                "wrist": (now_ns - int(wrist.host_monotonic_ns)) / 1e9,
                "gripper": (now_ns - int(gripper.host_monotonic_ns)) / 1e9,
            }
            if any(age < 0.0 or age > self.config.max_state_age_s for age in ages_s.values()):
                raise TimeoutError(f"执行器缓存过期：ages_s={ages_s}")
            images = {role: np.ascontiguousarray(frame.color).copy() for role, frame in frames.items()}
            raw_timestamps = {role: int(frame.device_timestamp_ns) for role, frame in frames.items()}
            host_timestamps = {role: int(frame.host_timestamp_ns) for role, frame in frames.items()}
            sequences = {role: int(frame.sequence) for role, frame in frames.items()}
            camera_skew_ms = (max(host_timestamps.values()) - min(host_timestamps.values())) / 1e6
            health = {
                "ok": True,
                "ur5": {"healthy": True, "state_age_s": ages_s["ur5"]},
                "wrist": {
                    "healthy": True,
                    "state_age_s": ages_s["wrist"],
                    "mode": wrist.mode,
                    "coordinate": "yaml_servo_zero_relative_raw",
                    "servo_zero_raw": wrist.output_state.servo_zero_raw,
                    "hardware_limits_raw": wrist.output_state.hardware_limits_raw,
                    "motor_position_raw": wrist.output_state.motor_position_raw,
                    "encoder_abs_deg": wrist.output_state.encoder_abs_deg,
                    "encoder_valid": wrist.output_state.encoder_valid,
                },
                "gripper": {
                    "healthy": True,
                    "state_age_s": ages_s["gripper"],
                    "state_source": self._gripper_state_source,
                    "backend": gripper.backend,
                },
                "cameras": {"healthy": True, "max_skew_ms": camera_skew_ms},
                "worker": {"healthy": True},
            }
            return {
                "images": images,
                "front": images["front"],
                "side": images["side"],
                "top": images["top"],
                "qpos": qpos,
                "tcp_pose": np.asarray(ur.tcp_pose, dtype=np.float32).copy(),
                "timestamp_ns": now_ns,
                "monotonic_ns": now_ns,
                "camera_timestamps_ns": raw_timestamps,
                "camera_raw_timestamps_ns": raw_timestamps.copy(),
                "camera_host_timestamps_ns": host_timestamps,
                "camera_sequences": sequences,
                "camera_frames": MappingProxyType(dict(frames)),
                "state_timestamps_ns": {
                    "ur5": int(ur.monotonic_ns),
                    "wrist": int(wrist.host_monotonic_ns),
                    "gripper": int(gripper.host_monotonic_ns),
                },
                "hardware_health": health,
                "action_receipt": receipt,
            }
        except BaseException as exc:
            self._request_fail_closed(exc)
            raise

    def report_input_failure(self, failure: BaseException) -> None:
        """SpaceMouse/UI 线程报告异常并立即请求 owner 线程 fail-closed。"""

        if not isinstance(failure, BaseException):
            raise TypeError("failure 必须是异常对象")
        self._request_fail_closed(failure)

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise TeleopHardwareError(f"硬件 worker 已 fail-closed：{failure}") from failure

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """请求 owner 线程停机并等待它在所属线程中关闭全部执行器。"""

        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s 必须是正有限数")
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout_s)
        if thread.is_alive():
            raise TimeoutError("actuator owner worker 未在期限内停止")

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> TeleopHardwareWorker:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _run(self) -> None:
        ur: _UR | None = None
        wrist: _Wrist | None = None
        gripper: GripperInterface | None = None
        cameras: Any | None = None
        ready_signalled = False
        try:
            ur = self._ur_factory(self._config, enable_motion=self.enable_motion)
            wrist = self._wrist_factory(self._config)
            gripper = self._gripper_factory(self._config)
            cameras = self._camera_factory(self._camera_config())
            self._cameras = cameras

            cameras.connect()
            ur.connect()
            wrist_state = wrist.connect(enable_motion=True)
            wrist.resume_master()
            gripper.connect()
            ur_state = ur.read()
            gripper_state = gripper.get()
            now_ns = int(self._monotonic_ns())
            with self._lock:
                # 连接/HOME 可能较久；从 ready 时刻开始计算首个零命令租约。
                self._command = TeleopCommand((0.0,) * 6, None, float(gripper_state.target), now_ns)
                self._latest_ur = ur_state
                self._latest_wrist = wrist_state
                self._latest_gripper = gripper_state
            self._ready_event.set()
            ready_signalled = True
            self._control_loop(ur, wrist, gripper)
        except BaseException as exc:
            self._store_failure(exc)
        finally:
            # stop/close 必须留在创建连接的 owner 线程。
            for device in (ur, wrist, gripper):
                if device is not None:
                    with suppress(Exception):
                        device.stop()
            for device in (gripper, wrist, ur):
                if device is not None:
                    with suppress(Exception):
                        device.close()
            if cameras is not None:
                with suppress(Exception):
                    cameras.close()
            if not ready_signalled:
                self._ready_event.set()
            self._finished_event.set()

    def _control_loop(self, ur: _UR, wrist: _Wrist, gripper: GripperInterface) -> None:
        ur_period = 1.0 / self.config.ur_hz
        wrist_period = 1.0 / self.config.wrist_hz
        gripper_period = 1.0 / self.config.gripper_hz
        next_ur = self._monotonic()
        next_wrist = next_ur
        next_gripper = next_ur
        last_ur = (0.0,) * 6
        last_wrist = _executed_wrist_target(self._latest_wrist) if self._latest_wrist is not None else (0.0, 0.0)
        last_gripper = (
            float(self._latest_gripper.target) if self._latest_gripper is not None else 0.0
        )
        sent_ur_ns = sent_wrist_ns = sent_gripper_ns = int(self._monotonic_ns())
        last_requested_gripper = last_gripper
        wrist_override_was_active = False
        wrist_resume_required = False
        wrist_refreshed_after_maintenance = True

        while not self._stop_event.is_set():
            now = self._monotonic()
            now_ns = int(self._monotonic_ns())
            command, episode, home, wrist_home, jog, resume, fatal = self._command_snapshot()
            if fatal is not None:
                raise fatal
            age_ns = now_ns - command.issued_monotonic_ns
            if age_ns < 0 or age_ns > int(self.config.max_command_age_s * 1e9):
                raise TimeoutError(
                    f"遥操作命令过期：{age_ns / 1e6:.1f}ms > {self.config.max_command_age_s * 1e3:.1f}ms"
                )
            if episode and (home or wrist_home or jog != 0):
                raise TeleopHardwareError("episode 中检测到 HOME/J6 维护动作请求")
            maintenance_action = home or wrist_home or jog != 0
            if maintenance_action:
                # speedJ/HOME 不属于本项目的 9 维训练 action；维护期间禁止
                # 录制线程误把旧的 speedL receipt 当作本周期实际命令。
                with self._lock:
                    self._latest_receipt = None
                wrist_refreshed_after_maintenance = False

            if resume and not maintenance_action:
                wrist.resume_master()
                wrist_override_was_active = False
                wrist_resume_required = False
                with self._lock:
                    self._resume_master_requested = False

            if wrist_home:
                ur.stop()
                last_ur = (0.0,) * 6
                wrist_state = wrist.home(enable_motion=True)
                last_wrist = _executed_wrist_target(wrist_state)
                wrist_override_was_active = False
                wrist_resume_required = True
                sent_wrist_ns = int(self._monotonic_ns())
                with self._lock:
                    self._latest_wrist = wrist_state
                    self._wrist_home_pending = False
                    self._home_in_progress = True
                next_wrist = self._monotonic() + wrist_period

            if now >= next_ur:
                ur_state = ur.read()
                if home:
                    result = ur.task_home_step(
                        self.config.task_home_rad,
                        max_speed_rad_s=self.config.home_speed_rad_s,
                        tolerance_rad=self.config.home_tolerance_rad,
                        proportional_gain=self.config.home_proportional_gain,
                        state_timestamp_ns=ur_state.monotonic_ns,
                    )
                    last_ur = tuple(float(value) for value in result.velocity_rad_s)
                    if bool(result.reached):
                        reached_ns = int(self._monotonic_ns())
                        with self._lock:
                            if self._home_reached_since_ns is None:
                                self._home_reached_since_ns = reached_ns
                            stable = reached_ns - self._home_reached_since_ns >= int(
                                self.config.home_stable_s * 1e9
                            )
                            if stable:
                                self._home_requested = False
                                self._home_in_progress = False
                                self._home_reached_since_ns = None
                        if stable:
                            ur.stop()
                            last_ur = (0.0,) * 6
                    else:
                        with self._lock:
                            self._home_reached_since_ns = None
                elif jog != 0:
                    result = ur.jog_joint6(
                        jog,
                        speed_rad_s=self.config.joint6_jog_speed_rad_s,
                        state_timestamp_ns=ur_state.monotonic_ns,
                    )
                    last_ur = tuple(float(value) for value in result.velocity_rad_s)
                else:
                    sent = ur.send(command.ur_speed_l, state_timestamp_ns=ur_state.monotonic_ns)
                    last_ur = tuple(float(value) for value in sent)
                sent_ur_ns = int(self._monotonic_ns())
                with self._lock:
                    self._latest_ur = ur_state
                next_ur = _advance_deadline(next_ur, ur_period, self._monotonic())

            if not maintenance_action and not wrist_resume_required and now >= next_wrist:
                if command.wrist_velocity_deg_s is None:
                    if wrist_override_was_active:
                        wrist.clear_spacemouse_velocity()
                        # 以松 Ctrl 时的主腕姿态重新建零，避免突然跳回进入
                        # override 前的旧基线；之后恢复 ESP32 主腕跟随。
                        wrist.resume_master()
                        wrist_override_was_active = False
                else:
                    wrist.set_spacemouse_velocity(*command.wrist_velocity_deg_s)
                    wrist_override_was_active = True
                wrist_state = wrist.step()
                last_wrist = _executed_wrist_target(wrist_state)
                sent_wrist_ns = int(self._monotonic_ns())
                with self._lock:
                    self._latest_wrist = wrist_state
                wrist_refreshed_after_maintenance = True
                next_wrist = _advance_deadline(next_wrist, wrist_period, self._monotonic())

            if now >= next_gripper:
                if not math.isclose(command.gripper_target, last_requested_gripper, abs_tol=1e-12):
                    executed_gripper = gripper.set_position(command.gripper_target)
                    last_requested_gripper = command.gripper_target
                    last_gripper = float(
                        command.gripper_target if executed_gripper is None else executed_gripper
                    )
                    sent_gripper_ns = int(self._monotonic_ns())
                gripper_state = gripper.get()
                # target 是驱动记录的、已经下发到底层的绝对目标。
                last_gripper = float(gripper_state.target)
                with self._lock:
                    self._latest_gripper = gripper_state
                next_gripper = _advance_deadline(next_gripper, gripper_period, self._monotonic())

            if maintenance_action or wrist_resume_required or not wrist_refreshed_after_maintenance:
                pass
            elif all(math.isfinite(value) for value in (*last_ur, *last_wrist, last_gripper)):
                self._publish_receipt(last_ur, last_wrist, last_gripper, sent_ur_ns, sent_wrist_ns, sent_gripper_ns)
            else:
                raise ValueError("底层执行器返回的 action receipt 含 NaN/Inf")

            deadline = min(next_ur, next_wrist, next_gripper)
            self._stop_event.wait(max(0.0, min(ur_period, deadline - self._monotonic())))

    def _publish_receipt(
        self,
        ur: tuple[float, ...],
        wrist: tuple[float, float],
        gripper: float,
        ur_ns: int,
        wrist_ns: int,
        gripper_ns: int,
    ) -> None:
        action = (*ur, *wrist, gripper)
        if len(action) != 9:
            raise AssertionError("内部 action receipt 不是 9 维")
        with self._lock:
            self._receipt_sequence += 1
            self._latest_receipt = ActionReceipt(
                action=action,
                host_monotonic_ns=int(self._monotonic_ns()),
                sequence=self._receipt_sequence,
                ur_sent_monotonic_ns=ur_ns,
                wrist_sent_monotonic_ns=wrist_ns,
                gripper_sent_monotonic_ns=gripper_ns,
            )

    def _command_snapshot(
        self,
    ) -> tuple[TeleopCommand, bool, bool, bool, int, bool, BaseException | None]:
        with self._lock:
            return (
                self._command,
                self._episode_active,
                self._home_requested,
                self._wrist_home_pending,
                self._joint6_direction,
                self._resume_master_requested,
                self._fatal_command_error,
            )

    def _request_fail_closed(self, failure: BaseException) -> None:
        with self._lock:
            if self._fatal_command_error is None:
                self._fatal_command_error = failure
            if self._failure is None:
                self._failure = failure
        self._stop_event.set()

    def _store_failure(self, failure: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = failure
        self._stop_event.set()

    def _camera_config(self) -> dict[str, Any]:
        hardware = _mapping(self._config.get("hardware"), "hardware")
        cameras = dict(_mapping(hardware.get("cameras"), "hardware.cameras"))
        timing = _mapping(_mapping(self._config.get("safety"), "safety").get("timing"), "safety.timing")
        cameras.setdefault("max_camera_skew_ms", float(timing.get("max_camera_skew_ms", 50.0)))
        cameras.setdefault("max_frame_age_ms", float(timing.get("camera_timeout_ms", 500.0)))
        cameras.setdefault("read_timeout_s", float(timing.get("camera_timeout_ms", 500.0)) / 1000.0)
        return cameras


def _default_ur_factory(config: Mapping[str, Any], *, enable_motion: bool) -> UR5Controller:
    return UR5Controller(UR5Config.from_mapping(config, enable_motion=enable_motion))


def _default_wrist_factory(config: Mapping[str, Any]) -> WristMasterSlaveController:
    return WristMasterSlaveController(config)


def _default_gripper_factory(config: Mapping[str, Any]) -> GripperInterface:
    return create_gripper(config)


def _default_camera_factory(config: Mapping[str, Any]) -> ThreeCameraCapture:
    return ThreeCameraCapture.from_mapping(config)


def _executed_wrist_target(state: Any) -> tuple[float, float]:
    """优先使用 OpenRB 量化后的 J1/J2 相对 raw，而不是未量化请求。"""

    source = getattr(state, "target_relative_raw", None)
    if source is None:
        output = getattr(state, "output_state", None)
        source = getattr(output, "target_relative_raw", None) if output is not None else None
    return _finite_tuple(source, 2, "OpenRB 实际最后下发目标")


def spacemouse_wrist_velocity_deg_s(
    sample: SpaceMouseSample,
    project_config: Mapping[str, Any],
) -> tuple[float, float] | None:
    """把 Ctrl 模式帽输入转换为 worker 所需的腕速度。

    未按 Ctrl 时返回 ``None``，从而保留 ESP32 主腕跟随。返回值绝不是
    ``build_teleop_action`` 中积分后的绝对 wrist target。
    """

    if not sample.connected or sample.stale:
        raise TimeoutError("SpaceMouse 状态过期或断开")
    motion = np.asarray(sample.motion, dtype=np.float64).reshape(-1)
    if motion.shape != (6,) or not np.isfinite(motion).all():
        raise ValueError("SpaceMouse motion 必须是 6 维有限数")
    teleop = _mapping(project_config.get("teleop"), "teleop")
    modes = _mapping(teleop.get("modes"), "teleop.modes")
    wrist = _mapping(modes.get("wrist"), "teleop.modes.wrist")
    modifier = str(wrist.get("modifier", "ctrl"))
    if not sample.pressed(modifier):
        return None
    deadzone = float(wrist.get("deadzone", 0.18))
    if not math.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
        raise ValueError("teleop wrist deadzone 必须位于 [0, 1)")
    response = np.sign(motion[:2]) * np.clip(
        (np.abs(motion[:2]) - deadzone) / (1.0 - deadzone),
        0.0,
        1.0,
    )
    maximum = _positive(wrist.get("max_speed_deg_s", 45.0), "teleop.modes.wrist.max_speed_deg_s")
    signs = np.asarray([float(wrist.get("cap_x_sign", -1.0)), float(wrist.get("cap_y_sign", 1.0))])
    if not np.isin(signs, (-1.0, 1.0)).all():
        raise ValueError("Ctrl wrist cap_x_sign/cap_y_sign 必须为 -1 或 +1")
    axes = (str(wrist.get("cap_x_to_axis", "j2")), str(wrist.get("cap_y_to_axis", "j1")))
    if set(axes) != {"j1", "j2"}:
        raise ValueError("Ctrl wrist cap_x_to_axis/cap_y_to_axis 必须恰好映射到 j1/j2")
    mapped = {axis: float(value) for axis, value in zip(axes, response * signs * maximum, strict=True)}
    return (mapped["j1"], mapped["j2"])


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 mapping")
    return value


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是正有限数")
    return result


def _finite_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes | Mapping):
        raise ValueError(f"{name} 必须是 {size} 维序列")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 {size} 维数值序列") from exc
    if len(result) != size:
        raise ValueError(f"{name} 必须是 {size} 维序列")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} 含 NaN/Inf")
    return result


def _require_joint_limits_for_motion(config: Mapping[str, Any]) -> None:
    safety = _mapping(config.get("safety"), "safety")
    ur = _mapping(safety.get("ur5"), "safety.ur5")
    lower = ur.get("joint_min_rad")
    upper = ur.get("joint_max_rad")
    low = _finite_tuple(lower, 6, "safety.ur5.joint_min_rad")
    high = _finite_tuple(upper, 6, "safety.ur5.joint_max_rad")
    if any(a >= b for a, b in zip(low, high, strict=True)):
        raise ValueError("UR5 每个关节软限位下界必须小于上界")


def _advance_deadline(previous: float, period: float, now: float) -> float:
    """丢弃已经错过的周期，避免串口抖动后突发补发命令。"""

    return max(previous + period, now + period)


__all__ = [
    "ActionReceipt",
    "MaintenanceStatus",
    "TeleopCommand",
    "TeleopHardwareError",
    "TeleopHardwareWorker",
    "TeleopWorkerConfig",
    "spacemouse_wrist_velocity_deg_s",
]
