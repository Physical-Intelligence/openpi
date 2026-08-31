# ruff: noqa: RUF001, RUF002, RUF003
"""UR5 + 双轴腕 + 夹爪 + 三相机的项目内统一硬件实现。

构造本类只解析 YAML mapping 并创建无副作用的 Python 对象。只有显式调用
``connect`` 才连接设备；``enable_motion`` 默认为 ``False``，因此默认只能
读取 observation。真实动作顺序固定为 UR5 ``speedL``、手腕绝对目标、夹爪
绝对目标，成功和部分失败都有同周期 command receipt 可供审计。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from examples.ur5_twinwrist.cameras.models import CameraFrame
from examples.ur5_twinwrist.cameras.models import CameraRigConfig
from examples.ur5_twinwrist.cameras.realsense import ThreeCameraCapture
from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.config_loader import validate_project_config
from examples.ur5_twinwrist.controller.gripper import GripperInterface
from examples.ur5_twinwrist.controller.gripper import create_gripper
from examples.ur5_twinwrist.controller.ur5 import UR5Config
from examples.ur5_twinwrist.controller.ur5 import UR5Controller
from examples.ur5_twinwrist.controller.ur5 import UR5State
from examples.ur5_twinwrist.controller.wrist import OpenRBWrist
from examples.ur5_twinwrist.controller.wrist import WristConfig
from examples.ur5_twinwrist.controller.wrist import WristState

EXPECTED_CAMERA_ROLES = ("front", "side", "top")
ACTION_DIMENSION = 9


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 mapping")
    return value


def _finite_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (dimension,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须是 {dimension} 维有限数")
    return result


def _milliseconds_to_seconds(value: Any, name: str) -> float:
    milliseconds = float(value)
    if not math.isfinite(milliseconds) or milliseconds <= 0.0:
        raise ValueError(f"{name} 必须是正有限数")
    return milliseconds / 1000.0


@dataclass(frozen=True)
class LocalHardwareConfig:
    """从五份项目 YAML 合并出的统一运行配置。"""

    ur5: UR5Config
    wrist: WristConfig
    cameras: CameraRigConfig
    gripper: Mapping[str, Any]
    enable_motion: bool
    max_command_age_s: float
    ur_control_hz: float
    wrist_command_hz: float
    gripper_command_hz: float
    ur_task_home_rad: tuple[float, ...]
    ur_home_speed_rad_s: float
    ur_home_proportional_gain: float
    ur_home_tolerance_rad: float
    ur_home_stable_s: float
    ur_joint6_jog_speed_rad_s: float

    @classmethod
    def from_mapping(
        cls,
        project: Mapping[str, Any],
        *,
        enable_motion: bool = False,
    ) -> LocalHardwareConfig:
        # 真实运动要求 PolyScope 关节软限位已经填入；只读模式允许 null。
        validate_project_config(dict(project), require_real_ready=enable_motion)
        hardware = _mapping(project.get("hardware"), "hardware")
        safety = _mapping(project.get("safety"), "safety")
        hardware_wrist = _mapping(hardware.get("wrist"), "hardware.wrist")
        hardware_ur5 = _mapping(hardware.get("ur5"), "hardware.ur5")
        safety_ur5 = _mapping(safety.get("ur5"), "safety.ur5")
        timing = _mapping(safety.get("timing"), "safety.timing")
        poses = _mapping(project.get("poses"), "poses")
        poses_ur5 = _mapping(poses.get("ur5"), "poses.ur5")

        wrist = WristConfig.from_mapping(project)
        cameras = _camera_config(project)
        gripper = _mapping(hardware.get("gripper"), "hardware.gripper")
        max_command_age_s = min(
            _milliseconds_to_seconds(timing.get("max_command_age_ms", 250.0), "safety.timing.max_command_age_ms"),
            _milliseconds_to_seconds(
                timing.get("observation_timeout_ms", 200.0),
                "safety.timing.observation_timeout_ms",
            ),
        )
        ur_control_hz = float(hardware_ur5.get("control_hz", 125.0))
        wrist_command_hz = float(hardware_wrist.get("command_hz", 30.0))
        gripper_command_hz = float(gripper.get("command_hz", 30.0))
        for name, value in (
            ("hardware.ur5.control_hz", ur_control_hz),
            ("hardware.wrist.command_hz", wrist_command_hz),
            ("hardware.gripper.command_hz", gripper_command_hz),
        ):
            if not math.isfinite(value) or not 0.0 < value <= ur_control_hz:
                raise ValueError(f"{name} 必须位于 (0, UR control_hz]")
        home_target = _finite_vector(poses_ur5.get("task_home_rad"), 6, "poses.ur5.task_home_rad")
        home_speed = float(poses_ur5.get("home_speed_rad_s", 0.50))
        home_gain = float(poses_ur5.get("home_proportional_gain", 1.50))
        home_tolerance = float(poses_ur5.get("home_tolerance_rad", 0.010))
        home_stable = float(poses_ur5.get("home_stable_s", 0.30))
        joint6_speed = float(safety_ur5.get("joint6_jog_speed_rad_s", 0.20))
        for name, value in (
            ("poses.ur5.home_speed_rad_s", home_speed),
            ("poses.ur5.home_proportional_gain", home_gain),
            ("poses.ur5.home_tolerance_rad", home_tolerance),
            ("poses.ur5.home_stable_s", home_stable),
            ("safety.ur5.joint6_jog_speed_rad_s", joint6_speed),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是正有限数")
        return cls(
            ur5=UR5Config.from_mapping(project, enable_motion=enable_motion),
            wrist=wrist,
            cameras=cameras,
            gripper=dict(gripper),
            enable_motion=bool(enable_motion),
            max_command_age_s=max_command_age_s,
            ur_control_hz=ur_control_hz,
            wrist_command_hz=wrist_command_hz,
            gripper_command_hz=gripper_command_hz,
            ur_task_home_rad=tuple(float(value) for value in home_target),
            ur_home_speed_rad_s=home_speed,
            ur_home_proportional_gain=home_gain,
            ur_home_tolerance_rad=home_tolerance,
            ur_home_stable_s=home_stable,
            ur_joint6_jog_speed_rad_s=joint6_speed,
        )


def _camera_config(project: Mapping[str, Any]) -> CameraRigConfig:
    hardware = _mapping(project.get("hardware"), "hardware")
    safety = _mapping(project.get("safety"), "safety")
    camera = _mapping(hardware.get("cameras"), "hardware.cameras")
    timing = _mapping(safety.get("timing"), "safety.timing")
    devices = camera.get("devices")
    if not isinstance(devices, Sequence) or isinstance(devices, str | bytes):
        raise ValueError("hardware.cameras.devices 必须是列表")
    common = {
        "width": int(camera.get("width", 640)),
        "height": int(camera.get("height", 480)),
        "fps": int(camera.get("fps", 30)),
        "enable_depth": bool(camera.get("enable_depth", False)),
    }
    merged_devices = [
        {**common, **dict(_mapping(item, f"hardware.cameras.devices[{index}]"))} for index, item in enumerate(devices)
    ]
    camera_timeout_s = _milliseconds_to_seconds(
        timing.get("camera_timeout_ms", 500.0), "safety.timing.camera_timeout_ms"
    )
    return CameraRigConfig.from_mapping(
        {
            "devices": merged_devices,
            "max_camera_skew_ms": timing.get("max_camera_skew_ms", 50.0),
            "max_frame_age_ms": timing.get("camera_timeout_ms", 500.0),
            "read_timeout_s": camera_timeout_s,
            "connect_timeout_s": camera.get("connect_timeout_s", 5.0),
            "provider_wait_timeout_s": min(0.25, camera_timeout_s),
            "stop_timeout_s": camera.get("stop_timeout_s", 3.0),
            "queue_size": camera.get("queue_size", 8),
        }
    )


UR5Factory = Callable[[UR5Config], Any]
WristFactory = Callable[[WristConfig], Any]
GripperFactory = Callable[[Mapping[str, Any]], Any]
CameraFactory = Callable[[CameraRigConfig], Any]


def _make_ur5(config: UR5Config) -> UR5Controller:
    return UR5Controller(config)


def _make_wrist(config: WristConfig) -> OpenRBWrist:
    return OpenRBWrist(config)


def _make_gripper(config: Mapping[str, Any]) -> GripperInterface:
    return create_gripper(config)


def _make_cameras(config: CameraRigConfig) -> ThreeCameraCapture:
    return ThreeCameraCapture(config)


@dataclass(frozen=True)
class HardwareFactories:
    """Fake 测试或现场替换后端时使用的无副作用工厂集合。"""

    ur5: UR5Factory = _make_ur5
    wrist: WristFactory = _make_wrist
    gripper: GripperFactory = _make_gripper
    cameras: CameraFactory = _make_cameras


@dataclass(frozen=True)
class CommandReceipt:
    """一次 9D action 的顺序执行回执。"""

    sequence: int
    observation_sequence: int
    observation_monotonic_ns: int
    ur5_state_monotonic_ns: int
    command_monotonic_ns: int
    requested_action: np.ndarray
    ur5_speedl: np.ndarray
    wrist_target_relative_raw: tuple[float, float]
    gripper_target: float
    completed_devices: tuple[str, ...]
    ok: bool
    error: str | None = None
    wrist_command_sent: bool = True
    gripper_command_sent: bool = True
    recordable: bool = True

    @property
    def executed_action(self) -> np.ndarray:
        """返回可写入训练集的 9D 实际命令，绝不返回未下发的辅助目标。"""

        return np.asarray(
            [*self.ur5_speedl, *self.wrist_target_relative_raw, self.gripper_target],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "observation_sequence": self.observation_sequence,
            "observation_monotonic_ns": self.observation_monotonic_ns,
            "ur5_state_monotonic_ns": self.ur5_state_monotonic_ns,
            "command_monotonic_ns": self.command_monotonic_ns,
            "requested_action": self.requested_action.copy(),
            "executed_action": self.executed_action,
            "ur5_speedl": self.ur5_speedl.copy(),
            "wrist_target_relative_raw": self.wrist_target_relative_raw,
            "gripper_target": self.gripper_target,
            "completed_devices": self.completed_devices,
            "ok": self.ok,
            "error": self.error,
            "wrist_command_sent": self.wrist_command_sent,
            "gripper_command_sent": self.gripper_command_sent,
            "recordable": self.recordable,
        }


@dataclass(frozen=True)
class MaintenanceReceipt:
    """HOME/J6 ``speedJ`` 维护回执；明确禁止写入训练 episode。"""

    sequence: int
    kind: str
    ur5_state_monotonic_ns: int
    command_monotonic_ns: int
    joint_velocity_rad_s: np.ndarray
    target_reached: bool | None
    stable: bool | None
    recordable: bool = field(default=False, init=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "ur5_state_monotonic_ns": self.ur5_state_monotonic_ns,
            "command_monotonic_ns": self.command_monotonic_ns,
            "joint_velocity_rad_s": self.joint_velocity_rad_s.copy(),
            "target_reached": self.target_reached,
            "stable": self.stable,
            "recordable": False,
        }


class LocalRobotHardware:
    """本项目统一 RobotHardware，所有串口 I/O 由 connect 线程拥有。"""

    def __init__(
        self,
        config: LocalHardwareConfig | Mapping[str, Any],
        *,
        enable_motion: bool | None = None,
        factories: HardwareFactories | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        resolved_enable_motion = bool(enable_motion) if enable_motion is not None else False
        self.config = (
            config
            if isinstance(config, LocalHardwareConfig)
            else LocalHardwareConfig.from_mapping(config, enable_motion=resolved_enable_motion)
        )
        if (
            isinstance(config, LocalHardwareConfig)
            and enable_motion is not None
            and bool(enable_motion) != config.enable_motion
        ):
            raise ValueError("enable_motion 必须与 LocalHardwareConfig.enable_motion 一致")
        factory = factories or HardwareFactories()
        # 下面只构造对象；各底层类保证构造时不导入 vendor 包或打开硬件。
        self.ur5 = factory.ur5(self.config.ur5)
        self.wrist = factory.wrist(self.config.wrist)
        self.gripper = factory.gripper(self.config.gripper)
        self.cameras = factory.cameras(self.config.cameras)
        self._clock_ns = monotonic_ns
        self._owner_thread_id: int | None = None
        self._connected = False
        self._started_components: list[str] = []
        self._fault: BaseException | None = None
        self._stopped = False
        self._observation_sequence = 0
        self._command_sequence = 0
        self._maintenance_sequence = 0
        self._last_observation_ns: int | None = None
        self._last_ur_state: UR5State | Any | None = None
        self._last_wrist_state: WristState | Any | None = None
        self._last_wrist_sent_target: tuple[float, float] | None = None
        self._last_gripper_sent_target: float | None = None
        self._last_wrist_command_ns: int | None = None
        self._last_gripper_command_ns: int | None = None
        self._last_command_receipt: CommandReceipt | None = None
        self._last_maintenance_receipt: MaintenanceReceipt | None = None
        self._home_reached_since_ns: int | None = None
        self._episode_active = False
        self._maintenance_motion_active = False

        # 只有显式 --enable-motion 的会话才允许调用执行器 stop。只读会话
        # close/trip 仅停止相机线程并关闭连接，绝不向腕/夹爪/UR 写命令。
        self._actuator_stop_allowed = bool(self.config.enable_motion)

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str | Path,
        *,
        enable_motion: bool = False,
        factories: HardwareFactories | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> LocalRobotHardware:
        """读取项目 YAML 并构造统一层；不会连接设备。"""

        project = load_project_config(config_dir, require_real_ready=enable_motion)
        return cls(
            project,
            enable_motion=enable_motion,
            factories=factories,
            monotonic_ns=monotonic_ns,
        )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_command_receipt(self) -> CommandReceipt | None:
        return self._last_command_receipt

    @property
    def last_maintenance_receipt(self) -> MaintenanceReceipt | None:
        return self._last_maintenance_receipt

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def connect(self) -> None:
        """按 UR、腕、夹爪、相机顺序连接；不会回零或发送目标。"""

        if self._connected or self._started_components:
            raise RuntimeError("统一硬件已经连接或正在连接")
        if self._fault is not None:
            raise RuntimeError(f"统一硬件已 fail-closed: {self._fault}") from self._fault
        self._owner_thread_id = threading.get_ident()
        try:
            for name in ("ur5", "wrist", "gripper", "cameras"):
                self._started_components.append(name)
                getattr(self, name).connect()
            self._connected = True
            self._stopped = False
            self._episode_active = False
            self._maintenance_motion_active = False
            self._home_reached_since_ns = None
        except BaseException as exc:
            self._trip(exc, close=True)
            raise

    def get_observation(self) -> dict[str, Any]:
        """读取一组三相机帧和相干的 9D 机器人状态。"""

        self._require_ready(allow_stopped=False)
        try:
            frames = self.cameras.read()
            self._validate_frame_roles(frames)
            # 把 UR 状态放在串口读取之后，使紧随其后的 speedL 使用最新快照。
            wrist_state = self.wrist.get()
            gripper_state = self.gripper.get()
            ur_state = self.ur5.read()
            qpos = np.asarray(
                [*ur_state.qpos_rad, *wrist_state.position_relative_raw, gripper_state.position],
                dtype=np.float32,
            )
            if qpos.shape != (9,) or not np.isfinite(qpos).all():
                raise RuntimeError("统一硬件 qpos 不是 9 维有限数")
            tcp_pose = _finite_vector(ur_state.tcp_pose, 6, "UR5 tcp_pose").astype(np.float32)
            now_ns = int(self._clock_ns())
            if now_ns < 0:
                raise RuntimeError("主机单调时间戳不得为负")
            self._observation_sequence += 1
            self._last_observation_ns = now_ns
            self._last_ur_state = ur_state
            self._last_wrist_state = wrist_state
            # 设备反馈中的 target 是本进程当前知道的“实际最后发送目标”。
            # 多速率控制跳过腕/夹爪发送时，回执必须继续报告这两个缓存值，
            # 不能把尚未下发的 requested_action 伪装成已执行 action。
            self._last_wrist_sent_target = tuple(
                float(value)
                for value in _finite_vector(wrist_state.target_relative_raw, 2, "手腕相对 raw 反馈目标")
            )
            self._last_gripper_sent_target = float(gripper_state.target)
            if not math.isfinite(self._last_gripper_sent_target):
                raise RuntimeError("夹爪反馈目标不是有限数")
            health = self._hardware_health(ur_state, wrist_state, gripper_state, frames)
            if not health["ok"]:
                raise RuntimeError(f"统一硬件健康检查失败: {health}")
            images = {role: np.ascontiguousarray(frames[role].color).copy() for role in EXPECTED_CAMERA_ROLES}
            raw_timestamps = {role: int(frames[role].device_timestamp_ns) for role in EXPECTED_CAMERA_ROLES}
            host_timestamps = {role: int(frames[role].host_timestamp_ns) for role in EXPECTED_CAMERA_ROLES}
            camera_sequences = {role: int(frames[role].sequence) for role in EXPECTED_CAMERA_ROLES}
            return {
                "images": images,
                "front": images["front"],
                "side": images["side"],
                "top": images["top"],
                "qpos": qpos,
                "tcp_pose": tcp_pose,
                "timestamp_ns": now_ns,
                "monotonic_ns": now_ns,
                "camera_timestamps_ns": raw_timestamps,
                "camera_raw_timestamps_ns": raw_timestamps.copy(),
                "camera_host_timestamps_ns": host_timestamps,
                "camera_sequences": camera_sequences,
                "sequence": self._observation_sequence,
                "state_timestamps_ns": {
                    "ur5": int(ur_state.monotonic_ns),
                    "wrist": int(wrist_state.host_monotonic_ns),
                    "gripper": int(gripper_state.host_monotonic_ns),
                },
                "hardware_health": health,
            }
        except BaseException as exc:
            self._trip(exc)
            raise

    def send_action(self, action: Sequence[float] | np.ndarray) -> CommandReceipt:
        """兼容接口：每次强制发送 UR5、腕和夹爪。

        action 语义为 ``speedL 6 + wrist J1/J2 YAML-zero-relative raw 2 + gripper absolute 1``。
        125 Hz 遥操作循环应优先调用 :meth:`control_step`，由它按 YAML
        ``command_hz`` 节流腕和夹爪。
        """

        return self.control_step(action, force_auxiliary=True)

    def control_step(
        self,
        action: Sequence[float] | np.ndarray,
        *,
        force_auxiliary: bool = False,
    ) -> CommandReceipt:
        """执行一个 125 Hz 主控制步，并返回“实际已发送命令”回执。

        每一步都会重新读取 UR5 实际状态，再发送 ``speedL``。腕和夹爪按
        ``wrist_command_hz`` / ``gripper_command_hz`` 独立节流；跳过某设备
        时，回执中的目标保持该设备实际最后一次成功发送的目标，而不是本次
        尚未下发的请求。相机和完整 observation 仍由 10 Hz 录制循环调用
        :meth:`get_observation`。

        ``force_auxiliary=True`` 保留旧 ``send_action`` 的三设备同周期行为。
        维护模式 ``speedJ`` 活跃时禁止进入本接口，必须先显式
        :meth:`stop_maintenance_motion` 并重新采集 observation。
        """

        self._require_ready(allow_stopped=False)
        completed: list[str] = []
        parsed: np.ndarray | None = None
        receipt_sequence = self._command_sequence + 1
        ur_state: Any | None = None
        wrist_sent = False
        gripper_sent = False
        try:
            if not self.config.enable_motion:
                raise PermissionError("真实动作未启用；必须由上层显式 --enable-motion")
            if self._maintenance_motion_active:
                raise RuntimeError("UR5 speedJ 维护模式仍活跃；请先 stop_maintenance_motion 并重新观测")
            parsed = _finite_vector(action, ACTION_DIMENSION, "robot action")
            now_ns = int(self._clock_ns())
            wrist_due = bool(force_auxiliary) or self._auxiliary_due(
                self._last_wrist_command_ns,
                self.config.wrist_command_hz,
                now_ns,
            )
            gripper_due = bool(force_auxiliary) or self._auxiliary_due(
                self._last_gripper_command_ns,
                self.config.gripper_command_hz,
                now_ns,
            )
            self._validate_action_before_send(parsed, validate_wrist_step=wrist_due)
            assert self._last_observation_ns is not None
            ur_state = self.ur5.read()
            self._last_ur_state = ur_state
            ur_command = np.asarray(
                self.ur5.send(
                    parsed[:6],
                    state_timestamp_ns=int(ur_state.monotonic_ns),
                ),
                dtype=np.float64,
            )
            completed.append("ur5")
            requested_wrist = (float(parsed[6]), float(parsed[7]))
            requested_gripper = float(parsed[8])
            command_ns = int(self._clock_ns())
            if wrist_due:
                wrist_result = _finite_vector(self.wrist.set(requested_wrist), 2, "手腕实际下发目标")
                self._last_wrist_sent_target = tuple(float(value) for value in wrist_result)
                self._last_wrist_command_ns = command_ns
                wrist_sent = True
                completed.append("wrist")
            if gripper_due:
                gripper_result = float(self.gripper.set(requested_gripper))
                if not math.isfinite(gripper_result) or not 0.0 <= gripper_result <= 1.0:
                    raise RuntimeError("夹爪驱动返回了非法实际下发目标")
                self._last_gripper_sent_target = gripper_result
                self._last_gripper_command_ns = command_ns
                gripper_sent = True
                completed.append("gripper")
            assert self._last_wrist_sent_target is not None
            assert self._last_gripper_sent_target is not None
            self._command_sequence = receipt_sequence
            receipt = CommandReceipt(
                sequence=receipt_sequence,
                observation_sequence=self._observation_sequence,
                observation_monotonic_ns=self._last_observation_ns,
                ur5_state_monotonic_ns=int(ur_state.monotonic_ns),
                command_monotonic_ns=command_ns,
                requested_action=parsed.copy(),
                ur5_speedl=ur_command.copy(),
                wrist_target_relative_raw=self._last_wrist_sent_target,
                gripper_target=self._last_gripper_sent_target,
                completed_devices=tuple(completed),
                ok=True,
                wrist_command_sent=wrist_sent,
                gripper_command_sent=gripper_sent,
            )
            self._last_command_receipt = receipt
            return receipt
        except BaseException as exc:
            if parsed is not None and self._last_observation_ns is not None:
                fallback_wrist = (float(parsed[6]), float(parsed[7]))
                fallback_gripper = float(parsed[8])
                state_ns = int(
                    getattr(
                        ur_state if ur_state is not None else self._last_ur_state,
                        "monotonic_ns",
                        self._last_observation_ns,
                    )
                )
                self._last_command_receipt = CommandReceipt(
                    sequence=receipt_sequence,
                    observation_sequence=self._observation_sequence,
                    observation_monotonic_ns=self._last_observation_ns,
                    ur5_state_monotonic_ns=state_ns,
                    command_monotonic_ns=int(self._clock_ns()),
                    requested_action=parsed.copy(),
                    ur5_speedl=parsed[:6].copy(),
                    wrist_target_relative_raw=self._last_wrist_sent_target or fallback_wrist,
                    gripper_target=(
                        self._last_gripper_sent_target
                        if self._last_gripper_sent_target is not None
                        else fallback_gripper
                    ),
                    completed_devices=tuple(completed),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    wrist_command_sent=wrist_sent,
                    gripper_command_sent=gripper_sent,
                    recordable=False,
                )
            self._trip(exc)
            raise

    def set_episode_active(self, *, active: bool) -> None:
        """显式标记训练 episode 生命周期，隔离维护 ``speedJ``。

        录制器应在 episode 开始前调用 ``set_episode_active(active=True)``，
        结束后传 ``active=False``。进入 episode 时若 HOME/J6 尚在输出，会先停止 UR5；
        下一次 recordable action 之前仍必须重新调用 ``get_observation``。
        """

        self._require_ready(allow_stopped=False)
        if not isinstance(active, bool):
            raise TypeError("active 必须是 bool")
        try:
            if active and self._maintenance_motion_active:
                self.stop_maintenance_motion()
            if active and not self._episode_active:
                self._last_observation_ns = None
            self._episode_active = active
        except BaseException as exc:
            self._trip(exc)
            raise

    def jog_joint6(self, direction: int) -> MaintenanceReceipt:
        """执行一次配置速度的 J6 ``speedJ`` 步；回执永远不可录制。"""

        try:
            self._require_maintenance_ready()
            state = self.ur5.read()
            self._last_ur_state = state
            result = self.ur5.jog_joint6(
                direction,
                speed_rad_s=self.config.ur_joint6_jog_speed_rad_s,
                state_timestamp_ns=int(state.monotonic_ns),
            )
            self._maintenance_motion_active = True
            self._home_reached_since_ns = None
            self._last_observation_ns = None
            self._maintenance_sequence += 1
            receipt = MaintenanceReceipt(
                sequence=self._maintenance_sequence,
                kind="joint6_jog",
                ur5_state_monotonic_ns=int(result.state_timestamp_ns),
                command_monotonic_ns=int(self._clock_ns()),
                joint_velocity_rad_s=_finite_vector(result.velocity_rad_s, 6, "UR5 J6 实际速度").copy(),
                target_reached=None,
                stable=None,
            )
            self._last_maintenance_receipt = receipt
            return receipt
        except BaseException as exc:
            self._trip(exc)
            raise

    def task_home_step(self) -> MaintenanceReceipt:
        """朝 YAML ``task_home_rad`` 执行一个 125 Hz ``speedJ`` 步。"""

        try:
            self._require_maintenance_ready()
            state = self.ur5.read()
            self._last_ur_state = state
            result = self.ur5.task_home_step(
                self.config.ur_task_home_rad,
                max_speed_rad_s=self.config.ur_home_speed_rad_s,
                tolerance_rad=self.config.ur_home_tolerance_rad,
                proportional_gain=self.config.ur_home_proportional_gain,
                state_timestamp_ns=int(state.monotonic_ns),
            )
            self._maintenance_motion_active = True
            self._last_observation_ns = None
            command_ns = int(self._clock_ns())
            reached = bool(result.reached)
            if reached:
                if self._home_reached_since_ns is None:
                    self._home_reached_since_ns = command_ns
                stable = command_ns - self._home_reached_since_ns >= int(self.config.ur_home_stable_s * 1e9)
            else:
                self._home_reached_since_ns = None
                stable = False
            self._maintenance_sequence += 1
            receipt = MaintenanceReceipt(
                sequence=self._maintenance_sequence,
                kind="task_home",
                ur5_state_monotonic_ns=int(result.state_timestamp_ns),
                command_monotonic_ns=command_ns,
                joint_velocity_rad_s=_finite_vector(result.velocity_rad_s, 6, "UR5 HOME 实际速度").copy(),
                target_reached=reached,
                stable=stable,
            )
            self._last_maintenance_receipt = receipt
            return receipt
        except BaseException as exc:
            self._trip(exc)
            raise

    def stop_maintenance_motion(self) -> None:
        """停止 HOME/J6 的 UR ``speedJ``，不停止相机、腕或夹爪。"""

        self._require_ready(allow_stopped=False)
        if not self._maintenance_motion_active:
            return
        try:
            self.ur5.stop()
            self._maintenance_motion_active = False
            self._home_reached_since_ns = None
            self._last_observation_ns = None
        except BaseException as exc:
            self._trip(exc)
            raise

    def stop(self) -> None:
        """停止本会话允许的运动和相机线程；对已关闭对象幂等。

        只读会话不会调用 UR/腕/夹爪 ``stop``，因为这些方法本身会写入
        执行器。只有显式启用 ``--enable-motion`` 的会话才有停执行器权限。
        """

        self._assert_owner_if_started()
        errors = self._stop_components()
        self._stopped = True
        if errors:
            raise RuntimeError(f"停止统一硬件失败: {errors[0]}") from errors[0]

    def close(self) -> None:
        """停止并按相反顺序关闭全部设备；可重复调用。"""

        self._assert_owner_if_started()
        with suppress(Exception):
            self._stop_components()
        errors: list[BaseException] = []
        for name in reversed(self._started_components):
            try:
                getattr(self, name).close()
            except BaseException as exc:
                errors.append(exc)
        self._started_components.clear()
        self._connected = False
        self._stopped = True
        self._episode_active = False
        self._maintenance_motion_active = False
        self._home_reached_since_ns = None
        self._owner_thread_id = None
        if errors:
            raise RuntimeError(f"关闭统一硬件失败: {errors[0]}") from errors[0]

    def healthy(self) -> bool:
        if not self._connected or self._fault is not None or self._stopped:
            return False
        checks = []
        for name in ("ur5", "wrist", "gripper"):
            check = getattr(getattr(self, name), "healthy", None)
            checks.append(bool(check()) if callable(check) else True)
        return all(checks)

    def __enter__(self) -> LocalRobotHardware:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _validate_action_before_send(self, action: np.ndarray, *, validate_wrist_step: bool) -> None:
        if self._last_observation_ns is None or self._last_ur_state is None:
            raise RuntimeError("发送动作前必须先调用 get_observation")
        age_ns = int(self._clock_ns()) - self._last_observation_ns
        max_age_ns = int(self.config.max_command_age_s * 1e9)
        if age_ns < 0 or age_ns > max_age_ns:
            raise TimeoutError(f"统一 observation 过期: {age_ns / 1e6:.1f} ms > {max_age_ns / 1e6:.1f} ms")
        wrist_state = self._last_wrist_state
        if wrist_state is None:
            raise RuntimeError("缺少手腕状态快照")
        if not wrist_state.active or not wrist_state.zero_valid or wrist_state.fault:
            raise RuntimeError("手腕 YAML 舵机零位无效或 J1/J2 状态不健康")
        if not self.wrist.healthy():
            raise RuntimeError("手腕控制器不健康")
        if not self.gripper.healthy():
            raise RuntimeError("夹爪控制器不健康")
        wrist_target = action[6:8]
        bounds_min = np.asarray(self.config.wrist.relative_min_raw, dtype=np.float64)
        bounds_max = np.asarray(self.config.wrist.relative_max_raw, dtype=np.float64)
        if np.any(wrist_target < bounds_min) or np.any(wrist_target > bounds_max):
            raise ValueError("手腕 action 超出 YAML 软件限位")
        if validate_wrist_step:
            if self._last_wrist_sent_target is None:
                raise RuntimeError("缺少手腕实际最后发送目标")
            last_target = _finite_vector(self._last_wrist_sent_target, 2, "手腕实际最后发送目标")
            if np.any(np.abs(wrist_target - last_target) > self.config.wrist.max_step_raw + 1e-12):
                raise ValueError("手腕 action 单步超过 YAML max_step_raw")
        if not 0.0 <= action[8] <= 1.0:
            raise ValueError("夹爪 action 必须位于 [0, 1]")

    @staticmethod
    def _auxiliary_due(last_command_ns: int | None, rate_hz: float, now_ns: int) -> bool:
        if last_command_ns is None:
            return True
        elapsed_ns = now_ns - last_command_ns
        if elapsed_ns < 0:
            raise RuntimeError("单调时钟倒退，禁止继续控制")
        return elapsed_ns >= math.ceil(1e9 / rate_hz)

    def _require_maintenance_ready(self) -> None:
        self._require_ready(allow_stopped=False)
        if not self.config.enable_motion:
            raise PermissionError("UR5 维护运动未启用；必须显式 --enable-motion")
        if self._episode_active:
            raise RuntimeError("训练 episode 活跃时禁止 HOME/J6 speedJ")

    def _hardware_health(
        self,
        ur_state: Any,
        wrist_state: Any,
        gripper_state: Any,
        frames: Mapping[str, CameraFrame],
    ) -> dict[str, Any]:
        ur_ok = bool(ur_state.healthy)
        wrist_ok = bool(
            self.wrist.healthy()
            and not wrist_state.fault
            and math.isfinite(float(wrist_state.source_age_s))
            and float(wrist_state.source_age_s) <= self.config.wrist.state_max_age_s
        )
        gripper_ok = bool(
            self.gripper.healthy()
            and math.isfinite(float(gripper_state.position))
            and 0.0 <= float(gripper_state.position) <= 1.0
        )
        host_times = [int(frames[role].host_timestamp_ns) for role in EXPECTED_CAMERA_ROLES]
        camera_skew_ms = (max(host_times) - min(host_times)) / 1_000_000.0
        cameras_ok = bool(camera_skew_ms <= self.config.cameras.max_camera_skew_ms)
        return {
            "ok": bool(ur_ok and wrist_ok and gripper_ok and cameras_ok),
            "emergency_stop": bool(ur_state.emergency_stopped),
            "protective_stop": bool(ur_state.protective_stopped),
            "motion_enabled": self.config.enable_motion,
            "motion_ready": bool(
                self.config.enable_motion and wrist_state.active and wrist_state.zero_valid and not wrist_state.fault
            ),
            "ur5": {"ok": ur_ok, "robot_mode": ur_state.robot_mode, "safety_mode": ur_state.safety_mode},
            "wrist": {
                "ok": wrist_ok,
                "active": bool(wrist_state.active),
                "zero_valid": bool(wrist_state.zero_valid),
                "sequence": int(wrist_state.sequence),
                "coordinate": "yaml_servo_zero_relative_raw",
                "servo_zero_raw": tuple(int(value) for value in wrist_state.servo_zero_raw),
                "hardware_limits_raw": wrist_state.hardware_limits_raw,
                "motor_position_raw": tuple(int(value) for value in wrist_state.motor_position_raw),
                "motor_goal_raw": tuple(int(value) for value in wrist_state.motor_goal_raw),
                "encoder_abs_deg": tuple(float(value) for value in wrist_state.encoder_abs_deg),
                "encoder_valid": tuple(bool(value) for value in wrist_state.encoder_valid),
            },
            "gripper": {
                "ok": gripper_ok,
                "backend": str(gripper_state.backend),
                "sequence": int(gripper_state.sequence),
            },
            "cameras": {
                "ok": cameras_ok,
                "max_skew_ms": camera_skew_ms,
                "serials": {role: frames[role].serial for role in EXPECTED_CAMERA_ROLES},
            },
            "fault": None,
        }

    @staticmethod
    def _validate_frame_roles(frames: Mapping[str, CameraFrame]) -> None:
        if set(frames) != set(EXPECTED_CAMERA_ROLES):
            raise RuntimeError(f"相机帧角色必须是 {EXPECTED_CAMERA_ROLES}, 实际 {tuple(frames)}")
        for role in EXPECTED_CAMERA_ROLES:
            frame = frames[role]
            if frame.role != role:
                raise RuntimeError(f"相机 role 错配: key={role}, frame.role={frame.role}")

    def _require_ready(self, *, allow_stopped: bool) -> None:
        self._assert_owner_if_started()
        if self._fault is not None:
            raise RuntimeError(f"统一硬件已 fail-closed: {self._fault}") from self._fault
        if not self._connected:
            raise RuntimeError("统一硬件尚未连接")
        if self._stopped and not allow_stopped:
            raise RuntimeError("统一硬件已经停止")

    def _assert_owner_if_started(self) -> None:
        if self._owner_thread_id is not None and self._owner_thread_id != threading.get_ident():
            raise RuntimeError("只有 connect 所在线程可以访问统一硬件")

    def _stop_components(self) -> list[BaseException]:
        errors: list[BaseException] = []
        for name in ("ur5", "wrist", "gripper", "cameras"):
            if name not in self._started_components:
                continue
            if name != "cameras" and not self._actuator_stop_allowed:
                continue
            try:
                getattr(self, name).stop()
            except BaseException as exc:
                errors.append(exc)
        return errors

    def _trip(self, failure: BaseException, *, close: bool = False) -> None:
        self._fault = failure
        self._stopped = True
        self._episode_active = False
        self._maintenance_motion_active = False
        self._home_reached_since_ns = None
        self._stop_components()
        if close:
            for name in reversed(self._started_components):
                with suppress(Exception):
                    getattr(self, name).close()
            self._started_components.clear()
            self._connected = False
            self._owner_thread_id = None


RobotHardware = LocalRobotHardware

__all__ = [
    "ACTION_DIMENSION",
    "CommandReceipt",
    "HardwareFactories",
    "LocalHardwareConfig",
    "LocalRobotHardware",
    "MaintenanceReceipt",
    "RobotHardware",
]
