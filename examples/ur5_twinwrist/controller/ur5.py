# ruff: noqa: RUF001, RUF002, RUF003
"""项目内 UR5 RTDE 基础控制层。

这个模块只定义配置、状态校验和显式控制方法。导入模块不会导入
``ur_rtde``、不会连接机器人，也不会发送运动；调用 :meth:`connect` 后才
连接 RTDE Receive。``enable_motion=True`` 时会在 ``connect`` 内预连接
RTDE Control、随后重新读取状态，但只有显式调用 :meth:`send` 才发送
``speedL``。

实现语义来自已验证旧工程的：

* ``src/slai_mi/devices/ur5/runtime.py``；
* ``src/slai_mi/devices/ur5/worker.py``；
* ``src/slai_mi/site_adapter.py::_apply_legacy_ur5_command``。

旧工程只作为只读参考，本模块不导入它。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import importlib
import math
import time
from typing import Any

import numpy as np

RUNNING_ROBOT_MODE = 7
NORMAL_SAFETY_MODE = 1


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数")
    return result


def _optional_vector(values: Any, size: int, name: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须包含 {size} 个有限数")
    return tuple(float(value) for value in result)


def _vector6(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须包含 6 个有限数")
    return result


def joint_home_velocity(
    current_joints: Sequence[float] | np.ndarray,
    target_joints: Sequence[float] | np.ndarray,
    max_speed_rad_s: float,
    *,
    tolerance_rad: float = 0.005,
    proportional_gain: float = 1.5,
) -> tuple[np.ndarray, bool]:
    """生成旧工程同语义的同步限速 HOME 关节速度。"""

    current = _vector6(current_joints, "UR5 当前关节")
    target = _vector6(target_joints, "UR5 task home")
    for name, value in (
        ("max_speed_rad_s", max_speed_rad_s),
        ("tolerance_rad", tolerance_rad),
        ("proportional_gain", proportional_gain),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须是正有限数")
    error = target - current
    if float(np.abs(error).max()) <= tolerance_rad:
        return np.zeros(6, dtype=np.float64), True
    velocity = error * proportional_gain
    peak = float(np.abs(velocity).max())
    if peak > max_speed_rad_s:
        velocity *= max_speed_rad_s / peak
    return velocity, False


@dataclass(frozen=True)
class UR5Config:
    """UR5 连接与软件安全参数。

    ``host`` 没有现场默认值，必须由 YAML/TOML 或环境配置传入。
    ``enable_motion`` 默认关闭；它是控制层内的第二道运动许可，不替代
    上层命令行的 ``--enable-motion`` 确认。
    """

    host: str
    enable_motion: bool = False
    acceleration: float = 0.50
    command_duration_s: float = 0.008
    stop_deceleration: float = 1.0
    max_state_age_s: float = 0.10
    max_linear_speed_m_s: float = 0.25
    max_angular_speed_rad_s: float = 0.60
    max_joint_speed_rad_s: float = 0.50
    max_joint_step_rad: float = 0.02
    prediction_horizon_s: float = 0.25
    min_tcp_z_m: float | None = None
    joint_soft_lower_rad: tuple[float, ...] | None = None
    joint_soft_upper_rad: tuple[float, ...] | None = None
    workspace_min_xyz_m: tuple[float, ...] | None = None
    workspace_max_xyz_m: tuple[float, ...] | None = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        enable_motion: bool | None = None,
    ) -> UR5Config:
        """从完整配置或其中的 ``ur5`` 段构造配置。

        可直接传入 ``config_loader.load_project_config`` 的结果。该情况下会
        合并 ``hardware.ur5``、``safety.ur5`` 和 ``safety.timing``。运动许可
        建议只由命令行传入 ``enable_motion``，不要永久写进 YAML。
        """

        source = cls._flatten_mapping(values)
        if not isinstance(source, Mapping):
            raise TypeError("ur5 配置必须是 mapping")
        control_hz = _finite_float(source.get("control_hz", 125.0), "control_hz")
        duration_default = 1.0 / control_hz
        age_default = (
            _finite_float(source["max_state_age_ms"], "max_state_age_ms") / 1000.0
            if "max_state_age_ms" in source
            else 0.10
        )
        config = cls(
            host=str(source.get("host", "")),
            enable_motion=(bool(source.get("enable_motion", False)) if enable_motion is None else bool(enable_motion)),
            acceleration=_finite_float(source.get("acceleration", 0.50), "acceleration"),
            command_duration_s=_finite_float(source.get("command_duration_s", duration_default), "command_duration_s"),
            stop_deceleration=_finite_float(source.get("stop_deceleration", 1.0), "stop_deceleration"),
            max_state_age_s=_finite_float(source.get("max_state_age_s", age_default), "max_state_age_s"),
            max_linear_speed_m_s=_finite_float(
                source.get("max_linear_speed_m_s", source.get("max_linear_m_s", 0.25)),
                "max_linear_speed_m_s",
            ),
            max_angular_speed_rad_s=_finite_float(
                source.get("max_angular_speed_rad_s", source.get("max_angular_rad_s", 0.60)),
                "max_angular_speed_rad_s",
            ),
            max_joint_speed_rad_s=_finite_float(source.get("max_joint_speed_rad_s", 0.50), "max_joint_speed_rad_s"),
            max_joint_step_rad=_finite_float(source.get("max_joint_step_rad", 0.02), "max_joint_step_rad"),
            prediction_horizon_s=_finite_float(source.get("prediction_horizon_s", 0.25), "prediction_horizon_s"),
            min_tcp_z_m=(
                None if source.get("min_tcp_z_m") is None else _finite_float(source["min_tcp_z_m"], "min_tcp_z_m")
            ),
            joint_soft_lower_rad=_optional_vector(
                source.get("joint_soft_lower_rad", source.get("joint_min_rad")),
                6,
                "joint_soft_lower_rad",
            ),
            joint_soft_upper_rad=_optional_vector(
                source.get("joint_soft_upper_rad", source.get("joint_max_rad")),
                6,
                "joint_soft_upper_rad",
            ),
            workspace_min_xyz_m=_optional_vector(
                source.get("workspace_min_xyz_m", source.get("workspace_min_m")),
                3,
                "workspace_min_xyz_m",
            ),
            workspace_max_xyz_m=_optional_vector(
                source.get("workspace_max_xyz_m", source.get("workspace_max_m")),
                3,
                "workspace_max_xyz_m",
            ),
        )
        config.validate()
        return config

    @staticmethod
    def _flatten_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
        if "hardware" not in values and "safety" not in values:
            return values.get("ur5", values)
        hardware = values.get("hardware", {})
        safety = values.get("safety", {})
        if not isinstance(hardware, Mapping) or not isinstance(safety, Mapping):
            raise TypeError("hardware/safety 配置必须是 mapping")
        hardware_ur5 = hardware.get("ur5", {})
        safety_ur5 = safety.get("ur5", {})
        timing = safety.get("timing", {})
        if not all(isinstance(item, Mapping) for item in (hardware_ur5, safety_ur5, timing)):
            raise TypeError("hardware.ur5、safety.ur5 和 safety.timing 必须是 mapping")
        return {**hardware_ur5, **safety_ur5, **timing}

    def validate(self) -> None:
        if not self.host.strip():
            raise ValueError("UR5 host 不能为空")
        for name, value, lower, upper in (
            ("acceleration", self.acceleration, 0.0, 1.0),
            ("command_duration_s", self.command_duration_s, 0.0, 0.1),
            ("stop_deceleration", self.stop_deceleration, 0.0, 2.0),
            ("max_state_age_s", self.max_state_age_s, 0.0, 1.0),
            ("max_linear_speed_m_s", self.max_linear_speed_m_s, 0.0, 0.25),
            ("max_angular_speed_rad_s", self.max_angular_speed_rad_s, 0.0, 1.0),
            ("max_joint_speed_rad_s", self.max_joint_speed_rad_s, 0.0, 1.0),
            ("max_joint_step_rad", self.max_joint_step_rad, 0.0, 0.5),
            ("prediction_horizon_s", self.prediction_horizon_s, 0.0, 1.0),
        ):
            if not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"{name} 必须位于 ({lower}, {upper}] 内")
        self._validate_pair(
            self.joint_soft_lower_rad,
            self.joint_soft_upper_rad,
            6,
            "joint_soft",
        )
        self._validate_pair(
            self.workspace_min_xyz_m,
            self.workspace_max_xyz_m,
            3,
            "workspace_xyz",
        )
        if self.enable_motion:
            if self.joint_soft_lower_rad is None or self.joint_soft_upper_rad is None:
                raise ValueError("UR5 真实运动必须配置 6 维关节软件软限位")
            if self.workspace_min_xyz_m is None or self.workspace_max_xyz_m is None:
                raise ValueError("UR5 真实运动必须配置 base-frame 工作空间边界")
        if self.min_tcp_z_m is not None:
            if not math.isfinite(self.min_tcp_z_m):
                raise ValueError("min_tcp_z_m 必须是有限数")
            if self.workspace_max_xyz_m is not None and self.min_tcp_z_m > self.workspace_max_xyz_m[2]:
                raise ValueError("min_tcp_z_m 不得高于工作空间 z 上界")

    @staticmethod
    def _validate_pair(
        lower: tuple[float, ...] | None,
        upper: tuple[float, ...] | None,
        size: int,
        name: str,
    ) -> None:
        if (lower is None) != (upper is None):
            raise ValueError(f"{name} 上下限必须同时设置")
        if lower is None or upper is None:
            return
        low = _vector6(lower, name) if size == 6 else np.asarray(lower, dtype=np.float64)
        high = _vector6(upper, name) if size == 6 else np.asarray(upper, dtype=np.float64)
        if low.shape != (size,) or high.shape != (size,):
            raise ValueError(f"{name} 上下限必须各有 {size} 维")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError(f"{name} 每个下限必须小于对应上限")


@dataclass(frozen=True)
class UR5State:
    """同一次 RTDE 读取形成的 UR5 状态快照。"""

    qpos_rad: np.ndarray
    tcp_pose: np.ndarray
    tcp_speed: np.ndarray
    monotonic_ns: int
    robot_mode: int
    safety_mode: int
    emergency_stopped: bool
    protective_stopped: bool

    @property
    def healthy(self) -> bool:
        return bool(
            not self.emergency_stopped
            and not self.protective_stopped
            and self.robot_mode == RUNNING_ROBOT_MODE
            and self.safety_mode == NORMAL_SAFETY_MODE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "qpos": self.qpos_rad.copy(),
            "tcp_pose": self.tcp_pose.copy(),
            "tcp_speed": self.tcp_speed.copy(),
            "monotonic_ns": self.monotonic_ns,
            "robot_mode": self.robot_mode,
            "safety_mode": self.safety_mode,
            "emergency_stopped": self.emergency_stopped,
            "protective_stopped": self.protective_stopped,
            "healthy": self.healthy,
        }


@dataclass(frozen=True)
class JointVelocityResult:
    """一次非训练 ``speedJ`` 命令的结果。"""

    velocity_rad_s: np.ndarray
    state_timestamp_ns: int
    reached: bool | None = None


class UR5Controller:
    """单进程 UR5 ``speedL`` 控制器，异常后保持 fail-closed。

    测试可注入 ``receiver_factory`` 和 ``control_factory``。生产环境不传时，
        两个 ``ur_rtde`` 模块仅在调用 ``connect`` 时延迟导入。
    """

    def __init__(
        self,
        config: UR5Config | Mapping[str, Any],
        *,
        receiver_factory: Callable[[str], Any] | None = None,
        control_factory: Callable[[str], Any] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config if isinstance(config, UR5Config) else UR5Config.from_mapping(config)
        self.config.validate()
        self._receiver_factory = receiver_factory
        self._control_factory = control_factory
        self._clock_ns = monotonic_ns
        self._receiver: Any | None = None
        self._control: Any | None = None
        self._last_state: UR5State | None = None
        self._fault: BaseException | None = None

    @property
    def connected(self) -> bool:
        return self._receiver is not None

    @property
    def motion_enabled(self) -> bool:
        return self.config.enable_motion

    def connect(self) -> None:
        """连接 RTDE；不会发送运动或自动回零。

        未启用运动时只连接 Receive。启用运动时也预连接 Control，再读取
        一次新状态，避免首次 Control 握手耗时使 100 ms 状态门禁永久失败。
        """

        if self.connected:
            raise RuntimeError("UR5 已连接")
        if self._receiver_factory is None:
            module = importlib.import_module("rtde_receive")
            self._receiver_factory = module.RTDEReceiveInterface
        receiver = self._receiver_factory(self.config.host)
        try:
            if hasattr(receiver, "isConnected") and not receiver.isConnected():
                raise ConnectionError(f"RTDE Receive 无法连接 {self.config.host}")
            self._receiver = receiver
            state = self.read()
            if self.config.enable_motion:
                self._ensure_control(state)
                # RTDEControl 构造可能超过 max_state_age；以握手完成后的实际
                # 状态作为首个可发送快照，不能放宽状态超时门禁。
                self.read()
        except BaseException:
            control = self._control
            self._control = None
            if control is not None:
                with suppress(Exception):
                    control.stopScript()
                with suppress(Exception):
                    control.disconnect()
            with suppress(Exception):
                receiver.disconnect()
            self._receiver = None
            self._last_state = None
            raise

    def read(self) -> UR5State:
        """读取并验证一个状态快照；急停、保护停或非法值立即报错。"""

        receiver = self._require_receiver()
        try:
            state = UR5State(
                qpos_rad=_vector6(receiver.getActualQ(), "UR5 实际关节"),
                tcp_pose=_vector6(receiver.getActualTCPPose(), "UR5 TCP 位姿"),
                tcp_speed=_vector6(receiver.getActualTCPSpeed(), "UR5 TCP 速度"),
                monotonic_ns=int(self._clock_ns()),
                robot_mode=int(receiver.getRobotMode()),
                safety_mode=int(receiver.getSafetyMode()),
                emergency_stopped=bool(receiver.isEmergencyStopped()),
                protective_stopped=bool(receiver.isProtectiveStopped()),
            )
            if not state.healthy:
                raise RuntimeError(
                    "UR5 健康检查失败: "
                    f"emergency={state.emergency_stopped}, "
                    f"protective={state.protective_stopped}, "
                    f"robot_mode={state.robot_mode}, safety_mode={state.safety_mode}"
                )
            self._validate_state_envelope(state)
            self._last_state = state
            return state
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def read_state(self) -> UR5State:
        """``read`` 的语义化别名。"""

        return self.read()

    def send(
        self,
        twist: Sequence[float] | np.ndarray,
        *,
        state_timestamp_ns: int | None = None,
    ) -> np.ndarray:
        """显式发送一个 base-frame ``speedL`` 命令。

        六维依次为 ``vx, vy, vz [m/s], wx, wy, wz [rad/s]``。返回实际发送
        的副本，便于记录最终 action。任何非法、过期或越界命令都会先停止
        已存在的控制连接，再抛出异常。
        """

        try:
            if not self.config.enable_motion:
                raise PermissionError("UR5 运动未启用；上层必须显式提供 --enable-motion")
            if self._fault is not None:
                raise RuntimeError(f"UR5 控制器已故障闭锁: {self._fault}") from self._fault
            command = _vector6(twist, "UR5 speedL 命令")
            self._validate_speed(command)
            state = self._require_fresh_state(state_timestamp_ns)
            self._validate_projected_workspace(state, command)
            control = self._ensure_control(state)
            # 首次创建 RTDEControl 可能阻塞；连接完成后必须重新检查同一状态
            # 是否仍新鲜，不能拿连接前的旧快照直接开始运动。
            self._require_fresh_state(state.monotonic_ns)
            result = control.speedL(
                command.tolist(),
                self.config.acceleration,
                self.config.command_duration_s,
            )
            if result is False:
                raise RuntimeError("UR5 speedL 返回 false")
            return command.copy()
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def send_action(
        self,
        action: Sequence[float] | np.ndarray,
        *,
        state_timestamp_ns: int | None = None,
    ) -> np.ndarray:
        """``send`` 的 pipeline 接口别名。"""

        return self.send(action, state_timestamp_ns=state_timestamp_ns)

    def send_joint_velocity(
        self,
        velocity_rad_s: Sequence[float] | np.ndarray,
        *,
        state_timestamp_ns: int | None = None,
    ) -> JointVelocityResult:
        """显式发送维护模式 ``speedJ``，不得作为训练 action 记录。

        调用签名与已验证旧 worker 一致：
        ``speedJ(qd, acceleration, command_duration_s)``。命令会检查最新实际
        关节、软件软限位、预测位置和 UR 控制器自身安全限位。
        """

        try:
            if not self.config.enable_motion:
                raise PermissionError("UR5 运动未启用；上层必须显式提供 --enable-motion")
            if self._fault is not None:
                raise RuntimeError(f"UR5 控制器已故障闭锁: {self._fault}") from self._fault
            velocity = _vector6(velocity_rad_s, "UR5 speedJ 命令")
            if float(np.abs(velocity).max()) > self.config.max_joint_speed_rad_s + 1e-12:
                raise ValueError(
                    "UR5 关节速度超过 YAML 限制: "
                    f"{float(np.abs(velocity).max()):.6f} > "
                    f"{self.config.max_joint_speed_rad_s:.6f} rad/s"
                )
            command_step = float(np.abs(velocity).max()) * self.config.command_duration_s
            if command_step > self.config.max_joint_step_rad + 1e-12:
                raise ValueError(
                    "UR5 speedJ 单命令关节变化超过 YAML 限制: "
                    f"{command_step:.6f} > {self.config.max_joint_step_rad:.6f} rad"
                )
            state = self._require_fresh_state(state_timestamp_ns)
            horizon_s = max(self.config.prediction_horizon_s, 2.0 * self.config.command_duration_s)
            projected = state.qpos_rad + velocity * horizon_s
            self._validate_joint_target(projected)
            control = self._ensure_control(state)
            self._require_fresh_state(state.monotonic_ns)
            if hasattr(control, "isJointsWithinSafetyLimits") and not control.isJointsWithinSafetyLimits(
                projected.tolist()
            ):
                raise RuntimeError("UR5 speedJ 预测关节不在控制器安全范围内")
            result = control.speedJ(
                velocity.tolist(),
                self.config.acceleration,
                self.config.command_duration_s,
            )
            if result is False:
                raise RuntimeError("UR5 speedJ 返回 false")
            return JointVelocityResult(
                velocity_rad_s=velocity.copy(),
                state_timestamp_ns=state.monotonic_ns,
            )
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def jog_joint6(
        self,
        direction: int,
        *,
        speed_rad_s: float,
        state_timestamp_ns: int | None = None,
    ) -> JointVelocityResult:
        """发送一次 J6 jog：``-1``=键1顺时针，``+1``=键2逆时针。"""

        try:
            if isinstance(direction, bool) or direction not in {-1, 0, 1}:
                raise ValueError("J6 jog direction 必须是 -1、0 或 +1")
            speed = float(speed_rad_s)
            if not math.isfinite(speed) or speed <= 0.0:
                raise ValueError("J6 jog speed_rad_s 必须是正有限数")
            velocity = np.zeros(6, dtype=np.float64)
            velocity[5] = direction * min(speed, self.config.max_joint_speed_rad_s)
            return self.send_joint_velocity(velocity, state_timestamp_ns=state_timestamp_ns)
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def task_home_step(
        self,
        target_joints_rad: Sequence[float] | np.ndarray,
        *,
        max_speed_rad_s: float,
        tolerance_rad: float,
        proportional_gain: float = 1.5,
        state_timestamp_ns: int | None = None,
    ) -> JointVelocityResult:
        """朝 task home 执行一个 125 Hz ``speedJ`` 控制步。

        调用方应仅在 T/HOME 按住且 episode 未录制时重复调用；松开按键后
        调用 ``stop``。此方法不使用 ``moveJ``，因为旧工程真实闭环采用可随
        按键释放立即停止的逐步 ``speedJ``。
        """

        try:
            if not self.config.enable_motion:
                raise PermissionError("UR5 运动未启用；上层必须显式提供 --enable-motion")
            if self._fault is not None:
                raise RuntimeError(f"UR5 控制器已故障闭锁: {self._fault}") from self._fault
            state = self._require_fresh_state(state_timestamp_ns)
            target = _vector6(target_joints_rad, "UR5 task home")
            self._validate_joint_target(target)
            control = self._ensure_control(state)
            if hasattr(control, "isJointsWithinSafetyLimits") and not control.isJointsWithinSafetyLimits(
                target.tolist()
            ):
                raise RuntimeError("UR5 task home 不在控制器安全范围内")
            velocity, reached = joint_home_velocity(
                state.qpos_rad,
                target,
                min(float(max_speed_rad_s), self.config.max_joint_speed_rad_s),
                tolerance_rad=float(tolerance_rad),
                proportional_gain=float(proportional_gain),
            )
            sent = self.send_joint_velocity(
                velocity,
                state_timestamp_ns=state.monotonic_ns,
            )
            return JointVelocityResult(
                velocity_rad_s=sent.velocity_rad_s,
                state_timestamp_ns=sent.state_timestamp_ns,
                reached=reached,
            )
        except BaseException as exc:
            # send_joint_velocity 已经闭锁；纯计算/目标检查异常也必须闭锁。
            self._fail_closed(exc)
            raise

    def stop(self) -> None:
        """尽力停止当前速度命令；未连接控制通道时为空操作。"""

        control = self._control
        if control is None:
            return
        with suppress(Exception):
            control.speedL([0.0] * 6, self.config.stop_deceleration, 0.02)
        with suppress(Exception):
            control.speedStop(self.config.stop_deceleration)
        with suppress(Exception):
            control.stopL(self.config.stop_deceleration)

    def close(self) -> None:
        """停止并关闭 RTDE；可重复调用。"""

        self.stop()
        control, receiver = self._control, self._receiver
        self._control = None
        self._receiver = None
        self._last_state = None
        if control is not None:
            with suppress(Exception):
                control.stopScript()
            with suppress(Exception):
                control.disconnect()
        if receiver is not None:
            with suppress(Exception):
                receiver.disconnect()

    def healthy(self) -> bool:
        state = self._last_state
        if self._fault is not None or not self.connected or state is None or not state.healthy:
            return False
        age_ns = int(self._clock_ns()) - state.monotonic_ns
        return 0 <= age_ns <= int(self.config.max_state_age_s * 1e9)

    def __enter__(self) -> UR5Controller:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _require_receiver(self) -> Any:
        if self._receiver is None:
            raise RuntimeError("UR5 尚未连接")
        return self._receiver

    def _require_fresh_state(self, timestamp_ns: int | None) -> UR5State:
        self._require_receiver()
        state = self._last_state
        if state is None:
            raise RuntimeError("发送动作前必须先读取 UR5 状态")
        expected = state.monotonic_ns if timestamp_ns is None else int(timestamp_ns)
        if expected != state.monotonic_ns:
            raise RuntimeError("动作引用的 UR5 状态不是当前最新快照")
        age_ns = int(self._clock_ns()) - expected
        max_age_ns = int(self.config.max_state_age_s * 1e9)
        if age_ns < 0 or age_ns > max_age_ns:
            raise TimeoutError(f"UR5 状态过期: {age_ns / 1e6:.1f} ms > {max_age_ns / 1e6:.1f} ms")
        if not state.healthy:
            raise RuntimeError("UR5 状态不健康")
        return state

    def _ensure_control(self, state: UR5State) -> Any:
        if self._control is None:
            if self._control_factory is None:
                module = importlib.import_module("rtde_control")
                self._control_factory = module.RTDEControlInterface
            control = self._control_factory(self.config.host)
            if hasattr(control, "isConnected") and not control.isConnected():
                with suppress(Exception):
                    control.disconnect()
                raise ConnectionError(f"RTDE Control 无法连接 {self.config.host}")
            self._control = control
        control = self._control
        if hasattr(control, "isProgramRunning") and not control.isProgramRunning():
            raise RuntimeError("UR5 RTDE control program 未运行")
        if hasattr(control, "isPoseWithinSafetyLimits") and not control.isPoseWithinSafetyLimits(
            state.tcp_pose.tolist()
        ):
            raise RuntimeError("当前 UR5 TCP 位姿不在控制器安全范围内")
        return control

    def _validate_state_envelope(self, state: UR5State) -> None:
        lower = self.config.joint_soft_lower_rad
        upper = self.config.joint_soft_upper_rad
        if (
            lower is not None
            and upper is not None
            and (np.any(state.qpos_rad < np.asarray(lower)) or np.any(state.qpos_rad > np.asarray(upper)))
        ):
            raise RuntimeError("UR5 实际关节超出软件软限位")
        xyz_min = self.config.workspace_min_xyz_m
        xyz_max = self.config.workspace_max_xyz_m
        if xyz_min is not None and xyz_max is not None:
            position = state.tcp_pose[:3]
            effective_min = np.asarray(xyz_min).copy()
            if self.config.min_tcp_z_m is not None:
                effective_min[2] = max(effective_min[2], self.config.min_tcp_z_m)
            if np.any(position < effective_min) or np.any(position > np.asarray(xyz_max)):
                raise RuntimeError("UR5 TCP 超出软件工作空间")

    def _validate_joint_target(self, target: Sequence[float] | np.ndarray) -> None:
        joints = _vector6(target, "UR5 预测关节")
        lower = self.config.joint_soft_lower_rad
        upper = self.config.joint_soft_upper_rad
        if lower is None or upper is None:
            raise RuntimeError("未配置 UR5 关节软限位，禁止 speedJ/task home")
        if np.any(joints < np.asarray(lower)) or np.any(joints > np.asarray(upper)):
            raise RuntimeError("UR5 speedJ/task home 将越出软件关节软限位")

    def _validate_speed(self, command: np.ndarray) -> None:
        linear = float(np.linalg.norm(command[:3]))
        angular = float(np.linalg.norm(command[3:]))
        if linear > self.config.max_linear_speed_m_s + 1e-12:
            raise ValueError(f"UR5 线速度范数 {linear:.6f} 超过 {self.config.max_linear_speed_m_s:.6f} m/s")
        if angular > self.config.max_angular_speed_rad_s + 1e-12:
            raise ValueError(f"UR5 角速度范数 {angular:.6f} 超过 {self.config.max_angular_speed_rad_s:.6f} rad/s")

    def _validate_projected_workspace(self, state: UR5State, command: np.ndarray) -> None:
        xyz_min = self.config.workspace_min_xyz_m
        xyz_max = self.config.workspace_max_xyz_m
        if xyz_min is None or xyz_max is None:
            return
        effective_min = np.asarray(xyz_min).copy()
        if self.config.min_tcp_z_m is not None:
            effective_min[2] = max(effective_min[2], self.config.min_tcp_z_m)
        projected = state.tcp_pose[:3] + command[:3] * self.config.prediction_horizon_s
        if np.any(projected < effective_min) or np.any(projected > np.asarray(xyz_max)):
            raise RuntimeError("UR5 speedL 预测位置将越出软件工作空间")

    def _fail_closed(self, exc: BaseException) -> None:
        self._fault = exc
        self.stop()


__all__ = [
    "NORMAL_SAFETY_MODE",
    "RUNNING_ROBOT_MODE",
    "JointVelocityResult",
    "UR5Config",
    "UR5Controller",
    "UR5State",
    "joint_home_velocity",
]
