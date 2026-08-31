# ruff: noqa: RUF001, RUF002, RUF003
"""项目内 OpenRB 两自由度手腕控制器。

本文件只保留真机已经使用过的 OpenRB 文本协议，不依赖旧工程的 Python
包。导入本模块不会导入 ``serial``，也不会打开串口；只有显式调用
``connect()`` 才会延迟导入 pyserial 并建立连接。

训练与推理公开坐标统一使用 ``[J1, J2]`` 舵机 raw 相对值：实际/目标
Dynamixel raw 减去 ``poses.yaml`` 中的 ``servo_zero_raw``。AS5048A 末端
绝对编码器只保留为诊断观测，不参与 HOME、训练 state 或推理 action。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
import threading
import time
from typing import Any


class WristError(RuntimeError):
    """手腕连接、协议或安全检查失败。"""


class WristTimeoutError(WristError, TimeoutError):
    """OpenRB 未在规定时间内返回完整应答。"""


class DeviceOwnershipError(WristError):
    """串口 I/O 被连接线程以外的线程调用。"""


def _require_by_id(port: str) -> str:
    """拒绝会随 USB 枚举顺序变化的 ttyUSB/ttyACM 路径。"""
    candidate = Path(port)
    prefix = Path("/dev/serial/by-id")
    if candidate.parent != prefix or not candidate.name or candidate.name in {".", ".."}:
        raise ValueError("手腕串口必须使用 /dev/serial/by-id/<设备名>")
    return str(candidate)


def _as_int(mapping: Mapping[str, Any], name: str, default: int) -> int:
    return int(mapping.get(name, default))


def _as_float(mapping: Mapping[str, Any], name: str, default: float) -> float:
    return float(mapping.get(name, default))


@dataclass(frozen=True)
class WristHardwareLimits:
    """OpenRB ``GET_LIMITS`` 返回的 J1/J2 舵机 raw 边界。"""

    j1_min_raw: int
    j1_max_raw: int
    j1_firmware_zero_raw: int
    j2_min_raw: int
    j2_max_raw: int
    j2_firmware_zero_raw: int

    @classmethod
    def from_fields(cls, fields: Mapping[str, str]) -> WristHardwareLimits:
        names = ("j1_min", "j1_max", "j1_zero", "j2_min", "j2_max", "j2_zero")
        try:
            values = [int(fields[name]) for name in names]
        except (KeyError, ValueError) as exc:
            raise WristError("GET_LIMITS 缺少合法 J1/J2 raw 字段") from exc
        result = cls(*values)
        result.validate()
        return result

    def validate(self) -> None:
        values = (
            self.j1_min_raw,
            self.j1_max_raw,
            self.j1_firmware_zero_raw,
            self.j2_min_raw,
            self.j2_max_raw,
            self.j2_firmware_zero_raw,
        )
        if any(not 0 <= value <= 4095 for value in values):
            raise WristError("OpenRB GET_LIMITS raw 值必须在 [0,4095]")
        if not self.j1_min_raw < self.j1_max_raw or not self.j2_min_raw < self.j2_max_raw:
            raise WristError("OpenRB GET_LIMITS 上下界非法")
        if not self.j1_min_raw <= self.j1_firmware_zero_raw <= self.j1_max_raw:
            raise WristError("OpenRB 固件 J1 zero 超出限位")
        if not self.j2_min_raw <= self.j2_firmware_zero_raw <= self.j2_max_raw:
            raise WristError("OpenRB 固件 J2 zero 超出限位")


@dataclass(frozen=True)
class WristConfig:
    """两轴腕的 YAML 舵机 raw 坐标与安全参数。"""

    port: str
    servo_zero_raw: tuple[int, int]
    baudrate: int = 115_200
    timeout_s: float = 0.20
    state_max_age_s: float = 0.20
    j1_min_raw: int = 2781
    j1_max_raw: int = 3568
    j2_min_raw: int = 1822
    j2_max_raw: int = 3333
    max_step_raw: int = 18
    start_tolerance_raw: int = 30
    settle_tolerance_raw: int = 30
    settle_samples: int = 3
    settle_timeout_s: float = 18.0
    poll_period_s: float = 0.05
    home_timeout_s: float = 30.0
    profile_velocity_raw: int = 50
    profile_acceleration_raw: int = 15

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _require_by_id(self.port))
        if self.baudrate <= 0:
            raise ValueError("baudrate 必须为正数")
        if len(self.servo_zero_raw) != 2 or any(not 0 <= int(value) <= 4095 for value in self.servo_zero_raw):
            raise ValueError("servo_zero_raw 必须是两个 [0,4095] 整数")
        for name in ("timeout_s", "state_max_age_s", "poll_period_s", "home_timeout_s", "settle_timeout_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
        if not 0 <= self.j1_min_raw < self.j1_max_raw <= 4095:
            raise ValueError("J1 YAML raw 限位非法")
        if not 0 <= self.j2_min_raw < self.j2_max_raw <= 4095:
            raise ValueError("J2 YAML raw 限位非法")
        if not self.j1_min_raw <= self.servo_zero_raw[0] <= self.j1_max_raw:
            raise ValueError("YAML J1 servo zero 超出 YAML 限位")
        if not self.j2_min_raw <= self.servo_zero_raw[1] <= self.j2_max_raw:
            raise ValueError("YAML J2 servo zero 超出 YAML 限位")
        for name in ("max_step_raw", "start_tolerance_raw", "settle_tolerance_raw", "settle_samples"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("profile_velocity_raw", "profile_acceleration_raw"):
            if not 0 <= int(getattr(self, name)) <= 32767:
                raise ValueError(f"{name} 必须在 [0,32767]")

    @property
    def relative_min_raw(self) -> tuple[float, float]:
        return (
            float(self.j1_min_raw - self.servo_zero_raw[0]),
            float(self.j2_min_raw - self.servo_zero_raw[1]),
        )

    @property
    def relative_max_raw(self) -> tuple[float, float]:
        return (
            float(self.j1_max_raw - self.servo_zero_raw[0]),
            float(self.j2_max_raw - self.servo_zero_raw[1]),
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> WristConfig:
        """接受五份项目 YAML 或便于测试的扁平 mapping；不访问硬件。"""
        if isinstance(config.get("hardware"), Mapping):
            hardware = config["hardware"]
            safety_root = config.get("safety", {})
            poses_root = config.get("poses", {})
            assert isinstance(hardware, Mapping)
            wrist_hardware = hardware.get("wrist", {})
            wrist_safety = safety_root.get("wrist", {}) if isinstance(safety_root, Mapping) else {}
            wrist_pose = poses_root.get("wrist", {}) if isinstance(poses_root, Mapping) else {}
            if not all(isinstance(value, Mapping) for value in (wrist_hardware, wrist_safety, wrist_pose)):
                raise ValueError("hardware/safety/poses.wrist 必须是 mapping")
            merged: dict[str, Any] = dict(wrist_safety)
            merged.update(wrist_hardware)
            merged.update(wrist_pose)
            config = merged
        port = config.get("port", config.get("openrb_port", config.get("controller_port")))
        if port is None:
            raise ValueError("手腕配置缺少 port/openrb_port/controller_port")
        zero_value = config.get("servo_zero_raw")
        if not isinstance(zero_value, list | tuple) or len(zero_value) != 2:
            raise ValueError("手腕配置缺少两维 servo_zero_raw")
        profile = config.get("dynamixel_profile", {})
        if not isinstance(profile, Mapping):
            raise ValueError("dynamixel_profile 必须是 mapping")
        return cls(
            port=str(port),
            servo_zero_raw=(int(zero_value[0]), int(zero_value[1])),
            baudrate=_as_int(config, "baudrate", _as_int(config, "baud", 115_200)),
            timeout_s=_as_float(config, "timeout_s", _as_float(config, "read_timeout_s", 0.20)),
            state_max_age_s=_as_float(config, "state_max_age_s", _as_float(config, "feedback_timeout_s", 0.20)),
            j1_min_raw=_as_int(config, "j1_min_raw", 2781),
            j1_max_raw=_as_int(config, "j1_max_raw", 3568),
            j2_min_raw=_as_int(config, "j2_min_raw", 1822),
            j2_max_raw=_as_int(config, "j2_max_raw", 3333),
            max_step_raw=_as_int(config, "max_step_raw", 18),
            start_tolerance_raw=_as_int(config, "start_tolerance_raw", 30),
            settle_tolerance_raw=_as_int(config, "settle_tolerance_raw", 30),
            settle_samples=_as_int(config, "settle_samples", 3),
            settle_timeout_s=_as_float(config, "settle_timeout_s", 18.0),
            poll_period_s=_as_float(config, "poll_period_s", 0.05),
            home_timeout_s=_as_float(config, "home_timeout_s", 30.0),
            profile_velocity_raw=int(profile.get("velocity_raw", config.get("profile_velocity_raw", 50))),
            profile_acceleration_raw=int(profile.get("acceleration_raw", config.get("profile_acceleration_raw", 15))),
        )


@dataclass(frozen=True)
class WristState:
    """一次 OpenRB 反馈；训练坐标按 ``[J1,J2]`` YAML 零位相对 raw 排列。"""

    position_relative_raw: tuple[float, float]
    target_relative_raw: tuple[float, float]
    encoder_abs_deg: tuple[float, float]
    encoder_valid: tuple[bool, bool]
    motor_position_raw: tuple[int, int]
    motor_goal_raw: tuple[int, int]
    servo_zero_raw: tuple[int, int]
    hardware_limits_raw: tuple[tuple[int, int], tuple[int, int]]
    host_monotonic_ns: int
    board_ms: int
    sequence: int
    source_age_s: float
    active: bool
    zero_valid: bool
    fault: bool
    fault_reason: str


SerialFactory = Callable[..., Any]


class OpenRBWrist:
    """OpenRB 舵机位置开环协议的单串口拥有者。

    ``connect`` 所在线程成为 I/O 拥有者。其余线程可以读取上层缓存，但
    不允许调用本类的 I/O 方法，从结构上阻止多个线程交叉读串口。
    """

    def __init__(
        self,
        config: WristConfig | Mapping[str, Any],
        *,
        serial_factory: SerialFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config if isinstance(config, WristConfig) else WristConfig.from_mapping(config)
        self._serial_factory = serial_factory
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        self._sleep = sleeper
        self._serial: Any | None = None
        self._owner_thread_id: int | None = None
        self._lock = threading.RLock()
        self._failure: BaseException | None = None
        self._motion_enabled = False
        self._last_state: WristState | None = None
        self._last_target_relative_raw: tuple[float, float] | None = None
        self._hardware_limits: WristHardwareLimits | None = None
        self._direct_mode_ready = False

    @property
    def servo_zero_raw(self) -> tuple[int, int]:
        return self.config.servo_zero_raw

    @property
    def hardware_limits(self) -> WristHardwareLimits | None:
        return self._hardware_limits

    def connect(self) -> None:
        """只读连接并核对固件模式；此方法不会让手腕运动。"""
        with self._lock:
            if self._serial is not None:
                raise RuntimeError("手腕已经连接")
            if self._failure is not None:
                raise WristError(f"手腕已 fail-closed：{self._failure}") from self._failure
            factory = self._serial_factory
            if factory is None:
                try:
                    import serial  # pyserial 必须延迟导入
                except ImportError as exc:
                    raise RuntimeError("缺少 pyserial；请在机器人运行环境安装锁定版本") from exc
                factory = serial.Serial
            try:
                port = factory(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.timeout_s,
                    write_timeout=self.config.timeout_s,
                )
                self._serial = port
                self._owner_thread_id = threading.get_ident()
                self._hardware_limits = self._read_hardware_limits_unlocked()
                self._validate_yaml_zero_against_hardware_unlocked()
                state = self._read_state_unlocked()
                self._last_state = state
                self._last_target_relative_raw = state.position_relative_raw
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def get(self) -> WristState:
        """读取舵机 J1/J2 raw，并减去 YAML 零位后返回训练 state。"""
        with self._lock:
            self._require_owner_unlocked()
            try:
                state = self._read_state_unlocked()
                if state.fault:
                    raise WristError(f"OpenRB output-loop fault: {state.fault_reason}")
                if state.source_age_s > self.config.state_max_age_s:
                    raise WristError(f"手腕状态过期：{state.source_age_s:.3f}s > {self.config.state_max_age_s:.3f}s")
                self._last_state = state
                return state
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def get_position(self) -> tuple[float, float]:
        """返回实际 ``(J1,J2)`` YAML 零位相对 raw。"""
        return self.get().position_relative_raw

    def set(self, position_relative_raw: tuple[float, float] | list[float]) -> tuple[float, float]:
        """把相对 raw 加上 YAML 零位后下发，并返回量化后的相对 raw receipt。"""
        if len(position_relative_raw) != 2:
            raise ValueError("手腕目标必须恰好包含 J1、J2 两个相对 raw")
        values = tuple(float(value) for value in position_relative_raw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("手腕目标包含 NaN 或 Inf")
        quantized = (float(round(values[0])), float(round(values[1])))
        with self._lock:
            self._require_owner_unlocked()
            try:
                state = self._last_state
                if state is None:
                    raise WristError("发送目标前缺少 J1/J2 舵机反馈")
                host_age_s = (self._monotonic_ns() - state.host_monotonic_ns) / 1e9
                if host_age_s < 0.0 or host_age_s > self.config.state_max_age_s:
                    raise WristError(f"手腕缓存状态过期：{host_age_s:.3f}s")
                if not state.zero_valid or state.fault:
                    raise WristError("YAML 舵机零位无效或舵机状态不健康，禁止发送目标")
                reference = self._last_target_relative_raw
                assert reference is not None
                if any(
                    abs(target - current) > self.config.max_step_raw
                    for target, current in zip(quantized, reference, strict=True)
                ):
                    raise ValueError(
                        f"手腕单步超过 {self.config.max_step_raw} raw：current={reference}, target={quantized}"
                    )
                absolute = self._absolute_target_unlocked(quantized)
                self._enter_direct_mode_unlocked()
                self._exchange_unlocked(f"SET_ARM_TARGET_STREAM {absolute[0]} {absolute[1]}")
                self._last_target_relative_raw = quantized
                self._last_state = replace(
                    state,
                    target_relative_raw=quantized,
                    motor_goal_raw=absolute,
                )
                self._motion_enabled = True
                return quantized
            except ValueError:
                raise
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def set_position(self, position_relative_raw: tuple[float, float] | list[float]) -> tuple[float, float]:
        """``set`` 的可读别名。"""
        return self.set(position_relative_raw)

    def home(self, *, enable_motion: bool = False) -> WristState:
        """直接回到 YAML J1/J2 servo raw 零位，不使用 AS5048A 闭环。

        默认拒绝运动。调用方必须同时通过更上层 ``--enable-motion`` 门禁，
        并显式传入 ``enable_motion=True``。
        """
        if not enable_motion:
            raise PermissionError("HOME 会运动真实手腕，必须显式 enable_motion=True")
        with self._lock:
            self._require_owner_unlocked()
            try:
                self._enter_direct_mode_unlocked(hold=False)
                zero = self.config.servo_zero_raw
                self._exchange_unlocked(f"START_ARM_MOVE_TO {zero[0]} {zero[1]}")
                deadline = self._monotonic() + self.config.home_timeout_s
                while self._monotonic() < deadline:
                    status = self._exchange_unlocked("GET_MOTION_STATUS")
                    if status.get("active") == "0":
                        if status.get("status") == "failed":
                            raise WristError(f"回 YAML servo zero 失败：{status.get('error_reason', 'unknown')}")
                        break
                    self._sleep(self.config.poll_period_s)
                else:
                    raise WristTimeoutError("回 YAML servo zero 超时")
                self._exchange_unlocked("HOLD_ALL")
                self._last_target_relative_raw = (0.0, 0.0)
                consecutive = 0
                deadline = self._monotonic() + self.config.settle_timeout_s
                last: WristState | None = None
                while self._monotonic() < deadline:
                    last = self._read_state_unlocked()
                    if last.fault or not last.zero_valid:
                        raise WristError("回 YAML servo zero 后 OpenRB 舵机状态不健康")
                    if last.source_age_s > self.config.state_max_age_s:
                        raise WristError(
                            f"舵机状态过期：{last.source_age_s:.3f}s > {self.config.state_max_age_s:.3f}s"
                        )
                    if max(abs(value) for value in last.position_relative_raw) <= self.config.settle_tolerance_raw:
                        consecutive += 1
                        if consecutive >= self.config.settle_samples:
                            self._last_state = last
                            return last
                    else:
                        consecutive = 0
                    self._sleep(self.config.poll_period_s)
                raise WristTimeoutError(f"YAML servo zero 未收敛：last={last}")
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def stop(self) -> None:
        """停止固件运动并让舵机位置环保持当前位置。"""
        with self._lock:
            self._require_owner_unlocked(allow_failed=True)
            errors: list[BaseException] = []
            for command in ("STOP_MOTION", "STOP_ALL_VELOCITY", "HOLD_ALL"):
                try:
                    self._exchange_unlocked(command)
                except Exception as exc:  # 后续命令仍做 best effort
                    errors.append(exc)
            self._motion_enabled = False
            self._direct_mode_ready = False
            if errors:
                self._trip_unlocked(errors[0])
                raise WristError(f"手腕 stop 失败：{errors[0]}") from errors[0]

    def close(self) -> None:
        """仅关闭串口；上层在运动会话结束时应先调用 ``stop``。"""
        with self._lock:
            self._assert_owner_if_connected_unlocked()
            port, self._serial = self._serial, None
            self._owner_thread_id = None
            if port is not None:
                port.close()

    def healthy(self) -> bool:
        """返回串口已连接且尚未触发 fail-closed。"""
        with self._lock:
            return self._serial is not None and self._failure is None

    def _enter_direct_mode_unlocked(self, *, hold: bool = True) -> None:
        if self._direct_mode_ready:
            return
        self._exchange_unlocked("SET_OUTPUT_CL_ENABLE 0 CONFIRM")
        self._exchange_unlocked("STOP_MOTION")
        self._exchange_unlocked("STOP_ALL_VELOCITY")
        if hold:
            self._exchange_unlocked("HOLD_ALL")
        self._exchange_unlocked(
            f"SET_ARM_PROFILE {self.config.profile_velocity_raw} {self.config.profile_acceleration_raw}"
        )
        self._direct_mode_ready = True

    def _read_hardware_limits_unlocked(self) -> WristHardwareLimits:
        names = ("j1_min", "j1_max", "j1_zero", "j2_min", "j2_max", "j2_zero")
        votes: dict[str, dict[int, int]] = {name: {} for name in names}
        last_error: BaseException | None = None
        for attempt in range(12):
            try:
                fields = self._exchange_unlocked("GET_LIMITS")
                for name in names:
                    try:
                        value = int(fields[name])
                    except (KeyError, ValueError):
                        continue
                    if 0 <= value <= 4095:
                        votes[name][value] = votes[name].get(value, 0) + 1
                consensus: dict[str, str] = {}
                for name in names:
                    winners = [value for value, count in votes[name].items() if count >= 2]
                    if len(winners) > 1:
                        raise WristError(f"GET_LIMITS {name} 出现冲突值：{votes[name]}")
                    if len(winners) == 1:
                        consensus[name] = str(winners[0])
                if len(consensus) == len(names):
                    return WristHardwareLimits.from_fields(consensus)
                last_error = WristError("GET_LIMITS 尚未形成两次一致读数")
            except (WristError, WristTimeoutError) as exc:
                last_error = exc
            if attempt < 11:
                self._sleep(0.02)
        raise WristError(f"GET_LIMITS 连续校验失败：{last_error}; votes={votes}") from last_error

    def _validate_yaml_zero_against_hardware_unlocked(self) -> None:
        limits = self._hardware_limits
        if limits is None:
            raise WristError("尚未读取 OpenRB GET_LIMITS")
        zero1, zero2 = self.config.servo_zero_raw
        if not limits.j1_min_raw <= zero1 <= limits.j1_max_raw:
            raise WristError(f"YAML J1 servo zero={zero1} 超出 OpenRB 限位")
        if not limits.j2_min_raw <= zero2 <= limits.j2_max_raw:
            raise WristError(f"YAML J2 servo zero={zero2} 超出 OpenRB 限位")
        if max(self.config.j1_min_raw, limits.j1_min_raw) >= min(self.config.j1_max_raw, limits.j1_max_raw):
            raise WristError("J1 YAML 与 OpenRB 限位没有有效交集")
        if max(self.config.j2_min_raw, limits.j2_min_raw) >= min(self.config.j2_max_raw, limits.j2_max_raw):
            raise WristError("J2 YAML 与 OpenRB 限位没有有效交集")

    def _absolute_target_unlocked(self, relative: tuple[float, float]) -> tuple[int, int]:
        absolute = (
            round(relative[0] + self.config.servo_zero_raw[0]),
            round(relative[1] + self.config.servo_zero_raw[1]),
        )
        limits = self._hardware_limits
        if limits is None:
            raise WristError("尚未读取 OpenRB GET_LIMITS")
        j1_min = max(self.config.j1_min_raw, limits.j1_min_raw)
        j1_max = min(self.config.j1_max_raw, limits.j1_max_raw)
        j2_min = max(self.config.j2_min_raw, limits.j2_min_raw)
        j2_max = min(self.config.j2_max_raw, limits.j2_max_raw)
        if not j1_min <= absolute[0] <= j1_max or not j2_min <= absolute[1] <= j2_max:
            raise ValueError(
                f"手腕相对 raw 目标越界：relative={relative}, absolute={absolute}, "
                f"J1=[{j1_min},{j1_max}], J2=[{j2_min},{j2_max}]"
            )
        return absolute

    def _read_state_unlocked(self) -> WristState:
        fields = self._exchange_unlocked("GET_WRIST_STATE")
        required = ("joint_age_ms", "j1_ok", "j2_ok", "j1_pos", "j2_pos", "seq", "st")
        missing = [name for name in required if name not in fields]
        if missing:
            raise WristError(f"GET_WRIST_STATE 缺少字段：{', '.join(missing)}")
        try:
            joint_age_ms = int(fields["joint_age_ms"])
            j1_raw = int(fields["j1_pos"])
            j2_raw = int(fields["j2_pos"])
            sequence = int(fields["seq"])
            status_bits = int(fields["st"])
        except ValueError as exc:
            raise WristError("GET_WRIST_STATE 含非法整数") from exc
        if fields["j1_ok"] != "1" or fields["j2_ok"] != "1":
            raise WristError("GET_WRIST_STATE 舵机 J1/J2 反馈无效")
        age_s = joint_age_ms / 1000.0
        if age_s < 0.0 or age_s > self.config.state_max_age_s:
            raise WristError(f"舵机状态过期：{age_s:.3f}s")
        self._absolute_target_unlocked(
            (
                float(j1_raw - self.config.servo_zero_raw[0]),
                float(j2_raw - self.config.servo_zero_raw[1]),
            )
        )
        relative = (
            float(j1_raw - self.config.servo_zero_raw[0]),
            float(j2_raw - self.config.servo_zero_raw[1]),
        )
        target = self._last_target_relative_raw or relative
        goal = (
            round(target[0] + self.config.servo_zero_raw[0]),
            round(target[1] + self.config.servo_zero_raw[1]),
        )
        limits = self._hardware_limits
        if limits is None:
            raise WristError("尚未读取 OpenRB GET_LIMITS")
        encoder_valid = (fields.get("enc0_ok") == "1", fields.get("enc1_ok") == "1")
        try:
            encoder = (
                int(fields["enc0_deg"]) / 100.0 if encoder_valid[0] else math.nan,
                int(fields["enc1_deg"]) / 100.0 if encoder_valid[1] else math.nan,
            )
        except (KeyError, ValueError) as exc:
            raise WristError("GET_WRIST_STATE 编码器诊断字段非法") from exc
        return WristState(
            position_relative_raw=relative,
            target_relative_raw=target,
            encoder_abs_deg=encoder,
            encoder_valid=encoder_valid,
            motor_position_raw=(j1_raw, j2_raw),
            motor_goal_raw=goal,
            servo_zero_raw=self.config.servo_zero_raw,
            hardware_limits_raw=(
                (limits.j1_min_raw, limits.j1_max_raw),
                (limits.j2_min_raw, limits.j2_max_raw),
            ),
            host_monotonic_ns=self._monotonic_ns(),
            board_ms=int(fields.get("ms", "0")),
            sequence=sequence,
            source_age_s=age_s,
            active=True,
            zero_valid=True,
            fault=False,
            fault_reason=f"status_bits={status_bits}",
        )

    def _exchange_unlocked(self, command: str) -> dict[str, str]:
        port = self._require_serial_unlocked()
        expected = command.split()[0].upper()
        port.reset_input_buffer()
        port.write((command.strip() + "\n").encode("ascii"))
        port.flush()
        deadline = self._monotonic() + self.config.timeout_s
        while self._monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            try:
                text = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise WristError(f"OpenRB 返回非 ASCII 数据：{raw!r}") from exc
            parts = text.split()
            if not parts or parts[0].upper() == "INFO":
                continue
            if parts[0].upper() not in {"OK", "ERR"} or len(parts) < 2:
                continue
            response_command = parts[1].upper()
            if response_command not in {expected, "UNKNOWN"}:
                continue
            fields = self._parse_fields(parts[2:])
            if parts[0].upper() == "ERR":
                raise WristError(f"OpenRB 拒绝 {expected}：{text}")
            return fields
        raise WristTimeoutError(f"OpenRB 等待 {expected} 应答超时")

    @staticmethod
    def _parse_fields(parts: list[str]) -> dict[str, str]:
        fields: dict[str, str] = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        return fields

    def _require_owner_unlocked(self, *, allow_failed: bool = False) -> None:
        self._assert_owner_if_connected_unlocked()
        if self._serial is None:
            raise RuntimeError("手腕尚未连接")
        if self._failure is not None and not allow_failed:
            raise WristError(f"手腕已 fail-closed：{self._failure}") from self._failure

    def _assert_owner_if_connected_unlocked(self) -> None:
        if self._serial is not None and self._owner_thread_id != threading.get_ident():
            raise DeviceOwnershipError("只有 connect() 所在线程可以访问 OpenRB 串口")

    def _require_serial_unlocked(self) -> Any:
        if self._serial is None:
            raise RuntimeError("手腕尚未连接")
        return self._serial

    def _trip_unlocked(self, failure: BaseException) -> None:
        """通信异常时 best-effort 停止输出，然后永久关闭本实例。"""
        if self._failure is None:
            self._failure = failure
        port, self._serial = self._serial, None
        self._owner_thread_id = None
        if port is None:
            return
        if self._motion_enabled:
            for command in (b"STOP_MOTION\n", b"STOP_ALL_VELOCITY\n", b"HOLD_ALL\n"):
                try:
                    port.write(command)
                    port.flush()
                except Exception:
                    break
        with suppress(Exception):
            port.close()
        self._motion_enabled = False
        self._direct_mode_ready = False


def create_wrist(
    config: WristConfig | Mapping[str, Any],
    *,
    serial_factory: SerialFactory | None = None,
) -> OpenRBWrist:
    """由 mapping/dataclass 创建项目内两轴腕驱动；不会自动连接。"""
    return OpenRBWrist(config, serial_factory=serial_factory)


@dataclass(frozen=True)
class MasterWristConfig:
    """ESP32 主腕到 J1/J2 相对 raw 的映射和 SpaceMouse 覆盖参数。"""

    port: str
    baudrate: int = 115_200
    serial_timeout_s: float = 0.05
    response_timeout_s: float = 2.0
    stream_period_ms: int = 10
    source_max_age_s: float = 0.20
    j1_sign: int = 1
    j2_sign: int = 1
    j1_raw_per_deg: float = 10.0
    j2_raw_per_deg: float = 14.0
    input_deadband_deg: float = 0.50
    one_euro_min_cutoff: float = 1.8
    one_euro_beta: float = 0.08
    one_euro_d_cutoff: float = 1.0
    target_deadband_raw: int = 2
    max_velocity_raw_s: float = 900.0
    max_accel_raw_s2: float = 4500.0
    max_jerk_raw_s3: float = 30000.0
    command_hz: float = 50.0
    state_hz: float = 10.0
    override_max_velocity_deg_s: float = 45.0
    override_lease_s: float = 0.10
    override_accel_deg_s2: float = 300.0
    override_decel_deg_s2: float = 500.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _require_by_id(self.port))
        if self.baudrate <= 0:
            raise ValueError("master baudrate 必须为正数")
        if not 1 <= self.stream_period_ms <= 1000:
            raise ValueError("stream_period_ms 必须在 [1, 1000]")
        for name in (
            "serial_timeout_s",
            "response_timeout_s",
            "source_max_age_s",
            "command_hz",
            "state_hz",
            "override_max_velocity_deg_s",
            "override_lease_s",
            "override_accel_deg_s2",
            "override_decel_deg_s2",
            "one_euro_min_cutoff",
            "one_euro_d_cutoff",
            "max_velocity_raw_s",
            "max_accel_raw_s2",
            "max_jerk_raw_s3",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
        if self.j1_sign not in {-1, 1} or self.j2_sign not in {-1, 1}:
            raise ValueError("J1/J2 sign 必须是 -1 或 1")
        if self.j1_raw_per_deg <= 0.0 or self.j2_raw_per_deg <= 0.0 or self.input_deadband_deg < 0.0:
            raise ValueError("主腕 raw_per_deg 必须为正，deadband 不得为负")
        if self.state_hz > self.command_hz:
            raise ValueError("state_hz 不得高于 command_hz")
        if not math.isfinite(self.one_euro_beta) or self.one_euro_beta < 0.0:
            raise ValueError("one_euro_beta 必须是有限非负数")
        if self.target_deadband_raw < 0:
            raise ValueError("target_deadband_raw 不得为负")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> MasterWristConfig:
        """接受 hardware.wrist 或 config_loader 返回的整个项目配置。"""
        teleop_wrist: Mapping[str, Any] = {}
        if isinstance(config.get("hardware"), Mapping):
            hardware = config["hardware"]
            assert isinstance(hardware, Mapping)
            wrist = hardware.get("wrist", {})
            if not isinstance(wrist, Mapping):
                raise ValueError("hardware.wrist 必须是 mapping")
            teleop = config.get("teleop", {})
            if isinstance(teleop, Mapping) and isinstance(teleop.get("modes"), Mapping):
                modes = teleop["modes"]
                assert isinstance(modes, Mapping)
                candidate = modes.get("wrist", {})
                if isinstance(candidate, Mapping):
                    teleop_wrist = candidate
            config = wrist
        port = config.get("port", config.get("master_port"))
        if port is None:
            raise ValueError("主腕配置缺少 port/master_port")
        mapping = config.get("master_mapping", {})
        if not isinstance(mapping, Mapping):
            raise ValueError("master_mapping 必须是 mapping")
        filter_config = config.get("filter", {})
        if not isinstance(filter_config, Mapping):
            raise ValueError("filter 必须是 mapping")
        limiter = config.get("target_limiter", {})
        if not isinstance(limiter, Mapping):
            raise ValueError("target_limiter 必须是 mapping")
        return cls(
            port=str(port),
            baudrate=int(config.get("baudrate", config.get("baud", 115_200))),
            serial_timeout_s=float(config.get("serial_timeout_s", 0.05)),
            response_timeout_s=float(config.get("response_timeout_s", 2.0)),
            stream_period_ms=int(config.get("stream_period_ms", 10)),
            source_max_age_s=float(config.get("source_max_age_s", config.get("read_timeout_s", 0.20))),
            j1_sign=int(mapping.get("j1_sign", 1)),
            j2_sign=int(mapping.get("j2_sign", 1)),
            j1_raw_per_deg=float(mapping.get("j1_raw_per_deg", 10.0)),
            j2_raw_per_deg=float(mapping.get("j2_raw_per_deg", 14.0)),
            input_deadband_deg=float(mapping.get("input_deadband_deg", 0.50)),
            one_euro_min_cutoff=float(filter_config.get("one_euro_min_cutoff", 1.8)),
            one_euro_beta=float(filter_config.get("one_euro_beta", 0.08)),
            one_euro_d_cutoff=float(filter_config.get("one_euro_d_cutoff", 1.0)),
            target_deadband_raw=int(limiter.get("deadband_raw", 2)),
            max_velocity_raw_s=float(limiter.get("max_velocity_raw_s", 900.0)),
            max_accel_raw_s2=float(limiter.get("max_accel_raw_s2", 4500.0)),
            max_jerk_raw_s3=float(limiter.get("max_jerk_raw_s3", 30000.0)),
            command_hz=float(config.get("command_hz", 50.0)),
            state_hz=float(config.get("state_hz", 10.0)),
            override_max_velocity_deg_s=float(
                config.get("override_max_velocity_deg_s", teleop_wrist.get("max_speed_deg_s", 45.0))
            ),
            override_lease_s=float(config.get("override_lease_s", 0.10)),
            override_accel_deg_s2=float(config.get("override_accel_deg_s2", 300.0)),
            override_decel_deg_s2=float(config.get("override_decel_deg_s2", 500.0)),
        )


@dataclass(frozen=True)
class MasterWristState:
    """ESP32 主腕当前动态零位映射出的 ``[J1,J2]`` 相对 raw 目标。"""

    target_relative_raw: tuple[float, float]
    encoder_abs_deg: tuple[float, float]
    encoder_relative_deg: tuple[float, float]
    host_monotonic_ns: int
    sequence: int
    board_sequence: int
    zero_valid: bool


@dataclass(frozen=True)
class _MasterLine:
    ok: bool
    command: str
    fields: dict[str, str]
    raw: str


class _AngleUnwrapper:
    def __init__(self) -> None:
        self._previous_deg: float | None = None
        self._relative_deg = 0.0

    def reset(self, absolute_deg: float) -> None:
        self._previous_deg = float(absolute_deg)
        self._relative_deg = 0.0

    def update(self, absolute_deg: float) -> float:
        value = float(absolute_deg)
        if self._previous_deg is None:
            self.reset(value)
            return 0.0
        delta = value - self._previous_deg
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
        self._relative_deg += delta
        self._previous_deg = value
        return self._relative_deg


def _deadband(value: float, width: float) -> float:
    if abs(value) <= width:
        return 0.0
    return value - width if value > 0.0 else value + width


class _LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.value = 0.0

    def filter(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.value = float(value)
            self.initialized = True
            return self.value
        self.value = alpha * float(value) + (1.0 - alpha) * self.value
        return self.value


class _OneEuroFilter:
    """与 open_loop_record 相同的 One-Euro 目标滤波器。"""

    def __init__(self, *, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = _LowPassFilter()
        self.dx_filter = _LowPassFilter()
        self.previous_raw: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / max(1e-6, dt))

    def reset(self, value: float) -> None:
        self.x_filter = _LowPassFilter()
        self.dx_filter = _LowPassFilter()
        self.previous_raw = float(value)
        self.x_filter.filter(value, 1.0)
        self.dx_filter.filter(0.0, 1.0)

    def filter(self, value: float, dt: float) -> float:
        if self.previous_raw is None:
            self.reset(value)
            return float(value)
        derivative = (float(value) - self.previous_raw) / max(1e-6, dt)
        self.previous_raw = float(value)
        filtered_derivative = self.dx_filter.filter(derivative, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        return self.x_filter.filter(value, self._alpha(cutoff, dt))


class _JerkLimitedTarget:
    """相对 raw 坐标的速度、加速度和 jerk 限制器。"""

    def __init__(
        self,
        *,
        min_raw: float,
        max_raw: float,
        max_velocity_raw_s: float,
        max_accel_raw_s2: float,
        max_jerk_raw_s3: float,
    ) -> None:
        self.min_raw = float(min_raw)
        self.max_raw = float(max_raw)
        self.max_velocity_raw_s = float(max_velocity_raw_s)
        self.max_accel_raw_s2 = float(max_accel_raw_s2)
        self.max_jerk_raw_s3 = float(max_jerk_raw_s3)
        self.position: float | None = None
        self.velocity = 0.0
        self.acceleration = 0.0

    def reset(self, raw: float) -> None:
        self.position = max(self.min_raw, min(self.max_raw, float(raw)))
        self.velocity = 0.0
        self.acceleration = 0.0

    def update(self, desired_raw: float, dt: float) -> float:
        desired = max(self.min_raw, min(self.max_raw, float(desired_raw)))
        if self.position is None:
            self.reset(desired)
            return desired
        dt = max(1e-6, float(dt))
        error = desired - self.position
        desired_velocity = max(-self.max_velocity_raw_s, min(self.max_velocity_raw_s, error / dt))
        desired_acceleration = max(
            -self.max_accel_raw_s2,
            min(self.max_accel_raw_s2, (desired_velocity - self.velocity) / dt),
        )
        max_accel_step = self.max_jerk_raw_s3 * dt
        self.acceleration += max(
            -max_accel_step,
            min(max_accel_step, desired_acceleration - self.acceleration),
        )
        self.acceleration = max(-self.max_accel_raw_s2, min(self.max_accel_raw_s2, self.acceleration))
        self.velocity += self.acceleration * dt
        self.velocity = max(-self.max_velocity_raw_s, min(self.max_velocity_raw_s, self.velocity))
        next_position = self.position + self.velocity * dt
        if (desired - self.position) * (desired - next_position) <= 0.0:
            next_position = desired
            self.velocity = 0.0
            self.acceleration = 0.0
        self.position = max(self.min_raw, min(self.max_raw, next_position))
        return self.position


class MasterWristReader:
    """ESP32 主腕 ``OK TELE`` 流的单线程串口拥有者。"""

    def __init__(
        self,
        config: MasterWristConfig | Mapping[str, Any],
        *,
        serial_factory: SerialFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config if isinstance(config, MasterWristConfig) else MasterWristConfig.from_mapping(config)
        self._serial_factory = serial_factory
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        self._serial: Any | None = None
        self._owner_thread_id: int | None = None
        self._lock = threading.RLock()
        self._failure: BaseException | None = None
        self._enc0 = _AngleUnwrapper()
        self._enc1 = _AngleUnwrapper()
        self._sequence = 0
        self._last_board_sequence: int | None = None
        self._streaming = False

    def connect(self) -> None:
        """连接后先 STOP 并设置 10ms 周期，不启动机器人或主腕数据流。"""
        with self._lock:
            if self._serial is not None:
                raise RuntimeError("主腕已经连接")
            factory = self._serial_factory
            if factory is None:
                try:
                    import serial
                except ImportError as exc:
                    raise RuntimeError("缺少 pyserial；请安装 hardware 依赖组") from exc
                factory = serial.Serial
            try:
                self._serial = factory(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.serial_timeout_s,
                    write_timeout=self.config.response_timeout_s,
                )
                self._owner_thread_id = threading.get_ident()
                self._command_unlocked("STOP", {"STOP"}, allow_stopped_tele=True)
                self._command_unlocked(f"SET_PERIOD {self.config.stream_period_ms}", {"SET_PERIOD"})
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def start_stream(self) -> MasterWristState:
        """以第一帧绝对编码器值建立本次主腕动态零位。"""
        with self._lock:
            self._require_owner_unlocked()
            try:
                started = self._command_unlocked("START", {"START", "TELE"})
                if not {"enc0_deg", "enc1_deg"}.issubset(started.fields):
                    started = self._next_tele_unlocked(self.config.response_timeout_s)
                enc0, enc1 = self._encoder_fields(started.fields)
                self._enc0.reset(enc0)
                self._enc1.reset(enc1)
                self._last_board_sequence = None
                self._streaming = True
                return self._state_from_line(started, baseline=True)
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def get(self) -> MasterWristState:
        """等待下一帧 TELE，并映射成 ``[J1,J2]`` 相对 raw 目标。"""
        with self._lock:
            self._require_owner_unlocked()
            if not self._streaming:
                raise RuntimeError("主腕尚未 START")
            try:
                return self._state_from_line(self._next_tele_unlocked(self.config.source_max_age_s))
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def get_position(self) -> tuple[float, float]:
        return self.get().target_relative_raw

    def stop(self) -> None:
        """停止 ESP32 TELE 流；不会向 OpenRB 发送命令。"""
        with self._lock:
            self._require_owner_unlocked(allow_failed=True)
            try:
                self._command_unlocked("STOP", {"STOP"}, allow_stopped_tele=True)
                self._streaming = False
            except Exception as exc:
                self._trip_unlocked(exc)
                raise

    def close(self) -> None:
        with self._lock:
            if self._serial is not None and self._owner_thread_id != threading.get_ident():
                raise DeviceOwnershipError("只有连接线程可以关闭主腕串口")
            port, self._serial = self._serial, None
            self._owner_thread_id = None
            if port is not None:
                port.close()

    def healthy(self) -> bool:
        with self._lock:
            return self._serial is not None and self._failure is None

    def _state_from_line(self, line: _MasterLine, *, baseline: bool = False) -> MasterWristState:
        enc0, enc1 = self._encoder_fields(line.fields)
        relative0 = 0.0 if baseline else self._enc0.update(enc0)
        relative1 = 0.0 if baseline else self._enc1.update(enc1)
        zero_valid = line.fields.get("zero_valid")
        if zero_valid not in {"0", "1"}:
            raise WristError("ESP32 主腕 TELE 缺少合法 zero_valid")
        if zero_valid != "1":
            raise WristError("ESP32 主腕 zero_valid=0")
        try:
            board_sequence = int(line.fields["seq"])
        except (KeyError, ValueError) as exc:
            raise WristError("ESP32 主腕 TELE 缺少合法 seq") from exc
        if self._last_board_sequence is not None and board_sequence <= self._last_board_sequence:
            raise WristError(f"ESP32 主腕 sequence 重复或倒退：{board_sequence} <= {self._last_board_sequence}")
        if "age_ms" not in line.fields:
            if not baseline:
                raise WristError("ESP32 主腕 TELE 缺少 age_ms")
        else:
            try:
                age_s = int(line.fields["age_ms"]) / 1000.0
            except ValueError as exc:
                raise WristError("ESP32 主腕 TELE age_ms 非法") from exc
            if age_s < 0.0 or age_s > self.config.source_max_age_s:
                raise WristError("ESP32 主腕 TELE 状态过期")
        self._last_board_sequence = board_sequence
        self._sequence += 1
        j1_relative_raw = (
            self.config.j1_sign
            * self.config.j1_raw_per_deg
            * _deadband(relative0, self.config.input_deadband_deg)
        )
        j2_relative_raw = (
            self.config.j2_sign
            * self.config.j2_raw_per_deg
            * _deadband(relative1, self.config.input_deadband_deg)
        )
        return MasterWristState(
            target_relative_raw=(j1_relative_raw, j2_relative_raw),
            encoder_abs_deg=(enc0, enc1),
            encoder_relative_deg=(relative0, relative1),
            host_monotonic_ns=self._monotonic_ns(),
            sequence=self._sequence,
            board_sequence=board_sequence,
            zero_valid=True,
        )

    @staticmethod
    def _encoder_fields(fields: Mapping[str, str]) -> tuple[float, float]:
        try:
            enc0 = int(fields["enc0_deg"]) / 100.0
            enc1 = int(fields["enc1_deg"]) / 100.0
        except (KeyError, ValueError) as exc:
            raise WristError("ESP32 TELE 缺少合法 enc0_deg/enc1_deg") from exc
        if not 0.0 <= enc0 <= 360.0 or not 0.0 <= enc1 <= 360.0:
            raise WristError("ESP32 绝对编码器超出 [0, 360]°")
        return enc0, enc1

    def _command_unlocked(
        self,
        command: str,
        expected: set[str],
        *,
        allow_stopped_tele: bool = False,
    ) -> _MasterLine:
        port = self._require_serial_unlocked()
        port.write((command.strip() + "\n").encode("ascii"))
        port.flush()
        deadline = self._monotonic() + self.config.response_timeout_s
        while self._monotonic() < deadline:
            parsed = self._read_line_unlocked()
            if parsed is None:
                continue
            if not parsed.ok:
                raise WristError(f"ESP32 拒绝命令：{parsed.raw}")
            if parsed.command in expected:
                return parsed
            if allow_stopped_tele and parsed.command == "TELE" and parsed.fields.get("st") == "0":
                return parsed
        raise WristTimeoutError(f"ESP32 等待 {expected} 应答超时")

    def _next_tele_unlocked(self, timeout_s: float) -> _MasterLine:
        deadline = self._monotonic() + timeout_s
        while self._monotonic() < deadline:
            parsed = self._read_line_unlocked()
            if parsed is None:
                continue
            if not parsed.ok:
                raise WristError(f"ESP32 TELE 错误：{parsed.raw}")
            if parsed.command == "TELE":
                return parsed
        raise WristTimeoutError("ESP32 主腕 TELE 超时")

    def _read_line_unlocked(self) -> _MasterLine | None:
        raw = self._require_serial_unlocked().readline()
        if not raw:
            return None
        text = raw.decode("ascii", errors="strict").strip()
        parts = text.split()
        if len(parts) < 2 or parts[0] not in {"OK", "ERR"}:
            return None
        fields: dict[str, str] = {}
        for token in parts[2:]:
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        return _MasterLine(parts[0] == "OK", parts[1].upper(), fields, text)

    def _require_owner_unlocked(self, *, allow_failed: bool = False) -> None:
        if self._serial is None:
            if self._failure is not None and not allow_failed:
                raise WristError(f"主腕已 fail-closed：{self._failure}") from self._failure
            raise RuntimeError("主腕尚未连接")
        if self._owner_thread_id != threading.get_ident():
            raise DeviceOwnershipError("只有 connect() 所在线程可以访问主腕串口")
        if self._failure is not None and not allow_failed:
            raise WristError(f"主腕已 fail-closed：{self._failure}") from self._failure

    def _require_serial_unlocked(self) -> Any:
        if self._serial is None:
            raise RuntimeError("主腕尚未连接")
        return self._serial

    def _trip_unlocked(self, failure: BaseException) -> None:
        if self._failure is None:
            self._failure = failure
        port, self._serial = self._serial, None
        self._owner_thread_id = None
        if port is None:
            return
        with suppress(Exception):
            port.write(b"STOP\n")
            port.flush()
        with suppress(Exception):
            port.close()
        self._streaming = False


@dataclass(frozen=True)
class WristMasterSlaveState:
    """主从腕控制器发布给数采线程的不可变缓存。"""

    actual_relative_raw: tuple[float, float]
    target_relative_raw: tuple[float, float]
    host_monotonic_ns: int
    sequence: int
    mode: str
    master_state: MasterWristState | None
    output_state: WristState


class WristMasterSlaveController:
    """项目内同步主腕→从腕控制器。

    本类不启动隐藏 I/O 线程。``connect``、``step``、``home``、``stop``
    和 ``close`` 必须由 RobotHardware 的同一个硬件 worker 调用；UI 与
    录制器只调用 ``get`` 读取缓存。这样主腕与 OpenRB 两个串口都有唯一
    I/O 拥有者。
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        wrist_config: WristConfig | None = None,
        master_config: MasterWristConfig | None = None,
        wrist_factory: Callable[[WristConfig], Any] | None = None,
        master_factory: Callable[[MasterWristConfig], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if config is None and (wrist_config is None or master_config is None):
            raise ValueError("必须提供项目 config 或 wrist/master 两份 dataclass")
        self.wrist_config = wrist_config or WristConfig.from_mapping(config or {})
        self.master_config = master_config or MasterWristConfig.from_mapping(config or {})
        self._wrist_factory = wrist_factory or (lambda value: OpenRBWrist(value))
        self._master_factory = master_factory or (lambda value: MasterWristReader(value))
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        self._owner_thread_id: int | None = None
        self._wrist: Any | None = None
        self._master: Any | None = None
        self._lock = threading.Lock()
        self._latest: WristMasterSlaveState | None = None
        self._failure: BaseException | None = None
        self._sequence = 0
        self._parked = True
        self._master_active = False
        self._override_active = False
        self._override_requested_deg_s = (0.0, 0.0)  # [J1 physical, J2 physical]
        self._override_applied_deg_s = (0.0, 0.0)
        self._override_updated_s = 0.0
        self._policy_target_relative_raw: tuple[float, float] | None = None
        self._command_target_relative_raw = (0.0, 0.0)
        self._master_anchor_relative_raw = (0.0, 0.0)
        self._master_filters = tuple(
            _OneEuroFilter(
                min_cutoff=self.master_config.one_euro_min_cutoff,
                beta=self.master_config.one_euro_beta,
                d_cutoff=self.master_config.one_euro_d_cutoff,
            )
            for _ in range(2)
        )
        self._target_limiters = tuple(
            _JerkLimitedTarget(
                min_raw=minimum,
                max_raw=maximum,
                max_velocity_raw_s=self.master_config.max_velocity_raw_s,
                max_accel_raw_s2=self.master_config.max_accel_raw_s2,
                max_jerk_raw_s3=self.master_config.max_jerk_raw_s3,
            )
            for minimum, maximum in zip(
                self.wrist_config.relative_min_raw,
                self.wrist_config.relative_max_raw,
                strict=True,
            )
        )
        self._last_output: WristState | None = None
        self._next_state_read_s = 0.0
        self._last_step_s = 0.0

    def connect(self, *, enable_motion: bool = False) -> WristMasterSlaveState:
        """建立两串口连接并 HOME 从腕；HOME 后保持 parked，不跟随主腕。"""
        if not enable_motion:
            raise PermissionError("主从腕 connect 包含真实 HOME，必须显式 enable_motion=True")
        if self._owner_thread_id is not None:
            raise RuntimeError("主从腕已经连接")
        self._owner_thread_id = threading.get_ident()
        wrist = self._wrist_factory(self.wrist_config)
        master = self._master_factory(self.master_config)
        self._wrist, self._master = wrist, master
        try:
            master.connect()
            wrist.connect()
            output = wrist.home(enable_motion=True)
            self._last_output = output
            self._command_target_relative_raw = (0.0, 0.0)
            self._reset_target_pipeline((0.0, 0.0))
            self._last_step_s = self._monotonic()
            self._next_state_read_s = self._last_step_s + 1.0 / self.master_config.state_hz
            self._parked = True
            return self._publish(output, "parked", None)
        except Exception as exc:
            self._fail_closed(exc)
            raise

    def resume_master(self) -> MasterWristState:
        """显式重新绑定主腕当前姿态为零并开始跟随。"""
        self._require_owner()
        assert self._master is not None
        try:
            # Ctrl 覆盖期间 ESP32 仍可能持续发送 TELE。重新绑定零点前必须
            # 先 STOP 清空旧流，再 START 取得新的基准帧；直接重复 START
            # 既可能被固件拒绝，也可能把串口积压的旧 TELE 当成新零点。
            if self._master_active:
                self._master.stop()
                self._master_active = False
            state = self._master.start_stream()
            self._master_anchor_relative_raw = self._command_target_relative_raw
            for filter_, anchor in zip(self._master_filters, self._master_anchor_relative_raw, strict=True):
                filter_.reset(anchor)
            for limiter, current in zip(self._target_limiters, self._command_target_relative_raw, strict=True):
                limiter.reset(current)
            with self._lock:
                self._override_active = False
                self._override_requested_deg_s = (0.0, 0.0)
                self._override_applied_deg_s = (0.0, 0.0)
                self._policy_target_relative_raw = None
            self._master_active = True
            self._parked = False
            return state
        except Exception as exc:
            self._fail_closed(exc)
            raise

    def step(self) -> WristMasterSlaveState:
        """执行一次 50Hz 仲裁、限步、目标发送和反馈缓存更新。"""
        self._require_owner()
        assert self._wrist is not None
        assert self._master is not None
        now = self._monotonic()
        dt = min(0.05, max(0.001, now - self._last_step_s))
        self._last_step_s = now
        master_state: MasterWristState | None = None
        try:
            with self._lock:
                policy = self._policy_target_relative_raw
                override = self._override_active
                requested = self._override_requested_deg_s
                updated = self._override_updated_s
            if self._parked:
                output = self._read_output_if_due(now)
                return self._publish(output, "parked", None)
            if policy is not None:
                desired_relative_raw = policy
                mode = "policy"
            elif override:
                if now - updated > self.master_config.override_lease_s:
                    requested = (0.0, 0.0)
                applied = self._slew_override(requested, dt)
                raw_per_deg = (
                    self.master_config.j1_raw_per_deg,
                    self.master_config.j2_raw_per_deg,
                )
                desired_relative_raw = tuple(
                    target + velocity * scale * dt
                    for target, velocity, scale in zip(
                        self._command_target_relative_raw,
                        applied,
                        raw_per_deg,
                        strict=True,
                    )
                )
                mode = "spacemouse"
            else:
                master_state = self._master.get()
                raw_target = tuple(
                    anchor + relative
                    for anchor, relative in zip(
                        self._master_anchor_relative_raw,
                        master_state.target_relative_raw,
                        strict=True,
                    )
                )
                desired_relative_raw = tuple(
                    filter_.filter(value, dt)
                    for filter_, value in zip(self._master_filters, raw_target, strict=True)
                )
                mode = "master"
            limited_relative_raw = tuple(
                limiter.update(value, dt)
                for limiter, value in zip(self._target_limiters, desired_relative_raw, strict=True)
            )
            target_relative_raw = self._bounded_step(limited_relative_raw)
            changed = any(
                abs(target - current) >= self.master_config.target_deadband_raw
                for target, current in zip(
                    target_relative_raw,
                    self._command_target_relative_raw,
                    strict=True,
                )
            )
            if changed:
                sent_relative_raw = self._wrist.set(target_relative_raw)
                self._command_target_relative_raw = tuple(float(value) for value in sent_relative_raw)
            output = self._read_output_if_due(now)
            return self._publish(output, mode, master_state)
        except Exception as exc:
            self._fail_closed(exc)
            raise

    def set_spacemouse_velocity(self, j1_deg_s: float, j2_deg_s: float) -> None:
        """Ctrl 模式输入；调用线程只更新缓存，不访问串口。"""
        values = (float(j1_deg_s), float(j2_deg_s))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SpaceMouse 手腕速度包含 NaN/Inf")
        limit = self.master_config.override_max_velocity_deg_s
        with self._lock:
            self._override_requested_deg_s = tuple(max(-limit, min(limit, value)) for value in values)
            self._override_updated_s = self._monotonic()
            if not self._override_active:
                self._override_applied_deg_s = (0.0, 0.0)
            self._override_active = True
            self._policy_target_relative_raw = None

    def clear_spacemouse_velocity(self) -> None:
        """Ctrl 松开后速度归零并保持最后目标，不自动跳回主腕。"""
        with self._lock:
            self._override_requested_deg_s = (0.0, 0.0)
            self._override_applied_deg_s = (0.0, 0.0)

    def set_policy_target(self, target_relative_raw: tuple[float, float] | list[float]) -> None:
        """设置 policy 输出的 ``[J1,J2]`` YAML 零位相对 raw 目标。"""
        if len(target_relative_raw) != 2:
            raise ValueError("policy wrist relative raw target 必须为 2 维")
        values = tuple(float(value) for value in target_relative_raw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("policy wrist target 包含 NaN/Inf")
        with self._lock:
            self._policy_target_relative_raw = values
            self._override_active = False

    def clear_policy_target(self) -> None:
        with self._lock:
            self._policy_target_relative_raw = None
            self._override_active = True
            self._override_requested_deg_s = (0.0, 0.0)
            self._override_applied_deg_s = (0.0, 0.0)

    def home(self, *, enable_motion: bool = False) -> WristMasterSlaveState:
        """停止主腕流并让 J1/J2 回 YAML servo zero；结束后保持 parked。"""
        if not enable_motion:
            raise PermissionError("主从腕 HOME 必须显式 enable_motion=True")
        self._require_owner()
        assert self._wrist is not None
        assert self._master is not None
        try:
            if self._master_active:
                self._master.stop()
            self._master_active = False
            output = self._wrist.home(enable_motion=True)
            self._last_output = output
            self._command_target_relative_raw = (0.0, 0.0)
            self._reset_target_pipeline((0.0, 0.0))
            now = self._monotonic()
            self._next_state_read_s = now + 1.0 / self.master_config.state_hz
            self._parked = True
            return self._publish(output, "parked", None)
        except Exception as exc:
            self._fail_closed(exc)
            raise

    def get(self) -> WristMasterSlaveState:
        """仅复制缓存，不访问两个串口，可由录制/UI 线程调用。"""
        with self._lock:
            failure = self._failure
            latest = self._latest
        if failure is not None:
            raise WristError(f"主从腕已 fail-closed：{failure}") from failure
        if latest is None:
            raise RuntimeError("主从腕尚无状态")
        age_s = (self._monotonic_ns() - latest.host_monotonic_ns) / 1e9
        if age_s > self.wrist_config.state_max_age_s:
            raise WristError(f"主从腕缓存过期：{age_s:.3f}s")
        return latest

    def stop(self) -> None:
        """由硬件 owner 停止两条控制链；任何失败都会关闭连接。"""
        self._require_owner(allow_failed=True)
        errors: list[BaseException] = []
        if self._master is not None and self._master_active:
            try:
                self._master.stop()
            except Exception as exc:
                errors.append(exc)
        self._master_active = False
        if self._wrist is not None:
            try:
                self._wrist.stop()
            except Exception as exc:
                errors.append(exc)
        self._parked = True
        if errors:
            self._fail_closed(errors[0])
            raise WristError(f"主从腕 stop 失败：{errors[0]}") from errors[0]

    def close(self) -> None:
        """停止后由 owner 关闭两个串口。"""
        if self._owner_thread_id is None:
            return
        self._require_owner(allow_failed=True)
        if self._failure is None:
            self.stop()
        master, wrist = self._master, self._wrist
        self._master = self._wrist = None
        self._owner_thread_id = None
        if master is not None:
            master.close()
        if wrist is not None:
            wrist.close()

    def healthy(self) -> bool:
        with self._lock:
            return self._failure is None and self._owner_thread_id is not None

    def _bounded_step(self, desired_relative_raw: tuple[float, float]) -> tuple[float, float]:
        bounds = (
            (self.wrist_config.relative_min_raw[0], self.wrist_config.relative_max_raw[0]),
            (self.wrist_config.relative_min_raw[1], self.wrist_config.relative_max_raw[1]),
        )
        result = []
        for desired, current, (minimum, maximum) in zip(
            desired_relative_raw,
            self._command_target_relative_raw,
            bounds,
            strict=True,
        ):
            clipped = max(minimum, min(maximum, desired))
            delta = max(-self.wrist_config.max_step_raw, min(self.wrist_config.max_step_raw, clipped - current))
            result.append(current + delta)
        return (result[0], result[1])

    def _reset_target_pipeline(self, position_relative_raw: tuple[float, float]) -> None:
        self._master_anchor_relative_raw = tuple(float(value) for value in position_relative_raw)
        for filter_, value in zip(self._master_filters, position_relative_raw, strict=True):
            filter_.reset(value)
        for limiter, value in zip(self._target_limiters, position_relative_raw, strict=True):
            limiter.reset(value)

    def _read_output_if_due(self, now_s: float) -> WristState:
        if self._wrist is None:
            raise RuntimeError("从腕尚未连接")
        if self._last_output is None or now_s >= self._next_state_read_s:
            self._last_output = self._wrist.get()
            period = 1.0 / self.master_config.state_hz
            self._next_state_read_s = max(self._next_state_read_s + period, now_s + period)
        return self._last_output

    def _slew_override(self, desired: tuple[float, float], dt: float) -> tuple[float, float]:
        values = []
        for current, target in zip(self._override_applied_deg_s, desired, strict=True):
            reversing = current != 0.0 and math.copysign(1.0, current) != math.copysign(1.0, target)
            effective = 0.0 if reversing else target
            braking = abs(effective) < abs(current)
            rate = self.master_config.override_decel_deg_s2 if braking else self.master_config.override_accel_deg_s2
            delta = max(-rate * dt, min(rate * dt, effective - current))
            values.append(current + delta)
        applied = (values[0], values[1])
        with self._lock:
            self._override_applied_deg_s = applied
        return applied

    def _publish(
        self,
        output: WristState,
        mode: str,
        master: MasterWristState | None,
    ) -> WristMasterSlaveState:
        self._sequence += 1
        state = WristMasterSlaveState(
            actual_relative_raw=output.position_relative_raw,
            target_relative_raw=self._command_target_relative_raw,
            host_monotonic_ns=output.host_monotonic_ns,
            sequence=self._sequence,
            mode=mode,
            master_state=master,
            output_state=output,
        )
        with self._lock:
            self._latest = state
        return state

    def _require_owner(self, *, allow_failed: bool = False) -> None:
        if self._owner_thread_id is None:
            if self._failure is not None and not allow_failed:
                raise WristError(f"主从腕已 fail-closed：{self._failure}") from self._failure
            raise RuntimeError("主从腕尚未连接")
        if self._owner_thread_id != threading.get_ident():
            raise DeviceOwnershipError("只有硬件 owner 线程可以执行主从腕 I/O")
        if self._failure is not None and not allow_failed:
            raise WristError(f"主从腕已 fail-closed：{self._failure}") from self._failure

    def _fail_closed(self, failure: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = failure
        if self._master is not None:
            with suppress(Exception):
                if self._master_active:
                    self._master.stop()
            with suppress(Exception):
                self._master.close()
        if self._wrist is not None:
            with suppress(Exception):
                self._wrist.stop()
            with suppress(Exception):
                self._wrist.close()
        self._master = self._wrist = None
        self._owner_thread_id = None
        self._master_active = False
        self._parked = True
