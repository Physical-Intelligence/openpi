# ruff: noqa: RUF001, RUF002
"""项目内幻尔与飞特单自由度夹爪驱动。

两个后端都使用同一归一化约定：``0.0`` 为完全打开，``1.0`` 为完全
闭合。模块导入不依赖 pyserial；只有没有注入 fake ``serial_factory``
且显式调用 ``connect()`` 时才延迟导入 pyserial。

该文件是底层串口拥有者，不是 UI 接口。``connect`` 所在线程必须独占
后续 I/O；采集器、控制器和 UI 应读取该拥有者发布的缓存，而不能并发
调用 ``get``。所有完整串口事务仍由可重入锁保护，协议或串口异常会
best-effort 关闭力矩并永久 fail-closed。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Protocol, runtime_checkable


class GripperError(RuntimeError):
    """夹爪连接、协议或安全检查失败。"""


class GripperTimeoutError(GripperError, TimeoutError):
    """夹爪没有在超时前返回合法数据。"""


class DeviceOwnershipError(GripperError):
    """非串口拥有线程试图执行 I/O。"""


def _require_by_id(port: str) -> str:
    candidate = Path(port)
    if candidate.parent != Path("/dev/serial/by-id") or not candidate.name or candidate.name in {".", ".."}:
        raise ValueError("夹爪串口必须使用 /dev/serial/by-id/<设备名>")
    return str(candidate)


def _normalise(raw: int, opened: int, closed: int) -> float:
    return float(min(1.0, max(0.0, (raw - opened) / (closed - opened))))


def _denormalise(value: float, opened: int, closed: int) -> int:
    return round(opened + value * (closed - opened))


def _finite_unit(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("夹爪位置必须是 [0, 1] 内的有限数")
    return result


@dataclass(frozen=True)
class GripperState:
    """两个后端共用的反馈结构。"""

    position: float
    target: float
    raw_position: int
    raw_target: int
    temperature_c: int
    voltage_v: float
    torque_enabled: bool
    host_monotonic_ns: int
    sequence: int
    backend: str
    moving: bool | None = None
    speed_raw: int | None = None
    load_raw: int | None = None
    current_raw: int | None = None


@runtime_checkable
class GripperInterface(Protocol):
    """上层只依赖此生命周期；具体协议由 backend 隔离。"""

    def connect(self) -> None: ...

    def get(self) -> GripperState: ...

    def get_position(self) -> float: ...

    def set(self, value: float) -> float: ...

    def set_position(self, value: float) -> float: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def healthy(self) -> bool: ...


@dataclass(frozen=True)
class HiwonderConfig:
    """幻尔/LewanSoul 总线舵机配置。"""

    port: str
    servo_id: int = 1
    baudrate: int = 115_200
    open_position: int = 630
    closed_position: int = 860
    feedback_position_bias: int = 19
    full_stroke_time_ms: int = 1000
    min_move_time_ms: int = 20
    max_temperature_c: int = 60
    timeout_s: float = 0.30

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _require_by_id(self.port))
        if not 0 <= self.servo_id <= 253:
            raise ValueError("servo_id 必须在 [0, 253]")
        if self.baudrate <= 0:
            raise ValueError("baudrate 必须为正数")
        if not 0 <= self.open_position <= 1000 or not 0 <= self.closed_position <= 1000:
            raise ValueError("幻尔端点必须在 [0, 1000]")
        if self.open_position == self.closed_position:
            raise ValueError("open_position 与 closed_position 不能相同")
        if not -1000 <= self.feedback_position_bias <= 1000:
            raise ValueError("feedback_position_bias 必须在 [-1000, 1000]")
        if not 1 <= self.min_move_time_ms <= self.full_stroke_time_ms <= 65_535:
            raise ValueError("移动时间必须满足 1 <= min <= full <= 65535")
        if self.max_temperature_c <= 0:
            raise ValueError("max_temperature_c 必须为正数")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s 必须是有限正数")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> HiwonderConfig:
        return cls(
            port=str(config["port"]),
            servo_id=int(config.get("servo_id", 1)),
            baudrate=int(config.get("baudrate", config.get("baud", 115_200))),
            open_position=int(config.get("open_position", 630)),
            closed_position=int(config.get("closed_position", 860)),
            feedback_position_bias=int(config.get("feedback_position_bias", 19)),
            full_stroke_time_ms=int(config.get("full_stroke_time_ms", 1000)),
            min_move_time_ms=int(config.get("min_move_time_ms", 20)),
            max_temperature_c=int(config.get("max_temperature_c", 60)),
            timeout_s=float(config.get("timeout_s", 0.30)),
        )


@dataclass(frozen=True)
class FeetechConfig:
    """飞特 STS3215 总线舵机配置。"""

    port: str
    servo_id: int = 13
    baudrate: int = 1_000_000
    open_position: int = 100
    closed_position: int = 3995
    speed: int = 250
    acceleration: int = 8
    max_temperature_c: int = 70
    max_load_raw: int = 750
    max_current_raw: int = 600
    timeout_s: float = 0.08

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _require_by_id(self.port))
        if not 0 <= self.servo_id <= 253:
            raise ValueError("servo_id 必须在 [0, 253]")
        if self.baudrate <= 0:
            raise ValueError("baudrate 必须为正数")
        if not 0 <= self.open_position <= 4095 or not 0 <= self.closed_position <= 4095:
            raise ValueError("飞特端点必须在 [0, 4095]")
        if self.open_position == self.closed_position:
            raise ValueError("open_position 与 closed_position 不能相同")
        if not 0 <= self.speed <= 65_535 or not 0 <= self.acceleration <= 255:
            raise ValueError("speed/acceleration 超出协议范围")
        if self.max_temperature_c <= 0:
            raise ValueError("max_temperature_c 必须为正数")
        if not 1 <= self.max_load_raw <= 1023 or not 1 <= self.max_current_raw <= 1023:
            raise ValueError("load/current 安全阈值必须在 [1, 1023]")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s 必须是有限正数")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> FeetechConfig:
        return cls(
            port=str(config["port"]),
            servo_id=int(config.get("servo_id", 13)),
            baudrate=int(config.get("baudrate", config.get("baud", 1_000_000))),
            open_position=int(config.get("open_position", 100)),
            closed_position=int(config.get("closed_position", 3995)),
            speed=int(config.get("speed", 250)),
            acceleration=int(config.get("acceleration", 8)),
            max_temperature_c=int(config.get("max_temperature_c", 70)),
            max_load_raw=int(config.get("max_load_raw", 750)),
            max_current_raw=int(config.get("max_current_raw", 600)),
            timeout_s=float(config.get("timeout_s", 0.08)),
        )


SerialFactory = Callable[..., Any]


class _OwnedSerialDevice:
    """两个夹爪后端共用的所有权与 fail-closed 基础设施。"""

    def __init__(self, *, serial_factory: SerialFactory | None) -> None:
        self._serial_factory = serial_factory
        self._serial: Any | None = None
        self._owner_thread_id: int | None = None
        self._lock = threading.RLock()
        self._failure: BaseException | None = None

    def _factory(self) -> SerialFactory:
        if self._serial_factory is not None:
            return self._serial_factory
        try:
            import serial  # pyserial 必须延迟导入
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial；请在机器人运行环境安装锁定版本") from exc
        return serial.Serial

    def _require_owner_unlocked(self, *, allow_failed: bool = False) -> Any:
        if self._serial is None:
            if self._failure is not None and not allow_failed:
                raise GripperError(f"夹爪已 fail-closed：{self._failure}") from self._failure
            raise RuntimeError("夹爪尚未连接")
        if self._owner_thread_id != threading.get_ident():
            raise DeviceOwnershipError("只有 connect() 所在线程可以访问夹爪串口")
        if self._failure is not None and not allow_failed:
            raise GripperError(f"夹爪已 fail-closed：{self._failure}") from self._failure
        return self._serial

    def _healthy(self) -> bool:
        with self._lock:
            return self._serial is not None and self._failure is None

    def _close_unlocked(self) -> None:
        port, self._serial = self._serial, None
        self._owner_thread_id = None
        if port is not None:
            port.close()

    def _trip_unlocked(self, failure: BaseException, torque_off_packet: bytes | None) -> None:
        if self._failure is None:
            self._failure = failure
        port, self._serial = self._serial, None
        self._owner_thread_id = None
        if port is None:
            return
        if torque_off_packet is not None:
            try:
                port.write(torque_off_packet)
                port.flush()
            except Exception:
                pass
        with suppress(Exception):
            port.close()


class HiwonderGripper(_OwnedSerialDevice):
    """幻尔/LewanSoul 串口总线夹爪。"""

    driver_name = "hiwonder"
    _MOVE_TIME_WRITE = 1
    _TEMP_READ = 26
    _VOLTAGE_READ = 27
    _POSITION_READ = 28
    _LOAD_WRITE = 31
    _LOAD_READ = 32

    def __init__(
        self,
        config: HiwonderConfig | Mapping[str, Any],
        *,
        serial_factory: SerialFactory | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__(serial_factory=serial_factory)
        self.config = config if isinstance(config, HiwonderConfig) else HiwonderConfig.from_mapping(config)
        self._monotonic_ns = monotonic_ns
        self._target_raw = self.config.open_position
        self._sequence = 0

    def connect(self) -> None:
        """打开串口并读取当前位置；不会发送运动目标。"""
        with self._lock:
            if self._serial is not None:
                raise RuntimeError("夹爪已经连接")
            if self._failure is not None:
                raise GripperError(f"夹爪已 fail-closed：{self._failure}") from self._failure
            try:
                self._serial = self._factory()(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.timeout_s,
                    write_timeout=self.config.timeout_s,
                )
                self._owner_thread_id = threading.get_ident()
                self._target_raw = self._read_register_unlocked(self._POSITION_READ, 2, signed=True)
            except Exception as exc:
                self._trip_unlocked(exc, self._packet(self._LOAD_WRITE, 0))
                raise

    def get(self) -> GripperState:
        """读取位置、温度、电压及力矩开关状态。"""
        with self._lock:
            self._require_owner_unlocked()
            try:
                raw_position = self._read_register_unlocked(self._POSITION_READ, 2, signed=True)
                temperature = self._read_register_unlocked(self._TEMP_READ, 1)
                voltage_mv = self._read_register_unlocked(self._VOLTAGE_READ, 2)
                load = self._read_register_unlocked(self._LOAD_READ, 1)
                if temperature >= self.config.max_temperature_c:
                    self._require_owner_unlocked().write(self._packet(self._LOAD_WRITE, 0))
                    self._require_owner_unlocked().flush()
                    raise GripperError(f"幻尔温度保护：{temperature}°C")
                self._sequence += 1
                return GripperState(
                    position=_normalise(
                        raw_position - self.config.feedback_position_bias,
                        self.config.open_position,
                        self.config.closed_position,
                    ),
                    target=_normalise(
                        self._target_raw,
                        self.config.open_position,
                        self.config.closed_position,
                    ),
                    raw_position=raw_position,
                    raw_target=self._target_raw,
                    temperature_c=temperature,
                    voltage_v=voltage_mv / 1000.0,
                    torque_enabled=bool(load),
                    host_monotonic_ns=self._monotonic_ns(),
                    sequence=self._sequence,
                    backend=self.driver_name,
                )
            except Exception as exc:
                self._trip_unlocked(exc, self._packet(self._LOAD_WRITE, 0))
                raise

    def get_position(self) -> float:
        """返回实际归一化位置。"""
        return self.get().position

    def set(self, value: float) -> float:
        """发送绝对位置，并返回实际下发 raw 目标对应的归一化值。"""
        position = _finite_unit(value)
        target = _denormalise(position, self.config.open_position, self.config.closed_position)
        with self._lock:
            self._require_owner_unlocked()
            try:
                state = self.get()
                if state.temperature_c >= self.config.max_temperature_c:
                    raise GripperError("幻尔温度保护已触发")
                span = abs(self.config.closed_position - self.config.open_position)
                duration = max(
                    self.config.min_move_time_ms,
                    round(abs(target - self._target_raw) / span * self.config.full_stroke_time_ms),
                )
                port = self._require_owner_unlocked()
                port.write(self._packet(self._LOAD_WRITE, 1))
                port.write(
                    self._packet(
                        self._MOVE_TIME_WRITE,
                        target & 0xFF,
                        (target >> 8) & 0xFF,
                        duration & 0xFF,
                        (duration >> 8) & 0xFF,
                    )
                )
                port.flush()
                self._target_raw = target
                return _normalise(target, self.config.open_position, self.config.closed_position)
            except Exception as exc:
                self._trip_unlocked(exc, self._packet(self._LOAD_WRITE, 0))
                raise

    def set_position(self, value: float) -> float:
        return self.set(value)

    def stop(self) -> None:
        """关闭舵机力矩，阻止继续执行目标。"""
        with self._lock:
            port = self._require_owner_unlocked(allow_failed=True)
            try:
                port.write(self._packet(self._LOAD_WRITE, 0))
                port.flush()
            except Exception as exc:
                self._trip_unlocked(exc, None)
                raise

    def close(self) -> None:
        """仅关闭串口；运动会话结束时上层应先调用 ``stop``。"""
        with self._lock:
            if self._serial is not None and self._owner_thread_id != threading.get_ident():
                raise DeviceOwnershipError("只有串口拥有线程可以关闭夹爪")
            self._close_unlocked()

    def healthy(self) -> bool:
        return self._healthy()

    @classmethod
    def packet(cls, servo_id: int, command: int, *data: int) -> bytes:
        if not 0 <= servo_id <= 253 or any(not 0 <= item <= 255 for item in (command, *data)):
            raise ValueError("幻尔 packet 字节超出范围")
        body = bytes((servo_id, len(data) + 3, command, *data))
        return b"\x55\x55" + body + bytes((~sum(body) & 0xFF,))

    def _packet(self, command: int, *data: int) -> bytes:
        return self.packet(self.config.servo_id, command, *data)

    def _read_register_unlocked(self, command: int, size: int, *, signed: bool = False) -> int:
        port = self._require_owner_unlocked()
        port.reset_input_buffer()
        port.write(self._packet(command))
        port.flush()
        reply = port.read(size + 6)
        if len(reply) != size + 6:
            raise GripperTimeoutError(f"幻尔 ID {self.config.servo_id} 读取 command={command} 超时")
        if reply[:2] != b"\x55\x55" or reply[2] != self.config.servo_id or reply[4] != command:
            raise GripperError(f"幻尔应答头错误：{reply.hex(' ')}")
        if reply[-1] != (~sum(reply[2:-1]) & 0xFF):
            raise GripperError("幻尔应答校验和错误")
        return int.from_bytes(reply[5:-1], "little", signed=signed)


class _FeetechBus:
    """STS/SCS 半双工包协议的最小实现。"""

    _INST_PING = 0x01
    _INST_READ = 0x02
    _INST_WRITE = 0x03
    INST_WRITE = _INST_WRITE

    def __init__(
        self,
        serial_port: Any,
        *,
        timeout_s: float,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        self.serial = serial_port
        self.timeout_s = timeout_s
        self._monotonic = monotonic
        self._sleep = sleeper

    @staticmethod
    def packet(servo_id: int, instruction: int, parameters: Iterable[int] = ()) -> bytes:
        params = list(parameters)
        if not 0 <= servo_id <= 254 or not 0 <= instruction <= 255:
            raise ValueError("飞特 ID/instruction 超出范围")
        if any(not 0 <= item <= 255 for item in params):
            raise ValueError("飞特 packet 参数必须为字节")
        body = [servo_id, len(params) + 2, instruction, *params]
        return bytes((0xFF, 0xFF, *body, (~sum(body)) & 0xFF))

    @staticmethod
    def _extract(data: bytes) -> bytes | None:
        for start in range(max(0, len(data) - 5)):
            if data[start : start + 2] != b"\xff\xff":
                continue
            length = data[start + 3]
            end = start + 4 + length
            if length >= 2 and end <= len(data):
                packet = data[start:end]
                if sum(packet[2:]) & 0xFF == 0xFF:
                    return packet
        return None

    def exchange(self, request: bytes) -> bytes:
        self.serial.reset_input_buffer()
        self.serial.write(request)
        self.serial.flush()
        deadline = self._monotonic() + self.timeout_s
        received = bytearray()
        while self._monotonic() < deadline:
            waiting = int(self.serial.in_waiting)
            if waiting:
                received.extend(self.serial.read(waiting))
                response = self._extract(bytes(received))
                if response is not None and response != request:
                    return response
            self._sleep(0.001)
        raise GripperTimeoutError(f"飞特应答超时：{received.hex(' ')}")

    def ping(self, servo_id: int) -> None:
        response = self.exchange(self.packet(servo_id, self._INST_PING))
        self._check(response, servo_id, 0)

    def read(self, servo_id: int, address: int, size: int) -> bytes:
        response = self.exchange(self.packet(servo_id, self._INST_READ, (address, size)))
        self._check(response, servo_id, size)
        return response[5:-1]

    def write(self, servo_id: int, address: int, data: Iterable[int]) -> None:
        response = self.exchange(self.packet(servo_id, self._INST_WRITE, (address, *data)))
        self._check(response, servo_id, 0)

    @staticmethod
    def _check(response: bytes, servo_id: int, size: int) -> None:
        if response[2] != servo_id:
            raise GripperError(f"飞特应答 ID 错误：{response[2]}")
        if response[4] != 0:
            raise GripperError(f"飞特舵机错误码：0x{response[4]:02x}")
        if len(response[5:-1]) != size:
            raise GripperError(f"飞特应答长度错误：期望 {size}，实际 {len(response[5:-1])}")


class FeetechGripper(_OwnedSerialDevice):
    """飞特 STS3215 夹爪。"""

    driver_name = "feetech"
    _ADDR_TORQUE_ENABLE = 40
    _ADDR_GOAL_ACCELERATION = 41
    _ADDR_PRESENT_POSITION = 56

    def __init__(
        self,
        config: FeetechConfig | Mapping[str, Any],
        *,
        serial_factory: SerialFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(serial_factory=serial_factory)
        self.config = config if isinstance(config, FeetechConfig) else FeetechConfig.from_mapping(config)
        self._monotonic = monotonic
        self._monotonic_ns = monotonic_ns
        self._sleep = sleeper
        self._bus: _FeetechBus | None = None
        self._target_raw = self.config.open_position
        self._sequence = 0

    def connect(self) -> None:
        """打开串口、PING 舵机并读取当前位置；不会发送运动目标。"""
        with self._lock:
            if self._serial is not None:
                raise RuntimeError("夹爪已经连接")
            if self._failure is not None:
                raise GripperError(f"夹爪已 fail-closed：{self._failure}") from self._failure
            try:
                self._serial = self._factory()(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    timeout=0,
                    write_timeout=self.config.timeout_s,
                )
                self._owner_thread_id = threading.get_ident()
                self._bus = _FeetechBus(
                    self._serial,
                    timeout_s=self.config.timeout_s,
                    monotonic=self._monotonic,
                    sleeper=self._sleep,
                )
                self._bus.ping(self.config.servo_id)
                self._target_raw = int.from_bytes(
                    self._bus.read(self.config.servo_id, self._ADDR_PRESENT_POSITION, 2), "little"
                )
            except Exception as exc:
                self._trip_feetech_unlocked(exc)
                raise

    def get(self) -> GripperState:
        """用一个连续寄存器块获取相干反馈，并检查温度/负载/电流。"""
        with self._lock:
            self._require_owner_unlocked()
            try:
                bus = self._require_bus_unlocked()
                data = bus.read(self.config.servo_id, self._ADDR_PRESENT_POSITION, 15)
                torque = bool(bus.read(self.config.servo_id, self._ADDR_TORQUE_ENABLE, 1)[0])
                raw_position = int.from_bytes(data[0:2], "little")
                speed = int.from_bytes(data[2:4], "little")
                load = int.from_bytes(data[4:6], "little")
                voltage = data[6] / 10.0
                temperature = data[7]
                moving = bool(data[10])
                current = int.from_bytes(data[13:15], "little")
                failures = []
                if temperature >= self.config.max_temperature_c:
                    failures.append(f"temperature={temperature}C")
                if load & 0x03FF >= self.config.max_load_raw:
                    failures.append(f"load={load & 0x03FF}")
                if current & 0x03FF >= self.config.max_current_raw:
                    failures.append(f"current={current & 0x03FF}")
                if failures:
                    raise GripperError("飞特安全保护：" + ", ".join(failures))
                self._sequence += 1
                return GripperState(
                    position=_normalise(raw_position, self.config.open_position, self.config.closed_position),
                    target=_normalise(self._target_raw, self.config.open_position, self.config.closed_position),
                    raw_position=raw_position,
                    raw_target=self._target_raw,
                    temperature_c=temperature,
                    voltage_v=voltage,
                    torque_enabled=torque,
                    host_monotonic_ns=self._monotonic_ns(),
                    sequence=self._sequence,
                    backend=self.driver_name,
                    moving=moving,
                    speed_raw=speed,
                    load_raw=load,
                    current_raw=current,
                )
            except Exception as exc:
                self._trip_feetech_unlocked(exc)
                raise

    def get_position(self) -> float:
        return self.get().position

    def set(self, value: float) -> float:
        """发送绝对位置，并返回实际下发 raw 目标对应的归一化值。"""
        position = _finite_unit(value)
        target = _denormalise(position, self.config.open_position, self.config.closed_position)
        with self._lock:
            self._require_owner_unlocked()
            try:
                state = self.get()
                bus = self._require_bus_unlocked()
                if not state.torque_enabled:
                    bus.write(self.config.servo_id, self._ADDR_TORQUE_ENABLE, (1,))
                bus.write(
                    self.config.servo_id,
                    self._ADDR_GOAL_ACCELERATION,
                    (
                        self.config.acceleration,
                        target & 0xFF,
                        (target >> 8) & 0xFF,
                        0,
                        0,
                        self.config.speed & 0xFF,
                        (self.config.speed >> 8) & 0xFF,
                    ),
                )
                self._target_raw = target
                return _normalise(target, self.config.open_position, self.config.closed_position)
            except Exception as exc:
                self._trip_feetech_unlocked(exc)
                raise

    def set_position(self, value: float) -> float:
        return self.set(value)

    def stop(self) -> None:
        """关闭 STS3215 力矩。"""
        with self._lock:
            self._require_owner_unlocked(allow_failed=True)
            try:
                self._require_bus_unlocked().write(self.config.servo_id, self._ADDR_TORQUE_ENABLE, (0,))
            except Exception as exc:
                self._trip_feetech_unlocked(exc)
                raise

    def close(self) -> None:
        with self._lock:
            if self._serial is not None and self._owner_thread_id != threading.get_ident():
                raise DeviceOwnershipError("只有串口拥有线程可以关闭夹爪")
            self._bus = None
            self._close_unlocked()

    def healthy(self) -> bool:
        return self._healthy()

    def _require_bus_unlocked(self) -> _FeetechBus:
        if self._bus is None:
            raise RuntimeError("飞特总线尚未连接")
        return self._bus

    def _torque_off_packet(self) -> bytes:
        return _FeetechBus.packet(
            self.config.servo_id,
            _FeetechBus.INST_WRITE,
            (self._ADDR_TORQUE_ENABLE, 0),
        )

    def _trip_feetech_unlocked(self, failure: BaseException) -> None:
        self._bus = None
        self._trip_unlocked(failure, self._torque_off_packet())


def create_gripper(
    config: Mapping[str, Any],
    *,
    serial_factory: SerialFactory | None = None,
) -> GripperInterface:
    """按 ``backend``/``driver`` 选择后端；不会自动连接或运动。"""
    if isinstance(config.get("hardware"), Mapping):
        hardware = config["hardware"]
        assert isinstance(hardware, Mapping)
        nested = hardware.get("gripper")
        if not isinstance(nested, Mapping):
            raise ValueError("hardware.gripper 必须是 mapping")
        config = nested
    backend = str(config.get("backend", config.get("driver", ""))).strip().lower()
    adapters = config.get("adapters")
    selected: Mapping[str, Any] = config
    if isinstance(adapters, Mapping):
        candidate = adapters.get(backend)
        if candidate is None and backend in {"feetech", "feetech_sts3215", "sts3215"}:
            candidate = adapters.get("feetech_sts3215", adapters.get("feetech"))
        if candidate is None and backend in {"hiwonder", "lewansoul", "lobot"}:
            candidate = adapters.get("hiwonder")
        if not isinstance(candidate, Mapping):
            raise ValueError(f"gripper.adapters 缺少 backend={backend!r} 的配置")
        selected = candidate
    if backend in {"hiwonder", "lewansoul", "lobot"}:
        return HiwonderGripper(selected, serial_factory=serial_factory)
    if backend in {"feetech", "feetech_sts3215", "sts3215"}:
        return FeetechGripper(selected, serial_factory=serial_factory)
    raise ValueError("gripper backend 必须是 hiwonder 或 feetech")
