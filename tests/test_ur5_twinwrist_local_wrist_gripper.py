from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from examples.ur5_twinwrist.controller.gripper import DeviceOwnershipError as GripperOwnershipError
from examples.ur5_twinwrist.controller.gripper import FeetechConfig
from examples.ur5_twinwrist.controller.gripper import FeetechGripper
from examples.ur5_twinwrist.controller.gripper import GripperError
from examples.ur5_twinwrist.controller.gripper import HiwonderConfig
from examples.ur5_twinwrist.controller.gripper import HiwonderGripper
from examples.ur5_twinwrist.controller.gripper import create_gripper
from examples.ur5_twinwrist.controller.wrist import DeviceOwnershipError as WristOwnershipError
from examples.ur5_twinwrist.controller.wrist import MasterWristConfig
from examples.ur5_twinwrist.controller.wrist import MasterWristReader
from examples.ur5_twinwrist.controller.wrist import MasterWristState
from examples.ur5_twinwrist.controller.wrist import OpenRBWrist
from examples.ur5_twinwrist.controller.wrist import WristConfig
from examples.ur5_twinwrist.controller.wrist import WristMasterSlaveController
from examples.ur5_twinwrist.controller.wrist import WristState

_BY_ID = "/dev/serial/by-id/fake-device"


class _HiwonderSerial:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.writes: list[bytes] = []
        self.pending = b""
        self.closed = False
        self.position = 649
        self.temperature = 25
        self.voltage_mv = 7400
        self.torque = 1
        self.fail_reads = False

    @staticmethod
    def _reply(servo_id: int, command: int, data: bytes) -> bytes:
        body = bytes((servo_id, len(data) + 3, command)) + data
        return b"\x55\x55" + body + bytes((~sum(body) & 0xFF,))

    def reset_input_buffer(self) -> None:
        self.pending = b""

    def write(self, packet: bytes) -> int:
        self.writes.append(packet)
        if packet[:2] != b"\x55\x55":
            return len(packet)
        servo_id, command = packet[2], packet[4]
        if command == 28:
            self.pending = self._reply(servo_id, command, self.position.to_bytes(2, "little", signed=True))
        elif command == 26:
            self.pending = self._reply(servo_id, command, bytes((self.temperature,)))
        elif command == 27:
            self.pending = self._reply(servo_id, command, self.voltage_mv.to_bytes(2, "little"))
        elif command == 32:
            self.pending = self._reply(servo_id, command, bytes((self.torque,)))
        elif command == 31:
            self.torque = packet[5]
        elif command == 1:
            # 现场标定显示反馈值比命令值稳定高 19 raw。
            self.position = int.from_bytes(packet[5:7], "little") + 19
        return len(packet)

    def flush(self) -> None:
        return None

    def read(self, size: int) -> bytes:
        if self.fail_reads:
            return b""
        result, self.pending = self.pending[:size], self.pending[size:]
        return result

    def close(self) -> None:
        self.closed = True


def _feetech_status(servo_id: int, data: bytes = b"", error: int = 0) -> bytes:
    body = bytes((servo_id, len(data) + 2, error)) + data
    return b"\xff\xff" + body + bytes((~sum(body) & 0xFF,))


class _FeetechSerial:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.writes: list[bytes] = []
        self.pending = b""
        self.closed = False
        self.position = 100
        self.torque = 1
        self.temperature = 25
        self.load = 10
        self.current = 12

    @property
    def in_waiting(self) -> int:
        return len(self.pending)

    def reset_input_buffer(self) -> None:
        self.pending = b""

    def write(self, packet: bytes) -> int:
        self.writes.append(packet)
        servo_id, instruction = packet[2], packet[4]
        parameters = packet[5:-1]
        if instruction == 1:
            self.pending = _feetech_status(servo_id)
        elif instruction == 2:
            address, size = parameters
            self.pending = _feetech_status(servo_id, self._read_registers(address, size))
        elif instruction == 3:
            address, payload = parameters[0], parameters[1:]
            if address == 40:
                self.torque = payload[0]
            elif address == 41:
                self.position = int.from_bytes(payload[1:3], "little")
            self.pending = _feetech_status(servo_id)
        return len(packet)

    def _read_registers(self, address: int, size: int) -> bytes:
        if address == 40 and size == 1:
            return bytes((self.torque,))
        if address == 56 and size == 2:
            return self.position.to_bytes(2, "little")
        if address == 56 and size == 15:
            data = bytearray(15)
            data[0:2] = self.position.to_bytes(2, "little")
            data[2:4] = (3).to_bytes(2, "little")
            data[4:6] = self.load.to_bytes(2, "little")
            data[6] = 74
            data[7] = self.temperature
            data[10] = 0
            data[13:15] = self.current.to_bytes(2, "little")
            return bytes(data)
        raise AssertionError(f"unexpected read address={address} size={size}")

    def flush(self) -> None:
        return None

    def read(self, size: int) -> bytes:
        result, self.pending = self.pending[:size], self.pending[size:]
        return result

    def close(self) -> None:
        self.closed = True


class _WristSerial:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.writes: list[str] = []
        self.pending = b""
        self.closed = False
        self.target = (3278, 2547)
        self.motion_reads = 0
        self.fail_on: str | None = None
        self.state_age_ms = 1

    def reset_input_buffer(self) -> None:
        self.pending = b""

    def write(self, data: bytes) -> int:
        command = data.decode("ascii").strip()
        self.writes.append(command)
        if command == self.fail_on:
            raise OSError("fake serial disconnect")
        name = command.split()[0]
        if name == "GET_LIMITS":
            fields = "j1_min=2781 j1_max=3568 j1_zero=3278 j2_min=1822 j2_max=3333 j2_zero=2547"
        elif name == "GET_MOTION_STATUS":
            self.motion_reads += 1
            active = 1 if self.motion_reads == 1 else 0
            fields = f"active={active} status={'active' if active else 'complete'} error_reason=none"
        elif name == "GET_WRIST_STATE":
            fields = (
                f"seq=7 age_ms=1 joint_age_ms={self.state_age_ms} "
                f"j1_pos={self.target[0]} j1_ok=1 j2_pos={self.target[1]} j2_ok=1 "
                "enc0_deg=6677 enc0_ok=1 enc1_deg=16376 enc1_ok=1 st=0"
            )
        else:
            if name in {"SET_ARM_TARGET_STREAM", "START_ARM_MOVE_TO"}:
                self.target = (int(command.split()[1]), int(command.split()[2]))
            fields = "accepted=1"
        self.pending = f"OK {name} {fields}\n".encode("ascii")
        return len(data)

    def flush(self) -> None:
        return None

    def readline(self) -> bytes:
        result, self.pending = self.pending, b""
        return result

    def close(self) -> None:
        self.closed = True


def test_by_id_paths_are_mandatory() -> None:
    with pytest.raises(ValueError, match="by-id"):
        HiwonderConfig(port="/dev/ttyUSB0")
    with pytest.raises(ValueError, match="by-id"):
        FeetechConfig(port="/dev/ttyUSB1")
    with pytest.raises(ValueError, match="by-id"):
        WristConfig(port="/dev/ttyACM0", servo_zero_raw=(3278, 2547))


def test_project_yaml_shape_maps_to_local_controllers() -> None:
    project = {
        "hardware": {
            "wrist": {
                "controller_port": _BY_ID,
                "baud": 115_200,
                "read_timeout_s": 0.2,
            },
            "gripper": {
                "backend": "hiwonder",
                "adapters": {"hiwonder": {"port": _BY_ID}},
            },
        },
        "safety": {
            "wrist": {
                "j1_min_raw": 2781,
                "j1_max_raw": 3568,
                "j2_min_raw": 1822,
                "j2_max_raw": 3333,
                "max_step_raw": 18,
                "feedback_timeout_s": 0.15,
            }
        },
        "poses": {"wrist": {"servo_zero_raw": [3278, 2547]}},
    }

    wrist = WristConfig.from_mapping(project)
    assert wrist.relative_min_raw == pytest.approx((-497.0, -725.0))
    assert wrist.relative_max_raw == pytest.approx((290.0, 786.0))
    assert wrist.max_step_raw == 18
    assert wrist.servo_zero_raw == (3278, 2547)
    assert wrist.state_max_age_s == pytest.approx(0.15)
    assert isinstance(create_gripper(project, serial_factory=lambda **_kwargs: _HiwonderSerial()), HiwonderGripper)


def test_hiwonder_protocol_normalisation_and_stop() -> None:
    fake = _HiwonderSerial()
    gripper = HiwonderGripper(HiwonderConfig(port=_BY_ID), serial_factory=lambda **_kwargs: fake)

    gripper.connect()
    assert gripper.healthy()
    state = gripper.get()
    assert state.position == pytest.approx(0.0)
    assert state.voltage_v == pytest.approx(7.4)

    sent = gripper.set(0.501)
    assert sent == pytest.approx(0.5)
    assert gripper.get_position() == pytest.approx(0.5, abs=0.01)
    assert fake.torque == 1
    gripper.stop()
    assert fake.torque == 0
    gripper.close()
    assert fake.closed


def test_hiwonder_disconnect_fails_closed_and_cannot_reopen() -> None:
    fake = _HiwonderSerial()
    gripper = HiwonderGripper(HiwonderConfig(port=_BY_ID), serial_factory=lambda **_kwargs: fake)
    gripper.connect()
    fake.fail_reads = True

    with pytest.raises(Exception, match="超时"):
        gripper.get()
    assert fake.closed
    assert fake.torque == 0
    assert not gripper.healthy()
    with pytest.raises(GripperError, match="fail-closed"):
        gripper.get()


def test_feetech_protocol_safety_and_factory() -> None:
    fake = _FeetechSerial()
    config = {
        "backend": "feetech",
        "port": _BY_ID,
        "servo_id": 13,
        "open_position": 100,
        "closed_position": 3995,
    }
    gripper = create_gripper(config, serial_factory=lambda **_kwargs: fake)
    assert isinstance(gripper, FeetechGripper)

    gripper.connect()
    sent = gripper.set_position(0.501)
    assert sent == pytest.approx((fake.position - 100) / (3995 - 100))
    assert fake.position == round(100 + 0.501 * (3995 - 100))
    assert gripper.get_position() == pytest.approx(sent)
    gripper.stop()
    assert fake.torque == 0
    gripper.close()
    assert fake.closed


def test_feetech_overtemperature_fails_closed() -> None:
    fake = _FeetechSerial()
    gripper = FeetechGripper(FeetechConfig(port=_BY_ID), serial_factory=lambda **_kwargs: fake)
    gripper.connect()
    fake.temperature = 80

    with pytest.raises(GripperError, match="temperature"):
        gripper.get()
    assert not gripper.healthy()
    assert fake.closed
    assert fake.torque == 0


def test_serial_owner_is_the_connecting_thread() -> None:
    fake = _HiwonderSerial()
    gripper = HiwonderGripper(HiwonderConfig(port=_BY_ID), serial_factory=lambda **_kwargs: fake)
    gripper.connect()
    errors: list[BaseException] = []
    thread = threading.Thread(target=lambda: _capture_error(gripper.get, errors))
    thread.start()
    thread.join()
    assert isinstance(errors[0], GripperOwnershipError)
    assert gripper.healthy()
    gripper.close()

    wrist_serial = _WristSerial()
    wrist = OpenRBWrist(
        WristConfig(port=_BY_ID, servo_zero_raw=(3278, 2547)),
        serial_factory=lambda **_kwargs: wrist_serial,
    )
    wrist.connect()
    errors = []
    thread = threading.Thread(target=lambda: _capture_error(wrist.get, errors))
    thread.start()
    thread.join()
    assert isinstance(errors[0], WristOwnershipError)
    assert wrist.healthy()
    wrist.close()


def _capture_error(operation: Any, errors: list[BaseException]) -> None:
    try:
        operation()
    except BaseException as exc:
        errors.append(exc)


def test_wrist_connect_is_read_only_and_home_requires_explicit_motion_gate() -> None:
    fake = _WristSerial()
    wrist = OpenRBWrist(
        WristConfig(port=_BY_ID, servo_zero_raw=(3278, 2547)),
        serial_factory=lambda **_kwargs: fake,
    )
    wrist.connect()
    with pytest.raises(PermissionError, match="enable_motion"):
        wrist.home()
    assert fake.writes == ["GET_LIMITS", "GET_LIMITS", "GET_WRIST_STATE"]
    wrist.close()


def test_wrist_yaml_servo_zero_set_and_stop_protocol() -> None:
    fake = _WristSerial()
    wrist = OpenRBWrist(
        WristConfig(port=_BY_ID, servo_zero_raw=(3278, 2547)),
        serial_factory=lambda **_kwargs: fake,
    )
    wrist.connect()

    state = wrist.home(enable_motion=True)
    assert state.position_relative_raw == pytest.approx((0.0, 0.0))
    assert state.servo_zero_raw == (3278, 2547)
    assert state.encoder_abs_deg == pytest.approx((66.77, 163.76))
    assert "START_ARM_MOVE_TO 3278 2547" in fake.writes
    assert "SET_OUTPUT_CL_ENABLE 0 CONFIRM" in fake.writes
    assert not any(command.startswith("SET_OUTPUT_ZERO") for command in fake.writes)

    state_reads_before_set = fake.writes.count("GET_WRIST_STATE")
    sent = wrist.set((10.4, -10.4))
    assert fake.target == (3288, 2537)
    assert sent == pytest.approx((10.0, -10.0))
    assert fake.writes.count("GET_WRIST_STATE") == state_reads_before_set
    with pytest.raises(ValueError, match="单步"):
        wrist.set((40.0, 0.0))
    wrist.stop()
    assert fake.writes[-3:] == ["STOP_MOTION", "STOP_ALL_VELOCITY", "HOLD_ALL"]
    wrist.close()


def test_wrist_serial_exception_stops_and_fails_closed() -> None:
    fake = _WristSerial()
    wrist = OpenRBWrist(
        WristConfig(port=_BY_ID, servo_zero_raw=(3278, 2547)),
        serial_factory=lambda **_kwargs: fake,
    )
    wrist.connect()
    wrist.home(enable_motion=True)
    wrist.set((1.0, 0.0))
    fake.fail_on = "GET_WRIST_STATE"

    with pytest.raises(OSError, match="disconnect"):
        wrist.get()
    assert not wrist.healthy()
    assert fake.closed
    assert fake.writes[-3:] == ["STOP_MOTION", "STOP_ALL_VELOCITY", "HOLD_ALL"]


def test_wrist_negative_source_age_fails_closed() -> None:
    fake = _WristSerial()
    wrist = OpenRBWrist(
        WristConfig(port=_BY_ID, servo_zero_raw=(3278, 2547)),
        serial_factory=lambda **_kwargs: fake,
    )
    wrist.connect()
    fake.state_age_ms = -1

    with pytest.raises(Exception, match="状态过期"):
        wrist.get()
    assert not wrist.healthy()
    assert fake.closed


class _MasterSerial:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.writes: list[str] = []
        self.pending: list[bytes] = []
        self.telemetry: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        command = data.decode("ascii").strip()
        self.writes.append(command)
        name = command.split()[0]
        if name == "STOP":
            self.pending.append(b"OK STOP sending=0\n")
        elif name == "SET_PERIOD":
            self.pending.append(b"OK SET_PERIOD period_ms=10\n")
        elif name == "START":
            self.pending.append(b"OK START seq=1 enc0_deg=1000 enc1_deg=2000 zero_valid=1\n")
        return len(data)

    def flush(self) -> None:
        return None

    def readline(self) -> bytes:
        if self.pending:
            return self.pending.pop(0)
        if self.telemetry:
            return self.telemetry.pop(0)
        return b""

    def close(self) -> None:
        self.closed = True


def test_master_wrist_reader_protocol_dynamic_zero_and_mapping() -> None:
    fake = _MasterSerial()
    reader = MasterWristReader(MasterWristConfig(port=_BY_ID), serial_factory=lambda **_kwargs: fake)

    reader.connect()
    baseline = reader.start_stream()
    assert baseline.target_relative_raw == pytest.approx((0.0, 0.0))
    fake.telemetry.append(b"OK TELE seq=2 enc0_deg=1100 enc1_deg=1800 zero_valid=1 age_ms=1\n")
    state = reader.get()
    assert state.target_relative_raw == pytest.approx((5.0, -21.0))
    assert state.encoder_relative_deg == pytest.approx((1.0, -2.0))
    assert fake.writes[:3] == ["STOP", "SET_PERIOD 10", "START"]
    reader.stop()
    reader.close()
    assert fake.closed


def test_master_wrist_duplicate_sequence_fails_closed() -> None:
    fake = _MasterSerial()
    reader = MasterWristReader(MasterWristConfig(port=_BY_ID), serial_factory=lambda **_kwargs: fake)
    reader.connect()
    reader.start_stream()
    fake.telemetry.append(b"OK TELE seq=1 enc0_deg=1001 enc1_deg=2001 zero_valid=1\n")

    with pytest.raises(Exception, match="sequence"):
        reader.get()
    assert not reader.healthy()
    assert fake.closed
    assert fake.writes[-1] == "STOP"


@pytest.mark.parametrize(
    ("telemetry", "message"),
    [
        (b"OK TELE seq=2 enc0_deg=1001 enc1_deg=2001 age_ms=1\n", "zero_valid"),
        (b"OK TELE seq=2 enc0_deg=1001 enc1_deg=2001 zero_valid=1\n", "age_ms"),
    ],
)
def test_master_wrist_requires_zero_and_age_evidence(telemetry: bytes, message: str) -> None:
    fake = _MasterSerial()
    reader = MasterWristReader(MasterWristConfig(port=_BY_ID), serial_factory=lambda **_kwargs: fake)
    reader.connect()
    reader.start_stream()
    fake.telemetry.append(telemetry)

    with pytest.raises(Exception, match=message):
        reader.get()
    assert not reader.healthy()
    assert fake.closed
    assert fake.writes[-1] == "STOP"


class _FakeOutputWrist:
    def __init__(self, config: WristConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, int]] = []
        self.target = (0.0, 0.0)
        self.sequence = 0

    def _record(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))

    def connect(self) -> None:
        self._record("connect")

    def home(self, *, enable_motion: bool = False) -> WristState:
        assert enable_motion
        self._record("home")
        self.target = (0.0, 0.0)
        return self.get()

    def set(self, target: tuple[float, float]) -> tuple[float, float]:
        self._record("set")
        self.target = tuple(target)
        return self.target

    def get(self) -> WristState:
        self._record("get")
        self.sequence += 1
        absolute = (
            round(self.target[0] + self.config.servo_zero_raw[0]),
            round(self.target[1] + self.config.servo_zero_raw[1]),
        )
        return WristState(
            position_relative_raw=self.target,
            target_relative_raw=self.target,
            encoder_abs_deg=(66.77, 163.76),
            encoder_valid=(True, True),
            motor_position_raw=absolute,
            motor_goal_raw=absolute,
            servo_zero_raw=self.config.servo_zero_raw,
            hardware_limits_raw=(
                (self.config.j1_min_raw, self.config.j1_max_raw),
                (self.config.j2_min_raw, self.config.j2_max_raw),
            ),
            host_monotonic_ns=time.monotonic_ns(),
            board_ms=100,
            sequence=self.sequence,
            source_age_s=0.001,
            active=True,
            zero_valid=True,
            fault=False,
            fault_reason="none",
        )

    def stop(self) -> None:
        self._record("stop")

    def close(self) -> None:
        self._record("close")


class _FakeMasterReader:
    def __init__(self, config: MasterWristConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, int]] = []
        self.sequence = 0
        self.target_relative_raw = (20.0, -10.0)

    def _record(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))

    def connect(self) -> None:
        self._record("connect")

    def start_stream(self) -> MasterWristState:
        self._record("start_stream")
        self.sequence += 1
        return self._state((0.0, 0.0))

    def get(self) -> MasterWristState:
        self._record("get")
        self.sequence += 1
        return self._state(self.target_relative_raw)

    def _state(self, value_raw: tuple[float, float]) -> MasterWristState:
        return MasterWristState(
            target_relative_raw=value_raw,
            encoder_abs_deg=(10.0, 20.0),
            encoder_relative_deg=(value_raw[0] / 10.0, value_raw[1] / 14.0),
            host_monotonic_ns=time.monotonic_ns(),
            sequence=self.sequence,
            board_sequence=self.sequence,
            zero_valid=True,
        )

    def stop(self) -> None:
        self._record("stop")

    def close(self) -> None:
        self._record("close")


def test_master_slave_controller_is_parked_after_home_and_arbitrates_targets() -> None:
    clock = [10.0]
    wrist_config = WristConfig(port=_BY_ID, servo_zero_raw=(3278, 2547), max_step_raw=5)
    master_config = MasterWristConfig(
        port="/dev/serial/by-id/fake-master",
        target_deadband_raw=0,
        max_velocity_raw_s=1_000_000.0,
        max_accel_raw_s2=1_000_000_000.0,
        max_jerk_raw_s3=1_000_000_000_000.0,
    )
    output_devices: list[_FakeOutputWrist] = []
    master_devices: list[_FakeMasterReader] = []

    def output_factory(config: WristConfig) -> _FakeOutputWrist:
        output_devices.append(_FakeOutputWrist(config))
        return output_devices[-1]

    def master_factory(config: MasterWristConfig) -> _FakeMasterReader:
        master_devices.append(_FakeMasterReader(config))
        return master_devices[-1]

    controller = WristMasterSlaveController(
        wrist_config=wrist_config,
        master_config=master_config,
        wrist_factory=output_factory,
        master_factory=master_factory,
        monotonic=lambda: clock[0],
    )
    with pytest.raises(PermissionError, match="enable_motion"):
        controller.connect()
    assert not output_devices
    assert not master_devices

    initial = controller.connect(enable_motion=True)
    assert initial.mode == "parked"
    assert not any(name == "start_stream" for name, _thread in master_devices[0].calls)
    controller.resume_master()
    clock[0] += 0.02
    followed = controller.step()
    assert followed.mode == "master"
    assert 0.0 < followed.target_relative_raw[0] <= 5.0
    assert -5.0 <= followed.target_relative_raw[1] < 0.0
    output_gets = sum(name == "get" for name, _thread in output_devices[0].calls)
    for _ in range(3):
        clock[0] += 0.02
        controller.step()
    assert sum(name == "get" for name, _thread in output_devices[0].calls) == output_gets
    clock[0] += 0.04
    controller.step()
    assert sum(name == "get" for name, _thread in output_devices[0].calls) == output_gets + 1

    controller.set_spacemouse_velocity(45.0, 0.0)
    clock[0] += 0.02
    overridden = controller.step()
    assert overridden.mode == "spacemouse"
    controller.clear_spacemouse_velocity()
    clock[0] += 0.02
    assert controller.step().mode == "spacemouse"

    # Rebinding must STOP the existing TELE stream before START. Repeating
    # START while the ESP32 is still streaming can consume a stale baseline.
    controller.resume_master()
    assert [name for name, _thread in master_devices[0].calls[-2:]] == ["stop", "start_stream"]

    controller.set_policy_target((100.0, 0.0))
    clock[0] += 0.02
    assert controller.step().mode == "policy"
    controller.home(enable_motion=True)
    assert controller.get().mode == "parked"
    controller.close()

    owner_threads = {thread_id for device in (*output_devices, *master_devices) for _, thread_id in device.calls}
    assert owner_threads == {threading.get_ident()}
