from dataclasses import dataclass
import threading

import pytest

from examples.ur5_twinwrist.gripper_worker import SingleOwnerGripper


@dataclass(frozen=True)
class _State:
    position: int
    normalized_position: float
    target_position: int
    normalized_target_position: float
    temperature_c: int = 25
    voltage_v: float = 7.4
    load_enabled: bool = True
    host_timestamp_s: float = 0.0
    sequence: int = 0


class _FakeSerialError(Exception):
    pass


class _FakeGripper:
    driver_name = "fake"
    port = "/dev/serial/by-id/fake"
    servo_id = 7
    open_position = 100
    closed_position = 900

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.target = 0.0
        self.sequence = 0
        self.read_count = 0
        self.fail_next_read = False

    def _record(self, operation: str) -> None:
        self.calls.append((operation, threading.get_ident()))

    def open(self) -> None:
        self._record("open")

    def close(self) -> None:
        self._record("close")

    def command(self, value: int) -> None:
        self._record("command")
        self.command_position(float(value))

    def command_position(self, value: float) -> None:
        self._record("command_position")
        self.target = value

    def unload(self) -> None:
        self._record("unload")

    def read_state(self, *, enforce_safety: bool = True) -> _State:
        assert enforce_safety
        self._record("read_state")
        self.read_count += 1
        if self.fail_next_read:
            self.fail_next_read = False
            raise _FakeSerialError("serial read failed")
        self.sequence += 1
        position = round(self.open_position + self.target * 800)
        return _State(
            position=position,
            normalized_position=self.target,
            target_position=position,
            normalized_target_position=self.target,
            sequence=self.sequence,
        )


def _proxy(delegate: _FakeGripper) -> SingleOwnerGripper:
    return SingleOwnerGripper(
        delegate,
        poll_interval_s=60.0,
        stale_timeout_s=120.0,
    )


def test_all_device_calls_use_one_owner_and_reads_use_cache():
    main_thread = threading.get_ident()
    delegate = _FakeGripper()
    gripper = _proxy(delegate)

    gripper.open()
    initial_reads = delegate.read_count
    for _ in range(10):
        assert gripper.read_state().normalized_position == 0.0
    assert delegate.read_count == initial_reads

    gripper.command_position(0.25)
    assert gripper.read_state().normalized_target_position == 0.25
    gripper.command(1)
    assert gripper.read_state(enforce_safety=False).normalized_position == 1.0
    gripper.unload()
    assert gripper.healthy()
    gripper.close()

    owner_threads = {thread_id for _, thread_id in delegate.calls}
    assert len(owner_threads) == 1
    assert main_thread not in owner_threads
    assert delegate.calls[0][0] == "open"
    assert delegate.calls[-1][0] == "close"
    assert gripper.driver_name == "fake"
    assert gripper.port == "/dev/serial/by-id/fake"
    assert gripper.servo_id == 7


def test_feedback_exception_propagates_and_permanently_fails_closed():
    delegate = _FakeGripper()
    gripper = _proxy(delegate)
    gripper.open()
    delegate.fail_next_read = True

    with pytest.raises(_FakeSerialError, match="serial read failed"):
        gripper.command_position(0.5)
    assert not gripper.healthy()
    with pytest.raises(_FakeSerialError, match="serial read failed"):
        gripper.read_state()
    with pytest.raises(_FakeSerialError, match="serial read failed"):
        gripper.command(0)
    with pytest.raises(_FakeSerialError, match="serial read failed"):
        gripper.close()

    owner_threads = {thread_id for _, thread_id in delegate.calls}
    assert len(owner_threads) == 1
    assert delegate.calls[-1][0] == "close"
