from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from examples.ur5_twinwrist.cameras.models import CameraFrame
from examples.ur5_twinwrist.controller.gripper import GripperState
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouseSample
from examples.ur5_twinwrist.controller.ur5 import JointVelocityResult
from examples.ur5_twinwrist.controller.ur5 import UR5State
from examples.ur5_twinwrist.teleop_hardware import TeleopHardwareError
from examples.ur5_twinwrist.teleop_hardware import TeleopHardwareWorker
from examples.ur5_twinwrist.teleop_hardware import spacemouse_wrist_velocity_deg_s


def _config(
    *,
    command_age_ms: float = 1_000.0,
    with_joint_limits: bool = True,
    home_stable_s: float = 0.01,
) -> dict[str, Any]:
    lower = [-6.2] * 6 if with_joint_limits else None
    upper = [6.2] * 6 if with_joint_limits else None
    return {
        "hardware": {
            "ur5": {"host": "192.0.2.10", "control_hz": 100.0},
            "wrist": {"command_hz": 50.0},
            "gripper": {"command_hz": 30.0},
            "cameras": {
                "devices": [
                    {"role": "front", "serial": "front-serial"},
                    {"role": "side", "serial": "side-serial"},
                    {"role": "top", "serial": "top-serial"},
                ]
            },
        },
        "safety": {
            "timing": {
                "max_command_age_ms": command_age_ms,
                "camera_timeout_ms": 100.0,
                "max_camera_skew_ms": 20.0,
            },
            "ur5": {
                "joint_min_rad": lower,
                "joint_max_rad": upper,
                "joint6_jog_speed_rad_s": 0.2,
            },
        },
        "poses": {
            "ur5": {
                "task_home_rad": [0.0] * 6,
                "home_speed_rad_s": 0.2,
                "home_tolerance_rad": 0.01,
                "home_stable_s": home_stable_s,
            }
        },
        "teleop": {
            "modes": {
                "wrist": {
                    "deadzone": 0.2,
                    "max_speed_deg_s": 40.0,
                    "cap_x_to_axis": "j2",
                    "cap_x_sign": -1,
                    "cap_y_to_axis": "j1",
                    "cap_y_sign": 1,
                }
            }
        },
    }


class _OwnerChecked:
    def __init__(self) -> None:
        self.owner: int | None = None
        self.io_threads: set[int] = set()
        self.stopped = False
        self.closed = False

    def _claim(self) -> None:
        self.owner = threading.get_ident()
        self.io_threads.add(threading.get_ident())

    def _io(self) -> None:
        self.io_threads.add(threading.get_ident())
        assert threading.get_ident() == self.owner

    def stop(self) -> None:
        self._io()
        self.stopped = True

    def close(self) -> None:
        self._io()
        self.closed = True

    def healthy(self) -> bool:
        return self.owner is not None and not self.closed


class FakeUR(_OwnerChecked):
    def __init__(self, *, fail_send: bool = False, home_pattern: list[bool] | None = None) -> None:
        super().__init__()
        self.fail_send = fail_send
        self.sent: list[tuple[float, ...]] = []
        self.home_pattern = list(home_pattern or [True])
        self.home_calls = 0

    def connect(self) -> None:
        self._claim()

    def read(self) -> UR5State:
        self._io()
        return UR5State(
            qpos_rad=np.arange(6, dtype=np.float64) / 10.0,
            tcp_pose=np.zeros(6),
            tcp_speed=np.zeros(6),
            monotonic_ns=time.monotonic_ns(),
            robot_mode=7,
            safety_mode=1,
            emergency_stopped=False,
            protective_stopped=False,
        )

    def send(self, twist: Any, *, state_timestamp_ns: int | None = None) -> np.ndarray:
        self._io()
        assert state_timestamp_ns is not None
        if self.fail_send:
            raise OSError("fake RTDE failure")
        result = np.asarray(twist, dtype=np.float64)
        self.sent.append(tuple(float(value) for value in result))
        return result

    def jog_joint6(
        self,
        direction: int,
        *,
        speed_rad_s: float,
        state_timestamp_ns: int | None = None,
    ) -> JointVelocityResult:
        self._io()
        velocity = np.zeros(6)
        velocity[5] = direction * speed_rad_s
        return JointVelocityResult(velocity, int(state_timestamp_ns or 0))

    def task_home_step(
        self,
        target_joints_rad: Any,
        *,
        max_speed_rad_s: float,
        tolerance_rad: float,
        proportional_gain: float = 1.5,
        state_timestamp_ns: int | None = None,
    ) -> JointVelocityResult:
        self._io()
        assert len(target_joints_rad) == 6
        assert max_speed_rad_s > 0.0
        assert tolerance_rad > 0.0
        assert proportional_gain > 0.0
        reached = self.home_pattern[min(self.home_calls, len(self.home_pattern) - 1)]
        self.home_calls += 1
        return JointVelocityResult(np.zeros(6), int(state_timestamp_ns or 0), reached=reached)


class FakeWrist(_OwnerChecked):
    def __init__(self) -> None:
        super().__init__()
        self.target_relative_raw = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.sequence = 0
        self.home_calls = 0
        self.resume_calls = 0

    def _state(self, mode: str) -> SimpleNamespace:
        self.sequence += 1
        quantized_target = tuple(float(round(value)) for value in self.target_relative_raw)
        motor_raw = (round(quantized_target[0] + 3278), round(quantized_target[1] + 2547))
        return SimpleNamespace(
            actual_relative_raw=quantized_target,
            target_relative_raw=quantized_target,
            output_state=SimpleNamespace(
                target_relative_raw=quantized_target,
                servo_zero_raw=(3278, 2547),
                hardware_limits_raw=((2781, 3568), (1822, 3333)),
                motor_position_raw=motor_raw,
                encoder_abs_deg=(66.77, 163.76),
                encoder_valid=(True, True),
            ),
            host_monotonic_ns=time.monotonic_ns(),
            sequence=self.sequence,
            mode=mode,
        )

    def connect(self, *, enable_motion: bool = False) -> SimpleNamespace:
        assert enable_motion
        self._claim()
        return self._state("parked")

    def resume_master(self) -> SimpleNamespace:
        self._io()
        self.resume_calls += 1
        return self._state("master")

    def set_spacemouse_velocity(self, j1_deg_s: float, j2_deg_s: float) -> None:
        self._io()
        self.velocity = (j1_deg_s, j2_deg_s)

    def clear_spacemouse_velocity(self) -> None:
        self._io()
        self.velocity = (0.0, 0.0)

    def step(self) -> SimpleNamespace:
        self._io()
        self.target_relative_raw = (
            self.velocity[0] * 10.0 / 50.0,
            self.velocity[1] * 14.0 / 50.0,
        )
        return self._state("spacemouse")

    def home(self, *, enable_motion: bool = False) -> SimpleNamespace:
        self._io()
        assert enable_motion
        self.home_calls += 1
        self.target_relative_raw = (0.0, 0.0)
        return self._state("parked")


class FakeGripper(_OwnerChecked):
    def __init__(self) -> None:
        super().__init__()
        self.position = 0.25
        self.target = 0.25
        self.sequence = 0

    def connect(self) -> None:
        self._claim()

    def get(self) -> GripperState:
        self._io()
        self.position = self.target
        self.sequence += 1
        return GripperState(
            position=self.position,
            target=self.target,
            raw_position=0,
            raw_target=0,
            temperature_c=25,
            voltage_v=7.4,
            torque_enabled=True,
            host_monotonic_ns=time.monotonic_ns(),
            sequence=self.sequence,
            backend="fake",
        )

    def get_position(self) -> float:
        return self.get().position

    def set(self, value: float) -> float:
        return self.set_position(value)

    def set_position(self, value: float) -> float:
        self._io()
        self.target = round(value * 10.0) / 10.0
        return self.target


class FakeCameras:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.connect_thread: int | None = None
        self.raise_on_read = False

    def connect(self) -> None:
        self.connect_thread = threading.get_ident()
        self.connected = True

    def read(self, timeout_s: float) -> dict[str, CameraFrame]:
        assert self.connected
        assert timeout_s > 0.0
        if self.raise_on_read:
            raise TimeoutError("fake camera timeout")
        now = time.monotonic_ns()
        return {
            role: CameraFrame(
                role=role,
                serial=f"{role}-serial",
                sequence=1,
                device_timestamp_ms=123.0 + index,
                host_timestamp_ns=now + index,
                color=np.full((3, 4, 3), index, dtype=np.uint8),
            )
            for index, role in enumerate(("front", "side", "top"))
        }

    def close(self) -> None:
        self.closed = True


class Rig:
    def __init__(self, *, fail_send: bool = False, home_pattern: list[bool] | None = None) -> None:
        self.ur = FakeUR(fail_send=fail_send, home_pattern=home_pattern)
        self.wrist = FakeWrist()
        self.gripper = FakeGripper()
        self.cameras = FakeCameras()
        self.factory_calls = 0

    def worker(self, config: dict[str, Any], *, enabled: bool = True) -> TeleopHardwareWorker:
        def ur_factory(_config: Any, *, enable_motion: bool) -> FakeUR:
            self.factory_calls += 1
            assert enable_motion is enabled
            return self.ur

        def wrist_factory(_config: Any) -> FakeWrist:
            self.factory_calls += 1
            return self.wrist

        def gripper_factory(_config: Any) -> FakeGripper:
            self.factory_calls += 1
            return self.gripper

        def camera_factory(values: Any) -> FakeCameras:
            self.factory_calls += 1
            assert values["max_camera_skew_ms"] == 20.0
            return self.cameras

        return TeleopHardwareWorker(
            config,
            enable_motion=enabled,
            ur_factory=ur_factory,
            wrist_factory=wrist_factory,
            gripper_factory=gripper_factory,
            camera_factory=camera_factory,
        )


def _wait_until(predicate: Any, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_construction_has_no_side_effect_and_motion_gate_is_strict() -> None:
    rig = Rig()
    worker = rig.worker(_config(), enabled=False)
    assert rig.factory_calls == 0
    with pytest.raises(PermissionError, match="enable_motion"):
        worker.connect()
    assert rig.factory_calls == 0


def test_real_motion_requires_explicit_joint_soft_limits() -> None:
    worker = Rig().worker(_config(with_joint_limits=False))
    with pytest.raises(ValueError, match="joint_min_rad"):
        worker.connect()


def test_owner_thread_cached_receipt_and_complete_observation() -> None:
    rig = Rig()
    worker = rig.worker(_config())
    worker.connect(timeout_s=1.0)
    command = (0.01, -0.02, 0.03, 0.1, -0.2, 0.3)
    worker.update_command(command, wrist_velocity_deg_s=(10.0, -5.0), gripper_target=0.8)

    def received_expected() -> bool:
        try:
            receipt = worker.latest_action_receipt()
        except RuntimeError:
            return False
        return receipt.action[:6] == command and receipt.action[8] == pytest.approx(0.8)

    _wait_until(received_expected)
    observation = worker.get_observation()
    assert observation["qpos"][:6] == pytest.approx(tuple(np.arange(6) / 10.0))
    assert observation["qpos"][6:8] == pytest.approx((2.0, -1.0))
    assert observation["qpos"][8] == pytest.approx(0.8)
    assert set(observation["images"]) == {"front", "side", "top"}
    assert all(frame.shape == (3, 4, 3) for frame in observation["images"].values())
    assert len(observation["action_receipt"].action) == 9
    assert observation["hardware_health"]["ok"]
    assert observation["hardware_health"]["gripper"]["state_source"] == "measured"
    assert set(observation["camera_timestamps_ns"]) == {"front", "side", "top"}
    assert observation["camera_sequences"] == {"front": 1, "side": 1, "top": 1}

    worker.close()
    actuator_thread = rig.ur.owner
    assert actuator_thread is not None
    assert actuator_thread != threading.get_ident()
    assert rig.ur.io_threads == {actuator_thread}
    assert rig.wrist.io_threads == {actuator_thread}
    assert rig.gripper.io_threads == {actuator_thread}
    assert rig.cameras.connect_thread == actuator_thread
    assert rig.ur.stopped
    assert rig.ur.closed
    assert rig.wrist.stopped
    assert rig.wrist.closed
    assert rig.gripper.stopped
    assert rig.gripper.closed
    assert rig.cameras.closed


def test_episode_gate_blocks_home_and_joint6_jog() -> None:
    rig = Rig()
    worker = rig.worker(_config())
    worker.connect(timeout_s=1.0)
    worker.set_episode_active(active=True)
    with pytest.raises(PermissionError, match="HOME"):
        worker.request_home()
    with pytest.raises(PermissionError, match="J6"):
        worker.set_joint6_jog(1)
    worker.set_joint6_jog(0)
    worker.set_episode_active(active=False)
    worker.request_home()
    _wait_until(lambda: rig.wrist.home_calls == 1)
    _wait_until(lambda: not worker.maintenance_status().active)
    worker.close()


def test_releasing_ctrl_rebases_and_resumes_master_following() -> None:
    rig = Rig()
    worker = rig.worker(_config())
    worker.connect(timeout_s=1.0)
    assert rig.wrist.resume_calls == 1  # connect 后首次启动主腕跟随。
    worker.update_command((0.0,) * 6, wrist_velocity_deg_s=(10.0, 0.0), gripper_target=0.25)
    _wait_until(lambda: rig.wrist.velocity == (10.0, 0.0))
    worker.update_command((0.0,) * 6, wrist_velocity_deg_s=None, gripper_target=0.25)
    _wait_until(lambda: rig.wrist.resume_calls == 2)
    assert rig.wrist.velocity == (0.0, 0.0)
    worker.close()


def test_full_home_stays_parked_until_explicit_master_resume_after_ctrl() -> None:
    rig = Rig()
    worker = rig.worker(_config(home_stable_s=0.05))
    worker.connect(timeout_s=1.0)
    worker.update_command((0.0,) * 6, wrist_velocity_deg_s=(10.0, 0.0), gripper_target=0.25)
    _wait_until(lambda: rig.wrist.velocity == (10.0, 0.0))
    worker.request_home()
    worker.update_command((0.0,) * 6, wrist_velocity_deg_s=None, gripper_target=0.25)
    _wait_until(lambda: rig.wrist.home_calls == 1)
    time.sleep(0.02)
    assert worker.maintenance_status().active
    assert rig.wrist.resume_calls == 1
    _wait_until(lambda: not worker.maintenance_status().active)
    assert rig.wrist.resume_calls == 1
    worker.request_wrist_master_resume()
    _wait_until(lambda: rig.wrist.resume_calls == 2)
    worker.close()


def test_action_receipt_uses_quantized_wrist_and_gripper_commands() -> None:
    rig = Rig()
    worker = rig.worker(_config())
    worker.connect(timeout_s=1.0)
    # Fake wrist 模拟 OpenRB 舵机 raw 整数量化, Fake gripper 模拟量化到 0.1。
    worker.update_command((0.0,) * 6, wrist_velocity_deg_s=(12.34, 0.0), gripper_target=0.83)

    def quantized_receipt_ready() -> bool:
        receipt = worker.latest_action_receipt()
        return math.isclose(receipt.action[8], 0.8, abs_tol=1e-12) and math.isclose(
            receipt.action[6],
            2.0,
            abs_tol=1e-12,
        )

    _wait_until(quantized_receipt_ready)
    receipt = worker.latest_action_receipt()
    assert receipt.action[6] != pytest.approx(12.34 * 10.0 / 50.0, abs=1e-8)
    assert receipt.action[8] != pytest.approx(0.83)
    worker.close()


def test_home_must_remain_reached_for_stable_window_and_resets_when_it_leaves() -> None:
    rig = Rig(home_pattern=[True, False, True])
    worker = rig.worker(_config(home_stable_s=0.025))
    worker.connect(timeout_s=1.0)
    worker.request_home()
    _wait_until(lambda: not worker.maintenance_status().active)
    # 第一次 reached 后立刻离开容差必须重置, 之后至少连续约 3 个 100Hz
    # 周期才能满足 25ms, 不能沿用第一次 reached 的计时。
    assert rig.ur.home_calls >= 5
    worker.close()


@pytest.mark.parametrize("failure_kind", ["stale", "nan", "driver"])
def test_stale_invalid_or_driver_failure_is_fail_closed(failure_kind: str) -> None:
    config = _config(command_age_ms=50.0 if failure_kind == "stale" else 1_000.0)
    rig = Rig(fail_send=failure_kind == "driver")
    worker = rig.worker(config)
    connect_failed = False
    try:
        worker.connect(timeout_s=1.0)
    except TeleopHardwareError:
        connect_failed = True
        if failure_kind != "driver":
            raise
    if connect_failed:
        assert rig.ur.stopped
        assert rig.wrist.stopped
        assert rig.gripper.stopped
        return
    if failure_kind == "nan":
        with pytest.raises(ValueError, match="NaN"):
            worker.update_command([0.0, 0.0, np.nan, 0.0, 0.0, 0.0], wrist_velocity_deg_s=None, gripper_target=0.0)

    def failed() -> bool:
        try:
            worker.raise_if_failed()
        except TeleopHardwareError:
            return True
        return False

    _wait_until(failed)
    worker.stop()
    assert rig.ur.stopped
    assert rig.wrist.stopped
    assert rig.gripper.stopped


def test_action_receipt_dataclasses_do_not_alias_input_arrays() -> None:
    rig = Rig()
    worker = rig.worker(_config())
    worker.connect(timeout_s=1.0)
    source = np.ones(6)
    command = worker.update_command(source, wrist_velocity_deg_s=None, gripper_target=0.4)
    source[:] = 99.0
    assert command.ur_speed_l == (1.0,) * 6
    worker.close()


def test_ctrl_maps_to_wrist_velocity_and_non_ctrl_keeps_master_following() -> None:
    def sample(*, ctrl: bool) -> SpaceMouseSample:
        return SpaceMouseSample(
            motion=np.asarray([0.6, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            buttons={},
            named_buttons={"ctrl": ctrl},
            monotonic_ns=time.monotonic_ns(),
            motion_timestamp_ns=time.monotonic_ns(),
            stale=False,
            connected=True,
        )

    assert spacemouse_wrist_velocity_deg_s(sample(ctrl=False), _config()) is None
    # deadzone=0.2: cap X→J2(-) 得 -20 deg/s, cap Y→J1(+) 得 -40 deg/s。
    velocity = spacemouse_wrist_velocity_deg_s(sample(ctrl=True), _config())
    assert velocity == pytest.approx((-40.0, -20.0))


def test_spacemouse_failure_requests_immediate_shutdown() -> None:
    rig = Rig()
    worker = rig.worker(_config())
    worker.connect(timeout_s=1.0)
    worker.report_input_failure(ConnectionError("SpaceMouse disconnected"))
    _wait_until(lambda: not worker.connected)
    with pytest.raises(TeleopHardwareError, match="SpaceMouse"):
        worker.raise_if_failed()
    assert rig.ur.stopped


def test_camera_failure_requests_immediate_shutdown() -> None:
    rig = Rig()
    worker = rig.worker(_config())
    worker.connect(timeout_s=1.0)
    rig.cameras.raise_on_read = True
    with pytest.raises(TimeoutError, match="camera"):
        worker.get_observation()
    _wait_until(lambda: not worker.connected)
    with pytest.raises(TeleopHardwareError, match="camera"):
        worker.raise_if_failed()
    assert rig.ur.stopped
