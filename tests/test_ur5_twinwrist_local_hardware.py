# ruff: noqa: RUF001, RUF002
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from examples.ur5_twinwrist.cameras.models import CameraFrame
from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.local_hardware import HardwareFactories
from examples.ur5_twinwrist.local_hardware import LocalHardwareConfig
from examples.ur5_twinwrist.local_hardware import LocalRobotHardware


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 10_000_000_000

    def __call__(self) -> int:
        return self.now_ns


class _FakeUR5:
    def __init__(self, config, log, clock) -> None:
        self.config = config
        self.log = log
        self.clock = clock
        self.connected = False
        self.stop_count = 0
        self.home_reached = False
        self.last_jog = None
        self.last_home = None
        self.state = SimpleNamespace(
            qpos_rad=np.arange(6, dtype=np.float64) / 10.0,
            tcp_pose=np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0]),
            tcp_speed=np.zeros(6),
            monotonic_ns=10_000_000_000,
            robot_mode=7,
            safety_mode=1,
            emergency_stopped=False,
            protective_stopped=False,
            healthy=True,
        )

    def connect(self) -> None:
        self.log.append("ur5.connect")
        self.connected = True

    def read(self):
        self.log.append("ur5.read")
        self.state.monotonic_ns = self.clock()
        return self.state

    def send(self, value, *, state_timestamp_ns):
        assert state_timestamp_ns == self.state.monotonic_ns
        self.log.append("ur5.send")
        return np.asarray(value, dtype=np.float64)

    def jog_joint6(self, direction, *, speed_rad_s, state_timestamp_ns):
        assert state_timestamp_ns == self.state.monotonic_ns
        velocity = np.zeros(6, dtype=np.float64)
        velocity[5] = direction * speed_rad_s
        self.last_jog = (direction, speed_rad_s, state_timestamp_ns)
        self.log.append("ur5.jog_joint6")
        return SimpleNamespace(
            velocity_rad_s=velocity,
            state_timestamp_ns=state_timestamp_ns,
            reached=None,
        )

    def task_home_step(
        self,
        target,
        *,
        max_speed_rad_s,
        tolerance_rad,
        proportional_gain=1.5,
        state_timestamp_ns,
    ):
        assert state_timestamp_ns == self.state.monotonic_ns
        assert proportional_gain > 0.0
        self.last_home = (
            tuple(target),
            max_speed_rad_s,
            tolerance_rad,
            proportional_gain,
            state_timestamp_ns,
        )
        self.log.append("ur5.task_home_step")
        return SimpleNamespace(
            velocity_rad_s=np.zeros(6) if self.home_reached else np.full(6, 0.01),
            state_timestamp_ns=state_timestamp_ns,
            reached=self.home_reached,
        )

    def stop(self) -> None:
        self.log.append("ur5.stop")
        self.stop_count += 1

    def close(self) -> None:
        self.log.append("ur5.close")
        self.connected = False

    def healthy(self) -> bool:
        return self.connected


class _FakeWrist:
    def __init__(self, config, log) -> None:
        self.config = config
        self.log = log
        self.connected = False
        self.fail_set = False
        self.returned_target = None
        self.stop_count = 0
        self.state = SimpleNamespace(
            position_relative_raw=(10.0, -10.0),
            target_relative_raw=(10.0, -10.0),
            encoder_abs_deg=(66.77, 163.76),
            encoder_valid=(True, True),
            motor_position_raw=(3288, 2537),
            motor_goal_raw=(3288, 2537),
            servo_zero_raw=(3278, 2547),
            hardware_limits_raw=((2781, 3568), (1822, 3333)),
            host_monotonic_ns=9_999_000_000,
            sequence=4,
            source_age_s=0.001,
            active=True,
            zero_valid=True,
            fault=False,
            fault_reason="none",
        )

    def connect(self) -> None:
        self.log.append("wrist.connect")
        self.connected = True

    def get(self):
        self.log.append("wrist.get")
        return self.state

    def set(self, value):
        self.log.append("wrist.set")
        if self.fail_set:
            raise OSError("wrist serial failed")
        result = tuple(value) if self.returned_target is None else tuple(self.returned_target)
        self.state.target_relative_raw = result
        return result

    def stop(self) -> None:
        self.log.append("wrist.stop")
        self.stop_count += 1

    def close(self) -> None:
        self.log.append("wrist.close")
        self.connected = False

    def healthy(self) -> bool:
        return self.connected


class _FakeGripper:
    def __init__(self, config, log) -> None:
        self.config = config
        self.log = log
        self.connected = False
        self.stop_count = 0
        self.returned_target = None
        self.state = SimpleNamespace(
            position=0.25,
            target=0.25,
            host_monotonic_ns=9_998_000_000,
            sequence=8,
            backend=str(config["backend"]),
        )

    def connect(self) -> None:
        self.log.append("gripper.connect")
        self.connected = True

    def get(self):
        self.log.append("gripper.get")
        return self.state

    def set(self, value):
        self.log.append("gripper.set")
        result = float(value) if self.returned_target is None else float(self.returned_target)
        self.state.target = result
        return result

    def stop(self) -> None:
        self.log.append("gripper.stop")
        self.stop_count += 1

    def close(self) -> None:
        self.log.append("gripper.close")
        self.connected = False

    def healthy(self) -> bool:
        return self.connected


class _FakeCameras:
    def __init__(self, config, log, clock) -> None:
        self.config = config
        self.log = log
        self.clock = clock
        self.connected = False
        self.fail_read = False
        self.stop_count = 0

    def connect(self) -> None:
        self.log.append("cameras.connect")
        self.connected = True

    def read(self):
        self.log.append("cameras.read")
        if self.fail_read:
            raise TimeoutError("camera timeout")
        frames = {}
        for index, camera in enumerate(self.config.cameras):
            frames[camera.role] = CameraFrame(
                role=camera.role,
                serial=camera.serial,
                sequence=100 + index,
                device_timestamp_ms=1_000.0 + index,
                host_timestamp_ns=self.clock() - index * 1_000_000,
                color=np.full((camera.height, camera.width, 3), 32 + index, np.uint8),
            )
        return frames

    def stop(self) -> None:
        self.log.append("cameras.stop")
        self.stop_count += 1

    def close(self) -> None:
        self.log.append("cameras.close")
        self.connected = False


def _project(*, real_ready: bool) -> dict:
    config = deepcopy(load_project_config())
    if real_ready:
        config["safety"]["ur5"]["joint_min_rad"] = [-6.0] * 6
        config["safety"]["ur5"]["joint_max_rad"] = [6.0] * 6
    return config


def _hardware(*, enable_motion: bool, clock: _Clock | None = None):
    clock = clock or _Clock()
    log: list[str] = []
    holders = {}

    def ur5_factory(config):
        holders["ur5"] = _FakeUR5(config, log, clock)
        return holders["ur5"]

    def wrist_factory(config):
        holders["wrist"] = _FakeWrist(config, log)
        return holders["wrist"]

    def gripper_factory(config):
        holders["gripper"] = _FakeGripper(config, log)
        return holders["gripper"]

    def camera_factory(config):
        holders["cameras"] = _FakeCameras(config, log, clock)
        return holders["cameras"]

    factories = HardwareFactories(
        ur5=ur5_factory,
        wrist=wrist_factory,
        gripper=gripper_factory,
        cameras=camera_factory,
    )
    hardware = LocalRobotHardware(
        _project(real_ready=enable_motion),
        enable_motion=enable_motion,
        factories=factories,
        monotonic_ns=clock,
    )
    holders["ur5"].state.monotonic_ns = clock()
    return hardware, holders, log, clock


def test_construction_is_pure_and_yaml_drives_all_factories():
    hardware, devices, log, _clock = _hardware(enable_motion=False)
    assert log == []
    assert not hardware.connected
    assert devices["ur5"].config.host == "192.168.1.106"
    assert not devices["ur5"].config.enable_motion
    assert devices["wrist"].config.port.startswith("/dev/serial/by-id/")
    assert devices["gripper"].config["backend"] == "hiwonder"
    assert [item.role for item in devices["cameras"].config.cameras] == ["front", "side", "top"]
    assert devices["cameras"].config.max_camera_skew_ms == 50.0


def test_connect_and_observation_contract():
    hardware, devices, log, _clock = _hardware(enable_motion=False)
    hardware.connect()
    assert log[:4] == ["ur5.connect", "wrist.connect", "gripper.connect", "cameras.connect"]
    observation = hardware.get_observation()
    assert set(observation["images"]) == {"front", "side", "top"}
    assert observation["front"] is observation["images"]["front"]
    assert observation["side"] is observation["images"]["side"]
    assert observation["top"] is observation["images"]["top"]
    assert all(image.dtype == np.uint8 for image in observation["images"].values())
    np.testing.assert_allclose(observation["qpos"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 10, -10, 0.25])
    assert observation["tcp_pose"].shape == (6,)
    assert observation["timestamp_ns"] == observation["monotonic_ns"]
    assert observation["camera_timestamps_ns"]["front"] == 1_000_000_000
    assert observation["camera_raw_timestamps_ns"] == observation["camera_timestamps_ns"]
    assert observation["camera_host_timestamps_ns"]["side"] == observation["monotonic_ns"] - 1_000_000
    assert observation["camera_sequences"] == {"front": 100, "side": 101, "top": 102}
    assert observation["sequence"] == 1
    assert observation["hardware_health"]["ok"]
    assert observation["hardware_health"]["wrist"]["servo_zero_raw"] == (3278, 2547)
    assert observation["hardware_health"]["wrist"]["coordinate"] == "yaml_servo_zero_relative_raw"
    assert not observation["hardware_health"]["motion_enabled"]
    hardware.close()
    assert devices["ur5"].stop_count == 0
    assert devices["wrist"].stop_count == 0
    assert devices["gripper"].stop_count == 0
    assert devices["cameras"].stop_count >= 1
    assert "ur5.stop" not in log
    assert "wrist.stop" not in log
    assert "gripper.stop" not in log


def test_motion_gate_and_invalid_action_fail_closed():
    hardware, devices, _log, _clock = _hardware(enable_motion=False)
    hardware.connect()
    hardware.get_observation()
    with pytest.raises(PermissionError, match="enable-motion"):
        hardware.send_action(np.zeros(9))
    assert devices["ur5"].stop_count == 0
    assert devices["wrist"].stop_count == 0
    assert devices["gripper"].stop_count == 0
    assert not hardware.healthy()

    invalid, invalid_devices, _log, _clock = _hardware(enable_motion=True)
    invalid.connect()
    invalid.get_observation()
    with pytest.raises(ValueError, match="9 维有限数"):
        invalid.send_action([0, 0, 0, 0, 0, 0, 0, np.nan, 0])
    assert invalid_devices["ur5"].stop_count >= 1


def test_send_action_order_and_success_receipt():
    hardware, devices, log, _clock = _hardware(enable_motion=True)
    hardware.connect()
    hardware.get_observation()
    devices["wrist"].returned_target = (11.0, -9.0)
    devices["gripper"].returned_target = 0.299
    log.clear()
    action = np.array([0.01, 0, 0, 0, 0, 0.02, 11.0, -9.0, 0.3])
    receipt = hardware.send_action(action)
    assert log == ["ur5.read", "ur5.send", "wrist.set", "gripper.set"]
    assert receipt.ok
    assert receipt.completed_devices == ("ur5", "wrist", "gripper")
    assert receipt.wrist_command_sent
    assert receipt.gripper_command_sent
    assert receipt.recordable
    np.testing.assert_allclose(receipt.requested_action, action)
    np.testing.assert_allclose(receipt.ur5_speedl, action[:6])
    assert receipt.wrist_target_relative_raw == pytest.approx((11.0, -9.0))
    assert receipt.gripper_target == pytest.approx(0.299)
    np.testing.assert_allclose(receipt.executed_action[6:], [11.0, -9.0, 0.299])
    assert hardware.last_command_receipt is receipt


def test_stale_observation_stops_before_first_command():
    hardware, devices, log, clock = _hardware(enable_motion=True)
    hardware.connect()
    hardware.get_observation()
    log.clear()
    clock.now_ns += 201_000_000
    with pytest.raises(TimeoutError, match="observation 过期"):
        hardware.send_action([0, 0, 0, 0, 0, 0, 11.0, -9.0, 0.25])
    assert "ur5.send" not in log
    assert devices["ur5"].stop_count >= 1


def test_partial_command_receipt_and_camera_failure_fail_closed():
    hardware, devices, log, _clock = _hardware(enable_motion=True)
    hardware.connect()
    hardware.get_observation()
    devices["wrist"].fail_set = True
    log.clear()
    with pytest.raises(OSError, match="wrist serial failed"):
        hardware.send_action([0, 0, 0, 0, 0, 0, 11.0, -9.0, 0.25])
    receipt = hardware.last_command_receipt
    assert receipt is not None
    assert not receipt.ok
    assert receipt.completed_devices == ("ur5",)
    assert not receipt.recordable
    assert log[:3] == ["ur5.read", "ur5.send", "wrist.set"]
    assert "gripper.set" not in log
    assert devices["ur5"].stop_count >= 1

    camera_failure, camera_devices, _log, _clock = _hardware(enable_motion=False)
    camera_failure.connect()
    camera_devices["cameras"].fail_read = True
    with pytest.raises(TimeoutError, match="camera timeout"):
        camera_failure.get_observation()
    assert camera_devices["ur5"].stop_count == 0
    assert camera_devices["wrist"].stop_count == 0
    assert camera_devices["gripper"].stop_count == 0
    assert not camera_failure.healthy()


def test_control_step_refreshes_ur_and_throttles_auxiliary_devices():
    hardware, _devices, log, clock = _hardware(enable_motion=True)
    hardware.connect()
    hardware.get_observation()
    log.clear()

    first_action = np.array([0.01, 0, 0, 0, 0, 0, 11.0, -9.0, 0.3])
    first = hardware.control_step(first_action)
    assert log == ["ur5.read", "ur5.send", "wrist.set", "gripper.set"]
    assert first.wrist_command_sent
    assert first.gripper_command_sent
    assert first.wrist_target_relative_raw == pytest.approx((11.0, -9.0))
    assert first.gripper_target == pytest.approx(0.3)

    log.clear()
    clock.now_ns += 8_000_000
    pending_action = np.array([0.02, 0, 0, 0, 0, 0, 12.0, -8.0, 0.8])
    skipped = hardware.control_step(pending_action)
    assert log == ["ur5.read", "ur5.send"]
    assert skipped.completed_devices == ("ur5",)
    assert not skipped.wrist_command_sent
    assert not skipped.gripper_command_sent
    # 回执必须是实际最后发送值; 不能把 pending_action 伪装成已执行值。
    assert skipped.wrist_target_relative_raw == pytest.approx((11.0, -9.0))
    assert skipped.gripper_target == pytest.approx(0.3)
    np.testing.assert_allclose(skipped.requested_action, pending_action)
    np.testing.assert_allclose(skipped.executed_action, [0.02, 0, 0, 0, 0, 0, 11.0, -9.0, 0.3])

    log.clear()
    clock.now_ns += 26_000_000
    sent = hardware.control_step(pending_action)
    assert log == ["ur5.read", "ur5.send", "wrist.set", "gripper.set"]
    assert sent.wrist_command_sent
    assert sent.gripper_command_sent
    assert sent.wrist_target_relative_raw == pytest.approx((12.0, -8.0))
    assert sent.gripper_target == pytest.approx(0.8)


def test_maintenance_home_and_j6_are_config_driven_and_nonrecordable():
    hardware, devices, log, clock = _hardware(enable_motion=True)
    hardware.connect()
    log.clear()

    jog = hardware.jog_joint6(-1)
    assert log == ["ur5.read", "ur5.jog_joint6"]
    assert jog.kind == "joint6_jog"
    assert jog.recordable is False
    assert jog.target_reached is None
    assert jog.stable is None
    assert devices["ur5"].last_jog[0] == -1
    assert devices["ur5"].last_jog[1] == pytest.approx(hardware.config.ur_joint6_jog_speed_rad_s)
    assert hardware.last_command_receipt is None
    assert hardware.last_maintenance_receipt is jog

    hardware.stop_maintenance_motion()
    devices["ur5"].home_reached = True
    first_home = hardware.task_home_step()
    assert first_home.kind == "task_home"
    assert first_home.target_reached is True
    assert first_home.stable is False
    target, speed, tolerance, gain, _timestamp = devices["ur5"].last_home
    assert target == hardware.config.ur_task_home_rad
    assert speed == pytest.approx(hardware.config.ur_home_speed_rad_s)
    assert tolerance == pytest.approx(hardware.config.ur_home_tolerance_rad)
    assert gain == pytest.approx(hardware.config.ur_home_proportional_gain)

    clock.now_ns += int(hardware.config.ur_home_stable_s * 1e9) + 1
    stable_home = hardware.task_home_step()
    assert stable_home.target_reached is True
    assert stable_home.stable is True
    assert stable_home.recordable is False


def test_maintenance_gate_rejects_readonly_and_active_episode_fail_closed():
    readonly, readonly_devices, _log, _clock = _hardware(enable_motion=False)
    readonly.connect()
    with pytest.raises(PermissionError, match="enable-motion"):
        readonly.jog_joint6(1)
    assert readonly_devices["ur5"].stop_count == 0
    assert readonly_devices["wrist"].stop_count == 0
    assert readonly_devices["gripper"].stop_count == 0

    recording, devices, _log, _clock = _hardware(enable_motion=True)
    recording.connect()
    recording.set_episode_active(active=True)
    with pytest.raises(RuntimeError, match="episode 活跃"):
        recording.task_home_step()
    assert devices["ur5"].stop_count >= 1
    assert devices["wrist"].stop_count >= 1
    assert devices["gripper"].stop_count >= 1
    assert not recording.healthy()


def test_local_config_requires_real_joint_limits_for_motion():
    with pytest.raises(ValueError, match="joint_min_rad"):
        LocalHardwareConfig.from_mapping(_project(real_ready=False), enable_motion=True)


def test_prebuilt_motion_config_does_not_require_duplicate_flag():
    project = _project(real_ready=True)
    config = LocalHardwareConfig.from_mapping(project, enable_motion=True)
    _hardware_with_config = LocalRobotHardware(
        config,
        factories=HardwareFactories(
            ur5=lambda value: SimpleNamespace(config=value),
            wrist=lambda value: SimpleNamespace(config=value),
            gripper=lambda value: SimpleNamespace(config=value),
            cameras=lambda value: SimpleNamespace(config=value),
        ),
    )
    assert _hardware_with_config.config.enable_motion
