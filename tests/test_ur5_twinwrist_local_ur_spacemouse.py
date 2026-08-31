# ruff: noqa: N802
from __future__ import annotations

import sys

import numpy as np
import pytest

from examples.ur5_twinwrist.controller.spacemouse import ButtonEvent
from examples.ur5_twinwrist.controller.spacemouse import MotionEvent
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouse
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouseConfig
from examples.ur5_twinwrist.controller.spacemouse import motion_to_ur5_twist
from examples.ur5_twinwrist.controller.spacemouse import normalize_motion
from examples.ur5_twinwrist.controller.ur5 import UR5Config
from examples.ur5_twinwrist.controller.ur5 import UR5Controller
from examples.ur5_twinwrist.controller.ur5 import joint_home_velocity


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.now_ns


class _FakeReceiver:
    def __init__(self, _host: str) -> None:
        self.q = np.zeros(6)
        self.pose = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
        self.speed = np.zeros(6)
        self.emergency = False
        self.protective = False
        self.disconnected = False

    def isConnected(self) -> bool:
        return True

    def getActualQ(self):
        return self.q

    def getActualTCPPose(self):
        return self.pose

    def getActualTCPSpeed(self):
        return self.speed

    def getRobotMode(self) -> int:
        return 7

    def getSafetyMode(self) -> int:
        return 1

    def isEmergencyStopped(self) -> bool:
        return self.emergency

    def isProtectiveStopped(self) -> bool:
        return self.protective

    def disconnect(self) -> None:
        self.disconnected = True


class _FakeControl:
    def __init__(self, _host: str) -> None:
        self.commands: list[list[float]] = []
        self.joint_commands: list[list[float]] = []
        self.stop_count = 0
        self.disconnected = False
        self.joints_safe = True

    def isConnected(self) -> bool:
        return True

    def isProgramRunning(self) -> bool:
        return True

    def isPoseWithinSafetyLimits(self, _pose) -> bool:
        return True

    def speedL(self, twist, _acceleration, _duration) -> bool:
        self.commands.append(list(twist))
        return True

    def speedJ(self, velocity, _acceleration, _duration) -> bool:
        self.joint_commands.append(list(velocity))
        return True

    def isJointsWithinSafetyLimits(self, _joints) -> bool:
        return self.joints_safe

    def speedStop(self, _deceleration) -> bool:
        self.stop_count += 1
        return True

    def stopL(self, _deceleration) -> bool:
        return True

    def stopScript(self) -> bool:
        return True

    def disconnect(self) -> None:
        self.disconnected = True


class _FakeSpaceMouseBackend:
    def __init__(self, events=()) -> None:
        self.events = list(events)
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def poll_event(self):
        return self.events.pop(0) if self.events else None

    def close(self) -> None:
        self.closed = True


def _ur_controller(
    clock: _Clock,
    *,
    enable_motion: bool = True,
    on_control_create=None,
    command_duration_s: float = 0.008,
    max_joint_step_rad: float = 0.02,
    min_tcp_z_m: float | None = None,
):
    receiver = _FakeReceiver("ignored")
    control = _FakeControl("ignored")
    config = UR5Config(
        host="configured-host",
        enable_motion=enable_motion,
        command_duration_s=command_duration_s,
        max_state_age_s=0.1,
        max_linear_speed_m_s=0.2,
        max_angular_speed_rad_s=0.5,
        max_joint_step_rad=max_joint_step_rad,
        min_tcp_z_m=min_tcp_z_m,
        joint_soft_lower_rad=(-3.0,) * 6,
        joint_soft_upper_rad=(3.0,) * 6,
        workspace_min_xyz_m=(0.2, -0.5, 0.1),
        workspace_max_xyz_m=(0.8, 0.5, 0.8),
    )

    def control_factory(_host):
        if on_control_create is not None:
            on_control_create()
        return control

    controller = UR5Controller(
        config,
        receiver_factory=lambda _host: receiver,
        control_factory=control_factory,
        monotonic_ns=clock,
    )
    return controller, receiver, control


def test_import_has_no_vendor_hardware_side_effects():
    assert "rtde_receive" not in sys.modules
    assert "rtde_control" not in sys.modules
    assert "spnav" not in sys.modules


def test_ur5_connect_is_read_only_and_send_records_final_speedl():
    clock = _Clock()
    controller, _receiver, control = _ur_controller(clock)
    controller.connect()
    assert control.commands == []

    sent = controller.send([0.05, 0.0, 0.0, 0.0, 0.0, 0.1])
    np.testing.assert_allclose(sent, [0.05, 0.0, 0.0, 0.0, 0.0, 0.1])
    np.testing.assert_allclose(control.commands[-1], sent)
    controller.close()
    assert control.disconnected


def test_ur5_motion_permission_nan_and_stale_fail_closed():
    clock = _Clock()
    disabled, _receiver, disabled_control = _ur_controller(clock, enable_motion=False)
    disabled.connect()
    with pytest.raises(PermissionError, match="enable-motion"):
        disabled.send(np.zeros(6))
    assert disabled_control.commands == []

    controller, _receiver, control = _ur_controller(clock)
    controller.connect()
    with pytest.raises(ValueError, match="6 个有限数"):
        controller.send([np.nan, 0, 0, 0, 0, 0])
    assert all(command == [0.0] * 6 for command in control.commands)

    fresh, _receiver, fresh_control = _ur_controller(clock)
    fresh.connect()
    clock.now_ns += 100_000_001
    with pytest.raises(TimeoutError, match="状态过期"):
        fresh.send(np.zeros(6))
    assert all(command == [0.0] * 6 for command in fresh_control.commands)


def test_ur5_speed_workspace_and_health_guards():
    clock = _Clock()
    controller, receiver, control = _ur_controller(clock)
    controller.connect()
    with pytest.raises(ValueError, match="线速度范数"):
        controller.send([0.2, 0.2, 0, 0, 0, 0])
    assert all(command == [0.0] * 6 for command in control.commands)

    workspace_controller, workspace_receiver, workspace_control = _ur_controller(clock)
    workspace_receiver.pose[0] = 0.205
    workspace_controller.connect()
    with pytest.raises(RuntimeError, match="预测位置"):
        workspace_controller.send([-0.05, 0, 0, 0, 0, 0])
    assert all(command == [0.0] * 6 for command in workspace_control.commands)

    health_controller, health_receiver, _control = _ur_controller(clock)
    health_controller.connect()
    health_receiver.emergency = True
    with pytest.raises(RuntimeError, match="健康检查失败"):
        health_controller.read()

    z_controller, z_receiver, _z_control = _ur_controller(clock, min_tcp_z_m=0.31)
    z_receiver.pose[2] = 0.30
    with pytest.raises(RuntimeError, match="软件工作空间"):
        z_controller.connect()


def test_ur5_motion_config_requires_all_software_envelopes() -> None:
    with pytest.raises(ValueError, match="关节软件软限位"):
        UR5Config(
            host="configured-host",
            enable_motion=True,
            workspace_min_xyz_m=(0.2, -0.5, 0.1),
            workspace_max_xyz_m=(0.8, 0.5, 0.8),
        ).validate()
    with pytest.raises(ValueError, match="工作空间边界"):
        UR5Config(
            host="configured-host",
            enable_motion=True,
            joint_soft_lower_rad=(-3.0,) * 6,
            joint_soft_upper_rad=(3.0,) * 6,
        ).validate()


def test_ur5_motion_connect_refreshes_state_after_slow_control_handshake() -> None:
    clock = _Clock()

    def delay_control_connection() -> None:
        clock.now_ns += 100_000_001

    controller, _receiver, control = _ur_controller(clock, on_control_create=delay_control_connection)
    controller.connect()
    sent = controller.send([0.05, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(sent, [0.05, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(control.commands, [[0.05, 0, 0, 0, 0, 0]])


def test_joint_home_velocity_matches_legacy_synchronized_limiter():
    velocity, reached = joint_home_velocity(
        np.zeros(6),
        [1.0, -0.5, 0.25, 0.0, 0.0, 0.0],
        0.5,
        tolerance_rad=0.01,
    )
    assert not reached
    np.testing.assert_allclose(velocity, [0.5, -0.25, 0.125, 0, 0, 0])
    stopped, reached = joint_home_velocity(
        np.zeros(6),
        [0.005] * 6,
        0.5,
        tolerance_rad=0.01,
    )
    assert reached
    np.testing.assert_array_equal(stopped, np.zeros(6))


def test_speedj_j6_and_task_home_use_explicit_nonrecording_path():
    clock = _Clock()
    controller, receiver, control = _ur_controller(clock)
    controller.connect()

    jog = controller.jog_joint6(-1, speed_rad_s=0.2)
    np.testing.assert_allclose(jog.velocity_rad_s, [0, 0, 0, 0, 0, -0.2])
    np.testing.assert_allclose(control.joint_commands[-1], jog.velocity_rad_s)
    assert control.commands == []

    receiver.q = np.array([0.5, -0.25, 0.0, 0.0, 0.0, 0.0])
    state = controller.read()
    home = controller.task_home_step(
        np.zeros(6),
        max_speed_rad_s=0.5,
        tolerance_rad=0.01,
        state_timestamp_ns=state.monotonic_ns,
    )
    assert not home.reached
    np.testing.assert_allclose(home.velocity_rad_s, [-0.5, 0.25, 0, 0, 0, 0])
    assert len(control.joint_commands) == 2


def test_speedj_permission_stale_nan_and_joint_limits_fail_closed():
    clock = _Clock()
    disabled, _receiver, disabled_control = _ur_controller(clock, enable_motion=False)
    disabled.connect()
    with pytest.raises(PermissionError, match="enable-motion"):
        disabled.jog_joint6(1, speed_rad_s=0.2)
    assert disabled_control.joint_commands == []

    invalid, _receiver, invalid_control = _ur_controller(clock)
    invalid.connect()
    with pytest.raises(ValueError, match="6 个有限数"):
        invalid.send_joint_velocity([0, 0, 0, 0, 0, np.nan])
    assert invalid_control.joint_commands == []

    stale, _receiver, stale_control = _ur_controller(clock)
    stale.connect()
    clock.now_ns += 100_000_001
    with pytest.raises(TimeoutError, match="状态过期"):
        stale.jog_joint6(1, speed_rad_s=0.2)
    assert stale_control.joint_commands == []

    boundary, boundary_receiver, boundary_control = _ur_controller(clock)
    boundary_receiver.q[5] = 2.99
    boundary.connect()
    with pytest.raises(RuntimeError, match="软件关节软限位"):
        boundary.jog_joint6(1, speed_rad_s=0.2)
    assert boundary_control.joint_commands == []

    step_limited, _receiver, step_control = _ur_controller(
        clock,
        command_duration_s=0.1,
        max_joint_step_rad=0.01,
    )
    step_limited.connect()
    with pytest.raises(ValueError, match="单命令关节变化"):
        step_limited.send_joint_velocity([0.2, 0, 0, 0, 0, 0])
    assert step_control.joint_commands == []


def test_spacemouse_axis_button_mapping_and_stale_zero():
    clock = _Clock()
    config = SpaceMouseConfig.from_mapping(
        {
            "max_raw_value": 500,
            "deadzone": 0.12,
            "stale_timeout_s": 0.1,
            "axis_mapping": ["-z", "+x", "+y", "-rz", "+rx", "+ry"],
            "button_codes": {"shift": 24, "ctrl": 25},
        }
    )
    backend = _FakeSpaceMouseBackend(
        [
            MotionEvent((100.0, 200.0, 300.0), (50.0, 150.0, 250.0)),
            ButtonEvent(code=24, pressed=True),
        ]
    )
    mouse = SpaceMouse(config, backend=backend, monotonic_ns=clock)
    assert not backend.opened
    mouse.connect()
    sample = mouse.read()
    np.testing.assert_allclose(sample.motion, [-0.6, 0.2, 0.4, -0.5, 0.0, 0.3])
    assert sample.pressed("shift")
    assert sample.healthy

    clock.now_ns += 100_000_001
    stale = mouse.read()
    assert stale.stale
    assert not stale.healthy
    np.testing.assert_array_equal(stale.motion, np.zeros(6))
    mouse.close()
    assert backend.closed


def test_project_yaml_shape_maps_into_local_controllers():
    project_buttons = {name: {"code": code} for name, code in SpaceMouseConfig().button_codes.items()}
    project = {
        "hardware": {
            "ur5": {"host": "robot.local", "control_hz": 125},
            "spacemouse": {"stale_timeout_ms": 100},
        },
        "safety": {
            "timing": {"max_state_age_ms": 80},
            "ur5": {
                "max_linear_m_s": 0.20,
                "max_angular_rad_s": 0.50,
                "min_tcp_z_m": 0.15,
                "max_joint_step_rad": 0.015,
                "workspace_min_m": [0.1, -0.5, 0.1],
                "workspace_max_m": [0.8, 0.5, 0.8],
                "joint_min_rad": [-3.0] * 6,
                "joint_max_rad": [3.0] * 6,
            },
        },
        "teleop": {
            "axes": {
                "raw_full_scale": 500,
                "deadzone": 0.12,
                "output_order": ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"],
                "mapping": {
                    "tcp_x": {"source": "z", "sign": -1},
                    "tcp_y": {"source": "x", "sign": 1},
                    "tcp_z": {"source": "y", "sign": 1},
                    "tcp_rx": {"source": "rz", "sign": -1},
                    "tcp_ry": {"source": "rx", "sign": 1},
                    "tcp_rz": {"source": "ry", "sign": 1},
                },
            },
            "buttons": project_buttons,
        },
    }
    ur5 = UR5Config.from_mapping(project, enable_motion=True)
    assert ur5.host == "robot.local"
    assert ur5.command_duration_s == pytest.approx(0.008)
    assert ur5.max_state_age_s == pytest.approx(0.08)
    assert ur5.max_linear_speed_m_s == pytest.approx(0.20)
    assert ur5.min_tcp_z_m == pytest.approx(0.15)
    assert ur5.max_joint_step_rad == pytest.approx(0.015)
    assert ur5.enable_motion

    mouse = SpaceMouseConfig.from_mapping(project)
    assert mouse.axis_mapping == ("-z", "+x", "+y", "-rz", "+rx", "+ry")
    assert mouse.button_codes == SpaceMouseConfig().button_codes
    assert mouse.stale_timeout_s == pytest.approx(0.1)

    backend = _FakeSpaceMouseBackend(
        [
            MotionEvent((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ButtonEvent(code=24, pressed=True),
        ]
    )
    device = SpaceMouse(mouse, backend=backend, monotonic_ns=_Clock())
    device.connect()
    sample = device.read()
    assert sample.pressed("shift")
    assert not sample.pressed("ctrl")


def test_spacemouse_yaml_code_swap_really_swaps_one_two_behavior() -> None:
    swapped_buttons = {name: {"code": code} for name, code in SpaceMouseConfig().button_codes.items()}
    swapped_buttons["one"]["code"] = 13
    swapped_buttons["two"]["code"] = 12
    project = {
        "hardware": {"spacemouse": {"stale_timeout_ms": 100}},
        "teleop": {
            "axes": {
                "raw_full_scale": 500,
                "deadzone": 0.12,
                "output_order": ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"],
                "mapping": {
                    "tcp_x": {"source": "z", "sign": -1},
                    "tcp_y": {"source": "x", "sign": 1},
                    "tcp_z": {"source": "y", "sign": 1},
                    "tcp_rx": {"source": "rz", "sign": -1},
                    "tcp_ry": {"source": "rx", "sign": 1},
                    "tcp_rz": {"source": "ry", "sign": 1},
                },
            },
            # 故意交换默认 bnum; 调用方查询 one/two 时必须跟随 YAML。
            "buttons": swapped_buttons,
        },
    }
    config = SpaceMouseConfig.from_mapping(project)
    backend = _FakeSpaceMouseBackend(
        [
            MotionEvent((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ButtonEvent(code=12, pressed=True),
        ]
    )
    device = SpaceMouse(config, backend=backend, monotonic_ns=_Clock())
    device.connect()
    sample = device.read()
    assert sample.pressed("two")
    assert not sample.pressed("one")


def test_spacemouse_project_mapping_rejects_missing_canonical_button() -> None:
    with pytest.raises(ValueError, match="canonical.*rotation_lock"):
        SpaceMouseConfig.from_mapping(
            {
                "hardware": {"spacemouse": {}},
                "teleop": {
                    "axes": {"output_order": [], "mapping": {}},
                    "buttons": {
                        name: {"code": code}
                        for name, code in SpaceMouseConfig().button_codes.items()
                        if name != "rotation_lock"
                    },
                },
            }
        )


def test_spacemouse_invalid_input_and_ur5_twist_mapping():
    config = SpaceMouseConfig()
    with pytest.raises(ValueError, match="6 个有限数"):
        normalize_motion([0, 0, np.inf, 0, 0, 0], config)

    motion = np.array([1.0, 1.0, 0.0, 0.0, 0.5, 0.0])
    translation = motion_to_ur5_twist(
        motion,
        {},
        translation_speed_m_s=0.1,
        rotation_speed_rad_s=0.5,
    )
    assert np.linalg.norm(translation[:3]) == pytest.approx(0.1)
    np.testing.assert_array_equal(translation[3:], np.zeros(3))

    rotation = motion_to_ur5_twist(
        motion,
        {"shift": True},
        translation_speed_m_s=0.1,
        rotation_speed_rad_s=0.5,
    )
    np.testing.assert_allclose(rotation, [0, 0, 0, 0, 0.25, 0])
    suppressed = motion_to_ur5_twist(
        motion,
        {"ctrl": True},
        translation_speed_m_s=0.1,
        rotation_speed_rad_s=0.5,
    )
    np.testing.assert_array_equal(suppressed, np.zeros(6))
