from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, ClassVar

import numpy as np
import pytest

from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouseSample
from examples.ur5_twinwrist.teleop_hardware import ActionReceipt
from examples.ur5_twinwrist.teleop_hardware import MaintenanceStatus
from examples.ur5_twinwrist.teleop_session import TeleopCollectionSession


def _sample(*, stale: bool = False, motion: Any = None, **buttons: bool) -> SpaceMouseSample:
    now = time.monotonic_ns()
    named = {
        name: bool(buttons.get(name, False))
        for name in ("menu", "fit", "esc", "rotation_lock", "t", "one", "two", "three", "four", "shift", "ctrl")
    }
    return SpaceMouseSample(
        motion=np.asarray(np.zeros(6) if motion is None else motion, dtype=np.float32),
        buttons={},
        named_buttons=named,
        monotonic_ns=now,
        motion_timestamp_ns=None if stale else now,
        stale=stale,
        connected=True,
    )


class FakeMouse:
    def __init__(self, samples: list[SpaceMouseSample]) -> None:
        self.samples = samples
        self.index = 0
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def read(self, *, fail_on_stale: bool = False) -> SpaceMouseSample:
        assert self.connected
        assert not fail_on_stale
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        # 时间戳必须代表本次 keepalive, 不能使用测试样本创建时间。
        return SpaceMouseSample(
            motion=sample.motion.copy(),
            buttons=sample.buttons,
            named_buttons=sample.named_buttons,
            monotonic_ns=time.monotonic_ns(),
            motion_timestamp_ns=sample.motion_timestamp_ns,
            stale=sample.stale,
            connected=sample.connected,
        )

    def close(self) -> None:
        self.closed = True


class FakeHardware:
    def __init__(self, *, camera_delay_s: float = 0.0) -> None:
        self.camera_delay_s = camera_delay_s
        self.connected = False
        self.stopped = False
        self.closed = False
        self.episode_active = False
        self.sequence = 0
        self.observation_sequence = 0
        self.home_ticks = 0
        self.resume_ticks = 0
        self.failure: BaseException | None = None
        self.keepalives_during_camera_wait: list[int] = []
        self.events: list[str] = []
        self.commands: list[tuple[tuple[float, ...], Any, float]] = []
        self._receipt = self._new_receipt((0.0,) * 9)

    def _new_receipt(self, action: Any) -> ActionReceipt:
        self.sequence += 1
        now = time.monotonic_ns()
        return ActionReceipt(tuple(float(value) for value in action), now, self.sequence, now, now, now)

    def connect(self) -> None:
        self.connected = True
        self.events.append("hardware.connect")

    def update_command(
        self,
        ur_speed_l: Any,
        *,
        wrist_velocity_deg_s: Any,
        gripper_target: float,
        issued_monotonic_ns: int | None = None,
    ) -> None:
        assert issued_monotonic_ns is not None
        twist = tuple(float(value) for value in ur_speed_l)
        wrist = (0.0, 0.0) if wrist_velocity_deg_s is None else tuple(
            np.radians(value) / 50.0 for value in wrist_velocity_deg_s
        )
        self.commands.append((twist, wrist_velocity_deg_s, gripper_target))
        self._receipt = self._new_receipt((*twist, *wrist, gripper_target))

    def get_observation(self) -> dict[str, Any]:
        before = len(self.commands)
        if self.camera_delay_s:
            time.sleep(self.camera_delay_s)
        self.keepalives_during_camera_wait.append(len(self.commands) - before)
        self.observation_sequence += 1
        now = time.monotonic_ns()
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        return {
            "images": {role: image.copy() for role in ("front", "side", "top")},
            "qpos": np.zeros(9, dtype=np.float32),
            "timestamp_ns": now,
            "camera_timestamps_ns": dict.fromkeys(("front", "side", "top"), now),
            "camera_host_timestamps_ns": dict.fromkeys(("front", "side", "top"), now),
            "camera_sequences": dict.fromkeys(("front", "side", "top"), self.observation_sequence),
            "hardware_health": {"gripper": {"state_source": "measured"}},
            "action_receipt": self._receipt,
        }

    def latest_action_receipt(self) -> ActionReceipt:
        return self._receipt

    def set_episode_active(self, *, active: bool) -> None:
        self.episode_active = active
        self.events.append(f"episode={active}")

    def set_joint6_jog(self, direction: int) -> None:
        self.events.append(f"jog={direction}")

    def request_home(self) -> None:
        assert not self.episode_active
        self.home_ticks = 2
        self.events.append("home.request")

    def request_wrist_master_resume(self) -> None:
        self.resume_ticks = 1
        self.events.append("master.resume")

    def maintenance_status(self) -> MaintenanceStatus:
        home = self.home_ticks > 0
        resume = self.resume_ticks > 0
        if self.home_ticks:
            self.home_ticks -= 1
        elif self.resume_ticks:
            self.resume_ticks -= 1
        return MaintenanceStatus(home, home, home, 0, resume)

    def report_input_failure(self, failure: BaseException) -> None:
        self.failure = failure

    def raise_if_failed(self) -> None:
        if self.failure is not None:
            raise RuntimeError(f"fake hardware failed: {self.failure}") from self.failure

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeRecorder:
    instances: ClassVar[list[FakeRecorder]] = []

    def __init__(self, root: str | Path, _project: Any) -> None:
        self.root = Path(root)
        self.active = False
        self.frames: list[tuple[dict[str, Any], Any]] = []
        self.events: list[str] = []
        self.closed = False
        type(self).instances.append(self)

    def start(self, observation: dict[str, Any]) -> Path:
        self.active = True
        self.events.append("start")
        assert "action_receipt" in observation
        return self.root / "episode_0000.tmp.hdf5"

    def append_if_due(self, observation: dict[str, Any], action: Any) -> bool:
        assert action is observation["action_receipt"].action
        self.frames.append((observation, action))
        return True

    def save(self) -> Path:
        assert self.active
        assert self.frames
        self.active = False
        self.events.append("save")
        return self.root / "episode_0000.hdf5"

    def reject(self, reason: str) -> Path:
        self.active = False
        self.events.append(f"reject:{reason}")
        return self.root / "rejected" / "episode_0000.hdf5"

    def close(self) -> None:
        self.closed = True


def _project() -> dict[str, Any]:
    project = deepcopy(load_project_config())
    project["hardware"]["ur5"]["control_hz"] = 500.0
    project["collection"]["capture"]["record_hz"] = 100.0
    return project


def test_menu_homes_and_opens_before_start_then_fit_saves_before_home(tmp_path: Path) -> None:
    FakeRecorder.instances.clear()
    samples = [
        _sample(),
        _sample(menu=True),
        *[_sample() for _ in range(20)],
        _sample(fit=True),
        *[_sample(fit=True) for _ in range(10)],
    ]
    mouse = FakeMouse(samples)
    hardware = FakeHardware(camera_delay_s=0.03)
    output: list[str] = []
    session = TeleopCollectionSession(
        _project(),
        episodes=1,
        spacemouse=mouse,
        hardware=hardware,
        output=tmp_path,
        recorder_factory=FakeRecorder,
        console=output.append,
    )
    result = session.run()
    recorder = FakeRecorder.instances[0]
    assert result == {"successful_episodes": 1, "requested_episodes": 1}
    assert recorder.events == ["start", "save"]
    first_home = hardware.events.index("home.request")
    episode_start = hardware.events.index("episode=True")
    assert first_home < episode_start
    assert hardware.events.count("home.request") == 2
    assert any(command[2] == 0.0 for command in hardware.commands[:episode_start])
    assert max(hardware.keepalives_during_camera_wait) >= 5
    assert recorder.frames
    assert hardware.stopped
    assert hardware.closed
    assert mouse.closed
    assert any("原子封存" in line for line in output)


def test_stale_motion_is_zero_keepalive_not_disconnect(tmp_path: Path) -> None:
    mouse = FakeMouse([_sample(stale=True, motion=[1, 1, 1, 1, 1, 1])])
    hardware = FakeHardware()
    mouse.connect()
    hardware.connect()
    session = TeleopCollectionSession(
        _project(),
        episodes=1,
        spacemouse=mouse,
        hardware=hardware,
        output=tmp_path,
        recorder_factory=FakeRecorder,
    )
    session._read_and_keepalive()  # noqa: SLF001
    assert hardware.commands[-1][0] == (0.0,) * 6
    assert hardware.failure is None


def test_stale_motion_lease_clears_continuous_joint6_jog(tmp_path: Path) -> None:
    sample = _sample(stale=True, one=True)
    session = TeleopCollectionSession(
        _project(),
        episodes=1,
        spacemouse=FakeMouse([sample]),
        hardware=FakeHardware(),
        output=tmp_path,
        recorder_factory=FakeRecorder,
    )
    session._process_buttons(  # noqa: SLF001
        sample.named_buttons,
        continuous_inputs_enabled=False,
    )
    assert session.hardware.events[-1] == "jog=0"


def test_joint6_press_during_episode_rejects_and_fails_closed(tmp_path: Path) -> None:
    FakeRecorder.instances.clear()
    samples = [
        _sample(),
        _sample(menu=True),
        *[_sample() for _ in range(15)],
        _sample(one=True),
    ]
    mouse = FakeMouse(samples)
    hardware = FakeHardware()
    session = TeleopCollectionSession(
        _project(),
        episodes=1,
        spacemouse=mouse,
        hardware=hardware,
        output=tmp_path,
        recorder_factory=FakeRecorder,
    )
    with pytest.raises(PermissionError, match="episode"):
        session.run()
    recorder = FakeRecorder.instances[0]
    assert any(event.startswith("reject:") for event in recorder.events)
    assert isinstance(hardware.failure, PermissionError)
    assert hardware.stopped
