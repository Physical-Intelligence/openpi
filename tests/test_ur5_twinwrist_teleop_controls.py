from copy import deepcopy
import time

import numpy as np

from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.controller.spacemouse import SpaceMouseSample
from examples.ur5_twinwrist.teleop_controls import EpisodeButtonState
from examples.ur5_twinwrist.teleop_controls import EpisodeCommand
from examples.ur5_twinwrist.teleop_controls import build_teleop_action


def _sample(motion=None, **pressed: bool) -> SpaceMouseSample:
    now = time.monotonic_ns()
    named = {
        name: bool(pressed.get(name, False))
        for name in ("menu", "fit", "esc", "rotation_lock", "shift", "ctrl", "alt", "three", "four")
    }
    return SpaceMouseSample(
        motion=np.asarray(motion if motion is not None else np.zeros(6), dtype=np.float32),
        buttons={},
        named_buttons=named,
        monotonic_ns=now,
        motion_timestamp_ns=now,
        stale=False,
        connected=True,
    )


def test_episode_buttons_use_rising_edges() -> None:
    state = EpisodeButtonState()
    assert state.update({"menu": True}) is EpisodeCommand.START
    assert state.update({"menu": True}) is None
    assert state.update({}) is None
    assert state.update({"fit": True}) is EpisodeCommand.SAVE
    assert state.update({}) is None
    assert state.update({"fit": True}) is EpisodeCommand.HOME
    assert state.update({}) is None
    assert state.update({"rotation_lock": True}) is EpisodeCommand.FINALIZE


def test_action_is_final_9d_command_not_raw_spacemouse_axes() -> None:
    config = load_project_config()
    action = build_teleop_action(
        _sample([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        [0.0, 0.0],
        0.0,
        config,
        dt_s=0.008,
    )
    assert action.shape == (9,)
    # sample.motion is already the normalized YAML output [tcp_x, ...].
    # The recorded command is +X speedL with units, not the raw value 1.0.
    np.testing.assert_allclose(action[:3], [0.107, 0.0, 0.0], atol=1e-6)
    assert np.all(action[3:] == 0.0)


def test_shift_selects_rotation_and_ctrl_selects_wrist() -> None:
    config = load_project_config()
    rotated = build_teleop_action(
        _sample([0, 0, 0, 0.5, 0, 0], shift=True),
        [0, 0],
        0,
        config,
        dt_s=0.008,
    )
    np.testing.assert_allclose(rotated[:3], 0.0)
    np.testing.assert_allclose(rotated[3:], [0.27, 0, 0, 0, 0, 0], atol=1e-6)

    wrist = build_teleop_action(
        _sample([1, 1, 0, 0, 0, 0], ctrl=True),
        [0, 0],
        0,
        config,
        dt_s=0.05,
    )
    np.testing.assert_allclose(wrist[:6], 0.0)
    # cap Y→J1(+), cap X→J2(-); 每步再受 safety.max_step_raw=18 限制。
    np.testing.assert_allclose(wrist[6:8], [18.0, -18.0], atol=1e-7)


def test_yaml_modifiers_are_runtime_authoritative() -> None:
    config = deepcopy(load_project_config())
    config["teleop"]["modes"]["rotation"]["modifier"] = "alt"
    rotated = build_teleop_action(
        _sample([0, 0, 0, 0.5, 0, 0], alt=True),
        [0, 0],
        0,
        config,
        dt_s=0.008,
    )
    np.testing.assert_allclose(rotated[3:6], [0.27, 0, 0], atol=1e-6)
    not_rotated = build_teleop_action(
        _sample([0, 0, 0, 0.5, 0, 0], shift=True),
        [0, 0],
        0,
        config,
        dt_s=0.008,
    )
    np.testing.assert_allclose(not_rotated, 0.0)


def test_gripper_is_absolute_binary_action_and_conflict_holds() -> None:
    config = load_project_config()
    closed = build_teleop_action(_sample(three=True), [0, 0], 0, config, dt_s=0.008)
    assert closed[8] == 1.0
    held = build_teleop_action(_sample(three=True, four=True), [0, 0], 0.4, config, dt_s=0.008)
    assert held[8] == np.float32(0.4)


def test_stale_spacemouse_fails_closed() -> None:
    config = load_project_config()
    sample = _sample()
    sample = SpaceMouseSample(
        motion=sample.motion,
        buttons=sample.buttons,
        named_buttons=sample.named_buttons,
        monotonic_ns=sample.monotonic_ns,
        motion_timestamp_ns=sample.motion_timestamp_ns,
        stale=True,
        connected=True,
    )
    try:
        build_teleop_action(sample, [0, 0], 0, config, dt_s=0.008)
    except TimeoutError:
        pass
    else:
        raise AssertionError("stale SpaceMouse must fail closed")
