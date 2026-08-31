# ruff: noqa: RUF001, RUF002, RUF003
"""纯函数形式的遥操作映射与 Episode 按键状态机。

本模块不连接任何硬件。它把项目 YAML 中的 SpaceMouse 语义转换为最终
9 维控制命令：UR5 ``speedL`` 六维、手腕 J1/J2 YAML 零位相对 raw
目标两维、夹爪绝对位置一维。
``build_teleop_action`` 保留给纯变换测试和旧调用兼容。真实数采循环使用
``teleop_session.py``：Ctrl 产生腕速度覆盖、未按 Ctrl 时跟随 ESP32 主腕；
HDF5 只保存 ``TeleopHardwareWorker`` 返回的实际 action receipt。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .controller.spacemouse import SpaceMouseSample
from .controller.spacemouse import motion_to_ur5_twist


class EpisodeCommand(str, Enum):
    """由 SpaceMouse 边沿事件产生的采集生命周期命令。"""

    START = "start"
    SAVE = "save"
    DISCARD = "discard"
    HOME = "home"
    FINALIZE = "finalize"


@dataclass
class EpisodeButtonState:
    """Menu/Fit/Esc/RotationLock 的边沿触发状态机。

    与已验证旧流程保持一致：待机时 Menu 开始；录制时 Fit 保存、Esc
    丢弃；待机时 Fit 请求回零；RotationLock 结束整个采集进程。
    """

    recording: bool = False

    def __post_init__(self) -> None:
        self._previous: dict[str, bool] = {}

    def update(self, buttons: Mapping[str, bool]) -> EpisodeCommand | None:
        tracked = ("menu", "fit", "esc", "rotation_lock")
        current = {name: bool(buttons.get(name, False)) for name in tracked}
        rising = {name: current[name] and not self._previous.get(name, False) for name in tracked}
        self._previous = current
        if rising["rotation_lock"]:
            self.recording = False
            return EpisodeCommand.FINALIZE
        if current["menu"] and current["fit"]:
            return None
        if self.recording:
            if rising["esc"]:
                self.recording = False
                return EpisodeCommand.DISCARD
            if rising["fit"]:
                self.recording = False
                return EpisodeCommand.SAVE
            return None
        if rising["menu"]:
            self.recording = True
            return EpisodeCommand.START
        if rising["fit"]:
            return EpisodeCommand.HOME
        return None

    def abort(self) -> None:
        self.recording = False

    def synchronize(self, buttons: Mapping[str, bool]) -> None:
        """连接后吞掉当前按键电平，防止按住的键被误判为新边沿。"""

        self._previous = {
            name: bool(buttons.get(name, False))
            for name in ("menu", "fit", "esc", "rotation_lock")
        }


def _finite_vector(values: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须包含 {size} 个有限数")
    return result


def _response_outside_deadzone(value: np.ndarray, deadzone: float) -> np.ndarray:
    if not math.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
        raise ValueError("手腕 deadzone 必须位于 [0, 1)")
    magnitude = np.abs(value)
    scaled = np.clip((magnitude - deadzone) / (1.0 - deadzone), 0.0, 1.0)
    return np.sign(value) * scaled


def wrist_target_from_spacemouse(
    current_target_relative_raw: object,
    sample: SpaceMouseSample,
    project_config: Mapping[str, Mapping[str, object]],
    *,
    dt_s: float,
) -> np.ndarray:
    """Ctrl 按住时把帽子的前两维积分为 J1/J2 相对 raw 目标。"""

    target = _finite_vector(current_target_relative_raw, 2, "手腕当前相对 raw 目标")
    if not math.isfinite(dt_s) or dt_s < 0.0:
        raise ValueError("dt_s 必须是非负有限数")
    teleop = project_config["teleop"]
    safety = project_config["safety"]
    hardware = project_config["hardware"]
    poses = project_config["poses"]
    mode = teleop["modes"]["wrist"]
    if not sample.pressed(str(mode.get("modifier", "ctrl"))):
        return target.astype(np.float32)
    deadzone = float(mode["deadzone"])
    maximum_speed = float(mode["max_speed_deg_s"])
    response = _response_outside_deadzone(_finite_vector(sample.motion[:2], 2, "手腕帽输入"), deadzone)
    signs = np.asarray([float(mode["cap_x_sign"]), float(mode["cap_y_sign"])], dtype=np.float64)
    axes = (str(mode["cap_x_to_axis"]), str(mode["cap_y_to_axis"]))
    if set(axes) != {"j1", "j2"}:
        raise ValueError("Ctrl wrist cap 映射必须恰好包含 j1/j2")
    velocity_by_axis = {
        axis: float(value)
        for axis, value in zip(axes, response * signs * maximum_speed, strict=True)
    }
    mapping = hardware["wrist"]["master_mapping"]
    delta = np.asarray(
        [
            velocity_by_axis["j1"] * float(mapping["j1_raw_per_deg"]),
            velocity_by_axis["j2"] * float(mapping["j2_raw_per_deg"]),
        ],
        dtype=np.float64,
    ) * min(dt_s, 0.05)
    maximum_step = float(safety["wrist"]["max_step_raw"])
    delta = np.clip(delta, -maximum_step, maximum_step)
    zero = _finite_vector(poses["wrist"]["servo_zero_raw"], 2, "手腕 YAML servo zero")
    lower = np.asarray(
        [safety["wrist"]["j1_min_raw"], safety["wrist"]["j2_min_raw"]], dtype=np.float64
    ) - zero
    upper = np.asarray(
        [safety["wrist"]["j1_max_raw"], safety["wrist"]["j2_max_raw"]], dtype=np.float64
    ) - zero
    return np.clip(target + delta, lower, upper).astype(np.float32)


def gripper_target_from_buttons(
    current_target: float,
    sample: SpaceMouseSample,
    poses: Mapping[str, object],
) -> float:
    """按住 3 闭合、按住 4 打开；冲突时保持，不产生隐式运动。"""

    value = float(current_target)
    if not math.isfinite(value):
        raise ValueError("夹爪当前目标必须是有限数")
    close = sample.pressed("three")
    opened = sample.pressed("four")
    if close != opened:
        value = float(poses["gripper"]["closed_value"] if close else poses["gripper"]["open_value"])
    return float(np.clip(value, 0.0, 1.0))


def build_teleop_action(
    sample: SpaceMouseSample,
    current_wrist_target_relative_raw: object,
    current_gripper_target: float,
    project_config: Mapping[str, Mapping[str, object]],
    *,
    dt_s: float,
) -> np.ndarray:
    """生成测试用候选 9 维命令；不得直接作为真实数采 action 保存。"""

    if not sample.connected or sample.stale or not np.isfinite(sample.motion).all():
        raise TimeoutError("SpaceMouse 状态过期或不健康，禁止继续运动")
    teleop = project_config["teleop"]
    rotation_modifier = str(teleop["modes"]["rotation"]["modifier"])
    wrist_modifier = str(teleop["modes"]["wrist"]["modifier"])
    twist = motion_to_ur5_twist(
        sample.motion,
        sample.named_buttons,
        translation_speed_m_s=float(teleop["modes"]["translation"]["linear_speed_m_s"]),
        rotation_speed_rad_s=float(teleop["modes"]["rotation"]["angular_speed_rad_s"]),
        rotation_button=rotation_modifier,
        suppress_buttons=(wrist_modifier, "rear", "t"),
    )
    wrist = wrist_target_from_spacemouse(
        current_wrist_target_relative_raw,
        sample,
        project_config,
        dt_s=dt_s,
    )
    gripper = gripper_target_from_buttons(current_gripper_target, sample, project_config["poses"])
    action = np.concatenate((twist, wrist, np.asarray([gripper], dtype=np.float64)))
    if action.shape != (9,) or not np.isfinite(action).all():
        raise RuntimeError("遥操作映射产生了非法 9 维 action")
    return action.astype(np.float32)


__all__ = [
    "EpisodeButtonState",
    "EpisodeCommand",
    "build_teleop_action",
    "gripper_target_from_buttons",
    "wrist_target_from_spacemouse",
]
