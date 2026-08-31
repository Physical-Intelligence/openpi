"""读取并校验 UR5 双轴腕项目的 YAML 配置。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = ("hardware.yaml", "teleop.yaml", "poses.yaml", "safety.yaml", "collection.yaml")
DEFAULT_CONFIG_DIR = Path(__file__).with_name("config")
REQUIRED_TELEOP_BUTTONS = {
    "menu",
    "fit",
    "t",
    "rear",
    "front",
    "roll_cw",
    "one",
    "two",
    "three",
    "four",
    "esc",
    "alt",
    "shift",
    "ctrl",
    "rotation_lock",
}


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取配置文件 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"配置文件必须是 YAML mapping: {path}")
    if value.get("schema_version") != 1:
        raise ValueError(f"不支持的 schema_version: {path}")
    return value


def _require_finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} 必须包含 {length} 个数")
    result = [float(item) for item in value]
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise ValueError(f"{name} 含 NaN 或 Inf")
    return result


def _require_finite_scalar(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} 含 NaN 或 Inf")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} 必须小于等于 {maximum}")
    return result


def _validate_wrist_hardware(wrist: Any, teleop: dict[str, Any]) -> None:
    """校验 ESP32 主腕到 OpenRB 舵机 raw 坐标的显式运行合同。"""
    if not isinstance(wrist, dict):
        raise ValueError("hardware.wrist 必须是 mapping")
    required = {
        "master_port",
        "controller_port",
        "baud",
        "serial_timeout_s",
        "response_timeout_s",
        "source_max_age_s",
        "stream_period_ms",
        "command_hz",
        "state_hz",
        "control_mode",
        "coordinate",
        "master_mapping",
        "filter",
        "dynamixel_profile",
        "target_limiter",
        "override_max_velocity_deg_s",
        "override_lease_s",
        "override_accel_deg_s2",
        "override_decel_deg_s2",
        "read_timeout_s",
    }
    missing = sorted(required - wrist.keys())
    if missing:
        raise ValueError(f"hardware.wrist 缺少显式运行参数: {', '.join(missing)}")
    baud = _require_finite_scalar(wrist["baud"], "hardware.wrist.baud", minimum=1.0)
    if not baud.is_integer():
        raise ValueError("hardware.wrist.baud 必须是正整数")
    if wrist["control_mode"] != "servo_position_open_loop":
        raise ValueError("hardware.wrist.control_mode 必须是 servo_position_open_loop")
    if wrist["coordinate"] != "yaml_servo_zero_relative_raw":
        raise ValueError("hardware.wrist.coordinate 必须是 yaml_servo_zero_relative_raw")

    serial_timeout_s = _require_finite_scalar(
        wrist["serial_timeout_s"], "hardware.wrist.serial_timeout_s", minimum=0.001
    )
    response_timeout_s = _require_finite_scalar(
        wrist["response_timeout_s"], "hardware.wrist.response_timeout_s", minimum=0.001
    )
    source_max_age_s = _require_finite_scalar(
        wrist["source_max_age_s"], "hardware.wrist.source_max_age_s", minimum=0.001
    )
    _require_finite_scalar(wrist["read_timeout_s"], "hardware.wrist.read_timeout_s", minimum=0.001)
    if response_timeout_s < serial_timeout_s:
        raise ValueError("hardware.wrist.response_timeout_s 不得小于 serial_timeout_s")
    if source_max_age_s < serial_timeout_s:
        raise ValueError("hardware.wrist.source_max_age_s 不得小于 serial_timeout_s")

    period_ms = _require_finite_scalar(
        wrist["stream_period_ms"], "hardware.wrist.stream_period_ms", minimum=1.0, maximum=1000.0
    )
    if not float(period_ms).is_integer():
        raise ValueError("hardware.wrist.stream_period_ms 必须是整数毫秒")
    command_hz = _require_finite_scalar(wrist["command_hz"], "hardware.wrist.command_hz", minimum=0.1)
    state_hz = _require_finite_scalar(wrist["state_hz"], "hardware.wrist.state_hz", minimum=0.1)
    master_hz = 1000.0 / period_ms
    if command_hz > master_hz:
        raise ValueError("hardware.wrist.command_hz 不得高于主腕 TELE 帧率")
    if state_hz > command_hz:
        raise ValueError("hardware.wrist.state_hz 不得高于 command_hz")

    mapping = wrist["master_mapping"]
    if not isinstance(mapping, dict):
        raise ValueError("hardware.wrist.master_mapping 必须是 mapping")
    mapping_required = {
        "j1_source",
        "j1_sign",
        "j1_raw_per_deg",
        "j2_source",
        "j2_sign",
        "j2_raw_per_deg",
        "input_deadband_deg",
    }
    mapping_missing = sorted(mapping_required - mapping.keys())
    if mapping_missing:
        raise ValueError(f"hardware.wrist.master_mapping 缺少: {', '.join(mapping_missing)}")
    if mapping["j1_source"] != "enc0" or mapping["j2_source"] != "enc1":
        raise ValueError("当前舵机 raw 驱动固定要求主腕 Enc0→J1、Enc1→J2")
    j1_sign = _require_finite_scalar(mapping["j1_sign"], "hardware.wrist.master_mapping.j1_sign")
    j2_sign = _require_finite_scalar(mapping["j2_sign"], "hardware.wrist.master_mapping.j2_sign")
    if j1_sign not in {-1.0, 1.0} or j2_sign not in {-1.0, 1.0}:
        raise ValueError("hardware.wrist.master_mapping J1/J2 sign 必须是 -1 或 1")
    _require_finite_scalar(
        mapping["j1_raw_per_deg"],
        "hardware.wrist.master_mapping.j1_raw_per_deg",
        minimum=0.001,
    )
    _require_finite_scalar(
        mapping["j2_raw_per_deg"],
        "hardware.wrist.master_mapping.j2_raw_per_deg",
        minimum=0.001,
    )
    _require_finite_scalar(
        mapping["input_deadband_deg"],
        "hardware.wrist.master_mapping.input_deadband_deg",
        minimum=0.0,
        maximum=30.0,
    )

    override_speed = _require_finite_scalar(
        wrist["override_max_velocity_deg_s"],
        "hardware.wrist.override_max_velocity_deg_s",
        minimum=0.001,
    )
    _require_finite_scalar(wrist["override_lease_s"], "hardware.wrist.override_lease_s", minimum=0.001)
    _require_finite_scalar(wrist["override_accel_deg_s2"], "hardware.wrist.override_accel_deg_s2", minimum=0.001)
    _require_finite_scalar(wrist["override_decel_deg_s2"], "hardware.wrist.override_decel_deg_s2", minimum=0.001)
    try:
        teleop_speed = teleop["modes"]["wrist"]["max_speed_deg_s"]
    except (KeyError, TypeError) as exc:
        raise ValueError("teleop.modes.wrist.max_speed_deg_s 缺失") from exc
    teleop_speed = _require_finite_scalar(
        teleop_speed,
        "teleop.modes.wrist.max_speed_deg_s",
        minimum=0.001,
    )
    if not math.isclose(override_speed, teleop_speed, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("hardware.wrist.override_max_velocity_deg_s 必须与 teleop.modes.wrist.max_speed_deg_s 相同")

    filter_config = wrist["filter"]
    if not isinstance(filter_config, dict):
        raise ValueError("hardware.wrist.filter 必须是 mapping")
    for key in ("one_euro_min_cutoff", "one_euro_d_cutoff"):
        _require_finite_scalar(filter_config.get(key), f"hardware.wrist.filter.{key}", minimum=0.001)
    _require_finite_scalar(
        filter_config.get("one_euro_beta"),
        "hardware.wrist.filter.one_euro_beta",
        minimum=0.0,
    )

    profile = wrist["dynamixel_profile"]
    if not isinstance(profile, dict):
        raise ValueError("hardware.wrist.dynamixel_profile 必须是 mapping")
    for key in ("velocity_raw", "acceleration_raw"):
        value = _require_finite_scalar(
            profile.get(key),
            f"hardware.wrist.dynamixel_profile.{key}",
            minimum=0.0,
            maximum=32767.0,
        )
        if not value.is_integer():
            raise ValueError(f"hardware.wrist.dynamixel_profile.{key} 必须是整数")

    limiter = wrist["target_limiter"]
    if not isinstance(limiter, dict):
        raise ValueError("hardware.wrist.target_limiter 必须是 mapping")
    deadband_raw = _require_finite_scalar(
        limiter.get("deadband_raw"),
        "hardware.wrist.target_limiter.deadband_raw",
        minimum=0.0,
    )
    if not deadband_raw.is_integer():
        raise ValueError("hardware.wrist.target_limiter.deadband_raw 必须是整数")
    for key in ("max_velocity_raw_s", "max_accel_raw_s2", "max_jerk_raw_s3"):
        _require_finite_scalar(limiter.get(key), f"hardware.wrist.target_limiter.{key}", minimum=0.001)


def validate_project_config(config: dict[str, dict[str, Any]], *, require_real_ready: bool = False) -> None:
    hardware = config["hardware"]
    cameras = hardware["cameras"]
    roles = [str(item.get("role")) for item in cameras["devices"]]
    serials = [str(item.get("serial")) for item in cameras["devices"]]
    if set(roles) != {"front", "side", "top"} or len(roles) != 3:
        raise ValueError("hardware.cameras.devices 必须恰好定义 front/side/top")
    if len(set(serials)) != 3 or any(not value for value in serials):
        raise ValueError("三台相机 serial 必须非空且互不重复")
    for key in ("width", "height", "fps"):
        value = _require_finite_scalar(cameras.get(key), f"hardware.cameras.{key}", minimum=1.0)
        if not value.is_integer():
            raise ValueError(f"hardware.cameras.{key} 必须是正整数")
    if cameras.get("pixel_format") != "rgb8":
        raise ValueError("当前三相机/HDF5 数据合同只支持 hardware.cameras.pixel_format=rgb8")
    for key in ("connect_timeout_s", "provider_wait_timeout_s", "stop_timeout_s"):
        _require_finite_scalar(cameras.get(key), f"hardware.cameras.{key}", minimum=0.001)
    queue_size = _require_finite_scalar(
        cameras.get("queue_size"), "hardware.cameras.queue_size", minimum=2.0
    )
    if not queue_size.is_integer():
        raise ValueError("hardware.cameras.queue_size 必须是大于等于 2 的整数")

    spacemouse = hardware["spacemouse"]
    max_events = _require_finite_scalar(
        spacemouse.get("max_events_per_read"),
        "hardware.spacemouse.max_events_per_read",
        minimum=1.0,
        maximum=4096.0,
    )
    if not max_events.is_integer():
        raise ValueError("hardware.spacemouse.max_events_per_read 必须是整数")

    teleop = config["teleop"]
    _validate_wrist_hardware(hardware.get("wrist"), teleop)

    gripper = hardware["gripper"]
    backend = str(gripper["backend"])
    if backend not in gripper["adapters"]:
        raise ValueError(f"夹爪 backend {backend!r} 没有对应 adapters 配置")
    ports = [
        str(hardware["wrist"]["master_port"]),
        str(hardware["wrist"]["controller_port"]),
        str(gripper["adapters"][backend]["port"]),
    ]
    if any(not value.startswith("/dev/serial/by-id/") for value in ports):
        raise ValueError("手腕和夹爪必须使用 /dev/serial/by-id/ 持久路径")

    raw_axes = set(teleop["axes"]["raw_order"])
    mapping = teleop["axes"]["mapping"]
    if set(mapping) != set(teleop["axes"]["output_order"]):
        raise ValueError("axes.mapping 必须覆盖全部 output_order")
    if any(item["source"] not in raw_axes or item["sign"] not in (-1, 1) for item in mapping.values()):
        raise ValueError("轴映射 source/sign 无效")
    buttons = teleop["buttons"]
    if set(buttons) != REQUIRED_TELEOP_BUTTONS:
        missing = sorted(REQUIRED_TELEOP_BUTTONS - set(buttons))
        extra = sorted(set(buttons) - REQUIRED_TELEOP_BUTTONS)
        raise ValueError(f"teleop.buttons canonical 键名不一致: missing={missing}, extra={extra}")
    button_codes = [int(item["code"]) for item in buttons.values()]
    if len(set(button_codes)) != len(button_codes):
        raise ValueError("SpaceMouse button code 不能重复")
    modes = teleop["modes"]
    if modes["translation"].get("modifier") is not None:
        raise ValueError("当前平移模式要求 teleop.modes.translation.modifier=null")
    rotation_modifier = str(modes["rotation"].get("modifier", ""))
    wrist_modifier = str(modes["wrist"].get("modifier", ""))
    if rotation_modifier not in buttons or wrist_modifier not in buttons:
        raise ValueError("旋转/手腕 modifier 必须是 teleop.buttons 中的 canonical 键名")
    if rotation_modifier == wrist_modifier:
        raise ValueError("旋转与手腕 modifier 不能使用同一个键")
    wrist_mode = modes["wrist"]
    cap_axes = (str(wrist_mode.get("cap_x_to_axis", "")), str(wrist_mode.get("cap_y_to_axis", "")))
    if set(cap_axes) != {"j1", "j2"}:
        raise ValueError("teleop.modes.wrist 的 cap_x/cap_y 必须恰好映射到 j1/j2")
    for key in ("cap_x_sign", "cap_y_sign"):
        sign = _require_finite_scalar(wrist_mode.get(key), f"teleop.modes.wrist.{key}")
        if sign not in {-1.0, 1.0}:
            raise ValueError(f"teleop.modes.wrist.{key} 必须是 -1 或 1")
    if not math.isclose(float(teleop["axes"].get("clip", 1.0)), 1.0, abs_tol=1e-12):
        raise ValueError("当前 SpaceMouse 归一化合同要求 teleop.axes.clip=1.0")
    episode = teleop["episode"]
    if episode.get("physical_toggle_chord_enabled") is not False:
        raise ValueError("第一版不支持组合键: physical_toggle_chord_enabled 必须为 false")
    if episode.get("reject_joint_jog_during_recording") is not True:
        raise ValueError("安全合同要求 reject_joint_jog_during_recording=true")

    poses = config["poses"]
    _require_finite_vector(poses["ur5"]["task_home_rad"], 6, "poses.ur5.task_home_rad")
    _require_finite_scalar(
        poses["ur5"]["home_proportional_gain"],
        "poses.ur5.home_proportional_gain",
        minimum=0.001,
    )
    _require_finite_scalar(poses["ur5"]["home_stable_s"], "poses.ur5.home_stable_s", minimum=0.001)
    _require_finite_scalar(
        poses["ur5"]["full_home_timeout_s"],
        "poses.ur5.full_home_timeout_s",
        minimum=1.0,
    )
    wrist_pose = poses["wrist"]
    if wrist_pose.get("coordinate") != "yaml_servo_zero_relative_raw":
        raise ValueError("poses.wrist.coordinate 必须是 yaml_servo_zero_relative_raw")
    if wrist_pose.get("zero_method") != "yaml_servo_raw":
        raise ValueError("poses.wrist.zero_method 必须是 yaml_servo_raw")
    if wrist_pose.get("state_order") != ["j1", "j2"]:
        raise ValueError("poses.wrist.state_order 必须是 [j1, j2]")
    servo_zero = _require_finite_vector(wrist_pose.get("servo_zero_raw"), 2, "poses.wrist.servo_zero_raw")
    if any(not value.is_integer() or not 0 <= value <= 4095 for value in servo_zero):
        raise ValueError("poses.wrist.servo_zero_raw 必须是两个 [0,4095] 整数")
    start_tolerance = _require_finite_scalar(
        wrist_pose.get("start_tolerance_raw"),
        "poses.wrist.start_tolerance_raw",
        minimum=0.0,
    )
    if not start_tolerance.is_integer():
        raise ValueError("poses.wrist.start_tolerance_raw 必须是整数")
    home_target = _require_finite_vector(
        wrist_pose.get("home_target_relative_raw"),
        2,
        "poses.wrist.home_target_relative_raw",
    )
    if home_target != [0.0, 0.0]:
        raise ValueError("poses.wrist.home_target_relative_raw 必须是 [0,0]")
    collection = config["collection"]
    if int(collection["state"]["dimension"]) != 9 or len(collection["state"]["fields"]) != 9:
        raise ValueError("state 必须是 9 维")
    if int(collection["action"]["dimension"]) != 9 or len(collection["action"]["fields"]) != 9:
        raise ValueError("action 必须是 9 维")
    expected_state_fields = [
        *(f"ur5_actual_joint_{index}_rad" for index in range(6)),
        "wrist_j1_actual_relative_raw",
        "wrist_j2_actual_relative_raw",
        "gripper_actual_normalized",
    ]
    expected_action_fields = [
        "tcp_vx_m_s",
        "tcp_vy_m_s",
        "tcp_vz_m_s",
        "tcp_wx_rad_s",
        "tcp_wy_rad_s",
        "tcp_wz_rad_s",
        "wrist_j1_target_relative_raw",
        "wrist_j2_target_relative_raw",
        "gripper_target_normalized",
    ]
    if collection["state"]["fields"] != expected_state_fields:
        raise ValueError("collection.state.fields 与 J1/J2 相对 raw 数据合同不一致")
    if collection["action"]["fields"] != expected_action_fields:
        raise ValueError("collection.action.fields 与 J1/J2 相对 raw 数据合同不一致")
    expected_semantics = "tcp_speedL_6+wrist_j1_j2_yaml_zero_relative_raw_2+gripper_absolute_1"
    if collection["action"].get("semantics") != expected_semantics:
        raise ValueError(f"collection.action.semantics 必须是 {expected_semantics}")
    if float(collection["capture"]["record_hz"]) > float(collection["capture"]["camera_fps"]):
        raise ValueError("record_hz 不能高于 camera_fps")
    if int(collection["capture"]["camera_fps"]) != int(cameras["fps"]):
        raise ValueError("collection.capture.camera_fps 必须与 hardware.cameras.fps 相同")
    expected_shape = [int(cameras["height"]), int(cameras["width"]), 3]
    if collection["capture"].get("image_shape") != expected_shape:
        raise ValueError(f"collection.capture.image_shape 必须为 {expected_shape}")

    safety = config["safety"]
    safety_wrist = safety["wrist"]
    if safety_wrist.get("coordinate") != "yaml_servo_zero_relative_raw":
        raise ValueError("safety.wrist.coordinate 必须是 yaml_servo_zero_relative_raw")
    raw_limits: list[float] = []
    for key in ("j1_min_raw", "j1_max_raw", "j2_min_raw", "j2_max_raw"):
        value = _require_finite_scalar(safety_wrist.get(key), f"safety.wrist.{key}", minimum=0, maximum=4095)
        if not value.is_integer():
            raise ValueError(f"safety.wrist.{key} 必须是整数")
        raw_limits.append(value)
    j1_min, j1_max, j2_min, j2_max = raw_limits
    if not j1_min < j1_max or not j2_min < j2_max:
        raise ValueError("safety.wrist J1/J2 raw 下界必须小于上界")
    if not j1_min <= servo_zero[0] <= j1_max or not j2_min <= servo_zero[1] <= j2_max:
        raise ValueError("poses.wrist.servo_zero_raw 超出 safety.wrist raw 限位")
    max_step_raw = _require_finite_scalar(
        safety_wrist.get("max_step_raw"),
        "safety.wrist.max_step_raw",
        minimum=1.0,
    )
    if not max_step_raw.is_integer():
        raise ValueError("safety.wrist.max_step_raw 必须是整数")
    maximum_step_from_velocity = float(hardware["wrist"]["target_limiter"]["max_velocity_raw_s"]) / float(
        hardware["wrist"]["command_hz"]
    )
    if max_step_raw > math.ceil(maximum_step_from_velocity):
        raise ValueError("safety.wrist.max_step_raw 不得超过 max_velocity_raw_s/command_hz")
    for key in ("feedback_timeout_s", "poll_period_s", "home_timeout_s", "settle_timeout_s"):
        _require_finite_scalar(safety_wrist.get(key), f"safety.wrist.{key}", minimum=0.001)
    for key in ("settle_tolerance_raw", "settle_samples"):
        value = _require_finite_scalar(safety_wrist.get(key), f"safety.wrist.{key}", minimum=1.0)
        if not value.is_integer():
            raise ValueError(f"safety.wrist.{key} 必须是整数")
    for key in (
        "acceleration",
        "command_duration_s",
        "stop_deceleration",
        "prediction_horizon_s",
        "max_linear_m_s",
        "max_angular_rad_s",
        "max_joint_step_rad",
        "max_joint_speed_rad_s",
    ):
        _require_finite_scalar(safety["ur5"][key], f"safety.ur5.{key}", minimum=0.001)
    translation_speed = _require_finite_scalar(
        teleop["modes"]["translation"]["linear_speed_m_s"],
        "teleop.modes.translation.linear_speed_m_s",
        minimum=0.001,
    )
    rotation_speed = _require_finite_scalar(
        teleop["modes"]["rotation"]["angular_speed_rad_s"],
        "teleop.modes.rotation.angular_speed_rad_s",
        minimum=0.001,
    )
    if not math.isclose(translation_speed, float(safety["ur5"]["teleop_linear_m_s"]), abs_tol=1e-12):
        raise ValueError("teleop 平移速度必须与 safety.ur5.teleop_linear_m_s 相同")
    if not math.isclose(rotation_speed, float(safety["ur5"]["teleop_angular_rad_s"]), abs_tol=1e-12):
        raise ValueError("teleop 旋转速度必须与 safety.ur5.teleop_angular_rad_s 相同")
    _require_finite_vector(safety["ur5"]["workspace_min_m"], 3, "safety.ur5.workspace_min_m")
    _require_finite_vector(safety["ur5"]["workspace_max_m"], 3, "safety.ur5.workspace_max_m")
    if require_real_ready:
        if safety["ur5"].get("joint_min_rad") is None or safety["ur5"].get("joint_max_rad") is None:
            raise ValueError("真实运动前必须从 PolyScope 填写 UR5 joint_min_rad/joint_max_rad")
        _require_finite_vector(safety["ur5"]["joint_min_rad"], 6, "safety.ur5.joint_min_rad")
        _require_finite_vector(safety["ur5"]["joint_max_rad"], 6, "safety.ur5.joint_max_rad")


def load_project_config(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    *,
    require_real_ready: bool = False,
) -> dict[str, dict[str, Any]]:
    """加载五份 YAML; 只做解析和一致性检查, 不连接任何硬件。"""
    root = Path(config_dir).expanduser().resolve()
    result = {Path(name).stem: _load_mapping(root / name) for name in CONFIG_FILES}
    validate_project_config(result, require_real_ready=require_real_ready)
    return result


def config_hash(config: dict[str, dict[str, Any]]) -> str:
    """生成跨机器稳定的配置 SHA-256。"""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--require-real-ready", action="store_true")
    args = parser.parse_args()
    config = load_project_config(args.config_dir, require_real_ready=args.require_real_ready)
    print(
        json.dumps(
            {
                "ok": True,
                "config_dir": str(args.config_dir.resolve()),
                "sha256": config_hash(config),
                "files": list(CONFIG_FILES),
                "gripper_backend": config["hardware"]["gripper"]["backend"],
                "camera_serials": {item["role"]: item["serial"] for item in config["hardware"]["cameras"]["devices"]},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
