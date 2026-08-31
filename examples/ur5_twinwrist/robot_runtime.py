# ruff: noqa: RUF001, RUF002, RUF003
"""统一安全过滤、FakeHardware 与项目内真机运行入口。"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import argparse
import concurrent.futures
from contextlib import suppress
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np

try:
    from .config_loader import DEFAULT_CONFIG_DIR
    from .config_loader import config_hash
    from .config_loader import load_project_config
except ImportError:  # 兼容 ``uv run examples/.../robot_runtime.py``
    from config_loader import DEFAULT_CONFIG_DIR
    from config_loader import config_hash
    from config_loader import load_project_config

MOTION_CONFIRMATION = "I_UNDERSTAND_REAL_ROBOT_MOTION"


class RobotHardware(ABC):
    """训练/推理代码看到的最小硬件合同。"""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_observation(self) -> dict[str, Any]: ...

    @abstractmethod
    def send_action(self, action: np.ndarray) -> Any: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class FakeHardware(RobotHardware):
    """不导入、不连接 vendor 驱动的软件验收硬件。"""

    def __init__(
        self,
        shape: tuple[int, int, int] = (480, 640, 3),
        *,
        servo_zero_raw: tuple[int, int] = (0, 0),
    ) -> None:
        self.shape = shape
        self.servo_zero_raw = tuple(int(value) for value in servo_zero_raw)
        if len(self.servo_zero_raw) != 2:
            raise ValueError("FakeHardware servo_zero_raw 必须是 2 维")
        self.qpos = np.zeros(9, np.float32)
        self.sent_actions: list[np.ndarray] = []
        self.connected = False
        self._sequence = 0

    def connect(self) -> None:
        self.connected = True

    def get_observation(self) -> dict[str, Any]:
        if not self.connected:
            raise RuntimeError("FakeHardware 尚未连接")
        now = time.monotonic_ns()
        self._sequence += 1
        image = np.full(self.shape, 32, np.uint8)
        return {
            "images": {name: image.copy() for name in ("front", "side", "top")},
            "qpos": self.qpos.copy(),
            "tcp_pose": np.asarray([0, 0, 0.2, 0, 0, 0], np.float32),
            "timestamp_ns": now,
            "monotonic_ns": now,
            "camera_timestamps_ns": dict.fromkeys(("front", "side", "top"), now),
            "camera_host_timestamps_ns": dict.fromkeys(("front", "side", "top"), now),
            "camera_sequences": dict.fromkeys(("front", "side", "top"), self._sequence),
            "hardware_health": {
                "ok": True,
                "emergency_stop": False,
                "protective_stop": False,
                "motion_enabled": False,
                "wrist": {
                    "coordinate": "yaml_servo_zero_relative_raw",
                    "servo_zero_raw": self.servo_zero_raw,
                    "motor_position_raw": self.servo_zero_raw,
                },
                "gripper": {"state_source": "commanded"},
            },
        }

    def send_action(self, action: np.ndarray) -> np.ndarray:
        parsed = np.asarray(action, np.float32).copy()
        if parsed.shape != (9,) or not np.isfinite(parsed).all():
            raise ValueError("FakeHardware action 必须是 9 维有限数")
        # 模拟 Dynamixel/OpenRB 的整数 raw 量化，使 Fake HDF5 也记录“实际下发回执”。
        parsed[6:8] = np.rint(parsed[6:8])
        self.sent_actions.append(parsed)
        self.qpos[6:] = parsed[6:]
        return parsed

    def stop(self) -> None:
        return None

    def close(self) -> None:
        self.connected = False


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeError(f"{name} 必须是 {size} 维有限数")
    return result


def _limit_norm(values: np.ndarray, maximum: float) -> None:
    norm = float(np.linalg.norm(values))
    if norm > maximum and norm > 0.0:
        values *= maximum / norm


def safety_filter(
    action: np.ndarray,
    observation: dict[str, Any],
    config: dict[str, Any],
    *,
    now_ns: int | None = None,
) -> np.ndarray:
    """把任意 9D 候选动作变为 YAML 合同允许的最终硬件命令。

    该函数不访问硬件。不可修复的状态问题直接抛异常；速度、腕目标和夹爪
    位置在有限安全范围内裁剪。真实发送前仍由每个底层控制器再次校验。
    """

    action = _finite_vector(action, 9, "action").copy()
    safety = config.get("safety", config)
    timing = safety.get("timing", safety)
    ur5 = safety.get("ur5", safety)
    wrist = safety.get("wrist", safety)
    current_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
    observation_ns = int(observation.get("timestamp_ns", observation.get("monotonic_ns", -1)))
    age_ns = current_ns - observation_ns
    timeout_ns = int(float(timing.get("observation_timeout_ms", 200.0)) * 1e6)
    if age_ns < 0 or age_ns > timeout_ns:
        raise RuntimeError(f"observation 已过期: {age_ns / 1e6:.1f} ms")

    health = observation.get("hardware_health", {})
    if not health.get("ok", False):
        raise RuntimeError("hardware_health.ok=false")
    if health.get("emergency_stop", False) or health.get("protective_stop", False):
        raise RuntimeError("急停或保护停已触发")

    qpos = _finite_vector(observation.get("qpos"), 9, "observation.qpos")
    joint_min = ur5.get("joint_min_rad")
    joint_max = ur5.get("joint_max_rad")
    if (joint_min is None) != (joint_max is None):
        raise RuntimeError("UR5 关节软限位上下界必须同时设置")
    if joint_min is not None:
        lower = _finite_vector(joint_min, 6, "UR5 关节下限")
        upper = _finite_vector(joint_max, 6, "UR5 关节上限")
        if np.any(qpos[:6] < lower) or np.any(qpos[:6] > upper):
            raise RuntimeError("UR5 实际关节越过 YAML 软限位")

    linear_limit = float(ur5.get("max_linear_m_s", ur5.get("max_tcp_linear_m_s", 0.10)))
    angular_limit = float(ur5.get("max_angular_rad_s", ur5.get("max_tcp_angular_rad_s", 0.30)))
    _limit_norm(action[:3], linear_limit)
    _limit_norm(action[3:6], angular_limit)

    tcp = _finite_vector(observation.get("tcp_pose"), 6, "observation.tcp_pose")
    xyz_min = _finite_vector(ur5.get("workspace_min_m", [-0.8, -0.8, 0.08]), 3, "工作空间下限")
    xyz_max = _finite_vector(ur5.get("workspace_max_m", [0.8, 0.8, 1.2]), 3, "工作空间上限")
    if np.any(tcp[:3] < xyz_min) or np.any(tcp[:3] > xyz_max):
        raise RuntimeError("当前 TCP 已在 YAML 工作空间之外")
    prediction_horizon_s = float(ur5.get("prediction_horizon_s", 0.25))
    if not math.isfinite(prediction_horizon_s) or prediction_horizon_s <= 0.0:
        raise RuntimeError("UR5 prediction_horizon_s 必须是正有限数")
    projected = tcp[:3] + action[:3] * prediction_horizon_s
    if np.any(projected < xyz_min) or np.any(projected > xyz_max):
        raise RuntimeError("当前 speedL 命令会把 TCP 推出工作空间")

    poses = config.get("poses", {})
    wrist_pose = poses.get("wrist", {}) if isinstance(poses, dict) else {}
    servo_zero = _finite_vector(wrist_pose.get("servo_zero_raw"), 2, "手腕 YAML servo zero")
    wrist_health = health.get("wrist", {})
    if wrist_health:
        if wrist_health.get("coordinate") != "yaml_servo_zero_relative_raw":
            raise RuntimeError("手腕反馈坐标不是 yaml_servo_zero_relative_raw")
        reported_zero = _finite_vector(wrist_health.get("servo_zero_raw"), 2, "手腕反馈 servo zero")
        if not np.array_equal(reported_zero, servo_zero):
            raise RuntimeError(
                f"手腕反馈 servo zero {reported_zero.tolist()} 与 YAML {servo_zero.tolist()} 不一致"
            )
    wrist_min = np.asarray(
        [float(wrist["j1_min_raw"]), float(wrist["j2_min_raw"])], dtype=np.float64
    ) - servo_zero
    wrist_max = np.asarray(
        [float(wrist["j1_max_raw"]), float(wrist["j2_max_raw"])], dtype=np.float64
    ) - servo_zero
    wrist_target = np.clip(action[6:8], wrist_min, wrist_max)
    max_step = float(wrist["max_step_raw"])
    action[6:8] = np.clip(wrist_target, qpos[6:8] - max_step, qpos[6:8] + max_step)
    action[8] = np.clip(action[8], 0.0, 1.0)
    return action.astype(np.float32)


def build_policy_observation(observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    """按训练字段构造 policy server 输入，腕维度保持 YAML 零位相对 raw。"""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("infer 模式需要非空 prompt")
    images = observation.get("images")
    if not isinstance(images, dict) or set(images) != {"front", "side", "top"}:
        raise ValueError("policy observation 必须包含 front/side/top 三路图像")
    result: dict[str, Any] = {
        "observation.state": _finite_vector(observation.get("qpos"), 9, "observation.qpos").astype(
            np.float32
        ),
        "prompt": prompt.strip(),
    }
    for role in ("front", "side", "top"):
        image = np.asarray(images[role])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"policy 图像 {role} 必须是 HWC uint8 RGB")
        result[f"observation.images.{role}"] = np.ascontiguousarray(image)
    return result


def validate_action_chunk(result: dict[str, Any]) -> np.ndarray:
    """严格校验 policy server 返回的腕相对 raw action chunk。"""

    if not isinstance(result, dict) or "actions" not in result:
        raise ValueError("policy server 返回值缺少 actions")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 9:
        raise ValueError(f"policy action chunk 必须是 (H,9)，实际 {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("policy action chunk 含 NaN 或 Inf")
    return actions


def _query_policy(client: Any, observation: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """调用官方 openpi-client，并在超时时关闭 websocket 使机器人 fail-closed。"""

    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("policy timeout 必须是正有限数")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-policy")
    future = executor.submit(client.infer, observation)
    try:
        result = future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        connection = getattr(client, "_ws", None)
        if connection is not None:
            with suppress(Exception):
                connection.close()
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"policy 请求超过 {timeout_s:.3f}s") from exc
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    if not isinstance(result, dict):
        raise ValueError("policy server 返回值不是 mapping")
    return result


def _connect_policy_client(host: str, port: int, timeout_s: float) -> Any:
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("第一版 policy server 只允许 localhost")
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=timeout_s) as response:
            if response.status != 200:
                raise ConnectionError(f"policy server health status={response.status}")
    except (OSError, urllib.error.URLError) as exc:
        raise ConnectionError(f"policy server {host}:{port} 不可达") from exc
    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy(host=host, port=port)


def _configured_hardware(args: argparse.Namespace, project: dict[str, Any]) -> RobotHardware:
    if args.fake:
        return FakeHardware(
            servo_zero_raw=tuple(int(value) for value in project["poses"]["wrist"]["servo_zero_raw"])
        )
    try:
        from .local_hardware import LocalRobotHardware
    except ImportError:
        from local_hardware import LocalRobotHardware

    return LocalRobotHardware(project, enable_motion=args.enable_motion)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "infer"), default="infer")
    parser.add_argument("--shadow", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-timeout-s", type=float)
    parser.add_argument("--chunk-steps", type=int, default=1)
    parser.add_argument("--gripper-backend", choices=("hiwonder", "feetech"))
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args(argv)

    shadow = True if args.mode == "infer" and args.shadow is None else bool(args.shadow)
    if args.enable_motion and args.confirm != MOTION_CONFIRMATION:
        parser.error(f"真实运动还必须提供 --confirm {MOTION_CONFIRMATION}")
    if not shadow and not args.enable_motion:
        parser.error("非 shadow 模式必须显式提供 --enable-motion")
    if not 1 <= args.chunk_steps <= 4:
        parser.error("--chunk-steps 必须位于 [1,4]；首次真机保持 1")
    project = load_project_config(args.config_dir, require_real_ready=args.enable_motion)
    if args.gripper_backend:
        project = deepcopy(project)
        project["hardware"]["gripper"]["backend"] = args.gripper_backend

    hardware = _configured_hardware(args, project)
    hardware.connect()
    try:
        observation = hardware.get_observation()
        policy_timing = project["safety"]["timing"]
        policy_timeout_s = (
            float(policy_timing["policy_timeout_ms"]) / 1000.0
            if args.policy_timeout_s is None
            else float(args.policy_timeout_s)
        )
        action_chunk: np.ndarray
        server_timing: dict[str, Any] | None = None
        if args.mode == "infer":
            client = _connect_policy_client(args.host, args.port, policy_timeout_s)
            policy_result = _query_policy(
                client,
                build_policy_observation(observation, args.prompt or ""),
                policy_timeout_s,
            )
            action_chunk = validate_action_chunk(policy_result)
            server_timing = policy_result.get("server_timing")
        else:
            action_chunk = np.zeros((1, 9), dtype=np.float32)
        executed: list[list[float]] = []
        for index in range(min(args.chunk_steps, len(action_chunk))):
            if index:
                observation = hardware.get_observation()
            action = safety_filter(action_chunk[index], observation, project)
            executed.append(action.tolist())
            if not shadow and args.enable_motion:
                hardware.send_action(action)
        result = {
            "mode": args.mode,
            "shadow": shadow,
            "fake": args.fake,
            "config_sha256": config_hash(project),
            "wrist_coordinate": "yaml_servo_zero_relative_raw",
            "wrist_servo_zero_raw": project["poses"]["wrist"]["servo_zero_raw"],
            "action_chunk_shape": list(action_chunk.shape),
            "candidate_first_action": action_chunk[0].tolist(),
            "filtered_actions": executed,
            "server_timing": server_timing,
            "motion_sent": bool(not shadow and args.enable_motion and executed),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException:
        hardware.stop()
        raise
    finally:
        hardware.close()


if __name__ == "__main__":
    raise SystemExit(main())
