"""DROID 格式观测采集器 — 从 Isaac Sim 场景提取观测数据。

输出格式与 pi05_droid 策略预期完全一致:
    {
        "observation/exterior_image_1_left": np.ndarray(224, 224, 3),  # uint8
        "observation/wrist_image_left":      np.ndarray(224, 224, 3),  # uint8
        "observation/joint_position":        np.ndarray(7,),           # float64
        "observation/gripper_position":      np.ndarray(1,),           # float64
        "prompt": str,                                                # 任务指令
    }
"""

from typing import Dict, Optional
import numpy as np
from isaacsim.sensors.camera import Camera


class DroidObservationCollector:
    FRANKA_ARM_DOF = 7
    GRIPPER_OPEN = 0.05  # 夹爪完全张开时的关节值(m)

    def __init__(
        self,
        exterior_camera: Camera,
        wrist_camera: Camera,
        franka,
    ):
        self._exterior = exterior_camera
        self._wrist = wrist_camera
        self._franka = franka

    def initialize(self, physics_sim_view=None):
        for cam in (self._exterior, self._wrist):
            cam.initialize(physics_sim_view=physics_sim_view)

    def get_observation(self, prompt: str = "") -> Dict:
        joint_positions = self._franka.get_joint_positions()

        arm_joints = joint_positions[: self.FRANKA_ARM_DOF].astype(np.float64)
        finger_joints = joint_positions[self.FRANKA_ARM_DOF:]

        # 归一化夹爪: 0.0 (闭合) ~ 1.0 (张开)
        gripper_pos = np.clip(np.mean(finger_joints) / self.GRIPPER_OPEN, 0.0, 1.0)

        exterior_img = self._capture_rgb(self._exterior)
        wrist_img = self._capture_rgb(self._wrist)

        return {
            "observation/exterior_image_1_left": exterior_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": arm_joints,
            "observation/gripper_position": np.array([gripper_pos], dtype=np.float64),
            "prompt": prompt,
        }

    @staticmethod
    def _capture_rgb(camera: Camera) -> np.ndarray:
        data = camera.get_rgb(device="cpu")
        if data is None:
            return np.zeros(tuple(camera._resolution) + (3,), dtype=np.uint8)
        return data.astype(np.uint8)