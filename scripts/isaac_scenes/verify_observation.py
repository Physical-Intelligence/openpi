"""验证观测采集 — 加载场景，运行几帧，检查 DROID 观测格式。

用法:
    python scripts/isaac_scenes/verify_observation.py
"""
import os
import sys

# Blackwell (RTX 5060 Ti) + 驱动 580.65.06 的 GPU 着色器编译崩溃修复。
# 必须在 SimulationApp 创建前设置，否则 GUI 模式下 Vulkan viewport 初始化为黑屏/卡死。
os.environ.setdefault("__GL_SHADER_DISK_CACHE", "0")

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "active_gpu": 0,
    "multi_gpu": False,
    "open_usd": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "franka_tabletop.usd"
    ),
})

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera
from droid_observation import DroidObservationCollector

world = World(stage_units_in_meters=1.0)

# 从已加载的场景中包装已有 prim 并注册到场景
franka = world.scene.add(
    Franka(prim_path="/World/Franka", name="franka")
)
exterior_cam = Camera(
    prim_path="/World/ExteriorCamera",
    name="exterior_camera",
    resolution=(224, 224),
)
wrist_cam = Camera(
    prim_path="/World/Franka/panda_hand/wrist_camera",
    name="wrist_camera",
    resolution=(224, 224),
)

collector = DroidObservationCollector(exterior_cam, wrist_cam, franka)

world.reset()
collector.initialize()

# GUI 模式下场景渲染初始化需要时间，空跑几帧让 viewport 完成首帧
import time
warmup_start = time.time()
for i in range(60):
    world.step(render=True)
elapsed = time.time() - warmup_start
print(f"Warmup: 60 frames in {elapsed:.1f}s", flush=True)

# 再等待确保渲染管线完全就绪
time.sleep(5.0)

for i in range(10):
    world.step(render=True)

obs = collector.get_observation(prompt="pick up the red cube")

# ── 验证 ──────────────────────────────────────────────────────────
errors = []
expected_keys = {
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
    "prompt",
}
actual_keys = set(obs.keys())
if actual_keys != expected_keys:
    errors.append(f"Keys mismatch: got {actual_keys - expected_keys}, missing {expected_keys - actual_keys}")

ext = obs["observation/exterior_image_1_left"]
wrist = obs["observation/wrist_image_left"]
joints = obs["observation/joint_position"]
gripper = obs["observation/gripper_position"]
prompt = obs["prompt"]

if ext.shape != (224, 224, 3):
    errors.append(f"Exterior shape: {ext.shape}, expected (224,224,3)")
if ext.dtype != np.uint8:
    errors.append(f"Exterior dtype: {ext.dtype}, expected uint8")
if wrist.shape != (224, 224, 3):
    errors.append(f"Wrist shape: {wrist.shape}, expected (224,224,3)")
if wrist.dtype != np.uint8:
    errors.append(f"Wrist dtype: {wrist.dtype}, expected uint8")
if joints.shape != (7,):
    errors.append(f"Joint pos shape: {joints.shape}, expected (7,)")
if joints.dtype != np.float64:
    errors.append(f"Joint pos dtype: {joints.dtype}, expected float64")
if gripper.shape != (1,):
    errors.append(f"Gripper shape: {gripper.shape}, expected (1,)")
if not (0.0 <= float(gripper[0]) <= 1.0):
    errors.append(f"Gripper value {float(gripper[0])} out of [0,1]")
if prompt != "pick up the red cube":
    errors.append(f"Prompt mismatch: {prompt!r}")

if errors:
    print("FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All checks passed.")
    print(f"  Exterior:  {ext.shape} {ext.dtype} range=[{ext.min()}, {ext.max()}]")
    print(f"  Wrist:     {wrist.shape} {wrist.dtype} range=[{wrist.min()}, {wrist.max()}]")
    print(f"  Joints:    {joints.shape} {joints.dtype} values={np.round(joints, 3)}")
    print(f"  Gripper:   {float(gripper[0]):.4f}")
    print(f"  Prompt:    {prompt!r}")

# syntheticdata plugin 在手动 cam.destroy() 时可能崩溃，
# 直接让 simulation_app.close() 处理所有清理
simulation_app.close()