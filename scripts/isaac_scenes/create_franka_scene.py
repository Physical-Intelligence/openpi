"""创建基础 Franka Panda 桌面场景并保存为 USD 文件。

包括:
  - Franka Panda 机器人（2.1）
  - 外部相机 + 腕部相机，224×224（2.2）
  - 桌面彩色方块（2.3）

用法:
    python scripts/isaac_scenes/create_franka_scene.py
"""
import os
import numpy as np

os.environ.setdefault("__GL_SHADER_DISK_CACHE", "0")
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
    "active_gpu": 0,
    "multi_gpu": False,
})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera


def _look_at_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """计算相机从 eye 指向 target 的四元数 (scalar-first: w,x,y,z)。

    USD 相机沿 -Z 轴朝向目标，+Y 为上方向。
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    camera_up = np.cross(forward, right)
    # 旋转矩阵: columns = right, camera_up, forward
    rot = np.column_stack((right, camera_up, forward))
    # 矩阵转四元数 (scalar-first)
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# ── 2.1 Franka Panda ──────────────────────────────────────────────
franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0]),
    )
)

# ── 2.1 桌面 ──────────────────────────────────────────────────────
table = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.5, 0.0, 0.0]),
        scale=np.array([0.5, 0.5, 0.05]),
        color=np.array([0.5, 0.3, 0.2]),
    )
)

# ── 2.2 外部相机 (左) — observation/exterior_image_1_left ─────────
exterior_eye = np.array([0.8, 0.3, 0.6])
exterior_target = np.array([0.35, 0.0, 0.25])
exterior_camera = world.scene.add(
    Camera(
        prim_path="/World/ExteriorCamera",
        name="exterior_camera",
        resolution=(224, 224),
        position=exterior_eye,
        orientation=_look_at_quat(exterior_eye, exterior_target),
    )
)

# ── 2.2 腕部相机 (左) — observation/wrist_image_left ──────────────
wrist_prim_path = "/World/Franka/panda_hand/wrist_camera"
wrist_camera = world.scene.add(
    Camera(
        prim_path=wrist_prim_path,
        name="wrist_camera",
        resolution=(224, 224),
        translation=np.array([0.0, 0.05, -0.05]),
    )
)

# ── 2.3 仿真物体 ──────────────────────────────────────────────────
objects = []
for name, pos, color in [
    ("red_cube",   (0.40,  0.10, 0.35), (1.0, 0.0, 0.0)),
    ("blue_cube",  (0.50, -0.10, 0.35), (0.0, 0.0, 1.0)),
    ("green_cube", (0.45,  0.00, 0.35), (0.0, 1.0, 0.0)),
]:
    obj = world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/Objects/{name}",
            name=name,
            position=np.array(pos),
            scale=np.array([0.03, 0.03, 0.03]),
            color=np.array(color),
        )
    )
    objects.append(obj)

# ── 初始化物理 ─────────────────────────────────────────────────────
world.reset()

# ── 保存 USD ──────────────────────────────────────────────────────
from omni.usd import get_context

# 先保存为 USDA（文本格式，可人工验证场景结构）
output_dir = os.path.dirname(os.path.abspath(__file__))
get_context().save_as_stage(os.path.join(output_dir, "franka_tabletop.usda"))
# 再保存为 USD（二进制格式，供后续脚本高效加载）
get_context().save_as_stage(os.path.join(output_dir, "franka_tabletop.usd"))
print(f"Scene saved to {output_dir}/franka_tabletop.usd(a)")

simulation_app.close()