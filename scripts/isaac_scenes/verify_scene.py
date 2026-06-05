"""验证场景：掉落方块测试 — 确认 Isaac Sim 物理/渲染管线正常。

用法:
    python scripts/isaac_scenes/verify_scene.py          # headless
    python scripts/isaac_scenes/verify_scene.py --gui    # GUI
"""
import argparse
import os

# Blackwell (RTX 5060 Ti) + 驱动 580.65.06: 禁用 GL 着色器磁盘缓存，
# 避免 libnvidia-gpucomp.so 在 Vulkan/RTX 初始化时崩溃或黑屏卡死。
os.environ.setdefault("__GL_SHADER_DISK_CACHE", "0")
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true", help="启用 GUI 渲染 (Blackwell: 不稳定)")
args = parser.parse_args()

config = {
    "headless": not args.gui,
    "active_gpu": 0,
    "multi_gpu": False,
}
if not args.gui:
    config["disable_viewport_updates"] = True

sim = SimulationApp(config)

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World()
world.scene.add_default_ground_plane()
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/cube",
        name="cube",
        position=np.array([0, 0, 1.0]),
        scale=np.array([0.5, 0.5, 0.5]),
    )
)
world.reset()

for i in range(200):
    world.step(render=True)

print("Verification complete: 200 steps rendered successfully.")
sim.close()