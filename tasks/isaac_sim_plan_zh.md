# Isaac Sim 环境部署与推理集成计划

> **目标**：在 Ubuntu 24.04 上安装 Isaac Sim 4.5+，搭建 Franka Panda 机器人仿真场景，通过 WebSocket 桥接至 openpi π₀.₅ 策略服务器，实现从仿真相机到动作执行的完整推理闭环。

---

## 0. 架构总览

```
┌──────────────────────────────┐      WebSocket (msgpack-numpy)       ┌──────────────────────────────┐
│        Isaac Sim 环境         │  ─────────────观测数据─────────────> │      openpi .venv/            │
│                              │                                      │                              │
│  • USD 场景 (Franka Panda)   │  <─────────────动作数据────────────── │  • 策略服务器                  │
│  • 相机渲染 → obs 字典        │                                      │  • π₀.₅ 推理 (JAX + GPU)      │
│  • 关节控制器 → 执行动作      │                                      │  • pi05_droid 权重             │
│                              │                                      │                              │
│  Python: isaacsim (pip)      │                                      │  Python: .venv/ (已有)        │
│  依赖: PyTorch, numpy, etc.  │                                      │  依赖: JAX 0.5.3, flax, etc.  │
└──────────────────────────────┘                                      └──────────────────────────────┘
```

**核心设计决策**：两个 Python 环境完全隔离，通过 WebSocket 通信。Isaac Sim 不需要 JAX，openpi 不需要 Isaac Sim 的仿真库。这是已打通的协议（见推理复现计划第三阶段）。

---

## 第一阶段：Isaac Sim 安装与环境配置

### 1.1 安装方式选择

采用 **pip 安装 Isaac Lab**（基于 Isaac Sim 4.5+），原因：
- 无需 Omniverse Launcher GUI 即可安装核心组件
- 支持 headless 模式，适合服务器/远程环境
- pip 依赖管理更透明，便于与现有项目共存

### 1.2 创建独立虚拟环境

```bash
# 方案 A（推荐）：在项目下创建专用 venv
python3.11 -m venv .venv-isaac

# 方案 B：使用 conda（如果已有 conda）
conda create -n isaacsim python=3.11
```

**Python 版本说明**：Isaac Sim 4.5+ 要求 Python 3.10 或 3.11。需先确认 Isaac Sim 4.5 的具体 Python 版本要求。

### 1.3 安装 Isaac Sim pip 包

```bash
source .venv-isaac/bin/activate

# 如果使用国内镜像（推荐）：
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Isaac Sim（根据官方最新文档调整包名）
pip install isaacsim-rl isaacsim-core  # Isaac Lab + 核心仿真

# 或安装完整 Isaac Sim（包含所有扩展）
pip install isaacsim
```

**注意事项**：
- Isaac Sim 首次运行时会自动下载 NVIDIA Omniverse Kit 运行时（数 GB），确保磁盘空间充足
- 当前 /home/srh/ 可用空间约 234 GB，预计足够
- 可能需要配置 `NVIDIA_ISAAC_SIM_PATH` 等环境变量

### 1.4 验证 GPU 渲染

```bash
source .venv-isaac/bin/activate

# 启动 headless 模式并验证 GPU 可用
python -c "
from isaacsim import SimulationApp

simulation_app = SimulationApp({'headless': True})
import omni.usd
print('Isaac Sim GPU rendering ready')

# 检查 GPU
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} (compute {cap[0]}.{cap[1]})')

# 显式验证 Blackwell (sm_120) 支持
assert 'sm_120' in torch.cuda.get_arch_list(), \
    'PyTorch 不支持 sm_120 (Blackwell)，需要升级到 cu128+ 版本'
print('Blackwell (sm_120) support: OK')
simulation_app.close()
"
```

**预期结果**：两张 RTX 5060 Ti 均可见，无 sm_120 警告，Blackwell 架构支持确认通过。

> **注意**：RTX 5060 Ti 为 Blackwell 架构 (compute capability 12.0 / sm_120)，需要 PyTorch ≥ 2.7.0+cu128 或 PyTorch ≥ 2.11.0+cu129/cu130。Isaac Sim 5.1.0 默认安装的 `torch==2.7.0+cu126` 不支持此 GPU，需手动升级。(已验证 `torch==2.12.0+cu130` 与 Isaac Sim 5.1.0 兼容。)

### 1.5 环境变量与启动脚本

创建 `scripts/isaac_env.sh` 统一管理环境变量：

```bash
# 激活 openpi 环境（策略服务器）
alias openpi-env='source /home/srh/VLA/openpi/.venv/bin/activate'

# 激活 Isaac Sim 环境
alias isaac-env='source /home/srh/VLA/openpi/.venv-isaac/bin/activate'
```

---

## 第二阶段：Franka Panda 基础场景搭建

### 2.1 创建 USD 场景

创建 `scripts/isaac_scenes/franka_tabletop.usd` 或使用 Python 脚本程序化生成：

```python
"""scripts/isaac_scenes/create_franka_scene.py — 创建基础 Franka Panda 桌面场景。"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
import numpy as np

# 初始化世界
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 加载 Franka Panda
# Isaac Sim 内置 Franka 资产路径
franka_usd = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
franka = world.scene.add(Robot(
    prim_path="/World/Franka",
    name="franka",
    usd_path=franka_usd,
    position=np.array([0.0, 0.0, 0.0]),
))

# 添加桌面
from omni.isaac.core.objects import DynamicCuboid
table = world.scene.add(DynamicCuboid(
    prim_path="/World/Table",
    name="table",
    position=np.array([0.5, 0.0, 0.0]),
    scale=np.array([0.5, 0.5, 0.05]),
    color=np.array([0.5, 0.3, 0.2]),
))

# 初始化物理
world.reset()

# 保存 USD 场景
from omni.usd import get_context
get_context().save_as_stage("/home/srh/VLA/openpi/scripts/isaac_scenes/franka_tabletop.usd")
print("Scene saved.")

simulation_app.close()
```

### 2.2 相机配置

DROID 格式需要两个相机视角：

| 相机 | 键名 | 位置 | 分辨率 |
|------|------|------|--------|
| 外部相机 1 (左) | `observation/exterior_image_1_left` | 机器人前方/侧面 | 224×224×3 |
| 腕部相机 (左) | `observation/wrist_image_left` | Franka 末端执行器 | 224×224×3 |

```python
# 添加外部相机
exterior_camera = world.scene.add(Camera(
    prim_path="/World/ExteriorCamera",
    name="exterior_camera",
    position=np.array([0.8, 0.3, 0.6]),
    target=np.array([0.3, 0.0, 0.2]),
    resolution=(224, 224),
))

# 添加腕部相机（附着于 Franka 末端）
wrist_camera = world.scene.add(Camera(
    prim_path="/World/Franka/panda_hand/wrist_camera",
    name="wrist_camera",
    position=np.array([0.0, 0.05, -0.05]),
    orientation=np.array([0, 0, 0, 1]),
    resolution=(224, 224),
))
```

### 2.3 添加仿真物体

在桌面放置可交互物体，用于后续 pick-and-place 等任务：

```python
objects = []
for name, pos, color in [
    ("red_cube", (0.4, 0.1, 0.35), (1.0, 0.0, 0.0)),
    ("blue_cube", (0.5, -0.1, 0.35), (0.0, 0.0, 1.0)),
    ("green_cube", (0.45, 0.0, 0.35), (0.0, 1.0, 0.0)),
]:
    obj = world.scene.add(DynamicCuboid(
        prim_path=f"/World/Objects/{name}",
        name=name,
        position=np.array(pos),
        scale=np.array([0.03, 0.03, 0.03]),
        color=np.array(color),
    ))
    objects.append(obj)
```

---

## 第三阶段：观测管线 — Isaac Sim → DROID 格式

### 3.1 观测采集类

创建 `scripts/isaac_scenes/droid_observation.py`：

```python
"""DROID 格式观测采集器 — 从 Isaac Sim 场景提取观测数据。"""
import numpy as np
from typing import Dict


class DroidObservationCollector:
    """从仿真场景采集 DROID 格式的观测数据。

    输出格式与 pi05_droid 策略预期完全一致：
    {
        "observation/exterior_image_1_left": np.ndarray(224, 224, 3),  # uint8
        "observation/wrist_image_left":      np.ndarray(224, 224, 3),  # uint8
        "observation/joint_position":        np.ndarray(7,),           # float64
        "observation/gripper_position":      np.ndarray(1,),           # float64
        "prompt": str,                                                # 任务指令
    }
    """

    def __init__(self, exterior_camera, wrist_camera, franka_articulation):
        self._exterior = exterior_camera
        self._wrist = wrist_camera
        self._franka = franka_articulation

    def get_observation(self, prompt: str = "") -> Dict:
        # 获取关节位置（Franka 7 个关节）
        joint_positions = self._franka.get_joint_positions()  # (7,)

        # 获取夹爪位置
        gripper_pos = self._franka.get_gripper_position()  # scalar

        # 渲染相机图像
        exterior_img = self._capture_rgb(self._exterior)  # (224, 224, 3) uint8
        wrist_img = self._capture_rgb(self._wrist)        # (224, 224, 3) uint8

        return {
            "observation/exterior_image_1_left": exterior_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_positions.astype(np.float64),
            "observation/gripper_position": np.array([gripper_pos], dtype=np.float64),
            "prompt": prompt,
        }

    def _capture_rgb(self, camera) -> np.ndarray:
        """触发渲染并获取 RGBA → RGB uint8 图像。"""
        # Isaac Sim 相机渲染管线
        # 具体 API 取决于 Isaac Sim 4.5 的 Camera 接口
        ...
```

### 3.2 图像格式验证

确保输出的图像与真实 DROID 数据格式完全一致：

| 属性 | 要求 |
|------|------|
| 分辨率 | 224 × 224 |
| 通道 | RGB (3 通道) |
| 数据类型 | uint8 |
| 值范围 | [0, 255] |
| 色彩空间 | sRGB（与 DROID 数据集一致） |
| 图像方向 | 标准（不翻转；如果 DROID 预期翻转则在此处理） |

---

## 第四阶段：WebSocket 桥接 — Isaac Sim ↔ 策略服务器

### 4.1 策略客户端封装

创建 `scripts/isaac_scenes/policy_bridge.py`：

```python
"""策略桥接器 — Isaac Sim ↔ 策略服务器 WebSocket 通信。"""
import sys
sys.path.insert(0, "/home/srh/VLA/openpi/packages/openpi-client/src")

import numpy as np
from openpi_client import websocket_client_policy
from openpi_client.action_chunk_broker import ActionChunkBroker


class PolicyBridge:
    """管理与策略服务器的 WebSocket 连接，封装观测→动作调用。"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port
        )
        metadata = self._policy.get_server_metadata()
        action_horizon = metadata.get("action_horizon", 15)
        self._broker = ActionChunkBroker(action_horizon=action_horizon)

    def infer(self, obs: dict) -> np.ndarray:
        """获取单个动作。

        参数:
            obs: DROID 格式观测字典
        返回:
            action: (8,) numpy 数组 [joint_vel(7), gripper(1)]
        """
        result = self._policy.infer(obs)
        action_chunk = result["actions"]  # (15, 8)
        return self._broker.get_action(action_chunk)

    def close(self):
        self._policy.close()
```

### 4.2 启动顺序

```bash
# 终端 1：启动 openpi 策略服务器
source .venv/bin/activate
python scripts/serve_policy.py --env droid --port 8000

# 终端 2：运行 Isaac Sim 主循环
source .venv-isaac/bin/activate
python scripts/isaac_scenes/main_loop.py
```

**启动检查清单**：
- [ ] 策略服务器输出 "Creating server..." 就绪日志
- [ ] Isaac Sim 主循环打印 "Connected to policy server at ws://localhost:8000"
- [ ] 首次推理完成（含 JIT 编译预热）

---

## 第五阶段：动作执行 — 策略输出 → Franka 关节控制

### 5.1 动作解码

策略服务器返回 `(15, 8)` 动作数组，需要：

```python
def decode_action(raw_action: np.ndarray) -> dict:
    """将 openpi 原始动作解码为 Franka 控制指令。

    参数:
        raw_action: (8,) numpy 数组
            - raw_action[:7]: 归一化的关节速度（或位置增量）
            - raw_action[7]: 夹爪指令 (0=合, 1=开)
    返回:
        {
            "joint_velocities": np.ndarray(7,),  # 关节速度 (rad/s)
            "gripper_command": float,             # 0.0 (合) 或 1.0 (开)
        }
    """
    # 速度缩放（需要根据 pi05_droid 的归一化范围确定缩放因子）
    velocity_scale = 1.0  # TBD: 根据实际归一化范围调整
    joint_velocities = raw_action[:7] * velocity_scale

    # 夹爪二值化
    gripper_command = 1.0 if raw_action[7] > 0.5 else 0.0

    return {
        "joint_velocities": joint_velocities,
        "gripper_command": gripper_command,
    }
```

### 5.2 Franka 控制器

```python
class FrankaController:
    """Isaac Sim 中 Franka Panda 的关节速度控制器。"""

    def __init__(self, franka_articulation):
        self._franka = franka_articulation
        self._dof = 7

    def apply_action(self, decoded: dict):
        """应用动作到仿真机器人。"""
        # 设置关节速度目标
        self._franka.set_joint_velocity_targets(decoded["joint_velocities"])

        # 设置夹爪
        self._franka.set_gripper_position(decoded["gripper_command"])
```

### 5.3 控制频率

DROID 原始控制频率为 15 Hz。仿真中的控制循环：

```python
import time

SIMULATION_DT = 1.0 / 60.0    # 物理步长 60 Hz
POLICY_DT = 1.0 / 15.0        # 策略控制频率 15 Hz

while simulation_app.is_running():
    t = time.time()

    # 物理步进
    world.step(render=True)

    # 每 POLICY_DT 秒调用一次策略
    if t - last_policy_time >= POLICY_DT:
        obs = collector.get_observation(prompt="pick up the red cube")
        action = bridge.infer(obs)
        decoded = decode_action(action)
        controller.apply_action(decoded)
        last_policy_time = t
```

---

## 第六阶段：端到端推理主循环

### 6.1 主循环脚本

创建 `scripts/isaac_scenes/main_loop.py`：

```python
"""端到端推理主循环 — Isaac Sim 仿真 + π₀.₅ 策略实时控制。

用法:
    # Headless（默认）
    python scripts/isaac_scenes/main_loop.py

    # GUI 模式（可视化调试）
    python scripts/isaac_scenes/main_loop.py --gui

    # 录制模式（保存观测/动作对）
    python scripts/isaac_scenes/main_loop.py --record

环境变量:
    ISAAC_HEADLESS=0      启用 GUI
    POLICY_HOST=localhost  策略服务器地址
    POLICY_PORT=8000       策略服务器端口
"""
import argparse
import time
import numpy as np

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="启用 GUI 渲染")
    parser.add_argument("--record", action="store_true", help="录制观测/动作数据")
    parser.add_argument("--num_steps", type=int, default=100, help="推理步数")
    parser.add_argument("--prompt", type=str, default="", help="任务指令")
    args = parser.parse_args()

    # 初始化
    simulation_app = SimulationApp({"headless": not args.gui})

    # TODO: 创建场景、加载 Franka、配置相机、连接策略服务器

    # 主循环
    for step in range(args.num_steps):
        t0 = time.time()

        # 1. 采集观测
        obs = collector.get_observation(prompt=args.prompt)

        # 2. 调用策略
        action = bridge.infer(obs)

        # 3. 执行动作
        controller.apply_action(decode_action(action))

        # 4. 物理步进
        world.step(render=True)

        elapsed = (time.time() - t0) * 1000
        if step % 10 == 0:
            print(f"Step {step:3d}: {elapsed:.1f}ms total")

    print(f"\n平均推理延迟: {np.mean(latencies):.1f}ms")
    print(f"平均循环耗时: {np.mean(loop_times):.1f}ms")

    simulation_app.close()


if __name__ == "__main__":
    main()
```

### 6.2 验证清单

端到端测试完成后确认：

- [ ] Isaac Sim 成功启动（headless 模式），无 CUDA/渲染错误
- [ ] Franka Panda 模型正确加载，关节可控制
- [ ] 两个相机渲染正常，图像分辨率 224×224
- [ ] 观测字典格式与 `DroidInputs` 完全匹配
- [ ] WebSocket 连接成功，客户端收到服务器元数据
- [ ] 首次推理返回有效的 (15, 8) 动作张量
- [ ] 动作值在合理范围内（非 NaN，非无穷大）
- [ ] Franka 关节随策略输出运动
- [ ] 单次循环耗时 < ~200ms（含渲染 + 推理 + 物理）
- [ ] GUI 模式可正常可视化（手动验证）

---

## 第七阶段：调试与性能分析

### 7.1 延迟分解

```
总循环耗时 = 相机渲染 + 网络传输(obs) + 服务端推理 + 网络传输(action) + 物理步进
```

各阶段预期耗时：

| 阶段 | 预期 | 备注 |
|------|------|------|
| 相机渲染 | ~10ms | GPU 渲染 224×224 |
| 网络传输 (往返) | <5ms | 本地回环 |
| 服务端推理 | <100ms | JIT 编译后 |
| 物理步进 | ~5ms | 60 Hz 固定步长 |
| **总计** | **~120ms** | 可支持 8 Hz 实时控制 |

### 7.2 调试要点

1. **GPU 显存**：Isaac Sim 渲染 + JAX 推理各占一张 GPU？还是共享一张？
   - RTX 5060 Ti (16 GB)：Isaac Sim 预计占用 2-4 GB 渲染，JAX 模型占用 6-10 GB
   - 理想情况：Isaac Sim 用 GPU 0（渲染），JAX 用 GPU 1（推理）
2. **nvidia-smi 监控**：分别观察两张 GPU 的显存和利用率
3. **Isaac Sim 日志**：`~/.nvidia-omniverse/logs/` 下的 Kit 日志

---

## 依赖与兼容性风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Isaac Sim 4.5 pip 包 Python 版本要求 | 可能与系统 Python 不兼容 | 使用独立 venv，确认 Python 版本要求后安装 |
| CUDA 13.0 兼容性 | Isaac Sim 可能要求特定 CUDA 版本 | 检查 Isaac Sim 4.5 的 CUDA 要求；driver 向后兼容 |
| 国内下载速度 | NVIDIA Omniverse 运行时首次下载缓慢 | 配置代理或寻找国内镜像源 |
| 双 GPU 分配 | Isaac Sim 与 openpi 竞争同一 GPU | 通过 CUDA_VISIBLE_DEVICES 隔离 |
| RTX 5060 Ti 太新 | Isaac Sim 可能未测试此 GPU | 验证渲染可用后再深入开发 |

---

## 文件结构总览

完成后的项目文件结构：

```
scripts/
├── isaac_env.sh                  # 环境变量与别名
├── isaac_scenes/
│   ├── create_franka_scene.py    # 场景生成脚本
│   ├── franka_tabletop.usd       # 保存的 USD 场景（程序化生成）
│   ├── droid_observation.py      # DROID 格式观测采集器
│   ├── policy_bridge.py          # WebSocket 策略桥接器
│   └── main_loop.py              # 端到端推理主循环
├── serve_policy.py               # (已有) 策略服务器入口
└── verify_inference.py           # (已有) 推理冒烟测试

.venv/                             # (已有) openpi Python 环境
.venv-isaac/                       # (新增) Isaac Sim Python 环境
```

---

## 时间估算

| 阶段 | 内容 | 预估耗时 |
|------|------|---------|
| 1 | Isaac Sim pip 安装 + GPU 验证 | 1-2 小时（含下载） |
| 2 | Franka 场景搭建 + 相机配置 | 1 小时 |
| 3 | DROID 观测管线 | 1 小时 |
| 4 | WebSocket 桥接 | 0.5 小时（已有基础设施） |
| 5 | 动作解码 + Franka 控制 | 1 小时 |
| 6 | 端到端主循环 + 测试 | 1 小时 |
| **总计** | | **5-7 小时**（含调试） |

---

## 后续扩展方向

- [ ] 多物体 pick-and-place 任务环境
- [ ] 域随机化（光照、纹理、物体位姿）
- [ ] 多视角相机（深度图、分割掩码）
- [ ] 并行仿真（多个 Franka 场景同时运行）
- [ ] Isaac Sim → 真实机器人 sim2real 迁移
- [ ] π₀.₅ 自回归模式推理（用于长周期任务）