# π₀.₅ 轻量级推理复现计划

> **目标**：(1) 确认硬件支持 π₀.₅ 推理；(2) 使用 VSCode 调试器单步跟踪模型执行细节；(3) 打通 server-client 架构，为后续 Isaac Sim 集成做准备。

---

## 环境确认

| 项目 | 状态 |
|------|------|
| GPU | 2× RTX 5060 Ti (16 GB 每卡) |
| JAX | 0.5.3 + CUDA（双卡均已识别） |
| Python 虚拟环境 | `.venv/` 位于工作区根目录 |
| 模型权重 | `pi05_droid` 已缓存至 `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/` |
| VSCode | debugpy 可用，已有基础 launch.json |

---

## 第一阶段：本地推理冒烟测试

**目标**：在不启动任何服务端的情况下，确认模型能正常加载并输出有效动作。

### 步骤 1.1 — 创建测试脚本

创建 `scripts/verify_inference.py`：

```python
"""最小化 π₀.₅ 推理验证 —— 无需服务器，无需机器人。"""
import logging
import time

import jax
import numpy as np

from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO)

def main():
    # 硬件检查
    devices = jax.devices()
    print(f"JAX 设备: {devices}")
    assert len(devices) > 0 and "cuda" in str(devices[0]).lower(), \
        f"需要 CUDA 设备，当前设备: {devices}"

    # 从本地缓存加载 pi05_droid
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "gs://openpi-assets/checkpoints/pi05_droid"
    print(f"加载配置: pi05_droid (action_dim={config.model.action_dim}, "
          f"action_horizon={config.model.action_horizon})")

    t0 = time.time()
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"模型加载耗时: {time.time() - t0:.1f}s")

    # 首次推理（包含 JIT 编译）
    example = droid_policy.make_droid_example()
    t0 = time.time()
    result = policy.infer(example)
    jit_time = time.time() - t0
    print(f"首次推理（含 JIT 编译）: {jit_time:.1f}s")

    # 第二次推理（JIT 之后，测量真实推理速度）
    example2 = droid_policy.make_droid_example()
    t0 = time.time()
    result = policy.infer(example2)
    infer_time = (time.time() - t0) * 1000
    print(f"第二次推理（JIT 后）: {infer_time:.1f}ms")

    # 验证输出
    expected_shape = (config.model.action_horizon, 8)  # DROID: 15×8
    assert result["actions"].shape == expected_shape, \
        f"预期形状 {expected_shape}，实际得到 {result['actions'].shape}"
    print(f"动作形状: {result['actions'].shape} ✓")
    print(f"动作值范围: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

    print("\n=== 推理验证通过 ===")
    print(f"模型: π₀.₅ (pi05_droid)")
    print(f"推理延迟: {infer_time:.1f}ms")
    print(f"GPU 显存: 通过 nvidia-smi 查看")

if __name__ == "__main__":
    main()
```

### 步骤 1.2 — 终端运行

```bash
source .venv/bin/activate
python scripts/verify_inference.py
```

**预期输出**：
- JAX 报告 2 个 CUDA 设备
- 首次推理: 5-30s（JIT 编译耗时）
- 第二次推理: < 500ms
- 动作形状: `(15, 8)` — 确认正确

### 步骤 1.3 — 检查 GPU 显存

```bash
nvidia-smi
```

确认显存使用远低于 16 GB（pi05_droid 预计使用 6-10 GB）。

---

## 第二阶段：VSCode 单步调试

**目标**：单步进入模型关键函数，理解执行流程。

### 步骤 2.1 — 配置 VSCode launch.json

将 `.vscode/launch.json` 替换为：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "验证推理 (调试)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/verify_inference.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${workspaceFolder}/packages/openpi-client/src"
            },
            "justMyCode": false
        },
        {
            "name": "启动策略服务 (调试)",
            "type": "debugpy",
            "request": "launch",
            "module": "openpi.serving.websocket_policy_server",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${workspaceFolder}/packages/openpi-client/src"
            },
            "justMyCode": false,
            "args": []
        },
        {
            "name": "简易客户端 (调试)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/examples/simple_client/main.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${workspaceFolder}/packages/openpi-client/src"
            },
            "justMyCode": false,
            "args": ["--env", "droid", "--num_steps", "5"]
        }
    ]
}
```

### 步骤 2.2 — 推荐断点位置

按执行顺序排列的推荐断点：

| # | 文件 | 行号/函数 | 观察要点 |
|---|------|----------|---------|
| 1 | `src/openpi/policies/policy.py` | `Policy.infer()` 函数首行 | 入口：接收原始 observation 字典 |
| 2 | `src/openpi/models/pi0.py` | `Pi0.sample_actions()` 函数首行 | 模型输入：`Observation` 结构化对象 |
| 3 | `src/openpi/models/pi0.py` | `Pi0.embed_inputs()` 内部 | 图片 + 文本 → token 嵌入的过程 |
| 4 | `src/openpi/models/pi0.py` | `Pi0.embed_suffix()` 内部 | 动作噪声 + 时间步 → 动作专家 token |
| 5 | `src/openpi/models/gemma.py` | Transformer block 中的注意力计算 | VLM 与动作专家的自注意力交互 |
| 6 | `src/openpi/models/pi0.py` | Flow matching 去噪循环 | 10 步 Euler 积分的每一步 |
| 7 | `src/openpi/policies/droid_policy.py` | `DroidOutputs.__call__()` | 模型原始输出 → DROID 动作格式 |

### 步骤 2.3 — 调试操作指南

1. 在 VSCode 中打开 `scripts/verify_inference.py`
2. 在第二次 `result = policy.infer(example2)` 处设置断点
3. 按 F5 → 选择 "验证推理 (调试)"
4. 等待 JIT 预热完成（首次 infer 调用结束）
5. 断点命中后，**单步进入 (F11)** 第二次 `policy.infer()` 调用
6. 跟踪执行流程：`Policy.infer()` → `Pi0.sample_actions()` → flow matching 循环
7. `sample_actions` 返回后，在 VARIABLES 面板中检查 `result["actions"]`

---

## 第三阶段：Server-Client 架构验证

**目标**：在分离进程中运行服务端和客户端，验证 WebSocket 通信。

### 步骤 3.1 — 启动策略服务器

终端 1：
```bash
source .venv/bin/activate
python scripts/serve_policy.py --env droid --port 8000
```

**等待输出**：`"Creating server (host: ..., ip: ...)"` — 服务器就绪。

### 步骤 3.2 — 运行简易客户端

终端 2：
```bash
source .venv/bin/activate
python examples/simple_client/main.py --env droid --port 8000 --num_steps 10
```

**预期输出**：
```
Server metadata: ...
Running policy: 100% |████████████| 10/10
Timing Statistics
┌─────────────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ Metric           │ Mean  │ Std   │ P25   │ P50   │ P75   │ P90   │ P95   │ P99   │
├─────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ client_infer_ms  │ 12.3  │ 2.1   │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │
│ server_infer_ms  │ 8.5   │ 1.2   │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │
└─────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

### 步骤 3.3 — 验证通信协议

在服务端开启录制模式，检查实际传输的消息：

终端 1：
```bash
source .venv/bin/activate
python scripts/serve_policy.py --env droid --port 8000 --record
```

此时会在 `policy_records/` 目录下生成 `.npy` 文件，记录每次推理的输入/输出数据对。

### 步骤 3.4 — 理解消息格式

WebSocket 协议使用 **msgpack-numpy** 编码：

```python
# 客户端发送的观测数据格式：
{
    "observation/exterior_image_1_left": np.ndarray(224, 224, 3),  # uint8
    "observation/wrist_image_left":      np.ndarray(224, 224, 3),  # uint8
    "observation/joint_position":        np.ndarray(7,),           # float64
    "observation/gripper_position":      np.ndarray(1,),           # float64
    "prompt": "pick up the red cube",                              # str
}

# 服务端返回的动作数据格式：
{
    "actions":         np.ndarray(15, 8),  # float32, 已归一化
    "policy_timing":   {"infer_ms": 8.5},
    "server_timing":   {...},
}
```

Isaac Sim 需要按此格式生产观测数据并消费动作数据。

---

## 第四阶段：Isaac Sim 集成准备

### 步骤 4.1 — 定义 Isaac Sim 观测接口

Isaac Sim 环境需提供符合 `DroidInputs` 格式的观测数据：

```python
observation = {
    "observation/exterior_image_1_left": camera_rgb,      # (224, 224, 3) uint8
    "observation/wrist_image_left":      wrist_camera_rgb, # (224, 224, 3) uint8
    "observation/joint_position":        joint_angles,     # (7,) float
    "observation/gripper_position":      gripper_pos,      # (1,) float
    "prompt": "pick up the red cube",                      # str
}
```

### 步骤 4.2 — 定义动作执行接口

服务端返回 `(15, 8)` 归一化动作数组。Isaac Sim 控制器需：

1. 提取当前时间步动作：`action = actions[step_index]`（或使用 `ActionChunkBroker`）
2. 逆归一化 / 缩放至机器人关节范围
3. 作为关节速度目标值执行：`robot.set_joint_velocity_targets(action[:7])`
4. 执行夹爪指令：`gripper.set_position(1.0 if action[7] > 0.5 else 0.0)`

### 步骤 4.3 — 客户端集成模板

```python
"""Isaac Sim → π₀.₅ 策略服务器集成模板。"""
from openpi_client import websocket_client_policy

# 连接服务器（可位于同一台机器或远程）
policy = websocket_client_policy.WebsocketClientPolicy(
    host="localhost",  # 或 GPU 服务器的 IP 地址
    port=8000,
)

def get_action(obs_dict: dict) -> np.ndarray:
    """从 π₀.₅ 策略服务器获取动作。
    参数:
        obs_dict: DROID 格式的观测字典
    返回:
        action: (8,) numpy 数组 [关节速度(7), 夹爪(1)]
    """
    result = policy.infer(obs_dict)
    # 从预测的动作块中取第一个动作
    return result["actions"][0]  # (8,)
```

---

## 预期时间线

| 阶段 | 步骤 | 预计耗时 |
|------|------|---------|
| 1 | 本地推理冒烟测试 | 5 分钟 |
| 2 | VSCode 调试设置与跟踪 | 15 分钟 |
| 3 | Server-client 验证 | 10 分钟 |
| 4 | Isaac Sim 集成准备 | N/A（后续） |

**总计**：约 30 分钟完成全部验证。

---

## 验证清单

- [ ] `verify_inference.py` 无错误运行，输出 `(15, 8)` 动作
- [ ] JIT 编译后推理延迟 < 500ms
- [ ] 推理期间 GPU 显存占用 < 16 GB
- [ ] VSCode 调试器成功单步进入 `Policy.infer()`
- [ ] VSCode 调试器成功单步进入 `Pi0.sample_actions()`
- [ ] VSCode 调试器 VARIABLES 面板显示动作张量数值
- [ ] 策略服务器无错误启动
- [ ] 简易客户端成功连接并接收有效动作
- [ ] 耗时统计中 server_infer_ms < 100ms
- [ ] `--record` 模式产出 policy_records/ 文件

---

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `src/openpi/models/pi0.py` | π₀/π₀.₅ 核心模型（`Pi0` 类），`pi05` 标志控制版本差异 |
| `src/openpi/models/pi0_config.py` | 模型配置，`Pi0Config` 数据类 |
| `src/openpi/models/model.py` | 基类 `BaseModel`、`Observation`、`Actions` |
| `src/openpi/policies/policy.py` | `Policy` 推理封装（transform + model.infer） |
| `src/openpi/policies/policy_config.py` | `create_trained_policy()` 策略工厂函数 |
| `src/openpi/policies/droid_policy.py` | DROID 数据格式转换（`DroidInputs` / `DroidOutputs`） |
| `src/openpi/serving/websocket_policy_server.py` | WebSocket 策略服务器 |
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | WebSocket 客户端 |
| `scripts/serve_policy.py` | 服务器启动入口脚本 |
| `examples/simple_client/main.py` | 客户端测试脚本（随机观测，无需真实机器人） |
| `examples/inference.ipynb` | 本地推理 Jupyter 示例 |