# OpenPI 训练与推理入门

本文是 UR5 双轴腕实验的中文项目文档，用本项目术语解释官方 OpenPI 流程，
不是 Physical Intelligence 官方 OpenPI 原文。具体 CLI 参数仍以当前仓库脚本 `--help` 为准。

## 一条数据如何进入模型

```text
真机原始 HDF5
→ validate_dataset.py
→ convert_to_lerobot.py
→ 锁定版 LeRobot 本地数据集
→ compute_norm_stats.py
→ JAX π0.5 训练
→ Orbax checkpoint
→ serve_policy.py
→ localhost openpi-client
→ safety_filter
→ shadow 或真实低速执行
```

## LeRobot 是什么

LeRobot 在本项目中主要负责训练数据的标准目录、episode 索引、图像和字段读取。真机控制循环不直接
写 LeRobot，而是先写简单 HDF5，原因是视频编码或数据集 consolidate 不应阻塞机器人控制线程。

转换后的字段固定为：

```text
observation.state
action
observation.images.front
observation.images.side
observation.images.top
task
```

本项目只使用 `uv.lock` 锁定的 LeRobot commit，不参考最新版 API 猜写法。

## norm stats 是什么

state/action 每一维的数值范围不同。norm stats 是从当前训练数据重新计算的归一化统计，让模型看到
尺度合理的输入和目标。更换数据集、action 语义或相机映射后，旧 norm stats 不能直接沿用。

官方说明在 `docs/norm_stats.md`；本项目执行顺序记录在 `docs/wrist/PIPELINE_RUNBOOK.md`。

## `LEROBOT_DATA_ROOT` 是什么

`train_h100.sh` 不会猜测 LeRobot 数据在哪里，启动前必须显式设置：

```bash
export LEROBOT_DATA_ROOT=/远程绝对路径/data/lerobot
export DATASET_REPO_ID=local/ur5_twinwrist_task2
```

脚本实际检查的数据目录是：

```text
$LEROBOT_DATA_ROOT/$DATASET_REPO_ID
```

因此 `LEROBOT_DATA_ROOT` 应指向包含 `local/` 的根目录，而不是直接指向
`.../local/ur5_twinwrist_task2`。脚本会再把它设置为 `HF_LEROBOT_HOME`，然后调用官方
`scripts/compute_norm_stats.py` 和 `scripts/train.py`。

## 为什么模型 action_dim 是 32，而机器人只有 9 维

π0.5 配置使用：

```text
action_dim = 32
action_horizon = 10
```

真实 state/action 只有前 9 维。OpenPI 的 `PadStatesAndActions` 补零到 32 维，模型输出后
`ur5_twinwrist_policy.py` 只取前 9 维。模型核心没有被修改。

## policy server 是什么

目标架构中，4090 上的 policy server 负责加载 JAX checkpoint 和执行模型。机器人客户端通过
localhost 发送三张图像、9 维状态和 prompt，server 返回一个 action chunk。

第一版只执行 chunk 第一步，并默认 shadow：打印动作但不发硬件。只有经过 timeout、stale、NaN/Inf、
关节/工作空间/腕/夹爪限位等安全检查后，显式 `--enable-motion` 才允许发送。

当前 `run_infer_4090.sh` 会启动 localhost policy server、等待 `/healthz`、再运行
`robot_runtime.py`。客户端发送三路图像、9 维 state 和 prompt，严格要求返回 `(H,9)`；腕第 6/7
维保持 YAML 零位相对 raw。默认 `FAKE=1` 且 shadow，`FAKE=0` 才打开本地真机只读组合。
policy timeout、NaN/Inf、零位不一致或状态过期都会停止；真 checkpoint 和真机 shadow 仍待现场验收。

官方远程推理协议在 `docs/remote_inference.md`。本实验虽然使用相同 client/server 协议，但第一版
server 和机器人客户端都在同一台 4090，host 固定为 localhost，不依赖公网。

## JAX 与 PyTorch

本项目训练和推理明确使用 `backend=jax`，checkpoint 是 Orbax/JAX 格式。不进行 JAX 与 PyTorch
权重转换，也不修改官方 π0/π0.5 模型核心。
