# OpenPI SO-101 红蓝方块联合训练设计

## 目标

使用一个 OpenPI π0 LoRA 模型联合训练以下两个本地 LeRobot v2.0 数据集：

- `YukiiLiu/so101_red_cube_box_formal_clean_v20_100src`
- `YukiiLiu/so101_blue_cube_box_formal_clean_v20_100src`

训练启动成功的判据是：容器能同时加载两个数据集，完成合并归一化统计，JAX 在 RTX 5080 上进入训练循环，日志产生有限的 loss/gradient 指标，并按配置写出检查点。

## 已确认约束

- 使用一个模型，而不是分别训练两个模型。
- 保留两个原始数据集，不生成或改写第三个合并数据集。
- 使用两个数据集各自的任务文本，使模型区分 red cube 与 blue cube。
- SO-101 的 6 维绝对动作中，前 5 个关节转换为相对当前状态的 delta action；第 6 维夹爪保持绝对值。
- 每 2,500 步保存一次检查点。
- RTX 5080 只有 16 GB 显存，先使用 LoRA、batch size 1、关闭 W&B。
- Docker 容器挂载 `F:\lerobot_data` 到 `/lerobot_data`，并设置 `HF_LEROBOT_HOME=/lerobot_data`。

## 方案选择

采用 LeRobot 已存在的 `MultiLeRobotDataset`，在 OpenPI 数据加载层增加多 repo 支持。该方案直接串联两个原始数据集，不复制视频、不重编号 episode，也不修改训练核心循环。

未采用的方案：

- 物理合并数据集：需要重写 episode、task、index、parquet 和视频路径，容易损坏元数据。
- 两个 DataLoader 交替采样：侵入训练循环，增加 checkpoint/resume 一致性风险。

## 架构

### 数据配置

为 `DataConfig`/`DataConfigFactory` 增加可选的多 repo 描述。单 repo 的现有行为保持不变；提供两个 repo 时，数据加载器创建 `MultiLeRobotDataset`。

联合训练配置命名为 `pi0_so101_red_blue_lora`，主要参数：

- π0 LoRA：`gemma_2b_lora` + `gemma_300m_lora`
- batch size：1
- train steps：20,000
- save interval：2,500
- W&B：关闭
- base checkpoint：`gs://openpi-assets/checkpoints/pi0_base/params`
- checkpoint 目录：宿主机持久化挂载

### SO-101 数据变换

新增 SO-101 专用数据配置，执行以下映射：

- `observation.images.front` → 主视角 `cam_high`
- `observation.images.wrist` →腕部视角 `cam_left_wrist`
- `observation.state` → `state`
- `action` → `actions`
- LeRobot 样本自带的 `task` 字符串 → `prompt`

复用 ALOHA 图像与 padding 管线，但关闭 ALOHA 专用关节方向和夹爪单位适配。数据经过模型前，对动作掩码 `[True, True, True, True, True, False]` 应用 `DeltaActions`；推理输出使用相同掩码的 `AbsoluteActions` 恢复绝对目标。

### 多数据集任务文本

`MultiLeRobotDataset` 的两个子数据集都使用 `task_index=0`，因此不能共享单一的 task-index 映射。联合加载时直接读取每个子数据集样本已经附带的 `task` 字符串，并将其转换为 `prompt`，避免红蓝任务文本混淆。

### 归一化统计

`compute_norm_stats.py` 使用同一多数据集配置遍历联合数据。生成的统计写入联合配置专属 assets 目录，并在正式训练时加载。统计计算和训练必须使用完全相同的 repack、SO-101 输入变换和 delta 掩码。

## 数据流

1. 宿主机两个数据集通过 Docker volume 映射到 `/lerobot_data/YukiiLiu/...`。
2. `MultiLeRobotDataset` 按帧串联红色与蓝色数据集。
3. 每个样本保留其原始 `task` 字符串。
4. SO-101 repack 将双相机、状态、动作和 prompt 转成 OpenPI 输入格式。
5. 前 5 维绝对关节动作转换为 delta，第 6 维夹爪保持绝对。
6. 联合统计归一化后，模型 tokenization/padding 生成训练 batch。
7. JAX 在 RTX 5080 上执行 LoRA 更新并按 2,500 步保存检查点。

## 错误处理与停止条件

- 任一数据集目录或 metadata 缺失：在统计或训练前立即失败并指出 repo。
- 两个数据集特征、fps 或动作维度不兼容：多数据集构造阶段立即失败。
- prompt 丢失或任务文本混淆：数据变换测试失败，不进入训练。
- 归一化统计包含非有限值或极小尺度：停止训练并报告异常维度。
- JAX OOM：保持数据与训练设计不变，先记录峰值显存和失败位置，再评估减少图像/序列负载或改用更低显存实现；不静默改变任务范围。
- 正式训练只有在首批 loss、grad norm 均为有限值后才视为启动成功。

## 测试与验证

实现采用测试驱动：

1. 先添加失败测试，证明当前加载器不支持两个 repo。
2. 添加多 repo 最小实现，使联合数据长度等于两个数据集帧数之和。
3. 添加失败测试，证明不能用共享 `task_index=0` 生成正确红蓝 prompt。
4. 添加基于样本 `task` 字符串的 prompt 变换并验证红、蓝文本各自正确。
5. 验证 SO-101 映射产生主相机、腕部相机、6 维 state/action。
6. 验证 delta 掩码只改变前 5 维，第 6 维夹爪保持不变，并可被输出变换恢复。
7. 运行现有相关单元测试，确保单 repo 行为无回归。
8. 在容器内执行真实数据 smoke test、合并统计计算和至少一个训练 step。

## 非目标

- 不修改两个原始数据集。
- 不同时训练第二个独立模型。
- 不改变 OpenPI 模型架构或训练核心算法。
- 不启用 W&B 网络上传。
- 不在本任务中实现 SO-101 机器人在线推理或部署。
