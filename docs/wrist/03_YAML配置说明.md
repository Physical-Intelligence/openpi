# YAML 配置说明

本文是 UR5 双轴腕实验的中文项目文档，不是 Physical Intelligence 官方 OpenPI 原文。

配置目录：

```text
examples/ur5_twinwrist/config/
```

## 五份配置各管什么

| 文件 | 负责内容 | 常改参数 |
|---|---|---|
| `hardware.yaml` | 设备身份、连接地址和设备协议参数 | UR IP、相机 serial、串口 by-id、主腕映射、夹爪 backend |
| `teleop.yaml` | 人机操作语义 | 轴映射、deadzone、键位 code/action、episode 按键 |
| `poses.yaml` | Home 和零位 | UR task-home、腕 J1/J2 舵机 raw 零位、夹爪 open/closed |
| `safety.yaml` | 所有安全边界 | 速度、工作空间、关节限位、timeout、camera skew |
| `collection.yaml` | 数据合同 | prompt、30/10 Hz、HDF5 路径、9D state/action 字段 |

不要在多个文件重复定义同一个参数。例如 UR IP 只放 `hardware.yaml`，task-home 只放
`poses.yaml`，最大速度只放 `safety.yaml`。唯一的显式例外是 Ctrl 腕速度：
`hardware.wrist.override_max_velocity_deg_s` 和 `teleop.modes.wrist.max_speed_deg_s` 分别面向驱动和
操作员阅读，loader 强制它们相同，避免两个值情况下静默选一个。

## 当前关键参数速查

### 设备和采样

| 参数 | 当前值 | 配置来源 |
|---|---:|---|
| UR5 地址 | `192.168.1.106` | `hardware.yaml` |
| UR 控制率 | 125 Hz | `hardware.yaml` |
| D435 图像 | 640×480 RGB8、30 Hz | `hardware.yaml` |
| HDF5 记录率 | 10 Hz | `collection.yaml` |
| 相机角色 | front / side / top | `hardware.yaml` |
| SpaceMouse stale | 250 ms | `hardware.yaml` |
| SpaceMouse 单次最多消费事件 | 256 | `hardware.yaml` |
| 夹爪命令率 | 30 Hz | `hardware.yaml` |
| 组合键/双击开关 | `false` | `teleop.yaml -> episode.physical_toggle_chord_enabled` |

### 三相机运行参数

| YAML 键 | 当前值 | 含义 |
|---|---:|---|
| `width / height / fps` | `640 / 480 / 30` | 每台 D435 实际传入 provider 的 RGB 规格 |
| `pixel_format` | `rgb8` | HDF5 最终保存 RGB `uint8` |
| `enable_depth` | `false` | 第一版不开深度流 |
| `connect_timeout_s` | 5.0 s | 等待三个采集线程启动 |
| `provider_wait_timeout_s` | 0.25 s | 单次等待 provider 新帧 |
| `stop_timeout_s` | 3.0 s | 关闭时等待采集线程退出 |
| `queue_size` | 8 | 每个角色的最近帧队列长度 |

这些顶层相机参数现在会真正传给 front/side/top 的每个 provider。
`config_loader.py` 同时强制 `collection.capture.image_shape` 与上述高/宽/三通道一致，
并强制 `collection.capture.camera_fps` 与 `hardware.cameras.fps` 一致，避免改了 YAML
却仍用旧图像规格。

### ESP32 主腕→OpenRB 从腕

`hardware.yaml` 已显式列出 `MasterWristConfig` 的全部运行参数，不再依赖看不见的
Python 默认值：

| YAML 键 | 当前值 | 含义 |
|---|---:|---|
| `master_port` | `/dev/serial/by-id/...` | ESP32 主腕持久设备路径 |
| `controller_port` | `/dev/serial/by-id/...` | OpenRB 从腕持久设备路径 |
| `baud` | 115200 | 两条串口当前波特率 |
| `serial_timeout_s` | 0.05 s | ESP32 单次 `readline` 等待 |
| `response_timeout_s` | 2.0 s | STOP/START/SET_PERIOD 完整命令应答等待 |
| `source_max_age_s` | 0.20 s | 等待新 TELE 的最大时间 |
| `stream_period_ms` | 10 ms | ESP32 TELE 目标频率 100 Hz |
| `command_hz` | 50 Hz | 上层主从腕 `step()` 目标调用率 |
| `state_hz` | 10 Hz | 舵机状态/数据记录目标率；由外层 owner 调度 |
| `master_mapping.j1_source/sign` | `enc0 / +1` | Enc0 正方向映射到 J1 raw 增大 |
| `master_mapping.j2_source/sign` | `enc1 / +1` | Enc1 正方向映射到 J2 raw 增大 |
| `j1_raw_per_deg / j2_raw_per_deg` | 10 / 14 | 主腕角位移换算为从腕相对 raw |
| `input_deadband_deg` | 0.5° | 主腕相对角小于此值输出零 |
| `filter.*` | 1.8 / 0.08 / 1.0 | One-Euro min cutoff / beta / derivative cutoff |
| `dynamixel_profile.*` | 50 / 15 raw | OpenRB 舵机 profile velocity / acceleration |
| `target_limiter.deadband_raw` | 2 raw | 小于此变化暂不发新目标 |
| `target_limiter.max_velocity_raw_s` | 900 raw/s | 相对 raw 目标速度上限 |
| `target_limiter.max_accel_raw_s2` | 4500 raw/s² | 相对 raw 目标加速度上限 |
| `target_limiter.max_jerk_raw_s3` | 30000 raw/s³ | 相对 raw 目标 jerk 上限 |
| `override_max_velocity_deg_s` | 45°/s | Ctrl SpaceMouse 覆盖最大速度 |
| `override_lease_s` | 0.10 s | Ctrl 速度命令过期时间 |
| `override_accel_deg_s2` | 300°/s² | Ctrl 覆盖加速斜坡 |
| `override_decel_deg_s2` | 500°/s² | Ctrl 覆盖减速斜坡 |
| `read_timeout_s` | 0.20 s | OpenRB 从腕单次反馈超时 |

`command_hz/state_hz` 不会在 controller 内启动隐藏线程；统一的 RobotHardware owner
负责调度。主腕目标链按 50 Hz 工作，HDF5/完整 observation 按 10 Hz 取从腕
`GET_WRIST_STATE.j1_pos/j2_pos`。串口始终只有 owner worker 能读写。

### Episode 按键配置

`teleop.yaml -> buttons` 的 YAML key（例如 `menu`/`fit`/`one`）是程序查询的
canonical 语义，`code` 是 SpaceMouse 实际上报键码。重绑物理键时只改 `code`；
`label` 和 `action` 都是给人阅读的说明，改它们不会改程序行为。当前
Menu/Fit/Esc/T/1/2/RotationLock 的完整行为见
[遥操作键位与零位](02_遥操作键位与零位.md)。第一版为了保持单键语义清晰，
`physical_toggle_chord_enabled` 必须保持 `false`。

`TeleopCollectionSession` 的开始条件是唯一的：Menu 完成完整 Home/主腕 resume，
并等到首个完整零 `speedL` action receipt 后直接启动 episode。

### UR 和时间门禁

| 参数 | 当前值 | 说明 |
|---|---:|---|
| 遥操作平移速度 | 0.107 m/s | 三维向量范数 |
| 遥操作旋转速度 | 0.54 rad/s | 三维向量范数 |
| worker 线速度硬上限 | 0.25 m/s | 三维向量范数 |
| worker 角速度硬上限 | 0.60 rad/s | 三维向量范数 |
| speedL acceleration | 0.50 | RTDE 参数 |
| command duration | 0.008 s | 对应约 125 Hz |
| stop deceleration | 1.0 | 异常/退出时 RTDE 停止减速参数 |
| 安全预测时域 | 0.25 s | 用当前 speedL/speedJ 预测 TCP/关节越界 |
| J6 jog | 0.20 rad/s | 录制窗口内禁止 |
| Home 比例增益 | 1.50 | 关节误差到 speedJ 的 P 增益 |
| 完整 Home 总超时 | 120 s | `poses.yaml -> ur5.full_home_timeout_s` |
| TCP 最低 Z | 0.08093 m | 现场仍需复核 |
| observation timeout | 200 ms | 推理/执行门禁 |
| policy timeout | 1000 ms | 超时停止/保持 |
| camera timeout | 500 ms | 无合格三帧则失败 |
| camera max skew | 50 ms | 比较 host monotonic 时间 |
| state / command max age | 100 / 250 ms | 过期即失败 |
| UR joint limits | `null` | 必须从 PolyScope 填写 |

### 双轴腕和夹爪

| 参数 | 当前值 |
|---|---:|
| YAML 舵机零位 | J1 `3278`、J2 `2547` raw |
| J1 绝对/相对范围 | `[2781,3568]` / `[-497,+290]` raw |
| J2 绝对/相对范围 | `[1822,3333]` / `[-725,+786]` raw |
| 单步最大腕变化 | 18 raw |
| 腕 shaper | 900 raw/s、4500 raw/s²、30000 raw/s³；当前控制器已执行 |
| 腕反馈 timeout / poll | 0.20 s / 0.05 s |
| 腕 Home timeout | 30 s |
| Home 收敛 | 30 raw，连续 3 次，最多 18 s |
| 夹爪归一化范围 | `[0,1]`，0 开、1 闭 |
| 夹爪最大变化率 | 1.0/s |

这些是当前软件配置，不代表全部已经通过现场标定。腕运行时仍会查询板端 limits，UR joint limits
为 `null` 时必须阻止真实运动。AS5048A 角度只作诊断；训练和推理均使用舵机 raw 减 YAML 零位。

## 配置校验

只解析 YAML，不连接设备：

```bash
uv run examples/ur5_twinwrist/config_loader.py
```

它检查：

- 三个 camera role 是否恰好为 front/side/top。
- 三个 serial 是否唯一。
- 手腕和夹爪是否使用 `/dev/serial/by-id/`。
- 主腕串口 timeout、10 ms TELE 周期和 50/10 Hz 是否全部显式定义。
- 映射是否严格为 Enc0→J1(+)、Enc1→J2(+)，raw_per_deg/deadband 是否有限且合法。
- YAML `servo_zero_raw` 是否为两个合法 raw，是否落在 J1/J2 软限位内。
- Ctrl cap X/Y 是否恰好各映射到 J1/J2，符号是否为 ±1。
- One-Euro、Dynamixel profile 和 velocity/acceleration/jerk 参数是否合法。
- TELE 帧率、command/state 频率是否前后一致。
- `hardware.yaml` 与 `teleop.yaml` 的 Ctrl 腕最大速度是否完全相同。
- 夹爪 backend 是否有对应 adapter。
- 轴映射是否覆盖所有输出轴。
- button code 是否重复。
- UR Home 是否 6 维、腕 Home 是否 2 维。
- state/action 是否都是 9 维。
- record_hz 是否不高于 camera_fps。
- 顶层相机分辨率/fps 是否与 collection 的 `image_shape/camera_fps` 一致。
- `teleop.yaml` 的平移/旋转速度是否与 `safety.yaml` 的遥操上限一致。
- SpaceMouse 事件数、相机线程 timeout/队列和完整 Home timeout 是否是有限合法值。

真实运动前的严格检查：

```bash
uv run examples/ur5_twinwrist/config_loader.py --require-real-ready
```

目前这个命令会故意失败，因为 `safety.yaml` 中的 UR5 `joint_min_rad/joint_max_rad` 还是 `null`。
应从当前 PolyScope 安全设置准确抄入，而不是填写宽泛的 `[-2π,2π]`。

## 配置 hash

loader 会把五份 YAML 规范化并计算一个 SHA-256。原始 HDF5、训练快照和推理启动记录都应保存
这个 hash。4090 与 H100 必须同时核对：

```text
Git commit
uv.lock SHA-256
五份 YAML config SHA-256
action space 定义
```

任意一项不同，就不应把两台机器称为同一个实验版本。

## 修改建议

1. 每次只改一个参数类别。
2. 修改前后运行 config loader 和对应 Fake 测试。
3. 修改真实速度、限位、Home 后，先 shadow/read-only，再低速小步验收。
4. 设备换 USB 口时通常不需要改 by-id；换了设备本体才更新 serial。
5. 正式采集前把配置和代码一起 commit，不要用 `+dirty` 状态采正式数据。
