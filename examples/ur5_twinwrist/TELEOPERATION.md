# TASK2 遥操作与原始数采手册

本文只覆盖当前最小闭环的前半段：SpaceMouse → UR5 + 外置两轴腕 + 夹爪 → 三台
RealSense → 10 Hz 原始 HDF5。训练和推理统一为 `backend=jax`，但本阶段不运行真机推理。

## 当前结论

- 普通遥操作最终是 UR RTDE `speedL`，所以训练 action 是
  `[TCP twist 6, wrist absolute target 2, gripper absolute target 1]`，不是原始 SpaceMouse 值。
- 相机采集仍为 30 Hz；新 HDF5 sink 用单调主机时钟降采样到默认 10 Hz。
- 新 adapter 已加入：夹爪唯一串口 worker、HOME 帽输入清零、控制周期完成后发布 command receipt、Fit 前封存、
  `speedJ`/stale/skew/duplicate/NaN/Inf/floor-clamp 歧义拒收、异常 emergency-save 拒收。
- 本轮只读 preflight 只发现配置的 3 台相机中的 1 台，三个串口 by-id 均不在；因此
  `--enable-motion` 当前会在打开硬件前被拒绝。本轮没有发送任何真机运动命令。
- legacy 工作树有 118 条状态变化，关键连续数采配置中也有未提交文件。每次采集必须保存 preflight
  JSON 和 HDF5 内的 config hash；在锁定这些配置前，数据不能称为完全可复现。

## 唯一推荐入口

```bash
cd /home/user/haitao_files/robowrist/pi0.5/openpi

# 1. 只读：配置、git、串口节点、相机 serial、spacenavd、进程占用
uv run examples/ur5_twinwrist/teleop_preflight.py \
  > data/teleop_preflight_$(date +%Y%m%dT%H%M%S).json

# 2. 只生成计划：不打开硬件、不运动
uv run examples/ur5_twinwrist/teleop_collect.py \
  --episodes 1 --record-hz 10 --no-dashboard

# 3. 真机模板：preflight 全绿且现场安全检查完成后才能通过
uv run examples/ur5_twinwrist/teleop_collect.py \
  --continuous \
  --record-hz 10 \
  --gripper-backend hiwonder \
  --output data/raw/task2_$(date +%Y%m%dT%H%M%S) \
  --enable-motion \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

不要使用 legacy 的 `sc9/sc9x/scwx`：它们仍传旧 `task2.yaml`，而当前 strategy 要求
`task2_9dof_continuous`。用户级 `slai-wrist-collection.service` 的工作目录是
`/home/user/shiyi/Sensecore_H100`，也不是本次审计的 legacy 根目录；运行新入口前必须保持该服务 inactive。

## 代码结构

```text
openpi/examples/ur5_twinwrist/teleop_collect.py
  ├─ 默认 dry-run；--enable-motion + 精确确认字符串才向下传 --execute-real
  ├─ teleop_preflight.py：真实运动前硬门禁
  └─ legacy slai_mi.apps.collect_real
      └─ RealCollectionWorkflow：Menu/Fit/Esc、HOME、录制线程生命周期
          └─ slai_mi.site_adapter.make_collection
              ├─ ControlledUR5SpaceMouse：125 Hz 联合控制线程
              │   ├─ SpaceMouseProcess：spnav worker 与 stale 清零
              │   ├─ UR5OnlySession → UR5DriverProcess → worker.py → speedL/speedJ
              │   ├─ WristMasterSlaveController：OpenRB 动态 HOME、FE/RU 闭环
              │   └─ legacy gripper driver：Hiwonder 或 Feetech
              ├─ RealSenseCapture：每台相机独立线程、按 serial 绑定
              ├─ FrameSynchronizer：从队列中选择满足 skew 的最新三帧组合
              ├─ StationSynchronizer：组装相机、UR、腕、夹爪和命令样本
              └─ EpisodeRecorder：生成 9D state/action 与 telemetry
                  └─ legacy_collection_adapter.py
                      ├─ HomingSafeSpaceMouse：T/协调 HOME 时帽输入强制为 0
                      ├─ AtomicControlledSpaceMouse：发布已完成的 125 Hz 命令周期 receipt
                      ├─ SingleOwnerGripper：唯一线程拥有串口，其他线程只读缓存
                      └─ RawHDF5Dataset：30 Hz 输入门禁并降采样至 10 Hz
                          └─ hdf5_writer_process.py（官方 OpenPI .venv）
                              ├─ episode_XXXX.tmp.hdf5
                              ├─ episode_XXXX.hdf5
                              └─ rejected/episode_XXXX.hdf5
```

legacy 硬件 Python 环境没有 `h5py`；HDF5 writer 使用官方 OpenPI `.venv` 独立进程。两边不复制
虚拟环境、不混装依赖，也没有复制 legacy OpenPI 核心。

## SpaceMouse 轴映射

spnav 原始顺序为 `[x, y, z, rx, ry, rz]`。代码先除以 500、clip 到 `[-1,1]`、应用
0.12 deadzone，再映射为：

```text
output = [-raw_z, raw_x, raw_y, -raw_rz, raw_rx, raw_ry]
```

| 操作 | UR/腕行为 | 当前参数 |
|---|---|---|
| 帽平移 | 只用 output XYZ；UR base-frame `speedL` 平移 | 最大向量范数 0.107 m/s |
| Shift + 帽 | 只用 output Rx/Ry/Rz；平移清零 | 最大向量范数 0.54 rad/s |
| Ctrl + 帽 X/Y | X → FE（符号 -1），Y → RU（符号 +1）；UR twist 清零 | 二次 deadzone 0.18，最大 45 deg/s |
| 1 / 2 | UR5 自带第 6 关节负/正 jog；帽必须居中 | -/+0.20 rad/s |
| 3 按住 | 夹爪连续闭合 | +1.0 normalized/s，命令最多 30 Hz |
| 4 按住 | 夹爪连续打开 | -1.0 normalized/s，命令最多 30 Hz |

Shift 是“旋转模式”，不是 boost。YAML 中的 precision/boost 数值当前不可达，因为
`select_speed_limits()` 固定返回 TRAINING。物理帽向前/后对应 raw 正负没有可信文档；接机器人前先做纯输入 echo：

```bash
cd /home/user/shiyi/slai-manipulation
.venv-lerobot-v3/bin/python -m slai_mi.apps.spacemouse_echo --duration-s 30
```

这个命令只打开 SpaceMouse，不打开 UR、腕、夹爪或相机。

## 全部键位

| 物理键 | bnum | 当前 9DoF 行为 |
|---|---:|---|
| Menu | 0 | idle 时先协调 HOME，完成后 armed；外腕 resume |
| Fit | 1 | 结束任务；新 sink 在 Fit 上升沿先封存，随后 legacy 在 episode 外 HOME |
| T / Top | 2 | UR 返回 task-home，外腕 park；不是保存键 |
| R / Rear | 4 | 外腕 soft-home；按住期间 UR 帽命令为 0 |
| F / Front | 5 | 当前 9DoF 无动作 |
| Roll CW | 8 | 当前 9DoF 无动作 |
| 1 | 12 | UR 第 6 关节 -0.20 rad/s jog |
| 2 | 13 | UR 第 6 关节 +0.20 rad/s jog |
| 3 | 14 | 夹爪闭合 |
| 4 | 15 | 夹爪打开 |
| Esc | 22 | 丢弃当前 episode 到 `rejected/`，然后 HOME |
| Alt | 23 | 未绑定 |
| Shift | 24 | 帽切到旋转模式 |
| Ctrl | 25 | 帽切到外腕 FE/RU jog，UR 清零 |
| Rotation Lock | 26 | finalize；当前未提交段丢弃，已完成 episode 保留 |

注意两个陷阱：

1. `Button.HOME = Button.FOUR` 是遗留别名，物理 4 实际是“夹爪打开”；真正 task-home 是 T。
2. `configs/controls/spacemouse_standard.yaml` 的 `bindings:` 只是人类可读说明，运行时代码没有读取它；
   改 YAML 不会重绑定按键。

Ctrl 松开后外腕保持最后目标，不会自动恢复 ESP32 主腕跟随。要恢复跟随，先用 T/Fit/R 进入 park/home，
再按 Menu resume。

## Episode 生命周期（新 HDF5 语义）

1. 进程启动先协调 HOME。
2. Menu 再次 HOME，完成后进入 armed。
3. 只有安全过滤后的 `twist.z < -1e-4 m/s` 才创建 recorder；首个通过门禁的帧创建
   `episode_XXXX.tmp.hdf5`。
4. Fit 上升沿立即封存任务段，不写 Fit 帧，也不写后续 reset/HOME 帧。
5. legacy 继续执行协调 HOME；全部设备到位后才调用正常 save，writer flush/fsync 后原子 rename。
6. Esc、Rotation Lock 未提交段、Ctrl-C、相机/串口/状态/同步异常全部进入 `rejected/`；
   legacy 的 `emergency_save_active_episode()` 被显式识别，永远不能成为 success。
7. Menu+Fit 同时按下并完全释放，两次、且在 5 秒内，是物理 start/exit toggle；旧 README 写三次是错的。

录制窗口内按 1/2 或 T 会产生 `speedJ`；新 sink 发现非零 `telemetry.ur5_target_qd` 后立即拒收整段。
不要把关节 jog 混入一个任务示范。

## 零位、Home 与 Park

### UR5

项目没有定义 UR5“机械零位”。当前软件 task-home 是以下关节目标：

```text
rad: [ 3.2977943420, -1.5669592063, -0.9337509314,
      -2.2119277159,  1.5755716562, -1.0345171134 ]
deg: [188.950, -89.780, -53.500, -126.734, 90.274, -59.273]
```

来源是 `configs/poses/tasks/task2_continuous_start.yaml`。它只能称为 task-home；历史
`task2_current_zero.yaml` 明确说是只读姿态快照，不是验证过的机械零。HOME 使用 `speedJ`，最大
0.50 rad/s，最大关节误差不超过 0.010 rad 并稳定 0.30 s 才完成。普通遥操作使用 `speedL`。

可选 `--home-preset last-point` 的 UR 目标为：

```text
[3.2988367081, -1.3855517546, -1.6734727065,
 -1.6501811186, 1.5775959492, -1.0311530272] rad
```

首轮 TASK2 不建议使用该可选值。

### 外置两轴腕

需要区分三种零：

1. OpenRB 电机 raw Home：固件候选 J1=3210、J2=2565。
2. 从腕输出零：每次 `HOME_ALL` 后读 Enc0/Enc1，并把当次值安装成 FE/RU `[0,0]`。
3. ESP32 主腕零：每次 resume 时把主腕当下姿态当相对零。

完整流程是 `HOME_ALL → 读取 Enc0/Enc1 → SET_OUTPUT_ZERO_ABS_CDEG → FE/RU target=0/0`。
最近诊断 baseline 为 Enc0=66.77°、Enc1=163.76°，会在下一次 Home 被覆盖，不是永久校准。

当前 vendor firmware 源码候选范围为 FE `[-54,+63]°`、RU `[-26,+42]°`。真正运行时从板端
`GET_OUTPUT_CL` 读取；未查询当前板端前不能把候选值当现场验收值。固件和 active YAML 都是
`Enc0 → RU, Enc1 → FE`；legacy `apps/task2_state.py` 的显示标签相反，不能据此判断物理轴。
9D 主数据走 controller 的 `state.fe/state.ru`，顺序仍是 `[FE,RU]`。

### 夹爪

夹爪没有机械 homing；任务 Home 发送绝对位置 0。训练约定固定为 `0=open, 1=closed`。

| backend | port 来源 | ID/baud | open/closed raw | 其他关键参数 |
|---|---|---|---|---|
| Hiwonder（当前） | `/dev/serial/by-id/...` | 1 / 115200 | 630 / 860 | feedback bias 19；timeout 0.3 s；max 60°C |
| Feetech | `/dev/serial/by-id/...` | 13 / 1,000,000 | 100 / 3995 | speed 250；accel 8；timeout 0.08 s |

`gripper.open()` 是打开串口，不是张开；`gripper.close()` 是关闭串口，不是闭合。真实位置命令是
`command_position(0..1)`。新 `SingleOwnerGripper` 让所有 open/read/write/close 都在同一 worker
线程执行，控制、HOME、Recorder 只拿缓存；任何 SerialException/timeout 会永久 fail-closed。
默认 backend 来自 legacy `configs/hardware.yaml`；新入口可用 `--gripper-backend hiwonder|feetech`
做进程内覆盖，不会改写 legacy YAML。

## 关键参数与唯一控制源

| 当前参数 | 数值 | 控制源 |
|---|---:|---|
| UR 低层控制率 | 125 Hz | `configs/controls/spacemouse_standard.yaml` |
| 相机采集 | 30 Hz, 640×480 RGB | continuous input schema |
| HDF5 记录 | 10 Hz 默认 | `teleop_collect.py --record-hz` |
| UR 平移/旋转 | 0.107 m/s / 0.54 rad/s | SpaceMouse control profile |
| UR worker 硬上限 | 0.25 m/s / 0.60 rad/s | `configs/hardware.yaml` |
| UR TCP floor | Z=0.08093 m | `configs/hardware.yaml`；现场需复核 |
| UR relative envelope | 0 mm / 0°，即关闭 | SpaceMouse control profile |
| SpaceMouse deadzone | 0.12 | SpaceMouse control profile |
| 外腕 master / command / feedback | 100 / 50 / 20 Hz | `runtime/wrist_output_v2.yaml` |
| 外腕 target v/a/jerk | 60 deg/s / 800 deg/s² / 10000 deg/s³ | wrist runtime config |
| camera/state/command max age | 100 / 100 / 250 ms | continuous input schema |
| camera max skew | 100 ms（当前 legacy 值） | continuous input schema；建议现场收紧后重测 |

关键文件：

| 用途 | legacy 文件 |
|---|---|
| UR IP、硬上限、相机 serial、腕/夹爪 by-id | `configs/hardware.yaml` |
| 当前任务/prompt/引用 | `configs/tasks/task2_continuous.yaml` |
| UR/腕/夹爪 task-home | `configs/poses/tasks/task2_continuous_start.yaml` |
| SpaceMouse 频率、速度、deadzone | `configs/controls/spacemouse_standard.yaml` |
| 9D strategy 与所需设备 | `configs/strategies/ur5e_wrist_gripper_9dof_collection.yaml` |
| state/action/相机/sync 阈值 | `configs/input_schemas/ur5e_wrist_gripper_9dof_continuous.yaml` |
| 外腕滤波、控制器、rate、settling | `runtime/wrist_output_v2.yaml` |
| 联合控制实现 | `src/slai_mi/site_adapter.py` |
| UR 进程代理与最终 worker guard | `src/slai_mi/devices/ur5/process.py`, `worker.py` |
| 外腕实现 | `src/slai_mi/devices/wrist_sensor/teleop.py` |
| 两种夹爪驱动 | `src/slai_mi/devices/gripper/` |
| 三相机配对 | `src/slai_mi/devices/cameras/realsense_capture.py` |
| Episode 状态机 | `src/slai_mi/runtime/real_workflows.py` |

独立工具：

| 工具 | 是否运动 | 说明 |
|---|---|---|
| 新 `teleop_preflight.py` | 否 | 配置、设备存在性、进程占用和 hash |
| legacy `spacemouse_echo` | 否 | 只看轴/按钮 |
| legacy `task2_state.py` | 否 | 只读 UR/腕/夹爪；腕 Enc0/Enc1 标签有已知错误 |
| legacy `task2_home.py` | 是 | 只有 `--execute-real` + confirm 才执行联合 Home |
| legacy `park_wrist.py` | 是 | 只有显式真机门禁才执行 `HOME_ALL → HOLD_ALL` |
| legacy `gripper.py status` | 否 | 打开并读取串口；服务 active 时禁止并发使用 |
| 新 `gripper_stress_test.py` | 默认否 | 必须显式 `--enable-motion` 才开合 |

## 原始 HDF5

成功文件至少包含：

```text
/observations/qpos             float32 [N,9]
/action                        float32 [N,9]
/observations/images/front     uint8 [N,H,W,3]
/observations/images/side      uint8 [N,H,W,3]
/observations/images/top       uint8 [N,H,W,3]
/timestamps/control            int64 [N]  monotonic host ns
/timestamps/{front,side,top}   int64 [N]  raw RealSense device timestamps
/timestamps_host/{...}         int64 [N]  aligned host monotonic timestamps
```

跨相机 skew 必须用 `timestamps_host`，不能直接比较三台设备各自的 raw clock。属性包含 task、fps、
success、OpenPI/legacy git identity、相机 serial/role、夹爪 backend/state source、action space 和 config hash。
同一输出目录由非阻塞文件锁保证只能有一个 writer；进程硬中断遗留的 `*.tmp.hdf5` 会在下次启动时
保留并移到 `rejected/`，不会覆盖或静默删除。未压缩 640×480×3×3、10 Hz RGB 约为 27.6 MB/s
（约 1.66 GB/min），批量采集前必须确认磁盘余量。

```bash
# 成功 episode 验证
uv run examples/ur5_twinwrist/validate_dataset.py data/raw/TASK_DIR \
  --max-camera-skew-ms 50

# 单条结构/属性
uv run examples/ur5_twinwrist/inspect_episode.py \
  data/raw/TASK_DIR/episode_0000.hdf5
```

validator 默认还检查 9D shape、float32/uint8、长度、空/黑帧、NaN/Inf、单调时间戳、重复帧、
state/action 范围及 TCP 速度范数。腕/UR 的现场精确边界可用 `--state-min/--state-max` 和
`--action-min/--action-max` 传入 9 个值覆盖保守默认值。

## 第一条真机 episode 的现场顺序

1. 清空工作区，确认 UR 急停、示教器 safety、TCP/tool、payload、工作空间和低速模式。
2. 接齐三台相机与三个串口设备；运行 preflight，必须 `ready=true`。
3. 保存 preflight JSON；确认相机 serial 与 front/side/top 物理安装对应。
4. 停止任何旧 collector/service；运行 SpaceMouse echo，逐轴和全部键位核对。
5. 用只读状态工具核对 UR task-home 误差、板端腕 bounds、夹爪状态；不要把 task-home 叫机械零。
6. 启动新采集入口。启动 HOME 期间松开帽并保持人员可随时急停。
7. Menu → 等 HOME → 首次小幅向下动作开始录制 → 完成任务 → Fit。
8. 等 HOME 完成和原子 rename；任何 error 都只看 `rejected/`，不能手工改 success。
9. 立即运行 validator 和 inspect；首条数据人工播放三路图像、检查 9D state/action 曲线后再批量采集。

## 仍需真实硬件确认

- UR PolyScope 实际 joint limits、安全平面、TCP/tool、payload；仓库没有锁定六关节软限位数值。
- 三个 camera role 对应真实前/侧/顶位置；当前 alias 是 front=primary、side=secondary、top=wrist。
- OpenRB 板端 firmware 与 `GET_LIMITS/GET_OUTPUT_CL` 返回值，以及 FE/RU 真实正方向。
- task-home 是否由现场操作者确认，动态腕 Home 重复性，夹爪开闭端点和 Hiwonder USB 稳定性。
- 当前 legacy 关键配置先形成可追溯 commit；否则同一 HDF5 的 `legacy_git_commit` 会带 `+dirty`。
