# Pipeline 审计记录

本文是 UR5 双轴腕实验的中文项目文档，不是 Physical Intelligence 官方 OpenPI 原文。

审计日期：2026-08-31。训练和推理后端明确为 `backend=jax`。

本文记录已经通过代码或软件测试验证的事实，并把“已实现”“仅有历史证据”“仍需真机确认”分开。

## OpenPI 基线

- 官方递归 clone：`/home/user/haitao_files/robowrist/pi0.5/openpi`。
- 初始官方 commit：`215abfb217dbac7d5f1273282331b9b1866c0479`。
- 自定义分支：`twinpath-pi05`。
- 第一阶段自定义提交：`0385f7870c3737a50c8985bcd2f9edd42d139989`。
- 官方子模块：Aloha `d1dc83a`，LIBERO `f78abd6`。
- 初始官方 `uv.lock` SHA-256：
  `793488b5a55bb87200db90a61fd0af51922b686d94e1da4f4c587ab119b37d74`。
- 锁定 LeRobot：Hugging Face Git commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。
- 主机默认 Python 曾为 3.13.13；项目 `.venv` 使用 CPython 3.11.15，uv 为 0.12.3。
- 审计 4090 为 RTX 4090、驱动 580.95.05；本轮没有检查远端 H100。
- JAX 0.5.3、NumPy 1.26.4、Torch 2.7.1、Transformers 4.53.2，均未升级。

为把真机模块放进同一项目环境，当前开发工作树增加了版本固定的 hardware dependency group：
`pyserial==3.5`、`pyrealsense2==2.56.5.9235`、`ur-rtde==1.6.3`。这会形成新的项目
`pyproject.toml`/`uv.lock` 身份；提交后 4090 与 H100 必须使用完全相同文件，H100 只是不安装
hardware group，不能继续拿上面的“初始官方 lock hash”当当前项目 hash。

## 用户主动精简的官方目录

用户确认 Droid、Aloha 和 Aloha-sim 相关删除是主动精简，不再视为意外数据丢失。
`src/openpi/policies/aloha_policy.py` 删除后，`training/config.py` 已加最小可选 import 守卫：
只允许缺失这一个被主动精简的模块，其他 `ModuleNotFoundError` 仍原样抛出。
因此 UR5 自定义 config 和官方 CLI 可正常 import；若误选 Aloha data config，则会给出
“本精简工作树已省略 Aloha policy”的明确错误，不会静默使用空实现。

审计和实现过程中没有 reset、checkout、恢复或删除用户这些修改。

## “旧参考工程”的身份

旧参考工程是：

```text
/home/user/shiyi/slai-manipulation
```

它不是 π0.5 的术语，也不是必须永久依赖的库，只是已经实现过真机功能的历史项目。审计时 HEAD 为
`3f94334a29b4f94270f284c29dba01713dcc63a5`，同时存在大量用户修改。整个提取过程保持该目录只读，
没有复制其虚拟环境、OpenPI 核心或训练代码。

当前项目的默认 dry-run、Fake、本地硬件组合和数据链都不导入 `slai_mi`，也不需要
旧虚拟环境。`legacy_adapter.py`/`legacy_collection_adapter.py` 是隔离的历史迁移文件，
不在新 `teleop_collect.py` 默认运行链上。

历史实现的主要调用图：

```text
collect_real / real_workflows
  → site_adapter.StationSession + safety supervisor
    → SpaceMouse client/device/mapping
    → UR5 worker/runtime → RTDE speedL
      └─ Home 和特殊 J6 jog 使用 speedJ
    → ESP32 主腕 → resume 时建立输入相对零 → Enc0→J1(+)、Enc1→J2(+)
      → One-Euro + velocity/acceleration/jerk shaper
      → OpenRB 从腕 → YAML servo zero + J1/J2 相对 raw 目标
      └─ Ctrl SpaceMouse 可覆盖主腕；旧历史实现松开后保持最后目标
    → Hiwonder 或 Feetech 串口夹爪
  → RealSenseCapture → 每 serial 独立线程 → FrameSynchronizer
  → StationSynchronizer → Recorder / LeRobot writer
```

历史工程中可作为行为证据的模块：

- UR：`src/slai_mi/devices/ur5/` 和 `site_adapter.py` 的 RTDE 调用。
- 双轴腕：`src/slai_mi/devices/wrist_sensor/`。
- 夹爪：`src/slai_mi/devices/gripper/hiwonder.py`、`feetech_sts3215.py`。
- SpaceMouse：`src/slai_mi/devices/spacemouse/`。
- 三相机：`src/slai_mi/devices/cameras/realsense_capture.py`。
- 安全：速度、workspace、heartbeat、stale 和 flight recorder。

## 已提取到当前项目的本地模块

```text
examples/ur5_twinwrist/controller/ur5.py
examples/ur5_twinwrist/controller/spacemouse.py
examples/ur5_twinwrist/controller/wrist.py
examples/ur5_twinwrist/controller/gripper.py
examples/ur5_twinwrist/cameras/
examples/ur5_twinwrist/config/*.yaml
examples/ur5_twinwrist/config_loader.py
examples/ur5_twinwrist/local_hardware.py
examples/ur5_twinwrist/teleop_hardware.py
examples/ur5_twinwrist/teleop_controls.py
```

共同设计原则：

- import 模块不会打开 USB、串口、RTDE 或发送运动；
- `pyserial`、`pyrealsense2`、`ur_rtde` 只在显式 `connect()` 路径延迟导入；
- 参数由 dataclass/mapping 和五份 YAML 输入；
- Fake backend/provider 可在无真机时测试；
- 设备异常会 fail-closed；
- UR/腕/夹爪/相机身份不由枚举顺序猜测。

本地腕实现不是只有 OpenRB 输出驱动，它现在包含三层：

```text
MasterWristReader
  → ESP32 STOP / SET_PERIOD / START / TELE，首帧建主腕相对零
OpenRBWrist
  → 只读 connect/GET_LIMITS、YAML servo zero Home、J1/J2 舵机 raw 目标与反馈
WristMasterSlaveController
  → policy > Ctrl SpaceMouse > ESP32 主腕仲裁，单 owner 串口 I/O 和状态缓存
```

`WristMasterSlaveController.connect(enable_motion=True)` 会真实执行从腕 HOME，因此必须受全局
`--enable-motion` 门禁保护。Home 完成后控制器仍为 parked，必须显式
`resume_master()` 才会以当下主腕姿态建零并开始跟随。底层 controller 的
`clear_spacemouse_velocity()` 单独调用时只清速并保持最后目标。完整遥操 worker
则会检测 Ctrl 松开边沿，先 clear 保持，再以松开时的主腕姿态调用
`resume_master()`，安全重建相对零后自动恢复主腕跟随。

所有 `MasterWristConfig` 参数已显式放入 `config/hardware.yaml`：ESP32 串口 timeout、
10 ms TELE 周期、50/10 Hz 控制/记录目标、Enc0→J1(+)/Enc1→J2(+)、10/14 raw/deg、
0.5° deadband、One-Euro、Dynamixel profile、velocity/acceleration/jerk shaper，以及 Ctrl
覆盖的 lease/速度/加减速斜坡。`config_loader.py` 会拒绝缺键、错轴、错符号、非有限数、
不相容频率、非法 servo zero，以及与 `teleop.yaml` 不一致的 Ctrl 最大速度。

截至本次审计，底层提取、统一硬件 worker、`TeleopCollectionSession`、125 Hz
SpaceMouse keepalive、按键生命周期、独立 10 Hz recorder 线程和
`teleop_collect.py::_run_real_collection` 已组成真实软件路径。可注入 Fake 测试已验证
四门禁先于硬件构造、Menu Home/开始、Fit 原子封存后 Home、Esc 拒收、J6/T 录制
禁入和 action receipt 录制。这仍不是真机验收；当前 preflight 会因关节限位未填和
旧采集进程占用而在打开硬件前拒绝。

## import 副作用和并发风险

新本地硬件模块导入时无设备副作用，CLI 都应由 `if __name__ == "__main__"` 保护。

旧夹爪虽然每个事务有锁，但控制、Home 和 Recorder 多个业务线程都会调用 `read_state()`；锁只能
防止字节交错，不能保证只有一个业务 owner。当前项目使用唯一 owner worker：open/read/write/close
都在一个线程，其他模块只读缓存。任一 SerialException/timeout 会永久 fail-closed。

主从腕同样要求一个 hardware owner 线程串行调用 `connect/step/home/stop/close`；UI 和
Recorder 只调用 `get()` 读不发串口 I/O 的不可变缓存。ESP32 TELE 超时、重复/倒退序号、
`zero_valid=0`、OpenRB 故障或任一串口异常都会 fail-closed 两条腕控制链。

当前主从腕已按 `open_loop_record` 参考迁入 One-Euro、完整 velocity/acceleration/jerk
target shaper、2 raw 发送 deadband、18 raw 单步限位和 Ctrl 速度斜坡。未迁入历史
AS5048A soft-home PI/前馈流程，因为新数据合同明确以舵机位置示数为真值，末端编码器只作诊断。

旧控制线程会分两步发布输入状态和命令字段，Recorder 有机会混入相邻 125 Hz 周期。当前
`local_hardware.py` 和 `teleop_hardware.py` 已在完整执行器周期后发布不可变
command receipt；`teleop_session.py` 的 recorder 线程已从 observation 取该 receipt 保存，
不保存 SpaceMouse 原始轴或上层请求值。

## 实际 state/action 合同

9 维 state：

```text
[UR actual_q 6 rad, wrist (j1_pos/j2_pos - YAML servo zero) 2 raw, gripper actual 1 normalized]
```

普通 SpaceMouse 控制最终调用 RTDE `speedL`，所以 9 维 action 是：

```text
[TCP velocity 6, wrist J1/J2 YAML-zero-relative raw target 2, gripper absolute target 1]
```

前六维不是关节目标，也不是 SpaceMouse raw axes。相机采集 30 Hz，原始 HDF5 默认记录 10 Hz，
底层遥操作循环为 125 Hz。完整逐维说明见[动作空间与数据格式](05_动作空间与数据格式.md)。

## 历史数据和旧同步路径发现的问题

1. 实际生产相机配对用的是 `devices/cameras/realsense_capture.py::FrameSynchronizer`，不是功能更丰富但
   未接入生产的 `collection/synchronization.py::RealFrameSynchronizer`。
2. 旧 `StationSynchronizer` 在阻塞相机读取前取得 `now`，UR 和 SpaceMouse 得到的是组装时间，不是
   真实设备采样时间；只有相机、腕和夹爪携带设备/缓存时间。
3. 旧采集约 30 Hz 直接写 LeRobot，没有 30 Hz capture / 10 Hz HDF5 分层、`.tmp`、原子 rename 和
   `rejected/`。
4. 只读抽查最近旧数据共 12,217 帧，其中 300 帧腕状态 age 超过 100 ms，最大 334.8 ms，却仍标记
   valid，说明旧 telemetry limit 不是严格录制门禁。
5. 同一数据有 1,761 帧 `target_qd` 非零但 TCP action 为零，因为 Home/J6 jog 使用 `speedJ`。
6. 旧 Fit 会把后续协调 Home 写进任务段；异常路径还可能 emergency-save 为成功 episode。
7. 相机角色和机器人状态通道都曾使用 `wrist` 名称；把来源转成一个 dict 会发生名称冲突，必须保留
   明确角色和有序索引。
8. 三台 RealSense 的 raw device clock 相互独立；跨相机 skew 只能比较同一主机
   `time.monotonic_ns()` 时间轴。

## 当前项目已实施的软件门禁

下列门禁已在对应本地模块和真实软件循环中实现并通过 Fake 测试；
不能据此声称它们已完成真机故障注入验收。

- 每个完整控制周期结束后才发布 command receipt。
- episode 门禁会拒绝 HOME/J6 `speedJ`，维护动作不发布可录制的 9 维 receipt。
- Menu 会先开夹爪/完整 Home/resume 再开始；Fit 先原子封存再 Home；
  Esc/RotationLock 会拒收活动 episode。
- `teleop.yaml` 明确禁用组合键/双击 toggle；真实运行状态机以
  `TeleopCollectionSession` 为准，不使用早期纯函数 `EpisodeButtonState` 猜测完整 Home 语义。
- NaN/Inf、stale、camera timeout/skew、重复序号、串口异常全部 fail-closed。
- 夹爪串口只有一个 worker 可读写。
- 相机按 serial 固定角色，每台独立线程；同步器选择满足 skew/age 的最新完整组合。
- HDF5 使用 `.tmp`、flush/fsync、原子 rename；异常和崩溃残留进入 `rejected/`。
- 真实运动需要 `--enable-motion` 和精确确认字符串；推理默认 shadow。
- π0.5 使用 `action_dim=32`、`action_horizon=10`，真实 9 维由现有 transform pad/crop，不改模型核心。

## 软件验收记录

- 第一阶段专项测试：22 passed。
- 本地三相机专项 Fake 测试：6 passed。
- 加入本地相机后的 wrist pipeline 回归：28 passed。
- FakeHardware 成功写入、关闭、重新加载并原子 rename 12 帧 HDF5。
- Fake HDF5 成功转换为锁定版 LeRobot，重新加载并随机读取至少 10 帧。
- 三相机并排预览视频和 state/action 曲线成功生成。
- 9D → 32D transform 和 32D → 9D crop 测试通过。
- 新 YAML 配置解析、一致性和 hash 测试通过。
- 主腕显式参数、错轴/错符号、非法频率、Ctrl 速度不一致等配置门禁均有负向测试。
- 文档收口后重跑配置、遥操映射、hardware worker、session、collect 入口、
  preflight、SpaceMouse echo 和 camera preview 专项：49 passed，没有连接或运动真实设备。
- 本地 UR、SpaceMouse、腕、两种夹爪和三相机均有不连接真实设备的 Fake 测试。
- 舵机示数合同更新后，同一工作树一次运行 `tests/test_ur5_twinwrist*.py`：136 passed；
  ruff 和 shell 语法检查通过。
- 新 Fake HDF5 已验证腕 action 为整数相对 raw，validator 无错误；锁定版 LeRobot 转换、
  12 帧重载、三相机预览视频和 state/action 曲线生成通过。
- 使用本地官方 websocket server + openpi-client 完成一次 `(10,9)` Fake shadow 请求；
  第 6/7 维 `[5,-6]` 保持相对 raw，`motion_sent=false`。
- 2026-08-31 只读 preflight 确认 3/3 相机、3 个 serial by-id、`spacenavd` 和 UR5
  `30004` 端口通过。但 required 门禁仍有三项失败：PolyScope joint min/max 为
  `null`；PID 1476 的旧 collection frontend 和 PID 7552 的旧 SpaceMouse worker 触发
  `no_other_collector=false`。未自动停止或 kill 任何用户进程。
- 所有上述验收都没有发送真实机器人运动。

测试数会随合并继续增长；最终应以同一 commit 上的完整 pytest 报告为准，不能把不同工作树的数字
拼成一次验收。

## 最小剩余修改清单

1. 由用户通过正常退出方式停止旧 collection frontend/SpaceMouse worker，再重跑
   preflight；新脚本不自动 kill 它们。
2. 将 PolyScope 精确 joint limits 写入 YAML，使 real-ready 在现场条件满足时通过。
3. 在现场完成已接线遥操路径的只读、低速单轴、按键生命周期和故障注入验收。
4. 完成第一条真机 HDF5、validator、人工视频/曲线检查和 Hiwonder 100 次压力测试。
5. 用真机低速单轴验证已迁入的 One-Euro/jerk shaper；AS5048A soft-home 保持停用。
6. 在 H100 完成 100-step JAX smoke、checkpoint reload，再运行正式训练。
7. 用真实 Orbax checkpoint 重跑已接线的 localhost client；先 Fake shadow，再真机 shadow，
   最后单步低速 rollout。

## 仍需真机确认

- UR PolyScope 精确 joint limits、TCP/tool、payload、安全平面、workspace 和 task-home。
- `front/side/top` 与三台 camera serial 的真实安装对应。
- OpenRB 当前固件、板端限位、J1/J2 raw 正方向和 YAML servo zero Home 重复性。
- 夹爪端点、实际反馈可靠性及 Hiwonder 100 次压力测试。
- 三台 D435 同时接入后的 USB 带宽、skew、掉帧和黑帧。
- 本地 orchestrator 的状态时间戳、最终 applied-action receipt 与真实执行完全一致。
- policy timeout、相机/串口故障、急停和程序异常时的真实 stop/hold 行为。
