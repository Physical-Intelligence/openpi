# UR5 双轴腕 π0.5 Pipeline 命令手册

本文是本项目的可执行命令清单，不是官方 OpenPI 文档。训练和推理后端始终明确为
`backend=jax`，不做 JAX/PyTorch 权重转换。

详细原理请先看本目录的 01–06 文档。官方 CLI 的最终参数仍以当前工作树的 `--help` 为准。

## 0. 当前完成度

| 环节 | 软件状态 | 真机/远端状态 |
|---|---|---|
| 项目 YAML 和 hash | 已实现并有 Fake 测试 | PolyScope joint limits 待填写 |
| UR/SpaceMouse/腕/夹爪本地底层 | 已提取并有 Fake 测试 | 连接节点/网络只读预检通过，待低速验收 |
| 三相机本地采集/同步 | 已实现 Fake 测试和只读 preview | 3/3 serial 枚举通过，待三路实际画面验收 |
| 遥操真实软件路径 | worker/session/recorder/按键已接入 `teleop_collect.py`，有 Fake 测试 | 当前 preflight 门禁阻止，未做低速真机验收 |
| 原始 HDF5 | Fake 写入、原子 rename、重载已通过 | 待首条真机 episode |
| LeRobot 离线转换 | Fake 数据转换和重载已通过 | 待真机数据转换 |
| π0.5 transform/config | 9→32→9 测试已通过 | 100-step H100 smoke 待执行 |
| policy server/client | localhost client、timeout、(H,9) chunk 和 shadow 已接线 | 真 checkpoint/server 请求待验收 |
| 真实推理执行 | 默认阻止 | 待所有前序门禁通过后低速验收 |

2026-08-31 最新只读预检中，3/3 相机、3 个 serial by-id、`spacenavd` 和 UR5
`30004` 端口已通过。但 required 门禁仍有三项失败：

- `safety.yaml` 的 `joint_min_rad/joint_max_rad` 仍为 `null`。
- PID 1476 的旧 collection frontend 仍在运行。
- PID 7552 的旧 SpaceMouse worker 仍在运行。

后两项使 `no_other_collector=false`。PID 只是当次快照，应以新 preflight 为准；
由用户正常停止旧前端/worker 后重跑预检，不允许新脚本自动 kill。
当前 `ready=false`，真实运动不得启动。

## 1. 两台机器必须保持相同的身份

4090 和 H100 必须使用完全相同的：

- 自定义 Git commit；
- OpenPI 基线历史；
- `pyproject.toml` 和 `uv.lock`；
- 五份 YAML；
- 9 维 state/action 定义；
- 相机字段和角色映射。

在两台机器分别运行并保存输出：

```bash
git rev-parse HEAD
git status --short
sha256sum pyproject.toml uv.lock
sha256sum examples/ur5_twinwrist/config/*.yaml
sha256sum docs/wrist/05_动作空间与数据格式.md
uv run examples/ur5_twinwrist/config_loader.py
```

两台机器各自创建 `.venv`，绝不复制虚拟环境。

## 2. 4090 环境

```bash
cd /home/user/haitao_files/robowrist/pi0.5/openpi
uv venv --python 3.11
uv sync --frozen --group hardware
```

确认核心和真机依赖：

```bash
uv run python -c "import openpi, jax, lerobot, serial, pyrealsense2, rtde_control, rtde_receive; print('imports ok')"
uv run python -c "import jax; print(jax.__version__, jax.devices())"
```

SpaceMouse 还依赖系统服务：

```bash
systemctl is-active spacenavd
ldconfig -p | rg libspnav
```

## 3. 数采前只读门禁

```bash
uv run examples/ur5_twinwrist/config_loader.py

mkdir -p data/preflight
uv run examples/ur5_twinwrist/teleop_preflight.py \
  --config-dir examples/ur5_twinwrist/config \
  > data/preflight/teleop_$(date +%Y%m%dT%H%M%S).json
```

需要真实运动时，再检查已校准的关节限位：

```bash
uv run examples/ur5_twinwrist/config_loader.py --require-real-ready
```

任何命令非零退出、preflight `ready=false`、设备缺失、另一个 collector 存在或 config hash 不符，
都必须停止。完整现场步骤见[遥操作数采操作手册](04_遥操作数采操作手册.md)。

项目内 SpaceMouse 只读 echo：

```bash
uv run python -m examples.ur5_twinwrist.spacemouse_echo \
  --read-input \
  --duration-s 30
```

三相机工具默认只打印 `role→serial`，显式 `--capture` 才打开三路并保存单图/拼图。
两种模式都永不连接或运动机器人：

```bash
uv run python -m examples.ur5_twinwrist.cameras.preview

uv run python -m examples.ur5_twinwrist.cameras.preview \
  --capture \
  --output data/camera_preview
```

## 4. 软件级 Fake 验收

```bash
uv run pytest -q \
  tests/test_ur5_twinwrist_config.py \
  tests/test_ur5_twinwrist_teleop_controls.py \
  tests/test_ur5_twinwrist_teleop_hardware.py \
  tests/test_ur5_twinwrist_teleop_session.py \
  tests/test_ur5_twinwrist_teleop_collect.py \
  tests/test_ur5_twinwrist_preflight.py \
  tests/test_ur5_twinwrist_spacemouse_echo.py \
  tests/test_ur5_twinwrist_camera_preview.py \
  tests/test_ur5_twinwrist_local_ur_spacemouse.py \
  tests/test_ur5_twinwrist_local_wrist_gripper.py \
  tests/test_ur5_twinwrist_local_cameras.py \
  tests/test_ur5_twinwrist_dataset.py \
  tests/test_ur5_twinwrist_transforms.py
```

录制、验证、查看一条 Fake HDF5：

```bash
FAKE_ROOT=/tmp/ur5_twinwrist_fake
uv run examples/ur5_twinwrist/teleop_collect.py \
  --config-dir examples/ur5_twinwrist/config \
  --output "$FAKE_ROOT" \
  --episodes 1 \
  --fake \
  --fake-frames 12
uv run examples/ur5_twinwrist/validate_dataset.py "$FAKE_ROOT" --max-camera-skew-ms 50
uv run examples/ur5_twinwrist/inspect_episode.py "$FAKE_ROOT/episode_0000.hdf5"
```

转换为当前 `uv.lock` 锁定版本的 LeRobot，并重新加载/生成预览：

```bash
uv run examples/ur5_twinwrist/convert_to_lerobot.py \
  --source "$FAKE_ROOT" \
  --repo-id local/ur5_twinwrist_fake \
  --output-root /tmp/lerobot
```

转换结束后应看到非零 frames/episodes，并生成：

```text
three_camera_preview.mp4
state_action.png
```

## 5. 4090 遥操作数采

先 dry-run，不打开硬件：

```bash
uv run examples/ur5_twinwrist/teleop_collect.py \
  --config-dir examples/ur5_twinwrist/config \
  --output data/raw/ur5_twinwrist \
  --episodes 1
```

当前 CLI 仅接受：

```text
--config-dir  --output  --episodes  --fake  --fake-frames
--gripper-backend  --skip-camera-enumeration  --enable-motion  --confirm
```

`record_hz` 从 `config/collection.yaml` 读取，不是 CLI 参数。
`teleop_collect.py::_run_real_collection` 已接入本地 worker/session/recorder，但目前
PolyScope 限位未填且旧采集进程触发互斥门禁。所以下列命令仅记录将来
preflight 全部通过后的真实 CLI 形式，现在不应执行：

```bash
RAW_ROOT=data/raw/task2_$(date +%Y%m%dT%H%M%S)

uv run examples/ur5_twinwrist/teleop_collect.py \
  --config-dir examples/ur5_twinwrist/config \
  --episodes 1 \
  --gripper-backend hiwonder \
  --output "$RAW_ROOT" \
  --enable-motion \
  --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
```

这条真实路径在四道门禁全部通过之前不会构造 SpaceMouse/硬件对象：
`--enable-motion`、精确确认字符串、严格 YAML safety 校验、实时 preflight `ready=true`。
禁止为了启动而删除其中任意门禁。

结束后立即验证：

```bash
uv run examples/ur5_twinwrist/validate_dataset.py "$RAW_ROOT" --max-camera-skew-ms 50
uv run examples/ur5_twinwrist/inspect_episode.py "$RAW_ROOT/episode_0000.hdf5"
```

所有失败、丢弃、中断、串口异常和安全门禁 episode 都保留在 `rejected/`，不得删除后伪装成
成功率更高的数据集。

## 6. 真机 HDF5 转为 LeRobot

只转换 validator 通过的成功目录：

```bash
LEROBOT_ROOT=data/lerobot
DATASET_REPO_ID=local/ur5_twinwrist_task2

uv run examples/ur5_twinwrist/convert_to_lerobot.py \
  --source "$RAW_ROOT" \
  --repo-id "$DATASET_REPO_ID" \
  --output-root "$LEROBOT_ROOT"
```

人工播放三相机并排视频、查看 state/action 曲线，并核对每条 episode 长度。字段必须是：

```text
observation.state
action
observation.images.front
observation.images.side
observation.images.top
task
```

## 7. 数据同步到 H100

脚本使用 rsync 的 `--partial --append-verify`，支持断点续传且没有 `--delete`。先 dry-run：

```bash
export H100_HOST=填写主机
export H100_USER=填写用户名
export LOCAL_DATA_ROOT=/绝对路径/data/lerobot/local/ur5_twinwrist_task2
export H100_DATA_ROOT=/远端绝对路径/data/lerobot/local/ur5_twinwrist_task2

examples/ur5_twinwrist/sync_data_to_h100.sh --dry-run
```

检查目标无误后再实际传输：

```bash
examples/ur5_twinwrist/sync_data_to_h100.sh
```

传输完成后在 H100 重新比较 Git、lock、YAML、动作合同 hash，并重新加载随机至少 10 帧。

## 8. H100 环境和 CLI 核对

H100 使用自己的 Python 3.11 `.venv`，不安装硬件组：

```bash
cd /远端绝对路径/openpi
uv venv --python 3.11
uv sync --frozen
```

训练前必须先运行当前本地 CLI help，不凭记忆猜参数：

```bash
uv run scripts/train.py --help
uv run scripts/compute_norm_stats.py --help
```

当前 `training/config.py` 已对用户主动省略的 `aloha_policy.py` 做了最小可选
import 守卫，UR5 自定义 config/CLI 可正常 import；误选 Aloha config 会明确报错。
如果上述 `--help` 仍失败，应按真实 traceback 修复当前工作树，不能绕开官方训练入口，
也不能恢复为另一个 OpenPI 副本。

## 9. norm stats 和 100-step smoke training

`train_h100.sh` 会检查 GPU 数、batch 是否能被设备数整除、保存环境快照，并先计算 norm stats。
第一次只运行 100 step。`LEROBOT_DATA_ROOT` 必须是包含 `local/` 的根目录，
而不是数据集叶子目录；脚本检查的完整路径为
`$LEROBOT_DATA_ROOT/$DATASET_REPO_ID`：

```bash
export LEROBOT_DATA_ROOT=/远程绝对路径/data/lerobot
export DATASET_REPO_ID=local/ur5_twinwrist_task2
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_STEPS=100 \
BATCH_SIZE=16 \
FSDP_DEVICES=4 \
  examples/ur5_twinwrist/train_h100.sh
```

验收条件：

- 明确为 OpenPI/JAX 路径；
- 四张指定 GPU 实际参与；
- loss 和 gradient 有限；
- 100 step 正常结束；
- Orbax checkpoint 元数据完整；
- checkpoint 能被实际重新加载；
- 数据集、norm stats、代码和配置身份均有记录。

仅看到进程 `RUNNING`、GPU 占用或旧日志不算成功。

## 10. 正式 10,000-step 训练

100-step smoke 完整通过后才运行：

```bash
export LEROBOT_DATA_ROOT=/远程绝对路径/data/lerobot
export DATASET_REPO_ID=local/ur5_twinwrist_task2
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_STEPS=10000 \
BATCH_SIZE=16 \
FSDP_DEVICES=4 \
  examples/ur5_twinwrist/train_h100.sh
```

正式验收还要求持续推进且有限的 loss/gradient、完整保存的最终 Orbax checkpoint 和当前 run 的
环境/配置证据。本地命令手册不会自动提交、启动或停止 CCI/ACP；任何远端资源操作都需要单独明确授权。

## 11. checkpoint 同步回 4090

先 dry-run：

```bash
export H100_HOST=填写主机
export H100_USER=填写用户名
export H100_CHECKPOINT_ROOT=/远端绝对路径/checkpoints/选定step
export LOCAL_CHECKPOINT_ROOT=/本地绝对路径/checkpoints/选定step

examples/ur5_twinwrist/sync_checkpoint_to_4090.sh --dry-run
```

核对路径后传输：

```bash
examples/ur5_twinwrist/sync_checkpoint_to_4090.sh
```

传回后要核对完整文件 manifest/SHA-256，并在 4090 实际加载 checkpoint；只看目录存在不算成功。

## 12. 4090 Fake shadow inference

当前 wrapper 启动 localhost policy server 后运行的是 `FakeHardware` shadow 客户端。这是软件验收，
不是已经接通真实机器人的证明：

```bash
CHECKPOINT=/本地绝对路径/orbax_checkpoint \
PROMPT='完成 TASK2 操作任务' \
FAKE=1 \
  examples/ur5_twinwrist/run_infer_4090.sh
```

验收应包括：server 成功加载 checkpoint、客户端请求返回、action chunk 形状正确、所有值有限、
最终裁剪为 9 维，并且 FakeHardware 的 `sent_actions` 在 shadow 模式保持为空。

## 13. 真机 shadow 和低速 rollout

`robot_runtime.py` 已接到本地 `LocalRobotHardware` 和官方 openpi-client。真机 shadow 不发送动作，
但会打开 UR/腕/夹爪/相机并读取状态，所以仍只能在设备身份和占用检查通过后运行：

```bash
CHECKPOINT=/本地绝对路径/orbax_checkpoint \
PROMPT='完成 TASK2 操作任务' \
FAKE=0 \
  examples/ur5_twinwrist/run_infer_4090.sh
```

此命令固定带 `--shadow` 和 `--chunk-steps 1`。它只能证明 observation→server→action chunk→
safety filter 的真机只读链，不能证明运动执行已通过。

仍必须按以下顺序验收：

1. 真机只读 observation，不启动 policy server，不运动。
2. policy server + 真机 client，保持 shadow，只打印动作。
3. 验证 policy timeout、stale、NaN/Inf、相机、串口、急停会停止/保持。
4. 第一次 rollout 每个 action chunk 只执行第一步。
5. 在示教器低速、空工作区和可急停条件下做单轴/小范围动作。
6. 稳定后才允许每次执行 2–4 步，且仍经过统一 safety filter。

## 14. 回滚

- 先停止 robot runtime 和 policy server，确认 UR、腕、夹爪已 stop/hold。
- 当前集成隔离在 `twinpath-pi05`；需要比较官方基线时，新建 worktree/branch，不对脏工作树执行
  `git reset --hard` 或破坏性 checkout。
- 旧参考工程保持只读。当前默认运行不依赖它，回滚也不是自动切换到旧虚拟环境；
  如确需调用隔离的历史 adapter，必须单独审核和记录身份。
- 保留 raw、`rejected/`、转换数据、norm stats、checkpoint 和报告，任何回滚都不删除实验证据。
- rsync 脚本不使用 `--delete`；先 dry-run，传输中断后可续传。
