# UR5 双轴腕 π0.5 项目中文文档

这个目录里的文件全部是本项目新增或整理的中文文档，不是 Physical Intelligence 官方 OpenPI 原文。

建议按下面顺序阅读：

1. [项目结构与代码来源](01_项目结构与代码来源.md)：先看清每个目录负责什么，以及“旧参考工程”是什么意思。
2. [遥操作键位与零位](02_遥操作键位与零位.md)：SpaceMouse 键位、轴映射、UR/手腕/夹爪零位。
3. [YAML 配置说明](03_YAML配置说明.md)：哪些参数可以修改，修改后由谁读取。
4. [遥操作数采操作手册](04_遥操作数采操作手册.md)：从设备检查到保存第一条 HDF5。
5. [动作空间与数据格式](05_动作空间与数据格式.md)：9 维 state/action 和 HDF5 字段。
6. [OpenPI 训练与推理入门](06_OpenPI训练与推理入门.md)：LeRobot、norm stats、训练、policy server 的关系。
7. [Pipeline 审计记录](PIPELINE_AUDIT.md)：已经验证的事实、代码风险和仍需真机确认的事项。
8. [Pipeline 命令手册](PIPELINE_RUNBOOK.md)：4090 数采、H100 训练、4090 推理命令。
9. [手腕舵机示数链说明](07_手腕舵机示数数采与推理.md)：J1/J2 raw、YAML 零位、数采和推理换算。

## 如何区分官方文件和本项目文件

本目录 `docs/wrist/` 中的文件全部是 **UR5 双轴腕实验的项目文档**，不是
Physical Intelligence 发布的 OpenPI 官方原文。以下文件才是当前仓库中保留的
官方 OpenPI 文档或入口，本项目没有把它们改写成中文：

- `docs/remote_inference.md`
- `docs/norm_stats.md`
- `examples/ur5/README.md`
- `examples/libero/`
- `scripts/train.py`
- `scripts/compute_norm_stats.py`
- `scripts/serve_policy.py`

以下目录是本项目代码：

- `examples/ur5_twinwrist/`
- `src/openpi/policies/ur5_twinwrist_policy.py`
- `tests/test_ur5_twinwrist_*.py`
- `docs/wrist/`

判断方法很简单：与 UR5 双轴腕实验有关的中文说明只认本目录；查 OpenPI 官方行为时再看官方原文。
`examples/ur5_twinwrist/README.md`、`ACTION_SPACE.md` 和 `TELEOPERATION.md` 只保留短索引，避免同一
参数在两个地方形成相互矛盾的副本。仓库根 `docs/` 不再另外保存本项目的 PIPELINE 文档。

## 当前安全状态

- 训练和推理后端固定为 `backend=jax`。
- 所有真机入口默认不运动。
- 真实运动必须显式提供 `--enable-motion` 和确认字符串。
- 当前配置中的 UR5 精确关节软限位仍未从 PolyScope 填入，因此项目内 `real_ready` 检查应当失败。
- 2026-08-31 最新只读预检已确认 3/3 相机、3 个 serial by-id、`spacenavd`
  和 UR5 `30004` 端口通过；硬件可见性检查没有失败。
- 当前仍有三项 required 门禁失败：`safety.yaml` 的
  `joint_min_rad/joint_max_rad` 为 `null`；PID 1476 的旧 collection frontend 和
  PID 7552 的旧 SpaceMouse worker 被识别为互斥占用。因此 `ready=false`，真实运动不得启动。
  PID 会变化，应以每次新 preflight 报告为准；新脚本绝不能自动 kill 用户进程。
- `teleop_hardware.py`、`teleop_session.py` 与 `teleop_collect.py` 的真实循环已经完成
  项目内软件接线，并有依赖注入的 Fake 测试；它尚未做低速真机 episode 和故障注入验收，
  不能声称真机遥操数采已验证。
