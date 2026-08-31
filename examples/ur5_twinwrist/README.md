# UR5 双轴腕实验代码索引

这里是本项目新增的 UR5、双轴腕、夹爪、SpaceMouse、三相机数采与 π0.5 适配代码，
不是 Physical Intelligence 官方示例。完整中文说明统一放在
[`docs/wrist/`](../../docs/wrist/README.md)，请从那里开始阅读。

- 遥操作数采：[04_遥操作数采操作手册.md](../../docs/wrist/04_遥操作数采操作手册.md)
- 键位与零位：[02_遥操作键位与零位.md](../../docs/wrist/02_遥操作键位与零位.md)
- YAML 参数：[03_YAML配置说明.md](../../docs/wrist/03_YAML配置说明.md)
- 动作与数据格式：[05_动作空间与数据格式.md](../../docs/wrist/05_动作空间与数据格式.md)
- 全链路命令：[PIPELINE_RUNBOOK.md](../../docs/wrist/PIPELINE_RUNBOOK.md)

官方 OpenPI 文档仍在仓库原有位置，例如 `docs/remote_inference.md`、
`docs/norm_stats.md` 和 `examples/ur5/README.md`；不要把它们与本目录的项目适配代码混淆。

快速定位：`controller/` 放 UR5、SpaceMouse、主从腕和两种夹爪驱动；`cameras/`
放 D435 采集、同步和只读 preview；`config/` 的五份 YAML 是 IP、serial、键位、零位、安全边界
和数据字段的人类可编辑来源。

文档中的 `legacy` 仅指 `/home/user/shiyi/slai-manipulation` 旧工程的只读历史参考。
当前默认 dry-run/Fake/本地硬件模块不导入或运行它。
`teleop_hardware.py` + `teleop_session.py` 已接入 `teleop_collect.py` 的真实软件路径并有
Fake 测试，但当前 preflight 仍会因 PolyScope 关节限位未填和旧采集进程占用而拒绝启动；
该路径没有做真机低速验收。
