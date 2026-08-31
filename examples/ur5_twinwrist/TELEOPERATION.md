# 遥操作文档索引

本文件只是兼容旧链接的中文索引。遥操作正文已经统一归档到：

- [遥操作键位与零位](../../docs/wrist/02_遥操作键位与零位.md)
- [YAML 配置说明](../../docs/wrist/03_YAML配置说明.md)
- [遥操作数采操作手册](../../docs/wrist/04_遥操作数采操作手册.md)
- [动作空间与数据格式](../../docs/wrist/05_动作空间与数据格式.md)

这些内容都是 UR5 双轴腕实验的项目说明，不是 Physical Intelligence 官方 OpenPI 文档。
官方 UR5 示例仍是 [`examples/ur5/README.md`](../ur5/README.md)。

键位只从 `config/teleop.yaml` 读取，UR task-home/手腕 J1/J2 舵机 raw 零位/夹爪 0开、1闭只从
`config/poses.yaml` 解释，设备身份和安全参数分别在 `config/hardware.yaml` 和
`config/safety.yaml`。`teleop_collect.py` 的 dry-run、`--fake` 和真实软件路径都不依赖
旧参考工程；真实 worker/session/recorder 已接线且有 Fake 测试，但 preflight 当前仍拒绝启动，
也尚未做真机低速验收。
