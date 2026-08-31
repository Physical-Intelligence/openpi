# 动作空间文档索引

本文件只是兼容旧链接的中文索引，不再保存动作合同正文。

UR5 双轴腕实验的 9 维 state/action、单位、范围、正方向、绝对/速度语义、HDF5 字段和
π0.5 的 9→32 维处理，统一以
[`docs/wrist/05_动作空间与数据格式.md`](../../docs/wrist/05_动作空间与数据格式.md)
为准。

该说明属于本项目，不是官方 OpenPI 文档。官方模型代码和说明仍位于 `src/openpi/models/`
和仓库原有 `docs/` 文件中。

当前本地硬件层已用 action receipt 表示实际下发的 9 维命令，并接入
`teleop_collect.py` 的 recorder 线程；该路径已有 Fake 测试，但尚未用真机 HDF5 验证。
