# archive — 已废弃的探索脚本（不维护）

本目录存放早期方法探索与对照实验脚本，**已停止维护、不保证可运行**，仅供考古参考：

| 脚本 | 内容 | 废弃原因 |
|---|---|---|
| `run_accumulated_wire.py` | 多帧累积 + 预训练分割器（AerialLidarClassifier）可行性评估 | 依赖外部仓库 `/home/xjt/experiment/ObjectDistance/...` 绝对路径，实测 0 recall |
| `run_segformer_wire.py` | SegFormer 航测导线分割尝试 | 同上（外部插件依赖 + 绝对路径） |
| `run_litept_mls_wire.py` | LitePT-L MLS 移动测绘导线分割尝试 | 同上 |
| `bench_geom_wire.py` | 3 种传统几何法对照基准 | 无 CLI，硬编码 ROI 目录与 GT stem，仅适配原实验数据 |

其中 `bench_geom_wire.py` 还会 `import run_accumulated_wire`（本目录内互引用）。

正式管线请使用仓库根目录的 `range_person_wire.py`（主入口）、`ttpla_wire_3d.py`（导线分支）、
`open_vocab_3d_detect.py`（开集目标分支）与 `ranging_api.py`（Python 库接口）。
