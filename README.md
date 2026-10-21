# 导线 + 人体 识别与测距（wire & person detection + ranging）

视觉（相机）+ 点云（LiDAR / mmWave 雷达）融合的目标识别与测距。本目录把「人（模板匹配）+ 导线（悬链线）」两条检测路径与测距步骤整理成自包含包。

- **人**：YOLO-World 2D 框 → SAM2 掩膜 → VGGT 单目深度模板匹配（`vggt_target_template` → DBSCAN 聚类 → 逐簇 shape-Chamfer 选最近形状簇）。
- **导线**：TTPLA 2D 线掩膜 → 骨架 → 带宽 → 点云候选 → **逐帧 PCA 局部线性度（k-NN）+ 最近相干深度簇（gap 统计）** → 3D PCHIP 曲线（默认，无绝对深度/高度常量）；可选静态悬链线路径（多帧累积 → RANSAC 粗线 → 悬链线拟合，`--wire_mode catenary`）。
- **测距**：在 lidar 系下给出 `person_distance`（传感器→人）、`wire_distance`（传感器→导线最近点）、`wire_person_min_distance`（导线→人体表面最近 3D 距离）。

## 目录结构

```
wire_person_ranging/
├── range_person_wire.py        # ★ 主入口：人(模板匹配)+导线(悬链线) 逐帧测距
├── open_vocab_3d_detect.py     # 几何/投影/模板匹配原语 + open-vocab 检测管线
├── ttpla_wire_3d.py            # TTPLA 前端（run_ttpla_line / skeleton_polyline）
├── ttpla_wire_3d_catenary.py   # 悬链线导线路径（collect_candidates / fit_catenary）
├── bench_geom_wire.py          # (参考) 3 种传统几何法对照基准
├── run_accumulated_wire.py     # (参考) 多帧累积 + 预训练分割器可行性评估
├── run_segformer_wire.py       # (参考) SegFormer 航测导线分割
├── run_litept_mls_wire.py      # (参考) LitePT-L MLS 移动测绘导线分割
├── export_roi_for_matlab.py    # (参考) 导出 2D 掩膜对应 3D ROI（供 MATLAB）
├── annotated_wire_detect.py    # (参考) 手标导线/人体辅助脚本
├── download_weights.py         # (参考) 权重下载
├── requirements.txt
├── run_ranging.sh              # 一键跑 28 帧测距
├── data/                       # 完整原始/派生数据
│   ├── images/                 # 28 张相机图 (1280×720 JPEG)
│   ├── lidar/                  # 28 帧 LiDAR 点云 (KITTI .bin, xyz+intensity)
│   ├── extrinsics/             # 28 帧外参 (4×4, T_cam_to_lidar)
│   ├── radar_depth/            # 28 帧 mmWave 雷达深度 (720×1280 float32, log 归一化)
│   ├── vggt_depth/             # 28 帧 VGGT 单目深度 (720×1280 depth+conf, 派生)
│   └── calibration.csv         # 传感器轨迹标定 (tx,ty,tz,rx,ry,rz)
├── weights/
│   ├── yoloworld/yolov8m-worldv2.pt   # open-vocab 检测器（默认）
│   ├── sam2/sam2.1_hiera_base_plus.pt # SAM2 掩膜精化
│   ├── ttpla/yolo11s_seg_1280.pt      # TTPLA 导线分割
│   └── clip/ViT-B-32.pt               # CLIP 文本编码器（YOLO-World set_classes 需要）
└── precomputed/
    ├── catenary_curve.txt      # 预计算的悬链线导线（lidar 系，28 帧累积拟合）
    └── catenary_fit.json       # 悬链线参数 (a/s0/c/垂度/内点/端点)
```

## 数据说明

| 数据 | 传感器 | 说明 |
|---|---|---|
| `images/` | 相机 | 无人机挂载相机，1280×720 |
| `lidar/` | LiDAR | `.bin` KITTI 格式，`(N,4)` float32 = xyz + intensity |
| `radar_depth/` | mmWave 雷达 | 归一化 log 深度图（本测距路径用 LiDAR，此为完整原始数据留存） |
| `extrinsics/` | 标定 | `T_cam_to_lidar` 4×4；代码 `load_extrinsic` 取逆得到 `T_lidar_to_cam` |
| `vggt_depth/` | 派生 | VGGT 单目深度（canonical，需逐帧 scale 对齐到 metric；模板匹配只用其形状，scale 无关） |

坐标约定：`range_person_wire.py` 输出的所有距离/中心均为 **lidar 系**（米）；`distance = ||center||` 即传感器原点到目标中心的欧氏距离。

## 快速开始

```bash
# 环境：conda env ov3d（torch 2.5.1+cu118 等），依赖见 requirements.txt
conda activate ov3d

# 一键跑满 28 帧（逐帧检测导线 + 逐帧模板匹配人 + 测距，端到端）
./run_ranging.sh                      # 或：
python range_person_wire.py --gpu 0   # 默认 --frames all --wire_mode per_frame

# 静态悬链线导线（读 precomputed，或从 28 帧重新拟合）
python range_person_wire.py --wire_mode catenary --gpu 0
python range_person_wire.py --wire_mode catenary --wire_curve recompute --gpu 0
```

> 注意：**请在包根目录运行**（`run_ranging.sh` 已自动 `cd` 到脚本目录）。YOLO-World 的
> `set_classes` 会按 `<cwd>/weights/clip/` 查找 CLIP 编码器，首次运行若该目录无
> `ViT-B-32.pt` 会自动联网下载；本包已随附该权重，故在根目录运行可完全离线。

输出：

- `viz_range/results.json` — 28 帧完整推理结果（人中心/extents/score/距离、全部候选、导线端点、导线→人最近距离、最近人体点/导线点）。
- `viz_range/summary.csv` — 逐帧一行汇总（人 x/y/z、人距离、导线距离、导线→人最近距离、score、点数）。
- `viz_range/<stem>/range.json / range.txt / range_2d.jpg / range.ply` — 每帧详情与可视化（人框 + 导线重投影 + 最近人体点→导线点红线 + 距离标注；`range.ply` 含整帧原始点云灰底 + 人绿 + 导线蓝 + 红线）。

## 结果（28 帧，lidar 系）

**默认 `--wire_mode per_frame`（逐帧导线检测，端到端）：**

| 指标 | 值 |
|---|---|
| 导线 → 人 最近 3D 距离 | **mean 0.840 m**，std 0.035 m（min 0.731 / max 0.910），27/28 帧 |
| 人 传感器距离 | mean 4.602 m，std 0.015 m |
| 导线 传感器距离 | 逐帧检测（GT 帧 5.194 m；最近点随帧微变） |
| 人 检测置信度 | 0.88–0.90，每帧 25–84 点 |

**静态悬链线 `--wire_mode catenary`（对照）：**

| 指标 | 值 |
|---|---|
| 导线 → 人 最近 3D 距离 | mean 0.853 m，std 0.020 m（min 0.829 / max 0.908） |
| 人 传感器距离 | mean 4.601 m，std 0.016 m |
| 导线 传感器距离 | 5.165 m（静态最近点） |

> 距离取「导线曲线 → 人体表面最近点」的最小欧氏距离（不是人中心→导线），因此比旧口径（人中心→导线 ~1.15 m）更小。第 18/20 帧人点数偏少（34/25）且人略向导线侧移（y≈-1.65），属正常波动。
>
> 逐帧模式 1 帧（322151）导线仅 4 个 lidar 点，`wire_too_few_inliers` 跳过（与 `ttpla_wire_3d.py` 单帧检测一致）；其余 27 帧导线端点稳定 ≈ [4.36,-1.94,3.0]→[4.12,-2.78,1.5]（对齐手标 GT）。

## 关键实现说明

- **逐帧导线检测（默认）**：候选点先做 **PCA 局部线性度**（`_local_linearity`，k-NN 邻域 k=12——导线局部 1D、地板/墙 2D，但 lidar 沿扫描线采样使小邻域下背景也显 1D，故 k 必须够大），再对剩下的 1D 点按深度做 **gap 统计切簇**（`_nearest_coherent_depth`：显著 gap = `median(gap)+4·1.4826·MAD(gap)`），取**最近且点数≥3**的簇，跳过更近的单点。两个阈值都从数据长出，**无绝对深度/高度常量**。
- **导线静态假设（仅 `catenary` 模式）**：28 帧为近静止序列（体 7cm 漂移 / 2.7s），故导线几何只用累积点云拟合一次（`precomputed/catenary_curve.txt`）；多帧累积无配准、直接 `concatenate`（1cm 去重）。
- **模板匹配人**：VGGT 单目深度对细线不可靠，但对人体可用——先在 SAM2 掩膜内用 `_two_mode_threshold` 做近/远深度 k-means 二分类得到前景，再 unproject 成点云、SOR 去杂、FPS 下采样成模板；对 LiDAR 候选做 DBSCAN 聚类后逐簇算 scale 不变的 shape-Chamfer，取最近形状簇。
- **悬链线导线**：RANSAC 粗 3D 线 → 水平方向 dh → `z(s)=a·cosh((s−s0)/a)+c` 最小二乘（soft_l1，a>0 约束）→ 内点重拟合 → 稠密曲线；物理垂度 = 曲线相对两端弦的最大竖直偏差（实测 10.3 cm）。

## 参考脚本（探索/对比，非主流程）

`bench_geom_wire.py`、`run_segformer_wire.py`、`run_litept_mls_wire.py`、`run_accumulated_wire.py`、`export_roi_for_matlab.py`、`annotated_wire_detect.py` 为方法探索与对照，含第三方依赖（AerialLidarClassifier 插件、MATLAB 等）与部分绝对路径，仅作留存参考，不保证开箱即用。
