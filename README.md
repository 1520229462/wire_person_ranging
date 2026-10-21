# wire_person_ranging — 视觉 + 点云融合的开集目标识别与三维测距

> **English abstract** — This repository detects a person (open-vocabulary,
> category text driven) and a power wire in a fused camera + LiDAR frame and
> measures their 3D distances in the lidar frame. It ships three entry
> points: a CLI (`range_person_wire.py`), a Python API
> (`ranging_api.RangingSession`), and a ROS2 node (`ros2_ranging/`, standard
> messages only, ROS2 Humble). Per-frame inputs: undistorted image, LiDAR
> point cloud, camera intrinsics + extrinsics, and the open-vocabulary
> category text(s) (e.g. `person`); the wire is a separate fixed branch
> (TTPLA segmentation + PCA-linearity template, no scene-specific
> constants). Verified on 28 frames: wire→person mean 0.840 m vs manual GT
> 0.848 m (GT frame 0.905 m). YOLO-World / TTPLA weights are AGPL-3.0
> (derived) — review license obligations for closed-source use; the TTPLA
> checkpoint and VGGT depth cache are distributed privately (see
> [权重下载](#3-权重下载)).

| 人分支 | 导线分支 |
|---|---|
| YOLO-World 开集检测（类别文本，默认 `person`）→ SAM2 掩膜 → VGGT 深度形状模板匹配（DBSCAN 聚类 → 逐簇 shape-Chamfer 最近形状簇） | TTPLA 线分割 → 骨架 → 带宽 → 点云候选 → **PCA 局部线性度（k-NN）+ 最近相干深度簇（gap 统计）** → 3D PCHIP 曲线 + 端点切线外推 |

测距输出（lidar 系，米）：`person_distance`（传感器→人中心）、
`wire_distance`（传感器→导线最近点）、`wire_person_min_distance`
（导线→人体表面最近 3D 距离，核心量）。

---

## 目录

1. [概览](#1-概览)
2. [环境搭建](#2-环境搭建)
3. [权重下载](#3-权重下载)
4. [数据契约](#4-数据契约)
5. [快速开始](#5-快速开始)
6. [Python API](#6-python-api)
7. [ROS2 集成](#7-ros2-集成)
8. [参数表](#8-参数表)
9. [硬件规格](#9-硬件规格)
10. [验证与基准](#10-验证与基准)
11. [许可与已知限制](#11-许可与已知限制)
12. [破坏性变更（Breaking changes）](#12-破坏性变更breaking-changes)

---

## 1. 概览

### 1.1 管线

```
图像(1280×720 无畸变) ─┬─ YOLO-World(开集类别) → SAM2 掩膜 → VGGT 形状模板匹配 → 人表面点云 + 中心
                       │
点云(KITTI .bin)  ─────┼─ TTPLA 线掩膜 → 骨架 → 带宽 → 候选点 → PCA 线性度 → 最近相干深度簇
                       │        → 3D PCHIP 曲线（导线，独立固定分支）
相机内参 + 外参 ───────┴─ 投影/反投影（内参可覆盖；外参 = T_cam_to_lidar，代码内取逆）
                              │
                              └─→ 测距（lidar 系）→ results.json / summary.csv / 每帧 range_2d.jpg + range.ply
```

人分支的类别由**开集类别文本**驱动（`--categories "person,worker"`），换目标
不需要重训。导线是另一个固定分支：默认 `--wire_mode per_frame` 每帧用 TTPLA
重检测，无绝对深度/高度常量（先验是「导线是近处的长 1D 结构」，不是场景
阈值）；可选 `catenary` 模式用静态悬链线曲线（仅适合导线不动的场景）。

### 1.2 三种使用方式

| 方式 | 入口 | 适用 |
|---|---|---|
| CLI | `range_person_wire.py` | 离线批量处理、验证 |
| Python API | `ranging_api.RangingSession` | 嵌入自己的程序（无需 ROS2） |
| ROS2 节点 | `ros2_ranging/`（标准消息） | 机器人/无人机实时系统 |

三者共享同一套数值内核（`process_frame_arrays` / `detect_wire_curve_arrays`），
输出 schema 一致；CLI 与 API 的等价性由 `tests/check_equivalence.py` 逐字节
验证（见 [验证与基准](#10-验证与基准)）。

### 1.3 目录结构

```
wire_person_ranging/
├── range_person_wire.py        # ★ CLI 主入口：逐帧人+导线测距
├── ranging_api.py              # ★ Python API：RangingSession
├── deploy_utils.py             # 部署辅助（CLIP 权重目录、内参解析、汇总输出）
├── open_vocab_3d_detect.py     # 投影/外参/模板匹配原语 + 开集检测管线
├── ttpla_wire_3d.py            # 导线逐帧检测（detect_wire_curve[_arrays]）
├── ttpla_wire_3d_catenary.py   # 悬链线路径（catenary 模式）
├── manual_wire_person_ref.py   # 手标参考导线/人体（仅本场景对照）
├── ros2_ranging/               # ★ ROS2 包（Humble，标准消息，见其 README）
├── tests/                      # compare_runs.py / check_equivalence.py / generate_golden.py
├── benchmarks/baseline/        # 冻结的 28 帧基准输出（回归门禁用）
├── scripts/regression_check.py # 跨环境容差回归门禁
├── archive/                    # 方法探索脚本（不再维护，见 archive/README.md）
├── run_ranging.sh              # 一键 CLI 运行（自动选 env / GPU）
├── download_weights.py         # 权重下载 + 哈希校验
├── requirements.txt            # 核心依赖（+ requirements-owlv2.txt 可选）
├── data/{images,lidar,extrinsics,vggt_depth}/   # 数据契约见 §4
├── weights/{yoloworld,sam2,ttpla,clip}/         # 权重（.gitignore，下载见 §3）
└── precomputed/catenary_curve.txt               # catenary 模式静态曲线
```

---

## 2. 环境搭建

验证环境：**Python 3.10**（3.8 亦可）+ **CUDA 11.8 + torch 2.5.1**。
建议用 conda：

```bash
conda create -n wpr python=3.10 -y
conda activate wpr
# 1) 先装 torch（按你的 CUDA 版本；无 GPU 用 CPU 版）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
# 2) 其余依赖
pip install -r requirements.txt
```

> `requirements.txt` 已 pin 关键版本（ultralytics 8.4.163、sam2 1.1.0、
> openai-clip 1.0.1）。openai-clip 1.0.1 与 setuptools>=81 不兼容，已一并
> pin `setuptools<81`。可选：`pip install -r requirements-owlv2.txt`
> 启用 OWLv2 兜底检测器（`--detector owlv2`，主路径不需要）。

---

## 3. 权重下载

权重不进 git。下载并校验（全部文件按 md5/sha256 核对，见
`weights/README.md`）：

```bash
python download_weights.py
```

| 文件 | 用途 | 来源 | 许可 |
|---|---|---|---|
| `weights/yoloworld/yolov8m-worldv2.pt` | 开集检测器（默认） | ultralytics release | AGPL-3.0 |
| `weights/sam2/sam2.1_hiera_base_plus.pt` | SAM2 掩膜 | facebook/sam2.1 | Apache-2.0 |
| `weights/clip/ViT-B-32.pt` | CLIP 文本编码器（YOLO-World `set_classes` 必需） | OpenAI | MIT |
| `weights/ttpla/yolo11s_seg_1280.pt` | **导线分割（`per_frame` 默认必需）** |  | AGPL-3.0（衍生） |
| `weights/owlv2/`（可选） | OWLv2 兜底检测器 | google/owlv2 | Apache-2.0 |

**私有分发项（TTPLA checkpoint 与 VGGT 深度缓存）**：这两个文件不公开发布，
由原作者私下提供：

1. 向原作者索取 `yolo11s_seg_1280.pt`（TTPLA）与 `vggt_depth_28frames.zip`
   （可选深度缓存）。
2. TTPLA 放入 `weights/ttpla/`；深度 zip 放入 `data/` 后运行
   `python download_weights.py --with-vggt-depth` 解压。
3. 重新运行 `python download_weights.py` 校验哈希。

深度缓存**不是必需**：缺失时人分支自动退化为最近簇兜底（结果带
`"degraded": true`，精度略降）。若日后提供公开地址，原作者设置
`WPR_TTPLA_URL` / `WPR_VGGT_URL` 即可自动下载。

---

## 4. 数据契约

一帧 = 同 stem 的四个文件（`<stem>` 为任意帧 ID 字符串，如
`img_1700603346.892936`）：

| 路径 | 格式 | 内容 |
|---|---|---|
| `data/images/<stem>.jpg` | JPEG，**1280×720 无畸变 RGB** | 相机图像 |
| `data/lidar/<stem>.bin` | KITTI：`(N,4)` float32 = xyz + intensity | 点云（lidar 系） |
| `data/extrinsics/<stem>.txt` | 4×4 文本矩阵 = **T_cam_to_lidar**（相机在 lidar 系下的位姿） | 外参 |
| `data/vggt_depth/<stem>.npz` | `(720,1280)` float32 深度图（可选） | VGGT 单目深度缓存 |

关键约定：

- **内参**：内置 1280×720 标定 `fx=622.6716 fy=694.2052 cx=644.2907 cy=522.0188`
  （`open_vocab_3d_detect.py` 常量，单一事实来源）。**其他分辨率或不同相机
  必须传 `--intrinsics fx,fy,cx,cy`**（CLI）/ `intrinsics=(...)`（API）/
  CameraInfo 消息（ROS2），否则投影全错。图像尺寸不是 1280×720 时程序会打
  stderr 警告。
- **外参方向**：文件存 `T_cam_to_lidar`（相机原点在 lidar 系的位置/姿态）；
  代码在 `load_extrinsic` 里取逆得到 `T_lidar_to_cam`。ROS2 侧约定相反：
  发布 TF 时把相机帧作为 lidar 帧的**子帧**（见 §7）。
- **坐标**：所有输出距离/中心在 **lidar 系**（米），`distance = ||center||`。
- **测距口径**：`wire_person_min_distance_m` = 导线曲线 → **人体表面最近点**
  的最小欧氏距离（不是人中心）。
- **雷达数据**：`data/radar_depth/` 与 `data/calibration.csv` 为历史存档，
  **当前代码不读取**（测距用 LiDAR）。

第三方数据接入：把你们的帧按上述布局放好（`--data_dir` 指向该目录），
`--frames` 指定 stem 列表即可。目录结构是唯一的接入要求。

---

## 5. 快速开始

```bash
conda activate wpr        # 或 run_ranging.sh 自动选 env
cd <repo>                 # 任意 cwd 均可（CLIP 路径已由 deploy_utils 处理）

# 28 帧端到端（逐帧导线 + 模板匹配人 + 测距）
python range_person_wire.py --gpu 0
# 或指定帧 / 子集
python range_person_wire.py --frames img_1700603346.892936,img_1700603347.024661

# 开集类别文本（换目标不重训）
python range_person_wire.py --categories "person,worker"

# 你们的相机内参（fx,fy,cx,cy 逗号分隔）
python range_person_wire.py --intrinsics 700,700,640,360

# 静态悬链线导线（导线不动时可用；换场景需 --wire_curve recompute）
python range_person_wire.py --wire_mode catenary
```

一键脚本（自动选 env、GPU、预检依赖）：`./run_ranging.sh [gpu_id] [额外参数]`

**退出码**：`0` 完成（≥1 帧成功）；`1` 致命（无权重 / 0 帧成功 / 全部失败 /
`--strict` 下存在软错误帧）；`2` 用法错误（argparse）。

**输出**（`--out_dir`，默认 `viz_range/`）：

| 文件 | 内容 |
|---|---|
| `results.json` | 每帧完整结果（keyed by stem）：人中心/extents/score/点数/距离、全部候选、导线端点/长度/距离、导线→人最近距离、最近人体点/导线点；失败帧带 `error` |
| `summary.csv` | 逐帧一行 11 列（人 xyz、人距离、导线距离、导线→人距离、ref/bias 列、score、点数） |
| `<stem>/range.json` | 该帧结果副本 |
| `<stem>/range.txt` | 该帧人类可读摘要 |
| `<stem>/range_2d.jpg` | 可视化：人框 + 导线重投影 + 最近点连线 + 距离标注 |
| `<stem>/range.ply` | 整帧点云灰底 + 人（绿）+ 导线（蓝）+ 连线（红），MeshLab 可开 |

> `--write_ref` 默认**关闭**：ref/bias 两列来自原作者为本演示场景手工标注的
> 参考导线，第三方数据上无意义。显式打开会打 stderr 警告，ref 计算失败时
> 留空（`--strict` 下转致命）。

---

## 6. Python API

```python
import numpy as np
from PIL import Image
import open_vocab_3d_detect as ovd
from ranging_api import RangingSession

# 建会话（参数与 CLI 同名同默认；模型加载 1-2 分钟，只建一次）
session = RangingSession(categories="person",          # 开集类别文本
                         wire_mode="per_frame",         # 导线分支
                         intrinsics=None,               # None=内置标定
                         device="cuda:0")

# 方式 A：按文件（仓库 data/ 布局）
result = session.process_frame_files("img_1700603346.892936")

# 方式 B：纯内存（自己的数据，无文件 I/O）
image_rgb   = np.array(Image.open("img.jpg").convert("RGB"))   # (720,1280,3)
points      = ovd.load_lidar("scan.bin")                       # (N,3) lidar 系
T_lidar_to_cam = ovd.load_extrinsic("ext.txt")                 # 4×4（内部取逆）
result = session.process("my_frame_1", image_rgb, points, T_lidar_to_cam)

print(result["wire_person_min_distance_m"])            # 0.905
print(result["person"]["center"])                      # [4.166, -1.598, 1.125]

session.close()
```

要点：

- `process()` 不做文件 I/O——图像、点云、外参（`T_lidar_to_cam`）都由调用方
  给；`process_frame_files()` 是仓库数据布局的便捷壳。二者数值完全一致
  （`tests/check_equivalence.py` 逐字节验证）。
- 每帧输入 = 点云 + 图像 + 相机内外参 + 开集类别文本；导线是独立固定分支。
- `vggt_depth` 可选：传入 `(H,W)` float32 深度图启用形状模板匹配；None 则
  退化为最近簇兜底，结果带 `"degraded": true`。
- `out_dir` 可选：传入则写该帧的 range.json/range.txt/range_2d.jpg/range.ply。
- 批量：`session.run(stems=None, strict=False)` 跑全部帧并写 results.json +
  summary.csv（与 CLI 字节一致）。
- 无 ROS2 环境时用 API 即可；ROS2 节点的处理核心只是它的线程安全包装。

---

## 7. ROS2 集成

`ros2_ranging/`（目标 **ROS2 Humble**，**只依赖标准消息**，无需自定义接口）。
构建、话题/TF 契约、参数表、已知限制见 [ros2_ranging/README.md](ros2_ranging/README.md)：

```bash
cd ros2_ranging && colcon build --symlink-install && source install/setup.bash
ros2 launch ros2_ranging sample_demo.launch.py     # 回放仓库 28 帧演示
ros2 launch ros2_ranging ranging.launch.py         # 只起节点，接你们的驱动
```

节点订阅 `Image + PointCloud2 + CameraInfo(可选)`，外参从 **TF2 每帧读取**
（相机帧 = lidar 帧的子帧），发布 `person_distance / wire_distance /
person_wire_distance`（Float32）+ 完整 JSON（String）。**注意**：本开发服务器
未装 ROS2，rclpy 部分经编译 + 静态审查 + 无 ROS2 核心数值验证（与 CLI 输出
一致），但未在真实 ROS2 环境运行过；集成时如遇问题请提 issue。

---

## 8. 参数表

### 8.1 CLI（`range_person_wire.py`，默认值 = 已验证配置）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data_dir` / `--out_dir` | `data` / `viz_range` | 数据/输出目录（§4 布局） |
| `--frames` | `all` | 帧 stem 列表（逗号分隔） |
| `--categories` | `person` | **开集类别文本**（逗号分隔，YOLO-World） |
| `--intrinsics` | `None` | `fx,fy,cx,cy`；None = 内置 1280×720 标定 |
| `--gpu` | `0` | GPU 编号（无 GPU 自动 CPU） |
| `--strict` | off | 软错误帧（缺文件/检测失败）转致命，退出码 1 |
| `--write_ref` | **off** | 手标参考导线距离（仅本演示场景，第三方勿开） |
| `--wire_mode` | `per_frame` | `per_frame` 逐帧 TTPLA / `catenary` 静态曲线 |
| `--wire_curve` | `precomputed/catenary_curve.txt` | [catenary] 曲线文件；`recompute` 从帧重拟合 |
| `--ttpla_weights` | `weights/ttpla/...` | TTPLA 权重路径 |
| `--conf` | `0.5` | TTPLA 分割置信度 |
| `--dilate` | `150` | 骨架带宽半宽（px；须大于外参误差偏移） |
| `--lin_k` | `12` | PCA 线性度 k-NN（导线 1D vs 背景 2D 的判别） |
| `--lin_thr` | `0.9` | 线性度阈值（0..1） |
| `--gap_kmad` | `4.0` | 深度簇 gap 显著性（MAD 倍数） |
| `--min_size` | `3` | 深度簇最小点数 |
| `--wire_min_points` | `6` | 导线最少 inlier/带宽点 |
| `--n_sample` | `100` | 导线曲线采样点 |
| `--extend_m` / `--extend_to_z` | `1.0` / `3.0` | 上端切线外推长度 / 高度上限 |
| `--extend_top_k` / `--extend_step` | `5` / `0.05` | 外推切线拟合点数 / 步长 |
| `--mask_max_area` | `20000` | TTPLA 掩膜面积上限（防误检，0 关） |
| `--yolo_weights` / `--sam2` / `--sam2_ckpt` | 默认路径 / on / 默认路径 | 人分支权重与开关 |
| `--imgsz` / `--box_threshold` | `640` / `0.1` | YOLO-World 输入尺寸 / 框阈值 |
| `--min_points` | `10` | 框内最少 lidar 点 |
| `--dbscan_eps` / `--dbscan_min_samples` | `1.0` / `3` | DBSCAN 参数（m / 点） |
| `--min_dist` | `0.5` | 相机系近场剔除（m） |
| `--match_k` / `--tmpl_open` / `--tmpl_sor_m` | `128` / `2` / `2.0` | VGGT 模板匹配参数 |
| `--vggt_depth_dir` | `data/vggt_depth` | 深度缓存目录（缺帧自动退化） |

### 8.2 三入口参数映射

| 作用 | CLI | Python API | ROS2 参数 |
|---|---|---|---|
| 开集类别 | `--categories` | `RangingSession(categories=...)` | `categories` |
| 内参 | `--intrinsics fx,fy,cx,cy` | `intrinsics=(...)` 或 process 时覆盖 | `intrinsics` / CameraInfo |
| 导线分支 | `--wire_mode` | `wire_mode=...` | `wire_mode` |
| 静态曲线 | `--wire_curve` | `wire_curve=<path>` | `wire_curve` |
| 每帧可视化 | `--out_dir` | `out_dir` / process 的 `out_dir=` | `out_dir` |
| 深度缓存 | `--vggt_depth_dir` | `vggt_depth_dir` | `vggt_depth_dir` |
| 软错误转致命 | `--strict` | `run(strict=True)` | 服务返回 success=false |
| 设备 | `--gpu` | `device=` | `device` |
| 其余算法参数 | 同名 flag | 同名 kwarg | 同名参数透传 |

---

## 9. 硬件规格

| 项 | 实测值 |
|---|---|
| GPU 显存 | **~3 GB**（YOLO-World + SAM2 + TTPLA 同驻），单卡 RTX 2080 Ti 11GB 余量充足 |
| 单帧推理 | **~1 秒**（含 YOLO-World + SAM2 + TTPLA + 模板匹配；CPU 模式慢一个量级） |
| 模型加载 | 1-2 分钟（首次运行，每进程一次） |
| 输入 | 1280×720 无畸变 RGB + KITTI `.bin`（xyz+intensity） |

---

## 10. 验证与基准

### 10.1 28 帧结果（随附数据，lidar 系）

默认 `per_frame`（逐帧导线，端到端）：

| 指标 | 值 |
|---|---|
| 导线→人 最近 3D 距离 | **mean 0.840 m**，std 0.035（min 0.731 / max 0.910），27/28 帧 |
| 手标 GT | mean **0.848 m**（GT 帧 `img_1700603346.892936`：**0.905 m**） |
| 人 传感器距离 | mean 4.602 m，std 0.015 |
| 检测置信度 | 0.88–0.90，每帧 25–84 点 |

对照 `catenary`（静态悬链线）：导线→人 mean 0.853 m（GT 帧 0.908 m）。
误差帧 `img_1700603348.322151`：导线仅 4 个 lidar 点 → `wire_too_few_inliers`
跳过（已知、预期）。

### 10.2 回归门禁（本仓库自带）

| 工具 | 用途 |
|---|---|
| `tests/compare_runs.py <before> <after>` | **字节级**对比两次运行：所有输出 md5 全等 + range_2d.jpg 逐像素等 + 误差帧集等 + GT 0.905 / 均值 0.8403 断言（`--tol` 模式做跨环境容差对比） |
| `tests/check_equivalence.py` | CLI ↔ RangingSession 等价性：28 帧逐字节（含 ref 路径）、内存/文件路径一致、Namespace 参数覆盖、内参单一来源断言 |
| `scripts/regression_check.py <run>` | 对冻结基准 `benchmarks/baseline/` 的容差门禁（跨 GPU/驱动环境用）：导线→人每帧 ≤0.02 m、均值 ≤0.005 m |
| `tests/generate_golden.py` | 投影/外参/骨架等原语的金标准快照 |

CI/换机器时跑 `compare_runs.py --tol 0.02` + `regression_check.py` 即可判断
是否回归。注意 GPU 浮点非确定性：字节级门禁失败时**先原样重跑一次**再下结论。

---

## 11. 许可与已知限制

- 本仓库代码：**Apache-2.0**。
- 权重许可（务必评估后再商用）：YOLO-World、TTPLA 微调权重为 **AGPL-3.0**
  （衍生），闭源商用集成需自行评估合规义务；SAM2 / OWLv2 为 Apache-2.0，
  CLIP 为 MIT。详见 `weights/README.md`。
- TTPLA checkpoint 与 VGGT 深度缓存为**私有分发**（§3），请勿公开上传。

已知限制：

- **ROS2 节点未在真实 ROS2 环境验证**（开发服务器无 ROS2）：rclpy 部分静态
  审查通过，数值核心已用真实数据验证（与 CLI 一致）。
- 人分支在无 VGGT 深度缓存时退化为最近簇兜底（`degraded`），精度略降。
- 内参按 1280×720 标定；其他分辨率必须显式传入内参（§4）。
- 导线检测假设「导线是近处最长的 1D 结构」；多导线/近距离缠绕场景需自行
  验证（可调 `--lin_k`/`--gap_kmad`/`--min_size`）。
- `catenary` 模式假设导线静止；换场景需 `--wire_curve recompute`（程序会
  对「曲线来源帧不在当前帧集」打警告）。
- `data/radar_depth/`、`data/calibration.csv` 为存档，当前代码不读取。

---

## 12. 破坏性变更（Breaking changes）

为第三方部署引入的接口变化（旧调用需更新）：

| 变更 | 影响 | 迁移 |
|---|---|---|
| `--write_ref` 默认 `True`→**`False`** | 旧命令不再输出 ref/bias 列 | 需要手标对照时显式 `--write_ref`（第三方数据不应使用） |
| 退出码语义化 | 0/1/2（旧版全 0） | 部署脚本请按 `$?` 判断：`2`=用法错误，`1`=致命 |
| 新增 `--strict` | 软错误帧默认不致命 | CI 里加 `--strict` 可让任何帧失败使任务失败 |
| 新增 `--intrinsics` | 分辨率≠1280×720 时必须提供 | 传 `fx,fy,cx,cy` |
| 新增 `--categories` | 开集类别可配置（旧版硬编码 person） | 无需改动，默认 `person` 行为不变 |
| 缺深度缓存打警告 + `degraded` 标记 | 结果 JSON 多一个键（仅缺缓存时） | 下游解析需容忍新键 |
| `weights/` 不再随仓库 | 需运行 `download_weights.py` | 见 §3 |
| 参考脚本移入 `archive/` | 目录移动 | `archive/README.md` 有清单与替代方案 |

> 数值内核未变：`tests/compare_runs.py` 证明重构前后 28 帧输出**逐字节一致**
> （GT 0.905 m / 均值 0.840 m 不变）。
