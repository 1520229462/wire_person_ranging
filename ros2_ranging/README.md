# ros2_ranging — wire_person_ranging 的 ROS2 集成

把仓库根目录的测距管线（人 + 导线三维测距）包成 ROS2（目标 Humble，
Python 3.10）节点。**只使用标准消息**（sensor_msgs / std_msgs / std_srvs /
tf2），无自定义接口，不需要额外编译消息包。

```
ros2_ranging/
  ros2_ranging/
    ranging_node.py      rclpy 节点（订阅 image+cloud+camera_info，发布距离）
    callback_core.py     与 rclpy 无关的处理核心（无 ROS2 也可单测）
    converters.py        消息 <-> 数组转换
    sample_replay.py     把仓库 data/ 序列回放成话题 + TF（演示/联调）
  config/ranging_params.yaml   节点默认参数
  launch/ranging.launch.py      单独启动节点
  launch/sample_demo.launch.py  回放 + 节点一起启动（演示）
```

## 1. 前置条件

1. 仓库根目录的 Python 环境（`requirements.txt`；本仓库验证于 Python 3.8
   与 3.10，torch 2.5.1+cu118）。
2. 权重已就位（`python download_weights.py`；TTPLA 为私有分发，见
   `weights/README.md`）。
3. ROS2 **Humble**（Ubuntu 22.04）。节点只用 rclpy 常用 API + 标准包，
   Jazzy/Iron 大概率同样可用（未验证）。

`ranging_node` 默认按“仓库内布局”找到 `ranging_api.py`（`ros2_ranging/` 的上
一级目录）。若把包拷出去单独使用，设环境变量 `WPR_REPO_ROOT` 指向仓库根。

## 2. 构建与运行

```bash
cd <repo>/ros2_ranging
colcon build --symlink-install
source install/setup.bash

# 演示：回放仓库 data/ 的 28 帧图像+点云+外参，节点实时测距
ros2 launch ros2_ranging sample_demo.launch.py

# 生产：只起节点，接你们自己的相机/雷达驱动
ros2 launch ros2_ranging ranging.launch.py
```

运行期间查看结果：

```bash
ros2 topic echo /ranging_node/person_wire_distance   # 人-导线最小距离 (m)
ros2 topic echo /ranging_node/result                 # 完整 JSON 结果
ros2 service call /ranging_node/range_once std_srvs/srv/Trigger {}  # 单次触发
```

模型加载约 1-2 分钟（GPU）后打印 `ranging_node ready`。首次回放帧的处理
可能慢于回放速率，`ApproximateTimeSynchronizer` 队列会丢弃积压的旧帧，属
正常现象。

## 3. 输入契约（你们的驱动需要满足的）

| 输入 | 类型 | 要求 |
|---|---|---|
| `image_topic` | sensor_msgs/Image | **无畸变** RGB（rgb8 / bgr8 / mono8）。分辨率与内参一致：内置内参按 1280x720 标定，其他分辨率必须提供 `--intrinsics` 或 CameraInfo |
| `cloud_topic` | sensor_msgs/PointCloud2 | 至少含 x/y/z float32 字段（KITTI 式点云即可），点位于 `frame_id_lidar` 帧 |
| TF | tf2 | 每帧发布 `frame_id_camera`（相机坐标系）在 `frame_id_lidar`（雷达坐标系）下的位姿。即：**相机帧是雷达帧的子帧**。仓库 `data/extrinsics/<stem>.txt` 存的是 T_cam_to_lidar，节点内部取逆 |
| `camera_info_topic` | sensor_msgs/CameraInfo | 可选。提供时其 K 矩阵覆盖 `intrinsics` 参数 |

处理时机：`image` 与 `cloud` 按时间戳近似同步（`sync_slop` 秒）。每帧
推理约 1 秒，请按此控制输入频率（`sample_replay` 默认 1 Hz）。节点订阅
默认 QoS（reliable/volatile）；若你们的驱动按 sensor_data QoS
（best_effort）发布，把 `ranging_node.py` 里 `message_filters.Subscriber`
的 `qos_profile` 改成对应 profile 即可。

## 4. 输出契约

| 输出 | 类型 | 含义（单位 m，雷达坐标系） |
|---|---|---|
| `~/person_distance` | std_msgs/Float32 | 传感器 → 人体中心距离 |
| `~/wire_distance` | std_msgs/Float32 | 传感器 → 导线最近点距离 |
| `~/person_wire_distance` | std_msgs/Float32 | 人 ↔ 导线最近距离（核心量） |
| `~/result` | std_msgs/String | 完整结果 JSON（与 CLI 的 range.json 同 schema） |
| `~/range_once` | std_srvs/Trigger | 对最近一帧同步对跑一次并返回成败 |

## 5. 参数表（与 CLI 一一对应）

| ROS2 参数 | 默认 | 对应 CLI / 说明 |
|---|---|---|
| `categories` | `"person"` | `--categories` 开集类别文本，逗号分隔 |
| `wire_mode` | `"per_frame"` | `--wire_mode`：per_frame（TTPLA 每帧检测）/ catenary（静态曲线） |
| `wire_curve` | `""` | `--wire_curve`（catenary 模式下的曲线文件路径） |
| `intrinsics` | `""` | `--intrinsics` `'fx,fy,cx,cy'`；空 = 内置 1280x720 标定 |
| `out_dir` | `""` | 每帧可视化输出目录（range.json/range.txt/range_2d.jpg/range.ply）；空 = 不写盘 |
| `vggt_depth_dir` | `""` | 深度缓存目录；空 = 人分支退化为最近簇兜底（`degraded`） |
| `sam2` | `true` | `--sam2` 掩膜精化开关 |
| `device` | `"cuda:0"` | 推理设备，无 GPU 用 `"cpu"` |
| `frame_id_lidar` / `frame_id_camera` | `"lidar"` / `"camera"` | TF 帧名 |
| `image_topic` / `cloud_topic` / `camera_info_topic` | `/camera/image` 等 | 话题名 |
| `sync_slop` | `0.1` | 图像/点云近似同步窗口（秒） |
| `publish_on_error` | `true` | 帧失败时是否仍发布 `~/result`（含 error 字段） |

其余算法参数（`--conf`/`--lin_k`/`--min_points` 等）沿用 CLI 默认值；如需
调整，在启动参数里按同名传入即可（会透传给 RangingSession），例如
`parameters=[{"categories": "person,worker", "dilate": 120}]`。

## 6. 与 Python API 的关系

节点处理核心 `callback_core.RangingCore` 只是 `ranging_api.RangingSession`
的一层线程安全封装。不用 ROS2 的集成方可以直接用
`RangingSession.process(image_rgb, points, T_lidar_to_cam, ...)`（见仓库
README 的 Python API 章节）。

## 7. 已知限制

- 本开发服务器未安装 ROS2，`ranging_node.py` / `sample_replay.py` 的 rclpy
  部分经 `py_compile` + 静态审查 + 与 rclpy 文档核对，但**未在真实 ROS2
  环境运行验证**；`callback_core` 与 `converters` 的数值部分已在本机用
  真实数据验证（与 CLI 输出一致）。
- 同步推理在回调线程内执行；生产环境如需更高吞吐，请自行把 `_run` 挪进
  执行器线程池（本仓库刻意保持最简）。
- YOLO-World / TTPLA 权重为 AGPL-3.0（衍生），闭源商用集成需自行评估合规
  义务（见 `weights/README.md`）。
