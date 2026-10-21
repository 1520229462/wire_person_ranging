"""Message <-> array converters for the ranging node (standard messages only).

All conversions go through stable standard packages: sensor_msgs_py for
PointCloud2, cv_bridge for Image, plain numpy for transforms.
"""
import numpy as np


def cloud2_to_points(msg):
    """sensor_msgs/PointCloud2 -> (N,3) float64 points (first 3 fields).

    The cloud must carry at least x/y/z float32 fields (KITTI-style lidar
    output satisfies this). Raises ValueError otherwise.
    """
    from sensor_msgs_py import point_cloud2
    names = [f.name for f in msg.fields]
    for need in ("x", "y", "z"):
        if need not in names:
            raise ValueError(
                f"PointCloud2 has no '{need}' field (got {names})")
    data = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
    return np.asarray(data, dtype=np.float64).reshape(-1, 3)


def image_to_rgb(bridge, msg):
    """sensor_msgs/Image -> contiguous (H,W,3) uint8 RGB numpy array.

    Accepts rgb8 / bgr8 / mono8 encodings.
    """
    import cv2
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if msg.encoding in ("bgr8", "8UC3"):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    elif msg.encoding in ("mono8", "8UC1"):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif msg.encoding not in ("rgb8",):
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    return np.ascontiguousarray(cv_img)


def camera_info_to_intrinsics(msg):
    """sensor_msgs/CameraInfo -> (fx, fy, cx, cy) from the K matrix."""
    k = msg.k
    return (float(k[0]), float(k[4]), float(k[2]), float(k[5]))


def tf_transform_to_matrix(transform):
    """geometry_msgs/TransformStamped -> 4x4 numpy matrix (target <- source)."""
    from geometry_msgs.msg import TransformStamped
    assert isinstance(transform, TransformStamped), \
        "lookup_transform() returned a Transform, expected TransformStamped"
    t = transform.transform.translation
    q = transform.transform.rotation
    # quaternion -> rotation matrix
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T
