"""Replay the bundled data/ sequence onto ROS2 topics (demo / integration test).

Publishes, per frame:
  - the undistorted image  -> sensor_msgs/Image (rgb8)
  - the lidar scan         -> sensor_msgs/PointCloud2 (x/y/z/intensity)
  - the extrinsics         -> TF: camera frame as a CHILD of the lidar frame
                              (data/extrinsics/<stem>.txt stores T_cam_to_lidar,
                              the pose of the camera in the lidar frame; TF
                              wants the camera frame expressed in the lidar
                              frame, i.e. T_lidar_to_cam = its inverse)
and once at startup:
  - the intrinsics         -> sensor_msgs/CameraInfo (optional override for
                              the ranging node)

Params:
  data_dir       where the repo's data/ lives (default: <repo>/data)
  rate           frame publish rate (Hz, default 1.0; models take ~1s/frame)
  loop           replay in a loop (default false)
  frame_prefix   TF child frame name (default 'camera')
  frame_id_lidar TF parent frame name (default 'lidar')
  intrinsics     'fx,fy,cx,cy' or '' for the built-in 1280x720 calibration

Run together with the ranging node:
  ros2 launch ros2_ranging sample_demo.launch.py
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import tf2_ros

REPO_ROOT = os.environ.get(
    "WPR_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SampleReplay(Node):

    def __init__(self):
        super().__init__("sample_replay")
        self.declare_parameter("data_dir", os.path.join(REPO_ROOT, "data"))
        self.declare_parameter("rate", 1.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("frame_id_lidar", "lidar")
        self.declare_parameter("frame_id_camera", "camera")
        self.declare_parameter("intrinsics", "")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("cloud_topic", "/lidar/points")
        self.declare_parameter("camera_info_topic", "/camera/info")

        self.data_dir = self.get_parameter("data_dir").value
        self.frame_lidar = self.get_parameter("frame_id_lidar").value
        self.frame_cam = self.get_parameter("frame_id_camera").value
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.pub_image = self.create_publisher(
            Image, self.get_parameter("image_topic").value, 10)
        self.pub_cloud = self.create_publisher(
            PointCloud2, self.get_parameter("cloud_topic").value, 10)
        self.pub_info = self.create_publisher(
            CameraInfo, self.get_parameter("camera_info_topic").value, 10)

        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in self._glob("images/*.jpg"))
        if not stems:
            raise RuntimeError(f"no frames under {self.data_dir}/images")
        self.stems = stems
        self.i = 0
        self.get_logger().info(f"{len(stems)} frames -> topics, "
                               f"TF {self.frame_lidar} <- {self.frame_cam}")
        self.publish_camera_info()
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter("rate").value), self._tick)

    def _glob(self, rel):
        import glob
        return glob.glob(os.path.join(self.data_dir, rel))

    # ------------------------------------------------------------------ #
    def publish_camera_info(self):
        s = self.get_parameter("intrinsics").value
        if s:
            fx, fy, cx, cy = (float(t) for t in s.split(","))
        else:
            sys.path.insert(0, REPO_ROOT)
            from deploy_utils import DEFAULT_INTRINSICS
            fx, fy, cx, cy = DEFAULT_INTRINSICS
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_cam
        msg.height, msg.width = 720, 1280
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.distortion_model = "plumb_bob"   # images are undistorted
        self.pub_info.publish(msg)
        self.get_logger().info(f"CameraInfo published (K={msg.k[:5]}...)")

    def _tick(self):
        stem = self.stems[self.i]
        self.i = (self.i + 1) % len(self.stems)
        if self.i == 0 and not self.get_parameter("loop").value:
            self.get_logger().info("sequence finished (loop=false)")
            return
        stamp = self.get_clock().now().to_msg()

        # extrinsics -> TF (camera child of lidar)
        T_cam_to_lidar = np.loadtxt(
            os.path.join(self.data_dir, "extrinsics", stem + ".txt"),
            dtype=np.float64)
        T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)
        t = T_lidar_to_cam[:3, 3]
        R = T_lidar_to_cam[:3, :3]
        from tf_transformations import quaternion_from_matrix  # tf2 dep
        q = quaternion_from_matrix(np.vstack([
            np.hstack([R, np.zeros((3, 1))]),
            [0, 0, 0, 1]]))
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.frame_lidar
        tf.child_frame_id = self.frame_cam
        tf.transform.translation.x, tf.transform.translation.y = t[0], t[1]
        tf.transform.translation.z = t[2]
        tf.transform.rotation.x, tf.transform.rotation.y = q[0], q[1]
        tf.transform.rotation.z, tf.transform.rotation.w = q[2], q[3]
        self.tf_broadcaster.sendTransform(tf)

        # image
        from PIL import Image as PILImage
        image_np = np.array(PILImage.open(
            os.path.join(self.data_dir, "images", stem + ".jpg")).convert("RGB"))
        img_msg = self.bridge.cv2_to_imgmsg(image_np, encoding="rgb8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.frame_cam
        self.pub_image.publish(img_msg)

        # lidar cloud (x/y/z/intensity float32)
        arr = np.fromfile(os.path.join(self.data_dir, "lidar", stem + ".bin"),
                          dtype=np.float32).reshape(-1, 4)
        cloud_msg = point_cloud2.create_cloud_xyz32(
            stamp, self.frame_lidar, arr[:, :3])
        # re-add intensity field
        fields = [point_cloud2.PointField(name="x", offset=0,
                                          datatype=point_cloud2.PointField.FLOAT32,
                                          count=1),
                  point_cloud2.PointField(name="y", offset=4,
                                          datatype=point_cloud2.PointField.FLOAT32,
                                          count=1),
                  point_cloud2.PointField(name="z", offset=8,
                                          datatype=point_cloud2.PointField.FLOAT32,
                                          count=1),
                  point_cloud2.PointField(name="intensity", offset=12,
                                          datatype=point_cloud2.PointField.FLOAT32,
                                          count=1)]
        cloud_msg.fields = fields
        cloud_msg.point_step = 16
        cloud_msg.row_step = 16 * len(arr)
        cloud_msg.width = len(arr)
        cloud_msg.height = 1
        cloud_msg.is_bigendian = False
        cloud_msg.is_dense = True
        cloud_msg.data = arr.astype(np.float32).tobytes()
        self.pub_cloud.publish(cloud_msg)
        self.get_logger().info(f"published {stem}", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SampleReplay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[sample_replay] {e}", file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
