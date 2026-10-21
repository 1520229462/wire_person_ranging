"""ROS2 node: person + wire ranging from image + point cloud + TF extrinsics.

Only standard messages are used (no custom interfaces):

  Subscribes (topic names via params):
    image       sensor_msgs/Image          undistorted RGB (rgb8/bgr8/mono8)
    cloud       sensor_msgs/PointCloud2    lidar points (x/y/z fields), in
                                           frame `frame_id_lidar`
    camera_info sensor_msgs/CameraInfo     optional intrinsics override
  Publishes:
    ~/person_distance        std_msgs/Float32   ||person_center||
    ~/wire_distance          std_msgs/Float32   min ||wire_curve||
    ~/person_wire_distance   std_msgs/Float32   min person <-> wire
    ~/result                 std_msgs/String    full JSON result dict
  Service:
    ~/range_once             std_srvs/Trigger   single-shot on the latest pair

Extrinsics come from TF2 per frame:
    lookup_transform(frame_id_lidar, frame_id_camera, stamp)
i.e. publish the camera frame as a CHILD of the lidar frame (the
data/extrinsics/*.txt files store the inverse, T_cam_to_lidar; see
sample_replay.py).

The pipeline runs synchronously in the image/cloud callback. Model inference
takes on the order of a second per frame, so drive the input at a modest rate
(sample_replay defaults to 1 Hz); incoming pairs while busy are dropped by
the ApproximateTimeSynchronizer's queue.
"""
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
import message_filters
import tf2_ros
from cv_bridge import CvBridge

from .callback_core import RangingCore
from .converters import (camera_info_to_intrinsics, cloud2_to_points,
                         image_to_rgb, tf_transform_to_matrix)


class RangingNode(Node):

    def __init__(self):
        super().__init__("ranging_node")

        # --- parameters (mirror the CLI flags) --------------------------- #
        self.declare_parameter("categories", "person")
        self.declare_parameter("wire_mode", "per_frame")
        self.declare_parameter("wire_curve", "")
        self.declare_parameter("intrinsics", "")      # 'fx,fy,cx,cy' or ''
        self.declare_parameter("out_dir", "")          # per-frame viz dump
        self.declare_parameter("vggt_depth_dir", "")   # optional depth cache
        self.declare_parameter("sam2", True)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("frame_id_lidar", "lidar")
        self.declare_parameter("frame_id_camera", "camera")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("cloud_topic", "/lidar/points")
        self.declare_parameter("camera_info_topic", "/camera/info")
        self.declare_parameter("sync_slop", 0.1)
        self.declare_parameter("publish_on_error", True)

        self.frame_id_lidar = self.get_parameter("frame_id_lidar").value
        self.frame_id_camera = self.get_parameter("frame_id_camera").value
        self.publish_on_error = self.get_parameter("publish_on_error").value
        self.bridge = CvBridge()

        # --- TF2 for per-frame extrinsics -------------------------------- #
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- the processing core (models load here, ~2-3 min) ------------ #
        wire_curve = self.get_parameter("wire_curve").value or None
        out_dir = self.get_parameter("out_dir").value or None
        vggt_dir = self.get_parameter("vggt_depth_dir").value or None
        self.get_logger().info("building RangingSession (loading models)...")
        self.core = RangingCore(
            categories=self.get_parameter("categories").value,
            wire_mode=self.get_parameter("wire_mode").value,
            wire_curve=wire_curve,
            intrinsics=self.get_parameter("intrinsics").value or None,
            out_dir=out_dir,
            vggt_depth_dir=vggt_dir,
            sam2=self.get_parameter("sam2").value,
            device=self.get_parameter("device").value)
        self.get_logger().info("models loaded")

        # --- publishers / service ---------------------------------------- #
        self.pub_person = self.create_publisher(Float32, "~/person_distance", 10)
        self.pub_wire = self.create_publisher(Float32, "~/wire_distance", 10)
        self.pub_pw = self.create_publisher(
            Float32, "~/person_wire_distance", 10)
        self.pub_result = self.create_publisher(String, "~/result", 10)
        self.create_service(Trigger, "~/range_once", self._on_range_once)

        # --- subscriptions ------------------------------------------------ #
        self.latest_image = None
        self.latest_cloud = None
        self.camera_info = None    # optional intrinsics override
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._on_camera_info, 10)
        img_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("image_topic").value)
        cloud_sub = message_filters.Subscriber(
            self, PointCloud2, self.get_parameter("cloud_topic").value)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, cloud_sub], queue_size=4,
            slop=self.get_parameter("sync_slop").value)
        self.sync.registerCallback(self._on_pair)
        self.get_logger().info("ranging_node ready")

    # ------------------------------------------------------------------ #
    def _on_camera_info(self, msg):
        self.camera_info = camera_info_to_intrinsics(msg)

    def _on_pair(self, img_msg, cloud_msg):
        self.latest_image = img_msg
        self.latest_cloud = cloud_msg
        self._run(img_msg, cloud_msg)

    def _on_range_once(self, request, response):
        if self.latest_image is None:
            response.success = False
            response.message = "no synchronized image+cloud pair yet"
            return response
        try:
            result = self._run(self.latest_image, self.latest_cloud)
            response.success = "error" not in result
            response.message = result.get("error", "ok")
        except Exception as e:  # noqa: BLE001 - report to the caller
            response.success = False
            response.message = str(e)
        return response

    # ------------------------------------------------------------------ #
    def _run(self, img_msg, cloud_msg):
        """Convert messages -> run the core -> publish. Returns the result
        dict (never raises for detection failures)."""
        frame_id = f"{img_msg.header.stamp.sec}.{img_msg.header.stamp.nanosec}"
        try:
            image_rgb = image_to_rgb(self.bridge, img_msg)
            points = cloud2_to_points(cloud_msg)
            # extrinsics per frame: camera frame expressed in the lidar frame
            tf = self.tf_buffer.lookup_transform(
                self.frame_id_lidar, self.frame_id_camera,
                img_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=1.0))
            T_lidar_to_cam = tf_transform_to_matrix(tf)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"{frame_id}: input conversion failed: {e}")
            return {"frame": frame_id, "error": f"input:{e}"}

        # a received CameraInfo overrides the node's intrinsics parameter
        result = self.core.process(
            frame_id, image_rgb, points, T_lidar_to_cam,
            intrinsics=self.camera_info)
        return self._publish(frame_id, result)

    def _publish(self, frame_id, result):
        if "error" in result:
            self.get_logger().warn(f"{frame_id}: {result['error']}")
            if self.publish_on_error:
                self.pub_result.publish(String(data=json.dumps(result)))
            return result
        p = result["person"]
        self.pub_person.publish(Float32(data=float(p["distance_m"])))
        self.pub_wire.publish(
            Float32(data=float(result["wire"]["distance_m"])))
        self.pub_pw.publish(
            Float32(data=float(result["wire_person_min_distance_m"])))
        self.pub_result.publish(String(data=json.dumps(result)))
        self.get_logger().info(
            f"{frame_id}: person {p['distance_m']}m wire "
            f"{result['wire']['distance_m']}m person-wire "
            f"{result['wire_person_min_distance_m']}m")
        return result


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RangingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.core.close()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
