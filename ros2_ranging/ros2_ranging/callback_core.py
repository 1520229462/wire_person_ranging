"""rclpy-free core of the ROS2 ranging node.

Holds a `ranging_api.RangingSession` and processes (image, cloud,
T_lidar_to_cam) tuples. No ROS2 imports anywhere in this module, so it can
be imported, unit-tested and profiled on machines without ROS2 installed
(this repo's development server included).
"""
import os
import sys
import threading

# The ROS2 package lives INSIDE the wire_person_ranging repo
# (wire_person_ranging/ros2_ranging/), so the repo root is two levels up.
# WPR_REPO_ROOT overrides the layout (e.g. installed copy elsewhere).
REPO_ROOT = os.environ.get(
    "WPR_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import ranging_api  # noqa: E402


class RangingCore:
    """Thread-safe wrapper around RangingSession for ROS2 callbacks.

    All keyword arguments are passed through to
    `ranging_api.RangingSession` (identical names/defaults as the CLI).
    `set_categories` rebuilds the detector for new open-vocabulary texts
    (weights reload -- call it sparingly).
    """

    def __init__(self, **session_kwargs):
        self._lock = threading.Lock()
        self._kw = dict(session_kwargs)
        self._session = None
        self._ensure_session()

    def _ensure_session(self):
        if self._session is None:
            self._session = ranging_api.RangingSession(**self._kw)

    # ------------------------------------------------------------------ #
    def process(self, frame_id, image_rgb, points, T_lidar_to_cam,
                vggt_depth=None, out_dir=None, intrinsics=None):
        """Run ranging on one in-memory frame. Returns the result dict
        (adds ``error`` on failure); never raises for detection failures."""
        with self._lock:
            self._ensure_session()
            return self._session.process(
                frame_id, image_rgb, points, T_lidar_to_cam,
                vggt_depth=vggt_depth, out_dir=out_dir,
                intrinsics=intrinsics)

    def set_categories(self, categories):
        """Rebuild the open-vocab detector with new category texts."""
        with self._lock:
            old = self._session
            self._kw["categories"] = categories
            self._session = None
            self._ensure_session()
            if old is not None:
                old.close()

    def close(self):
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
