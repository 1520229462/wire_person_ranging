"""ROS2 integration of the wire_person_ranging ranging pipeline.

`ranging_node` subscribes image + point cloud + (optional) camera info and
publishes person/wire/person-wire distances; `sample_replay` replays the
bundled data/ sequence onto those topics for a demo. `callback_core` holds
the rclpy-free processing core (unit-testable without ROS2 installed).
"""
