"""Launch the ranging node alone (your own image + cloud publishers feed it).

  ros2 launch ros2_ranging ranging.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ros2_ranging",
            executable="ranging_node",
            name="ranging_node",
            output="screen",
            parameters=["config/ranging_params.yaml"],
        ),
    ])
