"""Demo: replay the bundled data/ sequence into the ranging node.

  ros2 launch ros2_ranging sample_demo.launch.py

Results land on /ranging_node/person_distance etc. and are logged.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ros2_ranging",
            executable="sample_replay",
            name="sample_replay",
            output="screen",
            parameters=[{
                "rate": 1.0,
                "loop": False,
            }],
        ),
        Node(
            package="ros2_ranging",
            executable="ranging_node",
            name="ranging_node",
            output="screen",
            parameters=["config/ranging_params.yaml"],
        ),
    ])
