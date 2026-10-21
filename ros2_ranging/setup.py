import os

from setuptools import setup

package_name = "ros2_ranging"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"),
         ["config/ranging_params.yaml"]),
        (os.path.join("share", package_name, "launch"),
         ["launch/ranging.launch.py", "launch/sample_demo.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="wire_person_ranging",
    maintainer_email="noreply@example.com",
    description=("ROS2 node wrapping the wire_person_ranging open-vocabulary "
                 "3D ranging pipeline"),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ranging_node = ros2_ranging.ranging_node:main",
            "sample_replay = ros2_ranging.sample_replay:main",
        ],
    },
)
