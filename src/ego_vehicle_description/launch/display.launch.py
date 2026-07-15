import os

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(
        get_package_share_directory(
            "ego_vehicle_description"
        )
    )

    urdf_path = package_share / "urdf" / "ego_vehicle.urdf"
    robot_description = urdf_path.read_text(encoding="utf-8")

    rviz_config_path = os.path.join(package_share, 'rviz', 'my_config.rviz')

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="ego_vehicle_robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=['-d', rviz_config_path],
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
            }
        ],
    )

    return LaunchDescription(
        [
            robot_state_publisher,
            rviz,
        ]
    )
