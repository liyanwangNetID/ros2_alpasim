import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    vehicle_share = Path(
        get_package_share_directory(
            "ego_vehicle_description"
        )
    )

    bridge_share = Path(
        get_package_share_directory(
            "alpasim_bridge"
        )
    )

    urdf_path = (
        vehicle_share
        / "urdf"
        / "ego_vehicle.urdf"
    )

    actor_config_path = (
        bridge_share
        / "config"
        / "actor_export.yaml"
    )


    robot_description = urdf_path.read_text(encoding="utf-8")

    rviz_config_path = os.path.join(vehicle_share, 'rviz', 'my_config.rviz')

    
    # Ego state, camera images, calibration, clock and ego TF.
    ego_state_publisher = Node(
        package="alpasim_bridge",
        executable="ego_state_publisher",
        name="alpasim_sensor_publisher",
        output="screen",
    )

    # Current actors, history, ground-truth future and prediction placeholder.
    actor_state_publisher = Node(
        package="alpasim_bridge",
        executable="actor_state_publisher",
        name="actor_state_publisher",
        output="screen",
        parameters=[
            str(actor_config_path),
        ],
    )

    # Fixed transforms in the ego vehicle URDF.
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
            ego_state_publisher,
            actor_state_publisher,
            robot_state_publisher,
            rviz,
        ]
    )
