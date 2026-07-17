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

    actor_marker_config_path = (
        bridge_share
        / "config"
        / "actor_markers.yaml"
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

    map_server_config_path = (
        bridge_share
        / "config"
        / "map_server.yaml"
    )

    map_marker_config_path = (
        bridge_share
        / "config"
        / "map_markers.yaml"
    )

    ground_truth_server_config_path = (
        bridge_share
        / "config"
        / "ground_truth_trajectory_server.yaml"
    )

    navigation_state_config_path = (
        bridge_share
        / "config"
        / "navigation_state.yaml"
    )

    ground_truth_future_config_path = (
        bridge_share
        / "config"
        / "ground_truth_future.yaml"
    )

    executed_path_config_path = (
        bridge_share
        / "config"
        / "executed_path.yaml"
    )

    navigation_marker_config_path = (
        bridge_share
        / "config"
        / "navigation_markers.yaml"
    )


    robot_description = urdf_path.read_text(encoding="utf-8")

    rviz_config_path = os.path.join(vehicle_share, 'rviz', 'my_config.rviz')



    ground_truth_trajectory_server = Node(
        package="alpasim_bridge",
        executable="ground_truth_trajectory_server",
        name="ground_truth_trajectory_server",
        output="screen",
        parameters=[
            str(ground_truth_server_config_path),
        ],
    )

    navigation_state_publisher = Node(
        package="alpasim_bridge",
        executable="navigation_state_publisher",
        name="navigation_state_publisher",
        output="screen",
        parameters=[
            str(navigation_state_config_path),
            {
                "use_sim_time": True,
            },
        ],
    )

    ground_truth_future_publisher = Node(
        package="alpasim_bridge",
        executable="ground_truth_future_publisher",
        name="ground_truth_future_publisher",
        output="screen",
        parameters=[
            str(ground_truth_future_config_path),
            {
                "use_sim_time": True,
            },
        ],
    )

    executed_path_publisher = Node(
        package="alpasim_bridge",
        executable="executed_path_publisher",
        name="executed_path_publisher",
        output="screen",
        parameters=[
            str(executed_path_config_path),
            {
                "use_sim_time": True,
            },
        ],
    )

    navigation_marker_publisher = Node(
        package="alpasim_bridge",
        executable="navigation_marker_publisher",
        name="navigation_marker_publisher",
        output="screen",
        parameters=[
            str(navigation_marker_config_path),
            {
                "use_sim_time": True,
            },
        ],
    )


    
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

    map_server = Node(
        package="alpasim_bridge",
        executable="map_server",
        name="alpasim_map_server",
        output="screen",
        parameters=[
            str(map_server_config_path),
        ],
    )

    map_marker_publisher = Node(
        package="alpasim_bridge",
        executable="map_marker_publisher",
        name="alpasim_map_marker_publisher",
        output="screen",
        parameters=[
            str(map_marker_config_path),
            {
                "use_sim_time": True,
            },
        ],
    )

    actor_marker_publisher = Node(
        package="alpasim_bridge",
        executable="actor_marker_publisher",
        name="actor_marker_publisher",
        output="screen",
        parameters=[
            str(actor_marker_config_path),
            {
                "use_sim_time": True,
            },
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
            actor_marker_publisher,

            map_server,
            map_marker_publisher,

            ground_truth_trajectory_server,
            navigation_state_publisher,
            ground_truth_future_publisher,
            executed_path_publisher,
            navigation_marker_publisher,

            robot_state_publisher,
            rviz,
        ]
    )
