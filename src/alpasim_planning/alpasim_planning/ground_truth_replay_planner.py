#!/usr/bin/env python3

"""Publish recorded ground-truth Ego future as an external planning trajectory.

Input
-----
Topic:
    /alpasim/ground_truth/ego/future_trajectory

Type:
    alpasim_msgs/msg/EgoTrajectory

Input frame:
    map

The input contains the recorded ground-truth Ego trajectory at approximately
10 Hz. The first point normally corresponds to time_from_reference = 0.

Output
------
Topic:
    /alpasim/planning/ego/trajectory

Type:
    alpasim_msgs/msg/EgoTrajectory

Output frame:
    base_link

The output trajectory uses the unified external-planner contract:

    waypoint frequency: 2 Hz
    waypoint interval:  0.5 s
    requested horizon:  3.0 s
    maximum points:     6

Processing
----------
For each ground-truth future trajectory:

1. Look up the transform from map to the current base_link at the trajectory
   reference timestamp.
2. Resample the recorded 10 Hz future trajectory to:
       0.5, 1.0, 1.5, 2.0, 2.5 and 3.0 seconds.
3. Transform each sampled GT pose from map into the actual current base_link.
4. Rotate velocity and acceleration vectors into base_link.
5. Publish the resulting EgoTrajectory to the External Driver.

Using the actual base_link transform means that small tracking errors do not
accumulate. The generated trajectory continually points from the actual
simulated Ego pose toward the recorded ground-truth future path.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from alpasim_msgs.msg import (
    EgoTrajectory,
    TrajectoryPoint,
)
from builtin_interfaces.msg import (
    Duration as DurationMsg,
    Time as TimeMsg,
)
from geometry_msgs.msg import Quaternion, Vector3
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)


def duration_to_nanoseconds(
    duration: DurationMsg,
) -> int:
    """Convert a ROS Duration message to integer nanoseconds."""

    return (
        int(duration.sec) * 1_000_000_000
        + int(duration.nanosec)
    )


def duration_from_seconds(
    seconds: float,
) -> DurationMsg:
    """Convert non-negative seconds to a ROS Duration message."""

    if seconds < 0.0:
        raise ValueError(
            "Duration must be non-negative"
        )

    total_nanoseconds = int(
        round(seconds * 1_000_000_000.0)
    )

    message = DurationMsg()
    message.sec = int(
        total_nanoseconds // 1_000_000_000
    )
    message.nanosec = int(
        total_nanoseconds % 1_000_000_000
    )

    return message


def add_seconds_to_time(
    reference: TimeMsg,
    seconds: float,
) -> TimeMsg:
    """Add non-negative seconds to a ROS Time message."""

    total_nanoseconds = (
        int(reference.sec) * 1_000_000_000
        + int(reference.nanosec)
        + int(round(seconds * 1_000_000_000.0))
    )

    message = TimeMsg()
    message.sec = int(
        total_nanoseconds // 1_000_000_000
    )
    message.nanosec = int(
        total_nanoseconds % 1_000_000_000
    )

    return message


def wrap_to_pi(
    angle: float,
) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def interpolate_angle(
    angle_a: float,
    angle_b: float,
    ratio: float,
) -> float:
    """Interpolate two angles using the shortest angular displacement."""

    delta = wrap_to_pi(
        angle_b - angle_a
    )

    return wrap_to_pi(
        angle_a + ratio * delta
    )


def quaternion_to_yaw(
    quaternion: Quaternion,
) -> float:
    """Extract planar yaw from a quaternion."""

    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)

    sin_yaw = 2.0 * (
        w * z + x * y
    )

    cos_yaw = 1.0 - 2.0 * (
        y * y + z * z
    )

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )


def quaternion_from_yaw(
    yaw: float,
) -> Quaternion:
    """Create a planar quaternion from yaw."""

    quaternion = Quaternion()

    half_yaw = 0.5 * yaw

    quaternion.x = 0.0
    quaternion.y = 0.0
    quaternion.z = math.sin(half_yaw)
    quaternion.w = math.cos(half_yaw)

    return quaternion


def interpolate_scalar(
    value_a: float,
    value_b: float,
    ratio: float,
) -> float:
    """Linearly interpolate two scalar values."""

    return (
        float(value_a)
        + ratio
        * (
            float(value_b)
            - float(value_a)
        )
    )


def interpolate_vector(
    vector_a,
    vector_b,
    ratio: float,
) -> np.ndarray:
    """Linearly interpolate two three-dimensional ROS vectors."""

    array_a = np.asarray(
        [
            float(vector_a.x),
            float(vector_a.y),
            float(vector_a.z),
        ],
        dtype=np.float64,
    )

    array_b = np.asarray(
        [
            float(vector_b.x),
            float(vector_b.y),
            float(vector_b.z),
        ],
        dtype=np.float64,
    )

    return array_a + ratio * (
        array_b - array_a
    )


class GroundTruthReplayPlanner(Node):
    """Replay the recorded Ego ground-truth future through External Driver."""

    def __init__(self) -> None:
        super().__init__(
            "ground_truth_replay_planner"
        )

        self.declare_parameter(
            "input_topic",
            (
                "/alpasim/ground_truth/ego/"
                "future_trajectory"
            ),
        )

        self.declare_parameter(
            "output_topic",
            "/alpasim/planning/ego/trajectory",
        )

        self.declare_parameter(
            "map_frame",
            "map",
        )

        self.declare_parameter(
            "base_frame",
            "base_link",
        )

        self.declare_parameter(
            "waypoint_frequency_hz",
            2.0,
        )

        self.declare_parameter(
            "trajectory_horizon_s",
            3.0,
        )

        self.declare_parameter(
            "tf_timeout_s",
            0.1,
        )

        input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )

        output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )

        self.map_frame = str(
            self.get_parameter(
                "map_frame"
            ).value
        )

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.waypoint_frequency_hz = float(
            self.get_parameter(
                "waypoint_frequency_hz"
            ).value
        )

        self.trajectory_horizon_s = float(
            self.get_parameter(
                "trajectory_horizon_s"
            ).value
        )

        self.tf_timeout_s = float(
            self.get_parameter(
                "tf_timeout_s"
            ).value
        )

        if self.waypoint_frequency_hz <= 0.0:
            raise ValueError(
                "waypoint_frequency_hz must be positive"
            )

        if self.trajectory_horizon_s <= 0.0:
            raise ValueError(
                "trajectory_horizon_s must be positive"
            )

        if self.tf_timeout_s <= 0.0:
            raise ValueError(
                "tf_timeout_s must be positive"
            )

        waypoint_count_float = (
            self.trajectory_horizon_s
            * self.waypoint_frequency_hz
        )

        self.maximum_waypoint_count = int(
            round(waypoint_count_float)
        )

        if self.maximum_waypoint_count <= 0:
            raise ValueError(
                "The configured horizon must contain at least one waypoint"
            )

        self.waypoint_interval_s = (
            1.0 / self.waypoint_frequency_hz
        )

        self.target_times_s = [
            index * self.waypoint_interval_s
            for index in range(
                1,
                self.maximum_waypoint_count + 1,
            )
        ]

        trajectory_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            EgoTrajectory,
            output_topic,
            trajectory_qos,
        )

        self.subscription = self.create_subscription(
            EgoTrajectory,
            input_topic,
            self.trajectory_callback,
            trajectory_qos,
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.message_count = 0
        self.publish_count = 0
        self.last_reference_timestamp_ns: Optional[
            int
        ] = None

        self.get_logger().info(
            "Ground-truth Replay Planner initialized: "
            f"input={input_topic}, "
            f"output={output_topic}, "
            f"frames={self.map_frame}->{self.base_frame}, "
            f"waypoint_frequency="
            f"{self.waypoint_frequency_hz:.2f} Hz, "
            f"horizon={self.trajectory_horizon_s:.2f} s, "
            f"maximum_points="
            f"{self.maximum_waypoint_count}"
        )

    def trajectory_callback(
        self,
        message: EgoTrajectory,
    ) -> None:
        """Transform and resample one ground-truth future trajectory."""

        self.message_count += 1

        if message.pose_frame_id != self.map_frame:
            self.get_logger().warning(
                "Ignoring ground-truth trajectory in unexpected "
                f"pose frame {message.pose_frame_id!r}; "
                f"expected {self.map_frame!r}"
            )
            return

        if len(message.points) < 2:
            self.get_logger().warning(
                "Ignoring ground-truth trajectory with fewer "
                "than two points"
            )
            return

        reference_timestamp_ns = (
            int(message.reference_stamp.sec)
            * 1_000_000_000
            + int(message.reference_stamp.nanosec)
        )

        if (
            self.last_reference_timestamp_ns is not None
            and reference_timestamp_ns
            < self.last_reference_timestamp_ns
        ):
            rewind_s = (
                self.last_reference_timestamp_ns
                - reference_timestamp_ns
            ) / 1_000_000_000.0

            self.get_logger().info(
                "Detected AlpaSim rollout timestamp reset: "
                f"rewind={rewind_s:.3f} s. "
                "Resetting Replay Planner state and TF buffer."
            )

            self.publish_count = 0

            # The same scene can restart with exactly the same simulation
            # timestamps. Clear dynamic TF history from the previous rollout;
            # otherwise tf2 rejects the new map -> base_link transforms as
            # TF_OLD_DATA until simulation time catches up with the old rollout.
            self.tf_buffer.clear()

        self.last_reference_timestamp_ns = (
            reference_timestamp_ns
        )

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.map_frame,
                Time.from_msg(
                    message.reference_stamp
                ),
                timeout=Duration(
                    seconds=self.tf_timeout_s
                ),
            )
        except TransformException as exc:
            self.get_logger().warning(
                "Could not transform ground-truth trajectory "
                f"from {self.map_frame!r} to "
                f"{self.base_frame!r} at reference time: {exc}"
            )
            return

        source_times_s = np.asarray(
            [
                duration_to_nanoseconds(
                    point.time_from_reference
                )
                / 1_000_000_000.0
                for point in message.points
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(source_times_s)
        ):
            self.get_logger().warning(
                "Ignoring GT trajectory containing non-finite times"
            )
            return

        if np.any(
            np.diff(source_times_s) <= 0.0
        ):
            self.get_logger().warning(
                "Ignoring GT trajectory with non-increasing "
                "time_from_reference values"
            )
            return

        maximum_source_time_s = float(
            source_times_s[-1]
        )

        available_target_times_s = [
            target_time_s
            for target_time_s in self.target_times_s
            if (
                target_time_s
                <= maximum_source_time_s + 1e-6
            )
        ]

        if not available_target_times_s:
            self.get_logger().warning(
                "Ground-truth future does not contain a "
                "0.5-second future waypoint"
            )
            return

        transform_translation = np.asarray(
            [
                float(
                    transform.transform.translation.x
                ),
                float(
                    transform.transform.translation.y
                ),
                float(
                    transform.transform.translation.z
                ),
            ],
            dtype=np.float64,
        )

        transform_yaw = quaternion_to_yaw(
            transform.transform.rotation
        )

        cos_yaw = math.cos(
            transform_yaw
        )

        sin_yaw = math.sin(
            transform_yaw
        )

        rotation_map_to_base = np.asarray(
            [
                [
                    cos_yaw,
                    -sin_yaw,
                    0.0,
                ],
                [
                    sin_yaw,
                    cos_yaw,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        output_points: list[
            TrajectoryPoint
        ] = []

        for target_time_s in available_target_times_s:
            interpolated = self.interpolate_gt_point(
                message=message,
                source_times_s=source_times_s,
                target_time_s=target_time_s,
            )

            position_map = interpolated[
                "position"
            ]

            position_base = (
                rotation_map_to_base
                @ position_map
                + transform_translation
            )

            velocity_base = (
                rotation_map_to_base
                @ interpolated[
                    "linear_velocity"
                ]
            )

            acceleration_base = (
                rotation_map_to_base
                @ interpolated[
                    "linear_acceleration"
                ]
            )

            yaw_base = wrap_to_pi(
                transform_yaw
                + interpolated["yaw"]
            )

            point = TrajectoryPoint()

            point.stamp = add_seconds_to_time(
                message.reference_stamp,
                target_time_s,
            )

            point.time_from_reference = (
                duration_from_seconds(
                    target_time_s
                )
            )

            point.pose.position.x = float(
                position_base[0]
            )
            point.pose.position.y = float(
                position_base[1]
            )
            point.pose.position.z = float(
                position_base[2]
            )

            point.pose.orientation = (
                quaternion_from_yaw(
                    yaw_base
                )
            )

            point.linear_velocity = Vector3(
                x=float(velocity_base[0]),
                y=float(velocity_base[1]),
                z=float(velocity_base[2]),
            )

            point.linear_acceleration = Vector3(
                x=float(acceleration_base[0]),
                y=float(acceleration_base[1]),
                z=float(acceleration_base[2]),
            )

            point.yaw = float(
                yaw_base
            )

            point.yaw_rate = float(
                interpolated["yaw_rate"]
            )

            point.yaw_acceleration = float(
                interpolated[
                    "yaw_acceleration"
                ]
            )

            point.speed = float(
                interpolated["speed"]
            )

            output_points.append(point)

        output = EgoTrajectory()

        output.reference_stamp = (
            message.reference_stamp
        )

        output.pose_frame_id = (
            self.base_frame
        )

        output.dynamics_frame_id = (
            self.base_frame
        )

        # Preserve the GT source classification while identifying
        # this ROS adapter through the producer field.
        output.source = message.source
        output.producer = (
            "ground_truth_replay_planner"
        )
        output.is_model_generated = False
        output.force_gt_active = (
            message.force_gt_active
        )

        output.requested_duration = (
            duration_from_seconds(
                self.trajectory_horizon_s
            )
        )

        actual_duration_s = float(
            available_target_times_s[-1]
        )

        output.actual_duration = (
            duration_from_seconds(
                actual_duration_s
            )
        )

        output.points = output_points

        self.publisher.publish(output)

        self.publish_count += 1

        if (
            self.publish_count == 1
            or self.publish_count % 50 == 0
        ):
            self.get_logger().info(
                "Published GT replay trajectory: "
                f"count={self.publish_count}, "
                f"points={len(output_points)}, "
                f"duration={actual_duration_s:.1f} s, "
                f"final_x="
                f"{output_points[-1].pose.position.x:.3f}, "
                f"final_y="
                f"{output_points[-1].pose.position.y:.3f}"
            )

    @staticmethod
    def interpolate_gt_point(
        message: EgoTrajectory,
        source_times_s: np.ndarray,
        target_time_s: float,
    ) -> dict[str, object]:
        """Interpolate the GT trajectory to one target future time."""

        upper_index = int(
            np.searchsorted(
                source_times_s,
                target_time_s,
                side="left",
            )
        )

        if upper_index <= 0:
            lower_index = 0
            upper_index = 1
        elif upper_index >= len(
            source_times_s
        ):
            upper_index = (
                len(source_times_s) - 1
            )
            lower_index = (
                upper_index - 1
            )
        else:
            lower_index = (
                upper_index - 1
            )

        lower_time_s = float(
            source_times_s[lower_index]
        )

        upper_time_s = float(
            source_times_s[upper_index]
        )

        if math.isclose(
            target_time_s,
            upper_time_s,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            ratio = 1.0
        elif math.isclose(
            upper_time_s,
            lower_time_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            ratio = 0.0
        else:
            ratio = (
                target_time_s
                - lower_time_s
            ) / (
                upper_time_s
                - lower_time_s
            )

        ratio = float(
            np.clip(
                ratio,
                0.0,
                1.0,
            )
        )

        lower = message.points[
            lower_index
        ]

        upper = message.points[
            upper_index
        ]

        lower_position = np.asarray(
            [
                float(
                    lower.pose.position.x
                ),
                float(
                    lower.pose.position.y
                ),
                float(
                    lower.pose.position.z
                ),
            ],
            dtype=np.float64,
        )

        upper_position = np.asarray(
            [
                float(
                    upper.pose.position.x
                ),
                float(
                    upper.pose.position.y
                ),
                float(
                    upper.pose.position.z
                ),
            ],
            dtype=np.float64,
        )

        position = (
            lower_position
            + ratio
            * (
                upper_position
                - lower_position
            )
        )

        lower_yaw = float(
            lower.yaw
        )

        upper_yaw = float(
            upper.yaw
        )

        if not math.isfinite(lower_yaw):
            lower_yaw = quaternion_to_yaw(
                lower.pose.orientation
            )

        if not math.isfinite(upper_yaw):
            upper_yaw = quaternion_to_yaw(
                upper.pose.orientation
            )

        return {
            "position": position,
            "yaw": interpolate_angle(
                lower_yaw,
                upper_yaw,
                ratio,
            ),
            "linear_velocity": interpolate_vector(
                lower.linear_velocity,
                upper.linear_velocity,
                ratio,
            ),
            "linear_acceleration": interpolate_vector(
                lower.linear_acceleration,
                upper.linear_acceleration,
                ratio,
            ),
            "yaw_rate": interpolate_scalar(
                lower.yaw_rate,
                upper.yaw_rate,
                ratio,
            ),
            "yaw_acceleration": interpolate_scalar(
                lower.yaw_acceleration,
                upper.yaw_acceleration,
                ratio,
            ),
            "speed": interpolate_scalar(
                lower.speed,
                upper.speed,
                ratio,
            ),
        }


def main(args=None) -> None:
    """Run the ground-truth replay planner."""

    rclpy.init(args=args)

    node = GroundTruthReplayPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()