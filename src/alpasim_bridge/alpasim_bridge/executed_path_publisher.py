#!/usr/bin/env python3

"""Build and publish the physics-corrected executed ego path."""

from __future__ import annotations

from collections import deque

import math

import rclpy
from builtin_interfaces.msg import Duration, Time
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from alpasim_msgs.msg import (
    EgoState,
    EgoTrajectory,
    TrajectoryPoint,
)

def yaw_from_quaternion(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    """Extract ROS-convention yaw from a quaternion."""
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )

def time_to_microseconds(value: Time) -> int:
    return (
        int(value.sec) * 1_000_000
        + int(value.nanosec) // 1000
    )


def duration_from_microseconds(
    value: int,
) -> Duration:
    result = Duration()

    seconds = value // 1_000_000
    remaining_us = value - seconds * 1_000_000

    result.sec = int(seconds)
    result.nanosec = int(remaining_us * 1000)

    return result


class ExecutedPathPublisher(Node):
    """Accumulate EgoState messages into an executed ego trajectory."""

    def __init__(self) -> None:
        super().__init__(
            "executed_path_publisher"
        )

        self.declare_parameter(
            "input_topic",
            "/alpasim/ego_state",
        )
        self.declare_parameter(
            "output_topic",
            "/alpasim/ego/executed_path",
        )
        self.declare_parameter(
            "history_duration_sec",
            0.0,
        )
        self.declare_parameter(
            "sampling_interval_sec",
            0.1,
        )
        self.declare_parameter(
            "maximum_points",
            512,
        )
        self.declare_parameter(
            "publish_rate_hz",
            5.0,
        )

        input_topic = str(
            self.get_parameter("input_topic").value
        )
        output_topic = str(
            self.get_parameter("output_topic").value
        )

        self.history_duration_sec = float(
            self.get_parameter(
                "history_duration_sec"
            ).value
        )
        self.sampling_interval_sec = float(
            self.get_parameter(
                "sampling_interval_sec"
            ).value
        )
        self.maximum_points = int(
            self.get_parameter(
                "maximum_points"
            ).value
        )
        publish_rate_hz = float(
            self.get_parameter(
                "publish_rate_hz"
            ).value
        )

        if self.history_duration_sec < 0.0:
            raise ValueError(
                "history_duration_sec must be non-negative"
            )

        if self.sampling_interval_sec < 0.0:
            raise ValueError(
                "sampling_interval_sec must be non-negative"
            )

        if self.maximum_points <= 0:
            raise ValueError(
                "maximum_points must be positive"
            )

        if publish_rate_hz <= 0.0:
            raise ValueError(
                "publish_rate_hz must be positive"
            )

        self.history_duration_us = int(
            round(
                self.history_duration_sec
                * 1_000_000.0
            )
        )
        self.sampling_interval_us = int(
            round(
                self.sampling_interval_sec
                * 1_000_000.0
            )
        )

        ego_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        trajectory_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.publisher = self.create_publisher(
            EgoTrajectory,
            output_topic,
            trajectory_qos,
        )

        self.subscription = self.create_subscription(
            EgoState,
            input_topic,
            self.ego_state_callback,
            ego_qos,
        )

        self.samples: deque[
            tuple[int, TrajectoryPoint]
        ] = deque()

        self.last_received_timestamp_us: int | None = None
        self.last_stored_timestamp_us: int | None = None
        self.latest_timestamp_us: int | None = None

        self.pose_frame_id: str | None = None
        self.dynamics_frame_id: str | None = None
        self.child_frame_id: str | None = None

        self.timer = self.create_timer(
            1.0 / publish_rate_hz,
            self.publish_executed_path,
        )

        self.publish_count = 0

        self.get_logger().info(
            f"Executed-path input: {input_topic}"
        )
        self.get_logger().info(
            f"Executed-path output: {output_topic}"
        )

    def make_trajectory_point(
        self,
        message: EgoState,
    ) -> TrajectoryPoint:
        """Convert one EgoState message into a trajectory sample."""
        point = TrajectoryPoint()

        point.stamp = message.stamp

        # EgoState stores position and orientation separately,
        # while TrajectoryPoint uses geometry_msgs/Pose.
        point.pose.position.x = float(
            message.position.x
        )
        point.pose.position.y = float(
            message.position.y
        )
        point.pose.position.z = float(
            message.position.z
        )

        point.pose.orientation.x = float(
            message.orientation.x
        )
        point.pose.orientation.y = float(
            message.orientation.y
        )
        point.pose.orientation.z = float(
            message.orientation.z
        )
        point.pose.orientation.w = float(
            message.orientation.w
        )

        point.linear_velocity.x = float(
            message.linear_velocity.x
        )
        point.linear_velocity.y = float(
            message.linear_velocity.y
        )
        point.linear_velocity.z = float(
            message.linear_velocity.z
        )

        point.linear_acceleration.x = float(
            message.linear_acceleration.x
        )
        point.linear_acceleration.y = float(
            message.linear_acceleration.y
        )
        point.linear_acceleration.z = float(
            message.linear_acceleration.z
        )

        point.yaw = yaw_from_quaternion(
            x=float(message.orientation.x),
            y=float(message.orientation.y),
            z=float(message.orientation.z),
            w=float(message.orientation.w),
        )

        # EgoState provides full angular vectors. For a ground vehicle,
        # yaw derivatives are the z-axis angular components.
        point.yaw_rate = float(
            message.angular_velocity.z
        )
        point.yaw_acceleration = float(
            message.angular_acceleration.z
        )

        point.speed = float(message.speed)

        return point

    def ego_state_callback(
        self,
        message: EgoState,
    ) -> None:
        timestamp_us = time_to_microseconds(
            message.stamp
        )

        incoming_pose_frame_id = str(
            message.pose_frame_id
        )
        incoming_dynamics_frame_id = str(
            message.dynamics_frame_id
        )
        incoming_child_frame_id = str(
            message.child_frame_id
        )

        
        if not incoming_pose_frame_id:
            self.get_logger().error(
                "Ignoring EgoState with an empty pose_frame_id"
            )
            return

        if not incoming_dynamics_frame_id:
            self.get_logger().error(
                "Ignoring EgoState with an empty dynamics_frame_id"
            )
            return

        if not incoming_child_frame_id:
            self.get_logger().error(
                "Ignoring EgoState with an empty child_frame_id"
            )
            return

        if (
            self.last_received_timestamp_us is not None
            and timestamp_us
            < self.last_received_timestamp_us
        ):
            self.get_logger().info(
                "Ego timestamp moved backwards; "
                "clearing executed-path history"
            )

            self.samples.clear()
            self.last_stored_timestamp_us = None

        frame_changed = (
            self.pose_frame_id is not None
            and (
                incoming_pose_frame_id
                != self.pose_frame_id
                or incoming_dynamics_frame_id
                != self.dynamics_frame_id
                or incoming_child_frame_id
                != self.child_frame_id
            )
        )

        if frame_changed:
            self.get_logger().warning(
                "EgoState frame metadata changed; "
                "clearing executed-path history"
            )

            self.samples.clear()
            self.last_stored_timestamp_us = None

        self.pose_frame_id = incoming_pose_frame_id
        self.dynamics_frame_id = (
            incoming_dynamics_frame_id
        )
        self.child_frame_id = incoming_child_frame_id

        self.last_received_timestamp_us = timestamp_us
        self.latest_timestamp_us = timestamp_us


        if (
            self.last_stored_timestamp_us is not None
            and self.sampling_interval_us > 0
            and timestamp_us
            - self.last_stored_timestamp_us
            < self.sampling_interval_us
        ):
            return

        point = self.make_trajectory_point(
            message
        )

        self.samples.append(
            (timestamp_us, point)
        )
        self.last_stored_timestamp_us = timestamp_us

        self.trim_history()

    def trim_history(self) -> None:
        if self.latest_timestamp_us is None:
            return

        if self.history_duration_us > 0:
            earliest_allowed_us = (
                self.latest_timestamp_us
                - self.history_duration_us
            )

            while (
                self.samples
                and self.samples[0][0]
                < earliest_allowed_us
            ):
                self.samples.popleft()

        while len(self.samples) > self.maximum_points:
            self.samples.popleft()

    def publish_executed_path(self) -> None:
        if not self.samples:
            return

        self.trim_history()

        reference_timestamp_us = self.samples[0][0]
        final_timestamp_us = self.samples[-1][0]

        message = EgoTrajectory()

        message.reference_stamp = self.samples[0][1].stamp
        if (
            self.pose_frame_id is None
            or self.dynamics_frame_id is None
        ):
            return

        message.pose_frame_id = self.pose_frame_id
        message.dynamics_frame_id = (
            self.dynamics_frame_id
        )

        message.source = (
            EgoTrajectory.SOURCE_EXECUTED
        )
        message.producer = (
            "alpasim_physics_corrected_ego"
        )
        message.is_model_generated = False
        message.force_gt_active = False

        duration_us = max(
            0,
            final_timestamp_us
            - reference_timestamp_us,
        )

        message.requested_duration = (
            duration_from_microseconds(
                duration_us
            )
        )
        message.actual_duration = (
            duration_from_microseconds(
                duration_us
            )
        )

        message.points = []

        for timestamp_us, stored_point in self.samples:
            point = TrajectoryPoint()
            point.stamp = stored_point.stamp
            point.pose = stored_point.pose
            point.linear_velocity = (
                stored_point.linear_velocity
            )
            point.linear_acceleration = (
                stored_point.linear_acceleration
            )
            point.yaw = stored_point.yaw
            point.yaw_rate = stored_point.yaw_rate
            point.yaw_acceleration = (
                stored_point.yaw_acceleration
            )
            point.speed = stored_point.speed

            point.time_from_reference = (
                duration_from_microseconds(
                    timestamp_us
                    - reference_timestamp_us
                )
            )

            message.points.append(point)

        self.publisher.publish(message)

        self.publish_count += 1

        if self.publish_count == 1:
            self.get_logger().info(
                "Published first executed path: "
                f"points={len(message.points)}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExecutedPathPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
