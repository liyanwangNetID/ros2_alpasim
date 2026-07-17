#!/usr/bin/env python3

"""Publish a rolling future slice of the recording GT ego trajectory."""

from __future__ import annotations

import rclpy
from builtin_interfaces.msg import Duration, Time
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock

from alpasim_msgs.msg import EgoTrajectory
from alpasim_msgs.srv import GetGroundTruthEgoTrajectory


def time_to_microseconds(value: Time) -> int:
    return (
        int(value.sec) * 1_000_000
        + int(value.nanosec) // 1000
    )


def duration_from_seconds(value: float) -> Duration:
    total_nanoseconds = int(round(value * 1_000_000_000.0))

    message = Duration()
    message.sec = int(total_nanoseconds // 1_000_000_000)
    message.nanosec = int(
        total_nanoseconds % 1_000_000_000
    )
    return message


class GroundTruthFuturePublisher(Node):
    """Request and publish the recording GT future trajectory."""

    def __init__(self) -> None:
        super().__init__(
            "ground_truth_future_publisher"
        )

        self.declare_parameter(
            "service_name",
            (
                "/alpasim/navigation/"
                "get_ground_truth_ego_trajectory"
            ),
        )
        self.declare_parameter(
            "output_topic",
            (
                "/alpasim/ground_truth/ego/"
                "future_trajectory"
            ),
        )
        self.declare_parameter(
            "clock_topic",
            "/clock",
        )
        self.declare_parameter(
            "future_duration_sec",
            6.4,
        )
        self.declare_parameter(
            "sampling_interval_sec",
            0.1,
        )
        self.declare_parameter(
            "max_points",
            65,
        )
        self.declare_parameter(
            "publish_rate_hz",
            2.0,
        )

        self.service_name = str(
            self.get_parameter("service_name").value
        )
        output_topic = str(
            self.get_parameter("output_topic").value
        )
        clock_topic = str(
            self.get_parameter("clock_topic").value
        )

        self.future_duration_sec = float(
            self.get_parameter(
                "future_duration_sec"
            ).value
        )
        self.sampling_interval_sec = float(
            self.get_parameter(
                "sampling_interval_sec"
            ).value
        )
        self.max_points = int(
            self.get_parameter("max_points").value
        )
        publish_rate_hz = float(
            self.get_parameter(
                "publish_rate_hz"
            ).value
        )

        if self.future_duration_sec < 0.0:
            raise ValueError(
                "future_duration_sec must be non-negative"
            )

        if self.sampling_interval_sec < 0.0:
            raise ValueError(
                "sampling_interval_sec must be non-negative"
            )

        if self.max_points <= 0:
            raise ValueError(
                "max_points must be positive"
            )

        if publish_rate_hz <= 0.0:
            raise ValueError(
                "publish_rate_hz must be positive"
            )

        trajectory_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        clock_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.publisher = self.create_publisher(
            EgoTrajectory,
            output_topic,
            trajectory_qos,
        )

        self.clock_subscription = (
            self.create_subscription(
                Clock,
                clock_topic,
                self.clock_callback,
                clock_qos,
            )
        )

        self.client = self.create_client(
            GetGroundTruthEgoTrajectory,
            self.service_name,
        )

        self.latest_clock: Time | None = None
        self.latest_clock_us: int | None = None
        self.request_in_progress = False
        self.waiting_log_emitted = False
        self.publish_count = 0

        self.timer = self.create_timer(
            1.0 / publish_rate_hz,
            self.request_future_trajectory,
        )

        self.get_logger().info(
            f"GT trajectory service: {self.service_name}"
        )
        self.get_logger().info(
            f"Publishing GT future to {output_topic}"
        )
        self.get_logger().info(
            "GT future configuration: "
            f"duration={self.future_duration_sec:.3f}s, "
            f"sampling={self.sampling_interval_sec:.3f}s, "
            f"max_points={self.max_points}, "
            f"rate={publish_rate_hz:.3f}Hz"
        )

    def clock_callback(self, message: Clock) -> None:
        timestamp_us = time_to_microseconds(
            message.clock
        )

        if (
            self.latest_clock_us is not None
            and timestamp_us < self.latest_clock_us
        ):
            self.get_logger().info(
                "Simulation clock moved backwards; "
                "detected a new rollout"
            )

        self.latest_clock = message.clock
        self.latest_clock_us = timestamp_us

    def request_future_trajectory(self) -> None:
        if self.request_in_progress:
            return

        if self.latest_clock is None:
            return

        if not self.client.service_is_ready():
            if not self.waiting_log_emitted:
                self.get_logger().info(
                    f"Waiting for service {self.service_name}"
                )
                self.waiting_log_emitted = True
            return

        self.waiting_log_emitted = False

        request = (
            GetGroundTruthEgoTrajectory.Request()
        )

        request.reference_stamp = self.latest_clock
        request.future_duration = duration_from_seconds(
            self.future_duration_sec
        )
        request.sampling_interval = (
            duration_from_seconds(
                self.sampling_interval_sec
            )
        )
        request.max_points = self.max_points

        # Every request asks for a different temporal window.
        # Therefore known_revision must remain zero.
        request.known_revision = 0

        self.request_in_progress = True

        future = self.client.call_async(request)
        future.add_done_callback(
            self.handle_service_response
        )

    def handle_service_response(self, future) -> None:
        self.request_in_progress = False

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"GT trajectory service call failed: {exc}"
            )
            return

        if not response.success:
            self.get_logger().debug(
                f"GT future unavailable: {response.message}"
            )
            return

        if response.not_modified:
            return

        if not response.trajectory.points:
            self.get_logger().debug(
                "GT service returned an empty trajectory"
            )
            return

        self.publisher.publish(
            response.trajectory
        )

        self.publish_count += 1

        if self.publish_count == 1:
            self.get_logger().info(
                "Published first GT future trajectory: "
                f"points={len(response.trajectory.points)}, "
                f"frame="
                f"{response.trajectory.pose_frame_id}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthFuturePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
