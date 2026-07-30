#!/usr/bin/env python3

"""Publish a synthetic ego trajectory for closed-loop integration testing."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from alpasim_msgs.msg import EgoTrajectory, TrajectoryPoint


class SyntheticTrajectoryPlanner(Node):
    """Publish a fixed-curvature trajectory in the current base_link frame."""

    def __init__(self) -> None:
        super().__init__("synthetic_trajectory_planner")

        self.declare_parameter(
            "output_topic",
            "/alpasim/planning/ego/trajectory",
        )
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("horizon_sec", 4.0)
        self.declare_parameter("sample_period_sec", 0.5)
        self.declare_parameter("target_speed_mps", 5.0)
        # self.declare_parameter("curvature_per_meter", 0.025)
        self.declare_parameter("curvature_per_meter", 0.0)

        output_topic = str(
            self.get_parameter("output_topic").value
        )
        publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )
        self.horizon_sec = float(
            self.get_parameter("horizon_sec").value
        )
        self.sample_period_sec = float(
            self.get_parameter("sample_period_sec").value
        )
        self.target_speed_mps = float(
            self.get_parameter("target_speed_mps").value
        )
        self.curvature_per_meter = float(
            self.get_parameter("curvature_per_meter").value
        )

        if publish_rate_hz <= 0.0:
            raise ValueError(
                "publish_rate_hz must be positive"
            )

        if self.horizon_sec <= 0.0:
            raise ValueError(
                "horizon_sec must be positive"
            )

        if self.sample_period_sec <= 0.0:
            raise ValueError(
                "sample_period_sec must be positive"
            )

        if self.target_speed_mps < 0.0:
            raise ValueError(
                "target_speed_mps must be non-negative"
            )

        self.point_count = max(
            1,
            int(
                round(
                    self.horizon_sec
                    / self.sample_period_sec
                )
            ),
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            EgoTrajectory,
            output_topic,
            qos,
        )

        self.timer = self.create_timer(
            1.0 / publish_rate_hz,
            self.publish_trajectory,
        )

        self.publish_count = 0

        self.get_logger().info(
            "Synthetic trajectory planner initialized: "
            f"topic={output_topic}, "
            f"rate={publish_rate_hz:.2f} Hz, "
            f"horizon={self.horizon_sec:.2f} s, "
            f"sample_period={self.sample_period_sec:.2f} s, "
            f"points={self.point_count}, "
            f"speed={self.target_speed_mps:.2f} m/s, "
            f"curvature={self.curvature_per_meter:.4f} 1/m"
        )

    @staticmethod
    def duration_from_seconds(
        seconds: float,
    ):
        """Convert non-negative seconds to builtin_interfaces/Duration."""
        from builtin_interfaces.msg import Duration

        total_nanoseconds = int(
            round(seconds * 1_000_000_000.0)
        )

        message = Duration()
        message.sec = int(
            total_nanoseconds // 1_000_000_000
        )
        message.nanosec = int(
            total_nanoseconds % 1_000_000_000
        )

        return message

    def publish_trajectory(self) -> None:
        message = EgoTrajectory()

        message.reference_stamp = (
            self.get_clock().now().to_msg()
        )

        # Transitional integration contract:
        # points are relative to the current ego rig frame.
        message.pose_frame_id = "base_link"
        message.dynamics_frame_id = "base_link"

        message.source = (
            EgoTrajectory.SOURCE_MODEL_PLANNING
        )
        message.producer = (
            "synthetic_trajectory_planner"
        )
        message.is_model_generated = False
        message.force_gt_active = False

        message.requested_duration = (
            self.duration_from_seconds(
                self.horizon_sec
            )
        )
        message.actual_duration = (
            self.duration_from_seconds(
                self.horizon_sec
            )
        )

        points: list[TrajectoryPoint] = []

        for index in range(
            1,
            self.point_count + 1,
        ):
            time_seconds = (
                index * self.sample_period_sec
            )
            arc_length = (
                self.target_speed_mps
                * time_seconds
            )
            heading = (
                self.curvature_per_meter
                * arc_length
            )

            if (
                abs(self.curvature_per_meter)
                < 1.0e-9
            ):
                x = arc_length
                y = 0.0
            else:
                radius = (
                    1.0
                    / self.curvature_per_meter
                )

                x = radius * math.sin(heading)
                y = radius * (
                    1.0 - math.cos(heading)
                )

            point = TrajectoryPoint()

            point.time_from_reference = (
                self.duration_from_seconds(
                    time_seconds
                )
            )

            point.pose.position.x = float(x)
            point.pose.position.y = float(y)
            point.pose.position.z = 0.0

            point.pose.orientation.x = 0.0
            point.pose.orientation.y = 0.0
            point.pose.orientation.z = math.sin(
                heading / 2.0
            )
            point.pose.orientation.w = math.cos(
                heading / 2.0
            )

            point.linear_velocity.x = (
                self.target_speed_mps
                * math.cos(heading)
            )
            point.linear_velocity.y = (
                self.target_speed_mps
                * math.sin(heading)
            )
            point.linear_velocity.z = 0.0

            point.linear_acceleration.x = 0.0
            point.linear_acceleration.y = 0.0
            point.linear_acceleration.z = 0.0

            point.yaw = float(heading)
            point.yaw_rate = float(
                self.target_speed_mps
                * self.curvature_per_meter
            )
            point.yaw_acceleration = 0.0
            point.speed = self.target_speed_mps

            points.append(point)

        message.points = points
        self.publisher.publish(message)

        self.publish_count += 1

        if self.publish_count == 1:
            final_point = points[-1]

            self.get_logger().info(
                "Published first synthetic trajectory: "
                f"points={len(points)}, "
                f"final_x="
                f"{final_point.pose.position.x:.3f}, "
                f"final_y="
                f"{final_point.pose.position.y:.3f}, "
                f"final_yaw={final_point.yaw:.3f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticTrajectoryPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
