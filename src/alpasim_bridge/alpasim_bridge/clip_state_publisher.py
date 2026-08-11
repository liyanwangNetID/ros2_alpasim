#!/usr/bin/env python3

"""Publish whether an AlpaSim clip is currently active.

Input
-----
Topic:
    /alpasim/ground_truth/ego/future_trajectory

Type:
    alpasim_msgs/msg/EgoTrajectory

Output
------
Topic:
    /alpasim/simulation/clip_active

Type:
    std_msgs/msg/Bool

Semantics
---------
    True:
        Ground-truth future messages are currently being received, so an
        AlpaSim clip is considered active.

    False:
        No clip has started, the previous clip has ended, or AlpaSim is
        waiting for the next clip.

The state is published continuously at a configurable wall-clock frequency.
Clip activity is determined using wall-clock monotonic time, not simulation
time. This makes the node robust when a new rollout restarts or rewinds the
simulation timestamp.
"""

from __future__ import annotations

import time

import rclpy
from alpasim_msgs.msg import EgoTrajectory
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool


class ClipStatePublisher(Node):
    """Publish the active/inactive state of the current AlpaSim clip."""

    def __init__(self) -> None:
        super().__init__(
            "clip_state_publisher"
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
            "/alpasim/simulation/clip_active",
        )

        self.declare_parameter(
            "publish_rate_hz",
            10.0,
        )

        self.declare_parameter(
            "clip_timeout_s",
            2.0,
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

        self.publish_rate_hz = float(
            self.get_parameter(
                "publish_rate_hz"
            ).value
        )

        self.clip_timeout_s = float(
            self.get_parameter(
                "clip_timeout_s"
            ).value
        )

        if self.publish_rate_hz <= 0.0:
            raise ValueError(
                "publish_rate_hz must be positive"
            )

        if self.clip_timeout_s <= 0.0:
            raise ValueError(
                "clip_timeout_s must be positive"
            )

        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Keep the latest clip state available for compatible late-joining
        # subscribers.
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            Bool,
            output_topic,
            output_qos,
        )

        self.subscription = self.create_subscription(
            EgoTrajectory,
            input_topic,
            self.trajectory_callback,
            input_qos,
        )

        self.last_message_monotonic_s: (
            float | None
        ) = None

        self.clip_active = False

        self.received_message_count = 0
        self.published_message_count = 0

        timer_period_s = (
            1.0 / self.publish_rate_hz
        )

        self.timer = self.create_timer(
            timer_period_s,
            self.publish_state,
        )

        self.get_logger().info(
            "Clip State Publisher initialized: "
            f"input={input_topic}, "
            f"output={output_topic}, "
            f"publish_rate={self.publish_rate_hz:.2f} Hz, "
            f"clip_timeout={self.clip_timeout_s:.2f} s"
        )

    def trajectory_callback(
        self,
        message: EgoTrajectory,
    ) -> None:
        """Record receipt of a current ground-truth trajectory."""

        del message

        self.last_message_monotonic_s = (
            time.monotonic()
        )

        self.received_message_count += 1

        if not self.clip_active:
            self.clip_active = True

            self.get_logger().info(
                "AlpaSim clip started: "
                "publishing clip_active=true"
            )

    def determine_clip_active(
        self,
    ) -> bool:
        """Determine clip state using elapsed wall-clock time."""

        if (
            self.last_message_monotonic_s
            is None
        ):
            return False

        message_age_s = (
            time.monotonic()
            - self.last_message_monotonic_s
        )

        return (
            message_age_s
            <= self.clip_timeout_s
        )

    def publish_state(self) -> None:
        """Publish the current clip state at the configured frequency."""

        active_now = (
            self.determine_clip_active()
        )

        if (
            self.clip_active
            and not active_now
        ):
            message_age_s = (
                time.monotonic()
                - self.last_message_monotonic_s
                if (
                    self.last_message_monotonic_s
                    is not None
                )
                else float("inf")
            )

            self.get_logger().info(
                "AlpaSim clip ended or became inactive: "
                f"no GT future message for "
                f"{message_age_s:.3f} s; "
                "publishing clip_active=false"
            )

        self.clip_active = active_now

        output = Bool()
        output.data = self.clip_active

        self.publisher.publish(output)

        self.published_message_count += 1


def main(args=None) -> None:
    """Run the Clip State Publisher."""

    rclpy.init(args=args)

    node = ClipStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()