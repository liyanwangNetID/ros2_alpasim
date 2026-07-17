#!/usr/bin/env python3

"""Publish RViz markers for Route and ego trajectories."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import Marker, MarkerArray

from alpasim_msgs.msg import (
    EgoTrajectory,
    Route,
)


class NavigationMarkerPublisher(Node):
    """Visualize Route, GT future, model plan and executed path."""

    ROUTE_MARKER_ID = 1
    GT_FUTURE_MARKER_ID = 2
    MODEL_PLAN_MARKER_ID = 3
    EXECUTED_PATH_MARKER_ID = 4

    def __init__(self) -> None:
        super().__init__(
            "navigation_marker_publisher"
        )

        self.declare_parameter(
            "route_topic",
            "/alpasim/route/map",
        )
        self.declare_parameter(
            "ground_truth_topic",
            (
                "/alpasim/ground_truth/ego/"
                "future_trajectory"
            ),
        )
        self.declare_parameter(
            "model_plan_topic",
            "/alpasim/planning/ego/trajectory",
        )
        self.declare_parameter(
            "executed_path_topic",
            "/alpasim/ego/executed_path",
        )
        self.declare_parameter(
            "marker_topic",
            "/alpasim/navigation/markers",
        )

        self.declare_parameter(
            "route_width",
            0.28,
        )
        self.declare_parameter(
            "ground_truth_width",
            0.20,
        )
        self.declare_parameter(
            "model_plan_width",
            0.26,
        )
        self.declare_parameter(
            "executed_path_width",
            0.12,
        )

        self.declare_parameter(
            "route_z_offset",
            0.16,
        )
        self.declare_parameter(
            "ground_truth_z_offset",
            0.20,
        )
        self.declare_parameter(
            "model_plan_z_offset",
            0.24,
        )
        self.declare_parameter(
            "executed_path_z_offset",
            0.12,
        )

        route_topic = str(
            self.get_parameter("route_topic").value
        )
        ground_truth_topic = str(
            self.get_parameter(
                "ground_truth_topic"
            ).value
        )
        model_plan_topic = str(
            self.get_parameter(
                "model_plan_topic"
            ).value
        )
        executed_path_topic = str(
            self.get_parameter(
                "executed_path_topic"
            ).value
        )
        marker_topic = str(
            self.get_parameter(
                "marker_topic"
            ).value
        )

        self.route_width = float(
            self.get_parameter("route_width").value
        )
        self.ground_truth_width = float(
            self.get_parameter(
                "ground_truth_width"
            ).value
        )
        self.model_plan_width = float(
            self.get_parameter(
                "model_plan_width"
            ).value
        )
        self.executed_path_width = float(
            self.get_parameter(
                "executed_path_width"
            ).value
        )

        self.route_z_offset = float(
            self.get_parameter(
                "route_z_offset"
            ).value
        )
        self.ground_truth_z_offset = float(
            self.get_parameter(
                "ground_truth_z_offset"
            ).value
        )
        self.model_plan_z_offset = float(
            self.get_parameter(
                "model_plan_z_offset"
            ).value
        )
        self.executed_path_z_offset = float(
            self.get_parameter(
                "executed_path_z_offset"
            ).value
        )

        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.publisher = self.create_publisher(
            MarkerArray,
            marker_topic,
            marker_qos,
        )

        self.route_subscription = self.create_subscription(
            Route,
            route_topic,
            self.route_callback,
            input_qos,
        )

        self.ground_truth_subscription = (
            self.create_subscription(
                EgoTrajectory,
                ground_truth_topic,
                self.ground_truth_callback,
                input_qos,
            )
        )

        self.model_plan_subscription = (
            self.create_subscription(
                EgoTrajectory,
                model_plan_topic,
                self.model_plan_callback,
                input_qos,
            )
        )

        self.executed_path_subscription = (
            self.create_subscription(
                EgoTrajectory,
                executed_path_topic,
                self.executed_path_callback,
                input_qos,
            )
        )

        self.latest_markers: dict[int, Marker] = {}

        self.get_logger().info(
            f"Publishing navigation markers to {marker_topic}"
        )

    def copy_point(
        self,
        source: Point,
        z_offset: float,
    ) -> Point:
        point = Point()
        point.x = float(source.x)
        point.y = float(source.y)
        point.z = float(source.z) + z_offset
        return point

    def make_line_strip(
        self,
        marker_id: int,
        namespace: str,
        frame_id: str,
        points: list[Point],
        width: float,
        color: tuple[float, float, float, float],
        z_offset: float,
    ) -> Marker:
        marker = Marker()

        marker.header.frame_id = frame_id
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0

        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = width

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        marker.points = [
            self.copy_point(point, z_offset)
            for point in points
        ]

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        return marker

    def publish_current_markers(self) -> None:
        marker_array = MarkerArray()
        marker_array.markers = list(
            self.latest_markers.values()
        )
        self.publisher.publish(marker_array)

    def remove_marker(self, marker_id: int) -> None:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "navigation"
        marker.id = marker_id
        marker.action = Marker.DELETE

        self.latest_markers[marker_id] = marker
        self.publish_current_markers()
        self.latest_markers.pop(marker_id, None)

    def route_callback(self, message: Route) -> None:
        points = [
            point.position
            for point in message.points
            if point.valid
        ]

        if len(points) < 2:
            self.remove_marker(
                self.ROUTE_MARKER_ID
            )
            return

        marker = self.make_line_strip(
            marker_id=self.ROUTE_MARKER_ID,
            namespace="navigation/route",
            frame_id=message.frame_id,
            points=points,
            width=self.route_width,
            color=(1.00, 0.10, 0.62, 1.00),
            z_offset=self.route_z_offset,
        )

        self.latest_markers[
            self.ROUTE_MARKER_ID
        ] = marker

        self.publish_current_markers()

    def ground_truth_callback(
        self,
        message: EgoTrajectory,
    ) -> None:
        points = [
            point.pose.position
            for point in message.points
        ]

        if len(points) < 2:
            self.remove_marker(
                self.GT_FUTURE_MARKER_ID
            )
            return

        marker = self.make_line_strip(
            marker_id=self.GT_FUTURE_MARKER_ID,
            namespace="navigation/ground_truth_future",
            frame_id=message.pose_frame_id,
            points=points,
            width=self.ground_truth_width,
            color=(0.10, 1.00, 0.20, 1.00),
            z_offset=self.ground_truth_z_offset,
        )

        self.latest_markers[
            self.GT_FUTURE_MARKER_ID
        ] = marker

        self.publish_current_markers()

    def model_plan_callback(
        self,
        message: EgoTrajectory,
    ) -> None:
        points = [
            point.pose.position
            for point in message.points
        ]

        if len(points) < 2:
            self.remove_marker(
                self.MODEL_PLAN_MARKER_ID
            )
            return

        marker = self.make_line_strip(
            marker_id=self.MODEL_PLAN_MARKER_ID,
            namespace="navigation/model_plan",
            frame_id=message.pose_frame_id,
            points=points,
            width=self.model_plan_width,
            color=(1.00, 0.80, 0.00, 1.00),
            z_offset=self.model_plan_z_offset,
        )

        self.latest_markers[
            self.MODEL_PLAN_MARKER_ID
        ] = marker

        self.publish_current_markers()

    def executed_path_callback(
        self,
        message: EgoTrajectory,
    ) -> None:
        points = [
            point.pose.position
            for point in message.points
        ]

        if len(points) < 2:
            return

        marker = self.make_line_strip(
            marker_id=self.EXECUTED_PATH_MARKER_ID,
            namespace="navigation/executed_path",
            frame_id=message.pose_frame_id,
            points=points,
            width=self.executed_path_width,
            color=(1.00, 1.00, 1.00, 0.95),
            z_offset=self.executed_path_z_offset,
        )

        self.latest_markers[
            self.EXECUTED_PATH_MARKER_ID
        ] = marker

        self.publish_current_markers()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationMarkerPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
