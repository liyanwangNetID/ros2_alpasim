#!/usr/bin/env python3
"""Publish subdued grey-white RViz columns for static occupancy."""
from __future__ import annotations

import math

import rclpy
from alpasim_msgs.msg import StaticOccupancy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import Marker, MarkerArray

# Quantization keeps Marker count small while retaining a readable 3D shape.
HEIGHT_BINS_M = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
DISPLAY_HEIGHTS_M = (0.18, 0.38, 0.75, 1.5, 3.0, 6.0, 12.0, 20.0)


def height_bin(height: float) -> int:
    for index, upper in enumerate(HEIGHT_BINS_M):
        if height <= upper:
            return index
    return len(HEIGHT_BINS_M) - 1


def build_markers(
    occupancy: StaticOccupancy,
    color: tuple[float, float, float, float],
    min_display_height_m: float = 0.30,
) -> MarkerArray:
    points: list[list[Point]] = [[] for _ in HEIGHT_BINS_M]
    for grid_i, grid_j, min_z, max_z in zip(
        occupancy.grid_i,
        occupancy.grid_j,
        occupancy.min_z,
        occupancy.max_z,
        strict=True,
    ):
        height = max(0.001, float(max_z) - float(min_z))
        if height < min_display_height_m:
            continue
        point = Point()
        point.x = (float(grid_i) + 0.5) * float(occupancy.grid_size_xy_m)
        point.y = (float(grid_j) + 0.5) * float(occupancy.grid_size_xy_m)
        point.z = 0.5 * (float(min_z) + float(max_z))
        points[height_bin(height)].append(point)
    output = MarkerArray()
    delete_all = Marker()
    delete_all.header.frame_id = occupancy.frame_id
    delete_all.action = Marker.DELETEALL
    output.markers.append(delete_all)
    for index, bucket in enumerate(points):
        if not bucket:
            continue
        marker = Marker()
        marker.header.frame_id = occupancy.frame_id
        marker.ns = "static_occupancy/columns"
        marker.id = index
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(occupancy.grid_size_xy_m) * 0.96
        marker.scale.y = float(occupancy.grid_size_xy_m) * 0.96
        marker.scale.z = DISPLAY_HEIGHTS_M[index]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.points = bucket
        output.markers.append(marker)
    return output


class StaticOccupancyMarkerPublisher(Node):
    """Convert static occupancy to a lightweight, neutral RViz visualization."""

    def __init__(self) -> None:
        super().__init__("static_occupancy_marker_publisher")
        self.declare_parameter("input_topic", "/alpasim/static_occupancy")
        self.declare_parameter("marker_topic", "/alpasim/static_occupancy/markers")
        self.declare_parameter("color_r", 0.82)
        self.declare_parameter("color_g", 0.84)
        self.declare_parameter("color_b", 0.86)
        self.declare_parameter("alpha", 0.38)
        self.declare_parameter("min_display_height_m", 0.30)
        input_topic = str(self.get_parameter("input_topic").value)
        marker_topic = str(self.get_parameter("marker_topic").value)
        self.min_display_height_m = float(
            self.get_parameter("min_display_height_m").value
        )
        self.color = (
            float(self.get_parameter("color_r").value),
            float(self.get_parameter("color_g").value),
            float(self.get_parameter("color_b").value),
            float(self.get_parameter("alpha").value),
        )
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(MarkerArray, marker_topic, qos)
        self.subscription = self.create_subscription(
            StaticOccupancy, input_topic, self.callback, qos
        )
        self.get_logger().info(
            f"Static occupancy markers: {input_topic} -> {marker_topic}; "
            f"grey-white alpha={self.color[3]:.2f}"
        )

    def callback(self, occupancy: StaticOccupancy) -> None:
        markers = build_markers(
            occupancy,
            self.color,
            self.min_display_height_m,
        )
        self.publisher.publish(markers)
        self.get_logger().info(
            f"Published static occupancy visualization: scene={occupancy.scene_id}, "
            f"columns={len(occupancy.grid_i)}, markers={len(markers.markers) - 1}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticOccupancyMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
