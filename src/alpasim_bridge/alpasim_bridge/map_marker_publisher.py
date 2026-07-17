#!/usr/bin/env python3

"""Request an AlpaSim VectorMap and publish RViz MarkerArray visualization."""

from __future__ import annotations

import zlib

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

from alpasim_msgs.msg import MapPolyline, VectorMap
from alpasim_msgs.srv import GetVectorMap


def stable_marker_id(
    namespace: str,
    source_id: str,
) -> int:
    """Generate deterministic positive int32 marker IDs."""
    value = zlib.crc32(
        f"{namespace}:{source_id}".encode("utf-8")
    )
    return int(value & 0x7FFFFFFF)


def copy_point(
    source: Point,
    z_offset: float = 0.0,
) -> Point:
    """Copy a geometry point while applying a visualization-only z offset."""
    point = Point()
    point.x = float(source.x)
    point.y = float(source.y)
    point.z = float(source.z) + z_offset
    return point


class MapMarkerPublisher(Node):
    """Publish RViz markers for the cached AlpaSim vector map."""

    def __init__(self) -> None:
        super().__init__(
            "alpasim_map_marker_publisher"
        )

        self.declare_parameter(
            "service_name",
            "/alpasim/map/get_vector_map",
        )
        self.declare_parameter(
            "marker_topic",
            "/alpasim/map/markers",
        )
        self.declare_parameter(
            "request_period_sec",
            1.0,
        )
        self.declare_parameter(
            "republish_period_sec",
            2.0,
        )

        self.declare_parameter(
            "show_lane_centerlines",
            True,
        )
        self.declare_parameter(
            "show_lane_boundaries",
            True,
        )
        self.declare_parameter(
            "show_road_edges",
            True,
        )
        self.declare_parameter(
            "show_wait_lines",
            True,
        )
        self.declare_parameter(
            "show_traffic_signs",
            True,
        )
        self.declare_parameter(
            "show_traffic_sign_labels",
            False,
        )

        self.declare_parameter(
            "lane_centerline_width",
            0.10,
        )
        self.declare_parameter(
            "lane_boundary_width",
            0.06,
        )
        self.declare_parameter(
            "road_edge_width",
            0.12,
        )
        self.declare_parameter(
            "wait_line_width",
            0.20,
        )
        self.declare_parameter(
            "visualization_z_offset",
            0.04,
        )

        republish_period_sec = float(
            self.get_parameter(
                "republish_period_sec"
            ).value
        )

        if republish_period_sec <= 0.0:
            raise ValueError(
                "republish_period_sec must be positive"
            )

        self.service_name = str(
            self.get_parameter("service_name").value
        )
        marker_topic = str(
            self.get_parameter("marker_topic").value
        )
        request_period_sec = float(
            self.get_parameter(
                "request_period_sec"
            ).value
        )

        self.show_lane_centerlines = bool(
            self.get_parameter(
                "show_lane_centerlines"
            ).value
        )
        self.show_lane_boundaries = bool(
            self.get_parameter(
                "show_lane_boundaries"
            ).value
        )
        self.show_road_edges = bool(
            self.get_parameter("show_road_edges").value
        )
        self.show_wait_lines = bool(
            self.get_parameter("show_wait_lines").value
        )
        self.show_traffic_signs = bool(
            self.get_parameter(
                "show_traffic_signs"
            ).value
        )
        self.show_traffic_sign_labels = bool(
            self.get_parameter(
                "show_traffic_sign_labels"
            ).value
        )

        self.lane_centerline_width = float(
            self.get_parameter(
                "lane_centerline_width"
            ).value
        )
        self.lane_boundary_width = float(
            self.get_parameter(
                "lane_boundary_width"
            ).value
        )
        self.road_edge_width = float(
            self.get_parameter(
                "road_edge_width"
            ).value
        )
        self.wait_line_width = float(
            self.get_parameter(
                "wait_line_width"
            ).value
        )
        self.z_offset = float(
            self.get_parameter(
                "visualization_z_offset"
            ).value
        )

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            marker_topic,
            marker_qos,
        )

        self.map_client = self.create_client(
            GetVectorMap,
            self.service_name,
        )

        self.known_revision = 0
        self.request_in_progress = False
        self.map_loaded = False
        self.wait_log_emitted = False

        self.cached_marker_array: (
            MarkerArray | None
        ) = None

        self.request_timer = self.create_timer(
            max(request_period_sec, 0.1),
            self.request_map,
        )

        self.republish_timer = self.create_timer(
            max(republish_period_sec, 0.1),
            self.republish_cached_markers,
        )

        self.get_logger().info(
            f"Map service client: {self.service_name}"
        )
        self.get_logger().info(
            f"Map MarkerArray topic: {marker_topic}"
        )

    def republish_cached_markers(self) -> None:
        """Republish the cached static map markers.

        This restores the map if another visualization node or RViz
        cleared its marker state between rollouts.
        """
        if self.cached_marker_array is None:
            return

        self.marker_publisher.publish(
            self.cached_marker_array
        )

    def request_map(self) -> None:
        """Request the current map revision."""
        if self.request_in_progress:
            return

        if not self.map_client.service_is_ready():
            if not self.wait_log_emitted:
                self.get_logger().info(
                    f"Waiting for map service "
                    f"{self.service_name}"
                )
                self.wait_log_emitted = True
            return

        self.wait_log_emitted = False

        request = GetVectorMap.Request()
        request.requested_scene_id = ""
        request.known_revision = self.known_revision

        self.request_in_progress = True

        future = self.map_client.call_async(request)
        future.add_done_callback(
            self.handle_map_response
        )

    def handle_map_response(self, future) -> None:
        """Handle an asynchronous GetVectorMap response."""
        self.request_in_progress = False

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"Map service request failed: {exc}"
            )
            return

        if not response.success:
            self.get_logger().info(
                f"Map not ready: {response.message}"
            )
            return

        if response.not_modified:
            if self.cached_marker_array is not None:
                self.marker_publisher.publish(
                    self.cached_marker_array
                )
            return

        vector_map = response.vector_map

        if not vector_map.scene_id:
            self.get_logger().error(
                "Map service returned success but no map data"
            )
            return

        marker_array = self.build_marker_array(
            vector_map
        )

        self.cached_marker_array = marker_array

        self.marker_publisher.publish(
            self.cached_marker_array
        )

        self.known_revision = int(
            response.revision
        )
        self.map_loaded = True

        self.get_logger().info(
            "Published VectorMap markers: "
            f"scene={vector_map.scene_id}, "
            f"revision={vector_map.revision}, "
            f"lanes={len(vector_map.lanes)}, "
            f"road_edges={len(vector_map.road_edges)}, "
            f"traffic_signs="
            f"{len(vector_map.traffic_signs)}, "
            f"wait_lines={len(vector_map.wait_lines)}, "
            f"markers={len(marker_array.markers)}"
        )

    def make_line_strip(
        self,
        frame_id: str,
        namespace: str,
        element_id: str,
        polyline: MapPolyline,
        width: float,
        color: tuple[float, float, float, float],
        z_offset: float,
    ) -> Marker | None:
        """Create a LINE_STRIP Marker from a MapPolyline."""
        if len(polyline.points) < 2:
            return None

        marker = Marker()

        marker.header.frame_id = frame_id
        # Zero stamp means use the latest available transform.
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0

        marker.ns = namespace
        marker.id = stable_marker_id(
            namespace,
            element_id,
        )

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = float(width)

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])

        marker.points = [
            copy_point(
                point,
                z_offset=z_offset,
            )
            for point in polyline.points
        ]

        # Zero duration makes static markers persistent.
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        return marker

    def make_traffic_sign_marker(
        self,
        vector_map: VectorMap,
        traffic_sign,
    ) -> Marker:
        """Create a simple traffic-sign location marker."""
        marker = Marker()

        marker.header.frame_id = vector_map.frame_id
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0

        marker.ns = "map/traffic_signs"
        marker.id = stable_marker_id(
            marker.ns,
            traffic_sign.id,
        )

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position = copy_point(
            traffic_sign.position,
            self.z_offset,
        )
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.45
        marker.scale.y = 0.45
        marker.scale.z = 0.45

        marker.color.r = 1.0
        marker.color.g = 0.10
        marker.color.b = 0.85
        marker.color.a = 0.95

        return marker

    def make_traffic_sign_label(
        self,
        vector_map: VectorMap,
        traffic_sign,
    ) -> Marker:
        """Create a text label for a traffic sign."""
        marker = Marker()

        marker.header.frame_id = vector_map.frame_id
        marker.header.stamp.sec = 0
        marker.header.stamp.nanosec = 0

        marker.ns = "map/traffic_sign_labels"
        marker.id = stable_marker_id(
            marker.ns,
            traffic_sign.id,
        )

        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position = copy_point(
            traffic_sign.position,
            0.7 + self.z_offset,
        )
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.35

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95

        marker.text = (
            f"{traffic_sign.sign_type}\n"
            f"#{traffic_sign.id}"
        )

        return marker

    def wait_line_color(
        self,
        wait_line_type: str,
        is_implicit: bool,
    ) -> tuple[float, float, float, float]:
        """Return a visualization color for one wait-line category."""
        normalized_type = wait_line_type.upper()

        if normalized_type == "STOP":
            color = (1.00, 0.85, 0.00, 1.00)
        elif normalized_type == "YIELD":
            color = (1.00, 0.40, 0.00, 1.00)
        else:
            color = (0.65, 0.65, 0.65, 0.90)

        if is_implicit:
            return (
                color[0],
                color[1],
                color[2],
                0.45,
            )

        return color

    def build_marker_array(
        self,
        vector_map: VectorMap,
    ) -> MarkerArray:
        """Convert a full VectorMap into RViz markers."""
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = vector_map.frame_id
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if self.show_lane_centerlines:
            for lane in vector_map.lanes:
                marker = self.make_line_strip(
                    frame_id=vector_map.frame_id,
                    namespace="map/lane_centerlines",
                    element_id=lane.id,
                    polyline=lane.centerline,
                    width=self.lane_centerline_width,
                    color=(0.00, 0.85, 0.75, 0.85),
                    z_offset=self.z_offset,
                )

                if marker is not None:
                    marker_array.markers.append(marker)

        if self.show_lane_boundaries:
            for lane in vector_map.lanes:
                left_marker = self.make_line_strip(
                    frame_id=vector_map.frame_id,
                    namespace="map/lane_left_boundaries",
                    element_id=lane.id,
                    polyline=lane.left_boundary,
                    width=self.lane_boundary_width,
                    color=(0.90, 0.90, 0.90, 0.55),
                    z_offset=self.z_offset + 0.01,
                )

                if left_marker is not None:
                    marker_array.markers.append(
                        left_marker
                    )

                right_marker = self.make_line_strip(
                    frame_id=vector_map.frame_id,
                    namespace="map/lane_right_boundaries",
                    element_id=lane.id,
                    polyline=lane.right_boundary,
                    width=self.lane_boundary_width,
                    color=(0.90, 0.90, 0.90, 0.55),
                    z_offset=self.z_offset + 0.01,
                )

                if right_marker is not None:
                    marker_array.markers.append(
                        right_marker
                    )

        if self.show_road_edges:
            for road_edge in vector_map.road_edges:
                marker = self.make_line_strip(
                    frame_id=vector_map.frame_id,
                    namespace="map/road_edges",
                    element_id=road_edge.id,
                    polyline=road_edge.polyline,
                    width=self.road_edge_width,
                    color=(1.00, 0.12, 0.12, 0.85),
                    z_offset=self.z_offset + 0.02,
                )

                if marker is not None:
                    marker_array.markers.append(marker)

        if self.show_wait_lines:
            for wait_line in vector_map.wait_lines:
                marker = self.make_line_strip(
                    frame_id=vector_map.frame_id,
                    namespace=(
                        "map/wait_lines/"
                        f"{wait_line.wait_line_type.lower()}"
                    ),
                    element_id=wait_line.id,
                    polyline=wait_line.polyline,
                    width=self.wait_line_width,
                    color=self.wait_line_color(
                        wait_line.wait_line_type,
                        wait_line.is_implicit,
                    ),
                    z_offset=self.z_offset + 0.05,
                )

                if marker is not None:
                    marker_array.markers.append(marker)

        if self.show_traffic_signs:
            for traffic_sign in vector_map.traffic_signs:
                marker_array.markers.append(
                    self.make_traffic_sign_marker(
                        vector_map,
                        traffic_sign,
                    )
                )

                if self.show_traffic_sign_labels:
                    marker_array.markers.append(
                        self.make_traffic_sign_label(
                            vector_map,
                            traffic_sign,
                        )
                    )

        return marker_array


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapMarkerPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
