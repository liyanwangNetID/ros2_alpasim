#!/usr/bin/env python3

from __future__ import annotations

import zlib

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from alpasim_msgs.msg import ActorStateArray
from visualization_msgs.msg import Marker, MarkerArray


# RGBA colors by native AlpaSim label_class.
ACTOR_COLORS = {
    "automobile": (0.10, 0.45, 1.00, 0.65),
    "heavy_truck": (1.00, 0.45, 0.05, 0.70),
    "trailer": (0.65, 0.20, 0.90, 0.70),
    "pedestrian": (0.95, 0.15, 0.15, 0.75),
    "bicycle": (0.10, 0.85, 0.25, 0.75),
    "motorcycle": (0.95, 0.85, 0.10, 0.75),
}

DEFAULT_COLOR = (0.55, 0.55, 0.55, 0.65)


def stable_marker_id(track_id: str, suffix: str = "") -> int:
    """Generate a deterministic non-negative RViz marker ID."""
    value = zlib.crc32(
        f"{track_id}:{suffix}".encode("utf-8")
    )

    # Marker.id is signed int32. Keep it in the positive range.
    return int(value & 0x7FFFFFFF)


class ActorMarkerPublisher(Node):
    """Visualize AlpaSim ground-truth traffic actors in RViz2."""

    def __init__(self) -> None:
        super().__init__("actor_marker_publisher")

        self.declare_parameter(
            "input_topic",
            "/alpasim/actors/current",
        )
        self.declare_parameter(
            "marker_topic",
            "/alpasim/actors/markers",
        )

        self.declare_parameter("show_labels", True)
        self.declare_parameter("show_velocity_arrows", True)

        self.declare_parameter("box_alpha", 0.65)
        self.declare_parameter("label_height_offset", 0.5)
        self.declare_parameter("velocity_arrow_scale", 0.5)
        self.declare_parameter("marker_lifetime_sec", 1.0)

        input_topic = str(
            self.get_parameter("input_topic").value
        )
        marker_topic = str(
            self.get_parameter("marker_topic").value
        )

        self.show_labels = bool(
            self.get_parameter("show_labels").value
        )
        self.show_velocity_arrows = bool(
            self.get_parameter(
                "show_velocity_arrows"
            ).value
        )

        self.box_alpha = float(
            self.get_parameter("box_alpha").value
        )
        self.label_height_offset = float(
            self.get_parameter(
                "label_height_offset"
            ).value
        )
        self.velocity_arrow_scale = float(
            self.get_parameter(
                "velocity_arrow_scale"
            ).value
        )
        self.marker_lifetime_sec = float(
            self.get_parameter(
                "marker_lifetime_sec"
            ).value
        )

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        actor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            marker_topic,
            marker_qos,
        )

        self.actor_subscription = self.create_subscription(
            ActorStateArray,
            input_topic,
            self.actor_callback,
            actor_qos,
        )

        self.message_count = 0

        self.get_logger().info(
            f"Subscribing to {input_topic}"
        )
        self.get_logger().info(
            f"Publishing actor markers to {marker_topic}"
        )

    def make_duration(self):
        """Create a builtin_interfaces/Duration-compatible value."""
        lifetime = rclpy.duration.Duration(
            seconds=self.marker_lifetime_sec
        )
        return lifetime.to_msg()

    def actor_callback(
        self,
        actor_array: ActorStateArray,
    ) -> None:
        marker_array = MarkerArray()

        # Clear markers from actors that disappeared since the last update.
        delete_marker = Marker()
        delete_marker.header.frame_id = (
            actor_array.pose_frame_id
        )
        delete_marker.header.stamp = actor_array.stamp
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(delete_marker)

        for actor in actor_array.actors:
            marker_array.markers.append(
                self.make_box_marker(
                    actor_array,
                    actor,
                )
            )

            if self.show_labels:
                marker_array.markers.append(
                    self.make_label_marker(
                        actor_array,
                        actor,
                    )
                )

            if (
                self.show_velocity_arrows
                and actor.speed > 0.05
            ):
                marker_array.markers.append(
                    self.make_velocity_marker(
                        actor_array,
                        actor,
                    )
                )

        self.marker_publisher.publish(marker_array)

        self.message_count += 1

        if self.message_count == 1:
            self.get_logger().info(
                "Published first actor MarkerArray: "
                f"{len(actor_array.actors)} actors, "
                f"{len(marker_array.markers)} markers, "
                f"frame={actor_array.pose_frame_id}"
            )

    def make_box_marker(
        self,
        actor_array: ActorStateArray,
        actor,
    ) -> Marker:
        """Create an oriented AABB box marker."""
        marker = Marker()

        marker.header.frame_id = (
            actor_array.pose_frame_id
        )
        marker.header.stamp = actor_array.stamp

        marker.ns = f"actors/boxes/{actor.label_class}"
        marker.id = stable_marker_id(
            actor.track_id,
            "box",
        )

        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose = actor.pose

        # The actor pose is already the center of its AABB.
        marker.scale.x = max(
            float(actor.dimensions.x),
            0.05,
        )
        marker.scale.y = max(
            float(actor.dimensions.y),
            0.05,
        )
        marker.scale.z = max(
            float(actor.dimensions.z),
            0.05,
        )

        color = ACTOR_COLORS.get(
            actor.label_class,
            DEFAULT_COLOR,
        )

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])

        # Allow the YAML parameter to override default class alpha.
        marker.color.a = min(
            max(self.box_alpha, 0.0),
            1.0,
        )

        marker.lifetime = self.make_duration()

        return marker

    def make_label_marker(
        self,
        actor_array: ActorStateArray,
        actor,
    ) -> Marker:
        """Show track ID, semantic class and vehicle speed."""
        marker = Marker()

        marker.header.frame_id = (
            actor_array.pose_frame_id
        )
        marker.header.stamp = actor_array.stamp

        marker.ns = "actors/labels"
        marker.id = stable_marker_id(
            actor.track_id,
            "label",
        )

        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = actor.pose.position.x
        marker.pose.position.y = actor.pose.position.y
        marker.pose.position.z = (
            actor.pose.position.z
            + 0.5 * actor.dimensions.z
            + self.label_height_offset
        )
        marker.pose.orientation.w = 1.0

        # For TEXT_VIEW_FACING, scale.z controls text height.
        marker.scale.z = 0.45

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95

        static_text = " static" if actor.is_static else ""

        marker.text = (
            f"{actor.label_class} #{actor.track_id}"
            f"{static_text}\n"
            f"{actor.speed:.1f} m/s"
        )

        marker.lifetime = self.make_duration()

        return marker

    def make_velocity_marker(
        self,
        actor_array: ActorStateArray,
        actor,
    ) -> Marker:
        """Create an arrow representing map-frame linear velocity."""
        marker = Marker()

        marker.header.frame_id = (
            actor_array.dynamics_frame_id
        )
        marker.header.stamp = actor_array.stamp

        marker.ns = "actors/velocity"
        marker.id = stable_marker_id(
            actor.track_id,
            "velocity",
        )

        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = actor.pose.position

        velocity = actor.linear_velocity

        from geometry_msgs.msg import Point

        start_point = Point()
        start_point.x = float(start.x)
        start_point.y = float(start.y)
        start_point.z = (
            float(start.z)
            + 0.5 * float(actor.dimensions.z)
        )

        end_point = Point()
        end_point.x = (
            start_point.x
            + self.velocity_arrow_scale
            * float(velocity.x)
        )
        end_point.y = (
            start_point.y
            + self.velocity_arrow_scale
            * float(velocity.y)
        )
        end_point.z = (
            start_point.z
            + self.velocity_arrow_scale
            * float(velocity.z)
        )

        marker.points = [
            start_point,
            end_point,
        ]

        # ARROW with points:
        # scale.x = shaft diameter
        # scale.y = head diameter
        # scale.z = head length
        marker.scale.x = 0.10
        marker.scale.y = 0.24
        marker.scale.z = 0.30

        marker.color.r = 0.10
        marker.color.g = 1.00
        marker.color.b = 0.20
        marker.color.a = 0.95

        marker.lifetime = self.make_duration()

        return marker

    def destroy_node(self) -> None:
        # Clear markers when this node exits normally.
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActorMarkerPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
