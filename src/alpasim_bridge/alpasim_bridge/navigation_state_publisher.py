#!/usr/bin/env python3

"""Receive dynamic navigation updates from AlpaSim Runtime."""

from __future__ import annotations

import json
import math
import queue
import socket
import struct
import threading
from typing import Any

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
    EgoTrajectory,
    Route,
    RoutePoint,
    TrajectoryPoint,
)


def read_exact(
    sock: socket.socket,
    size: int,
) -> bytes:
    """Receive exactly size bytes or raise ConnectionError."""
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError(
                "Navigation TCP connection closed"
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def time_from_microseconds(value: int) -> Time:
    """Convert an absolute timestamp in microseconds to ROS Time."""
    message = Time()
    message.sec = int(value // 1_000_000)
    message.nanosec = int(
        (value % 1_000_000) * 1000
    )
    return message


def duration_from_microseconds(
    value: int,
) -> Duration:
    """Convert signed microseconds to a normalized ROS Duration."""
    message = Duration()

    seconds = value // 1_000_000
    remaining_us = value - seconds * 1_000_000

    message.sec = int(seconds)
    message.nanosec = int(
        remaining_us * 1000
    )

    return message


def require_finite(
    value: Any,
    field_name: str,
) -> float:
    """Convert a numeric value to float and require it to be finite."""
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{field_name} is not finite: {result}"
        )

    return result


def route_generator_value(
    generator_type: str,
) -> int:
    """Convert Runtime RouteGeneratorType text to Route constants."""
    normalized = str(generator_type).upper()

    # Supports values such as:
    # MAP
    # RECORDED
    # RouteGeneratorType.MAP
    # RouteGeneratorType.RECORDED
    if normalized.endswith("RECORDED"):
        return Route.GENERATOR_RECORDED

    if normalized.endswith("MAP"):
        return Route.GENERATOR_MAP

    return Route.GENERATOR_UNKNOWN


def trajectory_source_value(
    source_name: str,
) -> int:
    """Convert Runtime plan source text to EgoTrajectory constants."""
    normalized = str(source_name).upper()

    if normalized == "MODEL_PLANNING":
        return EgoTrajectory.SOURCE_MODEL_PLANNING

    if normalized == "CONTROLLER_REFERENCE":
        return EgoTrajectory.SOURCE_CONTROLLER_REFERENCE

    return EgoTrajectory.SOURCE_UNKNOWN


def make_route_point(
    source: dict[str, Any],
) -> RoutePoint:
    """Convert one Runtime Route point to a ROS RoutePoint."""
    message = RoutePoint()
    message.valid = bool(source.get("valid", False))

    if not message.valid:
        # prepare_for_policy() may pad the Route with NaNs.
        # Runtime replaces them with finite zero values plus valid=false.
        message.position.x = 0.0
        message.position.y = 0.0
        message.position.z = 0.0
        message.longitudinal_distance = 0.0
        return message

    position = source["position"]

    message.position.x = require_finite(
        position["x"],
        "route.position.x",
    )
    message.position.y = require_finite(
        position["y"],
        "route.position.y",
    )
    message.position.z = require_finite(
        position["z"],
        "route.position.z",
    )

    message.longitudinal_distance = require_finite(
        source["longitudinal_distance"],
        "route.longitudinal_distance",
    )

    return message


def make_route(
    source: dict[str, Any],
    *,
    reference_timestamp_us: int,
    sequence: int,
    generator_type: str,
    producer: str,
) -> Route:
    """Convert one Runtime Route representation to a ROS Route."""
    message = Route()

    message.reference_stamp = time_from_microseconds(
        reference_timestamp_us
    )

    message.frame_id = str(source["frame_id"])
    message.source_frame_id = str(
        source["source_frame_id"]
    )

    message.generator_type = route_generator_value(
        generator_type
    )
    message.producer = str(producer)
    message.sequence = int(sequence)

    message.lookahead_distance = require_finite(
        source["lookahead_distance"],
        "route.lookahead_distance",
    )

    message.expected_point_count = int(
        source["expected_point_count"]
    )

    message.points = [
        make_route_point(point)
        for point in source.get("points", [])
    ]

    if (
        message.expected_point_count > 0
        and len(message.points)
        != message.expected_point_count
    ):
        raise ValueError(
            "Route point count does not match expected_point_count: "
            f"{len(message.points)} != "
            f"{message.expected_point_count}"
        )

    return message


def populate_pose(
    message_pose,
    source: dict[str, Any],
) -> None:
    """Populate a geometry_msgs/Pose from Runtime trajectory JSON."""
    position = source["position"]
    orientation = source["orientation"]

    message_pose.position.x = require_finite(
        position["x"],
        "trajectory.position.x",
    )
    message_pose.position.y = require_finite(
        position["y"],
        "trajectory.position.y",
    )
    message_pose.position.z = require_finite(
        position["z"],
        "trajectory.position.z",
    )

    message_pose.orientation.x = require_finite(
        orientation["x"],
        "trajectory.orientation.x",
    )
    message_pose.orientation.y = require_finite(
        orientation["y"],
        "trajectory.orientation.y",
    )
    message_pose.orientation.z = require_finite(
        orientation["z"],
        "trajectory.orientation.z",
    )
    message_pose.orientation.w = require_finite(
        orientation["w"],
        "trajectory.orientation.w",
    )


def populate_vector3(
    message_vector,
    source: dict[str, Any],
    field_name: str,
) -> None:
    """Populate geometry_msgs/Vector3 from Runtime JSON."""
    message_vector.x = require_finite(
        source["x"],
        f"{field_name}.x",
    )
    message_vector.y = require_finite(
        source["y"],
        f"{field_name}.y",
    )
    message_vector.z = require_finite(
        source["z"],
        f"{field_name}.z",
    )


def make_trajectory_point(
    source: dict[str, Any],
    reference_timestamp_us: int,
) -> TrajectoryPoint:
    """Convert one Runtime trajectory sample to TrajectoryPoint."""
    message = TrajectoryPoint()

    timestamp_us = int(source["timestamp_us"])

    message.stamp = time_from_microseconds(
        timestamp_us
    )
    message.time_from_reference = (
        duration_from_microseconds(
            timestamp_us - reference_timestamp_us
        )
    )

    populate_pose(
        message.pose,
        source,
    )

    populate_vector3(
        message.linear_velocity,
        source["linear_velocity"],
        "linear_velocity",
    )

    populate_vector3(
        message.linear_acceleration,
        source["linear_acceleration"],
        "linear_acceleration",
    )

    message.yaw = require_finite(
        source["yaw"],
        "trajectory.yaw",
    )
    message.yaw_rate = require_finite(
        source["yaw_rate"],
        "trajectory.yaw_rate",
    )
    message.yaw_acceleration = require_finite(
        source["yaw_acceleration"],
        "trajectory.yaw_acceleration",
    )
    message.speed = require_finite(
        source["speed"],
        "trajectory.speed",
    )

    return message


def make_planned_trajectory(
    source: dict[str, Any],
    *,
    reference_timestamp_us: int,
    force_gt_active: bool,
) -> EgoTrajectory:
    """Convert Runtime planned/controller trajectory to EgoTrajectory."""
    message = EgoTrajectory()

    message.reference_stamp = time_from_microseconds(
        reference_timestamp_us
    )

    message.pose_frame_id = str(
        source.get("pose_frame_id", "map")
    )
    message.dynamics_frame_id = str(
        source.get("dynamics_frame_id", "map")
    )

    source_name = str(
        source.get("source", "UNKNOWN")
    )

    message.source = trajectory_source_value(
        source_name
    )
    message.producer = str(
        source.get("producer", "unknown")
    )
    message.is_model_generated = bool(
        source.get("is_model_generated", False)
    )
    message.force_gt_active = bool(
        force_gt_active
    )

    start_timestamp_us = int(
        source.get(
            "start_timestamp_us",
            reference_timestamp_us,
        )
    )
    end_timestamp_us = int(
        source.get(
            "end_timestamp_us",
            start_timestamp_us,
        )
    )

    if end_timestamp_us < start_timestamp_us:
        raise ValueError(
            "Planned trajectory end timestamp precedes start: "
            f"{end_timestamp_us} < {start_timestamp_us}"
        )

    requested_duration_us = max(
        0,
        end_timestamp_us - reference_timestamp_us,
    )
    actual_duration_us = max(
        0,
        end_timestamp_us - start_timestamp_us,
    )

    message.requested_duration = (
        duration_from_microseconds(
            requested_duration_us
        )
    )
    message.actual_duration = (
        duration_from_microseconds(
            actual_duration_us
        )
    )

    message.points = [
        make_trajectory_point(
            point,
            reference_timestamp_us,
        )
        for point in source.get("points", [])
    ]

    # Runtime serializer requires strictly increasing timestamps.
    previous_timestamp_us: int | None = None

    for point in source.get("points", []):
        timestamp_us = int(point["timestamp_us"])

        if (
            previous_timestamp_us is not None
            and timestamp_us <= previous_timestamp_us
        ):
            raise ValueError(
                "Planned trajectory timestamps are not "
                "strictly increasing"
            )

        previous_timestamp_us = timestamp_us

    return message


class NavigationStatePublisher(Node):
    """Receive Runtime navigation packets and publish ROS navigation state."""

    def __init__(self) -> None:
        super().__init__(
            "navigation_state_publisher"
        )

        self.declare_parameter(
            "navigation_tcp_host",
            "127.0.0.1",
        )
        self.declare_parameter(
            "navigation_tcp_port",
            15005,
        )

        self.declare_parameter(
            "route_map_topic",
            "/alpasim/route/map",
        )
        self.declare_parameter(
            "route_model_input_topic",
            "/alpasim/route/model_input",
        )
        self.declare_parameter(
            "planned_trajectory_topic",
            "/alpasim/planning/ego/trajectory",
        )

        self.declare_parameter(
            "publish_route_map",
            True,
        )
        self.declare_parameter(
            "publish_route_model_input",
            True,
        )
        self.declare_parameter(
            "publish_planned_trajectory",
            True,
        )

        self.navigation_tcp_host = str(
            self.get_parameter(
                "navigation_tcp_host"
            ).value
        )
        self.navigation_tcp_port = int(
            self.get_parameter(
                "navigation_tcp_port"
            ).value
        )

        route_map_topic = str(
            self.get_parameter(
                "route_map_topic"
            ).value
        )
        route_model_input_topic = str(
            self.get_parameter(
                "route_model_input_topic"
            ).value
        )
        planned_trajectory_topic = str(
            self.get_parameter(
                "planned_trajectory_topic"
            ).value
        )

        self.publish_route_map_enabled = bool(
            self.get_parameter(
                "publish_route_map"
            ).value
        )
        self.publish_route_model_input_enabled = bool(
            self.get_parameter(
                "publish_route_model_input"
            ).value
        )
        self.publish_planned_trajectory_enabled = bool(
            self.get_parameter(
                "publish_planned_trajectory"
            ).value
        )

        navigation_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.route_map_publisher = self.create_publisher(
            Route,
            route_map_topic,
            navigation_qos,
        )

        self.route_model_input_publisher = (
            self.create_publisher(
                Route,
                route_model_input_topic,
                navigation_qos,
            )
        )

        self.planned_trajectory_publisher = (
            self.create_publisher(
                EgoTrajectory,
                planned_trajectory_topic,
                navigation_qos,
            )
        )

        self.packet_queue: queue.Queue[
            dict[str, Any]
        ] = queue.Queue(maxsize=8)

        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        self.server_socket.bind(
            (
                self.navigation_tcp_host,
                self.navigation_tcp_port,
            )
        )
        self.server_socket.listen(1)

        self.stop_event = threading.Event()

        self.receiver_thread = threading.Thread(
            target=self.navigation_tcp_server_loop,
            name="alpasim-navigation-tcp-server",
            daemon=True,
        )
        self.receiver_thread.start()

        # All ROS publication occurs from the executor thread.
        self.timer = self.create_timer(
            0.002,
            self.poll_navigation_packets,
        )

        self.packet_count = 0
        self.last_sequence = 0
        self.last_reference_timestamp_us: int | None = None

        self.get_logger().info(
            "Navigation publisher listening at "
            f"tcp://{self.navigation_tcp_host}:"
            f"{self.navigation_tcp_port}"
        )
        self.get_logger().info(
            f"Route map topic: {route_map_topic}"
        )
        self.get_logger().info(
            "Route model-input topic: "
            f"{route_model_input_topic}"
        )
        self.get_logger().info(
            "Planned trajectory topic: "
            f"{planned_trajectory_topic}"
        )

    def navigation_tcp_server_loop(self) -> None:
        """Accept Runtime connections and receive navigation packets."""
        while not self.stop_event.is_set():
            try:
                connection, address = (
                    self.server_socket.accept()
                )
            except OSError:
                break

            self.get_logger().info(
                f"Navigation Runtime connected from {address}"
            )

            try:
                with connection:
                    while not self.stop_event.is_set():
                        payload_length = struct.unpack(
                            "!I",
                            read_exact(connection, 4),
                        )[0]

                        if payload_length <= 0:
                            raise ValueError(
                                "Navigation payload length "
                                "must be positive"
                            )

                        payload = read_exact(
                            connection,
                            payload_length,
                        )

                        packet = json.loads(
                            payload.decode("utf-8")
                        )

                        try:
                            self.packet_queue.put_nowait(
                                packet
                            )
                        except queue.Full:
                            # Dynamic data: preserve the newest state.
                            try:
                                self.packet_queue.get_nowait()
                            except queue.Empty:
                                pass

                            try:
                                self.packet_queue.put_nowait(
                                    packet
                                )
                            except queue.Full:
                                pass

            except (
                ConnectionError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                struct.error,
                ValueError,
            ) as exc:
                if not self.stop_event.is_set():
                    self.get_logger().warning(
                        "Navigation TCP connection ended: "
                        f"{exc}"
                    )

    def poll_navigation_packets(self) -> None:
        """Process pending packets without starving other callbacks."""
        for _ in range(8):
            try:
                packet = self.packet_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self.process_navigation_packet(
                    packet
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                self.get_logger().error(
                    f"Invalid navigation packet: {exc}"
                )

    def process_navigation_packet(
        self,
        packet: dict[str, Any],
    ) -> None:
        """Convert one Runtime navigation update and publish it."""
        if (
            packet.get("message_type")
            != "navigation_update"
        ):
            self.get_logger().warning(
                "Ignoring unsupported navigation "
                f"message type: "
                f"{packet.get('message_type')}"
            )
            return

        sequence = int(packet["sequence"])
        reference_timestamp_us = int(
            packet["reference_timestamp_us"]
        )

        # A new rollout may reset simulation time and Runtime sequence.
        # Accept it and reset local monotonicity tracking.
        if (
            self.last_reference_timestamp_us is not None
            and reference_timestamp_us
            < self.last_reference_timestamp_us
        ):
            self.get_logger().info(
                "Navigation simulation time moved backwards; "
                "treating packet as a new rollout"
            )
            self.last_sequence = 0

        if (
            self.last_sequence != 0
            and sequence <= self.last_sequence
            and (
                self.last_reference_timestamp_us is not None
                and reference_timestamp_us
                >= self.last_reference_timestamp_us
            )
        ):
            self.get_logger().warning(
                "Ignoring stale navigation update: "
                f"sequence={sequence}, "
                f"last_sequence={self.last_sequence}"
            )
            return

        generator_type = str(
            packet.get(
                "route_generator_type",
                "UNKNOWN",
            )
        )
        force_gt_active = bool(
            packet.get("force_gt_active", False)
        )

        published_route_points = 0
        planned_point_count = 0

        route_map_source = packet.get("route_map")

        if (
            self.publish_route_map_enabled
            and route_map_source is not None
        ):
            route_map = make_route(
                route_map_source,
                reference_timestamp_us=(
                    reference_timestamp_us
                ),
                sequence=sequence,
                generator_type=generator_type,
                producer="alpasim_route_generator",
            )

            if route_map.frame_id != "map":
                raise ValueError(
                    "route_map must be expressed in map, "
                    f"got {route_map.frame_id!r}"
                )

            self.route_map_publisher.publish(
                route_map
            )

            published_route_points = len(
                route_map.points
            )

        route_model_input_source = packet.get(
            "route_model_input"
        )

        if (
            self.publish_route_model_input_enabled
            and route_model_input_source is not None
        ):
            route_model_input = make_route(
                route_model_input_source,
                reference_timestamp_us=(
                    reference_timestamp_us
                ),
                sequence=sequence,
                generator_type=generator_type,
                producer=(
                    "alpasim_route_generator_model_input"
                ),
            )

            if (
                route_model_input.frame_id
                != "base_link"
            ):
                raise ValueError(
                    "route_model_input must be expressed "
                    "in base_link, got "
                    f"{route_model_input.frame_id!r}"
                )

            self.route_model_input_publisher.publish(
                route_model_input
            )

        planned_source = packet.get(
            "planned_trajectory"
        )

        if (
            self.publish_planned_trajectory_enabled
            and planned_source is not None
        ):
            planned_trajectory = (
                make_planned_trajectory(
                    planned_source,
                    reference_timestamp_us=(
                        reference_timestamp_us
                    ),
                    force_gt_active=(
                        force_gt_active
                    ),
                )
            )

            if (
                planned_trajectory.pose_frame_id
                != "map"
            ):
                raise ValueError(
                    "planned trajectory must be expressed "
                    "in map, got "
                    f"{planned_trajectory.pose_frame_id!r}"
                )

            self.planned_trajectory_publisher.publish(
                planned_trajectory
            )

            planned_point_count = len(
                planned_trajectory.points
            )

        self.last_sequence = sequence
        self.last_reference_timestamp_us = (
            reference_timestamp_us
        )
        self.packet_count += 1

        if self.packet_count == 1:
            self.get_logger().info(
                "Published first navigation update: "
                f"sequence={sequence}, "
                f"reference_time="
                f"{reference_timestamp_us / 1e6:.6f}s, "
                f"route_generator={generator_type}, "
                f"route_points={published_route_points}, "
                f"planned_points={planned_point_count}, "
                f"force_gt_active={force_gt_active}"
            )

    def destroy_node(self) -> None:
        self.stop_event.set()

        try:
            self.server_socket.close()
        except OSError:
            pass

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
