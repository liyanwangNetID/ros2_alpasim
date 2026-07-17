#!/usr/bin/env python3

"""ROS 2 vector-map server for AlpaSim."""

from __future__ import annotations

import copy
import json
import queue
import socket
import struct
import threading
from typing import Any

import rclpy
from rclpy.node import Node

from alpasim_msgs.msg import (
    MapLane,
    MapPolyline,
    MapRoadEdge,
    MapTrafficSign,
    MapWaitLine,
    VectorMap,
)
from alpasim_msgs.srv import GetVectorMap
from geometry_msgs.msg import Point


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
                "Map TCP connection closed"
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def make_point(values: dict[str, Any]) -> Point:
    """Convert a Runtime JSON point into geometry_msgs/Point."""
    point = Point()
    point.x = float(values["x"])
    point.y = float(values["y"])
    point.z = float(values["z"])
    return point


def make_polyline(
    source: dict[str, Any],
) -> MapPolyline:
    """Convert Runtime polyline JSON into MapPolyline."""
    message = MapPolyline()

    message.points = [
        make_point(point)
        for point in source.get("points", [])
    ]

    message.headings = [
        float(value)
        for value in source.get("headings", [])
    ]

    if (
        message.headings
        and len(message.headings) != len(message.points)
    ):
        raise ValueError(
            "Polyline heading count does not match point count: "
            f"{len(message.headings)} != {len(message.points)}"
        )

    return message


def make_lane(
    source: dict[str, Any],
) -> MapLane:
    """Convert Runtime lane JSON into MapLane."""
    message = MapLane()

    message.id = str(source["id"])

    message.centerline = make_polyline(
        source["centerline"]
    )
    message.left_boundary = make_polyline(
        source["left_boundary"]
    )
    message.right_boundary = make_polyline(
        source["right_boundary"]
    )

    message.successor_ids = [
        str(value)
        for value in source.get("successor_ids", [])
    ]
    message.predecessor_ids = [
        str(value)
        for value in source.get("predecessor_ids", [])
    ]
    message.left_adjacent_ids = [
        str(value)
        for value in source.get("left_adjacent_ids", [])
    ]
    message.right_adjacent_ids = [
        str(value)
        for value in source.get("right_adjacent_ids", [])
    ]

    message.traffic_sign_ids = [
        str(value)
        for value in source.get("traffic_sign_ids", [])
    ]
    message.wait_line_ids = [
        str(value)
        for value in source.get("wait_line_ids", [])
    ]
    message.road_area_ids = [
        str(value)
        for value in source.get("road_area_ids", [])
    ]

    return message


def make_road_edge(
    source: dict[str, Any],
) -> MapRoadEdge:
    """Convert Runtime road-edge JSON into MapRoadEdge."""
    message = MapRoadEdge()
    message.id = str(source["id"])
    message.polyline = make_polyline(
        source["polyline"]
    )
    return message


def make_traffic_sign(
    source: dict[str, Any],
) -> MapTrafficSign:
    """Convert Runtime traffic-sign JSON into MapTrafficSign."""
    message = MapTrafficSign()

    message.id = str(source["id"])
    message.sign_type = str(source["sign_type"])
    message.position = make_point(
        source["position"]
    )

    return message


def make_wait_line(
    source: dict[str, Any],
) -> MapWaitLine:
    """Convert Runtime wait-line JSON into MapWaitLine."""
    message = MapWaitLine()

    message.id = str(source["id"])
    message.wait_line_type = str(
        source["wait_line_type"]
    )
    message.is_implicit = bool(
        source["is_implicit"]
    )
    message.polyline = make_polyline(
        source["polyline"]
    )

    return message


def make_vector_map(
    source: dict[str, Any],
    revision: int,
) -> VectorMap:
    """Convert complete Runtime JSON into VectorMap."""
    message = VectorMap()

    message.frame_id = str(
        source.get("frame_id", "map")
    )
    message.scene_id = str(source["scene_id"])
    message.map_id = str(source["map_id"])
    message.revision = int(revision)

    extent = source["extent"]
    message.minimum = make_point(
        extent["minimum"]
    )
    message.maximum = make_point(
        extent["maximum"]
    )

    message.lanes = [
        make_lane(lane)
        for lane in source.get("lanes", [])
    ]

    message.road_edges = [
        make_road_edge(edge)
        for edge in source.get("road_edges", [])
    ]

    message.traffic_signs = [
        make_traffic_sign(sign)
        for sign in source.get("traffic_signs", [])
    ]

    message.wait_lines = [
        make_wait_line(wait_line)
        for wait_line in source.get("wait_lines", [])
    ]

    return message


class AlpaSimMapServer(Node):
    """Receive, cache and serve the current AlpaSim vector map."""

    def __init__(self) -> None:
        super().__init__("alpasim_map_server")

        self.declare_parameter(
            "map_tcp_host",
            "127.0.0.1",
        )
        self.declare_parameter(
            "map_tcp_port",
            15003,
        )
        self.declare_parameter(
            "service_name",
            "/alpasim/map/get_vector_map",
        )

        self.map_tcp_host = str(
            self.get_parameter("map_tcp_host").value
        )
        self.map_tcp_port = int(
            self.get_parameter("map_tcp_port").value
        )
        self.service_name = str(
            self.get_parameter("service_name").value
        )

        self.cached_map: VectorMap | None = None
        self.cached_source_revision = 0

        self.revision = 0
        self.map_receive_count = 0

        self.packet_queue: queue.Queue[
            dict[str, Any]
        ] = queue.Queue(maxsize=2)

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
                self.map_tcp_host,
                self.map_tcp_port,
            )
        )
        self.server_socket.listen(1)

        self.stop_event = threading.Event()

        self.receiver_thread = threading.Thread(
            target=self.map_tcp_server_loop,
            name="alpasim-map-tcp-server",
            daemon=True,
        )
        self.receiver_thread.start()

        # Convert/cache maps only from the ROS executor thread.
        self.timer = self.create_timer(
            0.05,
            self.poll_map_packets,
        )

        self.service = self.create_service(
            GetVectorMap,
            self.service_name,
            self.handle_get_vector_map,
        )

        self.get_logger().info(
            "Map server listening for Runtime at "
            f"tcp://{self.map_tcp_host}:{self.map_tcp_port}"
        )
        self.get_logger().info(
            f"Map service: {self.service_name}"
        )

    def map_tcp_server_loop(self) -> None:
        """Receive framed JSON map packets from Runtime."""
        while not self.stop_event.is_set():
            try:
                connection, address = (
                    self.server_socket.accept()
                )
            except OSError:
                break

            self.get_logger().info(
                f"Map Runtime connected from {address}"
            )

            try:
                with connection:
                    while not self.stop_event.is_set():
                        payload_length = struct.unpack(
                            "!I",
                            read_exact(connection, 4),
                        )[0]

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
                            # Keep the newest complete map.
                            try:
                                self.packet_queue.get_nowait()
                            except queue.Empty:
                                pass

                            self.packet_queue.put_nowait(
                                packet
                            )

            except (
                ConnectionError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                struct.error,
            ) as exc:
                if not self.stop_event.is_set():
                    self.get_logger().warning(
                        f"Map TCP connection ended: {exc}"
                    )

    def poll_map_packets(self) -> None:
        """Convert pending Runtime packets into ROS maps."""
        while True:
            try:
                packet = self.packet_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self.process_map_packet(packet)
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                self.get_logger().error(
                    f"Invalid vector-map packet: {exc}"
                )

    def process_map_packet(
        self,
        packet: dict[str, Any],
    ) -> None:
        """Validate and cache one Runtime map."""
        if packet.get("message_type") != "vector_map":
            self.get_logger().warning(
                "Ignoring unsupported map message type: "
                f"{packet.get('message_type')}"
            )
            return

        scene_id = str(packet["scene_id"])
        map_id = str(packet["map_id"])
        source_revision = int(
            packet.get("source_revision", 1)
        )

        # Ignore an exact retransmission of the cached source map.
        if (
            self.cached_map is not None
            and self.cached_map.scene_id == scene_id
            and self.cached_map.map_id == map_id
            and self.cached_source_revision
            == source_revision
        ):
            self.get_logger().info(
                "Ignoring duplicate vector map: "
                f"scene={scene_id}, "
                f"source_revision={source_revision}"
            )
            return

        next_revision = self.revision + 1

        converted_map = make_vector_map(
            packet,
            revision=next_revision,
        )

        self.cached_map = converted_map
        self.cached_source_revision = source_revision
        self.revision = next_revision
        self.map_receive_count += 1

        self.get_logger().info(
            "Cached VectorMap: "
            f"scene={converted_map.scene_id}, "
            f"map_id={converted_map.map_id}, "
            f"revision={converted_map.revision}, "
            f"lanes={len(converted_map.lanes)}, "
            f"road_edges={len(converted_map.road_edges)}, "
            f"traffic_signs="
            f"{len(converted_map.traffic_signs)}, "
            f"wait_lines={len(converted_map.wait_lines)}"
        )

    def handle_get_vector_map(
        self,
        request: GetVectorMap.Request,
        response: GetVectorMap.Response,
    ) -> GetVectorMap.Response:
        """Return the cached vector map when available."""
        if self.cached_map is None:
            response.success = False
            response.not_modified = False
            response.message = (
                "No vector map is currently loaded"
            )
            response.revision = 0
            return response

        if (
            request.requested_scene_id
            and request.requested_scene_id
            != self.cached_map.scene_id
        ):
            response.success = False
            response.not_modified = False
            response.message = (
                "Requested scene does not match loaded scene: "
                f"requested={request.requested_scene_id}, "
                f"loaded={self.cached_map.scene_id}"
            )
            response.revision = self.revision
            return response

        response.success = True
        response.revision = self.revision

        if (
            request.known_revision != 0
            and request.known_revision
            == self.revision
        ):
            response.not_modified = True
            response.message = (
                "Client already has the current map revision"
            )
            return response

        response.not_modified = False
        response.message = (
            "Returning complete vector map"
        )

        # Copy the cached ROS message into the response.
        response.vector_map = copy.deepcopy(
            self.cached_map
        )

        return response

    def destroy_node(self) -> None:
        self.stop_event.set()

        try:
            self.server_socket.close()
        except OSError:
            pass

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AlpaSimMapServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
