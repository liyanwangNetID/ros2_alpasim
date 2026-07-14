#!/usr/bin/env python3

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from alpasim_msgs.msg import EgoState
from sensor_msgs.msg import CompressedImage


CAMERA_TOPICS = {
    "camera_cross_left_120fov":
        "/alpasim/camera/cross_left/image/compressed",

    "camera_front_wide_120fov":
        "/alpasim/camera/front_wide/image/compressed",

    "camera_cross_right_120fov":
        "/alpasim/camera/cross_right/image/compressed",

    "camera_rear_left_70fov":
        "/alpasim/camera/rear/image/compressed",
}


def detect_image_format(data: bytes) -> str:
    """Infer common encoded image formats from file magic bytes."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"

    return "unknown"


def read_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly size bytes or raise ConnectionError."""
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError("Camera TCP connection closed")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


class EgoStatePublisher(Node):
    def __init__(self) -> None:
        super().__init__("alpasim_sensor_publisher")

        self.declare_parameter("ego_udp_host", "127.0.0.1")
        self.declare_parameter("ego_udp_port", 15000)

        self.declare_parameter("camera_tcp_host", "127.0.0.1")
        self.declare_parameter("camera_tcp_port", 15001)

        ego_host = str(self.get_parameter("ego_udp_host").value)
        ego_port = int(self.get_parameter("ego_udp_port").value)

        camera_host = str(self.get_parameter("camera_tcp_host").value)
        camera_port = int(self.get_parameter("camera_tcp_port").value)

        self.ego_publisher = self.create_publisher(
            EgoState,
            "/alpasim/ego_state",
            10,
        )

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.camera_publishers = {
            logical_id: self.create_publisher(
                CompressedImage,
                topic,
                camera_qos,
            )
            for logical_id, topic in CAMERA_TOPICS.items()
        }

        self.ego_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.ego_socket.bind((ego_host, ego_port))
        self.ego_socket.setblocking(False)

        self.camera_queue: queue.Queue[
            tuple[dict[str, Any], bytes]
        ] = queue.Queue(maxsize=32)

        self.camera_server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self.camera_server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        self.camera_server_socket.bind((camera_host, camera_port))
        self.camera_server_socket.listen(1)

        self.stop_event = threading.Event()

        self.camera_thread = threading.Thread(
            target=self.camera_server_loop,
            name="alpasim-camera-tcp-server",
            daemon=True,
        )
        self.camera_thread.start()

        # Publish only from the ROS executor thread.
        self.timer = self.create_timer(0.001, self.poll_inputs)

        self.ego_count = 0
        self.camera_counts = {
            logical_id: 0
            for logical_id in CAMERA_TOPICS
        }

        self.get_logger().info(
            f"Listening for ego state at udp://{ego_host}:{ego_port}"
        )
        self.get_logger().info(
            f"Listening for camera frames at tcp://{camera_host}:{camera_port}"
        )

        for logical_id, topic in CAMERA_TOPICS.items():
            self.get_logger().info(
                f"{logical_id} -> {topic}"
            )

    def camera_server_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                connection, address = self.camera_server_socket.accept()
            except OSError:
                break

            self.get_logger().info(
                f"Camera Runtime connected from {address}"
            )

            try:
                with connection:
                    while not self.stop_event.is_set():
                        header_length = struct.unpack(
                            "!I",
                            read_exact(connection, 4),
                        )[0]

                        header_bytes = read_exact(
                            connection,
                            header_length,
                        )

                        image_length = struct.unpack(
                            "!Q",
                            read_exact(connection, 8),
                        )[0]

                        image_bytes = read_exact(
                            connection,
                            image_length,
                        )

                        header = json.loads(
                            header_bytes.decode("utf-8")
                        )

                        item = (header, image_bytes)

                        try:
                            self.camera_queue.put_nowait(item)
                        except queue.Full:
                            # Keep the most recent camera data.
                            try:
                                self.camera_queue.get_nowait()
                            except queue.Empty:
                                pass

                            try:
                                self.camera_queue.put_nowait(item)
                            except queue.Full:
                                pass

            except (
                ConnectionError,
                OSError,
                json.JSONDecodeError,
                struct.error,
            ) as exc:
                if not self.stop_event.is_set():
                    self.get_logger().warning(
                        f"Camera connection ended: {exc}"
                    )

    def poll_inputs(self) -> None:
        self.poll_ego_udp()
        self.poll_camera_queue()

    def poll_ego_udp(self) -> None:
        while True:
            try:
                payload, _ = self.ego_socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError as exc:
                self.get_logger().error(
                    f"Ego UDP receive error: {exc}"
                )
                break

            try:
                state = json.loads(payload.decode("utf-8"))
                self.publish_ego_state(state)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ) as exc:
                self.get_logger().warning(
                    f"Invalid ego-state packet: {exc}"
                )

    def poll_camera_queue(self) -> None:
        # Limit work per callback so camera traffic cannot starve ego state.
        for _ in range(16):
            try:
                header, image_bytes = self.camera_queue.get_nowait()
            except queue.Empty:
                break

            self.publish_camera_image(header, image_bytes)

    def publish_ego_state(self, state: dict[str, Any]) -> None:
        msg = EgoState()

        timestamp_us = int(state["timestamp_us"])

        msg.stamp.sec = timestamp_us // 1_000_000
        msg.stamp.nanosec = (
            timestamp_us % 1_000_000
        ) * 1000

        msg.frame_id = str(
            state.get("frame_id", "alpasim_local")
        )

        position = state["position"]
        msg.position.x = float(position["x"])
        msg.position.y = float(position["y"])
        msg.position.z = float(position["z"])

        orientation = state["orientation"]
        msg.orientation.x = float(orientation["x"])
        msg.orientation.y = float(orientation["y"])
        msg.orientation.z = float(orientation["z"])
        msg.orientation.w = float(orientation["w"])

        linear_velocity = state["linear_velocity"]
        msg.linear_velocity.x = float(linear_velocity["x"])
        msg.linear_velocity.y = float(linear_velocity["y"])
        msg.linear_velocity.z = float(linear_velocity["z"])

        angular_velocity = state["angular_velocity"]
        msg.angular_velocity.x = float(angular_velocity["x"])
        msg.angular_velocity.y = float(angular_velocity["y"])
        msg.angular_velocity.z = float(angular_velocity["z"])

        linear_acceleration = state["linear_acceleration"]
        msg.linear_acceleration.x = float(
            linear_acceleration["x"]
        )
        msg.linear_acceleration.y = float(
            linear_acceleration["y"]
        )
        msg.linear_acceleration.z = float(
            linear_acceleration["z"]
        )

        angular_acceleration = state["angular_acceleration"]
        msg.angular_acceleration.x = float(
            angular_acceleration["x"]
        )
        msg.angular_acceleration.y = float(
            angular_acceleration["y"]
        )
        msg.angular_acceleration.z = float(
            angular_acceleration["z"]
        )

        msg.speed = float(state["speed"])

        self.ego_publisher.publish(msg)
        self.ego_count += 1

    def publish_camera_image(
        self,
        header: dict[str, Any],
        image_bytes: bytes,
    ) -> None:
        logical_id = str(header["camera_logical_id"])

        publisher = self.camera_publishers.get(logical_id)

        if publisher is None:
            self.get_logger().warning(
                f"Unknown camera logical ID: {logical_id}"
            )
            return

        timestamp_us = int(header["end_timestamp_us"])

        msg = CompressedImage()
        msg.header.stamp.sec = timestamp_us // 1_000_000
        msg.header.stamp.nanosec = (
            timestamp_us % 1_000_000
        ) * 1000

        msg.header.frame_id = logical_id
        msg.format = detect_image_format(image_bytes)
        msg.data = image_bytes

        publisher.publish(msg)

        self.camera_counts[logical_id] += 1

        count = self.camera_counts[logical_id]

        if count == 1:
            self.get_logger().info(
                f"First {logical_id} frame: "
                f"format={msg.format}, "
                f"bytes={len(image_bytes)}, "
                f"t={timestamp_us / 1e6:.6f}s"
            )

    def destroy_node(self) -> None:
        self.stop_event.set()

        try:
            self.ego_socket.close()
        except OSError:
            pass

        try:
            self.camera_server_socket.close()
        except OSError:
            pass

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EgoStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
