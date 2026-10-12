#!/usr/bin/env python3
"""Receive, verify, cache, publish, and serve AlpaSim static occupancy."""
from __future__ import annotations

import hashlib
import io
import json
import queue
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from alpasim_msgs.msg import StaticOccupancy
from alpasim_msgs.srv import GetStaticOccupancy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

_HEADER_LENGTH = struct.Struct("!I")
_PAYLOAD_LENGTH = struct.Struct("!Q")
_REQUIRED_ARRAYS = (
    "grid_i", "grid_j", "min_z", "max_z", "point_count", "flags"
)


@dataclass(frozen=True)
class ReceivedStaticOccupancy:
    header: dict[str, Any]
    arrays: dict[str, np.ndarray]


def read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Static occupancy TCP connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_static_occupancy(
    header_bytes: bytes,
    payload: bytes,
) -> ReceivedStaticOccupancy:
    header = json.loads(header_bytes.decode("utf-8"))
    if header.get("message_type") != "static_occupancy":
        raise ValueError("Unsupported static occupancy message type")
    if header.get("representation") != "single_interval_2p5d_static_occupancy":
        raise ValueError("Unsupported static occupancy representation")
    if bool(header.get("semantic_labels")):
        raise ValueError("Static occupancy must not contain semantic labels")
    if int(header.get("payload_size", -1)) != len(payload):
        raise ValueError("Static occupancy payload size mismatch")
    checksum = hashlib.sha256(payload).hexdigest()
    if header.get("payload_sha256") != checksum:
        raise ValueError("Static occupancy checksum mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        missing = sorted(set(_REQUIRED_ARRAYS) - set(source.files))
        if missing:
            raise ValueError(f"Static occupancy arrays missing: {missing}")
        arrays = {name: np.asarray(source[name]).copy() for name in _REQUIRED_ARRAYS}
    count = len(arrays["grid_i"])
    if count == 0 or any(len(value) != count for value in arrays.values()):
        raise ValueError("Static occupancy arrays have inconsistent lengths")
    if count != int(header.get("column_count", -1)):
        raise ValueError("Static occupancy column count mismatch")
    if not np.isfinite(arrays["min_z"]).all() or not np.isfinite(arrays["max_z"]).all():
        raise ValueError("Static occupancy contains non-finite Z values")
    if np.any(arrays["max_z"] < arrays["min_z"]):
        raise ValueError("Static occupancy contains invalid vertical intervals")
    if float(header.get("grid_size_xy_m", 0.0)) <= 0.0:
        raise ValueError("Static occupancy grid size must be positive")
    return ReceivedStaticOccupancy(header=header, arrays=arrays)


def make_message(
    received: ReceivedStaticOccupancy,
    revision: int,
) -> StaticOccupancy:
    header = received.header
    arrays = received.arrays
    message = StaticOccupancy()
    message.schema_version = str(header["schema_version"])
    message.scene_id = str(header["scene_id"])
    message.scene_uuid = str(header["scene_uuid"])
    message.frame_id = str(header["frame_id"])
    message.representation = str(header["representation"])
    message.semantic_labels = bool(header["semantic_labels"])
    message.grid_size_xy_m = float(header["grid_size_xy_m"])
    message.revision = int(revision)
    message.payload_sha256 = str(header["payload_sha256"])
    message.grid_i = arrays["grid_i"].astype(np.int32, copy=False).tolist()
    message.grid_j = arrays["grid_j"].astype(np.int32, copy=False).tolist()
    message.min_z = arrays["min_z"].astype(np.float32, copy=False).tolist()
    message.max_z = arrays["max_z"].astype(np.float32, copy=False).tolist()
    message.point_count = arrays["point_count"].astype(np.uint32, copy=False).tolist()
    message.flags = arrays["flags"].astype(np.uint16, copy=False).tolist()
    return message


@dataclass
class PendingSnapshot:
    received: ReceivedStaticOccupancy
    ready: threading.Event
    result: StaticOccupancy | None = None
    error: BaseException | None = None


class StaticOccupancyServer(Node):
    """Provide a startup-barrier TCP receiver and ROS static occupancy API."""

    def __init__(self) -> None:
        super().__init__("static_occupancy_server")
        self.declare_parameter("tcp_host", "0.0.0.0")
        self.declare_parameter("tcp_port", 15006)
        self.declare_parameter("service_name", "/alpasim/static_occupancy/get")
        self.declare_parameter("topic_name", "/alpasim/static_occupancy")
        self.declare_parameter("cache_commit_timeout_s", 15.0)
        self.tcp_host = str(self.get_parameter("tcp_host").value)
        self.tcp_port = int(self.get_parameter("tcp_port").value)
        self.cache_commit_timeout_s = float(
            self.get_parameter("cache_commit_timeout_s").value
        )
        service_name = str(self.get_parameter("service_name").value)
        topic_name = str(self.get_parameter("topic_name").value)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(StaticOccupancy, topic_name, qos)
        self.service = self.create_service(
            GetStaticOccupancy, service_name, self.handle_request
        )
        self.pending: queue.Queue[PendingSnapshot] = queue.Queue(maxsize=1)
        self.cache_lock = threading.RLock()
        self.cached_message: StaticOccupancy | None = None
        self.revision = 0
        self.timer = self.create_timer(0.01, self.commit_pending)
        self.stop_event = threading.Event()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.tcp_host, self.tcp_port))
        self.server_socket.listen(4)
        self.receiver_thread = threading.Thread(
            target=self.server_loop,
            daemon=True,
            name="static-occupancy-tcp-server",
        )
        self.receiver_thread.start()
        self.get_logger().info(
            f"Static occupancy server listening at tcp://{self.tcp_host}:{self.tcp_port}"
        )
        self.get_logger().info(f"Service: {service_name}; topic: {topic_name}")

    def server_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                connection, address = self.server_socket.accept()
            except OSError:
                return
            self.get_logger().info(f"Runtime connected from {address}")
            try:
                with connection:
                    header_size = _HEADER_LENGTH.unpack(read_exact(connection, 4))[0]
                    payload_size = _PAYLOAD_LENGTH.unpack(read_exact(connection, 8))[0]
                    if header_size <= 0 or header_size > 1_000_000:
                        raise ValueError(f"Invalid header size: {header_size}")
                    if payload_size <= 0 or payload_size > 1_000_000_000:
                        raise ValueError(f"Invalid payload size: {payload_size}")
                    received = decode_static_occupancy(
                        read_exact(connection, header_size),
                        read_exact(connection, payload_size),
                    )
                    pending = PendingSnapshot(received, threading.Event())
                    self.pending.put(pending, timeout=self.cache_commit_timeout_s)
                    if not pending.ready.wait(self.cache_commit_timeout_s):
                        raise TimeoutError("ROS executor did not commit static occupancy")
                    if pending.error is not None:
                        raise pending.error
                    assert pending.result is not None
                    ack = {
                        "message_type": "static_occupancy_ack",
                        "status": "ready",
                        "scene_id": pending.result.scene_id,
                        "scene_uuid": pending.result.scene_uuid,
                        "schema_version": pending.result.schema_version,
                        "payload_sha256": pending.result.payload_sha256,
                    }
                    encoded = json.dumps(ack, separators=(",", ":")).encode("utf-8")
                    connection.sendall(_HEADER_LENGTH.pack(len(encoded)) + encoded)
            except (
                ConnectionError, OSError, TimeoutError, ValueError,
                json.JSONDecodeError, struct.error, queue.Full,
            ) as exc:
                if not self.stop_event.is_set():
                    self.get_logger().error(f"Static occupancy receive failed: {exc}")

    def commit_pending(self) -> None:
        try:
            pending = self.pending.get_nowait()
        except queue.Empty:
            return
        try:
            checksum = str(pending.received.header["payload_sha256"])
            with self.cache_lock:
                duplicate = (
                    self.cached_message is not None
                    and self.cached_message.payload_sha256 == checksum
                )
                if duplicate:
                    message = self.cached_message
                else:
                    self.revision += 1
                    message = make_message(pending.received, self.revision)
                    self.cached_message = message
                    self.publisher.publish(message)
            pending.result = message
            if duplicate:
                self.get_logger().info(
                    f"Confirmed duplicate static occupancy: scene={message.scene_id}, "
                    f"revision={message.revision}"
                )
            else:
                self.get_logger().info(
                    f"Cached static occupancy: scene={message.scene_id}, "
                    f"revision={message.revision}, columns={len(message.grid_i)}"
                )
        except BaseException as exc:
            pending.error = exc
        finally:
            pending.ready.set()
            self.pending.task_done()

    def handle_request(
        self,
        request: GetStaticOccupancy.Request,
        response: GetStaticOccupancy.Response,
    ) -> GetStaticOccupancy.Response:
        with self.cache_lock:
            message = self.cached_message
            if message is None:
                response.success = False
                response.message = "No static occupancy is loaded"
                return response
            if request.requested_scene_id and request.requested_scene_id != message.scene_id:
                response.success = False
                response.message = "Requested scene does not match cached static occupancy"
                response.revision = message.revision
                return response
            response.success = True
            response.revision = message.revision
            if request.known_revision == message.revision and request.known_revision != 0:
                response.not_modified = True
                response.message = "Client already knows the current revision"
                return response
            response.not_modified = False
            response.message = "Returning current static occupancy"
            response.static_occupancy = message
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
    node = StaticOccupancyServer()
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
