#!/usr/bin/env python3

"""Cache and serve the complete AlpaSim recording GT ego trajectory."""

from __future__ import annotations

import copy
import json
import math
import queue
import socket
import struct
import threading
from bisect import bisect_left
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration, Time
from rclpy.node import Node

from alpasim_msgs.msg import (
    EgoTrajectory,
    TrajectoryPoint,
)
from alpasim_msgs.srv import (
    GetGroundTruthEgoTrajectory,
)


def read_exact(
    sock: socket.socket,
    size: int,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError(
                "GT trajectory TCP connection closed"
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def time_to_microseconds(value: Time) -> int:
    return (
        int(value.sec) * 1_000_000
        + int(value.nanosec) // 1000
    )


def duration_to_microseconds(
    value: Duration,
) -> int:
    return (
        int(value.sec) * 1_000_000
        + int(value.nanosec) // 1000
    )


def time_from_microseconds(value: int) -> Time:
    result = Time()
    result.sec = int(value // 1_000_000)
    result.nanosec = int(
        (value % 1_000_000) * 1000
    )
    return result


def duration_from_microseconds(
    value: int,
) -> Duration:
    result = Duration()

    seconds = value // 1_000_000
    remaining_us = value - seconds * 1_000_000

    result.sec = int(seconds)
    result.nanosec = int(remaining_us * 1000)
    return result


def interpolate_scalar(
    first: float,
    second: float,
    alpha: float,
) -> float:
    return (
        float(first)
        + alpha * (float(second) - float(first))
    )


def interpolate_vector(
    first: dict[str, Any],
    second: dict[str, Any],
    alpha: float,
) -> dict[str, float]:
    return {
        axis: interpolate_scalar(
            first[axis],
            second[axis],
            alpha,
        )
        for axis in ("x", "y", "z")
    }


def interpolate_quaternion(
    first: dict[str, Any],
    second: dict[str, Any],
    alpha: float,
) -> dict[str, float]:
    """Shortest-path normalized linear quaternion interpolation."""

    q0 = [
        float(first[key])
        for key in ("x", "y", "z", "w")
    ]
    q1 = [
        float(second[key])
        for key in ("x", "y", "z", "w")
    ]

    dot = sum(
        a * b
        for a, b in zip(q0, q1)
    )

    if dot < 0.0:
        q1 = [-value for value in q1]

    result = [
        (1.0 - alpha) * a + alpha * b
        for a, b in zip(q0, q1)
    ]

    norm = math.sqrt(
        sum(value * value for value in result)
    )

    if norm <= 1e-12:
        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0,
        }

    normalized = [
        value / norm
        for value in result
    ]

    return dict(
        zip(
            ("x", "y", "z", "w"),
            normalized,
        )
    )


def interpolate_point(
    first: dict[str, Any],
    second: dict[str, Any],
    timestamp_us: int,
) -> dict[str, Any]:
    t0 = int(first["timestamp_us"])
    t1 = int(second["timestamp_us"])

    if t1 <= t0:
        return copy.deepcopy(first)

    alpha = (
        (timestamp_us - t0)
        / float(t1 - t0)
    )
    alpha = min(max(alpha, 0.0), 1.0)

    velocity = interpolate_vector(
        first["linear_velocity"],
        second["linear_velocity"],
        alpha,
    )

    speed = math.sqrt(
        velocity["x"] ** 2
        + velocity["y"] ** 2
        + velocity["z"] ** 2
    )

    return {
        "timestamp_us": timestamp_us,
        "position": interpolate_vector(
            first["position"],
            second["position"],
            alpha,
        ),
        "orientation": interpolate_quaternion(
            first["orientation"],
            second["orientation"],
            alpha,
        ),
        "linear_velocity": velocity,
        "linear_acceleration": interpolate_vector(
            first["linear_acceleration"],
            second["linear_acceleration"],
            alpha,
        ),
        "yaw": interpolate_scalar(
            first["yaw"],
            second["yaw"],
            alpha,
        ),
        "yaw_rate": interpolate_scalar(
            first["yaw_rate"],
            second["yaw_rate"],
            alpha,
        ),
        "yaw_acceleration": interpolate_scalar(
            first["yaw_acceleration"],
            second["yaw_acceleration"],
            alpha,
        ),
        "speed": speed,
    }


def make_trajectory_point(
    source: dict[str, Any],
    reference_timestamp_us: int,
) -> TrajectoryPoint:
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

    message.pose.position.x = float(
        source["position"]["x"]
    )
    message.pose.position.y = float(
        source["position"]["y"]
    )
    message.pose.position.z = float(
        source["position"]["z"]
    )

    message.pose.orientation.x = float(
        source["orientation"]["x"]
    )
    message.pose.orientation.y = float(
        source["orientation"]["y"]
    )
    message.pose.orientation.z = float(
        source["orientation"]["z"]
    )
    message.pose.orientation.w = float(
        source["orientation"]["w"]
    )

    message.linear_velocity.x = float(
        source["linear_velocity"]["x"]
    )
    message.linear_velocity.y = float(
        source["linear_velocity"]["y"]
    )
    message.linear_velocity.z = float(
        source["linear_velocity"]["z"]
    )

    message.linear_acceleration.x = float(
        source["linear_acceleration"]["x"]
    )
    message.linear_acceleration.y = float(
        source["linear_acceleration"]["y"]
    )
    message.linear_acceleration.z = float(
        source["linear_acceleration"]["z"]
    )

    message.yaw = float(source["yaw"])
    message.yaw_rate = float(
        source["yaw_rate"]
    )
    message.yaw_acceleration = float(
        source["yaw_acceleration"]
    )
    message.speed = float(source["speed"])

    return message


class GroundTruthTrajectoryServer(Node):
    """Serve temporal slices of the recording GT ego trajectory."""

    def __init__(self) -> None:
        super().__init__(
            "ground_truth_trajectory_server"
        )

        self.declare_parameter(
            "tcp_host",
            "127.0.0.1",
        )
        self.declare_parameter(
            "tcp_port",
            15004,
        )
        self.declare_parameter(
            "service_name",
            (
                "/alpasim/navigation/"
                "get_ground_truth_ego_trajectory"
            ),
        )
        self.declare_parameter(
            "default_max_points",
            256,
        )

        self.tcp_host = str(
            self.get_parameter("tcp_host").value
        )
        self.tcp_port = int(
            self.get_parameter("tcp_port").value
        )
        self.service_name = str(
            self.get_parameter("service_name").value
        )
        self.default_max_points = int(
            self.get_parameter(
                "default_max_points"
            ).value
        )

        self.cached_packet: dict[str, Any] | None = None
        self.cached_timestamps: list[int] = []
        self.revision = 0
        self.cached_source_revision = 0

        self.packet_queue: queue.Queue[
            dict[str, Any]
        ] = queue.Queue(maxsize=1)

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
            (self.tcp_host, self.tcp_port)
        )
        self.server_socket.listen(1)

        self.stop_event = threading.Event()

        self.receiver_thread = threading.Thread(
            target=self.tcp_server_loop,
            daemon=True,
            name="ground-truth-trajectory-tcp-server",
        )
        self.receiver_thread.start()

        self.timer = self.create_timer(
            0.05,
            self.poll_packets,
        )

        self.service = self.create_service(
            GetGroundTruthEgoTrajectory,
            self.service_name,
            self.handle_request
        )

        self.get_logger().info(
            "GT trajectory server listening at "
            f"tcp://{self.tcp_host}:{self.tcp_port}"
        )
        self.get_logger().info(
            f"Service: {self.service_name}"
        )

    def tcp_server_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                connection, address = (
                    self.server_socket.accept()
                )
            except OSError:
                break

            self.get_logger().info(
                f"Runtime connected from {address}"
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
                        f"GT TCP connection ended: {exc}"
                    )

    def poll_packets(self) -> None:
        while True:
            try:
                packet = self.packet_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self.process_packet(packet)
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                self.get_logger().error(
                    f"Invalid GT trajectory packet: {exc}"
                )

    def process_packet(
        self,
        packet: dict[str, Any],
    ) -> None:
        if (
            packet.get("message_type")
            != "ground_truth_ego_trajectory"
        ):
            self.get_logger().warning(
                "Ignoring unsupported message type: "
                f"{packet.get('message_type')}"
            )
            return

        points = packet["points"]

        if not points:
            raise ValueError(
                "Ground-truth trajectory contains no points"
            )

        timestamps = [
            int(point["timestamp_us"])
            for point in points
        ]

        if any(
            second <= first
            for first, second in zip(
                timestamps,
                timestamps[1:],
            )
        ):
            raise ValueError(
                "Ground-truth timestamps are not increasing"
            )

        source_revision = int(
            packet.get("source_revision", 1)
        )

        if (
            self.cached_packet is not None
            and self.cached_packet["scene_id"]
            == packet["scene_id"]
            and self.cached_source_revision
            == source_revision
        ):
            self.get_logger().info(
                "Ignoring duplicate GT trajectory"
            )
            return

        self.cached_packet = packet
        self.cached_timestamps = timestamps
        self.cached_source_revision = source_revision
        self.revision += 1

        self.get_logger().info(
            "Cached recording GT ego trajectory: "
            f"scene={packet['scene_id']}, "
            f"revision={self.revision}, "
            f"points={len(points)}, "
            f"duration="
            f"{(timestamps[-1] - timestamps[0]) / 1e6:.3f}s"
        )

    def sample_at(
        self,
        timestamp_us: int,
    ) -> dict[str, Any]:
        """Interpolate one point from the cached source."""
        assert self.cached_packet is not None

        points = self.cached_packet["points"]
        timestamps = self.cached_timestamps

        index = bisect_left(
            timestamps,
            timestamp_us,
        )

        if index == 0:
            result = copy.deepcopy(points[0])
            result["timestamp_us"] = timestamp_us
            return result

        if index >= len(points):
            result = copy.deepcopy(points[-1])
            result["timestamp_us"] = timestamp_us
            return result

        if timestamps[index] == timestamp_us:
            return copy.deepcopy(points[index])

        return interpolate_point(
            points[index - 1],
            points[index],
            timestamp_us,
        )

    def select_query_timestamps(
        self,
        start_us: int,
        end_us: int,
        sample_interval_us: int,
        max_points: int,
    ) -> list[int]:
        """Build bounded, ordered output timestamps."""
        if sample_interval_us <= 0:
            timestamps = [
                timestamp_us
                for timestamp_us
                in self.cached_timestamps
                if start_us <= timestamp_us <= end_us
            ]

            if (
                not timestamps
                or timestamps[0] != start_us
            ):
                timestamps.insert(0, start_us)

            if (
                timestamps[-1] != end_us
                and len(timestamps) < max_points
            ):
                timestamps.append(end_us)

        else:
            timestamps = list(
                range(
                    start_us,
                    end_us + 1,
                    sample_interval_us,
                )
            )

            if (
                timestamps[-1] != end_us
                and len(timestamps) < max_points
            ):
                timestamps.append(end_us)

        if len(timestamps) <= max_points:
            return timestamps

        # Uniformly retain endpoints and intermediate samples.
        if max_points == 1:
            return [timestamps[0]]

        last_index = len(timestamps) - 1

        selected_indices = [
            round(
                index * last_index
                / float(max_points - 1)
            )
            for index in range(max_points)
        ]

        return [
            timestamps[index]
            for index in selected_indices
        ]

    def handle_request(
        self,
        request: GetGroundTruthEgoTrajectory.Request,
        response: GetGroundTruthEgoTrajectory.Response,
    ) -> GetGroundTruthEgoTrajectory.Response:
        if self.cached_packet is None:
            response.success = False
            response.not_modified = False
            response.message = (
                "No recording GT ego trajectory is loaded"
            )
            response.revision = 0
            return response

        source_start_us = self.cached_timestamps[0]
        source_end_us = self.cached_timestamps[-1]

        response.recording_start_stamp = (
            time_from_microseconds(source_start_us)
        )
        response.recording_end_stamp = (
            time_from_microseconds(source_end_us)
        )
        response.revision = self.revision

        if (
            request.known_revision != 0
            and request.known_revision == self.revision
        ):
            response.success = True
            response.not_modified = True
            response.message = (
                "Client already knows the current source revision"
            )
            return response

        reference_us = time_to_microseconds(
            request.reference_stamp
        )
        requested_duration_us = max(
            0,
            duration_to_microseconds(
                request.future_duration
            ),
        )

        requested_end_us = (
            reference_us + requested_duration_us
        )

        if (
            reference_us < source_start_us
            or reference_us > source_end_us
        ):
            response.success = False
            response.not_modified = False
            response.message = (
                "Reference timestamp is outside the "
                "recording trajectory range"
            )
            return response

        actual_end_us = min(
            requested_end_us,
            source_end_us,
        )

        sample_interval_us = (
            duration_to_microseconds(
                request.sampling_interval
            )
        )

        max_points = int(request.max_points)

        if max_points == 0:
            max_points = self.default_max_points

        max_points = max(max_points, 1)

        query_timestamps = (
            self.select_query_timestamps(
                start_us=reference_us,
                end_us=actual_end_us,
                sample_interval_us=sample_interval_us,
                max_points=max_points,
            )
        )

        trajectory = EgoTrajectory()
        trajectory.reference_stamp = (
            time_from_microseconds(reference_us)
        )
        trajectory.pose_frame_id = str(
            self.cached_packet["pose_frame_id"]
        )
        trajectory.dynamics_frame_id = str(
            self.cached_packet["dynamics_frame_id"]
        )
        trajectory.source = (
            EgoTrajectory
            .SOURCE_RECORDING_GROUND_TRUTH
        )
        trajectory.producer = (
            "alpasim_recording_ground_truth"
        )
        trajectory.is_model_generated = False
        trajectory.force_gt_active = False
        trajectory.requested_duration = (
            duration_from_microseconds(
                requested_duration_us
            )
        )
        trajectory.actual_duration = (
            duration_from_microseconds(
                actual_end_us - reference_us
            )
        )

        trajectory.points = [
            make_trajectory_point(
                self.sample_at(timestamp_us),
                reference_us,
            )
            for timestamp_us in query_timestamps
        ]

        response.success = True
        response.not_modified = False
        response.message = (
            "Returning recording GT future trajectory"
        )
        response.trajectory = trajectory

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
    node = GroundTruthTrajectoryServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
