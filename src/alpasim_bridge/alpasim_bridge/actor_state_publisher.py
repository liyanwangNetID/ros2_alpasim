#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
import queue
import socket
import struct
import threading
from collections import deque
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
    ActorState,
    ActorStateArray,
    ActorTrajectory,
    ActorTrajectoryArray,
    TrajectoryPoint,
)


def read_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly size bytes from a TCP socket."""
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = sock.recv(remaining)

        if not chunk:
            raise ConnectionError("Actor TCP connection closed")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def time_from_microseconds(timestamp_us: int) -> Time:
    """Convert an unsigned microsecond timestamp to builtin_interfaces/Time."""
    msg = Time()
    msg.sec = int(timestamp_us // 1_000_000)
    msg.nanosec = int((timestamp_us % 1_000_000) * 1000)
    return msg


def duration_from_microseconds(delta_us: int) -> Duration:
    """Convert signed microseconds into a normalized ROS Duration."""
    msg = Duration()

    # Python floor division gives a normalized representation:
    # -500000 us -> sec=-1, nanosec=500000000.
    seconds = delta_us // 1_000_000
    remaining_us = delta_us - seconds * 1_000_000

    msg.sec = int(seconds)
    msg.nanosec = int(remaining_us * 1000)

    return msg


def duration_from_seconds(seconds: float) -> Duration:
    """Convert floating-point seconds to builtin_interfaces/Duration."""
    total_us = int(round(seconds * 1_000_000.0))
    return duration_from_microseconds(total_us)


def populate_pose(ros_pose, state: dict[str, Any]) -> None:
    """Populate geometry_msgs/Pose from Runtime actor JSON."""
    position = state["position"]
    orientation = state["orientation"]

    ros_pose.position.x = float(position["x"])
    ros_pose.position.y = float(position["y"])
    ros_pose.position.z = float(position["z"])

    ros_pose.orientation.x = float(orientation["x"])
    ros_pose.orientation.y = float(orientation["y"])
    ros_pose.orientation.z = float(orientation["z"])
    ros_pose.orientation.w = float(orientation["w"])


def populate_vector3(ros_vector, values: dict[str, Any]) -> None:
    """Populate geometry_msgs/Vector3 from a JSON dictionary."""
    ros_vector.x = float(values["x"])
    ros_vector.y = float(values["y"])
    ros_vector.z = float(values["z"])


def make_trajectory_point(
    state: dict[str, Any],
    reference_timestamp_us: int,
) -> TrajectoryPoint:
    """Convert one Runtime actor point into a ROS trajectory point."""
    point = TrajectoryPoint()

    timestamp_us = int(state["timestamp_us"])

    point.stamp = time_from_microseconds(timestamp_us)
    point.time_from_reference = duration_from_microseconds(
        timestamp_us - reference_timestamp_us
    )

    populate_pose(point.pose, state)

    populate_vector3(
        point.linear_velocity,
        state["linear_velocity"],
    )
    populate_vector3(
        point.linear_acceleration,
        state["linear_acceleration"],
    )

    point.yaw = float(state["yaw"])
    point.yaw_rate = float(state["yaw_rate"])
    point.yaw_acceleration = float(state["yaw_acceleration"])
    point.speed = float(state["speed"])

    return point


def downsample_json_points(
    points: list[dict[str, Any]],
    minimum_period_us: int,
    max_points: int,
) -> list[dict[str, Any]]:
    """Downsample ordered points while preserving first and final samples."""
    if not points:
        return []

    ordered = sorted(
        points,
        key=lambda item: int(item["timestamp_us"]),
    )

    selected = [ordered[0]]
    last_timestamp_us = int(ordered[0]["timestamp_us"])

    for point in ordered[1:]:
        timestamp_us = int(point["timestamp_us"])

        if timestamp_us - last_timestamp_us >= minimum_period_us:
            selected.append(point)
            last_timestamp_us = timestamp_us

        if len(selected) >= max_points:
            break

    # Preserve the final available point when there is room and it was skipped.
    if (
        len(selected) < max_points
        and int(selected[-1]["timestamp_us"])
        != int(ordered[-1]["timestamp_us"])
    ):
        selected.append(ordered[-1])

    return selected[:max_points]


class ActorStatePublisher(Node):
    """Receive actor snapshots and publish actor ROS interfaces."""

    def __init__(self) -> None:
        super().__init__("actor_state_publisher")

        self.declare_parameter("actor_tcp_host", "127.0.0.1")
        self.declare_parameter("actor_tcp_port", 15002)

        self.declare_parameter(
            "actor_history_duration_sec",
            2.0,
        )
        self.declare_parameter(
            "actor_future_duration_sec",
            5.0,
        )
        self.declare_parameter(
            "actor_trajectory_sample_period_sec",
            0.5,
        )
        self.declare_parameter(
            "actor_trajectory_max_points",
            64,
        )
        self.declare_parameter(
            "actor_stale_timeout_sec",
            5.0,
        )

        self.declare_parameter("publish_actor_current", True)
        self.declare_parameter("publish_actor_history", True)
        self.declare_parameter(
            "publish_actor_ground_truth_future",
            True,
        )
        self.declare_parameter(
            "publish_actor_prediction_placeholder",
            True,
        )

        self.actor_tcp_host = str(
            self.get_parameter("actor_tcp_host").value
        )
        self.actor_tcp_port = int(
            self.get_parameter("actor_tcp_port").value
        )

        self.history_duration_sec = float(
            self.get_parameter(
                "actor_history_duration_sec"
            ).value
        )
        self.future_duration_sec = float(
            self.get_parameter(
                "actor_future_duration_sec"
            ).value
        )
        self.sample_period_sec = float(
            self.get_parameter(
                "actor_trajectory_sample_period_sec"
            ).value
        )
        self.max_points = int(
            self.get_parameter(
                "actor_trajectory_max_points"
            ).value
        )
        self.stale_timeout_sec = float(
            self.get_parameter(
                "actor_stale_timeout_sec"
            ).value
        )

        self.publish_current_enabled = bool(
            self.get_parameter("publish_actor_current").value
        )
        self.publish_history_enabled = bool(
            self.get_parameter("publish_actor_history").value
        )
        self.publish_future_enabled = bool(
            self.get_parameter(
                "publish_actor_ground_truth_future"
            ).value
        )
        self.publish_prediction_enabled = bool(
            self.get_parameter(
                "publish_actor_prediction_placeholder"
            ).value
        )

        if self.history_duration_sec < 0.0:
            raise ValueError(
                "actor_history_duration_sec must be non-negative"
            )

        if self.future_duration_sec < 0.0:
            raise ValueError(
                "actor_future_duration_sec must be non-negative"
            )

        if self.sample_period_sec <= 0.0:
            raise ValueError(
                "actor_trajectory_sample_period_sec must be positive"
            )

        if self.max_points <= 0:
            raise ValueError(
                "actor_trajectory_max_points must be positive"
            )

        self.history_duration_us = int(
            round(self.history_duration_sec * 1_000_000.0)
        )
        self.future_duration_us = int(
            round(self.future_duration_sec * 1_000_000.0)
        )
        self.sample_period_us = int(
            round(self.sample_period_sec * 1_000_000.0)
        )
        self.stale_timeout_us = int(
            round(self.stale_timeout_sec * 1_000_000.0)
        )

        calculated_history_points = (
            int(
                math.floor(
                    self.history_duration_sec
                    / self.sample_period_sec
                )
            )
            + 1
        )

        self.history_max_points = max(
            1,
            min(calculated_history_points, self.max_points),
        )

        actor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.current_publisher = self.create_publisher(
            ActorStateArray,
            "/alpasim/actors/current",
            actor_qos,
        )

        self.history_publisher = self.create_publisher(
            ActorTrajectoryArray,
            "/alpasim/actors/history",
            actor_qos,
        )

        self.future_publisher = self.create_publisher(
            ActorTrajectoryArray,
            "/alpasim/ground_truth/actors/future",
            actor_qos,
        )

        self.prediction_publisher = self.create_publisher(
            ActorTrajectoryArray,
            "/alpasim/prediction/actors",
            actor_qos,
        )

        # Per-track FIFO history buffers.
        self.history_buffers: dict[
            str,
            deque[dict[str, Any]]
        ] = {}

        # Last simulation timestamp at which each actor was observed.
        self.last_seen_timestamp_us: dict[str, int] = {}

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
            (self.actor_tcp_host, self.actor_tcp_port)
        )
        self.server_socket.listen(1)

        self.stop_event = threading.Event()

        self.receiver_thread = threading.Thread(
            target=self.actor_server_loop,
            name="alpasim-actor-tcp-server",
            daemon=True,
        )
        self.receiver_thread.start()

        # Only publish ROS messages from the executor thread.
        self.timer = self.create_timer(
            0.002,
            self.poll_actor_packets,
        )

        self.packet_count = 0

        self.get_logger().info(
            "Actor publisher listening at "
            f"tcp://{self.actor_tcp_host}:{self.actor_tcp_port}"
        )
        self.get_logger().info(
            "Actor trajectory configuration: "
            f"history={self.history_duration_sec:.3f}s, "
            f"future={self.future_duration_sec:.3f}s, "
            f"sample_period={self.sample_period_sec:.3f}s, "
            f"history_max_points={self.history_max_points}, "
            f"absolute_max_points={self.max_points}"
        )

    def actor_server_loop(self) -> None:
        """Accept Runtime connections and receive actor JSON packets."""
        while not self.stop_event.is_set():
            try:
                connection, address = self.server_socket.accept()
            except OSError:
                break

            self.get_logger().info(
                f"Actor Runtime connected from {address}"
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
                            self.packet_queue.put_nowait(packet)
                        except queue.Full:
                            # Keep the newest simulation state.
                            try:
                                self.packet_queue.get_nowait()
                            except queue.Empty:
                                pass

                            try:
                                self.packet_queue.put_nowait(packet)
                            except queue.Full:
                                pass

            except (
                ConnectionError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                struct.error,
            ) as exc:
                if not self.stop_event.is_set():
                    self.get_logger().warning(
                        f"Actor TCP connection ended: {exc}"
                    )

    def poll_actor_packets(self) -> None:
        """Process pending Runtime actor packets."""
        # Bound work per executor callback.
        for _ in range(8):
            try:
                packet = self.packet_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self.process_actor_packet(packet)
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                self.get_logger().error(
                    f"Invalid actor packet: {exc}"
                )

    def process_actor_packet(
        self,
        packet: dict[str, Any],
    ) -> None:
        """Update buffers and publish all configured actor interfaces."""
        if packet.get("message_type") != "actor_snapshot":
            self.get_logger().warning(
                "Ignoring unsupported actor message type: "
                f"{packet.get('message_type')}"
            )
            return

        reference_timestamp_us = int(packet["timestamp_us"])
        pose_frame_id = str(
            packet.get("pose_frame_id", "map")
        )
        dynamics_frame_id = str(
            packet.get("dynamics_frame_id", "map")
        )
        actors = packet.get("actors", [])

        # Update actor FIFO history.
        for actor in actors:
            track_id = str(actor["track_id"])
            current_state = actor["current_state"]

            buffer = self.history_buffers.get(track_id)

            if buffer is None:
                buffer = deque(
                    maxlen=self.history_max_points
                )
                self.history_buffers[track_id] = buffer

            current_timestamp_us = int(
                current_state["timestamp_us"]
            )

            # Avoid duplicate samples at the same timestamp.
            if (
                not buffer
                or int(buffer[-1]["state"]["timestamp_us"])
                != current_timestamp_us
            ):
                buffer.append(
                    {
                        "track_id": track_id,
                        "label_class": str(
                            actor["label_class"]
                        ),
                        "is_static": bool(
                            actor["is_static"]
                        ),
                        "dimensions": copy.deepcopy(
                            actor["dimensions"]
                        ),
                        "state": copy.deepcopy(
                            current_state
                        ),
                    }
                )

            self.last_seen_timestamp_us[track_id] = (
                reference_timestamp_us
            )

        self.remove_stale_actor_buffers(
            reference_timestamp_us
        )

        if self.publish_current_enabled:
            current_msg = self.build_current_message(
                reference_timestamp_us,
                pose_frame_id,
                dynamics_frame_id,
                actors,
            )
            self.current_publisher.publish(current_msg)

        if self.publish_history_enabled:
            history_msg = self.build_history_message(
                reference_timestamp_us,
                pose_frame_id,
                dynamics_frame_id,
            )
            self.history_publisher.publish(history_msg)

        future_msg = None

        if (
            self.publish_future_enabled
            or self.publish_prediction_enabled
        ):
            future_msg = self.build_future_message(
                reference_timestamp_us,
                pose_frame_id,
                dynamics_frame_id,
                actors,
            )

        if self.publish_future_enabled and future_msg is not None:
            self.future_publisher.publish(future_msg)

        if (
            self.publish_prediction_enabled
            and future_msg is not None
        ):
            prediction_msg = copy.deepcopy(future_msg)

            prediction_msg.source = (
                ActorTrajectoryArray
                .SOURCE_GROUND_TRUTH_PLACEHOLDER
            )
            prediction_msg.producer = (
                "ground_truth_placeholder"
            )
            prediction_msg.is_model_generated = False

            self.prediction_publisher.publish(
                prediction_msg
            )

        self.packet_count += 1

        if self.packet_count == 1:
            self.get_logger().info(
                "Published first actor packet: "
                f"actors={len(actors)}, "
                f"reference_time="
                f"{reference_timestamp_us / 1e6:.6f}s"
            )

    def remove_stale_actor_buffers(
        self,
        reference_timestamp_us: int,
    ) -> None:
        """Delete history buffers for actors absent for too long."""
        stale_ids = [
            track_id
            for track_id, last_seen_us
            in self.last_seen_timestamp_us.items()
            if (
                reference_timestamp_us - last_seen_us
                > self.stale_timeout_us
            )
        ]

        for track_id in stale_ids:
            self.last_seen_timestamp_us.pop(
                track_id,
                None,
            )
            self.history_buffers.pop(
                track_id,
                None,
            )

    def build_actor_state(
        self,
        actor: dict[str, Any],
    ) -> ActorState:
        """Convert one actor's current Runtime state."""
        msg = ActorState()

        msg.track_id = str(actor["track_id"])
        msg.label_class = str(actor["label_class"])
        msg.is_static = bool(actor["is_static"])

        dimensions = actor["dimensions"]
        populate_vector3(msg.dimensions, dimensions)

        state = actor["current_state"]
        populate_pose(msg.pose, state)

        populate_vector3(
            msg.linear_velocity,
            state["linear_velocity"],
        )
        populate_vector3(
            msg.linear_acceleration,
            state["linear_acceleration"],
        )

        msg.yaw = float(state["yaw"])
        msg.yaw_rate = float(state["yaw_rate"])
        msg.yaw_acceleration = float(
            state["yaw_acceleration"]
        )
        msg.speed = float(state["speed"])

        return msg

    def build_current_message(
        self,
        reference_timestamp_us: int,
        pose_frame_id: str,
        dynamics_frame_id: str,
        actors: list[dict[str, Any]],
    ) -> ActorStateArray:
        """Build current ground-truth actor snapshot."""
        msg = ActorStateArray()

        msg.stamp = time_from_microseconds(
            reference_timestamp_us
        )
        msg.pose_frame_id = pose_frame_id
        msg.dynamics_frame_id = dynamics_frame_id

        msg.actors = [
            self.build_actor_state(actor)
            for actor in actors
        ]

        return msg

    def build_actor_trajectory(
        self,
        track_id: str,
        label_class: str,
        is_static: bool,
        dimensions: dict[str, Any],
        points: list[dict[str, Any]],
        reference_timestamp_us: int,
    ) -> ActorTrajectory:
        """Build one ROS actor trajectory."""
        trajectory = ActorTrajectory()

        trajectory.track_id = track_id
        trajectory.label_class = label_class
        trajectory.is_static = is_static

        populate_vector3(
            trajectory.dimensions,
            dimensions,
        )

        trajectory.points = [
            make_trajectory_point(
                point,
                reference_timestamp_us,
            )
            for point in points
        ]

        return trajectory

    def initialize_trajectory_array(
        self,
        reference_timestamp_us: int,
        pose_frame_id: str,
        dynamics_frame_id: str,
        source: int,
        producer: str,
        is_model_generated: bool,
        requested_duration_sec: float,
    ) -> ActorTrajectoryArray:
        """Create common trajectory array metadata."""
        msg = ActorTrajectoryArray()

        msg.reference_stamp = time_from_microseconds(
            reference_timestamp_us
        )

        msg.pose_frame_id = pose_frame_id
        msg.dynamics_frame_id = dynamics_frame_id

        msg.source = source
        msg.producer = producer
        msg.is_model_generated = is_model_generated

        msg.requested_duration = duration_from_seconds(
            requested_duration_sec
        )
        msg.actual_duration = duration_from_seconds(0.0)
        msg.sampling_interval = duration_from_seconds(
            self.sample_period_sec
        )

        return msg

    def build_history_message(
        self,
        reference_timestamp_us: int,
        pose_frame_id: str,
        dynamics_frame_id: str,
    ) -> ActorTrajectoryArray:
        """Build ground-truth actor histories from FIFO buffers."""
        msg = self.initialize_trajectory_array(
            reference_timestamp_us,
            pose_frame_id,
            dynamics_frame_id,
            (
                ActorTrajectoryArray
                .SOURCE_GROUND_TRUTH_HISTORY
            ),
            "alpasim_ground_truth",
            False,
            self.history_duration_sec,
        )

        earliest_timestamp_us = reference_timestamp_us

        for track_id, buffer in self.history_buffers.items():
            if not buffer:
                continue

            valid_entries = [
                entry
                for entry in buffer
                if (
                    reference_timestamp_us
                    - self.history_duration_us
                    <= int(entry["state"]["timestamp_us"])
                    <= reference_timestamp_us
                )
            ]

            if not valid_entries:
                continue

            raw_points = [
                entry["state"]
                for entry in valid_entries
            ]

            points = downsample_json_points(
                raw_points,
                self.sample_period_us,
                self.max_points,
            )

            if not points:
                continue

            metadata = valid_entries[-1]

            trajectory = self.build_actor_trajectory(
                track_id=track_id,
                label_class=metadata["label_class"],
                is_static=metadata["is_static"],
                dimensions=metadata["dimensions"],
                points=points,
                reference_timestamp_us=reference_timestamp_us,
            )

            msg.trajectories.append(trajectory)

            earliest_timestamp_us = min(
                earliest_timestamp_us,
                int(points[0]["timestamp_us"]),
            )

        actual_duration_us = max(
            0,
            reference_timestamp_us - earliest_timestamp_us,
        )

        msg.actual_duration = duration_from_microseconds(
            actual_duration_us
        )

        return msg

    def build_future_message(
        self,
        reference_timestamp_us: int,
        pose_frame_id: str,
        dynamics_frame_id: str,
        actors: list[dict[str, Any]],
    ) -> ActorTrajectoryArray:
        """Build available future ground-truth trajectories."""
        msg = self.initialize_trajectory_array(
            reference_timestamp_us,
            pose_frame_id,
            dynamics_frame_id,
            (
                ActorTrajectoryArray
                .SOURCE_GROUND_TRUTH_FUTURE
            ),
            "alpasim_ground_truth",
            False,
            self.future_duration_sec,
        )

        requested_end_us = (
            reference_timestamp_us
            + self.future_duration_us
        )
        latest_timestamp_us = reference_timestamp_us

        for actor in actors:
            raw_future_points = actor.get(
                "available_future_points",
                [],
            )

            bounded_points = [
                point
                for point in raw_future_points
                if (
                    reference_timestamp_us
                    <= int(point["timestamp_us"])
                    <= requested_end_us
                )
            ]

            points = downsample_json_points(
                bounded_points,
                self.sample_period_us,
                self.max_points,
            )

            if not points:
                continue

            trajectory = self.build_actor_trajectory(
                track_id=str(actor["track_id"]),
                label_class=str(actor["label_class"]),
                is_static=bool(actor["is_static"]),
                dimensions=actor["dimensions"],
                points=points,
                reference_timestamp_us=reference_timestamp_us,
            )

            msg.trajectories.append(trajectory)

            latest_timestamp_us = max(
                latest_timestamp_us,
                int(points[-1]["timestamp_us"]),
            )

        msg.actual_duration = duration_from_microseconds(
            max(
                0,
                latest_timestamp_us - reference_timestamp_us,
            )
        )

        return msg

    def destroy_node(self) -> None:
        self.stop_event.set()

        try:
            self.server_socket.close()
        except OSError:
            pass

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActorStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
