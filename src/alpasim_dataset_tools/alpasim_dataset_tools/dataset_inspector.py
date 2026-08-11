#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from alpasim_msgs.msg import (
    ActorStateArray,
    ActorTrajectoryArray,
    EgoState,
    EgoTrajectory,
    Route,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String


CAMERA_NAMES = (
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right",
)


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def duration_to_ns(duration: Any) -> int:
    return int(duration.sec) * 1_000_000_000 + int(duration.nanosec)


class DatasetInspector(Node):
    """Inspect AlpaSim streams before implementing the final recorder."""

    def __init__(self) -> None:
        super().__init__("alpasim_dataset_inspector")

        self.declare_parameter(
            "output_directory",
            "/tmp/alpasim_dataset_inspection",
        )
        self.declare_parameter("report_period_sec", 5.0)

        self.output_directory = Path(
            str(self.get_parameter("output_directory").value)
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.report_period_sec = float(
            self.get_parameter("report_period_sec").value
        )

        self._stats_lock = threading.Lock()
        self._subscriptions: list[Any] = []

        self.message_counts: dict[str, int] = defaultdict(int)
        self.first_ros_stamp_ns: dict[str, int] = {}
        self.last_ros_stamp_ns: dict[str, int] = {}
        self.first_wall_time: dict[str, float] = {}
        self.latest_summary: dict[str, dict[str, Any]] = {}
        self.saved_calibrations: set[str] = set()
        self._camera_timestamps_ns: dict[str, list[int]] = {
            camera_name: [] for camera_name in CAMERA_NAMES
        }

        # This block was missing in the previous version.
        self._camera_callback_groups = {
            camera_name: MutuallyExclusiveCallbackGroup()
            for camera_name in CAMERA_NAMES
        }

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.transient_local_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._create_camera_subscriptions()
        self._create_dynamic_subscriptions()
        self.report_timer = self.create_timer(
            self.report_period_sec,
            self.print_report,
        )

        self.get_logger().info("AlpaSim dataset inspector started")
        self.get_logger().info(
            f"Inspection output: {self.output_directory}"
        )

    def _create_camera_subscriptions(self) -> None:
        for camera_name in CAMERA_NAMES:
            image_topic = f"/alpasim/camera/{camera_name}/image"
            calibration_topic = (
                f"/alpasim/camera/{camera_name}/calibration"
            )

            image_subscription = self.create_subscription(
                Image,
                image_topic,
                lambda msg, topic=image_topic, name=camera_name: (
                    self.image_callback(msg, topic, name)
                ),
                self.sensor_qos,
                callback_group=self._camera_callback_groups[camera_name],
            )
            calibration_subscription = self.create_subscription(
                String,
                calibration_topic,
                lambda msg, topic=calibration_topic, name=camera_name: (
                    self.calibration_callback(msg, topic, name)
                ),
                self.transient_local_qos,
            )
            self._subscriptions.extend(
                [image_subscription, calibration_subscription]
            )

    def _create_dynamic_subscriptions(self) -> None:
        topic_specs = [
            ("/clock", Clock, self.clock_callback),
            ("/alpasim/ego_state", EgoState, self.ego_state_callback),
            (
                "/alpasim/ego/executed_path",
                EgoTrajectory,
                self.ego_trajectory_callback,
            ),
            (
                "/alpasim/ground_truth/ego/future_trajectory",
                EgoTrajectory,
                self.ego_trajectory_callback,
            ),
            (
                "/alpasim/planning/ego/trajectory",
                EgoTrajectory,
                self.ego_trajectory_callback,
            ),
            (
                "/alpasim/actors/current",
                ActorStateArray,
                self.actor_state_callback,
            ),
            (
                "/alpasim/actors/history",
                ActorTrajectoryArray,
                self.actor_trajectory_callback,
            ),
            (
                "/alpasim/ground_truth/actors/future",
                ActorTrajectoryArray,
                self.actor_trajectory_callback,
            ),
            ("/alpasim/route/map", Route, self.route_callback),
            (
                "/alpasim/route/model_input",
                Route,
                self.route_callback,
            ),
        ]

        for topic, message_type, callback in topic_specs:
            subscription = self.create_subscription(
                message_type,
                topic,
                lambda msg, topic_name=topic, cb=callback: cb(
                    msg, topic_name
                ),
                self.reliable_qos,
            )
            self._subscriptions.append(subscription)

    def register_message(
        self,
        topic: str,
        ros_stamp_ns: int | None,
    ) -> None:
        with self._stats_lock:
            self.message_counts[topic] += 1
            if topic not in self.first_wall_time:
                self.first_wall_time[topic] = time.monotonic()
            if ros_stamp_ns is not None:
                if topic not in self.first_ros_stamp_ns:
                    self.first_ros_stamp_ns[topic] = ros_stamp_ns
                self.last_ros_stamp_ns[topic] = ros_stamp_ns

    def calibration_callback(
        self,
        msg: String,
        topic: str,
        camera_name: str,
    ) -> None:
        self.register_message(topic, None)

        with self._stats_lock:
            if camera_name in self.saved_calibrations:
                return

        try:
            calibration = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f"Invalid calibration JSON for {camera_name}: {exc}"
            )
            return

        output_path = (
            self.output_directory
            / f"{camera_name}_calibration.json"
        )
        output_path.write_text(
            json.dumps(calibration, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        available_camera = calibration.get("available_camera", {})

        with self._stats_lock:
            self.latest_summary[topic] = {
                "camera_name": camera_name,
                "logical_id": available_camera.get("logical_id"),
                "top_level_keys": sorted(calibration.keys()),
                "available_camera_keys": sorted(
                    available_camera.keys()
                ),
                "json_character_count": len(msg.data),
                "saved_path": str(output_path),
            }
            self.saved_calibrations.add(camera_name)

        self.get_logger().info(
            f"Saved calibration for {camera_name}: {output_path}"
        )

    def image_callback(
        self,
        msg: Image,
        topic: str,
        camera_name: str,
    ) -> None:
        stamp_ns = stamp_to_ns(msg.header.stamp)
        self.register_message(topic, stamp_ns)

        with self._stats_lock:
            self._camera_timestamps_ns[camera_name].append(stamp_ns)
            self.latest_summary[topic] = {
                "camera_name": camera_name,
                "stamp_ns": stamp_ns,
                "frame_id": msg.header.frame_id,
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": msg.encoding,
                "is_bigendian": int(msg.is_bigendian),
                "step": int(msg.step),
                "data_bytes": len(msg.data),
            }

    def clock_callback(self, msg: Clock, topic: str) -> None:
        stamp_ns = stamp_to_ns(msg.clock)
        self.register_message(topic, stamp_ns)
        with self._stats_lock:
            self.latest_summary[topic] = {
                "simulation_time_ns": stamp_ns
            }

    def ego_state_callback(self, msg: EgoState, topic: str) -> None:
        stamp_ns = stamp_to_ns(msg.stamp)
        self.register_message(topic, stamp_ns)
        with self._stats_lock:
            self.latest_summary[topic] = {
                "stamp_ns": stamp_ns,
                "pose_frame_id": msg.pose_frame_id,
                "child_frame_id": msg.child_frame_id,
                "dynamics_frame_id": msg.dynamics_frame_id,
                "speed": float(msg.speed),
            }

    def ego_trajectory_callback(
        self,
        msg: EgoTrajectory,
        topic: str,
    ) -> None:
        reference_stamp_ns = stamp_to_ns(msg.reference_stamp)

        # executed_path has a fixed clip-start reference stamp, so use
        # the newest point timestamp when estimating publication rate.
        if topic == "/alpasim/ego/executed_path" and msg.points:
            frequency_stamp_ns = stamp_to_ns(msg.points[-1].stamp)
        else:
            frequency_stamp_ns = reference_stamp_ns

        self.register_message(topic, frequency_stamp_ns)

        first_offset_ns = None
        last_offset_ns = None
        if msg.points:
            first_offset_ns = duration_to_ns(
                msg.points[0].time_from_reference
            )
            last_offset_ns = duration_to_ns(
                msg.points[-1].time_from_reference
            )

        with self._stats_lock:
            self.latest_summary[topic] = {
                "reference_stamp_ns": reference_stamp_ns,
                "pose_frame_id": msg.pose_frame_id,
                "dynamics_frame_id": msg.dynamics_frame_id,
                "source": int(msg.source),
                "producer": msg.producer,
                "is_model_generated": bool(msg.is_model_generated),
                "force_gt_active": bool(msg.force_gt_active),
                "point_count": len(msg.points),
                "first_offset_ns": first_offset_ns,
                "last_offset_ns": last_offset_ns,
            }

    def actor_state_callback(
        self,
        msg: ActorStateArray,
        topic: str,
    ) -> None:
        stamp_ns = stamp_to_ns(msg.stamp)
        self.register_message(topic, stamp_ns)
        with self._stats_lock:
            self.latest_summary[topic] = {
                "stamp_ns": stamp_ns,
                "pose_frame_id": msg.pose_frame_id,
                "dynamics_frame_id": msg.dynamics_frame_id,
                "actor_count": len(msg.actors),
            }

    def actor_trajectory_callback(
        self,
        msg: ActorTrajectoryArray,
        topic: str,
    ) -> None:
        stamp_ns = stamp_to_ns(msg.reference_stamp)
        self.register_message(topic, stamp_ns)
        total_points = sum(
            len(trajectory.points)
            for trajectory in msg.trajectories
        )
        with self._stats_lock:
            self.latest_summary[topic] = {
                "reference_stamp_ns": stamp_ns,
                "pose_frame_id": msg.pose_frame_id,
                "dynamics_frame_id": msg.dynamics_frame_id,
                "source": int(msg.source),
                "producer": msg.producer,
                "is_model_generated": bool(msg.is_model_generated),
                "trajectory_count": len(msg.trajectories),
                "total_point_count": total_points,
            }

    def route_callback(self, msg: Route, topic: str) -> None:
        stamp_ns = stamp_to_ns(msg.reference_stamp)
        self.register_message(topic, stamp_ns)
        valid_point_count = sum(1 for point in msg.points if point.valid)
        with self._stats_lock:
            self.latest_summary[topic] = {
                "reference_stamp_ns": stamp_ns,
                "frame_id": msg.frame_id,
                "source_frame_id": msg.source_frame_id,
                "generator_type": int(msg.generator_type),
                "producer": msg.producer,
                "sequence": int(msg.sequence),
                "lookahead_distance": float(msg.lookahead_distance),
                "expected_point_count": int(
                    msg.expected_point_count
                ),
                "point_count": len(msg.points),
                "valid_point_count": valid_point_count,
            }

    def camera_interval_statistics(
        self,
        camera_name: str,
    ) -> dict[str, Any]:
        with self._stats_lock:
            timestamps = sorted(
                self._camera_timestamps_ns[camera_name]
            )

        unique_timestamps = sorted(set(timestamps))
        result: dict[str, Any] = {
            "frame_count": len(timestamps),
            "unique_frame_count": len(unique_timestamps),
            "duplicate_timestamp_count": (
                len(timestamps) - len(unique_timestamps)
            ),
            "minimum_interval_ms": None,
            "median_interval_ms": None,
            "maximum_interval_ms": None,
            "intervals_over_150ms": 0,
            "estimated_missing_100ms_steps": 0,
        }
        if len(unique_timestamps) < 2:
            return result

        intervals_ns = sorted(
            current - previous
            for previous, current in zip(
                unique_timestamps[:-1],
                unique_timestamps[1:],
            )
        )
        middle = len(intervals_ns) // 2
        if len(intervals_ns) % 2 == 0:
            median_ns = (
                intervals_ns[middle - 1] + intervals_ns[middle]
            ) / 2.0
        else:
            median_ns = float(intervals_ns[middle])

        result.update(
            {
                "minimum_interval_ms": min(intervals_ns) / 1e6,
                "median_interval_ms": median_ns / 1e6,
                "maximum_interval_ms": max(intervals_ns) / 1e6,
                "intervals_over_150ms": sum(
                    interval > 150_000_000
                    for interval in intervals_ns
                ),
                "estimated_missing_100ms_steps": sum(
                    max(0, round(interval / 100_000_000) - 1)
                    for interval in intervals_ns
                ),
            }
        )
        return result

    def camera_sync_statistics(self) -> dict[str, Any]:
        with self._stats_lock:
            timestamp_sets = {
                camera_name: set(
                    self._camera_timestamps_ns[camera_name]
                )
                for camera_name in CAMERA_NAMES
            }

        union_timestamps = set().union(*timestamp_sets.values())
        common_timestamps = set.intersection(
            *timestamp_sets.values()
        )
        match_ratio = (
            len(common_timestamps) / len(union_timestamps)
            if union_timestamps
            else None
        )
        return {
            "union_timestamp_count": len(union_timestamps),
            "four_camera_matched_count": len(common_timestamps),
            "four_camera_match_ratio": match_ratio,
            "missing_by_camera": {
                camera_name: len(
                    union_timestamps - timestamp_sets[camera_name]
                )
                for camera_name in CAMERA_NAMES
            },
        }

    def estimated_hz(self, topic: str) -> float | None:
        with self._stats_lock:
            count = self.message_counts.get(topic, 0)
            first_stamp = self.first_ros_stamp_ns.get(topic)
            last_stamp = self.last_ros_stamp_ns.get(topic)
            first_wall = self.first_wall_time.get(topic)

        if count < 2:
            return None
        if (
            first_stamp is not None
            and last_stamp is not None
            and last_stamp > first_stamp
        ):
            duration_sec = (last_stamp - first_stamp) / 1e9
            return (count - 1) / duration_sec
        if first_wall is None:
            return None
        wall_duration = time.monotonic() - first_wall
        return (count - 1) / wall_duration if wall_duration > 0 else None

    def print_report(self) -> None:
        with self._stats_lock:
            counts = dict(self.message_counts)
            latest_summary = json.loads(
                json.dumps(self.latest_summary)
            )
            saved_calibrations = sorted(self.saved_calibrations)

        report = {
            "message_counts": counts,
            "estimated_hz": {
                topic: self.estimated_hz(topic)
                for topic in sorted(counts)
            },
            "latest_summary": latest_summary,
            "saved_calibrations": saved_calibrations,
            "camera_interval_statistics": {
                camera_name: self.camera_interval_statistics(
                    camera_name
                )
                for camera_name in CAMERA_NAMES
            },
            "camera_sync_statistics": self.camera_sync_statistics(),
        }

        report_path = self.output_directory / "inspection_report.json"
        temporary_path = report_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(report_path)

        self.get_logger().info(
            "Inspection report: "
            f"topics={len(counts)}, "
            f"calibrations={len(saved_calibrations)}/"
            f"{len(CAMERA_NAMES)}, "
            f"path={report_path}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetInspector()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.print_report()
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
