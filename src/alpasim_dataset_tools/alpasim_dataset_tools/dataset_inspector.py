#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import rclpy
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
    """Convert builtin_interfaces/Time to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class DatasetInspector(Node):
    """Inspect AlpaSim topics before implementing dataset recording."""

    def __init__(self) -> None:
        super().__init__("alpasim_dataset_inspector")

        self.declare_parameter(
            "output_directory",
            "/tmp/alpasim_dataset_inspection",
        )
        self.declare_parameter(
            "report_period_sec",
            5.0,
        )

        self.output_directory = Path(
            str(
                self.get_parameter(
                    "output_directory"
                ).value
            )
        )
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.report_period_sec = float(
            self.get_parameter(
                "report_period_sec"
            ).value
        )

        self.message_counts: dict[str, int] = defaultdict(int)
        self.first_ros_stamp_ns: dict[str, int] = {}
        self.last_ros_stamp_ns: dict[str, int] = {}
        self.first_wall_time: dict[str, float] = {}
        self.latest_summary: dict[str, dict[str, Any]] = {}

        self.saved_calibrations: set[str] = set()
        self._subscriptions = []

        # sensor_msgs/Image publishers commonly use sensor-data QoS.
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

        self.get_logger().info(
            "AlpaSim dataset inspector started"
        )
        self.get_logger().info(
            f"Inspection output: {self.output_directory}"
        )

    def _create_camera_subscriptions(self) -> None:
        for camera_name in CAMERA_NAMES:
            image_topic = (
                f"/alpasim/camera/{camera_name}/image"
            )
            calibration_topic = (
                f"/alpasim/camera/"
                f"{camera_name}/calibration"
            )

            image_subscription = self.create_subscription(
                Image,
                image_topic,
                lambda msg, topic=image_topic, name=camera_name:
                    self.image_callback(msg, topic, name),
                self.sensor_qos,
            )

            calibration_subscription = self.create_subscription(
                String,
                calibration_topic,
                lambda msg, topic=calibration_topic, name=camera_name:
                    self.calibration_callback(msg, topic, name),
                self.transient_local_qos,
            )

            self._subscriptions.extend(
                [
                    image_subscription,
                    calibration_subscription,
                ]
            )

    def _create_dynamic_subscriptions(self) -> None:
        topic_specs = [
            (
                "/clock",
                Clock,
                self.clock_callback,
            ),
            (
                "/alpasim/ego_state",
                EgoState,
                self.ego_state_callback,
            ),
            (
                "/alpasim/ego/executed_path",
                EgoTrajectory,
                self.ego_trajectory_callback,
            ),
            (
                "/alpasim/ground_truth/ego/"
                "future_trajectory",
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
            (
                "/alpasim/route/map",
                Route,
                self.route_callback,
            ),
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
                lambda msg, topic_name=topic, cb=callback:
                    cb(msg, topic_name),
                self.reliable_qos,
            )
            self._subscriptions.append(subscription)

    def register_message(
        self,
        topic: str,
        ros_stamp_ns: int | None,
    ) -> None:
        self.message_counts[topic] += 1

        if topic not in self.first_wall_time:
            self.first_wall_time[topic] = time.monotonic()

        if ros_stamp_ns is None:
            return

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

        if camera_name in self.saved_calibrations:
            return

        try:
            calibration = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f"Invalid calibration JSON for "
                f"{camera_name}: {exc}"
            )
            return

        output_path = (
            self.output_directory
            / f"{camera_name}_calibration.json"
        )

        output_path.write_text(
            json.dumps(
                calibration,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        available_camera = calibration.get(
            "available_camera",
            {},
        )

        self.latest_summary[topic] = {
            "camera_name": camera_name,
            "logical_id": available_camera.get(
                "logical_id"
            ),
            "top_level_keys": sorted(
                calibration.keys()
            ),
            "available_camera_keys": sorted(
                available_camera.keys()
            ),
            "json_character_count": len(msg.data),
            "saved_path": str(output_path),
        }

        self.saved_calibrations.add(camera_name)

        self.get_logger().info(
            f"Saved calibration for {camera_name}: "
            f"{output_path}"
        )

    def image_callback(
        self,
        msg: Image,
        topic: str,
        camera_name: str,
    ) -> None:
        stamp_ns = stamp_to_ns(msg.header.stamp)
        self.register_message(topic, stamp_ns)

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

    def clock_callback(
        self,
        msg: Clock,
        topic: str,
    ) -> None:
        stamp_ns = stamp_to_ns(msg.clock)
        self.register_message(topic, stamp_ns)

        self.latest_summary[topic] = {
            "simulation_time_ns": stamp_ns,
        }

    def ego_state_callback(
        self,
        msg: EgoState,
        topic: str,
    ) -> None:
        stamp_ns = stamp_to_ns(msg.stamp)
        self.register_message(topic, stamp_ns)

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
        stamp_ns = stamp_to_ns(msg.reference_stamp)
        self.register_message(topic, stamp_ns)

        first_offset_ns = None
        last_offset_ns = None

        if msg.points:
            first_offset_ns = (
                int(msg.points[0].time_from_reference.sec)
                * 1_000_000_000
                + int(
                    msg.points[0]
                    .time_from_reference.nanosec
                )
            )
            last_offset_ns = (
                int(msg.points[-1].time_from_reference.sec)
                * 1_000_000_000
                + int(
                    msg.points[-1]
                    .time_from_reference.nanosec
                )
            )

        self.latest_summary[topic] = {
            "reference_stamp_ns": stamp_ns,
            "pose_frame_id": msg.pose_frame_id,
            "dynamics_frame_id": msg.dynamics_frame_id,
            "source": int(msg.source),
            "producer": msg.producer,
            "is_model_generated": bool(
                msg.is_model_generated
            ),
            "force_gt_active": bool(
                msg.force_gt_active
            ),
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

        self.latest_summary[topic] = {
            "reference_stamp_ns": stamp_ns,
            "pose_frame_id": msg.pose_frame_id,
            "dynamics_frame_id": msg.dynamics_frame_id,
            "source": int(msg.source),
            "producer": msg.producer,
            "is_model_generated": bool(
                msg.is_model_generated
            ),
            "trajectory_count": len(
                msg.trajectories
            ),
            "total_point_count": total_points,
        }

    def route_callback(
        self,
        msg: Route,
        topic: str,
    ) -> None:
        stamp_ns = stamp_to_ns(msg.reference_stamp)
        self.register_message(topic, stamp_ns)

        valid_points = [
            point
            for point in msg.points
            if point.valid
        ]

        self.latest_summary[topic] = {
            "reference_stamp_ns": stamp_ns,
            "frame_id": msg.frame_id,
            "source_frame_id": msg.source_frame_id,
            "generator_type": int(
                msg.generator_type
            ),
            "producer": msg.producer,
            "sequence": int(msg.sequence),
            "lookahead_distance": float(
                msg.lookahead_distance
            ),
            "expected_point_count": int(
                msg.expected_point_count
            ),
            "point_count": len(msg.points),
            "valid_point_count": len(valid_points),
        }

    def estimated_hz(self, topic: str) -> float | None:
        count = self.message_counts.get(topic, 0)

        if count < 2:
            return None

        first_stamp = self.first_ros_stamp_ns.get(
            topic
        )
        last_stamp = self.last_ros_stamp_ns.get(
            topic
        )

        if (
            first_stamp is not None
            and last_stamp is not None
            and last_stamp > first_stamp
        ):
            duration_sec = (
                last_stamp - first_stamp
            ) / 1_000_000_000.0

            return (count - 1) / duration_sec

        first_wall = self.first_wall_time.get(topic)

        if first_wall is None:
            return None

        wall_duration = time.monotonic() - first_wall

        if wall_duration <= 0.0:
            return None

        return (count - 1) / wall_duration

    def print_report(self) -> None:
        report = {
            "message_counts": dict(
                self.message_counts
            ),
            "estimated_hz": {},
            "latest_summary": self.latest_summary,
            "saved_calibrations": sorted(
                self.saved_calibrations
            ),
        }

        for topic in sorted(self.message_counts):
            frequency = self.estimated_hz(topic)
            report["estimated_hz"][topic] = frequency

        report_path = (
            self.output_directory
            / "inspection_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        self.get_logger().info(
            "Inspection report: "
            f"topics={len(self.message_counts)}, "
            f"calibrations="
            f"{len(self.saved_calibrations)}/"
            f"{len(CAMERA_NAMES)}, "
            f"path={report_path}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetInspector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_report()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()