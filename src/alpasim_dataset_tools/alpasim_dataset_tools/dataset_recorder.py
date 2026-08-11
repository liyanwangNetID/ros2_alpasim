#!/usr/bin/env python3
"""Record one AlpaSim replay clip into a model-independent dataset folder.

First-version lifecycle:
  1. Start this node before starting one AlpaSim clip.
  2. Let the clip run.
  3. Press Ctrl+C after the clip ends.
  4. The node validates and renames test_clip_NNN.tmp to test_clip_NNN.

Images are saved independently as JPEG files with their original ROS
simulation timestamps. No camera synchronization or training downsampling is
performed here.
"""
from __future__ import annotations

import json
import os
import queue
import re
from collections import defaultdict
import threading
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
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
from rosidl_runtime_py.convert import message_to_ordereddict

from alpasim_msgs.msg import (
    ActorStateArray,
    ActorTrajectoryArray,
    EgoState,
    EgoTrajectory,
    Route,
)
from alpasim_msgs.srv import GetGroundTruthEgoTrajectory, GetVectorMap
from builtin_interfaces.msg import Duration, Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String


CAMERA_NAMES = (
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right",
)

DYNAMIC_TOPIC_SPECS = (
    ("/alpasim/ego_state", EgoState, "ego/ego_state.jsonl"),
    (
        "/alpasim/ground_truth/ego/future_trajectory",
        EgoTrajectory,
        "ego/ground_truth_future.jsonl",
    ),
    (
        "/alpasim/planning/ego/trajectory",
        EgoTrajectory,
        "ego/planner_output.jsonl",
    ),
    (
        "/alpasim/actors/current",
        ActorStateArray,
        "actors/current.jsonl",
    ),
    (
        "/alpasim/actors/history",
        ActorTrajectoryArray,
        "actors/history.jsonl",
    ),
    (
        "/alpasim/ground_truth/actors/future",
        ActorTrajectoryArray,
        "actors/future.jsonl",
    ),
    ("/alpasim/route/map", Route, "route/map_route.jsonl"),
    (
        "/alpasim/route/model_input",
        Route,
        "route/navigation_route_local.jsonl",
    ),
)


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_time(value_ns: int) -> Time:
    msg = Time()
    msg.sec = int(value_ns // 1_000_000_000)
    msg.nanosec = int(value_ns % 1_000_000_000)
    return msg


def ros_message_to_dict(message: Any) -> dict[str, Any]:
    return dict(message_to_ordereddict(message))


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


class JsonlWriter:
    """Thread-safe line-oriented JSON writer."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.count = 0

    def append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
            self.count += 1

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()


@dataclass
class ImageTask:
    frame_index: int
    stamp_ns: int
    frame_id: str
    width: int
    height: int
    encoding: str
    step: int
    data: bytes


class CameraWriter:
    """Encode one camera stream in a dedicated background thread."""

    def __init__(
        self,
        camera_name: str,
        output_directory: Path,
        jpeg_quality: int,
        queue_size: int,
    ) -> None:
        self.camera_name = camera_name
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = int(jpeg_quality)
        self.queue: queue.Queue[ImageTask | None] = queue.Queue(
            maxsize=int(queue_size)
        )
        self.timestamps_writer = JsonlWriter(
            self.output_directory / "timestamps.jsonl"
        )
        self.received_count = 0
        self.saved_count = 0
        self.dropped_queue_full = 0
        self.encode_failures = 0
        self._counter_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"jpeg-writer-{camera_name}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, task: ImageTask) -> bool:
        with self._counter_lock:
            self.received_count += 1
        try:
            self.queue.put_nowait(task)
            return True
        except queue.Full:
            with self._counter_lock:
                self.dropped_queue_full += 1
            return False

    def _decode_rgb(self, task: ImageTask) -> np.ndarray:
        raw = np.frombuffer(task.data, dtype=np.uint8)

        if task.encoding in ("rgb8", "bgr8"):
            channels = 3
        elif task.encoding in ("rgba8", "bgra8"):
            channels = 4
        elif task.encoding in ("mono8", "8UC1"):
            channels = 1
        else:
            raise ValueError(
                f"Unsupported image encoding: {task.encoding}"
            )

        expected_minimum = task.step * task.height
        if raw.size < expected_minimum:
            raise ValueError(
                f"Image buffer too small: {raw.size} < {expected_minimum}"
            )

        # Respect row padding expressed by sensor_msgs/Image.step.
        rows = raw[:expected_minimum].reshape(task.height, task.step)
        pixel_bytes = rows[:, : task.width * channels]
        if channels == 1:
            image = pixel_bytes.reshape(task.height, task.width)
        else:
            image = pixel_bytes.reshape(
                task.height, task.width, channels
            )

        if task.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif task.encoding == "rgba8":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif task.encoding == "bgra8":
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _run(self) -> None:
        encode_parameters = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            self.jpeg_quality,
        ]
        while True:
            task = self.queue.get()
            try:
                if task is None:
                    return

                image = self._decode_rgb(task)
                success, encoded = cv2.imencode(
                    ".jpg", image, encode_parameters
                )
                if not success:
                    raise RuntimeError("cv2.imencode returned false")

                file_name = f"{task.frame_index:06d}.jpg"
                output_path = self.output_directory / file_name
                output_path.write_bytes(encoded.tobytes())

                self.timestamps_writer.append(
                    {
                        "frame_index": task.frame_index,
                        "stamp_ns": task.stamp_ns,
                        "frame_id": task.frame_id,
                        "width": task.width,
                        "height": task.height,
                        "encoding": task.encoding,
                        "step": task.step,
                        "image_path": file_name,
                        "jpeg_bytes": int(encoded.size),
                    }
                )
                with self._counter_lock:
                    self.saved_count += 1
            except Exception:
                with self._counter_lock:
                    self.encode_failures += 1
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.queue.join()
        self.queue.put(None)
        self._thread.join(timeout=30.0)
        self.timestamps_writer.close()

    def statistics(self) -> dict[str, int]:
        with self._counter_lock:
            return {
                "received_count": self.received_count,
                "saved_count": self.saved_count,
                "dropped_queue_full": self.dropped_queue_full,
                "encode_failures": self.encode_failures,
            }


class DatasetRecorder(Node):
    """Record one manually started AlpaSim clip."""

    def __init__(self) -> None:
        super().__init__("alpasim_dataset_recorder")

        self.declare_parameter(
            "output_root", "/home/lab/data_from_alpasim"
        )
        self.declare_parameter("clip_prefix", "test_clip_")
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("image_queue_size", 64)
        self.declare_parameter("report_period_sec", 5.0)
        self.declare_parameter("gt_service_duration_sec", 3600)

        self.output_root = Path(
            str(self.get_parameter("output_root").value)
        ).expanduser()
        self.clip_prefix = str(
            self.get_parameter("clip_prefix").value
        )
        self.jpeg_quality = int(
            self.get_parameter("jpeg_quality").value
        )
        self.image_queue_size = int(
            self.get_parameter("image_queue_size").value
        )
        self.report_period_sec = float(
            self.get_parameter("report_period_sec").value
        )
        self.gt_service_duration_sec = int(
            self.get_parameter("gt_service_duration_sec").value
        )

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.clips_root = self.output_root
        self.clips_root.mkdir(parents=True, exist_ok=True)
        self.clip_number, self.clip_name, self.temporary_directory = (
            self._allocate_clip_directory()
        )
        self.final_directory = self.clips_root / self.clip_name
        self._create_directory_structure()

        self._state_lock = threading.Lock()
        self._subscriptions: list[Any] = []
        self._closed = False
        self._first_sim_time_ns: int | None = None
        self._last_sim_time_ns: int | None = None
        self._first_wall_time = time.time()
        self._last_wall_time: float | None = None
        self._topic_counts: dict[str, int] = defaultdict(int)
        self._calibrations_saved: set[str] = set()
        self._last_executed_stamp_ns: int | None = None
        self._last_executed_message: dict[str, Any] | None = None

        self._map_request_started = False
        self._map_request_completed = False
        self._map_request_error: str | None = None
        self._gt_request_started = False
        self._gt_request_completed = False
        self._gt_request_error: str | None = None

        self._jsonl_writers = {
            topic: JsonlWriter(self.temporary_directory / relative_path)
            for topic, _, relative_path in DYNAMIC_TOPIC_SPECS
        }
        self._executed_incremental_writer = JsonlWriter(
            self.temporary_directory
            / "ego"
            / "executed_path_points.jsonl"
        )

        self._camera_groups = {
            camera_name: MutuallyExclusiveCallbackGroup()
            for camera_name in CAMERA_NAMES
        }
        self._camera_writers = {
            camera_name: CameraWriter(
                camera_name=camera_name,
                output_directory=(
                    self.temporary_directory
                    / "cameras"
                    / camera_name
                ),
                jpeg_quality=self.jpeg_quality,
                queue_size=self.image_queue_size,
            )
            for camera_name in CAMERA_NAMES
        }
        self._camera_frame_indices = {
            camera_name: 0 for camera_name in CAMERA_NAMES
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
            depth=20,
        )
        self.transient_local_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._create_camera_subscriptions()
        self._create_dynamic_subscriptions()

        self.map_client = self.create_client(
            GetVectorMap,
            "/alpasim/map/get_vector_map",
        )
        self.gt_client = self.create_client(
            GetGroundTruthEgoTrajectory,
            "/alpasim/navigation/get_ground_truth_ego_trajectory",
        )
        self.service_timer = self.create_timer(
            0.5, self._try_service_requests
        )
        self.report_timer = self.create_timer(
            self.report_period_sec, self._log_progress
        )

        self._write_initial_metadata()
        self.get_logger().info(
            f"Recording clip into: {self.temporary_directory}"
        )
        self.get_logger().info(
            "Start one AlpaSim clip; press Ctrl+C after it ends."
        )

    def _allocate_clip_directory(
        self,
    ) -> tuple[int, str, Path]:
        pattern = re.compile(
            rf"^{re.escape(self.clip_prefix)}(\d+)$"
        )

        used_numbers: set[int] = set()

        for path in self.output_root.iterdir():
            if not path.is_dir():
                continue

            match = pattern.match(path.name)
            if match:
                used_numbers.add(int(match.group(1)))

        number = 1

        while number in used_numbers:
            number += 1

        while True:
            name = f"{self.clip_prefix}{number:03d}"
            temporary = self.output_root / f"{name}.tmp"
            final = self.output_root / name

            if final.exists() or temporary.exists():
                number += 1
                continue

            try:
                temporary.mkdir(exist_ok=False)
            except FileExistsError:
                number += 1
                continue

            return number, name, temporary

    def _remove_temporary_clip_directories(
        self,
        exclude: Path | None = None,
    ) -> list[str]:
        pattern = re.compile(
            rf"^{re.escape(self.clip_prefix)}\d+\.tmp$"
        )

        removed_directories: list[str] = []

        for path in sorted(self.output_root.iterdir()):
            if not path.is_dir():
                continue

            if exclude is not None and path == exclude:
                continue

            if not pattern.fullmatch(path.name):
                continue

            try:
                shutil.rmtree(path)
                removed_directories.append(path.name)

                self.get_logger().info(
                    f"Removed stale temporary directory: {path}"
                )
            except OSError as exc:
                self.get_logger().error(
                    f"Failed to remove temporary directory "
                    f"{path}: {exc}"
                )

        return removed_directories

    def _create_directory_structure(self) -> None:
        for relative in (
            "calibration",
            "cameras",
            "ego",
            "actors",
            "route",
            "map",
        ):
            (self.temporary_directory / relative).mkdir(
                parents=True, exist_ok=True
            )

    def _write_initial_metadata(self) -> None:
        write_json_atomic(
            self.temporary_directory / "metadata.json",
            {
                "dataset_format_version": "0.1",
                "clip_number": self.clip_number,
                "clip_name": self.clip_name,
                "status": "recording",
                "output_root": str(self.output_root),
                "image_storage": "JPEG_PER_FRAME",
                "jpeg_quality": self.jpeg_quality,
                "save_all_received_frames": True,
                "camera_names": list(CAMERA_NAMES),
                "created_wall_time_unix": self._first_wall_time,
            },
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
                lambda msg, name=camera_name, topic=image_topic: (
                    self._image_callback(msg, name, topic)
                ),
                self.sensor_qos,
                callback_group=self._camera_groups[camera_name],
            )
            calibration_subscription = self.create_subscription(
                String,
                calibration_topic,
                lambda msg, name=camera_name, topic=calibration_topic: (
                    self._calibration_callback(msg, name, topic)
                ),
                self.transient_local_qos,
            )
            self._subscriptions.extend(
                [image_subscription, calibration_subscription]
            )

    def _create_dynamic_subscriptions(self) -> None:
        clock_subscription = self.create_subscription(
            Clock,
            "/clock",
            self._clock_callback,
            self.reliable_qos,
        )
        self._subscriptions.append(clock_subscription)

        executed_subscription = self.create_subscription(
            EgoTrajectory,
            "/alpasim/ego/executed_path",
            self._executed_path_callback,
            self.reliable_qos,
        )
        self._subscriptions.append(executed_subscription)

        for topic, message_type, _ in DYNAMIC_TOPIC_SPECS:
            subscription = self.create_subscription(
                message_type,
                topic,
                lambda msg, topic_name=topic: (
                    self._dynamic_message_callback(msg, topic_name)
                ),
                self.reliable_qos,
            )
            self._subscriptions.append(subscription)

    def _increment_topic_count(self, topic: str) -> None:
        with self._state_lock:
            self._topic_counts[topic] += 1

    def _clock_callback(self, msg: Clock) -> None:
        stamp_ns = stamp_to_ns(msg.clock)
        with self._state_lock:
            self._topic_counts["/clock"] += 1
            if self._first_sim_time_ns is None:
                self._first_sim_time_ns = stamp_ns
            self._last_sim_time_ns = stamp_ns

    def _calibration_callback(
        self,
        msg: String,
        camera_name: str,
        topic: str,
    ) -> None:
        self._increment_topic_count(topic)
        with self._state_lock:
            if camera_name in self._calibrations_saved:
                return
        try:
            calibration = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f"Invalid calibration JSON for {camera_name}: {exc}"
            )
            return

        write_json_atomic(
            self.temporary_directory
            / "calibration"
            / f"{camera_name}.json",
            calibration,
        )
        with self._state_lock:
            self._calibrations_saved.add(camera_name)

    def _image_callback(
        self,
        msg: Image,
        camera_name: str,
        topic: str,
    ) -> None:
        self._increment_topic_count(topic)
        with self._state_lock:
            frame_index = self._camera_frame_indices[camera_name]
            self._camera_frame_indices[camera_name] += 1

        task = ImageTask(
            frame_index=frame_index,
            stamp_ns=stamp_to_ns(msg.header.stamp),
            frame_id=msg.header.frame_id,
            width=int(msg.width),
            height=int(msg.height),
            encoding=msg.encoding,
            step=int(msg.step),
            data=bytes(msg.data),
        )
        if not self._camera_writers[camera_name].submit(task):
            self.get_logger().warning(
                f"Image queue full for {camera_name}; frame dropped"
            )

    def _dynamic_message_callback(
        self,
        msg: Any,
        topic: str,
    ) -> None:
        self._increment_topic_count(topic)
        self._jsonl_writers[topic].append(
            {
                "topic": topic,
                "message": ros_message_to_dict(msg),
            }
        )

    def _executed_path_callback(self, msg: EgoTrajectory) -> None:
        topic = "/alpasim/ego/executed_path"
        self._increment_topic_count(topic)
        message_dict = ros_message_to_dict(msg)

        new_points = []
        for point, point_dict in zip(msg.points, message_dict["points"]):
            stamp_ns = stamp_to_ns(point.stamp)
            if (
                self._last_executed_stamp_ns is None
                or stamp_ns > self._last_executed_stamp_ns
            ):
                new_points.append(point_dict)
                self._last_executed_stamp_ns = stamp_ns

        for point_dict in new_points:
            self._executed_incremental_writer.append(point_dict)

        with self._state_lock:
            self._last_executed_message = message_dict

    def _try_service_requests(self) -> None:
        if (
            not self._map_request_started
            and self.map_client.service_is_ready()
        ):
            self._map_request_started = True
            request = GetVectorMap.Request()
            request.requested_scene_id = ""
            request.known_revision = 0
            future = self.map_client.call_async(request)
            future.add_done_callback(self._handle_map_response)

        with self._state_lock:
            first_sim_time_ns = self._first_sim_time_ns

        if (
            first_sim_time_ns is not None
            and not self._gt_request_started
            and self.gt_client.service_is_ready()
        ):
            self._gt_request_started = True
            request = GetGroundTruthEgoTrajectory.Request()
            request.reference_stamp = ns_to_time(first_sim_time_ns)
            request.future_duration = Duration(
                sec=self.gt_service_duration_sec,
                nanosec=0,
            )
            request.sampling_interval = Duration(sec=0, nanosec=0)
            request.max_points = 0
            request.known_revision = 0
            future = self.gt_client.call_async(request)
            future.add_done_callback(self._handle_gt_response)

    def _handle_map_response(self, future: Any) -> None:
        try:
            response = future.result()
            if not response.success:
                raise RuntimeError(response.message)
            if response.not_modified:
                raise RuntimeError(
                    "Map service returned not_modified without local cache"
                )
            write_json_atomic(
                self.temporary_directory / "map" / "vector_map.json",
                ros_message_to_dict(response.vector_map),
            )
            self._map_request_completed = True
        except Exception as exc:
            self._map_request_error = str(exc)
            self.get_logger().error(f"Vector map request failed: {exc}")

    def _handle_gt_response(self, future: Any) -> None:
        try:
            response = future.result()
            if not response.success:
                raise RuntimeError(response.message)
            if response.not_modified:
                raise RuntimeError(
                    "GT service returned not_modified without local cache"
                )
            write_json_atomic(
                self.temporary_directory
                / "ego"
                / "complete_recording_ground_truth.json",
                {
                    "revision": int(response.revision),
                    "recording_start_stamp": ros_message_to_dict(
                        response.recording_start_stamp
                    ),
                    "recording_end_stamp": ros_message_to_dict(
                        response.recording_end_stamp
                    ),
                    "trajectory": ros_message_to_dict(
                        response.trajectory
                    ),
                },
            )
            self._gt_request_completed = True
        except Exception as exc:
            self._gt_request_error = str(exc)
            self.get_logger().error(
                f"Ground-truth trajectory request failed: {exc}"
            )

    def _log_progress(self) -> None:
        camera_stats = {
            name: writer.statistics()
            for name, writer in self._camera_writers.items()
        }
        self.get_logger().info(
            f"Recording {self.clip_name}: cameras={camera_stats}, "
            f"map={self._map_request_completed}, "
            f"complete_gt={self._gt_request_completed}"
        )

    def _validation_payload(self) -> dict[str, Any]:
        camera_stats = {
            name: writer.statistics()
            for name, writer in self._camera_writers.items()
        }
        with self._state_lock:
            topic_counts = dict(self._topic_counts)
            calibrations = sorted(self._calibrations_saved)
            first_sim_time_ns = self._first_sim_time_ns
            last_sim_time_ns = self._last_sim_time_ns

        checks = {
            "all_four_calibrations": (
                set(calibrations) == set(CAMERA_NAMES)
            ),
            "all_four_cameras_saved_images": all(
                stats["saved_count"] > 0
                for stats in camera_stats.values()
            ),
            "no_camera_queue_drops": all(
                stats["dropped_queue_full"] == 0
                for stats in camera_stats.values()
            ),
            "no_camera_encode_failures": all(
                stats["encode_failures"] == 0
                for stats in camera_stats.values()
            ),
            "has_clock": topic_counts.get("/clock", 0) > 0,
            "has_ego_state": topic_counts.get(
                "/alpasim/ego_state", 0
            ) > 0,
            "has_executed_path": topic_counts.get(
                "/alpasim/ego/executed_path", 0
            ) > 0,
            "has_actor_current": topic_counts.get(
                "/alpasim/actors/current", 0
            ) > 0,
            "has_navigation_route": topic_counts.get(
                "/alpasim/route/model_input", 0
            ) > 0,
            "has_vector_map": self._map_request_completed,
            "has_complete_gt_service": self._gt_request_completed,
        }

        # Complete GT service is retained as an important diagnostic, but a
        # clip can still be finalized when the online GT topic was recorded.
        required_check_names = (
            "all_four_calibrations",
            "all_four_cameras_saved_images",
            "no_camera_queue_drops",
            "no_camera_encode_failures",
            "has_clock",
            "has_ego_state",
            "has_executed_path",
            "has_actor_current",
            "has_navigation_route",
            "has_vector_map",
        )
        valid = all(checks[name] for name in required_check_names)

        return {
            "valid": valid,
            "checks": checks,
            "required_checks": list(required_check_names),
            "camera_statistics": camera_stats,
            "topic_counts": topic_counts,
            "calibrations_saved": calibrations,
            "first_sim_time_ns": first_sim_time_ns,
            "last_sim_time_ns": last_sim_time_ns,
            "sim_duration_sec": (
                (last_sim_time_ns - first_sim_time_ns) / 1e9
                if first_sim_time_ns is not None
                and last_sim_time_ns is not None
                else None
            ),
            "map_service_error": self._map_request_error,
            "gt_service_error": self._gt_request_error,
        }

    def close_and_finalize(self) -> tuple[bool, Path]:
        if self._closed:
            return False, self.temporary_directory
        self._closed = True

        self.get_logger().info("Closing camera writer queues...")
        for writer in self._camera_writers.values():
            writer.close()
        for writer in self._jsonl_writers.values():
            writer.close()
        self._executed_incremental_writer.close()

        with self._state_lock:
            executed_message = self._last_executed_message
        if executed_message is not None:
            write_json_atomic(
                self.temporary_directory
                / "ego"
                / "executed_path_final.json",
                executed_message,
            )

        self._last_wall_time = time.time()
        validation = self._validation_payload()
        write_json_atomic(
            self.temporary_directory / "validation.json",
            validation,
        )

        metadata = {
            "dataset_format_version": "0.1",
            "clip_number": self.clip_number,
            "clip_name": self.clip_name,
            "status": "complete" if validation["valid"] else "invalid",
            "image_storage": "JPEG_PER_FRAME",
            "jpeg_quality": self.jpeg_quality,
            "save_all_received_frames": True,
            "camera_names": list(CAMERA_NAMES),
            "created_wall_time_unix": self._first_wall_time,
            "closed_wall_time_unix": self._last_wall_time,
            "validation_file": "validation.json",
        }
        write_json_atomic(
            self.temporary_directory / "metadata.json", metadata
        )

        if validation["valid"]:
            self.temporary_directory.replace(
                self.final_directory
            )

            self.get_logger().info(
                f"Clip finalized successfully: "
                f"{self.final_directory}"
            )

            self._remove_temporary_clip_directories(
                exclude=self.final_directory
            )

            return True, self.final_directory

        self.get_logger().warning(
            "Clip validation failed. Temporary data is retained for "
            f"inspection: {self.temporary_directory}"
        )

        return False, self.temporary_directory


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetRecorder()
    executor = MultiThreadedExecutor(num_threads=10)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.close_and_finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
