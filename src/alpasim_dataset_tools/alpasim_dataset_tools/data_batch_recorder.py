#!/usr/bin/env python3
"""Batch AlpaSim dataset recorder controlled by clip_active edges.

Lifecycle:
- Node stays alive while batch simulation is running.
- false -> true on /alpasim/simulation/clip_active starts a new clip.
- true -> false finalizes the current clip.
- The smallest unused test_clip_NNN number is allocated.
- Failed clips retain their .tmp directory for diagnosis.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from collections import defaultdict
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
from std_msgs.msg import Bool, String


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
    ("/alpasim/actors/current", ActorStateArray, "actors/current.jsonl"),
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
    result = Time()
    result.sec = int(value_ns // 1_000_000_000)
    result.nanosec = int(value_ns % 1_000_000_000)
    return result


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
            output_directory / "timestamps.jsonl"
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

    @staticmethod
    def _decode_image(task: ImageTask) -> np.ndarray:
        raw = np.frombuffer(task.data, dtype=np.uint8)
        if task.encoding in ("rgb8", "bgr8"):
            channels = 3
        elif task.encoding in ("rgba8", "bgra8"):
            channels = 4
        elif task.encoding in ("mono8", "8UC1"):
            channels = 1
        else:
            raise ValueError(f"Unsupported image encoding: {task.encoding}")

        required_bytes = task.step * task.height
        if raw.size < required_bytes:
            raise ValueError(
                f"Image buffer too small: {raw.size} < {required_bytes}"
            )
        rows = raw[:required_bytes].reshape(task.height, task.step)
        pixels = rows[:, : task.width * channels]
        if channels == 1:
            image = pixels.reshape(task.height, task.width)
        else:
            image = pixels.reshape(task.height, task.width, channels)

        if task.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif task.encoding == "rgba8":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif task.encoding == "bgra8":
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _run(self) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while True:
            task = self.queue.get()
            try:
                if task is None:
                    return
                image = self._decode_image(task)
                success, encoded = cv2.imencode(".jpg", image, params)
                if not success:
                    raise RuntimeError("cv2.imencode returned false")
                file_name = f"{task.frame_index:06d}.jpg"
                (self.output_directory / file_name).write_bytes(
                    encoded.tobytes()
                )
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


class RecordingSession:
    def __init__(
        self,
        clip_number: int,
        clip_name: str,
        temporary_directory: Path,
        final_directory: Path,
        jpeg_quality: int,
        image_queue_size: int,
    ) -> None:
        self.clip_number = clip_number
        self.clip_name = clip_name
        self.temporary_directory = temporary_directory
        self.final_directory = final_directory
        self.started_wall_time = time.time()
        self.closed_wall_time: float | None = None
        self.accepting_messages = True

        for relative in (
            "calibration",
            "cameras",
            "ego",
            "actors",
            "route",
            "map",
        ):
            (temporary_directory / relative).mkdir(
                parents=True, exist_ok=True
            )

        self.topic_counts: dict[str, int] = defaultdict(int)
        self.calibrations_saved: set[str] = set()
        self.first_sim_time_ns: int | None = None
        self.last_sim_time_ns: int | None = None
        self.last_executed_stamp_ns: int | None = None
        self.last_executed_message: dict[str, Any] | None = None
        self.camera_frame_indices = {
            camera_name: 0 for camera_name in CAMERA_NAMES
        }
        self.jsonl_writers = {
            topic: JsonlWriter(temporary_directory / relative_path)
            for topic, _, relative_path in DYNAMIC_TOPIC_SPECS
        }
        self.executed_writer = JsonlWriter(
            temporary_directory / "ego" / "executed_path_points.jsonl"
        )
        self.camera_writers = {
            camera_name: CameraWriter(
                camera_name,
                temporary_directory / "cameras" / camera_name,
                jpeg_quality,
                image_queue_size,
            )
            for camera_name in CAMERA_NAMES
        }

        self.map_request_started = False
        self.map_request_completed = False
        self.map_request_error: str | None = None
        self.gt_request_started = False
        self.gt_request_completed = False
        self.gt_request_error: str | None = None

        write_json_atomic(
            temporary_directory / "metadata.json",
            {
                "dataset_format_version": "0.2-batch",
                "clip_number": clip_number,
                "clip_name": clip_name,
                "status": "recording",
                "image_storage": "JPEG_PER_FRAME",
                "jpeg_quality": jpeg_quality,
                "save_all_received_frames": True,
                "camera_names": list(CAMERA_NAMES),
                "created_wall_time_unix": self.started_wall_time,
            },
        )

    def camera_statistics(self) -> dict[str, dict[str, int]]:
        return {
            name: writer.statistics()
            for name, writer in self.camera_writers.items()
        }

    def close_writers(self) -> None:
        self.accepting_messages = False
        for writer in self.camera_writers.values():
            writer.close()
        for writer in self.jsonl_writers.values():
            writer.close()
        self.executed_writer.close()


class DatasetBatchRecorder(Node):
    def __init__(self) -> None:
        super().__init__("alpasim_data_batch_recorder")

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
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.clip_prefix = str(self.get_parameter("clip_prefix").value)
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

        self._lifecycle_lock = threading.RLock()
        self._subscription_handles: list[Any] = []
        self._session: RecordingSession | None = None
        self._last_clip_active = False
        self._calibration_cache: dict[str, dict[str, Any]] = {}

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

        self._camera_groups = {
            camera_name: MutuallyExclusiveCallbackGroup()
            for camera_name in CAMERA_NAMES
        }
        self._create_subscriptions()

        self.map_client = self.create_client(
            GetVectorMap, "/alpasim/map/get_vector_map"
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

        self.get_logger().info(
            "Batch recorder ready; waiting for false -> true on "
            "/alpasim/simulation/clip_active"
        )

    def _create_subscriptions(self) -> None:
        self._subscription_handles.append(
            self.create_subscription(
                Bool,
                "/alpasim/simulation/clip_active",
                self._clip_active_callback,
                self.reliable_qos,
            )
        )
        self._subscription_handles.append(
            self.create_subscription(
                Clock, "/clock", self._clock_callback, self.reliable_qos
            )
        )
        self._subscription_handles.append(
            self.create_subscription(
                EgoTrajectory,
                "/alpasim/ego/executed_path",
                self._executed_path_callback,
                self.reliable_qos,
            )
        )

        for camera_name in CAMERA_NAMES:
            image_topic = f"/alpasim/camera/{camera_name}/image"
            calibration_topic = (
                f"/alpasim/camera/{camera_name}/calibration"
            )
            self._subscription_handles.append(
                self.create_subscription(
                    Image,
                    image_topic,
                    lambda msg, name=camera_name, topic=image_topic: (
                        self._image_callback(msg, name, topic)
                    ),
                    self.sensor_qos,
                    callback_group=self._camera_groups[camera_name],
                )
            )
            self._subscription_handles.append(
                self.create_subscription(
                    String,
                    calibration_topic,
                    lambda msg, name=camera_name, topic=calibration_topic: (
                        self._calibration_callback(msg, name, topic)
                    ),
                    self.transient_local_qos,
                )
            )

        for topic, message_type, _ in DYNAMIC_TOPIC_SPECS:
            self._subscription_handles.append(
                self.create_subscription(
                    message_type,
                    topic,
                    lambda msg, topic_name=topic: (
                        self._dynamic_message_callback(msg, topic_name)
                    ),
                    self.reliable_qos,
                )
            )

    def _allocate_clip_directory(
        self,
    ) -> tuple[int, str, Path, Path]:
        pattern = re.compile(
            rf"^{re.escape(self.clip_prefix)}(\d+)(?:\.tmp)?$"
        )
        used_numbers: set[int] = set()
        for path in self.output_root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.fullmatch(path.name)
            if match:
                used_numbers.add(int(match.group(1)))

        number = 1
        while number in used_numbers:
            number += 1

        while True:
            name = f"{self.clip_prefix}{number:03d}"
            temporary = self.output_root / f"{name}.tmp"
            final = self.output_root / name
            try:
                temporary.mkdir(exist_ok=False)
            except FileExistsError:
                number += 1
                continue
            if final.exists():
                temporary.rmdir()
                number += 1
                continue
            return number, name, temporary, final

    def _clip_active_callback(self, msg: Bool) -> None:
        new_state = bool(msg.data)
        with self._lifecycle_lock:
            previous_state = self._last_clip_active
            self._last_clip_active = new_state

        if not previous_state and new_state:
            self._start_session()
        elif previous_state and not new_state:
            self._stop_session()

    def _start_session(self) -> None:
        with self._lifecycle_lock:
            if self._session is not None:
                self.get_logger().warning(
                    "Start edge ignored because a session is already active"
                )
                return
            number, name, temporary, final = (
                self._allocate_clip_directory()
            )
            self._session = RecordingSession(
                number,
                name,
                temporary,
                final,
                self.jpeg_quality,
                self.image_queue_size,
            )

            for camera_name, calibration in (
                self._calibration_cache.items()
            ):
                write_json_atomic(
                    self._session.temporary_directory
                    / "calibration"
                    / f"{camera_name}.json",
                    calibration,
                )
                self._session.calibrations_saved.add(camera_name)

            self.get_logger().info(
                f"Started recording {name}: {temporary}"
            )

    def _stop_session(self) -> None:
        with self._lifecycle_lock:
            session = self._session
            if session is None:
                return
            session.accepting_messages = False
            self._session = None

        self.get_logger().info(
            f"Stopping {session.clip_name}; flushing writers..."
        )
        self._finalize_session(session)

    def _clock_callback(self, msg: Clock) -> None:
        with self._lifecycle_lock:
            session = self._session
            if session is None or not session.accepting_messages:
                return
            stamp_ns = stamp_to_ns(msg.clock)
            session.topic_counts["/clock"] += 1
            if session.first_sim_time_ns is None:
                session.first_sim_time_ns = stamp_ns
            session.last_sim_time_ns = stamp_ns

    def _calibration_callback(
        self,
        msg: String,
        camera_name: str,
        topic: str,
    ) -> None:
        try:
            calibration = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f"Invalid calibration JSON for {camera_name}: {exc}"
            )
            return

        with self._lifecycle_lock:
            # Calibration belongs to the recorder, not only one session.
            self._calibration_cache[camera_name] = calibration

            session = self._session
            if session is None or not session.accepting_messages:
                return

            session.topic_counts[topic] += 1

            if camera_name in session.calibrations_saved:
                return

            write_json_atomic(
                session.temporary_directory
                / "calibration"
                / f"{camera_name}.json",
                calibration,
            )
            session.calibrations_saved.add(camera_name)

    def _image_callback(
        self,
        msg: Image,
        camera_name: str,
        topic: str,
    ) -> None:
        with self._lifecycle_lock:
            session = self._session
            if session is None or not session.accepting_messages:
                return
            frame_index = session.camera_frame_indices[camera_name]
            session.camera_frame_indices[camera_name] += 1
            session.topic_counts[topic] += 1
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
            accepted = session.camera_writers[camera_name].submit(task)
        if not accepted:
            self.get_logger().warning(
                f"Image queue full for {camera_name}; frame dropped"
            )

    def _dynamic_message_callback(
        self,
        msg: Any,
        topic: str,
    ) -> None:
        payload = {
            "topic": topic,
            "message": ros_message_to_dict(msg),
        }
        with self._lifecycle_lock:
            session = self._session
            if session is None or not session.accepting_messages:
                return
            session.topic_counts[topic] += 1
            session.jsonl_writers[topic].append(payload)

    def _executed_path_callback(self, msg: EgoTrajectory) -> None:
        message_dict = ros_message_to_dict(msg)
        with self._lifecycle_lock:
            session = self._session
            if session is None or not session.accepting_messages:
                return
            topic = "/alpasim/ego/executed_path"
            session.topic_counts[topic] += 1
            for point, point_dict in zip(
                msg.points, message_dict["points"]
            ):
                point_stamp_ns = stamp_to_ns(point.stamp)
                if (
                    session.last_executed_stamp_ns is None
                    or point_stamp_ns > session.last_executed_stamp_ns
                ):
                    session.executed_writer.append(point_dict)
                    session.last_executed_stamp_ns = point_stamp_ns
            session.last_executed_message = message_dict

    def _try_service_requests(self) -> None:
        with self._lifecycle_lock:
            session = self._session
            if session is None or not session.accepting_messages:
                return

            if (
                not session.map_request_started
                and self.map_client.service_is_ready()
            ):
                session.map_request_started = True
                request = GetVectorMap.Request()
                request.requested_scene_id = ""
                request.known_revision = 0
                future = self.map_client.call_async(request)
                future.add_done_callback(
                    lambda result, target=session: (
                        self._handle_map_response(result, target)
                    )
                )

            if (
                session.first_sim_time_ns is not None
                and not session.gt_request_started
                and self.gt_client.service_is_ready()
            ):
                session.gt_request_started = True
                request = GetGroundTruthEgoTrajectory.Request()
                request.reference_stamp = ns_to_time(
                    session.first_sim_time_ns
                )
                request.future_duration = Duration(
                    sec=self.gt_service_duration_sec, nanosec=0
                )
                request.sampling_interval = Duration(sec=0, nanosec=0)
                request.max_points = 0
                request.known_revision = 0
                future = self.gt_client.call_async(request)
                future.add_done_callback(
                    lambda result, target=session: (
                        self._handle_gt_response(result, target)
                    )
                )

    def _handle_map_response(
        self,
        future: Any,
        session: RecordingSession,
    ) -> None:
        try:
            response = future.result()
            if not response.success:
                raise RuntimeError(response.message)
            if response.not_modified:
                raise RuntimeError(
                    "Map service returned not_modified without local cache"
                )
            write_json_atomic(
                session.temporary_directory / "map" / "vector_map.json",
                ros_message_to_dict(response.vector_map),
            )
            session.map_request_completed = True
        except Exception as exc:
            session.map_request_error = str(exc)
            self.get_logger().error(
                f"Vector map request failed for {session.clip_name}: {exc}"
            )

    def _handle_gt_response(
        self,
        future: Any,
        session: RecordingSession,
    ) -> None:
        try:
            response = future.result()
            if not response.success:
                raise RuntimeError(response.message)
            if response.not_modified:
                raise RuntimeError(
                    "GT service returned not_modified without local cache"
                )
            write_json_atomic(
                session.temporary_directory
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
            session.gt_request_completed = True
        except Exception as exc:
            session.gt_request_error = str(exc)
            self.get_logger().error(
                f"GT request failed for {session.clip_name}: {exc}"
            )

    def _build_validation(
        self,
        session: RecordingSession,
    ) -> dict[str, Any]:
        camera_stats = session.camera_statistics()
        checks = {
            "all_four_calibrations": (
                set(session.calibrations_saved) == set(CAMERA_NAMES)
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
            "has_clock": session.topic_counts.get("/clock", 0) > 0,
            "has_ego_state": session.topic_counts.get(
                "/alpasim/ego_state", 0
            ) > 0,
            "has_executed_path": session.topic_counts.get(
                "/alpasim/ego/executed_path", 0
            ) > 0,
            "has_actor_current": session.topic_counts.get(
                "/alpasim/actors/current", 0
            ) > 0,
            "has_navigation_route": session.topic_counts.get(
                "/alpasim/route/model_input", 0
            ) > 0,
            "has_vector_map": session.map_request_completed,
            "has_complete_gt_service": session.gt_request_completed,
        }
        required = (
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
        return {
            "valid": all(checks[name] for name in required),
            "checks": checks,
            "required_checks": list(required),
            "camera_statistics": camera_stats,
            "topic_counts": dict(session.topic_counts),
            "calibrations_saved": sorted(session.calibrations_saved),
            "first_sim_time_ns": session.first_sim_time_ns,
            "last_sim_time_ns": session.last_sim_time_ns,
            "sim_duration_sec": (
                (
                    session.last_sim_time_ns
                    - session.first_sim_time_ns
                )
                / 1e9
                if session.first_sim_time_ns is not None
                and session.last_sim_time_ns is not None
                else None
            ),
            "map_service_error": session.map_request_error,
            "gt_service_error": session.gt_request_error,
        }

    def _finalize_session(self, session: RecordingSession) -> None:
        session.close_writers()
        if session.last_executed_message is not None:
            write_json_atomic(
                session.temporary_directory
                / "ego"
                / "executed_path_final.json",
                session.last_executed_message,
            )

        session.closed_wall_time = time.time()
        validation = self._build_validation(session)
        write_json_atomic(
            session.temporary_directory / "validation.json",
            validation,
        )
        write_json_atomic(
            session.temporary_directory / "metadata.json",
            {
                "dataset_format_version": "0.2-batch",
                "clip_number": session.clip_number,
                "clip_name": session.clip_name,
                "status": (
                    "complete" if validation["valid"] else "invalid"
                ),
                "image_storage": "JPEG_PER_FRAME",
                "jpeg_quality": self.jpeg_quality,
                "save_all_received_frames": True,
                "camera_names": list(CAMERA_NAMES),
                "created_wall_time_unix": session.started_wall_time,
                "closed_wall_time_unix": session.closed_wall_time,
                "validation_file": "validation.json",
            },
        )

        if validation["valid"]:
            session.temporary_directory.replace(session.final_directory)
            self.get_logger().info(
                f"Finalized {session.clip_name}: {session.final_directory}"
            )
        else:
            self.get_logger().warning(
                f"Validation failed for {session.clip_name}; retained "
                f"temporary data: {session.temporary_directory}"
            )

    def _log_progress(self) -> None:
        with self._lifecycle_lock:
            session = self._session
            if session is None:
                return
            stats = session.camera_statistics()
            self.get_logger().info(
                f"Recording {session.clip_name}: cameras={stats}, "
                f"map={session.map_request_completed}, "
                f"complete_gt={session.gt_request_completed}"
            )

    def shutdown_active_session(self) -> None:
        with self._lifecycle_lock:
            session = self._session
            if session is None:
                return
            session.accepting_messages = False
            self._session = None
        self.get_logger().warning(
            f"Node shutdown while {session.clip_name} was active; "
            "finalizing received data."
        )
        self._finalize_session(session)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = DatasetBatchRecorder()

    executor = MultiThreadedExecutor(
        num_threads=10
    )
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_active_session()

        executor.remove_node(node)

        executor.shutdown(
            timeout_sec=10.0
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
