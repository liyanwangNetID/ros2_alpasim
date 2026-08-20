#!/usr/bin/env python3
"""Read one finalized AlpaSim dataset clip through a unified temporal API.

The reader is independent of ROS 2. It loads lightweight indexes eagerly and
loads larger JSON/JSONL data lazily. Raw clip files are read-only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from coordinate_utils import (
    Pose2D,
    interpolate_angle,
    interpolate_scalar,
    map_pose_to_anchor_ego,
    pose2d_from_pose_mapping,
)
from temporal_index import (
    MultiCameraSequence,
    TemporalIndex,
    synchronize_camera_sequence,
)


CAMERA_NAMES = (
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right",
)
DEFAULT_FRAME_OFFSETS_NS = (-400_000_000, -200_000_000, 0)
DEFAULT_CAMERA_TOLERANCE_NS = 50_000_000
DEFAULT_EGO_HISTORY_POINTS = 16
DEFAULT_EGO_INTERVAL_NS = 100_000_000
DEFAULT_ROUTE_MAX_AGE_NS = 200_000_000
DEFAULT_ACTOR_TOLERANCE_NS = 100_000_000


class ClipReaderError(RuntimeError):
    """Base error for malformed or unavailable clip data."""


class ClipDataError(ClipReaderError):
    """Raised when a required raw-data field is malformed."""


@dataclass(frozen=True, slots=True)
class CameraFrameRecord:
    camera_name: str
    frame_index: int
    stamp_ns: int
    frame_id: str
    width: int
    height: int
    encoding: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class EgoStateSample:
    stamp_ns: int
    pose: Pose2D
    speed: float
    longitudinal_acceleration: float
    yaw_rate: float


@dataclass(frozen=True, slots=True)
class EgoHistoryWaypoint:
    target_stamp_ns: int
    relative_x: float
    relative_y: float
    sin_relative_yaw: float
    cos_relative_yaw: float
    speed: float
    longitudinal_acceleration: float
    yaw_rate: float
    relative_time: float


@dataclass(frozen=True, slots=True)
class TimedMessage:
    stamp_ns: int
    time_error_ns: int
    message: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FutureTrajectory:
    anchor_ns: int
    end_ns: int
    points: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AnchorValidation:
    anchor_ns: int
    usable: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...]


def stamp_mapping_to_ns(stamp: Mapping[str, Any]) -> int:
    try:
        sec = stamp["sec"]
        nanosec = stamp["nanosec"]
    except KeyError as exc:
        raise ClipDataError(f"timestamp missing field: {exc}") from exc
    if isinstance(sec, bool) or not isinstance(sec, int):
        raise ClipDataError("timestamp sec must be an integer")
    if isinstance(nanosec, bool) or not isinstance(nanosec, int):
        raise ClipDataError("timestamp nanosec must be an integer")
    if not 0 <= nanosec < 1_000_000_000:
        raise ClipDataError("timestamp nanosec is outside [0, 1e9)")
    return sec * 1_000_000_000 + nanosec


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ClipReaderError(f"required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ClipDataError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClipDataError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ClipDataError(
                        f"invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ClipDataError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                records.append(value)
    except FileNotFoundError as exc:
        raise ClipReaderError(f"required file not found: {path}") from exc
    return records


def _message_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    message = record.get("message")
    if not isinstance(message, dict):
        raise ClipDataError("JSONL topic row has no object-valued message")
    return message


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClipDataError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ClipDataError(f"{name} must be finite")
    return result


class DrivingClipReader:
    """Unified reader for one finalized test_clip_NNN directory."""

    def __init__(self, clip_directory: str | Path) -> None:
        self.clip_directory = Path(clip_directory).expanduser().resolve()
        if not self.clip_directory.is_dir():
            raise ClipReaderError(
                f"clip directory does not exist: {self.clip_directory}"
            )
        self.clip_id = self.clip_directory.name
        self.metadata = _read_json(self.clip_directory / "metadata.json")
        self.validation = _read_json(self.clip_directory / "validation.json")
        if self.validation.get("valid") is not True:
            raise ClipReaderError(f"clip is not validation-valid: {self.clip_id}")

        self.start_ns = int(self.validation["first_sim_time_ns"])
        self.end_ns = int(self.validation["last_sim_time_ns"])
        self.duration_ns = self.end_ns - self.start_ns
        if self.duration_ns <= 0:
            raise ClipDataError("clip duration must be positive")

        self._camera_indexes = self._load_camera_indexes()
        self._executed_samples: tuple[EgoStateSample, ...] | None = None
        self._executed_index: TemporalIndex[EgoStateSample] | None = None
        self._route_records: tuple[dict[str, Any], ...] | None = None
        self._route_index: TemporalIndex[dict[str, Any]] | None = None
        self._actor_records: tuple[dict[str, Any], ...] | None = None
        self._actor_index: TemporalIndex[dict[str, Any]] | None = None
        self._complete_gt: dict[str, Any] | None = None
        self._complete_gt_points: tuple[dict[str, Any], ...] | None = None
        self._complete_gt_index: TemporalIndex[dict[str, Any]] | None = None
        self._vector_map: dict[str, Any] | None = None

    @property
    def camera_indexes(self) -> Mapping[str, TemporalIndex[CameraFrameRecord]]:
        return self._camera_indexes

    def _load_camera_indexes(
        self,
    ) -> dict[str, TemporalIndex[CameraFrameRecord]]:
        indexes: dict[str, TemporalIndex[CameraFrameRecord]] = {}
        for camera_name in CAMERA_NAMES:
            camera_directory = (
                self.clip_directory / "cameras" / camera_name
            )
            rows = _read_jsonl(camera_directory / "timestamps.jsonl")
            records: list[CameraFrameRecord] = []
            for row in rows:
                image_name = row.get("image_path")
                if not isinstance(image_name, str) or not image_name:
                    raise ClipDataError(
                        f"invalid image_path for camera {camera_name}"
                    )
                image_path = camera_directory / image_name
                if not image_path.is_file():
                    raise ClipReaderError(f"image file not found: {image_path}")
                records.append(
                    CameraFrameRecord(
                        camera_name=camera_name,
                        frame_index=int(row["frame_index"]),
                        stamp_ns=int(row["stamp_ns"]),
                        frame_id=str(row.get("frame_id", "")),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        encoding=str(row["encoding"]),
                        image_path=image_path,
                    )
                )
            indexes[camera_name] = TemporalIndex(
                [record.stamp_ns for record in records],
                records,
                name=f"{self.clip_id}:{camera_name}",
            )
        return indexes

    def get_multicamera_frames(
        self,
        anchor_ns: int,
        *,
        offsets_ns: Sequence[int] = DEFAULT_FRAME_OFFSETS_NS,
        tolerance_ns: int = DEFAULT_CAMERA_TOLERANCE_NS,
        prefer_exact: bool = True,
    ) -> MultiCameraSequence[CameraFrameRecord] | None:
        return synchronize_camera_sequence(
            self._camera_indexes,
            anchor_ns,
            offsets_ns,
            tolerance_ns=tolerance_ns,
            prefer_exact=prefer_exact,
            require_unique_per_camera=True,
        )

    def _ensure_executed_path(self) -> None:
        if self._executed_index is not None:
            return
        rows = _read_jsonl(
            self.clip_directory / "ego" / "executed_path_points.jsonl"
        )
        samples: list[EgoStateSample] = []
        for row in rows:
            pose = pose2d_from_pose_mapping(row["pose"])
            acceleration = row.get("linear_acceleration", {})
            samples.append(
                EgoStateSample(
                    stamp_ns=stamp_mapping_to_ns(row["stamp"]),
                    pose=pose,
                    speed=_finite(row.get("speed", 0.0), "speed"),
                    longitudinal_acceleration=_finite(
                        acceleration.get("x", 0.0),
                        "linear_acceleration.x",
                    ),
                    yaw_rate=_finite(row.get("yaw_rate", 0.0), "yaw_rate"),
                )
            )
        self._executed_samples = tuple(samples)
        self._executed_index = TemporalIndex(
            [sample.stamp_ns for sample in samples],
            samples,
            name=f"{self.clip_id}:executed_path",
        )

    def _interpolate_ego(self, target_ns: int) -> EgoStateSample | None:
        self._ensure_executed_path()
        assert self._executed_index is not None
        exact = self._executed_index.exact(target_ns)
        if exact is not None:
            return exact.value
        before = self._executed_index.at_or_before(target_ns)
        after = self._executed_index.at_or_after(target_ns)
        if before is None or after is None:
            return None
        span = after.timestamp_ns - before.timestamp_ns
        if span <= 0:
            return None
        ratio = (target_ns - before.timestamp_ns) / span
        first = before.value
        second = after.value
        return EgoStateSample(
            stamp_ns=target_ns,
            pose=Pose2D(
                x=interpolate_scalar(first.pose.x, second.pose.x, ratio),
                y=interpolate_scalar(first.pose.y, second.pose.y, ratio),
                yaw=interpolate_angle(first.pose.yaw, second.pose.yaw, ratio),
            ),
            speed=interpolate_scalar(first.speed, second.speed, ratio),
            longitudinal_acceleration=interpolate_scalar(
                first.longitudinal_acceleration,
                second.longitudinal_acceleration,
                ratio,
            ),
            yaw_rate=interpolate_scalar(
                first.yaw_rate,
                second.yaw_rate,
                ratio,
            ),
        )

    def get_ego_history(
        self,
        anchor_ns: int,
        *,
        num_waypoints: int = DEFAULT_EGO_HISTORY_POINTS,
        interval_ns: int = DEFAULT_EGO_INTERVAL_NS,
    ) -> tuple[EgoHistoryWaypoint, ...] | None:
        if num_waypoints <= 0:
            raise ValueError("num_waypoints must be positive")
        if interval_ns <= 0:
            raise ValueError("interval_ns must be positive")
        target_stamps = tuple(
            anchor_ns - interval_ns * offset
            for offset in range(num_waypoints - 1, -1, -1)
        )
        samples: list[EgoStateSample] = []
        for target_stamp in target_stamps:
            sample = self._interpolate_ego(target_stamp)
            if sample is None:
                return None
            samples.append(sample)
        anchor = samples[-1]
        waypoints: list[EgoHistoryWaypoint] = []
        for target_stamp, sample in zip(target_stamps, samples):
            relative = map_pose_to_anchor_ego(sample.pose, anchor.pose)
            waypoints.append(
                EgoHistoryWaypoint(
                    target_stamp_ns=target_stamp,
                    relative_x=relative.relative_x,
                    relative_y=relative.relative_y,
                    sin_relative_yaw=math.sin(relative.relative_yaw),
                    cos_relative_yaw=math.cos(relative.relative_yaw),
                    speed=sample.speed,
                    longitudinal_acceleration=(
                        sample.longitudinal_acceleration
                    ),
                    yaw_rate=sample.yaw_rate,
                    relative_time=(target_stamp - anchor_ns) / 1e9,
                )
            )
        return tuple(waypoints)

    def _ensure_routes(self) -> None:
        if self._route_index is not None:
            return
        rows = _read_jsonl(
            self.clip_directory
            / "route"
            / "navigation_route_local.jsonl"
        )
        messages = tuple(_message_envelope(row) for row in rows)
        self._route_records = messages
        self._route_index = TemporalIndex(
            [stamp_mapping_to_ns(message["reference_stamp"]) for message in messages],
            messages,
            name=f"{self.clip_id}:navigation_route",
        )

    def get_navigation_route_at(
        self,
        anchor_ns: int,
        *,
        maximum_age_ns: int = DEFAULT_ROUTE_MAX_AGE_NS,
    ) -> TimedMessage | None:
        self._ensure_routes()
        assert self._route_index is not None
        match = self._route_index.at_or_before(
            anchor_ns,
            tolerance_ns=maximum_age_ns,
        )
        if match is None:
            return None
        return TimedMessage(
            stamp_ns=match.timestamp_ns,
            time_error_ns=match.error_ns,
            message=match.value,
        )

    def _ensure_actors(self) -> None:
        if self._actor_index is not None:
            return
        rows = _read_jsonl(self.clip_directory / "actors" / "current.jsonl")
        messages = tuple(_message_envelope(row) for row in rows)
        self._actor_records = messages
        self._actor_index = TemporalIndex(
            [stamp_mapping_to_ns(message["stamp"]) for message in messages],
            messages,
            name=f"{self.clip_id}:actors_current",
        )

    def get_actors_at(
        self,
        anchor_ns: int,
        *,
        tolerance_ns: int = DEFAULT_ACTOR_TOLERANCE_NS,
    ) -> TimedMessage | None:
        self._ensure_actors()
        assert self._actor_index is not None
        match = self._actor_index.nearest(
            anchor_ns,
            tolerance_ns=tolerance_ns,
        )
        if match is None:
            return None
        return TimedMessage(
            stamp_ns=match.timestamp_ns,
            time_error_ns=match.error_ns,
            message=match.value,
        )

    def _ensure_complete_gt(self) -> None:
        if self._complete_gt_index is not None:
            return
        complete_gt = _read_json(
            self.clip_directory
            / "ego"
            / "complete_recording_ground_truth.json"
        )
        trajectory = complete_gt.get("trajectory")
        if not isinstance(trajectory, dict):
            raise ClipDataError("complete GT has no trajectory object")
        raw_points = trajectory.get("points")
        if not isinstance(raw_points, list):
            raise ClipDataError("complete GT trajectory has no points list")
        points = tuple(
            point for point in raw_points if isinstance(point, dict)
        )
        self._complete_gt = complete_gt
        self._complete_gt_points = points
        self._complete_gt_index = TemporalIndex(
            [stamp_mapping_to_ns(point["stamp"]) for point in points],
            points,
            name=f"{self.clip_id}:complete_gt",
        )

    def get_future_ego_trajectory(
        self,
        anchor_ns: int,
        *,
        horizon_ns: int = 3_000_000_000,
        include_anchor: bool = True,
    ) -> FutureTrajectory | None:
        if horizon_ns <= 0:
            raise ValueError("horizon_ns must be positive")
        self._ensure_complete_gt()
        assert self._complete_gt_index is not None
        assert self._complete_gt_points is not None
        start = self._complete_gt_index.at_or_after(anchor_ns)
        end_ns = anchor_ns + horizon_ns
        last_timestamp_ns = self._complete_gt_index.last_timestamp_ns
        if (
            start is None
            or last_timestamp_ns is None
            or last_timestamp_ns < end_ns
        ):
            return None
        end = self._complete_gt_index.at_or_before(end_ns)
        if end is None:
            return None
        first_index = start.index
        if not include_anchor and start.timestamp_ns == anchor_ns:
            first_index += 1
        selected = self._complete_gt_points[first_index : end.index + 1]
        if not selected:
            return None
        return FutureTrajectory(
            anchor_ns=anchor_ns,
            end_ns=end_ns,
            points=tuple(selected),
        )

    def get_vector_map(self) -> dict[str, Any]:
        if self._vector_map is None:
            self._vector_map = _read_json(
                self.clip_directory / "map" / "vector_map.json"
            )
        return self._vector_map

    def validate_anchor(
        self,
        anchor_ns: int,
        *,
        future_horizon_ns: int = 3_000_000_000,
    ) -> AnchorValidation:
        checks = {
            "anchor_within_clip": self.start_ns <= anchor_ns <= self.end_ns,
            "multicamera_frames": self.get_multicamera_frames(anchor_ns) is not None,
            "ego_history": self.get_ego_history(anchor_ns) is not None,
            "navigation_route": self.get_navigation_route_at(anchor_ns) is not None,
            "actors_current": self.get_actors_at(anchor_ns) is not None,
            "future_ego_trajectory": self.get_future_ego_trajectory(
                anchor_ns,
                horizon_ns=future_horizon_ns,
            )
            is not None,
        }
        reasons = tuple(name for name, passed in checks.items() if not passed)
        return AnchorValidation(
            anchor_ns=anchor_ns,
            usable=not reasons,
            checks=checks,
            reasons=reasons,
        )

    @staticmethod
    def ego_history_to_dicts(
        history: Sequence[EgoHistoryWaypoint],
    ) -> list[dict[str, Any]]:
        return [asdict(waypoint) for waypoint in history]
