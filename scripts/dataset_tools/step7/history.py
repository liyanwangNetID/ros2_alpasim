"""Step 7F Actor short-history feature domain.

Contains historical sampling, compact export, validation, and summary contracts.
Only current and past Actor snapshots are accepted.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
from step2.coordinate_utils import (
    Pose2D,
    map_pose_to_anchor_ego,
    pose2d_from_pose_mapping,
    shortest_angular_distance,
)
from typing import Any, Mapping
from collections import Counter


DEFAULT_HISTORY_DURATION_NS = 1_500_000_000


DEFAULT_MAXIMUM_SAMPLE_GAP_NS = 250_000_000


DEFAULT_MINIMUM_SAMPLE_COUNT = 3


DEFAULT_MINIMUM_HISTORY_SPAN_NS = 500_000_000


HISTORY_STATUSES = frozenset({
    "usable",
    "insufficient_samples",
    "insufficient_span",
    "discontinuous",
    "anchor_actor_missing",
})


@dataclass(frozen=True, slots=True)
class ActorHistoryPoint:
    stamp_ns: int
    relative_time_s: float
    relative_x_m: float
    relative_y_m: float
    planar_distance_m: float
    relative_yaw_rad: float
    actor_speed_mps: float
    actor_yaw_rate_rps: float


@dataclass(frozen=True, slots=True)
class ActorShortHistoryFeatures:
    track_id: str
    actor_class: str
    status: str
    sample_count: int
    history_span_s: float
    maximum_sample_gap_s: float | None
    longitudinal_displacement_m: float | None
    lateral_displacement_m: float | None
    distance_change_m: float | None
    mean_longitudinal_velocity_mps: float | None
    mean_lateral_velocity_mps: float | None
    mean_distance_rate_mps: float | None
    heading_change_rad: float | None
    mean_actor_speed_mps: float | None
    mean_actor_yaw_rate_rps: float | None
    points: tuple[ActorHistoryPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["points"] = [asdict(point) for point in self.points]
        return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _actors_by_id(message: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if message.get("pose_frame_id") != "map":
        raise ValueError("Actor snapshot pose_frame_id must be 'map'")
    if message.get("dynamics_frame_id") != "map":
        raise ValueError("Actor snapshot dynamics_frame_id must be 'map'")
    actors = message.get("actors")
    if not isinstance(actors, list):
        raise TypeError("Actor snapshot actors must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for actor in actors:
        if not isinstance(actor, Mapping):
            raise TypeError("Actor must be a mapping")
        track_id = str(actor.get("track_id", ""))
        if not track_id:
            raise ValueError("Actor track_id must be non-empty")
        if track_id in result:
            raise ValueError("Actor snapshot contains duplicate track_id values")
        result[track_id] = actor
    return result


def _point(actor: Mapping[str, Any], stamp_ns: int, anchor_ns: int, anchor_pose: Pose2D) -> ActorHistoryPoint:
    pose = pose2d_from_pose_mapping(actor["pose"])
    relative = map_pose_to_anchor_ego(pose, anchor_pose)
    return ActorHistoryPoint(
        stamp_ns=stamp_ns,
        relative_time_s=(stamp_ns - anchor_ns) / 1e9,
        relative_x_m=relative.relative_x,
        relative_y_m=relative.relative_y,
        planar_distance_m=math.hypot(relative.relative_x, relative.relative_y),
        relative_yaw_rad=relative.relative_yaw,
        actor_speed_mps=_finite(actor.get("speed"), "actor.speed"),
        actor_yaw_rate_rps=_finite(actor.get("yaw_rate"), "actor.yaw_rate"),
    )


def compute_actor_short_history_features(
    *,
    track_id: str,
    actor_class: str,
    anchor_ns: int,
    anchor_ego_pose: Pose2D,
    snapshots: Sequence[Any],
    maximum_sample_gap_ns: int = DEFAULT_MAXIMUM_SAMPLE_GAP_NS,
    minimum_sample_count: int = DEFAULT_MINIMUM_SAMPLE_COUNT,
    minimum_history_span_ns: int = DEFAULT_MINIMUM_HISTORY_SPAN_NS,
) -> ActorShortHistoryFeatures:
    """Compute one current Actor's fixed-Anchor-frame history features."""
    if not track_id or not actor_class:
        raise ValueError("track_id and actor_class must be non-empty")
    if maximum_sample_gap_ns <= 0 or minimum_sample_count <= 0 or minimum_history_span_ns < 0:
        raise ValueError("history quality parameters are invalid")

    points = []
    anchor_actor_found = False
    previous_stamp = None
    for snapshot in snapshots:
        stamp_ns = int(snapshot.stamp_ns)
        if stamp_ns > anchor_ns:
            raise ValueError("Actor history must not contain future snapshots")
        if previous_stamp is not None and stamp_ns <= previous_stamp:
            raise ValueError("Actor history snapshots must be strictly ordered")
        previous_stamp = stamp_ns
        actors = _actors_by_id(snapshot.message)
        actor = actors.get(track_id)
        if actor is None:
            continue
        if str(actor.get("label_class")) != actor_class:
            raise ValueError("Actor label_class changed within history window")
        points.append(_point(actor, stamp_ns, anchor_ns, anchor_ego_pose))
        if stamp_ns == anchor_ns:
            anchor_actor_found = True

    point_values = tuple(points)
    sample_count = len(point_values)
    span_ns = point_values[-1].stamp_ns - point_values[0].stamp_ns if sample_count >= 2 else 0
    gaps = [b.stamp_ns - a.stamp_ns for a, b in zip(point_values, point_values[1:])]
    maximum_gap_ns = max(gaps) if gaps else None

    if not anchor_actor_found:
        status = "anchor_actor_missing"
    elif sample_count < minimum_sample_count:
        status = "insufficient_samples"
    elif span_ns < minimum_history_span_ns:
        status = "insufficient_span"
    elif maximum_gap_ns is not None and maximum_gap_ns > maximum_sample_gap_ns:
        status = "discontinuous"
    else:
        status = "usable"

    metrics: dict[str, float | None] = {
        "longitudinal_displacement_m": None,
        "lateral_displacement_m": None,
        "distance_change_m": None,
        "mean_longitudinal_velocity_mps": None,
        "mean_lateral_velocity_mps": None,
        "mean_distance_rate_mps": None,
        "heading_change_rad": None,
        "mean_actor_speed_mps": None,
        "mean_actor_yaw_rate_rps": None,
    }
    if status == "usable":
        first, last = point_values[0], point_values[-1]
        span_s = span_ns / 1e9
        longitudinal = last.relative_x_m - first.relative_x_m
        lateral = last.relative_y_m - first.relative_y_m
        distance_change = last.planar_distance_m - first.planar_distance_m
        metrics.update({
            "longitudinal_displacement_m": longitudinal,
            "lateral_displacement_m": lateral,
            "distance_change_m": distance_change,
            "mean_longitudinal_velocity_mps": longitudinal / span_s,
            "mean_lateral_velocity_mps": lateral / span_s,
            "mean_distance_rate_mps": distance_change / span_s,
            "heading_change_rad": shortest_angular_distance(first.relative_yaw_rad, last.relative_yaw_rad),
            "mean_actor_speed_mps": sum(p.actor_speed_mps for p in point_values) / sample_count,
            "mean_actor_yaw_rate_rps": sum(p.actor_yaw_rate_rps for p in point_values) / sample_count,
        })

    return ActorShortHistoryFeatures(
        track_id=track_id,
        actor_class=actor_class,
        status=status,
        sample_count=sample_count,
        history_span_s=span_ns / 1e9,
        maximum_sample_gap_s=None if maximum_gap_ns is None else maximum_gap_ns / 1e9,
        points=point_values,
        **metrics,
    )


def compute_snapshot_actor_short_histories(
    *, anchor_ns: int, anchor_ego_pose: Pose2D, snapshots: Sequence[Any]
) -> tuple[ActorShortHistoryFeatures, ...]:
    """Compute histories for every Actor in the exact Anchor snapshot."""
    exact = [item for item in snapshots if int(item.stamp_ns) == anchor_ns]
    if len(exact) != 1:
        raise ValueError("exactly one Anchor Actor snapshot is required")
    current = _actors_by_id(exact[0].message)
    return tuple(
        compute_actor_short_history_features(
            track_id=track_id,
            actor_class=str(actor["label_class"]),
            anchor_ns=anchor_ns,
            anchor_ego_pose=anchor_ego_pose,
            snapshots=snapshots,
        )
        for track_id, actor in sorted(current.items())
    )


SCHEMA_VERSION = "step7f-actor-short-history-v01"


METRIC_FIELDS = (
    "longitudinal_displacement_m",
    "lateral_displacement_m",
    "distance_change_m",
    "mean_longitudinal_velocity_mps",
    "mean_lateral_velocity_mps",
    "mean_distance_rate_mps",
    "heading_change_rad",
    "mean_actor_speed_mps",
    "mean_actor_yaw_rate_rps",
)


def actor_short_history_export_row(
    *,
    keyframe: Mapping[str, Any],
    feature: ActorShortHistoryFeatures,
    history_duration_ns: int,
) -> dict[str, Any]:
    """Build one compact row without repeating every raw history point."""
    for field in ("anchor_id", "clip_id", "anchor_ns"):
        if field not in keyframe:
            raise ValueError(f"keyframe is missing {field}")
    row = {
        "schema_version": SCHEMA_VERSION,
        "anchor_id": str(keyframe["anchor_id"]),
        "clip_id": str(keyframe["clip_id"]),
        "anchor_ns": int(keyframe["anchor_ns"]),
        "track_id": feature.track_id,
        "label_class": feature.actor_class,
        "history_duration_ns": int(history_duration_ns),
        "history_status": feature.status,
        "sample_count": feature.sample_count,
        "history_span_s": feature.history_span_s,
        "maximum_sample_gap_s": feature.maximum_sample_gap_s,
    }
    for field in METRIC_FIELDS:
        row[field] = getattr(feature, field)
    return row


def validate_actor_short_history_export_row(row: Mapping[str, Any]) -> None:
    """Validate the compact production contract used by downstream Step 7."""
    required = {
        "schema_version", "anchor_id", "clip_id", "anchor_ns", "track_id",
        "label_class", "history_duration_ns", "history_status", "sample_count",
        "history_span_s", "maximum_sample_gap_s", *METRIC_FIELDS,
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"history export row is missing fields: {missing}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected history export schema")
    if "points" in row or "history" in row:
        raise ValueError("compact history export must not embed raw history points")
    usable = row["history_status"] == "usable"
    metric_values = [row[field] for field in METRIC_FIELDS]
    if usable and any(value is None for value in metric_values):
        raise ValueError("usable history row must contain every motion metric")
    if not usable and any(value is not None for value in metric_values):
        raise ValueError("non-usable history row must not contain motion metrics")


def summarize_actor_short_history_rows(
    *, keyframes: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    anchor_ids = [str(item["anchor_id"]) for item in keyframes]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("keyframe anchor_id values must be unique")
    identities = [(str(row["anchor_id"]), str(row["track_id"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("history row identities must be unique")
    unknown = sorted({anchor for anchor, _ in identities} - set(anchor_ids))
    if unknown:
        raise ValueError("history rows contain unknown anchor_id values")
    status_counts = Counter(str(row["history_status"]) for row in rows)
    class_counts = Counter(str(row["label_class"]) for row in rows)
    exported = {anchor for anchor, _ in identities}
    return {
        "schema_version": "step7f-actor-short-history-summary-v01",
        "keyframe_count": len(anchor_ids),
        "actor_row_count": len(rows),
        "anchor_with_actor_rows_count": len(exported),
        "anchor_without_actor_rows_count": len(set(anchor_ids) - exported),
        "history_status_counts": dict(sorted(status_counts.items())),
        "actor_class_counts": dict(sorted(class_counts.items())),
    }
