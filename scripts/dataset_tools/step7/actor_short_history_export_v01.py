"""Compact Step 7F Actor short-history export rows."""
from __future__ import annotations

from typing import Any, Mapping

from step7.actor_short_history_v01 import ActorShortHistoryFeatures

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
