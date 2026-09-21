from step7.actor_short_history_export_v01 import (
    METRIC_FIELDS,
    actor_short_history_export_row,
    validate_actor_short_history_export_row,
)
from step7.actor_short_history_v01 import ActorShortHistoryFeatures


def feature(status="usable"):
    usable = status == "usable"
    value = 1.0 if usable else None
    return ActorShortHistoryFeatures(
        track_id="1",
        actor_class="automobile",
        status=status,
        sample_count=6,
        history_span_s=0.5,
        maximum_sample_gap_s=0.1,
        longitudinal_displacement_m=value,
        lateral_displacement_m=value,
        distance_change_m=value,
        mean_longitudinal_velocity_mps=value,
        mean_lateral_velocity_mps=value,
        mean_distance_rate_mps=value,
        heading_change_rad=value,
        mean_actor_speed_mps=value,
        mean_actor_yaw_rate_rps=value,
        points=(),
    )


def test_compact_row_has_metrics_but_no_raw_points():
    row = actor_short_history_export_row(
        keyframe={"anchor_id": "a", "clip_id": "c", "anchor_ns": 10},
        feature=feature(),
        history_duration_ns=1_500_000_000,
    )
    validate_actor_short_history_export_row(row)
    assert "points" not in row
    assert "history" not in row
    assert row["mean_distance_rate_mps"] == 1.0
    assert all(field in row for field in METRIC_FIELDS)


def test_nonusable_row_has_explicit_null_metrics():
    row = actor_short_history_export_row(
        keyframe={"anchor_id": "a", "clip_id": "c", "anchor_ns": 10},
        feature=feature("insufficient_span"),
        history_duration_ns=1_500_000_000,
    )
    validate_actor_short_history_export_row(row)
    assert all(row[field] is None for field in METRIC_FIELDS)
