from types import SimpleNamespace
import pytest
from step2.coordinate_utils import Pose2D, yaw_to_quaternion
from step7.history import (
    compute_actor_short_history_features,
    compute_snapshot_actor_short_histories,
)
from step7.history import (
    METRIC_FIELDS,
    actor_short_history_export_row,
    validate_actor_short_history_export_row,
)
from step7.history import ActorShortHistoryFeatures
from step7.history import summarize_actor_short_history_rows


def actor(track, x, y=0.0, yaw=0.0, label="automobile"):
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    return {
        "track_id": track, "label_class": label, "is_static": False,
        "pose": {"position": {"x": x, "y": y, "z": 0.0},
                 "orientation": {"x": qx, "y": qy, "z": qz, "w": qw}},
        "speed": 2.0, "yaw_rate": 0.1,
    }


def snapshot(stamp, actors):
    return SimpleNamespace(
        stamp_ns=stamp,
        message={"pose_frame_id": "map", "dynamics_frame_id": "map", "actors": actors},
    )


def test_usable_history_is_in_fixed_anchor_frame():
    values = tuple(snapshot(t, [actor("1", t / 1e9)]) for t in (0, 250_000_000, 500_000_000))
    result = compute_actor_short_history_features(
        track_id="1", actor_class="automobile", anchor_ns=500_000_000,
        anchor_ego_pose=Pose2D(0.5, 0.0, 0.0), snapshots=values,
    )
    assert result.status == "usable"
    assert result.sample_count == 3
    assert result.points[0].relative_x_m == pytest.approx(-0.5)
    assert result.points[-1].relative_x_m == pytest.approx(0.0)
    assert result.mean_longitudinal_velocity_mps == pytest.approx(1.0)
    assert result.mean_distance_rate_mps == pytest.approx(-1.0)


def test_missing_samples_are_not_interpolated():
    values = (snapshot(0, [actor("1", 0)]), snapshot(500_000_000, [actor("1", 1)]))
    result = compute_actor_short_history_features(
        track_id="1", actor_class="automobile", anchor_ns=500_000_000,
        anchor_ego_pose=Pose2D(0, 0, 0), snapshots=values,
    )
    assert result.status == "insufficient_samples"
    assert result.mean_longitudinal_velocity_mps is None


def test_discontinuous_track_is_explicit():
    values = tuple(snapshot(t, [actor("1", 0)]) for t in (0, 100_000_000, 500_000_000))
    result = compute_actor_short_history_features(
        track_id="1", actor_class="automobile", anchor_ns=500_000_000,
        anchor_ego_pose=Pose2D(0, 0, 0), snapshots=values,
    )
    assert result.status == "discontinuous"


def test_future_snapshot_is_rejected():
    with pytest.raises(ValueError, match="future"):
        compute_actor_short_history_features(
            track_id="1", actor_class="automobile", anchor_ns=0,
            anchor_ego_pose=Pose2D(0, 0, 0), snapshots=(snapshot(1, [actor("1", 0)]),),
        )


def test_snapshot_export_contains_only_current_actors_sorted():
    values = tuple(
        snapshot(t, [actor("2", 1), actor("1", 2)])
        for t in (0, 250_000_000, 500_000_000)
    )
    result = compute_snapshot_actor_short_histories(
        anchor_ns=500_000_000, anchor_ego_pose=Pose2D(0, 0, 0), snapshots=values
    )
    assert tuple(item.track_id for item in result) == ("1", "2")


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


def test_summary_preserves_rowless_keyframes_and_statuses():
    result = summarize_actor_short_history_rows(
        keyframes=({"anchor_id": "a"}, {"anchor_id": "b"}),
        rows=({"anchor_id": "a", "track_id": "1", "history_status": "usable", "label_class": "automobile"},),
    )
    assert result["keyframe_count"] == 2
    assert result["actor_row_count"] == 1
    assert result["anchor_without_actor_rows_count"] == 1
    assert result["history_status_counts"] == {"usable": 1}


def test_duplicate_identity_is_rejected():
    row = {"anchor_id": "a", "track_id": "1", "history_status": "usable", "label_class": "automobile"}
    with pytest.raises(ValueError, match="identities must be unique"):
        summarize_actor_short_history_rows(keyframes=({"anchor_id": "a"},), rows=(row, dict(row)))
