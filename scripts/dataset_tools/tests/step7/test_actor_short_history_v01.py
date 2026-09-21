from types import SimpleNamespace

import pytest

from step2.coordinate_utils import Pose2D, yaw_to_quaternion
from step7.actor_short_history_v01 import (
    compute_actor_short_history_features,
    compute_snapshot_actor_short_histories,
)


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
