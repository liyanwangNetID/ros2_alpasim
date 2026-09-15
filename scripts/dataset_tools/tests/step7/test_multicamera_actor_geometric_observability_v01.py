from types import SimpleNamespace

import pytest

from step7 import multicamera_actor_geometric_observability_v01 as target
from step7.actor_box_image_projection_v01 import ActorCameraProjection
from step7.scene_fact_schema_v01 import CAMERA_NAMES


def actor(track_id, actor_class="automobile"):
    return {"track_id": track_id, "label_class": actor_class}


def projection(camera_name, track_id, actor_class, valid=True):
    return ActorCameraProjection(
        camera_name=camera_name,
        track_id=track_id,
        actor_class=actor_class,
        corner_count=8,
        edge_count=12,
        edge_samples_per_edge=0,
        camera_sample_count=1 if valid else 0,
        positive_depth_sample_count=1 if valid else 0,
        within_fov_sample_count=1 if valid else 0,
        inside_image_sample_count=1 if valid else 0,
        projected_bbox=None,
        clipped_bbox=None,
        projected_area_px=0.0,
        inside_image_area_px=0.0,
        inside_image_ratio=0.0,
        projected_hull=(),
        clipped_hull=(),
        projected_hull_area_px=0.0,
        inside_image_hull_area_px=(600.0 if valid else 0.0),
        inside_image_hull_ratio=(1.0 if valid else 0.0),
        projected_height_px=(20.0 if valid else 0.0),
        minimum_depth_m=(10.0 if valid else None),
        maximum_depth_m=(12.0 if valid else None),
        truncated=False,
        projection_valid=valid,
        failure_reason=(None if valid else "box_outside_camera_fov"),
    )


def calibrations():
    return {
        name: SimpleNamespace(camera_name=name)
        for name in CAMERA_NAMES
    }


def install_projection_stub(monkeypatch, invalid_camera=None):
    calls = []

    def project(value, *, calibration, **kwargs):
        calls.append((value["track_id"], calibration.camera_name, kwargs))
        return projection(
            calibration.camera_name,
            str(value["track_id"]),
            str(value["label_class"]),
            valid=(calibration.camera_name != invalid_camera),
        )

    monkeypatch.setattr(target, "project_actor_box_to_camera", project)
    return calls


def call(monkeypatch, **overrides):
    calls = install_projection_stub(monkeypatch)
    values = {
        "actors": (actor("2"), actor("1", "person")),
        "recorded_ego_message": {"ego": True},
        "calibrations": calibrations(),
    }
    values.update(overrides)
    result = target.build_multicamera_actor_geometric_observability(**values)
    return result, calls


def test_builds_all_actor_projections_and_observability(monkeypatch):
    result, calls = call(monkeypatch)

    assert result.actor_count == 2
    assert tuple(item.track_id for item in result.actor_results) == ("1", "2")
    assert tuple(item.track_id for item in result.actor_observability) == ("1", "2")
    assert len(calls) == 2 * len(CAMERA_NAMES)
    assert all(
        item.observability.observability_status == "candidate_visible"
        for item in result.actor_results
    )


def test_projection_and_camera_order_is_canonical(monkeypatch):
    result, _ = call(monkeypatch)

    for item in result.actor_results:
        assert tuple(
            projection.camera_name for projection in item.projections
        ) == tuple(CAMERA_NAMES)
        assert item.observability.visible_in_cameras == tuple(CAMERA_NAMES)


def test_projection_parameters_are_forwarded(monkeypatch):
    result, calls = call(
        monkeypatch,
        samples_per_edge=9,
        near_plane_m=0.01,
        maximum_chord_error_px=0.5,
        maximum_adaptive_depth=11,
    )

    assert result.actor_count == 2
    for _, _, kwargs in calls:
        assert kwargs["samples_per_edge"] == 9
        assert kwargs["near_plane_m"] == 0.01
        assert kwargs["maximum_chord_error_px"] == 0.5
        assert kwargs["maximum_adaptive_depth"] == 11


def test_invalid_projection_is_aggregated_not_dropped(monkeypatch):
    invalid_camera = CAMERA_NAMES[-1]
    install_projection_stub(monkeypatch, invalid_camera=invalid_camera)

    result = target.build_multicamera_actor_geometric_observability(
        actors=(actor("1"),),
        recorded_ego_message={"ego": True},
        calibrations=calibrations(),
    )

    assert result.actor_count == 1
    assert invalid_camera not in result.actor_observability[0].visible_in_cameras
    assert result.actor_observability[0].observability_status == "candidate_visible"


def test_missing_camera_is_rejected(monkeypatch):
    source = calibrations()
    del source[CAMERA_NAMES[0]]

    with pytest.raises(ValueError, match="missing"):
        call(monkeypatch, calibrations=source)


def test_extra_camera_is_rejected(monkeypatch):
    source = calibrations()
    source["extra"] = SimpleNamespace(camera_name="extra")

    with pytest.raises(ValueError, match="extra"):
        call(monkeypatch, calibrations=source)


def test_duplicate_track_ids_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="must be unique"):
        call(monkeypatch, actors=(actor("1"), actor("1")))


def test_missing_track_id_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="usable track_id"):
        call(monkeypatch, actors=({"label_class": "automobile"},))


def test_missing_actor_class_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="usable label_class"):
        call(monkeypatch, actors=({"track_id": "1"},))


def test_empty_actor_set_is_supported(monkeypatch):
    result, calls = call(monkeypatch, actors=())

    assert result.actor_count == 0
    assert result.actor_results == ()
    assert result.actor_observability == ()
    assert calls == []
