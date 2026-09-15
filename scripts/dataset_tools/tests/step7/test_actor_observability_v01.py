from __future__ import annotations

from dataclasses import replace

import pytest

from step7.actor_box_image_projection_v01 import ActorCameraProjection
from step7.actor_observability_v01 import aggregate_actor_observability
from step7.scene_fact_schema_v01 import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION


def projection(
    camera_name: str,
    *,
    track_id: str = "18",
    actor_class: str = "automobile",
    valid: bool = True,
    failure_reason: str | None = None,
    area: float = 1000.0,
    height: float = 20.0,
    ratio: float = 1.0,
) -> ActorCameraProjection:
    return ActorCameraProjection(
        camera_name=camera_name,
        track_id=track_id,
        actor_class=actor_class,
        corner_count=8,
        edge_count=12,
        edge_samples_per_edge=0,
        camera_sample_count=8,
        positive_depth_sample_count=8 if valid else 0,
        within_fov_sample_count=8 if valid else 0,
        inside_image_sample_count=8 if valid else 0,
        projected_bbox=None,
        clipped_bbox=None,
        projected_area_px=area if valid else 0.0,
        inside_image_area_px=area if valid else 0.0,
        inside_image_ratio=ratio if valid else 0.0,
        projected_hull=(),
        clipped_hull=(),
        projected_hull_area_px=area if valid else 0.0,
        inside_image_hull_area_px=area if valid else 0.0,
        inside_image_hull_ratio=ratio if valid else 0.0,
        projected_height_px=height if valid else 0.0,
        minimum_depth_m=10.0 if valid else None,
        maximum_depth_m=12.0 if valid else None,
        truncated=valid and ratio < 1.0,
        projection_valid=valid,
        failure_reason=failure_reason,
    )


def all_projections(**overrides):
    return {
        name: overrides.get(name, projection(name))
        for name in CAMERA_NAMES
    }


def test_aggregates_visible_cameras_in_canonical_order():
    values = all_projections(
        front_wide=projection("front_wide", height=4.9),
        cross_left=projection("cross_left", height=6.0),
    )
    result = aggregate_actor_observability(values)
    assert result.observability_status == "candidate_visible"
    assert result.visible_in_cameras == (
        "front_tele",
        "cross_left",
        "cross_right",
    )
    assert result.actor_to_actor_occlusion_evaluated is False
    assert result.static_occlusion_evaluated is False


def test_all_conclusive_camera_failures_produce_not_visible():
    values = all_projections(
        **{
            name: projection(
                name,
                valid=False,
                failure_reason="box_outside_camera_fov",
            )
            for name in CAMERA_NAMES
        }
    )
    result = aggregate_actor_observability(values)
    assert result.observability_status == "not_visible"
    assert result.visible_in_cameras == ()
    assert all(
        not item.geometric_observability_candidate
        for item in result.camera_observability
    )


def test_valid_but_small_projections_produce_not_visible():
    values = all_projections(
        front_wide=projection("front_wide", height=4.9),
        front_tele=projection("front_tele", area=511.0, height=15.9),
        cross_left=projection("cross_left", height=5.9),
        cross_right=projection("cross_right", height=5.9),
    )
    result = aggregate_actor_observability(values)
    assert result.observability_status == "not_visible"
    assert result.visible_in_cameras == ()
    assert all(item.failure_reason for item in result.camera_observability)


def test_to_dict_uses_json_ready_lists_and_version():
    result = aggregate_actor_observability(all_projections())
    value = result.to_dict()
    assert value["observability_format_version"] == OBSERVABILITY_FORMAT_VERSION
    assert value["visible_in_cameras"] == list(CAMERA_NAMES)
    assert isinstance(value["camera_observability"], list)
    assert len(value["camera_observability"]) == 4


def test_missing_camera_is_rejected():
    values = all_projections()
    del values["cross_right"]
    with pytest.raises(ValueError, match="Expected exactly CAMERA_NAMES"):
        aggregate_actor_observability(values)


def test_extra_camera_is_rejected():
    values = all_projections()
    values["rear"] = projection("rear")
    with pytest.raises(ValueError, match="Expected exactly CAMERA_NAMES"):
        aggregate_actor_observability(values)


def test_mismatched_projection_camera_is_rejected():
    values = all_projections()
    values["front_wide"] = projection("cross_left")
    with pytest.raises(ValueError, match="Projection camera mismatch"):
        aggregate_actor_observability(values)


def test_mixed_track_ids_are_rejected():
    values = all_projections()
    values["front_wide"] = projection("front_wide", track_id="other")
    with pytest.raises(ValueError, match="same track_id"):
        aggregate_actor_observability(values)


def test_mixed_actor_classes_are_rejected():
    values = all_projections()
    values["front_wide"] = projection(
        "front_wide",
        actor_class="person",
    )
    with pytest.raises(ValueError, match="same actor_class"):
        aggregate_actor_observability(values)


def test_invalid_projection_requires_failure_reason():
    values = all_projections()
    values["front_wide"] = projection("front_wide", valid=False)
    with pytest.raises(ValueError, match="must provide a failure_reason"):
        aggregate_actor_observability(values)


def test_valid_projection_forbids_failure_reason():
    values = all_projections()
    values["front_wide"] = projection(
        "front_wide",
        valid=True,
        failure_reason="unexpected",
    )
    with pytest.raises(ValueError, match="must not provide a failure_reason"):
        aggregate_actor_observability(values)
