"""Adapter tests between ActorCameraProjection and observability rules."""

from __future__ import annotations

import pytest

from step7.actor_box_image_projection_v01 import ActorCameraProjection
from step7.actor_observability_rules_v01 import (
    FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO,
    FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT,
    evaluate_geometric_observability,
)


def make_projection(
    *,
    camera_name: str,
    area: float,
    height: float,
    ratio: float,
) -> ActorCameraProjection:
    """Construct a minimal valid-shaped projection for rule adaptation tests."""
    return ActorCameraProjection(
        camera_name=camera_name,
        track_id="test-track",
        actor_class="automobile",
        corner_count=8,
        edge_count=12,
        edge_samples_per_edge=0,
        camera_sample_count=8,
        positive_depth_sample_count=8,
        within_fov_sample_count=8,
        inside_image_sample_count=8,
        projected_bbox=None,
        clipped_bbox=None,
        projected_area_px=area,
        inside_image_area_px=area,
        inside_image_ratio=ratio,
        projected_hull=(),
        clipped_hull=(),
        projected_hull_area_px=area,
        inside_image_hull_area_px=area,
        inside_image_hull_ratio=ratio,
        projected_height_px=height,
        minimum_depth_m=10.0,
        maximum_depth_m=12.0,
        truncated=ratio < 1.0,
        projection_valid=True,
        failure_reason=None,
    )


def evaluate_projection(projection: ActorCameraProjection):
    """Pass production projection fields directly into the pure rule."""
    assert projection.projection_valid is True
    return evaluate_geometric_observability(
        camera_name=projection.camera_name,
        inside_image_hull_area_px=projection.inside_image_hull_area_px,
        projected_height_px=projection.projected_height_px,
        inside_image_hull_ratio=projection.inside_image_hull_ratio,
    )


@pytest.mark.parametrize(
    ("camera_name", "height"),
    (("front_wide", 5.0), ("cross_left", 6.0), ("cross_right", 6.0)),
)
def test_projection_fields_directly_drive_height_policies(camera_name, height):
    projection = make_projection(
        camera_name=camera_name,
        area=0.0,
        height=height,
        ratio=0.0,
    )
    decision = evaluate_projection(projection)
    assert decision.candidate is True
    assert decision.failure_reason is None


def test_projection_fields_directly_drive_height_rejection():
    projection = make_projection(
        camera_name="cross_left",
        area=10000.0,
        height=5.99,
        ratio=1.0,
    )
    decision = evaluate_projection(projection)
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT


def test_projection_fields_directly_drive_front_tele_area_branch():
    projection = make_projection(
        camera_name="front_tele",
        area=512.0,
        height=10.0,
        ratio=0.10,
    )
    decision = evaluate_projection(projection)
    assert decision.candidate is True
    assert decision.failure_reason is None


def test_projection_fields_directly_drive_front_tele_slender_height_branch():
    projection = make_projection(
        camera_name="front_tele",
        area=126.3,
        height=17.06,
        ratio=1.0,
    )
    decision = evaluate_projection(projection)
    assert decision.candidate is True
    assert decision.failure_reason is None


def test_projection_fields_directly_drive_front_tele_fragment_rejection():
    projection = make_projection(
        camera_name="front_tele",
        area=10000.0,
        height=200.0,
        ratio=0.099,
    )
    decision = evaluate_projection(projection)
    assert decision.candidate is False
    assert (
        decision.failure_reason
        == FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO
    )


def test_projection_to_dict_preserves_observability_field_names_and_values():
    projection = make_projection(
        camera_name="front_tele",
        area=512.0,
        height=16.0,
        ratio=0.10,
    )
    value = projection.to_dict()
    assert value["camera_name"] == "front_tele"
    assert value["inside_image_hull_area_px"] == 512.0
    assert value["projected_height_px"] == 16.0
    assert value["inside_image_hull_ratio"] == 0.10
