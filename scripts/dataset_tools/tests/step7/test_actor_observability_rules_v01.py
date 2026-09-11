from __future__ import annotations

import math

import pytest

from actor_observability_rules_v01 import (
    FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO,
    FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT,
    FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT,
    evaluate_geometric_observability,
)


def evaluate(
    camera_name: str,
    *,
    area: float = 100.0,
    height: float = 10.0,
    ratio: float = 1.0,
):
    return evaluate_geometric_observability(
        camera_name=camera_name,
        inside_image_hull_area_px=area,
        projected_height_px=height,
        inside_image_hull_ratio=ratio,
    )


@pytest.mark.parametrize(
    ("camera_name", "threshold"),
    (("front_wide", 5.0), ("cross_left", 6.0), ("cross_right", 6.0)),
)
def test_height_policy_boundary_is_inclusive(camera_name, threshold):
    decision = evaluate(camera_name, height=threshold)
    assert decision.candidate is True
    assert decision.failure_reason is None


@pytest.mark.parametrize(
    ("camera_name", "threshold"),
    (("front_wide", 5.0), ("cross_left", 6.0), ("cross_right", 6.0)),
)
def test_height_policy_rejects_value_immediately_below_boundary(
    camera_name,
    threshold,
):
    decision = evaluate(camera_name, height=math.nextafter(threshold, 0.0))
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT


def test_wide_and_cross_height_policy_does_not_require_area_or_ratio():
    decision = evaluate("front_wide", area=0.0, height=5.0, ratio=0.0)
    assert decision.candidate is True
    assert decision.failure_reason is None


def test_front_tele_area_branch_passes_at_boundary():
    decision = evaluate("front_tele", area=512.0, height=0.0, ratio=0.10)
    assert decision.candidate is True
    assert decision.failure_reason is None


def test_front_tele_height_branch_preserves_complete_slender_target():
    decision = evaluate("front_tele", area=126.3, height=17.06, ratio=1.0)
    assert decision.candidate is True
    assert decision.failure_reason is None


def test_front_tele_rejects_when_both_scale_features_are_below_threshold():
    decision = evaluate(
        "front_tele",
        area=math.nextafter(512.0, 0.0),
        height=math.nextafter(16.0, 0.0),
        ratio=1.0,
    )
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT


def test_front_tele_ratio_boundary_is_inclusive():
    decision = evaluate("front_tele", area=512.0, height=16.0, ratio=0.10)
    assert decision.candidate is True
    assert decision.failure_reason is None


def test_front_tele_rejects_scale_pass_with_ratio_below_boundary():
    decision = evaluate(
        "front_tele",
        area=1000.0,
        height=100.0,
        ratio=math.nextafter(0.10, 0.0),
    )
    assert decision.candidate is False
    assert (
        decision.failure_reason
        == FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO
    )


def test_front_tele_does_not_allow_large_area_to_bypass_ratio_guard():
    decision = evaluate("front_tele", area=10000.0, height=200.0, ratio=0.01)
    assert decision.candidate is False
    assert (
        decision.failure_reason
        == FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO
    )


@pytest.mark.parametrize("camera_name", ("rear", "", "FRONT_WIDE"))
def test_unknown_camera_is_rejected(camera_name):
    with pytest.raises(ValueError, match="Unsupported camera_name"):
        evaluate(camera_name)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("area", -1.0, "inside_image_hull_area_px"),
        ("height", -1.0, "projected_height_px"),
        ("ratio", -0.01, "inside_image_hull_ratio"),
        ("ratio", 1.01, "inside_image_hull_ratio"),
    ),
)
def test_invalid_numeric_range_is_rejected(field, value, message):
    arguments = {"area": 100.0, "height": 10.0, "ratio": 1.0}
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        evaluate("front_wide", **arguments)
