from __future__ import annotations

import math

import pytest

from step7.triangle_perspective_depth_v01 import (
    interpolate_perspective_camera_depth,
)


def test_equal_vertex_depths_remain_constant():
    result = interpolate_perspective_camera_depth(
        (5.0, 5.0, 5.0),
        (0.2, 0.3, 0.5),
    )
    assert result.reciprocal_depth_per_m == pytest.approx(0.2)
    assert result.depth_m == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("weights", "expected_depth"),
    (
        ((1.0, 0.0, 0.0), 2.0),
        ((0.0, 1.0, 0.0), 4.0),
        ((0.0, 0.0, 1.0), 8.0),
    ),
)
def test_triangle_vertices_reproduce_vertex_depth(weights, expected_depth):
    result = interpolate_perspective_camera_depth(
        (2.0, 4.0, 8.0),
        weights,
    )
    assert result.depth_m == pytest.approx(expected_depth)


def test_edge_sample_uses_reciprocal_depth():
    result = interpolate_perspective_camera_depth(
        (2.0, 4.0, 8.0),
        (0.5, 0.5, 0.0),
    )
    assert result.reciprocal_depth_per_m == pytest.approx(0.375)
    assert result.depth_m == pytest.approx(8.0 / 3.0)
    assert result.depth_m != pytest.approx(3.0)


def test_centroid_uses_harmonic_mean_of_depths():
    result = interpolate_perspective_camera_depth(
        (2.0, 4.0, 8.0),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    )
    expected = 3.0 / (1.0 / 2.0 + 1.0 / 4.0 + 1.0 / 8.0)
    assert result.depth_m == pytest.approx(expected)


def test_matching_vertex_and_weight_reordering_preserves_result():
    forward = interpolate_perspective_camera_depth(
        (2.0, 4.0, 8.0),
        (0.2, 0.3, 0.5),
    )
    reversed_order = interpolate_perspective_camera_depth(
        (2.0, 8.0, 4.0),
        (0.2, 0.5, 0.3),
    )
    assert reversed_order == forward


def test_small_weight_roundoff_within_tolerance_is_accepted():
    result = interpolate_perspective_camera_depth(
        (2.0, 4.0, 8.0),
        (-1e-13, 0.5, 0.5000000000001),
        weight_tolerance=1e-12,
    )
    assert result.depth_m > 0.0


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_depth_count_is_rejected(count):
    with pytest.raises(ValueError, match="exactly three"):
        interpolate_perspective_camera_depth(
            tuple(1.0 for _ in range(count)),
            (1.0, 0.0, 0.0),
        )


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_weight_count_is_rejected(count):
    with pytest.raises(ValueError, match="exactly three"):
        interpolate_perspective_camera_depth(
            (1.0, 2.0, 3.0),
            tuple(1.0 / count for _ in range(count)) if count else (),
        )


@pytest.mark.parametrize("depths", ((0.0, 2.0, 3.0), (-1.0, 2.0, 3.0)))
def test_nonpositive_depth_is_rejected(depths):
    with pytest.raises(ValueError, match="positive"):
        interpolate_perspective_camera_depth(depths, (1.0, 0.0, 0.0))


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_depth_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        interpolate_perspective_camera_depth(
            (value, 2.0, 3.0),
            (1.0, 0.0, 0.0),
        )


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_weight_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        interpolate_perspective_camera_depth(
            (1.0, 2.0, 3.0),
            (value, 0.0, 1.0),
        )


def test_weights_not_summing_to_one_are_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        interpolate_perspective_camera_depth(
            (1.0, 2.0, 3.0),
            (0.2, 0.3, 0.4),
        )


@pytest.mark.parametrize(
    "weights",
    ((-0.1, 0.5, 0.6), (1.1, -0.05, -0.05)),
)
def test_weights_outside_closed_triangle_are_rejected(weights):
    with pytest.raises(ValueError, match="closed triangle"):
        interpolate_perspective_camera_depth(
            (1.0, 2.0, 3.0),
            weights,
        )


@pytest.mark.parametrize("tolerance", (0.0, -1.0, math.nan, math.inf))
def test_invalid_weight_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="weight_tolerance"):
        interpolate_perspective_camera_depth(
            (1.0, 2.0, 3.0),
            (1.0, 0.0, 0.0),
            weight_tolerance=tolerance,
        )
