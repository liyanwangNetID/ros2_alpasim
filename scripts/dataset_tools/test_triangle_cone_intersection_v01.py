from __future__ import annotations

import math

import pytest

from camera_projection_v01 import Vector3
from triangle_cone_intersection_v01 import (
    triangle_intersects_angular_fov_cone,
)


def test_inside_vertex_intersects_and_reports_vertex():
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(0.1, 0.0, 2.0),
            Vector3(4.0, 0.0, 2.0),
            Vector3(4.0, 1.0, 2.0),
        ),
        max_angle_rad=0.5,
    )
    assert result.intersects
    assert result.minimizing_location_type == "vertex"
    assert result.minimizing_feature_indices == (0,)


def test_separated_triangle_does_not_intersect():
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(10.0, 0.0, 2.0),
            Vector3(12.0, 0.0, 2.0),
            Vector3(11.0, 1.0, 2.0),
        ),
        max_angle_rad=0.5,
    )
    assert not result.intersects
    assert result.margin_squared > 0.0


def test_edge_crossing_axis_reports_edge_or_interior():
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(-4.0, 0.0, 2.0),
            Vector3(4.0, 0.0, 2.0),
            Vector3(4.0, 2.0, 2.0),
        ),
        max_angle_rad=0.2,
    )
    assert result.intersects
    assert result.minimum_normalized_radius_squared == pytest.approx(0.0)
    assert result.minimizing_location_type == "edge"
    assert result.minimizing_feature_indices == (0, 1)


def test_origin_inside_normalized_triangle_reports_interior():
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(-3.0, -2.0, 2.0),
            Vector3(3.0, -2.0, 2.0),
            Vector3(0.0, 4.0, 2.0),
        ),
        max_angle_rad=0.1,
    )
    assert result.intersects
    assert result.minimizing_location_type == "interior"
    assert result.closest_normalized_point == (0.0, 0.0)


def test_cone_boundary_is_included():
    maximum = 0.5
    tangent = math.tan(maximum)
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(tangent * 2.0, 0.0, 2.0),
            Vector3(4.0, 0.0, 2.0),
            Vector3(4.0, 1.0, 2.0),
        ),
        max_angle_rad=maximum,
    )
    assert result.intersects
    assert result.margin_squared == pytest.approx(0.0, abs=1e-12)


def test_varying_depth_uses_perspective_normalization():
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(1.0, 0.0, 10.0),
            Vector3(3.0, 1.0, 1.0),
            Vector3(3.0, -1.0, 1.0),
        ),
        max_angle_rad=0.2,
    )
    assert result.intersects
    assert result.minimizing_location_type == "vertex"
    assert result.minimizing_feature_indices == (0,)


def test_degenerate_collinear_triangle_uses_edge_minimum():
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(-2.0, 1.0, 2.0),
            Vector3(0.0, 1.0, 2.0),
            Vector3(2.0, 1.0, 2.0),
        ),
        max_angle_rad=0.6,
    )
    assert result.intersects
    assert result.minimum_normalized_radius_squared == pytest.approx(0.25)


def test_result_fields_are_consistent():
    maximum = 0.3
    result = triangle_intersects_angular_fov_cone(
        (
            Vector3(2.0, 0.0, 2.0),
            Vector3(3.0, 0.0, 2.0),
            Vector3(2.0, 1.0, 2.0),
        ),
        max_angle_rad=maximum,
    )
    assert result.cone_radius_squared == pytest.approx(math.tan(maximum) ** 2)
    assert result.margin_squared == pytest.approx(
        result.minimum_normalized_radius_squared - result.cone_radius_squared
    )


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    triangle = tuple(Vector3(float(index), 0.0, 1.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        triangle_intersects_angular_fov_cone(
            triangle, max_angle_rad=0.5
        )


@pytest.mark.parametrize(
    "angle", (0.0, -1.0, math.pi / 2.0, math.nan, math.inf)
)
def test_invalid_angle_is_rejected(angle):
    with pytest.raises(ValueError, match="max_angle_rad"):
        triangle_intersects_angular_fov_cone(
            (
                Vector3(0.0, 0.0, 1.0),
                Vector3(0.1, 0.0, 1.0),
                Vector3(0.0, 0.1, 1.0),
            ),
            max_angle_rad=angle,
        )


@pytest.mark.parametrize("z", (0.0, -1.0))
def test_nonpositive_depth_is_rejected(z):
    with pytest.raises(ValueError, match="positive depth"):
        triangle_intersects_angular_fov_cone(
            (
                Vector3(0.0, 0.0, z),
                Vector3(0.1, 0.0, 1.0),
                Vector3(0.0, 0.1, 1.0),
            ),
            max_angle_rad=0.5,
        )


def test_nonfinite_coordinate_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        triangle_intersects_angular_fov_cone(
            (
                Vector3(math.nan, 0.0, 1.0),
                Vector3(0.1, 0.0, 1.0),
                Vector3(0.0, 0.1, 1.0),
            ),
            max_angle_rad=0.5,
        )
