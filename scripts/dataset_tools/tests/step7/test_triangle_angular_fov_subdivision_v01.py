from __future__ import annotations

import math

import pytest

from camera_projection_v01 import Vector3
from triangle_angular_fov_subdivision_v01 import (
    subdivide_triangle_to_angular_fov,
)


class BoundaryProjection:
    def __init__(self, u, v):
        self.u = u
        self.v = v
        self.positive_z = True
        self.within_fov = True
        self.valid = True


class NormalizedCalibration:
    def __init__(self, scale=100.0):
        self.scale = scale

    def project_camera_point(self, point):
        return BoundaryProjection(
            self.scale * point.x / point.z,
            self.scale * point.y / point.z,
        )


def inside_triangle():
    return (
        Vector3(-0.1, -0.1, 2.0),
        Vector3(0.1, -0.1, 2.0),
        Vector3(0.0, 0.1, 2.0),
    )


def mixed_triangle():
    return (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )


def test_fully_inside_is_accepted_without_subdivision():
    result = subdivide_triangle_to_angular_fov(
        inside_triangle(), max_angle_rad=0.5, maximum_depth=4
    )
    assert len(result.accepted_inside_triangles) == 1
    assert result.accepted_inside_triangles[0].vertices_camera == inside_triangle()
    assert result.accepted_inside_triangles[0].subdivision_depth == 0
    assert result.rejected_outside_triangle_count == 0
    assert result.boundary_unresolved_triangles == ()
    assert result.maximum_depth_reached == 0
    assert not result.stopped_by_depth_limit


def test_none_fov_accepts_positive_depth_triangle():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(), max_angle_rad=None, maximum_depth=0
    )
    assert len(result.accepted_inside_triangles) == 1
    assert not result.stopped_by_depth_limit


def test_mixed_triangle_subdivides():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(), max_angle_rad=0.5, maximum_depth=1
    )
    total = (
        len(result.accepted_inside_triangles)
        + len(result.boundary_unresolved_triangles)
        + result.rejected_outside_triangle_count
    )
    assert total == 4
    assert result.maximum_depth_reached == 1
    assert result.boundary_unresolved_triangles
    assert result.stopped_by_depth_limit


def test_depth_zero_mixed_triangle_is_unresolved():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(), max_angle_rad=0.5, maximum_depth=0
    )
    assert len(result.boundary_unresolved_triangles) == 1
    unresolved = result.boundary_unresolved_triangles[0]
    assert unresolved.subdivision_depth == 0
    assert unresolved.sample_inside_count > 0
    assert unresolved.edge_intersection_count > 0
    assert result.stopped_by_depth_limit


def test_separated_outside_triangle_is_rejected_at_limit():
    triangle = (
        Vector3(10.0, 0.0, 2.0),
        Vector3(12.0, 0.0, 2.0),
        Vector3(11.0, 1.0, 2.0),
    )
    result = subdivide_triangle_to_angular_fov(
        triangle, max_angle_rad=0.5, maximum_depth=0
    )
    assert result.accepted_inside_triangles == ()
    assert result.boundary_unresolved_triangles == ()
    assert result.rejected_outside_triangle_count == 1
    assert not result.stopped_by_depth_limit


def test_outside_triangle_is_rejected_before_subdivision():
    triangle = (
        Vector3(10.0, 0.0, 2.0),
        Vector3(12.0, 0.0, 2.0),
        Vector3(11.0, 1.0, 2.0),
    )
    result = subdivide_triangle_to_angular_fov(
        triangle, max_angle_rad=0.5, maximum_depth=8
    )
    assert result.rejected_outside_triangle_count == 1
    assert result.maximum_depth_reached == 0
    assert result.accepted_inside_triangles == ()
    assert result.boundary_unresolved_triangles == ()
    assert not result.stopped_by_depth_limit


def test_all_vertices_outside_but_interior_intersection_is_not_early_rejected():
    triangle = (
        Vector3(-3.0, -2.0, 2.0),
        Vector3(3.0, -2.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )
    result = subdivide_triangle_to_angular_fov(
        triangle, max_angle_rad=0.1, maximum_depth=1
    )
    assert result.accepted_inside_triangles or result.boundary_unresolved_triangles


def test_boundary_triangle_never_silently_becomes_inside():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(), max_angle_rad=0.5, maximum_depth=2
    )
    assert result.boundary_unresolved_triangles
    assert result.stopped_by_depth_limit


def test_child_winding_is_preserved_for_accepted_children():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(), max_angle_rad=0.5, maximum_depth=2
    )
    for item in result.accepted_inside_triangles:
        first, second, third = item.vertices_camera
        signed = (
            (second.x - first.x) * (third.y - first.y)
            - (second.y - first.y) * (third.x - first.x)
        )
        assert signed > 0.0


def test_small_boundary_is_reported_as_approximated():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(),
        max_angle_rad=0.5,
        maximum_depth=8,
        calibration=NormalizedCalibration(scale=1.0),
        maximum_boundary_extent_px=10.0,
    )
    assert result.boundary_approximated_triangles
    assert result.boundary_depth_limited_triangles == ()
    assert result.boundary_unmeasurable_triangles == ()
    assert not result.stopped_by_depth_limit


def test_large_boundary_at_depth_limit_is_reported_separately():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(),
        max_angle_rad=0.5,
        maximum_depth=0,
        calibration=NormalizedCalibration(scale=1000.0),
        maximum_boundary_extent_px=0.01,
    )
    assert result.boundary_approximated_triangles == ()
    assert len(result.boundary_depth_limited_triangles) == 1
    assert result.boundary_unmeasurable_triangles == ()
    assert result.stopped_by_depth_limit


def test_unmeasurable_boundary_is_not_treated_as_zero_extent():
    triangle = (
        Vector3(-3.0, -2.0, 2.0),
        Vector3(3.0, -2.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )
    result = subdivide_triangle_to_angular_fov(
        triangle,
        max_angle_rad=0.1,
        maximum_depth=0,
        calibration=NormalizedCalibration(),
        maximum_boundary_extent_px=1.0,
    )
    assert result.boundary_approximated_triangles == ()
    assert result.boundary_depth_limited_triangles == ()
    assert len(result.boundary_unmeasurable_triangles) == 1
    assert result.stopped_by_depth_limit


def test_legacy_unresolved_property_contains_two_unresolved_categories():
    result = subdivide_triangle_to_angular_fov(
        mixed_triangle(),
        max_angle_rad=0.5,
        maximum_depth=0,
    )
    assert result.boundary_unresolved_triangles == (
        result.boundary_depth_limited_triangles
        + result.boundary_unmeasurable_triangles
    )


def test_boundary_limit_requires_calibration():
    with pytest.raises(ValueError, match="calibration is required"):
        subdivide_triangle_to_angular_fov(
            mixed_triangle(),
            max_angle_rad=0.5,
            maximum_depth=2,
            maximum_boundary_extent_px=1.0,
        )


@pytest.mark.parametrize("limit", (0.0, -1.0, math.nan, math.inf))
def test_invalid_boundary_extent_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="maximum_boundary_extent_px"):
        subdivide_triangle_to_angular_fov(
            mixed_triangle(),
            max_angle_rad=0.5,
            maximum_depth=2,
            calibration=NormalizedCalibration(),
            maximum_boundary_extent_px=limit,
        )


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    triangle = tuple(Vector3(float(index), 0.0, 1.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        subdivide_triangle_to_angular_fov(
            triangle, max_angle_rad=0.5, maximum_depth=1
        )


@pytest.mark.parametrize("depth", (-1,))
def test_negative_depth_is_rejected(depth):
    with pytest.raises(ValueError, match="non-negative"):
        subdivide_triangle_to_angular_fov(
            inside_triangle(), max_angle_rad=0.5, maximum_depth=depth
        )


@pytest.mark.parametrize("depth", (True, 1.5, "2"))
def test_noninteger_depth_is_rejected(depth):
    with pytest.raises(TypeError, match="integer"):
        subdivide_triangle_to_angular_fov(
            inside_triangle(), max_angle_rad=0.5, maximum_depth=depth
        )


@pytest.mark.parametrize("z", (0.0, -1.0))
def test_nonpositive_vertex_depth_is_rejected(z):
    triangle = (
        Vector3(0.0, 0.0, z),
        Vector3(0.1, 0.0, 1.0),
        Vector3(0.0, 0.1, 1.0),
    )
    with pytest.raises(ValueError, match="positive depth"):
        subdivide_triangle_to_angular_fov(
            triangle, max_angle_rad=0.5, maximum_depth=1
        )


def test_nonfinite_vertex_is_rejected():
    triangle = (
        Vector3(math.nan, 0.0, 1.0),
        Vector3(0.1, 0.0, 1.0),
        Vector3(0.0, 0.1, 1.0),
    )
    with pytest.raises(ValueError, match="finite"):
        subdivide_triangle_to_angular_fov(
            triangle, max_angle_rad=0.5, maximum_depth=1
        )
