from __future__ import annotations

import math

import pytest

from step7.camera_projection_v01 import Vector3
from step7.safe_angular_fov_polygon_v01 import build_safe_angular_fov_polygon
from step7.triangle_angular_fov_diagnostics_v01 import point_is_within_angular_fov


def test_fully_inside_triangle_is_preserved_as_one_triangle():
    source = (
        Vector3(-0.2, -0.1, 2.0),
        Vector3(0.2, -0.1, 2.0),
        Vector3(0.0, 0.2, 2.0),
    )
    result = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    assert result.has_measurable_polygon
    assert not result.is_degenerate
    assert len(result.ordered_vertices_camera) == 3
    assert len(result.triangles_camera) == 1
    assert set(result.ordered_vertices_camera) == set(source)


def test_one_inside_vertex_produces_safe_clipped_triangle():
    source = (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )
    result = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    assert len(result.ordered_vertices_camera) == 3
    assert len(result.triangles_camera) == 1
    assert all(
        point_is_within_angular_fov(point, max_angle_rad=0.5)
        for point in result.ordered_vertices_camera
    )


def test_two_inside_vertices_produce_quadrilateral_and_two_triangles():
    source = (
        Vector3(-0.2, 0.0, 2.0),
        Vector3(0.2, 0.0, 2.0),
        Vector3(4.0, 0.5, 2.0),
    )
    result = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    assert len(result.ordered_vertices_camera) == 4
    assert len(result.triangles_camera) == 2
    assert all(
        point_is_within_angular_fov(point, max_angle_rad=0.5)
        for point in result.ordered_vertices_camera
    )


def test_outside_separated_triangle_returns_empty_polygon():
    result = build_safe_angular_fov_polygon((
        Vector3(10.0, 0.0, 2.0),
        Vector3(12.0, 0.0, 2.0),
        Vector3(11.0, 1.0, 2.0),
    ), max_angle_rad=0.5)
    assert not result.has_measurable_polygon
    assert result.ordered_vertices_camera == ()
    assert result.triangles_camera == ()


def test_interior_only_cone_intersection_is_explicitly_empty():
    result = build_safe_angular_fov_polygon((
        Vector3(-3.0, -2.0, 2.0),
        Vector3(3.0, -2.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    ), max_angle_rad=0.1)
    assert not result.has_measurable_polygon
    assert result.triangles_camera == ()


def test_duplicate_clipped_endpoints_are_removed_with_tolerance():
    source = (
        Vector3(-0.2, -0.1, 2.0),
        Vector3(0.2, -0.1, 2.0),
        Vector3(0.0, 0.2, 2.0),
    )
    result = build_safe_angular_fov_polygon(source, max_angle_rad=None)
    assert len(result.ordered_vertices_camera) == 3


def test_reversed_winding_has_same_polygon_vertex_set():
    source = (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )
    forward = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    reverse = build_safe_angular_fov_polygon(
        (source[0], source[2], source[1]), max_angle_rad=0.5
    )
    assert set(reverse.ordered_vertices_camera) == set(forward.ordered_vertices_camera)
    assert len(reverse.triangles_camera) == len(forward.triangles_camera)


def test_collinear_source_triangle_is_rejected():
    with pytest.raises(ValueError, match="non-degenerate"):
        build_safe_angular_fov_polygon((
            Vector3(0.0, 0.0, 2.0),
            Vector3(1.0, 0.0, 2.0),
            Vector3(2.0, 0.0, 2.0),
        ), max_angle_rad=0.5)


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    source = tuple(Vector3(float(index), 0.0, 2.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly three"):
        build_safe_angular_fov_polygon(source, max_angle_rad=0.5)


@pytest.mark.parametrize("tolerance", (0.0, -1.0, math.nan, math.inf))
def test_invalid_point_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="point_tolerance"):
        build_safe_angular_fov_polygon((
            Vector3(0.0, 0.0, 2.0),
            Vector3(1.0, 0.0, 2.0),
            Vector3(0.0, 1.0, 2.0),
        ), max_angle_rad=0.5, point_tolerance=tolerance)


def test_nonpositive_depth_is_rejected():
    with pytest.raises(ValueError, match="positive depth"):
        build_safe_angular_fov_polygon((
            Vector3(0.0, 0.0, 0.0),
            Vector3(1.0, 0.0, 2.0),
            Vector3(0.0, 1.0, 2.0),
        ), max_angle_rad=0.5)
