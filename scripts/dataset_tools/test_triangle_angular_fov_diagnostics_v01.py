from __future__ import annotations

import math

import pytest

from camera_projection_v01 import Vector3
from triangle_angular_fov_diagnostics_v01 import (
    FOV_EDGE_INDEX_PAIRS,
    classify_triangle_angular_fov,
    point_is_within_angular_fov,
)


def theta(point: Vector3) -> float:
    return math.atan2(math.hypot(point.x, point.y), point.z)


def test_point_inside_cone():
    assert point_is_within_angular_fov(
        Vector3(0.2, 0.1, 2.0), max_angle_rad=0.5
    )


def test_point_on_cone_boundary_is_inside():
    maximum = 0.5
    point = Vector3(math.tan(maximum) * 2.0, 0.0, 2.0)
    assert point_is_within_angular_fov(point, max_angle_rad=maximum)


@pytest.mark.parametrize("z", (0.0, -1.0))
def test_nonpositive_depth_is_outside(z):
    assert not point_is_within_angular_fov(
        Vector3(0.0, 0.0, z), max_angle_rad=0.5
    )


def test_none_angle_accepts_finite_positive_depth_point():
    assert point_is_within_angular_fov(
        Vector3(100.0, 0.0, 1.0), max_angle_rad=None
    )


def test_fully_inside_triangle_has_original_edges():
    triangle = (
        Vector3(-0.2, -0.1, 2.0),
        Vector3(0.2, -0.1, 2.0),
        Vector3(0.0, 0.2, 2.0),
    )
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.5)
    assert result.vertex_inside == (True, True, True)
    assert result.all_vertices_inside
    assert result.any_vertex_inside
    assert not result.every_edge_has_no_fov_segment
    assert tuple(
        (item.first_vertex_index, item.second_vertex_index)
        for item in result.edge_intersections
    ) == FOV_EDGE_INDEX_PAIRS
    for edge in result.edge_intersections:
        assert edge.clipped_segment is not None

        actual_first, actual_second = (
            edge.clipped_segment
        )
        expected_first = triangle[
            edge.first_vertex_index
        ]
        expected_second = triangle[
            edge.second_vertex_index
        ]

        for actual, expected in (
            (actual_first, expected_first),
            (actual_second, expected_second),
        ):
            assert actual.x == pytest.approx(
                expected.x,
                abs=1e-12,
            )
            assert actual.y == pytest.approx(
                expected.y,
                abs=1e-12,
            )
            assert actual.z == pytest.approx(
                expected.z,
                abs=1e-12,
            )


def test_one_inside_vertex_clips_two_incident_edges():
    maximum = 0.5
    triangle = (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )
    result = classify_triangle_angular_fov(triangle, max_angle_rad=maximum)
    assert result.vertex_inside == (True, False, False)
    assert not result.all_vertices_inside
    assert result.any_vertex_inside
    assert result.edge_intersections[0].clipped_segment is not None
    assert result.edge_intersections[1].clipped_segment is None
    assert result.edge_intersections[2].clipped_segment is not None
    for edge in (result.edge_intersections[0], result.edge_intersections[2]):
        clipped = edge.clipped_segment
        assert clipped is not None
        assert any(abs(theta(point) - maximum) < 1e-10 for point in clipped)


def test_two_outside_vertices_edge_can_cross_cone():
    triangle = (
        Vector3(-4.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(4.0, 1.0, 2.0),
    )
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.5)
    assert result.vertex_inside == (False, False, False)
    assert not result.any_vertex_inside
    assert result.edge_intersections[0].clipped_segment is not None
    assert not result.every_edge_has_no_fov_segment


def test_fully_outside_separated_triangle_has_no_edge_segments():
    triangle = (
        Vector3(10.0, 0.0, 2.0),
        Vector3(12.0, 0.0, 2.0),
        Vector3(11.0, 1.0, 2.0),
    )
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.5)
    assert result.vertex_inside == (False, False, False)
    assert not result.any_vertex_inside
    assert result.every_edge_has_no_fov_segment


def test_no_edge_segment_flag_is_documented_as_non_proof_of_empty_interior():
    # This triangle surrounds the optical axis at z=2. Its three vertices and
    # edges lie outside a sufficiently small circular cone, while its interior
    # contains the cone axis. The diagnostic must expose only conservative
    # facts, not label the whole triangle as outside.
    triangle = (
        Vector3(-4.0, -3.0, 2.0),
        Vector3(4.0, -3.0, 2.0),
        Vector3(0.0, 5.0, 2.0),
    )
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.2)
    assert not result.any_vertex_inside
    # At least one long edge may cross the cone depending on geometry. The key
    # contract is that no `triangle_outside` conclusion exists in the result.
    assert not hasattr(result, "triangle_outside")


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    triangle = tuple(Vector3(float(i), 0.0, 1.0) for i in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        classify_triangle_angular_fov(triangle, max_angle_rad=0.5)


@pytest.mark.parametrize("angle", (0.0, -1.0, math.pi / 2.0, math.nan, math.inf))
def test_invalid_angle_is_rejected(angle):
    with pytest.raises(ValueError, match="max_angle_rad"):
        point_is_within_angular_fov(
            Vector3(0.0, 0.0, 1.0), max_angle_rad=angle
        )


def test_nonfinite_point_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        point_is_within_angular_fov(
            Vector3(math.nan, 0.0, 1.0), max_angle_rad=0.5
        )
