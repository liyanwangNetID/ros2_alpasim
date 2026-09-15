from __future__ import annotations

import math

import pytest

from step7.actor_box_image_projection_v01 import clip_segment_to_positive_z
from step7.camera_projection_v01 import Vector3
from step7.triangle_near_plane_clipping_v01 import (
    clip_polygon_to_positive_z,
    clip_triangle_to_positive_z,
)

NEAR = 0.1


def area_xy(triangle) -> float:
    first, second, third = triangle
    return 0.5 * abs(
        (second.x - first.x) * (third.y - first.y)
        - (second.y - first.y) * (third.x - first.x)
    )


def signed_area_xy(triangle) -> float:
    first, second, third = triangle
    return 0.5 * (
        (second.x - first.x) * (third.y - first.y)
        - (second.y - first.y) * (third.x - first.x)
    )


def test_fully_inside_triangle_is_preserved_exactly():
    triangle = (
        Vector3(0.0, 0.0, 1.0),
        Vector3(2.0, 0.0, 1.0),
        Vector3(0.0, 2.0, 1.0),
    )
    assert clip_triangle_to_positive_z(triangle, near_plane_m=NEAR) == (triangle,)


def test_fully_outside_triangle_is_removed():
    triangle = (
        Vector3(0.0, 0.0, 0.0),
        Vector3(2.0, 0.0, 0.05),
        Vector3(0.0, 2.0, -1.0),
    )
    assert clip_triangle_to_positive_z(triangle, near_plane_m=NEAR) == ()


def test_one_inside_vertex_produces_one_triangle():
    triangle = (
        Vector3(0.0, 0.0, 1.0),
        Vector3(2.0, 0.0, 0.0),
        Vector3(0.0, 2.0, 0.0),
    )
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert len(clipped) == 1
    assert all(vertex.z >= NEAR for vertex in clipped[0])
    assert sum(vertex.z == NEAR for vertex in clipped[0]) == 2


def test_two_inside_vertices_produce_two_triangles():
    triangle = (
        Vector3(0.0, 0.0, 1.0),
        Vector3(2.0, 0.0, 1.0),
        Vector3(0.0, 2.0, 0.0),
    )
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert len(clipped) == 2
    assert all(vertex.z >= NEAR for item in clipped for vertex in item)


def test_vertices_exactly_on_near_plane_are_inside():
    triangle = (
        Vector3(0.0, 0.0, NEAR),
        Vector3(2.0, 0.0, NEAR),
        Vector3(0.0, 2.0, 1.0),
    )
    assert clip_triangle_to_positive_z(triangle, near_plane_m=NEAR) == (triangle,)


def test_intersection_matches_existing_segment_clipper():
    inside = Vector3(1.0, 2.0, 1.0)
    outside = Vector3(5.0, 6.0, 0.0)
    segment = clip_segment_to_positive_z(inside, outside, near_plane_m=NEAR)
    assert segment is not None
    _, expected_intersection = segment

    polygon = clip_polygon_to_positive_z(
        (inside, outside, Vector3(-1.0, 3.0, 0.0)),
        near_plane_m=NEAR,
    )
    assert expected_intersection in polygon


def test_clipped_triangles_preserve_positive_winding():
    triangle = (
        Vector3(0.0, 0.0, 1.0),
        Vector3(2.0, 0.0, 1.0),
        Vector3(0.0, 2.0, 0.0),
    )
    assert signed_area_xy(triangle) > 0.0
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert all(signed_area_xy(item) > 0.0 for item in clipped)


def test_clipped_area_matches_expected_polygon_area():
    triangle = (
        Vector3(0.0, 0.0, 1.0),
        Vector3(2.0, 0.0, 1.0),
        Vector3(0.0, 2.0, 0.0),
    )
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert sum(area_xy(item) for item in clipped) == pytest.approx(1.98)


@pytest.mark.parametrize("near", (0.0, -1.0))
def test_non_positive_near_plane_is_rejected(near):
    triangle = (Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1))
    with pytest.raises(ValueError, match="must be positive"):
        clip_triangle_to_positive_z(triangle, near_plane_m=near)


@pytest.mark.parametrize("near", (math.nan, math.inf, -math.inf))
def test_non_finite_near_plane_is_rejected(near):
    triangle = (Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1))
    with pytest.raises(ValueError, match="must be finite"):
        clip_triangle_to_positive_z(triangle, near_plane_m=near)


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_triangle_vertex_count_is_rejected(count):
    vertices = tuple(Vector3(float(i), 0.0, 1.0) for i in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        clip_triangle_to_positive_z(vertices, near_plane_m=NEAR)
