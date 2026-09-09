from __future__ import annotations

import math

import pytest

from box_surface_geometry_v01 import triangulate_box_surfaces
from camera_projection_v01 import Vector3
from triangle_facing_v01 import (
    is_triangle_front_facing,
    select_front_facing_triangles,
    triangle_normal,
)


def test_front_facing_triangle_is_selected():
    triangle = (
        Vector3(-1.0, -1.0, 5.0),
        Vector3(0.0, 1.0, 5.0),
        Vector3(1.0, -1.0, 5.0),
    )
    assert is_triangle_front_facing(triangle) is True
    assert select_front_facing_triangles((triangle,)) == (triangle,)


def test_reversed_triangle_is_back_facing():
    triangle = (
        Vector3(-1.0, -1.0, 5.0),
        Vector3(1.0, -1.0, 5.0),
        Vector3(0.0, 1.0, 5.0),
    )
    assert is_triangle_front_facing(triangle) is False
    assert select_front_facing_triangles((triangle,)) == ()


def test_edge_on_triangle_is_not_front_facing():
    triangle = (
        Vector3(1.0, -1.0, 4.0),
        Vector3(1.0, 1.0, 4.0),
        Vector3(1.0, 0.0, 6.0),
    )
    assert is_triangle_front_facing(triangle) is False


def test_axis_aligned_box_in_front_exposes_camera_facing_surface():
    center = Vector3(0.0, 0.0, 5.0)
    local = (
        Vector3(-1, -1, -1), Vector3(1, -1, -1),
        Vector3(-1, 1, -1), Vector3(1, 1, -1),
        Vector3(-1, -1, 1), Vector3(1, -1, 1),
        Vector3(-1, 1, 1), Vector3(1, 1, 1),
    )
    corners = tuple(Vector3(p.x + center.x, p.y + center.y, p.z + center.z) for p in local)
    surfaces = triangulate_box_surfaces(corners)
    selected = select_front_facing_triangles(item.vertices for item in surfaces)
    assert len(selected) == 2
    negative_z = {item.vertices for item in surfaces if item.face_name == "negative_z"}
    assert set(selected) == negative_z


def test_selection_preserves_input_order():
    front_first = (
        Vector3(-1, -1, 5), Vector3(0, 1, 5), Vector3(1, -1, 5)
    )
    back = tuple(reversed(front_first))
    front_second = (
        Vector3(-2, -1, 8), Vector3(0, 2, 8), Vector3(2, -1, 8)
    )
    assert select_front_facing_triangles((front_first, back, front_second)) == (
        front_first,
        front_second,
    )


def test_triangle_normal_follows_winding():
    triangle = (
        Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1)
    )
    normal = triangle_normal(triangle)
    reversed_normal = triangle_normal(tuple(reversed(triangle)))
    assert normal == Vector3(0, 0, 1)
    assert reversed_normal == Vector3(0, 0, -1)


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    triangle = tuple(Vector3(float(i), 0.0, 1.0) for i in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        is_triangle_front_facing(triangle)


def test_degenerate_triangle_is_rejected():
    triangle = (
        Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(2, 0, 1)
    )
    with pytest.raises(ValueError, match="non-zero area"):
        is_triangle_front_facing(triangle)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_non_finite_vertex_is_rejected(value):
    triangle = (
        Vector3(0, 0, 1), Vector3(1, value, 1), Vector3(0, 1, 1)
    )
    with pytest.raises(ValueError, match="must be finite"):
        is_triangle_front_facing(triangle)


@pytest.mark.parametrize("tolerance", (-1.0, math.nan, math.inf))
def test_invalid_tolerance_is_rejected(tolerance):
    triangle = (
        Vector3(-1, -1, 5), Vector3(0, 1, 5), Vector3(1, -1, 5)
    )
    with pytest.raises(ValueError, match="tolerance"):
        is_triangle_front_facing(triangle, tolerance=tolerance)
