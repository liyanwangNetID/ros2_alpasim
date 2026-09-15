from __future__ import annotations

import math

import pytest

from step7.box_surface_geometry_v01 import triangulate_box_surfaces
from step7.camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from step7.camera_projection_v01 import Vector3
from step7.triangle_facing_v01 import is_triangle_front_facing


def translated_box(center: Vector3, half_extent: float = 1.0):
    return tuple(
        Vector3(
            center.x + x * half_extent,
            center.y + y * half_extent,
            center.z + z * half_extent,
        )
        for z in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for x in (-1.0, 1.0)
    )


def test_box_fully_in_front_returns_camera_facing_near_surface():
    corners = translated_box(Vector3(0.0, 0.0, 5.0))
    result = prepare_camera_facing_box_triangles(corners)
    assert len(result) == 2
    assert {item.face_name for item in result} == {"negative_z"}
    assert all(vertex.z == 4.0 for item in result for vertex in item.vertices_camera)


def test_output_matches_independent_facing_selection_when_no_clipping_occurs():
    corners = translated_box(Vector3(2.0, 1.0, 8.0))
    source = triangulate_box_surfaces(corners)
    expected = [item for item in source if is_triangle_front_facing(item.vertices)]
    actual = prepare_camera_facing_box_triangles(corners)
    assert len(actual) == len(expected)
    assert [item.face_name for item in actual] == [item.face_name for item in expected]
    assert [item.vertices_camera for item in actual] == [item.vertices for item in expected]


def test_composition_applies_near_clipping_after_facing_selection():
    # The box crosses the near plane, but the camera origin is outside the box
    # on the negative-x side. This gives camera-facing side triangles that cross
    # the near plane without placing the camera inside a closed convex box.
    corners = translated_box(Vector3(2.5, 0.0, 0.75))
    result = prepare_camera_facing_box_triangles(corners, near_plane_m=0.1)
    assert result
    assert all(vertex.z >= 0.1 for item in result for vertex in item.vertices_camera)
    assert any(vertex.z == 0.1 for item in result for vertex in item.vertices_camera)


def test_source_metadata_is_preserved_when_clipping_splits_triangle():
    corners = translated_box(Vector3(2.5, 0.0, 0.75))
    result = prepare_camera_facing_box_triangles(corners, near_plane_m=0.1)
    groups = {}
    for item in result:
        key = (item.face_name, item.source_corner_indices)
        groups.setdefault(key, 0)
        groups[key] += 1
    assert any(count == 2 for count in groups.values())


def test_camera_inside_box_has_no_outward_front_facing_surface():
    corners = translated_box(Vector3(0.0, 0.0, 0.75))
    assert prepare_camera_facing_box_triangles(corners, near_plane_m=0.1) == ()


def test_box_fully_behind_near_plane_produces_no_triangles():
    corners = translated_box(Vector3(0.0, 0.0, -5.0))
    assert prepare_camera_facing_box_triangles(corners, near_plane_m=0.1) == ()


def test_output_order_is_deterministic():
    corners = translated_box(Vector3(2.0, -1.0, 8.0))
    first = prepare_camera_facing_box_triangles(corners)
    second = prepare_camera_facing_box_triangles(corners)
    assert first == second


@pytest.mark.parametrize("count", (0, 7, 9))
def test_wrong_corner_count_is_rejected(count):
    corners = tuple(Vector3(float(index), 0.0, 1.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly eight corners"):
        prepare_camera_facing_box_triangles(corners)


@pytest.mark.parametrize("near", (0.0, -1.0, math.nan, math.inf))
def test_invalid_near_plane_is_rejected(near):
    corners = translated_box(Vector3(0.0, 0.0, 5.0))
    with pytest.raises(ValueError, match="near_plane_m"):
        prepare_camera_facing_box_triangles(corners, near_plane_m=near)


@pytest.mark.parametrize("tolerance", (-1.0, math.nan, math.inf))
def test_invalid_facing_tolerance_is_rejected(tolerance):
    corners = translated_box(Vector3(0.0, 0.0, 5.0))
    with pytest.raises(ValueError, match="facing_tolerance"):
        prepare_camera_facing_box_triangles(corners, facing_tolerance=tolerance)
