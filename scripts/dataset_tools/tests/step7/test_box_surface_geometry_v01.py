from __future__ import annotations

import pytest

from step7.actor_box_image_projection_v01 import BOX_EDGE_INDEX_PAIRS
from step7.actor_box_projection_v01 import local_box_corners
from step7.box_surface_geometry_v01 import (
    BOX_FACE_INDEX_QUADS,
    BOX_TRIANGLE_INDEX_TRIPLES,
    triangulate_box_surfaces,
)
from step7.camera_projection_v01 import Vector3


def subtract(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        first.x - second.x,
        first.y - second.y,
        first.z - second.z,
    )


def cross(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    )


def dot(first: Vector3, second: Vector3) -> float:
    return first.x * second.x + first.y * second.y + first.z * second.z


def test_face_and_triangle_counts_are_fixed():
    assert len(BOX_FACE_INDEX_QUADS) == 6
    assert len(BOX_TRIANGLE_INDEX_TRIPLES) == 12
    assert len(triangulate_box_surfaces(local_box_corners(Vector3(2, 4, 6)))) == 12


def test_each_face_produces_exactly_two_triangles():
    face_names = [name for name, _ in BOX_FACE_INDEX_QUADS]
    triangle_names = [name for name, _ in BOX_TRIANGLE_INDEX_TRIPLES]
    assert len(set(face_names)) == 6
    for face_name in face_names:
        assert triangle_names.count(face_name) == 2


def test_all_indices_are_valid_and_each_corner_is_used():
    indices = [index for _, triangle in BOX_TRIANGLE_INDEX_TRIPLES for index in triangle]
    assert set(indices) == set(range(8))
    assert all(0 <= index < 8 for index in indices)


def test_face_boundaries_match_existing_box_edges():
    undirected_edges = {frozenset(edge) for edge in BOX_EDGE_INDEX_PAIRS}
    for _, quad in BOX_FACE_INDEX_QUADS:
        boundary = zip(quad, quad[1:] + quad[:1])
        assert all(frozenset(edge) in undirected_edges for edge in boundary)


def test_triangle_winding_points_away_from_box_center():
    corners = local_box_corners(Vector3(2.0, 4.0, 6.0))
    for triangle in triangulate_box_surfaces(corners):
        first, second, third = triangle.vertices
        normal = cross(subtract(second, first), subtract(third, first))
        centroid = Vector3(
            (first.x + second.x + third.x) / 3.0,
            (first.y + second.y + third.y) / 3.0,
            (first.z + second.z + third.z) / 3.0,
        )
        assert dot(normal, centroid) > 0.0, triangle


def test_triangles_preserve_supplied_transformed_vertices():
    corners = tuple(Vector3(float(i), float(i + 10), float(i + 20)) for i in range(8))
    triangles = triangulate_box_surfaces(corners)
    for triangle in triangles:
        assert triangle.vertices == tuple(corners[i] for i in triangle.corner_indices)


@pytest.mark.parametrize("corner_count", (0, 7, 9))
def test_wrong_corner_count_is_rejected(corner_count):
    corners = tuple(Vector3(float(i), 0.0, 0.0) for i in range(corner_count))
    with pytest.raises(ValueError, match="exactly eight corners"):
        triangulate_box_surfaces(corners)
