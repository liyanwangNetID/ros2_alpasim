#!/usr/bin/env python3
"""Angular-FOV diagnostics for camera-frame triangles.

This module deliberately does not claim to exactly clip a triangle against the
F-theta cone. A plane triangle and a circular cone can intersect along a curved
conic section, including cases not decidable from vertex states alone. The
module therefore exposes conservative point and edge diagnostics to support a
later adaptive surface-clipping algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from camera_projection_v01 import Vector3
from ftheta_fov_clipping_v01 import clip_segment_to_angular_fov

FOV_EDGE_INDEX_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0))
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TriangleEdgeFovIntersection:
    first_vertex_index: int
    second_vertex_index: int
    clipped_segment: tuple[Vector3, Vector3] | None


@dataclass(frozen=True, slots=True)
class TriangleAngularFovDiagnostics:
    vertex_inside: tuple[bool, bool, bool]
    edge_intersections: tuple[TriangleEdgeFovIntersection, ...]
    all_vertices_inside: bool
    any_vertex_inside: bool
    every_edge_has_no_fov_segment: bool


def _validate_max_angle(max_angle_rad: float | None) -> float | None:
    if max_angle_rad is None:
        return None
    maximum = float(max_angle_rad)
    if not math.isfinite(maximum) or not 0.0 < maximum < math.pi / 2.0:
        raise ValueError("max_angle_rad must be in (0, pi/2)")
    return maximum


def point_is_within_angular_fov(
    point: Vector3,
    *,
    max_angle_rad: float | None,
) -> bool:
    """Return whether a positive-depth camera point lies in the angular cone.

    The cone boundary is included using the same numerical tolerance as the
    existing analytic segment clipper. When max_angle_rad is None, every
    positive-depth finite point is accepted.
    """
    values = (point.x, point.y, point.z)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("point coordinates must be finite")
    if point.z <= 0.0:
        return False
    maximum = _validate_max_angle(max_angle_rad)
    if maximum is None:
        return True
    tangent_squared = math.tan(maximum) ** 2
    cone_value = point.x * point.x + point.y * point.y - tangent_squared * point.z * point.z
    scale = max(
        1.0,
        point.x * point.x + point.y * point.y,
        tangent_squared * point.z * point.z,
    )
    return cone_value <= _EPSILON * scale


def classify_triangle_angular_fov(
    triangle_camera: Sequence[Vector3],
    *,
    max_angle_rad: float | None,
) -> TriangleAngularFovDiagnostics:
    """Return conservative vertex and edge diagnostics for one triangle.

    `every_edge_has_no_fov_segment` must not be interpreted as proof that the
    triangle interior misses the cone. Exact surface classification requires
    interior sampling or adaptive triangle subdivision.
    """
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    maximum = _validate_max_angle(max_angle_rad)
    vertices = tuple(triangle_camera)
    vertex_inside = tuple(
        point_is_within_angular_fov(point, max_angle_rad=maximum)
        for point in vertices
    )
    edges = tuple(
        TriangleEdgeFovIntersection(
            first_vertex_index=first_index,
            second_vertex_index=second_index,
            clipped_segment=clip_segment_to_angular_fov(
                vertices[first_index],
                vertices[second_index],
                max_angle_rad=maximum,
            ),
        )
        for first_index, second_index in FOV_EDGE_INDEX_PAIRS
    )
    return TriangleAngularFovDiagnostics(
        vertex_inside=vertex_inside,  # type: ignore[arg-type]
        edge_intersections=edges,
        all_vertices_inside=all(vertex_inside),
        any_vertex_inside=any(vertex_inside),
        every_edge_has_no_fov_segment=all(
            edge.clipped_segment is None for edge in edges
        ),
    )
