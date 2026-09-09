#!/usr/bin/env python3
"""Build safe in-FOV polygon geometry from one boundary triangle.

The intersection of a triangle and a convex angular-FOV cone is convex. This
module collects original vertices inside the cone plus endpoints of clipped
triangle-edge segments, removes tolerance-equivalent duplicates, orders the
points in the source-triangle plane, and triangulates the polygon as a fan.

If the cone intersects only the triangle interior, with no inside source vertex
or clipped edge segment, the result is explicitly unmeasurable and empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from camera_projection_v01 import Vector3
from triangle_angular_fov_diagnostics_v01 import classify_triangle_angular_fov

Triangle3D = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True, slots=True)
class SafeAngularFovPolygon:
    ordered_vertices_camera: tuple[Vector3, ...]
    triangles_camera: tuple[Triangle3D, ...]
    has_measurable_polygon: bool
    is_degenerate: bool


def _subtract(first: Vector3, second: Vector3) -> tuple[float, float, float]:
    return first.x - second.x, first.y - second.y, first.z - second.z


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1e-15:
        raise ValueError("source triangle must be non-degenerate")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _points_close(first: Vector3, second: Vector3, tolerance: float) -> bool:
    return all(
        math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
        for a, b in ((first.x, second.x), (first.y, second.y), (first.z, second.z))
    )


def build_safe_angular_fov_polygon(
    triangle_camera: Sequence[Vector3],
    *,
    max_angle_rad: float | None,
    point_tolerance: float = 1e-12,
) -> SafeAngularFovPolygon:
    """Return ordered in-FOV evidence polygon and fan triangulation."""
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    tolerance = float(point_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("point_tolerance must be positive and finite")

    vertices: Triangle3D = tuple(triangle_camera)  # type: ignore[assignment]
    for point in vertices:
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            raise ValueError("triangle coordinates must be finite")
        if point.z <= 0.0:
            raise ValueError("triangle vertices must have positive depth")

    first, second, third = vertices
    axis_u = _normalize(_subtract(second, first))
    third_offset = _subtract(third, first)
    orthogonal = tuple(
        value - _dot(third_offset, axis_u) * axis
        for value, axis in zip(third_offset, axis_u)
    )
    axis_v = _normalize(orthogonal)  # validates source non-degeneracy

    diagnostics = classify_triangle_angular_fov(
        vertices,
        max_angle_rad=max_angle_rad,
    )
    unique: list[Vector3] = []

    def append_unique(candidate: Vector3) -> None:
        if not any(_points_close(existing, candidate, tolerance) for existing in unique):
            unique.append(candidate)

    for point, inside in zip(vertices, diagnostics.vertex_inside):
        if inside:
            append_unique(point)
    for edge in diagnostics.edge_intersections:
        if edge.clipped_segment is not None:
            for point in edge.clipped_segment:
                append_unique(point)

    if not unique:
        return SafeAngularFovPolygon((), (), False, False)
    if len(unique) < 3:
        return SafeAngularFovPolygon(tuple(unique), (), True, True)

    centroid = Vector3(
        sum(point.x for point in unique) / len(unique),
        sum(point.y for point in unique) / len(unique),
        sum(point.z for point in unique) / len(unique),
    )

    def angle(point: Vector3) -> float:
        offset = _subtract(point, centroid)
        return math.atan2(_dot(offset, axis_v), _dot(offset, axis_u))

    ordered = tuple(sorted(unique, key=angle))
    anchor = ordered[0]
    triangles = tuple(
        (anchor, ordered[index], ordered[index + 1])
        for index in range(1, len(ordered) - 1)
    )
    return SafeAngularFovPolygon(ordered, triangles, True, False)
