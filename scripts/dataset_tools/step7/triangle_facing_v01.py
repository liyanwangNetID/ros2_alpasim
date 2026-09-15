"""Camera-facing triangle selection for Step 7E Actor occlusion.

Input triangles are expected to use outward-facing winding. The camera optical
center is the origin of the camera frame. A triangle is front-facing when its
outward normal points toward the camera.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from step7.camera_projection_v01 import Vector3

Triangle3D = tuple[Vector3, Vector3, Vector3]


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        first.x - second.x,
        first.y - second.y,
        first.z - second.z,
    )


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    )


def _dot(first: Vector3, second: Vector3) -> float:
    return first.x * second.x + first.y * second.y + first.z * second.z


def _validate_triangle(triangle: Sequence[Vector3]) -> Triangle3D:
    if len(triangle) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    vertices = tuple(triangle)
    for vertex in vertices:
        if not all(math.isfinite(value) for value in (vertex.x, vertex.y, vertex.z)):
            raise ValueError("triangle vertices must be finite")
    return vertices  # type: ignore[return-value]


def triangle_normal(triangle: Sequence[Vector3]) -> Vector3:
    """Return the unnormalized winding-derived triangle normal."""
    first, second, third = _validate_triangle(triangle)
    normal = _cross(_subtract(second, first), _subtract(third, first))
    if _dot(normal, normal) <= 1e-24:
        raise ValueError("triangle must have non-zero area")
    return normal


def is_triangle_front_facing(
    triangle: Sequence[Vector3],
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether an outward-wound triangle faces the camera origin.

    Edge-on triangles within the tolerance are not considered front-facing.
    """
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    first, second, third = _validate_triangle(triangle)
    normal = triangle_normal((first, second, third))
    centroid = Vector3(
        (first.x + second.x + third.x) / 3.0,
        (first.y + second.y + third.y) / 3.0,
        (first.z + second.z + third.z) / 3.0,
    )
    view_vector = Vector3(-centroid.x, -centroid.y, -centroid.z)
    return _dot(normal, view_vector) > tolerance_value


def select_front_facing_triangles(
    triangles: Iterable[Sequence[Vector3]],
    *,
    tolerance: float = 1e-12,
) -> tuple[Triangle3D, ...]:
    """Return front-facing triangles while preserving input order."""
    selected: list[Triangle3D] = []
    for triangle in triangles:
        validated = _validate_triangle(triangle)
        if is_triangle_front_facing(validated, tolerance=tolerance):
            selected.append(validated)
    return tuple(selected)
