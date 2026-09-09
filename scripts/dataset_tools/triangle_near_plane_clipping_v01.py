"""Near-plane clipping for camera-frame triangles used by Step 7E.

Triangles are clipped against z >= near_plane_m. The output preserves the
input winding and contains zero, one, or two triangles.
"""

from __future__ import annotations

import math
from typing import Sequence

from camera_projection_v01 import Vector3


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _intersection(first: Vector3, second: Vector3, near: float) -> Vector3:
    denominator = second.z - first.z
    if abs(denominator) <= 1e-12:
        raise ValueError("cannot intersect an edge parallel to the near plane")
    ratio = (near - first.z) / denominator
    return Vector3(
        first.x + ratio * (second.x - first.x),
        first.y + ratio * (second.y - first.y),
        near,
    )


def clip_polygon_to_positive_z(
    vertices: Sequence[Vector3],
    *,
    near_plane_m: float = 1e-3,
) -> tuple[Vector3, ...]:
    """Clip a winding-ordered polygon against z >= near_plane_m."""
    near = _finite(near_plane_m, "near_plane_m")
    if near <= 0.0:
        raise ValueError("near_plane_m must be positive")
    if len(vertices) < 3:
        raise ValueError("a polygon must contain at least three vertices")

    output: list[Vector3] = []
    previous = vertices[-1]
    previous_inside = previous.z >= near

    for current in vertices:
        current_inside = current.z >= near
        if current_inside:
            if not previous_inside:
                output.append(_intersection(previous, current, near))
            output.append(current)
        elif previous_inside:
            output.append(_intersection(previous, current, near))
        previous = current
        previous_inside = current_inside

    return tuple(output)


def clip_triangle_to_positive_z(
    triangle: Sequence[Vector3],
    *,
    near_plane_m: float = 1e-3,
) -> tuple[tuple[Vector3, Vector3, Vector3], ...]:
    """Clip one triangle and return zero, one, or two winding-preserving triangles."""
    if len(triangle) != 3:
        raise ValueError("a triangle must contain exactly three vertices")

    polygon = clip_polygon_to_positive_z(
        triangle,
        near_plane_m=near_plane_m,
    )
    if len(polygon) < 3:
        return ()
    if len(polygon) == 3:
        return ((polygon[0], polygon[1], polygon[2]),)
    if len(polygon) == 4:
        return (
            (polygon[0], polygon[1], polygon[2]),
            (polygon[0], polygon[2], polygon[3]),
        )
    raise RuntimeError(f"triangle clipping unexpectedly produced {len(polygon)} vertices")
