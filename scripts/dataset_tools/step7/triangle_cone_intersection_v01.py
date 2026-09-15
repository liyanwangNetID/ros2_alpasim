#!/usr/bin/env python3
"""Exact positive-depth triangle intersection with an angular FOV cone.

For z > 0, perspective normalization maps a 3D triangle exactly to the 2D
triangle formed by (x/z, y/z) at its vertices. The angular FOV cone becomes a
2D disk of radius tan(max_angle_rad). Therefore triangle/cone intersection is
reduced to the exact closest-point query from the origin to that normalized
2D triangle.

This module only classifies intersection and reports the minimizing location.
It does not create clipped triangles, project F-theta pixels, or rasterize.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from step7.camera_projection_v01 import Vector3

LocationType = Literal["vertex", "edge", "interior"]
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TriangleConeIntersection:
    intersects: bool
    minimum_normalized_radius_squared: float
    cone_radius_squared: float
    margin_squared: float
    minimizing_location_type: LocationType
    minimizing_feature_indices: tuple[int, ...]
    closest_normalized_point: tuple[float, float]


def _validate_max_angle(max_angle_rad: float) -> float:
    maximum = float(max_angle_rad)
    if not math.isfinite(maximum) or not 0.0 < maximum < math.pi / 2.0:
        raise ValueError("max_angle_rad must be in (0, pi/2)")
    return maximum


def _normalized_point(point: Vector3) -> tuple[float, float]:
    if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
        raise ValueError("triangle coordinates must be finite")
    if point.z <= 0.0:
        raise ValueError("triangle vertices must have positive depth")
    return point.x / point.z, point.y / point.z


def _cross_2d(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _origin_in_triangle(points: tuple[tuple[float, float], ...]) -> bool:
    first, second, third = points
    values = (
        _cross_2d(first, second, (0.0, 0.0)),
        _cross_2d(second, third, (0.0, 0.0)),
        _cross_2d(third, first, (0.0, 0.0)),
    )
    has_positive = any(value > _EPSILON for value in values)
    has_negative = any(value < -_EPSILON for value in values)
    on_boundary = any(
        abs(value) <= _EPSILON
        for value in values
    )
    return (
        not (has_positive and has_negative)
        and not on_boundary
    )


def _closest_on_segment(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    delta_x = second[0] - first[0]
    delta_y = second[1] - first[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= _EPSILON:
        return first, 0.0
    ratio = -(
        first[0] * delta_x + first[1] * delta_y
    ) / length_squared
    ratio = min(1.0, max(0.0, ratio))
    return (
        first[0] + ratio * delta_x,
        first[1] + ratio * delta_y,
    ), ratio


def triangle_intersects_angular_fov_cone(
    triangle_camera: Sequence[Vector3],
    *,
    max_angle_rad: float,
) -> TriangleConeIntersection:
    """Return exact intersection with the positive angular FOV cone.

    The cone boundary is included. Degenerate normalized triangles are handled
    through their vertices and edges.
    """
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    maximum = _validate_max_angle(max_angle_rad)
    points = tuple(_normalized_point(point) for point in triangle_camera)
    cone_radius_squared = math.tan(maximum) ** 2

    area_twice = abs(_cross_2d(points[0], points[1], points[2]))
    if area_twice > _EPSILON and _origin_in_triangle(points):
        minimum_squared = 0.0
        closest = (0.0, 0.0)
        location_type: LocationType = "interior"
        indices: tuple[int, ...] = (0, 1, 2)
    else:
        candidates: list[
            tuple[float, tuple[float, float], LocationType, tuple[int, ...]]
        ] = []
        for index, point in enumerate(points):
            squared = point[0] * point[0] + point[1] * point[1]
            candidates.append((squared, point, "vertex", (index,)))
        for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
            point, ratio = _closest_on_segment(
                points[first_index], points[second_index]
            )
            squared = point[0] * point[0] + point[1] * point[1]
            if ratio <= _EPSILON:
                feature_type: LocationType = "vertex"
                feature_indices = (first_index,)
            elif ratio >= 1.0 - _EPSILON:
                feature_type = "vertex"
                feature_indices = (second_index,)
            else:
                feature_type = "edge"
                feature_indices = (first_index, second_index)
            candidates.append(
                (squared, point, feature_type, feature_indices)
            )
        minimum_squared, closest, location_type, indices = min(
            candidates,
            key=lambda item: (item[0], len(item[3]), item[3]),
        )

    scale = max(1.0, minimum_squared, cone_radius_squared)
    intersects = minimum_squared <= cone_radius_squared + _EPSILON * scale
    return TriangleConeIntersection(
        intersects=intersects,
        minimum_normalized_radius_squared=minimum_squared,
        cone_radius_squared=cone_radius_squared,
        margin_squared=minimum_squared - cone_radius_squared,
        minimizing_location_type=location_type,
        minimizing_feature_indices=indices,
        closest_normalized_point=closest,
    )
