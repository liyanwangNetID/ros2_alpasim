#!/usr/bin/env python3
"""Barycentric coordinates for one 2D triangle and sample point.

This module computes affine barycentric weights in pixel or raster coordinates.
It supports either triangle winding, includes the triangle boundary, and rejects
degenerate triangles. It does not interpolate depth or perform perspective
correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BarycentricCoordinates:
    first_weight: float
    second_weight: float
    third_weight: float
    inside_closed_triangle: bool

    @property
    def weight_sum(self) -> float:
        return self.first_weight + self.second_weight + self.third_weight


def triangle_barycentric_coordinates(
    triangle: Sequence[Point2D],
    point: Point2D,
    *,
    degeneracy_epsilon: float = _EPSILON,
) -> BarycentricCoordinates:
    """Return affine barycentric weights of a point relative to a 2D triangle."""
    if len(triangle) != 3:
        raise ValueError("a triangle must contain exactly three points")

    epsilon = float(degeneracy_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("degeneracy_epsilon must be positive and finite")

    vertices = tuple(triangle)
    values = tuple(
        coordinate
        for item in (*vertices, point)
        for coordinate in (item.x, item.y)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("triangle and sample-point coordinates must be finite")

    first, second, third = vertices
    denominator = (
        (second.y - third.y) * (first.x - third.x)
        + (third.x - second.x) * (first.y - third.y)
    )

    coordinate_scale = max(1.0, *(abs(value) for value in values))
    area_tolerance = epsilon * coordinate_scale * coordinate_scale
    if abs(denominator) <= area_tolerance:
        raise ValueError("triangle must be non-degenerate")

    first_weight = (
        (second.y - third.y) * (point.x - third.x)
        + (third.x - second.x) * (point.y - third.y)
    ) / denominator
    second_weight = (
        (third.y - first.y) * (point.x - third.x)
        + (first.x - third.x) * (point.y - third.y)
    ) / denominator
    third_weight = 1.0 - first_weight - second_weight

    weight_tolerance = epsilon * 8.0
    inside = all(
        -weight_tolerance <= weight <= 1.0 + weight_tolerance
        for weight in (first_weight, second_weight, third_weight)
    )

    return BarycentricCoordinates(
        first_weight=first_weight,
        second_weight=second_weight,
        third_weight=third_weight,
        inside_closed_triangle=inside,
    )
