#!/usr/bin/env python3
"""Conservative raster-cell coverage for one 2D projected triangle.

The function maps pixel coordinates to a lower-resolution raster using explicit
scale factors and returns cells whose closed rectangular area intersects the
closed triangle. This module does not interpolate depth, resolve ownership,
clip to a camera image, or compare Actors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PixelPoint:
    u: float
    v: float


@dataclass(frozen=True, slots=True)
class RasterCell:
    column: int
    row: int


@dataclass(frozen=True, slots=True)
class RasterizedTriangleCoverage:
    cells: tuple[RasterCell, ...]
    candidate_column_range: tuple[int, int] | None
    candidate_row_range: tuple[int, int] | None


_EPSILON = 1e-12


def _cross(first: PixelPoint, second: PixelPoint, third: PixelPoint) -> float:
    return (
        (second.u - first.u) * (third.v - first.v)
        - (second.v - first.v) * (third.u - first.u)
    )


def _point_in_closed_triangle(point: PixelPoint, triangle: tuple[PixelPoint, ...]) -> bool:
    first, second, third = triangle
    values = (
        _cross(first, second, point),
        _cross(second, third, point),
        _cross(third, first, point),
    )
    return not (
        any(value > _EPSILON for value in values)
        and any(value < -_EPSILON for value in values)
    )


def _orientation(first: PixelPoint, second: PixelPoint, third: PixelPoint) -> int:
    value = _cross(first, second, third)
    if value > _EPSILON:
        return 1
    if value < -_EPSILON:
        return -1
    return 0


def _on_segment(first: PixelPoint, second: PixelPoint, point: PixelPoint) -> bool:
    return (
        min(first.u, second.u) - _EPSILON <= point.u <= max(first.u, second.u) + _EPSILON
        and min(first.v, second.v) - _EPSILON <= point.v <= max(first.v, second.v) + _EPSILON
        and _orientation(first, second, point) == 0
    )


def _closed_segments_intersect(
    first_a: PixelPoint,
    first_b: PixelPoint,
    second_a: PixelPoint,
    second_b: PixelPoint,
) -> bool:
    o1 = _orientation(first_a, first_b, second_a)
    o2 = _orientation(first_a, first_b, second_b)
    o3 = _orientation(second_a, second_b, first_a)
    o4 = _orientation(second_a, second_b, first_b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(first_a, first_b, second_a))
        or (o2 == 0 and _on_segment(first_a, first_b, second_b))
        or (o3 == 0 and _on_segment(second_a, second_b, first_a))
        or (o4 == 0 and _on_segment(second_a, second_b, first_b))
    )


def _cell_intersects_triangle(
    column: int,
    row: int,
    triangle: tuple[PixelPoint, ...],
) -> bool:
    corners = (
        PixelPoint(float(column), float(row)),
        PixelPoint(float(column + 1), float(row)),
        PixelPoint(float(column + 1), float(row + 1)),
        PixelPoint(float(column), float(row + 1)),
    )
    if any(_point_in_closed_triangle(corner, triangle) for corner in corners):
        return True
    if any(
        column - _EPSILON <= vertex.u <= column + 1 + _EPSILON
        and row - _EPSILON <= vertex.v <= row + 1 + _EPSILON
        for vertex in triangle
    ):
        return True
    triangle_edges = ((0, 1), (1, 2), (2, 0))
    cell_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    return any(
        _closed_segments_intersect(
            triangle[first_triangle], triangle[second_triangle],
            corners[first_cell], corners[second_cell],
        )
        for first_triangle, second_triangle in triangle_edges
        for first_cell, second_cell in cell_edges
    )


def rasterize_projected_triangle_cells(
    triangle_pixels: Sequence[PixelPoint],
    *,
    image_width_px: int,
    image_height_px: int,
    raster_width: int,
    raster_height: int,
) -> RasterizedTriangleCoverage:
    """Return raster cells whose closed area intersects the projected triangle."""
    if len(triangle_pixels) != 3:
        raise ValueError("a triangle must contain exactly three pixel points")
    dimensions = (image_width_px, image_height_px, raster_width, raster_height)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
        raise TypeError("image and raster dimensions must be integers")
    if any(value <= 0 for value in dimensions):
        raise ValueError("image and raster dimensions must be positive")
    triangle_source = tuple(triangle_pixels)
    for point in triangle_source:
        if not math.isfinite(point.u) or not math.isfinite(point.v):
            raise ValueError("triangle pixel coordinates must be finite")

    scale_u = raster_width / image_width_px
    scale_v = raster_height / image_height_px
    triangle = tuple(
        PixelPoint(point.u * scale_u, point.v * scale_v)
        for point in triangle_source
    )
    minimum_u = min(point.u for point in triangle)
    minimum_v = min(point.v for point in triangle)
    maximum_u = max(point.u for point in triangle)
    maximum_v = max(point.v for point in triangle)

    # Closed-cell conservative coverage must examine the cell immediately
    # before an integer-valued minimum. For example, a triangle vertex at
    # (2, 2) touches both cells ending at that boundary and cells beginning
    # at that boundary.
    min_u = max(
        0,
        math.ceil(minimum_u) - 1,
    )
    min_v = max(
        0,
        math.ceil(minimum_v) - 1,
    )
    max_u = min(
        raster_width - 1,
        math.floor(maximum_u),
    )
    max_v = min(
        raster_height - 1,
        math.floor(maximum_v),
    )
    if max_u < min_u or max_v < min_v:
        return RasterizedTriangleCoverage((), None, None)

    cells = tuple(
        RasterCell(column=column, row=row)
        for row in range(min_v, max_v + 1)
        for column in range(min_u, max_u + 1)
        if _cell_intersects_triangle(column, row, triangle)
    )
    return RasterizedTriangleCoverage(
        cells=cells,
        candidate_column_range=(min_u, max_u),
        candidate_row_range=(min_v, max_v),
    )
