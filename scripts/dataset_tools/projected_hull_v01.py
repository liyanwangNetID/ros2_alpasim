#!/usr/bin/env python3
"""Step 7D.5 polygon geometry for projected Actor boxes.

This module computes a convex hull from projected F-theta edge samples, clips
that hull to the image rectangle, and reports polygon areas. The polygon is a
geometric projection of the Actor AABB, not a visible-pixel or segmentation
mask.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class Point2D:
    u: float
    v: float


@dataclass(frozen=True, slots=True)
class ProjectedHullGeometry:
    projected_hull: tuple[Point2D, ...]
    clipped_hull: tuple[Point2D, ...]
    projected_hull_area_px: float
    inside_image_hull_area_px: float
    inside_image_hull_ratio: float
    truncated_by_image: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _cross(origin: Point2D, first: Point2D, second: Point2D) -> float:
    return (
        (first.u - origin.u) * (second.v - origin.v)
        - (first.v - origin.v) * (second.u - origin.u)
    )


def convex_hull(points: Sequence[Point2D]) -> tuple[Point2D, ...]:
    """Return a counter-clockwise convex hull using the monotonic chain."""
    unique = sorted(
        {
            Point2D(_finite(point.u, "point.u"), _finite(point.v, "point.v"))
            for point in points
        },
        key=lambda point: (point.u, point.v),
    )
    if len(unique) <= 1:
        return tuple(unique)

    lower: list[Point2D] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[Point2D] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return tuple(lower[:-1] + upper[:-1])


def polygon_area(points: Sequence[Point2D]) -> float:
    """Return the unsigned polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    doubled = 0.0
    for first, second in zip(points, (*points[1:], points[0])):
        doubled += first.u * second.v - second.u * first.v
    return 0.5 * abs(doubled)


def _clip_polygon(
    polygon: Sequence[Point2D],
    *,
    inside,
    intersection,
) -> tuple[Point2D, ...]:
    if not polygon:
        return ()
    output: list[Point2D] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return tuple(output)


def clip_polygon_to_image(
    polygon: Sequence[Point2D],
    *,
    width: int,
    height: int,
) -> tuple[Point2D, ...]:
    """Clip a polygon to [0,width-1] x [0,height-1]."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    right = float(width - 1)
    bottom = float(height - 1)

    result = tuple(polygon)
    result = _clip_polygon(
        result,
        inside=lambda point: point.u >= 0.0,
        intersection=lambda first, second: Point2D(
            0.0,
            first.v + (second.v - first.v) * (0.0 - first.u) / (second.u - first.u),
        ),
    )
    result = _clip_polygon(
        result,
        inside=lambda point: point.u <= right,
        intersection=lambda first, second: Point2D(
            right,
            first.v + (second.v - first.v) * (right - first.u) / (second.u - first.u),
        ),
    )
    result = _clip_polygon(
        result,
        inside=lambda point: point.v >= 0.0,
        intersection=lambda first, second: Point2D(
            first.u + (second.u - first.u) * (0.0 - first.v) / (second.v - first.v),
            0.0,
        ),
    )
    result = _clip_polygon(
        result,
        inside=lambda point: point.v <= bottom,
        intersection=lambda first, second: Point2D(
            first.u + (second.u - first.u) * (bottom - first.v) / (second.v - first.v),
            bottom,
        ),
    )
    return result


def summarize_projected_hull(
    projected_points: Sequence[Point2D],
    *,
    width: int,
    height: int,
) -> ProjectedHullGeometry:
    """Build, clip, and measure a projected AABB hull."""
    hull = convex_hull(projected_points)
    clipped = clip_polygon_to_image(hull, width=width, height=height)
    projected_area = polygon_area(hull)
    inside_area = polygon_area(clipped)
    ratio = inside_area / projected_area if projected_area > 0.0 else 0.0
    return ProjectedHullGeometry(
        projected_hull=hull,
        clipped_hull=clipped,
        projected_hull_area_px=projected_area,
        inside_image_hull_area_px=inside_area,
        inside_image_hull_ratio=ratio,
        truncated_by_image=(
            projected_area > 0.0 and inside_area < projected_area - 1e-9
        ),
    )
