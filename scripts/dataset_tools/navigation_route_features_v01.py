#!/usr/bin/env python3
"""Pure geometry helpers for Step 6 navigation route features."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

FEATURE_FORMAT_VERSION = "0.1-draft"
PROFILER_VERSION = "0.1.0"


def normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def valid_local_points(message: Mapping[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in message.get("points", []):
        if not bool(item.get("valid", False)):
            continue
        position = item.get("position", {})
        x = position.get("x")
        y = position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        points.append((float(x), float(y)))
    return points


def route_geometry_features(
    points: Sequence[tuple[float, float]],
    *,
    minimum_segment_length_m: float = 0.5,
) -> dict[str, Any]:
    if len(points) < 2:
        raise ValueError("navigation route requires at least two valid points")

    headings: list[float] = []
    segment_lengths: list[float] = []
    path_length = 0.0
    for first, second in zip(points, points[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        path_length += length
        if length < minimum_segment_length_m:
            continue
        headings.append(math.atan2(dy, dx))
        segment_lengths.append(length)

    if not headings:
        raise ValueError("navigation route has no valid heading segments")

    unwrapped = [headings[0]]
    for heading in headings[1:]:
        unwrapped.append(unwrapped[-1] + normalize_angle(heading - unwrapped[-1]))

    start_heading = unwrapped[0]
    end_heading = unwrapped[-1]
    signed_heading_change = end_heading - start_heading
    absolute_heading_change = sum(
        abs(current - previous)
        for previous, current in zip(unwrapped, unwrapped[1:])
    )
    heading_excursion = max(abs(value - start_heading) for value in unwrapped)

    forward_points = [point for point in points if point[0] >= -1.0]
    forward_fraction = len(forward_points) / len(points)
    final_x, final_y = points[-1]
    maximum_left = max(point[1] for point in points)
    maximum_right = min(point[1] for point in points)

    return {
        "valid_point_count": len(points),
        "valid_heading_segment_count": len(headings),
        "route_path_length_m": path_length,
        "route_start_heading_rad": start_heading,
        "route_end_heading_rad": end_heading,
        "route_signed_heading_change_rad": signed_heading_change,
        "route_absolute_heading_change_rad": absolute_heading_change,
        "route_maximum_heading_excursion_rad": heading_excursion,
        "final_local_x_m": final_x,
        "final_local_y_m": final_y,
        "maximum_left_lateral_offset_m": maximum_left,
        "maximum_right_lateral_offset_m": maximum_right,
        "forward_point_fraction": forward_fraction,
    }
