#!/usr/bin/env python3
"""Road-level heading features around an observed Route branch.

Pure geometry only. This module does not classify Navigation actions.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

FEATURE_VERSION = "0.1-draft"


def normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def cumulative_distances(
    points: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    if not points:
        return tuple()
    values = [0.0]
    for first, second in zip(points, points[1:]):
        values.append(
            values[-1]
            + math.hypot(second[0] - first[0], second[1] - first[1])
        )
    return tuple(values)


def interpolate_at_distance(
    points: Sequence[tuple[float, float]],
    distances: Sequence[float],
    target_m: float,
) -> tuple[float, float]:
    if not points or len(points) != len(distances):
        raise ValueError("points and distances must be non-empty and aligned")
    target = min(max(float(target_m), distances[0]), distances[-1])
    for index in range(1, len(points)):
        if distances[index] < target:
            continue
        span = distances[index] - distances[index - 1]
        if span <= 1e-9:
            return points[index]
        ratio = (target - distances[index - 1]) / span
        return (
            points[index - 1][0]
            + ratio * (points[index][0] - points[index - 1][0]),
            points[index - 1][1]
            + ratio * (points[index][1] - points[index - 1][1]),
        )
    return points[-1]


def chord_heading(
    points: Sequence[tuple[float, float]],
    distances: Sequence[float],
    start_m: float,
    end_m: float,
) -> float:
    if end_m <= start_m:
        raise ValueError("heading window must have positive length")
    first = interpolate_at_distance(points, distances, start_m)
    second = interpolate_at_distance(points, distances, end_m)
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    if math.hypot(dx, dy) <= 1e-6:
        raise ValueError("heading window has negligible displacement")
    return math.atan2(dy, dx)


def extract_road_level_features(
    points: Sequence[tuple[float, float]],
    *,
    branch_distance_m: float,
    pre_window_m: float = 15.0,
    post_offset_m: float = 10.0,
    post_window_m: float = 20.0,
) -> dict[str, Any]:
    """Measure Route direction before and after one observed branch.

    The post window begins after a configurable offset so initially parallel
    successor lanes do not dominate the road-level direction estimate.
    """
    if len(points) < 2:
        raise ValueError("Route requires at least two points")
    distances = cumulative_distances(points)
    total = distances[-1]
    branch_distance = float(branch_distance_m)
    pre_start = max(0.0, branch_distance - pre_window_m)
    pre_end = min(branch_distance, total)
    post_start = min(branch_distance + post_offset_m, total)
    post_end = min(post_start + post_window_m, total)

    pre_length = pre_end - pre_start
    post_length = post_end - post_start
    status = "available"
    reasons: list[str] = []
    if pre_length < 5.0:
        status = "unavailable"
        reasons.append("insufficient_pre_branch_route_length")
    if post_length < 10.0:
        status = "unavailable"
        reasons.append("insufficient_post_branch_route_length")

    result: dict[str, Any] = {
        "status": status,
        "reasons": reasons,
        "route_length_m": total,
        "branch_distance_m": branch_distance,
        "pre_window_start_m": pre_start,
        "pre_window_end_m": pre_end,
        "pre_window_length_m": pre_length,
        "post_window_start_m": post_start,
        "post_window_end_m": post_end,
        "post_window_length_m": post_length,
    }
    if status != "available":
        return result

    pre_heading = chord_heading(points, distances, pre_start, pre_end)
    post_heading = chord_heading(points, distances, post_start, post_end)
    change = normalize_angle(post_heading - pre_heading)
    result.update({
        "pre_branch_heading_rad": pre_heading,
        "route_post_branch_heading_rad": post_heading,
        "route_road_level_heading_change_rad": change,
        "route_road_level_heading_change_deg": math.degrees(change),
    })
    return result
