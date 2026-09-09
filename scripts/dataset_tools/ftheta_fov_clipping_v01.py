#!/usr/bin/env python3
"""Analytic clipping of camera-frame segments to an F-theta angular FOV.

For the recorded cameras max_angle is below pi/2. The valid angular domain is
therefore the convex cone:

    z > 0
    x^2 + y^2 <= tan(max_angle)^2 * z^2

A straight 3D segment can be clipped to that cone by solving one quadratic
inequality in the segment parameter. Near-plane clipping remains separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from camera_projection_v01 import Vector3

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SegmentInterval:
    start_ratio: float
    end_ratio: float


def interpolate_vector3(first: Vector3, second: Vector3, ratio: float) -> Vector3:
    return Vector3(
        first.x + ratio * (second.x - first.x),
        first.y + ratio * (second.y - first.y),
        first.z + ratio * (second.z - first.z),
    )


def _quadratic_value(a: float, b: float, c: float, value: float) -> float:
    return (a * value + b) * value + c


def quadratic_nonpositive_interval(
    a: float,
    b: float,
    c: float,
) -> SegmentInterval | None:
    """Solve a*t^2+b*t+c <= 0 over t in [0,1]."""
    values = (a, b, c)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quadratic coefficients must be finite")

    if abs(a) <= _EPSILON:
        if abs(b) <= _EPSILON:
            return SegmentInterval(0.0, 1.0) if c <= 0.0 else None
        boundary = -c / b
        if b > 0.0:
            start, end = 0.0, min(1.0, boundary)
        else:
            start, end = max(0.0, boundary), 1.0
        return SegmentInterval(start, end) if end >= start else None

    discriminant = b * b - 4.0 * a * c
    if discriminant < -_EPSILON:
        return (
            SegmentInterval(0.0, 1.0)
            if _quadratic_value(a, b, c, 0.5) <= 0.0
            else None
        )

    root_term = math.sqrt(max(0.0, discriminant))
    first_root = (-b - root_term) / (2.0 * a)
    second_root = (-b + root_term) / (2.0 * a)
    low_root = min(first_root, second_root)
    high_root = max(first_root, second_root)

    candidate_boundaries = [0.0, 1.0]
    if 0.0 < low_root < 1.0:
        candidate_boundaries.append(low_root)
    if 0.0 < high_root < 1.0:
        candidate_boundaries.append(high_root)
    candidate_boundaries = sorted(set(candidate_boundaries))

    valid_intervals: list[tuple[float, float]] = []
    for start, end in zip(candidate_boundaries, candidate_boundaries[1:]):
        midpoint = 0.5 * (start + end)
        if _quadratic_value(a, b, c, midpoint) <= _EPSILON:
            valid_intervals.append((start, end))

    for boundary in candidate_boundaries:
        if abs(_quadratic_value(a, b, c, boundary)) <= _EPSILON:
            valid_intervals.append((boundary, boundary))

    if not valid_intervals:
        return None
    return SegmentInterval(
        min(item[0] for item in valid_intervals),
        max(item[1] for item in valid_intervals),
    )


def clip_segment_to_angular_fov(
    first: Vector3,
    second: Vector3,
    *,
    max_angle_rad: float | None,
) -> tuple[Vector3, Vector3] | None:
    """Clip a positive-depth segment to theta <= max_angle_rad.

    When max_angle_rad is None, the original segment is returned. Angles must
    lie strictly between zero and pi/2 for the convex-cone formulation.
    """
    if max_angle_rad is None:
        return first, second
    maximum = float(max_angle_rad)
    if not math.isfinite(maximum) or not 0.0 < maximum < math.pi / 2.0:
        raise ValueError("max_angle_rad must be in (0, pi/2)")

    dx = second.x - first.x
    dy = second.y - first.y
    dz = second.z - first.z
    tangent_squared = math.tan(maximum) ** 2

    a = dx * dx + dy * dy - tangent_squared * dz * dz
    b = 2.0 * (
        first.x * dx + first.y * dy - tangent_squared * first.z * dz
    )
    c = (
        first.x * first.x
        + first.y * first.y
        - tangent_squared * first.z * first.z
    )

    interval = quadratic_nonpositive_interval(a, b, c)
    if interval is None:
        return None
    return (
        interpolate_vector3(first, second, interval.start_ratio),
        interpolate_vector3(first, second, interval.end_ratio),
    )
