#!/usr/bin/env python3
"""Perspective-correct camera-depth interpolation over a projected triangle.

Given screen-space barycentric weights and positive camera-frame vertex depths,
this module interpolates reciprocal depth and converts it back to camera depth.
It does not compute barycentric coordinates, rasterize cells, or update a
Z-buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

_DEFAULT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PerspectiveDepthInterpolation:
    reciprocal_depth_per_m: float
    depth_m: float


def interpolate_perspective_camera_depth(
    vertex_depths_m: Sequence[float],
    barycentric_weights: Sequence[float],
    *,
    weight_tolerance: float = _DEFAULT_TOLERANCE,
) -> PerspectiveDepthInterpolation:
    """Interpolate positive camera depth from screen-space barycentric weights.

    The weights must describe a point in the closed projected triangle: each
    weight is within [0, 1] up to tolerance and their sum is one up to tolerance.
    """
    if len(vertex_depths_m) != 3:
        raise ValueError("vertex_depths_m must contain exactly three values")
    if len(barycentric_weights) != 3:
        raise ValueError("barycentric_weights must contain exactly three values")

    tolerance = float(weight_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("weight_tolerance must be positive and finite")

    depths = tuple(float(value) for value in vertex_depths_m)
    weights = tuple(float(value) for value in barycentric_weights)
    if not all(math.isfinite(value) for value in depths):
        raise ValueError("vertex depths must be finite")
    if not all(value > 0.0 for value in depths):
        raise ValueError("vertex depths must be positive")
    if not all(math.isfinite(value) for value in weights):
        raise ValueError("barycentric weights must be finite")

    weight_sum = sum(weights)
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("barycentric weights must sum to one")
    if not all(
        -tolerance <= weight <= 1.0 + tolerance
        for weight in weights
    ):
        raise ValueError("barycentric weights must describe the closed triangle")

    reciprocal_depth = sum(
        weight / depth
        for weight, depth in zip(weights, depths)
    )
    if not math.isfinite(reciprocal_depth) or reciprocal_depth <= 0.0:
        raise ValueError("interpolated reciprocal depth must be positive and finite")

    depth = 1.0 / reciprocal_depth
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("interpolated depth must be positive and finite")

    return PerspectiveDepthInterpolation(
        reciprocal_depth_per_m=reciprocal_depth,
        depth_m=depth,
    )
