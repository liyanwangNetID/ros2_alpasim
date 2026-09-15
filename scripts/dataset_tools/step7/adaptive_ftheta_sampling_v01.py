#!/usr/bin/env python3
"""Adaptive sampling of projected F-theta line segments.

This module operates on one camera-frame 3D segment. It recursively subdivides
the segment until the projected midpoint is sufficiently close to the chord
between the projected endpoints, or until the maximum recursion depth is
reached. Integration with full Actor-box projection is intentionally deferred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from step7.camera_projection_v01 import FthetaCameraCalibration, PixelProjection, Vector3


@dataclass(frozen=True, slots=True)
class AdaptiveProjectedEdge:
    camera_points: tuple[Vector3, ...]
    projections: tuple[PixelProjection, ...]
    maximum_observed_chord_error_px: float
    maximum_depth_reached: int
    stopped_by_depth_limit: bool


def _interpolate(first: Vector3, second: Vector3, ratio: float) -> Vector3:
    return Vector3(
        first.x + ratio * (second.x - first.x),
        first.y + ratio * (second.y - first.y),
        first.z + ratio * (second.z - first.z),
    )


def point_to_segment_distance_px(
    point_u: float,
    point_v: float,
    first_u: float,
    first_v: float,
    second_u: float,
    second_v: float,
) -> float:
    """Return Euclidean distance from a 2D point to a closed segment."""
    values = (point_u, point_v, first_u, first_v, second_u, second_v)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("pixel coordinates must be finite")

    delta_u = second_u - first_u
    delta_v = second_v - first_v
    squared_length = delta_u * delta_u + delta_v * delta_v
    if squared_length <= 1e-18:
        return math.hypot(point_u - first_u, point_v - first_v)

    ratio = (
        (point_u - first_u) * delta_u
        + (point_v - first_v) * delta_v
    ) / squared_length
    ratio = min(1.0, max(0.0, ratio))
    nearest_u = first_u + ratio * delta_u
    nearest_v = first_v + ratio * delta_v
    return math.hypot(point_u - nearest_u, point_v - nearest_v)


def _projectable(projection: PixelProjection) -> bool:
    return (
        projection.positive_z
        and projection.within_fov
        and math.isfinite(projection.u)
        and math.isfinite(projection.v)
    )


def sample_projected_edge_adaptive(
    first_camera: Vector3,
    second_camera: Vector3,
    calibration: FthetaCameraCalibration,
    *,
    maximum_chord_error_px: float,
    maximum_depth: int,
) -> AdaptiveProjectedEdge:
    """Adaptively sample one positive-depth, in-FOV camera-frame edge.

    Both endpoints and every recursively examined midpoint must be projectable.
    Near-plane clipping remains the responsibility of the caller.
    """
    error_limit = float(maximum_chord_error_px)
    if not math.isfinite(error_limit) or error_limit <= 0.0:
        raise ValueError("maximum_chord_error_px must be positive and finite")
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int):
        raise TypeError("maximum_depth must be an integer")
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")

    first_projection = calibration.project_camera_point(first_camera)
    second_projection = calibration.project_camera_point(second_camera)
    if not _projectable(first_projection) or not _projectable(second_projection):
        raise ValueError("edge endpoints must be positive-depth and within FOV")

    maximum_observed_error = 0.0
    maximum_depth_reached = 0
    stopped_by_depth_limit = False

    def recurse(
        first_point: Vector3,
        first_pixel: PixelProjection,
        second_point: Vector3,
        second_pixel: PixelProjection,
        depth: int,
    ) -> list[tuple[Vector3, PixelProjection]]:
        nonlocal maximum_observed_error
        nonlocal maximum_depth_reached
        nonlocal stopped_by_depth_limit

        maximum_depth_reached = max(maximum_depth_reached, depth)
        midpoint = _interpolate(first_point, second_point, 0.5)
        midpoint_pixel = calibration.project_camera_point(midpoint)
        if not _projectable(midpoint_pixel):
            raise ValueError("edge midpoint is not positive-depth and within FOV")

        chord_error = point_to_segment_distance_px(
            midpoint_pixel.u,
            midpoint_pixel.v,
            first_pixel.u,
            first_pixel.v,
            second_pixel.u,
            second_pixel.v,
        )
        maximum_observed_error = max(maximum_observed_error, chord_error)

        if chord_error <= error_limit:
            return [
                (first_point, first_pixel),
                (midpoint, midpoint_pixel),
                (second_point, second_pixel),
            ]

        if depth >= maximum_depth:
            stopped_by_depth_limit = True
            return [
                (first_point, first_pixel),
                (midpoint, midpoint_pixel),
                (second_point, second_pixel),
            ]

        left = recurse(
            first_point,
            first_pixel,
            midpoint,
            midpoint_pixel,
            depth + 1,
        )
        right = recurse(
            midpoint,
            midpoint_pixel,
            second_point,
            second_pixel,
            depth + 1,
        )
        return left[:-1] + right

    pairs = recurse(
        first_camera,
        first_projection,
        second_camera,
        second_projection,
        0,
    )
    return AdaptiveProjectedEdge(
        camera_points=tuple(pair[0] for pair in pairs),
        projections=tuple(pair[1] for pair in pairs),
        maximum_observed_chord_error_px=maximum_observed_error,
        maximum_depth_reached=maximum_depth_reached,
        stopped_by_depth_limit=stopped_by_depth_limit,
    )
