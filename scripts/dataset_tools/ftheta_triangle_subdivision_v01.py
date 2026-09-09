"""Adaptive subdivision of F-theta projected camera-frame triangles.

A triangle is treated as locally linear only when the true F-theta projections
of its three edge midpoints and centroid agree with pixel-space barycentric
predictions within a configured error. Otherwise it is split into four child
triangles. This module does not clip against the near plane or image boundary,
rasterize pixels, interpolate depth, or compare Actors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from camera_projection_v01 import PixelProjection, Vector3

Triangle3D = tuple[Vector3, Vector3, Vector3]


class CameraPointProjector(Protocol):
    def project_camera_point(self, point: Vector3) -> PixelProjection: ...


@dataclass(frozen=True, slots=True)
class SubdividedFthetaTriangle:
    vertices_camera: Triangle3D
    vertex_projections: tuple[PixelProjection, PixelProjection, PixelProjection]
    subdivision_depth: int
    maximum_test_error_px: float


@dataclass(frozen=True, slots=True)
class AdaptiveFthetaTriangleSubdivision:
    triangles: tuple[SubdividedFthetaTriangle, ...]
    maximum_observed_error_px: float
    maximum_depth_reached: int
    stopped_by_depth_limit: bool


def _validate_point(point: Vector3, *, near_plane_m: float) -> None:
    if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
        raise ValueError("triangle vertices and subdivision points must be finite")
    if point.z < near_plane_m:
        raise ValueError("triangle vertex is behind near_plane_m")


def _projectable(projection: PixelProjection) -> bool:
    return (
        projection.positive_z
        and projection.within_fov
        and math.isfinite(projection.u)
        and math.isfinite(projection.v)
    )


def _project(point: Vector3, calibration: CameraPointProjector) -> PixelProjection:
    projection = calibration.project_camera_point(point)
    if not _projectable(projection):
        raise ValueError(
            "triangle vertex or subdivision test point must be positive-depth "
            "and within FOV"
        )
    return projection


def _interpolate(first: Vector3, second: Vector3, ratio: float) -> Vector3:
    return Vector3(
        first.x + ratio * (second.x - first.x),
        first.y + ratio * (second.y - first.y),
        first.z + ratio * (second.z - first.z),
    )


def _pixel_error(
    actual: PixelProjection,
    expected_u: float,
    expected_v: float,
) -> float:
    return math.hypot(actual.u - expected_u, actual.v - expected_v)


def subdivide_ftheta_triangle_adaptive(
    triangle_camera: Sequence[Vector3],
    calibration: CameraPointProjector,
    *,
    maximum_projection_error_px: float,
    maximum_depth: int,
    near_plane_m: float = 1e-3,
) -> AdaptiveFthetaTriangleSubdivision:
    """Recursively approximate one F-theta projected triangle.

    Every examined point must already be positive-depth and within the camera
    FOV. FOV and image-boundary clipping remain caller responsibilities.
    """
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    error_limit = float(maximum_projection_error_px)
    if not math.isfinite(error_limit) or error_limit <= 0.0:
        raise ValueError("maximum_projection_error_px must be positive and finite")
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int):
        raise TypeError("maximum_depth must be an integer")
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")
    near = float(near_plane_m)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_plane_m must be positive and finite")

    root: Triangle3D = tuple(triangle_camera)  # type: ignore[assignment]
    for point in root:
        _validate_point(point, near_plane_m=near)

    cache: dict[Vector3, PixelProjection] = {}

    def project(point: Vector3) -> PixelProjection:
        if point not in cache:
            _validate_point(point, near_plane_m=near)
            cache[point] = _project(point, calibration)
        return cache[point]

    maximum_observed_error = 0.0
    maximum_depth_reached = 0
    stopped_by_depth_limit = False

    def recurse(vertices: Triangle3D, depth: int) -> list[SubdividedFthetaTriangle]:
        nonlocal maximum_observed_error
        nonlocal maximum_depth_reached
        nonlocal stopped_by_depth_limit

        maximum_depth_reached = max(maximum_depth_reached, depth)
        first, second, third = vertices
        p_first, p_second, p_third = tuple(project(point) for point in vertices)

        midpoint_ab = _interpolate(first, second, 0.5)
        midpoint_bc = _interpolate(second, third, 0.5)
        midpoint_ca = _interpolate(third, first, 0.5)
        centroid = Vector3(
            (first.x + second.x + third.x) / 3.0,
            (first.y + second.y + third.y) / 3.0,
            (first.z + second.z + third.z) / 3.0,
        )
        p_ab = project(midpoint_ab)
        p_bc = project(midpoint_bc)
        p_ca = project(midpoint_ca)
        p_centroid = project(centroid)

        errors = (
            _pixel_error(p_ab, 0.5 * (p_first.u + p_second.u), 0.5 * (p_first.v + p_second.v)),
            _pixel_error(p_bc, 0.5 * (p_second.u + p_third.u), 0.5 * (p_second.v + p_third.v)),
            _pixel_error(p_ca, 0.5 * (p_third.u + p_first.u), 0.5 * (p_third.v + p_first.v)),
            _pixel_error(
                p_centroid,
                (p_first.u + p_second.u + p_third.u) / 3.0,
                (p_first.v + p_second.v + p_third.v) / 3.0,
            ),
        )
        local_error = max(errors)
        maximum_observed_error = max(maximum_observed_error, local_error)
        projections = (p_first, p_second, p_third)

        if local_error <= error_limit:
            return [
                SubdividedFthetaTriangle(
                    vertices_camera=vertices,
                    vertex_projections=projections,
                    subdivision_depth=depth,
                    maximum_test_error_px=local_error,
                )
            ]
        if depth >= maximum_depth:
            stopped_by_depth_limit = True
            return [
                SubdividedFthetaTriangle(
                    vertices_camera=vertices,
                    vertex_projections=projections,
                    subdivision_depth=depth,
                    maximum_test_error_px=local_error,
                )
            ]

        children: tuple[Triangle3D, ...] = (
            (first, midpoint_ab, midpoint_ca),
            (midpoint_ab, second, midpoint_bc),
            (midpoint_ca, midpoint_bc, third),
            (midpoint_ab, midpoint_bc, midpoint_ca),
        )
        leaves: list[SubdividedFthetaTriangle] = []
        for child in children:
            leaves.extend(recurse(child, depth + 1))
        return leaves

    leaves = recurse(root, 0)
    return AdaptiveFthetaTriangleSubdivision(
        triangles=tuple(leaves),
        maximum_observed_error_px=maximum_observed_error,
        maximum_depth_reached=maximum_depth_reached,
        stopped_by_depth_limit=stopped_by_depth_limit,
    )
