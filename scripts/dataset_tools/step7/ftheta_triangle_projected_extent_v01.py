#!/usr/bin/env python3
"""Projected pixel extent of one positive-depth, in-FOV camera triangle.

This module projects the three triangle vertices with the supplied F-theta
calibration and summarizes the vertex-defined 2D triangle extent. It does not
claim to bound an un-subdivided nonlinear F-theta surface between vertices.
Callers should use it only after, or together with, projection-error control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from step7.camera_projection_v01 import PixelProjection, Vector3


class CameraPointProjector(Protocol):
    def project_camera_point(self, point: Vector3) -> PixelProjection: ...


@dataclass(frozen=True, slots=True)
class FthetaTriangleProjectedExtent:
    vertex_projections: tuple[PixelProjection, PixelProjection, PixelProjection]
    min_u_px: float
    min_v_px: float
    max_u_px: float
    max_v_px: float
    width_px: float
    height_px: float
    maximum_projected_edge_length_px: float
    projected_triangle_area_px2: float


def _validate_vertex(point: Vector3, *, near_plane_m: float) -> None:
    if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
        raise ValueError("triangle vertex coordinates must be finite")
    if point.z < near_plane_m:
        raise ValueError("triangle vertex is behind near_plane_m")


def _validate_projection(projection: PixelProjection) -> None:
    if not projection.positive_z:
        raise ValueError("projected triangle vertex must have positive depth")
    if not projection.within_fov:
        raise ValueError("projected triangle vertex must be within FOV")
    if not math.isfinite(projection.u) or not math.isfinite(projection.v):
        raise ValueError("projected pixel coordinates must be finite")


def _distance(first: PixelProjection, second: PixelProjection) -> float:
    return math.hypot(second.u - first.u, second.v - first.v)


def summarize_ftheta_triangle_projected_extent(
    triangle_camera: Sequence[Vector3],
    calibration: CameraPointProjector,
    *,
    near_plane_m: float = 1e-3,
) -> FthetaTriangleProjectedExtent:
    """Project three vertices and summarize their pixel-space triangle extent."""
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    near = float(near_plane_m)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_plane_m must be positive and finite")

    vertices = tuple(triangle_camera)
    for vertex in vertices:
        _validate_vertex(vertex, near_plane_m=near)

    projections = tuple(
        calibration.project_camera_point(vertex)
        for vertex in vertices
    )
    for projection in projections:
        _validate_projection(projection)

    u_values = tuple(projection.u for projection in projections)
    v_values = tuple(projection.v for projection in projections)
    min_u = min(u_values)
    min_v = min(v_values)
    max_u = max(u_values)
    max_v = max(v_values)

    first, second, third = projections
    edge_lengths = (
        _distance(first, second),
        _distance(second, third),
        _distance(third, first),
    )
    signed_twice_area = (
        (second.u - first.u) * (third.v - first.v)
        - (second.v - first.v) * (third.u - first.u)
    )

    return FthetaTriangleProjectedExtent(
        vertex_projections=projections,  # type: ignore[arg-type]
        min_u_px=min_u,
        min_v_px=min_v,
        max_u_px=max_u,
        max_v_px=max_v,
        width_px=max_u - min_u,
        height_px=max_v - min_v,
        maximum_projected_edge_length_px=max(edge_lengths),
        projected_triangle_area_px2=0.5 * abs(signed_twice_area),
    )
