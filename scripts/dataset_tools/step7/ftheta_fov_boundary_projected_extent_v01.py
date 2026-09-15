#!/usr/bin/env python3
"""Projected footprint of triangle evidence inside an angular FOV cone.

For a positive-depth camera-frame triangle, this module collects:
- original vertices that are inside the angular FOV; and
- endpoints of analytically clipped triangle-edge segments.

It projects the unique collected points and summarizes their pixel footprint.
An empty evidence set is reported explicitly because a triangle interior can
intersect or contain the FOV cone even when no vertex or edge segment is inside.
This module does not classify or clip the full triangle surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from step7.camera_projection_v01 import PixelProjection, Vector3
from step7.triangle_angular_fov_diagnostics_v01 import classify_triangle_angular_fov


class CameraPointProjector(Protocol):
    def project_camera_point(self, point: Vector3) -> PixelProjection: ...


@dataclass(frozen=True, slots=True)
class AngularFovBoundaryProjectedExtent:
    camera_points: tuple[Vector3, ...]
    projections: tuple[PixelProjection, ...]
    point_count: int
    width_px: float | None
    height_px: float | None
    maximum_pair_distance_px: float | None
    has_measurable_extent: bool


def _points_close(
    first: Vector3,
    second: Vector3,
) -> bool:
    return all(
        math.isclose(
            first_value,
            second_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for first_value, second_value in (
            (first.x, second.x),
            (first.y, second.y),
            (first.z, second.z),
        )
    )


def _append_unique_point(
    points: list[Vector3],
    candidate: Vector3,
) -> None:
    if not any(
        _points_close(existing, candidate)
        for existing in points
    ):
        points.append(candidate)


def _validate_point(point: Vector3) -> None:
    if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
        raise ValueError("triangle coordinates must be finite")
    if point.z <= 0.0:
        raise ValueError("triangle vertices must have positive depth")


def summarize_angular_fov_boundary_projected_extent(
    triangle_camera: Sequence[Vector3],
    calibration: CameraPointProjector,
    *,
    max_angle_rad: float | None,
) -> AngularFovBoundaryProjectedExtent:
    """Project unique in-FOV vertex and edge-intersection evidence points."""
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    vertices = tuple(triangle_camera)
    for vertex in vertices:
        _validate_point(vertex)

    diagnostics = classify_triangle_angular_fov(
        vertices,
        max_angle_rad=max_angle_rad,
    )

    unique_points: list[Vector3] = []

    for vertex, inside in zip(
        vertices,
        diagnostics.vertex_inside,
    ):
        if inside:
            _append_unique_point(
                unique_points,
                vertex,
            )

    for edge in diagnostics.edge_intersections:
        if edge.clipped_segment is None:
            continue

        for point in edge.clipped_segment:
            _append_unique_point(
                unique_points,
                point,
            )

    points = tuple(unique_points)
    projections_list: list[PixelProjection] = []
    for point in points:
        projection = calibration.project_camera_point(point)
        if not projection.positive_z:
            raise ValueError("evidence point projection must have positive depth")
        if not projection.within_fov:
            raise ValueError("evidence point projection must be within FOV")
        if not math.isfinite(projection.u) or not math.isfinite(projection.v):
            raise ValueError("evidence point pixel coordinates must be finite")
        projections_list.append(projection)
    projections = tuple(projections_list)

    if not projections:
        return AngularFovBoundaryProjectedExtent(
            camera_points=points,
            projections=projections,
            point_count=0,
            width_px=None,
            height_px=None,
            maximum_pair_distance_px=None,
            has_measurable_extent=False,
        )

    u_values = [item.u for item in projections]
    v_values = [item.v for item in projections]
    maximum_distance = 0.0
    for first_index, first in enumerate(projections):
        for second in projections[first_index + 1:]:
            maximum_distance = max(
                maximum_distance,
                math.hypot(second.u - first.u, second.v - first.v),
            )

    return AngularFovBoundaryProjectedExtent(
        camera_points=points,
        projections=projections,  # type: ignore[arg-type]
        point_count=len(points),
        width_px=max(u_values) - min(u_values),
        height_px=max(v_values) - min(v_values),
        maximum_pair_distance_px=maximum_distance,
        has_measurable_extent=True,
    )
