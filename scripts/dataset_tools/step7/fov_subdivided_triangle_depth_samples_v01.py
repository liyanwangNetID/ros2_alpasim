#!/usr/bin/env python3
"""Adapt angular-FOV subdivision output to safe raster depth samples.

Fully inside leaves are sampled directly. Pixel-scale boundary leaves are first
converted to safe in-FOV polygons, triangulated, and then sampled. Unmeasurable,
degenerate, and depth-limited boundary leaves remain explicit diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from step7.camera_projection_v01 import Vector3
from step7.camera_triangle_depth_samples_v01 import sample_camera_triangle_depths
from step7.projected_triangle_depth_samples_v01 import ProjectedTriangleDepthSamples
from step7.safe_angular_fov_polygon_v01 import build_safe_angular_fov_polygon
from step7.triangle_angular_fov_subdivision_v01 import (
    AngularFovTriangleSubdivision,
    subdivide_triangle_to_angular_fov,
)


@dataclass(frozen=True, slots=True)
class FovSubdividedTriangleDepthSamples:
    subdivision: AngularFovTriangleSubdivision
    accepted_triangle_samples: tuple[ProjectedTriangleDepthSamples, ...]
    boundary_triangle_samples: tuple[ProjectedTriangleDepthSamples, ...]
    all_triangle_samples: tuple[ProjectedTriangleDepthSamples, ...]
    sampled_inside_triangle_count: int
    sampled_boundary_triangle_count: int
    measurable_boundary_polygon_count: int
    unmeasurable_boundary_polygon_count: int
    degenerate_boundary_polygon_count: int
    unresolved_boundary_count: int


def sample_fov_subdivided_camera_triangle_depths(
    triangle_camera: Sequence[Vector3],
    calibration,
    *,
    max_angle_rad: float | None,
    maximum_depth: int,
    maximum_boundary_extent_px: float,
    image_width_px: int,
    image_height_px: int,
    raster_width: int,
    raster_height: int,
    near_plane_m: float = 1e-3,
) -> FovSubdividedTriangleDepthSamples:
    """Subdivide by angular FOV and sample safe inside geometry."""
    subdivision = subdivide_triangle_to_angular_fov(
        triangle_camera,
        max_angle_rad=max_angle_rad,
        maximum_depth=maximum_depth,
        calibration=calibration,
        maximum_boundary_extent_px=maximum_boundary_extent_px,
    )

    def sample_triangle(vertices):
        return sample_camera_triangle_depths(
            vertices,
            calibration,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            raster_width=raster_width,
            raster_height=raster_height,
            near_plane_m=near_plane_m,
        )

    accepted_samples = tuple(
        sample_triangle(accepted.vertices_camera)
        for accepted in subdivision.accepted_inside_triangles
    )

    boundary_samples: list[ProjectedTriangleDepthSamples] = []
    measurable_count = 0
    unmeasurable_count = 0
    degenerate_count = 0

    for boundary in subdivision.boundary_approximated_triangles:
        polygon = build_safe_angular_fov_polygon(
            boundary.vertices_camera,
            max_angle_rad=max_angle_rad,
        )
        if not polygon.has_measurable_polygon:
            unmeasurable_count += 1
            continue
        if polygon.is_degenerate:
            degenerate_count += 1
            continue
        measurable_count += 1
        boundary_samples.extend(
            sample_triangle(vertices)
            for vertices in polygon.triangles_camera
        )

    boundary_samples_tuple = tuple(boundary_samples)
    all_samples = accepted_samples + boundary_samples_tuple
    return FovSubdividedTriangleDepthSamples(
        subdivision=subdivision,
        accepted_triangle_samples=accepted_samples,
        boundary_triangle_samples=boundary_samples_tuple,
        all_triangle_samples=all_samples,
        sampled_inside_triangle_count=len(accepted_samples),
        sampled_boundary_triangle_count=len(boundary_samples_tuple),
        measurable_boundary_polygon_count=measurable_count,
        unmeasurable_boundary_polygon_count=unmeasurable_count,
        degenerate_boundary_polygon_count=degenerate_count,
        unresolved_boundary_count=(
            len(subdivision.boundary_depth_limited_triangles)
            + len(subdivision.boundary_unmeasurable_triangles)
            + unmeasurable_count
            + degenerate_count
        ),
    )
