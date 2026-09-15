#!/usr/bin/env python3
"""Build one Actor surface-depth raster from prepared camera triangles.

Each input triangle is processed through angular-FOV subdivision, safe boundary
polygon construction, raster-center depth sampling, and finally per-cell nearest
surface merging. The module handles one Actor only and does not perform
multi-Actor Z-buffer competition or visibility classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from step7.actor_surface_depth_raster_v01 import (
    ActorSurfaceDepthRaster,
    merge_actor_surface_depth_samples,
)
from step7.camera_projection_v01 import Vector3
from step7.fov_subdivided_triangle_depth_samples_v01 import (
    FovSubdividedTriangleDepthSamples,
    sample_fov_subdivided_camera_triangle_depths,
)

Triangle3D = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True, slots=True)
class ActorCameraSurfaceDepthRaster:
    surface_raster: ActorSurfaceDepthRaster
    triangle_results: tuple[FovSubdividedTriangleDepthSamples, ...]
    source_triangle_count: int
    generated_triangle_sample_count: int
    unresolved_boundary_count: int
    zero_center_sample_triangle_count: int


def build_actor_camera_surface_depth_raster(
    triangles_camera: Sequence[Sequence[Vector3]],
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
    depth_tolerance_m: float = 1e-9,
) -> ActorCameraSurfaceDepthRaster:
    """Process and merge all prepared camera-frame triangles for one Actor."""
    source = tuple(tuple(triangle) for triangle in triangles_camera)
    results = tuple(
        sample_fov_subdivided_camera_triangle_depths(
            triangle,
            calibration,
            max_angle_rad=max_angle_rad,
            maximum_depth=maximum_depth,
            maximum_boundary_extent_px=maximum_boundary_extent_px,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            raster_width=raster_width,
            raster_height=raster_height,
            near_plane_m=near_plane_m,
        )
        for triangle in source
    )
    generated_samples = tuple(
        sample
        for result in results
        for sample in result.all_triangle_samples
    )
    raster = merge_actor_surface_depth_samples(
        generated_samples,
        depth_tolerance_m=depth_tolerance_m,
    )
    return ActorCameraSurfaceDepthRaster(
        surface_raster=raster,
        triangle_results=results,
        source_triangle_count=len(source),
        generated_triangle_sample_count=len(generated_samples),
        unresolved_boundary_count=sum(
            result.unresolved_boundary_count for result in results
        ),
        zero_center_sample_triangle_count=sum(
            sample.center_sampled_cell_count == 0
            for sample in generated_samples
        ),
    )
