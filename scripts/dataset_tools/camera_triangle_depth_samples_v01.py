#!/usr/bin/env python3
"""Adapt one camera-frame triangle to projected raster depth samples.

The adapter projects three positive-depth camera-frame vertices with the supplied
camera calibration, validates that all projected vertices are inside the angular
FOV, and delegates raster center sampling and perspective-correct depth
interpolation to projected_triangle_depth_samples_v01.

This module does not perform near-plane clipping, FOV boundary subdivision,
F-theta projection-error subdivision, Actor surface merging, or Z-buffering.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

from camera_projection_v01 import PixelProjection, Vector3
from projected_triangle_depth_samples_v01 import (
    ProjectedTriangleDepthSamples,
    sample_projected_triangle_depths,
)
from projected_triangle_raster_cells_v01 import PixelPoint


class CameraPointProjector(Protocol):
    def project_camera_point(self, point: Vector3) -> PixelProjection: ...


def sample_camera_triangle_depths(
    triangle_camera: Sequence[Vector3],
    calibration: CameraPointProjector,
    *,
    image_width_px: int,
    image_height_px: int,
    raster_width: int,
    raster_height: int,
    near_plane_m: float = 1e-3,
) -> ProjectedTriangleDepthSamples:
    """Project one in-FOV camera triangle and sample its raster-center depths."""
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three camera points")

    near = float(near_plane_m)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_plane_m must be positive and finite")

    vertices = tuple(triangle_camera)
    for vertex in vertices:
        if not all(math.isfinite(value) for value in (vertex.x, vertex.y, vertex.z)):
            raise ValueError("camera triangle coordinates must be finite")
        if vertex.z < near:
            raise ValueError("camera triangle vertex is behind near_plane_m")

    projections = tuple(
        calibration.project_camera_point(vertex)
        for vertex in vertices
    )
    for projection in projections:
        if not projection.positive_z:
            raise ValueError("projected triangle vertex must have positive depth")
        if not projection.within_fov:
            raise ValueError("projected triangle vertex must be within FOV")
        if not math.isfinite(projection.u) or not math.isfinite(projection.v):
            raise ValueError("projected pixel coordinates must be finite")

    return sample_projected_triangle_depths(
        tuple(PixelPoint(item.u, item.v) for item in projections),
        tuple(vertex.z for vertex in vertices),
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        raster_width=raster_width,
        raster_height=raster_height,
    )
