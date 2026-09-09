"""Compose Actor-box surface preparation for Step 7E occlusion.

Pipeline for camera-frame Actor-box corners:
1. triangulate the six outward-wound box surfaces;
2. select front-facing source triangles;
3. clip selected triangles against z >= near_plane_m.

This module does not project, rasterize, or compare depths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from box_surface_geometry_v01 import triangulate_box_surfaces
from camera_projection_v01 import Vector3
from triangle_facing_v01 import is_triangle_front_facing
from triangle_near_plane_clipping_v01 import clip_triangle_to_positive_z


@dataclass(frozen=True, slots=True)
class CameraFacingBoxTriangle:
    face_name: str
    source_corner_indices: tuple[int, int, int]
    vertices_camera: tuple[Vector3, Vector3, Vector3]


def prepare_camera_facing_box_triangles(
    corners_camera: Sequence[Vector3],
    *,
    near_plane_m: float = 1e-3,
    facing_tolerance: float = 1e-12,
) -> tuple[CameraFacingBoxTriangle, ...]:
    """Return near-clipped triangles from camera-facing Actor-box surfaces.

    Source-face visibility is determined before near-plane clipping. A source
    triangle can therefore produce zero, one, or two output triangles.
    Output order follows the fixed box topology and then clipping fan order.
    """
    near = float(near_plane_m)
    tolerance = float(facing_tolerance)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_plane_m must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("facing_tolerance must be finite and non-negative")

    prepared: list[CameraFacingBoxTriangle] = []
    for source in triangulate_box_surfaces(corners_camera):
        if not is_triangle_front_facing(
            source.vertices,
            tolerance=tolerance,
        ):
            continue
        clipped = clip_triangle_to_positive_z(
            source.vertices,
            near_plane_m=near,
        )
        for triangle in clipped:
            prepared.append(
                CameraFacingBoxTriangle(
                    face_name=source.face_name,
                    source_corner_indices=source.corner_indices,
                    vertices_camera=triangle,
                )
            )
    return tuple(prepared)
