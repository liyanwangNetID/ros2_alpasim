"""Build per-camera Actor occlusion evidence from current Actor geometry.

This Step 7E composition module converts current Actor boxes into prepared
camera-facing surface triangles, builds one surface-depth raster per Actor,
resolves their shared Z-buffer, and returns threshold-free occlusion evidence.
It does not select observability labels, aggregate cameras, use future Actor
state, evaluate static-scene occlusion, or freeze production thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from step7.actor_box_projection_v01 import actor_box_corners_in_rig
from step7.actor_camera_occlusion_pipeline_v01 import (
    CameraActorOcclusionPipelineResult,
    build_camera_actor_occlusion_pipeline,
)
from step7.actor_camera_surface_depth_raster_v01 import (
    build_actor_camera_surface_depth_raster,
)
from step7.actor_depth_zbuffer_v01 import ActorDepthRasterInput
from step7.camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles


@dataclass(frozen=True, slots=True)
class ActorCameraSurfaceDiagnostic:
    track_id: str
    prepared_source_triangle_count: int
    generated_triangle_sample_count: int
    occupied_cell_count: int
    zero_center_sample_triangle_count: int
    unresolved_boundary_count: int


@dataclass(frozen=True, slots=True)
class ActorCameraOcclusionFromGeometryResult:
    camera_name: str
    raster_width: int
    raster_height: int
    actor_count: int
    surface_diagnostics: tuple[ActorCameraSurfaceDiagnostic, ...]
    occlusion: CameraActorOcclusionPipelineResult


def _actor_id(actor: dict[str, Any]) -> str:
    value = actor.get("track_id")
    if value is None or str(value) == "":
        raise ValueError("Actor is missing a usable track_id")
    return str(value)


def build_actor_camera_occlusion_from_geometry(
    *,
    actors: Sequence[dict[str, Any]],
    recorded_ego_message: dict[str, Any],
    calibration: Any,
    camera_name: str,
    image_width_px: int,
    image_height_px: int,
    raster_width: int,
    raster_height: int,
    maximum_depth: int,
    maximum_boundary_extent_px: float,
    near_plane_m: float = 1e-3,
    depth_tolerance_m: float = 1e-9,
) -> ActorCameraOcclusionFromGeometryResult:
    """Build all current-Actor camera rasters and shared occlusion evidence."""

    source = tuple(actors)
    track_ids = tuple(_actor_id(actor) for actor in source)
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Actor track_id values must be unique")
    if getattr(calibration, "max_angle_rad", None) is None:
        raise ValueError("calibration.max_angle_rad is required")

    inputs: list[ActorDepthRasterInput] = []
    diagnostics: list[ActorCameraSurfaceDiagnostic] = []

    for actor in sorted(source, key=_actor_id):
        track_id = _actor_id(actor)
        corners_rig = actor_box_corners_in_rig(
            actor,
            recorded_ego_message=recorded_ego_message,
        )
        corners_camera = tuple(
            calibration.rig_point_to_camera(point)
            for point in corners_rig
        )
        prepared = prepare_camera_facing_box_triangles(
            corners_camera,
            near_plane_m=near_plane_m,
        )
        result = build_actor_camera_surface_depth_raster(
                tuple(item.vertices_camera for item in prepared),
                calibration,
                max_angle_rad=calibration.max_angle_rad,
                maximum_depth=maximum_depth,
                maximum_boundary_extent_px=maximum_boundary_extent_px,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                raster_width=raster_width,
                raster_height=raster_height,
                near_plane_m=near_plane_m,
                depth_tolerance_m=depth_tolerance_m,
            )
        inputs.append(
            ActorDepthRasterInput(track_id, result.surface_raster)
        )
        diagnostics.append(
            ActorCameraSurfaceDiagnostic(
                track_id=track_id,
                prepared_source_triangle_count=len(prepared),
                generated_triangle_sample_count=(
                    result.generated_triangle_sample_count
                ),
                occupied_cell_count=(
                    result.surface_raster.occupied_cell_count
                ),
                zero_center_sample_triangle_count=(
                    result.zero_center_sample_triangle_count
                ),
                unresolved_boundary_count=(
                    result.unresolved_boundary_count
                ),
            )
        )

    occlusion = build_camera_actor_occlusion_pipeline(
        camera_name=camera_name,
        raster_width=raster_width,
        raster_height=raster_height,
        actor_rasters=tuple(inputs),
        depth_tolerance_m=depth_tolerance_m,
        static_occlusion_evaluated=False,
    )

    evidence_ids = tuple(
        item.track_id for item in occlusion.actor_evidence
    )
    diagnostic_ids = tuple(item.track_id for item in diagnostics)
    if evidence_ids != diagnostic_ids:
        raise RuntimeError(
            "surface diagnostics and occlusion evidence Actor IDs differ"
        )

    return ActorCameraOcclusionFromGeometryResult(
        camera_name=camera_name,
        raster_width=raster_width,
        raster_height=raster_height,
        actor_count=len(source),
        surface_diagnostics=tuple(diagnostics),
        occlusion=occlusion,
    )
