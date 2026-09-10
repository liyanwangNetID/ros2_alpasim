"""Build complete Step 7E geometric and occlusion evidence from geometry.

This composition module runs four-camera geometric projection and four-camera
Actor-to-Actor occlusion from the same current Actor and Ego inputs, then joins
both results into threshold-free combined evidence. It does not select final
observability labels or evaluate static-scene occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from actor_geometric_occlusion_pipeline_v01 import (
    ActorGeometricOcclusionPipelineResult,
    build_actor_geometric_occlusion_pipeline,
)
from multicamera_actor_geometric_observability_v01 import (
    MulticameraActorGeometricObservabilityResult,
    build_multicamera_actor_geometric_observability,
)
from multicamera_actor_occlusion_from_geometry_v01 import (
    CameraOcclusionBuildInput,
    MulticameraActorOcclusionResult,
    build_multicamera_actor_occlusion_from_geometry,
)
from scene_fact_schema_v01 import CAMERA_NAMES


@dataclass(frozen=True, slots=True)
class CameraGeometricOcclusionBuildInput:
    calibration: Any
    image_width_px: int
    image_height_px: int


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionFromGeometryResult:
    actor_count: int
    geometric: MulticameraActorGeometricObservabilityResult
    occlusion: MulticameraActorOcclusionResult
    combined: ActorGeometricOcclusionPipelineResult


def build_actor_geometric_occlusion_from_geometry(
    *,
    actors: Sequence[Mapping[str, Any]],
    recorded_ego_message: Mapping[str, Any],
    cameras: Mapping[str, CameraGeometricOcclusionBuildInput],
    raster_width: int,
    raster_height: int,
    maximum_depth: int,
    maximum_boundary_extent_px: float,
    samples_per_edge: int | None = None,
    near_plane_m: float = 1e-3,
    maximum_chord_error_px: float = 1.0,
    maximum_adaptive_depth: int = 14,
    depth_tolerance_m: float = 1e-9,
) -> ActorGeometricOcclusionFromGeometryResult:
    """Run and join geometric and Actor-occlusion evidence from one snapshot."""

    expected = set(CAMERA_NAMES)
    actual = set(cameras)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}"
        )

    source = tuple(actors)
    calibrations = {
        name: cameras[name].calibration for name in CAMERA_NAMES
    }
    occlusion_cameras = {
        name: CameraOcclusionBuildInput(
            calibration=cameras[name].calibration,
            image_width_px=cameras[name].image_width_px,
            image_height_px=cameras[name].image_height_px,
        )
        for name in CAMERA_NAMES
    }

    geometric = build_multicamera_actor_geometric_observability(
        actors=source,
        recorded_ego_message=recorded_ego_message,
        calibrations=calibrations,
        samples_per_edge=samples_per_edge,
        near_plane_m=near_plane_m,
        maximum_chord_error_px=maximum_chord_error_px,
        maximum_adaptive_depth=maximum_adaptive_depth,
    )
    occlusion = build_multicamera_actor_occlusion_from_geometry(
        actors=source,
        recorded_ego_message=recorded_ego_message,
        cameras=occlusion_cameras,
        raster_width=raster_width,
        raster_height=raster_height,
        maximum_depth=maximum_depth,
        maximum_boundary_extent_px=maximum_boundary_extent_px,
        near_plane_m=near_plane_m,
        depth_tolerance_m=depth_tolerance_m,
    )
    combined = build_actor_geometric_occlusion_pipeline(
        geometric_observability=geometric.actor_observability,
        multicamera_occlusion=occlusion,
    )

    counts = (geometric.actor_count, occlusion.actor_count, combined.actor_count)
    if counts != (len(source), len(source), len(source)):
        raise RuntimeError(
            "geometric, occlusion, and combined Actor counts are inconsistent"
        )

    geometric_ids = tuple(
        item.track_id for item in geometric.actor_observability
    )
    occlusion_ids = tuple(item.track_id for item in occlusion.actor_summaries)
    combined_ids = tuple(
        item.track_id for item in combined.combined_evidence.actor_evidence
    )
    if geometric_ids != occlusion_ids or geometric_ids != combined_ids:
        raise RuntimeError(
            "geometric, occlusion, and combined Actor IDs are inconsistent"
        )

    return ActorGeometricOcclusionFromGeometryResult(
        actor_count=len(source),
        geometric=geometric,
        occlusion=occlusion,
        combined=combined,
    )
