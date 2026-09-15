"""Build complete Step 7E evidence with missing-surface projection context.

This composition layer reuses the complete current-geometry pipeline and then
attaches per-camera projection context for geometric candidates that have no
sampled surface. It does not recompute geometry, apply visibility thresholds,
or select final Actor visibility labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from step7.actor_geometric_occlusion_from_geometry_v01 import (
    ActorGeometricOcclusionFromGeometryResult,
    CameraGeometricOcclusionBuildInput,
    build_actor_geometric_occlusion_from_geometry,
)
from step7.actor_geometric_occlusion_projection_context_v01 import (
    ActorGeometricOcclusionProjectionContextResult,
    build_actor_geometric_occlusion_projection_context,
)


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionWithProjectionContextResult:
    actor_count: int
    pipeline: ActorGeometricOcclusionFromGeometryResult
    projection_context: ActorGeometricOcclusionProjectionContextResult


def build_actor_geometric_occlusion_with_projection_context(
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
) -> ActorGeometricOcclusionWithProjectionContextResult:
    """Run the existing pipeline and attach threshold-free projection context."""

    source = tuple(actors)
    pipeline = build_actor_geometric_occlusion_from_geometry(
        actors=source,
        recorded_ego_message=recorded_ego_message,
        cameras=cameras,
        raster_width=raster_width,
        raster_height=raster_height,
        maximum_depth=maximum_depth,
        maximum_boundary_extent_px=maximum_boundary_extent_px,
        samples_per_edge=samples_per_edge,
        near_plane_m=near_plane_m,
        maximum_chord_error_px=maximum_chord_error_px,
        maximum_adaptive_depth=maximum_adaptive_depth,
        depth_tolerance_m=depth_tolerance_m,
    )
    context = build_actor_geometric_occlusion_projection_context(
        geometric=pipeline.geometric,
        combined_evidence=(
            pipeline.combined.combined_evidence.actor_evidence
        ),
    )

    if pipeline.actor_count != len(source):
        raise RuntimeError("pipeline Actor count differs from input")
    if context.actor_count != pipeline.actor_count:
        raise RuntimeError(
            "projection-context Actor count differs from pipeline"
        )
    pipeline_ids = tuple(
        item.track_id
        for item in pipeline.combined.combined_evidence.actor_evidence
    )
    context_ids = tuple(item.track_id for item in context.actor_context)
    if context_ids != pipeline_ids:
        raise RuntimeError(
            "projection-context Actor IDs differ from combined evidence"
        )

    return ActorGeometricOcclusionWithProjectionContextResult(
        actor_count=pipeline.actor_count,
        pipeline=pipeline,
        projection_context=context,
    )
