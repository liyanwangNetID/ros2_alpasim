"""Attach missing-surface projection context to complete Actor evidence.

This Step 7E adapter combines the existing multicamera geometric projection
result with the combined geometric-occlusion evidence set. It emits projection
context only for geometric candidate cameras without sampled surface evidence.
No geometry is recomputed and no final visibility label is selected.
"""

from __future__ import annotations

from dataclasses import dataclass

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from candidate_without_sampled_surface_projection_evidence_v01 import (
    CandidateWithoutSampledSurfaceProjectionEvidence,
    build_candidate_without_sampled_surface_projection_evidence,
)
from multicamera_actor_geometric_observability_v01 import (
    MulticameraActorGeometricObservabilityResult,
)


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionProjectionContext:
    track_id: str
    combined_evidence: ActorGeometricOcclusionEvidence
    candidate_without_sampled_surface_projection_evidence: tuple[
        CandidateWithoutSampledSurfaceProjectionEvidence, ...
    ]


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionProjectionContextResult:
    actor_count: int
    actor_context: tuple[ActorGeometricOcclusionProjectionContext, ...]
    missing_surface_projection_context_count: int


def build_actor_geometric_occlusion_projection_context(
    *,
    geometric: MulticameraActorGeometricObservabilityResult,
    combined_evidence: tuple[ActorGeometricOcclusionEvidence, ...],
) -> ActorGeometricOcclusionProjectionContextResult:
    """Join all Actor projections with combined evidence by track_id."""

    geometric_results = tuple(geometric.actor_results)
    geometric_by_id = {item.track_id: item for item in geometric_results}
    if len(geometric_by_id) != len(geometric_results):
        raise ValueError("geometric result contains duplicate track_id values")

    combined = tuple(combined_evidence)
    combined_by_id = {item.track_id: item for item in combined}
    if len(combined_by_id) != len(combined):
        raise ValueError("combined evidence contains duplicate track_id values")
    if geometric.actor_count != len(geometric_results):
        raise ValueError("geometric actor_count and actor_results are inconsistent")
    if set(geometric_by_id) != set(combined_by_id):
        missing_geometric = sorted(set(combined_by_id) - set(geometric_by_id))
        missing_combined = sorted(set(geometric_by_id) - set(combined_by_id))
        raise ValueError(
            "geometric and combined Actor sets must match; "
            f"missing_geometric={missing_geometric}, "
            f"missing_combined={missing_combined}"
        )

    contexts = []
    context_count = 0
    for track_id in sorted(combined_by_id):
        geometric_item = geometric_by_id[track_id]
        projections_by_camera = {
            projection.camera_name: projection
            for projection in geometric_item.projections
        }
        projection_context = (
            build_candidate_without_sampled_surface_projection_evidence(
                combined_evidence=combined_by_id[track_id],
                projections_by_camera=projections_by_camera,
            )
        )
        context_count += len(projection_context)
        contexts.append(
            ActorGeometricOcclusionProjectionContext(
                track_id=track_id,
                combined_evidence=combined_by_id[track_id],
                candidate_without_sampled_surface_projection_evidence=(
                    projection_context
                ),
            )
        )

    return ActorGeometricOcclusionProjectionContextResult(
        actor_count=len(contexts),
        actor_context=tuple(contexts),
        missing_surface_projection_context_count=context_count,
    )
