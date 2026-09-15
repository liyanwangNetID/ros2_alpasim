"""Build a threshold-free Actor occlusion resolution profile.

This Step 7E composition module compares complete per-camera Actor evidence sets
between one candidate raster and one reference raster, then returns both the
per-Actor comparisons and their case-level summary. It does not decide whether
a raster is acceptable or select observability labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from step7.actor_occlusion_evidence_v01 import CameraActorOcclusionEvidence
from step7.actor_occlusion_resolution_comparison_v01 import (
    ActorOcclusionResolutionComparison,
    compare_actor_occlusion_resolutions,
)
from step7.actor_occlusion_resolution_stability_v01 import (
    ActorOcclusionResolutionStabilityEvidence,
    build_actor_occlusion_resolution_stability_evidence,
)
from step7.occlusion_resolution_comparison_summary_v01 import (
    OcclusionResolutionComparisonSummary,
    summarize_occlusion_resolution_comparisons,
)


@dataclass(frozen=True, slots=True)
class OcclusionResolutionProfile:
    comparisons: tuple[ActorOcclusionResolutionComparison, ...]
    stability_evidence: tuple[
        ActorOcclusionResolutionStabilityEvidence, ...
    ]
    summary: OcclusionResolutionComparisonSummary


def build_occlusion_resolution_profile(
    *,
    candidate_evidence: Sequence[CameraActorOcclusionEvidence],
    reference_evidence: Sequence[CameraActorOcclusionEvidence],
) -> OcclusionResolutionProfile:
    """Compare two complete Actor evidence sets in deterministic track order."""

    candidate = tuple(candidate_evidence)
    reference = tuple(reference_evidence)

    candidate_by_id = {item.track_id: item for item in candidate}
    reference_by_id = {item.track_id: item for item in reference}
    if len(candidate_by_id) != len(candidate):
        raise ValueError("candidate evidence contains duplicate track_id values")
    if len(reference_by_id) != len(reference):
        raise ValueError("reference evidence contains duplicate track_id values")
    if set(candidate_by_id) != set(reference_by_id):
        missing_from_candidate = sorted(set(reference_by_id) - set(candidate_by_id))
        missing_from_reference = sorted(set(candidate_by_id) - set(reference_by_id))
        raise ValueError(
            "candidate and reference Actor sets must match; "
            f"missing_from_candidate={missing_from_candidate}, "
            f"missing_from_reference={missing_from_reference}"
        )
    if not candidate_by_id:
        raise ValueError("Actor evidence sets must not be empty")

    comparisons = tuple(
        compare_actor_occlusion_resolutions(
            candidate=candidate_by_id[track_id],
            reference=reference_by_id[track_id],
        )
        for track_id in sorted(candidate_by_id)
    )
    stability_evidence = tuple(
        build_actor_occlusion_resolution_stability_evidence(item)
        for item in comparisons
    )
    summary = summarize_occlusion_resolution_comparisons(comparisons)

    if summary.actor_count != len(comparisons):
        raise RuntimeError("resolution profile summary count is inconsistent")
    comparison_ids = tuple(item.track_id for item in comparisons)
    stability_ids = tuple(item.track_id for item in stability_evidence)
    if stability_ids != comparison_ids:
        raise RuntimeError(
            "resolution stability evidence Actor IDs are inconsistent"
        )

    return OcclusionResolutionProfile(
        comparisons=comparisons,
        stability_evidence=stability_evidence,
        summary=summary,
    )
