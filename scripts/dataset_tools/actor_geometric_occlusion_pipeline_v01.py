"""Compose complete Step 7E geometric and occlusion evidence sets.

This module joins an existing complete ActorObservability set with one complete
multicamera current-geometry occlusion result. It preserves both upstream
results and produces deterministic combined evidence for every Actor. It does
not select final observability labels or apply occlusion thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from actor_geometric_occlusion_evidence_set_v01 import (
    ActorGeometricOcclusionEvidenceSet,
    build_actor_geometric_occlusion_evidence_set,
)
from actor_geometric_occlusion_evidence_summary_v01 import (
    ActorGeometricOcclusionEvidenceSummary,
    summarize_actor_geometric_occlusion_evidence,
)
from actor_observability_v01 import ActorObservability
from multicamera_actor_occlusion_from_geometry_v01 import (
    MulticameraActorOcclusionResult,
)


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionPipelineResult:
    actor_count: int
    geometric_observability: tuple[ActorObservability, ...]
    multicamera_occlusion: MulticameraActorOcclusionResult
    combined_evidence: ActorGeometricOcclusionEvidenceSet
    combined_summary: ActorGeometricOcclusionEvidenceSummary


def build_actor_geometric_occlusion_pipeline(
    *,
    geometric_observability: Sequence[ActorObservability],
    multicamera_occlusion: MulticameraActorOcclusionResult,
) -> ActorGeometricOcclusionPipelineResult:
    """Join complete upstream geometric and multicamera occlusion results."""

    geometric = tuple(geometric_observability)
    if multicamera_occlusion.actor_count != len(
        multicamera_occlusion.actor_summaries
    ):
        raise ValueError(
            "multicamera occlusion actor_count and summaries are inconsistent"
        )
    if multicamera_occlusion.actor_count != len(
        multicamera_occlusion.actor_evidence
    ):
        raise ValueError(
            "multicamera occlusion actor_count and evidence are inconsistent"
        )

    combined = build_actor_geometric_occlusion_evidence_set(
        geometric_observability=geometric,
        occlusion_summaries=multicamera_occlusion.actor_summaries,
    )
    if combined.actor_count != multicamera_occlusion.actor_count:
        raise ValueError(
            "geometric and multicamera occlusion Actor counts must match"
        )

    summary = summarize_actor_geometric_occlusion_evidence(
        combined.actor_evidence
    )
    if summary.actor_count != combined.actor_count:
        raise RuntimeError(
            "combined evidence summary Actor count is inconsistent"
        )

    geometric_ids = tuple(sorted(item.track_id for item in geometric))
    combined_ids = tuple(item.track_id for item in combined.actor_evidence)
    summary_ids = tuple(
        item.track_id for item in multicamera_occlusion.actor_summaries
    )
    if combined_ids != geometric_ids or combined_ids != summary_ids:
        raise RuntimeError(
            "pipeline geometric, occlusion, and combined Actor IDs differ"
        )

    return ActorGeometricOcclusionPipelineResult(
        actor_count=combined.actor_count,
        geometric_observability=tuple(
            sorted(geometric, key=lambda item: item.track_id)
        ),
        multicamera_occlusion=multicamera_occlusion,
        combined_evidence=combined,
        combined_summary=summary,
    )
