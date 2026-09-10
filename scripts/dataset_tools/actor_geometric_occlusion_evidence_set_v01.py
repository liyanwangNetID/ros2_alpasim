"""Join geometric observability and occlusion summaries for all Actors.

This Step 7E composition module matches existing ActorObservability records with
threshold-free multicamera Actor occlusion summaries by track_id. It returns
one combined evidence record per Actor in deterministic order. It does not
select final observability labels or apply visibility thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
    build_actor_geometric_occlusion_evidence,
)
from actor_multicamera_occlusion_summary_v01 import (
    ActorMulticameraOcclusionSummary,
)
from actor_observability_v01 import ActorObservability


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionEvidenceSet:
    actor_count: int
    actor_evidence: tuple[ActorGeometricOcclusionEvidence, ...]


def build_actor_geometric_occlusion_evidence_set(
    *,
    geometric_observability: Sequence[ActorObservability],
    occlusion_summaries: Sequence[ActorMulticameraOcclusionSummary],
) -> ActorGeometricOcclusionEvidenceSet:
    """Join complete geometric and occlusion Actor sets by track_id."""

    geometric = tuple(geometric_observability)
    occlusion = tuple(occlusion_summaries)

    geometric_by_id = {item.track_id: item for item in geometric}
    occlusion_by_id = {item.track_id: item for item in occlusion}
    if len(geometric_by_id) != len(geometric):
        raise ValueError(
            "geometric observability contains duplicate track_id values"
        )
    if len(occlusion_by_id) != len(occlusion):
        raise ValueError(
            "occlusion summaries contain duplicate track_id values"
        )
    if set(geometric_by_id) != set(occlusion_by_id):
        missing_geometric = sorted(
            set(occlusion_by_id) - set(geometric_by_id)
        )
        missing_occlusion = sorted(
            set(geometric_by_id) - set(occlusion_by_id)
        )
        raise ValueError(
            "geometric and occlusion Actor sets must match; "
            f"missing_geometric={missing_geometric}, "
            f"missing_occlusion={missing_occlusion}"
        )

    evidence = tuple(
        build_actor_geometric_occlusion_evidence(
            geometric=geometric_by_id[track_id],
            occlusion=occlusion_by_id[track_id],
        )
        for track_id in sorted(geometric_by_id)
    )

    if tuple(item.track_id for item in evidence) != tuple(
        sorted(geometric_by_id)
    ):
        raise RuntimeError(
            "combined geometric-occlusion evidence order is inconsistent"
        )

    return ActorGeometricOcclusionEvidenceSet(
        actor_count=len(evidence),
        actor_evidence=evidence,
    )
