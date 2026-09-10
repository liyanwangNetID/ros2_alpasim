"""Summarize combined geometric and Actor-occlusion evidence for Step 7E.

This module aggregates a complete set of threshold-free combined Actor evidence
records. It reports evidence availability and camera-set relationships without
selecting final observability labels or applying visibility thresholds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionEvidenceSummary:
    actor_count: int
    evidence_status_counts: tuple[tuple[str, int], ...]
    geometric_candidate_actor_count: int
    geometric_candidate_with_sampled_surface_actor_count: int
    geometric_candidate_with_winning_cells_actor_count: int
    geometric_candidate_without_sampled_surface_actor_count: int
    geometric_candidate_fully_occluded_actor_count: int
    actor_with_occluder_count: int
    actor_to_actor_occlusion_complete_count: int
    static_occlusion_evaluated_count: int
    geometric_candidate_without_sampled_surface_track_ids: tuple[str, ...]
    geometric_candidate_fully_occluded_track_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["evidence_status_counts"] = {
            key: count for key, count in self.evidence_status_counts
        }
        for key in (
            "geometric_candidate_without_sampled_surface_track_ids",
            "geometric_candidate_fully_occluded_track_ids",
            "reasons",
        ):
            value[key] = list(value[key])
        return value


def summarize_actor_geometric_occlusion_evidence(
    evidence: Sequence[ActorGeometricOcclusionEvidence],
) -> ActorGeometricOcclusionEvidenceSummary:
    """Aggregate a deterministic set of combined evidence records."""

    source = tuple(evidence)
    track_ids = tuple(item.track_id for item in source)
    if any(not track_id for track_id in track_ids):
        raise ValueError("track_id values must be non-empty")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track_id values must be unique")

    status_counts = Counter(item.evidence_status for item in source)
    candidate = tuple(
        item for item in source if item.geometric_candidate_camera_names
    )
    candidate_with_surface = tuple(
        item
        for item in source
        if item.geometric_candidate_with_sampled_surface_camera_names
    )
    candidate_with_winners = tuple(
        item
        for item in source
        if item.geometric_candidate_with_winning_cells_camera_names
    )
    candidate_without_surface = tuple(
        item
        for item in source
        if item.geometric_candidate_without_sampled_surface_camera_names
    )
    candidate_fully_occluded = tuple(
        item
        for item in source
        if item.geometric_candidate_fully_occluded_camera_names
    )

    reasons = []
    if any(not item.actor_to_actor_occlusion_evaluated for item in source):
        reasons.append("one_or_more_actors_have_incomplete_occlusion_evidence")
    if any(item.static_occlusion_evaluated for item in source):
        reasons.append("one_or_more_actors_have_static_occlusion_evidence")

    return ActorGeometricOcclusionEvidenceSummary(
        actor_count=len(source),
        evidence_status_counts=tuple(sorted(status_counts.items())),
        geometric_candidate_actor_count=len(candidate),
        geometric_candidate_with_sampled_surface_actor_count=len(
            candidate_with_surface
        ),
        geometric_candidate_with_winning_cells_actor_count=len(
            candidate_with_winners
        ),
        geometric_candidate_without_sampled_surface_actor_count=len(
            candidate_without_surface
        ),
        geometric_candidate_fully_occluded_actor_count=len(
            candidate_fully_occluded
        ),
        actor_with_occluder_count=sum(
            bool(item.occluding_actor_ids) for item in source
        ),
        actor_to_actor_occlusion_complete_count=sum(
            item.actor_to_actor_occlusion_evaluated for item in source
        ),
        static_occlusion_evaluated_count=sum(
            item.static_occlusion_evaluated for item in source
        ),
        geometric_candidate_without_sampled_surface_track_ids=tuple(
            sorted(item.track_id for item in candidate_without_surface)
        ),
        geometric_candidate_fully_occluded_track_ids=tuple(
            sorted(item.track_id for item in candidate_fully_occluded)
        ),
        reasons=tuple(reasons),
    )
