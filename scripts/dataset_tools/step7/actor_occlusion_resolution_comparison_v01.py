"""Compare Actor occlusion evidence across two raster resolutions.

This Step 7E module records threshold-free stability diagnostics between one
candidate raster and one reference raster for the same Actor and camera. It
does not decide whether a resolution is acceptable, select observability
labels, or define visible-fraction and winning-cell thresholds.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from step7.actor_occlusion_evidence_v01 import CameraActorOcclusionEvidence


@dataclass(frozen=True, slots=True)
class ActorOcclusionResolutionComparison:
    camera_name: str
    track_id: str
    candidate_raster_width: int
    candidate_raster_height: int
    reference_raster_width: int
    reference_raster_height: int
    candidate_evidence_status: str
    reference_evidence_status: str
    both_evaluated: bool
    sampled_surface_presence_changed: bool
    winning_presence_changed: bool | None
    candidate_occupied_cell_count: int
    reference_occupied_cell_count: int
    candidate_winning_cell_count: int
    reference_winning_cell_count: int
    candidate_visible_fraction: float | None
    reference_visible_fraction: float | None
    absolute_visible_fraction_delta: float | None
    candidate_occluding_actor_ids: tuple[str, ...]
    reference_occluding_actor_ids: tuple[str, ...]
    occluding_actor_set_changed: bool | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["candidate_occluding_actor_ids"] = list(
            self.candidate_occluding_actor_ids
        )
        value["reference_occluding_actor_ids"] = list(
            self.reference_occluding_actor_ids
        )
        value["reasons"] = list(self.reasons)
        return value


def compare_actor_occlusion_resolutions(
    *,
    candidate: CameraActorOcclusionEvidence,
    reference: CameraActorOcclusionEvidence,
) -> ActorOcclusionResolutionComparison:
    """Compare two validated evidence records for the same Actor and camera."""

    if candidate.camera_name != reference.camera_name:
        raise ValueError("candidate and reference camera_name values must match")
    if candidate.track_id != reference.track_id:
        raise ValueError("candidate and reference track_id values must match")
    if candidate.static_occlusion_evaluated or reference.static_occlusion_evaluated:
        raise ValueError("static occlusion must remain unevaluated in Step 7E v0.1")

    candidate_evaluated = candidate.evidence_status == "evaluated"
    reference_evaluated = reference.evidence_status == "evaluated"
    both_evaluated = candidate_evaluated and reference_evaluated

    candidate_has_surface = candidate.occupied_cell_count > 0
    reference_has_surface = reference.occupied_cell_count > 0
    sampled_surface_presence_changed = (
        candidate_has_surface != reference_has_surface
    )

    reasons = []
    if not candidate.actor_to_actor_occlusion_evaluated:
        reasons.append("candidate_occlusion_not_evaluated")
    if not reference.actor_to_actor_occlusion_evaluated:
        reasons.append("reference_occlusion_not_evaluated")
    if not both_evaluated:
        reasons.append("visible_fraction_not_comparable")

    if both_evaluated:
        candidate_has_winner = candidate.winning_cell_count > 0
        reference_has_winner = reference.winning_cell_count > 0
        winning_presence_changed = (
            candidate_has_winner != reference_has_winner
        )
        absolute_delta = abs(
            candidate.visible_fraction - reference.visible_fraction
        )
        if not math.isfinite(absolute_delta):
            raise ValueError("visible-fraction delta must be finite")
        occluder_changed = (
            candidate.occluding_actor_ids
            != reference.occluding_actor_ids
        )
    else:
        winning_presence_changed = None
        absolute_delta = None
        occluder_changed = None

    return ActorOcclusionResolutionComparison(
        camera_name=candidate.camera_name,
        track_id=candidate.track_id,
        candidate_raster_width=candidate.raster_width,
        candidate_raster_height=candidate.raster_height,
        reference_raster_width=reference.raster_width,
        reference_raster_height=reference.raster_height,
        candidate_evidence_status=candidate.evidence_status,
        reference_evidence_status=reference.evidence_status,
        both_evaluated=both_evaluated,
        sampled_surface_presence_changed=sampled_surface_presence_changed,
        winning_presence_changed=winning_presence_changed,
        candidate_occupied_cell_count=candidate.occupied_cell_count,
        reference_occupied_cell_count=reference.occupied_cell_count,
        candidate_winning_cell_count=candidate.winning_cell_count,
        reference_winning_cell_count=reference.winning_cell_count,
        candidate_visible_fraction=candidate.visible_fraction,
        reference_visible_fraction=reference.visible_fraction,
        absolute_visible_fraction_delta=absolute_delta,
        candidate_occluding_actor_ids=candidate.occluding_actor_ids,
        reference_occluding_actor_ids=reference.occluding_actor_ids,
        occluding_actor_set_changed=occluder_changed,
        reasons=tuple(reasons),
    )
