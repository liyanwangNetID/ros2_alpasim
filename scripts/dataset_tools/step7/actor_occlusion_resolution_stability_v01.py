"""Derive threshold-free resolution-stability evidence for one Actor.

This Step 7E module converts an Actor occlusion resolution comparison into an
explicit stability record. It only classifies discrete evidence changes such
as sampled-surface presence, winning presence, and occluder-set changes. It
does not apply a visible-fraction tolerance, choose a production raster, or
select a final observability label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from step7.actor_occlusion_resolution_comparison_v01 import (
    ActorOcclusionResolutionComparison,
)


@dataclass(frozen=True, slots=True)
class ActorOcclusionResolutionStabilityEvidence:
    camera_name: str
    track_id: str
    candidate_raster_width: int
    candidate_raster_height: int
    reference_raster_width: int
    reference_raster_height: int
    stability_status: str
    comparable: bool
    sampled_surface_presence_stable: bool | None
    winning_presence_stable: bool | None
    occluding_actor_set_stable: bool | None
    absolute_visible_fraction_delta: float | None
    candidate_winning_cell_count: int
    reference_winning_cell_count: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def build_actor_occlusion_resolution_stability_evidence(
    comparison: ActorOcclusionResolutionComparison,
) -> ActorOcclusionResolutionStabilityEvidence:
    """Build a discrete, threshold-free stability record."""

    reasons = []
    if not comparison.both_evaluated:
        status = "not_comparable"
        surface_stable = None
        winning_stable = None
        occluder_stable = None
        reasons.append("occlusion_evidence_not_comparable")
    else:
        surface_stable = not comparison.sampled_surface_presence_changed
        winning_stable = not comparison.winning_presence_changed
        occluder_stable = not comparison.occluding_actor_set_changed

        if not surface_stable:
            status = "sampled_surface_presence_changed"
            reasons.append("sampled_surface_presence_changed")
        elif not winning_stable:
            status = "winning_presence_changed"
            reasons.append("winning_presence_changed")
        elif not occluder_stable:
            status = "occluding_actor_set_changed"
            reasons.append("occluding_actor_set_changed")
        else:
            status = "discrete_evidence_stable"

    return ActorOcclusionResolutionStabilityEvidence(
        camera_name=comparison.camera_name,
        track_id=comparison.track_id,
        candidate_raster_width=comparison.candidate_raster_width,
        candidate_raster_height=comparison.candidate_raster_height,
        reference_raster_width=comparison.reference_raster_width,
        reference_raster_height=comparison.reference_raster_height,
        stability_status=status,
        comparable=comparison.both_evaluated,
        sampled_surface_presence_stable=surface_stable,
        winning_presence_stable=winning_stable,
        occluding_actor_set_stable=occluder_stable,
        absolute_visible_fraction_delta=(
            comparison.absolute_visible_fraction_delta
        ),
        candidate_winning_cell_count=(
            comparison.candidate_winning_cell_count
        ),
        reference_winning_cell_count=(
            comparison.reference_winning_cell_count
        ),
        reasons=tuple(reasons),
    )
