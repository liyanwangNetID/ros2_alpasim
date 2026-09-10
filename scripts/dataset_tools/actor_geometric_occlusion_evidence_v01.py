"""Combine geometric observability and Actor occlusion evidence for Step 7E.

This module joins the existing four-camera geometric candidate result with the
threshold-free multicamera Actor-to-Actor occlusion summary. It preserves both
sources and derives camera-set relationships without selecting a final
observability label or applying winning-cell and visible-fraction thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from actor_multicamera_occlusion_summary_v01 import (
    ActorMulticameraOcclusionSummary,
)
from actor_observability_v01 import ActorObservability
from scene_fact_schema_v01 import CAMERA_NAMES


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionEvidence:
    track_id: str
    actor_class: str
    geometric_observability_status: str
    geometric_candidate_camera_names: tuple[str, ...]
    occlusion_evaluated_camera_names: tuple[str, ...]
    occlusion_winning_camera_names: tuple[str, ...]
    geometric_candidate_with_sampled_surface_camera_names: tuple[str, ...]
    geometric_candidate_with_winning_cells_camera_names: tuple[str, ...]
    geometric_candidate_without_sampled_surface_camera_names: tuple[str, ...]
    geometric_candidate_fully_occluded_camera_names: tuple[str, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool
    maximum_visible_fraction: float | None
    total_occupied_cell_count: int
    total_winning_cell_count: int
    total_occluded_cell_count: int
    occluding_actor_ids: tuple[str, ...]
    evidence_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "geometric_candidate_camera_names",
            "occlusion_evaluated_camera_names",
            "occlusion_winning_camera_names",
            "geometric_candidate_with_sampled_surface_camera_names",
            "geometric_candidate_with_winning_cells_camera_names",
            "geometric_candidate_without_sampled_surface_camera_names",
            "geometric_candidate_fully_occluded_camera_names",
            "occluding_actor_ids",
            "reasons",
        ):
            value[key] = list(value[key])
        return value


def _ordered_intersection(first: tuple[str, ...], second: set[str]) -> tuple[str, ...]:
    return tuple(value for value in first if value in second)


def build_actor_geometric_occlusion_evidence(
    *,
    geometric: ActorObservability,
    occlusion: ActorMulticameraOcclusionSummary,
) -> ActorGeometricOcclusionEvidence:
    """Join geometric candidates with threshold-free occlusion diagnostics."""

    if geometric.track_id != occlusion.track_id:
        raise ValueError("geometric and occlusion track_id values must match")

    canonical = tuple(CAMERA_NAMES)
    geometric_candidates = tuple(geometric.visible_in_cameras)
    if any(name not in CAMERA_NAMES for name in geometric_candidates):
        raise ValueError("geometric candidate camera is not canonical")
    if geometric_candidates != tuple(
        name for name in canonical if name in set(geometric_candidates)
    ):
        raise ValueError("geometric candidate cameras must use canonical order")

    evaluated_set = set(occlusion.evaluated_camera_names)
    winning_set = set(occlusion.winning_camera_names)
    no_surface_set = set(occlusion.no_sampled_surface_camera_names)
    fully_occluded_set = set(occlusion.fully_occluded_camera_names)

    candidate_with_surface = _ordered_intersection(
        geometric_candidates,
        evaluated_set,
    )
    candidate_with_winners = _ordered_intersection(
        geometric_candidates,
        winning_set,
    )
    candidate_without_surface = _ordered_intersection(
        geometric_candidates,
        no_surface_set,
    )
    candidate_fully_occluded = _ordered_intersection(
        geometric_candidates,
        fully_occluded_set,
    )

    reasons = []
    if not geometric_candidates:
        reasons.append("no_geometric_candidate_camera")
    if not occlusion.actor_to_actor_occlusion_evaluated:
        reasons.append("actor_to_actor_occlusion_not_fully_evaluated")
    if geometric_candidates and not candidate_with_surface:
        reasons.append("no_geometric_candidate_has_sampled_surface")

    if not occlusion.actor_to_actor_occlusion_evaluated:
        evidence_status = "occlusion_incomplete"
    elif not geometric_candidates:
        evidence_status = "no_geometric_candidate"
    elif not candidate_with_surface:
        evidence_status = "candidate_without_sampled_surface"
    else:
        evidence_status = "combined_evidence_available"

    return ActorGeometricOcclusionEvidence(
        track_id=geometric.track_id,
        actor_class=geometric.actor_class,
        geometric_observability_status=geometric.observability_status,
        geometric_candidate_camera_names=geometric_candidates,
        occlusion_evaluated_camera_names=occlusion.evaluated_camera_names,
        occlusion_winning_camera_names=occlusion.winning_camera_names,
        geometric_candidate_with_sampled_surface_camera_names=(
            candidate_with_surface
        ),
        geometric_candidate_with_winning_cells_camera_names=(
            candidate_with_winners
        ),
        geometric_candidate_without_sampled_surface_camera_names=(
            candidate_without_surface
        ),
        geometric_candidate_fully_occluded_camera_names=(
            candidate_fully_occluded
        ),
        actor_to_actor_occlusion_evaluated=(
            occlusion.actor_to_actor_occlusion_evaluated
        ),
        static_occlusion_evaluated=occlusion.static_occlusion_evaluated,
        maximum_visible_fraction=occlusion.maximum_visible_fraction,
        total_occupied_cell_count=occlusion.total_occupied_cell_count,
        total_winning_cell_count=occlusion.total_winning_cell_count,
        total_occluded_cell_count=occlusion.total_occluded_cell_count,
        occluding_actor_ids=occlusion.occluding_actor_ids,
        evidence_status=evidence_status,
        reasons=tuple(reasons),
    )
