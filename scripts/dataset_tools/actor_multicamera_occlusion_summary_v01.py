"""Summarize four-camera Actor occlusion evidence without label thresholds.

This Step 7E module derives deterministic cross-camera diagnostic features from
per-camera Z-buffer evidence. It does not select observability labels, apply
minimum winning-cell or visible-fraction thresholds, evaluate static-scene
occlusion, or merge camera images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from actor_occlusion_evidence_v01 import CameraActorOcclusionEvidence
from scene_fact_schema_v01 import CAMERA_NAMES


@dataclass(frozen=True, slots=True)
class ActorMulticameraOcclusionSummary:
    track_id: str
    evaluated_camera_names: tuple[str, ...]
    no_sampled_surface_camera_names: tuple[str, ...]
    winning_camera_names: tuple[str, ...]
    fully_occluded_camera_names: tuple[str, ...]
    occluded_camera_names: tuple[str, ...]
    evaluated_camera_count: int
    winning_camera_count: int
    total_occupied_cell_count: int
    total_winning_cell_count: int
    total_occluded_cell_count: int
    maximum_visible_fraction: float | None
    occluding_actor_ids: tuple[str, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "evaluated_camera_names",
            "no_sampled_surface_camera_names",
            "winning_camera_names",
            "fully_occluded_camera_names",
            "occluded_camera_names",
            "occluding_actor_ids",
            "reasons",
        ):
            value[key] = list(value[key])
        return value


def summarize_actor_multicamera_occlusion(
    *,
    track_id: str,
    camera_evidence: Sequence[CameraActorOcclusionEvidence],
) -> ActorMulticameraOcclusionSummary:
    """Derive threshold-free cross-camera features for one Actor."""

    if not isinstance(track_id, str) or not track_id:
        raise ValueError("track_id must be a non-empty string")

    source = tuple(camera_evidence)
    expected_order = tuple(CAMERA_NAMES)
    actual_order = tuple(item.camera_name for item in source)
    if actual_order != expected_order:
        raise ValueError(
            f"camera evidence must follow CAMERA_NAMES; actual={actual_order}"
        )
    if any(item.track_id != track_id for item in source):
        raise ValueError("camera evidence track_id values must match track_id")
    if any(item.static_occlusion_evaluated for item in source):
        raise ValueError("static occlusion must remain unevaluated in Step 7E v0.1")

    allowed_statuses = {"evaluated", "no_sampled_surface", "not_evaluated"}
    if any(item.evidence_status not in allowed_statuses for item in source):
        raise ValueError("unexpected camera occlusion evidence status")

    evaluated = tuple(item for item in source if item.evidence_status == "evaluated")
    no_surface = tuple(
        item for item in source if item.evidence_status == "no_sampled_surface"
    )
    not_evaluated = tuple(
        item for item in source if item.evidence_status == "not_evaluated"
    )

    winning = tuple(item for item in evaluated if item.winning_cell_count > 0)
    fully_occluded = tuple(
        item for item in evaluated if item.winning_cell_count == 0
    )
    occluded = tuple(
        item for item in evaluated if item.occluded_cell_count > 0
    )

    reasons = []
    if not_evaluated:
        reasons.append("one_or_more_cameras_not_evaluated")
    if not evaluated:
        reasons.append("no_camera_has_sampled_surface")

    occluders = tuple(
        sorted(
            {
                actor_id
                for item in evaluated
                for actor_id in item.occluding_actor_ids
            }
        )
    )
    maximum_visible_fraction = (
        max(item.visible_fraction for item in evaluated)
        if evaluated
        else None
    )

    return ActorMulticameraOcclusionSummary(
        track_id=track_id,
        evaluated_camera_names=tuple(item.camera_name for item in evaluated),
        no_sampled_surface_camera_names=tuple(
            item.camera_name for item in no_surface
        ),
        winning_camera_names=tuple(item.camera_name for item in winning),
        fully_occluded_camera_names=tuple(
            item.camera_name for item in fully_occluded
        ),
        occluded_camera_names=tuple(item.camera_name for item in occluded),
        evaluated_camera_count=len(evaluated),
        winning_camera_count=len(winning),
        total_occupied_cell_count=sum(
            item.occupied_cell_count for item in evaluated
        ),
        total_winning_cell_count=sum(
            item.winning_cell_count for item in evaluated
        ),
        total_occluded_cell_count=sum(
            item.occluded_cell_count for item in evaluated
        ),
        maximum_visible_fraction=maximum_visible_fraction,
        occluding_actor_ids=occluders,
        actor_to_actor_occlusion_evaluated=(
            len(not_evaluated) == 0
        ),
        static_occlusion_evaluated=False,
        reasons=tuple(reasons),
    )
