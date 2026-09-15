"""Aggregate Actor occlusion resolution comparisons without thresholds.

This Step 7E module summarizes one camera/case comparison between a candidate
and reference raster. It preserves change counts and Actor IDs but does not
decide whether a raster is acceptable or select observability labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence

from step7.actor_occlusion_resolution_comparison_v01 import (
    ActorOcclusionResolutionComparison,
)


@dataclass(frozen=True, slots=True)
class OcclusionResolutionComparisonSummary:
    camera_name: str
    candidate_raster_width: int
    candidate_raster_height: int
    reference_raster_width: int
    reference_raster_height: int
    actor_count: int
    comparable_actor_count: int
    sampled_surface_presence_change_count: int
    winning_presence_change_count: int
    occluding_actor_set_change_count: int
    nonzero_visible_fraction_delta_count: int
    maximum_absolute_visible_fraction_delta: float | None
    median_absolute_visible_fraction_delta: float | None
    sampled_surface_presence_change_track_ids: tuple[str, ...]
    winning_presence_change_track_ids: tuple[str, ...]
    occluding_actor_set_change_track_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "sampled_surface_presence_change_track_ids",
            "winning_presence_change_track_ids",
            "occluding_actor_set_change_track_ids",
            "reasons",
        ):
            value[key] = list(value[key])
        return value


def summarize_occlusion_resolution_comparisons(
    comparisons: Sequence[ActorOcclusionResolutionComparison],
) -> OcclusionResolutionComparisonSummary:
    """Summarize a homogeneous set of per-Actor resolution comparisons."""

    source = tuple(comparisons)
    if not source:
        raise ValueError("comparisons must not be empty")

    first = source[0]
    identity = (
        first.camera_name,
        first.candidate_raster_width,
        first.candidate_raster_height,
        first.reference_raster_width,
        first.reference_raster_height,
    )
    for item in source[1:]:
        current = (
            item.camera_name,
            item.candidate_raster_width,
            item.candidate_raster_height,
            item.reference_raster_width,
            item.reference_raster_height,
        )
        if current != identity:
            raise ValueError(
                "all comparisons must use the same camera and raster pair"
            )

    track_ids = tuple(item.track_id for item in source)
    if any(not track_id for track_id in track_ids):
        raise ValueError("comparison track_id values must be non-empty")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("comparison track_id values must be unique")

    comparable = tuple(item for item in source if item.both_evaluated)
    surface_changes = tuple(
        sorted(
            item.track_id
            for item in source
            if item.sampled_surface_presence_changed
        )
    )
    winner_changes = tuple(
        sorted(
            item.track_id
            for item in comparable
            if item.winning_presence_changed
        )
    )
    occluder_changes = tuple(
        sorted(
            item.track_id
            for item in comparable
            if item.occluding_actor_set_changed
        )
    )
    deltas = tuple(
        item.absolute_visible_fraction_delta
        for item in comparable
    )
    nonzero_delta_count = sum(value > 0.0 for value in deltas)

    reasons = []
    if len(comparable) != len(source):
        reasons.append("one_or_more_actors_not_comparable")

    return OcclusionResolutionComparisonSummary(
        camera_name=first.camera_name,
        candidate_raster_width=first.candidate_raster_width,
        candidate_raster_height=first.candidate_raster_height,
        reference_raster_width=first.reference_raster_width,
        reference_raster_height=first.reference_raster_height,
        actor_count=len(source),
        comparable_actor_count=len(comparable),
        sampled_surface_presence_change_count=len(surface_changes),
        winning_presence_change_count=len(winner_changes),
        occluding_actor_set_change_count=len(occluder_changes),
        nonzero_visible_fraction_delta_count=nonzero_delta_count,
        maximum_absolute_visible_fraction_delta=(
            max(deltas) if deltas else None
        ),
        median_absolute_visible_fraction_delta=(
            median(deltas) if deltas else None
        ),
        sampled_surface_presence_change_track_ids=surface_changes,
        winning_presence_change_track_ids=winner_changes,
        occluding_actor_set_change_track_ids=occluder_changes,
        reasons=tuple(reasons),
    )
