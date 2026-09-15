#!/usr/bin/env python3
"""Visibility fractions derived from multi-Actor Z-buffer summaries.

This module converts occupied, winning, and occluded raster-cell counts into
visibility statistics. Actors without sampled surface cells are represented
explicitly with undefined fractions rather than fabricated percentages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from step7.actor_depth_zbuffer_v01 import ActorZBufferSummary


@dataclass(frozen=True, slots=True)
class ActorVisibilityStatistic:
    actor_id: str
    occupied_cell_count: int
    winning_cell_count: int
    occluded_cell_count: int
    has_sampled_surface: bool
    visible_fraction: float | None
    occluded_fraction: float | None
    fully_visible: bool
    fully_occluded: bool


def calculate_actor_visibility_statistics(
    summaries: Sequence[ActorZBufferSummary],
) -> tuple[ActorVisibilityStatistic, ...]:
    """Validate Z-buffer counts and calculate deterministic visibility ratios."""
    source = tuple(summaries)
    actor_ids = tuple(item.actor_id for item in source)
    if any(not isinstance(actor_id, str) or not actor_id for actor_id in actor_ids):
        raise ValueError("actor_id must be a non-empty string")
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError("actor_id values must be unique")

    results: list[ActorVisibilityStatistic] = []
    for summary in source:
        counts = (
            summary.occupied_cell_count,
            summary.winning_cell_count,
            summary.occluded_cell_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("visibility counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("visibility counts must be non-negative")
        if summary.winning_cell_count + summary.occluded_cell_count != summary.occupied_cell_count:
            raise ValueError("winning and occluded counts must equal occupied count")

        has_surface = summary.occupied_cell_count > 0
        if has_surface:
            visible_fraction = (
                summary.winning_cell_count / summary.occupied_cell_count
            )
            occluded_fraction = (
                summary.occluded_cell_count / summary.occupied_cell_count
            )
            fully_visible = summary.occluded_cell_count == 0
            fully_occluded = summary.winning_cell_count == 0
        else:
            visible_fraction = None
            occluded_fraction = None
            fully_visible = False
            fully_occluded = False

        results.append(
            ActorVisibilityStatistic(
                actor_id=summary.actor_id,
                occupied_cell_count=summary.occupied_cell_count,
                winning_cell_count=summary.winning_cell_count,
                occluded_cell_count=summary.occluded_cell_count,
                has_sampled_surface=has_surface,
                visible_fraction=visible_fraction,
                occluded_fraction=occluded_fraction,
                fully_visible=fully_visible,
                fully_occluded=fully_occluded,
            )
        )

    return tuple(sorted(results, key=lambda item: item.actor_id))
