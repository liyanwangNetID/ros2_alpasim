"""Adapt shared Actor depth Z-buffer results into Step 7E occlusion evidence.

This module extracts threshold-free, per-camera Actor-to-Actor occlusion evidence
from validated Actor surface rasters and one shared Z-buffer. It does not select
observability labels, aggregate cameras, evaluate static-scene occlusion, or
freeze raster and visibility thresholds.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from actor_depth_zbuffer_v01 import ActorDepthRasterInput, ActorDepthZBuffer
from actor_occlusion_evidence_v01 import (
    CameraActorOcclusionEvidence,
    build_camera_actor_occlusion_evidence,
)


def build_camera_occlusion_evidence_from_zbuffer(
    *,
    camera_name: str,
    raster_width: int,
    raster_height: int,
    actor_rasters: Sequence[ActorDepthRasterInput],
    zbuffer: ActorDepthZBuffer,
    static_occlusion_evaluated: bool = False,
) -> tuple[CameraActorOcclusionEvidence, ...]:
    """Build one threshold-free evidence record for every input Actor.

    Occluding Actor IDs are derived from target-occupied cells won by another
    Actor. The adapter validates that Z-buffer summaries and winners are
    consistent with the supplied Actor surface rasters.
    """

    inputs = tuple(actor_rasters)
    input_ids = tuple(item.actor_id for item in inputs)
    if any(not isinstance(actor_id, str) or not actor_id for actor_id in input_ids):
        raise ValueError("actor_id must be a non-empty string")
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("actor_id values must be unique")

    summaries = {item.actor_id: item for item in zbuffer.actor_summaries}
    if len(summaries) != len(zbuffer.actor_summaries):
        raise ValueError("Z-buffer summaries contain duplicate actor_id values")
    if set(summaries) != set(input_ids):
        raise ValueError("Z-buffer summary Actor IDs must match input Actor IDs")

    winners_by_cell = {item.cell: item for item in zbuffer.cell_winners}
    if len(winners_by_cell) != len(zbuffer.cell_winners):
        raise ValueError("Z-buffer contains duplicate winner cells")
    if len(winners_by_cell) != zbuffer.occupied_union_cell_count:
        raise ValueError("Z-buffer occupied union count is inconsistent")
    if any(item.actor_id not in summaries for item in zbuffer.cell_winners):
        raise ValueError("Z-buffer winner references an unknown Actor")

    occluders_by_actor: dict[str, set[str]] = defaultdict(set)

    for actor_input in inputs:
        actor_id = actor_input.actor_id
        surface = actor_input.surface_raster
        if surface.occupied_cell_count != len(surface.cell_depths):
            raise ValueError("surface raster occupied count is inconsistent")

        seen_cells = set()
        for sample in surface.cell_depths:
            if sample.cell in seen_cells:
                raise ValueError("surface raster contains duplicate cells")
            seen_cells.add(sample.cell)

            winner = winners_by_cell.get(sample.cell)
            if winner is None:
                raise ValueError("target occupied cell has no Z-buffer winner")
            if winner.actor_id != actor_id:
                occluders_by_actor[actor_id].add(winner.actor_id)

        summary = summaries[actor_id]
        if summary.occupied_cell_count != surface.occupied_cell_count:
            raise ValueError("Z-buffer summary occupied count is inconsistent")
        if summary.winning_cell_count + summary.occluded_cell_count != summary.occupied_cell_count:
            raise ValueError("Z-buffer summary counts are inconsistent")

    evidence = []
    for actor_input in sorted(inputs, key=lambda item: item.actor_id):
        summary = summaries[actor_input.actor_id]
        evidence.append(
            build_camera_actor_occlusion_evidence(
                camera_name=camera_name,
                track_id=actor_input.actor_id,
                raster_width=raster_width,
                raster_height=raster_height,
                occupied_cell_count=summary.occupied_cell_count,
                winning_cell_count=summary.winning_cell_count,
                occluded_cell_count=summary.occluded_cell_count,
                occluding_actor_ids=tuple(occluders_by_actor[actor_input.actor_id]),
                actor_to_actor_occlusion_evaluated=True,
                static_occlusion_evaluated=static_occlusion_evaluated,
                reasons=(),
            )
        )

    return tuple(evidence)
