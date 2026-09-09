#!/usr/bin/env python3
"""Deterministic multi-Actor depth competition on a shared raster.

Each input Actor already contains its nearest surface depth per occupied cell.
This module compares Actors cell by cell, keeps the nearest depth, and records
per-Actor occupied and winning-cell counts. It does not calculate visibility
ratios or build Actor surfaces from geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from actor_surface_depth_raster_v01 import (
    ActorSurfaceCellDepth,
    ActorSurfaceDepthRaster,
)
from projected_triangle_raster_cells_v01 import RasterCell


@dataclass(frozen=True, slots=True)
class ActorDepthRasterInput:
    actor_id: str
    surface_raster: ActorSurfaceDepthRaster


@dataclass(frozen=True, slots=True)
class ZBufferCellWinner:
    cell: RasterCell
    actor_id: str
    depth_m: float
    source_surface: ActorSurfaceCellDepth


@dataclass(frozen=True, slots=True)
class ActorZBufferSummary:
    actor_id: str
    occupied_cell_count: int
    winning_cell_count: int
    occluded_cell_count: int


@dataclass(frozen=True, slots=True)
class ActorDepthZBuffer:
    cell_winners: tuple[ZBufferCellWinner, ...]
    actor_summaries: tuple[ActorZBufferSummary, ...]
    occupied_union_cell_count: int
    contested_cell_count: int


def resolve_actor_depth_zbuffer(
    actor_rasters: Sequence[ActorDepthRasterInput],
    *,
    depth_tolerance_m: float = 1e-9,
) -> ActorDepthZBuffer:
    """Resolve nearest Actor per cell with order-independent tie handling.

    A candidate wins if it is nearer by more than depth_tolerance_m. Depths
    within tolerance are tied and the lexicographically smaller actor_id wins.
    Actor IDs must be non-empty and unique.
    """
    tolerance = float(depth_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("depth_tolerance_m must be non-negative and finite")

    inputs = tuple(actor_rasters)
    actor_ids = tuple(item.actor_id for item in inputs)
    if any(not isinstance(actor_id, str) or not actor_id for actor_id in actor_ids):
        raise ValueError("actor_id must be a non-empty string")
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError("actor_id values must be unique")

    candidates_by_cell: dict[RasterCell, list[tuple[str, ActorSurfaceCellDepth]]] = {}
    occupied_counts: dict[str, int] = {}

    for actor_input in inputs:
        depths = actor_input.surface_raster.cell_depths
        if actor_input.surface_raster.occupied_cell_count != len(depths):
            raise ValueError("surface raster occupied_cell_count is inconsistent")
        seen_cells: set[RasterCell] = set()
        for surface in depths:
            if surface.cell in seen_cells:
                raise ValueError("surface raster contains duplicate cells")
            seen_cells.add(surface.cell)
            if not math.isfinite(surface.depth_m) or surface.depth_m <= 0.0:
                raise ValueError("surface depths must be positive and finite")
            candidates_by_cell.setdefault(surface.cell, []).append(
                (actor_input.actor_id, surface)
            )
        occupied_counts[actor_input.actor_id] = len(depths)

    winners: list[ZBufferCellWinner] = []
    winning_counts = {actor_id: 0 for actor_id in actor_ids}
    contested_count = 0

    for cell in sorted(candidates_by_cell, key=lambda item: (item.row, item.column)):
        candidates = candidates_by_cell[cell]
        if len(candidates) > 1:
            contested_count += 1
        minimum_depth = min(surface.depth_m for _, surface in candidates)
        tied = [
            (actor_id, surface)
            for actor_id, surface in candidates
            if surface.depth_m <= minimum_depth + tolerance
        ]
        winner_actor_id, winner_surface = min(tied, key=lambda item: item[0])
        winners.append(
            ZBufferCellWinner(
                cell=cell,
                actor_id=winner_actor_id,
                depth_m=winner_surface.depth_m,
                source_surface=winner_surface,
            )
        )
        winning_counts[winner_actor_id] += 1

    summaries = tuple(
        ActorZBufferSummary(
            actor_id=actor_id,
            occupied_cell_count=occupied_counts[actor_id],
            winning_cell_count=winning_counts[actor_id],
            occluded_cell_count=(
                occupied_counts[actor_id] - winning_counts[actor_id]
            ),
        )
        for actor_id in sorted(actor_ids)
    )
    return ActorDepthZBuffer(
        cell_winners=tuple(winners),
        actor_summaries=summaries,
        occupied_union_cell_count=len(winners),
        contested_cell_count=contested_count,
    )
