#!/usr/bin/env python3
"""Merge depth samples from multiple projected triangles of one Actor.

For every raster cell, the nearest positive camera-depth sample is retained.
The winning triangle index and source sample are preserved for diagnostics.
This module does not compare different Actors or compute visibility ratios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from step7.projected_triangle_depth_samples_v01 import ProjectedTriangleDepthSamples
from step7.projected_triangle_raster_cells_v01 import RasterCell


@dataclass(frozen=True, slots=True)
class ActorSurfaceCellDepth:
    cell: RasterCell
    depth_m: float
    reciprocal_depth_per_m: float
    winning_triangle_index: int
    sample_u_px: float
    sample_v_px: float
    barycentric_weights: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ActorSurfaceDepthRaster:
    cell_depths: tuple[ActorSurfaceCellDepth, ...]
    occupied_cell_count: int
    input_triangle_count: int
    input_depth_sample_count: int
    replaced_sample_count: int
    discarded_farther_or_equal_sample_count: int


def merge_actor_surface_depth_samples(
    triangle_samples: Sequence[ProjectedTriangleDepthSamples],
    *,
    depth_tolerance_m: float = 1e-9,
) -> ActorSurfaceDepthRaster:
    """Keep the nearest triangle sample in each cell for one Actor.

    A new sample replaces the current winner only when it is nearer by more
    than depth_tolerance_m. Equal or tolerance-equivalent samples preserve the
    earlier triangle index, giving deterministic input-order tie handling.
    """
    tolerance = float(depth_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("depth_tolerance_m must be non-negative and finite")

    winners: dict[RasterCell, ActorSurfaceCellDepth] = {}
    input_sample_count = 0
    replaced_count = 0
    discarded_count = 0

    for triangle_index, result in enumerate(triangle_samples):
        for sample in result.center_sampled_depths:
            input_sample_count += 1
            values = (
                sample.depth_m,
                sample.reciprocal_depth_per_m,
                sample.sample_u_px,
                sample.sample_v_px,
                *sample.barycentric_weights,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("triangle depth samples must be finite")
            if sample.depth_m <= 0.0 or sample.reciprocal_depth_per_m <= 0.0:
                raise ValueError("triangle depth samples must have positive depth")
            if not math.isclose(
                sample.depth_m * sample.reciprocal_depth_per_m,
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("depth and reciprocal depth must be consistent")

            candidate = ActorSurfaceCellDepth(
                cell=sample.cell,
                depth_m=sample.depth_m,
                reciprocal_depth_per_m=sample.reciprocal_depth_per_m,
                winning_triangle_index=triangle_index,
                sample_u_px=sample.sample_u_px,
                sample_v_px=sample.sample_v_px,
                barycentric_weights=sample.barycentric_weights,
            )
            current = winners.get(sample.cell)
            if current is None:
                winners[sample.cell] = candidate
            elif candidate.depth_m < current.depth_m - tolerance:
                winners[sample.cell] = candidate
                replaced_count += 1
            else:
                discarded_count += 1

    ordered = tuple(
        winners[cell]
        for cell in sorted(winners, key=lambda item: (item.row, item.column))
    )
    return ActorSurfaceDepthRaster(
        cell_depths=ordered,
        occupied_cell_count=len(ordered),
        input_triangle_count=len(triangle_samples),
        input_depth_sample_count=input_sample_count,
        replaced_sample_count=replaced_count,
        discarded_farther_or_equal_sample_count=discarded_count,
    )
