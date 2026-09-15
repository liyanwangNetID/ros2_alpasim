#!/usr/bin/env python3
"""Center-sampled perspective depths for one projected triangle.

The module first obtains conservative raster-cell coverage, then samples each
covered cell at its center. Only centers inside the closed projected triangle
produce depth samples. A degenerate projected triangle retains conservative
coverage diagnostics but produces zero center samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from step7.projected_triangle_raster_cells_v01 import (
    PixelPoint,
    RasterCell,
    rasterize_projected_triangle_cells,
)
from step7.triangle_barycentric_coordinates_v01 import (
    Point2D,
    triangle_barycentric_coordinates,
    triangle_is_degenerate,
)
from step7.triangle_perspective_depth_v01 import (
    interpolate_perspective_camera_depth,
)


@dataclass(frozen=True, slots=True)
class RasterDepthSample:
    cell: RasterCell
    sample_u_px: float
    sample_v_px: float
    barycentric_weights: tuple[float, float, float]
    reciprocal_depth_per_m: float
    depth_m: float


@dataclass(frozen=True, slots=True)
class ProjectedTriangleDepthSamples:
    conservative_coverage_cells: tuple[RasterCell, ...]
    center_sampled_depths: tuple[RasterDepthSample, ...]
    conservative_cell_count: int
    center_sampled_cell_count: int


def sample_projected_triangle_depths(
    triangle_pixels: Sequence[PixelPoint],
    vertex_depths_m: Sequence[float],
    *,
    image_width_px: int,
    image_height_px: int,
    raster_width: int,
    raster_height: int,
) -> ProjectedTriangleDepthSamples:
    """Return center samples and perspective-correct depths for one triangle."""
    if len(triangle_pixels) != 3:
        raise ValueError("a triangle must contain exactly three pixel points")
    if len(vertex_depths_m) != 3:
        raise ValueError("vertex_depths_m must contain exactly three values")

    pixels = tuple(triangle_pixels)
    depths = tuple(float(value) for value in vertex_depths_m)
    if not all(math.isfinite(value) for value in depths):
        raise ValueError("vertex depths must be finite")
    if not all(value > 0.0 for value in depths):
        raise ValueError("vertex depths must be positive")

    coverage = rasterize_projected_triangle_cells(
        pixels,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        raster_width=raster_width,
        raster_height=raster_height,
    )
    barycentric_triangle = tuple(Point2D(point.u, point.v) for point in pixels)

    if triangle_is_degenerate(barycentric_triangle):
        return ProjectedTriangleDepthSamples(
            conservative_coverage_cells=coverage.cells,
            center_sampled_depths=(),
            conservative_cell_count=len(coverage.cells),
            center_sampled_cell_count=0,
        )

    pixel_width_per_cell = image_width_px / raster_width
    pixel_height_per_cell = image_height_px / raster_height
    samples: list[RasterDepthSample] = []

    for cell in coverage.cells:
        sample_u = (cell.column + 0.5) * pixel_width_per_cell
        sample_v = (cell.row + 0.5) * pixel_height_per_cell
        barycentric = triangle_barycentric_coordinates(
            barycentric_triangle,
            Point2D(sample_u, sample_v),
        )
        if not barycentric.inside_closed_triangle:
            continue

        weights = (
            barycentric.first_weight,
            barycentric.second_weight,
            barycentric.third_weight,
        )
        interpolation = interpolate_perspective_camera_depth(depths, weights)
        samples.append(
            RasterDepthSample(
                cell=cell,
                sample_u_px=sample_u,
                sample_v_px=sample_v,
                barycentric_weights=weights,
                reciprocal_depth_per_m=interpolation.reciprocal_depth_per_m,
                depth_m=interpolation.depth_m,
            )
        )

    result_samples = tuple(samples)
    return ProjectedTriangleDepthSamples(
        conservative_coverage_cells=coverage.cells,
        center_sampled_depths=result_samples,
        conservative_cell_count=len(coverage.cells),
        center_sampled_cell_count=len(result_samples),
    )
