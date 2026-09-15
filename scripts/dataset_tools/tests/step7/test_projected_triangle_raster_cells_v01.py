from __future__ import annotations

import math

import pytest

from step7.projected_triangle_raster_cells_v01 import (
    PixelPoint,
    RasterCell,
    rasterize_projected_triangle_cells,
)


def rasterize(points, *, image=(8, 8), raster=(8, 8)):
    return rasterize_projected_triangle_cells(
        points,
        image_width_px=image[0], image_height_px=image[1],
        raster_width=raster[0], raster_height=raster[1],
    )


def test_small_triangle_inside_one_cell():
    result = rasterize((PixelPoint(2.1, 3.1), PixelPoint(2.8, 3.1), PixelPoint(2.1, 3.8)))
    assert result.cells == (RasterCell(2, 3),)


def test_triangle_crossing_four_cells_returns_row_major_order():
    result = rasterize((PixelPoint(1.5, 1.5), PixelPoint(2.5, 1.5), PixelPoint(1.5, 2.5)))
    assert result.cells == (
        RasterCell(1, 1), RasterCell(2, 1),
        RasterCell(1, 2), RasterCell(2, 2),
    )


def test_downsampling_uses_explicit_independent_scales():
    result = rasterize(
        (PixelPoint(4.2, 2.2), PixelPoint(5.8, 2.2), PixelPoint(4.2, 3.8)),
        image=(8, 8), raster=(4, 2),
    )
    assert result.cells == (RasterCell(2, 0),)


def test_fully_outside_triangle_returns_empty():
    result = rasterize((PixelPoint(-5.0, -5.0), PixelPoint(-4.0, -5.0), PixelPoint(-5.0, -4.0)))
    assert result.cells == ()
    assert result.candidate_column_range is None
    assert result.candidate_row_range is None


def test_partially_outside_triangle_is_clamped_to_raster():
    result = rasterize((PixelPoint(-1.0, 1.2), PixelPoint(1.2, 1.2), PixelPoint(1.2, 3.0)))
    assert result.cells
    assert all(cell.column >= 0 and cell.row >= 0 for cell in result.cells)


def test_reversed_winding_produces_same_cells():
    points = (PixelPoint(1.2, 1.2), PixelPoint(3.2, 1.2), PixelPoint(1.2, 3.2))
    assert rasterize(points).cells == rasterize((points[0], points[2], points[1])).cells


def test_degenerate_point_triangle_covers_containing_cell():
    point = PixelPoint(2.25, 4.25)
    assert rasterize((point, point, point)).cells == (RasterCell(2, 4),)


def test_boundary_touch_is_conservatively_included():
    result = rasterize((PixelPoint(2.0, 2.0), PixelPoint(3.0, 2.0), PixelPoint(2.0, 3.0)))
    assert RasterCell(1, 1) in result.cells
    assert RasterCell(2, 2) in result.cells


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_point_count_is_rejected(count):
    with pytest.raises(ValueError, match="exactly three"):
        rasterize(tuple(PixelPoint(float(i), 0.0) for i in range(count)))


@pytest.mark.parametrize("dimensions", ((0, 8, 8, 8), (8, -1, 8, 8), (8, 8, 0, 8), (8, 8, 8, 0)))
def test_nonpositive_dimension_is_rejected(dimensions):
    with pytest.raises(ValueError, match="positive"):
        rasterize_projected_triangle_cells(
            (PixelPoint(0.0, 0.0),) * 3,
            image_width_px=dimensions[0], image_height_px=dimensions[1],
            raster_width=dimensions[2], raster_height=dimensions[3],
        )


def test_boolean_dimension_is_rejected():
    with pytest.raises(TypeError, match="integers"):
        rasterize_projected_triangle_cells(
            (PixelPoint(0.0, 0.0),) * 3,
            image_width_px=True, image_height_px=8,
            raster_width=8, raster_height=8,
        )


def test_nonfinite_coordinate_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        rasterize((PixelPoint(math.nan, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)))
