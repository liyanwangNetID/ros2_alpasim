from __future__ import annotations

import math

import pytest

from step7.projected_triangle_depth_samples_v01 import (
    sample_projected_triangle_depths,
)
from step7.projected_triangle_raster_cells_v01 import PixelPoint, RasterCell


def sample(points, depths=(2.0, 4.0, 8.0), *, image=(8, 8), raster=(8, 8)):
    return sample_projected_triangle_depths(
        points,
        depths,
        image_width_px=image[0],
        image_height_px=image[1],
        raster_width=raster[0],
        raster_height=raster[1],
    )


def test_single_cell_center_inside_produces_one_sample():
    result = sample(
        (PixelPoint(2.0, 3.0), PixelPoint(3.0, 3.0), PixelPoint(2.0, 4.0)),
        depths=(5.0, 5.0, 5.0),
    )
    assert result.center_sampled_cell_count == 1
    item = result.center_sampled_depths[0]
    assert item.cell == RasterCell(2, 3)
    assert item.sample_u_px == pytest.approx(2.5)
    assert item.sample_v_px == pytest.approx(3.5)
    assert item.depth_m == pytest.approx(5.0)


def test_conservative_cells_are_distinct_from_center_samples():
    result = sample(
        (PixelPoint(2.0, 2.0), PixelPoint(3.0, 2.0), PixelPoint(2.0, 3.0)),
        depths=(5.0, 5.0, 5.0),
    )
    assert result.conservative_cell_count > result.center_sampled_cell_count
    assert RasterCell(1, 1) in result.conservative_coverage_cells
    assert RasterCell(1, 1) not in tuple(
        item.cell for item in result.center_sampled_depths
    )


def test_center_on_triangle_boundary_is_sampled():
    result = sample(
        (PixelPoint(1.0, 1.0), PixelPoint(4.0, 1.0), PixelPoint(1.0, 4.0)),
        depths=(3.0, 3.0, 3.0),
    )
    cells = tuple(item.cell for item in result.center_sampled_depths)
    assert RasterCell(2, 2) in cells


def test_perspective_depth_matches_reciprocal_formula():
    result = sample(
        (PixelPoint(0.0, 0.0), PixelPoint(4.0, 0.0), PixelPoint(0.0, 4.0)),
        depths=(2.0, 4.0, 8.0),
    )
    item = next(sample for sample in result.center_sampled_depths if sample.cell == RasterCell(0, 0))
    assert item.barycentric_weights == pytest.approx((0.75, 0.125, 0.125))
    expected_reciprocal = 0.75 / 2.0 + 0.125 / 4.0 + 0.125 / 8.0
    assert item.reciprocal_depth_per_m == pytest.approx(expected_reciprocal)
    assert item.depth_m == pytest.approx(1.0 / expected_reciprocal)


def test_downsampled_cell_center_maps_back_to_image_pixels():
    result = sample(
        (PixelPoint(2.0, 2.0), PixelPoint(6.0, 2.0), PixelPoint(2.0, 6.0)),
        depths=(5.0, 5.0, 5.0),
        image=(8, 8),
        raster=(4, 2),
    )
    item = next(sample for sample in result.center_sampled_depths if sample.cell == RasterCell(1, 0))
    assert item.sample_u_px == pytest.approx(3.0)
    assert item.sample_v_px == pytest.approx(2.0)


def test_reversed_winding_with_matching_depths_preserves_cell_depths():
    points = (PixelPoint(0.0, 0.0), PixelPoint(4.0, 0.0), PixelPoint(0.0, 4.0))
    forward = sample(points, depths=(2.0, 4.0, 8.0))
    reverse = sample((points[0], points[2], points[1]), depths=(2.0, 8.0, 4.0))
    forward_map = {item.cell: item.depth_m for item in forward.center_sampled_depths}
    reverse_map = {item.cell: item.depth_m for item in reverse.center_sampled_depths}
    assert reverse_map == pytest.approx(forward_map)


def test_triangle_outside_raster_has_no_samples():
    result = sample(
        (PixelPoint(-5.0, -5.0), PixelPoint(-4.0, -5.0), PixelPoint(-5.0, -4.0)),
        depths=(2.0, 2.0, 2.0),
    )
    assert result.conservative_coverage_cells == ()
    assert result.center_sampled_depths == ()


def test_degenerate_projected_triangle_is_rejected():
    result = sample_projected_triangle_depths(
        (
            PixelPoint(10.0, 10.0),
            PixelPoint(20.0, 20.0),
            PixelPoint(30.0, 30.0),
        ),
        (10.0, 11.0, 12.0),
        image_width_px=100,
        image_height_px=100,
        raster_width=50,
        raster_height=50,
    )

    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_pixel_count_is_rejected(count):
    with pytest.raises(ValueError, match="exactly three pixel points"):
        sample(tuple(PixelPoint(float(index), 0.0) for index in range(count)))


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_depth_count_is_rejected(count):
    with pytest.raises(ValueError, match="exactly three values"):
        sample(
            (PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)),
            depths=tuple(1.0 for _ in range(count)),
        )


@pytest.mark.parametrize("depths", ((0.0, 1.0, 1.0), (-1.0, 1.0, 1.0)))
def test_nonpositive_depth_is_rejected(depths):
    with pytest.raises(ValueError, match="positive"):
        sample(
            (PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)),
            depths=depths,
        )


def test_nonfinite_depth_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        sample(
            (PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)),
            depths=(math.nan, 1.0, 1.0),
        )


def test_invalid_dimensions_are_delegated_to_rasterizer():
    with pytest.raises(ValueError, match="positive"):
        sample_projected_triangle_depths(
            (PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)),
            (1.0, 1.0, 1.0),
            image_width_px=8,
            image_height_px=8,
            raster_width=0,
            raster_height=8,
        )
