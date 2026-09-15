import pytest

from step7.projected_triangle_depth_samples_v01 import (
    sample_projected_triangle_depths,
)
from step7.projected_triangle_raster_cells_v01 import PixelPoint
from step7.triangle_barycentric_coordinates_v01 import Point2D, triangle_is_degenerate


def sample(points):
    return sample_projected_triangle_depths(
        points,
        (10.0, 11.0, 12.0),
        image_width_px=100,
        image_height_px=100,
        raster_width=50,
        raster_height=50,
    )


def test_exactly_collinear_projected_triangle_returns_zero_center_samples():
    result = sample((
        PixelPoint(10.0, 10.0),
        PixelPoint(20.0, 20.0),
        PixelPoint(30.0, 30.0),
    ))

    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0
    assert result.conservative_cell_count == len(result.conservative_coverage_cells)


def test_near_collinear_projected_triangle_uses_scale_aware_degeneracy():
    triangle = (
        Point2D(1000.0, 1000.0),
        Point2D(2000.0, 2000.0),
        Point2D(3000.0, 3000.0 + 1e-9),
    )

    assert triangle_is_degenerate(triangle)


def test_regular_projected_triangle_still_generates_depth_samples():
    result = sample((
        PixelPoint(10.0, 10.0),
        PixelPoint(40.0, 10.0),
        PixelPoint(10.0, 40.0),
    ))

    assert result.center_sampled_cell_count > 0
    assert all(value.depth_m > 0.0 for value in result.center_sampled_depths)
