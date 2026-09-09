from __future__ import annotations

from dataclasses import dataclass

import pytest

from actor_camera_surface_depth_raster_v01 import (
    build_actor_camera_surface_depth_raster,
)
from camera_projection_v01 import Vector3
from projected_triangle_raster_cells_v01 import RasterCell


@dataclass(frozen=True)
class FakeProjection:
    u: float
    v: float
    positive_z: bool = True
    within_fov: bool = True
    valid: bool = True


class NormalizedCalibration:
    def __init__(self, scale=4.0):
        self.scale = scale

    def project_camera_point(self, point):
        return FakeProjection(
            self.scale * point.x / point.z + 4.0,
            self.scale * point.y / point.z + 4.0,
        )


def build(triangles, *, extent=1.0, depth=4, calibration=None):
    return build_actor_camera_surface_depth_raster(
        triangles,
        calibration or NormalizedCalibration(),
        max_angle_rad=0.5,
        maximum_depth=depth,
        maximum_boundary_extent_px=extent,
        image_width_px=8,
        image_height_px=8,
        raster_width=8,
        raster_height=8,
    )


def inside_triangle(z=4.0):
    return (
        Vector3(-1.0, -1.0, z),
        Vector3(1.0, -1.0, z),
        Vector3(-1.0, 1.0, z),
    )


def boundary_triangle():
    return (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )


def test_empty_triangle_input_produces_empty_actor_raster():
    result = build(())
    assert result.source_triangle_count == 0
    assert result.generated_triangle_sample_count == 0
    assert result.surface_raster.occupied_cell_count == 0
    assert result.triangle_results == ()


def test_one_inside_triangle_produces_actor_surface_cells():
    result = build((inside_triangle(),))
    assert result.source_triangle_count == 1
    assert result.generated_triangle_sample_count == 1
    assert result.surface_raster.occupied_cell_count > 0
    assert result.unresolved_boundary_count == 0


def test_multiple_disjoint_triangles_are_merged():
    left = (
        Vector3(-1.5, -1.5, 4.0),
        Vector3(-0.5, -1.5, 4.0),
        Vector3(-1.5, -0.5, 4.0),
    )
    right = (
        Vector3(0.5, 0.5, 4.0),
        Vector3(1.5, 0.5, 4.0),
        Vector3(0.5, 1.5, 4.0),
    )
    result = build((left, right))
    assert result.source_triangle_count == 2
    assert result.generated_triangle_sample_count == 2
    assert result.surface_raster.input_triangle_count == 2
    assert result.surface_raster.occupied_cell_count > 0


def test_overlapping_actor_surfaces_keep_nearest_depth():
    far = inside_triangle(z=8.0)
    near = inside_triangle(z=4.0)
    result = build((far, near), calibration=NormalizedCalibration(scale=8.0))
    assert result.surface_raster.replaced_sample_count > 0
    assert all(
        cell.depth_m == pytest.approx(4.0)
        for cell in result.surface_raster.cell_depths
    )


def test_boundary_geometry_contributes_generated_triangle_samples():
    result = build(
        (boundary_triangle(),),
        extent=100.0,
        calibration=NormalizedCalibration(scale=4.0),
    )
    triangle_result = result.triangle_results[0]
    assert triangle_result.sampled_boundary_triangle_count > 0
    assert result.generated_triangle_sample_count == len(
        triangle_result.all_triangle_samples
    )
    assert result.unresolved_boundary_count == 0


def test_zero_center_sample_triangles_are_counted_but_do_not_add_cells():
    tiny = (
        Vector3(-0.01, -0.01, 4.0),
        Vector3(0.01, -0.01, 4.0),
        Vector3(-0.01, 0.01, 4.0),
    )
    result = build((tiny,), calibration=NormalizedCalibration(scale=1.0))
    assert result.generated_triangle_sample_count == 1
    assert result.zero_center_sample_triangle_count == 1
    assert result.surface_raster.occupied_cell_count == 0


def test_depth_limited_boundary_is_aggregated_as_unresolved():
    result = build(
        (boundary_triangle(),),
        depth=0,
        extent=1e-6,
        calibration=NormalizedCalibration(scale=100.0),
    )
    assert result.unresolved_boundary_count == 1
    assert result.generated_triangle_sample_count == 0


def test_output_cells_are_row_major_sorted():
    result = build((inside_triangle(),))
    cells = tuple(item.cell for item in result.surface_raster.cell_depths)
    assert cells == tuple(sorted(cells, key=lambda cell: (cell.row, cell.column)))


def test_invalid_boundary_limit_is_delegated():
    with pytest.raises(ValueError, match="maximum_boundary_extent_px"):
        build((inside_triangle(),), extent=0.0)


def test_invalid_depth_tolerance_is_delegated_after_sampling():
    with pytest.raises(ValueError, match="depth_tolerance_m"):
        build_actor_camera_surface_depth_raster(
            (),
            NormalizedCalibration(),
            max_angle_rad=0.5,
            maximum_depth=4,
            maximum_boundary_extent_px=1.0,
            image_width_px=8,
            image_height_px=8,
            raster_width=8,
            raster_height=8,
            depth_tolerance_m=-1.0,
        )


def test_invalid_source_triangle_is_delegated():
    with pytest.raises(ValueError, match="exactly three"):
        build(((Vector3(0.0, 0.0, 2.0),) * 2,))
