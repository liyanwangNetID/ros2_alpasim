"""Tests for the consolidated Step 7 raster and z-buffer domain."""
from __future__ import annotations
import math
import pytest
from step7.raster import PixelPoint, RasterCell, rasterize_projected_triangle_cells
from step7.raster import sample_projected_triangle_depths
from step7.raster import PixelPoint, RasterCell
from dataclasses import dataclass
from step7.projection import Vector3
from step7.raster import sample_fov_subdivided_camera_triangle_depths
from step7.raster import sample_camera_triangle_depths
from step7.raster import RasterCell
from step7.raster import merge_actor_surface_depth_samples
from step7.raster import ProjectedTriangleDepthSamples, RasterDepthSample
from step7.raster import build_actor_camera_surface_depth_raster
from step7.raster import ActorDepthRasterInput, resolve_actor_depth_zbuffer
from step7.raster import ActorSurfaceCellDepth, ActorSurfaceDepthRaster
from step7.raster import PixelPoint
from step7.geometry import Point2D, triangle_is_degenerate

def cells_rasterize(points, *, image=(8, 8), raster=(8, 8)):
    return rasterize_projected_triangle_cells(points, image_width_px=image[0], image_height_px=image[1], raster_width=raster[0], raster_height=raster[1])

def test_small_triangle_inside_one_cell():
    result = cells_rasterize((PixelPoint(2.1, 3.1), PixelPoint(2.8, 3.1), PixelPoint(2.1, 3.8)))
    assert result.cells == (RasterCell(2, 3),)

def test_triangle_crossing_four_cells_returns_row_major_order():
    result = cells_rasterize((PixelPoint(1.5, 1.5), PixelPoint(2.5, 1.5), PixelPoint(1.5, 2.5)))
    assert result.cells == (RasterCell(1, 1), RasterCell(2, 1), RasterCell(1, 2), RasterCell(2, 2))

def test_downsampling_uses_explicit_independent_scales():
    result = cells_rasterize((PixelPoint(4.2, 2.2), PixelPoint(5.8, 2.2), PixelPoint(4.2, 3.8)), image=(8, 8), raster=(4, 2))
    assert result.cells == (RasterCell(2, 0),)

def test_fully_outside_triangle_returns_empty():
    result = cells_rasterize((PixelPoint(-5.0, -5.0), PixelPoint(-4.0, -5.0), PixelPoint(-5.0, -4.0)))
    assert result.cells == ()
    assert result.candidate_column_range is None
    assert result.candidate_row_range is None

def test_partially_outside_triangle_is_clamped_to_raster():
    result = cells_rasterize((PixelPoint(-1.0, 1.2), PixelPoint(1.2, 1.2), PixelPoint(1.2, 3.0)))
    assert result.cells
    assert all((cell.column >= 0 and cell.row >= 0 for cell in result.cells))

def test_reversed_winding_produces_same_cells():
    points = (PixelPoint(1.2, 1.2), PixelPoint(3.2, 1.2), PixelPoint(1.2, 3.2))
    assert cells_rasterize(points).cells == cells_rasterize((points[0], points[2], points[1])).cells

def test_degenerate_point_triangle_covers_containing_cell():
    point = PixelPoint(2.25, 4.25)
    assert cells_rasterize((point, point, point)).cells == (RasterCell(2, 4),)

def test_boundary_touch_is_conservatively_included():
    result = cells_rasterize((PixelPoint(2.0, 2.0), PixelPoint(3.0, 2.0), PixelPoint(2.0, 3.0)))
    assert RasterCell(1, 1) in result.cells
    assert RasterCell(2, 2) in result.cells

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_point_count_is_rejected(count):
    with pytest.raises(ValueError, match='exactly three'):
        cells_rasterize(tuple((PixelPoint(float(i), 0.0) for i in range(count))))

@pytest.mark.parametrize('dimensions', ((0, 8, 8, 8), (8, -1, 8, 8), (8, 8, 0, 8), (8, 8, 8, 0)))
def test_nonpositive_dimension_is_rejected(dimensions):
    with pytest.raises(ValueError, match='positive'):
        rasterize_projected_triangle_cells((PixelPoint(0.0, 0.0),) * 3, image_width_px=dimensions[0], image_height_px=dimensions[1], raster_width=dimensions[2], raster_height=dimensions[3])

def test_boolean_dimension_is_rejected():
    with pytest.raises(TypeError, match='integers'):
        rasterize_projected_triangle_cells((PixelPoint(0.0, 0.0),) * 3, image_width_px=True, image_height_px=8, raster_width=8, raster_height=8)

def test_nonfinite_coordinate_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        cells_rasterize((PixelPoint(math.nan, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)))

def depth_sample(points, depths=(2.0, 4.0, 8.0), *, image=(8, 8), raster=(8, 8)):
    return sample_projected_triangle_depths(points, depths, image_width_px=image[0], image_height_px=image[1], raster_width=raster[0], raster_height=raster[1])

def test_single_cell_center_inside_produces_one_sample():
    result = depth_sample((PixelPoint(2.0, 3.0), PixelPoint(3.0, 3.0), PixelPoint(2.0, 4.0)), depths=(5.0, 5.0, 5.0))
    assert result.center_sampled_cell_count == 1
    item = result.center_sampled_depths[0]
    assert item.cell == RasterCell(2, 3)
    assert item.sample_u_px == pytest.approx(2.5)
    assert item.sample_v_px == pytest.approx(3.5)
    assert item.depth_m == pytest.approx(5.0)

def test_conservative_cells_are_distinct_from_center_samples():
    result = depth_sample((PixelPoint(2.0, 2.0), PixelPoint(3.0, 2.0), PixelPoint(2.0, 3.0)), depths=(5.0, 5.0, 5.0))
    assert result.conservative_cell_count > result.center_sampled_cell_count
    assert RasterCell(1, 1) in result.conservative_coverage_cells
    assert RasterCell(1, 1) not in tuple((item.cell for item in result.center_sampled_depths))

def test_center_on_triangle_boundary_is_sampled():
    result = depth_sample((PixelPoint(1.0, 1.0), PixelPoint(4.0, 1.0), PixelPoint(1.0, 4.0)), depths=(3.0, 3.0, 3.0))
    cells = tuple((item.cell for item in result.center_sampled_depths))
    assert RasterCell(2, 2) in cells

def test_perspective_depth_matches_reciprocal_formula():
    result = depth_sample((PixelPoint(0.0, 0.0), PixelPoint(4.0, 0.0), PixelPoint(0.0, 4.0)), depths=(2.0, 4.0, 8.0))
    item = next((depth_sample for depth_sample in result.center_sampled_depths if depth_sample.cell == RasterCell(0, 0)))
    assert item.barycentric_weights == pytest.approx((0.75, 0.125, 0.125))
    expected_reciprocal = 0.75 / 2.0 + 0.125 / 4.0 + 0.125 / 8.0
    assert item.reciprocal_depth_per_m == pytest.approx(expected_reciprocal)
    assert item.depth_m == pytest.approx(1.0 / expected_reciprocal)

def test_downsampled_cell_center_maps_back_to_image_pixels():
    result = depth_sample((PixelPoint(2.0, 2.0), PixelPoint(6.0, 2.0), PixelPoint(2.0, 6.0)), depths=(5.0, 5.0, 5.0), image=(8, 8), raster=(4, 2))
    item = next((depth_sample for depth_sample in result.center_sampled_depths if depth_sample.cell == RasterCell(1, 0)))
    assert item.sample_u_px == pytest.approx(3.0)
    assert item.sample_v_px == pytest.approx(2.0)

def test_reversed_winding_with_matching_depths_preserves_cell_depths():
    points = (PixelPoint(0.0, 0.0), PixelPoint(4.0, 0.0), PixelPoint(0.0, 4.0))
    forward = depth_sample(points, depths=(2.0, 4.0, 8.0))
    reverse = depth_sample((points[0], points[2], points[1]), depths=(2.0, 8.0, 4.0))
    forward_map = {item.cell: item.depth_m for item in forward.center_sampled_depths}
    reverse_map = {item.cell: item.depth_m for item in reverse.center_sampled_depths}
    assert reverse_map == pytest.approx(forward_map)

def test_triangle_outside_raster_has_no_samples():
    result = depth_sample((PixelPoint(-5.0, -5.0), PixelPoint(-4.0, -5.0), PixelPoint(-5.0, -4.0)), depths=(2.0, 2.0, 2.0))
    assert result.conservative_coverage_cells == ()
    assert result.center_sampled_depths == ()

def test_degenerate_projected_triangle_is_rejected():
    result = sample_projected_triangle_depths((PixelPoint(10.0, 10.0), PixelPoint(20.0, 20.0), PixelPoint(30.0, 30.0)), (10.0, 11.0, 12.0), image_width_px=100, image_height_px=100, raster_width=50, raster_height=50)
    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_pixel_count_is_rejected(count):
    with pytest.raises(ValueError, match='exactly three pixel points'):
        depth_sample(tuple((PixelPoint(float(index), 0.0) for index in range(count))))

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_depth_count_is_rejected(count):
    with pytest.raises(ValueError, match='exactly three values'):
        depth_sample((PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)), depths=tuple((1.0 for _ in range(count))))

@pytest.mark.parametrize('depths', ((0.0, 1.0, 1.0), (-1.0, 1.0, 1.0)))
def test_nonpositive_depth_is_rejected(depths):
    with pytest.raises(ValueError, match='positive'):
        depth_sample((PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)), depths=depths)

def test_nonfinite_depth_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        depth_sample((PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)), depths=(math.nan, 1.0, 1.0))

def test_invalid_dimensions_are_delegated_to_rasterizer():
    with pytest.raises(ValueError, match='positive'):
        sample_projected_triangle_depths((PixelPoint(0.0, 0.0), PixelPoint(1.0, 0.0), PixelPoint(0.0, 1.0)), (1.0, 1.0, 1.0), image_width_px=8, image_height_px=8, raster_width=0, raster_height=8)

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
        return FakeProjection(self.scale * point.x / point.z + 4.0, self.scale * point.y / point.z + 4.0)

def fov_call(triangle, *, depth=4, extent=1.0, calibration=None):
    calibration = calibration or NormalizedCalibration()
    return sample_fov_subdivided_camera_triangle_depths(triangle, calibration, max_angle_rad=0.5, maximum_depth=depth, maximum_boundary_extent_px=extent, image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)

def fov_inside_triangle():
    return (Vector3(-1.0, -1.0, 4.0), Vector3(1.0, -1.0, 4.0), Vector3(-1.0, 1.0, 4.0))

def fov_boundary_triangle():
    return (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(0.0, 4.0, 2.0))

def test_fully_inside_triangle_is_sampled_directly():
    result = fov_call(fov_inside_triangle())
    assert result.sampled_inside_triangle_count == 1
    assert result.sampled_boundary_triangle_count == 0
    assert len(result.all_triangle_samples) == 1
    assert result.all_triangle_samples[0].center_sampled_cell_count > 0
    assert result.unresolved_boundary_count == 0

def test_fully_outside_triangle_is_rejected_without_samples():
    result = fov_call((Vector3(10.0, 0.0, 2.0), Vector3(12.0, 0.0, 2.0), Vector3(11.0, 1.0, 2.0)))
    assert result.all_triangle_samples == ()
    assert result.subdivision.rejected_outside_triangle_count == 1

def test_boundary_approximation_is_converted_to_safe_samples():
    result = fov_call(fov_boundary_triangle(), depth=4, extent=100.0, calibration=NormalizedCalibration(scale=1.0))
    assert result.subdivision.boundary_approximated_triangles
    assert result.measurable_boundary_polygon_count > 0
    assert result.sampled_boundary_triangle_count > 0
    assert result.boundary_triangle_samples
    assert result.unmeasurable_boundary_polygon_count == 0
    assert result.degenerate_boundary_polygon_count == 0

def test_all_samples_concatenate_inside_then_boundary_samples():
    result = fov_call(fov_boundary_triangle(), depth=2, extent=1.0, calibration=NormalizedCalibration(scale=1.0))
    assert result.all_triangle_samples == result.accepted_triangle_samples + result.boundary_triangle_samples

def test_depth_limited_boundary_is_reported_as_unresolved():
    result = fov_call(fov_boundary_triangle(), depth=0, extent=1e-06, calibration=NormalizedCalibration(scale=100.0))
    assert result.unresolved_boundary_count == 1
    assert result.all_triangle_samples == ()

def test_all_generated_sample_depths_are_positive():
    result = fov_call(fov_boundary_triangle(), depth=4, extent=100.0, calibration=NormalizedCalibration(scale=4.0))
    depths = [sample.depth_m for triangle_samples in result.all_triangle_samples for sample in triangle_samples.center_sampled_depths]
    assert depths
    assert all((depth > 0.0 for depth in depths))

def test_boundary_limit_validation_is_delegated():
    with pytest.raises(ValueError, match='maximum_boundary_extent_px'):
        fov_call(fov_inside_triangle(), extent=0.0)

def test_wrong_triangle_size_is_delegated():
    with pytest.raises(ValueError, match='exactly three'):
        fov_call((Vector3(0.0, 0.0, 2.0),) * 2)

@dataclass(frozen=True)
class FakeProjection_2:
    u: float
    v: float
    positive_z: bool = True
    within_fov: bool = True
    valid: bool = True

class LinearCalibration:

    def project_camera_point(self, point):
        return FakeProjection_2(point.x, point.y)

class OutsideFovCalibration:

    def project_camera_point(self, point):
        return FakeProjection_2(point.x, point.y, within_fov=False, valid=False)

class NonfiniteCalibration:

    def project_camera_point(self, point):
        return FakeProjection_2(math.nan, point.y)

def camera_triangle():
    return (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 4.0), Vector3(0.0, 4.0, 8.0))

def test_camera_triangle_projects_and_samples_depth():
    result = sample_camera_triangle_depths(camera_triangle(), LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)
    item = next((sample for sample in result.center_sampled_depths if sample.cell == RasterCell(0, 0)))
    assert item.barycentric_weights == pytest.approx((0.75, 0.125, 0.125))
    expected_reciprocal = 0.75 / 2.0 + 0.125 / 4.0 + 0.125 / 8.0
    assert item.depth_m == pytest.approx(1.0 / expected_reciprocal)

def test_projection_order_matches_camera_vertex_depth_order():
    source = camera_triangle()
    forward = sample_camera_triangle_depths(source, LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)
    reverse = sample_camera_triangle_depths((source[0], source[2], source[1]), LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)
    forward_map = {item.cell: item.depth_m for item in forward.center_sampled_depths}
    reverse_map = {item.cell: item.depth_m for item in reverse.center_sampled_depths}
    assert reverse_map == pytest.approx(forward_map)

def test_downsampled_raster_is_delegated_correctly():
    result = sample_camera_triangle_depths((Vector3(2.0, 2.0, 5.0), Vector3(6.0, 2.0, 5.0), Vector3(2.0, 6.0, 5.0)), LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=4, raster_height=2)
    item = next((sample for sample in result.center_sampled_depths if sample.cell == RasterCell(1, 0)))
    assert item.sample_u_px == pytest.approx(3.0)
    assert item.sample_v_px == pytest.approx(2.0)
    assert item.depth_m == pytest.approx(5.0)

def test_vertex_exactly_on_near_plane_is_accepted():
    result = sample_camera_triangle_depths((Vector3(0.0, 0.0, 0.1), Vector3(2.0, 0.0, 1.0), Vector3(0.0, 2.0, 1.0)), LinearCalibration(), image_width_px=4, image_height_px=4, raster_width=4, raster_height=4, near_plane_m=0.1)
    assert result.center_sampled_depths

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_camera_point_count_is_rejected(count):
    source = tuple((Vector3(float(index), 0.0, 1.0) for index in range(count)))
    with pytest.raises(ValueError, match='exactly three camera points'):
        sample_camera_triangle_depths(source, LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)

@pytest.mark.parametrize('near', (0.0, -1.0, math.nan, math.inf))
def test_invalid_near_plane_is_rejected(near):
    with pytest.raises(ValueError, match='near_plane_m'):
        sample_camera_triangle_depths(camera_triangle(), LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8, near_plane_m=near)

def test_vertex_behind_near_plane_is_rejected_before_projection():
    with pytest.raises(ValueError, match='behind near_plane_m'):
        sample_camera_triangle_depths((Vector3(0.0, 0.0, 0.09), Vector3(1.0, 0.0, 1.0), Vector3(0.0, 1.0, 1.0)), LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8, near_plane_m=0.1)

def test_nonfinite_camera_coordinate_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        sample_camera_triangle_depths((Vector3(math.nan, 0.0, 1.0), Vector3(1.0, 0.0, 1.0), Vector3(0.0, 1.0, 1.0)), LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)

def test_outside_fov_projection_is_rejected():
    with pytest.raises(ValueError, match='within FOV'):
        sample_camera_triangle_depths(camera_triangle(), OutsideFovCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)

def test_nonfinite_projection_is_rejected():
    with pytest.raises(ValueError, match='pixel coordinates must be finite'):
        sample_camera_triangle_depths(camera_triangle(), NonfiniteCalibration(), image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)

def test_degenerate_projected_triangle_is_rejected_by_sampler():

    class DegenerateProjectionCalibration:

        def project_camera_point(self, point):

            class Projection:
                pass
            result = Projection()
            result.u = point.x * 10.0
            result.v = point.x * 10.0
            result.positive_z = True
            result.within_fov = True
            result.valid = True
            return result
    result = sample_camera_triangle_depths((Vector3(1.0, 0.0, 10.0), Vector3(2.0, 0.0, 10.0), Vector3(3.0, 0.0, 10.0)), DegenerateProjectionCalibration(), image_width_px=100, image_height_px=100, raster_width=50, raster_height=50)
    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0

def test_invalid_raster_dimensions_are_delegated():
    with pytest.raises(ValueError, match='positive'):
        sample_camera_triangle_depths(camera_triangle(), LinearCalibration(), image_width_px=8, image_height_px=8, raster_width=0, raster_height=8)

def surface_depth_sample(column, row, depth, *, u=None, v=None):
    return RasterDepthSample(cell=RasterCell(column, row), sample_u_px=float(column) + 0.5 if u is None else u, sample_v_px=float(row) + 0.5 if v is None else v, barycentric_weights=(0.5, 0.25, 0.25), reciprocal_depth_per_m=1.0 / depth, depth_m=depth)

def surface_triangle_result(*samples):
    cells = tuple((sample.cell for sample in samples))
    return ProjectedTriangleDepthSamples(conservative_coverage_cells=cells, center_sampled_depths=tuple(samples), conservative_cell_count=len(cells), center_sampled_cell_count=len(samples))

def test_empty_input_produces_empty_raster():
    result = merge_actor_surface_depth_samples(())
    assert result.cell_depths == ()
    assert result.occupied_cell_count == 0
    assert result.input_triangle_count == 0
    assert result.input_depth_sample_count == 0

def test_disjoint_triangle_cells_are_all_preserved_in_row_major_order():
    result = merge_actor_surface_depth_samples((surface_triangle_result(surface_depth_sample(2, 1, 5.0)), surface_triangle_result(surface_depth_sample(0, 0, 3.0)), surface_triangle_result(surface_depth_sample(1, 1, 4.0))))
    assert tuple((item.cell for item in result.cell_depths)) == (RasterCell(0, 0), RasterCell(1, 1), RasterCell(2, 1))
    assert result.occupied_cell_count == 3
    assert result.input_triangle_count == 3
    assert result.input_depth_sample_count == 3

def test_nearer_later_triangle_replaces_current_winner():
    result = merge_actor_surface_depth_samples((surface_triangle_result(surface_depth_sample(1, 2, 8.0)), surface_triangle_result(surface_depth_sample(1, 2, 3.0))))
    winner = result.cell_depths[0]
    assert winner.depth_m == pytest.approx(3.0)
    assert winner.winning_triangle_index == 1
    assert result.replaced_sample_count == 1
    assert result.discarded_farther_or_equal_sample_count == 0

def test_farther_later_triangle_is_discarded():
    result = merge_actor_surface_depth_samples((surface_triangle_result(surface_depth_sample(1, 2, 3.0)), surface_triangle_result(surface_depth_sample(1, 2, 8.0))))
    assert result.cell_depths[0].winning_triangle_index == 0
    assert result.replaced_sample_count == 0
    assert result.discarded_farther_or_equal_sample_count == 1

def test_equal_depth_preserves_earlier_triangle_deterministically():
    result = merge_actor_surface_depth_samples((surface_triangle_result(surface_depth_sample(1, 2, 3.0, u=10.0)), surface_triangle_result(surface_depth_sample(1, 2, 3.0, u=20.0))))
    winner = result.cell_depths[0]
    assert winner.winning_triangle_index == 0
    assert winner.sample_u_px == pytest.approx(10.0)
    assert result.discarded_farther_or_equal_sample_count == 1

def test_depth_difference_within_tolerance_preserves_earlier_triangle():
    result = merge_actor_surface_depth_samples((surface_triangle_result(surface_depth_sample(0, 0, 5.0)), surface_triangle_result(surface_depth_sample(0, 0, 5.0 - 5e-07))), depth_tolerance_m=1e-06)
    assert result.cell_depths[0].winning_triangle_index == 0

def test_zero_tolerance_accepts_any_strictly_nearer_sample():
    result = merge_actor_surface_depth_samples((surface_triangle_result(surface_depth_sample(0, 0, 5.0)), surface_triangle_result(surface_depth_sample(0, 0, 5.0 - 5e-07))), depth_tolerance_m=0.0)
    assert result.cell_depths[0].winning_triangle_index == 1

def test_winner_preserves_source_diagnostics():
    sample = surface_depth_sample(3, 4, 2.5, u=31.0, v=42.0)
    result = merge_actor_surface_depth_samples((surface_triangle_result(sample),))
    winner = result.cell_depths[0]
    assert winner.sample_u_px == pytest.approx(31.0)
    assert winner.sample_v_px == pytest.approx(42.0)
    assert winner.barycentric_weights == sample.barycentric_weights
    assert winner.reciprocal_depth_per_m == pytest.approx(0.4)

def test_triangle_with_no_center_samples_is_counted_but_adds_no_cells():
    empty = ProjectedTriangleDepthSamples(conservative_coverage_cells=(RasterCell(0, 0),), center_sampled_depths=(), conservative_cell_count=1, center_sampled_cell_count=0)
    result = merge_actor_surface_depth_samples((empty,))
    assert result.input_triangle_count == 1
    assert result.input_depth_sample_count == 0
    assert result.occupied_cell_count == 0

@pytest.mark.parametrize('tolerance', (-1.0, math.nan, math.inf))
def test_invalid_depth_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match='depth_tolerance_m'):
        merge_actor_surface_depth_samples((), depth_tolerance_m=tolerance)

@pytest.mark.parametrize('depth', (0.0, -1.0))
def test_nonpositive_depth_sample_is_rejected(depth):
    bad = RasterDepthSample(cell=RasterCell(0, 0), sample_u_px=0.5, sample_v_px=0.5, barycentric_weights=(1.0, 0.0, 0.0), reciprocal_depth_per_m=1.0, depth_m=depth)
    with pytest.raises(ValueError, match='positive depth'):
        merge_actor_surface_depth_samples((surface_triangle_result(bad),))

def test_nonfinite_sample_is_rejected():
    bad = RasterDepthSample(cell=RasterCell(0, 0), sample_u_px=math.nan, sample_v_px=0.5, barycentric_weights=(1.0, 0.0, 0.0), reciprocal_depth_per_m=0.5, depth_m=2.0)
    with pytest.raises(ValueError, match='finite'):
        merge_actor_surface_depth_samples((surface_triangle_result(bad),))

def test_inconsistent_reciprocal_depth_is_rejected():
    bad = RasterDepthSample(cell=RasterCell(0, 0), sample_u_px=0.5, sample_v_px=0.5, barycentric_weights=(1.0, 0.0, 0.0), reciprocal_depth_per_m=0.3, depth_m=2.0)
    with pytest.raises(ValueError, match='consistent'):
        merge_actor_surface_depth_samples((surface_triangle_result(bad),))

@dataclass(frozen=True)
class FakeProjection_3:
    u: float
    v: float
    positive_z: bool = True
    within_fov: bool = True
    valid: bool = True

class NormalizedCalibration_2:

    def __init__(self, scale=4.0):
        self.scale = scale

    def project_camera_point(self, point):
        return FakeProjection_3(self.scale * point.x / point.z + 4.0, self.scale * point.y / point.z + 4.0)

def actorcamera_build(triangles, *, extent=1.0, depth=4, calibration=None):
    return build_actor_camera_surface_depth_raster(triangles, calibration or NormalizedCalibration_2(), max_angle_rad=0.5, maximum_depth=depth, maximum_boundary_extent_px=extent, image_width_px=8, image_height_px=8, raster_width=8, raster_height=8)

def actorcamera_inside_triangle(z=4.0):
    return (Vector3(-1.0, -1.0, z), Vector3(1.0, -1.0, z), Vector3(-1.0, 1.0, z))

def actorcamera_boundary_triangle():
    return (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(0.0, 4.0, 2.0))

def test_empty_triangle_input_produces_empty_actor_raster():
    result = actorcamera_build(())
    assert result.source_triangle_count == 0
    assert result.generated_triangle_sample_count == 0
    assert result.surface_raster.occupied_cell_count == 0
    assert result.triangle_results == ()

def test_one_inside_triangle_produces_actor_surface_cells():
    result = actorcamera_build((actorcamera_inside_triangle(),))
    assert result.source_triangle_count == 1
    assert result.generated_triangle_sample_count == 1
    assert result.surface_raster.occupied_cell_count > 0
    assert result.unresolved_boundary_count == 0

def test_multiple_disjoint_triangles_are_merged():
    left = (Vector3(-1.5, -1.5, 4.0), Vector3(-0.5, -1.5, 4.0), Vector3(-1.5, -0.5, 4.0))
    right = (Vector3(0.5, 0.5, 4.0), Vector3(1.5, 0.5, 4.0), Vector3(0.5, 1.5, 4.0))
    result = actorcamera_build((left, right))
    assert result.source_triangle_count == 2
    assert result.generated_triangle_sample_count == 2
    assert result.surface_raster.input_triangle_count == 2
    assert result.surface_raster.occupied_cell_count > 0

def test_overlapping_actor_surfaces_keep_nearest_depth():
    far = actorcamera_inside_triangle(z=8.0)
    near = actorcamera_inside_triangle(z=4.0)
    result = actorcamera_build((far, near), calibration=NormalizedCalibration_2(scale=8.0))
    assert result.surface_raster.replaced_sample_count > 0
    assert all((cell.depth_m == pytest.approx(4.0) for cell in result.surface_raster.cell_depths))

def test_boundary_geometry_contributes_generated_triangle_samples():
    result = actorcamera_build((actorcamera_boundary_triangle(),), extent=100.0, calibration=NormalizedCalibration_2(scale=4.0))
    triangle_result = result.triangle_results[0]
    assert triangle_result.sampled_boundary_triangle_count > 0
    assert result.generated_triangle_sample_count == len(triangle_result.all_triangle_samples)
    assert result.unresolved_boundary_count == 0

def test_zero_center_sample_triangles_are_counted_but_do_not_add_cells():
    tiny = (Vector3(-0.01, -0.01, 4.0), Vector3(0.01, -0.01, 4.0), Vector3(-0.01, 0.01, 4.0))
    result = actorcamera_build((tiny,), calibration=NormalizedCalibration_2(scale=1.0))
    assert result.generated_triangle_sample_count == 1
    assert result.zero_center_sample_triangle_count == 1
    assert result.surface_raster.occupied_cell_count == 0

def test_depth_limited_boundary_is_aggregated_as_unresolved():
    result = actorcamera_build((actorcamera_boundary_triangle(),), depth=0, extent=1e-06, calibration=NormalizedCalibration_2(scale=100.0))
    assert result.unresolved_boundary_count == 1
    assert result.generated_triangle_sample_count == 0

def test_output_cells_are_row_major_sorted():
    result = actorcamera_build((actorcamera_inside_triangle(),))
    cells = tuple((item.cell for item in result.surface_raster.cell_depths))
    assert cells == tuple(sorted(cells, key=lambda cell: (cell.row, cell.column)))

def test_invalid_boundary_limit_is_delegated():
    with pytest.raises(ValueError, match='maximum_boundary_extent_px'):
        actorcamera_build((actorcamera_inside_triangle(),), extent=0.0)

def test_invalid_depth_tolerance_is_delegated_after_sampling():
    with pytest.raises(ValueError, match='depth_tolerance_m'):
        build_actor_camera_surface_depth_raster((), NormalizedCalibration_2(), max_angle_rad=0.5, maximum_depth=4, maximum_boundary_extent_px=1.0, image_width_px=8, image_height_px=8, raster_width=8, raster_height=8, depth_tolerance_m=-1.0)

def test_invalid_source_triangle_is_delegated():
    with pytest.raises(ValueError, match='exactly three'):
        actorcamera_build(((Vector3(0.0, 0.0, 2.0),) * 2,))

def zbuffer_surface(column, row, depth):
    return ActorSurfaceCellDepth(cell=RasterCell(column, row), depth_m=depth, reciprocal_depth_per_m=1.0 / depth, winning_triangle_index=0, sample_u_px=column + 0.5, sample_v_px=row + 0.5, barycentric_weights=(0.5, 0.25, 0.25))

def zbuffer_raster(*items, occupied_count=None):
    count = len(items) if occupied_count is None else occupied_count
    return ActorSurfaceDepthRaster(cell_depths=tuple(items), occupied_cell_count=count, input_triangle_count=1, input_depth_sample_count=len(items), replaced_sample_count=0, discarded_farther_or_equal_sample_count=0)

def zbuffer_actor(actor_id, *items):
    return ActorDepthRasterInput(actor_id, zbuffer_raster(*items))

def test_empty_input_returns_empty_zbuffer():
    result = resolve_actor_depth_zbuffer(())
    assert result.cell_winners == ()
    assert result.actor_summaries == ()
    assert result.occupied_union_cell_count == 0
    assert result.contested_cell_count == 0

def test_disjoint_actors_keep_all_cells_in_row_major_order():
    result = resolve_actor_depth_zbuffer((zbuffer_actor('b', zbuffer_surface(2, 1, 4.0)), zbuffer_actor('a', zbuffer_surface(0, 0, 3.0))))
    assert tuple((item.cell for item in result.cell_winners)) == (RasterCell(0, 0), RasterCell(2, 1))
    assert result.occupied_union_cell_count == 2
    assert result.contested_cell_count == 0

def test_nearer_actor_wins_contested_cell():
    result = resolve_actor_depth_zbuffer((zbuffer_actor('far', zbuffer_surface(1, 1, 8.0)), zbuffer_actor('near', zbuffer_surface(1, 1, 3.0))))
    assert result.cell_winners[0].actor_id == 'near'
    assert result.cell_winners[0].depth_m == pytest.approx(3.0)
    assert result.contested_cell_count == 1

def test_tie_uses_lexicographically_smaller_actor_id():
    result = resolve_actor_depth_zbuffer((zbuffer_actor('zeta', zbuffer_surface(1, 1, 3.0)), zbuffer_actor('alpha', zbuffer_surface(1, 1, 3.0))))
    assert result.cell_winners[0].actor_id == 'alpha'

def test_tolerance_equivalent_depths_use_actor_id_tie_break():
    result = resolve_actor_depth_zbuffer((zbuffer_actor('a', zbuffer_surface(1, 1, 5.0)), zbuffer_actor('b', zbuffer_surface(1, 1, 5.0 - 5e-07))), depth_tolerance_m=1e-06)
    assert result.cell_winners[0].actor_id == 'a'

def test_strictly_nearer_beyond_tolerance_wins_regardless_of_actor_id():
    result = resolve_actor_depth_zbuffer((zbuffer_actor('a', zbuffer_surface(1, 1, 5.0)), zbuffer_actor('z', zbuffer_surface(1, 1, 4.0))), depth_tolerance_m=0.1)
    assert result.cell_winners[0].actor_id == 'z'

def test_input_order_does_not_change_winners_or_summaries():
    first = zbuffer_actor('b', zbuffer_surface(0, 0, 2.0), zbuffer_surface(1, 0, 4.0))
    second = zbuffer_actor('a', zbuffer_surface(1, 0, 4.0), zbuffer_surface(2, 0, 3.0))
    forward = resolve_actor_depth_zbuffer((first, second))
    reverse = resolve_actor_depth_zbuffer((second, first))
    assert forward == reverse

def test_actor_summaries_count_occupied_winning_and_occluded_cells():
    result = resolve_actor_depth_zbuffer((zbuffer_actor('near', zbuffer_surface(0, 0, 2.0), zbuffer_surface(1, 0, 2.0)), zbuffer_actor('far', zbuffer_surface(1, 0, 5.0), zbuffer_surface(2, 0, 5.0))))
    summaries = {item.actor_id: item for item in result.actor_summaries}
    assert summaries['near'].occupied_cell_count == 2
    assert summaries['near'].winning_cell_count == 2
    assert summaries['near'].occluded_cell_count == 0
    assert summaries['far'].occupied_cell_count == 2
    assert summaries['far'].winning_cell_count == 1
    assert summaries['far'].occluded_cell_count == 1

def test_winner_preserves_source_surface_record():
    source = zbuffer_surface(3, 4, 2.5)
    result = resolve_actor_depth_zbuffer((zbuffer_actor('actor-1', source),))
    assert result.cell_winners[0].source_surface == source

def test_actor_with_empty_raster_receives_zero_summary():
    result = resolve_actor_depth_zbuffer((ActorDepthRasterInput('empty', zbuffer_raster()),))
    assert result.actor_summaries[0].occupied_cell_count == 0
    assert result.actor_summaries[0].winning_cell_count == 0
    assert result.actor_summaries[0].occluded_cell_count == 0

def test_duplicate_actor_ids_are_rejected():
    with pytest.raises(ValueError, match='unique'):
        resolve_actor_depth_zbuffer((zbuffer_actor('same'), zbuffer_actor('same')))

@pytest.mark.parametrize('actor_id', ('', None, 123))
def test_invalid_actor_id_is_rejected(actor_id):
    with pytest.raises(ValueError, match='non-empty string'):
        resolve_actor_depth_zbuffer((ActorDepthRasterInput(actor_id, zbuffer_raster()),))

@pytest.mark.parametrize('tolerance', (-1.0, math.nan, math.inf))
def test_zbuffer_invalid_depth_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match='depth_tolerance_m'):
        resolve_actor_depth_zbuffer((), depth_tolerance_m=tolerance)

def test_inconsistent_occupied_count_is_rejected():
    bad = ActorDepthRasterInput('a', zbuffer_raster(zbuffer_surface(0, 0, 2.0), occupied_count=2))
    with pytest.raises(ValueError, match='occupied_cell_count'):
        resolve_actor_depth_zbuffer((bad,))

def test_duplicate_cells_within_actor_raster_are_rejected():
    duplicate = zbuffer_surface(0, 0, 2.0)
    bad = ActorDepthRasterInput('a', zbuffer_raster(duplicate, duplicate))
    with pytest.raises(ValueError, match='duplicate cells'):
        resolve_actor_depth_zbuffer((bad,))

@pytest.mark.parametrize('depth', (0.0, -1.0, math.nan, math.inf))
def test_invalid_surface_depth_is_rejected(depth):
    bad_surface = ActorSurfaceCellDepth(cell=RasterCell(0, 0), depth_m=depth, reciprocal_depth_per_m=1.0, winning_triangle_index=0, sample_u_px=0.5, sample_v_px=0.5, barycentric_weights=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='positive and finite'):
        resolve_actor_depth_zbuffer((zbuffer_actor('a', bad_surface),))

def degenerate_depth_sample(points):
    return sample_projected_triangle_depths(points, (10.0, 11.0, 12.0), image_width_px=100, image_height_px=100, raster_width=50, raster_height=50)

def test_exactly_collinear_projected_triangle_returns_zero_center_samples():
    result = degenerate_depth_sample((PixelPoint(10.0, 10.0), PixelPoint(20.0, 20.0), PixelPoint(30.0, 30.0)))
    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0
    assert result.conservative_cell_count == len(result.conservative_coverage_cells)

def test_near_collinear_projected_triangle_uses_scale_aware_degeneracy():
    triangle = (Point2D(1000.0, 1000.0), Point2D(2000.0, 2000.0), Point2D(3000.0, 3000.0 + 1e-09))
    assert triangle_is_degenerate(triangle)

def test_regular_projected_triangle_still_generates_depth_samples():
    result = degenerate_depth_sample((PixelPoint(10.0, 10.0), PixelPoint(40.0, 10.0), PixelPoint(10.0, 40.0)))
    assert result.center_sampled_cell_count > 0
    assert all((value.depth_m > 0.0 for value in result.center_sampled_depths))
