"""Tests for the consolidated Step 7 geometry domain."""
from __future__ import annotations
import math
from dataclasses import dataclass
import pytest
from step7.projection import Vector3
from step7.geometry import summarize_angular_fov_boundary_projected_extent
from step7.geometry import build_safe_angular_fov_polygon
from step7.geometry import point_is_within_angular_fov
from step7.geometry import FOV_EDGE_INDEX_PAIRS, classify_triangle_angular_fov, point_is_within_angular_fov
from step7.geometry import subdivide_triangle_to_angular_fov
from step7.geometry import Point2D, triangle_barycentric_coordinates
from step7.geometry import triangle_intersects_angular_fov_cone
from step7.geometry import triangulate_box_surfaces
from step7.geometry import is_triangle_front_facing, select_front_facing_triangles, triangle_normal
from step7.projection import clip_segment_to_positive_z
from step7.geometry import clip_polygon_to_positive_z, clip_triangle_to_positive_z
from step7.geometry import interpolate_perspective_camera_depth
from step7.projection import BOX_EDGE_INDEX_PAIRS
from step7.projection import local_box_corners
from step7.geometry import BOX_FACE_INDEX_QUADS, BOX_TRIANGLE_INDEX_TRIPLES, triangulate_box_surfaces
from step7.geometry import prepare_camera_facing_box_triangles
from step7.geometry import is_triangle_front_facing

@dataclass(frozen=True)
class FakeProjection:
    u: float
    v: float
    positive_z: bool = True
    within_fov: bool = True
    valid: bool = True

class NormalizedCalibration:

    def project_camera_point(self, point):
        return FakeProjection(100.0 * point.x / point.z, 100.0 * point.y / point.z)

class InvalidCalibration:

    def project_camera_point(self, point):
        return FakeProjection(0.0, 0.0, within_fov=False, valid=False)

def test_fully_inside_triangle_uses_three_unique_vertices():
    triangle = (Vector3(-0.2, -0.1, 2.0), Vector3(0.2, -0.1, 2.0), Vector3(0.0, 0.2, 2.0))
    result = summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=0.5)
    assert result.point_count == 3
    assert result.has_measurable_extent
    assert result.width_px == pytest.approx(20.0)
    assert result.height_px == pytest.approx(15.0)
    assert result.maximum_pair_distance_px is not None

def test_one_inside_vertex_collects_boundary_intersections():
    triangle = (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(0.0, 4.0, 2.0))
    result = summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=0.5)
    assert result.point_count == 3
    assert result.has_measurable_extent
    assert result.maximum_pair_distance_px is not None
    assert result.maximum_pair_distance_px > 0.0

def test_two_outside_endpoints_crossing_cone_collects_two_points():
    triangle = (Vector3(-4.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(4.0, 2.0, 2.0))
    result = summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=0.2)
    assert result.point_count >= 2
    assert result.has_measurable_extent

def test_separated_outside_triangle_has_no_extent():
    triangle = (Vector3(10.0, 0.0, 2.0), Vector3(12.0, 0.0, 2.0), Vector3(11.0, 1.0, 2.0))
    result = summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=0.5)
    assert result.point_count == 0
    assert not result.has_measurable_extent
    assert result.width_px is None
    assert result.height_px is None
    assert result.maximum_pair_distance_px is None

def test_none_fov_deduplicates_shared_edge_endpoints():
    triangle = (Vector3(0.0, 0.0, 2.0), Vector3(1.0, 0.0, 2.0), Vector3(0.0, 1.0, 2.0))
    result = summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=None)
    assert result.point_count == 3

def test_single_unique_point_has_zero_extent():
    point = Vector3(0.0, 0.0, 2.0)
    result = summarize_angular_fov_boundary_projected_extent((point, point, point), NormalizedCalibration(), max_angle_rad=0.5)
    assert result.point_count == 1
    assert result.width_px == pytest.approx(0.0)
    assert result.height_px == pytest.approx(0.0)
    assert result.maximum_pair_distance_px == pytest.approx(0.0)

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    triangle = tuple((Vector3(float(index), 0.0, 1.0) for index in range(count)))
    with pytest.raises(ValueError, match='exactly three vertices'):
        summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=0.5)

@pytest.mark.parametrize('z', (0.0, -1.0))
def test_nonpositive_depth_is_rejected(z):
    triangle = (Vector3(0.0, 0.0, z), Vector3(0.1, 0.0, 1.0), Vector3(0.0, 0.1, 1.0))
    with pytest.raises(ValueError, match='positive depth'):
        summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=0.5)

def test_nonfinite_vertex_is_rejected():
    triangle = (Vector3(math.nan, 0.0, 1.0), Vector3(0.1, 0.0, 1.0), Vector3(0.0, 0.1, 1.0))
    with pytest.raises(ValueError, match='finite'):
        summarize_angular_fov_boundary_projected_extent(triangle, NormalizedCalibration(), max_angle_rad=0.5)

def test_invalid_projected_evidence_is_rejected():
    triangle = (Vector3(0.0, 0.0, 2.0), Vector3(0.1, 0.0, 2.0), Vector3(0.0, 0.1, 2.0))
    with pytest.raises(ValueError, match='within FOV'):
        summarize_angular_fov_boundary_projected_extent(triangle, InvalidCalibration(), max_angle_rad=0.5)

def test_fully_inside_triangle_is_preserved_as_one_triangle():
    source = (Vector3(-0.2, -0.1, 2.0), Vector3(0.2, -0.1, 2.0), Vector3(0.0, 0.2, 2.0))
    result = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    assert result.has_measurable_polygon
    assert not result.is_degenerate
    assert len(result.ordered_vertices_camera) == 3
    assert len(result.triangles_camera) == 1
    assert set(result.ordered_vertices_camera) == set(source)

def test_one_inside_vertex_produces_safe_clipped_triangle():
    source = (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(0.0, 4.0, 2.0))
    result = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    assert len(result.ordered_vertices_camera) == 3
    assert len(result.triangles_camera) == 1
    assert all((point_is_within_angular_fov(point, max_angle_rad=0.5) for point in result.ordered_vertices_camera))

def test_two_inside_vertices_produce_quadrilateral_and_two_triangles():
    source = (Vector3(-0.2, 0.0, 2.0), Vector3(0.2, 0.0, 2.0), Vector3(4.0, 0.5, 2.0))
    result = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    assert len(result.ordered_vertices_camera) == 4
    assert len(result.triangles_camera) == 2
    assert all((point_is_within_angular_fov(point, max_angle_rad=0.5) for point in result.ordered_vertices_camera))

def test_outside_separated_triangle_returns_empty_polygon():
    result = build_safe_angular_fov_polygon((Vector3(10.0, 0.0, 2.0), Vector3(12.0, 0.0, 2.0), Vector3(11.0, 1.0, 2.0)), max_angle_rad=0.5)
    assert not result.has_measurable_polygon
    assert result.ordered_vertices_camera == ()
    assert result.triangles_camera == ()

def test_interior_only_cone_intersection_is_explicitly_empty():
    result = build_safe_angular_fov_polygon((Vector3(-3.0, -2.0, 2.0), Vector3(3.0, -2.0, 2.0), Vector3(0.0, 4.0, 2.0)), max_angle_rad=0.1)
    assert not result.has_measurable_polygon
    assert result.triangles_camera == ()

def test_duplicate_clipped_endpoints_are_removed_with_tolerance():
    source = (Vector3(-0.2, -0.1, 2.0), Vector3(0.2, -0.1, 2.0), Vector3(0.0, 0.2, 2.0))
    result = build_safe_angular_fov_polygon(source, max_angle_rad=None)
    assert len(result.ordered_vertices_camera) == 3

def test_reversed_winding_has_same_polygon_vertex_set():
    source = (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(0.0, 4.0, 2.0))
    forward = build_safe_angular_fov_polygon(source, max_angle_rad=0.5)
    reverse = build_safe_angular_fov_polygon((source[0], source[2], source[1]), max_angle_rad=0.5)
    assert set(reverse.ordered_vertices_camera) == set(forward.ordered_vertices_camera)
    assert len(reverse.triangles_camera) == len(forward.triangles_camera)

def test_collinear_source_triangle_is_rejected():
    with pytest.raises(ValueError, match='non-degenerate'):
        build_safe_angular_fov_polygon((Vector3(0.0, 0.0, 2.0), Vector3(1.0, 0.0, 2.0), Vector3(2.0, 0.0, 2.0)), max_angle_rad=0.5)

@pytest.mark.parametrize('count', (0, 2, 4))
def test_safe_wrong_vertex_count_is_rejected(count):
    source = tuple((Vector3(float(index), 0.0, 2.0) for index in range(count)))
    with pytest.raises(ValueError, match='exactly three'):
        build_safe_angular_fov_polygon(source, max_angle_rad=0.5)

@pytest.mark.parametrize('tolerance', (0.0, -1.0, math.nan, math.inf))
def test_invalid_point_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match='point_tolerance'):
        build_safe_angular_fov_polygon((Vector3(0.0, 0.0, 2.0), Vector3(1.0, 0.0, 2.0), Vector3(0.0, 1.0, 2.0)), max_angle_rad=0.5, point_tolerance=tolerance)

def test_safe_nonpositive_depth_is_rejected():
    with pytest.raises(ValueError, match='positive depth'):
        build_safe_angular_fov_polygon((Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 2.0), Vector3(0.0, 1.0, 2.0)), max_angle_rad=0.5)

def diag_theta(point: Vector3) -> float:
    return math.atan2(math.hypot(point.x, point.y), point.z)

def test_point_inside_cone():
    assert point_is_within_angular_fov(Vector3(0.2, 0.1, 2.0), max_angle_rad=0.5)

def test_point_on_cone_boundary_is_inside():
    maximum = 0.5
    point = Vector3(math.tan(maximum) * 2.0, 0.0, 2.0)
    assert point_is_within_angular_fov(point, max_angle_rad=maximum)

@pytest.mark.parametrize('z', (0.0, -1.0))
def test_nonpositive_depth_is_outside(z):
    assert not point_is_within_angular_fov(Vector3(0.0, 0.0, z), max_angle_rad=0.5)

def test_none_angle_accepts_finite_positive_depth_point():
    assert point_is_within_angular_fov(Vector3(100.0, 0.0, 1.0), max_angle_rad=None)

def test_fully_inside_triangle_has_original_edges():
    triangle = (Vector3(-0.2, -0.1, 2.0), Vector3(0.2, -0.1, 2.0), Vector3(0.0, 0.2, 2.0))
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.5)
    assert result.vertex_inside == (True, True, True)
    assert result.all_vertices_inside
    assert result.any_vertex_inside
    assert not result.every_edge_has_no_fov_segment
    assert tuple(((item.first_vertex_index, item.second_vertex_index) for item in result.edge_intersections)) == FOV_EDGE_INDEX_PAIRS
    for edge in result.edge_intersections:
        assert edge.clipped_segment is not None
        actual_first, actual_second = edge.clipped_segment
        expected_first = triangle[edge.first_vertex_index]
        expected_second = triangle[edge.second_vertex_index]
        for actual, expected in ((actual_first, expected_first), (actual_second, expected_second)):
            assert actual.x == pytest.approx(expected.x, abs=1e-12)
            assert actual.y == pytest.approx(expected.y, abs=1e-12)
            assert actual.z == pytest.approx(expected.z, abs=1e-12)

def test_one_inside_vertex_clips_two_incident_edges():
    maximum = 0.5
    triangle = (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(0.0, 4.0, 2.0))
    result = classify_triangle_angular_fov(triangle, max_angle_rad=maximum)
    assert result.vertex_inside == (True, False, False)
    assert not result.all_vertices_inside
    assert result.any_vertex_inside
    assert result.edge_intersections[0].clipped_segment is not None
    assert result.edge_intersections[1].clipped_segment is None
    assert result.edge_intersections[2].clipped_segment is not None
    for edge in (result.edge_intersections[0], result.edge_intersections[2]):
        clipped = edge.clipped_segment
        assert clipped is not None
        assert any((abs(diag_theta(point) - maximum) < 1e-10 for point in clipped))

def test_two_outside_vertices_edge_can_cross_cone():
    triangle = (Vector3(-4.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(4.0, 1.0, 2.0))
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.5)
    assert result.vertex_inside == (False, False, False)
    assert not result.any_vertex_inside
    assert result.edge_intersections[0].clipped_segment is not None
    assert not result.every_edge_has_no_fov_segment

def test_fully_outside_separated_triangle_has_no_edge_segments():
    triangle = (Vector3(10.0, 0.0, 2.0), Vector3(12.0, 0.0, 2.0), Vector3(11.0, 1.0, 2.0))
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.5)
    assert result.vertex_inside == (False, False, False)
    assert not result.any_vertex_inside
    assert result.every_edge_has_no_fov_segment

def test_no_edge_segment_flag_is_documented_as_non_proof_of_empty_interior():
    triangle = (Vector3(-4.0, -3.0, 2.0), Vector3(4.0, -3.0, 2.0), Vector3(0.0, 5.0, 2.0))
    result = classify_triangle_angular_fov(triangle, max_angle_rad=0.2)
    assert not result.any_vertex_inside
    assert not hasattr(result, 'triangle_outside')

@pytest.mark.parametrize('count', (0, 2, 4))
def test_diag_wrong_vertex_count_is_rejected(count):
    triangle = tuple((Vector3(float(i), 0.0, 1.0) for i in range(count)))
    with pytest.raises(ValueError, match='exactly three vertices'):
        classify_triangle_angular_fov(triangle, max_angle_rad=0.5)

@pytest.mark.parametrize('angle', (0.0, -1.0, math.pi / 2.0, math.nan, math.inf))
def test_invalid_angle_is_rejected(angle):
    with pytest.raises(ValueError, match='max_angle_rad'):
        point_is_within_angular_fov(Vector3(0.0, 0.0, 1.0), max_angle_rad=angle)

def test_nonfinite_point_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        point_is_within_angular_fov(Vector3(math.nan, 0.0, 1.0), max_angle_rad=0.5)

class BoundaryProjection:

    def __init__(self, u, v):
        self.u = u
        self.v = v
        self.positive_z = True
        self.within_fov = True
        self.valid = True

class NormalizedCalibration_2:

    def __init__(self, scale=100.0):
        self.scale = scale

    def project_camera_point(self, point):
        return BoundaryProjection(self.scale * point.x / point.z, self.scale * point.y / point.z)

def subdiv_inside_triangle():
    return (Vector3(-0.1, -0.1, 2.0), Vector3(0.1, -0.1, 2.0), Vector3(0.0, 0.1, 2.0))

def subdiv_mixed_triangle():
    return (Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(0.0, 4.0, 2.0))

def test_fully_inside_is_accepted_without_subdivision():
    result = subdivide_triangle_to_angular_fov(subdiv_inside_triangle(), max_angle_rad=0.5, maximum_depth=4)
    assert len(result.accepted_inside_triangles) == 1
    assert result.accepted_inside_triangles[0].vertices_camera == subdiv_inside_triangle()
    assert result.accepted_inside_triangles[0].subdivision_depth == 0
    assert result.rejected_outside_triangle_count == 0
    assert result.boundary_unresolved_triangles == ()
    assert result.maximum_depth_reached == 0
    assert not result.stopped_by_depth_limit

def test_none_fov_accepts_positive_depth_triangle():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=None, maximum_depth=0)
    assert len(result.accepted_inside_triangles) == 1
    assert not result.stopped_by_depth_limit

def test_mixed_triangle_subdivides():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=1)
    total = len(result.accepted_inside_triangles) + len(result.boundary_unresolved_triangles) + result.rejected_outside_triangle_count
    assert total == 4
    assert result.maximum_depth_reached == 1
    assert result.boundary_unresolved_triangles
    assert result.stopped_by_depth_limit

def test_depth_zero_mixed_triangle_is_unresolved():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=0)
    assert len(result.boundary_unresolved_triangles) == 1
    unresolved = result.boundary_unresolved_triangles[0]
    assert unresolved.subdivision_depth == 0
    assert unresolved.sample_inside_count > 0
    assert unresolved.edge_intersection_count > 0
    assert result.stopped_by_depth_limit

def test_separated_outside_triangle_is_rejected_at_limit():
    triangle = (Vector3(10.0, 0.0, 2.0), Vector3(12.0, 0.0, 2.0), Vector3(11.0, 1.0, 2.0))
    result = subdivide_triangle_to_angular_fov(triangle, max_angle_rad=0.5, maximum_depth=0)
    assert result.accepted_inside_triangles == ()
    assert result.boundary_unresolved_triangles == ()
    assert result.rejected_outside_triangle_count == 1
    assert not result.stopped_by_depth_limit

def test_outside_triangle_is_rejected_before_subdivision():
    triangle = (Vector3(10.0, 0.0, 2.0), Vector3(12.0, 0.0, 2.0), Vector3(11.0, 1.0, 2.0))
    result = subdivide_triangle_to_angular_fov(triangle, max_angle_rad=0.5, maximum_depth=8)
    assert result.rejected_outside_triangle_count == 1
    assert result.maximum_depth_reached == 0
    assert result.accepted_inside_triangles == ()
    assert result.boundary_unresolved_triangles == ()
    assert not result.stopped_by_depth_limit

def test_all_vertices_outside_but_interior_intersection_is_not_early_rejected():
    triangle = (Vector3(-3.0, -2.0, 2.0), Vector3(3.0, -2.0, 2.0), Vector3(0.0, 4.0, 2.0))
    result = subdivide_triangle_to_angular_fov(triangle, max_angle_rad=0.1, maximum_depth=1)
    assert result.accepted_inside_triangles or result.boundary_unresolved_triangles

def test_boundary_triangle_never_silently_becomes_inside():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=2)
    assert result.boundary_unresolved_triangles
    assert result.stopped_by_depth_limit

def test_child_winding_is_preserved_for_accepted_children():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=2)
    for item in result.accepted_inside_triangles:
        first, second, third = item.vertices_camera
        signed = (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x)
        assert signed > 0.0

def test_small_boundary_is_reported_as_approximated():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=8, calibration=NormalizedCalibration_2(scale=1.0), maximum_boundary_extent_px=10.0)
    assert result.boundary_approximated_triangles
    assert result.boundary_depth_limited_triangles == ()
    assert result.boundary_unmeasurable_triangles == ()
    assert not result.stopped_by_depth_limit

def test_large_boundary_at_depth_limit_is_reported_separately():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=0, calibration=NormalizedCalibration_2(scale=1000.0), maximum_boundary_extent_px=0.01)
    assert result.boundary_approximated_triangles == ()
    assert len(result.boundary_depth_limited_triangles) == 1
    assert result.boundary_unmeasurable_triangles == ()
    assert result.stopped_by_depth_limit

def test_unmeasurable_boundary_is_not_treated_as_zero_extent():
    triangle = (Vector3(-3.0, -2.0, 2.0), Vector3(3.0, -2.0, 2.0), Vector3(0.0, 4.0, 2.0))
    result = subdivide_triangle_to_angular_fov(triangle, max_angle_rad=0.1, maximum_depth=0, calibration=NormalizedCalibration_2(), maximum_boundary_extent_px=1.0)
    assert result.boundary_approximated_triangles == ()
    assert result.boundary_depth_limited_triangles == ()
    assert len(result.boundary_unmeasurable_triangles) == 1
    assert result.stopped_by_depth_limit

def test_legacy_unresolved_property_contains_two_unresolved_categories():
    result = subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=0)
    assert result.boundary_unresolved_triangles == result.boundary_depth_limited_triangles + result.boundary_unmeasurable_triangles

def test_boundary_limit_requires_calibration():
    with pytest.raises(ValueError, match='calibration is required'):
        subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=2, maximum_boundary_extent_px=1.0)

@pytest.mark.parametrize('limit', (0.0, -1.0, math.nan, math.inf))
def test_invalid_boundary_extent_limit_is_rejected(limit):
    with pytest.raises(ValueError, match='maximum_boundary_extent_px'):
        subdivide_triangle_to_angular_fov(subdiv_mixed_triangle(), max_angle_rad=0.5, maximum_depth=2, calibration=NormalizedCalibration_2(), maximum_boundary_extent_px=limit)

@pytest.mark.parametrize('count', (0, 2, 4))
def test_subdiv_wrong_vertex_count_is_rejected(count):
    triangle = tuple((Vector3(float(index), 0.0, 1.0) for index in range(count)))
    with pytest.raises(ValueError, match='exactly three vertices'):
        subdivide_triangle_to_angular_fov(triangle, max_angle_rad=0.5, maximum_depth=1)

@pytest.mark.parametrize('depth', (-1,))
def test_negative_depth_is_rejected(depth):
    with pytest.raises(ValueError, match='non-negative'):
        subdivide_triangle_to_angular_fov(subdiv_inside_triangle(), max_angle_rad=0.5, maximum_depth=depth)

@pytest.mark.parametrize('depth', (True, 1.5, '2'))
def test_noninteger_depth_is_rejected(depth):
    with pytest.raises(TypeError, match='integer'):
        subdivide_triangle_to_angular_fov(subdiv_inside_triangle(), max_angle_rad=0.5, maximum_depth=depth)

@pytest.mark.parametrize('z', (0.0, -1.0))
def test_nonpositive_vertex_depth_is_rejected(z):
    triangle = (Vector3(0.0, 0.0, z), Vector3(0.1, 0.0, 1.0), Vector3(0.0, 0.1, 1.0))
    with pytest.raises(ValueError, match='positive depth'):
        subdivide_triangle_to_angular_fov(triangle, max_angle_rad=0.5, maximum_depth=1)

def test_subdiv_nonfinite_vertex_is_rejected():
    triangle = (Vector3(math.nan, 0.0, 1.0), Vector3(0.1, 0.0, 1.0), Vector3(0.0, 0.1, 1.0))
    with pytest.raises(ValueError, match='finite'):
        subdivide_triangle_to_angular_fov(triangle, max_angle_rad=0.5, maximum_depth=1)

def bary_triangle():
    return (Point2D(0.0, 0.0), Point2D(4.0, 0.0), Point2D(0.0, 4.0))

def test_vertices_have_unit_basis_weights():
    source = bary_triangle()
    expected = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for point, weights in zip(source, expected):
        result = triangle_barycentric_coordinates(source, point)
        assert (result.first_weight, result.second_weight, result.third_weight) == pytest.approx(weights)
        assert result.inside_closed_triangle

def test_centroid_has_equal_weights():
    result = triangle_barycentric_coordinates(bary_triangle(), Point2D(4.0 / 3.0, 4.0 / 3.0))
    assert result.first_weight == pytest.approx(1.0 / 3.0)
    assert result.second_weight == pytest.approx(1.0 / 3.0)
    assert result.third_weight == pytest.approx(1.0 / 3.0)
    assert result.weight_sum == pytest.approx(1.0)
    assert result.inside_closed_triangle

def test_edge_point_is_inside_closed_triangle():
    result = triangle_barycentric_coordinates(bary_triangle(), Point2D(2.0, 2.0))
    assert result.first_weight == pytest.approx(0.0)
    assert result.inside_closed_triangle

def test_outside_point_has_negative_weight():
    result = triangle_barycentric_coordinates(bary_triangle(), Point2D(3.0, 3.0))
    assert min(result.first_weight, result.second_weight, result.third_weight) < 0.0
    assert not result.inside_closed_triangle
    assert result.weight_sum == pytest.approx(1.0)

def test_reversed_winding_preserves_geometric_weights_for_matching_vertices():
    source = bary_triangle()
    point = Point2D(1.0, 1.0)
    forward = triangle_barycentric_coordinates(source, point)
    reverse = triangle_barycentric_coordinates((source[0], source[2], source[1]), point)
    assert reverse.first_weight == pytest.approx(forward.first_weight)
    assert reverse.second_weight == pytest.approx(forward.third_weight)
    assert reverse.third_weight == pytest.approx(forward.second_weight)
    assert reverse.inside_closed_triangle == forward.inside_closed_triangle

def test_affine_reconstruction_matches_sample_point():
    source = bary_triangle()
    point = Point2D(0.75, 1.25)
    result = triangle_barycentric_coordinates(source, point)
    reconstructed_x = sum((weight * vertex.x for weight, vertex in zip((result.first_weight, result.second_weight, result.third_weight), source)))
    reconstructed_y = sum((weight * vertex.y for weight, vertex in zip((result.first_weight, result.second_weight, result.third_weight), source)))
    assert reconstructed_x == pytest.approx(point.x)
    assert reconstructed_y == pytest.approx(point.y)

def test_collinear_triangle_is_rejected():
    with pytest.raises(ValueError, match='non-degenerate'):
        triangle_barycentric_coordinates((Point2D(0.0, 0.0), Point2D(1.0, 1.0), Point2D(2.0, 2.0)), Point2D(1.0, 1.0))

def test_coincident_triangle_is_rejected():
    point = Point2D(1.0, 1.0)
    with pytest.raises(ValueError, match='non-degenerate'):
        triangle_barycentric_coordinates((point, point, point), point)

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_triangle_size_is_rejected(count):
    source = tuple((Point2D(float(index), 0.0) for index in range(count)))
    with pytest.raises(ValueError, match='exactly three'):
        triangle_barycentric_coordinates(source, Point2D(0.0, 0.0))

@pytest.mark.parametrize('epsilon', (0.0, -1.0, math.nan, math.inf))
def test_invalid_degeneracy_epsilon_is_rejected(epsilon):
    with pytest.raises(ValueError, match='degeneracy_epsilon'):
        triangle_barycentric_coordinates(bary_triangle(), Point2D(1.0, 1.0), degeneracy_epsilon=epsilon)

def test_nonfinite_triangle_coordinate_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        triangle_barycentric_coordinates((Point2D(math.nan, 0.0), Point2D(1.0, 0.0), Point2D(0.0, 1.0)), Point2D(0.0, 0.0))

def test_nonfinite_sample_coordinate_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        triangle_barycentric_coordinates(bary_triangle(), Point2D(math.inf, 0.0))

def test_large_translated_triangle_remains_valid():
    source = (Point2D(1000000.0, 1000000.0), Point2D(1000004.0, 1000000.0), Point2D(1000000.0, 1000004.0))
    result = triangle_barycentric_coordinates(source, Point2D(1000001.0, 1000001.0))
    assert result.inside_closed_triangle
    assert result.weight_sum == pytest.approx(1.0)

def test_inside_vertex_intersects_and_reports_vertex():
    result = triangle_intersects_angular_fov_cone((Vector3(0.1, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(4.0, 1.0, 2.0)), max_angle_rad=0.5)
    assert result.intersects
    assert result.minimizing_location_type == 'vertex'
    assert result.minimizing_feature_indices == (0,)

def test_separated_triangle_does_not_intersect():
    result = triangle_intersects_angular_fov_cone((Vector3(10.0, 0.0, 2.0), Vector3(12.0, 0.0, 2.0), Vector3(11.0, 1.0, 2.0)), max_angle_rad=0.5)
    assert not result.intersects
    assert result.margin_squared > 0.0

def test_edge_crossing_axis_reports_edge_or_interior():
    result = triangle_intersects_angular_fov_cone((Vector3(-4.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(4.0, 2.0, 2.0)), max_angle_rad=0.2)
    assert result.intersects
    assert result.minimum_normalized_radius_squared == pytest.approx(0.0)
    assert result.minimizing_location_type == 'edge'
    assert result.minimizing_feature_indices == (0, 1)

def test_origin_inside_normalized_triangle_reports_interior():
    result = triangle_intersects_angular_fov_cone((Vector3(-3.0, -2.0, 2.0), Vector3(3.0, -2.0, 2.0), Vector3(0.0, 4.0, 2.0)), max_angle_rad=0.1)
    assert result.intersects
    assert result.minimizing_location_type == 'interior'
    assert result.closest_normalized_point == (0.0, 0.0)

def test_cone_boundary_is_included():
    maximum = 0.5
    tangent = math.tan(maximum)
    result = triangle_intersects_angular_fov_cone((Vector3(tangent * 2.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), Vector3(4.0, 1.0, 2.0)), max_angle_rad=maximum)
    assert result.intersects
    assert result.margin_squared == pytest.approx(0.0, abs=1e-12)

def test_varying_depth_uses_perspective_normalization():
    result = triangle_intersects_angular_fov_cone((Vector3(1.0, 0.0, 10.0), Vector3(3.0, 1.0, 1.0), Vector3(3.0, -1.0, 1.0)), max_angle_rad=0.2)
    assert result.intersects
    assert result.minimizing_location_type == 'vertex'
    assert result.minimizing_feature_indices == (0,)

def test_degenerate_collinear_triangle_uses_edge_minimum():
    result = triangle_intersects_angular_fov_cone((Vector3(-2.0, 1.0, 2.0), Vector3(0.0, 1.0, 2.0), Vector3(2.0, 1.0, 2.0)), max_angle_rad=0.6)
    assert result.intersects
    assert result.minimum_normalized_radius_squared == pytest.approx(0.25)

def test_result_fields_are_consistent():
    maximum = 0.3
    result = triangle_intersects_angular_fov_cone((Vector3(2.0, 0.0, 2.0), Vector3(3.0, 0.0, 2.0), Vector3(2.0, 1.0, 2.0)), max_angle_rad=maximum)
    assert result.cone_radius_squared == pytest.approx(math.tan(maximum) ** 2)
    assert result.margin_squared == pytest.approx(result.minimum_normalized_radius_squared - result.cone_radius_squared)

@pytest.mark.parametrize('count', (0, 2, 4))
def test_cone_wrong_vertex_count_is_rejected(count):
    triangle = tuple((Vector3(float(index), 0.0, 1.0) for index in range(count)))
    with pytest.raises(ValueError, match='exactly three vertices'):
        triangle_intersects_angular_fov_cone(triangle, max_angle_rad=0.5)

@pytest.mark.parametrize('angle', (0.0, -1.0, math.pi / 2.0, math.nan, math.inf))
def test_cone_invalid_angle_is_rejected(angle):
    with pytest.raises(ValueError, match='max_angle_rad'):
        triangle_intersects_angular_fov_cone((Vector3(0.0, 0.0, 1.0), Vector3(0.1, 0.0, 1.0), Vector3(0.0, 0.1, 1.0)), max_angle_rad=angle)

@pytest.mark.parametrize('z', (0.0, -1.0))
def test_cone_nonpositive_depth_is_rejected(z):
    with pytest.raises(ValueError, match='positive depth'):
        triangle_intersects_angular_fov_cone((Vector3(0.0, 0.0, z), Vector3(0.1, 0.0, 1.0), Vector3(0.0, 0.1, 1.0)), max_angle_rad=0.5)

def test_nonfinite_coordinate_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        triangle_intersects_angular_fov_cone((Vector3(math.nan, 0.0, 1.0), Vector3(0.1, 0.0, 1.0), Vector3(0.0, 0.1, 1.0)), max_angle_rad=0.5)

def test_front_facing_triangle_is_selected():
    triangle = (Vector3(-1.0, -1.0, 5.0), Vector3(0.0, 1.0, 5.0), Vector3(1.0, -1.0, 5.0))
    assert is_triangle_front_facing(triangle) is True
    assert select_front_facing_triangles((triangle,)) == (triangle,)

def test_reversed_triangle_is_back_facing():
    triangle = (Vector3(-1.0, -1.0, 5.0), Vector3(1.0, -1.0, 5.0), Vector3(0.0, 1.0, 5.0))
    assert is_triangle_front_facing(triangle) is False
    assert select_front_facing_triangles((triangle,)) == ()

def test_edge_on_triangle_is_not_front_facing():
    triangle = (Vector3(1.0, -1.0, 4.0), Vector3(1.0, 1.0, 4.0), Vector3(1.0, 0.0, 6.0))
    assert is_triangle_front_facing(triangle) is False

def test_axis_aligned_box_in_front_exposes_camera_facing_surface():
    center = Vector3(0.0, 0.0, 5.0)
    local = (Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(-1, 1, -1), Vector3(1, 1, -1), Vector3(-1, -1, 1), Vector3(1, -1, 1), Vector3(-1, 1, 1), Vector3(1, 1, 1))
    corners = tuple((Vector3(p.x + center.x, p.y + center.y, p.z + center.z) for p in local))
    surfaces = triangulate_box_surfaces(corners)
    selected = select_front_facing_triangles((item.vertices for item in surfaces))
    assert len(selected) == 2
    negative_z = {item.vertices for item in surfaces if item.face_name == 'negative_z'}
    assert set(selected) == negative_z

def test_selection_preserves_input_order():
    front_first = (Vector3(-1, -1, 5), Vector3(0, 1, 5), Vector3(1, -1, 5))
    back = tuple(reversed(front_first))
    front_second = (Vector3(-2, -1, 8), Vector3(0, 2, 8), Vector3(2, -1, 8))
    assert select_front_facing_triangles((front_first, back, front_second)) == (front_first, front_second)

def test_triangle_normal_follows_winding():
    triangle = (Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1))
    normal = triangle_normal(triangle)
    reversed_normal = triangle_normal(tuple(reversed(triangle)))
    assert normal == Vector3(0, 0, 1)
    assert reversed_normal == Vector3(0, 0, -1)

@pytest.mark.parametrize('count', (0, 2, 4))
def test_facing_wrong_vertex_count_is_rejected(count):
    triangle = tuple((Vector3(float(i), 0.0, 1.0) for i in range(count)))
    with pytest.raises(ValueError, match='exactly three vertices'):
        is_triangle_front_facing(triangle)

def test_degenerate_triangle_is_rejected():
    triangle = (Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(2, 0, 1))
    with pytest.raises(ValueError, match='non-zero area'):
        is_triangle_front_facing(triangle)

@pytest.mark.parametrize('value', (math.nan, math.inf, -math.inf))
def test_non_finite_vertex_is_rejected(value):
    triangle = (Vector3(0, 0, 1), Vector3(1, value, 1), Vector3(0, 1, 1))
    with pytest.raises(ValueError, match='must be finite'):
        is_triangle_front_facing(triangle)

@pytest.mark.parametrize('tolerance', (-1.0, math.nan, math.inf))
def test_invalid_tolerance_is_rejected(tolerance):
    triangle = (Vector3(-1, -1, 5), Vector3(0, 1, 5), Vector3(1, -1, 5))
    with pytest.raises(ValueError, match='tolerance'):
        is_triangle_front_facing(triangle, tolerance=tolerance)
NEAR = 0.1

def near_area_xy(triangle) -> float:
    first, second, third = triangle
    return 0.5 * abs((second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x))

def near_signed_area_xy(triangle) -> float:
    first, second, third = triangle
    return 0.5 * ((second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x))

def test_fully_inside_triangle_is_preserved_exactly():
    triangle = (Vector3(0.0, 0.0, 1.0), Vector3(2.0, 0.0, 1.0), Vector3(0.0, 2.0, 1.0))
    assert clip_triangle_to_positive_z(triangle, near_plane_m=NEAR) == (triangle,)

def test_fully_outside_triangle_is_removed():
    triangle = (Vector3(0.0, 0.0, 0.0), Vector3(2.0, 0.0, 0.05), Vector3(0.0, 2.0, -1.0))
    assert clip_triangle_to_positive_z(triangle, near_plane_m=NEAR) == ()

def test_one_inside_vertex_produces_one_triangle():
    triangle = (Vector3(0.0, 0.0, 1.0), Vector3(2.0, 0.0, 0.0), Vector3(0.0, 2.0, 0.0))
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert len(clipped) == 1
    assert all((vertex.z >= NEAR for vertex in clipped[0]))
    assert sum((vertex.z == NEAR for vertex in clipped[0])) == 2

def test_two_inside_vertices_produce_two_triangles():
    triangle = (Vector3(0.0, 0.0, 1.0), Vector3(2.0, 0.0, 1.0), Vector3(0.0, 2.0, 0.0))
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert len(clipped) == 2
    assert all((vertex.z >= NEAR for item in clipped for vertex in item))

def test_vertices_exactly_on_near_plane_are_inside():
    triangle = (Vector3(0.0, 0.0, NEAR), Vector3(2.0, 0.0, NEAR), Vector3(0.0, 2.0, 1.0))
    assert clip_triangle_to_positive_z(triangle, near_plane_m=NEAR) == (triangle,)

def test_intersection_matches_existing_segment_clipper():
    inside = Vector3(1.0, 2.0, 1.0)
    outside = Vector3(5.0, 6.0, 0.0)
    segment = clip_segment_to_positive_z(inside, outside, near_plane_m=NEAR)
    assert segment is not None
    _, expected_intersection = segment
    polygon = clip_polygon_to_positive_z((inside, outside, Vector3(-1.0, 3.0, 0.0)), near_plane_m=NEAR)
    assert expected_intersection in polygon

def test_clipped_triangles_preserve_positive_winding():
    triangle = (Vector3(0.0, 0.0, 1.0), Vector3(2.0, 0.0, 1.0), Vector3(0.0, 2.0, 0.0))
    assert near_signed_area_xy(triangle) > 0.0
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert all((near_signed_area_xy(item) > 0.0 for item in clipped))

def test_clipped_area_matches_expected_polygon_area():
    triangle = (Vector3(0.0, 0.0, 1.0), Vector3(2.0, 0.0, 1.0), Vector3(0.0, 2.0, 0.0))
    clipped = clip_triangle_to_positive_z(triangle, near_plane_m=NEAR)
    assert sum((near_area_xy(item) for item in clipped)) == pytest.approx(1.98)

@pytest.mark.parametrize('near', (0.0, -1.0))
def test_non_positive_near_plane_is_rejected(near):
    triangle = (Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1))
    with pytest.raises(ValueError, match='must be positive'):
        clip_triangle_to_positive_z(triangle, near_plane_m=near)

@pytest.mark.parametrize('near', (math.nan, math.inf, -math.inf))
def test_non_finite_near_plane_is_rejected(near):
    triangle = (Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1))
    with pytest.raises(ValueError, match='must be finite'):
        clip_triangle_to_positive_z(triangle, near_plane_m=near)

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_triangle_vertex_count_is_rejected(count):
    vertices = tuple((Vector3(float(i), 0.0, 1.0) for i in range(count)))
    with pytest.raises(ValueError, match='exactly three vertices'):
        clip_triangle_to_positive_z(vertices, near_plane_m=NEAR)

def test_equal_vertex_depths_remain_constant():
    result = interpolate_perspective_camera_depth((5.0, 5.0, 5.0), (0.2, 0.3, 0.5))
    assert result.reciprocal_depth_per_m == pytest.approx(0.2)
    assert result.depth_m == pytest.approx(5.0)

@pytest.mark.parametrize(('weights', 'expected_depth'), (((1.0, 0.0, 0.0), 2.0), ((0.0, 1.0, 0.0), 4.0), ((0.0, 0.0, 1.0), 8.0)))
def test_triangle_vertices_reproduce_vertex_depth(weights, expected_depth):
    result = interpolate_perspective_camera_depth((2.0, 4.0, 8.0), weights)
    assert result.depth_m == pytest.approx(expected_depth)

def test_edge_sample_uses_reciprocal_depth():
    result = interpolate_perspective_camera_depth((2.0, 4.0, 8.0), (0.5, 0.5, 0.0))
    assert result.reciprocal_depth_per_m == pytest.approx(0.375)
    assert result.depth_m == pytest.approx(8.0 / 3.0)
    assert result.depth_m != pytest.approx(3.0)

def test_centroid_uses_harmonic_mean_of_depths():
    result = interpolate_perspective_camera_depth((2.0, 4.0, 8.0), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    expected = 3.0 / (1.0 / 2.0 + 1.0 / 4.0 + 1.0 / 8.0)
    assert result.depth_m == pytest.approx(expected)

def test_matching_vertex_and_weight_reordering_preserves_result():
    forward = interpolate_perspective_camera_depth((2.0, 4.0, 8.0), (0.2, 0.3, 0.5))
    reversed_order = interpolate_perspective_camera_depth((2.0, 8.0, 4.0), (0.2, 0.5, 0.3))
    assert reversed_order == forward

def test_small_weight_roundoff_within_tolerance_is_accepted():
    result = interpolate_perspective_camera_depth((2.0, 4.0, 8.0), (-1e-13, 0.5, 0.5000000000001), weight_tolerance=1e-12)
    assert result.depth_m > 0.0

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_depth_count_is_rejected(count):
    with pytest.raises(ValueError, match='exactly three'):
        interpolate_perspective_camera_depth(tuple((1.0 for _ in range(count))), (1.0, 0.0, 0.0))

@pytest.mark.parametrize('count', (0, 2, 4))
def test_wrong_weight_count_is_rejected(count):
    with pytest.raises(ValueError, match='exactly three'):
        interpolate_perspective_camera_depth((1.0, 2.0, 3.0), tuple((1.0 / count for _ in range(count))) if count else ())

@pytest.mark.parametrize('depths', ((0.0, 2.0, 3.0), (-1.0, 2.0, 3.0)))
def test_depth_nonpositive_depth_is_rejected(depths):
    with pytest.raises(ValueError, match='positive'):
        interpolate_perspective_camera_depth(depths, (1.0, 0.0, 0.0))

@pytest.mark.parametrize('value', (math.nan, math.inf, -math.inf))
def test_nonfinite_depth_is_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        interpolate_perspective_camera_depth((value, 2.0, 3.0), (1.0, 0.0, 0.0))

@pytest.mark.parametrize('value', (math.nan, math.inf, -math.inf))
def test_nonfinite_weight_is_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        interpolate_perspective_camera_depth((1.0, 2.0, 3.0), (value, 0.0, 1.0))

def test_weights_not_summing_to_one_are_rejected():
    with pytest.raises(ValueError, match='sum to one'):
        interpolate_perspective_camera_depth((1.0, 2.0, 3.0), (0.2, 0.3, 0.4))

@pytest.mark.parametrize('weights', ((-0.1, 0.5, 0.6), (1.1, -0.05, -0.05)))
def test_weights_outside_closed_triangle_are_rejected(weights):
    with pytest.raises(ValueError, match='closed triangle'):
        interpolate_perspective_camera_depth((1.0, 2.0, 3.0), weights)

@pytest.mark.parametrize('tolerance', (0.0, -1.0, math.nan, math.inf))
def test_invalid_weight_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match='weight_tolerance'):
        interpolate_perspective_camera_depth((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), weight_tolerance=tolerance)

def box_subtract(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(first.x - second.x, first.y - second.y, first.z - second.z)

def box_cross(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x)

def box_dot(first: Vector3, second: Vector3) -> float:
    return first.x * second.x + first.y * second.y + first.z * second.z

def test_face_and_triangle_counts_are_fixed():
    assert len(BOX_FACE_INDEX_QUADS) == 6
    assert len(BOX_TRIANGLE_INDEX_TRIPLES) == 12
    assert len(triangulate_box_surfaces(local_box_corners(Vector3(2, 4, 6)))) == 12

def test_each_face_produces_exactly_two_triangles():
    face_names = [name for name, _ in BOX_FACE_INDEX_QUADS]
    triangle_names = [name for name, _ in BOX_TRIANGLE_INDEX_TRIPLES]
    assert len(set(face_names)) == 6
    for face_name in face_names:
        assert triangle_names.count(face_name) == 2

def test_all_indices_are_valid_and_each_corner_is_used():
    indices = [index for _, triangle in BOX_TRIANGLE_INDEX_TRIPLES for index in triangle]
    assert set(indices) == set(range(8))
    assert all((0 <= index < 8 for index in indices))

def test_face_boundaries_match_existing_box_edges():
    undirected_edges = {frozenset(edge) for edge in BOX_EDGE_INDEX_PAIRS}
    for _, quad in BOX_FACE_INDEX_QUADS:
        boundary = zip(quad, quad[1:] + quad[:1])
        assert all((frozenset(edge) in undirected_edges for edge in boundary))

def test_triangle_winding_points_away_from_box_center():
    corners = local_box_corners(Vector3(2.0, 4.0, 6.0))
    for triangle in triangulate_box_surfaces(corners):
        first, second, third = triangle.vertices
        normal = box_cross(box_subtract(second, first), box_subtract(third, first))
        centroid = Vector3((first.x + second.x + third.x) / 3.0, (first.y + second.y + third.y) / 3.0, (first.z + second.z + third.z) / 3.0)
        assert box_dot(normal, centroid) > 0.0, triangle

def test_triangles_preserve_supplied_transformed_vertices():
    corners = tuple((Vector3(float(i), float(i + 10), float(i + 20)) for i in range(8)))
    triangles = triangulate_box_surfaces(corners)
    for triangle in triangles:
        assert triangle.vertices == tuple((corners[i] for i in triangle.corner_indices))

@pytest.mark.parametrize('corner_count', (0, 7, 9))
def test_wrong_corner_count_is_rejected(corner_count):
    corners = tuple((Vector3(float(i), 0.0, 0.0) for i in range(corner_count)))
    with pytest.raises(ValueError, match='exactly eight corners'):
        triangulate_box_surfaces(corners)

def camera_translated_box(center: Vector3, half_extent: float=1.0):
    return tuple((Vector3(center.x + x * half_extent, center.y + y * half_extent, center.z + z * half_extent) for z in (-1.0, 1.0) for y in (-1.0, 1.0) for x in (-1.0, 1.0)))

def test_box_fully_in_front_returns_camera_facing_near_surface():
    corners = camera_translated_box(Vector3(0.0, 0.0, 5.0))
    result = prepare_camera_facing_box_triangles(corners)
    assert len(result) == 2
    assert {item.face_name for item in result} == {'negative_z'}
    assert all((vertex.z == 4.0 for item in result for vertex in item.vertices_camera))

def test_output_matches_independent_facing_selection_when_no_clipping_occurs():
    corners = camera_translated_box(Vector3(2.0, 1.0, 8.0))
    source = triangulate_box_surfaces(corners)
    expected = [item for item in source if is_triangle_front_facing(item.vertices)]
    actual = prepare_camera_facing_box_triangles(corners)
    assert len(actual) == len(expected)
    assert [item.face_name for item in actual] == [item.face_name for item in expected]
    assert [item.vertices_camera for item in actual] == [item.vertices for item in expected]

def test_composition_applies_near_clipping_after_facing_selection():
    corners = camera_translated_box(Vector3(2.5, 0.0, 0.75))
    result = prepare_camera_facing_box_triangles(corners, near_plane_m=0.1)
    assert result
    assert all((vertex.z >= 0.1 for item in result for vertex in item.vertices_camera))
    assert any((vertex.z == 0.1 for item in result for vertex in item.vertices_camera))

def test_source_metadata_is_preserved_when_clipping_splits_triangle():
    corners = camera_translated_box(Vector3(2.5, 0.0, 0.75))
    result = prepare_camera_facing_box_triangles(corners, near_plane_m=0.1)
    groups = {}
    for item in result:
        key = (item.face_name, item.source_corner_indices)
        groups.setdefault(key, 0)
        groups[key] += 1
    assert any((count == 2 for count in groups.values()))

def test_camera_inside_box_has_no_outward_front_facing_surface():
    corners = camera_translated_box(Vector3(0.0, 0.0, 0.75))
    assert prepare_camera_facing_box_triangles(corners, near_plane_m=0.1) == ()

def test_box_fully_behind_near_plane_produces_no_triangles():
    corners = camera_translated_box(Vector3(0.0, 0.0, -5.0))
    assert prepare_camera_facing_box_triangles(corners, near_plane_m=0.1) == ()

def test_output_order_is_deterministic():
    corners = camera_translated_box(Vector3(2.0, -1.0, 8.0))
    first = prepare_camera_facing_box_triangles(corners)
    second = prepare_camera_facing_box_triangles(corners)
    assert first == second

@pytest.mark.parametrize('count', (0, 7, 9))
def test_camera_wrong_corner_count_is_rejected(count):
    corners = tuple((Vector3(float(index), 0.0, 1.0) for index in range(count)))
    with pytest.raises(ValueError, match='exactly eight corners'):
        prepare_camera_facing_box_triangles(corners)

@pytest.mark.parametrize('near', (0.0, -1.0, math.nan, math.inf))
def test_invalid_near_plane_is_rejected(near):
    corners = camera_translated_box(Vector3(0.0, 0.0, 5.0))
    with pytest.raises(ValueError, match='near_plane_m'):
        prepare_camera_facing_box_triangles(corners, near_plane_m=near)

@pytest.mark.parametrize('tolerance', (-1.0, math.nan, math.inf))
def test_invalid_facing_tolerance_is_rejected(tolerance):
    corners = camera_translated_box(Vector3(0.0, 0.0, 5.0))
    with pytest.raises(ValueError, match='facing_tolerance'):
        prepare_camera_facing_box_triangles(corners, facing_tolerance=tolerance)
