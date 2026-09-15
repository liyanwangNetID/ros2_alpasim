from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from step7.camera_projection_v01 import Vector3
from step7.ftheta_fov_boundary_projected_extent_v01 import (
    summarize_angular_fov_boundary_projected_extent,
)


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
    triangle = (
        Vector3(-0.2, -0.1, 2.0),
        Vector3(0.2, -0.1, 2.0),
        Vector3(0.0, 0.2, 2.0),
    )
    result = summarize_angular_fov_boundary_projected_extent(
        triangle, NormalizedCalibration(), max_angle_rad=0.5
    )
    assert result.point_count == 3
    assert result.has_measurable_extent
    assert result.width_px == pytest.approx(20.0)
    assert result.height_px == pytest.approx(15.0)
    assert result.maximum_pair_distance_px is not None


def test_one_inside_vertex_collects_boundary_intersections():
    triangle = (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )
    result = summarize_angular_fov_boundary_projected_extent(
        triangle, NormalizedCalibration(), max_angle_rad=0.5
    )
    assert result.point_count == 3
    assert result.has_measurable_extent
    assert result.maximum_pair_distance_px is not None
    assert result.maximum_pair_distance_px > 0.0


def test_two_outside_endpoints_crossing_cone_collects_two_points():
    triangle = (
        Vector3(-4.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(4.0, 2.0, 2.0),
    )
    result = summarize_angular_fov_boundary_projected_extent(
        triangle, NormalizedCalibration(), max_angle_rad=0.2
    )
    assert result.point_count >= 2
    assert result.has_measurable_extent


def test_separated_outside_triangle_has_no_extent():
    triangle = (
        Vector3(10.0, 0.0, 2.0),
        Vector3(12.0, 0.0, 2.0),
        Vector3(11.0, 1.0, 2.0),
    )
    result = summarize_angular_fov_boundary_projected_extent(
        triangle, NormalizedCalibration(), max_angle_rad=0.5
    )
    assert result.point_count == 0
    assert not result.has_measurable_extent
    assert result.width_px is None
    assert result.height_px is None
    assert result.maximum_pair_distance_px is None


def test_none_fov_deduplicates_shared_edge_endpoints():
    triangle = (
        Vector3(0.0, 0.0, 2.0),
        Vector3(1.0, 0.0, 2.0),
        Vector3(0.0, 1.0, 2.0),
    )
    result = summarize_angular_fov_boundary_projected_extent(
        triangle, NormalizedCalibration(), max_angle_rad=None
    )
    assert result.point_count == 3


def test_single_unique_point_has_zero_extent():
    point = Vector3(0.0, 0.0, 2.0)
    result = summarize_angular_fov_boundary_projected_extent(
        (point, point, point), NormalizedCalibration(), max_angle_rad=0.5
    )
    assert result.point_count == 1
    assert result.width_px == pytest.approx(0.0)
    assert result.height_px == pytest.approx(0.0)
    assert result.maximum_pair_distance_px == pytest.approx(0.0)


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    triangle = tuple(Vector3(float(index), 0.0, 1.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        summarize_angular_fov_boundary_projected_extent(
            triangle, NormalizedCalibration(), max_angle_rad=0.5
        )


@pytest.mark.parametrize("z", (0.0, -1.0))
def test_nonpositive_depth_is_rejected(z):
    triangle = (
        Vector3(0.0, 0.0, z),
        Vector3(0.1, 0.0, 1.0),
        Vector3(0.0, 0.1, 1.0),
    )
    with pytest.raises(ValueError, match="positive depth"):
        summarize_angular_fov_boundary_projected_extent(
            triangle, NormalizedCalibration(), max_angle_rad=0.5
        )


def test_nonfinite_vertex_is_rejected():
    triangle = (
        Vector3(math.nan, 0.0, 1.0),
        Vector3(0.1, 0.0, 1.0),
        Vector3(0.0, 0.1, 1.0),
    )
    with pytest.raises(ValueError, match="finite"):
        summarize_angular_fov_boundary_projected_extent(
            triangle, NormalizedCalibration(), max_angle_rad=0.5
        )


def test_invalid_projected_evidence_is_rejected():
    triangle = (
        Vector3(0.0, 0.0, 2.0),
        Vector3(0.1, 0.0, 2.0),
        Vector3(0.0, 0.1, 2.0),
    )
    with pytest.raises(ValueError, match="within FOV"):
        summarize_angular_fov_boundary_projected_extent(
            triangle, InvalidCalibration(), max_angle_rad=0.5
        )
