from __future__ import annotations

from dataclasses import dataclass

import pytest

from step7.camera_projection_v01 import Vector3
from step7.fov_subdivided_triangle_depth_samples_v01 import (
    sample_fov_subdivided_camera_triangle_depths,
)


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


def call(triangle, *, depth=4, extent=1.0, calibration=None):
    calibration = calibration or NormalizedCalibration()
    return sample_fov_subdivided_camera_triangle_depths(
        triangle,
        calibration,
        max_angle_rad=0.5,
        maximum_depth=depth,
        maximum_boundary_extent_px=extent,
        image_width_px=8,
        image_height_px=8,
        raster_width=8,
        raster_height=8,
    )


def inside_triangle():
    return (
        Vector3(-1.0, -1.0, 4.0),
        Vector3(1.0, -1.0, 4.0),
        Vector3(-1.0, 1.0, 4.0),
    )


def boundary_triangle():
    return (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 2.0),
        Vector3(0.0, 4.0, 2.0),
    )


def test_fully_inside_triangle_is_sampled_directly():
    result = call(inside_triangle())
    assert result.sampled_inside_triangle_count == 1
    assert result.sampled_boundary_triangle_count == 0
    assert len(result.all_triangle_samples) == 1
    assert result.all_triangle_samples[0].center_sampled_cell_count > 0
    assert result.unresolved_boundary_count == 0


def test_fully_outside_triangle_is_rejected_without_samples():
    result = call((
        Vector3(10.0, 0.0, 2.0),
        Vector3(12.0, 0.0, 2.0),
        Vector3(11.0, 1.0, 2.0),
    ))
    assert result.all_triangle_samples == ()
    assert result.subdivision.rejected_outside_triangle_count == 1


def test_boundary_approximation_is_converted_to_safe_samples():
    result = call(
        boundary_triangle(),
        depth=4,
        extent=100.0,
        calibration=NormalizedCalibration(scale=1.0),
    )
    assert result.subdivision.boundary_approximated_triangles
    assert result.measurable_boundary_polygon_count > 0
    assert result.sampled_boundary_triangle_count > 0
    assert result.boundary_triangle_samples
    assert result.unmeasurable_boundary_polygon_count == 0
    assert result.degenerate_boundary_polygon_count == 0


def test_all_samples_concatenate_inside_then_boundary_samples():
    result = call(
        boundary_triangle(),
        depth=2,
        extent=1.0,
        calibration=NormalizedCalibration(scale=1.0),
    )
    assert result.all_triangle_samples == (
        result.accepted_triangle_samples
        + result.boundary_triangle_samples
    )


def test_depth_limited_boundary_is_reported_as_unresolved():
    result = call(
        boundary_triangle(),
        depth=0,
        extent=1e-6,
        calibration=NormalizedCalibration(scale=100.0),
    )
    assert result.unresolved_boundary_count == 1
    assert result.all_triangle_samples == ()


def test_all_generated_sample_depths_are_positive():
    result = call(
        boundary_triangle(),
        depth=4,
        extent=100.0,
        calibration=NormalizedCalibration(scale=4.0),
    )
    depths = [
        sample.depth_m
        for triangle_samples in result.all_triangle_samples
        for sample in triangle_samples.center_sampled_depths
    ]
    assert depths
    assert all(depth > 0.0 for depth in depths)


def test_boundary_limit_validation_is_delegated():
    with pytest.raises(ValueError, match="maximum_boundary_extent_px"):
        call(inside_triangle(), extent=0.0)


def test_wrong_triangle_size_is_delegated():
    with pytest.raises(ValueError, match="exactly three"):
        call((Vector3(0.0, 0.0, 2.0),) * 2)
