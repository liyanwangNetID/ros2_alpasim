from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from camera_projection_v01 import Vector3
from camera_triangle_depth_samples_v01 import sample_camera_triangle_depths
from projected_triangle_raster_cells_v01 import RasterCell


@dataclass(frozen=True)
class FakeProjection:
    u: float
    v: float
    positive_z: bool = True
    within_fov: bool = True
    valid: bool = True


class LinearCalibration:
    def project_camera_point(self, point):
        return FakeProjection(point.x, point.y)


class OutsideFovCalibration:
    def project_camera_point(self, point):
        return FakeProjection(point.x, point.y, within_fov=False, valid=False)


class NonfiniteCalibration:
    def project_camera_point(self, point):
        return FakeProjection(math.nan, point.y)


def triangle():
    return (
        Vector3(0.0, 0.0, 2.0),
        Vector3(4.0, 0.0, 4.0),
        Vector3(0.0, 4.0, 8.0),
    )


def test_camera_triangle_projects_and_samples_depth():
    result = sample_camera_triangle_depths(
        triangle(), LinearCalibration(),
        image_width_px=8, image_height_px=8,
        raster_width=8, raster_height=8,
    )
    item = next(
        sample for sample in result.center_sampled_depths
        if sample.cell == RasterCell(0, 0)
    )
    assert item.barycentric_weights == pytest.approx((0.75, 0.125, 0.125))
    expected_reciprocal = 0.75 / 2.0 + 0.125 / 4.0 + 0.125 / 8.0
    assert item.depth_m == pytest.approx(1.0 / expected_reciprocal)


def test_projection_order_matches_camera_vertex_depth_order():
    source = triangle()
    forward = sample_camera_triangle_depths(
        source, LinearCalibration(),
        image_width_px=8, image_height_px=8,
        raster_width=8, raster_height=8,
    )
    reverse = sample_camera_triangle_depths(
        (source[0], source[2], source[1]), LinearCalibration(),
        image_width_px=8, image_height_px=8,
        raster_width=8, raster_height=8,
    )
    forward_map = {item.cell: item.depth_m for item in forward.center_sampled_depths}
    reverse_map = {item.cell: item.depth_m for item in reverse.center_sampled_depths}
    assert reverse_map == pytest.approx(forward_map)


def test_downsampled_raster_is_delegated_correctly():
    result = sample_camera_triangle_depths(
        (
            Vector3(2.0, 2.0, 5.0),
            Vector3(6.0, 2.0, 5.0),
            Vector3(2.0, 6.0, 5.0),
        ),
        LinearCalibration(),
        image_width_px=8, image_height_px=8,
        raster_width=4, raster_height=2,
    )
    item = next(
        sample for sample in result.center_sampled_depths
        if sample.cell == RasterCell(1, 0)
    )
    assert item.sample_u_px == pytest.approx(3.0)
    assert item.sample_v_px == pytest.approx(2.0)
    assert item.depth_m == pytest.approx(5.0)


def test_vertex_exactly_on_near_plane_is_accepted():
    result = sample_camera_triangle_depths(
        (
            Vector3(0.0, 0.0, 0.1),
            Vector3(2.0, 0.0, 1.0),
            Vector3(0.0, 2.0, 1.0),
        ),
        LinearCalibration(),
        image_width_px=4, image_height_px=4,
        raster_width=4, raster_height=4,
        near_plane_m=0.1,
    )
    assert result.center_sampled_depths


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_camera_point_count_is_rejected(count):
    source = tuple(Vector3(float(index), 0.0, 1.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly three camera points"):
        sample_camera_triangle_depths(
            source, LinearCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=8, raster_height=8,
        )


@pytest.mark.parametrize("near", (0.0, -1.0, math.nan, math.inf))
def test_invalid_near_plane_is_rejected(near):
    with pytest.raises(ValueError, match="near_plane_m"):
        sample_camera_triangle_depths(
            triangle(), LinearCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=8, raster_height=8,
            near_plane_m=near,
        )


def test_vertex_behind_near_plane_is_rejected_before_projection():
    with pytest.raises(ValueError, match="behind near_plane_m"):
        sample_camera_triangle_depths(
            (
                Vector3(0.0, 0.0, 0.09),
                Vector3(1.0, 0.0, 1.0),
                Vector3(0.0, 1.0, 1.0),
            ),
            LinearCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=8, raster_height=8,
            near_plane_m=0.1,
        )


def test_nonfinite_camera_coordinate_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        sample_camera_triangle_depths(
            (
                Vector3(math.nan, 0.0, 1.0),
                Vector3(1.0, 0.0, 1.0),
                Vector3(0.0, 1.0, 1.0),
            ),
            LinearCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=8, raster_height=8,
        )


def test_outside_fov_projection_is_rejected():
    with pytest.raises(ValueError, match="within FOV"):
        sample_camera_triangle_depths(
            triangle(), OutsideFovCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=8, raster_height=8,
        )


def test_nonfinite_projection_is_rejected():
    with pytest.raises(ValueError, match="pixel coordinates must be finite"):
        sample_camera_triangle_depths(
            triangle(), NonfiniteCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=8, raster_height=8,
        )


def test_degenerate_projected_triangle_is_rejected_by_sampler():
    with pytest.raises(ValueError, match="non-degenerate"):
        sample_camera_triangle_depths(
            (
                Vector3(0.0, 0.0, 1.0),
                Vector3(1.0, 1.0, 2.0),
                Vector3(2.0, 2.0, 3.0),
            ),
            LinearCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=8, raster_height=8,
        )


def test_invalid_raster_dimensions_are_delegated():
    with pytest.raises(ValueError, match="positive"):
        sample_camera_triangle_depths(
            triangle(), LinearCalibration(),
            image_width_px=8, image_height_px=8,
            raster_width=0, raster_height=8,
        )
