from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from camera_projection_v01 import Vector3
from ftheta_triangle_projected_extent_v01 import (
    summarize_ftheta_triangle_projected_extent,
)


@dataclass(frozen=True)
class FakeProjection:
    u: float
    v: float
    positive_z: bool = True
    within_fov: bool = True
    valid: bool = True


class LinearCalibration:
    def project_camera_point(self, point):
        return FakeProjection(10.0 * point.x + 100.0, 20.0 * point.y + 50.0)


class OutsideFovCalibration:
    def project_camera_point(self, point):
        return FakeProjection(point.x, point.y, within_fov=False, valid=False)


class NonfiniteCalibration:
    def project_camera_point(self, point):
        return FakeProjection(math.nan, point.y)


def triangle():
    return (
        Vector3(0.0, 0.0, 5.0),
        Vector3(3.0, 0.0, 5.0),
        Vector3(0.0, 2.0, 5.0),
    )


def test_extent_values_for_right_triangle():
    result = summarize_ftheta_triangle_projected_extent(
        triangle(), LinearCalibration()
    )
    assert result.min_u_px == pytest.approx(100.0)
    assert result.max_u_px == pytest.approx(130.0)
    assert result.min_v_px == pytest.approx(50.0)
    assert result.max_v_px == pytest.approx(90.0)
    assert result.width_px == pytest.approx(30.0)
    assert result.height_px == pytest.approx(40.0)
    assert result.maximum_projected_edge_length_px == pytest.approx(50.0)
    assert result.projected_triangle_area_px2 == pytest.approx(600.0)


def test_vertex_projection_order_is_preserved():
    result = summarize_ftheta_triangle_projected_extent(
        triangle(), LinearCalibration()
    )
    assert [(item.u, item.v) for item in result.vertex_projections] == [
        (100.0, 50.0),
        (130.0, 50.0),
        (100.0, 90.0),
    ]


def test_reversed_winding_keeps_positive_area_and_same_extent():
    source = triangle()
    forward = summarize_ftheta_triangle_projected_extent(source, LinearCalibration())
    reverse = summarize_ftheta_triangle_projected_extent(
        (source[0], source[2], source[1]), LinearCalibration()
    )
    assert reverse.projected_triangle_area_px2 == pytest.approx(
        forward.projected_triangle_area_px2
    )
    assert reverse.width_px == pytest.approx(forward.width_px)
    assert reverse.height_px == pytest.approx(forward.height_px)


def test_degenerate_triangle_has_zero_area():
    result = summarize_ftheta_triangle_projected_extent(
        (
            Vector3(0.0, 0.0, 5.0),
            Vector3(1.0, 0.0, 5.0),
            Vector3(2.0, 0.0, 5.0),
        ),
        LinearCalibration(),
    )
    assert result.projected_triangle_area_px2 == pytest.approx(0.0)
    assert result.height_px == pytest.approx(0.0)
    assert result.maximum_projected_edge_length_px == pytest.approx(20.0)


def test_vertex_exactly_on_near_plane_is_accepted():
    source = (
        Vector3(0.0, 0.0, 0.1),
        Vector3(1.0, 0.0, 1.0),
        Vector3(0.0, 1.0, 1.0),
    )
    result = summarize_ftheta_triangle_projected_extent(
        source, LinearCalibration(), near_plane_m=0.1
    )
    assert result.width_px == pytest.approx(10.0)


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    source = tuple(Vector3(float(index), 0.0, 1.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        summarize_ftheta_triangle_projected_extent(source, LinearCalibration())


@pytest.mark.parametrize("near", (0.0, -1.0, math.nan, math.inf))
def test_invalid_near_plane_is_rejected(near):
    with pytest.raises(ValueError, match="near_plane_m"):
        summarize_ftheta_triangle_projected_extent(
            triangle(), LinearCalibration(), near_plane_m=near
        )


def test_vertex_behind_near_plane_is_rejected_before_projection():
    source = (
        Vector3(0.0, 0.0, 0.09),
        Vector3(1.0, 0.0, 1.0),
        Vector3(0.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="behind near_plane_m"):
        summarize_ftheta_triangle_projected_extent(
            source, LinearCalibration(), near_plane_m=0.1
        )


def test_nonfinite_vertex_is_rejected():
    source = (
        Vector3(math.nan, 0.0, 1.0),
        Vector3(1.0, 0.0, 1.0),
        Vector3(0.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="finite"):
        summarize_ftheta_triangle_projected_extent(source, LinearCalibration())


def test_outside_fov_projection_is_rejected():
    with pytest.raises(ValueError, match="within FOV"):
        summarize_ftheta_triangle_projected_extent(
            triangle(), OutsideFovCalibration()
        )


def test_nonfinite_projection_is_rejected():
    with pytest.raises(ValueError, match="pixel coordinates must be finite"):
        summarize_ftheta_triangle_projected_extent(
            triangle(), NonfiniteCalibration()
        )
