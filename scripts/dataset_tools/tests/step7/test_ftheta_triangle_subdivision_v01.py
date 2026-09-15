from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from step7.camera_projection_v01 import Vector3
from step7.ftheta_triangle_subdivision_v01 import subdivide_ftheta_triangle_adaptive


@dataclass(frozen=True)
class FakeProjection:
    u: float
    v: float
    positive_z: bool = True
    within_fov: bool = True
    valid: bool = True


class LinearCalibration:
    def __init__(self):
        self.calls = 0

    def project_camera_point(self, point):
        self.calls += 1
        return FakeProjection(10.0 * point.x, 20.0 * point.y)


class CurvedCalibration:
    def __init__(self):
        self.calls = 0

    def project_camera_point(self, point):
        self.calls += 1
        return FakeProjection(point.x * point.x, point.y * point.y)


class InvalidCalibration:
    def project_camera_point(self, point):
        return FakeProjection(0.0, 0.0, within_fov=False, valid=False)


def triangle():
    return (
        Vector3(0.0, 0.0, 5.0),
        Vector3(2.0, 0.0, 5.0),
        Vector3(0.0, 2.0, 5.0),
    )


def signed_area_xy(vertices):
    first, second, third = vertices
    return 0.5 * (
        (second.x - first.x) * (third.y - first.y)
        - (second.y - first.y) * (third.x - first.x)
    )


def test_linear_projection_returns_original_triangle():
    source = triangle()
    result = subdivide_ftheta_triangle_adaptive(
        source,
        LinearCalibration(),
        maximum_projection_error_px=0.01,
        maximum_depth=4,
    )
    assert len(result.triangles) == 1
    assert result.triangles[0].vertices_camera == source
    assert result.triangles[0].subdivision_depth == 0
    assert result.maximum_observed_error_px == pytest.approx(0.0)
    assert result.stopped_by_depth_limit is False


def test_curved_projection_subdivides_into_four_children():
    result = subdivide_ftheta_triangle_adaptive(
        triangle(),
        CurvedCalibration(),
        maximum_projection_error_px=0.6,
        maximum_depth=4,
    )
    assert len(result.triangles) == 4
    assert {item.subdivision_depth for item in result.triangles} == {1}
    assert result.maximum_depth_reached == 1
    assert result.stopped_by_depth_limit is False


def test_tighter_error_produces_at_least_as_many_leaf_triangles():
    loose = subdivide_ftheta_triangle_adaptive(
        triangle(), CurvedCalibration(),
        maximum_projection_error_px=0.6, maximum_depth=5,
    )
    tight = subdivide_ftheta_triangle_adaptive(
        triangle(), CurvedCalibration(),
        maximum_projection_error_px=0.1, maximum_depth=5,
    )
    assert len(tight.triangles) >= len(loose.triangles)
    assert len(tight.triangles) > 4


def test_child_triangles_preserve_positive_winding():
    result = subdivide_ftheta_triangle_adaptive(
        triangle(), CurvedCalibration(),
        maximum_projection_error_px=0.1, maximum_depth=3,
    )
    assert all(signed_area_xy(item.vertices_camera) > 0.0 for item in result.triangles)


def test_depth_limit_is_reported_and_leaf_retains_error():
    result = subdivide_ftheta_triangle_adaptive(
        triangle(), CurvedCalibration(),
        maximum_projection_error_px=1e-12, maximum_depth=1,
    )
    assert len(result.triangles) == 4
    assert result.maximum_depth_reached == 1
    assert result.stopped_by_depth_limit is True
    assert any(item.maximum_test_error_px > 1e-12 for item in result.triangles)


def test_vertex_projections_are_stored_with_each_leaf():
    result = subdivide_ftheta_triangle_adaptive(
        triangle(), LinearCalibration(),
        maximum_projection_error_px=0.01, maximum_depth=1,
    )
    leaf = result.triangles[0]
    assert len(leaf.vertex_projections) == 3
    assert [(p.u, p.v) for p in leaf.vertex_projections] == [
        (0.0, 0.0), (20.0, 0.0), (0.0, 40.0)
    ]


def test_projection_cache_avoids_reprojecting_shared_points():
    calibration = CurvedCalibration()
    subdivide_ftheta_triangle_adaptive(
        triangle(), calibration,
        maximum_projection_error_px=0.6, maximum_depth=2,
    )
    assert calibration.calls == 19


def test_vertex_exactly_on_near_plane_is_accepted():
    source = (
        Vector3(0.0, 0.0, 0.1),
        Vector3(2.0, 0.0, 1.0),
        Vector3(0.0, 2.0, 1.0),
    )
    result = subdivide_ftheta_triangle_adaptive(
        source, LinearCalibration(),
        maximum_projection_error_px=0.01, maximum_depth=1,
        near_plane_m=0.1,
    )
    assert result.triangles


def test_point_outside_fov_is_rejected():
    with pytest.raises(ValueError, match="within FOV"):
        subdivide_ftheta_triangle_adaptive(
            triangle(), InvalidCalibration(),
            maximum_projection_error_px=1.0, maximum_depth=1,
        )


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    source = tuple(Vector3(float(i), 0.0, 1.0) for i in range(count))
    with pytest.raises(ValueError, match="exactly three vertices"):
        subdivide_ftheta_triangle_adaptive(
            source, LinearCalibration(),
            maximum_projection_error_px=1.0, maximum_depth=1,
        )


@pytest.mark.parametrize("error", (0.0, -1.0, math.nan, math.inf))
def test_invalid_error_limit_is_rejected(error):
    with pytest.raises(ValueError, match="maximum_projection_error_px"):
        subdivide_ftheta_triangle_adaptive(
            triangle(), LinearCalibration(),
            maximum_projection_error_px=error, maximum_depth=1,
        )


@pytest.mark.parametrize("depth", (-1,))
def test_negative_maximum_depth_is_rejected(depth):
    with pytest.raises(ValueError, match="non-negative"):
        subdivide_ftheta_triangle_adaptive(
            triangle(), LinearCalibration(),
            maximum_projection_error_px=1.0, maximum_depth=depth,
        )


@pytest.mark.parametrize("depth", (True, 1.5, "2"))
def test_non_integer_maximum_depth_is_rejected(depth):
    with pytest.raises(TypeError, match="integer"):
        subdivide_ftheta_triangle_adaptive(
            triangle(), LinearCalibration(),
            maximum_projection_error_px=1.0, maximum_depth=depth,
        )


@pytest.mark.parametrize("near", (0.0, -1.0, math.nan, math.inf))
def test_invalid_near_plane_is_rejected(near):
    with pytest.raises(ValueError, match="near_plane_m"):
        subdivide_ftheta_triangle_adaptive(
            triangle(), LinearCalibration(),
            maximum_projection_error_px=1.0, maximum_depth=1,
            near_plane_m=near,
        )


def test_vertex_behind_near_plane_is_rejected_before_projection():
    source = (
        Vector3(0.0, 0.0, 0.09),
        Vector3(2.0, 0.0, 1.0),
        Vector3(0.0, 2.0, 1.0),
    )
    calibration = LinearCalibration()
    with pytest.raises(ValueError, match="behind near_plane_m"):
        subdivide_ftheta_triangle_adaptive(
            source, calibration,
            maximum_projection_error_px=1.0, maximum_depth=1,
            near_plane_m=0.1,
        )
    assert calibration.calls == 0
