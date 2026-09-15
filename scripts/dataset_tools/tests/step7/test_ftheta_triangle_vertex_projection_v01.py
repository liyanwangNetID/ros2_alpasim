from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from step7.camera_projection_v01 import Vector3
from step7.ftheta_triangle_vertex_projection_v01 import project_ftheta_triangle_vertices


@dataclass(frozen=True)
class FakePixelProjection:
    u: float
    v: float
    positive_z: bool
    within_fov: bool
    valid: bool


class FakeCalibration:
    camera_name = "front_wide"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    def project_camera_point(self, point):
        self.inputs.append(point)
        return next(self.outputs)


def triangle():
    return (
        Vector3(-1.0, -1.0, 5.0),
        Vector3(1.0, -1.0, 5.0),
        Vector3(0.0, 1.0, 6.0),
    )


def pixel(u, v, *, fov=True, valid=True, positive=True):
    return FakePixelProjection(u, v, positive, fov, valid)


def test_projects_three_vertices_in_input_order():
    calibration = FakeCalibration((pixel(10, 20), pixel(30, 20), pixel(20, 5)))
    source = triangle()
    result = project_ftheta_triangle_vertices(source, calibration=calibration)
    assert calibration.inputs == list(source)
    assert [item.vertex_index for item in result.vertices] == [0, 1, 2]
    assert [(item.u_px, item.v_px) for item in result.vertices] == [
        (10.0, 20.0), (30.0, 20.0), (20.0, 5.0)
    ]
    assert [item.camera_z_m for item in result.vertices] == [5.0, 5.0, 6.0]


def test_all_vertex_flags_are_aggregated():
    calibration = FakeCalibration((pixel(1, 2), pixel(3, 4), pixel(5, 6)))
    result = project_ftheta_triangle_vertices(triangle(), calibration=calibration)
    assert result.all_vertices_within_fov is True
    assert result.all_vertices_inside_image is True


def test_outside_vertex_changes_aggregate_flags_without_dropping_record():
    calibration = FakeCalibration((
        pixel(1, 2),
        pixel(3, 4, fov=True, valid=False),
        pixel(5, 6, fov=False, valid=False),
    ))
    result = project_ftheta_triangle_vertices(triangle(), calibration=calibration)
    assert len(result.vertices) == 3
    assert result.all_vertices_within_fov is False
    assert result.all_vertices_inside_image is False


def test_to_dict_is_json_ready():
    calibration = FakeCalibration((pixel(1, 2), pixel(3, 4), pixel(5, 6)))
    value = project_ftheta_triangle_vertices(triangle(), calibration=calibration).to_dict()
    assert value["camera_name"] == "front_wide"
    assert isinstance(value["vertices"], list)
    assert value["vertices"][2]["camera_z_m"] == 6.0


def test_vertex_exactly_on_near_plane_is_accepted():
    source = (Vector3(0, 0, 0.1), Vector3(1, 0, 1), Vector3(0, 1, 1))
    calibration = FakeCalibration((pixel(1, 2), pixel(3, 4), pixel(5, 6)))
    result = project_ftheta_triangle_vertices(
        source, calibration=calibration, near_plane_m=0.1
    )
    assert result.vertices[0].camera_z_m == 0.1


def test_vertex_behind_near_plane_is_rejected_before_projection():
    source = (Vector3(0, 0, 0.09), Vector3(1, 0, 1), Vector3(0, 1, 1))
    calibration = FakeCalibration((pixel(1, 2), pixel(3, 4), pixel(5, 6)))
    with pytest.raises(ValueError, match="behind near_plane_m"):
        project_ftheta_triangle_vertices(
            source, calibration=calibration, near_plane_m=0.1
        )
    assert calibration.inputs == []


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_vertex_count_is_rejected(count):
    source = tuple(Vector3(float(i), 0, 1) for i in range(count))
    calibration = FakeCalibration(())
    with pytest.raises(ValueError, match="exactly three vertices"):
        project_ftheta_triangle_vertices(source, calibration=calibration)


@pytest.mark.parametrize("near", (0.0, -1.0, math.nan, math.inf))
def test_invalid_near_plane_is_rejected(near):
    calibration = FakeCalibration(())
    with pytest.raises(ValueError, match="near_plane_m"):
        project_ftheta_triangle_vertices(
            triangle(), calibration=calibration, near_plane_m=near
        )


def test_empty_camera_name_is_rejected():
    calibration = FakeCalibration(())
    calibration.camera_name = ""
    with pytest.raises(ValueError, match="camera_name"):
        project_ftheta_triangle_vertices(triangle(), calibration=calibration)


def test_non_finite_input_vertex_is_rejected():
    source = (Vector3(math.nan, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1))
    calibration = FakeCalibration(())
    with pytest.raises(ValueError, match="vertices must be finite"):
        project_ftheta_triangle_vertices(source, calibration=calibration)


def test_non_finite_projected_coordinate_is_rejected():
    calibration = FakeCalibration((pixel(math.nan, 2), pixel(3, 4), pixel(5, 6)))
    with pytest.raises(ValueError, match="pixel coordinates must be finite"):
        project_ftheta_triangle_vertices(triangle(), calibration=calibration)


def test_non_positive_projection_is_rejected():
    calibration = FakeCalibration((pixel(1, 2, positive=False), pixel(3, 4), pixel(5, 6)))
    with pytest.raises(ValueError, match="non-positive z"):
        project_ftheta_triangle_vertices(triangle(), calibration=calibration)


def test_inside_image_without_fov_is_rejected():
    calibration = FakeCalibration((pixel(1, 2, fov=False, valid=True), pixel(3, 4), pixel(5, 6)))
    with pytest.raises(ValueError, match="within FOV"):
        project_ftheta_triangle_vertices(triangle(), calibration=calibration)
