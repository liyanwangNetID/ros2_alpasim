from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from step7 import actor_camera_occlusion_from_geometry_v01 as target


@dataclass(frozen=True)
class Cell:
    column: int
    row: int


@dataclass(frozen=True)
class Sample:
    cell: Cell
    depth_m: float


@dataclass(frozen=True)
class Raster:
    cell_depths: tuple[Sample, ...]
    occupied_cell_count: int


class Calibration:
    max_angle_rad = 1.0

    def rig_point_to_camera(self, point):
        return ("camera", point)


def actor(track_id):
    return {"track_id": track_id}


def install_geometry_stubs(monkeypatch):
    monkeypatch.setattr(
        target,
        "actor_box_corners_in_rig",
        lambda value, recorded_ego_message: (
            value["track_id"],
        ),
    )
    monkeypatch.setattr(
        target,
        "prepare_camera_facing_box_triangles",
        lambda corners, near_plane_m: (
            SimpleNamespace(vertices_camera=corners),
        ),
    )

    def build_surface(triangles, calibration, **kwargs):
        track_id = triangles[0][0][1]
        if track_id == "near":
            samples = (
                Sample(Cell(0, 0), 5.0),
                Sample(Cell(1, 0), 5.0),
            )
        elif track_id == "far":
            samples = (
                Sample(Cell(0, 0), 10.0),
                Sample(Cell(1, 0), 10.0),
                Sample(Cell(2, 0), 10.0),
            )
        else:
            samples = ()
        return SimpleNamespace(
            surface_raster=Raster(samples, len(samples)),
            generated_triangle_sample_count=1 if samples else 0,
            zero_center_sample_triangle_count=0 if samples else 1,
            unresolved_boundary_count=0,
        )

    monkeypatch.setattr(
        target,
        "build_actor_camera_surface_depth_raster",
        build_surface,
    )


def call(monkeypatch, **overrides):
    install_geometry_stubs(monkeypatch)
    values = {
        "actors": (actor("near"), actor("far"), actor("empty")),
        "recorded_ego_message": {"ego": True},
        "calibration": Calibration(),
        "camera_name": "cross_left",
        "image_width_px": 1920,
        "image_height_px": 1080,
        "raster_width": 480,
        "raster_height": 270,
        "maximum_depth": 8,
        "maximum_boundary_extent_px": 4.0,
    }
    values.update(overrides)
    return target.build_actor_camera_occlusion_from_geometry(**values)


def test_builds_surface_diagnostics_and_shared_evidence(monkeypatch):
    result = call(monkeypatch)
    evidence = {
        item.track_id: item
        for item in result.occlusion.actor_evidence
    }

    assert result.actor_count == 3
    assert evidence["far"].occupied_cell_count == 3
    assert evidence["far"].winning_cell_count == 1
    assert evidence["far"].occluding_actor_ids == ("near",)
    assert evidence["near"].visible_fraction == 1.0
    assert evidence["empty"].evidence_status == "no_sampled_surface"


def test_output_is_deterministic_by_track_id(monkeypatch):
    result = call(
        monkeypatch,
        actors=(actor("near"), actor("empty"), actor("far")),
    )

    assert tuple(
        item.track_id for item in result.surface_diagnostics
    ) == ("empty", "far", "near")
    assert tuple(
        item.track_id for item in result.occlusion.actor_evidence
    ) == ("empty", "far", "near")


def test_surface_diagnostics_preserve_zero_surface(monkeypatch):
    result = call(monkeypatch)
    diagnostics = {
        item.track_id: item
        for item in result.surface_diagnostics
    }

    assert diagnostics["empty"].occupied_cell_count == 0
    assert diagnostics["empty"].generated_triangle_sample_count == 0
    assert diagnostics["empty"].zero_center_sample_triangle_count == 1
    assert diagnostics["empty"].unresolved_boundary_count == 0


def test_result_metadata_matches_request(monkeypatch):
    result = call(monkeypatch)

    assert result.camera_name == "cross_left"
    assert result.raster_width == 480
    assert result.raster_height == 270
    assert result.occlusion.camera_name == "cross_left"


def test_duplicate_track_ids_are_rejected(monkeypatch):
    install_geometry_stubs(monkeypatch)

    with pytest.raises(ValueError, match="must be unique"):
        call(
            monkeypatch,
            actors=(actor("near"), actor("near")),
        )


def test_missing_track_id_is_rejected(monkeypatch):
    install_geometry_stubs(monkeypatch)

    with pytest.raises(ValueError, match="usable track_id"):
        call(monkeypatch, actors=({},))


def test_missing_calibration_angle_is_rejected(monkeypatch):
    install_geometry_stubs(monkeypatch)
    calibration = Calibration()
    calibration.max_angle_rad = None

    with pytest.raises(ValueError, match="max_angle_rad"):
        call(monkeypatch, calibration=calibration)


def test_static_occlusion_remains_false(monkeypatch):
    result = call(monkeypatch)

    assert all(
        not item.static_occlusion_evaluated
        for item in result.occlusion.actor_evidence
    )
