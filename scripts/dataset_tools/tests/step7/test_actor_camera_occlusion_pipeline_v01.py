from dataclasses import dataclass

import pytest

from step7.actor_camera_occlusion_pipeline_v01 import (
    build_camera_actor_occlusion_pipeline,
)
from step7.actor_depth_zbuffer_v01 import ActorDepthRasterInput


@dataclass(frozen=True)
class Cell:
    column: int
    row: int


@dataclass(frozen=True)
class SurfaceSample:
    cell: Cell
    depth_m: float


@dataclass(frozen=True)
class SurfaceRaster:
    cell_depths: tuple[SurfaceSample, ...]
    occupied_cell_count: int


def raster(*cells):
    samples = tuple(
        SurfaceSample(Cell(column, row), depth)
        for column, row, depth in cells
    )
    return SurfaceRaster(samples, len(samples))


def inputs():
    return (
        ActorDepthRasterInput(
            "near",
            raster((0, 0, 5.0), (1, 0, 5.0)),
        ),
        ActorDepthRasterInput(
            "far",
            raster(
                (0, 0, 10.0),
                (1, 0, 10.0),
                (2, 0, 10.0),
            ),
        ),
        ActorDepthRasterInput("empty", raster()),
    )


def call(**overrides):
    values = {
        "camera_name": "cross_left",
        "raster_width": 480,
        "raster_height": 270,
        "actor_rasters": inputs(),
        "depth_tolerance_m": 1e-9,
        "static_occlusion_evaluated": False,
    }
    values.update(overrides)
    return build_camera_actor_occlusion_pipeline(**values)


def test_pipeline_resolves_zbuffer_and_evidence():
    result = call()
    evidence = {
        item.track_id: item
        for item in result.actor_evidence
    }

    assert result.zbuffer.occupied_union_cell_count == 3
    assert result.zbuffer.contested_cell_count == 2
    assert evidence["far"].occupied_cell_count == 3
    assert evidence["far"].winning_cell_count == 1
    assert evidence["far"].occluded_cell_count == 2
    assert evidence["far"].visible_fraction == pytest.approx(1 / 3)
    assert evidence["far"].occluding_actor_ids == ("near",)
    assert evidence["near"].visible_fraction == 1.0
    assert evidence["near"].occluding_actor_ids == ()


def test_pipeline_preserves_zero_surface_actor():
    result = call()
    evidence = {
        item.track_id: item
        for item in result.actor_evidence
    }

    assert evidence["empty"].evidence_status == "no_sampled_surface"
    assert evidence["empty"].visible_fraction is None
    assert evidence["empty"].occluding_actor_ids == ()


def test_result_metadata_matches_request():
    result = call()

    assert result.camera_name == "cross_left"
    assert result.raster_width == 480
    assert result.raster_height == 270
    assert tuple(
        item.track_id for item in result.actor_evidence
    ) == ("empty", "far", "near")


def test_static_occlusion_flag_is_forwarded_without_evaluation_logic():
    result = call(static_occlusion_evaluated=True)

    assert all(
        item.static_occlusion_evaluated
        for item in result.actor_evidence
    )


def test_duplicate_actor_ids_are_rejected_by_shared_zbuffer():
    source = inputs()

    with pytest.raises(ValueError, match="must be unique"):
        call(actor_rasters=(source[0], source[0]))


def test_invalid_depth_tolerance_is_rejected():
    with pytest.raises(ValueError, match="non-negative and finite"):
        call(depth_tolerance_m=-1.0)


def test_invalid_raster_dimensions_are_rejected_by_evidence_contract():
    with pytest.raises(ValueError, match="must be positive"):
        call(raster_width=0)
