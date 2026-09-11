from dataclasses import dataclass

import pytest

from actor_depth_zbuffer_v01 import (
    ActorDepthRasterInput,
    ActorDepthZBuffer,
    ActorZBufferSummary,
    ZBufferCellWinner,
)
from actor_occlusion_evidence_adapter_v01 import (
    build_camera_occlusion_evidence_from_zbuffer,
)


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
    samples = tuple(SurfaceSample(Cell(column, row), depth) for column, row, depth in cells)
    return SurfaceRaster(samples, len(samples))


def shared_case():
    near = ActorDepthRasterInput("near", raster((0, 0, 5.0), (1, 0, 5.0)))
    far = ActorDepthRasterInput("far", raster((0, 0, 10.0), (1, 0, 10.0), (2, 0, 10.0)))
    winner_0 = ZBufferCellWinner(Cell(0, 0), "near", 5.0, near.surface_raster.cell_depths[0])
    winner_1 = ZBufferCellWinner(Cell(1, 0), "near", 5.0, near.surface_raster.cell_depths[1])
    winner_2 = ZBufferCellWinner(Cell(2, 0), "far", 10.0, far.surface_raster.cell_depths[2])
    zbuffer = ActorDepthZBuffer(
        cell_winners=(winner_0, winner_1, winner_2),
        actor_summaries=(
            ActorZBufferSummary("far", 3, 1, 2),
            ActorZBufferSummary("near", 2, 2, 0),
        ),
        occupied_union_cell_count=3,
        contested_cell_count=2,
    )
    return (near, far), zbuffer


def call(inputs=None, zbuffer=None):
    default_inputs, default_zbuffer = shared_case()
    return build_camera_occlusion_evidence_from_zbuffer(
        camera_name="cross_left",
        raster_width=480,
        raster_height=270,
        actor_rasters=default_inputs if inputs is None else inputs,
        zbuffer=default_zbuffer if zbuffer is None else zbuffer,
    )


def test_extracts_occluder_ids_and_counts():
    result = {item.track_id: item for item in call()}
    assert result["far"].occupied_cell_count == 3
    assert result["far"].winning_cell_count == 1
    assert result["far"].occluded_cell_count == 2
    assert result["far"].visible_fraction == pytest.approx(1 / 3)
    assert result["far"].occluding_actor_ids == ("near",)
    assert result["near"].occluding_actor_ids == ()


def test_output_is_sorted_by_track_id():
    assert tuple(item.track_id for item in call()) == ("far", "near")


def test_zero_surface_actor_is_preserved():
    empty = ActorDepthRasterInput("empty", raster())
    zbuffer = ActorDepthZBuffer(
        cell_winners=(),
        actor_summaries=(ActorZBufferSummary("empty", 0, 0, 0),),
        occupied_union_cell_count=0,
        contested_cell_count=0,
    )
    result = call(inputs=(empty,), zbuffer=zbuffer)
    assert result[0].evidence_status == "no_sampled_surface"
    assert result[0].visible_fraction is None


def test_summary_actor_set_must_match_inputs():
    inputs, zbuffer = shared_case()
    bad = ActorDepthZBuffer(
        cell_winners=zbuffer.cell_winners,
        actor_summaries=zbuffer.actor_summaries[:1],
        occupied_union_cell_count=zbuffer.occupied_union_cell_count,
        contested_cell_count=zbuffer.contested_cell_count,
    )
    with pytest.raises(ValueError, match="must match"):
        call(inputs=inputs, zbuffer=bad)


def test_missing_winner_for_target_cell_is_rejected():
    inputs, zbuffer = shared_case()
    bad = ActorDepthZBuffer(
        cell_winners=zbuffer.cell_winners[:2],
        actor_summaries=zbuffer.actor_summaries,
        occupied_union_cell_count=2,
        contested_cell_count=zbuffer.contested_cell_count,
    )
    with pytest.raises(ValueError, match="has no Z-buffer winner"):
        call(inputs=inputs, zbuffer=bad)


def test_duplicate_input_actor_ids_are_rejected():
    inputs, zbuffer = shared_case()
    with pytest.raises(ValueError, match="must be unique"):
        call(inputs=(inputs[0], inputs[0]), zbuffer=zbuffer)


def test_inconsistent_surface_count_is_rejected():
    inputs, zbuffer = shared_case()
    broken_surface = SurfaceRaster(inputs[0].surface_raster.cell_depths, 99)
    broken = ActorDepthRasterInput("near", broken_surface)
    with pytest.raises(ValueError, match="surface raster occupied count"):
        call(inputs=(broken, inputs[1]), zbuffer=zbuffer)


def test_unknown_winner_actor_is_rejected():
    inputs, zbuffer = shared_case()
    bad_first = ZBufferCellWinner(
        zbuffer.cell_winners[0].cell,
        "unknown",
        zbuffer.cell_winners[0].depth_m,
        zbuffer.cell_winners[0].source_surface,
    )
    bad = ActorDepthZBuffer(
        cell_winners=(bad_first, *zbuffer.cell_winners[1:]),
        actor_summaries=zbuffer.actor_summaries,
        occupied_union_cell_count=zbuffer.occupied_union_cell_count,
        contested_cell_count=zbuffer.contested_cell_count,
    )
    with pytest.raises(ValueError, match="unknown Actor"):
        call(inputs=inputs, zbuffer=bad)
