from __future__ import annotations

import math

import pytest

from step7.actor_depth_zbuffer_v01 import (
    ActorDepthRasterInput,
    resolve_actor_depth_zbuffer,
)
from step7.actor_surface_depth_raster_v01 import (
    ActorSurfaceCellDepth,
    ActorSurfaceDepthRaster,
)
from step7.projected_triangle_raster_cells_v01 import RasterCell


def surface(column, row, depth):
    return ActorSurfaceCellDepth(
        cell=RasterCell(column, row),
        depth_m=depth,
        reciprocal_depth_per_m=1.0 / depth,
        winning_triangle_index=0,
        sample_u_px=column + 0.5,
        sample_v_px=row + 0.5,
        barycentric_weights=(0.5, 0.25, 0.25),
    )


def raster(*items, occupied_count=None):
    count = len(items) if occupied_count is None else occupied_count
    return ActorSurfaceDepthRaster(
        cell_depths=tuple(items),
        occupied_cell_count=count,
        input_triangle_count=1,
        input_depth_sample_count=len(items),
        replaced_sample_count=0,
        discarded_farther_or_equal_sample_count=0,
    )


def actor(actor_id, *items):
    return ActorDepthRasterInput(actor_id, raster(*items))


def test_empty_input_returns_empty_zbuffer():
    result = resolve_actor_depth_zbuffer(())
    assert result.cell_winners == ()
    assert result.actor_summaries == ()
    assert result.occupied_union_cell_count == 0
    assert result.contested_cell_count == 0


def test_disjoint_actors_keep_all_cells_in_row_major_order():
    result = resolve_actor_depth_zbuffer((
        actor("b", surface(2, 1, 4.0)),
        actor("a", surface(0, 0, 3.0)),
    ))
    assert tuple(item.cell for item in result.cell_winners) == (
        RasterCell(0, 0), RasterCell(2, 1)
    )
    assert result.occupied_union_cell_count == 2
    assert result.contested_cell_count == 0


def test_nearer_actor_wins_contested_cell():
    result = resolve_actor_depth_zbuffer((
        actor("far", surface(1, 1, 8.0)),
        actor("near", surface(1, 1, 3.0)),
    ))
    assert result.cell_winners[0].actor_id == "near"
    assert result.cell_winners[0].depth_m == pytest.approx(3.0)
    assert result.contested_cell_count == 1


def test_tie_uses_lexicographically_smaller_actor_id():
    result = resolve_actor_depth_zbuffer((
        actor("zeta", surface(1, 1, 3.0)),
        actor("alpha", surface(1, 1, 3.0)),
    ))
    assert result.cell_winners[0].actor_id == "alpha"


def test_tolerance_equivalent_depths_use_actor_id_tie_break():
    result = resolve_actor_depth_zbuffer((
        actor("a", surface(1, 1, 5.0)),
        actor("b", surface(1, 1, 5.0 - 5e-7)),
    ), depth_tolerance_m=1e-6)
    assert result.cell_winners[0].actor_id == "a"


def test_strictly_nearer_beyond_tolerance_wins_regardless_of_actor_id():
    result = resolve_actor_depth_zbuffer((
        actor("a", surface(1, 1, 5.0)),
        actor("z", surface(1, 1, 4.0)),
    ), depth_tolerance_m=0.1)
    assert result.cell_winners[0].actor_id == "z"


def test_input_order_does_not_change_winners_or_summaries():
    first = actor("b", surface(0, 0, 2.0), surface(1, 0, 4.0))
    second = actor("a", surface(1, 0, 4.0), surface(2, 0, 3.0))
    forward = resolve_actor_depth_zbuffer((first, second))
    reverse = resolve_actor_depth_zbuffer((second, first))
    assert forward == reverse


def test_actor_summaries_count_occupied_winning_and_occluded_cells():
    result = resolve_actor_depth_zbuffer((
        actor("near", surface(0, 0, 2.0), surface(1, 0, 2.0)),
        actor("far", surface(1, 0, 5.0), surface(2, 0, 5.0)),
    ))
    summaries = {item.actor_id: item for item in result.actor_summaries}
    assert summaries["near"].occupied_cell_count == 2
    assert summaries["near"].winning_cell_count == 2
    assert summaries["near"].occluded_cell_count == 0
    assert summaries["far"].occupied_cell_count == 2
    assert summaries["far"].winning_cell_count == 1
    assert summaries["far"].occluded_cell_count == 1


def test_winner_preserves_source_surface_record():
    source = surface(3, 4, 2.5)
    result = resolve_actor_depth_zbuffer((actor("actor-1", source),))
    assert result.cell_winners[0].source_surface == source


def test_actor_with_empty_raster_receives_zero_summary():
    result = resolve_actor_depth_zbuffer((
        ActorDepthRasterInput("empty", raster()),
    ))
    assert result.actor_summaries[0].occupied_cell_count == 0
    assert result.actor_summaries[0].winning_cell_count == 0
    assert result.actor_summaries[0].occluded_cell_count == 0


def test_duplicate_actor_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        resolve_actor_depth_zbuffer((actor("same"), actor("same")))


@pytest.mark.parametrize("actor_id", ("", None, 123))
def test_invalid_actor_id_is_rejected(actor_id):
    with pytest.raises(ValueError, match="non-empty string"):
        resolve_actor_depth_zbuffer((ActorDepthRasterInput(actor_id, raster()),))


@pytest.mark.parametrize("tolerance", (-1.0, math.nan, math.inf))
def test_invalid_depth_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="depth_tolerance_m"):
        resolve_actor_depth_zbuffer((), depth_tolerance_m=tolerance)


def test_inconsistent_occupied_count_is_rejected():
    bad = ActorDepthRasterInput("a", raster(surface(0, 0, 2.0), occupied_count=2))
    with pytest.raises(ValueError, match="occupied_cell_count"):
        resolve_actor_depth_zbuffer((bad,))


def test_duplicate_cells_within_actor_raster_are_rejected():
    duplicate = surface(0, 0, 2.0)
    bad = ActorDepthRasterInput("a", raster(duplicate, duplicate))
    with pytest.raises(ValueError, match="duplicate cells"):
        resolve_actor_depth_zbuffer((bad,))


@pytest.mark.parametrize("depth", (0.0, -1.0, math.nan, math.inf))
def test_invalid_surface_depth_is_rejected(depth):
    bad_surface = ActorSurfaceCellDepth(
        cell=RasterCell(0, 0), depth_m=depth,
        reciprocal_depth_per_m=1.0, winning_triangle_index=0,
        sample_u_px=0.5, sample_v_px=0.5,
        barycentric_weights=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="positive and finite"):
        resolve_actor_depth_zbuffer((actor("a", bad_surface),))
