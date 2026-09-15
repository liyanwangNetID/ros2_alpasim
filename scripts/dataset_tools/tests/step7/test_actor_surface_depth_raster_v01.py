from __future__ import annotations

import math

import pytest

from step7.actor_surface_depth_raster_v01 import merge_actor_surface_depth_samples
from step7.projected_triangle_depth_samples_v01 import (
    ProjectedTriangleDepthSamples,
    RasterDepthSample,
)
from step7.projected_triangle_raster_cells_v01 import RasterCell


def depth_sample(column, row, depth, *, u=None, v=None):
    return RasterDepthSample(
        cell=RasterCell(column, row),
        sample_u_px=float(column) + 0.5 if u is None else u,
        sample_v_px=float(row) + 0.5 if v is None else v,
        barycentric_weights=(0.5, 0.25, 0.25),
        reciprocal_depth_per_m=1.0 / depth,
        depth_m=depth,
    )


def triangle_result(*samples):
    cells = tuple(sample.cell for sample in samples)
    return ProjectedTriangleDepthSamples(
        conservative_coverage_cells=cells,
        center_sampled_depths=tuple(samples),
        conservative_cell_count=len(cells),
        center_sampled_cell_count=len(samples),
    )


def test_empty_input_produces_empty_raster():
    result = merge_actor_surface_depth_samples(())
    assert result.cell_depths == ()
    assert result.occupied_cell_count == 0
    assert result.input_triangle_count == 0
    assert result.input_depth_sample_count == 0


def test_disjoint_triangle_cells_are_all_preserved_in_row_major_order():
    result = merge_actor_surface_depth_samples((
        triangle_result(depth_sample(2, 1, 5.0)),
        triangle_result(depth_sample(0, 0, 3.0)),
        triangle_result(depth_sample(1, 1, 4.0)),
    ))
    assert tuple(item.cell for item in result.cell_depths) == (
        RasterCell(0, 0), RasterCell(1, 1), RasterCell(2, 1)
    )
    assert result.occupied_cell_count == 3
    assert result.input_triangle_count == 3
    assert result.input_depth_sample_count == 3


def test_nearer_later_triangle_replaces_current_winner():
    result = merge_actor_surface_depth_samples((
        triangle_result(depth_sample(1, 2, 8.0)),
        triangle_result(depth_sample(1, 2, 3.0)),
    ))
    winner = result.cell_depths[0]
    assert winner.depth_m == pytest.approx(3.0)
    assert winner.winning_triangle_index == 1
    assert result.replaced_sample_count == 1
    assert result.discarded_farther_or_equal_sample_count == 0


def test_farther_later_triangle_is_discarded():
    result = merge_actor_surface_depth_samples((
        triangle_result(depth_sample(1, 2, 3.0)),
        triangle_result(depth_sample(1, 2, 8.0)),
    ))
    assert result.cell_depths[0].winning_triangle_index == 0
    assert result.replaced_sample_count == 0
    assert result.discarded_farther_or_equal_sample_count == 1


def test_equal_depth_preserves_earlier_triangle_deterministically():
    result = merge_actor_surface_depth_samples((
        triangle_result(depth_sample(1, 2, 3.0, u=10.0)),
        triangle_result(depth_sample(1, 2, 3.0, u=20.0)),
    ))
    winner = result.cell_depths[0]
    assert winner.winning_triangle_index == 0
    assert winner.sample_u_px == pytest.approx(10.0)
    assert result.discarded_farther_or_equal_sample_count == 1


def test_depth_difference_within_tolerance_preserves_earlier_triangle():
    result = merge_actor_surface_depth_samples((
        triangle_result(depth_sample(0, 0, 5.0)),
        triangle_result(depth_sample(0, 0, 5.0 - 5e-7)),
    ), depth_tolerance_m=1e-6)
    assert result.cell_depths[0].winning_triangle_index == 0


def test_zero_tolerance_accepts_any_strictly_nearer_sample():
    result = merge_actor_surface_depth_samples((
        triangle_result(depth_sample(0, 0, 5.0)),
        triangle_result(depth_sample(0, 0, 5.0 - 5e-7)),
    ), depth_tolerance_m=0.0)
    assert result.cell_depths[0].winning_triangle_index == 1


def test_winner_preserves_source_diagnostics():
    sample = depth_sample(3, 4, 2.5, u=31.0, v=42.0)
    result = merge_actor_surface_depth_samples((triangle_result(sample),))
    winner = result.cell_depths[0]
    assert winner.sample_u_px == pytest.approx(31.0)
    assert winner.sample_v_px == pytest.approx(42.0)
    assert winner.barycentric_weights == sample.barycentric_weights
    assert winner.reciprocal_depth_per_m == pytest.approx(0.4)


def test_triangle_with_no_center_samples_is_counted_but_adds_no_cells():
    empty = ProjectedTriangleDepthSamples(
        conservative_coverage_cells=(RasterCell(0, 0),),
        center_sampled_depths=(),
        conservative_cell_count=1,
        center_sampled_cell_count=0,
    )
    result = merge_actor_surface_depth_samples((empty,))
    assert result.input_triangle_count == 1
    assert result.input_depth_sample_count == 0
    assert result.occupied_cell_count == 0


@pytest.mark.parametrize("tolerance", (-1.0, math.nan, math.inf))
def test_invalid_depth_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="depth_tolerance_m"):
        merge_actor_surface_depth_samples((), depth_tolerance_m=tolerance)


@pytest.mark.parametrize("depth", (0.0, -1.0))
def test_nonpositive_depth_sample_is_rejected(depth):
    bad = RasterDepthSample(
        cell=RasterCell(0, 0), sample_u_px=0.5, sample_v_px=0.5,
        barycentric_weights=(1.0, 0.0, 0.0),
        reciprocal_depth_per_m=1.0, depth_m=depth,
    )
    with pytest.raises(ValueError, match="positive depth"):
        merge_actor_surface_depth_samples((triangle_result(bad),))


def test_nonfinite_sample_is_rejected():
    bad = RasterDepthSample(
        cell=RasterCell(0, 0), sample_u_px=math.nan, sample_v_px=0.5,
        barycentric_weights=(1.0, 0.0, 0.0),
        reciprocal_depth_per_m=0.5, depth_m=2.0,
    )
    with pytest.raises(ValueError, match="finite"):
        merge_actor_surface_depth_samples((triangle_result(bad),))


def test_inconsistent_reciprocal_depth_is_rejected():
    bad = RasterDepthSample(
        cell=RasterCell(0, 0), sample_u_px=0.5, sample_v_px=0.5,
        barycentric_weights=(1.0, 0.0, 0.0),
        reciprocal_depth_per_m=0.3, depth_m=2.0,
    )
    with pytest.raises(ValueError, match="consistent"):
        merge_actor_surface_depth_samples((triangle_result(bad),))
