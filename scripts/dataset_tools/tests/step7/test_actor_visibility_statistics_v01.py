from __future__ import annotations

import pytest

from step7.actor_depth_zbuffer_v01 import ActorZBufferSummary
from step7.actor_visibility_statistics_v01 import (
    calculate_actor_visibility_statistics,
)


def summary(actor_id, occupied, winning, occluded):
    return ActorZBufferSummary(
        actor_id=actor_id,
        occupied_cell_count=occupied,
        winning_cell_count=winning,
        occluded_cell_count=occluded,
    )


def test_empty_input_returns_empty_tuple():
    assert calculate_actor_visibility_statistics(()) == ()


def test_fully_visible_actor():
    result = calculate_actor_visibility_statistics((summary("a", 4, 4, 0),))[0]
    assert result.has_sampled_surface
    assert result.visible_fraction == pytest.approx(1.0)
    assert result.occluded_fraction == pytest.approx(0.0)
    assert result.fully_visible
    assert not result.fully_occluded


def test_fully_occluded_actor():
    result = calculate_actor_visibility_statistics((summary("a", 5, 0, 5),))[0]
    assert result.visible_fraction == pytest.approx(0.0)
    assert result.occluded_fraction == pytest.approx(1.0)
    assert not result.fully_visible
    assert result.fully_occluded


def test_partially_visible_actor():
    result = calculate_actor_visibility_statistics((summary("a", 10, 3, 7),))[0]
    assert result.visible_fraction == pytest.approx(0.3)
    assert result.occluded_fraction == pytest.approx(0.7)
    assert not result.fully_visible
    assert not result.fully_occluded
    assert result.visible_fraction + result.occluded_fraction == pytest.approx(1.0)


def test_no_sampled_surface_has_undefined_fractions_and_no_full_flags():
    result = calculate_actor_visibility_statistics((summary("a", 0, 0, 0),))[0]
    assert not result.has_sampled_surface
    assert result.visible_fraction is None
    assert result.occluded_fraction is None
    assert not result.fully_visible
    assert not result.fully_occluded


def test_results_are_sorted_by_actor_id():
    result = calculate_actor_visibility_statistics((
        summary("z", 1, 1, 0),
        summary("a", 1, 0, 1),
    ))
    assert tuple(item.actor_id for item in result) == ("a", "z")


def test_output_preserves_counts():
    result = calculate_actor_visibility_statistics((summary("a", 7, 2, 5),))[0]
    assert result.occupied_cell_count == 7
    assert result.winning_cell_count == 2
    assert result.occluded_cell_count == 5


def test_duplicate_actor_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        calculate_actor_visibility_statistics((
            summary("a", 1, 1, 0),
            summary("a", 1, 1, 0),
        ))


@pytest.mark.parametrize("actor_id", ("", None, 42))
def test_invalid_actor_id_is_rejected(actor_id):
    with pytest.raises(ValueError, match="non-empty string"):
        calculate_actor_visibility_statistics((summary(actor_id, 0, 0, 0),))


@pytest.mark.parametrize(
    "counts",
    ((-1, 0, 0), (1, -1, 2), (1, 2, -1)),
)
def test_negative_count_is_rejected(counts):
    with pytest.raises(ValueError, match="non-negative"):
        calculate_actor_visibility_statistics((summary("a", *counts),))


@pytest.mark.parametrize(
    "counts",
    ((2, 1, 0), (2, 0, 1), (1, 1, 1)),
)
def test_inconsistent_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="equal occupied"):
        calculate_actor_visibility_statistics((summary("a", *counts),))


@pytest.mark.parametrize(
    "counts",
    ((True, 1, 0), (1, 1.0, 0), (1, 1, "0")),
)
def test_noninteger_count_is_rejected(counts):
    with pytest.raises(TypeError, match="integers"):
        calculate_actor_visibility_statistics((summary("a", *counts),))
