from dataclasses import replace

import pytest

from step7.actor_occlusion_evidence_v01 import (
    build_camera_actor_occlusion_evidence,
)
from step7.actor_occlusion_resolution_comparison_v01 import (
    compare_actor_occlusion_resolutions,
)
from step7.occlusion_resolution_comparison_summary_v01 import (
    summarize_occlusion_resolution_comparisons,
)


def evidence(track, width, height, occupied, winning, occluded, occluders=()):
    return build_camera_actor_occlusion_evidence(
        camera_name="cross_left",
        track_id=track,
        raster_width=width,
        raster_height=height,
        occupied_cell_count=occupied,
        winning_cell_count=winning,
        occluded_cell_count=occluded,
        occluding_actor_ids=occluders,
        actor_to_actor_occlusion_evaluated=True,
        static_occlusion_evaluated=False,
    )


def comparison(track, candidate, reference):
    return compare_actor_occlusion_resolutions(
        candidate=evidence(track, 480, 270, *candidate),
        reference=evidence(track, 640, 360, *reference),
    )


def source():
    return (
        comparison("226", (60, 0, 60, ("17",)), (114, 4, 110, ("17",))),
        comparison("350", (33, 23, 10, ("80",)), (61, 48, 13, ("80",))),
        comparison("307", (60, 5, 55, ("18",)), (91, 9, 82, ("18", "80"))),
    )


def test_summarizes_change_counts_and_track_ids():
    result = summarize_occlusion_resolution_comparisons(source())

    assert result.actor_count == 3
    assert result.comparable_actor_count == 3
    assert result.sampled_surface_presence_change_count == 0
    assert result.winning_presence_change_count == 1
    assert result.occluding_actor_set_change_count == 1
    assert result.nonzero_visible_fraction_delta_count == 3
    assert result.winning_presence_change_track_ids == ("226",)
    assert result.occluding_actor_set_change_track_ids == ("307",)
    assert result.reasons == ()


def test_summarizes_visible_fraction_deltas():
    comparisons = source()
    result = summarize_occlusion_resolution_comparisons(comparisons)
    deltas = sorted(
        item.absolute_visible_fraction_delta
        for item in comparisons
    )

    assert result.maximum_absolute_visible_fraction_delta == max(deltas)
    assert result.median_absolute_visible_fraction_delta == deltas[1]


def test_noncomparable_actor_is_preserved_as_reason():
    comparable = source()[0]
    noncomparable = replace(
        source()[1],
        both_evaluated=False,
        absolute_visible_fraction_delta=None,
        winning_presence_changed=None,
        occluding_actor_set_changed=None,
    )
    result = summarize_occlusion_resolution_comparisons(
        (comparable, noncomparable)
    )

    assert result.actor_count == 2
    assert result.comparable_actor_count == 1
    assert result.reasons == ("one_or_more_actors_not_comparable",)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_occlusion_resolution_comparisons(())


def test_mixed_camera_or_raster_pair_is_rejected():
    first, second, _ = source()
    mixed = replace(second, camera_name="cross_right")

    with pytest.raises(ValueError, match="same camera and raster pair"):
        summarize_occlusion_resolution_comparisons((first, mixed))


def test_duplicate_track_ids_are_rejected():
    first = source()[0]

    with pytest.raises(ValueError, match="must be unique"):
        summarize_occlusion_resolution_comparisons((first, first))


def test_to_dict_uses_json_ready_lists():
    value = summarize_occlusion_resolution_comparisons(source()).to_dict()

    assert isinstance(value["winning_presence_change_track_ids"], list)
    assert isinstance(value["occluding_actor_set_change_track_ids"], list)
    assert isinstance(value["reasons"], list)
