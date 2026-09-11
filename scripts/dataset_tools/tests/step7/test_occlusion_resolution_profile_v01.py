import pytest

from actor_occlusion_evidence_v01 import (
    build_camera_actor_occlusion_evidence,
)
from occlusion_resolution_profile_v01 import (
    build_occlusion_resolution_profile,
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


def candidate():
    return (
        evidence("350", 480, 270, 33, 23, 10, ("80",)),
        evidence("226", 480, 270, 60, 0, 60, ("17",)),
        evidence("307", 480, 270, 60, 5, 55, ("18",)),
    )


def reference():
    return (
        evidence("307", 640, 360, 91, 9, 82, ("18", "80")),
        evidence("226", 640, 360, 114, 4, 110, ("17",)),
        evidence("350", 640, 360, 61, 48, 13, ("80",)),
    )


def test_builds_deterministic_comparisons_and_summary():
    result = build_occlusion_resolution_profile(
        candidate_evidence=candidate(),
        reference_evidence=reference(),
    )

    assert tuple(item.track_id for item in result.comparisons) == (
        "226",
        "307",
        "350",
    )
    assert result.summary.actor_count == 3
    assert result.summary.comparable_actor_count == 3
    assert result.summary.winning_presence_change_track_ids == ("226",)
    assert result.summary.occluding_actor_set_change_track_ids == ("307",)


def test_profile_preserves_track_226_boundary_change():
    result = build_occlusion_resolution_profile(
        candidate_evidence=candidate(),
        reference_evidence=reference(),
    )
    item = next(value for value in result.comparisons if value.track_id == "226")

    assert item.both_evaluated
    assert not item.sampled_surface_presence_changed
    assert item.winning_presence_changed
    assert item.candidate_winning_cell_count == 0
    assert item.reference_winning_cell_count == 4
    assert not item.occluding_actor_set_changed


def test_actor_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Actor sets must match"):
        build_occlusion_resolution_profile(
            candidate_evidence=candidate(),
            reference_evidence=reference()[:-1],
        )


def test_duplicate_candidate_actor_is_rejected():
    source = candidate()
    with pytest.raises(ValueError, match="candidate evidence contains duplicate"):
        build_occlusion_resolution_profile(
            candidate_evidence=(source[0], source[0]),
            reference_evidence=reference(),
        )


def test_duplicate_reference_actor_is_rejected():
    source = reference()
    with pytest.raises(ValueError, match="reference evidence contains duplicate"):
        build_occlusion_resolution_profile(
            candidate_evidence=candidate(),
            reference_evidence=(source[0], source[0]),
        )


def test_empty_sets_are_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        build_occlusion_resolution_profile(
            candidate_evidence=(),
            reference_evidence=(),
        )


def test_profile_builds_stability_evidence_in_track_order():
    result = build_occlusion_resolution_profile(
        candidate_evidence=candidate(),
        reference_evidence=reference(),
    )

    assert tuple(
        item.track_id for item in result.stability_evidence
    ) == ("226", "307", "350")
    by_id = {
        item.track_id: item for item in result.stability_evidence
    }
    assert by_id["226"].stability_status == "winning_presence_changed"
    assert by_id["307"].stability_status == "occluding_actor_set_changed"
    assert by_id["350"].stability_status == "discrete_evidence_stable"


def test_profile_stability_preserves_continuous_delta_without_threshold():
    result = build_occlusion_resolution_profile(
        candidate_evidence=candidate(),
        reference_evidence=reference(),
    )
    comparison_by_id = {
        item.track_id: item for item in result.comparisons
    }
    stability_by_id = {
        item.track_id: item for item in result.stability_evidence
    }

    for track_id in comparison_by_id:
        assert (
            stability_by_id[track_id].absolute_visible_fraction_delta
            == comparison_by_id[track_id].absolute_visible_fraction_delta
        )
