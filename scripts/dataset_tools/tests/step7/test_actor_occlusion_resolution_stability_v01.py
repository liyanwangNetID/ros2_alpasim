from dataclasses import replace

from step7.actor_occlusion_evidence_v01 import (
    build_camera_actor_occlusion_evidence,
)
from step7.actor_occlusion_resolution_comparison_v01 import (
    compare_actor_occlusion_resolutions,
)
from step7.actor_occlusion_resolution_stability_v01 import (
    build_actor_occlusion_resolution_stability_evidence,
)


def evidence(width, height, occupied, winning, occluded, occluders=()):
    return build_camera_actor_occlusion_evidence(
        camera_name="cross_right",
        track_id="143",
        raster_width=width,
        raster_height=height,
        occupied_cell_count=occupied,
        winning_cell_count=winning,
        occluded_cell_count=occluded,
        occluding_actor_ids=occluders,
        actor_to_actor_occlusion_evaluated=True,
        static_occlusion_evaluated=False,
    )


def base_comparison():
    return compare_actor_occlusion_resolutions(
        candidate=evidence(480, 270, 284, 0, 284, ("371", "492", "675")),
        reference=evidence(640, 360, 504, 1, 503, ("371", "492", "675")),
    )


def test_winning_presence_change_is_explicit():
    result = build_actor_occlusion_resolution_stability_evidence(
        base_comparison()
    )

    assert result.stability_status == "winning_presence_changed"
    assert result.comparable
    assert result.sampled_surface_presence_stable
    assert not result.winning_presence_stable
    assert result.occluding_actor_set_stable
    assert result.candidate_winning_cell_count == 0
    assert result.reference_winning_cell_count == 1
    assert result.reasons == ("winning_presence_changed",)


def test_discrete_stability_does_not_require_zero_fraction_delta():
    comparison = replace(
        base_comparison(),
        winning_presence_changed=False,
        candidate_winning_cell_count=10,
        reference_winning_cell_count=20,
        candidate_visible_fraction=0.25,
        reference_visible_fraction=0.30,
        absolute_visible_fraction_delta=0.05,
    )
    result = build_actor_occlusion_resolution_stability_evidence(comparison)

    assert result.stability_status == "discrete_evidence_stable"
    assert result.absolute_visible_fraction_delta == 0.05
    assert result.reasons == ()


def test_surface_change_has_highest_discrete_precedence():
    comparison = replace(
        base_comparison(),
        sampled_surface_presence_changed=True,
        occluding_actor_set_changed=True,
    )
    result = build_actor_occlusion_resolution_stability_evidence(comparison)

    assert result.stability_status == "sampled_surface_presence_changed"
    assert result.reasons == ("sampled_surface_presence_changed",)


def test_occluder_change_is_explicit_when_winner_presence_is_stable():
    comparison = replace(
        base_comparison(),
        winning_presence_changed=False,
        occluding_actor_set_changed=True,
    )
    result = build_actor_occlusion_resolution_stability_evidence(comparison)

    assert result.stability_status == "occluding_actor_set_changed"
    assert result.reasons == ("occluding_actor_set_changed",)


def test_noncomparable_evidence_has_undefined_stability_flags():
    comparison = replace(
        base_comparison(),
        both_evaluated=False,
        winning_presence_changed=None,
        occluding_actor_set_changed=None,
        absolute_visible_fraction_delta=None,
    )
    result = build_actor_occlusion_resolution_stability_evidence(comparison)

    assert result.stability_status == "not_comparable"
    assert not result.comparable
    assert result.sampled_surface_presence_stable is None
    assert result.winning_presence_stable is None
    assert result.occluding_actor_set_stable is None
    assert result.reasons == ("occlusion_evidence_not_comparable",)


def test_to_dict_uses_json_ready_reasons():
    value = build_actor_occlusion_resolution_stability_evidence(
        base_comparison()
    ).to_dict()

    assert isinstance(value["reasons"], list)
