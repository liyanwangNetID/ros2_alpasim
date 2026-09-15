import pytest

from step7.actor_occlusion_evidence_v01 import (
    build_camera_actor_occlusion_evidence,
)
from step7.actor_occlusion_resolution_comparison_v01 import (
    compare_actor_occlusion_resolutions,
)


def evidence(
    *,
    width,
    height,
    occupied,
    winning,
    occluded,
    occluders=(),
    evaluated=True,
    camera="cross_left",
    track="226",
):
    return build_camera_actor_occlusion_evidence(
        camera_name=camera,
        track_id=track,
        raster_width=width,
        raster_height=height,
        occupied_cell_count=occupied,
        winning_cell_count=winning,
        occluded_cell_count=occluded,
        occluding_actor_ids=occluders,
        actor_to_actor_occlusion_evaluated=evaluated,
        static_occlusion_evaluated=False,
    )


def test_compares_evaluated_resolution_evidence():
    candidate = evidence(
        width=480,
        height=270,
        occupied=60,
        winning=0,
        occluded=60,
        occluders=("17", "80"),
    )
    reference = evidence(
        width=640,
        height=360,
        occupied=114,
        winning=4,
        occluded=110,
        occluders=("17", "80"),
    )

    result = compare_actor_occlusion_resolutions(
        candidate=candidate,
        reference=reference,
    )

    assert result.both_evaluated
    assert not result.sampled_surface_presence_changed
    assert result.winning_presence_changed
    assert result.absolute_visible_fraction_delta == pytest.approx(4 / 114)
    assert not result.occluding_actor_set_changed
    assert result.reasons == ()


def test_detects_occluder_set_change():
    candidate = evidence(
        width=480,
        height=270,
        occupied=10,
        winning=5,
        occluded=5,
        occluders=("17",),
    )
    reference = evidence(
        width=640,
        height=360,
        occupied=20,
        winning=10,
        occluded=10,
        occluders=("17", "80"),
    )

    result = compare_actor_occlusion_resolutions(
        candidate=candidate,
        reference=reference,
    )

    assert not result.winning_presence_changed
    assert result.occluding_actor_set_changed
    assert result.absolute_visible_fraction_delta == 0.0


def test_surface_presence_change_is_preserved_without_fraction_comparison():
    candidate = evidence(
        width=480,
        height=270,
        occupied=0,
        winning=0,
        occluded=0,
    )
    reference = evidence(
        width=640,
        height=360,
        occupied=20,
        winning=10,
        occluded=10,
        occluders=("17",),
    )

    result = compare_actor_occlusion_resolutions(
        candidate=candidate,
        reference=reference,
    )

    assert not result.both_evaluated
    assert result.sampled_surface_presence_changed
    assert result.winning_presence_changed is None
    assert result.absolute_visible_fraction_delta is None
    assert result.occluding_actor_set_changed is None
    assert result.reasons == ("visible_fraction_not_comparable",)


def test_unevaluated_reference_preserves_reasons():
    candidate = evidence(
        width=480,
        height=270,
        occupied=10,
        winning=5,
        occluded=5,
        occluders=("17",),
    )
    reference = evidence(
        width=640,
        height=360,
        occupied=0,
        winning=0,
        occluded=0,
        evaluated=False,
    )

    result = compare_actor_occlusion_resolutions(
        candidate=candidate,
        reference=reference,
    )

    assert not result.both_evaluated
    assert result.reasons == (
        "reference_occlusion_not_evaluated",
        "visible_fraction_not_comparable",
    )


def test_camera_mismatch_is_rejected():
    candidate = evidence(
        width=480,
        height=270,
        occupied=10,
        winning=5,
        occluded=5,
    )
    reference = evidence(
        width=640,
        height=360,
        occupied=10,
        winning=5,
        occluded=5,
        camera="cross_right",
    )

    with pytest.raises(ValueError, match="camera_name"):
        compare_actor_occlusion_resolutions(
            candidate=candidate,
            reference=reference,
        )


def test_track_mismatch_is_rejected():
    candidate = evidence(
        width=480,
        height=270,
        occupied=10,
        winning=5,
        occluded=5,
    )
    reference = evidence(
        width=640,
        height=360,
        occupied=10,
        winning=5,
        occluded=5,
        track="other",
    )

    with pytest.raises(ValueError, match="track_id"):
        compare_actor_occlusion_resolutions(
            candidate=candidate,
            reference=reference,
        )


def test_to_dict_uses_json_ready_lists():
    result = compare_actor_occlusion_resolutions(
        candidate=evidence(
            width=480,
            height=270,
            occupied=10,
            winning=5,
            occluded=5,
            occluders=("17",),
        ),
        reference=evidence(
            width=640,
            height=360,
            occupied=20,
            winning=10,
            occluded=10,
            occluders=("17",),
        ),
    )
    value = result.to_dict()

    assert isinstance(value["candidate_occluding_actor_ids"], list)
    assert isinstance(value["reference_occluding_actor_ids"], list)
    assert isinstance(value["reasons"], list)
