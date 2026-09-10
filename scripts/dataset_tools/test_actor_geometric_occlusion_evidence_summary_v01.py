from dataclasses import replace

import pytest

from actor_geometric_occlusion_evidence_summary_v01 import (
    summarize_actor_geometric_occlusion_evidence,
)
from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)


def item(
    track_id,
    *,
    status="combined_evidence_available",
    candidates=("front_wide",),
    sampled=("front_wide",),
    winners=("front_wide",),
    without_surface=(),
    fully_occluded=(),
    occluders=(),
    complete=True,
    static=False,
):
    return ActorGeometricOcclusionEvidence(
        track_id=track_id,
        actor_class="automobile",
        geometric_observability_status=(
            "candidate_visible" if candidates else "not_visible"
        ),
        geometric_candidate_camera_names=candidates,
        occlusion_evaluated_camera_names=("front_wide",),
        occlusion_winning_camera_names=winners,
        geometric_candidate_with_sampled_surface_camera_names=sampled,
        geometric_candidate_with_winning_cells_camera_names=winners,
        geometric_candidate_without_sampled_surface_camera_names=without_surface,
        geometric_candidate_fully_occluded_camera_names=fully_occluded,
        actor_to_actor_occlusion_evaluated=complete,
        static_occlusion_evaluated=static,
        maximum_visible_fraction=1.0 if sampled else None,
        total_occupied_cell_count=10 if sampled else 0,
        total_winning_cell_count=10 if winners else 0,
        total_occluded_cell_count=0 if winners else (10 if sampled else 0),
        occluding_actor_ids=occluders,
        evidence_status=status,
        reasons=(),
    )


def source():
    return (
        item("1"),
        item(
            "2",
            winners=(),
            fully_occluded=("front_wide",),
            occluders=("9",),
        ),
        item(
            "3",
            status="candidate_without_sampled_surface",
            sampled=(),
            winners=(),
            without_surface=("front_wide",),
        ),
        item(
            "4",
            status="no_geometric_candidate",
            candidates=(),
            sampled=(),
            winners=(),
        ),
    )


def test_summarizes_combined_evidence_counts():
    result = summarize_actor_geometric_occlusion_evidence(source())

    assert result.actor_count == 4
    assert dict(result.evidence_status_counts) == {
        "candidate_without_sampled_surface": 1,
        "combined_evidence_available": 2,
        "no_geometric_candidate": 1,
    }
    assert result.geometric_candidate_actor_count == 3
    assert result.geometric_candidate_with_sampled_surface_actor_count == 2
    assert result.geometric_candidate_with_winning_cells_actor_count == 1
    assert result.geometric_candidate_without_sampled_surface_actor_count == 1
    assert result.geometric_candidate_fully_occluded_actor_count == 1
    assert result.actor_with_occluder_count == 1
    assert result.actor_to_actor_occlusion_complete_count == 4
    assert result.static_occlusion_evaluated_count == 0


def test_preserves_review_track_ids():
    result = summarize_actor_geometric_occlusion_evidence(source())

    assert result.geometric_candidate_without_sampled_surface_track_ids == ("3",)
    assert result.geometric_candidate_fully_occluded_track_ids == ("2",)


def test_incomplete_occlusion_is_reported():
    values = list(source())
    values[0] = replace(values[0], actor_to_actor_occlusion_evaluated=False)
    result = summarize_actor_geometric_occlusion_evidence(values)

    assert result.actor_to_actor_occlusion_complete_count == 3
    assert result.reasons == (
        "one_or_more_actors_have_incomplete_occlusion_evidence",
    )


def test_static_occlusion_presence_is_reported():
    values = list(source())
    values[0] = replace(values[0], static_occlusion_evaluated=True)
    result = summarize_actor_geometric_occlusion_evidence(values)

    assert result.static_occlusion_evaluated_count == 1
    assert result.reasons == (
        "one_or_more_actors_have_static_occlusion_evidence",
    )


def test_duplicate_track_ids_are_rejected():
    first = source()[0]
    with pytest.raises(ValueError, match="must be unique"):
        summarize_actor_geometric_occlusion_evidence((first, first))


def test_empty_input_produces_empty_summary():
    result = summarize_actor_geometric_occlusion_evidence(())

    assert result.actor_count == 0
    assert result.evidence_status_counts == ()
    assert result.reasons == ()


def test_to_dict_uses_mapping_and_lists():
    value = summarize_actor_geometric_occlusion_evidence(source()).to_dict()

    assert isinstance(value["evidence_status_counts"], dict)
    assert isinstance(
        value["geometric_candidate_without_sampled_surface_track_ids"],
        list,
    )
    assert isinstance(value["reasons"], list)
