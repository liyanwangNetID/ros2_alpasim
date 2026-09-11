from types import SimpleNamespace

import pytest

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from actor_geometric_occlusion_export_inputs_v01 import (
    prepare_actor_geometric_occlusion_export_inputs,
)


def evidence(track_id, actor_class="automobile"):
    return ActorGeometricOcclusionEvidence(
        track_id=track_id,
        actor_class=actor_class,
        geometric_observability_status="candidate_visible",
        geometric_candidate_camera_names=("front_wide",),
        occlusion_evaluated_camera_names=("front_wide",),
        occlusion_winning_camera_names=("front_wide",),
        geometric_candidate_with_sampled_surface_camera_names=("front_wide",),
        geometric_candidate_with_winning_cells_camera_names=("front_wide",),
        geometric_candidate_without_sampled_surface_camera_names=(),
        geometric_candidate_fully_occluded_camera_names=(),
        actor_to_actor_occlusion_evaluated=True,
        static_occlusion_evaluated=False,
        maximum_visible_fraction=1.0,
        total_occupied_cell_count=10,
        total_winning_cell_count=10,
        total_occluded_cell_count=0,
        occluding_actor_ids=(),
        evidence_status="combined_evidence_available",
        reasons=(),
    )


def keyframe():
    return {
        "anchor_id": "clip_100",
        "clip_id": "clip",
        "anchor_ns": 100,
    }


def pipeline(*track_ids):
    values = tuple(evidence(track_id) for track_id in track_ids)
    return SimpleNamespace(
        actor_count=len(values),
        combined_evidence=SimpleNamespace(actor_evidence=values),
    )


def test_prepares_export_inputs_in_track_order():
    result = prepare_actor_geometric_occlusion_export_inputs(
        keyframe=keyframe(),
        actors=(
            {"track_id": "2", "label_class": "automobile", "is_static": True},
            {"track_id": "1", "label_class": "automobile", "is_static": False},
        ),
        pipeline_result=pipeline("1", "2"),
    )

    assert tuple(item.evidence.track_id for item in result) == ("1", "2")
    assert tuple(item.is_static for item in result) == (False, True)
    assert all(item.keyframe is result[0].keyframe for item in result)


def test_actor_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match="sets must match"):
        prepare_actor_geometric_occlusion_export_inputs(
            keyframe=keyframe(),
            actors=(
                {"track_id": "1", "label_class": "automobile", "is_static": False},
            ),
            pipeline_result=pipeline("2"),
        )


def test_duplicate_actor_track_id_is_rejected():
    actor = {"track_id": "1", "label_class": "automobile", "is_static": False}
    with pytest.raises(ValueError, match="duplicate track_id"):
        prepare_actor_geometric_occlusion_export_inputs(
            keyframe=keyframe(),
            actors=(actor, actor),
            pipeline_result=pipeline("1"),
        )


def test_missing_is_static_is_rejected():
    with pytest.raises(ValueError, match="missing is_static"):
        prepare_actor_geometric_occlusion_export_inputs(
            keyframe=keyframe(),
            actors=({"track_id": "1", "label_class": "automobile"},),
            pipeline_result=pipeline("1"),
        )


def test_actor_class_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Actor class"):
        prepare_actor_geometric_occlusion_export_inputs(
            keyframe=keyframe(),
            actors=(
                {"track_id": "1", "label_class": "person", "is_static": False},
            ),
            pipeline_result=pipeline("1"),
        )


def test_pipeline_count_mismatch_is_rejected():
    value = pipeline("1")
    value.actor_count = 2
    with pytest.raises(ValueError, match="count are inconsistent"):
        prepare_actor_geometric_occlusion_export_inputs(
            keyframe=keyframe(),
            actors=(
                {"track_id": "1", "label_class": "automobile", "is_static": False},
            ),
            pipeline_result=value,
        )


def test_empty_anchor_is_supported():
    result = prepare_actor_geometric_occlusion_export_inputs(
        keyframe=keyframe(),
        actors=(),
        pipeline_result=pipeline(),
    )

    assert result == ()
