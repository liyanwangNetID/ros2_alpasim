from dataclasses import replace

import pytest

from actor_geometric_occlusion_evidence_set_v01 import (
    build_actor_geometric_occlusion_evidence_set,
)
from actor_multicamera_occlusion_summary_v01 import (
    ActorMulticameraOcclusionSummary,
)
from actor_observability_v01 import ActorObservability, CameraObservability
from scene_fact_schema_v01 import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION


def geometric(track_id, actor_class="automobile"):
    return ActorObservability(
        observability_format_version=OBSERVABILITY_FORMAT_VERSION,
        track_id=track_id,
        actor_class=actor_class,
        observability_status="candidate_visible",
        visible_in_cameras=(CAMERA_NAMES[0],),
        camera_observability=tuple(
            CameraObservability(
                camera_name=name,
                projection_valid=True,
                geometric_observability_candidate=(name == CAMERA_NAMES[0]),
                failure_reason=(None if name == CAMERA_NAMES[0] else "below"),
            )
            for name in CAMERA_NAMES
        ),
        actor_to_actor_occlusion_evaluated=False,
        static_occlusion_evaluated=False,
    )


def occlusion(track_id):
    return ActorMulticameraOcclusionSummary(
        track_id=track_id,
        evaluated_camera_names=(CAMERA_NAMES[0],),
        no_sampled_surface_camera_names=tuple(CAMERA_NAMES[1:]),
        winning_camera_names=(CAMERA_NAMES[0],),
        fully_occluded_camera_names=(),
        occluded_camera_names=(),
        evaluated_camera_count=1,
        winning_camera_count=1,
        total_occupied_cell_count=10,
        total_winning_cell_count=10,
        total_occluded_cell_count=0,
        maximum_visible_fraction=1.0,
        occluding_actor_ids=(),
        actor_to_actor_occlusion_evaluated=True,
        static_occlusion_evaluated=False,
        reasons=(),
    )


def test_joins_complete_actor_sets_in_track_order():
    result = build_actor_geometric_occlusion_evidence_set(
        geometric_observability=(geometric("2"), geometric("1")),
        occlusion_summaries=(occlusion("1"), occlusion("2")),
    )

    assert result.actor_count == 2
    assert tuple(item.track_id for item in result.actor_evidence) == (
        "1",
        "2",
    )
    assert all(
        item.evidence_status == "combined_evidence_available"
        for item in result.actor_evidence
    )


def test_combined_evidence_preserves_actor_class():
    result = build_actor_geometric_occlusion_evidence_set(
        geometric_observability=(geometric("1", "person"),),
        occlusion_summaries=(occlusion("1"),),
    )

    assert result.actor_evidence[0].actor_class == "person"


def test_geometric_duplicate_track_id_is_rejected():
    with pytest.raises(ValueError, match="geometric observability contains duplicate"):
        build_actor_geometric_occlusion_evidence_set(
            geometric_observability=(geometric("1"), geometric("1")),
            occlusion_summaries=(occlusion("1"),),
        )


def test_occlusion_duplicate_track_id_is_rejected():
    with pytest.raises(ValueError, match="occlusion summaries contain duplicate"):
        build_actor_geometric_occlusion_evidence_set(
            geometric_observability=(geometric("1"),),
            occlusion_summaries=(occlusion("1"), occlusion("1")),
        )


def test_actor_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Actor sets must match"):
        build_actor_geometric_occlusion_evidence_set(
            geometric_observability=(geometric("1"), geometric("2")),
            occlusion_summaries=(occlusion("1"), occlusion("3")),
        )


def test_empty_sets_produce_empty_result():
    result = build_actor_geometric_occlusion_evidence_set(
        geometric_observability=(),
        occlusion_summaries=(),
    )

    assert result.actor_count == 0
    assert result.actor_evidence == ()
