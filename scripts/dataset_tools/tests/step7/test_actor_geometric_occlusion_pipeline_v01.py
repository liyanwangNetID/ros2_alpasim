from types import SimpleNamespace

import pytest

from actor_geometric_occlusion_pipeline_v01 import (
    build_actor_geometric_occlusion_pipeline,
)
from actor_multicamera_occlusion_summary_v01 import (
    ActorMulticameraOcclusionSummary,
)
from actor_observability_v01 import ActorObservability, CameraObservability
from scene_fact_schema_v01 import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION


def geometric(track_id):
    return ActorObservability(
        observability_format_version=OBSERVABILITY_FORMAT_VERSION,
        track_id=track_id,
        actor_class="automobile",
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


def summary(track_id):
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


def multicamera(track_ids=("1", "2")):
    summaries = tuple(summary(track_id) for track_id in track_ids)
    evidence = tuple(SimpleNamespace(track_id=track_id) for track_id in track_ids)
    return SimpleNamespace(
        actor_count=len(track_ids),
        actor_summaries=summaries,
        actor_evidence=evidence,
    )


def test_composes_complete_results_in_track_order():
    result = build_actor_geometric_occlusion_pipeline(
        geometric_observability=(geometric("2"), geometric("1")),
        multicamera_occlusion=multicamera(),
    )

    assert result.actor_count == 2
    assert tuple(
        item.track_id for item in result.geometric_observability
    ) == ("1", "2")
    assert tuple(
        item.track_id for item in result.combined_evidence.actor_evidence
    ) == ("1", "2")


def test_preserves_multicamera_result_object():
    source = multicamera()
    result = build_actor_geometric_occlusion_pipeline(
        geometric_observability=(geometric("1"), geometric("2")),
        multicamera_occlusion=source,
    )

    assert result.multicamera_occlusion is source


def test_multicamera_summary_count_mismatch_is_rejected():
    source = multicamera()
    source.actor_summaries = source.actor_summaries[:1]

    with pytest.raises(ValueError, match="actor_count and summaries"):
        build_actor_geometric_occlusion_pipeline(
            geometric_observability=(geometric("1"), geometric("2")),
            multicamera_occlusion=source,
        )


def test_multicamera_evidence_count_mismatch_is_rejected():
    source = multicamera()
    source.actor_evidence = source.actor_evidence[:1]

    with pytest.raises(ValueError, match="actor_count and evidence"):
        build_actor_geometric_occlusion_pipeline(
            geometric_observability=(geometric("1"), geometric("2")),
            multicamera_occlusion=source,
        )


def test_actor_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Actor sets must match"):
        build_actor_geometric_occlusion_pipeline(
            geometric_observability=(geometric("1"), geometric("3")),
            multicamera_occlusion=multicamera(),
        )


def test_empty_results_are_supported():
    source = multicamera(())
    result = build_actor_geometric_occlusion_pipeline(
        geometric_observability=(),
        multicamera_occlusion=source,
    )

    assert result.actor_count == 0
    assert result.combined_evidence.actor_evidence == ()


def test_pipeline_builds_combined_evidence_summary():
    result = build_actor_geometric_occlusion_pipeline(
        geometric_observability=(geometric("2"), geometric("1")),
        multicamera_occlusion=multicamera(),
    )

    assert result.combined_summary.actor_count == 2
    assert dict(result.combined_summary.evidence_status_counts) == {
        "combined_evidence_available": 2,
    }
    assert result.combined_summary.geometric_candidate_actor_count == 2
    assert (
        result.combined_summary
        .geometric_candidate_with_sampled_surface_actor_count
        == 2
    )
    assert (
        result.combined_summary
        .geometric_candidate_with_winning_cells_actor_count
        == 2
    )
    assert result.combined_summary.reasons == ()


def test_empty_pipeline_builds_empty_summary():
    result = build_actor_geometric_occlusion_pipeline(
        geometric_observability=(),
        multicamera_occlusion=multicamera(()),
    )

    assert result.combined_summary.actor_count == 0
    assert result.combined_summary.evidence_status_counts == ()
