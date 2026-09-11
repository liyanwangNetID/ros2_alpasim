import json
from dataclasses import replace

import pytest

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from actor_geometric_occlusion_export_v01 import (
    SCHEMA_VERSION,
    actor_geometric_occlusion_export_record,
    encode_actor_geometric_occlusion_export_record,
)


def evidence():
    return ActorGeometricOcclusionEvidence(
        track_id="13",
        actor_class="automobile",
        geometric_observability_status="candidate_visible",
        geometric_candidate_camera_names=(
            "front_wide",
            "front_tele",
            "cross_left",
        ),
        occlusion_evaluated_camera_names=(
            "front_wide",
            "front_tele",
            "cross_left",
        ),
        occlusion_winning_camera_names=(
            "front_wide",
            "front_tele",
            "cross_left",
        ),
        geometric_candidate_with_sampled_surface_camera_names=(
            "front_wide",
            "front_tele",
            "cross_left",
        ),
        geometric_candidate_with_winning_cells_camera_names=(
            "front_wide",
            "front_tele",
            "cross_left",
        ),
        geometric_candidate_without_sampled_surface_camera_names=(),
        geometric_candidate_fully_occluded_camera_names=(),
        actor_to_actor_occlusion_evaluated=True,
        static_occlusion_evaluated=False,
        maximum_visible_fraction=0.25,
        total_occupied_cell_count=100,
        total_winning_cell_count=25,
        total_occluded_cell_count=75,
        occluding_actor_ids=("29",),
        evidence_status="combined_evidence_available",
        reasons=(),
    )


def keyframe():
    return {
        "anchor_id": "test_clip_001_9306612661000",
        "clip_id": "test_clip_001",
        "anchor_ns": 9306612661000,
    }


def test_builds_stable_json_ready_actor_row():
    row = actor_geometric_occlusion_export_record(
        keyframe=keyframe(),
        evidence=evidence(),
        is_static=False,
    )

    assert row["schema_version"] == SCHEMA_VERSION
    assert row["anchor_id"] == "test_clip_001_9306612661000"
    assert row["track_id"] == "13"
    assert row["label_class"] == "automobile"
    assert row["geometric_candidate_camera_names"] == [
        "front_wide",
        "front_tele",
        "cross_left",
    ]
    assert row["occluding_actor_ids"] == ["29"]
    assert row["maximum_visible_fraction"] == 0.25
    assert not row["static_occlusion_evaluated"]


def test_compact_encoder_round_trips_record():
    encoded = encode_actor_geometric_occlusion_export_record(
        keyframe=keyframe(),
        evidence=evidence(),
        is_static=True,
    )

    assert "\n" not in encoded
    assert ": " not in encoded
    row = json.loads(encoded)
    assert row["is_static"] is True
    assert row["track_id"] == "13"


def test_undefined_visible_fraction_is_preserved_as_null():
    row = actor_geometric_occlusion_export_record(
        keyframe=keyframe(),
        evidence=replace(
            evidence(),
            maximum_visible_fraction=None,
            total_occupied_cell_count=0,
            total_winning_cell_count=0,
            total_occluded_cell_count=0,
        ),
        is_static=False,
    )

    assert row["maximum_visible_fraction"] is None


def test_missing_keyframe_field_is_rejected():
    value = keyframe()
    del value["anchor_ns"]

    with pytest.raises(ValueError, match="missing anchor_ns"):
        actor_geometric_occlusion_export_record(
            keyframe=value,
            evidence=evidence(),
            is_static=False,
        )


def test_invalid_visible_fraction_is_rejected():
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        actor_geometric_occlusion_export_record(
            keyframe=keyframe(),
            evidence=replace(evidence(), maximum_visible_fraction=1.1),
            is_static=False,
        )


def test_cell_counts_must_close():
    with pytest.raises(ValueError, match="must close"):
        actor_geometric_occlusion_export_record(
            keyframe=keyframe(),
            evidence=replace(evidence(), total_occluded_cell_count=74),
            is_static=False,
        )


def test_static_occlusion_evidence_is_rejected():
    with pytest.raises(ValueError, match="must remain unevaluated"):
        actor_geometric_occlusion_export_record(
            keyframe=keyframe(),
            evidence=replace(evidence(), static_occlusion_evaluated=True),
            is_static=False,
        )
