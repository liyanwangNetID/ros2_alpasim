import json
from dataclasses import replace

import pytest

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from actor_geometric_occlusion_export_v02 import (
    SCHEMA_VERSION,
    actor_geometric_occlusion_projection_context_export_record,
    encode_actor_geometric_occlusion_projection_context_export_record,
)
from actor_geometric_occlusion_projection_context_v01 import (
    ActorGeometricOcclusionProjectionContext,
)
from candidate_without_sampled_surface_projection_evidence_v01 import (
    CandidateWithoutSampledSurfaceProjectionEvidence,
)


def keyframe():
    return {
        "anchor_id": "clip_100",
        "clip_id": "clip",
        "anchor_ns": 100,
    }


def evidence(*, missing=("cross_left",)):
    return ActorGeometricOcclusionEvidence(
        track_id="13",
        actor_class="automobile",
        geometric_observability_status="candidate_visible",
        geometric_candidate_camera_names=missing,
        occlusion_evaluated_camera_names=(),
        occlusion_winning_camera_names=(),
        geometric_candidate_with_sampled_surface_camera_names=(),
        geometric_candidate_with_winning_cells_camera_names=(),
        geometric_candidate_without_sampled_surface_camera_names=missing,
        geometric_candidate_fully_occluded_camera_names=(),
        actor_to_actor_occlusion_evaluated=True,
        static_occlusion_evaluated=False,
        maximum_visible_fraction=None,
        total_occupied_cell_count=0,
        total_winning_cell_count=0,
        total_occluded_cell_count=0,
        occluding_actor_ids=(),
        evidence_status="candidate_without_sampled_surface",
        reasons=("no_geometric_candidate_has_sampled_surface",),
    )


def projection_context(camera_name="cross_left"):
    return CandidateWithoutSampledSurfaceProjectionEvidence(
        camera_name=camera_name,
        track_id="13",
        actor_class="automobile",
        projection_valid=True,
        projection_truncated=True,
        inside_image_hull_area_px=3.5,
        projected_height_px=18.0,
        inside_image_hull_ratio=0.01,
        minimum_depth_m=20.0,
        maximum_depth_m=24.0,
        evidence_status="missing_surface_with_truncated_projection",
        reasons=(
            "geometric_candidate_without_sampled_surface",
            "projection_truncated_by_image",
        ),
    )


def context(*projection_values):
    value = evidence(missing=tuple(
        item.camera_name for item in projection_values
    ))
    return ActorGeometricOcclusionProjectionContext(
        track_id="13",
        combined_evidence=value,
        candidate_without_sampled_surface_projection_evidence=tuple(
            projection_values
        ),
    )


def test_extends_v01_row_with_projection_context():
    row = actor_geometric_occlusion_projection_context_export_record(
        keyframe=keyframe(),
        context=context(projection_context()),
        is_static=True,
    )

    assert row["schema_version"] == SCHEMA_VERSION
    assert row["track_id"] == "13"
    assert row["is_static"] is True
    values = row[
        "candidate_without_sampled_surface_projection_evidence"
    ]
    assert len(values) == 1
    assert values[0]["camera_name"] == "cross_left"
    assert values[0]["projection_truncated"] is True
    assert isinstance(values[0]["reasons"], list)


def test_empty_projection_context_is_exported_for_ordinary_actor():
    value = context()
    row = actor_geometric_occlusion_projection_context_export_record(
        keyframe=keyframe(),
        context=value,
        is_static=False,
    )

    assert row[
        "candidate_without_sampled_surface_projection_evidence"
    ] == []


def test_compact_encoder_round_trips():
    encoded = encode_actor_geometric_occlusion_projection_context_export_record(
        keyframe=keyframe(),
        context=context(projection_context()),
        is_static=False,
    )

    assert "\n" not in encoded
    assert ": " not in encoded
    assert json.loads(encoded)["schema_version"] == SCHEMA_VERSION


def test_context_track_mismatch_is_rejected():
    value = replace(context(projection_context()), track_id="other")

    with pytest.raises(ValueError, match="track_id values differ"):
        actor_geometric_occlusion_projection_context_export_record(
            keyframe=keyframe(),
            context=value,
            is_static=False,
        )


def test_context_camera_mismatch_is_rejected():
    value = ActorGeometricOcclusionProjectionContext(
        track_id="13",
        combined_evidence=evidence(missing=("cross_left",)),
        candidate_without_sampled_surface_projection_evidence=(
            projection_context("front_wide"),
        ),
    )

    with pytest.raises(ValueError, match="cameras differ"):
        actor_geometric_occlusion_projection_context_export_record(
            keyframe=keyframe(),
            context=value,
            is_static=False,
        )


def test_duplicate_context_camera_is_rejected():
    item = projection_context()
    value = ActorGeometricOcclusionProjectionContext(
        track_id="13",
        combined_evidence=evidence(
            missing=("cross_left", "cross_left")
        ),
        candidate_without_sampled_surface_projection_evidence=(item, item),
    )

    with pytest.raises(ValueError, match="duplicate camera"):
        actor_geometric_occlusion_projection_context_export_record(
            keyframe=keyframe(),
            context=value,
            is_static=False,
        )
