from dataclasses import replace

import pytest

from actor_box_image_projection_v01 import ActorCameraProjection
from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from candidate_without_sampled_surface_projection_evidence_v01 import (
    build_candidate_without_sampled_surface_projection_evidence,
)
from scene_fact_schema_v01 import CAMERA_NAMES


def projection(camera_name, *, truncated=False, valid=True):
    return ActorCameraProjection(
        camera_name=camera_name,
        track_id="13",
        actor_class="automobile",
        corner_count=8,
        edge_count=12,
        edge_samples_per_edge=0,
        camera_sample_count=36,
        positive_depth_sample_count=36,
        within_fov_sample_count=36,
        inside_image_sample_count=1,
        projected_bbox=None,
        clipped_bbox=None,
        projected_area_px=100.0,
        inside_image_area_px=2.0,
        inside_image_ratio=0.02,
        projected_hull=(),
        clipped_hull=(),
        projected_hull_area_px=90.0,
        inside_image_hull_area_px=1.5,
        inside_image_hull_ratio=0.016,
        projected_height_px=18.0,
        minimum_depth_m=20.0,
        maximum_depth_m=24.0,
        truncated=truncated,
        projection_valid=valid,
        failure_reason=(None if valid else "projected_box_outside_image"),
    )


def projections(**overrides):
    values = {name: projection(name) for name in CAMERA_NAMES}
    values.update(overrides)
    return values


def evidence():
    return ActorGeometricOcclusionEvidence(
        track_id="13",
        actor_class="automobile",
        geometric_observability_status="candidate_visible",
        geometric_candidate_camera_names=("cross_left",),
        occlusion_evaluated_camera_names=(),
        occlusion_winning_camera_names=(),
        geometric_candidate_with_sampled_surface_camera_names=(),
        geometric_candidate_with_winning_cells_camera_names=(),
        geometric_candidate_without_sampled_surface_camera_names=(
            "cross_left",
        ),
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


def test_reports_truncated_projection_context_without_causal_claim():
    result = build_candidate_without_sampled_surface_projection_evidence(
        combined_evidence=evidence(),
        projections_by_camera=projections(
            cross_left=projection("cross_left", truncated=True)
        ),
    )

    assert len(result) == 1
    assert result[0].camera_name == "cross_left"
    assert result[0].evidence_status == (
        "missing_surface_with_truncated_projection"
    )
    assert result[0].reasons == (
        "geometric_candidate_without_sampled_surface",
        "projection_truncated_by_image",
    )
    assert result[0].inside_image_hull_area_px == 1.5


def test_reports_untruncated_projection_separately():
    result = build_candidate_without_sampled_surface_projection_evidence(
        combined_evidence=evidence(),
        projections_by_camera=projections(),
    )

    assert result[0].evidence_status == (
        "missing_surface_with_untruncated_projection"
    )
    assert result[0].reasons == (
        "geometric_candidate_without_sampled_surface",
    )


def test_invalid_projection_has_explicit_status():
    result = build_candidate_without_sampled_surface_projection_evidence(
        combined_evidence=evidence(),
        projections_by_camera=projections(
            cross_left=projection("cross_left", valid=False)
        ),
    )

    assert result[0].evidence_status == "missing_surface_with_invalid_projection"
    assert result[0].reasons[-1] == "projection_invalid"


def test_non_missing_candidate_cameras_are_not_emitted():
    result = build_candidate_without_sampled_surface_projection_evidence(
        combined_evidence=replace(
            evidence(),
            geometric_candidate_camera_names=("front_wide", "cross_left"),
            geometric_candidate_with_sampled_surface_camera_names=(
                "front_wide",
            ),
        ),
        projections_by_camera=projections(),
    )

    assert tuple(item.camera_name for item in result) == ("cross_left",)


def test_projection_track_mismatch_is_rejected():
    values = projections()
    values["cross_left"] = replace(
        values["cross_left"],
        track_id="other",
    )

    with pytest.raises(ValueError, match="track_id differ"):
        build_candidate_without_sampled_surface_projection_evidence(
            combined_evidence=evidence(),
            projections_by_camera=values,
        )


def test_missing_camera_input_is_rejected():
    values = projections()
    del values[CAMERA_NAMES[0]]

    with pytest.raises(ValueError, match="missing"):
        build_candidate_without_sampled_surface_projection_evidence(
            combined_evidence=evidence(),
            projections_by_camera=values,
        )


def test_to_dict_uses_json_ready_reasons():
    result = build_candidate_without_sampled_surface_projection_evidence(
        combined_evidence=evidence(),
        projections_by_camera=projections(),
    )

    assert isinstance(result[0].to_dict()["reasons"], list)
