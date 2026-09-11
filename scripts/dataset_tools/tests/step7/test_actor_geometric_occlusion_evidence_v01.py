from dataclasses import replace

import pytest

from actor_geometric_occlusion_evidence_v01 import (
    build_actor_geometric_occlusion_evidence,
)
from actor_multicamera_occlusion_summary_v01 import (
    ActorMulticameraOcclusionSummary,
)
from actor_observability_v01 import ActorObservability, CameraObservability
from scene_fact_schema_v01 import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION


def geometric(*candidate_cameras):
    return ActorObservability(
        observability_format_version=OBSERVABILITY_FORMAT_VERSION,
        track_id="13",
        actor_class="automobile",
        observability_status=(
            "candidate_visible" if candidate_cameras else "not_visible"
        ),
        visible_in_cameras=tuple(candidate_cameras),
        camera_observability=tuple(
            CameraObservability(
                camera_name=name,
                projection_valid=True,
                geometric_observability_candidate=(name in candidate_cameras),
                failure_reason=(None if name in candidate_cameras else "below"),
            )
            for name in CAMERA_NAMES
        ),
        actor_to_actor_occlusion_evaluated=False,
        static_occlusion_evaluated=False,
    )


def occlusion(**overrides):
    values = {
        "track_id": "13",
        "evaluated_camera_names": tuple(CAMERA_NAMES[:3]),
        "no_sampled_surface_camera_names": (CAMERA_NAMES[3],),
        "winning_camera_names": (CAMERA_NAMES[0], CAMERA_NAMES[2]),
        "fully_occluded_camera_names": (CAMERA_NAMES[1],),
        "occluded_camera_names": (CAMERA_NAMES[1], CAMERA_NAMES[2]),
        "evaluated_camera_count": 3,
        "winning_camera_count": 2,
        "total_occupied_cell_count": 30,
        "total_winning_cell_count": 14,
        "total_occluded_cell_count": 16,
        "maximum_visible_fraction": 1.0,
        "occluding_actor_ids": ("18", "29"),
        "actor_to_actor_occlusion_evaluated": True,
        "static_occlusion_evaluated": False,
        "reasons": (),
    }
    values.update(overrides)
    return ActorMulticameraOcclusionSummary(**values)


def test_combines_geometric_candidates_with_occlusion_camera_sets():
    result = build_actor_geometric_occlusion_evidence(
        geometric=geometric(CAMERA_NAMES[0], CAMERA_NAMES[1], CAMERA_NAMES[3]),
        occlusion=occlusion(),
    )

    assert result.evidence_status == "combined_evidence_available"
    assert result.geometric_candidate_with_sampled_surface_camera_names == (
        CAMERA_NAMES[0],
        CAMERA_NAMES[1],
    )
    assert result.geometric_candidate_with_winning_cells_camera_names == (
        CAMERA_NAMES[0],
    )
    assert result.geometric_candidate_without_sampled_surface_camera_names == (
        CAMERA_NAMES[3],
    )
    assert result.geometric_candidate_fully_occluded_camera_names == (
        CAMERA_NAMES[1],
    )
    assert result.reasons == ()


def test_no_geometric_candidate_is_explicit():
    result = build_actor_geometric_occlusion_evidence(
        geometric=geometric(),
        occlusion=occlusion(),
    )

    assert result.evidence_status == "no_geometric_candidate"
    assert result.reasons == ("no_geometric_candidate_camera",)


def test_candidate_without_surface_is_explicit():
    result = build_actor_geometric_occlusion_evidence(
        geometric=geometric(CAMERA_NAMES[3]),
        occlusion=occlusion(),
    )

    assert result.evidence_status == "candidate_without_sampled_surface"
    assert result.geometric_candidate_without_sampled_surface_camera_names == (
        CAMERA_NAMES[3],
    )
    assert result.reasons == (
        "no_geometric_candidate_has_sampled_surface",
    )


def test_incomplete_occlusion_has_precedence():
    result = build_actor_geometric_occlusion_evidence(
        geometric=geometric(CAMERA_NAMES[0]),
        occlusion=occlusion(actor_to_actor_occlusion_evaluated=False),
    )

    assert result.evidence_status == "occlusion_incomplete"
    assert result.reasons == (
        "actor_to_actor_occlusion_not_fully_evaluated",
    )


def test_continuous_and_occluder_values_are_preserved():
    result = build_actor_geometric_occlusion_evidence(
        geometric=geometric(CAMERA_NAMES[0]),
        occlusion=occlusion(),
    )

    assert result.maximum_visible_fraction == 1.0
    assert result.total_occupied_cell_count == 30
    assert result.total_winning_cell_count == 14
    assert result.total_occluded_cell_count == 16
    assert result.occluding_actor_ids == ("18", "29")


def test_track_mismatch_is_rejected():
    with pytest.raises(ValueError, match="track_id values must match"):
        build_actor_geometric_occlusion_evidence(
            geometric=geometric(CAMERA_NAMES[0]),
            occlusion=replace(occlusion(), track_id="other"),
        )


def test_noncanonical_geometric_order_is_rejected():
    with pytest.raises(ValueError, match="canonical order"):
        build_actor_geometric_occlusion_evidence(
            geometric=geometric(CAMERA_NAMES[1], CAMERA_NAMES[0]),
            occlusion=occlusion(),
        )


def test_to_dict_uses_json_ready_lists():
    value = build_actor_geometric_occlusion_evidence(
        geometric=geometric(CAMERA_NAMES[0]),
        occlusion=occlusion(),
    ).to_dict()

    assert isinstance(value["geometric_candidate_camera_names"], list)
    assert isinstance(value["occlusion_winning_camera_names"], list)
    assert isinstance(value["occluding_actor_ids"], list)
    assert isinstance(value["reasons"], list)
