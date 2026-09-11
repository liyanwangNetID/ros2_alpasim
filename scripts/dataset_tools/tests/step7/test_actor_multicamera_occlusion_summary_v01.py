import pytest

from actor_multicamera_occlusion_summary_v01 import (
    summarize_actor_multicamera_occlusion,
)
from actor_occlusion_evidence_v01 import (
    build_camera_actor_occlusion_evidence,
)
from scene_fact_schema_v01 import CAMERA_NAMES


def evidence(
    camera_name,
    *,
    occupied=10,
    winning=5,
    occluded=5,
    occluders=("9",),
    evaluated=True,
    reasons=(),
):
    return build_camera_actor_occlusion_evidence(
        camera_name=camera_name,
        track_id="13",
        raster_width=480,
        raster_height=270,
        occupied_cell_count=occupied,
        winning_cell_count=winning,
        occluded_cell_count=occluded,
        occluding_actor_ids=occluders,
        actor_to_actor_occlusion_evaluated=evaluated,
        static_occlusion_evaluated=False,
        reasons=reasons,
    )


def complete_source():
    return (
        evidence(CAMERA_NAMES[0], occupied=10, winning=10, occluded=0, occluders=()),
        evidence(CAMERA_NAMES[1], occupied=10, winning=0, occluded=10, occluders=("29",)),
        evidence(CAMERA_NAMES[2], occupied=10, winning=4, occluded=6, occluders=("18",)),
        evidence(CAMERA_NAMES[3], occupied=0, winning=0, occluded=0, occluders=()),
    )


def test_summarizes_threshold_free_cross_camera_features():
    result = summarize_actor_multicamera_occlusion(
        track_id="13",
        camera_evidence=complete_source(),
    )

    assert result.evaluated_camera_names == tuple(CAMERA_NAMES[:3])
    assert result.no_sampled_surface_camera_names == (CAMERA_NAMES[3],)
    assert result.winning_camera_names == (CAMERA_NAMES[0], CAMERA_NAMES[2])
    assert result.fully_occluded_camera_names == (CAMERA_NAMES[1],)
    assert result.occluded_camera_names == (CAMERA_NAMES[1], CAMERA_NAMES[2])
    assert result.evaluated_camera_count == 3
    assert result.winning_camera_count == 2
    assert result.total_occupied_cell_count == 30
    assert result.total_winning_cell_count == 14
    assert result.total_occluded_cell_count == 16
    assert result.maximum_visible_fraction == 1.0
    assert result.occluding_actor_ids == ("18", "29")
    assert result.actor_to_actor_occlusion_evaluated
    assert not result.static_occlusion_evaluated
    assert result.reasons == ()


def test_all_no_surface_has_undefined_maximum_fraction():
    source = tuple(
        evidence(
            camera_name,
            occupied=0,
            winning=0,
            occluded=0,
            occluders=(),
        )
        for camera_name in CAMERA_NAMES
    )
    result = summarize_actor_multicamera_occlusion(
        track_id="13",
        camera_evidence=source,
    )

    assert result.evaluated_camera_count == 0
    assert result.winning_camera_count == 0
    assert result.maximum_visible_fraction is None
    assert result.actor_to_actor_occlusion_evaluated
    assert result.reasons == ("no_camera_has_sampled_surface",)


def test_not_evaluated_camera_is_preserved_as_reason():
    source = list(complete_source())
    source[3] = evidence(
        CAMERA_NAMES[3],
        occupied=0,
        winning=0,
        occluded=0,
        occluders=(),
        evaluated=False,
        reasons=("camera_input_unavailable",),
    )
    result = summarize_actor_multicamera_occlusion(
        track_id="13",
        camera_evidence=tuple(source),
    )

    assert not result.actor_to_actor_occlusion_evaluated
    assert result.reasons == ("one_or_more_cameras_not_evaluated",)


def test_camera_order_is_required():
    source = complete_source()
    with pytest.raises(ValueError, match="must follow CAMERA_NAMES"):
        summarize_actor_multicamera_occlusion(
            track_id="13",
            camera_evidence=tuple(reversed(source)),
        )


def test_track_id_mismatch_is_rejected():
    source = list(complete_source())
    wrong = build_camera_actor_occlusion_evidence(
        camera_name=CAMERA_NAMES[0],
        track_id="wrong",
        raster_width=480,
        raster_height=270,
        occupied_cell_count=10,
        winning_cell_count=10,
        occluded_cell_count=0,
        occluding_actor_ids=(),
        actor_to_actor_occlusion_evaluated=True,
    )
    source[0] = wrong

    with pytest.raises(ValueError, match="must match"):
        summarize_actor_multicamera_occlusion(
            track_id="13",
            camera_evidence=tuple(source),
        )


def test_to_dict_uses_json_ready_lists():
    result = summarize_actor_multicamera_occlusion(
        track_id="13",
        camera_evidence=complete_source(),
    )
    value = result.to_dict()

    assert isinstance(value["evaluated_camera_names"], list)
    assert isinstance(value["winning_camera_names"], list)
    assert isinstance(value["occluding_actor_ids"], list)
    assert isinstance(value["reasons"], list)
