from types import SimpleNamespace

import pytest

from step7 import multicamera_actor_occlusion_from_geometry_v01 as target
from step7.actor_occlusion_evidence_v01 import (
    build_camera_actor_occlusion_evidence,
)
from step7.scene_fact_schema_v01 import CAMERA_NAMES


def actor(track_id):
    return {"track_id": track_id}


def cameras():
    return {
        camera_name: target.CameraOcclusionBuildInput(
            calibration=object(),
            image_width_px=1920,
            image_height_px=1080,
        )
        for camera_name in CAMERA_NAMES
    }


def install_stub(monkeypatch, *, reverse_one_camera=False):
    def build_one(*, actors, camera_name, raster_width, raster_height, **kwargs):
        track_ids = sorted(str(actor["track_id"]) for actor in actors)
        if reverse_one_camera and camera_name == CAMERA_NAMES[0]:
            track_ids.reverse()
        evidence = tuple(
            build_camera_actor_occlusion_evidence(
                camera_name=camera_name,
                track_id=track_id,
                raster_width=raster_width,
                raster_height=raster_height,
                occupied_cell_count=10,
                winning_cell_count=5,
                occluded_cell_count=5,
                occluding_actor_ids=("other",),
                actor_to_actor_occlusion_evaluated=True,
                static_occlusion_evaluated=False,
            )
            for track_id in track_ids
        )
        return SimpleNamespace(
            camera_name=camera_name,
            actor_count=len(actors),
            occlusion=SimpleNamespace(actor_evidence=evidence),
        )

    monkeypatch.setattr(
        target,
        "build_actor_camera_occlusion_from_geometry",
        build_one,
    )


def call(monkeypatch, **overrides):
    install_stub(monkeypatch)
    values = {
        "actors": (actor("2"), actor("1")),
        "recorded_ego_message": {"ego": True},
        "cameras": cameras(),
        "raster_width": 480,
        "raster_height": 270,
        "maximum_depth": 8,
        "maximum_boundary_extent_px": 4.0,
    }
    values.update(overrides)
    return target.build_multicamera_actor_occlusion_from_geometry(**values)


def test_builds_all_cameras_in_canonical_order(monkeypatch):
    result = call(monkeypatch)

    assert tuple(
        item.camera_name for item in result.camera_results
    ) == tuple(CAMERA_NAMES)
    assert result.actor_count == 2


def test_groups_evidence_by_actor_and_camera(monkeypatch):
    result = call(monkeypatch)

    assert tuple(item.track_id for item in result.actor_evidence) == ("1", "2")
    for item in result.actor_evidence:
        assert tuple(
            evidence.camera_name for evidence in item.camera_evidence
        ) == tuple(CAMERA_NAMES)
        assert len(item.camera_evidence) == len(CAMERA_NAMES)


def test_missing_camera_is_rejected(monkeypatch):
    source = cameras()
    del source[CAMERA_NAMES[0]]

    with pytest.raises(ValueError, match="missing"):
        call(monkeypatch, cameras=source)


def test_extra_camera_is_rejected(monkeypatch):
    source = cameras()
    source["extra"] = target.CameraOcclusionBuildInput(
        calibration=object(),
        image_width_px=1,
        image_height_px=1,
    )

    with pytest.raises(ValueError, match="extra"):
        call(monkeypatch, cameras=source)


def test_duplicate_track_ids_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="must be unique"):
        call(monkeypatch, actors=(actor("1"), actor("1")))


def test_camera_actor_set_mismatch_is_rejected(monkeypatch):
    install_stub(monkeypatch, reverse_one_camera=True)

    with pytest.raises(RuntimeError, match="Actor IDs differ"):
        target.build_multicamera_actor_occlusion_from_geometry(
            actors=(actor("2"), actor("1")),
            recorded_ego_message={"ego": True},
            cameras=cameras(),
            raster_width=480,
            raster_height=270,
            maximum_depth=8,
            maximum_boundary_extent_px=4.0,
        )


def test_builds_threshold_free_actor_summaries(monkeypatch):
    result = call(monkeypatch)

    assert tuple(
        item.track_id for item in result.actor_summaries
    ) == ("1", "2")
    for item in result.actor_summaries:
        assert item.evaluated_camera_count == len(CAMERA_NAMES)
        assert item.winning_camera_count == len(CAMERA_NAMES)
        assert item.total_occupied_cell_count == 10 * len(CAMERA_NAMES)
        assert item.total_winning_cell_count == 5 * len(CAMERA_NAMES)
        assert item.total_occluded_cell_count == 5 * len(CAMERA_NAMES)
        assert item.maximum_visible_fraction == 0.5
        assert item.occluding_actor_ids == ("other",)
        assert item.actor_to_actor_occlusion_evaluated
        assert not item.static_occlusion_evaluated
        assert item.reasons == ()


def test_summary_camera_order_matches_grouped_evidence(monkeypatch):
    result = call(monkeypatch)

    grouped_by_id = {
        item.track_id: item for item in result.actor_evidence
    }
    for summary in result.actor_summaries:
        grouped = grouped_by_id[summary.track_id]
        assert summary.evaluated_camera_names == tuple(
            item.camera_name for item in grouped.camera_evidence
        )
