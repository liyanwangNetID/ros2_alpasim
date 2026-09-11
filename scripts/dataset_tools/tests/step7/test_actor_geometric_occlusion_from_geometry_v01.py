from types import SimpleNamespace

import pytest

import actor_geometric_occlusion_from_geometry_v01 as target
from scene_fact_schema_v01 import CAMERA_NAMES


def cameras():
    return {
        name: target.CameraGeometricOcclusionBuildInput(
            calibration=SimpleNamespace(camera_name=name),
            image_width_px=1920,
            image_height_px=1080,
        )
        for name in CAMERA_NAMES
    }


def install_stubs(monkeypatch):
    calls = []

    def geometric(**kwargs):
        calls.append(("geometric", kwargs))
        values = tuple(
            SimpleNamespace(track_id=str(actor["track_id"]))
            for actor in sorted(kwargs["actors"], key=lambda item: str(item["track_id"]))
        )
        return SimpleNamespace(
            actor_count=len(values),
            actor_observability=values,
        )

    def occlusion(**kwargs):
        calls.append(("occlusion", kwargs))
        values = tuple(
            SimpleNamespace(track_id=str(actor["track_id"]))
            for actor in sorted(kwargs["actors"], key=lambda item: str(item["track_id"]))
        )
        return SimpleNamespace(
            actor_count=len(values),
            actor_summaries=values,
            actor_evidence=values,
        )

    def combine(*, geometric_observability, multicamera_occlusion):
        calls.append(("combine", {
            "geometric_observability": geometric_observability,
            "multicamera_occlusion": multicamera_occlusion,
        }))
        evidence = SimpleNamespace(actor_evidence=geometric_observability)
        return SimpleNamespace(
            actor_count=len(geometric_observability),
            combined_evidence=evidence,
        )

    monkeypatch.setattr(
        target,
        "build_multicamera_actor_geometric_observability",
        geometric,
    )
    monkeypatch.setattr(
        target,
        "build_multicamera_actor_occlusion_from_geometry",
        occlusion,
    )
    monkeypatch.setattr(
        target,
        "build_actor_geometric_occlusion_pipeline",
        combine,
    )
    return calls


def call(monkeypatch, **overrides):
    calls = install_stubs(monkeypatch)
    values = {
        "actors": (
            {"track_id": "2", "label_class": "automobile"},
            {"track_id": "1", "label_class": "person"},
        ),
        "recorded_ego_message": {"ego": True},
        "cameras": cameras(),
        "raster_width": 480,
        "raster_height": 270,
        "maximum_depth": 8,
        "maximum_boundary_extent_px": 4.0,
    }
    values.update(overrides)
    return target.build_actor_geometric_occlusion_from_geometry(**values), calls


def test_runs_geometric_occlusion_and_combination(monkeypatch):
    result, calls = call(monkeypatch)

    assert result.actor_count == 2
    assert tuple(name for name, _ in calls) == (
        "geometric",
        "occlusion",
        "combine",
    )
    assert tuple(
        item.track_id
        for item in result.combined.combined_evidence.actor_evidence
    ) == ("1", "2")


def test_uses_same_calibration_objects_for_both_paths(monkeypatch):
    result, calls = call(monkeypatch)
    assert result.actor_count == 2
    geometric_call = calls[0][1]
    occlusion_call = calls[1][1]

    for name in CAMERA_NAMES:
        assert (
            geometric_call["calibrations"][name]
            is occlusion_call["cameras"][name].calibration
        )


def test_parameters_are_forwarded_to_correct_paths(monkeypatch):
    _, calls = call(
        monkeypatch,
        samples_per_edge=9,
        near_plane_m=0.01,
        maximum_chord_error_px=0.5,
        maximum_adaptive_depth=11,
        depth_tolerance_m=1e-8,
    )
    geometric_call = calls[0][1]
    occlusion_call = calls[1][1]

    assert geometric_call["samples_per_edge"] == 9
    assert geometric_call["maximum_chord_error_px"] == 0.5
    assert geometric_call["maximum_adaptive_depth"] == 11
    assert geometric_call["near_plane_m"] == 0.01
    assert occlusion_call["near_plane_m"] == 0.01
    assert occlusion_call["depth_tolerance_m"] == 1e-8


def test_camera_dimensions_are_forwarded_to_occlusion_path(monkeypatch):
    _, calls = call(monkeypatch)
    occlusion_call = calls[1][1]

    for name in CAMERA_NAMES:
        camera = occlusion_call["cameras"][name]
        assert camera.image_width_px == 1920
        assert camera.image_height_px == 1080


def test_missing_camera_is_rejected(monkeypatch):
    source = cameras()
    del source[CAMERA_NAMES[0]]

    with pytest.raises(ValueError, match="missing"):
        call(monkeypatch, cameras=source)


def test_extra_camera_is_rejected(monkeypatch):
    source = cameras()
    source["extra"] = target.CameraGeometricOcclusionBuildInput(
        calibration=object(),
        image_width_px=1,
        image_height_px=1,
    )

    with pytest.raises(ValueError, match="extra"):
        call(monkeypatch, cameras=source)


def test_empty_actor_set_is_supported(monkeypatch):
    result, _ = call(monkeypatch, actors=())

    assert result.actor_count == 0
    assert result.combined.combined_evidence.actor_evidence == ()
