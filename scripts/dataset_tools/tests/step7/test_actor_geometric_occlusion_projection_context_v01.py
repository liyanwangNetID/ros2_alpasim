from types import SimpleNamespace

import pytest

from step7 import actor_geometric_occlusion_projection_context_v01 as target


def geometric(*track_ids):
    actor_results = tuple(
        SimpleNamespace(
            track_id=track_id,
            projections=tuple(
                SimpleNamespace(camera_name=name)
                for name in ("front_wide", "front_tele", "cross_left", "cross_right")
            ),
        )
        for track_id in track_ids
    )
    return SimpleNamespace(
        actor_count=len(actor_results),
        actor_results=actor_results,
    )


def combined(*track_ids):
    return tuple(SimpleNamespace(track_id=track_id) for track_id in track_ids)


def install_builder(monkeypatch, counts=None):
    calls = []
    counts = counts or {}

    def build(*, combined_evidence, projections_by_camera):
        calls.append((combined_evidence.track_id, tuple(projections_by_camera)))
        return tuple(
            SimpleNamespace(camera_name=f"camera_{index}")
            for index in range(counts.get(combined_evidence.track_id, 0))
        )

    monkeypatch.setattr(
        target,
        "build_candidate_without_sampled_surface_projection_evidence",
        build,
    )
    return calls


def test_joins_all_actors_in_track_order(monkeypatch):
    calls = install_builder(monkeypatch, {"2": 1})
    result = target.build_actor_geometric_occlusion_projection_context(
        geometric=geometric("2", "1"),
        combined_evidence=combined("1", "2"),
    )

    assert result.actor_count == 2
    assert tuple(item.track_id for item in result.actor_context) == ("1", "2")
    assert result.missing_surface_projection_context_count == 1
    assert tuple(item[0] for item in calls) == ("1", "2")


def test_preserves_combined_evidence_objects(monkeypatch):
    install_builder(monkeypatch)
    source = combined("1")
    result = target.build_actor_geometric_occlusion_projection_context(
        geometric=geometric("1"),
        combined_evidence=source,
    )

    assert result.actor_context[0].combined_evidence is source[0]


def test_actor_set_mismatch_is_rejected(monkeypatch):
    install_builder(monkeypatch)
    with pytest.raises(ValueError, match="Actor sets must match"):
        target.build_actor_geometric_occlusion_projection_context(
            geometric=geometric("1"),
            combined_evidence=combined("2"),
        )


def test_geometric_count_mismatch_is_rejected(monkeypatch):
    install_builder(monkeypatch)
    value = geometric("1")
    value.actor_count = 2
    with pytest.raises(ValueError, match="are inconsistent"):
        target.build_actor_geometric_occlusion_projection_context(
            geometric=value,
            combined_evidence=combined("1"),
        )


def test_empty_results_are_supported(monkeypatch):
    calls = install_builder(monkeypatch)
    result = target.build_actor_geometric_occlusion_projection_context(
        geometric=geometric(),
        combined_evidence=(),
    )

    assert result.actor_count == 0
    assert result.actor_context == ()
    assert result.missing_surface_projection_context_count == 0
    assert calls == []
