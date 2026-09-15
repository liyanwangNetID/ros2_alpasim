from types import SimpleNamespace

import pytest

from step7 import actor_geometric_occlusion_with_projection_context_v01 as target


def install_stubs(monkeypatch, *, pipeline_count=2, context_count=2):
    calls = []
    evidence = tuple(
        SimpleNamespace(track_id=str(index))
        for index in range(pipeline_count)
    )
    pipeline = SimpleNamespace(
        actor_count=pipeline_count,
        geometric=SimpleNamespace(source="geometric"),
        combined=SimpleNamespace(
            combined_evidence=SimpleNamespace(actor_evidence=evidence)
        ),
    )

    def build_pipeline(**kwargs):
        calls.append(("pipeline", kwargs))
        return pipeline

    context_items = tuple(
        SimpleNamespace(track_id=str(index))
        for index in range(context_count)
    )

    def build_context(**kwargs):
        calls.append(("context", kwargs))
        return SimpleNamespace(
            actor_count=context_count,
            actor_context=context_items,
            missing_surface_projection_context_count=1,
        )

    monkeypatch.setattr(
        target,
        "build_actor_geometric_occlusion_from_geometry",
        build_pipeline,
    )
    monkeypatch.setattr(
        target,
        "build_actor_geometric_occlusion_projection_context",
        build_context,
    )
    return calls, pipeline


def arguments(actor_count=2):
    return {
        "actors": tuple({"track_id": str(index)} for index in range(actor_count)),
        "recorded_ego_message": {"ego": True},
        "cameras": {"camera": object()},
        "raster_width": 480,
        "raster_height": 270,
        "maximum_depth": 8,
        "maximum_boundary_extent_px": 4.0,
    }


def test_runs_pipeline_then_attaches_projection_context(monkeypatch):
    calls, pipeline = install_stubs(monkeypatch)
    result = target.build_actor_geometric_occlusion_with_projection_context(
        **arguments()
    )

    assert tuple(name for name, _ in calls) == ("pipeline", "context")
    assert result.actor_count == 2
    assert result.pipeline is pipeline
    assert result.projection_context.missing_surface_projection_context_count == 1
    assert calls[1][1]["geometric"] is pipeline.geometric
    assert calls[1][1]["combined_evidence"] is (
        pipeline.combined.combined_evidence.actor_evidence
    )


def test_projection_and_occlusion_parameters_are_forwarded(monkeypatch):
    calls, _ = install_stubs(monkeypatch)
    values = arguments()
    values.update({
        "samples_per_edge": 9,
        "near_plane_m": 0.01,
        "maximum_chord_error_px": 0.5,
        "maximum_adaptive_depth": 11,
        "depth_tolerance_m": 1e-8,
    })
    target.build_actor_geometric_occlusion_with_projection_context(**values)

    pipeline_call = calls[0][1]
    assert pipeline_call["samples_per_edge"] == 9
    assert pipeline_call["near_plane_m"] == 0.01
    assert pipeline_call["maximum_chord_error_px"] == 0.5
    assert pipeline_call["maximum_adaptive_depth"] == 11
    assert pipeline_call["depth_tolerance_m"] == 1e-8


def test_pipeline_count_mismatch_is_rejected(monkeypatch):
    install_stubs(monkeypatch, pipeline_count=1, context_count=1)
    with pytest.raises(RuntimeError, match="differs from input"):
        target.build_actor_geometric_occlusion_with_projection_context(
            **arguments(actor_count=2)
        )


def test_context_count_mismatch_is_rejected(monkeypatch):
    install_stubs(monkeypatch, pipeline_count=2, context_count=1)
    with pytest.raises(RuntimeError, match="Actor count differs"):
        target.build_actor_geometric_occlusion_with_projection_context(
            **arguments()
        )


def test_empty_input_is_supported(monkeypatch):
    calls, _ = install_stubs(monkeypatch, pipeline_count=0, context_count=0)
    result = target.build_actor_geometric_occlusion_with_projection_context(
        **arguments(actor_count=0)
    )

    assert result.actor_count == 0
    assert result.projection_context.actor_context == ()
    assert tuple(name for name, _ in calls) == ("pipeline", "context")
