import hashlib
import json
from types import SimpleNamespace

import pytest

from step7 import actor_geometric_occlusion_export_writer_v02 as target


def context(track_id, *, status="combined_evidence_available", count=0):
    projections = tuple(
        SimpleNamespace(
            evidence_status="missing_surface_with_truncated_projection",
            camera_name=("cross_left" if index == 0 else "cross_right"),
        )
        for index in range(count)
    )
    return SimpleNamespace(
        track_id=track_id,
        combined_evidence=SimpleNamespace(
            track_id=track_id,
            actor_class="automobile",
            evidence_status=status,
            actor_to_actor_occlusion_evaluated=True,
            static_occlusion_evaluated=False,
        ),
        candidate_without_sampled_surface_projection_evidence=projections,
    )


def item(anchor_id, track_id, *, count=0):
    return target.ActorGeometricOcclusionProjectionContextExportInput(
        keyframe={
            "anchor_id": anchor_id,
            "clip_id": "clip",
            "anchor_ns": 100,
        },
        context=context(track_id, count=count),
        is_static=False,
    )


def install_encoder(monkeypatch):
    def encode(*, keyframe, context, is_static):
        return json.dumps({
            "schema_version": target.SCHEMA_VERSION,
            "anchor_id": keyframe["anchor_id"],
            "track_id": context.track_id,
            "is_static": is_static,
        }, separators=(",", ":"))

    monkeypatch.setattr(
        target,
        "encode_actor_geometric_occlusion_projection_context_export_record",
        encode,
    )


def test_writes_sorted_v02_rows_and_context_summary(monkeypatch, tmp_path):
    install_encoder(monkeypatch)
    output = tmp_path / "evidence_v02.jsonl"
    summary_path = tmp_path / "evidence_v02.summary.json"

    summary = target.write_actor_geometric_occlusion_projection_context_evidence(
        inputs=(
            item("b", "2"),
            item("a", "9"),
            item("a", "1", count=1),
        ),
        output_path=output,
        summary_path=summary_path,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["anchor_id"], row["track_id"]) for row in rows] == [
        ("a", "1"),
        ("a", "9"),
        ("b", "2"),
    ]
    assert summary["row_count"] == 3
    assert summary["anchor_count"] == 2
    assert summary["missing_surface_projection_context_count"] == 1
    assert summary[
        "missing_surface_projection_context_status_counts"
    ] == {"missing_surface_with_truncated_projection": 1}
    expected_digest = hashlib.sha256(
        output.read_bytes()
    ).hexdigest()

    assert summary["output_sha256"] == expected_digest
    assert json.loads(
        summary_path.read_text()
    )["output_sha256"] == expected_digest
    assert json.loads(summary_path.read_text())["schema_version"] == (
        target.SCHEMA_VERSION
    )


def test_duplicate_identity_is_rejected_before_writing(monkeypatch, tmp_path):
    install_encoder(monkeypatch)
    value = item("a", "1")
    output = tmp_path / "evidence.jsonl"
    summary_path = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="duplicate"):
        target.write_actor_geometric_occlusion_projection_context_evidence(
            inputs=(value, value),
            output_path=output,
            summary_path=summary_path,
        )

    assert not output.exists()
    assert not summary_path.exists()


def test_empty_export_is_supported(monkeypatch, tmp_path):
    install_encoder(monkeypatch)
    output = tmp_path / "evidence.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = target.write_actor_geometric_occlusion_projection_context_evidence(
        inputs=(),
        output_path=output,
        summary_path=summary_path,
    )

    assert output.read_text() == ""
    assert summary["row_count"] == 0
    assert summary["missing_surface_projection_context_count"] == 0
    assert summary["output_sha256"] == hashlib.sha256(
        b""
    ).hexdigest()


def test_encoder_failure_leaves_no_outputs(monkeypatch, tmp_path):
    def fail(**kwargs):
        raise ValueError("invalid context")

    monkeypatch.setattr(
        target,
        "encode_actor_geometric_occlusion_projection_context_export_record",
        fail,
    )
    output = tmp_path / "evidence.jsonl"
    summary_path = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="invalid context"):
        target.write_actor_geometric_occlusion_projection_context_evidence(
            inputs=(item("a", "1"),),
            output_path=output,
            summary_path=summary_path,
        )

    assert not output.exists()
    assert not summary_path.exists()


def test_summary_separates_actor_and_camera_context_scopes(monkeypatch, tmp_path):
    install_encoder(monkeypatch)
    special = target.ActorGeometricOcclusionProjectionContextExportInput(
        keyframe={"anchor_id": "a", "clip_id": "clip", "anchor_ns": 100},
        context=context("3", status="candidate_without_sampled_surface", count=2),
        is_static=False,
    )
    summary = target.write_actor_geometric_occlusion_projection_context_evidence(
        inputs=(item("a", "1", count=1), item("a", "2"), special),
        output_path=tmp_path / "evidence.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    assert summary["actors_with_missing_surface_projection_context_by_evidence_status"] == {
        "candidate_without_sampled_surface": 1,
        "combined_evidence_available": 1,
    }
    assert summary["missing_surface_projection_context_parent_evidence_status_counts"] == {
        "candidate_without_sampled_surface": 2,
        "combined_evidence_available": 1,
    }
    assert summary["missing_surface_projection_context_camera_counts"] == {
        "cross_left": 2,
        "cross_right": 1,
    }
    assert summary["missing_surface_projection_context_count_per_actor_histogram"] == {
        "0": 1,
        "1": 1,
        "2": 1,
    }

