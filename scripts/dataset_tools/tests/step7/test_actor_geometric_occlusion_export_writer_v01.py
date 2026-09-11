import json
from dataclasses import replace

import pytest

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from actor_geometric_occlusion_export_writer_v01 import (
    ActorGeometricOcclusionExportInput,
    write_actor_geometric_occlusion_evidence,
)
from actor_geometric_occlusion_export_v01 import SCHEMA_VERSION


def evidence(track_id, actor_class="automobile", status="combined_evidence_available"):
    return ActorGeometricOcclusionEvidence(
        track_id=track_id,
        actor_class=actor_class,
        geometric_observability_status="candidate_visible",
        geometric_candidate_camera_names=("front_wide",),
        occlusion_evaluated_camera_names=("front_wide",),
        occlusion_winning_camera_names=("front_wide",),
        geometric_candidate_with_sampled_surface_camera_names=("front_wide",),
        geometric_candidate_with_winning_cells_camera_names=("front_wide",),
        geometric_candidate_without_sampled_surface_camera_names=(),
        geometric_candidate_fully_occluded_camera_names=(),
        actor_to_actor_occlusion_evaluated=True,
        static_occlusion_evaluated=False,
        maximum_visible_fraction=1.0,
        total_occupied_cell_count=10,
        total_winning_cell_count=10,
        total_occluded_cell_count=0,
        occluding_actor_ids=(),
        evidence_status=status,
        reasons=(),
    )


def item(anchor_id, track_id, actor_class="automobile"):
    clip_id, anchor_ns = anchor_id.rsplit("_", 1)
    return ActorGeometricOcclusionExportInput(
        keyframe={
            "anchor_id": anchor_id,
            "clip_id": clip_id,
            "anchor_ns": int(anchor_ns),
        },
        evidence=evidence(track_id, actor_class),
        is_static=False,
    )


def test_writes_deterministic_jsonl_and_summary(tmp_path):
    output = tmp_path / "evidence.jsonl"
    summary_path = tmp_path / "evidence.summary.json"
    summary = write_actor_geometric_occlusion_evidence(
        inputs=(
            item("clip_b_200", "2", "person"),
            item("clip_a_100", "9"),
            item("clip_a_100", "1"),
        ),
        output_path=output,
        summary_path=summary_path,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["anchor_id"], row["track_id"]) for row in rows] == [
        ("clip_a_100", "1"),
        ("clip_a_100", "9"),
        ("clip_b_200", "2"),
    ]
    assert summary.schema_version == SCHEMA_VERSION
    assert summary.row_count == 3
    assert summary.anchor_count == 2
    assert dict(summary.rows_by_actor_class) == {
        "automobile": 2,
        "person": 1,
    }
    saved = json.loads(summary_path.read_text())
    assert saved["duplicate_key_count"] == 0


def test_empty_export_writes_empty_jsonl(tmp_path):
    output = tmp_path / "evidence.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = write_actor_geometric_occlusion_evidence(
        inputs=(),
        output_path=output,
        summary_path=summary_path,
    )

    assert output.read_text() == ""
    assert summary.row_count == 0
    assert json.loads(summary_path.read_text())["anchor_count"] == 0


def test_duplicate_anchor_actor_key_is_rejected_before_writing(tmp_path):
    value = item("clip_a_100", "1")
    output = tmp_path / "evidence.jsonl"
    summary_path = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="duplicate"):
        write_actor_geometric_occlusion_evidence(
            inputs=(value, value),
            output_path=output,
            summary_path=summary_path,
        )

    assert not output.exists()
    assert not summary_path.exists()


def test_invalid_record_leaves_no_temporary_files(tmp_path):
    value = item("clip_a_100", "1")
    value = replace(
        value,
        evidence=replace(value.evidence, static_occlusion_evaluated=True),
    )
    output = tmp_path / "evidence.jsonl"
    summary_path = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="must remain unevaluated"):
        write_actor_geometric_occlusion_evidence(
            inputs=(value,),
            output_path=output,
            summary_path=summary_path,
        )

    assert not output.exists()
    assert not summary_path.exists()
    assert not output.with_suffix(".jsonl.tmp").exists()
    assert not summary_path.with_suffix(".json.tmp").exists()
