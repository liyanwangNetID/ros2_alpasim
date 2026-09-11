from dataclasses import replace

import pytest

from actor_export_anchor_coverage_summary_v01 import (
    ActorExportAnchorCoverageSummary,
)
from actor_export_anchor_snapshot_coverage_v01 import (
    ActorExportAnchorSnapshotCoverageSummary,
    RowlessAnchorSnapshotEvidence,
)
from step7e_v02_summary_anchor_coverage_v01 import (
    attach_anchor_snapshot_coverage_to_summary,
)


def coverage():
    base = ActorExportAnchorCoverageSummary(
        source_keyframe_count=3,
        anchor_with_actor_rows_count=2,
        anchor_without_actor_rows_count=1,
        anchor_without_actor_rows=("b",),
    )
    evidence = RowlessAnchorSnapshotEvidence(
        anchor_id="b",
        clip_id="clip",
        anchor_ns=2,
        reason="exact_snapshot_empty_actor_list",
        actor_count=0,
        detail=None,
    )
    return ActorExportAnchorSnapshotCoverageSummary(
        coverage=base,
        rowless_anchor_reason_counts=(("exact_snapshot_empty_actor_list", 1),),
        rowless_anchor_snapshot_evidence=(evidence,),
    )


def summary():
    return {
        "schema_version": "step7e-geometric-occlusion-evidence-v02",
        "row_count": 10,
        "anchor_count": 2,
        "evidence_status_counts": {"combined_evidence_available": 10},
    }


def test_attaches_source_and_rowless_anchor_coverage_without_mutating_input():
    source = summary()
    result = attach_anchor_snapshot_coverage_to_summary(
        summary=source,
        coverage=coverage(),
    )

    assert result["source_keyframe_count"] == 3
    assert result["anchor_with_actor_rows_count"] == 2
    assert result["anchor_without_actor_rows_count"] == 1
    assert result["anchor_without_actor_rows"] == ["b"]
    assert result["rowless_anchor_reason_counts"] == {
        "exact_snapshot_empty_actor_list": 1
    }
    assert result["rowless_anchor_snapshot_evidence"][0]["actor_count"] == 0
    assert "source_keyframe_count" not in source


def test_summary_anchor_count_must_match_anchors_with_rows():
    value = summary()
    value["anchor_count"] = 3

    with pytest.raises(ValueError, match="differs"):
        attach_anchor_snapshot_coverage_to_summary(
            summary=value,
            coverage=coverage(),
        )


def test_rowless_reason_counts_must_close():
    value = replace(coverage(), rowless_anchor_reason_counts=())

    with pytest.raises(ValueError, match="reason counts"):
        attach_anchor_snapshot_coverage_to_summary(
            summary=summary(),
            coverage=value,
        )


def test_rowless_snapshot_evidence_count_must_close():
    value = replace(coverage(), rowless_anchor_snapshot_evidence=())

    with pytest.raises(ValueError, match="evidence count"):
        attach_anchor_snapshot_coverage_to_summary(
            summary=summary(),
            coverage=value,
        )


def test_missing_required_summary_field_is_rejected():
    value = summary()
    del value["row_count"]

    with pytest.raises(ValueError, match="missing row_count"):
        attach_anchor_snapshot_coverage_to_summary(
            summary=value,
            coverage=coverage(),
        )
