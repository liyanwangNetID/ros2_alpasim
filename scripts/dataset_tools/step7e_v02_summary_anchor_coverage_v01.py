"""Attach source-Anchor snapshot coverage to a Step 7E v02 summary.

The adapter combines an existing JSONL-derived v02 summary with the reusable
Anchor snapshot coverage result. It validates row and Anchor closure, then adds
explicit source-Keyframe and rowless-Anchor fields. It does not read files,
recompute geometry, or alter Actor evidence rows.
"""

from __future__ import annotations

from typing import Any, Mapping

from actor_export_anchor_snapshot_coverage_v01 import (
    ActorExportAnchorSnapshotCoverageSummary,
)


def attach_anchor_snapshot_coverage_to_summary(
    *,
    summary: Mapping[str, Any],
    coverage: ActorExportAnchorSnapshotCoverageSummary,
) -> dict[str, Any]:
    """Return a copied v02 summary with validated Anchor coverage fields."""

    required = ("row_count", "anchor_count", "schema_version")
    for field in required:
        if field not in summary:
            raise ValueError(f"summary is missing {field}")

    result = dict(summary)
    anchor_coverage = coverage.coverage
    if int(result["anchor_count"]) != (
        anchor_coverage.anchor_with_actor_rows_count
    ):
        raise ValueError(
            "summary anchor_count differs from Anchors with Actor rows"
        )
    if (
        anchor_coverage.anchor_with_actor_rows_count
        + anchor_coverage.anchor_without_actor_rows_count
        != anchor_coverage.source_keyframe_count
    ):
        raise ValueError("source Anchor coverage counts do not close")
    if len(anchor_coverage.anchor_without_actor_rows) != (
        anchor_coverage.anchor_without_actor_rows_count
    ):
        raise ValueError("rowless Anchor list and count are inconsistent")
    if sum(dict(coverage.rowless_anchor_reason_counts).values()) != (
        anchor_coverage.anchor_without_actor_rows_count
    ):
        raise ValueError("rowless Anchor reason counts do not close")
    if len(coverage.rowless_anchor_snapshot_evidence) != (
        anchor_coverage.anchor_without_actor_rows_count
    ):
        raise ValueError("rowless Anchor snapshot evidence count does not close")

    result.update(coverage.to_dict())
    return result
