"""Classify rowless Step 7E Anchors from exact Actor snapshots.

This module accepts source Keyframes, exported Actor identities, and a reader
factory. It first computes Anchor coverage, then inspects only rowless Anchors
and reports whether each has one exact snapshot with an empty Actor list. It
does not run projection, occlusion, or final visibility classification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from step7.actor_export_anchor_coverage_summary_v01 import (
    ActorExportAnchorCoverageSummary,
    summarize_actor_export_anchor_coverage,
)


@dataclass(frozen=True, slots=True)
class RowlessAnchorSnapshotEvidence:
    anchor_id: str
    clip_id: str
    anchor_ns: int
    reason: str
    actor_count: int | None
    detail: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActorExportAnchorSnapshotCoverageSummary:
    coverage: ActorExportAnchorCoverageSummary
    rowless_anchor_reason_counts: tuple[tuple[str, int], ...]
    rowless_anchor_snapshot_evidence: tuple[
        RowlessAnchorSnapshotEvidence, ...
    ]

    def to_dict(self) -> dict[str, Any]:
        value = self.coverage.to_dict()
        value["rowless_anchor_reason_counts"] = dict(
            self.rowless_anchor_reason_counts
        )
        value["rowless_anchor_snapshot_evidence"] = [
            item.to_dict() for item in self.rowless_anchor_snapshot_evidence
        ]
        return value


def _inspect_exact_snapshot(reader, anchor_ns: int):
    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if len(snapshots) != 1:
        return "exact_snapshot_count_not_one", None, len(snapshots)
    snapshot = snapshots[0]
    if snapshot.stamp_ns != anchor_ns:
        return "snapshot_timestamp_not_exact", None, snapshot.stamp_ns
    actors = snapshot.message.get("actors")
    if not isinstance(actors, list):
        return "actors_field_not_list", None, type(actors).__name__
    if actors:
        return "exact_snapshot_has_actors", len(actors), None
    return "exact_snapshot_empty_actor_list", 0, None


def summarize_actor_export_anchor_snapshot_coverage(
    *,
    keyframes: Sequence[Mapping[str, Any]],
    exported_actor_identities: Sequence[tuple[str, str]],
    reader_factory: Callable[[str], Any],
) -> ActorExportAnchorSnapshotCoverageSummary:
    """Compute Anchor coverage and classify each rowless exact snapshot."""

    source = tuple(keyframes)
    coverage = summarize_actor_export_anchor_coverage(
        keyframes=source,
        exported_actor_identities=exported_actor_identities,
    )
    keyframes_by_id = {
        str(item["anchor_id"]): item for item in source
    }

    evidence = []
    reasons = Counter()
    current_clip_id = None
    reader = None
    for anchor_id in coverage.anchor_without_actor_rows:
        keyframe = keyframes_by_id[anchor_id]
        for field in ("clip_id", "anchor_ns"):
            if field not in keyframe:
                raise ValueError(
                    f"rowless keyframe {anchor_id} is missing {field}"
                )
        clip_id = str(keyframe["clip_id"])
        anchor_ns = int(keyframe["anchor_ns"])
        if clip_id != current_clip_id:
            current_clip_id = clip_id
            reader = reader_factory(clip_id)
        reason, actor_count, detail = _inspect_exact_snapshot(
            reader,
            anchor_ns,
        )
        reasons[reason] += 1
        evidence.append(
            RowlessAnchorSnapshotEvidence(
                anchor_id=anchor_id,
                clip_id=clip_id,
                anchor_ns=anchor_ns,
                reason=reason,
                actor_count=actor_count,
                detail=detail,
            )
        )

    return ActorExportAnchorSnapshotCoverageSummary(
        coverage=coverage,
        rowless_anchor_reason_counts=tuple(sorted(reasons.items())),
        rowless_anchor_snapshot_evidence=tuple(evidence),
    )
