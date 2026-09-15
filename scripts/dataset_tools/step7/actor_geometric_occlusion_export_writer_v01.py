"""Write Step 7E combined geometric-occlusion evidence atomically.

This module serializes complete per-Actor combined evidence rows to compact
JSONL and writes a deterministic summary JSON. It operates on already computed
evidence, so it does not rerun projection or occlusion geometry and does not
select final visibility labels.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from step7.actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from step7.actor_geometric_occlusion_export_v01 import (
    SCHEMA_VERSION,
    encode_actor_geometric_occlusion_export_record,
)


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionExportInput:
    keyframe: Mapping[str, Any]
    evidence: ActorGeometricOcclusionEvidence
    is_static: bool


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionExportSummary:
    schema_version: str
    row_count: int
    anchor_count: int
    evidence_status_counts: tuple[tuple[str, int], ...]
    rows_by_actor_class: tuple[tuple[str, int], ...]
    actor_to_actor_occlusion_evaluated_count: int
    static_occlusion_evaluated_count: int
    duplicate_key_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "description": (
                "Threshold-free Step 7E geometric and Actor-to-Actor "
                "occlusion evidence. No final visibility labels."
            ),
            "row_count": self.row_count,
            "anchor_count": self.anchor_count,
            "evidence_status_counts": dict(self.evidence_status_counts),
            "rows_by_actor_class": dict(self.rows_by_actor_class),
            "actor_to_actor_occlusion_evaluated_count": (
                self.actor_to_actor_occlusion_evaluated_count
            ),
            "static_occlusion_evaluated_count": (
                self.static_occlusion_evaluated_count
            ),
            "duplicate_key_count": self.duplicate_key_count,
        }


def _identity(item: ActorGeometricOcclusionExportInput) -> tuple[str, str]:
    keyframe = item.keyframe
    if "anchor_id" not in keyframe:
        raise ValueError("keyframe is missing anchor_id")
    return str(keyframe["anchor_id"]), item.evidence.track_id


def write_actor_geometric_occlusion_evidence(
    *,
    inputs: Sequence[ActorGeometricOcclusionExportInput],
    output_path: Path,
    summary_path: Path,
) -> ActorGeometricOcclusionExportSummary:
    """Validate, sort, and atomically write JSONL evidence and summary JSON."""

    source = tuple(inputs)
    identities = tuple(_identity(item) for item in source)
    duplicate_count = len(identities) - len(set(identities))
    if duplicate_count:
        raise ValueError("duplicate anchor_id/track_id export keys")

    ordered = tuple(sorted(source, key=_identity))
    lines = tuple(
        encode_actor_geometric_occlusion_export_record(
            keyframe=item.keyframe,
            evidence=item.evidence,
            is_static=item.is_static,
        )
        for item in ordered
    )

    status_counts = Counter(
        item.evidence.evidence_status for item in ordered
    )
    class_counts = Counter(item.evidence.actor_class for item in ordered)
    summary = ActorGeometricOcclusionExportSummary(
        schema_version=SCHEMA_VERSION,
        row_count=len(ordered),
        anchor_count=len({identity[0] for identity in identities}),
        evidence_status_counts=tuple(sorted(status_counts.items())),
        rows_by_actor_class=tuple(sorted(class_counts.items())),
        actor_to_actor_occlusion_evaluated_count=sum(
            item.evidence.actor_to_actor_occlusion_evaluated
            for item in ordered
        ),
        static_occlusion_evaluated_count=sum(
            item.evidence.static_occlusion_evaluated for item in ordered
        ),
        duplicate_key_count=0,
    )

    output_path = Path(output_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")

    try:
        output_temporary.write_text(
            "".join(line + "\n" for line in lines),
            encoding="utf-8",
        )
        summary_temporary.write_text(
            json.dumps(
                summary.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(output_temporary, output_path)
        os.replace(summary_temporary, summary_path)
    except Exception:
        output_temporary.unlink(missing_ok=True)
        summary_temporary.unlink(missing_ok=True)
        raise

    return summary
