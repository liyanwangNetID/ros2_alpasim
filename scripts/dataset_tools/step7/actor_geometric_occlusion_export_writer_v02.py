"""Atomically write Step 7E v02 Actor evidence with projection context.

The writer accepts already computed per-Actor projection-context records,
validates unique Anchor/Actor identities, writes deterministic compact JSONL,
and emits a summary JSON. It does not rerun geometry or alter the v01 product.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from step7.actor_geometric_occlusion_export_v02 import (
    SCHEMA_VERSION,
    encode_actor_geometric_occlusion_projection_context_export_record,
)
from step7.actor_geometric_occlusion_projection_context_v01 import (
    ActorGeometricOcclusionProjectionContext,
)


@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionProjectionContextExportInput:
    keyframe: Mapping[str, Any]
    context: ActorGeometricOcclusionProjectionContext
    is_static: bool


def _identity(
    item: ActorGeometricOcclusionProjectionContextExportInput,
) -> tuple[str, str]:
    if "anchor_id" not in item.keyframe:
        raise ValueError("keyframe is missing anchor_id")
    return str(item.keyframe["anchor_id"]), item.context.track_id


def write_actor_geometric_occlusion_projection_context_evidence(
    *,
    inputs: Sequence[
        ActorGeometricOcclusionProjectionContextExportInput
    ],
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Validate and atomically write deterministic v02 JSONL and summary."""

    source = tuple(inputs)
    identities = tuple(_identity(item) for item in source)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate anchor_id/track_id export keys")

    ordered = tuple(sorted(source, key=_identity))
    lines = tuple(
        encode_actor_geometric_occlusion_projection_context_export_record(
            keyframe=item.keyframe,
            context=item.context,
            is_static=item.is_static,
        )
        for item in ordered
    )
    output_text = "".join(
        line + "\n"
        for line in lines
    )
    output_sha256 = hashlib.sha256(
        output_text.encode("utf-8")
    ).hexdigest()

    status_counts = Counter(
        item.context.combined_evidence.evidence_status for item in ordered
    )
    class_counts = Counter(
        item.context.combined_evidence.actor_class for item in ordered
    )
    projection_context_status_counts = Counter(
        projection.evidence_status
        for item in ordered
        for projection in (
            item.context
            .candidate_without_sampled_surface_projection_evidence
        )
    )
    projection_context_camera_counts = Counter(
        projection.camera_name
        for item in ordered
        for projection in item.context.candidate_without_sampled_surface_projection_evidence
    )
    actors_with_projection_context_by_evidence_status = Counter(
        item.context.combined_evidence.evidence_status
        for item in ordered
        if item.context.candidate_without_sampled_surface_projection_evidence
    )
    projection_context_parent_evidence_status_counts = Counter(
        item.context.combined_evidence.evidence_status
        for item in ordered
        for _ in item.context.candidate_without_sampled_surface_projection_evidence
    )
    projection_context_count_per_actor_histogram = Counter(
        len(item.context.candidate_without_sampled_surface_projection_evidence)
        for item in ordered
    )
    projection_context_count = sum(
        len(
            item.context
            .candidate_without_sampled_surface_projection_evidence
        )
        for item in ordered
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Threshold-free Step 7E geometric and Actor-to-Actor occlusion "
            "evidence with missing-surface projection context. No final "
            "visibility labels."
        ),
        "row_count": len(ordered),
        "anchor_count": len({identity[0] for identity in identities}),
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "rows_by_actor_class": dict(sorted(class_counts.items())),
        "missing_surface_projection_context_count": projection_context_count,
        "missing_surface_projection_context_status_counts": dict(
            sorted(projection_context_status_counts.items())
        ),
        "missing_surface_projection_context_camera_counts": dict(
            sorted(projection_context_camera_counts.items())
        ),
        "actors_with_missing_surface_projection_context_by_evidence_status": dict(
            sorted(actors_with_projection_context_by_evidence_status.items())
        ),
        "missing_surface_projection_context_parent_evidence_status_counts": dict(
            sorted(projection_context_parent_evidence_status_counts.items())
        ),
        "missing_surface_projection_context_count_per_actor_histogram": {
            str(count): actor_count
            for count, actor_count in sorted(
                projection_context_count_per_actor_histogram.items()
            )
        },
        "actor_to_actor_occlusion_evaluated_count": sum(
            item.context.combined_evidence
            .actor_to_actor_occlusion_evaluated
            for item in ordered
        ),
        "static_occlusion_evaluated_count": sum(
            item.context.combined_evidence.static_occlusion_evaluated
            for item in ordered
        ),
        "duplicate_key_count": 0,
        "output_sha256": output_sha256,
    }

    output_path = Path(output_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    summary_temporary = summary_path.with_suffix(
        summary_path.suffix + ".tmp"
    )

    try:
        output_temporary.write_text(
            output_text,
            encoding="utf-8",
        )
        summary_temporary.write_text(
            json.dumps(
                summary,
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
