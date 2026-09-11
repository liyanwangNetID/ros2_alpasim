"""Serialize Step 7E combined geometric-occlusion evidence to JSONL rows.

The adapter converts one threshold-free ActorGeometricOcclusionEvidence value
into a stable per-Actor export record. It preserves geometric and Actor-to-Actor
occlusion evidence without selecting a final visibility label or evaluating
static-scene occlusion.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)


SCHEMA_VERSION = "step7e-geometric-occlusion-evidence-v01"


def actor_geometric_occlusion_export_record(
    *,
    keyframe: Mapping[str, Any],
    evidence: ActorGeometricOcclusionEvidence,
    is_static: bool,
) -> dict[str, Any]:
    """Return one stable JSON-ready row for an Anchor/Actor pair."""

    for field in ("anchor_id", "clip_id", "anchor_ns"):
        if field not in keyframe:
            raise ValueError(f"keyframe is missing {field}")
    if not evidence.track_id:
        raise ValueError("evidence track_id must be non-empty")
    if not evidence.actor_class:
        raise ValueError("evidence actor_class must be non-empty")
    if evidence.static_occlusion_evaluated:
        raise ValueError(
            "static occlusion must remain unevaluated in Step 7E v0.1"
        )

    maximum_visible_fraction = evidence.maximum_visible_fraction
    if maximum_visible_fraction is not None:
        value = float(maximum_visible_fraction)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "maximum_visible_fraction must be finite and within [0, 1]"
            )
        maximum_visible_fraction = value

    counts = {
        "total_occupied_cell_count": evidence.total_occupied_cell_count,
        "total_winning_cell_count": evidence.total_winning_cell_count,
        "total_occluded_cell_count": evidence.total_occluded_cell_count,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValueError("combined cell counts must be non-negative integers")
    if (
        evidence.total_winning_cell_count
        + evidence.total_occluded_cell_count
        != evidence.total_occupied_cell_count
    ):
        raise ValueError("winning and occluded cell counts must close")

    return {
        "schema_version": SCHEMA_VERSION,
        "anchor_id": str(keyframe["anchor_id"]),
        "clip_id": str(keyframe["clip_id"]),
        "anchor_ns": int(keyframe["anchor_ns"]),
        "track_id": evidence.track_id,
        "label_class": evidence.actor_class,
        "is_static": bool(is_static),
        "geometric_observability_status": (
            evidence.geometric_observability_status
        ),
        "geometric_candidate_camera_names": list(
            evidence.geometric_candidate_camera_names
        ),
        "occlusion_evaluated_camera_names": list(
            evidence.occlusion_evaluated_camera_names
        ),
        "occlusion_winning_camera_names": list(
            evidence.occlusion_winning_camera_names
        ),
        "geometric_candidate_with_sampled_surface_camera_names": list(
            evidence.geometric_candidate_with_sampled_surface_camera_names
        ),
        "geometric_candidate_with_winning_cells_camera_names": list(
            evidence.geometric_candidate_with_winning_cells_camera_names
        ),
        "geometric_candidate_without_sampled_surface_camera_names": list(
            evidence.geometric_candidate_without_sampled_surface_camera_names
        ),
        "geometric_candidate_fully_occluded_camera_names": list(
            evidence.geometric_candidate_fully_occluded_camera_names
        ),
        "maximum_visible_fraction": maximum_visible_fraction,
        "total_occupied_cell_count": evidence.total_occupied_cell_count,
        "total_winning_cell_count": evidence.total_winning_cell_count,
        "total_occluded_cell_count": evidence.total_occluded_cell_count,
        "occluding_actor_ids": list(evidence.occluding_actor_ids),
        "actor_to_actor_occlusion_evaluated": (
            evidence.actor_to_actor_occlusion_evaluated
        ),
        "static_occlusion_evaluated": evidence.static_occlusion_evaluated,
        "evidence_status": evidence.evidence_status,
        "reasons": list(evidence.reasons),
    }


def encode_actor_geometric_occlusion_export_record(
    *,
    keyframe: Mapping[str, Any],
    evidence: ActorGeometricOcclusionEvidence,
    is_static: bool,
) -> str:
    """Encode one export row using the project's compact JSONL convention."""

    return json.dumps(
        actor_geometric_occlusion_export_record(
            keyframe=keyframe,
            evidence=evidence,
            is_static=is_static,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
