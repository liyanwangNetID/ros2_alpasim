"""Serialize Step 7E combined evidence with projection context.

Version 02 extends the stable v01 per-Actor export row with explicit projection
context for geometric candidate cameras that lack sampled surface evidence. It
does not change v01 files in place, apply thresholds, or assign final visibility
labels.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from step7.actor_geometric_occlusion_export_v01 import (
    actor_geometric_occlusion_export_record,
)
from step7.actor_geometric_occlusion_projection_context_v01 import (
    ActorGeometricOcclusionProjectionContext,
)


SCHEMA_VERSION = "step7e-geometric-occlusion-evidence-v02"


def actor_geometric_occlusion_projection_context_export_record(
    *,
    keyframe: Mapping[str, Any],
    context: ActorGeometricOcclusionProjectionContext,
    is_static: bool,
) -> dict[str, Any]:
    """Return one JSON-ready v02 Actor row with projection context."""

    base = actor_geometric_occlusion_export_record(
        keyframe=keyframe,
        evidence=context.combined_evidence,
        is_static=is_static,
    )
    if context.track_id != context.combined_evidence.track_id:
        raise ValueError(
            "projection context and combined evidence track_id values differ"
        )

    projection_context = tuple(
        context.candidate_without_sampled_surface_projection_evidence
    )
    camera_names = tuple(item.camera_name for item in projection_context)
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("projection context contains duplicate camera names")

    expected_cameras = tuple(
        context.combined_evidence
        .geometric_candidate_without_sampled_surface_camera_names
    )
    if camera_names != expected_cameras:
        raise ValueError(
            "projection-context cameras differ from missing-surface cameras"
        )

    base["schema_version"] = SCHEMA_VERSION
    base[
        "candidate_without_sampled_surface_projection_evidence"
    ] = [item.to_dict() for item in projection_context]
    return base


def encode_actor_geometric_occlusion_projection_context_export_record(
    *,
    keyframe: Mapping[str, Any],
    context: ActorGeometricOcclusionProjectionContext,
    is_static: bool,
) -> str:
    """Encode one compact v02 JSONL row."""

    return json.dumps(
        actor_geometric_occlusion_projection_context_export_record(
            keyframe=keyframe,
            context=context,
            is_static=is_static,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
