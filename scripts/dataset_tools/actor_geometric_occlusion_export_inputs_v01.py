"""Prepare one Anchor's combined Actor evidence for JSONL export.

This adapter joins a complete current Actor snapshot with the already computed
ActorGeometricOcclusionPipelineResult. It preserves Actor is_static values and
produces deterministic writer inputs without rerunning geometry or selecting
final visibility labels.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from actor_geometric_occlusion_export_writer_v01 import (
    ActorGeometricOcclusionExportInput,
)
from actor_geometric_occlusion_pipeline_v01 import (
    ActorGeometricOcclusionPipelineResult,
)


def _actor_track_id(actor: Mapping[str, Any]) -> str:
    value = actor.get("track_id")
    if value is None or str(value) == "":
        raise ValueError("Actor is missing a usable track_id")
    return str(value)


def prepare_actor_geometric_occlusion_export_inputs(
    *,
    keyframe: Mapping[str, Any],
    actors: Sequence[Mapping[str, Any]],
    pipeline_result: ActorGeometricOcclusionPipelineResult,
) -> tuple[ActorGeometricOcclusionExportInput, ...]:
    """Join current Actors and combined evidence in deterministic track order."""

    for field in ("anchor_id", "clip_id", "anchor_ns"):
        if field not in keyframe:
            raise ValueError(f"keyframe is missing {field}")

    source = tuple(actors)
    actors_by_id = {_actor_track_id(actor): actor for actor in source}
    if len(actors_by_id) != len(source):
        raise ValueError("Actor snapshot contains duplicate track_id values")

    evidence = tuple(pipeline_result.combined_evidence.actor_evidence)
    evidence_by_id = {item.track_id: item for item in evidence}
    if len(evidence_by_id) != len(evidence):
        raise ValueError("combined evidence contains duplicate track_id values")
    if pipeline_result.actor_count != len(evidence):
        raise ValueError(
            "pipeline actor_count and combined evidence count are inconsistent"
        )
    if set(actors_by_id) != set(evidence_by_id):
        missing_actor = sorted(set(evidence_by_id) - set(actors_by_id))
        missing_evidence = sorted(set(actors_by_id) - set(evidence_by_id))
        raise ValueError(
            "Actor snapshot and combined evidence sets must match; "
            f"missing_actor={missing_actor}, "
            f"missing_evidence={missing_evidence}"
        )

    prepared = []
    for track_id in sorted(actors_by_id):
        actor = actors_by_id[track_id]
        if "is_static" not in actor:
            raise ValueError(f"Actor {track_id} is missing is_static")
        evidence_item = evidence_by_id[track_id]
        actor_class = actor.get("label_class")
        if actor_class is None or str(actor_class) != evidence_item.actor_class:
            raise ValueError(
                f"Actor class and combined evidence differ for track {track_id}"
            )
        prepared.append(
            ActorGeometricOcclusionExportInput(
                keyframe=keyframe,
                evidence=evidence_item,
                is_static=bool(actor["is_static"]),
            )
        )

    return tuple(prepared)
