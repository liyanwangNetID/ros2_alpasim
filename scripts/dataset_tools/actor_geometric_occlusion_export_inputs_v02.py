"""Prepare one Anchor's Step 7E v02 projection-context export inputs.

This adapter joins the current Actor snapshot with the enhanced projection-
context result. It preserves each Actor's is_static value and produces one
writer input per Actor in deterministic track order. It does not rerun geometry,
apply thresholds, or select final visibility labels.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from actor_geometric_occlusion_export_writer_v02 import (
    ActorGeometricOcclusionProjectionContextExportInput,
)
from actor_geometric_occlusion_with_projection_context_v01 import (
    ActorGeometricOcclusionWithProjectionContextResult,
)


def _track_id(actor: Mapping[str, Any]) -> str:
    value = actor.get("track_id")
    if value is None or str(value) == "":
        raise ValueError("Actor is missing a usable track_id")
    return str(value)


def prepare_actor_geometric_occlusion_projection_context_export_inputs(
    *,
    keyframe: Mapping[str, Any],
    actors: Sequence[Mapping[str, Any]],
    result: ActorGeometricOcclusionWithProjectionContextResult,
) -> tuple[ActorGeometricOcclusionProjectionContextExportInput, ...]:
    """Join one complete Actor snapshot with v02 projection context."""

    for field in ("anchor_id", "clip_id", "anchor_ns"):
        if field not in keyframe:
            raise ValueError(f"keyframe is missing {field}")

    source = tuple(actors)
    actors_by_id = {_track_id(actor): actor for actor in source}
    if len(actors_by_id) != len(source):
        raise ValueError("Actor snapshot contains duplicate track_id values")

    contexts = tuple(result.projection_context.actor_context)
    contexts_by_id = {item.track_id: item for item in contexts}
    if len(contexts_by_id) != len(contexts):
        raise ValueError("projection context contains duplicate track_id values")
    if result.actor_count != len(contexts):
        raise ValueError(
            "result actor_count and projection context count are inconsistent"
        )
    if result.projection_context.actor_count != len(contexts):
        raise ValueError(
            "projection-context actor_count and actor_context are inconsistent"
        )
    if set(actors_by_id) != set(contexts_by_id):
        missing_actor = sorted(set(contexts_by_id) - set(actors_by_id))
        missing_context = sorted(set(actors_by_id) - set(contexts_by_id))
        raise ValueError(
            "Actor snapshot and projection-context sets must match; "
            f"missing_actor={missing_actor}, "
            f"missing_context={missing_context}"
        )

    prepared = []
    for track_id in sorted(actors_by_id):
        actor = actors_by_id[track_id]
        if "is_static" not in actor:
            raise ValueError(f"Actor {track_id} is missing is_static")
        actor_class = actor.get("label_class")
        context = contexts_by_id[track_id]
        if actor_class is None or str(actor_class) != (
            context.combined_evidence.actor_class
        ):
            raise ValueError(
                f"Actor class and projection context differ for track {track_id}"
            )
        prepared.append(
            ActorGeometricOcclusionProjectionContextExportInput(
                keyframe=keyframe,
                context=context,
                is_static=bool(actor["is_static"]),
            )
        )

    return tuple(prepared)
