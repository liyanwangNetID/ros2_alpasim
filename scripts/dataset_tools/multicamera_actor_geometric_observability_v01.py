"""Build four-camera geometric observability for all current Actors.

This Step 7E module projects each current Actor box into every canonical camera
and aggregates the four projections into the existing geometric observability
contract. It does not evaluate Actor-to-Actor or static-scene occlusion and does
not select final Scene-Fact Actor roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from actor_box_image_projection_v01 import (
    ActorCameraProjection,
    project_actor_box_to_camera,
)
from actor_observability_v01 import (
    ActorObservability,
    aggregate_actor_observability,
)
from scene_fact_schema_v01 import CAMERA_NAMES


@dataclass(frozen=True, slots=True)
class MulticameraActorGeometricObservability:
    track_id: str
    actor_class: str
    projections: tuple[ActorCameraProjection, ...]
    observability: ActorObservability


@dataclass(frozen=True, slots=True)
class MulticameraActorGeometricObservabilityResult:
    actor_count: int
    actor_results: tuple[MulticameraActorGeometricObservability, ...]
    actor_observability: tuple[ActorObservability, ...]


def _track_id(actor: Mapping[str, Any]) -> str:
    value = actor.get("track_id")
    if value is None or str(value) == "":
        raise ValueError("Actor is missing a usable track_id")
    return str(value)


def _actor_class(actor: Mapping[str, Any]) -> str:
    value = actor.get("label_class")
    if value is None or str(value) == "":
        raise ValueError("Actor is missing a usable label_class")
    return str(value)


def build_multicamera_actor_geometric_observability(
    *,
    actors: Sequence[Mapping[str, Any]],
    recorded_ego_message: Mapping[str, Any],
    calibrations: Mapping[str, Any],
    samples_per_edge: int | None = None,
    near_plane_m: float = 1e-3,
    maximum_chord_error_px: float = 1.0,
    maximum_adaptive_depth: int = 14,
) -> MulticameraActorGeometricObservabilityResult:
    """Project and aggregate every Actor in deterministic track order."""

    expected_cameras = tuple(CAMERA_NAMES)
    expected_set = set(expected_cameras)
    actual_set = set(calibrations)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(
            f"Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}"
        )

    source = tuple(actors)
    track_ids = tuple(_track_id(actor) for actor in source)
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Actor track_id values must be unique")
    for actor in source:
        _actor_class(actor)

    actor_results = []
    observability_results = []

    for actor in sorted(source, key=_track_id):
        track_id = _track_id(actor)
        actor_class = _actor_class(actor)
        projections_by_camera = {}

        for camera_name in expected_cameras:
            projection = project_actor_box_to_camera(
                actor,
                recorded_ego_message=recorded_ego_message,
                calibration=calibrations[camera_name],
                samples_per_edge=samples_per_edge,
                near_plane_m=near_plane_m,
                maximum_chord_error_px=maximum_chord_error_px,
                maximum_adaptive_depth=maximum_adaptive_depth,
            )
            if projection.camera_name != camera_name:
                raise RuntimeError(
                    f"Projection camera mismatch for track {track_id}: "
                    f"expected={camera_name}, actual={projection.camera_name}"
                )
            if projection.track_id != track_id:
                raise RuntimeError(
                    f"Projection track_id mismatch for track {track_id}"
                )
            if projection.actor_class != actor_class:
                raise RuntimeError(
                    f"Projection actor_class mismatch for track {track_id}"
                )
            projections_by_camera[camera_name] = projection

        aggregated = aggregate_actor_observability(projections_by_camera)
        if aggregated.track_id != track_id:
            raise RuntimeError(
                f"Aggregated track_id mismatch for track {track_id}"
            )
        if aggregated.actor_class != actor_class:
            raise RuntimeError(
                f"Aggregated actor_class mismatch for track {track_id}"
            )

        ordered_projections = tuple(
            projections_by_camera[camera_name]
            for camera_name in expected_cameras
        )
        if tuple(item.camera_name for item in ordered_projections) != expected_cameras:
            raise RuntimeError("Projection order is not canonical")

        actor_results.append(
            MulticameraActorGeometricObservability(
                track_id=track_id,
                actor_class=actor_class,
                projections=ordered_projections,
                observability=aggregated,
            )
        )
        observability_results.append(aggregated)

    return MulticameraActorGeometricObservabilityResult(
        actor_count=len(source),
        actor_results=tuple(actor_results),
        actor_observability=tuple(observability_results),
    )
