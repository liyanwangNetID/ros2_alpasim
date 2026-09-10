"""Compose threshold-free Actor occlusion evidence across four cameras.

This Step 7E module runs the current-geometry occlusion pipeline independently
for every canonical camera, then groups per-camera evidence by Actor. It does
not select observability labels, merge visibility fractions across cameras,
use future state, evaluate static-scene occlusion, or freeze thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from actor_camera_occlusion_from_geometry_v01 import (
    ActorCameraOcclusionFromGeometryResult,
    build_actor_camera_occlusion_from_geometry,
)
from actor_multicamera_occlusion_summary_v01 import (
    ActorMulticameraOcclusionSummary,
    summarize_actor_multicamera_occlusion,
)
from actor_occlusion_evidence_v01 import CameraActorOcclusionEvidence
from scene_fact_schema_v01 import CAMERA_NAMES


@dataclass(frozen=True, slots=True)
class CameraOcclusionBuildInput:
    calibration: Any
    image_width_px: int
    image_height_px: int


@dataclass(frozen=True, slots=True)
class ActorMulticameraOcclusionEvidence:
    track_id: str
    camera_evidence: tuple[CameraActorOcclusionEvidence, ...]


@dataclass(frozen=True, slots=True)
class MulticameraActorOcclusionResult:
    actor_count: int
    camera_results: tuple[ActorCameraOcclusionFromGeometryResult, ...]
    actor_evidence: tuple[ActorMulticameraOcclusionEvidence, ...]
    actor_summaries: tuple[ActorMulticameraOcclusionSummary, ...]


def _track_id(actor: dict[str, Any]) -> str:
    value = actor.get("track_id")
    if value is None or str(value) == "":
        raise ValueError("Actor is missing a usable track_id")
    return str(value)


def build_multicamera_actor_occlusion_from_geometry(
    *,
    actors: Sequence[dict[str, Any]],
    recorded_ego_message: dict[str, Any],
    cameras: Mapping[str, CameraOcclusionBuildInput],
    raster_width: int,
    raster_height: int,
    maximum_depth: int,
    maximum_boundary_extent_px: float,
    near_plane_m: float = 1e-3,
    depth_tolerance_m: float = 1e-9,
) -> MulticameraActorOcclusionResult:
    """Build four independent camera Z-buffers and group evidence by Actor."""

    expected_cameras = tuple(CAMERA_NAMES)
    expected_set = set(expected_cameras)
    actual_set = set(cameras)
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
    ordered_track_ids = tuple(sorted(track_ids))

    camera_results = []
    evidence_by_actor: dict[str, list[CameraActorOcclusionEvidence]] = {
        track_id: [] for track_id in ordered_track_ids
    }

    for camera_name in expected_cameras:
        camera = cameras[camera_name]
        result = build_actor_camera_occlusion_from_geometry(
            actors=source,
            recorded_ego_message=recorded_ego_message,
            calibration=camera.calibration,
            camera_name=camera_name,
            image_width_px=camera.image_width_px,
            image_height_px=camera.image_height_px,
            raster_width=raster_width,
            raster_height=raster_height,
            maximum_depth=maximum_depth,
            maximum_boundary_extent_px=maximum_boundary_extent_px,
            near_plane_m=near_plane_m,
            depth_tolerance_m=depth_tolerance_m,
        )
        camera_results.append(result)

        evidence_ids = tuple(
            item.track_id for item in result.occlusion.actor_evidence
        )
        if evidence_ids != ordered_track_ids:
            raise RuntimeError(
                f"Camera {camera_name} evidence Actor IDs differ from inputs"
            )
        for item in result.occlusion.actor_evidence:
            if item.camera_name != camera_name:
                raise RuntimeError(
                    f"Camera evidence mismatch: expected {camera_name}, "
                    f"got {item.camera_name}"
                )
            evidence_by_actor[item.track_id].append(item)

    grouped = tuple(
        ActorMulticameraOcclusionEvidence(
            track_id=track_id,
            camera_evidence=tuple(evidence_by_actor[track_id]),
        )
        for track_id in ordered_track_ids
    )

    for item in grouped:
        camera_order = tuple(
            evidence.camera_name for evidence in item.camera_evidence
        )
        if camera_order != expected_cameras:
            raise RuntimeError(
                f"Actor {item.track_id} camera evidence order is invalid"
            )

    summaries = tuple(
        summarize_actor_multicamera_occlusion(
            track_id=item.track_id,
            camera_evidence=item.camera_evidence,
        )
        for item in grouped
    )
    summary_ids = tuple(item.track_id for item in summaries)
    if summary_ids != ordered_track_ids:
        raise RuntimeError(
            "multicamera occlusion summary Actor IDs differ from inputs"
        )

    return MulticameraActorOcclusionResult(
        actor_count=len(source),
        camera_results=tuple(camera_results),
        actor_evidence=grouped,
        actor_summaries=summaries,
    )
