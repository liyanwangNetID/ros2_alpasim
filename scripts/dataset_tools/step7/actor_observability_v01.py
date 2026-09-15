"""Actor-level geometric observability aggregation for Step 7E.

This module aggregates four per-camera ActorCameraProjection values. It does
not evaluate Actor-to-Actor or static-scene occlusion and does not construct a
final Scene-Fact Actor role.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from step7.actor_box_image_projection_v01 import ActorCameraProjection
from step7.actor_observability_rules_v01 import evaluate_geometric_observability
from step7.scene_fact_schema_v01 import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class CameraObservability:
    camera_name: str
    projection_valid: bool
    geometric_observability_candidate: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActorObservability:
    observability_format_version: str
    track_id: str
    actor_class: str
    observability_status: str
    visible_in_cameras: tuple[str, ...]
    camera_observability: tuple[CameraObservability, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["visible_in_cameras"] = list(self.visible_in_cameras)
        value["camera_observability"] = [
            item.to_dict() for item in self.camera_observability
        ]
        return value


def aggregate_actor_observability(
    projections_by_camera: Mapping[str, ActorCameraProjection],
) -> ActorObservability:
    """Aggregate exactly four same-Actor projections in canonical camera order."""
    expected = set(CAMERA_NAMES)
    actual = set(projections_by_camera)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}"
        )

    projections = [projections_by_camera[name] for name in CAMERA_NAMES]
    track_ids = {projection.track_id for projection in projections}
    actor_classes = {projection.actor_class for projection in projections}
    if len(track_ids) != 1:
        raise ValueError("All projections must have the same track_id")
    if len(actor_classes) != 1:
        raise ValueError("All projections must have the same actor_class")

    camera_results: list[CameraObservability] = []
    visible_in_cameras: list[str] = []

    for expected_camera, projection in zip(CAMERA_NAMES, projections):
        if projection.camera_name != expected_camera:
            raise ValueError(
                f"Projection camera mismatch: key={expected_camera!r}, "
                f"value={projection.camera_name!r}"
            )

        if not projection.projection_valid:
            if not projection.failure_reason:
                raise ValueError(
                    "Invalid projection must provide a failure_reason"
                )
            camera_results.append(
                CameraObservability(
                    camera_name=expected_camera,
                    projection_valid=False,
                    geometric_observability_candidate=False,
                    failure_reason=projection.failure_reason,
                )
            )
            continue

        if projection.failure_reason is not None:
            raise ValueError(
                "Valid projection must not provide a failure_reason"
            )

        decision = evaluate_geometric_observability(
            camera_name=expected_camera,
            inside_image_hull_area_px=projection.inside_image_hull_area_px,
            projected_height_px=projection.projected_height_px,
            inside_image_hull_ratio=projection.inside_image_hull_ratio,
        )
        if decision.candidate:
            visible_in_cameras.append(expected_camera)
        camera_results.append(
            CameraObservability(
                camera_name=expected_camera,
                projection_valid=True,
                geometric_observability_candidate=decision.candidate,
                failure_reason=decision.failure_reason,
            )
        )

    status = "candidate_visible" if visible_in_cameras else "not_visible"
    return ActorObservability(
        observability_format_version=OBSERVABILITY_FORMAT_VERSION,
        track_id=next(iter(track_ids)),
        actor_class=next(iter(actor_classes)),
        observability_status=status,
        visible_in_cameras=tuple(visible_in_cameras),
        camera_observability=tuple(camera_results),
        actor_to_actor_occlusion_evaluated=False,
        static_occlusion_evaluated=False,
    )
