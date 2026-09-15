"""Describe projection context for candidate cameras without sampled surface.

This Step 7E module joins one combined Actor evidence record with its canonical
per-camera ActorCameraProjection values. It produces explicit, threshold-free
records for geometric candidate cameras that lack sampled surface evidence.
Projection truncation is reported as context, not asserted as the cause of the
surface absence and not converted into a final visibility label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from step7.actor_box_image_projection_v01 import ActorCameraProjection
from step7.actor_geometric_occlusion_evidence_v01 import (
    ActorGeometricOcclusionEvidence,
)
from step7.scene_fact_schema_v01 import CAMERA_NAMES


@dataclass(frozen=True, slots=True)
class CandidateWithoutSampledSurfaceProjectionEvidence:
    camera_name: str
    track_id: str
    actor_class: str
    projection_valid: bool
    projection_truncated: bool
    inside_image_hull_area_px: float
    projected_height_px: float
    inside_image_hull_ratio: float
    minimum_depth_m: float | None
    maximum_depth_m: float | None
    evidence_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def build_candidate_without_sampled_surface_projection_evidence(
    *,
    combined_evidence: ActorGeometricOcclusionEvidence,
    projections_by_camera: Mapping[str, ActorCameraProjection],
) -> tuple[CandidateWithoutSampledSurfaceProjectionEvidence, ...]:
    """Build one projection-context record per missing-surface candidate camera."""

    expected = set(CAMERA_NAMES)
    actual = set(projections_by_camera)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}"
        )

    missing_surface_cameras = tuple(
        combined_evidence
        .geometric_candidate_without_sampled_surface_camera_names
    )
    candidate_set = set(combined_evidence.geometric_candidate_camera_names)
    if any(name not in candidate_set for name in missing_surface_cameras):
        raise ValueError(
            "missing-surface cameras must be geometric candidate cameras"
        )

    records = []
    for camera_name in CAMERA_NAMES:
        if camera_name not in missing_surface_cameras:
            continue
        projection = projections_by_camera[camera_name]
        if projection.camera_name != camera_name:
            raise ValueError("projection camera_name differs from mapping key")
        if projection.track_id != combined_evidence.track_id:
            raise ValueError("projection and combined evidence track_id differ")
        if projection.actor_class != combined_evidence.actor_class:
            raise ValueError("projection and combined evidence actor_class differ")

        reasons = ["geometric_candidate_without_sampled_surface"]
        if not projection.projection_valid:
            status = "missing_surface_with_invalid_projection"
            reasons.append("projection_invalid")
        elif projection.truncated:
            status = "missing_surface_with_truncated_projection"
            reasons.append("projection_truncated_by_image")
        else:
            status = "missing_surface_with_untruncated_projection"

        records.append(
            CandidateWithoutSampledSurfaceProjectionEvidence(
                camera_name=camera_name,
                track_id=combined_evidence.track_id,
                actor_class=combined_evidence.actor_class,
                projection_valid=projection.projection_valid,
                projection_truncated=projection.truncated,
                inside_image_hull_area_px=float(
                    projection.inside_image_hull_area_px
                ),
                projected_height_px=float(projection.projected_height_px),
                inside_image_hull_ratio=float(
                    projection.inside_image_hull_ratio
                ),
                minimum_depth_m=projection.minimum_depth_m,
                maximum_depth_m=projection.maximum_depth_m,
                evidence_status=status,
                reasons=tuple(reasons),
            )
        )

    return tuple(records)
