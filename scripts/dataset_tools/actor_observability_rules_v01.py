"""Pure geometric observability rules for Actor/camera projections.

The rules classify only geometric observability candidates. They do not model
Actor-to-Actor occlusion, image blur, exposure, or appearance quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT: Final = (
    "below_minimum_projected_height"
)
FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT: Final = (
    "below_primary_area_and_height"
)
FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO: Final = (
    "below_minimum_inside_image_hull_ratio"
)


@dataclass(frozen=True)
class GeometricObservabilityDecision:
    candidate: bool
    failure_reason: str | None


@dataclass(frozen=True)
class HeightPolicy:
    minimum_projected_height_px: float


@dataclass(frozen=True)
class TelePolicy:
    minimum_inside_image_hull_area_px: float
    minimum_projected_height_px: float
    minimum_inside_image_hull_ratio: float


HEIGHT_POLICIES: Final = {
    "front_wide": HeightPolicy(minimum_projected_height_px=5.0),
    "cross_left": HeightPolicy(minimum_projected_height_px=6.0),
    "cross_right": HeightPolicy(minimum_projected_height_px=6.0),
}

FRONT_TELE_POLICY: Final = TelePolicy(
    minimum_inside_image_hull_area_px=512.0,
    minimum_projected_height_px=16.0,
    minimum_inside_image_hull_ratio=0.10,
)

CAMERA_NAMES: Final = frozenset((*HEIGHT_POLICIES, "front_tele"))


def evaluate_geometric_observability(
    *,
    camera_name: str,
    inside_image_hull_area_px: float,
    projected_height_px: float,
    inside_image_hull_ratio: float,
) -> GeometricObservabilityDecision:
    """Evaluate one valid geometric projection.

    Inputs are expected to come from a valid ActorCameraProjection. Thresholds
    are inclusive, so a value exactly on a boundary passes that boundary.
    """
    if camera_name not in CAMERA_NAMES:
        raise ValueError(f"Unsupported camera_name: {camera_name!r}")
    if inside_image_hull_area_px < 0.0:
        raise ValueError("inside_image_hull_area_px must be non-negative")
    if projected_height_px < 0.0:
        raise ValueError("projected_height_px must be non-negative")
    if not 0.0 <= inside_image_hull_ratio <= 1.0:
        raise ValueError("inside_image_hull_ratio must be in [0, 1]")

    height_policy = HEIGHT_POLICIES.get(camera_name)
    if height_policy is not None:
        if projected_height_px < height_policy.minimum_projected_height_px:
            return GeometricObservabilityDecision(
                candidate=False,
                failure_reason=FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT,
            )
        return GeometricObservabilityDecision(
            candidate=True,
            failure_reason=None,
        )

    policy = FRONT_TELE_POLICY
    scale_pass = (
        inside_image_hull_area_px
        >= policy.minimum_inside_image_hull_area_px
        or projected_height_px >= policy.minimum_projected_height_px
    )
    if not scale_pass:
        return GeometricObservabilityDecision(
            candidate=False,
            failure_reason=FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT,
        )
    if inside_image_hull_ratio < policy.minimum_inside_image_hull_ratio:
        return GeometricObservabilityDecision(
            candidate=False,
            failure_reason=FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO,
        )
    return GeometricObservabilityDecision(candidate=True, failure_reason=None)
