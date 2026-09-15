"""Per-camera Actor-to-Actor occlusion evidence for Step 7E.

This module represents validated Z-buffer evidence. It does not select
observability labels, aggregate cameras, evaluate static-scene occlusion,
or freeze raster and visibility thresholds.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


OCCLUSION_EVIDENCE_FORMAT_VERSION = "0.1-draft"


@dataclass(frozen=True, slots=True)
class CameraActorOcclusionEvidence:
    occlusion_evidence_format_version: str
    camera_name: str
    track_id: str
    raster_width: int
    raster_height: int
    occupied_cell_count: int
    winning_cell_count: int
    occluded_cell_count: int
    visible_fraction: float | None
    occluding_actor_ids: tuple[str, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool
    evidence_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["occluding_actor_ids"] = list(self.occluding_actor_ids)
        value["reasons"] = list(self.reasons)
        return value


def build_camera_actor_occlusion_evidence(
    *,
    camera_name: str,
    track_id: str,
    raster_width: int,
    raster_height: int,
    occupied_cell_count: int,
    winning_cell_count: int,
    occluded_cell_count: int,
    occluding_actor_ids: Sequence[str],
    actor_to_actor_occlusion_evaluated: bool,
    static_occlusion_evaluated: bool = False,
    reasons: Sequence[str] = (),
) -> CameraActorOcclusionEvidence:
    """Validate counts and construct threshold-free occlusion evidence."""

    if not isinstance(camera_name, str) or not camera_name:
        raise ValueError("camera_name must be a non-empty string")
    if not isinstance(track_id, str) or not track_id:
        raise ValueError("track_id must be a non-empty string")

    dimensions = (raster_width, raster_height)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
        raise TypeError("raster dimensions must be integers")
    if any(value <= 0 for value in dimensions):
        raise ValueError("raster dimensions must be positive")

    counts = (occupied_cell_count, winning_cell_count, occluded_cell_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise TypeError("occlusion counts must be integers")
    if any(value < 0 for value in counts):
        raise ValueError("occlusion counts must be non-negative")
    if winning_cell_count + occluded_cell_count != occupied_cell_count:
        raise ValueError("winning and occluded counts must equal occupied count")

    occluders = tuple(occluding_actor_ids)
    if any(not isinstance(actor_id, str) or not actor_id for actor_id in occluders):
        raise ValueError("occluding Actor IDs must be non-empty strings")
    if track_id in occluders:
        raise ValueError("an Actor cannot occlude itself")
    if len(set(occluders)) != len(occluders):
        raise ValueError("occluding Actor IDs must be unique")
    occluders = tuple(sorted(occluders))

    reason_values = tuple(reasons)
    if any(not isinstance(reason, str) or not reason for reason in reason_values):
        raise ValueError("reasons must be non-empty strings")

    if actor_to_actor_occlusion_evaluated:
        if occupied_cell_count:
            visible_fraction = winning_cell_count / occupied_cell_count
            evidence_status = "evaluated"
        else:
            visible_fraction = None
            evidence_status = "no_sampled_surface"
    else:
        if any(counts):
            raise ValueError("unevaluated evidence cannot contain Z-buffer counts")
        if occluders:
            raise ValueError("unevaluated evidence cannot contain occluding Actor IDs")
        visible_fraction = None
        evidence_status = "not_evaluated"

    if visible_fraction is not None and (
        not math.isfinite(visible_fraction) or not 0.0 <= visible_fraction <= 1.0
    ):
        raise ValueError("visible_fraction must be finite and in [0, 1]")

    return CameraActorOcclusionEvidence(
        occlusion_evidence_format_version=OCCLUSION_EVIDENCE_FORMAT_VERSION,
        camera_name=camera_name,
        track_id=track_id,
        raster_width=raster_width,
        raster_height=raster_height,
        occupied_cell_count=occupied_cell_count,
        winning_cell_count=winning_cell_count,
        occluded_cell_count=occluded_cell_count,
        visible_fraction=visible_fraction,
        occluding_actor_ids=occluders,
        actor_to_actor_occlusion_evaluated=actor_to_actor_occlusion_evaluated,
        static_occlusion_evaluated=static_occlusion_evaluated,
        evidence_status=evidence_status,
        reasons=reason_values,
    )
