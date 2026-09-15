"""Build one shared per-camera Actor Z-buffer and occlusion evidence.

This Step 7E composition module accepts already prepared Actor surface rasters,
resolves their shared depth competition, and adapts the result into validated,
threshold-free per-camera occlusion evidence. It does not build Actor geometry,
select observability labels, aggregate cameras, or evaluate static occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from step7.actor_depth_zbuffer_v01 import (
    ActorDepthRasterInput,
    ActorDepthZBuffer,
    resolve_actor_depth_zbuffer,
)
from step7.actor_occlusion_evidence_adapter_v01 import (
    build_camera_occlusion_evidence_from_zbuffer,
)
from step7.actor_occlusion_evidence_v01 import CameraActorOcclusionEvidence


@dataclass(frozen=True, slots=True)
class CameraActorOcclusionPipelineResult:
    camera_name: str
    raster_width: int
    raster_height: int
    zbuffer: ActorDepthZBuffer
    actor_evidence: tuple[CameraActorOcclusionEvidence, ...]


def build_camera_actor_occlusion_pipeline(
    *,
    camera_name: str,
    raster_width: int,
    raster_height: int,
    actor_rasters: Sequence[ActorDepthRasterInput],
    depth_tolerance_m: float = 1e-9,
    static_occlusion_evaluated: bool = False,
) -> CameraActorOcclusionPipelineResult:
    """Resolve shared depth competition and produce evidence for every Actor."""

    inputs = tuple(actor_rasters)
    zbuffer = resolve_actor_depth_zbuffer(
        inputs,
        depth_tolerance_m=depth_tolerance_m,
    )
    evidence = build_camera_occlusion_evidence_from_zbuffer(
        camera_name=camera_name,
        raster_width=raster_width,
        raster_height=raster_height,
        actor_rasters=inputs,
        zbuffer=zbuffer,
        static_occlusion_evaluated=static_occlusion_evaluated,
    )

    input_ids = tuple(sorted(item.actor_id for item in inputs))
    evidence_ids = tuple(item.track_id for item in evidence)
    summary_ids = tuple(item.actor_id for item in zbuffer.actor_summaries)
    if evidence_ids != input_ids:
        raise RuntimeError("occlusion evidence Actor IDs do not match inputs")
    if summary_ids != input_ids:
        raise RuntimeError("Z-buffer summary Actor IDs do not match inputs")

    return CameraActorOcclusionPipelineResult(
        camera_name=camera_name,
        raster_width=raster_width,
        raster_height=raster_height,
        zbuffer=zbuffer,
        actor_evidence=evidence,
    )
