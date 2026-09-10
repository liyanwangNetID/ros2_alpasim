#!/usr/bin/env python3
"""Validate the composed real Actor-geometry occlusion pipeline for Step 7E.

This diagnostic runs the reusable geometry-to-occlusion module on the fixed
front_tele Keyframe previously used for all-Actor Z-buffer validation. It
checks the composed result against the established real-case counts without
selecting observability labels or evaluating static-scene occlusion.
"""

from __future__ import annotations

from actor_camera_occlusion_from_geometry_v01 import (
    build_actor_camera_occlusion_from_geometry,
)
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from profile_step7e_multicase_raster_stability_v01 import (
    exact_inputs,
    split_anchor_id,
)
from project_paths import ALPASIM_DATA_ROOT


ANCHOR_ID = "test_clip_001_9306612661000"
CAMERA_NAME = "front_tele"
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
DEPTH_TOLERANCE_M = 1e-9

EXPECTED = {
    "input_actor_count": 37,
    "evaluated_actor_count": 16,
    "no_sampled_surface_actor_count": 21,
    "fully_visible_actor_count": 3,
    "partially_occluded_actor_count": 6,
    "fully_occluded_actor_count": 7,
    "occupied_union_cell_count": 14796,
    "contested_cell_count": 4865,
}


def main() -> int:
    clip_id, anchor_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, anchor_ns)

    frame = reader.camera_indexes[CAMERA_NAME].exact(anchor_ns)
    if frame is None:
        raise RuntimeError(
            f"Exact camera frame unavailable: {ANCHOR_ID} {CAMERA_NAME}"
        )

    calibration = load_camera_calibration(
        reader.clip_directory / "calibration" / f"{CAMERA_NAME}.json",
        camera_name=CAMERA_NAME,
        source_width=frame.value.width,
        source_height=frame.value.height,
    )

    result = build_actor_camera_occlusion_from_geometry(
        actors=actors,
        recorded_ego_message=ego.message,
        calibration=calibration,
        camera_name=CAMERA_NAME,
        image_width_px=frame.value.width,
        image_height_px=frame.value.height,
        raster_width=RASTER_WIDTH,
        raster_height=RASTER_HEIGHT,
        maximum_depth=MAXIMUM_DEPTH,
        maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
        near_plane_m=NEAR_PLANE_M,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )

    evidence = result.occlusion.actor_evidence
    evaluated = [item for item in evidence if item.evidence_status == "evaluated"]
    no_surface = [
        item for item in evidence if item.evidence_status == "no_sampled_surface"
    ]
    fully_visible = [item for item in evaluated if item.occluded_cell_count == 0]
    fully_occluded = [item for item in evaluated if item.winning_cell_count == 0]
    partially_occluded = [
        item
        for item in evaluated
        if item.winning_cell_count > 0 and item.occluded_cell_count > 0
    ]

    actual = {
        "input_actor_count": result.actor_count,
        "evaluated_actor_count": len(evaluated),
        "no_sampled_surface_actor_count": len(no_surface),
        "fully_visible_actor_count": len(fully_visible),
        "partially_occluded_actor_count": len(partially_occluded),
        "fully_occluded_actor_count": len(fully_occluded),
        "occupied_union_cell_count": (
            result.occlusion.zbuffer.occupied_union_cell_count
        ),
        "contested_cell_count": result.occlusion.zbuffer.contested_cell_count,
    }

    if actual != EXPECTED:
        raise RuntimeError(
            f"Composed pipeline changed the real baseline: "
            f"actual={actual}, expected={EXPECTED}"
        )

    if len(result.surface_diagnostics) != result.actor_count:
        raise RuntimeError("Surface diagnostic count does not match Actor count")
    if len(evidence) != result.actor_count:
        raise RuntimeError("Occlusion evidence count does not match Actor count")

    diagnostic_ids = tuple(
        item.track_id for item in result.surface_diagnostics
    )
    evidence_ids = tuple(item.track_id for item in evidence)
    if diagnostic_ids != evidence_ids:
        raise RuntimeError("Surface diagnostics and evidence IDs differ")

    unresolved = sum(
        item.unresolved_boundary_count
        for item in result.surface_diagnostics
    )
    if unresolved != 0:
        raise RuntimeError(f"Unexpected unresolved boundary count: {unresolved}")

    track_13 = next(item for item in evidence if item.track_id == "13")
    track_29 = next(item for item in evidence if item.track_id == "29")
    if track_13.occluding_actor_ids != ("29",):
        raise RuntimeError("Track 13 occluder baseline changed")
    if track_29.occluding_actor_ids:
        raise RuntimeError("Track 29 should have no occluder in this case")

    print("Real composed Actor-geometry occlusion pipeline check")
    print("anchor:", ANCHOR_ID)
    print("camera:", CAMERA_NAME)
    print("raster:", f"{RASTER_WIDTH}x{RASTER_HEIGHT}")
    for key, value in actual.items():
        print(f"{key}:", value)
    print("unresolved_boundary_count:", unresolved)
    print(
        "track=13",
        f"occupied={track_13.occupied_cell_count}",
        f"winning={track_13.winning_cell_count}",
        f"occluded={track_13.occluded_cell_count}",
        f"visible_fraction={track_13.visible_fraction}",
        f"occluders={list(track_13.occluding_actor_ids)}",
    )
    print(
        "track=29",
        f"occupied={track_29.occupied_cell_count}",
        f"winning={track_29.winning_cell_count}",
        f"occluded={track_29.occluded_cell_count}",
        f"visible_fraction={track_29.visible_fraction}",
        f"occluders={list(track_29.occluding_actor_ids)}",
    )
    print(
        "PASS: reusable Actor-geometry pipeline reproduces the verified "
        "all-Actor real occlusion baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
