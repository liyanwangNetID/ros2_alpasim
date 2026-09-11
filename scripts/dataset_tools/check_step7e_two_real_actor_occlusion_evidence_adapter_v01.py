#!/usr/bin/env python3
"""Validate real two-Actor Z-buffer evidence adaptation for Step 7E.

This diagnostic reuses the previously verified real front_tele case with
track 13 behind track 29. It builds both Actor surface rasters, resolves the
shared Z-buffer, adapts the result into threshold-free per-camera occlusion
evidence, and checks consistency with the raw Z-buffer summaries.
"""

from __future__ import annotations

import math

from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_camera_surface_depth_raster_v01 import (
    build_actor_camera_surface_depth_raster,
)
from actor_depth_zbuffer_v01 import (
    ActorDepthRasterInput,
    resolve_actor_depth_zbuffer,
)
from actor_occlusion_evidence_adapter_v01 import (
    build_camera_occlusion_evidence_from_zbuffer,
)
from camera_facing_box_surfaces_v01 import (
    prepare_camera_facing_box_triangles,
)
from camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from profile_step7e_multicase_raster_stability_v01 import (
    actor_id,
    exact_inputs,
    split_anchor_id,
)
from project_paths import ALPASIM_DATA_ROOT


ANCHOR_ID = "test_clip_001_9306612661000"
CAMERA_NAME = "front_tele"
TARGET_TRACK_IDS = ("13", "29")
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
DEPTH_TOLERANCE_M = 1e-9


def build_surface_raster(*, actor, ego, calibration, image_width, image_height):
    corners_rig = actor_box_corners_in_rig(
        actor,
        recorded_ego_message=ego.message,
    )
    corners_camera = tuple(
        calibration.rig_point_to_camera(point)
        for point in corners_rig
    )
    prepared = prepare_camera_facing_box_triangles(
        corners_camera,
        near_plane_m=NEAR_PLANE_M,
    )
    result = build_actor_camera_surface_depth_raster(
        tuple(item.vertices_camera for item in prepared),
        calibration,
        max_angle_rad=calibration.max_angle_rad,
        maximum_depth=MAXIMUM_DEPTH,
        maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
        image_width_px=image_width,
        image_height_px=image_height,
        raster_width=RASTER_WIDTH,
        raster_height=RASTER_HEIGHT,
        near_plane_m=NEAR_PLANE_M,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )
    if result.unresolved_boundary_count:
        raise RuntimeError(
            f"Unresolved boundary for track {actor_id(actor)}: "
            f"{result.unresolved_boundary_count}"
        )
    if result.surface_raster.occupied_cell_count <= 0:
        raise RuntimeError(
            f"No occupied surface cells for track {actor_id(actor)}"
        )
    return result.surface_raster


def main() -> int:
    clip_id, anchor_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, anchor_ns)

    actors_by_id = {actor_id(actor): actor for actor in actors}
    missing = sorted(set(TARGET_TRACK_IDS) - set(actors_by_id))
    if missing:
        raise RuntimeError(f"Missing target Actors: {missing}")

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
    if calibration.max_angle_rad is None:
        raise RuntimeError("Camera calibration has no max_angle_rad")

    inputs = tuple(
        ActorDepthRasterInput(
            track_id,
            build_surface_raster(
                actor=actors_by_id[track_id],
                ego=ego,
                calibration=calibration,
                image_width=frame.value.width,
                image_height=frame.value.height,
            ),
        )
        for track_id in TARGET_TRACK_IDS
    )

    zbuffer = resolve_actor_depth_zbuffer(
        inputs,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )
    evidence = build_camera_occlusion_evidence_from_zbuffer(
        camera_name=CAMERA_NAME,
        raster_width=RASTER_WIDTH,
        raster_height=RASTER_HEIGHT,
        actor_rasters=inputs,
        zbuffer=zbuffer,
        static_occlusion_evaluated=False,
    )

    summaries = {item.actor_id: item for item in zbuffer.actor_summaries}
    evidence_by_id = {item.track_id: item for item in evidence}

    if tuple(item.track_id for item in evidence) != tuple(sorted(TARGET_TRACK_IDS)):
        raise RuntimeError("Evidence output order is not deterministic")
    if set(evidence_by_id) != set(TARGET_TRACK_IDS):
        raise RuntimeError("Evidence Actor IDs do not match target Actor IDs")

    for track_id in TARGET_TRACK_IDS:
        summary = summaries[track_id]
        item = evidence_by_id[track_id]
        if item.occupied_cell_count != summary.occupied_cell_count:
            raise RuntimeError(f"Occupied count mismatch for track {track_id}")
        if item.winning_cell_count != summary.winning_cell_count:
            raise RuntimeError(f"Winning count mismatch for track {track_id}")
        if item.occluded_cell_count != summary.occluded_cell_count:
            raise RuntimeError(f"Occluded count mismatch for track {track_id}")
        expected_fraction = summary.winning_cell_count / summary.occupied_cell_count
        if not math.isclose(
            item.visible_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"Visible fraction mismatch for track {track_id}")
        if item.camera_name != CAMERA_NAME:
            raise RuntimeError(f"Camera mismatch for track {track_id}")
        if item.raster_width != RASTER_WIDTH or item.raster_height != RASTER_HEIGHT:
            raise RuntimeError(f"Raster dimensions mismatch for track {track_id}")
        if not item.actor_to_actor_occlusion_evaluated:
            raise RuntimeError(f"Occlusion not marked evaluated for track {track_id}")
        if item.static_occlusion_evaluated:
            raise RuntimeError(f"Static occlusion incorrectly marked evaluated for track {track_id}")

    if evidence_by_id["13"].occluding_actor_ids != ("29",):
        raise RuntimeError(
            "Track 13 should be occluded only by track 29 in the two-Actor case"
        )
    if evidence_by_id["29"].occluding_actor_ids:
        raise RuntimeError("Track 29 should have no occluding Actor in this case")

    print("Real two-Actor occlusion-evidence adapter check")
    print("anchor:", ANCHOR_ID)
    print("camera:", CAMERA_NAME)
    print("raster:", f"{RASTER_WIDTH}x{RASTER_HEIGHT}")
    print("occupied_union_cell_count:", zbuffer.occupied_union_cell_count)
    print("contested_cell_count:", zbuffer.contested_cell_count)
    for item in evidence:
        print(
            f"track={item.track_id} "
            f"status={item.evidence_status} "
            f"occupied={item.occupied_cell_count} "
            f"winning={item.winning_cell_count} "
            f"occluded={item.occluded_cell_count} "
            f"visible_fraction={item.visible_fraction} "
            f"occluders={list(item.occluding_actor_ids)}"
        )

    print(
        "PASS: real two-Actor Z-buffer adapted into consistent "
        "threshold-free occlusion evidence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
