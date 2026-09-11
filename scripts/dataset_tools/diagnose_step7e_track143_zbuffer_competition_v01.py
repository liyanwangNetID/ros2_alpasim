#!/usr/bin/env python3
"""Diagnose the real cross_right track 143 Z-buffer boundary change.

Builds all current Actor surface rasters at 480x270 and 640x360, resolves each
shared Z-buffer, and reports the target Actor's winning samples, occluders,
depth gaps, and pixel-space competition ranges. No visibility threshold or
final observability label is applied.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_camera_surface_depth_raster_v01 import (
    build_actor_camera_surface_depth_raster,
)
from actor_depth_zbuffer_v01 import (
    ActorDepthRasterInput,
    resolve_actor_depth_zbuffer,
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


ANCHOR_ID = "test_clip_316_1404486225261000"
CAMERA_NAME = "cross_right"
TARGET_ACTOR_ID = "143"
RASTERS = ((480, 270), (640, 360))
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
DEPTH_TOLERANCE_M = 1e-9


def build_inputs(
    *,
    actors,
    ego,
    calibration,
    image_width,
    image_height,
    raster_width,
    raster_height,
):
    inputs = []
    surfaces_by_actor = {}

    for actor in sorted(actors, key=actor_id):
        track_id = actor_id(actor)
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
            raster_width=raster_width,
            raster_height=raster_height,
            near_plane_m=NEAR_PLANE_M,
            depth_tolerance_m=DEPTH_TOLERANCE_M,
        )
        if result.unresolved_boundary_count:
            raise RuntimeError(
                f"Unresolved boundary for Actor {track_id}: "
                f"{result.unresolved_boundary_count}"
            )
        surface = result.surface_raster
        surfaces_by_actor[track_id] = surface
        inputs.append(ActorDepthRasterInput(track_id, surface))

    return tuple(inputs), surfaces_by_actor


def pixel_bbox(samples):
    return (
        min(item.sample_u_px for item in samples),
        min(item.sample_v_px for item in samples),
        max(item.sample_u_px for item in samples),
        max(item.sample_v_px for item in samples),
    )


def diagnose_resolution(
    *,
    actors,
    ego,
    calibration,
    image_width,
    image_height,
    raster_width,
    raster_height,
):
    inputs, surfaces_by_actor = build_inputs(
        actors=actors,
        ego=ego,
        calibration=calibration,
        image_width=image_width,
        image_height=image_height,
        raster_width=raster_width,
        raster_height=raster_height,
    )
    target_surface = surfaces_by_actor.get(TARGET_ACTOR_ID)
    if target_surface is None or not target_surface.cell_depths:
        raise RuntimeError("Target Actor has no sampled surface")

    zbuffer = resolve_actor_depth_zbuffer(
        inputs,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )
    winner_by_cell = {
        winner.cell: winner for winner in zbuffer.cell_winners
    }

    winning_samples = []
    occluded_samples = []
    occluder_counts = Counter()
    occluder_depth_gaps = defaultdict(list)
    occluder_target_samples = defaultdict(list)
    tied_loss_count = 0

    for target_sample in target_surface.cell_depths:
        winner = winner_by_cell.get(target_sample.cell)
        if winner is None:
            raise RuntimeError(
                f"Missing winner for target cell {target_sample.cell}"
            )
        if winner.actor_id == TARGET_ACTOR_ID:
            winning_samples.append(target_sample)
            continue

        occluded_samples.append(target_sample)
        occluder_counts[winner.actor_id] += 1
        occluder_depth_gaps[winner.actor_id].append(
            target_sample.depth_m - winner.depth_m
        )
        occluder_target_samples[winner.actor_id].append(target_sample)
        if math.isclose(
            target_sample.depth_m,
            winner.depth_m,
            rel_tol=0.0,
            abs_tol=DEPTH_TOLERANCE_M,
        ):
            tied_loss_count += 1

    target_cells = target_surface.cell_depths
    target_cell_bbox = (
        min(item.cell.column for item in target_cells),
        min(item.cell.row for item in target_cells),
        max(item.cell.column for item in target_cells),
        max(item.cell.row for item in target_cells),
    )

    print()
    print("=" * 100)
    print(f"{raster_width}x{raster_height}")
    print("=" * 100)
    print("target_occupied_cell_count:", len(target_cells))
    print("target_winning_cell_count:", len(winning_samples))
    print("target_occluded_cell_count:", len(occluded_samples))
    print(
        "target_visible_fraction:",
        len(winning_samples) / len(target_cells),
    )
    print("target_cell_bbox:", target_cell_bbox)
    print("target_sample_pixel_bbox:", pixel_bbox(target_cells))
    print("tie_break_loss_count:", tied_loss_count)
    print("occluders:")

    for occluder_id, count in sorted(
        occluder_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        gaps = occluder_depth_gaps[occluder_id]
        samples = occluder_target_samples[occluder_id]
        print(
            f"  actor={occluder_id} "
            f"cells={count} "
            f"target_minus_winner_depth_m="
            f"({min(gaps):.6f}, {max(gaps):.6f}) "
            f"target_pixel_bbox={tuple(round(v, 3) for v in pixel_bbox(samples))}"
        )

    if winning_samples:
        print(
            "target_winning_pixel_bbox:",
            pixel_bbox(winning_samples),
        )
        print("target winning samples:")
        for sample in winning_samples[:20]:
            print(
                f"  cell=({sample.cell.column},{sample.cell.row}) "
                f"pixel=({sample.sample_u_px:.6f},"
                f"{sample.sample_v_px:.6f}) "
                f"depth={sample.depth_m:.6f}"
            )

    return {
        "occupied": len(target_cells),
        "winning": len(winning_samples),
        "occluded": len(occluded_samples),
        "tie_break_losses": tied_loss_count,
        "occluders": tuple(sorted(occluder_counts)),
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
    if calibration.max_angle_rad is None:
        raise RuntimeError("Camera calibration has no max_angle_rad")

    results = {}
    for raster_width, raster_height in RASTERS:
        results[(raster_width, raster_height)] = diagnose_resolution(
            actors=actors,
            ego=ego,
            calibration=calibration,
            image_width=frame.value.width,
            image_height=frame.value.height,
            raster_width=raster_width,
            raster_height=raster_height,
        )

    candidate = results[(480, 270)]
    reference = results[(640, 360)]
    if candidate["winning"] != 0:
        raise RuntimeError("Candidate winning-count baseline changed")
    if reference["winning"] != 1:
        raise RuntimeError("Reference winning-count baseline changed")
    if candidate["occluders"] != reference["occluders"]:
        raise RuntimeError("Occluder-set baseline changed")
    if candidate["tie_break_losses"] or reference["tie_break_losses"]:
        raise RuntimeError("Unexpected tie-break losses")

    print()
    print(
        "PASS: track 143 winning-presence change was localized to real "
        "pixel-space Z-buffer competition without depth ties."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
