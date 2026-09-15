#!/usr/bin/env python3
"""Validate one real Actor through the complete single-Actor surface raster path.

Diagnostic only. The script loads one fixed real Actor/camera case, prepares all
camera-facing box triangles, applies angular-FOV processing and safe boundary
geometry, samples raster-center depths, and merges the nearest surface per cell.
It writes JSON under /tmp and does not modify dataset products.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from step7.actor_box_projection_v01 import actor_box_corners_in_rig
from step7.actor_camera_surface_depth_raster_v01 import (
    build_actor_camera_surface_depth_raster,
)
from step7.camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT

ANCHOR_ID = "test_clip_001_9306612661000"
CAMERA_NAME = "front_tele"
TRACK_ID = "13"
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
OUTPUT = Path("/tmp/step7e_single_real_actor_surface_depth_raster.json")


def split_anchor_id(anchor_id: str) -> tuple[str, int]:
    clip_id, stamp_text = anchor_id.rsplit("_", 1)
    return clip_id, int(stamp_text)


def exact_inputs(reader: DrivingClipReader, stamp_ns: int):
    ego = reader.get_recorded_ego_state_at_or_before(stamp_ns)
    if ego is None or ego.stamp_ns != stamp_ns:
        raise RuntimeError(f"Exact recorded Ego state unavailable: {stamp_ns}")
    snapshots = reader.get_actor_snapshots(stamp_ns, duration_ns=0)
    if len(snapshots) != 1 or snapshots[0].stamp_ns != stamp_ns:
        raise RuntimeError(f"Exact Actor snapshot unavailable: {stamp_ns}")
    return ego, snapshots[0].message["actors"]


def find_actor(actors: list[dict[str, Any]], track_id: str) -> dict[str, Any]:
    matches = [actor for actor in actors if str(actor.get("track_id")) == track_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Actor track={track_id}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    clip_id, stamp_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, stamp_ns)
    actor = find_actor(actors, TRACK_ID)
    frame = reader.camera_indexes[CAMERA_NAME].exact(stamp_ns)
    if frame is None:
        raise RuntimeError(f"Exact camera frame unavailable: {ANCHOR_ID} {CAMERA_NAME}")

    calibration = load_camera_calibration(
        reader.clip_directory / "calibration" / f"{CAMERA_NAME}.json",
        camera_name=CAMERA_NAME,
        source_width=frame.value.width,
        source_height=frame.value.height,
    )
    if calibration.max_angle_rad is None:
        raise RuntimeError("Camera calibration is missing max_angle_rad")

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
    triangles_camera = tuple(item.vertices_camera for item in prepared)

    result = build_actor_camera_surface_depth_raster(
        triangles_camera,
        calibration,
        max_angle_rad=calibration.max_angle_rad,
        maximum_depth=MAXIMUM_DEPTH,
        maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
        image_width_px=frame.value.width,
        image_height_px=frame.value.height,
        raster_width=RASTER_WIDTH,
        raster_height=RASTER_HEIGHT,
        near_plane_m=NEAR_PLANE_M,
    )
    raster = result.surface_raster
    depths = [item.depth_m for item in raster.cell_depths]

    accepted_leaf_count = sum(
        len(item.subdivision.accepted_inside_triangles)
        for item in result.triangle_results
    )
    approximated_boundary_leaf_count = sum(
        len(item.subdivision.boundary_approximated_triangles)
        for item in result.triangle_results
    )
    boundary_triangle_sample_count = sum(
        item.sampled_boundary_triangle_count
        for item in result.triangle_results
    )

    if result.source_triangle_count != len(prepared):
        raise RuntimeError("Source triangle count is inconsistent")
    if result.unresolved_boundary_count != 0:
        raise RuntimeError("Unresolved FOV boundary diagnostics remain")
    if raster.occupied_cell_count <= 0:
        raise RuntimeError("Real Actor surface raster contains no occupied cells")
    if raster.occupied_cell_count != len(raster.cell_depths):
        raise RuntimeError("Occupied-cell count is inconsistent")
    if not all(math.isfinite(depth) and depth > 0.0 for depth in depths):
        raise RuntimeError("Actor surface depths must be positive and finite")
    cells = tuple(item.cell for item in raster.cell_depths)
    if cells != tuple(sorted(cells, key=lambda cell: (cell.row, cell.column))):
        raise RuntimeError("Actor surface cells are not row-major sorted")
    if len(set(cells)) != len(cells):
        raise RuntimeError("Actor surface raster contains duplicate cells")

    report = {
        "anchor_id": ANCHOR_ID,
        "clip_id": clip_id,
        "stamp_ns": stamp_ns,
        "camera_name": CAMERA_NAME,
        "track_id": TRACK_ID,
        "actor_class": str(actor.get("label_class", "")),
        "image_width_px": frame.value.width,
        "image_height_px": frame.value.height,
        "raster_width": RASTER_WIDTH,
        "raster_height": RASTER_HEIGHT,
        "near_plane_m": NEAR_PLANE_M,
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "prepared_source_triangle_count": len(prepared),
        "accepted_leaf_count": accepted_leaf_count,
        "approximated_boundary_leaf_count": approximated_boundary_leaf_count,
        "boundary_triangle_sample_count": boundary_triangle_sample_count,
        "generated_triangle_sample_count": result.generated_triangle_sample_count,
        "zero_center_sample_triangle_count": result.zero_center_sample_triangle_count,
        "unresolved_boundary_count": result.unresolved_boundary_count,
        "input_depth_sample_count": raster.input_depth_sample_count,
        "occupied_cell_count": raster.occupied_cell_count,
        "replaced_sample_count": raster.replaced_sample_count,
        "discarded_farther_or_equal_sample_count": raster.discarded_farther_or_equal_sample_count,
        "minimum_surface_depth_m": min(depths),
        "maximum_surface_depth_m": max(depths),
        "first_ten_cells": [
            {
                "column": item.cell.column,
                "row": item.cell.row,
                "depth_m": item.depth_m,
                "winning_triangle_index": item.winning_triangle_index,
                "sample_u_px": item.sample_u_px,
                "sample_v_px": item.sample_v_px,
            }
            for item in raster.cell_depths[:10]
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("Single real Actor surface-depth raster")
    print("  anchor:", ANCHOR_ID)
    print("  camera:", CAMERA_NAME)
    print("  track:", TRACK_ID)
    print("  source_triangles:", len(prepared))
    print("  accepted_leaves:", accepted_leaf_count)
    print("  boundary_leaves:", approximated_boundary_leaf_count)
    print("  generated_triangles:", result.generated_triangle_sample_count)
    print("  zero_center_triangles:", result.zero_center_sample_triangle_count)
    print("  input_depth_samples:", raster.input_depth_sample_count)
    print("  occupied_cells:", raster.occupied_cell_count)
    print("  replaced_samples:", raster.replaced_sample_count)
    print("  discarded_samples:", raster.discarded_farther_or_equal_sample_count)
    print("  surface_depth_range_m:", min(depths), max(depths))
    print("  unresolved:", result.unresolved_boundary_count)
    print("Output:", OUTPUT)
    print("PASS: one real Actor produced a valid merged surface-depth raster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
