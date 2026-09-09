#!/usr/bin/env python3
"""Classify zero-occupied Actors in one real keyframe/camera diagnostic.

The diagnostic separates: no prepared camera-facing surface, all prepared
triangles rejected outside angular FOV, generated geometry fully outside the
image raster, and image-intersecting geometry without a covered cell center.
It writes JSON under /tmp and does not modify dataset products.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_camera_surface_depth_raster_v01 import build_actor_camera_surface_depth_raster
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT

ANCHOR_ID = "test_clip_001_9306612661000"
CAMERA_NAME = "front_tele"
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
OUTPUT = Path("/tmp/step7e_zero_occupied_actor_classification.json")


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


def actor_id(actor: dict[str, Any]) -> str:
    value = actor.get("track_id")
    if value is None or str(value) == "":
        raise RuntimeError("Actor is missing a usable track_id")
    return str(value)


def classify_zero_actor(*, prepared_count: int, generated_count: int,
                        rejected_outside_count: int, conservative_count: int,
                        center_count: int, unresolved_count: int) -> str:
    if unresolved_count > 0:
        return "unresolved_boundary"
    if prepared_count == 0:
        return "no_prepared_camera_surface"
    if generated_count == 0:
        if rejected_outside_count == prepared_count:
            return "all_prepared_triangles_outside_angular_fov"
        return "no_generated_in_fov_geometry"
    if conservative_count == 0:
        return "generated_geometry_outside_image_raster"
    if center_count == 0:
        return "image_intersection_without_center_sample"
    raise RuntimeError("Zero-occupied Actor unexpectedly contains center samples")


def main() -> int:
    clip_id, stamp_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, stamp_ns)
    frame = reader.camera_indexes[CAMERA_NAME].exact(stamp_ns)
    if frame is None:
        raise RuntimeError("Exact camera frame unavailable")

    calibration = load_camera_calibration(
        reader.clip_directory / "calibration" / f"{CAMERA_NAME}.json",
        camera_name=CAMERA_NAME,
        source_width=frame.value.width,
        source_height=frame.value.height,
    )
    if calibration.max_angle_rad is None:
        raise RuntimeError("Camera calibration is missing max_angle_rad")

    zero_rows: list[dict[str, Any]] = []
    rasterized_count = 0

    for actor in sorted(actors, key=actor_id):
        track_id = actor_id(actor)
        corners_rig = actor_box_corners_in_rig(actor, recorded_ego_message=ego.message)
        corners_camera = tuple(calibration.rig_point_to_camera(point) for point in corners_rig)
        prepared = prepare_camera_facing_box_triangles(
            corners_camera, near_plane_m=NEAR_PLANE_M
        )
        result = build_actor_camera_surface_depth_raster(
            tuple(item.vertices_camera for item in prepared),
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
        if result.surface_raster.occupied_cell_count > 0:
            rasterized_count += 1
            continue

        generated = tuple(
            sample
            for triangle_result in result.triangle_results
            for sample in triangle_result.all_triangle_samples
        )
        rejected_outside_count = sum(
            triangle_result.subdivision.rejected_outside_triangle_count
            for triangle_result in result.triangle_results
        )
        accepted_leaf_count = sum(
            len(triangle_result.subdivision.accepted_inside_triangles)
            for triangle_result in result.triangle_results
        )
        boundary_leaf_count = sum(
            len(triangle_result.subdivision.boundary_approximated_triangles)
            for triangle_result in result.triangle_results
        )
        conservative_count = sum(item.conservative_cell_count for item in generated)
        center_count = sum(item.center_sampled_cell_count for item in generated)
        zero_conservative_triangles = sum(
            item.conservative_cell_count == 0 for item in generated
        )
        subcell_triangles = sum(
            item.conservative_cell_count > 0 and item.center_sampled_cell_count == 0
            for item in generated
        )
        reason = classify_zero_actor(
            prepared_count=len(prepared),
            generated_count=len(generated),
            rejected_outside_count=rejected_outside_count,
            conservative_count=conservative_count,
            center_count=center_count,
            unresolved_count=result.unresolved_boundary_count,
        )
        zero_rows.append({
            "track_id": track_id,
            "actor_class": str(actor.get("label_class", "")),
            "reason": reason,
            "prepared_source_triangle_count": len(prepared),
            "rejected_outside_triangle_count": rejected_outside_count,
            "accepted_leaf_count": accepted_leaf_count,
            "approximated_boundary_leaf_count": boundary_leaf_count,
            "generated_triangle_sample_count": len(generated),
            "zero_conservative_triangle_count": zero_conservative_triangles,
            "subcell_triangle_count": subcell_triangles,
            "conservative_cell_count": conservative_count,
            "center_sample_count": center_count,
            "unresolved_boundary_count": result.unresolved_boundary_count,
        })

    reasons = Counter(row["reason"] for row in zero_rows)
    aggregate = {
        "input_actor_count": len(actors),
        "rasterized_actor_count": rasterized_count,
        "zero_occupied_actor_count": len(zero_rows),
        "reason_counts": dict(sorted(reasons.items())),
    }
    if rasterized_count + len(zero_rows) != len(actors):
        raise RuntimeError("Actor status counts do not close")
    if any(row["center_sample_count"] != 0 for row in zero_rows):
        raise RuntimeError("Zero-occupied Actor contains center samples")

    report = {
        "anchor_id": ANCHOR_ID,
        "camera_name": CAMERA_NAME,
        "raster_width": RASTER_WIDTH,
        "raster_height": RASTER_HEIGHT,
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "aggregate": aggregate,
        "zero_occupied_actors": zero_rows,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("Zero-occupied real Actor classification")
    print("  anchor:", ANCHOR_ID)
    print("  camera:", CAMERA_NAME)
    print("  input_actor_count:", len(actors))
    print("  rasterized_actor_count:", rasterized_count)
    print("  zero_occupied_actor_count:", len(zero_rows))
    print("  reason_counts:")
    for reason, count in sorted(reasons.items()):
        print(f"    {reason}: {count}")
    print("  zero_rows:")
    for row in zero_rows:
        print(
            f"    track={row['track_id']} reason={row['reason']} "
            f"prepared={row['prepared_source_triangle_count']} "
            f"generated={row['generated_triangle_sample_count']} "
            f"conservative={row['conservative_cell_count']} "
            f"subcell_triangles={row['subcell_triangle_count']}"
        )
    print("Output:", OUTPUT)
    print("PASS: zero-occupied real Actors were classified without fabricated visibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
