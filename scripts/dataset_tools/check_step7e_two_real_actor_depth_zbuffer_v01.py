#!/usr/bin/env python3
"""Validate two real Actors through the shared depth Z-buffer path.

Diagnostic only. The script loads two fixed Actors from one keyframe and camera,
builds each Actor's complete surface-depth raster, resolves shared-cell depth
competition, and calculates visibility statistics. It writes JSON under /tmp
and does not modify dataset products.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_camera_surface_depth_raster_v01 import build_actor_camera_surface_depth_raster
from actor_depth_zbuffer_v01 import ActorDepthRasterInput, resolve_actor_depth_zbuffer
from actor_visibility_statistics_v01 import calculate_actor_visibility_statistics
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT

ANCHOR_ID = "test_clip_001_9306612661000"
CAMERA_NAME = "front_tele"
TRACK_IDS = ("13", "29")
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
DEPTH_TOLERANCE_M = 1e-9
OUTPUT = Path("/tmp/step7e_two_real_actor_depth_zbuffer.json")


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
        raise RuntimeError(f"Expected one Actor track={track_id}, found {len(matches)}")
    return matches[0]


def main() -> int:
    clip_id, stamp_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, stamp_ns)
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

    zbuffer_inputs: list[ActorDepthRasterInput] = []
    actor_reports: list[dict[str, Any]] = []

    for track_id in TRACK_IDS:
        actor = find_actor(actors, track_id)
        corners_rig = actor_box_corners_in_rig(actor, recorded_ego_message=ego.message)
        corners_camera = tuple(calibration.rig_point_to_camera(point) for point in corners_rig)
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
            image_width_px=frame.value.width,
            image_height_px=frame.value.height,
            raster_width=RASTER_WIDTH,
            raster_height=RASTER_HEIGHT,
            near_plane_m=NEAR_PLANE_M,
            depth_tolerance_m=DEPTH_TOLERANCE_M,
        )
        if result.unresolved_boundary_count != 0:
            raise RuntimeError(f"Unresolved boundary diagnostics for track={track_id}")
        if result.surface_raster.occupied_cell_count <= 0:
            raise RuntimeError(f"No occupied surface cells for track={track_id}")

        depths = [item.depth_m for item in result.surface_raster.cell_depths]
        if not all(math.isfinite(depth) and depth > 0.0 for depth in depths):
            raise RuntimeError(f"Invalid surface depth for track={track_id}")

        zbuffer_inputs.append(
            ActorDepthRasterInput(
                actor_id=track_id,
                surface_raster=result.surface_raster,
            )
        )
        actor_reports.append({
            "track_id": track_id,
            "actor_class": str(actor.get("label_class", "")),
            "prepared_source_triangle_count": len(prepared),
            "generated_triangle_sample_count": result.generated_triangle_sample_count,
            "zero_center_sample_triangle_count": result.zero_center_sample_triangle_count,
            "input_depth_sample_count": result.surface_raster.input_depth_sample_count,
            "occupied_cell_count": result.surface_raster.occupied_cell_count,
            "minimum_surface_depth_m": min(depths),
            "maximum_surface_depth_m": max(depths),
            "unresolved_boundary_count": result.unresolved_boundary_count,
        })

    zbuffer = resolve_actor_depth_zbuffer(
        tuple(zbuffer_inputs),
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )
    statistics = calculate_actor_visibility_statistics(zbuffer.actor_summaries)

    summary_by_id = {item.actor_id: item for item in zbuffer.actor_summaries}
    statistic_by_id = {item.actor_id: item for item in statistics}
    for actor_report in actor_reports:
        actor_id = actor_report["track_id"]
        summary = summary_by_id[actor_id]
        statistic = statistic_by_id[actor_id]
        if summary.occupied_cell_count != actor_report["occupied_cell_count"]:
            raise RuntimeError(f"Occupied count changed for track={actor_id}")
        if summary.winning_cell_count + summary.occluded_cell_count != summary.occupied_cell_count:
            raise RuntimeError(f"Visibility counts do not close for track={actor_id}")
        actor_report.update({
            "winning_cell_count": summary.winning_cell_count,
            "occluded_cell_count": summary.occluded_cell_count,
            "visible_fraction": statistic.visible_fraction,
            "occluded_fraction": statistic.occluded_fraction,
            "fully_visible": statistic.fully_visible,
            "fully_occluded": statistic.fully_occluded,
        })

    if zbuffer.occupied_union_cell_count != len(zbuffer.cell_winners):
        raise RuntimeError("Z-buffer union count is inconsistent")
    winner_counts = {track_id: 0 for track_id in TRACK_IDS}
    for winner in zbuffer.cell_winners:
        winner_counts[winner.actor_id] += 1
    if any(winner_counts[track_id] != summary_by_id[track_id].winning_cell_count for track_id in TRACK_IDS):
        raise RuntimeError("Winner records and Actor summaries are inconsistent")

    report = {
        "anchor_id": ANCHOR_ID,
        "clip_id": clip_id,
        "stamp_ns": stamp_ns,
        "camera_name": CAMERA_NAME,
        "track_ids": list(TRACK_IDS),
        "image_width_px": frame.value.width,
        "image_height_px": frame.value.height,
        "raster_width": RASTER_WIDTH,
        "raster_height": RASTER_HEIGHT,
        "near_plane_m": NEAR_PLANE_M,
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "depth_tolerance_m": DEPTH_TOLERANCE_M,
        "occupied_union_cell_count": zbuffer.occupied_union_cell_count,
        "contested_cell_count": zbuffer.contested_cell_count,
        "actors": actor_reports,
        "first_ten_contested_winners": [
            {
                "column": winner.cell.column,
                "row": winner.cell.row,
                "actor_id": winner.actor_id,
                "depth_m": winner.depth_m,
            }
            for winner in zbuffer.cell_winners
            if sum(
                winner.cell in {item.cell for item in actor_input.surface_raster.cell_depths}
                for actor_input in zbuffer_inputs
            ) > 1
        ][:10],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("Two real Actor depth Z-buffer")
    print("  anchor:", ANCHOR_ID)
    print("  camera:", CAMERA_NAME)
    print("  occupied_union_cells:", zbuffer.occupied_union_cell_count)
    print("  contested_cells:", zbuffer.contested_cell_count)
    for item in actor_reports:
        print(
            f"  track={item['track_id']} occupied={item['occupied_cell_count']} "
            f"winning={item['winning_cell_count']} "
            f"occluded={item['occluded_cell_count']} "
            f"visible_fraction={item['visible_fraction']} "
            f"depth_range=({item['minimum_surface_depth_m']}, "
            f"{item['maximum_surface_depth_m']})"
        )
    print("Output:", OUTPUT)
    print("PASS: two real Actors produced a valid shared depth Z-buffer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
