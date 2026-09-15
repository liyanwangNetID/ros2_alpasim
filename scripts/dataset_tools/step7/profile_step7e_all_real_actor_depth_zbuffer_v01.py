#!/usr/bin/env python3
"""Profile all rasterizable real Actors in one fixed keyframe and camera.

Diagnostic only. Each Actor is converted into a camera-facing surface-depth
raster. Actors with occupied cells are submitted to one shared Z-buffer and
visibility statistics are calculated. Actors without occupied cells and Actors
with unresolved boundary diagnostics remain explicit in the JSON report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from step7.actor_box_projection_v01 import actor_box_corners_in_rig
from step7.actor_camera_surface_depth_raster_v01 import build_actor_camera_surface_depth_raster
from step7.actor_depth_zbuffer_v01 import ActorDepthRasterInput, resolve_actor_depth_zbuffer
from step7.actor_visibility_statistics_v01 import calculate_actor_visibility_statistics
from step7.camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT

ANCHOR_ID = "test_clip_001_9306612661000"
CAMERA_NAME = "front_tele"
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
DEPTH_TOLERANCE_M = 1e-9
OUTPUT = Path("/tmp/step7e_all_real_actor_depth_zbuffer.json")


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


def stable_actor_id(actor: dict[str, Any]) -> str:
    value = actor.get("track_id")
    if value is None or str(value) == "":
        raise RuntimeError("Actor is missing a usable track_id")
    return str(value)


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

    actor_reports: list[dict[str, Any]] = []
    zbuffer_inputs: list[ActorDepthRasterInput] = []
    seen_ids: set[str] = set()

    for actor in sorted(actors, key=stable_actor_id):
        actor_id = stable_actor_id(actor)
        if actor_id in seen_ids:
            raise RuntimeError(f"Duplicate Actor track_id: {actor_id}")
        seen_ids.add(actor_id)

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
        raster = result.surface_raster
        depths = [item.depth_m for item in raster.cell_depths]
        if not all(math.isfinite(depth) and depth > 0.0 for depth in depths):
            raise RuntimeError(f"Invalid surface depth for track={actor_id}")

        status = (
            "unresolved_boundary"
            if result.unresolved_boundary_count > 0
            else "rasterized"
            if raster.occupied_cell_count > 0
            else "zero_occupied_cells"
        )
        report = {
            "track_id": actor_id,
            "actor_class": str(actor.get("label_class", "")),
            "status": status,
            "prepared_source_triangle_count": len(prepared),
            "generated_triangle_sample_count": result.generated_triangle_sample_count,
            "zero_center_sample_triangle_count": result.zero_center_sample_triangle_count,
            "input_depth_sample_count": raster.input_depth_sample_count,
            "occupied_cell_count": raster.occupied_cell_count,
            "unresolved_boundary_count": result.unresolved_boundary_count,
            "minimum_surface_depth_m": min(depths) if depths else None,
            "maximum_surface_depth_m": max(depths) if depths else None,
        }
        actor_reports.append(report)

        if status == "rasterized":
            zbuffer_inputs.append(
                ActorDepthRasterInput(actor_id=actor_id, surface_raster=raster)
            )

    zbuffer = resolve_actor_depth_zbuffer(
        tuple(zbuffer_inputs),
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )
    statistics = calculate_actor_visibility_statistics(zbuffer.actor_summaries)
    statistics_by_id = {item.actor_id: item for item in statistics}

    for report in actor_reports:
        statistic = statistics_by_id.get(report["track_id"])
        if statistic is None:
            report.update({
                "winning_cell_count": None,
                "occluded_cell_count": None,
                "visible_fraction": None,
                "occluded_fraction": None,
                "fully_visible": False,
                "fully_occluded": False,
            })
            continue
        report.update({
            "winning_cell_count": statistic.winning_cell_count,
            "occluded_cell_count": statistic.occluded_cell_count,
            "visible_fraction": statistic.visible_fraction,
            "occluded_fraction": statistic.occluded_fraction,
            "fully_visible": statistic.fully_visible,
            "fully_occluded": statistic.fully_occluded,
        })

    rasterized_reports = [item for item in actor_reports if item["status"] == "rasterized"]
    zero_reports = [item for item in actor_reports if item["status"] == "zero_occupied_cells"]
    unresolved_reports = [item for item in actor_reports if item["status"] == "unresolved_boundary"]
    partially_occluded = [
        item for item in rasterized_reports
        if item["occluded_cell_count"] not in (None, 0)
        and item["winning_cell_count"] not in (None, 0)
    ]
    fully_occluded = [item for item in rasterized_reports if item["fully_occluded"]]

    if zbuffer.occupied_union_cell_count != len(zbuffer.cell_winners):
        raise RuntimeError("Z-buffer union count is inconsistent")
    if any(item["unresolved_boundary_count"] != 0 for item in rasterized_reports):
        raise RuntimeError("Rasterized Actor unexpectedly contains unresolved boundaries")

    aggregate = {
        "input_actor_count": len(actor_reports),
        "rasterized_actor_count": len(rasterized_reports),
        "zero_occupied_actor_count": len(zero_reports),
        "unresolved_actor_count": len(unresolved_reports),
        "fully_visible_actor_count": sum(item["fully_visible"] for item in rasterized_reports),
        "partially_occluded_actor_count": len(partially_occluded),
        "fully_occluded_actor_count": len(fully_occluded),
        "occupied_union_cell_count": zbuffer.occupied_union_cell_count,
        "contested_cell_count": zbuffer.contested_cell_count,
        "sum_actor_occupied_cell_count": sum(item["occupied_cell_count"] for item in rasterized_reports),
        "sum_actor_winning_cell_count": sum(item["winning_cell_count"] for item in rasterized_reports),
        "sum_actor_occluded_cell_count": sum(item["occluded_cell_count"] for item in rasterized_reports),
    }
    if aggregate["sum_actor_winning_cell_count"] != aggregate["occupied_union_cell_count"]:
        raise RuntimeError("Winning-cell total does not equal occupied union")

    report = {
        "anchor_id": ANCHOR_ID,
        "clip_id": clip_id,
        "stamp_ns": stamp_ns,
        "camera_name": CAMERA_NAME,
        "image_width_px": frame.value.width,
        "image_height_px": frame.value.height,
        "raster_width": RASTER_WIDTH,
        "raster_height": RASTER_HEIGHT,
        "near_plane_m": NEAR_PLANE_M,
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "depth_tolerance_m": DEPTH_TOLERANCE_M,
        "aggregate": aggregate,
        "actors": actor_reports,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("All real Actor depth Z-buffer")
    print("  anchor:", ANCHOR_ID)
    print("  camera:", CAMERA_NAME)
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print("  visibility_rows:")
    for item in rasterized_reports:
        print(
            f"    track={item['track_id']} occupied={item['occupied_cell_count']} "
            f"winning={item['winning_cell_count']} occluded={item['occluded_cell_count']} "
            f"visible_fraction={item['visible_fraction']}"
        )
    print("Output:", OUTPUT)
    print("PASS: all rasterizable real Actors produced a valid shared depth Z-buffer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
