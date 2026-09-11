#!/usr/bin/env python3
"""Profile multi-Actor visibility stability across a small real case matrix.

Diagnostic only. Five fixed clip/camera cases are evaluated at four raster
resolutions. Each run builds all Actor surface rasters, resolves one shared
Z-buffer, and reports Actor visibility fractions without selecting production
parameters or modifying dataset products.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_camera_surface_depth_raster_v01 import build_actor_camera_surface_depth_raster
from actor_depth_zbuffer_v01 import ActorDepthRasterInput, resolve_actor_depth_zbuffer
from actor_visibility_statistics_v01 import calculate_actor_visibility_statistics
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT

CASES = (
    ("test_clip_001_9306612661000", "front_tele"),
    ("test_clip_001_9306612661000", "front_wide"),
    ("test_clip_063_18787721418000", "front_tele"),
    ("test_clip_564_95233721166000", "cross_left"),
    ("test_clip_316_1404486225261000", "cross_right"),
)
RASTERS = ((160, 90), (320, 180), (480, 270), (640, 360))
REFERENCE_RASTER = (640, 360)
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
DEPTH_TOLERANCE_M = 1e-9
OUTPUT = Path("/tmp/step7e_multicase_raster_stability.json")


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


def run_resolution(*, actors, ego, calibration, image_width, image_height,
                   raster_width, raster_height):
    inputs: list[ActorDepthRasterInput] = []
    rows: list[dict[str, Any]] = []

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
            image_width_px=image_width,
            image_height_px=image_height,
            raster_width=raster_width,
            raster_height=raster_height,
            near_plane_m=NEAR_PLANE_M,
            depth_tolerance_m=DEPTH_TOLERANCE_M,
        )
        occupied = result.surface_raster.occupied_cell_count
        status = (
            "unresolved_boundary" if result.unresolved_boundary_count
            else "rasterized" if occupied
            else "zero_occupied_cells"
        )
        rows.append({
            "track_id": track_id,
            "actor_class": str(actor.get("label_class", "")),
            "status": status,
            "occupied_cell_count": occupied,
            "unresolved_boundary_count": result.unresolved_boundary_count,
        })
        if status == "rasterized":
            inputs.append(ActorDepthRasterInput(track_id, result.surface_raster))

    zbuffer = resolve_actor_depth_zbuffer(
        tuple(inputs), depth_tolerance_m=DEPTH_TOLERANCE_M
    )
    stats = {
        item.actor_id: item
        for item in calculate_actor_visibility_statistics(zbuffer.actor_summaries)
    }
    for row in rows:
        item = stats.get(row["track_id"])
        row.update({
            "winning_cell_count": item.winning_cell_count if item else None,
            "occluded_cell_count": item.occluded_cell_count if item else None,
            "visible_fraction": item.visible_fraction if item else None,
        })

    return {
        "raster_width": raster_width,
        "raster_height": raster_height,
        "rasterized_actor_count": len(inputs),
        "zero_occupied_actor_count": sum(row["status"] == "zero_occupied_cells" for row in rows),
        "unresolved_actor_count": sum(row["status"] == "unresolved_boundary" for row in rows),
        "occupied_union_cell_count": zbuffer.occupied_union_cell_count,
        "contested_cell_count": zbuffer.contested_cell_count,
        "actors": rows,
    }


def main() -> int:
    case_reports = []
    all_deltas_160 = []
    all_deltas_320 = []
    all_deltas_480 = []

    for case_index, (anchor_id, camera_name) in enumerate(CASES, start=1):
        clip_id, stamp_ns = split_anchor_id(anchor_id)
        reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        ego, actors = exact_inputs(reader, stamp_ns)
        frame = reader.camera_indexes[camera_name].exact(stamp_ns)
        if frame is None:
            raise RuntimeError(f"Exact camera frame unavailable: {anchor_id} {camera_name}")
        calibration = load_camera_calibration(
            reader.clip_directory / "calibration" / f"{camera_name}.json",
            camera_name=camera_name,
            source_width=frame.value.width,
            source_height=frame.value.height,
        )
        if calibration.max_angle_rad is None:
            raise RuntimeError(f"Missing max_angle_rad: {anchor_id} {camera_name}")

        runs = [
            run_resolution(
                actors=actors, ego=ego, calibration=calibration,
                image_width=frame.value.width, image_height=frame.value.height,
                raster_width=width, raster_height=height,
            )
            for width, height in RASTERS
        ]
        by_resolution = {(run["raster_width"], run["raster_height"]): run for run in runs}
        reference = by_resolution[REFERENCE_RASTER]
        reference_by_actor = {
            row["track_id"]: row for row in reference["actors"]
            if row["visible_fraction"] is not None
        }

        comparisons = []
        for resolution in RASTERS[:-1]:
            run = by_resolution[resolution]
            current_by_actor = {
                row["track_id"]: row for row in run["actors"]
                if row["visible_fraction"] is not None
            }
            common = sorted(set(reference_by_actor) & set(current_by_actor))
            deltas = [
                abs(current_by_actor[key]["visible_fraction"] - reference_by_actor[key]["visible_fraction"])
                for key in common
            ]
            comparison = {
                "raster_width": resolution[0],
                "raster_height": resolution[1],
                "common_rasterized_actor_count": len(common),
                "reference_only_actor_count": len(set(reference_by_actor) - set(current_by_actor)),
                "candidate_only_actor_count": len(set(current_by_actor) - set(reference_by_actor)),
                "maximum_absolute_visible_fraction_delta": max(deltas) if deltas else None,
                "median_absolute_visible_fraction_delta": median(deltas) if deltas else None,
                "actor_deltas": [
                    {
                        "track_id": key,
                        "candidate_visible_fraction": current_by_actor[key]["visible_fraction"],
                        "reference_visible_fraction": reference_by_actor[key]["visible_fraction"],
                        "absolute_delta": abs(current_by_actor[key]["visible_fraction"] - reference_by_actor[key]["visible_fraction"]),
                    }
                    for key in common
                ],
            }
            comparisons.append(comparison)
            if resolution == (160, 90):
                all_deltas_160.extend(deltas)
            elif resolution == (320, 180):
                all_deltas_320.extend(deltas)
            elif resolution == (480, 270):
                all_deltas_480.extend(deltas)
            else:
                raise RuntimeError(
                    f"Unexpected candidate raster: {resolution}"
                )

        case_reports.append({
            "anchor_id": anchor_id,
            "camera_name": camera_name,
            "input_actor_count": len(actors),
            "runs": runs,
            "comparisons_to_640x360": comparisons,
        })
        print(f"Case {case_index}/{len(CASES)}: {anchor_id} {camera_name}")
        for run in runs:
            print(
                f"  {run['raster_width']}x{run['raster_height']}: "
                f"rasterized={run['rasterized_actor_count']} "
                f"zero={run['zero_occupied_actor_count']} "
                f"unresolved={run['unresolved_actor_count']} "
                f"union={run['occupied_union_cell_count']} "
                f"contested={run['contested_cell_count']}"
            )
        for item in comparisons:
            print(
                f"  vs 640x360 {item['raster_width']}x{item['raster_height']}: "
                f"common={item['common_rasterized_actor_count']} "
                f"max_delta={item['maximum_absolute_visible_fraction_delta']} "
                f"median_delta={item['median_absolute_visible_fraction_delta']}"
            )

    aggregate = {
        "case_count": len(case_reports),
        "comparison_actor_count_160x90": len(all_deltas_160),
        "maximum_absolute_visible_fraction_delta_160x90": max(all_deltas_160) if all_deltas_160 else None,
        "median_absolute_visible_fraction_delta_160x90": median(all_deltas_160) if all_deltas_160 else None,
        "comparison_actor_count_320x180": len(all_deltas_320),
        "maximum_absolute_visible_fraction_delta_320x180": max(all_deltas_320) if all_deltas_320 else None,
        "median_absolute_visible_fraction_delta_320x180": median(all_deltas_320) if all_deltas_320 else None,
        "comparison_actor_count_480x270": len(all_deltas_480),
        "maximum_absolute_visible_fraction_delta_480x270": max(all_deltas_480) if all_deltas_480 else None,
        "median_absolute_visible_fraction_delta_480x270": median(all_deltas_480) if all_deltas_480 else None,
    }
    report = {
        "rasters": [list(value) for value in RASTERS],
        "reference_raster": list(REFERENCE_RASTER),
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "near_plane_m": NEAR_PLANE_M,
        "aggregate": aggregate,
        "cases": case_reports,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("Aggregate")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print("Output:", OUTPUT)
    print("PASS: multicase raster-resolution stability profile completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
