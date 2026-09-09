#!/usr/bin/env python3
"""Profile safe FOV-boundary depth sampling on known real Actor/camera pairs.

Diagnostic only. For each prepared camera-facing Actor triangle, this script
runs angular-FOV subdivision, samples fully inside leaves, converts approximated
boundary leaves into safe in-FOV polygons, and records whether generated leaf
triangles contain raster-center samples. It writes JSON under /tmp and does not
modify dataset products.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from actor_box_projection_v01 import actor_box_corners_in_rig
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from fov_subdivided_triangle_depth_samples_v01 import (
    sample_fov_subdivided_camera_triangle_depths,
)
from project_paths import ALPASIM_DATA_ROOT

NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
OUTPUT = Path("/tmp/step7e_real_fov_boundary_depth_sampling_profile.json")

CASES = (
    ("test_clip_001_9306612661000", "front_tele", "13"),
    ("test_clip_001_9306612661000", "front_tele", "29"),
    ("test_clip_001_9306612661000", "front_wide", "27"),
    ("test_clip_063_18787721418000", "front_tele", "18"),
    ("test_clip_063_18787721418000", "front_tele", "98"),
    ("test_clip_185_222151025385000", "front_tele", "229"),
    ("test_clip_185_222151025385000", "front_tele", "238"),
    ("test_clip_410_2056042617247000", "front_tele", "157"),
)


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
    rows: list[dict[str, Any]] = []

    for case_index, (anchor_id, camera_name, track_id) in enumerate(CASES, start=1):
        clip_id, stamp_ns = split_anchor_id(anchor_id)
        reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        ego, actors = exact_inputs(reader, stamp_ns)
        actor = find_actor(actors, track_id)
        frame = reader.camera_indexes[camera_name].exact(stamp_ns)
        if frame is None:
            raise RuntimeError(
                f"Exact camera frame unavailable: {anchor_id} {camera_name}"
            )

        calibration = load_camera_calibration(
            reader.clip_directory / "calibration" / f"{camera_name}.json",
            camera_name=camera_name,
            source_width=frame.value.width,
            source_height=frame.value.height,
        )
        if calibration.max_angle_rad is None:
            raise RuntimeError(f"Missing max_angle_rad: {anchor_id} {camera_name}")

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

        totals = {
            "accepted_leaf_count": 0,
            "approximated_boundary_leaf_count": 0,
            "depth_limited_leaf_count": 0,
            "subdivision_unmeasurable_leaf_count": 0,
            "measurable_boundary_polygon_count": 0,
            "polygon_unmeasurable_count": 0,
            "polygon_degenerate_count": 0,
            "sampled_inside_triangle_count": 0,
            "sampled_boundary_triangle_count": 0,
            "inside_zero_center_sample_triangle_count": 0,
            "boundary_zero_center_sample_triangle_count": 0,
            "inside_zero_conservative_cell_triangle_count": 0,
            "boundary_zero_conservative_cell_triangle_count": 0,
            "inside_subcell_triangle_count": 0,
            "boundary_subcell_triangle_count": 0,
            "inside_conservative_cell_count": 0,
            "boundary_conservative_cell_count": 0,
            "inside_center_sample_count": 0,
            "boundary_center_sample_count": 0,
            "unresolved_boundary_count": 0,
        }

        for prepared_triangle in prepared:
            result = sample_fov_subdivided_camera_triangle_depths(
                prepared_triangle.vertices_camera,
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
            subdivision = result.subdivision
            totals["accepted_leaf_count"] += len(
                subdivision.accepted_inside_triangles
            )
            totals["approximated_boundary_leaf_count"] += len(
                subdivision.boundary_approximated_triangles
            )
            totals["depth_limited_leaf_count"] += len(
                subdivision.boundary_depth_limited_triangles
            )
            totals["subdivision_unmeasurable_leaf_count"] += len(
                subdivision.boundary_unmeasurable_triangles
            )
            totals["measurable_boundary_polygon_count"] += (
                result.measurable_boundary_polygon_count
            )
            totals["polygon_unmeasurable_count"] += (
                result.unmeasurable_boundary_polygon_count
            )
            totals["polygon_degenerate_count"] += (
                result.degenerate_boundary_polygon_count
            )
            totals["sampled_inside_triangle_count"] += (
                result.sampled_inside_triangle_count
            )
            totals["sampled_boundary_triangle_count"] += (
                result.sampled_boundary_triangle_count
            )
            totals["inside_zero_center_sample_triangle_count"] += sum(
                sample.center_sampled_cell_count == 0
                for sample in result.accepted_triangle_samples
            )
            totals["boundary_zero_center_sample_triangle_count"] += sum(
                sample.center_sampled_cell_count == 0
                for sample in result.boundary_triangle_samples
            )

            totals[
                "inside_zero_conservative_cell_triangle_count"
            ] += sum(
                sample.conservative_cell_count == 0
                for sample in result.accepted_triangle_samples
            )
            totals[
                "boundary_zero_conservative_cell_triangle_count"
            ] += sum(
                sample.conservative_cell_count == 0
                for sample in result.boundary_triangle_samples
            )

            totals["inside_subcell_triangle_count"] += sum(
                sample.conservative_cell_count > 0
                and sample.center_sampled_cell_count == 0
                for sample in result.accepted_triangle_samples
            )
            totals["boundary_subcell_triangle_count"] += sum(
                sample.conservative_cell_count > 0
                and sample.center_sampled_cell_count == 0
                for sample in result.boundary_triangle_samples
            )

            totals["inside_conservative_cell_count"] += sum(
                sample.conservative_cell_count
                for sample in result.accepted_triangle_samples
            )
            totals["boundary_conservative_cell_count"] += sum(
                sample.conservative_cell_count
                for sample in result.boundary_triangle_samples
            )

            totals["inside_center_sample_count"] += sum(
                sample.center_sampled_cell_count
                for sample in result.accepted_triangle_samples
            )
            totals["boundary_center_sample_count"] += sum(
                sample.center_sampled_cell_count
                for sample in result.boundary_triangle_samples
            )
            totals["unresolved_boundary_count"] += result.unresolved_boundary_count

        row = {
            "anchor_id": anchor_id,
            "camera_name": camera_name,
            "track_id": track_id,
            "actor_class": str(actor.get("label_class", "")),
            "source_triangle_count": len(prepared),
            **totals,
        }
        rows.append(row)
        print(
            f"Case {case_index}/8: {camera_name} track={track_id} "
            f"source={len(prepared)} accepted={totals['accepted_leaf_count']} "
            f"boundary={totals['approximated_boundary_leaf_count']} "
            f"boundary_triangles={totals['sampled_boundary_triangle_count']} "
            f"boundary_zero={totals['boundary_zero_center_sample_triangle_count']} "
            f"boundary_samples={totals['boundary_center_sample_count']} "
            f"unresolved={totals['unresolved_boundary_count']}"
        )

    aggregate_keys = tuple(
        key for key in rows[0]
        if key.endswith("_count") and key != "source_triangle_count"
    )
    aggregate = {
        "case_count": len(rows),
        "source_triangle_count": sum(row["source_triangle_count"] for row in rows),
        **{key: sum(row[key] for row in rows) for key in aggregate_keys},
    }
    report = {
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "near_plane_m": NEAR_PLANE_M,
        "raster_width": RASTER_WIDTH,
        "raster_height": RASTER_HEIGHT,
        "rows": rows,
        "aggregate": aggregate,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\nAggregate")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print("Output:", OUTPUT)

    if aggregate["depth_limited_leaf_count"] != 0:
        raise RuntimeError("Depth-limited FOV boundary leaves remain")
    if aggregate["subdivision_unmeasurable_leaf_count"] != 0:
        raise RuntimeError("Subdivision-unmeasurable FOV boundary leaves remain")
    if aggregate["polygon_unmeasurable_count"] != 0:
        raise RuntimeError("Safe boundary polygon construction was unmeasurable")
    if aggregate["polygon_degenerate_count"] != 0:
        raise RuntimeError("Safe boundary polygon construction was degenerate")
    if aggregate["unresolved_boundary_count"] != 0:
        raise RuntimeError("Unresolved boundary diagnostics remain")

    print("PASS: real FOV boundary depth-sampling profile completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
