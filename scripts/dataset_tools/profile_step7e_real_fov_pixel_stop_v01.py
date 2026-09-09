#!/usr/bin/env python3
"""Profile pixel-scale angular-FOV stopping on known real boundary pairs.

Step 7E-2B diagnostic only. The script scans boundary extent limits and maximum
recursion depths for eight previously identified Actor/camera pairs. It writes
only a JSON report under /tmp and does not modify annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from actor_box_projection_v01 import actor_box_corners_in_rig
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT
from triangle_angular_fov_subdivision_v01 import subdivide_triangle_to_angular_fov

NEAR_PLANE_M = 1e-3
BOUNDARY_EXTENTS_PX = (0.5, 1.0, 2.0, 4.0)
MAXIMUM_DEPTHS = (4, 6, 8)
BOUNDARY_CASES = (
    ("test_clip_001_9306612661000", "front_tele", "13"),
    ("test_clip_001_9306612661000", "front_tele", "29"),
    ("test_clip_001_9306612661000", "front_wide", "27"),
    ("test_clip_063_18787721418000", "front_tele", "18"),
    ("test_clip_063_18787721418000", "front_tele", "98"),
    ("test_clip_185_222151025385000", "front_tele", "229"),
    ("test_clip_185_222151025385000", "front_tele", "238"),
    ("test_clip_410_2056042617247000", "front_tele", "157"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/step7e_real_fov_pixel_stop_profile.json"),
    )
    return parser.parse_args()


def split_anchor_id(anchor_id: str) -> tuple[str, int]:
    clip_id, anchor_text = anchor_id.rsplit("_", 1)
    return clip_id, int(anchor_text)


def exact_inputs(reader: DrivingClipReader, anchor_ns: int):
    ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
    if ego is None or ego.stamp_ns != anchor_ns:
        raise RuntimeError(f"Exact recorded Ego state unavailable: {anchor_ns}")
    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
        raise RuntimeError(f"Exact Actor snapshot unavailable: {anchor_ns}")
    return ego, snapshots[0].message["actors"]


def find_actor(actors: list[dict[str, Any]], track_id: str) -> dict[str, Any]:
    matches = [actor for actor in actors if str(actor.get("track_id")) == track_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Actor track={track_id}, found {len(matches)}")
    return matches[0]


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for case_index, (anchor_id, camera_name, track_id) in enumerate(BOUNDARY_CASES, start=1):
        clip_id, anchor_ns = split_anchor_id(anchor_id)
        reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        ego, actors = exact_inputs(reader, anchor_ns)
        actor = find_actor(actors, track_id)
        frame = reader.camera_indexes[camera_name].exact(anchor_ns)
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

        corners_rig = actor_box_corners_in_rig(actor, recorded_ego_message=ego.message)
        corners_camera = tuple(calibration.rig_point_to_camera(point) for point in corners_rig)
        prepared = prepare_camera_facing_box_triangles(
            corners_camera,
            near_plane_m=NEAR_PLANE_M,
        )
        if not prepared:
            failures.append({
                "anchor_id": anchor_id,
                "camera_name": camera_name,
                "track_id": track_id,
                "reason": "no_prepared_triangles",
            })
            continue

        print(
            f"Case {case_index}/8: anchor={anchor_id} camera={camera_name} "
            f"track={track_id} source_triangles={len(prepared)}"
        )

        for extent_limit in BOUNDARY_EXTENTS_PX:
            for maximum_depth in MAXIMUM_DEPTHS:
                accepted = 0
                approximated = 0
                depth_limited = 0
                unmeasurable = 0
                rejected = 0
                deepest = 0
                approximation_extents: list[float] = []
                depth_limited_extents: list[float] = []

                for item in prepared:
                    result = subdivide_triangle_to_angular_fov(
                        item.vertices_camera,
                        max_angle_rad=calibration.max_angle_rad,
                        maximum_depth=maximum_depth,
                        calibration=calibration,
                        maximum_boundary_extent_px=extent_limit,
                    )
                    accepted += len(result.accepted_inside_triangles)
                    approximated += len(result.boundary_approximated_triangles)
                    depth_limited += len(result.boundary_depth_limited_triangles)
                    unmeasurable += len(result.boundary_unmeasurable_triangles)
                    rejected += result.rejected_outside_triangle_count
                    deepest = max(deepest, result.maximum_depth_reached)
                    approximation_extents.extend(
                        float(boundary.maximum_boundary_extent_px)
                        for boundary in result.boundary_approximated_triangles
                        if boundary.maximum_boundary_extent_px is not None
                    )
                    depth_limited_extents.extend(
                        float(boundary.maximum_boundary_extent_px)
                        for boundary in result.boundary_depth_limited_triangles
                        if boundary.maximum_boundary_extent_px is not None
                    )

                leaves = accepted + approximated + depth_limited + unmeasurable + rejected
                row = {
                    "anchor_id": anchor_id,
                    "clip_id": clip_id,
                    "anchor_ns": anchor_ns,
                    "camera_name": camera_name,
                    "track_id": track_id,
                    "actor_class": str(actor.get("label_class", "")),
                    "source_triangle_count": len(prepared),
                    "maximum_boundary_extent_px": extent_limit,
                    "maximum_depth": maximum_depth,
                    "accepted_inside_triangle_count": accepted,
                    "boundary_approximated_triangle_count": approximated,
                    "boundary_depth_limited_triangle_count": depth_limited,
                    "boundary_unmeasurable_triangle_count": unmeasurable,
                    "rejected_outside_triangle_count": rejected,
                    "leaf_triangle_count": leaves,
                    "leaf_growth_ratio": leaves / len(prepared),
                    "maximum_depth_reached": deepest,
                    "maximum_approximated_extent_px": max(approximation_extents, default=None),
                    "maximum_depth_limited_extent_px": max(depth_limited_extents, default=None),
                }
                rows.append(row)
                print(
                    f"  extent={extent_limit:.1f}px depth={maximum_depth} "
                    f"accepted={accepted} approximated={approximated} "
                    f"depth_limited={depth_limited} unmeasurable={unmeasurable} "
                    f"rejected={rejected} leaves={leaves} deepest={deepest}"
                )

    aggregate = []
    for extent_limit in BOUNDARY_EXTENTS_PX:
        for maximum_depth in MAXIMUM_DEPTHS:
            group = [
                row for row in rows
                if row["maximum_boundary_extent_px"] == extent_limit
                and row["maximum_depth"] == maximum_depth
            ]
            aggregate.append({
                "maximum_boundary_extent_px": extent_limit,
                "maximum_depth": maximum_depth,
                "case_count": len(group),
                "accepted_inside_triangle_count": sum(row["accepted_inside_triangle_count"] for row in group),
                "boundary_approximated_triangle_count": sum(row["boundary_approximated_triangle_count"] for row in group),
                "boundary_depth_limited_triangle_count": sum(row["boundary_depth_limited_triangle_count"] for row in group),
                "boundary_unmeasurable_triangle_count": sum(row["boundary_unmeasurable_triangle_count"] for row in group),
                "rejected_outside_triangle_count": sum(row["rejected_outside_triangle_count"] for row in group),
                "leaf_triangle_count": sum(row["leaf_triangle_count"] for row in group),
                "cases_with_depth_limited": sum(row["boundary_depth_limited_triangle_count"] > 0 for row in group),
                "cases_with_unmeasurable": sum(row["boundary_unmeasurable_triangle_count"] > 0 for row in group),
                "maximum_case_leaf_count": max((row["leaf_triangle_count"] for row in group), default=0),
                "maximum_depth_reached": max((row["maximum_depth_reached"] for row in group), default=0),
            })

    report = {
        "case_count": len(BOUNDARY_CASES),
        "boundary_extents_px": list(BOUNDARY_EXTENTS_PX),
        "maximum_depths": list(MAXIMUM_DEPTHS),
        "near_plane_m": NEAR_PLANE_M,
        "row_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "rows": rows,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\nAggregate")
    for item in aggregate:
        print(
            f"  extent={item['maximum_boundary_extent_px']:.1f}px "
            f"depth={item['maximum_depth']} cases={item['case_count']} "
            f"approximated={item['boundary_approximated_triangle_count']} "
            f"depth_limited={item['boundary_depth_limited_triangle_count']} "
            f"unmeasurable={item['boundary_unmeasurable_triangle_count']} "
            f"leaves={item['leaf_triangle_count']} "
            f"max_case_leaves={item['maximum_case_leaf_count']} "
            f"deepest={item['maximum_depth_reached']}"
        )
    print("Failures:", len(failures))
    print("Output:", args.output)

    expected_rows = len(BOUNDARY_CASES) * len(BOUNDARY_EXTENTS_PX) * len(MAXIMUM_DEPTHS)
    if failures:
        raise RuntimeError("Real FOV pixel-stop profiling had processing failures")
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(rows)}")
    print("PASS: real FOV pixel-scale stopping profile completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
