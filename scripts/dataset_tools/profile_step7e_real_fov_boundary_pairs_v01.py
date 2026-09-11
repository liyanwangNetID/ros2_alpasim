#!/usr/bin/env python3
"""Profile angular-FOV recursive subdivision on eight known real boundary pairs.

Diagnostic only. Reads real Keyframe data, prepares camera-facing Box surfaces,
and reports FOV classification complexity for maximum depths 2, 4, 6, 8, 10.
Writes only /tmp output and does not modify annotations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from actor_box_projection_v01 import actor_box_corners_in_rig
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT
from triangle_angular_fov_subdivision_v01 import subdivide_triangle_to_angular_fov

NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTHS = (2, 4, 6, 8, 10)

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
        default=Path("/tmp/step7e_real_fov_boundary_subdivision_profile.json"),
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

        corners_rig = actor_box_corners_in_rig(
            actor,
            recorded_ego_message=ego.message,
        )
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

        for maximum_depth in MAXIMUM_DEPTHS:
            accepted_count = 0
            rejected_count = 0
            unresolved_count = 0
            deepest = 0
            source_with_unresolved = 0
            unresolved_inside_counts: Counter[int] = Counter()
            unresolved_edge_counts: Counter[int] = Counter()

            for item in prepared:
                result = subdivide_triangle_to_angular_fov(
                    item.vertices_camera,
                    max_angle_rad=calibration.max_angle_rad,
                    maximum_depth=maximum_depth,
                )
                accepted_count += len(result.accepted_inside_triangles)
                rejected_count += result.rejected_outside_triangle_count
                unresolved_count += len(result.boundary_unresolved_triangles)
                deepest = max(deepest, result.maximum_depth_reached)
                source_with_unresolved += int(result.stopped_by_depth_limit)
                unresolved_inside_counts.update(
                    unresolved.sample_inside_count
                    for unresolved in result.boundary_unresolved_triangles
                )
                unresolved_edge_counts.update(
                    unresolved.edge_intersection_count
                    for unresolved in result.boundary_unresolved_triangles
                )

            leaf_count = accepted_count + rejected_count + unresolved_count
            row = {
                "anchor_id": anchor_id,
                "clip_id": clip_id,
                "anchor_ns": anchor_ns,
                "camera_name": camera_name,
                "track_id": track_id,
                "actor_class": str(actor.get("label_class", "")),
                "source_triangle_count": len(prepared),
                "maximum_depth": maximum_depth,
                "accepted_inside_triangle_count": accepted_count,
                "rejected_outside_triangle_count": rejected_count,
                "boundary_unresolved_triangle_count": unresolved_count,
                "leaf_triangle_count": leaf_count,
                "leaf_growth_ratio": leaf_count / len(prepared),
                "maximum_depth_reached": deepest,
                "source_triangles_with_unresolved": source_with_unresolved,
                "unresolved_sample_inside_count_distribution": dict(
                    sorted(unresolved_inside_counts.items())
                ),
                "unresolved_edge_intersection_count_distribution": dict(
                    sorted(unresolved_edge_counts.items())
                ),
            }
            rows.append(row)
            print(
                f"  depth={maximum_depth} accepted={accepted_count} "
                f"rejected={rejected_count} unresolved={unresolved_count} "
                f"leaves={leaf_count} growth={row['leaf_growth_ratio']:.2f} "
                f"source_unresolved={source_with_unresolved}"
            )

    aggregate = []
    for maximum_depth in MAXIMUM_DEPTHS:
        group = [row for row in rows if row["maximum_depth"] == maximum_depth]
        aggregate.append({
            "maximum_depth": maximum_depth,
            "case_count": len(group),
            "accepted_inside_triangle_count": sum(
                row["accepted_inside_triangle_count"] for row in group
            ),
            "rejected_outside_triangle_count": sum(
                row["rejected_outside_triangle_count"] for row in group
            ),
            "boundary_unresolved_triangle_count": sum(
                row["boundary_unresolved_triangle_count"] for row in group
            ),
            "leaf_triangle_count": sum(row["leaf_triangle_count"] for row in group),
            "cases_with_unresolved": sum(
                row["boundary_unresolved_triangle_count"] > 0 for row in group
            ),
            "maximum_case_leaf_count": max(
                (row["leaf_triangle_count"] for row in group), default=0
            ),
        })

    report = {
        "case_count": len(BOUNDARY_CASES),
        "maximum_depths": list(MAXIMUM_DEPTHS),
        "near_plane_m": NEAR_PLANE_M,
        "row_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "rows": rows,
        "aggregate_by_depth": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("Aggregate by depth")
    for item in aggregate:
        print(
            f"  depth={item['maximum_depth']} cases={item['case_count']} "
            f"accepted={item['accepted_inside_triangle_count']} "
            f"rejected={item['rejected_outside_triangle_count']} "
            f"unresolved={item['boundary_unresolved_triangle_count']} "
            f"cases_with_unresolved={item['cases_with_unresolved']} "
            f"leaves={item['leaf_triangle_count']} "
            f"max_case_leaves={item['maximum_case_leaf_count']}"
        )
    print("Failures:", len(failures))
    print("Output:", args.output)

    if failures:
        raise RuntimeError("Real FOV boundary profiling had processing failures")
    if len(rows) != len(BOUNDARY_CASES) * len(MAXIMUM_DEPTHS):
        raise RuntimeError("Profile row count did not close")
    print("PASS: eight real FOV boundary pairs profiled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
