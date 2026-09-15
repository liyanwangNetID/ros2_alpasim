#!/usr/bin/env python3
"""Profile real-data F-theta triangle subdivision complexity.

Step 7E-2B diagnostic only. This script reads deterministic sampled Keyframes,
selects geometrically observable Actor/camera pairs, prepares camera-facing Box
triangles, and compares adaptive triangle subdivision settings. It writes only
a JSON report under /tmp and does not modify dataset annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from step7.actor_box_projection_v01 import actor_box_corners_in_rig
from step7.actor_observability_rules_v01 import evaluate_geometric_observability
from step7.camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from step7.ftheta_triangle_subdivision_v01 import subdivide_ftheta_triangle_adaptive
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from step7.scene_fact_schema_v01 import CAMERA_NAMES

ERROR_LIMITS_PX = (2.0, 1.0, 0.5, 0.25)
MAXIMUM_DEPTHS = (4, 6, 8, 10)
NEAR_PLANE_M = 1e-3
DEFAULT_SAMPLE_KEYFRAMES = 24
DEFAULT_MAXIMUM_PAIRS_PER_CAMERA = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-keyframes", type=int, default=DEFAULT_SAMPLE_KEYFRAMES)
    parser.add_argument(
        "--maximum-pairs-per-camera",
        type=int,
        default=DEFAULT_MAXIMUM_PAIRS_PER_CAMERA,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/step7e_real_triangle_subdivision_profile.json"),
    )
    return parser.parse_args()


def load_keyframes() -> list[dict[str, Any]]:
    path = ANNOTATION_ROOT / "keyframes.jsonl"
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def deterministic_sample(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("sample-keyframes must be positive")
    if count >= len(records):
        return records
    indices = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
    return [records[index] for index in indices]


def exact_inputs(reader: DrivingClipReader, anchor_ns: int):
    ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
    if ego is None or ego.stamp_ns != anchor_ns:
        raise RuntimeError("Exact recorded Ego state unavailable")
    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
        raise RuntimeError("Exact Actor snapshot unavailable")
    return ego, snapshots[0].message["actors"]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round(fraction * (len(ordered) - 1)))
    return float(ordered[index])


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(max(values)),
    }


def main() -> int:
    args = parse_args()
    if args.maximum_pairs_per_camera <= 0:
        raise ValueError("maximum-pairs-per-camera must be positive")

    keyframes = deterministic_sample(load_keyframes(), args.sample_keyframes)
    selected_per_camera: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.perf_counter()

    for keyframe_index, keyframe in enumerate(keyframes, start=1):
        clip_id = str(keyframe["clip_id"])
        anchor_id = str(keyframe["anchor_id"])
        anchor_ns = int(keyframe["anchor_ns"])
        reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        ego, actors = exact_inputs(reader, anchor_ns)

        for camera_name in CAMERA_NAMES:
            if selected_per_camera[camera_name] >= args.maximum_pairs_per_camera:
                continue
            frame = reader.camera_indexes[camera_name].exact(anchor_ns)
            if frame is None:
                raise RuntimeError(f"Exact camera frame unavailable: {anchor_id} {camera_name}")
            calibration = load_camera_calibration(
                reader.clip_directory / "calibration" / f"{camera_name}.json",
                camera_name=camera_name,
                source_width=frame.value.width,
                source_height=frame.value.height,
            )

            for actor in actors:
                if selected_per_camera[camera_name] >= args.maximum_pairs_per_camera:
                    break

                corners_rig = actor_box_corners_in_rig(
                    actor,
                    recorded_ego_message=ego.message,
                )
                corners_camera = tuple(calibration.rig_point_to_camera(point) for point in corners_rig)

                # Reuse the production Box projection solely to choose current
                # geometric candidates. Import here to keep the profiling path explicit.
                from step7.actor_box_image_projection_v01 import project_actor_box_to_camera

                projection = project_actor_box_to_camera(
                    actor,
                    recorded_ego_message=ego.message,
                    calibration=calibration,
                )
                if not projection.projection_valid:
                    continue
                decision = evaluate_geometric_observability(
                    camera_name=camera_name,
                    inside_image_hull_area_px=projection.inside_image_hull_area_px,
                    projected_height_px=projection.projected_height_px,
                    inside_image_hull_ratio=projection.inside_image_hull_ratio,
                )
                if not decision.candidate:
                    continue

                prepared = prepare_camera_facing_box_triangles(
                    corners_camera,
                    near_plane_m=NEAR_PLANE_M,
                )
                if not prepared:
                    failures.append({
                        "anchor_id": anchor_id,
                        "camera_name": camera_name,
                        "track_id": str(actor["track_id"]),
                        "reason": "candidate_without_prepared_surfaces",
                    })
                    continue

                source_triangle_count = len(prepared)
                for error_limit in ERROR_LIMITS_PX:
                    for maximum_depth in MAXIMUM_DEPTHS:
                        leaf_count = 0
                        maximum_observed_error = 0.0
                        maximum_depth_reached = 0
                        depth_limited_source_triangles = 0

                        try:
                            for prepared_triangle in prepared:
                                result = subdivide_ftheta_triangle_adaptive(
                                    prepared_triangle.vertices_camera,
                                    calibration,
                                    maximum_projection_error_px=error_limit,
                                    maximum_depth=maximum_depth,
                                    near_plane_m=NEAR_PLANE_M,
                                )
                                leaf_count += len(result.triangles)
                                maximum_observed_error = max(
                                    maximum_observed_error,
                                    result.maximum_observed_error_px,
                                )
                                maximum_depth_reached = max(
                                    maximum_depth_reached,
                                    result.maximum_depth_reached,
                                )
                                depth_limited_source_triangles += int(
                                    result.stopped_by_depth_limit
                                )
                        except ValueError as exc:
                            failures.append({
                                "anchor_id": anchor_id,
                                "camera_name": camera_name,
                                "track_id": str(actor["track_id"]),
                                "error_limit_px": error_limit,
                                "maximum_depth": maximum_depth,
                                "reason": "subdivision_input_outside_fov",
                                "detail": str(exc),
                            })
                            break

                        rows.append({
                            "anchor_id": anchor_id,
                            "clip_id": clip_id,
                            "anchor_ns": anchor_ns,
                            "camera_name": camera_name,
                            "track_id": str(actor["track_id"]),
                            "actor_class": str(actor["label_class"]),
                            "minimum_depth_m": projection.minimum_depth_m,
                            "projected_height_px": projection.projected_height_px,
                            "source_triangle_count": source_triangle_count,
                            "error_limit_px": error_limit,
                            "maximum_depth": maximum_depth,
                            "leaf_triangle_count": leaf_count,
                            "leaf_growth_ratio": leaf_count / source_triangle_count,
                            "maximum_observed_error_px": maximum_observed_error,
                            "maximum_depth_reached": maximum_depth_reached,
                            "depth_limited_source_triangle_count": depth_limited_source_triangles,
                        })

                selected_per_camera[camera_name] += 1

        print(
            f"Progress: {keyframe_index}/{len(keyframes)} "
            f"selected={dict(selected_per_camera)} "
            f"elapsed_sec={time.perf_counter() - start:.1f}"
        )

    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["camera_name"], row["error_limit_px"], row["maximum_depth"])].append(row)

    summaries = []
    for (camera_name, error_limit, maximum_depth), group in sorted(grouped.items()):
        leaves = [float(item["leaf_triangle_count"]) for item in group]
        growth = [float(item["leaf_growth_ratio"]) for item in group]
        depths = [float(item["maximum_depth_reached"]) for item in group]
        limited_pairs = sum(
            int(item["depth_limited_source_triangle_count"] > 0)
            for item in group
        )
        summaries.append({
            "camera_name": camera_name,
            "error_limit_px": error_limit,
            "maximum_depth": maximum_depth,
            "pair_count": len(group),
            "leaf_triangle_count": summarize(leaves),
            "leaf_growth_ratio": summarize(growth),
            "maximum_depth_reached": summarize(depths),
            "depth_limited_pair_count": limited_pairs,
        })

    report = {
        "sampled_keyframe_count": len(keyframes),
        "selected_pairs_by_camera": dict(sorted(selected_per_camera.items())),
        "error_limits_px": list(ERROR_LIMITS_PX),
        "maximum_depths": list(MAXIMUM_DEPTHS),
        "near_plane_m": NEAR_PLANE_M,
        "row_count": len(rows),
        "failure_count": len(failures),
        "failures": failures[:100],
        "summaries": summaries,
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("Selected pairs by camera:", dict(sorted(selected_per_camera.items())))
    print("Rows:", len(rows))
    print("Failures:", len(failures))
    print()
    print("Compact summaries")
    for item in summaries:
        print(
            f"  {item['camera_name']} error={item['error_limit_px']:.2f}px "
            f"depth={item['maximum_depth']} pairs={item['pair_count']} "
            f"leaves median/p95/max="
            f"{item['leaf_triangle_count']['median']}/"
            f"{item['leaf_triangle_count']['p95']}/"
            f"{item['leaf_triangle_count']['max']} "
            f"growth p95={item['leaf_growth_ratio']['p95']} "
            f"limited={item['depth_limited_pair_count']}"
        )
    print("Output:", args.output)

    if failures:
        raise RuntimeError(
            "Real triangles crossed the angular FOV boundary; FOV surface "
            "clipping is required before subdivision. See the JSON report."
        )
    print("PASS: real F-theta triangle subdivision profile completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
