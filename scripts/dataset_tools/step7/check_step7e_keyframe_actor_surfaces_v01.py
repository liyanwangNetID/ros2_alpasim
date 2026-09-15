#!/usr/bin/env python3
"""Step 7E.19 validate prepared box surfaces for one complete Keyframe.

For every Actor and each selected camera, this diagnostic compares the existing
production ActorCameraProjection with camera-facing, near-clipped box surfaces.
It reports combinations that need investigation before rasterization work.
No dataset artifact is modified; the detailed report is written under /tmp.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from step7.actor_box_image_projection_v01 import project_actor_box_to_camera
from step7.actor_box_projection_v01 import actor_box_corners_in_rig
from step7.actor_observability_rules_v01 import evaluate_geometric_observability
from step7.camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from step7.scene_fact_schema_v01 import CAMERA_NAMES

KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
DEFAULT_OUTPUT = Path("/tmp/step7e_keyframe_actor_surface_validation.json")
NEAR_PLANE_M = 1e-3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-id",
        help="Anchor to inspect. The first Keyframe is used when omitted.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def select_keyframe(anchor_id: str | None) -> dict[str, Any]:
    first: dict[str, Any] | None = None
    with KEYFRAME_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("anchor_id", "clip_id", "anchor_ns"):
                if field not in record:
                    raise ValueError(
                        f"{KEYFRAME_PATH}:{line_number}: missing {field}"
                    )
            if first is None:
                first = record
            if anchor_id is not None and str(record["anchor_id"]) == anchor_id:
                return record
    if anchor_id is not None:
        raise ValueError(f"anchor_id not found: {anchor_id}")
    if first is None:
        raise RuntimeError("No Keyframes found")
    return first


def load_calibrations(reader: DrivingClipReader, anchor_ns: int):
    calibrations = {}
    for camera_name in CAMERA_NAMES:
        exact = reader.camera_indexes[camera_name].exact(anchor_ns)
        if exact is None:
            raise RuntimeError(f"No exact camera frame for {camera_name}")
        frame = exact.value
        calibrations[camera_name] = load_camera_calibration(
            reader.clip_directory / "calibration" / f"{camera_name}.json",
            camera_name=camera_name,
            source_width=frame.width,
            source_height=frame.height,
        )
    return calibrations


def classify_combination(
    *,
    projection_valid: bool,
    candidate: bool,
    triangle_count: int,
    corner_minimum_depth: float,
    corner_maximum_depth: float,
) -> str:
    has_surfaces = triangle_count > 0
    if candidate and has_surfaces:
        return "candidate_with_surfaces"
    if candidate and not has_surfaces:
        return "candidate_without_surfaces"
    if projection_valid and has_surfaces:
        return "valid_non_candidate_with_surfaces"
    if projection_valid and not has_surfaces:
        return "valid_non_candidate_without_surfaces"
    if has_surfaces:
        return "invalid_projection_with_surfaces"
    if corner_maximum_depth < NEAR_PLANE_M:
        return "behind_near_plane_without_surfaces"
    if corner_minimum_depth < NEAR_PLANE_M <= corner_maximum_depth:
        return "near_plane_crossing_without_surfaces"
    return "in_front_without_surfaces"


def main() -> int:
    arguments = parse_arguments()
    keyframe = select_keyframe(arguments.anchor_id)
    anchor_id = str(keyframe["anchor_id"])
    clip_id = str(keyframe["clip_id"])
    anchor_ns = int(keyframe["anchor_ns"])

    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    calibrations = load_calibrations(reader, anchor_ns)
    ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if ego is None or ego.stamp_ns != anchor_ns:
        raise RuntimeError("Exact recorded Ego state unavailable")
    if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
        raise RuntimeError("Exact Actor snapshot unavailable")
    actors = snapshots[0].message.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError("Actor snapshot has no actors list")

    combination_counts: Counter[str] = Counter()
    combination_counts_by_camera: dict[str, Counter[str]] = {
        name: Counter() for name in CAMERA_NAMES
    }
    triangle_count_distribution: Counter[int] = Counter()
    candidate_counts: Counter[str] = Counter()
    projection_valid_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    seen_tracks: set[str] = set()

    for actor in actors:
        track_id = str(actor["track_id"])
        actor_class = str(actor["label_class"])
        if track_id in seen_tracks:
            raise RuntimeError(f"Duplicate track_id: {track_id}")
        seen_tracks.add(track_id)

        corners_rig = actor_box_corners_in_rig(
            actor,
            recorded_ego_message=ego.message,
        )
        if len(corners_rig) != 8:
            raise RuntimeError("Expected exactly eight rig-frame corners")

        for camera_name in CAMERA_NAMES:
            calibration = calibrations[camera_name]
            corners_camera = tuple(
                calibration.rig_point_to_camera(corner)
                for corner in corners_rig
            )
            surfaces = prepare_camera_facing_box_triangles(
                corners_camera,
                near_plane_m=NEAR_PLANE_M,
            )
            projection = project_actor_box_to_camera(
                actor,
                recorded_ego_message=ego.message,
                calibration=calibration,
                near_plane_m=NEAR_PLANE_M,
            )

            if projection.camera_name != camera_name:
                raise RuntimeError("Projection camera mismatch")
            if projection.track_id != track_id:
                raise RuntimeError("Projection track mismatch")
            if projection.actor_class != actor_class:
                raise RuntimeError("Projection class mismatch")
            if any(
                vertex.z < NEAR_PLANE_M
                for triangle in surfaces
                for vertex in triangle.vertices_camera
            ):
                raise RuntimeError("Prepared vertex is behind the near plane")

            candidate = False
            observability_failure = None
            if projection.projection_valid:
                decision = evaluate_geometric_observability(
                    camera_name=camera_name,
                    inside_image_hull_area_px=(
                        projection.inside_image_hull_area_px
                    ),
                    projected_height_px=projection.projected_height_px,
                    inside_image_hull_ratio=(
                        projection.inside_image_hull_ratio
                    ),
                )
                candidate = decision.candidate
                observability_failure = decision.failure_reason

            minimum_depth = min(corner.z for corner in corners_camera)
            maximum_depth = max(corner.z for corner in corners_camera)
            combination = classify_combination(
                projection_valid=projection.projection_valid,
                candidate=candidate,
                triangle_count=len(surfaces),
                corner_minimum_depth=minimum_depth,
                corner_maximum_depth=maximum_depth,
            )
            combination_counts[combination] += 1
            combination_counts_by_camera[camera_name][combination] += 1
            triangle_count_distribution[len(surfaces)] += 1
            candidate_counts[f"{camera_name}|{candidate}"] += 1
            projection_valid_counts[
                f"{camera_name}|{projection.projection_valid}"
            ] += 1

            if combination in {
                "candidate_without_surfaces",
                "valid_non_candidate_without_surfaces",
                "near_plane_crossing_without_surfaces",
                "in_front_without_surfaces",
            }:
                issues.append(
                    {
                        "anchor_id": anchor_id,
                        "track_id": track_id,
                        "actor_class": actor_class,
                        "camera_name": camera_name,
                        "combination": combination,
                        "corner_depth_range_m": [minimum_depth, maximum_depth],
                        "prepared_triangle_count": len(surfaces),
                        "projection_valid": projection.projection_valid,
                        "projection_failure_reason": projection.failure_reason,
                        "geometric_observability_candidate": candidate,
                        "observability_failure_reason": observability_failure,
                    }
                )

    actor_count = len(actors)
    combination_total = sum(combination_counts.values())
    expected_total = actor_count * len(CAMERA_NAMES)
    if combination_total != expected_total:
        raise RuntimeError("Combination count does not close")
    if sum(triangle_count_distribution.values()) != expected_total:
        raise RuntimeError("Triangle-count distribution does not close")
    for camera_name in CAMERA_NAMES:
        if sum(combination_counts_by_camera[camera_name].values()) != actor_count:
            raise RuntimeError(f"Camera count does not close: {camera_name}")

    output = {
        "schema_version": "step7e-keyframe-actor-surface-validation-v01",
        "description": (
            "Diagnostic only. F-theta rasterization and depth comparison are "
            "not implemented in this report."
        ),
        "anchor_id": anchor_id,
        "clip_id": clip_id,
        "anchor_ns": anchor_ns,
        "near_plane_m": NEAR_PLANE_M,
        "actor_count": actor_count,
        "actor_camera_count": expected_total,
        "combination_counts": dict(sorted(combination_counts.items())),
        "combination_counts_by_camera": {
            name: dict(sorted(combination_counts_by_camera[name].items()))
            for name in CAMERA_NAMES
        },
        "triangle_count_distribution": {
            str(key): triangle_count_distribution[key]
            for key in sorted(triangle_count_distribution)
        },
        "projection_valid_counts": dict(sorted(projection_valid_counts.items())),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "issue_count": len(issues),
        "issues": issues,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Anchor ID:", anchor_id)
    print("Clip ID:", clip_id)
    print("Anchor ns:", anchor_ns)
    print("Actors:", actor_count)
    print("Actor/camera combinations:", expected_total)
    print("Combination counts:", dict(sorted(combination_counts.items())))
    print(
        "Triangle count distribution:",
        dict(sorted(triangle_count_distribution.items())),
    )
    print("Projection validity:", dict(sorted(projection_valid_counts.items())))
    print("Candidate counts:", dict(sorted(candidate_counts.items())))
    print("Issue count:", len(issues))
    for issue in issues[:20]:
        print("ISSUE:", json.dumps(issue, ensure_ascii=False, sort_keys=True))
    print("Diagnostic output:", arguments.output)
    if any(
        issue["combination"] == "candidate_without_surfaces"
        for issue in issues
    ):
        raise RuntimeError(
            "At least one geometric candidate has no prepared surface triangles"
        )
    print("PASS: complete-Keyframe Actor surface validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
