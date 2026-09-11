#!/usr/bin/env python3
"""Step 7E.17 check camera-facing surfaces for one real Actor.

The script selects one real Keyframe Actor, transforms its eight box corners
from map to rig to each camera, prepares camera-facing near-clipped triangles,
and compares the result with the existing production ActorCameraProjection.
It writes diagnostics only under /tmp and modifies no dataset artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from actor_box_image_projection_v01 import project_actor_box_to_camera
from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_observability_rules_v01 import evaluate_geometric_observability
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from scene_fact_schema_v01 import CAMERA_NAMES

KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
DEFAULT_OUTPUT = Path("/tmp/step7e_single_actor_camera_surfaces.json")
NEAR_PLANE_M = 1e-3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-id")
    parser.add_argument("--track-id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def select_keyframe(anchor_id: str | None) -> dict[str, Any]:
    first = None
    with KEYFRAME_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
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
    result = {}
    for camera_name in CAMERA_NAMES:
        exact = reader.camera_indexes[camera_name].exact(anchor_ns)
        if exact is None:
            raise RuntimeError(f"No exact camera frame for {camera_name}")
        frame = exact.value
        result[camera_name] = load_camera_calibration(
            reader.clip_directory / "calibration" / f"{camera_name}.json",
            camera_name=camera_name,
            source_width=frame.width,
            source_height=frame.height,
        )
    return result


def surface_summary(actor, ego_message, calibrations):
    corners_rig = actor_box_corners_in_rig(
        actor,
        recorded_ego_message=ego_message,
    )
    if len(corners_rig) != 8:
        raise RuntimeError("Expected eight rig-frame corners")

    camera_records = []
    has_candidate_camera = False
    has_surface_camera = False

    for camera_name in CAMERA_NAMES:
        calibration = calibrations[camera_name]
        corners_camera = tuple(
            calibration.rig_point_to_camera(corner) for corner in corners_rig
        )
        surfaces = prepare_camera_facing_box_triangles(
            corners_camera,
            near_plane_m=NEAR_PLANE_M,
        )
        projection = project_actor_box_to_camera(
            actor,
            recorded_ego_message=ego_message,
            calibration=calibration,
            near_plane_m=NEAR_PLANE_M,
        )

        if any(
            vertex.z < NEAR_PLANE_M
            for triangle in surfaces
            for vertex in triangle.vertices_camera
        ):
            raise RuntimeError("Prepared surface contains a vertex behind near plane")

        decision = None
        if projection.projection_valid:
            decision = evaluate_geometric_observability(
                camera_name=camera_name,
                inside_image_hull_area_px=projection.inside_image_hull_area_px,
                projected_height_px=projection.projected_height_px,
                inside_image_hull_ratio=projection.inside_image_hull_ratio,
            )
            has_candidate_camera |= decision.candidate

        has_surface_camera |= bool(surfaces)
        face_counts: dict[str, int] = {}
        for triangle in surfaces:
            face_counts[triangle.face_name] = face_counts.get(triangle.face_name, 0) + 1

        all_surface_depths = [
            vertex.z
            for triangle in surfaces
            for vertex in triangle.vertices_camera
        ]
        camera_records.append(
            {
                "camera_name": camera_name,
                "camera_corner_depth_range_m": [
                    min(corner.z for corner in corners_camera),
                    max(corner.z for corner in corners_camera),
                ],
                "prepared_triangle_count": len(surfaces),
                "prepared_face_triangle_counts": dict(sorted(face_counts.items())),
                "prepared_surface_depth_range_m": (
                    None
                    if not all_surface_depths
                    else [min(all_surface_depths), max(all_surface_depths)]
                ),
                "projection_valid": projection.projection_valid,
                "projection_failure_reason": projection.failure_reason,
                "projection_minimum_depth_m": projection.minimum_depth_m,
                "projection_maximum_depth_m": projection.maximum_depth_m,
                "geometric_observability_candidate": (
                    None if decision is None else decision.candidate
                ),
                "observability_failure_reason": (
                    None if decision is None else decision.failure_reason
                ),
            }
        )

    return camera_records, has_candidate_camera, has_surface_camera


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
    if not isinstance(actors, list) or not actors:
        raise RuntimeError("No Actors at selected Keyframe")

    selected_actor = None
    selected_records = None
    if arguments.track_id is not None:
        matches = [a for a in actors if str(a["track_id"]) == arguments.track_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one Actor for track_id={arguments.track_id!r}")
        selected_actor = matches[0]
        selected_records, _, _ = surface_summary(
            selected_actor, ego.message, calibrations
        )
    else:
        for actor in actors:
            records, has_candidate, has_surfaces = surface_summary(
                actor, ego.message, calibrations
            )
            if has_candidate and has_surfaces:
                selected_actor = actor
                selected_records = records
                break

    if selected_actor is None or selected_records is None:
        raise RuntimeError("No Actor with candidate projection and prepared surfaces found")

    if [item["camera_name"] for item in selected_records] != list(CAMERA_NAMES):
        raise RuntimeError("Camera record order is not canonical")
    if not any(item["prepared_triangle_count"] > 0 for item in selected_records):
        raise RuntimeError("Selected Actor produced no prepared triangles")
    if not any(
        item["geometric_observability_candidate"] is True
        for item in selected_records
    ):
        raise RuntimeError("Selected Actor has no geometric candidate camera")

    output = {
        "schema_version": "step7e-single-real-actor-camera-surfaces-v01",
        "description": "Diagnostic only; no projection, observability, or Scene-Fact artifact was modified.",
        "anchor_id": anchor_id,
        "clip_id": clip_id,
        "anchor_ns": anchor_ns,
        "track_id": str(selected_actor["track_id"]),
        "actor_class": str(selected_actor["label_class"]),
        "near_plane_m": NEAR_PLANE_M,
        "camera_surfaces": selected_records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Anchor ID:", anchor_id)
    print("Clip ID:", clip_id)
    print("Anchor ns:", anchor_ns)
    print("Track ID:", output["track_id"])
    print("Actor class:", output["actor_class"])
    print("Near plane m:", NEAR_PLANE_M)
    for item in selected_records:
        print(
            f"{item['camera_name']}: "
            f"triangles={item['prepared_triangle_count']} "
            f"faces={item['prepared_face_triangle_counts']} "
            f"corner_depth={item['camera_corner_depth_range_m']} "
            f"surface_depth={item['prepared_surface_depth_range_m']} "
            f"projection_valid={item['projection_valid']} "
            f"candidate={item['geometric_observability_candidate']} "
            f"failure={item['projection_failure_reason'] or item['observability_failure_reason']}"
        )
    print("Diagnostic output:", arguments.output)
    print("PASS: single real Actor camera-facing surfaces validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
