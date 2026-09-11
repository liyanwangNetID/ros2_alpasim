#!/usr/bin/env python3
"""Step 7E.20 inspect F-theta vertex projections for one real Actor.

The script selects one real Keyframe Actor with at least one geometric candidate
camera, prepares camera-facing near-clipped box triangles, projects every
triangle vertex with the production F-theta calibration, validates invariants,
and writes a diagnostic JSON file under /tmp.

It does not rasterize triangle interiors, compare depths, or modify dataset
artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from actor_box_image_projection_v01 import project_actor_box_to_camera
from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_observability_rules_v01 import evaluate_geometric_observability
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from ftheta_triangle_vertex_projection_v01 import (
    project_ftheta_triangle_vertices,
)
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from scene_fact_schema_v01 import CAMERA_NAMES

KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
DEFAULT_OUTPUT = Path(
    "/tmp/step7e_single_real_actor_ftheta_triangle_vertices.json"
)
NEAR_PLANE_M = 1e-3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-id",
        help="Anchor to inspect. The first Keyframe is used when omitted.",
    )
    parser.add_argument(
        "--track-id",
        help="Actor to inspect. A suitable Actor is selected when omitted.",
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


def inspect_actor(actor, ego_message, calibrations):
    corners_rig = actor_box_corners_in_rig(
        actor,
        recorded_ego_message=ego_message,
    )
    if len(corners_rig) != 8:
        raise RuntimeError("Expected exactly eight rig-frame corners")

    camera_records = []
    has_candidate_camera = False
    candidate_without_projected_triangles = []

    for camera_name in CAMERA_NAMES:
        calibration = calibrations[camera_name]
        corners_camera = tuple(
            calibration.rig_point_to_camera(corner)
            for corner in corners_rig
        )
        prepared = prepare_camera_facing_box_triangles(
            corners_camera,
            near_plane_m=NEAR_PLANE_M,
        )
        projection = project_actor_box_to_camera(
            actor,
            recorded_ego_message=ego_message,
            calibration=calibration,
            near_plane_m=NEAR_PLANE_M,
        )

        decision = None
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
            has_candidate_camera |= decision.candidate

        triangle_records = []
        vertex_flag_counts: Counter[str] = Counter()
        face_counts: Counter[str] = Counter()

        for triangle_index, triangle in enumerate(prepared):
            projected = project_ftheta_triangle_vertices(
                triangle.vertices_camera,
                calibration=calibration,
                near_plane_m=NEAR_PLANE_M,
            )
            if len(projected.vertices) != 3:
                raise RuntimeError("Projected triangle does not contain three vertices")

            for source_vertex, projected_vertex in zip(
                triangle.vertices_camera,
                projected.vertices,
            ):
                if projected_vertex.camera_x_m != float(source_vertex.x):
                    raise RuntimeError("Projected vertex x/order mismatch")
                if projected_vertex.camera_y_m != float(source_vertex.y):
                    raise RuntimeError("Projected vertex y/order mismatch")
                if projected_vertex.camera_z_m != float(source_vertex.z):
                    raise RuntimeError("Projected vertex z/order mismatch")
                if projected_vertex.camera_z_m < NEAR_PLANE_M:
                    raise RuntimeError("Projected vertex is behind near plane")
                if not math.isfinite(projected_vertex.u_px):
                    raise RuntimeError("Non-finite projected u coordinate")
                if not math.isfinite(projected_vertex.v_px):
                    raise RuntimeError("Non-finite projected v coordinate")
                vertex_flag_counts[
                    f"within_fov={projected_vertex.within_fov}"
                ] += 1
                vertex_flag_counts[
                    f"inside_image={projected_vertex.inside_image}"
                ] += 1

            face_counts[triangle.face_name] += 1
            triangle_records.append(
                {
                    "triangle_index": triangle_index,
                    "face_name": triangle.face_name,
                    "source_corner_indices": list(
                        triangle.source_corner_indices
                    ),
                    "projection": projected.to_dict(),
                }
            )

        candidate = bool(decision is not None and decision.candidate)
        if candidate and not triangle_records:
            candidate_without_projected_triangles.append(camera_name)

        camera_records.append(
            {
                "camera_name": camera_name,
                "projection_valid": projection.projection_valid,
                "projection_failure_reason": projection.failure_reason,
                "geometric_observability_candidate": (
                    None if decision is None else decision.candidate
                ),
                "observability_failure_reason": (
                    None if decision is None else decision.failure_reason
                ),
                "prepared_triangle_count": len(prepared),
                "projected_triangle_count": len(triangle_records),
                "projected_vertex_count": len(triangle_records) * 3,
                "face_triangle_counts": dict(sorted(face_counts.items())),
                "vertex_flag_counts": dict(
                    sorted(vertex_flag_counts.items())
                ),
                "triangles": triangle_records,
            }
        )

    return (
        camera_records,
        has_candidate_camera,
        candidate_without_projected_triangles,
    )


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
    selected_missing = None

    if arguments.track_id is not None:
        matches = [
            actor
            for actor in actors
            if str(actor["track_id"]) == arguments.track_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one Actor for track_id={arguments.track_id!r}"
            )
        selected_actor = matches[0]
        selected_records, has_candidate, selected_missing = inspect_actor(
            selected_actor,
            ego.message,
            calibrations,
        )
        if not has_candidate:
            raise RuntimeError("Selected Actor has no candidate camera")
    else:
        for actor in actors:
            records, has_candidate, missing = inspect_actor(
                actor,
                ego.message,
                calibrations,
            )
            if has_candidate:
                selected_actor = actor
                selected_records = records
                selected_missing = missing
                break

    if selected_actor is None or selected_records is None:
        raise RuntimeError("No Actor with a geometric candidate camera found")
    if selected_missing:
        raise RuntimeError(
            "Candidate cameras without projected triangles: "
            + ", ".join(selected_missing)
        )
    if [item["camera_name"] for item in selected_records] != list(CAMERA_NAMES):
        raise RuntimeError("Camera output order is not canonical")

    total_triangles = sum(
        item["projected_triangle_count"] for item in selected_records
    )
    total_vertices = sum(
        item["projected_vertex_count"] for item in selected_records
    )
    if total_vertices != total_triangles * 3:
        raise RuntimeError("Projected vertex count does not close")

    output = {
        "schema_version": (
            "step7e-single-real-actor-ftheta-triangle-vertices-v01"
        ),
        "description": (
            "Diagnostic only. Vertex projection is not triangle rasterization "
            "and is not an occlusion decision."
        ),
        "anchor_id": anchor_id,
        "clip_id": clip_id,
        "anchor_ns": anchor_ns,
        "track_id": str(selected_actor["track_id"]),
        "actor_class": str(selected_actor["label_class"]),
        "near_plane_m": NEAR_PLANE_M,
        "projected_triangle_count": total_triangles,
        "projected_vertex_count": total_vertices,
        "camera_triangle_projections": selected_records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
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
            f"projection_valid={item['projection_valid']} "
            f"candidate={item['geometric_observability_candidate']} "
            f"triangles={item['projected_triangle_count']} "
            f"vertices={item['projected_vertex_count']} "
            f"faces={item['face_triangle_counts']} "
            f"flags={item['vertex_flag_counts']} "
            f"failure={item['projection_failure_reason'] or item['observability_failure_reason']}"
        )
    print("Total projected triangles:", total_triangles)
    print("Total projected vertices:", total_vertices)
    print("Diagnostic output:", arguments.output)
    print("PASS: single real Actor F-theta triangle vertices validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
