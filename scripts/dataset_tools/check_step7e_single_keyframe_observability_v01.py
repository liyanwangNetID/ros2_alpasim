#!/usr/bin/env python3
"""Step 7E.15 shadow-check Actor observability on one real Keyframe.

The script selects one Keyframe, computes four production camera projections for
every Actor, aggregates Actor-level geometric observability, validates closure
and ordering invariants, and writes one diagnostic JSON file under /tmp.

It does not modify Scene Facts or dataset annotation artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from actor_box_image_projection_v01 import project_actor_box_to_camera
from actor_observability_v01 import aggregate_actor_observability
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from scene_fact_schema_v01 import CAMERA_NAMES

KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
DEFAULT_OUTPUT_PATH = Path("/tmp/step7e_single_keyframe_observability_shadow.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-id",
        help="Select this anchor_id. The first Keyframe is used when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Diagnostic JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
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
        raise RuntimeError(f"No Keyframes found in {KEYFRAME_PATH}")
    return first


def load_calibrations(
    reader: DrivingClipReader,
    anchor_ns: int,
):
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


def main() -> int:
    arguments = parse_arguments()
    keyframe = select_keyframe(arguments.anchor_id)
    anchor_id = str(keyframe["anchor_id"])
    clip_id = str(keyframe["clip_id"])
    anchor_ns = int(keyframe["anchor_ns"])

    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    calibrations = load_calibrations(reader, anchor_ns)

    ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
    if ego is None or ego.stamp_ns != anchor_ns:
        raise RuntimeError("Exact recorded Ego state unavailable")

    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
        raise RuntimeError("Exact Actor snapshot unavailable")
    actors = snapshots[0].message.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError("Actor snapshot has no actors list")

    actor_records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    visible_camera_counts: Counter[str] = Counter()
    projection_valid_counts: Counter[str] = Counter()
    failure_reason_counts: Counter[str] = Counter()
    seen_track_ids: set[str] = set()

    for actor in actors:
        track_id = str(actor["track_id"])
        if track_id in seen_track_ids:
            raise RuntimeError(f"Duplicate track_id in Actor snapshot: {track_id}")
        seen_track_ids.add(track_id)

        projections = {}
        for camera_name in CAMERA_NAMES:
            projection = project_actor_box_to_camera(
                actor,
                recorded_ego_message=ego.message,
                calibration=calibrations[camera_name],
            )
            if projection.camera_name != camera_name:
                raise RuntimeError("Projection camera name mismatch")
            if projection.track_id != track_id:
                raise RuntimeError("Projection track_id mismatch")
            if projection.actor_class != str(actor["label_class"]):
                raise RuntimeError("Projection actor_class mismatch")
            projections[camera_name] = projection

        aggregated = aggregate_actor_observability(projections)
        if aggregated.track_id != track_id:
            raise RuntimeError("Aggregated track_id mismatch")
        if aggregated.actor_class != str(actor["label_class"]):
            raise RuntimeError("Aggregated actor_class mismatch")
        if tuple(
            camera
            for camera in CAMERA_NAMES
            if camera in aggregated.visible_in_cameras
        ) != aggregated.visible_in_cameras:
            raise RuntimeError("visible_in_cameras is not in canonical order")
        if aggregated.observability_status == "candidate_visible":
            if not aggregated.visible_in_cameras:
                raise RuntimeError("candidate_visible Actor has no visible camera")
        elif aggregated.observability_status == "not_visible":
            if aggregated.visible_in_cameras:
                raise RuntimeError("not_visible Actor has visible cameras")
        else:
            raise RuntimeError(
                f"Unexpected pre-occlusion status: {aggregated.observability_status}"
            )

        camera_projection_summary = []
        for camera_name in CAMERA_NAMES:
            projection = projections[camera_name]
            camera_result = next(
                item
                for item in aggregated.camera_observability
                if item.camera_name == camera_name
            )
            projection_valid_counts[
                f"{camera_name}|{projection.projection_valid}"
            ] += 1
            if camera_result.failure_reason is not None:
                failure_reason_counts[
                    f"{camera_name}|{camera_result.failure_reason}"
                ] += 1
            camera_projection_summary.append(
                {
                    "camera_name": camera_name,
                    "projection_valid": projection.projection_valid,
                    "geometric_observability_candidate": (
                        camera_result.geometric_observability_candidate
                    ),
                    "failure_reason": camera_result.failure_reason,
                    "inside_image_hull_area_px": (
                        projection.inside_image_hull_area_px
                    ),
                    "projected_height_px": projection.projected_height_px,
                    "inside_image_hull_ratio": (
                        projection.inside_image_hull_ratio
                    ),
                }
            )

        actor_record = aggregated.to_dict()
        actor_record["camera_projections"] = camera_projection_summary
        actor_records.append(actor_record)
        status_counts[aggregated.observability_status] += 1
        visible_camera_counts.update(aggregated.visible_in_cameras)

    actor_count = len(actors)
    if len(actor_records) != actor_count:
        raise RuntimeError("Actor output count does not close")
    if sum(status_counts.values()) != actor_count:
        raise RuntimeError("Actor status counts do not close")
    if sum(projection_valid_counts.values()) != actor_count * len(CAMERA_NAMES):
        raise RuntimeError("Actor/camera projection counts do not close")

    output = {
        "schema_version": "step7e-single-keyframe-observability-shadow-v01",
        "description": (
            "Single-Keyframe diagnostic only. Actor-to-Actor and static-scene "
            "occlusion were not evaluated."
        ),
        "anchor_id": anchor_id,
        "clip_id": clip_id,
        "anchor_ns": anchor_ns,
        "actor_count": actor_count,
        "actor_camera_projection_count": actor_count * len(CAMERA_NAMES),
        "observability_status_counts": dict(sorted(status_counts.items())),
        "visible_actor_counts_by_camera": dict(
            sorted(visible_camera_counts.items())
        ),
        "projection_valid_counts": dict(
            sorted(projection_valid_counts.items())
        ),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "actors": actor_records,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, arguments.output)

    print("Anchor ID:", anchor_id)
    print("Clip ID:", clip_id)
    print("Anchor ns:", anchor_ns)
    print("Actors:", actor_count)
    print("Actor/camera projections:", actor_count * len(CAMERA_NAMES))
    print("Status counts:", dict(sorted(status_counts.items())))
    print(
        "Visible Actor counts by camera:",
        dict(sorted(visible_camera_counts.items())),
    )
    print("Projection validity:", dict(sorted(projection_valid_counts.items())))
    print("Failure reasons:", dict(sorted(failure_reason_counts.items())))
    print("Diagnostic output:", arguments.output)
    print("PASS: single-Keyframe Actor observability shadow aggregation completed.")
    return 0


if __name__ == "__main__":
    import os

    raise SystemExit(main())
