#!/usr/bin/env python3
"""Step 7E.4 export one row per valid Actor/camera projection.

The exporter traverses all selected Keyframes and calls the Step 7D production
projection API with its default adaptive policy. It writes valid geometric
projection evidence only. No observability classification, distance threshold,
or Actor-to-Actor occlusion estimate is applied.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from step7.actor_box_image_projection_v01 import project_actor_box_to_camera
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from step7.scene_fact_schema_v01 import CAMERA_NAMES

KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
OUTPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
SUMMARY_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.summary.json"
SCHEMA_VERSION = "step7e-projection-evidence-v01"


def read_keyframes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
            records.append(record)
    if not records:
        raise RuntimeError(f"No Keyframes found in {KEYFRAME_PATH}")
    return records


def point_list(points) -> list[dict[str, float]]:
    return [
        {"u": float(point.u), "v": float(point.v)}
        for point in points
    ]


def bbox_dict(bbox) -> dict[str, float]:
    return {
        "min_u": float(bbox.min_u),
        "min_v": float(bbox.min_v),
        "max_u": float(bbox.max_u),
        "max_v": float(bbox.max_v),
    }


def evidence_record(
    *,
    keyframe: dict[str, Any],
    camera_name: str,
    actor: dict[str, Any],
    projection,
) -> dict[str, Any]:
    if projection.projected_bbox is None or projection.clipped_bbox is None:
        raise RuntimeError("Valid projection is missing a 2D bounding box")
    if projection.minimum_depth_m is None or projection.maximum_depth_m is None:
        raise RuntimeError("Valid projection is missing camera-depth evidence")

    record = {
        "schema_version": SCHEMA_VERSION,
        "anchor_id": str(keyframe["anchor_id"]),
        "clip_id": str(keyframe["clip_id"]),
        "anchor_ns": int(keyframe["anchor_ns"]),
        "camera_name": camera_name,
        "track_id": str(actor["track_id"]),
        "label_class": str(actor["label_class"]),
        "is_static": bool(actor["is_static"]),
        "projection_valid": True,
        "sampling_mode": "adaptive",
        "camera_sample_count": int(projection.camera_sample_count),
        "projected_bbox": bbox_dict(projection.projected_bbox),
        "clipped_bbox": bbox_dict(projection.clipped_bbox),
        "projected_hull": point_list(projection.projected_hull),
        "clipped_hull": point_list(projection.clipped_hull),
        "projected_area_px": float(projection.projected_area_px),
        "inside_image_area_px": float(projection.inside_image_area_px),
        "inside_image_ratio": float(projection.inside_image_ratio),
        "projected_hull_area_px": float(projection.projected_hull_area_px),
        "inside_image_hull_area_px": float(
            projection.inside_image_hull_area_px
        ),
        "inside_image_hull_ratio": float(
            projection.inside_image_hull_ratio
        ),
        "projected_height_px": float(projection.projected_height_px),
        "minimum_depth_m": float(projection.minimum_depth_m),
        "maximum_depth_m": float(projection.maximum_depth_m),
        "truncated": bool(projection.truncated),
    }

    numeric_fields = (
        "projected_area_px",
        "inside_image_area_px",
        "inside_image_ratio",
        "projected_hull_area_px",
        "inside_image_hull_area_px",
        "inside_image_hull_ratio",
        "projected_height_px",
        "minimum_depth_m",
        "maximum_depth_m",
    )
    if not all(math.isfinite(record[field]) for field in numeric_fields):
        raise RuntimeError("Evidence record contains a non-finite number")
    if record["inside_image_hull_area_px"] <= 0.0:
        raise RuntimeError("Evidence record has non-positive inside Hull area")
    if not 0.0 < record["inside_image_hull_ratio"] <= 1.0 + 1e-9:
        raise RuntimeError("Evidence record has invalid inside Hull ratio")
    if len(record["projected_hull"]) < 3 or len(record["clipped_hull"]) < 3:
        raise RuntimeError("Evidence record has a degenerate Hull")
    return record


def main() -> int:
    started = time.monotonic()
    keyframes = read_keyframes()
    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")

    current_clip_id: str | None = None
    reader: DrivingClipReader | None = None
    calibrations = {}

    actor_count = 0
    pair_count = 0
    valid_count = 0
    invalid_reasons: Counter[str] = Counter()
    rows_by_camera: Counter[str] = Counter()
    rows_by_class: Counter[str] = Counter()
    rows_by_camera_class: Counter[str] = Counter()
    truncated_count = 0
    duplicate_keys: list[tuple[str, str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    processing_failures: list[dict[str, str]] = []

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with temporary_path.open("w", encoding="utf-8") as output:
        for index, keyframe in enumerate(keyframes, start=1):
            clip_id = str(keyframe["clip_id"])
            anchor_id = str(keyframe["anchor_id"])
            anchor_ns = int(keyframe["anchor_ns"])

            try:
                if clip_id != current_clip_id:
                    current_clip_id = clip_id
                    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
                    calibrations = {}
                    for camera_name in CAMERA_NAMES:
                        exact = reader.camera_indexes[camera_name].exact(anchor_ns)
                        if exact is None:
                            raise RuntimeError(
                                f"No exact camera frame for {camera_name}"
                            )
                        frame = exact.value
                        calibrations[camera_name] = load_camera_calibration(
                            reader.clip_directory
                            / "calibration"
                            / f"{camera_name}.json",
                            camera_name=camera_name,
                            source_width=frame.width,
                            source_height=frame.height,
                        )

                assert reader is not None
                ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
                snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
                if ego is None or ego.stamp_ns != anchor_ns:
                    raise RuntimeError("Exact recorded Ego state unavailable")
                if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
                    raise RuntimeError("Exact Actor snapshot unavailable")
                actors = snapshots[0].message.get("actors")
                if not isinstance(actors, list):
                    raise RuntimeError("Actor snapshot has no actors list")
                actor_count += len(actors)

                for camera_name in CAMERA_NAMES:
                    calibration = calibrations[camera_name]
                    for actor in actors:
                        pair_count += 1
                        projection = project_actor_box_to_camera(
                            actor,
                            recorded_ego_message=ego.message,
                            calibration=calibration,
                        )
                        if projection.edge_samples_per_edge != 0:
                            raise RuntimeError(
                                "Production projection did not use adaptive sampling"
                            )
                        if not projection.projection_valid:
                            invalid_reasons[
                                projection.failure_reason
                                or "missing_failure_reason"
                            ] += 1
                            continue

                        record = evidence_record(
                            keyframe=keyframe,
                            camera_name=camera_name,
                            actor=actor,
                            projection=projection,
                        )
                        unique_key = (
                            anchor_id,
                            camera_name,
                            str(actor["track_id"]),
                        )
                        if unique_key in seen_keys:
                            duplicate_keys.append(unique_key)
                        seen_keys.add(unique_key)

                        output.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        valid_count += 1
                        rows_by_camera[camera_name] += 1
                        rows_by_class[str(actor["label_class"])] += 1
                        rows_by_camera_class[
                            f"{camera_name}|{actor['label_class']}"
                        ] += 1
                        truncated_count += int(projection.truncated)

            except Exception as exc:
                processing_failures.append(
                    {
                        "anchor_id": anchor_id,
                        "clip_id": clip_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

            if index % 100 == 0 or index == len(keyframes):
                print(
                    f"Progress: {index}/{len(keyframes)} "
                    f"pairs={pair_count} rows={valid_count} "
                    f"elapsed_sec={time.monotonic() - started:.1f}",
                    flush=True,
                )

    if processing_failures or duplicate_keys:
        temporary_path.unlink(missing_ok=True)
        if processing_failures:
            print("First processing failures:")
            for failure in processing_failures[:20]:
                print(json.dumps(failure, ensure_ascii=False))
        if duplicate_keys:
            print("First duplicate keys:", duplicate_keys[:20])
        raise RuntimeError("Evidence export validation failed; temporary file removed")

    if pair_count != actor_count * len(CAMERA_NAMES):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Actor/camera pair count does not close")
    if valid_count + sum(invalid_reasons.values()) != pair_count:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Valid and invalid counts do not close")
    if "missing_failure_reason" in invalid_reasons:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("An invalid projection has no failure reason")

    os.replace(temporary_path, OUTPUT_PATH)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Valid Step 7D geometric projection evidence only. No "
            "observability classification or Actor-to-Actor occlusion."
        ),
        "source": {
            "keyframes_path": str(KEYFRAME_PATH),
            "keyframe_count": len(keyframes),
            "actor_count": actor_count,
            "actor_camera_pair_count": pair_count,
        },
        "output_path": str(OUTPUT_PATH),
        "valid_evidence_row_count": valid_count,
        "invalid_projection_count": sum(invalid_reasons.values()),
        "invalid_projection_reasons": dict(sorted(invalid_reasons.items())),
        "truncated_valid_row_count": truncated_count,
        "rows_by_camera": dict(sorted(rows_by_camera.items())),
        "rows_by_actor_class": dict(sorted(rows_by_class.items())),
        "rows_by_camera_and_actor_class": dict(
            sorted(rows_by_camera_class.items())
        ),
        "projection_policy": {
            "sampling": "adaptive",
            "maximum_chord_error_px": 1.0,
            "maximum_adaptive_depth": 14,
            "near_plane_m": 1e-3,
        },
        "processing_failure_count": 0,
        "duplicate_key_count": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("Keyframes:", len(keyframes))
    print("Actors:", actor_count)
    print("Actor/camera pairs:", pair_count)
    print("Evidence rows:", valid_count)
    print("Invalid projections:", sum(invalid_reasons.values()))
    print("Invalid reasons:", dict(sorted(invalid_reasons.items())))
    print("Truncated valid rows:", truncated_count)
    print("Rows by camera:", dict(sorted(rows_by_camera.items())))
    print("Duplicate keys:", len(duplicate_keys))
    print("Processing failures:", len(processing_failures))
    print("Evidence output:", OUTPUT_PATH)
    print("Summary output:", SUMMARY_PATH)
    print("PASS: Step 7E projection evidence exported atomically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
