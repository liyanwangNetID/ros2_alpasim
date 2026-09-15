#!/usr/bin/env python3
"""Step 7E.16 export full Actor observability intermediate records.

One JSONL row is written per Keyframe Actor. Each row contains four per-camera
projection/observability diagnostics and the Actor-level pre-occlusion status.
The output is intermediate evidence, not final Scene-Fact supervision.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from step7.actor_box_image_projection_v01 import project_actor_box_to_camera
from step7.actor_observability_v01 import aggregate_actor_observability
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT, INTERMEDIATE_ROOT, REPORT_ROOT
from step7.scene_fact_schema_v01 import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION

KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
OUTPUT_PATH = INTERMEDIATE_ROOT / "actor_observability_v0.1.jsonl"
SUMMARY_PATH = REPORT_ROOT / "actor_observability_summary_v0.1.json"
SCHEMA_VERSION = "actor-observability-intermediate-v0.1"


def read_keyframes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with KEYFRAME_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("anchor_id", "clip_id", "anchor_ns"):
                if field not in record:
                    raise ValueError(f"{KEYFRAME_PATH}:{line_number}: missing {field}")
            records.append(record)
    if not records:
        raise RuntimeError(f"No Keyframes found in {KEYFRAME_PATH}")
    return records


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


def projection_record(projection, camera_result) -> dict[str, Any]:
    return {
        "camera_name": projection.camera_name,
        "projection_valid": projection.projection_valid,
        "projection_failure_reason": projection.failure_reason,
        "geometric_observability_candidate": (
            camera_result.geometric_observability_candidate
        ),
        "observability_failure_reason": camera_result.failure_reason,
        "inside_image_hull_area_px": float(projection.inside_image_hull_area_px),
        "projected_height_px": float(projection.projected_height_px),
        "inside_image_hull_ratio": float(projection.inside_image_hull_ratio),
        "minimum_depth_m": (
            None if projection.minimum_depth_m is None
            else float(projection.minimum_depth_m)
        ),
        "maximum_depth_m": (
            None if projection.maximum_depth_m is None
            else float(projection.maximum_depth_m)
        ),
        "truncated": bool(projection.truncated),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    started = time.monotonic()
    keyframes = read_keyframes()
    INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")

    current_clip_id: str | None = None
    reader: DrivingClipReader | None = None
    calibrations = {}
    actor_count = 0
    projection_count = 0
    status_counts: Counter[str] = Counter()
    visible_camera_counts: Counter[str] = Counter()
    projection_failure_counts: Counter[str] = Counter()
    observability_failure_counts: Counter[str] = Counter()
    visible_camera_count_distribution: Counter[int] = Counter()
    duplicate_keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    processing_failures: list[dict[str, str]] = []

    with temporary_path.open("w", encoding="utf-8") as output:
        for index, keyframe in enumerate(keyframes, start=1):
            anchor_id = str(keyframe["anchor_id"])
            clip_id = str(keyframe["clip_id"])
            anchor_ns = int(keyframe["anchor_ns"])
            try:
                if clip_id != current_clip_id:
                    current_clip_id = clip_id
                    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
                    calibrations = load_calibrations(reader, anchor_ns)

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

                for actor in actors:
                    track_id = str(actor["track_id"])
                    unique_key = (anchor_id, track_id)
                    if unique_key in seen_keys:
                        duplicate_keys.append(unique_key)
                    seen_keys.add(unique_key)

                    projections = {
                        camera_name: project_actor_box_to_camera(
                            actor,
                            recorded_ego_message=ego.message,
                            calibration=calibrations[camera_name],
                        )
                        for camera_name in CAMERA_NAMES
                    }
                    aggregated = aggregate_actor_observability(projections)
                    camera_results = {
                        item.camera_name: item
                        for item in aggregated.camera_observability
                    }
                    camera_records = [
                        projection_record(
                            projections[camera_name],
                            camera_results[camera_name],
                        )
                        for camera_name in CAMERA_NAMES
                    ]
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "observability_format_version": OBSERVABILITY_FORMAT_VERSION,
                        "anchor_id": anchor_id,
                        "clip_id": clip_id,
                        "anchor_ns": anchor_ns,
                        "track_id": aggregated.track_id,
                        "actor_class": aggregated.actor_class,
                        "observability_status": aggregated.observability_status,
                        "visible_in_cameras": list(aggregated.visible_in_cameras),
                        "actor_to_actor_occlusion_evaluated": False,
                        "static_occlusion_evaluated": False,
                        "camera_observability": camera_records,
                    }
                    output.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    actor_count += 1
                    projection_count += len(CAMERA_NAMES)
                    status_counts[aggregated.observability_status] += 1
                    visible_camera_counts.update(aggregated.visible_in_cameras)
                    visible_camera_count_distribution[
                        len(aggregated.visible_in_cameras)
                    ] += 1
                    for camera_record in camera_records:
                        camera_name = camera_record["camera_name"]
                        projection_reason = camera_record["projection_failure_reason"]
                        observability_reason = camera_record[
                            "observability_failure_reason"
                        ]
                        if projection_reason is not None:
                            projection_failure_counts[
                                f"{camera_name}|{projection_reason}"
                            ] += 1
                        elif observability_reason is not None:
                            observability_failure_counts[
                                f"{camera_name}|{observability_reason}"
                            ] += 1
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
                    f"Progress: {index}/{len(keyframes)} actors={actor_count} "
                    f"elapsed_sec={time.monotonic() - started:.1f}",
                    flush=True,
                )

    if processing_failures or duplicate_keys:
        temporary_path.unlink(missing_ok=True)
        for failure in processing_failures[:20]:
            print(json.dumps(failure, ensure_ascii=False))
        if duplicate_keys:
            print("First duplicate keys:", duplicate_keys[:20])
        raise RuntimeError("Export validation failed; temporary file removed")
    if actor_count != len(seen_keys):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Unique Actor key count does not close")
    if projection_count != actor_count * len(CAMERA_NAMES):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Actor/camera projection count does not close")
    if sum(status_counts.values()) != actor_count:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Observability status count does not close")
    if status_counts["candidate_visible"] != sum(
        count
        for camera_count, count in visible_camera_count_distribution.items()
        if camera_count > 0
    ):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Candidate-visible and camera-count distribution disagree")

    os.replace(temporary_path, OUTPUT_PATH)
    output_sha256 = sha256_file(OUTPUT_PATH)
    summary = {
        "schema_version": "actor-observability-summary-v0.1",
        "input_keyframes_path": str(KEYFRAME_PATH),
        "output_path": str(OUTPUT_PATH),
        "keyframe_count": len(keyframes),
        "actor_record_count": actor_count,
        "actor_camera_projection_count": projection_count,
        "observability_status_counts": dict(sorted(status_counts.items())),
        "visible_actor_counts_by_camera": dict(sorted(visible_camera_counts.items())),
        "visible_camera_count_distribution": {
            str(key): visible_camera_count_distribution[key]
            for key in sorted(visible_camera_count_distribution)
        },
        "projection_failure_counts": dict(sorted(projection_failure_counts.items())),
        "observability_failure_counts": dict(
            sorted(observability_failure_counts.items())
        ),
        "actor_to_actor_occlusion_evaluated": False,
        "static_occlusion_evaluated": False,
        "duplicate_key_count": 0,
        "processing_failure_count": 0,
        "output_sha256": output_sha256,
        "elapsed_seconds": time.monotonic() - started,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("Keyframes:", len(keyframes))
    print("Actor records:", actor_count)
    print("Actor/camera projections:", projection_count)
    print("Status counts:", dict(sorted(status_counts.items())))
    print("Visible Actor counts by camera:", dict(sorted(visible_camera_counts.items())))
    print(
        "Visible camera count distribution:",
        dict(sorted(visible_camera_count_distribution.items())),
    )
    print("Duplicate keys:", len(duplicate_keys))
    print("Processing failures:", len(processing_failures))
    print("Output SHA-256:", output_sha256)
    print("Intermediate output:", OUTPUT_PATH)
    print("Summary output:", SUMMARY_PATH)
    print("PASS: full Actor observability intermediate exported atomically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
