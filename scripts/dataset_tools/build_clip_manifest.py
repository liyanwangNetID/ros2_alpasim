#!/usr/bin/env python3
"""Build a read-only manifest for recorded AlpaSim clips.

This is a standalone, one-shot Python script. It is not a ROS 2 node and does
not require rclpy. It scans finalized test_clip_NNN directories, writes one
JSON object per clip to a JSONL manifest, and writes an aggregate JSON report.
It never modifies files inside a clip directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "0.1.0"
MANIFEST_VERSION = "0.1"
CLIP_PATTERN = re.compile(r"^test_clip_([0-9]+)$")
CAMERA_NAMES = (
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right",
)

REQUIRED_RELATIVE_PATHS = (
    "metadata.json",
    "validation.json",
    "ego/ego_state.jsonl",
    "ego/executed_path_points.jsonl",
    "ego/executed_path_final.json",
    "ego/ground_truth_future.jsonl",
    "ego/planner_output.jsonl",
    "ego/complete_recording_ground_truth.json",
    "actors/current.jsonl",
    "actors/history.jsonl",
    "actors/future.jsonl",
    "route/map_route.jsonl",
    "route/navigation_route_local.jsonl",
    "map/vector_map.json",
)

JSONL_RELATIVE_PATHS = (
    "ego/ego_state.jsonl",
    "ego/executed_path_points.jsonl",
    "ego/ground_truth_future.jsonl",
    "ego/planner_output.jsonl",
    "actors/current.jsonl",
    "actors/history.jsonl",
    "actors/future.jsonl",
    "route/map_route.jsonl",
    "route/navigation_route_local.jsonl",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "file_not_found"
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}:line={exc.lineno}:column={exc.colno}"
    except OSError as exc:
        return None, f"read_error:{exc}"

    if not isinstance(data, dict):
        return None, "json_root_is_not_object"
    return data, None


def count_nonempty_lines(path: Path) -> tuple[int | None, str | None]:
    try:
        count = 0
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    count += 1
        return count, None
    except FileNotFoundError:
        return None, "file_not_found"
    except OSError as exc:
        return None, f"read_error:{exc}"


def read_timestamp_index(path: Path) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {
        "row_count": 0,
        "first_stamp_ns": None,
        "last_stamp_ns": None,
        "strictly_increasing": True,
        "unique_timestamps": True,
        "missing_image_count": 0,
        "unindexed_jpeg_count": 0,
        "median_interval_ms": None,
        "maximum_interval_ms": None,
    }
    errors: list[str] = []
    rows: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(
                        f"invalid_timestamp_json:line={line_number}:{exc.msg}"
                    )
                    continue
                if not isinstance(row, dict):
                    errors.append(
                        f"timestamp_row_not_object:line={line_number}"
                    )
                    continue
                rows.append(row)
    except FileNotFoundError:
        errors.append("timestamps_file_not_found")
        return result, errors
    except OSError as exc:
        errors.append(f"timestamps_read_error:{exc}")
        return result, errors

    stamps: list[int] = []
    indexed_names: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        stamp = row.get("stamp_ns")
        image_path = row.get("image_path")
        if not isinstance(stamp, int):
            errors.append(f"invalid_stamp_ns:row={row_number}")
            continue
        if not isinstance(image_path, str) or not image_path:
            errors.append(f"invalid_image_path:row={row_number}")
            continue
        stamps.append(stamp)
        indexed_names.add(image_path)
        if not (path.parent / image_path).is_file():
            result["missing_image_count"] += 1

    jpeg_names = {
        child.name
        for child in path.parent.iterdir()
        if child.is_file() and child.suffix.lower() == ".jpg"
    }
    result["unindexed_jpeg_count"] = len(jpeg_names - indexed_names)
    result["row_count"] = len(rows)

    if stamps:
        result["first_stamp_ns"] = stamps[0]
        result["last_stamp_ns"] = stamps[-1]
        result["strictly_increasing"] = all(
            current > previous
            for previous, current in zip(stamps, stamps[1:])
        )
        result["unique_timestamps"] = len(stamps) == len(set(stamps))
        intervals = [
            current - previous
            for previous, current in zip(stamps, stamps[1:])
            if current > previous
        ]
        if intervals:
            result["median_interval_ms"] = statistics.median(intervals) / 1e6
            result["maximum_interval_ms"] = max(intervals) / 1e6

    return result, errors


def directory_size_bytes(root: Path) -> int:
    total = 0
    for directory_path, _, file_names in os.walk(root):
        directory = Path(directory_path)
        for file_name in file_names:
            path = directory / file_name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def relative_file_status(clip_dir: Path) -> dict[str, bool]:
    return {
        relative_path: (clip_dir / relative_path).is_file()
        for relative_path in REQUIRED_RELATIVE_PATHS
    }


def inspect_clip(clip_dir: Path, dataset_root: Path) -> dict[str, Any]:
    match = CLIP_PATTERN.fullmatch(clip_dir.name)
    if match is None:
        raise ValueError(f"Unexpected clip directory name: {clip_dir.name}")

    clip_number = int(match.group(1))
    metadata, metadata_error = read_json(clip_dir / "metadata.json")
    validation, validation_error = read_json(clip_dir / "validation.json")

    file_status = relative_file_status(clip_dir)
    missing_required_files = sorted(
        relative_path
        for relative_path, exists in file_status.items()
        if not exists
    )

    jsonl_counts: dict[str, int | None] = {}
    jsonl_errors: dict[str, str] = {}
    for relative_path in JSONL_RELATIVE_PATHS:
        count, error = count_nonempty_lines(clip_dir / relative_path)
        jsonl_counts[relative_path] = count
        if error is not None:
            jsonl_errors[relative_path] = error

    cameras: dict[str, Any] = {}
    camera_errors: dict[str, list[str]] = {}
    for camera_name in CAMERA_NAMES:
        timestamp_path = (
            clip_dir / "cameras" / camera_name / "timestamps.jsonl"
        )
        camera_summary, errors = read_timestamp_index(timestamp_path)
        camera_summary["calibration_present"] = (
            clip_dir / "calibration" / f"{camera_name}.json"
        ).is_file()
        cameras[camera_name] = camera_summary
        if errors:
            camera_errors[camera_name] = errors

    validation_checks = {}
    validation_required_checks: list[str] = []
    validation_valid: bool | None = None
    first_sim_time_ns = None
    last_sim_time_ns = None
    sim_duration_sec = None
    recorder_topic_counts: dict[str, int] = {}
    camera_statistics: dict[str, Any] = {}

    if validation is not None:
        raw_checks = validation.get("checks", {})
        if isinstance(raw_checks, dict):
            validation_checks = raw_checks
        raw_required = validation.get("required_checks", [])
        if isinstance(raw_required, list):
            validation_required_checks = [
                str(value) for value in raw_required
            ]
        raw_valid = validation.get("valid")
        if isinstance(raw_valid, bool):
            validation_valid = raw_valid
        first_sim_time_ns = validation.get("first_sim_time_ns")
        last_sim_time_ns = validation.get("last_sim_time_ns")
        sim_duration_sec = validation.get("sim_duration_sec")
        raw_counts = validation.get("topic_counts", {})
        if isinstance(raw_counts, dict):
            recorder_topic_counts = raw_counts
        raw_camera_stats = validation.get("camera_statistics", {})
        if isinstance(raw_camera_stats, dict):
            camera_statistics = raw_camera_stats

    failed_validation_checks = sorted(
        name
        for name, passed in validation_checks.items()
        if passed is False
    )

    derived_checks = {
        "metadata_readable": metadata_error is None,
        "validation_readable": validation_error is None,
        "required_files_present": not missing_required_files,
        "camera_indexes_readable": not camera_errors,
        "camera_images_complete": all(
            summary["row_count"] > 0
            and summary["missing_image_count"] == 0
            and summary["unindexed_jpeg_count"] == 0
            for summary in cameras.values()
        ),
        "camera_timestamps_valid": all(
            summary["strictly_increasing"]
            and summary["unique_timestamps"]
            for summary in cameras.values()
        ),
        "route_available": (
            (jsonl_counts.get("route/map_route.jsonl") or 0) > 0
            and (
                jsonl_counts.get("route/navigation_route_local.jsonl")
                or 0
            )
            > 0
        ),
        "complete_ground_truth_present": file_status[
            "ego/complete_recording_ground_truth.json"
        ],
        "vector_map_present": file_status["map/vector_map.json"],
    }

    manifest_usable = all(
        derived_checks[name]
        for name in (
            "metadata_readable",
            "validation_readable",
            "required_files_present",
            "camera_indexes_readable",
            "camera_images_complete",
            "camera_timestamps_valid",
            "complete_ground_truth_present",
            "vector_map_present",
        )
    ) and validation_valid is True

    dataset_format_version = None
    recorder_status = None
    if metadata is not None:
        dataset_format_version = metadata.get("dataset_format_version")
        recorder_status = metadata.get("status")

    return {
        "manifest_version": MANIFEST_VERSION,
        "manifest_builder_version": SCRIPT_VERSION,
        "clip_id": clip_dir.name,
        "clip_number": clip_number,
        "clip_path": str(clip_dir.relative_to(dataset_root)),
        "dataset_format_version": dataset_format_version,
        "recorder_status": recorder_status,
        "validation_valid": validation_valid,
        "manifest_usable": manifest_usable,
        "first_sim_time_ns": first_sim_time_ns,
        "last_sim_time_ns": last_sim_time_ns,
        "sim_duration_sec": sim_duration_sec,
        "size_bytes": directory_size_bytes(clip_dir),
        "cameras": cameras,
        "camera_statistics": camera_statistics,
        "jsonl_line_counts": jsonl_counts,
        "recorder_topic_counts": recorder_topic_counts,
        "validation_checks": validation_checks,
        "validation_required_checks": validation_required_checks,
        "failed_validation_checks": failed_validation_checks,
        "derived_checks": derived_checks,
        "missing_required_files": missing_required_files,
        "errors": {
            "metadata": metadata_error,
            "validation": validation_error,
            "jsonl": jsonl_errors,
            "cameras": camera_errors,
        },
    }


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    sizes_mb = [record["size_bytes"] / 1_000_000 for record in records]
    durations = [
        float(record["sim_duration_sec"])
        for record in records
        if isinstance(record["sim_duration_sec"], (int, float))
    ]
    camera_counts: dict[str, list[int]] = {
        camera_name: [] for camera_name in CAMERA_NAMES
    }
    for record in records:
        for camera_name in CAMERA_NAMES:
            count = record["cameras"][camera_name]["row_count"]
            camera_counts[camera_name].append(int(count))

    validation_failures = Counter()
    derived_failures = Counter()
    metadata_versions = Counter()
    for record in records:
        for name in record["failed_validation_checks"]:
            validation_failures[name] += 1
        for name, passed in record["derived_checks"].items():
            if not passed:
                derived_failures[name] += 1
        metadata_versions[str(record["dataset_format_version"])] += 1

    return {
        "manifest_version": MANIFEST_VERSION,
        "manifest_builder_version": SCRIPT_VERSION,
        "generated_at": utc_now_iso(),
        "total_clips": len(records),
        "validation_valid_clips": sum(
            record["validation_valid"] is True for record in records
        ),
        "manifest_usable_clips": sum(
            record["manifest_usable"] is True for record in records
        ),
        "route_available_clips": sum(
            record["derived_checks"]["route_available"]
            for record in records
        ),
        "dataset_format_versions": dict(metadata_versions),
        "validation_failure_counts": dict(validation_failures),
        "derived_failure_counts": dict(derived_failures),
        "size_mb": {
            "total": sum(sizes_mb),
            "mean": statistics.mean(sizes_mb) if sizes_mb else None,
            "median": statistics.median(sizes_mb) if sizes_mb else None,
            "p90": percentile(sizes_mb, 0.90),
            "p95": percentile(sizes_mb, 0.95),
            "maximum": max(sizes_mb) if sizes_mb else None,
        },
        "duration_sec": {
            "mean": statistics.mean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "minimum": min(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
        "camera_frame_counts": {
            camera_name: {
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
            }
            for camera_name, values in camera_counts.items()
        },
    }


def safe_write_text(path: Path, text: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. Use --force."
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def discover_clips(dataset_root: Path) -> list[Path]:
    clips: list[tuple[int, Path]] = []
    for path in dataset_root.iterdir():
        if not path.is_dir():
            continue
        match = CLIP_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        clips.append((int(match.group(1)), path))
    clips.sort(key=lambda item: item[0])
    return [path for _, path in clips]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only JSONL manifest for AlpaSim clips."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/lab/data_from_alpasim"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "/home/lab/data_from_alpasim/manifests/clips_v0.1.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "/home/lab/data_from_alpasim/reports/"
            "clip_manifest_summary_v0.1.json"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest and summary outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root is not a directory: {dataset_root}")
        return 2

    clip_dirs = discover_clips(dataset_root)
    if not clip_dirs:
        print(f"ERROR: no finalized clip directories found in {dataset_root}")
        return 2

    records: list[dict[str, Any]] = []
    total = len(clip_dirs)
    for index, clip_dir in enumerate(clip_dirs, start=1):
        records.append(inspect_clip(clip_dir, dataset_root))
        if index == 1 or index % 50 == 0 or index == total:
            print(f"Inspected {index}/{total}: {clip_dir.name}")

    manifest_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for record in records
    )
    summary = build_summary(records)

    try:
        safe_write_text(args.manifest_output, manifest_text, args.force)
        safe_write_text(
            args.summary_output,
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            args.force,
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}")
        return 3

    digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    print("Manifest written:", args.manifest_output)
    print("Summary written:", args.summary_output)
    print("Manifest SHA-256:", digest)
    print("Total clips:", summary["total_clips"])
    print("Validation-valid clips:", summary["validation_valid_clips"])
    print("Manifest-usable clips:", summary["manifest_usable_clips"])
    print("Route-available clips:", summary["route_available_clips"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
