#!/usr/bin/env python3
"""Profile Anchor-time navigation route geometry for Step 6.

Only the route message available at or before each Keyframe Anchor is used.
No future Ego trajectory, future speed, future control, or Meta-action label is
used to derive these route features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clip_reader import DrivingClipReader
from navigation_route_features_v01 import (
    FEATURE_FORMAT_VERSION,
    PROFILER_VERSION,
    route_geometry_features,
    valid_local_points,
)
from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

DATA_ROOT = ALPASIM_DATA_ROOT
DEFAULT_KEYFRAMES = (
    ANNOTATION_ROOT / "keyframes.jsonl"
)
DEFAULT_OUTPUT = (
    INTERMEDIATE_ROOT
    / "navigation_route_features_v0.1.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT
    / "navigation_route_feature_summary_v0.1.json"
)
DEFAULT_ROUTE_MAX_AGE_NS = 200_000_000


def read_keyframes(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in seen:
                raise ValueError(f"duplicate Anchor at {path}:{line_number}: {anchor_id}")
            seen.add(anchor_id)
            records.append(record)
    records.sort(key=lambda value: (str(value["clip_id"]), int(value["anchor_ns"]), str(value["anchor_id"])))
    return records


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}; use --force")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-input", type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--maximum-route-age-ns", type=int, default=DEFAULT_ROUTE_MAX_AGE_NS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.maximum_route_age_ns < 0:
        raise ValueError("--maximum-route-age-ns must be non-negative")

    keyframes = read_keyframes(args.keyframe_input)
    readers: dict[str, DrivingClipReader] = {}
    output_records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    age_buckets: Counter[str] = Counter()
    error_reasons: Counter[str] = Counter()

    for index, keyframe in enumerate(keyframes, start=1):
        clip_id = str(keyframe["clip_id"])
        anchor_ns = int(keyframe["anchor_ns"])
        reader = readers.get(clip_id)
        if reader is None:
            reader = DrivingClipReader(args.dataset_root / clip_id)
            readers[clip_id] = reader

        timed = reader.get_navigation_route_at(
            anchor_ns,
            maximum_age_ns=args.maximum_route_age_ns,
        )

        base = {
            "feature_format_version": FEATURE_FORMAT_VERSION,
            "profiler_version": PROFILER_VERSION,
            "anchor_id": str(keyframe["anchor_id"]),
            "clip_id": clip_id,
            "anchor_ns": anchor_ns,
            "route_time_policy": "at_or_before_anchor",
            "maximum_route_age_ns": args.maximum_route_age_ns,
        }

        if timed is None:
            base.update({
                "quality_status": "unknown",
                "reasons": ["no_route_at_or_before_anchor_within_maximum_age"],
                "route_stamp_ns": None,
                "route_age_ns": None,
                "route": {},
            })
            status_counts["unknown"] += 1
            error_reasons["no_recent_route"] += 1
            output_records.append(base)
            continue

        message = timed.message
        points = valid_local_points(message)
        try:
            geometry = route_geometry_features(points)
            usable = (
                geometry["valid_point_count"] >= 3
                and geometry["route_path_length_m"] >= 5.0
                and geometry["forward_point_fraction"] >= 0.8
            )
            reasons = [] if usable else ["route_geometry_quality_gate_failed"]
            status = "usable" if usable else "unknown"
        except ValueError as error:
            geometry = {}
            reasons = ["route_geometry_unavailable", str(error)]
            status = "unknown"
            error_reasons["route_geometry_unavailable"] += 1

        route_age_ns = anchor_ns - int(timed.stamp_ns)
        if route_age_ns <= 50_000_000:
            age_buckets["0_to_50ms"] += 1
        elif route_age_ns <= 100_000_000:
            age_buckets["50_to_100ms"] += 1
        else:
            age_buckets["100_to_200ms"] += 1

        base.update({
            "quality_status": status,
            "reasons": reasons,
            "route_stamp_ns": int(timed.stamp_ns),
            "route_age_ns": route_age_ns,
            "route_sequence": message.get("sequence"),
            "route_frame_id": message.get("frame_id"),
            "route_source_frame_id": message.get("source_frame_id"),
            "route_lookahead_distance_m": message.get("lookahead_distance"),
            "route_expected_point_count": message.get("expected_point_count"),
            "route": geometry,
        })
        status_counts[status] += 1
        output_records.append(base)

        if index == 1 or index % 250 == 0 or index == len(keyframes):
            print(f"Processed {index}/{len(keyframes)} Keyframes: {clip_id}")

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output_records
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "feature_format_version": FEATURE_FORMAT_VERSION,
        "profiler_version": PROFILER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_keyframe_count": len(keyframes),
        "output_record_count": len(output_records),
        "route_time_policy": "at_or_before_anchor",
        "maximum_route_age_ns": args.maximum_route_age_ns,
        "quality_status_counts": dict(status_counts),
        "route_age_bucket_counts": dict(age_buckets),
        "error_reason_counts": dict(error_reasons),
        "output_sha256": sha256,
        "leakage_controls": {
            "future_ego_trajectory_used": False,
            "future_speed_used": False,
            "future_control_used": False,
            "meta_action_used_for_generation": False,
        },
    }

    atomic_write(args.output, output_text, args.force)
    atomic_write(args.summary_output, json.dumps(summary, ensure_ascii=False, indent=2) + "\n", args.force)

    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("SHA-256:", sha256)
    print("Keyframes:", len(keyframes))
    print("Records:", len(output_records))
    print("Quality:", dict(status_counts))
    print("Route age buckets:", dict(age_buckets))
    print("Errors:", dict(error_reasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
