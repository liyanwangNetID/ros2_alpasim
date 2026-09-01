#!/usr/bin/env python3
"""Profile lateral evidence and longitudinal motion features for Meta-actions.

Reads selected anchors and refined lane-matching features v0.2. For every
anchor, it extracts velocity-based future motion features from complete ground
truth and preserves map-based lateral evidence. It does not assign final
Meta-action labels or thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from clip_reader import DrivingClipReader, stamp_mapping_to_ns  # noqa: E402
from coordinate_utils import pose2d_from_pose_mapping  # noqa: E402
from project_paths import (  # noqa: E402
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

SCRIPT_VERSION = "0.2.0"
FEATURE_FORMAT_VERSION = "0.2-draft"
ROOT = ALPASIM_DATA_ROOT
DEFAULT_ANCHORS = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
DEFAULT_LANE_FEATURES = (
    INTERMEDIATE_ROOT / "lane_matching_features_v0.2.jsonl"
)
DEFAULT_OUTPUT = (
    INTERMEDIATE_ROOT / "meta_action_features_v0.2.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "meta_action_feature_summary_v0.2.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            records.append(value)
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


def finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return {
        "minimum": min(items) if items else None,
        "mean": statistics.mean(items) if items else None,
        "median": statistics.median(items) if items else None,
        "p10": percentile(items, 0.10),
        "p25": percentile(items, 0.25),
        "p75": percentile(items, 0.75),
        "p90": percentile(items, 0.90),
        "p95": percentile(items, 0.95),
        "maximum": max(items) if items else None,
    }


def trapezoid_duration_below(
    stamps_ns: Sequence[int],
    speeds: Sequence[float],
    threshold_mps: float,
) -> float:
    """Approximate duration whose interval-average speed is below threshold."""
    duration = 0.0
    for index in range(len(speeds) - 1):
        midpoint_speed = 0.5 * (speeds[index] + speeds[index + 1])
        if midpoint_speed <= threshold_mps:
            duration += (stamps_ns[index + 1] - stamps_ns[index]) / 1e9
    return duration


def longest_contiguous_duration_below(
    stamps_ns: Sequence[int],
    speeds: Sequence[float],
    threshold_mps: float,
) -> float:
    longest = 0.0
    current = 0.0
    for index in range(len(speeds) - 1):
        interval = (stamps_ns[index + 1] - stamps_ns[index]) / 1e9
        midpoint_speed = 0.5 * (speeds[index] + speeds[index + 1])
        if midpoint_speed <= threshold_mps:
            current += interval
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def _pose_derived_point_speeds(
    points: Sequence[Mapping[str, Any]],
    stamps_ns: Sequence[int],
) -> list[float]:
    """Derive one speed per pose using adjacent displacement and timestamps."""
    poses = [pose2d_from_pose_mapping(point["pose"]) for point in points]
    interval_speeds: list[float] = []
    interval_durations: list[float] = []
    for index in range(len(poses) - 1):
        duration_sec = (stamps_ns[index + 1] - stamps_ns[index]) / 1e9
        if duration_sec <= 0.0:
            raise ValueError("future trajectory timestamps must increase")
        distance_m = math.hypot(
            poses[index + 1].x - poses[index].x,
            poses[index + 1].y - poses[index].y,
        )
        interval_speeds.append(distance_m / duration_sec)
        interval_durations.append(duration_sec)

    if not interval_speeds:
        raise ValueError("future trajectory requires at least one interval")

    point_speeds = [interval_speeds[0]]
    for index in range(1, len(points) - 1):
        previous_duration = interval_durations[index - 1]
        next_duration = interval_durations[index]
        point_speeds.append(
            (
                interval_speeds[index - 1] * previous_duration
                + interval_speeds[index] * next_duration
            )
            / (previous_duration + next_duration)
        )
    point_speeds.append(interval_speeds[-1])
    return point_speeds


def _median_absolute_error(first: Sequence[float], second: Sequence[float]) -> float:
    return statistics.median(abs(a - b) for a, b in zip(first, second))


def extract_longitudinal_features(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(points) < 2:
        raise ValueError("future trajectory requires at least two points")
    stamps = [stamp_mapping_to_ns(point["stamp"]) for point in points]
    for previous, current in zip(stamps, stamps[1:]):
        if current <= previous:
            raise ValueError("future trajectory timestamps must increase")

    reported_speeds = [finite_float(point["speed"], "speed") for point in points]
    speeds = _pose_derived_point_speeds(points, stamps)
    speed_errors = [abs(a - b) for a, b in zip(reported_speeds, speeds)]
    ordered_errors = sorted(speed_errors)
    p95_error = ordered_errors[round((len(ordered_errors) - 1) * 0.95)]
    reported_zero_count = sum(value <= 0.01 for value in reported_speeds)
    reported_zero_fraction = reported_zero_count / len(reported_speeds)
    median_error = statistics.median(speed_errors)
    reported_speed_reliable = (
        reported_zero_fraction <= 0.05
        and median_error <= 1.0
        and p95_error <= 3.0
    )

    accelerations = []
    for point in points:
        acceleration = point.get("linear_acceleration", {})
        if isinstance(acceleration, Mapping) and "x" in acceleration:
            accelerations.append(
                finite_float(acceleration["x"], "linear_acceleration.x")
            )

    duration_sec = (stamps[-1] - stamps[0]) / 1e9
    speed_delta = speeds[-1] - speeds[0]
    derived_mean_acceleration = speed_delta / duration_sec if duration_sec > 0 else 0.0
    first_half_count = max(1, len(speeds) // 2)
    second_half_start = len(speeds) // 2
    return {
        "speed_source_used": "pose_time_derived",
        "trajectory_point_count": len(points),
        "trajectory_duration_sec": duration_sec,
        "initial_speed_mps": speeds[0],
        "final_speed_mps": speeds[-1],
        "minimum_speed_mps": min(speeds),
        "maximum_speed_mps": max(speeds),
        "mean_speed_mps": statistics.mean(speeds),
        "median_speed_mps": statistics.median(speeds),
        "speed_delta_mps": speed_delta,
        "absolute_speed_delta_mps": abs(speed_delta),
        "derived_mean_acceleration_mps2": derived_mean_acceleration,
        "reported_initial_speed_mps": reported_speeds[0],
        "reported_final_speed_mps": reported_speeds[-1],
        "reported_mean_speed_mps": statistics.mean(reported_speeds),
        "reported_minimum_speed_mps": min(reported_speeds),
        "reported_maximum_speed_mps": max(reported_speeds),
        "reported_zero_speed_count": reported_zero_count,
        "reported_zero_speed_fraction": reported_zero_fraction,
        "reported_vs_pose_median_absolute_error_mps": median_error,
        "reported_vs_pose_p95_absolute_error_mps": p95_error,
        "reported_speed_reliable": reported_speed_reliable,
        "reported_mean_longitudinal_acceleration_mps2": (
            statistics.mean(accelerations) if accelerations else None
        ),
        "reported_minimum_longitudinal_acceleration_mps2": (
            min(accelerations) if accelerations else None
        ),
        "reported_maximum_longitudinal_acceleration_mps2": (
            max(accelerations) if accelerations else None
        ),
        "first_half_mean_speed_mps": statistics.mean(speeds[:first_half_count]),
        "second_half_mean_speed_mps": statistics.mean(speeds[second_half_start:]),
        "second_half_minus_first_half_mean_speed_mps": (
            statistics.mean(speeds[second_half_start:])
            - statistics.mean(speeds[:first_half_count])
        ),
        "duration_below_0_1_mps_sec": trapezoid_duration_below(stamps, speeds, 0.1),
        "duration_below_0_3_mps_sec": trapezoid_duration_below(stamps, speeds, 0.3),
        "duration_below_0_5_mps_sec": trapezoid_duration_below(stamps, speeds, 0.5),
        "longest_duration_below_0_3_mps_sec": longest_contiguous_duration_below(
            stamps, speeds, 0.3
        ),
        "longest_duration_below_0_5_mps_sec": longest_contiguous_duration_below(
            stamps, speeds, 0.5
        ),
        "final_speed_below_0_1_mps": speeds[-1] <= 0.1,
        "final_speed_below_0_3_mps": speeds[-1] <= 0.3,
        "final_speed_below_0_5_mps": speeds[-1] <= 0.5,
        "initial_speed_below_0_3_mps": speeds[0] <= 0.3,
    }


def adjacent_summary(lane_feature: Mapping[str, Any]) -> dict[str, Any]:
    evidence = [
        item for item in lane_feature.get("adjacent_transition_evidence", [])
        if isinstance(item, Mapping)
    ]
    return {
        "transition_count": len(evidence),
        "directions": [str(item["direction"]) for item in evidence],
        "maximum_direct_target_duration_sec": max(
            (float(item["direct_target_duration_sec"]) for item in evidence),
            default=None,
        ),
        "maximum_corridor_duration_sec": max(
            (float(item["corridor_duration_sec"]) for item in evidence),
            default=None,
        ),
        "maximum_corridor_point_count": max(
            (int(item["corridor_point_count"]) for item in evidence),
            default=None,
        ),
        "any_corridor_reaches_horizon": any(
            bool(item["corridor_reaches_horizon"]) for item in evidence
        ),
        "any_return_to_source_lane": any(
            bool(item["returns_to_source_lane"]) for item in evidence
        ),
        "any_opposite_adjacent_after": any(
            bool(item["contains_opposite_adjacent_after"]) for item in evidence
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--anchor-input", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--lane-feature-input", type=Path, default=DEFAULT_LANE_FEATURES)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anchors = read_jsonl(args.anchor_input.expanduser().resolve())
    lane_features = read_jsonl(args.lane_feature_input.expanduser().resolve())
    lane_by_anchor = {str(item["anchor_id"]): item for item in lane_features}
    if len(lane_by_anchor) != len(lane_features):
        raise ValueError("duplicate anchor_id in lane features")

    if args.limit_clips is not None:
        if args.limit_clips <= 0:
            raise ValueError("--limit-clips must be positive")
        ordered: list[str] = []
        for anchor in anchors:
            clip_id = str(anchor["clip_id"])
            if clip_id not in ordered:
                ordered.append(clip_id)
        allowed = set(ordered[: args.limit_clips])
        anchors = [item for item in anchors if str(item["clip_id"]) in allowed]

    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        by_clip[str(anchor["clip_id"])].append(anchor)

    output: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    started = time.monotonic()
    for clip_index, (clip_id, clip_anchors) in enumerate(by_clip.items(), start=1):
        try:
            reader = DrivingClipReader(args.dataset_root / clip_id)
            for anchor in clip_anchors:
                anchor_id = str(anchor["anchor_id"])
                try:
                    lane_feature = lane_by_anchor[anchor_id]
                    horizon_ns = int(anchor.get("future_horizon_ns", 3_000_000_000))
                    future = reader.get_future_ego_trajectory(
                        int(anchor["anchor_ns"]), horizon_ns=horizon_ns
                    )
                    if future is None:
                        raise RuntimeError("future trajectory unavailable")
                    longitudinal = extract_longitudinal_features(future.points)
                    output.append(
                        {
                            "feature_format_version": FEATURE_FORMAT_VERSION,
                            "profiler_version": SCRIPT_VERSION,
                            "anchor_id": anchor_id,
                            "clip_id": clip_id,
                            "anchor_ns": int(anchor["anchor_ns"]),
                            "future_horizon_ns": horizon_ns,
                            "lateral_quality_gate": lane_feature["lateral_quality_gate"],
                            "lane_matching": {
                                "matched_fraction": lane_feature["matched_fraction"],
                                "confidence": lane_feature["confidence"],
                                "compressed_lane_sequence": lane_feature[
                                    "compressed_lane_sequence"
                                ],
                                "transitions": lane_feature["transitions"],
                                "adjacent_transition_evidence": lane_feature[
                                    "adjacent_transition_evidence"
                                ],
                                "adjacent_summary": adjacent_summary(lane_feature),
                            },
                            "longitudinal": longitudinal,
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "anchor_id": anchor_id,
                            "clip_id": clip_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
        except Exception as exc:
            for anchor in clip_anchors:
                errors.append(
                    {
                        "anchor_id": str(anchor["anchor_id"]),
                        "clip_id": clip_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if clip_index == 1 or clip_index % 50 == 0 or clip_index == len(by_clip):
            print(f"Processed {clip_index}/{len(by_clip)} clips: {clip_id}")

    adjacent_evidence = [
        evidence
        for item in output
        for evidence in item["lane_matching"]["adjacent_transition_evidence"]
    ]
    longitudinal_fields = (
        "initial_speed_mps",
        "final_speed_mps",
        "minimum_speed_mps",
        "maximum_speed_mps",
        "mean_speed_mps",
        "speed_delta_mps",
        "derived_mean_acceleration_mps2",
        "second_half_minus_first_half_mean_speed_mps",
        "duration_below_0_3_mps_sec",
        "duration_below_0_5_mps_sec",
        "longest_duration_below_0_3_mps_sec",
        "longest_duration_below_0_5_mps_sec",
    )
    summary = {
        "feature_format_version": FEATURE_FORMAT_VERSION,
        "profiler_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_anchor_file": str(args.anchor_input),
        "input_lane_feature_file": str(args.lane_feature_input),
        "elapsed_sec": time.monotonic() - started,
        "processing": {
            "attempted_anchor_count": len(anchors),
            "successful_anchor_count": len(output),
            "failed_anchor_count": len(errors),
            "processing_errors": errors,
        },
        "lateral_quality_gate_counts": dict(
            Counter(item["lateral_quality_gate"]["status"] for item in output)
        ),
        "adjacent_evidence": {
            "count": len(adjacent_evidence),
            "direction_counts": dict(Counter(item["direction"] for item in adjacent_evidence)),
            "direct_target_duration_sec": distribution(
                float(item["direct_target_duration_sec"]) for item in adjacent_evidence
            ),
            "corridor_duration_sec": distribution(
                float(item["corridor_duration_sec"]) for item in adjacent_evidence
            ),
            "corridor_point_count": distribution(
                float(item["corridor_point_count"]) for item in adjacent_evidence
            ),
            "corridor_reaches_horizon_count": sum(
                bool(item["corridor_reaches_horizon"]) for item in adjacent_evidence
            ),
            "returns_to_source_lane_count": sum(
                bool(item["returns_to_source_lane"]) for item in adjacent_evidence
            ),
            "opposite_adjacent_after_count": sum(
                bool(item["contains_opposite_adjacent_after"]) for item in adjacent_evidence
            ),
        },
        "longitudinal_distributions": {
            field: distribution(
                float(item["longitudinal"][field]) for item in output
            )
            for field in longitudinal_fields
        },
        "longitudinal_boolean_counts": {
            field: sum(bool(item["longitudinal"][field]) for item in output)
            for field in (
                "final_speed_below_0_1_mps",
                "final_speed_below_0_3_mps",
                "final_speed_below_0_5_mps",
                "initial_speed_below_0_3_mps",
                "reported_speed_reliable",
            )
        },
        "speed_source_counts": dict(
            Counter(item["longitudinal"]["speed_source_used"] for item in output)
        ),
    }

    feature_text = "".join(
        json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n"
        for item in output
    )
    atomic_write(args.feature_output, feature_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        args.force,
    )
    print("Feature output:", args.feature_output)
    print("Summary output:", args.summary_output)
    print("Feature SHA-256:", hashlib.sha256(feature_text.encode()).hexdigest())
    print("Successful anchors:", len(output))
    print("Processing errors:", len(errors))
    print("Adjacent evidence records:", len(adjacent_evidence))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
