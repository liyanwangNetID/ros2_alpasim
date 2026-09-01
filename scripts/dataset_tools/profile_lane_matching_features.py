#!/usr/bin/env python3
"""Profile lane-matching features for selected AlpaSim anchors.

This Step 4 offline tool reads candidate_anchors.jsonl, groups anchors by clip,
loads each clip and VectorMap once, matches each anchor's future ego trajectory
to lanes, and writes:

1. reusable per-anchor intermediate features as JSONL;
2. a human-readable aggregate summary as JSON.

Raw test_clip_NNN directories are read-only and are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from clip_reader import DrivingClipReader  # noqa: E402
from lane_matcher import (  # noqa: E402
    LaneMatcher,
    LaneMatcherConfig,
    trajectory_poses_from_gt_points,
)
from vector_map_reader import VectorMapReader  # noqa: E402
from project_paths import (  # noqa: E402
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)


SCRIPT_VERSION = "0.1.0"
FEATURE_FORMAT_VERSION = "0.1-draft"

DEFAULT_DATASET_ROOT = ALPASIM_DATA_ROOT
DEFAULT_ANCHOR_INPUT = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
DEFAULT_FEATURE_OUTPUT = (
    INTERMEDIATE_ROOT / "lane_matching_features_v0.1.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = (
    REPORT_ROOT / "lane_matching_summary_v0.1.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL row is not an object at {path}:{line_number}"
                    )
                records.append(value)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"input file not found: {path}") from exc
    return records


def atomic_write_text(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing output: {path}; use --force"
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
        file.write(content)
        file.flush()
        os.fsync(file.fileno())

    temporary_path.replace(path)


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return {
        "minimum": min(items) if items else None,
        "mean": statistics.mean(items) if items else None,
        "median": statistics.median(items) if items else None,
        "p90": percentile(items, 0.90),
        "p95": percentile(items, 0.95),
        "maximum": max(items) if items else None,
    }


def matcher_config_to_dict(config: LaneMatcherConfig) -> dict[str, Any]:
    return {
        "search_radius_m": config.search_radius_m,
        "maximum_heading_error_rad": config.maximum_heading_error_rad,
        "maximum_candidates_per_point": config.maximum_candidates_per_point,
        "polygon_outside_penalty": config.polygon_outside_penalty,
        "distance_weight": config.distance_weight,
        "heading_weight": config.heading_weight,
        "same_lane_transition_cost": config.same_lane_transition_cost,
        "successor_transition_cost": config.successor_transition_cost,
        "adjacent_transition_cost": config.adjacent_transition_cost,
        "predecessor_transition_cost": config.predecessor_transition_cost,
        "unrelated_transition_cost": config.unrelated_transition_cost,
        "unmatched_emission_cost": config.unmatched_emission_cost,
        "unmatched_transition_cost": config.unmatched_transition_cost,
        "minimum_match_confidence": config.minimum_match_confidence,
    }


def feature_record(
    anchor: dict[str, Any],
    result: Any,
    *,
    future_horizon_ns: int,
    map_id: str,
    map_revision: int | None,
    topology_warning_count: int,
) -> dict[str, Any]:
    transition_records = [
        {
            "source_lane_id": transition.source_lane_id,
            "target_lane_id": transition.target_lane_id,
            "relation": transition.relation,
            "source_point_index": transition.source_point_index,
            "target_point_index": transition.target_point_index,
        }
        for transition in result.transitions
    ]
    relation_counts = Counter(
        transition["relation"] for transition in transition_records
    )

    lane_point_counts: Counter[str] = Counter(
        point.lane_id
        for point in result.points
        if point.lane_id is not None
    )

    total_points = len(result.points)
    target_lane_id = (
        result.compressed_lane_sequence[-1]
        if result.compressed_lane_sequence
        else None
    )
    target_lane_point_count = (
        lane_point_counts.get(target_lane_id, 0)
        if target_lane_id is not None
        else 0
    )

    return {
        "feature_format_version": FEATURE_FORMAT_VERSION,
        "lane_matcher_version": SCRIPT_VERSION,
        "anchor_id": str(anchor["anchor_id"]),
        "clip_id": str(anchor["clip_id"]),
        "anchor_ns": int(anchor["anchor_ns"]),
        "future_horizon_ns": future_horizon_ns,
        "map_id": map_id,
        "map_revision": map_revision,
        "map_topology_warning_count": topology_warning_count,
        "trajectory_point_count": total_points,
        "matched_point_count": result.matched_point_count,
        "unmatched_point_count": result.unmatched_point_count,
        "matched_fraction": result.matched_fraction,
        "mean_distance_m": result.mean_distance_m,
        "maximum_distance_m": result.maximum_distance_m,
        "mean_heading_error_rad": result.mean_heading_error_rad,
        "total_cost": result.total_cost,
        "confidence": result.confidence,
        "compressed_lane_sequence": list(
            result.compressed_lane_sequence
        ),
        "lane_sequence_length": len(result.compressed_lane_sequence),
        "transitions": transition_records,
        "transition_relation_counts": dict(relation_counts),
        "start_lane_id": (
            result.compressed_lane_sequence[0]
            if result.compressed_lane_sequence
            else None
        ),
        "end_lane_id": target_lane_id,
        "target_lane_point_count": target_lane_point_count,
        "target_lane_fraction": (
            target_lane_point_count / total_points if total_points else 0.0
        ),
        "contains_left_adjacent_transition": (
            relation_counts["left_adjacent"] > 0
        ),
        "contains_right_adjacent_transition": (
            relation_counts["right_adjacent"] > 0
        ),
        "contains_predecessor_transition": (
            relation_counts["predecessor"] > 0
        ),
        "contains_unrelated_transition": (
            relation_counts["unrelated"] > 0
        ),
    }


def build_summary(
    features: list[dict[str, Any]],
    *,
    anchor_input: Path,
    matcher_config: LaneMatcherConfig,
    attempted_anchor_count: int,
    processed_clip_count: int,
    processing_errors: list[dict[str, str]],
    elapsed_sec: float,
) -> dict[str, Any]:
    transition_counts: Counter[str] = Counter()
    sequence_lengths: list[float] = []
    matched_fractions: list[float] = []
    confidences: list[float] = []
    mean_distances: list[float] = []
    maximum_distances: list[float] = []
    heading_errors: list[float] = []
    target_lane_fractions: list[float] = []

    fully_unmatched: list[str] = []
    low_match: list[str] = []
    low_confidence: list[str] = []
    anomalous_topology: list[str] = []
    left_adjacent: list[str] = []
    right_adjacent: list[str] = []

    for feature in features:
        transition_counts.update(feature["transition_relation_counts"])
        sequence_lengths.append(float(feature["lane_sequence_length"]))
        matched_fractions.append(float(feature["matched_fraction"]))
        confidences.append(float(feature["confidence"]))
        target_lane_fractions.append(float(feature["target_lane_fraction"]))

        if feature["mean_distance_m"] is not None:
            mean_distances.append(float(feature["mean_distance_m"]))
        if feature["maximum_distance_m"] is not None:
            maximum_distances.append(float(feature["maximum_distance_m"]))
        if feature["mean_heading_error_rad"] is not None:
            heading_errors.append(float(feature["mean_heading_error_rad"]))

        anchor_id = str(feature["anchor_id"])
        if feature["matched_fraction"] == 0.0:
            fully_unmatched.append(anchor_id)
        if feature["matched_fraction"] < 0.8:
            low_match.append(anchor_id)
        if feature["confidence"] < 0.5:
            low_confidence.append(anchor_id)
        if (
            feature["contains_predecessor_transition"]
            or feature["contains_unrelated_transition"]
        ):
            anomalous_topology.append(anchor_id)
        if feature["contains_left_adjacent_transition"]:
            left_adjacent.append(anchor_id)
        if feature["contains_right_adjacent_transition"]:
            right_adjacent.append(anchor_id)

    return {
        "feature_format_version": FEATURE_FORMAT_VERSION,
        "lane_matcher_version": SCRIPT_VERSION,
        "generated_at": utc_now_iso(),
        "input_anchor_file": str(anchor_input),
        "elapsed_sec": elapsed_sec,
        "matcher_configuration": matcher_config_to_dict(matcher_config),
        "processing": {
            "attempted_anchor_count": attempted_anchor_count,
            "successful_anchor_count": len(features),
            "failed_anchor_count": len(processing_errors),
            "processed_clip_count": processed_clip_count,
            "processing_errors": processing_errors,
        },
        "distributions": {
            "matched_fraction": distribution(matched_fractions),
            "confidence": distribution(confidences),
            "mean_distance_m": distribution(mean_distances),
            "maximum_distance_m": distribution(maximum_distances),
            "mean_heading_error_rad": distribution(heading_errors),
            "lane_sequence_length": distribution(sequence_lengths),
            "target_lane_fraction": distribution(target_lane_fractions),
        },
        "transition_relation_counts": dict(transition_counts),
        "quality_flags": {
            "fully_unmatched_count": len(fully_unmatched),
            "fully_unmatched_anchor_ids": fully_unmatched,
            "matched_fraction_below_0_8_count": len(low_match),
            "matched_fraction_below_0_8_anchor_ids": low_match,
            "confidence_below_0_5_count": len(low_confidence),
            "confidence_below_0_5_anchor_ids": low_confidence,
            "anomalous_topology_count": len(anomalous_topology),
            "anomalous_topology_anchor_ids": anomalous_topology,
            "left_adjacent_transition_count": len(left_adjacent),
            "left_adjacent_anchor_ids": left_adjacent,
            "right_adjacent_transition_count": len(right_adjacent),
            "right_adjacent_anchor_ids": right_adjacent,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reusable lane-matching features for selected anchors."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--anchor-input",
        type=Path,
        default=DEFAULT_ANCHOR_INPUT,
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=DEFAULT_FEATURE_OUTPUT,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="Process only the first N clips represented in the anchor file.",
    )
    parser.add_argument(
        "--limit-anchors",
        type=int,
        default=None,
        help="Process at most N anchors after clip filtering.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    anchor_input = args.anchor_input.expanduser().resolve()
    anchors = read_jsonl(anchor_input)

    if args.limit_clips is not None:
        if args.limit_clips <= 0:
            raise ValueError("--limit-clips must be positive")
        ordered_clip_ids: list[str] = []
        seen: set[str] = set()
        for anchor in anchors:
            clip_id = str(anchor["clip_id"])
            if clip_id not in seen:
                seen.add(clip_id)
                ordered_clip_ids.append(clip_id)
        allowed = set(ordered_clip_ids[: args.limit_clips])
        anchors = [
            anchor for anchor in anchors
            if str(anchor["clip_id"]) in allowed
        ]

    if args.limit_anchors is not None:
        if args.limit_anchors <= 0:
            raise ValueError("--limit-anchors must be positive")
        anchors = anchors[: args.limit_anchors]

    anchors_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        anchors_by_clip[str(anchor["clip_id"])].append(anchor)

    matcher_config = LaneMatcherConfig()
    features: list[dict[str, Any]] = []
    processing_errors: list[dict[str, str]] = []
    started = time.monotonic()
    processed_anchors = 0
    total_anchors = len(anchors)
    total_clips = len(anchors_by_clip)

    for clip_index, (clip_id, clip_anchors) in enumerate(
        anchors_by_clip.items(),
        start=1,
    ):
        try:
            clip_reader = DrivingClipReader(dataset_root / clip_id)
            vector_map = VectorMapReader.from_dict(
                clip_reader.get_vector_map()
            )
            matcher = LaneMatcher(vector_map, matcher_config)

            for anchor in clip_anchors:
                anchor_id = str(anchor["anchor_id"])
                anchor_ns = int(anchor["anchor_ns"])
                horizon_ns = int(
                    anchor.get("future_horizon_ns", 3_000_000_000)
                )
                try:
                    future = clip_reader.get_future_ego_trajectory(
                        anchor_ns,
                        horizon_ns=horizon_ns,
                    )
                    if future is None:
                        raise RuntimeError("future trajectory is unavailable")
                    trajectory = trajectory_poses_from_gt_points(
                        future.points
                    )
                    result = matcher.match(trajectory)
                    features.append(
                        feature_record(
                            anchor,
                            result,
                            future_horizon_ns=horizon_ns,
                            map_id=vector_map.map_id,
                            map_revision=vector_map.revision,
                            topology_warning_count=len(
                                vector_map.topology_warnings
                            ),
                        )
                    )
                except Exception as exc:
                    processing_errors.append(
                        {
                            "anchor_id": anchor_id,
                            "clip_id": clip_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                processed_anchors += 1

        except Exception as exc:
            for anchor in clip_anchors:
                processing_errors.append(
                    {
                        "anchor_id": str(anchor.get("anchor_id", "")),
                        "clip_id": clip_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                processed_anchors += 1

        if (
            clip_index == 1
            or clip_index % 25 == 0
            or clip_index == total_clips
        ):
            print(
                f"Processed clips {clip_index}/{total_clips}; "
                f"anchors {processed_anchors}/{total_anchors}: {clip_id}"
            )

    elapsed_sec = time.monotonic() - started
    summary = build_summary(
        features,
        anchor_input=anchor_input,
        matcher_config=matcher_config,
        attempted_anchor_count=total_anchors,
        processed_clip_count=total_clips,
        processing_errors=processing_errors,
        elapsed_sec=elapsed_sec,
    )

    feature_text = "".join(
        json.dumps(feature, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for feature in features
    )
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    atomic_write_text(args.feature_output, feature_text, args.force)
    atomic_write_text(args.summary_output, summary_text, args.force)

    digest = hashlib.sha256(feature_text.encode("utf-8")).hexdigest()
    print("Feature output:", args.feature_output)
    print("Summary output:", args.summary_output)
    print("Feature SHA-256:", digest)
    print("Attempted anchors:", total_anchors)
    print("Successful anchors:", len(features))
    print("Processing errors:", len(processing_errors))
    print(
        "Transition relation counts:",
        summary["transition_relation_counts"],
    )
    print(
        "Low match count:",
        summary["quality_flags"]["matched_fraction_below_0_8_count"],
    )
    print(
        "Anomalous topology count:",
        summary["quality_flags"]["anomalous_topology_count"],
    )
    return 1 if processing_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
