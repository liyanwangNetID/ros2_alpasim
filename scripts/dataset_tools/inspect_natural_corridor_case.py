#!/usr/bin/env python3
"""Inspect one real Anchor against its map-only natural lane corridor.

This diagnostic tool validates the natural-continuation concept before it is
integrated into lateral_action_features_v0.3. It reads existing lane-matching
features, Future GT, and VectorMap, then prints:

- actual future path distance and yaw change;
- Anchor projection on the initial matched lane;
- natural continuation lane sequence;
- branch candidates and chosen natural successor;
- actual branch choice relative to the natural continuation;
- total and half-window lane-relative heading changes.

No files are modified and no final Meta-action label is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from clip_reader import DrivingClipReader  # noqa: E402
from coordinate_utils import Point2D, normalize_angle, pose2d_from_pose_mapping  # noqa: E402
from natural_lane_corridor import (  # noqa: E402
    NaturalCorridorConfig,
    NaturalLaneCorridor,
    build_natural_lane_corridor,
    compare_actual_lane_sequence,
    recover_boundary_branch_comparisons,
)
from vector_map_reader import VectorMapReader  # noqa: E402
from project_paths import (  # noqa: E402
    ALPASIM_DATA_ROOT,
    INTERMEDIATE_ROOT,
)

ROOT = ALPASIM_DATA_ROOT
DEFAULT_LANE_FEATURES = (
    INTERMEDIATE_ROOT / "lane_matching_features_v0.2.jsonl"
)


def read_jsonl_record(path: Path, anchor_id: str) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            if str(value.get("anchor_id", "")) == anchor_id:
                return value
    return None


def trajectory_path_length(points: Sequence[Mapping[str, Any]]) -> float:
    poses = [pose2d_from_pose_mapping(point["pose"]) for point in points]
    return sum(
        math.hypot(current.x - previous.x, current.y - previous.y)
        for previous, current in zip(poses, poses[1:])
    )


def actual_yaw_change(points: Sequence[Mapping[str, Any]]) -> float:
    poses = [pose2d_from_pose_mapping(point["pose"]) for point in points]
    return normalize_angle(poses[-1].yaw - poses[0].yaw)


def corridor_heading_at_distance(
    corridor: NaturalLaneCorridor,
    distance_m: float,
) -> float:
    """Return the heading of the corridor segment containing a distance."""
    if not corridor.points:
        raise ValueError("natural corridor has no points")
    query = min(max(0.0, distance_m), corridor.total_distance_m)
    if query <= corridor.points[0].distance_m:
        return corridor.points[0].heading_rad
    for previous, current in zip(corridor.points, corridor.points[1:]):
        if previous.distance_m <= query <= current.distance_m:
            return current.heading_rad
    return corridor.points[-1].heading_rad


def cumulative_trajectory_distances(
    points: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    poses = [pose2d_from_pose_mapping(point["pose"]) for point in points]
    distances = [0.0]
    for previous, current in zip(poses, poses[1:]):
        distances.append(
            distances[-1]
            + math.hypot(current.x - previous.x, current.y - previous.y)
        )
    return tuple(distances)


def relative_heading_features(
    points: Sequence[Mapping[str, Any]],
    corridor: NaturalLaneCorridor,
) -> dict[str, Any]:
    """Compare ego yaw and reference heading at identical traveled distances."""
    poses = [pose2d_from_pose_mapping(point["pose"]) for point in points]
    distances = cumulative_trajectory_distances(points)
    reference_headings = [
        corridor_heading_at_distance(corridor, distance)
        for distance in distances
    ]
    relative_headings = [
        normalize_angle(pose.yaw - reference_heading)
        for pose, reference_heading in zip(poses, reference_headings)
    ]
    middle = len(poses) // 2

    actual_first = normalize_angle(poses[middle].yaw - poses[0].yaw)
    actual_second = normalize_angle(poses[-1].yaw - poses[middle].yaw)
    reference_first = normalize_angle(
        reference_headings[middle] - reference_headings[0]
    )
    reference_second = normalize_angle(
        reference_headings[-1] - reference_headings[middle]
    )
    relative_first = normalize_angle(
        relative_headings[middle] - relative_headings[0]
    )
    relative_second = normalize_angle(
        relative_headings[-1] - relative_headings[middle]
    )
    relative_total = normalize_angle(
        relative_headings[-1] - relative_headings[0]
    )
    return {
        "trajectory_cumulative_distances_m": list(distances),
        "reference_heading_rad": reference_headings,
        "relative_heading_rad": relative_headings,
        "actual_first_half_yaw_change_rad": actual_first,
        "actual_second_half_yaw_change_rad": actual_second,
        "actual_total_yaw_change_rad": normalize_angle(
            poses[-1].yaw - poses[0].yaw
        ),
        "reference_first_half_heading_change_rad": reference_first,
        "reference_second_half_heading_change_rad": reference_second,
        "reference_total_heading_change_rad": normalize_angle(
            reference_headings[-1] - reference_headings[0]
        ),
        "relative_heading_start_rad": relative_headings[0],
        "relative_heading_middle_rad": relative_headings[middle],
        "relative_heading_end_rad": relative_headings[-1],
        "relative_first_half_heading_change_rad": relative_first,
        "relative_second_half_heading_change_rad": relative_second,
        "relative_total_heading_change_rad": relative_total,
        "maximum_absolute_relative_heading_rad": max(
            abs(value) for value in relative_headings
        ),
    }


def degrees(value: float) -> float:
    return round(math.degrees(value), 3)


def corridor_to_dict(corridor: NaturalLaneCorridor) -> dict[str, Any]:
    return {
        "lane_ids": list(corridor.lane_ids),
        "total_distance_m": corridor.total_distance_m,
        "terminated_reason": corridor.terminated_reason,
        "branch_decisions": [
            {
                "branch_lane_id": decision.branch_lane_id,
                "chosen_natural_successor_lane_id": (
                    decision.chosen_successor_lane_id
                ),
                "reliability_status": decision.reliability_status,
                "reliability_reasons": list(decision.reliability_reasons),
                "score_margin": decision.score_margin,
                "candidates": [
                    {
                        "successor_lane_id": candidate.successor_lane_id,
                        "lane_ids": list(candidate.lane_ids),
                        "signed_heading_change_deg": degrees(
                            candidate.signed_heading_change_rad
                        ),
                        "absolute_heading_change_deg": degrees(
                            candidate.absolute_heading_change_rad
                        ),
                        "evaluated_distance_m": candidate.evaluated_distance_m,
                        "score": candidate.score,
                    }
                    for candidate in decision.candidates
                ],
            }
            for decision in corridor.branch_decisions
        ],
    }


def inspect_anchor(
    anchor_id: str,
    *,
    dataset_root: Path,
    lane_feature_input: Path,
    branch_evaluation_distance_m: float,
    maximum_lookahead_m: float,
) -> dict[str, Any]:
    feature = read_jsonl_record(lane_feature_input, anchor_id)
    if feature is None:
        raise KeyError(f"anchor not found in lane features: {anchor_id}")

    clip_id = str(feature["clip_id"])
    reader = DrivingClipReader(dataset_root / clip_id)
    vector_map = VectorMapReader.from_dict(reader.get_vector_map())
    horizon_ns = int(feature.get("future_horizon_ns", 3_000_000_000))
    future = reader.get_future_ego_trajectory(
        int(feature["anchor_ns"]), horizon_ns=horizon_ns
    )
    if future is None or len(future.points) < 2:
        raise RuntimeError("future trajectory is unavailable or too short")

    actual_lane_sequence = [
        str(value) for value in feature["compressed_lane_sequence"]
    ]
    if not actual_lane_sequence:
        raise RuntimeError("actual lane sequence is empty")

    first_pose = pose2d_from_pose_mapping(future.points[0]["pose"])
    first_point = Point2D(first_pose.x, first_pose.y)
    start_lane_id = actual_lane_sequence[0]
    projection = vector_map.project_to_lane(start_lane_id, first_point)
    path_length_m = trajectory_path_length(future.points)
    if path_length_m <= 0.0:
        raise RuntimeError("future trajectory path length is zero")

    config = NaturalCorridorConfig(
        maximum_lookahead_m=maximum_lookahead_m,
        branch_evaluation_distance_m=branch_evaluation_distance_m,
    )
    corridor = build_natural_lane_corridor(
        vector_map,
        start_lane_id,
        lookahead_distance_m=path_length_m,
        start_arc_length_m=projection.arc_length_m,
        config=config,
    )
    comparisons = compare_actual_lane_sequence(actual_lane_sequence, corridor)
    boundary_comparisons = recover_boundary_branch_comparisons(
        vector_map,
        start_lane_id,
        config=config,
    )
    relative = relative_heading_features(future.points, corridor)

    return {
        "anchor_id": anchor_id,
        "clip_id": clip_id,
        "anchor_ns": int(feature["anchor_ns"]),
        "future_path_length_m": path_length_m,
        "actual_total_yaw_change_deg": degrees(actual_yaw_change(future.points)),
        "actual_lane_sequence": actual_lane_sequence,
        "start_lane_id": start_lane_id,
        "start_projection_arc_length_m": projection.arc_length_m,
        "start_projection_distance_m": projection.distance_m,
        "natural_corridor": corridor_to_dict(corridor),
        "boundary_branch_comparisons": [
            {
                "branch_lane_id": comparison.branch_lane_id,
                "natural_successor_lane_id": comparison.natural_successor_lane_id,
                "actual_successor_lane_id": comparison.actual_successor_lane_id,
                "actual_matches_natural": comparison.actual_matches_natural,
                "actual_relation_to_natural": comparison.actual_relation_to_natural,
            }
            for comparison in boundary_comparisons
        ],
        "actual_branch_comparisons": [
            {
                "branch_lane_id": comparison.branch_lane_id,
                "natural_successor_lane_id": (
                    comparison.natural_successor_lane_id
                ),
                "actual_successor_lane_id": comparison.actual_successor_lane_id,
                "actual_matches_natural": comparison.actual_matches_natural,
                "actual_relation_to_natural": (
                    comparison.actual_relation_to_natural
                ),
            }
            for comparison in comparisons
        ],
        "relative_heading_degrees": {
            key.replace("_rad", "_deg"): degrees(value)
            for key, value in relative.items()
            if key.endswith("_rad") and isinstance(value, (int, float))
        },
        "relative_heading_series_degrees": [
            degrees(value) for value in relative["relative_heading_rad"]
        ],
        "reference_heading_series_degrees": [
            degrees(value) for value in relative["reference_heading_rad"]
        ],
        "trajectory_cumulative_distances_m": relative[
            "trajectory_cumulative_distances_m"
        ],
        "lateral_quality_gate": feature["lateral_quality_gate"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-id", action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument(
        "--lane-feature-input", type=Path, default=DEFAULT_LANE_FEATURES
    )
    parser.add_argument(
        "--branch-evaluation-distance-m", type=float, default=30.0
    )
    parser.add_argument("--maximum-lookahead-m", type=float, default=80.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [
        inspect_anchor(
            anchor_id,
            dataset_root=args.dataset_root.expanduser().resolve(),
            lane_feature_input=args.lane_feature_input.expanduser().resolve(),
            branch_evaluation_distance_m=args.branch_evaluation_distance_m,
            maximum_lookahead_m=args.maximum_lookahead_m,
        )
        for anchor_id in args.anchor_id
    ]
    text = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print("Saved:", args.output, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
