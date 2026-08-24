#!/usr/bin/env python3
"""Profile map-grounded lateral-action features for selected anchors.

Version 0.2 adds:
- low-speed-resistant path-heading features;
- explicit Level A, B, and C junction evidence;
- retained map, trajectory, lane-change, and quality-gate evidence.

The tool does not assign final lateral Meta-action labels. Raw clips are read
only. Final classification thresholds belong to a later rule-scan stage.
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

from clip_reader import DrivingClipReader  # noqa: E402
from coordinate_utils import (  # noqa: E402
    Point2D,
    map_point_to_anchor_ego,
    normalize_angle,
    pose2d_from_pose_mapping,
    unwrap_angles,
)
from vector_map_reader import VectorMapReader  # noqa: E402
from natural_lane_corridor import (  # noqa: E402
    NaturalCorridorConfig,
    assess_branch_candidate_reliability,
    build_natural_lane_corridor,
    compare_actual_lane_sequence,
    recover_boundary_branch_comparisons,
)
from inspect_natural_corridor_case import (  # noqa: E402
    relative_heading_features,
    trajectory_path_length,
)

SCRIPT_VERSION = "0.3.2"
FEATURE_FORMAT_VERSION = "0.3.2-draft"
DEFAULT_MINIMUM_PATH_SEGMENT_LENGTH_M = 0.1

ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_ANCHORS = (
    ROOT / "annotations" / "v0.1-draft" / "candidate_anchors.jsonl"
)
DEFAULT_LANE_FEATURES = (
    ROOT
    / "annotations"
    / "v0.1-draft"
    / "intermediate"
    / "lane_matching_features_v0.2.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "annotations"
    / "v0.1-draft"
    / "intermediate"
    / "lateral_action_features_v0.3.jsonl"
)
DEFAULT_SUMMARY = (
    ROOT / "reports" / "lateral_action_feature_summary_v0.3.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"row {line_number} in {path} is not an object"
                )
            records.append(value)
    return records


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}; use --force")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


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


def signed_and_absolute_change(
    angles: Sequence[float],
) -> tuple[float, float]:
    if len(angles) < 2:
        return 0.0, 0.0
    unwrapped = unwrap_angles(angles)
    increments = [
        current - previous
        for previous, current in zip(unwrapped, unwrapped[1:])
    ]
    return sum(increments), sum(abs(value) for value in increments)


def filtered_path_heading_features(
    points: Sequence[Point2D],
    *,
    minimum_segment_length_m: float,
) -> dict[str, Any]:
    """Compute path-heading changes using only physically meaningful segments."""
    if minimum_segment_length_m <= 0.0:
        raise ValueError("minimum_segment_length_m must be positive")

    headings: list[float] = []
    valid_distance_m = 0.0
    rejected_segment_count = 0
    for first, second in zip(points, points[1:]):
        dx = second.x - first.x
        dy = second.y - first.y
        distance_m = math.hypot(dx, dy)
        if distance_m < minimum_segment_length_m:
            rejected_segment_count += 1
            continue
        headings.append(math.atan2(dy, dx))
        valid_distance_m += distance_m

    signed_change, absolute_change = signed_and_absolute_change(headings)
    valid_segment_count = len(headings)
    return {
        "minimum_segment_length_m": minimum_segment_length_m,
        "valid_path_segment_count": valid_segment_count,
        "rejected_path_segment_count": rejected_segment_count,
        "valid_path_distance_m": valid_distance_m,
        "filtered_path_signed_heading_change_rad": signed_change,
        "filtered_path_absolute_heading_change_rad": absolute_change,
        "path_heading_reliable": valid_segment_count >= 2,
    }


def topology_evidence(
    vector_map: VectorMapReader,
    lane_sequence: Sequence[str],
) -> dict[str, Any]:
    valid_lane_ids = [
        lane_id for lane_id in lane_sequence
        if vector_map.get_lane(lane_id) is not None
    ]
    missing_lane_ids = [
        lane_id for lane_id in lane_sequence
        if vector_map.get_lane(lane_id) is None
    ]

    wait_line_lanes: set[str] = set()
    traffic_sign_lanes: set[str] = set()
    branching_lanes: set[str] = set()
    merging_lanes: set[str] = set()
    predecessor_branch_lanes: set[str] = set()
    successor_merge_lanes: set[str] = set()

    for lane_id in valid_lane_ids:
        lane = vector_map.require_lane(lane_id)
        if lane.wait_line_ids:
            wait_line_lanes.add(lane_id)
        if lane.traffic_sign_ids:
            traffic_sign_lanes.add(lane_id)
        if len(vector_map.valid_related_lane_ids(lane_id, "successor")) > 1:
            branching_lanes.add(lane_id)
        if len(vector_map.valid_related_lane_ids(lane_id, "predecessor")) > 1:
            merging_lanes.add(lane_id)

    if valid_lane_ids:
        first_lane_id = valid_lane_ids[0]
        last_lane_id = valid_lane_ids[-1]

        for predecessor_id in vector_map.valid_related_lane_ids(
            first_lane_id, "predecessor"
        ):
            if len(
                vector_map.valid_related_lane_ids(predecessor_id, "successor")
            ) > 1:
                predecessor_branch_lanes.add(predecessor_id)

        for successor_id in vector_map.valid_related_lane_ids(
            last_lane_id, "successor"
        ):
            if len(
                vector_map.valid_related_lane_ids(successor_id, "predecessor")
            ) > 1:
                successor_merge_lanes.add(successor_id)

    level_a_reasons: list[str] = []
    if wait_line_lanes:
        level_a_reasons.append("sequence_contains_wait_line_lane")
    if branching_lanes:
        level_a_reasons.append("sequence_contains_branching_lane")
    if merging_lanes:
        level_a_reasons.append("sequence_contains_merging_lane")

    level_b_reasons: list[str] = []
    if predecessor_branch_lanes:
        level_b_reasons.append("start_lane_follows_branching_predecessor")
    if successor_merge_lanes:
        level_b_reasons.append("end_lane_precedes_merging_successor")

    if level_a_reasons:
        evidence_level = "A"
        evidence_reasons = level_a_reasons
    elif level_b_reasons:
        evidence_level = "B"
        evidence_reasons = level_b_reasons
    else:
        evidence_level = "C"
        evidence_reasons = ["no_direct_or_boundary_topology_evidence"]

    return {
        "valid_lane_sequence_length": len(valid_lane_ids),
        "missing_lane_ids": missing_lane_ids,
        "wait_line_lane_ids": sorted(wait_line_lanes),
        "traffic_sign_lane_ids": sorted(traffic_sign_lanes),
        "branching_lane_ids": sorted(branching_lanes),
        "merging_lane_ids": sorted(merging_lanes),
        "boundary_predecessor_branch_lane_ids": sorted(
            predecessor_branch_lanes
        ),
        "boundary_successor_merge_lane_ids": sorted(successor_merge_lanes),
        "level_a_direct_evidence": bool(level_a_reasons),
        "level_b_boundary_topology_evidence": bool(level_b_reasons),
        "junction_evidence_level": evidence_level,
        "junction_evidence_reasons": evidence_reasons,
        "traffic_sign_only": bool(traffic_sign_lanes)
        and evidence_level == "C",
    }



def branch_comparison_to_dict(comparison: Any) -> dict[str, Any]:
    return {
        "branch_lane_id": comparison.branch_lane_id,
        "natural_successor_lane_id": comparison.natural_successor_lane_id,
        "actual_successor_lane_id": comparison.actual_successor_lane_id,
        "actual_matches_natural": comparison.actual_matches_natural,
        "actual_relation_to_natural": comparison.actual_relation_to_natural,
    }


def natural_corridor_features(
    future_points: Sequence[Mapping[str, Any]],
    lane_feature: Mapping[str, Any],
    vector_map: VectorMapReader,
) -> dict[str, Any]:
    lane_sequence = [
        str(value) for value in lane_feature.get("compressed_lane_sequence", [])
    ]
    if not lane_sequence:
        return {
            "status": "unavailable",
            "reasons": ["empty_lane_sequence"],
            "turn_evidence_status": "fallback_keep_direction",
        }

    poses = [pose2d_from_pose_mapping(point["pose"]) for point in future_points]
    path_length_m = trajectory_path_length(future_points)
    if path_length_m <= 0.0:
        return {
            "status": "unavailable",
            "reasons": ["zero_future_path_length"],
            "turn_evidence_status": "fallback_keep_direction",
        }

    start_lane_id = lane_sequence[0]
    projection = vector_map.project_to_lane(
        start_lane_id, Point2D(poses[0].x, poses[0].y)
    )
    config = NaturalCorridorConfig()
    corridor = build_natural_lane_corridor(
        vector_map,
        start_lane_id,
        lookahead_distance_m=path_length_m,
        start_arc_length_m=projection.arc_length_m,
        config=config,
    )
    if not corridor.points:
        return {
            "status": "unavailable",
            "reasons": ["natural_corridor_has_no_points"],
            "future_path_length_m": path_length_m,
            "start_lane_id": start_lane_id,
            "start_projection_arc_length_m": projection.arc_length_m,
            "start_projection_distance_m": projection.distance_m,
            "lane_ids": list(corridor.lane_ids),
            "total_distance_m": corridor.total_distance_m,
            "terminated_reason": corridor.terminated_reason,
            "branch_decisions": [],
            "boundary_branch_comparisons": [],
            "actual_branch_comparisons": [],
            "relative_heading": {},
            "turn_evidence_status": "fallback_keep_direction",
            "reliable_directional_relations": [],
            "fallback_reasons": ["natural_corridor_has_no_points"],
        }

    actual_comparisons = compare_actual_lane_sequence(lane_sequence, corridor)
    boundary_comparisons = recover_boundary_branch_comparisons(
        vector_map, start_lane_id, config=config
    )
    relative = relative_heading_features(future_points, corridor)

    decision_records = []
    unreliable_reasons: list[str] = []
    for decision in corridor.branch_decisions:
        decision_records.append(
            {
                "branch_lane_id": decision.branch_lane_id,
                "chosen_natural_successor_lane_id": decision.chosen_successor_lane_id,
                "reliability_status": decision.reliability_status,
                "reliability_reasons": list(decision.reliability_reasons),
                "score_margin": decision.score_margin,
            }
        )
        if decision.reliability_status != "reliable":
            unreliable_reasons.extend(decision.reliability_reasons)

    all_comparisons = tuple(boundary_comparisons) + tuple(actual_comparisons)
    reliable_directional_relations = [
        comparison.actual_relation_to_natural
        for comparison in all_comparisons
        if comparison.actual_relation_to_natural
        in ("left_of_natural", "right_of_natural")
    ]
    ambiguous_relations = [
        comparison.actual_relation_to_natural
        for comparison in all_comparisons
        if comparison.actual_relation_to_natural
        in (
            "natural_continuation_uncertain",
            "ambiguous_relative_direction",
            "actual_successor_not_candidate",
        )
    ]
    uncertain = bool(ambiguous_relations) or bool(unreliable_reasons)
    unreliable_reasons.extend(
        "branch_comparison_" + relation for relation in ambiguous_relations
    )

    if uncertain:
        turn_evidence_status = "fallback_keep_direction"
    elif reliable_directional_relations:
        turn_evidence_status = "directional_branch_observed"
    else:
        turn_evidence_status = "natural_or_unobserved_keep_direction"

    scalar_relative = {
        key: value
        for key, value in relative.items()
        if isinstance(value, (int, float))
    }
    return {
        "status": "available",
        "future_path_length_m": path_length_m,
        "start_lane_id": start_lane_id,
        "start_projection_arc_length_m": projection.arc_length_m,
        "start_projection_distance_m": projection.distance_m,
        "lane_ids": list(corridor.lane_ids),
        "total_distance_m": corridor.total_distance_m,
        "terminated_reason": corridor.terminated_reason,
        "branch_decisions": decision_records,
        "boundary_branch_comparisons": [
            branch_comparison_to_dict(value) for value in boundary_comparisons
        ],
        "actual_branch_comparisons": [
            branch_comparison_to_dict(value) for value in actual_comparisons
        ],
        "relative_heading": scalar_relative,
        "turn_evidence_status": turn_evidence_status,
        "reliable_directional_relations": reliable_directional_relations,
        "fallback_reasons": sorted(set(unreliable_reasons)),
    }


def extract_lateral_features(
    future_points: Sequence[Mapping[str, Any]],
    lane_feature: Mapping[str, Any],
    vector_map: VectorMapReader,
    *,
    minimum_path_segment_length_m: float,
) -> dict[str, Any]:
    if len(future_points) < 2:
        raise ValueError("future trajectory requires at least two points")

    poses = [pose2d_from_pose_mapping(point["pose"]) for point in future_points]
    map_points = [Point2D(pose.x, pose.y) for pose in poses]
    pose_yaws = [pose.yaw for pose in poses]
    yaw_signed, yaw_absolute = signed_and_absolute_change(pose_yaws)
    initial_yaw = pose_yaws[0]
    ego_yaw_excursions = [
        abs(normalize_angle(yaw - initial_yaw)) for yaw in pose_yaws
    ]
    ego_maximum_yaw_excursion = max(ego_yaw_excursions)
    filtered_path = filtered_path_heading_features(
        map_points,
        minimum_segment_length_m=minimum_path_segment_length_m,
    )

    anchor = poses[0]
    final_local = map_point_to_anchor_ego(
        poses[-1].x,
        poses[-1].y,
        anchor.x,
        anchor.y,
        anchor.yaw,
    )

    lane_sequence = [
        str(value)
        for value in lane_feature.get("compressed_lane_sequence", [])
    ]
    topology = topology_evidence(vector_map, lane_sequence)

    start_projection = None
    end_projection = None
    if lane_sequence and vector_map.get_lane(lane_sequence[0]):
        start_projection = vector_map.project_to_lane(
            lane_sequence[0], map_points[0]
        )
    if lane_sequence and vector_map.get_lane(lane_sequence[-1]):
        end_projection = vector_map.project_to_lane(
            lane_sequence[-1], map_points[-1]
        )

    map_heading_change = None
    if start_projection is not None and end_projection is not None:
        map_heading_change = normalize_angle(
            end_projection.heading_rad - start_projection.heading_rad
        )

    transitions = [
        item
        for item in lane_feature.get("transitions", [])
        if isinstance(item, Mapping)
    ]
    relation_counts = Counter(
        str(item.get("relation", "")) for item in transitions
    )

    result = {
        "trajectory_point_count": len(future_points),
        "trajectory_yaw_signed_change_rad": yaw_signed,
        "trajectory_yaw_absolute_change_rad": yaw_absolute,
        "ego_total_yaw_change_rad": yaw_signed,
        "ego_maximum_yaw_excursion_rad": ego_maximum_yaw_excursion,
        "ego_total_absolute_yaw_change_rad": yaw_absolute,
        "final_relative_x_m": final_local.x,
        "final_relative_y_m": final_local.y,
        "final_relative_yaw_rad": normalize_angle(
            poses[-1].yaw - anchor.yaw
        ),
        "start_lane_heading_rad": (
            start_projection.heading_rad if start_projection else None
        ),
        "end_lane_heading_rad": (
            end_projection.heading_rad if end_projection else None
        ),
        "map_corridor_heading_change_rad": map_heading_change,
        "lane_sequence": lane_sequence,
        "transition_relation_counts": dict(relation_counts),
        "same_lane_only": len(lane_sequence) == 1,
        "successor_only": bool(transitions)
        and set(relation_counts) <= {"successor"},
        "contains_adjacent_transition": (
            relation_counts["left_adjacent"] > 0
            or relation_counts["right_adjacent"] > 0
        ),
        "topology": topology,
        "adjacent_transition_evidence": lane_feature.get(
            "adjacent_transition_evidence", []
        ),
        "lateral_quality_gate": lane_feature["lateral_quality_gate"],
    }
    result.update(filtered_path)
    result["natural_corridor"] = natural_corridor_features(
        future_points,
        lane_feature,
        vector_map,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--anchor-input", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument(
        "--lane-feature-input", type=Path, default=DEFAULT_LANE_FEATURES
    )
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument(
        "--minimum-path-segment-length-m",
        type=float,
        default=DEFAULT_MINIMUM_PATH_SEGMENT_LENGTH_M,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_path_segment_length_m <= 0.0:
        raise ValueError("--minimum-path-segment-length-m must be positive")

    anchors = read_jsonl(args.anchor_input.expanduser().resolve())
    lane_features = read_jsonl(args.lane_feature_input.expanduser().resolve())
    lane_by_anchor = {str(item["anchor_id"]): item for item in lane_features}
    if len(lane_by_anchor) != len(lane_features):
        raise ValueError("duplicate anchor_id in lane features")

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
            anchor
            for anchor in anchors
            if str(anchor["clip_id"]) in allowed
        ]

    anchors_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        anchors_by_clip[str(anchor["clip_id"])].append(anchor)

    output: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    started = time.monotonic()

    for clip_index, (clip_id, clip_anchors) in enumerate(
        anchors_by_clip.items(), start=1
    ):
        try:
            reader = DrivingClipReader(args.dataset_root / clip_id)
            vector_map = VectorMapReader.from_dict(reader.get_vector_map())
            for anchor in clip_anchors:
                anchor_id = str(anchor["anchor_id"])
                try:
                    lane_feature = lane_by_anchor[anchor_id]
                    horizon_ns = int(
                        anchor.get("future_horizon_ns", 3_000_000_000)
                    )
                    future = reader.get_future_ego_trajectory(
                        int(anchor["anchor_ns"]), horizon_ns=horizon_ns
                    )
                    if future is None:
                        raise RuntimeError("future trajectory unavailable")
                    lateral = extract_lateral_features(
                        future.points,
                        lane_feature,
                        vector_map,
                        minimum_path_segment_length_m=(
                            args.minimum_path_segment_length_m
                        ),
                    )
                    output.append(
                        {
                            "feature_format_version": FEATURE_FORMAT_VERSION,
                            "profiler_version": SCRIPT_VERSION,
                            "anchor_id": anchor_id,
                            "clip_id": clip_id,
                            "anchor_ns": int(anchor["anchor_ns"]),
                            "future_horizon_ns": horizon_ns,
                            "map_id": vector_map.map_id,
                            "lateral": lateral,
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

        if (
            clip_index == 1
            or clip_index % 50 == 0
            or clip_index == len(anchors_by_clip)
        ):
            print(
                f"Processed {clip_index}/{len(anchors_by_clip)} clips: "
                f"{clip_id}"
            )

    numeric_fields = (
        "trajectory_yaw_signed_change_rad",
        "trajectory_yaw_absolute_change_rad",
        "ego_total_yaw_change_rad",
        "ego_maximum_yaw_excursion_rad",
        "ego_total_absolute_yaw_change_rad",
        "filtered_path_signed_heading_change_rad",
        "filtered_path_absolute_heading_change_rad",
        "valid_path_segment_count",
        "rejected_path_segment_count",
        "valid_path_distance_m",
        "final_relative_x_m",
        "final_relative_y_m",
        "final_relative_yaw_rad",
    )
    map_heading_values = [
        float(item["lateral"]["map_corridor_heading_change_rad"])
        for item in output
        if item["lateral"]["map_corridor_heading_change_rad"] is not None
    ]
    junction_counts = Counter(
        item["lateral"]["topology"]["junction_evidence_level"]
        for item in output
    )
    profile_counts = Counter()
    for item in output:
        lateral = item["lateral"]
        if lateral["same_lane_only"]:
            profile_counts["same_lane_only"] += 1
        if lateral["successor_only"]:
            profile_counts["successor_only"] += 1
        if lateral["contains_adjacent_transition"]:
            profile_counts["contains_adjacent_transition"] += 1
        if lateral["path_heading_reliable"]:
            profile_counts["path_heading_reliable"] += 1
        natural = lateral["natural_corridor"]
        profile_counts[
            "natural_corridor_" + str(natural["status"])
        ] += 1
        profile_counts[
            "turn_evidence_" + str(natural["turn_evidence_status"])
        ] += 1

    level_distributions: dict[str, Any] = {}
    for level in ("A", "B", "C"):
        members = [
            item["lateral"]
            for item in output
            if item["lateral"]["topology"]["junction_evidence_level"]
            == level
        ]
        level_distributions[level] = {
            "anchor_count": len(members),
            "trajectory_yaw_signed_change_rad": distribution(
                float(item["trajectory_yaw_signed_change_rad"])
                for item in members
            ),
            "absolute_trajectory_yaw_change_rad": distribution(
                abs(float(item["trajectory_yaw_signed_change_rad"]))
                for item in members
            ),
            "map_corridor_heading_change_rad": distribution(
                float(item["map_corridor_heading_change_rad"])
                for item in members
                if item["map_corridor_heading_change_rad"] is not None
            ),
            "absolute_map_corridor_heading_change_rad": distribution(
                abs(float(item["map_corridor_heading_change_rad"]))
                for item in members
                if item["map_corridor_heading_change_rad"] is not None
            ),
        }

    summary = {
        "feature_format_version": FEATURE_FORMAT_VERSION,
        "profiler_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_anchor_file": str(args.anchor_input),
        "input_lane_feature_file": str(args.lane_feature_input),
        "elapsed_sec": time.monotonic() - started,
        "configuration": {
            "minimum_path_segment_length_m": (
                args.minimum_path_segment_length_m
            )
        },
        "processing": {
            "attempted_anchor_count": len(anchors),
            "successful_anchor_count": len(output),
            "failed_anchor_count": len(errors),
            "processing_errors": errors,
        },
        "profile_counts": dict(profile_counts),
        "junction_evidence_level_counts": dict(junction_counts),
        "distributions": {
            field: distribution(
                float(item["lateral"][field]) for item in output
            )
            for field in numeric_fields
        },
        "map_corridor_heading_change_rad": distribution(map_heading_values),
        "junction_level_distributions": level_distributions,
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
    print(
        "Feature SHA-256:",
        hashlib.sha256(feature_text.encode("utf-8")).hexdigest(),
    )
    print("Successful anchors:", len(output))
    print("Processing errors:", len(errors))
    print("Profile counts:", dict(profile_counts))
    print("Junction evidence levels:", dict(junction_counts))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
