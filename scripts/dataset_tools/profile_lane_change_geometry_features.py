#!/usr/bin/env python3
"""Profile parallel-corridor and in-progress lane-change evidence.

This is a diagnostic/profiling tool. It does not modify Meta-action labels.
It combines:
- existing lane-matching / adjacent evidence;
- raw future GT poses;
- VectorMap lane centerlines and adjacency;
- lane_change_geometry.py.

Use --anchor-id repeatedly for targeted validation, then --all for full-scale
profiling after the confirmed cases behave correctly.
"""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lane_change_geometry import (
    InProgressLaneChangeConfig,
    ParallelCorridorConfig,
    append_polyline,
    compare_parallel_corridors,
    distance_preference,
    in_progress_lane_change_evidence,
    polyline_length,
    project_to_polyline,
    slice_polyline_from_projection,
    truncate_polyline,
)
from project_paths import (
    ALPASIM_DATA_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

VERSION = "0.3.4"
FORMAT_VERSION = "0.3-draft"
ROOT = ALPASIM_DATA_ROOT
DEFAULT_META_FEATURE_INPUT = (
    INTERMEDIATE_ROOT / "meta_action_features_v0.2.jsonl"
)
DEFAULT_LATERAL_FEATURE_INPUT = (
    INTERMEDIATE_ROOT / "lateral_action_features_v0.3.jsonl"
)
DEFAULT_OUTPUT = (
    INTERMEDIATE_ROOT / "lane_change_geometry_features_v0.1.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "lane_change_geometry_feature_summary_v0.1.json"
)
DEFAULT_ANCHORS = (
    "test_clip_096_227094511211000",   # observed keep_direction
    "test_clip_023_11825028592000",    # observed turn_left
    "test_clip_512_2241491722265000",  # observed turn_right
    "test_clip_337_13921605926000",    # observed in-progress right lane change
    "test_clip_420_96272302038000",    # observed in-progress left lane change
)


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in result:
                raise ValueError(f"duplicate anchor_id at {path}:{line_number}: {anchor_id}")
            result[anchor_id] = record
    return result


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


def plain_xy(point: Any) -> dict[str, float]:
    if isinstance(point, Mapping):
        if "position" in point:
            return plain_xy(point["position"])
        return {"x": float(point["x"]), "y": float(point["y"])}
    if hasattr(point, "position"):
        return plain_xy(point.position)
    return {"x": float(point.x), "y": float(point.y)}


def pose_xy_yaw(pose: Any) -> tuple[dict[str, float], float]:
    if isinstance(pose, Mapping):
        return plain_xy(pose), float(pose.get("yaw_rad", pose.get("yaw", 0.0)))
    return plain_xy(pose), float(getattr(pose, "yaw_rad", getattr(pose, "yaw", 0.0)))


def lane_centerline(lane: Any) -> Sequence[Any]:
    value = getattr(lane, "centerline", None)
    if value is None and isinstance(lane, Mapping):
        value = lane["centerline"]
    if isinstance(value, Mapping):
        value = value["points"]
    if hasattr(value, "points"):
        value = value.points
    return value


def lane_ids(lane: Any, name: str) -> tuple[str, ...]:
    value = getattr(lane, name, None)
    if value is None and isinstance(lane, Mapping):
        value = lane.get(name, ())
    return tuple(str(item) for item in (value or ()))


def find_lane_matching(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if "lane_matching" in record:
        return record["lane_matching"]
    lateral = record.get("lateral", {})
    if "lane_matching" in lateral:
        return lateral["lane_matching"]
    return lateral


def transitions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    lane_matching = find_lane_matching(record)
    return list(lane_matching.get("transitions", record.get("transitions", [])) or [])


def compressed_sequence(record: Mapping[str, Any]) -> list[str]:
    lane_matching = find_lane_matching(record)
    value = lane_matching.get(
        "compressed_lane_sequence",
        record.get("compressed_lane_sequence", []),
    )
    return [str(item) for item in (value or [])]


def adjacent_evidence(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    lane_matching = find_lane_matching(record)
    return list(lane_matching.get("adjacent_transition_evidence", []) or [])


def truthy_flag(value: Any, keys: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in keys and child is True:
                return True
            if truthy_flag(child, keys):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(truthy_flag(item, keys) for item in value)
    return False


def numeric_values(value: Any, keys: set[str]) -> list[float]:
    result: list[float] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in keys and isinstance(child, (int, float)):
                result.append(float(child))
            result.extend(numeric_values(child, keys))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            result.extend(numeric_values(child, keys))
    return result


def adjacent_transition_stable(record: Mapping[str, Any]) -> tuple[bool, list[str]]:
    evidence = adjacent_evidence(record)
    reasons: list[str] = []
    if not evidence:
        reasons.append("adjacent_persistence_evidence_unavailable")
        return False, reasons
    if truthy_flag(
        evidence,
        {"return_to_source", "returns_to_source", "opposite_adjacent", "ambiguous"},
    ):
        reasons.append("ambiguous_or_returning_adjacent_transition")
    corridor_counts = numeric_values(evidence, {"corridor_point_count"})
    corridor_durations = numeric_values(evidence, {"corridor_duration_sec"})
    stable_by_points = bool(corridor_counts) and max(corridor_counts) >= 4.0
    stable_by_time = bool(corridor_durations) and max(corridor_durations) >= 0.3
    if not (stable_by_points or stable_by_time):
        reasons.append("adjacent_target_not_persistent")
    return not reasons, reasons


def transition_direction(relation: str) -> str | None:
    if relation == "left_adjacent":
        return "left"
    if relation == "right_adjacent":
        return "right"
    return None


def build_downstream_corridor(
    vector_map: Any,
    start_lane_id: str,
    start_point: Mapping[str, float],
    lookahead_m: float = 35.0,
    maximum_lanes: int = 8,
) -> tuple[tuple[Any, ...], tuple[str, ...], str]:
    """Build a downstream centerline from a point, following unique successors.

    The builder does not use future GT to choose at a branch. It stops when the
    current lane has zero or multiple successors.
    """
    lane = vector_map.require_lane(start_lane_id)
    first = slice_polyline_from_projection(
        lane_centerline(lane), start_point, lookahead_m
    )
    if len(first) < 2:
        raise ValueError(f"lane {start_lane_id} has no downstream centerline")
    corridor = tuple(first)
    lane_path = [str(start_lane_id)]
    current = lane
    termination = "lookahead_reached" if polyline_length(corridor) >= lookahead_m else ""
    visited = {str(start_lane_id)}
    while polyline_length(corridor) < lookahead_m and len(lane_path) < maximum_lanes:
        successors = lane_ids(current, "successor_ids")
        if not successors:
            termination = "no_successor"
            break
        if len(successors) != 1:
            termination = "successor_branch"
            break
        successor_id = successors[0]
        if successor_id in visited:
            termination = "cycle"
            break
        successor = vector_map.get_lane(successor_id)
        if successor is None:
            termination = "missing_successor"
            break
        corridor = append_polyline(corridor, lane_centerline(successor))
        corridor = truncate_polyline(corridor, lookahead_m)
        lane_path.append(successor_id)
        visited.add(successor_id)
        current = successor
    if not termination:
        termination = "maximum_lanes" if len(lane_path) >= maximum_lanes else "lookahead_reached"
    return corridor, tuple(lane_path), termination


def lane_start_indices(sequence: Sequence[str], transition_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return the first observed GT point index for each compressed lane."""
    result: dict[str, int] = {}
    if sequence:
        result[str(sequence[0])] = 0
    for row in transition_rows:
        target = str(row.get("target_lane_id", ""))
        if target and target not in result:
            result[target] = int(row.get("target_point_index", 0))
    return result


def turn_direction_from_yaw(yaws: Sequence[float], start_index: int, minimum_deg: float = 3.0) -> str | None:
    if not yaws or start_index >= len(yaws) - 1:
        return None
    from lane_change_geometry import angle_difference
    change = math.degrees(angle_difference(yaws[start_index], yaws[-1]))
    if change >= minimum_deg:
        return "left"
    if change <= -minimum_deg:
        return "right"
    return None



def trajectory_length_xy(values: Sequence[Mapping[str, float]]) -> float:
    return sum(
        math.hypot(second["x"] - first["x"], second["y"] - first["y"])
        for first, second in zip(values, values[1:])
    )


def corridor_heading_residual(
    trajectory_xy: Sequence[Mapping[str, float]],
    trajectory_yaw: Sequence[float],
    corridor: Sequence[Any],
) -> dict[str, float]:
    """Compare observed ego heading change with a downstream corridor."""
    from lane_change_geometry import angle_difference, sample_polyline

    if len(trajectory_xy) < 2 or len(trajectory_yaw) < 2:
        raise ValueError("trajectory too short for heading residual")
    travelled = trajectory_length_xy(trajectory_xy)
    corridor_distance = min(travelled, polyline_length(corridor))
    _, start_heading = sample_polyline(corridor, 0.0)
    _, end_heading = sample_polyline(corridor, corridor_distance)
    ego_change = math.degrees(
        angle_difference(trajectory_yaw[0], trajectory_yaw[-1])
    )
    corridor_change = math.degrees(
        angle_difference(start_heading, end_heading)
    )
    return {
        "ego_heading_change_deg": ego_change,
        "corridor_heading_change_deg": corridor_change,
        "heading_change_residual_deg": ego_change - corridor_change,
        "travelled_distance_m": travelled,
        "corridor_distance_m": corridor_distance,
    }


def spatial_parallel_lane_candidates(
    vector_map: Any,
    xy: Sequence[Mapping[str, float]],
    yaw: Sequence[float],
    excluded_lane_ids: set[str],
    *,
    radius_m: float = 7.0,
    maximum_heading_error_deg: float = 20.0,
    limit_per_query: int = 12,
) -> list[tuple[str, int]]:
    """Find nearby same-direction lanes when adjacency topology is absent."""
    if len(xy) < 2:
        return []
    start_index = max(0, len(xy) // 3)
    query_indices = sorted(set(
        list(range(start_index, len(xy), max(1, (len(xy) - start_index) // 6)))
        + [len(xy) - 1]
    ))
    first_seen: dict[str, int] = {}
    for index in query_indices:
        nearby = vector_map.find_nearby_lanes(
            xy[index]["x"],
            xy[index]["y"],
            radius_m=radius_m,
            yaw_rad=yaw[index],
            maximum_heading_error_rad=math.radians(maximum_heading_error_deg),
            limit=limit_per_query,
        )
        for candidate in nearby:
            lane_id = str(candidate.lane_id)
            if lane_id in excluded_lane_ids:
                continue
            first_seen.setdefault(lane_id, index)
    return sorted(first_seen.items(), key=lambda item: (item[1], item[0]))


def spatial_candidate_direction(
    source_corridor: Sequence[Any],
    target_corridor: Sequence[Any],
) -> str | None:
    """Determine target side from target start projected on source corridor."""
    from lane_change_geometry import project_to_polyline, sample_polyline

    target_start, _ = sample_polyline(target_corridor, 0.0)
    projection = project_to_polyline(source_corridor, target_start)
    if projection.signed_offset_m > 0.5:
        return "left"
    if projection.signed_offset_m < -0.5:
        return "right"
    return None


def spatial_lane_fallback_enabled() -> bool:
    value = os.environ.get("ENABLE_SPATIAL_LANE_FALLBACK", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def analyze_anchor(
    feature_record: Mapping[str, Any],
    reader: Any,
    vector_map: Any,
    parallel_config: ParallelCorridorConfig,
    in_progress_config: InProgressLaneChangeConfig,
) -> dict[str, Any]:
    from lane_matcher import trajectory_poses_from_gt_points

    anchor_id = str(feature_record["anchor_id"])
    anchor_ns = int(feature_record["anchor_ns"])
    future = reader.get_future_ego_trajectory(anchor_ns)
    if future is None or len(future.points) < 2:
        raise ValueError("future trajectory unavailable or too short")
    poses = trajectory_poses_from_gt_points(future.points)
    xy: list[dict[str, float]] = []
    yaw: list[float] = []
    for pose in poses:
        point, heading = pose_xy_yaw(pose)
        xy.append(point)
        yaw.append(heading)

    sequence = compressed_sequence(feature_record)
    transition_rows = transitions(feature_record)
    start_indices = lane_start_indices(sequence, transition_rows)
    stable, stability_reasons = adjacent_transition_stable(feature_record)
    observed_results: list[dict[str, Any]] = []

    for transition in transition_rows:
        relation = str(transition.get("relation", ""))
        adjacency_direction = transition_direction(relation)
        if adjacency_direction is None:
            continue
        source_id = str(transition["source_lane_id"])
        target_id = str(transition["target_lane_id"])
        source_index = max(0, min(len(xy) - 1, int(transition.get("source_point_index", 0))))
        target_index = max(0, min(len(xy) - 1, int(transition.get("target_point_index", source_index))))
        decision_point = xy[source_index]
        try:
            (
                source_corridor,
                source_path,
                source_termination,
            ) = build_downstream_corridor(
                vector_map,
                source_id,
                decision_point,
            )

            (
                target_corridor,
                target_path,
                target_termination,
            ) = build_downstream_corridor(
                vector_map,
                target_id,
                xy[target_index],
            )

        except ValueError:
            # An observed adjacent transition may occur at the
            # end of a Lane, leaving no downstream centerline
            # available for geometric arbitration. Skip this
            # individual candidate without failing the Anchor.
            continue

        parallel = compare_parallel_corridors(
            source_corridor,
            target_corridor,
            parallel_config,
        )
        preference = distance_preference(xy[source_index:], source_corridor, target_corridor)
        source_heading_residual = corridor_heading_residual(
            xy[source_index:], yaw[source_index:], source_corridor
        )
        target_heading_residual = corridor_heading_residual(
            xy[source_index:], yaw[source_index:], target_corridor
        )
        target_residual = abs(target_heading_residual["heading_change_residual_deg"])
        source_residual = abs(source_heading_residual["heading_change_residual_deg"])
        lateral_section = feature_record.get("lateral", feature_record)
        topology_section = lateral_section.get("topology", {})
        junction_level = str(
            topology_section.get(
                "junction_evidence_level",
                lateral_section.get("junction_evidence_level", "C"),
            )
        )
        heading_threshold_deg = 8.0 if junction_level in {"A", "B"} else 12.0
        large_actual_turn = (
            abs(target_heading_residual["ego_heading_change_deg"])
            >= heading_threshold_deg
        )
        source_not_followed = source_residual >= heading_threshold_deg
        target_not_followed = target_residual >= heading_threshold_deg
        if (
            parallel.same_direction_parallel
            and large_actual_turn
            and source_not_followed
            and target_not_followed
        ):
            actual_turn_direction = turn_direction_from_yaw(yaw, source_index)
            interpretation = (
                f"turn_{actual_turn_direction}_candidate"
                if actual_turn_direction is not None
                else "turn_candidate_direction_unclear"
            )
            interpretation_reason = "ego_heading_diverges_from_parallel_target_corridor"
        elif parallel.same_direction_parallel and stable and preference.confirmed_switch:
            interpretation = f"change_lane_{adjacency_direction}"
            interpretation_reason = "parallel_corridor_and_confirmed_distance_preference_switch"
        elif parallel.same_direction_parallel:
            interpretation = "keep_direction"
            interpretation_reason = "parallel_corridor_without_confirmed_lane_preference_switch"
        else:
            actual_turn_direction = turn_direction_from_yaw(yaw, source_index)
            interpretation = (
                f"turn_{actual_turn_direction}_candidate"
                if actual_turn_direction is not None
                else "turn_candidate_direction_unclear"
            )
            interpretation_reason = "adjacent_target_is_not_parallel_downstream_corridor"
        observed_results.append({
            "direction_from_adjacency": adjacency_direction,
            "source_lane_id": source_id,
            "target_lane_id": target_id,
            "source_point_index": source_index,
            "target_point_index": target_index,
            "source_corridor_lane_ids": source_path,
            "target_corridor_lane_ids": target_path,
            "source_corridor_termination": source_termination,
            "target_corridor_termination": target_termination,
            "transition_stable": stable,
            "stability_reasons": stability_reasons,
            "parallel_corridor": parallel.to_dict(),
            "distance_preference": preference.to_dict(),
            "source_heading_residual": source_heading_residual,
            "target_heading_residual": target_heading_residual,
            "junction_evidence_level": junction_level,
            "heading_residual_threshold_deg": heading_threshold_deg,
            "interpretation": interpretation,
            "interpretation_reason": interpretation_reason,
        })

    in_progress_results: list[dict[str, Any]] = []
    if sequence and not observed_results:
        seen_pairs: set[tuple[str, str, str]] = set()
        for source_id in sequence:
            source_lane = vector_map.get_lane(source_id)
            if source_lane is None:
                continue
            source_index = max(0, min(len(xy) - 2, start_indices.get(source_id, 0)))
            point = xy[source_index]
            for direction, field_name in (
                ("left", "left_adjacent_ids"),
                ("right", "right_adjacent_ids"),
            ):
                for target_id in lane_ids(source_lane, field_name):
                    key = (source_id, target_id, direction)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    target_lane = vector_map.get_lane(target_id)
                    if target_lane is None:
                        continue
                    try:
                        source_corridor, source_path, source_termination = build_downstream_corridor(
                            vector_map, source_id, point
                        )
                        target_corridor, target_path, target_termination = build_downstream_corridor(
                            vector_map, target_id, point
                        )
                    except ValueError as exc:
                        in_progress_results.append({
                            "direction": direction,
                            "source_lane_id": source_id,
                            "target_lane_id": target_id,
                            "candidate": False,
                            "reasons": ["corridor_build_failed", str(exc)],
                        })
                        continue
                    parallel = compare_parallel_corridors(
                        source_corridor, target_corridor, parallel_config
                    )
                    result: dict[str, Any] = {
                        "direction": direction,
                        "source_lane_id": source_id,
                        "target_lane_id": target_id,
                        "source_point_index": source_index,
                        "source_corridor_lane_ids": source_path,
                        "target_corridor_lane_ids": target_path,
                        "source_corridor_termination": source_termination,
                        "target_corridor_termination": target_termination,
                        "parallel_corridor": parallel.to_dict(),
                    }
                    if not parallel.same_direction_parallel:
                        result.update({
                            "candidate": False,
                            "reasons": ["adjacent_corridor_not_parallel_same_direction"],
                        })
                    else:
                        evidence = in_progress_lane_change_evidence(
                            xy[source_index:], yaw[source_index:],
                            source_corridor, target_corridor,
                            direction, in_progress_config,
                        )
                        result.update(evidence.to_dict())
                    in_progress_results.append(result)

    if (
        spatial_lane_fallback_enabled()
        and sequence
        and not observed_results
        and not any(
            bool(item.get("candidate"))
            for item in in_progress_results
        )
    ):
        excluded_lane_ids = set(sequence)
        for lane_id in list(sequence):
            lane = vector_map.get_lane(lane_id)
            if lane is not None:
                excluded_lane_ids.update(lane_ids(lane, "successor_ids"))
                excluded_lane_ids.update(lane_ids(lane, "predecessor_ids"))
        for target_id, query_index in spatial_parallel_lane_candidates(
            vector_map, xy, yaw, excluded_lane_ids
        ):
            source_id = sequence[min(len(sequence) - 1, 1 if query_index >= len(xy) // 2 else 0)]
            try:
                source_corridor, source_path, source_termination = build_downstream_corridor(
                    vector_map, source_id, xy[query_index]
                )
                target_corridor, target_path, target_termination = build_downstream_corridor(
                    vector_map, target_id, xy[query_index]
                )
                parallel = compare_parallel_corridors(
                    source_corridor, target_corridor, parallel_config
                )
            except ValueError:
                continue
            if not parallel.same_direction_parallel:
                continue
            direction = spatial_candidate_direction(source_corridor, target_corridor)
            if direction is None:
                continue
            evidence = in_progress_lane_change_evidence(
                xy[query_index:], yaw[query_index:],
                source_corridor, target_corridor,
                direction, in_progress_config,
            )
            item = evidence.to_dict()
            item.update({
                "evidence_source": "spatial_parallel_corridor",
                "source_lane_id": source_id,
                "target_lane_id": target_id,
                "source_point_index": query_index,
                "source_corridor_lane_ids": source_path,
                "target_corridor_lane_ids": target_path,
                "source_corridor_termination": source_termination,
                "target_corridor_termination": target_termination,
                "parallel_corridor": parallel.to_dict(),
            })
            in_progress_results.append(item)

    candidates = [item for item in in_progress_results if bool(item.get("candidate"))]
    inferred = None
    if len(candidates) == 1:
        inferred = f"change_lane_{candidates[0]['direction']}"
    elif len(candidates) > 1:
        directions = {item["direction"] for item in candidates}
        inferred = (
            f"change_lane_{next(iter(directions))}"
            if len(directions) == 1
            else "ambiguous_in_progress_lane_change"
        )

    return {
        "feature_format_version": FORMAT_VERSION,
        "profiler_version": VERSION,
        "anchor_id": anchor_id,
        "clip_id": str(feature_record["clip_id"]),
        "anchor_ns": anchor_ns,
        "lane_sequence": sequence,
        "trajectory_point_count": len(xy),
        "observed_adjacent_transitions": observed_results,
        "in_progress_candidates": in_progress_results,
        "inferred_in_progress_action": inferred,
    }



def merge_feature_records(
    meta_record: Mapping[str, Any],
    lateral_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Join lane-matching and lateral records with strict identity checks."""
    for key in ("anchor_id", "clip_id", "anchor_ns", "future_horizon_ns"):
        if meta_record.get(key) != lateral_record.get(key):
            raise ValueError(
                f"feature identity mismatch for {key}: "
                f"{meta_record.get(key)!r} != {lateral_record.get(key)!r}"
            )
    lateral = lateral_record.get("lateral")
    if not isinstance(lateral, Mapping):
        raise ValueError("lateral feature record has no lateral object")
    lane_matching = meta_record.get("lane_matching")
    if not isinstance(lane_matching, Mapping):
        raise ValueError("meta feature record has no lane_matching object")
    merged = dict(meta_record)
    merged["lateral"] = dict(lateral)
    merged["map_id"] = lateral_record.get("map_id")
    merged["source_feature_versions"] = {
        "meta": meta_record.get("feature_format_version"),
        "lateral": lateral_record.get("feature_format_version"),
    }
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-feature-input",
        type=Path,
        default=DEFAULT_META_FEATURE_INPUT,
    )
    parser.add_argument(
        "--lateral-feature-input",
        type=Path,
        default=DEFAULT_LATERAL_FEATURE_INPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--anchor-id", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from clip_reader import DrivingClipReader
    from vector_map_reader import VectorMapReader
    meta_index = read_jsonl_index(args.meta_feature_input)
    lateral_index = read_jsonl_index(args.lateral_feature_input)
    meta_ids = set(meta_index)
    lateral_ids = set(lateral_index)
    if meta_ids != lateral_ids:
        raise ValueError(
            "dual input anchor sets differ; "
            f"missing lateral={sorted(meta_ids - lateral_ids)[:10]}, "
            f"missing meta={sorted(lateral_ids - meta_ids)[:10]}"
        )
    requested = list(args.anchor_id)
    if not requested and not args.all:
        requested = list(DEFAULT_ANCHORS)
    if args.all:
        requested = sorted(meta_index)
    missing = [anchor_id for anchor_id in requested if anchor_id not in meta_index]
    if missing:
        raise KeyError(f"anchors absent from dual inputs: {missing}")

    by_clip: dict[str, list[Mapping[str, Any]]] = {}
    for anchor_id in requested:
        record = merge_feature_records(
            meta_index[anchor_id],
            lateral_index[anchor_id],
        )
        by_clip.setdefault(str(record["clip_id"]), []).append(record)

    parallel_config = ParallelCorridorConfig()
    in_progress_config = InProgressLaneChangeConfig()
    output: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for clip_id in sorted(by_clip):
        try:
            reader = DrivingClipReader(ROOT / clip_id)
            vector_map = VectorMapReader.from_dict(reader.get_vector_map())
        except Exception as exc:
            for record in by_clip[clip_id]:
                errors.append({
                    "anchor_id": str(record["anchor_id"]),
                    "clip_id": clip_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })
            continue
        for record in by_clip[clip_id]:
            try:
                output.append(analyze_anchor(
                    record, reader, vector_map, parallel_config, in_progress_config
                ))
            except Exception as exc:
                errors.append({
                    "anchor_id": str(record["anchor_id"]),
                    "clip_id": clip_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

    interpretation_counts = Counter()
    in_progress_counts = Counter()
    for record in output:
        for item in record["observed_adjacent_transitions"]:
            interpretation_counts[item["interpretation"]] += 1
        in_progress_counts[str(record["inferred_in_progress_action"])] += 1
    summary = {
        "feature_format_version": FORMAT_VERSION,
        "profiler_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_anchor_count": len(requested),
        "successful_anchor_count": len(output),
        "failed_anchor_count": len(errors),
        "errors": errors,
        "observed_adjacent_interpretation_counts": dict(interpretation_counts),
        "in_progress_action_counts": dict(in_progress_counts),
        "parallel_config": parallel_config.__dict__,
        "in_progress_config": in_progress_config.__dict__,
    }
    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output
    )
    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )
    print("Feature output:", args.output)
    print("Summary output:", args.summary_output)
    print("Feature SHA-256:", hashlib.sha256(output_text.encode()).hexdigest())
    print("Successful anchors:", len(output))
    print("Processing errors:", len(errors))
    print("Observed adjacent interpretations:", dict(interpretation_counts))
    print("In-progress actions:", dict(in_progress_counts))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
