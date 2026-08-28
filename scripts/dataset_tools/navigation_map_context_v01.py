#!/usr/bin/env python3
"""Pure Step 6 map-context helpers for Anchor-time local routes."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from coordinate_utils import Point2D, Pose2D, anchor_ego_point_to_map
from lane_matcher import LaneMatcher, TrajectoryPose
from natural_lane_corridor import (
    NaturalCorridorConfig,
    build_natural_lane_corridor,
    compare_actual_lane_sequence,
)
from vector_map_reader import VectorMapReader

FEATURE_FORMAT_VERSION = '0.1-draft'
PROFILER_VERSION = '0.1.0'


def local_route_to_map_trajectory(
    points: Sequence[tuple[float, float]],
    anchor_pose: Pose2D,
    *,
    stamp_ns: int,
) -> tuple[TrajectoryPose, ...]:
    if len(points) < 2:
        raise ValueError('local route requires at least two points')
    mapped = [
        anchor_ego_point_to_map(x, y, anchor_pose.x, anchor_pose.y, anchor_pose.yaw)
        for x, y in points
    ]
    headings: list[float] = []
    for index, point in enumerate(mapped):
        if index + 1 < len(mapped):
            other = mapped[index + 1]
            heading = math.atan2(other.y - point.y, other.x - point.x)
        else:
            other = mapped[index - 1]
            heading = math.atan2(point.y - other.y, point.x - other.x)
        headings.append(heading)
    return tuple(
        TrajectoryPose(stamp_ns=stamp_ns + index, x=point.x, y=point.y, yaw=headings[index])
        for index, point in enumerate(mapped)
    )


def topology_context(
    vector_map: VectorMapReader,
    lane_sequence: Sequence[str],
) -> dict[str, Any]:
    valid = [lane_id for lane_id in lane_sequence if vector_map.get_lane(lane_id)]
    wait_lines: set[str] = set()
    branching: set[str] = set()
    merging: set[str] = set()
    traffic_signs: set[str] = set()
    for lane_id in valid:
        lane = vector_map.require_lane(lane_id)
        if lane.wait_line_ids:
            wait_lines.add(lane_id)
        if lane.traffic_sign_ids:
            traffic_signs.add(lane_id)
        if len(vector_map.valid_related_lane_ids(lane_id, 'successor')) > 1:
            branching.add(lane_id)
        if len(vector_map.valid_related_lane_ids(lane_id, 'predecessor')) > 1:
            merging.add(lane_id)
    direct = bool(wait_lines or branching or merging)
    return {
        'valid_lane_sequence_length': len(valid),
        'wait_line_lane_ids': sorted(wait_lines),
        'traffic_sign_lane_ids': sorted(traffic_signs),
        'branching_lane_ids': sorted(branching),
        'merging_lane_ids': sorted(merging),
        'intersection_context': direct,
        'branch_context': bool(branching),
    }


def extract_navigation_map_context(
    *,
    vector_map: VectorMapReader,
    anchor_pose: Pose2D,
    local_points: Sequence[tuple[float, float]],
    anchor_ns: int,
    lookahead_distance_m: float,
) -> dict[str, Any]:
    trajectory = local_route_to_map_trajectory(local_points, anchor_pose, stamp_ns=anchor_ns)
    match = LaneMatcher(vector_map).match(trajectory)
    lane_sequence = tuple(str(value) for value in match.compressed_lane_sequence)
    if not lane_sequence:
        return {
            'quality_status': 'unknown',
            'reasons': ['route_lane_sequence_empty'],
            'lane_match': match.to_dict(),
            'route_lane_sequence': [],
            'topology': {},
            'natural_corridor': {},
            'branch_comparisons': [],
        }

    first_pose = trajectory[0]
    projection = vector_map.project_to_lane(
        lane_sequence[0], Point2D(first_pose.x, first_pose.y)
    )
    corridor = build_natural_lane_corridor(
        vector_map,
        lane_sequence[0],
        lookahead_distance_m=lookahead_distance_m,
        start_arc_length_m=projection.arc_length_m,
        config=NaturalCorridorConfig(),
    )
    comparisons = compare_actual_lane_sequence(lane_sequence, corridor)
    comparison_records = [
        {
            'branch_lane_id': value.branch_lane_id,
            'natural_successor_lane_id': value.natural_successor_lane_id,
            'route_successor_lane_id': value.actual_successor_lane_id,
            'route_matches_natural': value.actual_matches_natural,
            'route_relation_to_natural': value.actual_relation_to_natural,
        }
        for value in comparisons
    ]
    relation_counts = Counter(
        value.actual_relation_to_natural for value in comparisons
    )
    unreliable = [
        reason
        for decision in corridor.branch_decisions
        if decision.reliability_status != 'reliable'
        for reason in decision.reliability_reasons
    ]
    directional = [
        value.actual_relation_to_natural
        for value in comparisons
        if value.actual_relation_to_natural in {'left_of_natural', 'right_of_natural'}
    ]
    topology = topology_context(vector_map, lane_sequence)

    reasons: list[str] = []
    status = 'usable'
    if match.matched_fraction < 0.8:
        status = 'unknown'
        reasons.append('route_lane_match_fraction_below_0_8')
    if not corridor.points:
        status = 'unknown'
        reasons.append('natural_corridor_has_no_points')
    if unreliable:
        reasons.extend('natural_branch_' + value for value in unreliable)

    return {
        'quality_status': status,
        'reasons': sorted(set(reasons)),
        'lane_match': {
            'matched_fraction': match.matched_fraction,
            'mean_distance_m': match.mean_distance_m,
            'maximum_distance_m': match.maximum_distance_m,
            'mean_heading_error_rad': match.mean_heading_error_rad,
            'confidence': match.confidence,
        },
        'route_lane_sequence': list(lane_sequence),
        'transition_relation_counts': dict(Counter(
            value.relation for value in match.transitions
        )),
        'topology': topology,
        'natural_corridor': {
            'start_lane_id': corridor.start_lane_id,
            'lane_ids': list(corridor.lane_ids),
            'total_distance_m': corridor.total_distance_m,
            'terminated_reason': corridor.terminated_reason,
            'branch_decision_count': len(corridor.branch_decisions),
            'unreliable_reasons': sorted(set(unreliable)),
        },
        'branch_comparisons': comparison_records,
        'branch_relation_counts': dict(relation_counts),
        'reliable_directional_relations': directional if not unreliable else [],
    }
