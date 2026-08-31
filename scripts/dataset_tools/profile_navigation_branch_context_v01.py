#!/usr/bin/env python3
"""Profile first observed Route branch and upcoming-intersection distance.

Uses only the navigation route available at or before the Anchor, Anchor-time
Ego pose, and VectorMap. It does not use future Ego motion or Meta-action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clip_reader import DrivingClipReader
from coordinate_utils import Point2D
from lane_matcher import LaneMatcher
from natural_lane_corridor import (
    NaturalCorridorConfig,
    assess_branch_candidate_reliability,
    build_natural_lane_corridor,
    compare_actual_lane_sequence,
)
from natural_corridor_family_guard_v01 import (
    FAMILY_GUARD_VERSION,
    evaluate_direction_family_guard,
)
from navigation_map_context_v01 import local_route_to_map_trajectory
from navigation_route_features_v01 import valid_local_points
from vector_map_reader import VectorMapReader

DATA_ROOT = Path('/home/lab/data_from_alpasim')
ANN = DATA_ROOT / 'annotations/v0.1-draft'
DEFAULT_KEYFRAMES = ANN / 'keyframes.jsonl'
DEFAULT_OUTPUT = ANN / 'intermediate/navigation_branch_context_v0.1.jsonl'
DEFAULT_SUMMARY = DATA_ROOT / 'reports/navigation_branch_context_summary_v0.1.json'
FORMAT_VERSION = '0.1-draft'
PROFILER_VERSION = '0.1.1'


def read_records(path: Path) -> list[dict[str, Any]]:
    result = []
    seen = set()
    with path.open('r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record['anchor_id'])
            if anchor_id in seen:
                raise ValueError(f'duplicate Anchor at {path}:{line_number}: {anchor_id}')
            seen.add(anchor_id)
            result.append(record)
    return sorted(result, key=lambda item: (str(item['clip_id']), int(item['anchor_ns']), str(item['anchor_id'])))


def cumulative_distances(points: list[tuple[float, float]]) -> list[float]:
    distances = [0.0]
    for first, second in zip(points, points[1:]):
        distances.append(distances[-1] + math.hypot(second[0] - first[0], second[1] - first[1]))
    return distances


def lane_has_intersection_evidence(vector_map: VectorMapReader, lane_id: str) -> tuple[bool, list[str]]:
    lane = vector_map.require_lane(lane_id)
    reasons = []
    if lane.wait_line_ids:
        reasons.append('wait_line')
    if len(vector_map.valid_related_lane_ids(lane_id, 'successor')) > 1:
        reasons.append('branching')
    if len(vector_map.valid_related_lane_ids(lane_id, 'predecessor')) > 1:
        reasons.append('merging')
    return bool(reasons), reasons


def atomic_write(path: Path, text: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f'output exists: {path}; use --force')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyframe-input', type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument('--dataset-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--summary-output', type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument('--maximum-route-age-ns', type=int, default=200_000_000)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyframes = read_records(args.keyframe_input)
    readers: dict[str, DrivingClipReader] = {}
    maps: dict[str, VectorMapReader] = {}
    output = []
    status_counts = Counter()
    first_relation_counts = Counter()
    first_reliability_counts = Counter()
    intersection_distance_buckets = Counter()
    branch_distance_buckets = Counter()
    error_counts = Counter()

    for index, keyframe in enumerate(keyframes, start=1):
        clip_id = str(keyframe['clip_id'])
        anchor_ns = int(keyframe['anchor_ns'])
        reader = readers.get(clip_id)
        if reader is None:
            reader = DrivingClipReader(args.dataset_root / clip_id)
            readers[clip_id] = reader
            maps[clip_id] = VectorMapReader.from_dict(reader.get_vector_map())
        vector_map = maps[clip_id]
        timed = reader.get_navigation_route_at(anchor_ns, maximum_age_ns=args.maximum_route_age_ns)
        ego_state = reader.get_ego_state_at(anchor_ns)
        ego_pose = ego_state.pose if ego_state is not None else None
        record = {
            'feature_format_version': FORMAT_VERSION,
            'profiler_version': PROFILER_VERSION,
            'anchor_id': str(keyframe['anchor_id']),
            'clip_id': clip_id,
            'anchor_ns': anchor_ns,
            'route_time_policy': 'at_or_before_anchor',
        }
        if timed is None or ego_pose is None:
            reasons = []
            if timed is None:
                reasons.append('recent_route_unavailable')
            if ego_pose is None:
                reasons.append('anchor_ego_pose_unavailable')
            record.update({'quality_status': 'unknown', 'reasons': reasons, 'branch_context': {}})
            error_counts.update(reasons)
            status_counts['unknown'] += 1
            output.append(record)
            continue

        local_points = valid_local_points(timed.message)
        try:
            trajectory = local_route_to_map_trajectory(local_points, ego_pose, stamp_ns=anchor_ns)
            match = LaneMatcher(vector_map).match(trajectory)
            lane_sequence = tuple(str(value) for value in match.compressed_lane_sequence)
            if not lane_sequence:
                raise ValueError('route_lane_sequence_empty')
            distances = cumulative_distances(local_points)
            first_pose = trajectory[0]
            projection = vector_map.project_to_lane(lane_sequence[0], Point2D(first_pose.x, first_pose.y))
            corridor = build_natural_lane_corridor(
                vector_map,
                lane_sequence[0],
                lookahead_distance_m=float(timed.message.get('lookahead_distance', 80.0)),
                start_arc_length_m=projection.arc_length_m,
                config=NaturalCorridorConfig(),
            )
            comparisons = compare_actual_lane_sequence(lane_sequence, corridor)
            decisions = {decision.branch_lane_id: decision for decision in corridor.branch_decisions}

            first_intersection = None
            for point_index, matched_point in enumerate(match.points):
                if matched_point.lane_id is None:
                    continue
                detected, evidence = lane_has_intersection_evidence(vector_map, str(matched_point.lane_id))
                if detected:
                    distance = distances[min(point_index, len(distances) - 1)]
                    first_intersection = {
                        'lane_id': str(matched_point.lane_id),
                        'route_distance_m': distance,
                        'evidence': evidence,
                    }
                    break

            observed = []
            for comparison in comparisons:
                if comparison.actual_successor_lane_id is None:
                    continue
                decision = decisions[comparison.branch_lane_id]
                reliability, reliability_reasons, score_margin = assess_branch_candidate_reliability(decision.candidates)
                transition = next((
                    value for value in match.transitions
                    if value.source_lane_id == comparison.branch_lane_id
                    and value.target_lane_id == comparison.actual_successor_lane_id
                ), None)
                distance = None
                if transition is not None:
                    distance = distances[min(max(transition.source_point_index, 0), len(distances) - 1)]
                family_guard = evaluate_direction_family_guard(
                    vector_map,
                    branch_lane_id=comparison.branch_lane_id,
                    natural_successor_lane_id=(
                        comparison.natural_successor_lane_id
                    ),
                    route_successor_lane_id=(
                        comparison.actual_successor_lane_id
                    ),
                    raw_relation=(
                        comparison.actual_relation_to_natural
                    ),
                    incoming_heading_rad=(
                        decision.incoming_heading_rad
                    ),
                    config=NaturalCorridorConfig(),
                )

                observed.append({
                    'branch_lane_id': comparison.branch_lane_id,
                    'route_distance_m': distance,
                    'natural_successor_lane_id': comparison.natural_successor_lane_id,
                    'route_successor_lane_id': comparison.actual_successor_lane_id,
                    'raw_route_relation_to_natural': (
                        comparison.actual_relation_to_natural
                    ),
                    'route_relation_to_natural': family_guard.relation,
                    'reliability_status': reliability,
                    'reliability_reasons': list(reliability_reasons),
                    'score_margin': score_margin,
                    'family_guard_version': FAMILY_GUARD_VERSION,
                    'family_guard_status': family_guard.status,
                    'family_guard_reason': family_guard.reason,
                    'family_relationship': (
                        family_guard.family_relationship
                    ),
                    'family_guard_observations': [
                        {
                            'horizon_m': observation.horizon_m,
                            'natural_family': (
                                observation.natural_family
                            ),
                            'route_family': (
                                observation.route_family
                            ),
                        }
                        for observation
                        in family_guard.observations
                    ],
                })
            observed.sort(key=lambda item: (float('inf') if item['route_distance_m'] is None else item['route_distance_m'], item['branch_lane_id']))
            first_observed = observed[0] if observed else None

            status = 'usable' if match.matched_fraction >= 0.8 else 'unknown'
            reasons = [] if status == 'usable' else ['route_lane_match_fraction_below_0_8']
            record.update({
                'quality_status': status,
                'reasons': reasons,
                'route_stamp_ns': int(timed.stamp_ns),
                'anchor_speed_mps': float(ego_state.speed),
                'branch_context': {
                    'route_lane_sequence': list(lane_sequence),
                    'lane_match_fraction': match.matched_fraction,
                    'first_intersection_evidence': first_intersection,
                    'observed_branch_count': len(observed),
                    'first_observed_branch': first_observed,
                    'observed_branches': observed,
                },
            })
            status_counts[status] += 1
            if first_observed is None:
                first_relation_counts['none'] += 1
                first_reliability_counts['none'] += 1
            else:
                first_relation_counts[str(first_observed['route_relation_to_natural'])] += 1
                first_reliability_counts[str(first_observed['reliability_status'])] += 1
                distance = first_observed['route_distance_m']
                if distance is not None:
                    bucket = '<=20m' if distance <= 20 else '<=40m' if distance <= 40 else '<=60m' if distance <= 60 else '>60m'
                    branch_distance_buckets[bucket] += 1
            if first_intersection is not None:
                distance = first_intersection['route_distance_m']
                bucket = '<=20m' if distance <= 20 else '<=40m' if distance <= 40 else '<=60m' if distance <= 60 else '>60m'
                intersection_distance_buckets[bucket] += 1
        except (ValueError, KeyError, RuntimeError) as error:
            record.update({'quality_status': 'unknown', 'reasons': ['navigation_branch_context_unavailable', str(error)], 'route_stamp_ns': int(timed.stamp_ns), 'branch_context': {}})
            status_counts['unknown'] += 1
            error_counts['navigation_branch_context_unavailable'] += 1
        output.append(record)
        if index == 1 or index % 250 == 0 or index == len(keyframes):
            print(f'Processed {index}/{len(keyframes)} Keyframes: {clip_id}')

    text = ''.join(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n' for record in output)
    sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
    summary = {
        'feature_format_version': FORMAT_VERSION,
        'profiler_version': PROFILER_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'input_keyframe_count': len(keyframes),
        'output_record_count': len(output),
        'quality_status_counts': dict(status_counts),
        'first_observed_branch_relation_counts': dict(first_relation_counts),
        'first_observed_branch_reliability_counts': dict(first_reliability_counts),
        'first_intersection_distance_bucket_counts': dict(intersection_distance_buckets),
        'first_branch_distance_bucket_counts': dict(branch_distance_buckets),
        'error_counts': dict(error_counts),
        'output_sha256': sha256,
        'leakage_controls': {
            'route_after_anchor_used': False,
            'future_ego_trajectory_used': False,
            'meta_action_used_for_generation': False,
        },
    }
    atomic_write(args.output, text, args.force)
    atomic_write(args.summary_output, json.dumps(summary, ensure_ascii=False, indent=2) + '\n', args.force)
    print('Output:', args.output)
    print('Summary:', args.summary_output)
    print('SHA-256:', sha256)
    print('Records:', len(output))
    print('Quality:', dict(status_counts))
    print('First branch relations:', dict(first_relation_counts))
    print('First branch reliability:', dict(first_reliability_counts))
    print('First intersection distances:', dict(intersection_distance_buckets))
    print('First branch distances:', dict(branch_distance_buckets))
    print('Errors:', dict(error_counts))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
