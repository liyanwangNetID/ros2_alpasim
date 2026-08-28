#!/usr/bin/env python3
"""Scan Navigation left/right candidates against Anchor-time route geometry.

Diagnostic only. Meta-action and future Ego behavior are not read.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path('/home/lab/data_from_alpasim/annotations/v0.1-draft')
NAVIGATION_PATH = ROOT / 'intermediate/navigation_candidates_v0.1.jsonl'
ROUTE_FEATURE_PATH = ROOT / 'intermediate/navigation_route_features_v0.1.jsonl'
BRANCH_FEATURE_PATH = ROOT / 'intermediate/navigation_branch_context_v0.1.jsonl'


def read_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record['anchor_id'])
            if anchor_id in result:
                raise ValueError(f'duplicate Anchor at {path}:{line_number}: {anchor_id}')
            result[anchor_id] = record
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        'count': len(values),
        'minimum': min(values) if values else None,
        'p10': percentile(values, 0.10),
        'p25': percentile(values, 0.25),
        'median': statistics.median(values) if values else None,
        'p75': percentile(values, 0.75),
        'p90': percentile(values, 0.90),
        'maximum': max(values) if values else None,
    }


def geometry_sign(value: float, epsilon_deg: float = 3.0) -> str:
    if value >= epsilon_deg:
        return 'left'
    if value <= -epsilon_deg:
        return 'right'
    return 'neutral'


def main() -> int:
    navigation = read_index(NAVIGATION_PATH)
    route_features = read_index(ROUTE_FEATURE_PATH)
    branch_features = read_index(BRANCH_FEATURE_PATH)
    if set(navigation) != set(route_features) or set(navigation) != set(branch_features):
        raise ValueError('Navigation diagnostic Anchor sets differ')

    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sign_counts = Counter()
    conflict_counts = Counter()
    source_counts = Counter()
    intersection_distance_counts = Counter()
    weakest: dict[str, list[tuple[float, str, float, float, str]]] = defaultdict(list)

    for anchor_id, navigation_record in navigation.items():
        item = navigation_record['navigation']
        action = str(item['action'])
        if action not in {'left', 'right'}:
            continue

        route = route_features[anchor_id]['route']
        signed_deg = math.degrees(float(route['route_signed_heading_change_rad']))
        excursion_deg = math.degrees(float(route['route_maximum_heading_excursion_rad']))
        final_y = float(route['final_local_y_m'])
        branch = branch_features[anchor_id].get('branch_context', {})
        first_branch = branch.get('first_observed_branch') or {}
        branch_distance = first_branch.get('route_distance_m')
        intersection = branch.get('first_intersection_evidence')
        intersection_distance = (
            intersection.get('route_distance_m')
            if isinstance(intersection, dict)
            else None
        )

        values[action]['signed_heading_deg'].append(signed_deg)
        values[action]['absolute_signed_heading_deg'].append(abs(signed_deg))
        values[action]['excursion_deg'].append(excursion_deg)
        values[action]['final_y_m'].append(final_y)
        if isinstance(branch_distance, (int, float)):
            values[action]['branch_distance_m'].append(float(branch_distance))
        if isinstance(intersection_distance, (int, float)):
            values[action]['intersection_distance_m'].append(float(intersection_distance))
            bucket = '<=20m' if intersection_distance <= 20 else '20_to_40m' if intersection_distance <= 40 else '>40m'
            intersection_distance_counts[(action, bucket)] += 1
        else:
            intersection_distance_counts[(action, 'none')] += 1

        sign = geometry_sign(signed_deg)
        sign_counts[(action, sign)] += 1
        expected_sign = action
        if sign == 'neutral':
            conflict = 'weak_geometry'
        elif sign != expected_sign:
            conflict = 'opposite_geometry'
        else:
            conflict = 'consistent_geometry'
        conflict_counts[(action, conflict)] += 1
        source_counts[(action, str(item['decision_source']))] += 1
        weakest[action].append((abs(signed_deg), anchor_id, signed_deg, final_y, conflict))

    print('=' * 78)
    print('STEP 6 NAVIGATION DIRECTION-CONFLICT SCAN')
    print('=' * 78)
    print('Candidate records:', len(navigation))
    print('Left/right candidates:', sum(len(values[action]['signed_heading_deg']) for action in values))

    for action in ('left', 'right'):
        print()
        print('Action:', action)
        for metric in (
            'signed_heading_deg',
            'absolute_signed_heading_deg',
            'excursion_deg',
            'final_y_m',
            'branch_distance_m',
            'intersection_distance_m',
        ):
            print(' ', metric, ':', describe(values[action][metric]))

    print()
    print('Geometry signs with 3-degree neutral band:')
    for key, count in sorted(sign_counts.items()):
        print(' ', key[0], '+', key[1], ':', count)

    print()
    print('Direction consistency:')
    for key, count in sorted(conflict_counts.items()):
        print(' ', key[0], '+', key[1], ':', count)

    print()
    print('Decision sources:')
    for key, count in sorted(source_counts.items()):
        print(' ', key[0], '+', key[1], ':', count)

    print()
    print('Intersection distances:')
    for key, count in sorted(intersection_distance_counts.items()):
        print(' ', key[0], '+', key[1], ':', count)

    print()
    print('Weakest geometry candidates per action:')
    for action in ('left', 'right'):
        print()
        print(' ', action)
        for magnitude, anchor_id, signed_deg, final_y, conflict in sorted(weakest[action])[:15]:
            print(
                '   ', anchor_id,
                'signed_heading_deg=', signed_deg,
                'final_y_m=', final_y,
                'classification=', conflict,
            )

    print()
    print('PASS: Navigation direction-conflict scan completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
