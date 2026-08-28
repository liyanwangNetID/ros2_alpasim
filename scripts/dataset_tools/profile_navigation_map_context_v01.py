#!/usr/bin/env python3
"""Profile VectorMap context for Anchor-time Navigation routes."""
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
from navigation_map_context_v01 import (
    FEATURE_FORMAT_VERSION,
    PROFILER_VERSION,
    extract_navigation_map_context,
)
from navigation_route_features_v01 import valid_local_points
from vector_map_reader import VectorMapReader

DATA_ROOT = Path('/home/lab/data_from_alpasim')
ANN = DATA_ROOT / 'annotations/v0.1-draft'
DEFAULT_KEYFRAMES = ANN / 'keyframes.jsonl'
DEFAULT_OUTPUT = ANN / 'intermediate/navigation_map_context_v0.1.jsonl'
DEFAULT_SUMMARY = DATA_ROOT / 'reports/navigation_map_context_summary_v0.1.json'


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
    return sorted(result, key=lambda value: (str(value['clip_id']), int(value['anchor_ns']), str(value['anchor_id'])))


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f'output exists: {path}; use --force')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
        file.write(content)
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
    directional_counts = Counter()
    context_counts = Counter()
    error_counts = Counter()

    for index, keyframe in enumerate(keyframes, start=1):
        clip_id = str(keyframe['clip_id'])
        anchor_ns = int(keyframe['anchor_ns'])
        reader = readers.get(clip_id)
        if reader is None:
            reader = DrivingClipReader(args.dataset_root / clip_id)
            readers[clip_id] = reader
            maps[clip_id] = VectorMapReader.from_dict(reader.get_vector_map())

        base = {
            'feature_format_version': FEATURE_FORMAT_VERSION,
            'profiler_version': PROFILER_VERSION,
            'anchor_id': str(keyframe['anchor_id']),
            'clip_id': clip_id,
            'anchor_ns': anchor_ns,
            'route_time_policy': 'at_or_before_anchor',
        }
        timed = reader.get_navigation_route_at(anchor_ns, maximum_age_ns=args.maximum_route_age_ns)
        ego_pose = reader.get_ego_pose_at(anchor_ns)
        if timed is None or ego_pose is None:
            reasons = []
            if timed is None:
                reasons.append('recent_navigation_route_unavailable')
            if ego_pose is None:
                reasons.append('anchor_ego_pose_unavailable')
            record = {**base, 'quality_status': 'unknown', 'reasons': reasons, 'map_context': {}}
            error_counts.update(reasons)
        else:
            points = valid_local_points(timed.message)
            try:
                context = extract_navigation_map_context(
                    vector_map=maps[clip_id],
                    anchor_pose=ego_pose,
                    local_points=points,
                    anchor_ns=anchor_ns,
                    lookahead_distance_m=float(timed.message.get('lookahead_distance', 80.0)),
                )
                record = {**base, 'quality_status': context['quality_status'], 'reasons': context['reasons'], 'route_stamp_ns': int(timed.stamp_ns), 'map_context': context}
            except (ValueError, KeyError, RuntimeError) as error:
                record = {**base, 'quality_status': 'unknown', 'reasons': ['navigation_map_context_unavailable', str(error)], 'route_stamp_ns': int(timed.stamp_ns), 'map_context': {}}
                error_counts['navigation_map_context_unavailable'] += 1

        status_counts[record['quality_status']] += 1
        context = record.get('map_context', {})
        topology = context.get('topology', {})
        context_counts['intersection' if topology.get('intersection_context') else 'non_intersection'] += 1
        relations = context.get('reliable_directional_relations', [])
        if not relations:
            directional_counts['none'] += 1
        else:
            directional_counts.update(relations)
        output.append(record)

        if index == 1 or index % 250 == 0 or index == len(keyframes):
            print(f'Processed {index}/{len(keyframes)} Keyframes: {clip_id}')

    text = ''.join(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n' for record in output)
    sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
    summary = {
        'feature_format_version': FEATURE_FORMAT_VERSION,
        'profiler_version': PROFILER_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'input_keyframe_count': len(keyframes),
        'output_record_count': len(output),
        'quality_status_counts': dict(status_counts),
        'reliable_directional_relation_counts': dict(directional_counts),
        'topology_context_counts': dict(context_counts),
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
    print('Directional relations:', dict(directional_counts))
    print('Topology contexts:', dict(context_counts))
    print('Errors:', dict(error_counts))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
