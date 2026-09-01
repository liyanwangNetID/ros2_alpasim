#!/usr/bin/env python3
"""Publish reviewed Step 6 Navigation candidates as navigation.jsonl."""
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

from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

ROOT = ALPASIM_DATA_ROOT
ANN = ANNOTATION_ROOT
DEFAULT_KEYFRAMES = ANNOTATION_ROOT / 'keyframes.jsonl'
DEFAULT_INPUT = (
    INTERMEDIATE_ROOT / 'navigation_candidates_v0.1.jsonl'
)
DEFAULT_OUTPUT = ANNOTATION_ROOT / 'navigation.jsonl'
DEFAULT_SUMMARY = (
    REPORT_ROOT / 'navigation_generation_summary_v0.1.json'
)
EXPECTED_GENERATOR_VERSION = '0.1.4'
EXPECTED_RULE_VERSION = 'navigation_rules_v0.1.4-candidate'
FINAL_RULE_VERSION = 'navigation_rules_v0.1.4'


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


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f'output exists: {path}; use --force')
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent,
        prefix=path.name + '.', suffix='.tmp', delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyframe-input', type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument('--candidate-input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--summary-output', type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyframes = read_index(args.keyframe_input)
    candidates = read_index(args.candidate_input)
    if set(keyframes) != set(candidates):
        raise ValueError('Navigation candidate and Keyframe Anchor sets differ')

    output: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for anchor_id in sorted(
        candidates,
        key=lambda value: (
            str(candidates[value]['clip_id']),
            int(candidates[value]['anchor_ns']),
            value,
        ),
    ):
        candidate = candidates[anchor_id]
        if candidate.get('generator_version') != EXPECTED_GENERATOR_VERSION:
            raise ValueError(f'{anchor_id}: unexpected generator version')
        if candidate.get('rule_version') != EXPECTED_RULE_VERSION:
            raise ValueError(f'{anchor_id}: unexpected candidate rule version')

        navigation = candidate['navigation']
        action = str(navigation['action'])
        text = navigation.get('text')
        quality = str(navigation['quality_status'])
        if action not in {'straight', 'left', 'right', 'unknown'}:
            raise ValueError(f'{anchor_id}: invalid Navigation action {action}')
        if action == 'unknown':
            if text is not None or quality != 'unknown':
                raise ValueError(f'{anchor_id}: invalid unknown Navigation')
        elif text is None or quality != 'usable':
            raise ValueError(f'{anchor_id}: invalid usable Navigation')

        record = dict(candidate)
        record['rule_version'] = FINAL_RULE_VERSION
        record['review_status'] = 'reviewed_v0.1'
        output.append(record)
        action_counts[action] += 1
        quality_counts[quality] += 1
        text_counts[str(text)] += 1
        source_counts[str(navigation['decision_source'])] += 1

    output_text = ''.join(
        json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'
        for record in output
    )
    sha256 = hashlib.sha256(output_text.encode('utf-8')).hexdigest()
    summary = {
        'navigation_format_version': '0.1-draft',
        'generator_version': EXPECTED_GENERATOR_VERSION,
        'rule_version': FINAL_RULE_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'review_status': 'reviewed_v0.1',
        'input_keyframe_count': len(keyframes),
        'output_record_count': len(output),
        'action_counts': dict(action_counts),
        'quality_status_counts': dict(quality_counts),
        'decision_source_counts': dict(source_counts),
        'text_counts': dict(text_counts),
        'output_sha256': sha256,
        'leakage_controls': {
            'future_ego_trajectory_used': False,
            'future_speed_used': False,
            'future_control_used': False,
            'meta_action_used_for_generation': False,
        },
    }
    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        args.force,
    )

    print('Output:', args.output)
    print('Summary:', args.summary_output)
    print('SHA-256:', sha256)
    print('Records:', len(output))
    print('Actions:', dict(action_counts))
    print('Quality:', dict(quality_counts))
    print('Review status: reviewed_v0.1')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
