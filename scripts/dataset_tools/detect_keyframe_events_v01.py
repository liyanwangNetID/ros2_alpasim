#!/usr/bin/env python3
"""Step 5A v0.1 event-candidate detector."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from keyframe_event_rules_v01 import (
    DETECTOR_VERSION,
    EVENT_FORMAT_VERSION,
    RULE_VERSION,
    detect_anchor_events,
)
from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

ROOT = ALPASIM_DATA_ROOT
DEFAULT_CANDIDATE_INPUT = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
DEFAULT_META_INPUT = (
    ANNOTATION_ROOT / "meta_actions_v0.2.jsonl"
)
DEFAULT_LATERAL_INPUT = (
    INTERMEDIATE_ROOT / "lateral_action_features_v0.3.jsonl"
)
DEFAULT_LONGITUDINAL_INPUT = (
    INTERMEDIATE_ROOT / "meta_action_features_v0.2.jsonl"
)
DEFAULT_GEOMETRY_INPUT = (
    INTERMEDIATE_ROOT / "lane_change_geometry_features_v0.1.jsonl"
)
DEFAULT_OUTPUT = (
    INTERMEDIATE_ROOT / "keyframe_event_candidates_v0.1.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "keyframe_event_candidate_summary_v0.1.json"
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
                raise ValueError(f"duplicate Anchor at {path}:{line_number}: {anchor_id}")
            result[anchor_id] = record
    return result


def require_identity(anchor_id: str, reference: Mapping[str, Any], *others: Mapping[str, Any]) -> None:
    for record in others:
        for key in ("anchor_id", "clip_id", "anchor_ns"):
            if reference.get(key) != record.get(key):
                raise ValueError(f"{anchor_id}: identity mismatch for {key}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-input", type=Path, default=DEFAULT_CANDIDATE_INPUT)
    parser.add_argument("--meta-input", type=Path, default=DEFAULT_META_INPUT)
    parser.add_argument("--lateral-input", type=Path, default=DEFAULT_LATERAL_INPUT)
    parser.add_argument("--longitudinal-input", type=Path, default=DEFAULT_LONGITUDINAL_INPUT)
    parser.add_argument("--geometry-input", type=Path, default=DEFAULT_GEOMETRY_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_jsonl_index(args.candidate_input)
    meta = read_jsonl_index(args.meta_input)
    lateral = read_jsonl_index(args.lateral_input)
    longitudinal = read_jsonl_index(args.longitudinal_input)
    geometry = read_jsonl_index(args.geometry_input)

    expected = set(candidates)
    for name, index in (
        ("meta", meta), ("lateral", lateral),
        ("longitudinal", longitudinal), ("geometry", geometry),
    ):
        if set(index) != expected:
            raise ValueError(
                f"{name} Anchor set differs; missing={sorted(expected-set(index))[:10]}, "
                f"extra={sorted(set(index)-expected)[:10]}"
            )

    clips: dict[str, list[str]] = defaultdict(list)
    for anchor_id, record in candidates.items():
        clips[str(record["clip_id"])].append(anchor_id)

    output_records: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    event_clips: dict[str, set[str]] = defaultdict(set)
    multi_event_anchor_count = 0

    for clip_id in sorted(clips):
        anchor_ids = sorted(clips[clip_id], key=lambda value: int(candidates[value]["anchor_ns"]))
        previous_id: str | None = None

        for anchor_id in anchor_ids:
            current_candidate = candidates[anchor_id]
            require_identity(
                anchor_id,
                current_candidate,
                meta[anchor_id], lateral[anchor_id], longitudinal[anchor_id], geometry[anchor_id],
            )

            events = detect_anchor_events(
                previous_meta=meta[previous_id] if previous_id else None,
                current_meta=meta[anchor_id],
                previous_lateral_features=lateral[previous_id] if previous_id else None,
                current_lateral_features=lateral[anchor_id],
                previous_longitudinal_features=longitudinal[previous_id] if previous_id else None,
                current_longitudinal_features=longitudinal[anchor_id],
                current_geometry_features=geometry[anchor_id],
            )

            if events:
                record = {
                    "event_format_version": EVENT_FORMAT_VERSION,
                    "detector_version": DETECTOR_VERSION,
                    "rule_version": RULE_VERSION,
                    "anchor_id": anchor_id,
                    "clip_id": clip_id,
                    "anchor_ns": int(current_candidate["anchor_ns"]),
                    "future_horizon_ns": int(current_candidate["future_horizon_ns"]),
                    "events": events,
                }
                output_records.append(record)
                if len(events) > 1:
                    multi_event_anchor_count += 1
                for event in events:
                    event_type = str(event["type"])
                    event_counts[event_type] += 1
                    confidence_counts[str(event["confidence"])] += 1
                    event_clips[event_type].add(clip_id)

            previous_id = anchor_id

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output_records
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "event_format_version": EVENT_FORMAT_VERSION,
        "detector_version": DETECTOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_anchor_count": len(expected),
        "event_anchor_count": len(output_records),
        "no_event_anchor_count": len(expected) - len(output_records),
        "multi_event_anchor_count": multi_event_anchor_count,
        "event_counts": dict(event_counts),
        "confidence_counts": dict(confidence_counts),
        "event_clip_counts": {key: len(value) for key, value in sorted(event_clips.items())},
        "output_sha256": sha256,
        "excluded_event_families": ["navigation_direction_change", "actor_interaction_change"],
    }

    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )

    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("SHA-256:", sha256)
    print("Total Anchors:", len(expected))
    print("Event Anchors:", len(output_records))
    print("No-event Anchors:", len(expected) - len(output_records))
    print("Multi-event Anchors:", multi_event_anchor_count)
    print("Event counts:", dict(event_counts))
    print("Confidence counts:", dict(confidence_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
