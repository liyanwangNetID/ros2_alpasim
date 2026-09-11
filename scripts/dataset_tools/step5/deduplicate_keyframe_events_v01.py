#!/usr/bin/env python3
"""Step 5B v0.1 exact Anchor-level deduplication and normalization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from project_paths import (
    ALPASIM_DATA_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

DEDUPLICATOR_VERSION = "0.1.0"


RULE_VERSION = "keyframe_event_dedup_rules_v0.1"


OUTPUT_FORMAT_VERSION = "0.1-draft"


def event_identity(event: Mapping[str, Any]) -> tuple[str, str | None]:
    event_type = str(event.get("type", ""))
    if not event_type:
        raise ValueError("event type must be a non-empty string")
    direction_value = event.get("direction")
    direction = None if direction_value is None else str(direction_value)
    return event_type, direction


def _stable_unique(values: Sequence[Any]) -> list[Any]:
    indexed = {json.dumps(value, sort_keys=True, ensure_ascii=False): value for value in values}
    return [indexed[key] for key in sorted(indexed)]


def merge_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("cannot merge an empty event group")

    identities = {event_identity(event) for event in events}
    if len(identities) != 1:
        raise ValueError(f"event identity mismatch: {sorted(identities)}")

    event_type, direction = next(iter(identities))
    confidences = {str(event.get("confidence", "unknown")) for event in events}
    confidence = "high" if confidences == {"high"} else "low"

    sources = sorted({str(event.get("source", "")) for event in events if event.get("source")})
    reasons = _stable_unique([
        reason
        for event in events
        for reason in event.get("reasons", [])
    ])

    metric_variants = _stable_unique([
        dict(event.get("metrics", {}))
        for event in events
    ])
    metrics: dict[str, Any]
    if len(metric_variants) == 1:
        metrics = metric_variants[0]
    else:
        metrics = {"evidence_variants": metric_variants}

    result: dict[str, Any] = {
        "type": event_type,
        "confidence": confidence,
        "source": sources[0] if len(sources) == 1 else "multiple_sources",
        "reasons": reasons,
        "metrics": metrics,
    }
    if direction is not None:
        result["direction"] = direction
    if len(sources) > 1:
        result["evidence_sources"] = sources
    if len(events) > 1:
        result["merged_duplicate_count"] = len(events)
    return result


def normalize_anchor_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None], list[Mapping[str, Any]]] = {}
    for event in events:
        groups.setdefault(event_identity(event), []).append(event)

    normalized = [merge_events(groups[key]) for key in sorted(groups, key=lambda item: (item[0], item[1] or ""))]
    return normalized


ROOT = ALPASIM_DATA_ROOT
DEFAULT_INPUT = (
    INTERMEDIATE_ROOT / "keyframe_event_candidates_v0.1.jsonl"
)
DEFAULT_OUTPUT = (
    INTERMEDIATE_ROOT
    / "keyframe_event_candidates_deduplicated_v0.1.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "keyframe_event_deduplication_summary_v0.1.json"
)


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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anchors: dict[str, dict[str, Any]] = {}
    input_record_count = 0
    input_event_count = 0
    duplicate_anchor_record_count = 0

    with args.input.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            events = record.get("events", [])
            if not isinstance(events, list):
                raise ValueError(f"{args.input}:{line_number} events is not a list")
            input_record_count += 1
            input_event_count += len(events)

            if anchor_id not in anchors:
                anchors[anchor_id] = {
                    "event_format_version": OUTPUT_FORMAT_VERSION,
                    "deduplicator_version": DEDUPLICATOR_VERSION,
                    "rule_version": RULE_VERSION,
                    "anchor_id": anchor_id,
                    "clip_id": str(record["clip_id"]),
                    "anchor_ns": int(record["anchor_ns"]),
                    "future_horizon_ns": int(record["future_horizon_ns"]),
                    "events": list(events),
                }
            else:
                duplicate_anchor_record_count += 1
                existing = anchors[anchor_id]
                for key in ("clip_id", "anchor_ns", "future_horizon_ns"):
                    if existing[key] != record[key]:
                        raise ValueError(f"{anchor_id}: conflicting duplicate Anchor field {key}")
                existing["events"].extend(events)

    output_records = []
    output_event_count = 0
    duplicate_event_count = 0
    event_counts: Counter[str] = Counter()

    for record in sorted(anchors.values(), key=lambda value: (value["clip_id"], value["anchor_ns"], value["anchor_id"])):
        before = len(record["events"])
        normalized = normalize_anchor_events(record["events"])
        after = len(normalized)
        duplicate_event_count += before - after
        output_event_count += after
        record["events"] = normalized
        output_records.append(record)
        for event in normalized:
            event_counts[str(event["type"])] += 1

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output_records
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "output_format_version": OUTPUT_FORMAT_VERSION,
        "deduplicator_version": DEDUPLICATOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "anchor_identity": "anchor_id",
            "event_identity": ["type", "direction"],
            "temporal_suppression": False,
            "different_anchor_ids_always_preserved": True,
            "abnormal_scene_filtering": False,
        },
        "input_record_count": input_record_count,
        "output_anchor_count": len(output_records),
        "duplicate_anchor_record_count": duplicate_anchor_record_count,
        "input_event_count": input_event_count,
        "output_event_count": output_event_count,
        "duplicate_event_count_removed": duplicate_event_count,
        "event_counts": dict(event_counts),
        "output_sha256": sha256,
    }

    atomic_write(args.output, output_text, args.force)
    atomic_write(args.summary_output, json.dumps(summary, ensure_ascii=False, indent=2) + "\n", args.force)

    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("SHA-256:", sha256)
    print("Input records:", input_record_count)
    print("Output Anchors:", len(output_records))
    print("Duplicate Anchor records merged:", duplicate_anchor_record_count)
    print("Input events:", input_event_count)
    print("Output events:", output_event_count)
    print("Duplicate events removed:", duplicate_event_count)
    print("Temporal suppression: disabled")
    print("Abnormal-scene filtering: disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
