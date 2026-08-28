#!/usr/bin/env python3
"""Frozen Step 5B v0.1 exact Anchor/event normalization rules.

No temporal suppression is performed. Different Anchor IDs are always kept.
Within one Anchor, events sharing (type, direction) are merged deterministically.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

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
