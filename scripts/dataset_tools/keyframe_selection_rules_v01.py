#!/usr/bin/env python3
"""Frozen Step 5C v0.1 keyframe-selection policy."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

SELECTOR_VERSION = "0.1.0"
RULE_VERSION = "keyframe_selection_rules_v0.1"
OUTPUT_FORMAT_VERSION = "0.1-draft"
DETERMINISTIC_SEED = "alpasim-keyframe-selection-v0.1"

STABLE_LATERAL_QUOTAS = {
    "turn_left": None,
    "turn_right": None,
    "change_lane_left": 50,
    "change_lane_right": 50,
}
STABLE_LONGITUDINAL_QUOTAS = {
    "accelerate": 100,
    "decelerate": 100,
    "stop": 100,
}
NORMAL_BASELINE_QUOTA = 500


def deterministic_rank(anchor_id: str, bucket: str) -> str:
    value = f"{DETERMINISTIC_SEED}|{bucket}|{anchor_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_with_clip_preference(
    records: Sequence[Mapping[str, Any]],
    *,
    quota: int | None,
    bucket: str,
    clip_id: Callable[[Mapping[str, Any]], str] = lambda record: str(record["clip_id"]),
    anchor_id: Callable[[Mapping[str, Any]], str] = lambda record: str(record["anchor_id"]),
) -> list[Mapping[str, Any]]:
    """Select deterministically, choosing at most one per Clip before filling."""
    ordered = sorted(records, key=lambda record: deterministic_rank(anchor_id(record), bucket))
    if quota is None or quota >= len(ordered):
        return ordered

    by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in ordered:
        by_clip[clip_id(record)].append(record)

    chosen: list[Mapping[str, Any]] = []
    chosen_ids: set[str] = set()
    clip_order = sorted(
        by_clip,
        key=lambda value: deterministic_rank(anchor_id(by_clip[value][0]), bucket + "|clip"),
    )
    for clip in clip_order:
        record = by_clip[clip][0]
        chosen.append(record)
        chosen_ids.add(anchor_id(record))
        if len(chosen) == quota:
            return chosen

    for record in ordered:
        if anchor_id(record) in chosen_ids:
            continue
        chosen.append(record)
        if len(chosen) == quota:
            return chosen
    return chosen
