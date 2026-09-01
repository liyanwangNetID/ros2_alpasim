#!/usr/bin/env python3
"""Scan Step 5C inputs before defining keyframe selection quotas."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from project_paths import (
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
)

ROOT = ANNOTATION_ROOT
CANDIDATE_PATH = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
META_PATH = (
    ANNOTATION_ROOT / "meta_actions_v0.2.jsonl"
)
EVENT_PATH = (
    INTERMEDIATE_ROOT
    / "keyframe_event_candidates_deduplicated_v0.1.jsonl"
)

EVENT_ORDER = (
    "turn_start",
    "lane_change_start",
    "lane_change_in_progress",
    "acceleration_start",
    "deceleration_start",
    "stop_start",
    "restart",
    "junction_approach",
    "junction_entry",
    "lateral_action_transition",
    "longitudinal_action_transition",
)


def read_index(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = record["anchor_id"]
            if anchor_id in records:
                raise RuntimeError(
                    f"Duplicate Anchor in {path} at line "
                    f"{line_number}: {anchor_id}"
                )
            records[anchor_id] = record
    return records


def main() -> int:
    candidates = read_index(CANDIDATE_PATH)
    meta = read_index(META_PATH)
    events = read_index(EVENT_PATH)

    candidate_ids = set(candidates)
    meta_ids = set(meta)
    event_ids = set(events)

    if candidate_ids != meta_ids:
        raise RuntimeError("Candidate and Meta-action Anchor sets differ")
    if not event_ids <= candidate_ids:
        raise RuntimeError("Event candidates contain unknown Anchors")

    baseline_ids = candidate_ids - event_ids

    event_anchor_counts = Counter()
    event_instance_counts = Counter()
    event_clip_sets = defaultdict(set)
    event_confidence_counts = defaultdict(Counter)
    event_direction_counts = defaultdict(Counter)
    event_lateral_counts = defaultdict(Counter)
    event_longitudinal_counts = defaultdict(Counter)
    events_per_anchor = Counter()

    for anchor_id, record in events.items():
        event_types_for_anchor = set()
        events_per_anchor[len(record["events"])] += 1

        for event in record["events"]:
            event_type = event["type"]
            event_types_for_anchor.add(event_type)
            event_instance_counts[event_type] += 1
            event_clip_sets[event_type].add(record["clip_id"])
            event_confidence_counts[event_type][event["confidence"]] += 1

            direction = event.get("direction")
            if direction is not None:
                event_direction_counts[event_type][direction] += 1

            event_lateral_counts[event_type][
                meta[anchor_id]["lateral"]["action"]
            ] += 1
            event_longitudinal_counts[event_type][
                meta[anchor_id]["longitudinal"]["action"]
            ] += 1

        for event_type in event_types_for_anchor:
            event_anchor_counts[event_type] += 1

    baseline_lateral_counts = Counter(
        meta[anchor_id]["lateral"]["action"] for anchor_id in baseline_ids
    )
    baseline_longitudinal_counts = Counter(
        meta[anchor_id]["longitudinal"]["action"]
        for anchor_id in baseline_ids
    )
    baseline_joint_counts = Counter(
        (
            meta[anchor_id]["lateral"]["action"],
            meta[anchor_id]["longitudinal"]["action"],
        )
        for anchor_id in baseline_ids
    )
    baseline_quality_counts = Counter(
        meta[anchor_id]["overall_quality_status"] for anchor_id in baseline_ids
    )
    baseline_clips = {
        candidates[anchor_id]["clip_id"] for anchor_id in baseline_ids
    }

    print("=" * 78)
    print("STEP 5C PRE-SELECTION DISTRIBUTION")
    print("=" * 78)
    print("Candidate Anchors:", len(candidate_ids))
    print("Event Anchors:", len(event_ids))
    print("Baseline Anchors:", len(baseline_ids))
    print("Event + baseline:", len(event_ids) + len(baseline_ids))
    print("Baseline Clips:", len(baseline_clips))

    assert len(candidate_ids) == 10231
    assert len(event_ids) == 2552
    assert len(baseline_ids) == 7679

    print()
    print("=" * 78)
    print("EVENT COVERAGE")
    print("=" * 78)

    for event_type in EVENT_ORDER:
        print()
        print("Event:", event_type)
        print("  Anchor count:", event_anchor_counts[event_type])
        print("  Event instances:", event_instance_counts[event_type])
        print("  Clip count:", len(event_clip_sets[event_type]))
        print("  Confidence:", dict(event_confidence_counts[event_type]))

        if event_direction_counts[event_type]:
            print("  Directions:", dict(event_direction_counts[event_type]))

        print("  Lateral actions:", dict(event_lateral_counts[event_type]))
        print(
            "  Longitudinal actions:",
            dict(event_longitudinal_counts[event_type]),
        )

    print()
    print("=" * 78)
    print("EVENTS PER ANCHOR")
    print("=" * 78)
    for event_count in sorted(events_per_anchor):
        print(
            f"{event_count} event(s):",
            events_per_anchor[event_count],
            "Anchors",
        )

    print()
    print("=" * 78)
    print("NON-EVENT BASELINE POOL")
    print("=" * 78)
    print("Lateral:", dict(baseline_lateral_counts))
    print("Longitudinal:", dict(baseline_longitudinal_counts))
    print("Quality:", dict(baseline_quality_counts))

    print()
    print("Most common baseline joint actions:")
    for (lateral_action, longitudinal_action), count in (
        baseline_joint_counts.most_common(15)
    ):
        print(
            " ",
            lateral_action,
            "+",
            longitudinal_action,
            ":",
            count,
        )

    print()
    print("PASS: Step 5C input distribution scan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
