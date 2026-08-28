#!/usr/bin/env python3
"""Step 5C v0.1 balanced keyframe selector."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from keyframe_selection_rules_v01 import (
    NORMAL_BASELINE_QUOTA,
    OUTPUT_FORMAT_VERSION,
    RULE_VERSION,
    SELECTOR_VERSION,
    STABLE_LATERAL_QUOTAS,
    STABLE_LONGITUDINAL_QUOTAS,
    select_with_clip_preference,
)

ROOT = Path("/home/lab/data_from_alpasim")
ANN = ROOT / "annotations/v0.1-draft"
DEFAULT_CANDIDATES = ANN / "candidate_anchors.jsonl"
DEFAULT_META = ANN / "meta_actions_v0.2.jsonl"
DEFAULT_EVENTS = ANN / "intermediate/keyframe_event_candidates_deduplicated_v0.1.jsonl"
DEFAULT_OUTPUT = ANN / "keyframes.jsonl"
DEFAULT_SUMMARY = ROOT / "reports/keyframe_selection_summary_v0.1.json"


def read_index(path: Path) -> dict[str, dict[str, Any]]:
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
    parser.add_argument("--candidate-input", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--meta-input", type=Path, default=DEFAULT_META)
    parser.add_argument("--event-input", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_index(args.candidate_input)
    meta = read_index(args.meta_input)
    events = read_index(args.event_input)
    candidate_ids = set(candidates)
    if set(meta) != candidate_ids:
        raise ValueError("Candidate and Meta-action Anchor sets differ")
    if not set(events) <= candidate_ids:
        raise ValueError("Event records contain unknown Anchors")

    selected: dict[str, dict[str, Any]] = {}

    def add(anchor_id: str, source: str, reason: str) -> None:
        if anchor_id not in selected:
            selected[anchor_id] = {
                "selection_sources": [],
                "selection_reasons": [],
            }
        if source not in selected[anchor_id]["selection_sources"]:
            selected[anchor_id]["selection_sources"].append(source)
        if reason not in selected[anchor_id]["selection_reasons"]:
            selected[anchor_id]["selection_reasons"].append(reason)

    # A. Keep every event Anchor.
    for anchor_id in events:
        add(anchor_id, "event_candidate", "contains_detected_keyframe_event")

    baseline_ids = candidate_ids - set(events)

    # B. Stable rare lateral states from the non-event, usable pool.
    for action, quota in STABLE_LATERAL_QUOTAS.items():
        pool = [
            candidates[anchor_id]
            for anchor_id in baseline_ids
            if anchor_id not in selected
            and meta[anchor_id]["overall_quality_status"] == "usable"
            and meta[anchor_id]["lateral"]["action"] == action
        ]
        chosen = select_with_clip_preference(
            pool, quota=quota, bucket=f"stable_lateral:{action}"
        )
        for record in chosen:
            add(
                str(record["anchor_id"]),
                "balanced_stable_lateral",
                f"stable_lateral_{action}",
            )

    # C. Stable longitudinal states while keeping direction.
    for action, quota in STABLE_LONGITUDINAL_QUOTAS.items():
        pool = [
            candidates[anchor_id]
            for anchor_id in baseline_ids
            if anchor_id not in selected
            and meta[anchor_id]["overall_quality_status"] == "usable"
            and meta[anchor_id]["lateral"]["action"] == "keep_direction"
            and meta[anchor_id]["longitudinal"]["action"] == action
        ]
        chosen = select_with_clip_preference(
            pool, quota=quota, bucket=f"stable_longitudinal:{action}"
        )
        if len(chosen) != quota:
            raise ValueError(f"insufficient pool for stable longitudinal {action}: {len(chosen)}")
        for record in chosen:
            add(
                str(record["anchor_id"]),
                "balanced_stable_longitudinal",
                f"stable_longitudinal_{action}",
            )

    # D. Normal straight and steady-speed baseline.
    pool = [
        candidates[anchor_id]
        for anchor_id in baseline_ids
        if anchor_id not in selected
        and meta[anchor_id]["overall_quality_status"] == "usable"
        and meta[anchor_id]["lateral"]["action"] == "keep_direction"
        and meta[anchor_id]["longitudinal"]["action"] == "maintain_speed"
    ]
    chosen = select_with_clip_preference(
        pool, quota=NORMAL_BASELINE_QUOTA, bucket="normal_driving_baseline"
    )
    if len(chosen) != NORMAL_BASELINE_QUOTA:
        raise ValueError(f"insufficient normal baseline pool: {len(chosen)}")
    for record in chosen:
        add(
            str(record["anchor_id"]),
            "normal_driving_baseline",
            "stable_keep_direction_and_maintain_speed",
        )

    output_records = []
    source_counts: Counter[str] = Counter()
    lateral_counts: Counter[str] = Counter()
    longitudinal_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for anchor_id in sorted(
        selected,
        key=lambda value: (
            str(candidates[value]["clip_id"]),
            int(candidates[value]["anchor_ns"]),
            value,
        ),
    ):
        candidate = candidates[anchor_id]
        meta_record = meta[anchor_id]
        selection = selected[anchor_id]
        event_record = events.get(anchor_id)
        record = {
            "keyframe_format_version": OUTPUT_FORMAT_VERSION,
            "selector_version": SELECTOR_VERSION,
            "rule_version": RULE_VERSION,
            "anchor_id": anchor_id,
            "clip_id": str(candidate["clip_id"]),
            "anchor_ns": int(candidate["anchor_ns"]),
            "future_horizon_ns": int(candidate["future_horizon_ns"]),
            "selection_sources": selection["selection_sources"],
            "selection_reasons": selection["selection_reasons"],
            "events": event_record["events"] if event_record else [],
            "meta_action": {
                "lateral": meta_record["lateral"]["action"],
                "longitudinal": meta_record["longitudinal"]["action"],
                "overall_quality_status": meta_record["overall_quality_status"],
            },
        }
        output_records.append(record)
        for source in record["selection_sources"]:
            source_counts[source] += 1
        lateral_counts[record["meta_action"]["lateral"]] += 1
        longitudinal_counts[record["meta_action"]["longitudinal"]] += 1
        quality_counts[record["meta_action"]["overall_quality_status"]] += 1

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output_records
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "keyframe_format_version": OUTPUT_FORMAT_VERSION,
        "selector_version": SELECTOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "keep_all_event_anchors": True,
            "stable_lateral_quotas": STABLE_LATERAL_QUOTAS,
            "stable_longitudinal_quotas": STABLE_LONGITUDINAL_QUOTAS,
            "normal_baseline_quota": NORMAL_BASELINE_QUOTA,
            "deterministic_selection": True,
            "prefer_distinct_clips": True,
            "temporal_suppression": False,
            "abnormal_scene_filtering": False,
        },
        "candidate_anchor_count": len(candidate_ids),
        "event_anchor_count": len(events),
        "selected_keyframe_count": len(output_records),
        "selection_source_counts": dict(source_counts),
        "lateral_action_counts": dict(lateral_counts),
        "longitudinal_action_counts": dict(longitudinal_counts),
        "quality_counts": dict(quality_counts),
        "output_sha256": sha256,
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
    print("Candidate Anchors:", len(candidate_ids))
    print("Event Anchors retained:", len(events))
    print("Selected Keyframes:", len(output_records))
    print("Selection sources:", dict(source_counts))
    print("Lateral actions:", dict(lateral_counts))
    print("Longitudinal actions:", dict(longitudinal_counts))
    print("Quality:", dict(quality_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
