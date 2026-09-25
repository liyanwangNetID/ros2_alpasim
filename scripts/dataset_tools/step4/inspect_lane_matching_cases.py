#!/usr/bin/env python3
"""Inspect diagnostic case groups from existing lane-matching features.

This Step 4 offline diagnostic tool reads the reusable intermediate feature
file produced by profile_lane_matching_features.py. It does not open raw clip
directories and does not rerun lane matching.

Outputs:
1. a compact JSON summary with category counts and common transition pairs;
2. a JSONL file containing deterministically sampled cases for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from project_paths import (
    ALPASIM_DATA_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)


SCRIPT_VERSION = "0.1.0"
DIAGNOSTIC_FORMAT_VERSION = "0.1-draft"
DEFAULT_DATASET_ROOT = ALPASIM_DATA_ROOT
DEFAULT_FEATURE_INPUT = (
    INTERMEDIATE_ROOT / "lane_matching_features_v0.1.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = (
    REPORT_ROOT / "lane_matching_case_summary_v0.1.json"
)
DEFAULT_CASE_OUTPUT = (
    REPORT_ROOT / "lane_matching_cases_v0.1.jsonl"
)

CATEGORY_ORDER = (
    "fully_unmatched",
    "low_match",
    "low_confidence_full_match",
    "unrelated_transition",
    "predecessor_transition",
    "left_adjacent_transition",
    "right_adjacent_transition",
    "both_adjacent_directions",
    "returns_to_previous_lane",
    "short_lived_adjacent_target",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL row is not an object at {path}:{line_number}"
                    )
                records.append(value)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"feature input not found: {path}") from exc
    return records


def atomic_write_text(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing output: {path}; use --force"
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return {
        "minimum": min(items) if items else None,
        "mean": statistics.mean(items) if items else None,
        "median": statistics.median(items) if items else None,
        "p90": percentile(items, 0.90),
        "p95": percentile(items, 0.95),
        "maximum": max(items) if items else None,
    }


def transition_relations(record: dict[str, Any]) -> tuple[str, ...]:
    transitions = record.get("transitions", [])
    if not isinstance(transitions, list):
        return tuple()
    return tuple(
        str(transition.get("relation", ""))
        for transition in transitions
        if isinstance(transition, dict)
    )


def has_return_to_previous_lane(sequence: list[str]) -> bool:
    """Detect A -> B -> A patterns, including longer intervening paths."""
    last_position: dict[str, int] = {}
    for index, lane_id in enumerate(sequence):
        if lane_id in last_position and index - last_position[lane_id] >= 2:
            return True
        last_position[lane_id] = index
    return False


def has_short_lived_adjacent_target(record: dict[str, Any]) -> bool:
    """Flag adjacent transitions whose final target has fewer than 3 points."""
    relations = transition_relations(record)
    if not any(
        relation in ("left_adjacent", "right_adjacent")
        for relation in relations
    ):
        return False
    point_count = int(record.get("target_lane_point_count", 0))
    return point_count < 3


def categories_for_record(record: dict[str, Any]) -> tuple[str, ...]:
    matched_fraction = float(record.get("matched_fraction", 0.0))
    confidence = float(record.get("confidence", 0.0))
    relations = transition_relations(record)
    relation_set = set(relations)
    raw_sequence = record.get("compressed_lane_sequence", [])
    sequence = [str(value) for value in raw_sequence] if isinstance(raw_sequence, list) else []

    categories: list[str] = []
    if matched_fraction == 0.0:
        categories.append("fully_unmatched")
    if matched_fraction < 0.8:
        categories.append("low_match")
    if matched_fraction == 1.0 and confidence < 0.5:
        categories.append("low_confidence_full_match")
    if "unrelated" in relation_set:
        categories.append("unrelated_transition")
    if "predecessor" in relation_set:
        categories.append("predecessor_transition")
    if "left_adjacent" in relation_set:
        categories.append("left_adjacent_transition")
    if "right_adjacent" in relation_set:
        categories.append("right_adjacent_transition")
    if "left_adjacent" in relation_set and "right_adjacent" in relation_set:
        categories.append("both_adjacent_directions")
    if has_return_to_previous_lane(sequence):
        categories.append("returns_to_previous_lane")
    if has_short_lived_adjacent_target(record):
        categories.append("short_lived_adjacent_target")
    return tuple(categories)


def transition_pair_counts(
    records: Iterable[dict[str, Any]],
    relation_filter: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        transitions = record.get("transitions", [])
        if not isinstance(transitions, list):
            continue
        for transition in transitions:
            if not isinstance(transition, dict):
                continue
            relation = str(transition.get("relation", ""))
            if relation not in relation_filter:
                continue
            source = str(transition.get("source_lane_id", ""))
            target = str(transition.get("target_lane_id", ""))
            counts[f"{source}->{target}:{relation}"] += 1
    return counts


def compact_case_record(
    record: dict[str, Any],
    category: str,
    sample_rank: int,
) -> dict[str, Any]:
    return {
        "diagnostic_format_version": DIAGNOSTIC_FORMAT_VERSION,
        "inspector_version": SCRIPT_VERSION,
        "category": category,
        "sample_rank": sample_rank,
        "anchor_id": record.get("anchor_id"),
        "clip_id": record.get("clip_id"),
        "anchor_ns": record.get("anchor_ns"),
        "matched_fraction": record.get("matched_fraction"),
        "confidence": record.get("confidence"),
        "mean_distance_m": record.get("mean_distance_m"),
        "maximum_distance_m": record.get("maximum_distance_m"),
        "mean_heading_error_rad": record.get("mean_heading_error_rad"),
        "trajectory_point_count": record.get("trajectory_point_count"),
        "matched_point_count": record.get("matched_point_count"),
        "unmatched_point_count": record.get("unmatched_point_count"),
        "compressed_lane_sequence": record.get("compressed_lane_sequence"),
        "transitions": record.get("transitions"),
        "target_lane_point_count": record.get("target_lane_point_count"),
        "target_lane_fraction": record.get("target_lane_fraction"),
        "map_id": record.get("map_id"),
        "map_topology_warning_count": record.get(
            "map_topology_warning_count"
        ),
    }


def sample_category(
    records: list[dict[str, Any]],
    sample_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(records) <= sample_count:
        return sorted(records, key=lambda record: str(record.get("anchor_id", "")))
    random_generator = random.Random(seed)
    indexes = sorted(random_generator.sample(range(len(records)), sample_count))
    return [records[index] for index in indexes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect diagnostic groups in lane-matching features."
    )
    parser.add_argument(
        "--feature-input",
        type=Path,
        default=DEFAULT_FEATURE_INPUT,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        default=DEFAULT_CASE_OUTPUT,
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260821,
    )
    parser.add_argument(
        "--top-transition-pairs",
        type=int,
        default=20,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_category <= 0:
        raise ValueError("--samples-per-category must be positive")
    if args.top_transition_pairs <= 0:
        raise ValueError("--top-transition-pairs must be positive")

    feature_input = args.feature_input.expanduser().resolve()
    records = read_jsonl(feature_input)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories_per_anchor: Counter[int] = Counter()
    clips_per_category: dict[str, set[str]] = defaultdict(set)

    for record in records:
        categories = categories_for_record(record)
        categories_per_anchor[len(categories)] += 1
        for category in categories:
            by_category[category].append(record)
            clips_per_category[category].add(str(record.get("clip_id", "")))

    case_records: list[dict[str, Any]] = []
    sampled_counts: dict[str, int] = {}
    for category_index, category in enumerate(CATEGORY_ORDER):
        category_records = by_category.get(category, [])
        sampled = sample_category(
            category_records,
            args.samples_per_category,
            args.seed + category_index,
        )
        sampled_counts[category] = len(sampled)
        for sample_rank, record in enumerate(sampled):
            case_records.append(
                compact_case_record(record, category, sample_rank)
            )

    anomalous_pairs = transition_pair_counts(
        records,
        {"unrelated", "predecessor"},
    )
    adjacent_pairs = transition_pair_counts(
        records,
        {"left_adjacent", "right_adjacent"},
    )

    category_summary: dict[str, Any] = {}
    for category in CATEGORY_ORDER:
        category_records = by_category.get(category, [])
        category_summary[category] = {
            "anchor_count": len(category_records),
            "clip_count": len(clips_per_category.get(category, set())),
            "sampled_count": sampled_counts.get(category, 0),
            "matched_fraction": distribution(
                float(record.get("matched_fraction", 0.0))
                for record in category_records
            ),
            "confidence": distribution(
                float(record.get("confidence", 0.0))
                for record in category_records
            ),
            "target_lane_fraction": distribution(
                float(record.get("target_lane_fraction", 0.0))
                for record in category_records
            ),
        }

    summary = {
        "diagnostic_format_version": DIAGNOSTIC_FORMAT_VERSION,
        "inspector_version": SCRIPT_VERSION,
        "generated_at": utc_now_iso(),
        "feature_input": str(feature_input),
        "feature_record_count": len(records),
        "sampling": {
            "samples_per_category": args.samples_per_category,
            "seed": args.seed,
        },
        "category_summary": category_summary,
        "category_membership_count_per_anchor": {
            str(key): value for key, value in sorted(categories_per_anchor.items())
        },
        "top_anomalous_transition_pairs": [
            {"pair": pair, "count": count}
            for pair, count in anomalous_pairs.most_common(
                args.top_transition_pairs
            )
        ],
        "top_adjacent_transition_pairs": [
            {"pair": pair, "count": count}
            for pair, count in adjacent_pairs.most_common(
                args.top_transition_pairs
            )
        ],
    }

    case_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for record in case_records
    )
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(args.case_output, case_text, args.force)
    atomic_write_text(args.summary_output, summary_text, args.force)

    digest = hashlib.sha256(case_text.encode("utf-8")).hexdigest()
    print("Feature records:", len(records))
    for category in CATEGORY_ORDER:
        print(
            f"{category}: {len(by_category.get(category, []))} "
            f"anchors; sampled {sampled_counts.get(category, 0)}"
        )
    print("Case output:", args.case_output)
    print("Summary output:", args.summary_output)
    print("Case SHA-256:", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
