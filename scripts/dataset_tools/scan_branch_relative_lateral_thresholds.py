#!/usr/bin/env python3
"""Scan branch-relative thresholds for lateral turn candidates.

Reads lateral_action_features_v0.3.jsonl only. It does not open raw clips and
does not generate final labels. A turn candidate must already have a reliable
left_of_natural or right_of_natural branch relation. Thresholds are applied to
lane-relative heading evidence, never to world-coordinate yaw sign.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_VERSION = "0.2.0"
SCAN_FORMAT_VERSION = "0.2-draft"
ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_INPUT = (
    ROOT / "annotations" / "v0.1-draft" / "intermediate"
    / "lateral_action_features_v0.3.jsonl"
)
DEFAULT_SUMMARY = ROOT / "reports" / "branch_relative_lateral_threshold_scan_v0.1.json"
DEFAULT_CASES = ROOT / "reports" / "branch_relative_lateral_threshold_cases_v0.1.jsonl"
DEFAULT_PROGRESS_THRESHOLDS_DEG = (0.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_DEVIATION_THRESHOLDS_DEG = (5.0, 10.0, 15.0, 20.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


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


def parse_thresholds(text: str, *, allow_zero: bool) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    minimum = 0.0 if allow_zero else 1e-12
    if not values or any(value < minimum for value in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"thresholds must be a non-empty list of {qualifier} values")
    return tuple(sorted(set(values)))


def unique_directional_relation(natural: Mapping[str, Any]) -> str | None:
    relations = set(str(value) for value in natural.get("reliable_directional_relations", []))
    if relations == {"left_of_natural"}:
        return "left"
    if relations == {"right_of_natural"}:
        return "right"
    return None


def directed_degrees(value_rad: float, direction: str) -> float:
    value_deg = math.degrees(float(value_rad))
    return value_deg if direction == "left" else -value_deg


def evidence_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    lateral = record["lateral"]
    natural = lateral["natural_corridor"]
    direction = unique_directional_relation(natural)
    relative = natural.get("relative_heading", {})
    if direction is None:
        raise ValueError("record does not have one reliable directional relation")

    first = directed_degrees(
        float(relative["relative_first_half_heading_change_rad"]), direction
    )
    second = directed_degrees(
        float(relative["relative_second_half_heading_change_rad"]), direction
    )
    total = directed_degrees(
        float(relative["relative_total_heading_change_rad"]), direction
    )
    start = directed_degrees(float(relative["relative_heading_start_rad"]), direction)
    middle = directed_degrees(float(relative["relative_heading_middle_rad"]), direction)
    end = directed_degrees(float(relative["relative_heading_end_rad"]), direction)
    progress = max(0.0, first, second, total, middle - start, end - start, end - middle)
    maximum_absolute = math.degrees(
        float(relative["maximum_absolute_relative_heading_rad"])
    )
    return {
        "direction": direction,
        "directed_first_half_change_deg": first,
        "directed_second_half_change_deg": second,
        "directed_total_change_deg": total,
        "directed_start_heading_deg": start,
        "directed_middle_heading_deg": middle,
        "directed_end_heading_deg": end,
        "maximum_directional_progress_deg": progress,
        "maximum_absolute_relative_heading_deg": maximum_absolute,
    }


STRAIGHT_TOTAL_YAW_DEG = 3.0
STRAIGHT_MAX_EXCURSION_DEG = 3.0
STRAIGHT_TOTAL_ABSOLUTE_YAW_DEG = 5.0


def straight_motion_override(record: Mapping[str, Any]) -> bool:
    lateral = record["lateral"]
    return (
        abs(math.degrees(float(lateral["ego_total_yaw_change_rad"])))
        <= STRAIGHT_TOTAL_YAW_DEG
        and math.degrees(float(lateral["ego_maximum_yaw_excursion_rad"]))
        <= STRAIGHT_MAX_EXCURSION_DEG
        and math.degrees(float(lateral["ego_total_absolute_yaw_change_rad"]))
        <= STRAIGHT_TOTAL_ABSOLUTE_YAW_DEG
    )


def base_eligibility(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    lateral = record["lateral"]
    natural = lateral["natural_corridor"]
    if not bool(lateral["lateral_quality_gate"]["passed"]):
        return False, "quality_gate_failed"
    if bool(lateral["contains_adjacent_transition"]):
        return False, "contains_adjacent_transition"
    if straight_motion_override(record):
        return False, "straight_motion_override"
    if natural.get("turn_evidence_status") != "directional_branch_observed":
        return False, "not_directional_branch_observed"
    if unique_directional_relation(natural) is None:
        return False, "ambiguous_directional_relation"
    relative = natural.get("relative_heading", {})
    required = (
        "relative_first_half_heading_change_rad",
        "relative_second_half_heading_change_rad",
        "relative_total_heading_change_rad",
        "relative_heading_start_rad",
        "relative_heading_middle_rad",
        "relative_heading_end_rad",
        "maximum_absolute_relative_heading_rad",
    )
    if any(key not in relative for key in required):
        return False, "relative_heading_unavailable"
    return True, None


def evaluate(
    records: Sequence[Mapping[str, Any]],
    *,
    progress_threshold_deg: float,
    deviation_threshold_deg: float,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    candidates: list[str] = []
    rejected_boundary: list[str] = []
    for record in records:
        eligible, reason = base_eligibility(record)
        if not eligible:
            counts[reason or "ineligible"] += 1
            continue
        counts["eligible_directional_branch"] += 1
        metrics = evidence_metrics(record)
        if metrics["maximum_directional_progress_deg"] < progress_threshold_deg:
            counts["below_progress_threshold"] += 1
            rejected_boundary.append(str(record["anchor_id"]))
            continue
        if metrics["maximum_absolute_relative_heading_deg"] < deviation_threshold_deg:
            counts["below_deviation_threshold"] += 1
            rejected_boundary.append(str(record["anchor_id"]))
            continue
        direction = str(metrics["direction"])
        counts[f"turn_{direction}_candidate"] += 1
        counts["candidate_total"] += 1
        candidates.append(str(record["anchor_id"]))
    return {
        "progress_threshold_deg": progress_threshold_deg,
        "deviation_threshold_deg": deviation_threshold_deg,
        "counts": dict(counts),
        "candidate_anchor_ids": candidates,
        "rejected_boundary_anchor_ids": rejected_boundary,
    }


def sample(values: Sequence[str], count: int, seed: int) -> list[str]:
    if len(values) <= count:
        return list(values)
    generator = random.Random(seed)
    return sorted(generator.sample(list(values), count))


def case_record(
    record: Mapping[str, Any],
    source: str,
    progress_threshold_deg: float,
    deviation_threshold_deg: float,
) -> dict[str, Any]:
    lateral = record["lateral"]
    natural = lateral["natural_corridor"]
    return {
        "scan_format_version": SCAN_FORMAT_VERSION,
        "scanner_version": SCRIPT_VERSION,
        "source": source,
        "anchor_id": record["anchor_id"],
        "clip_id": record["clip_id"],
        "anchor_ns": record["anchor_ns"],
        "junction_evidence_level": lateral["topology"]["junction_evidence_level"],
        "progress_threshold_deg": progress_threshold_deg,
        "deviation_threshold_deg": deviation_threshold_deg,
        "metrics": evidence_metrics(record),
        "boundary_branch_comparisons": natural.get("boundary_branch_comparisons", []),
        "actual_branch_comparisons": natural.get("actual_branch_comparisons", []),
        "lane_sequence": lateral["lane_sequence"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--progress-thresholds-deg",
        default=",".join(str(value) for value in DEFAULT_PROGRESS_THRESHOLDS_DEG),
    )
    parser.add_argument(
        "--deviation-thresholds-deg",
        default=",".join(str(value) for value in DEFAULT_DEVIATION_THRESHOLDS_DEG),
    )
    parser.add_argument("--samples-per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_group <= 0:
        raise ValueError("--samples-per-group must be positive")
    records = read_jsonl(args.feature_input.expanduser().resolve())
    progress_values = parse_thresholds(args.progress_thresholds_deg, allow_zero=True)
    deviation_values = parse_thresholds(args.deviation_thresholds_deg, allow_zero=False)
    rows = [
        evaluate(
            records,
            progress_threshold_deg=progress,
            deviation_threshold_deg=deviation,
        )
        for progress, deviation in itertools.product(progress_values, deviation_values)
    ]

    by_id = {str(record["anchor_id"]): record for record in records}
    cases: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = (
            f"progress_{row['progress_threshold_deg']:g}_"
            f"deviation_{row['deviation_threshold_deg']:g}"
        )
        selected = sample(
            row["candidate_anchor_ids"], args.samples_per_group, args.seed + index
        )
        rejected = sample(
            row["rejected_boundary_anchor_ids"],
            args.samples_per_group,
            args.seed + 1000 + index,
        )
        for anchor_id in selected:
            cases.append(
                case_record(
                    by_id[anchor_id],
                    source + "_candidate",
                    float(row["progress_threshold_deg"]),
                    float(row["deviation_threshold_deg"]),
                )
            )
        for anchor_id in rejected:
            cases.append(
                case_record(
                    by_id[anchor_id],
                    source + "_rejected_boundary",
                    float(row["progress_threshold_deg"]),
                    float(row["deviation_threshold_deg"]),
                )
            )
        value = dict(row)
        value.pop("candidate_anchor_ids")
        value.pop("rejected_boundary_anchor_ids")
        compact.append(value)

    base_counts = Counter()
    directional_metrics: list[dict[str, Any]] = []
    for record in records:
        eligible, reason = base_eligibility(record)
        if eligible:
            base_counts["eligible_directional_branch"] += 1
            metrics = evidence_metrics(record)
            base_counts[f"direction_{metrics['direction']}"] += 1
            directional_metrics.append(metrics)
        else:
            base_counts[reason or "ineligible"] += 1

    summary = {
        "scan_format_version": SCAN_FORMAT_VERSION,
        "scanner_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_input": str(args.feature_input),
        "feature_record_count": len(records),
        "principle": (
            "turn direction comes from reliable branch relation; world yaw sign "
            "is not used as a left/right label"
        ),
        "configuration": {
            "progress_thresholds_deg": list(progress_values),
            "deviation_thresholds_deg": list(deviation_values),
            "require_quality_gate": True,
            "exclude_adjacent_transitions": True,
            "straight_motion_override": {
                "maximum_total_yaw_change_deg": STRAIGHT_TOTAL_YAW_DEG,
                "maximum_yaw_excursion_deg": STRAIGHT_MAX_EXCURSION_DEG,
                "maximum_total_absolute_yaw_change_deg": (
                    STRAIGHT_TOTAL_ABSOLUTE_YAW_DEG
                ),
            },
        },
        "base_counts": dict(base_counts),
        "scan_rows": compact,
    }
    case_text = "".join(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n"
        for value in cases
    )
    atomic_write(
        args.summary_output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        args.force,
    )
    atomic_write(args.case_output, case_text, args.force)
    print("Feature records:", len(records))
    print("Base counts:", dict(base_counts))
    print("Threshold combinations:", len(rows))
    print("Sampled case records:", len(cases))
    print("Summary output:", args.summary_output)
    print("Case output:", args.case_output)
    print("Case SHA-256:", hashlib.sha256(case_text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
