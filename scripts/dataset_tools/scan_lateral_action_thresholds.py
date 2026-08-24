#!/usr/bin/env python3
"""Scan candidate thresholds for map-grounded lateral turn classification.

This Step 4 diagnostic tool reads lateral_action_features_v0.2.jsonl and
compares Level A and Level B threshold combinations. It does not write final
Meta-action labels and does not open raw clips.

Important: positive/negative turn directions are intentionally kept as signed
candidates until a real front-camera review confirms the dataset convention.
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
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_VERSION = "0.1.0"
SCAN_FORMAT_VERSION = "0.1-draft"
ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_INPUT = (
    ROOT / "annotations" / "v0.1-draft" / "intermediate"
    / "lateral_action_features_v0.2.jsonl"
)
DEFAULT_SUMMARY = ROOT / "reports" / "lateral_threshold_scan_v0.1.json"
DEFAULT_CASES = ROOT / "reports" / "lateral_threshold_scan_cases_v0.1.jsonl"

DEFAULT_LEVEL_A_THRESHOLDS_DEG = (10.0, 15.0, 20.0, 25.0)
DEFAULT_LEVEL_B_THRESHOLDS_DEG = (20.0, 25.0, 30.0, 35.0, 40.0)
DEFAULT_LEVEL_C_LARGE_ANGLE_DEG = 20.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object")
            records.append(value)
    return records


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


def parse_thresholds(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("thresholds must be a non-empty list of positive values")
    return tuple(sorted(set(values)))


def signs_agree(first: float, second: float) -> bool:
    return first != 0.0 and second != 0.0 and (first > 0.0) == (second > 0.0)


def eligible_turn_base(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    lateral = record["lateral"]
    gate = lateral["lateral_quality_gate"]
    if not bool(gate["passed"]):
        return False, "quality_gate_failed"
    if bool(lateral["contains_adjacent_transition"]):
        return False, "contains_adjacent_transition"
    if lateral.get("map_corridor_heading_change_rad") is None:
        return False, "map_heading_unavailable"
    return True, None


def evaluate_threshold_pair(
    records: Sequence[Mapping[str, Any]],
    *,
    evidence_level: str,
    ego_threshold_deg: float,
    map_threshold_deg: float,
) -> dict[str, Any]:
    ego_threshold = math.radians(ego_threshold_deg)
    map_threshold = math.radians(map_threshold_deg)
    counts: Counter[str] = Counter()
    candidate_ids: list[str] = []
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    margin_values_deg: list[float] = []

    for record in records:
        lateral = record["lateral"]
        if lateral["topology"]["junction_evidence_level"] != evidence_level:
            continue
        counts["level_total"] += 1
        eligible, reason = eligible_turn_base(record)
        if not eligible:
            counts[reason or "ineligible"] += 1
            continue

        ego_change = float(lateral["trajectory_yaw_signed_change_rad"])
        map_change = float(lateral["map_corridor_heading_change_rad"])
        if abs(ego_change) < ego_threshold:
            counts["below_ego_threshold"] += 1
            continue
        if abs(map_change) < map_threshold:
            counts["below_map_threshold"] += 1
            continue
        if not signs_agree(ego_change, map_change):
            counts["direction_disagreement"] += 1
            continue

        anchor_id = str(record["anchor_id"])
        candidate_ids.append(anchor_id)
        direction = "positive" if ego_change > 0.0 else "negative"
        counts[f"{direction}_candidate"] += 1
        if direction == "positive":
            positive_ids.append(anchor_id)
        else:
            negative_ids.append(anchor_id)
        counts["candidate_total"] += 1
        margin_values_deg.append(
            min(
                abs(math.degrees(ego_change)) - ego_threshold_deg,
                abs(math.degrees(map_change)) - map_threshold_deg,
            )
        )

    return {
        "evidence_level": evidence_level,
        "ego_threshold_deg": ego_threshold_deg,
        "map_threshold_deg": map_threshold_deg,
        "counts": dict(counts),
        "candidate_anchor_ids": candidate_ids,
        "positive_candidate_anchor_ids": positive_ids,
        "negative_candidate_anchor_ids": negative_ids,
        "minimum_margin_deg": min(margin_values_deg) if margin_values_deg else None,
        "median_margin_deg": (
            sorted(margin_values_deg)[len(margin_values_deg) // 2]
            if margin_values_deg else None
        ),
    }


def threshold_stability(scan_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_level: dict[str, list[Mapping[str, Any]]] = {}
    for row in scan_rows:
        by_level.setdefault(str(row["evidence_level"]), []).append(row)
    result: list[dict[str, Any]] = []
    for level, rows in by_level.items():
        lookup = {
            (float(row["ego_threshold_deg"]), float(row["map_threshold_deg"])): row
            for row in rows
        }
        for (ego, map_value), row in lookup.items():
            current = int(row["counts"].get("candidate_total", 0))
            neighbors = []
            for key, other in lookup.items():
                if (
                    (key[0] == ego and key[1] != map_value)
                    or (key[1] == map_value and key[0] != ego)
                ):
                    distance = abs(key[0] - ego) + abs(key[1] - map_value)
                    neighbors.append((distance, other))
            neighbors.sort(key=lambda value: value[0])
            closest = neighbors[:2]
            deltas = [
                abs(current - int(other["counts"].get("candidate_total", 0)))
                for _, other in closest
            ]
            result.append(
                {
                    "evidence_level": level,
                    "ego_threshold_deg": ego,
                    "map_threshold_deg": map_value,
                    "candidate_total": current,
                    "mean_closest_neighbor_count_delta": (
                        sum(deltas) / len(deltas) if deltas else None
                    ),
                }
            )
    return result


def level_c_large_angle_cases(
    records: Sequence[Mapping[str, Any]],
    threshold_deg: float,
) -> list[dict[str, Any]]:
    threshold = math.radians(threshold_deg)
    cases: list[dict[str, Any]] = []
    for record in records:
        lateral = record["lateral"]
        if lateral["topology"]["junction_evidence_level"] != "C":
            continue
        eligible, _ = eligible_turn_base(record)
        if not eligible:
            continue
        ego = float(lateral["trajectory_yaw_signed_change_rad"])
        map_value = float(lateral["map_corridor_heading_change_rad"])
        if (
            abs(ego) >= threshold
            and abs(map_value) >= threshold
            and signs_agree(ego, map_value)
        ):
            cases.append(
                {
                    "anchor_id": record["anchor_id"],
                    "clip_id": record["clip_id"],
                    "anchor_ns": record["anchor_ns"],
                    "ego_yaw_change_deg": math.degrees(ego),
                    "map_heading_change_deg": math.degrees(map_value),
                }
            )
    return cases


def sample_records(
    values: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    if len(values) <= count:
        return list(values)
    generator = random.Random(seed)
    indexes = sorted(generator.sample(range(len(values)), count))
    return [values[index] for index in indexes]


def make_case_record(
    source: str,
    record: Mapping[str, Any],
    *,
    ego_threshold_deg: float | None = None,
    map_threshold_deg: float | None = None,
) -> dict[str, Any]:
    lateral = record["lateral"]
    ego = float(lateral["trajectory_yaw_signed_change_rad"])
    map_value = lateral["map_corridor_heading_change_rad"]
    return {
        "scan_format_version": SCAN_FORMAT_VERSION,
        "scanner_version": SCRIPT_VERSION,
        "source": source,
        "anchor_id": record["anchor_id"],
        "clip_id": record["clip_id"],
        "anchor_ns": record["anchor_ns"],
        "junction_evidence_level": lateral["topology"]["junction_evidence_level"],
        "junction_evidence_reasons": lateral["topology"]["junction_evidence_reasons"],
        "ego_yaw_change_deg": math.degrees(ego),
        "map_heading_change_deg": (
            math.degrees(float(map_value)) if map_value is not None else None
        ),
        "signed_direction_candidate": "positive" if ego > 0.0 else "negative",
        "ego_threshold_deg": ego_threshold_deg,
        "map_threshold_deg": map_threshold_deg,
        "lane_sequence": lateral["lane_sequence"],
        "path_heading_reliable": lateral["path_heading_reliable"],
        "filtered_path_heading_change_deg": math.degrees(
            float(lateral["filtered_path_signed_heading_change_rad"])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--level-a-thresholds-deg",
        default=",".join(str(value) for value in DEFAULT_LEVEL_A_THRESHOLDS_DEG),
    )
    parser.add_argument(
        "--level-b-thresholds-deg",
        default=",".join(str(value) for value in DEFAULT_LEVEL_B_THRESHOLDS_DEG),
    )
    parser.add_argument(
        "--level-c-large-angle-deg",
        type=float,
        default=DEFAULT_LEVEL_C_LARGE_ANGLE_DEG,
    )
    parser.add_argument("--samples-per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_group <= 0:
        raise ValueError("--samples-per-group must be positive")
    if args.level_c_large_angle_deg <= 0.0:
        raise ValueError("--level-c-large-angle-deg must be positive")

    records = read_jsonl(args.feature_input.expanduser().resolve())
    level_a = parse_thresholds(args.level_a_thresholds_deg)
    level_b = parse_thresholds(args.level_b_thresholds_deg)

    scan_rows: list[dict[str, Any]] = []
    for ego_threshold, map_threshold in itertools.product(level_a, repeat=2):
        scan_rows.append(
            evaluate_threshold_pair(
                records,
                evidence_level="A",
                ego_threshold_deg=ego_threshold,
                map_threshold_deg=map_threshold,
            )
        )
    for ego_threshold, map_threshold in itertools.product(level_b, repeat=2):
        scan_rows.append(
            evaluate_threshold_pair(
                records,
                evidence_level="B",
                ego_threshold_deg=ego_threshold,
                map_threshold_deg=map_threshold,
            )
        )

    level_c_cases = level_c_large_angle_cases(
        records, args.level_c_large_angle_deg
    )
    record_by_id = {str(record["anchor_id"]): record for record in records}

    case_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(scan_rows):
        source = (
            f"level_{str(row['evidence_level']).lower()}_"
            f"ego_{row['ego_threshold_deg']:g}_map_{row['map_threshold_deg']:g}"
        )
        positive = [
            record_by_id[anchor_id]
            for anchor_id in row["positive_candidate_anchor_ids"]
        ]
        negative = [
            record_by_id[anchor_id]
            for anchor_id in row["negative_candidate_anchor_ids"]
        ]
        for direction, candidates in (("positive", positive), ("negative", negative)):
            sampled = sample_records(
                candidates, args.samples_per_group, args.seed + row_index
            )
            for record in sampled:
                case_records.append(
                    make_case_record(
                        source + "_" + direction,
                        record,
                        ego_threshold_deg=float(row["ego_threshold_deg"]),
                        map_threshold_deg=float(row["map_threshold_deg"]),
                    )
                )

    level_c_source_records = [
        record_by_id[str(item["anchor_id"])] for item in level_c_cases
    ]
    for record in sample_records(
        level_c_source_records, args.samples_per_group * 2, args.seed + 999
    ):
        case_records.append(
            make_case_record(
                f"level_c_large_angle_{args.level_c_large_angle_deg:g}",
                record,
                ego_threshold_deg=args.level_c_large_angle_deg,
                map_threshold_deg=args.level_c_large_angle_deg,
            )
        )

    compact_rows = []
    for row in scan_rows:
        compact = dict(row)
        compact.pop("candidate_anchor_ids")
        compact.pop("positive_candidate_anchor_ids")
        compact.pop("negative_candidate_anchor_ids")
        compact_rows.append(compact)

    summary = {
        "scan_format_version": SCAN_FORMAT_VERSION,
        "scanner_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_input": str(args.feature_input),
        "feature_record_count": len(records),
        "direction_label_mapping": {
            "positive": "unverified",
            "negative": "unverified",
        },
        "configuration": {
            "level_a_thresholds_deg": list(level_a),
            "level_b_thresholds_deg": list(level_b),
            "level_c_large_angle_deg": args.level_c_large_angle_deg,
            "require_ego_map_direction_agreement": True,
            "exclude_adjacent_transitions": True,
            "require_lateral_quality_gate": True,
        },
        "scan_rows": compact_rows,
        "stability": threshold_stability(scan_rows),
        "level_c_large_angle": {
            "count": len(level_c_cases),
            "anchor_ids": [item["anchor_id"] for item in level_c_cases],
        },
    }

    case_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in case_records
    )
    atomic_write(
        args.summary_output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        args.force,
    )
    atomic_write(args.case_output, case_text, args.force)

    print("Feature records:", len(records))
    print("Level A threshold combinations:", len(level_a) ** 2)
    print("Level B threshold combinations:", len(level_b) ** 2)
    print("Level C large-angle cases:", len(level_c_cases))
    print("Sampled case records:", len(case_records))
    print("Summary output:", args.summary_output)
    print("Case output:", args.case_output)
    print("Case SHA-256:", hashlib.sha256(case_text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
