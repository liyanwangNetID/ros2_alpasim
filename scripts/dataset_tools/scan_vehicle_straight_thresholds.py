#!/usr/bin/env python3
"""Scan ego-only straight-motion thresholds before turn classification.

Lane-change candidates are excluded first. The scan evaluates combinations of:
- absolute total yaw change;
- maximum yaw excursion from the initial heading;
- total absolute yaw variation.

It can also evaluate the reviewed scene groups without double-counting
neighboring Anchors from the same driving event.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_FEATURES = ROOT / "annotations/v0.1-draft/intermediate/lateral_action_features_v0.3.jsonl"
DEFAULT_REVIEWS = ROOT / "reports/lateral_branch_review_v0.1.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/vehicle_straight_threshold_scan_v0.1.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def parse_values(text: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(value.strip()) for value in text.split(",") if value.strip())))
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("threshold list must contain positive values")
    return values


def ego_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    lateral = record["lateral"]
    return {
        "absolute_total_yaw_change_deg": abs(math.degrees(float(lateral["ego_total_yaw_change_rad"]))),
        "maximum_yaw_excursion_deg": math.degrees(float(lateral["ego_maximum_yaw_excursion_rad"])),
        "total_absolute_yaw_change_deg": math.degrees(float(lateral["ego_total_absolute_yaw_change_rad"])),
    }


def eligible_before_straight(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    lateral = record["lateral"]
    if not bool(lateral["lateral_quality_gate"]["passed"]):
        return False, "quality_gate_failed"
    if bool(lateral["contains_adjacent_transition"]):
        return False, "lane_change_priority"
    return True, None


def is_straight(metrics: Mapping[str, float], total: float, excursion: float, absolute: float) -> bool:
    return (
        metrics["absolute_total_yaw_change_deg"] <= total
        and metrics["maximum_yaw_excursion_deg"] <= excursion
        and metrics["total_absolute_yaw_change_deg"] <= absolute
    )


def evaluate_reviews(
    features: Mapping[str, Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
    total: float,
    excursion: float,
    absolute: float,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    scenes = []
    for review in reviews:
        anchor_id = str(review["representative_anchor_id"])
        record = features[anchor_id]
        eligible, reason = eligible_before_straight(record)
        metrics = ego_metrics(record)
        predicted_straight = eligible and is_straight(metrics, total, excursion, absolute)
        observed_straight = str(review["observed_action"]) == "keep_direction"
        if not eligible:
            outcome = reason or "ineligible"
        elif predicted_straight and observed_straight:
            outcome = "true_straight"
        elif predicted_straight and not observed_straight:
            outcome = "turn_suppressed"
        elif not predicted_straight and observed_straight:
            outcome = "straight_not_overridden"
        else:
            outcome = "turn_preserved"
        counts[outcome] += 1
        scenes.append(
            {
                "scene_id": review["scene_id"],
                "representative_anchor_id": anchor_id,
                "observed_action": review["observed_action"],
                "predicted_straight": predicted_straight,
                "outcome": outcome,
                "metrics": metrics,
            }
        )
    return {"counts": dict(counts), "scenes": scenes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--review-input", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-thresholds-deg", default="3,5,8,10")
    parser.add_argument("--excursion-thresholds-deg", default="3,5,8,10")
    parser.add_argument("--absolute-thresholds-deg", default="5,10,15,20")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    feature_rows = read_jsonl(args.feature_input)
    features = {str(row["anchor_id"]): row for row in feature_rows}
    reviews = read_jsonl(args.review_input)
    totals = parse_values(args.total_thresholds_deg)
    excursions = parse_values(args.excursion_thresholds_deg)
    absolutes = parse_values(args.absolute_thresholds_deg)

    rows = []
    for total, excursion, absolute in itertools.product(totals, excursions, absolutes):
        reviewed = evaluate_reviews(features, reviews, total, excursion, absolute)
        all_counts: Counter[str] = Counter()
        for record in feature_rows:
            eligible, reason = eligible_before_straight(record)
            if not eligible:
                all_counts[reason or "ineligible"] += 1
                continue
            if is_straight(ego_metrics(record), total, excursion, absolute):
                all_counts["straight_override"] += 1
            else:
                all_counts["continue_to_turn_logic"] += 1
        rows.append(
            {
                "maximum_total_yaw_change_deg": total,
                "maximum_yaw_excursion_deg": excursion,
                "maximum_total_absolute_yaw_change_deg": absolute,
                "review_counts": reviewed["counts"],
                "review_scenes": reviewed["scenes"],
                "all_anchor_counts": dict(all_counts),
            }
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_record_count": len(feature_rows),
        "review_scene_count": len(reviews),
        "priority": [
            "quality_gate",
            "lane_change",
            "straight_motion_override",
            "branch_relative_turn",
            "keep_direction_fallback",
        ],
        "scan_rows": rows,
    }
    atomic_write(args.output, json.dumps(result, indent=2, ensure_ascii=False) + "\n", args.force)
    print("Feature records:", len(feature_rows))
    print("Review scenes:", len(reviews))
    print("Threshold combinations:", len(rows))
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
