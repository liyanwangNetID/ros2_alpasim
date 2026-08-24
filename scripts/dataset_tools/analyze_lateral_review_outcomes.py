#!/usr/bin/env python3
"""Compare reviewed true-turn and false-positive lateral scenes.

The script reads lateral_action_features_v0.3.jsonl and a compact review JSONL.
It aggregates by review scene, avoiding duplicate weighting from overlapping
Anchors, and reports topology/relative-heading differences that may explain why
straight road-following was classified as a turn candidate.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_FEATURES = ROOT / "annotations/v0.1-draft/intermediate/lateral_action_features_v0.3.jsonl"
DEFAULT_REVIEWS = ROOT / "reports/lateral_branch_review_v0.1.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/lateral_branch_review_analysis_v0.1.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            result.append(value)
    return result


def degrees(value: Any) -> float | None:
    if value is None:
        return None
    return math.degrees(float(value))


def first_directional_comparison(natural: Mapping[str, Any]) -> dict[str, Any] | None:
    comparisons = list(natural.get("boundary_branch_comparisons", [])) + list(
        natural.get("actual_branch_comparisons", [])
    )
    for comparison in comparisons:
        if comparison.get("actual_relation_to_natural") in (
            "left_of_natural", "right_of_natural"
        ):
            return dict(comparison)
    return None


def feature_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    lateral = record["lateral"]
    natural = lateral["natural_corridor"]
    relative = natural.get("relative_heading", {})
    comparison = first_directional_comparison(natural)
    topology = lateral["topology"]
    return {
        "anchor_id": record["anchor_id"],
        "clip_id": record["clip_id"],
        "junction_level": topology["junction_evidence_level"],
        "junction_reasons": topology["junction_evidence_reasons"],
        "same_lane_only": lateral["same_lane_only"],
        "successor_only": lateral["successor_only"],
        "lane_sequence_length": len(lateral["lane_sequence"]),
        "lane_sequence": lateral["lane_sequence"],
        "start_projection_distance_m": natural.get("start_projection_distance_m"),
        "future_path_length_m": natural.get("future_path_length_m"),
        "directional_comparison": comparison,
        "comparison_source": (
            "boundary" if comparison in natural.get("boundary_branch_comparisons", [])
            else "within_window" if comparison is not None else None
        ),
        "relative_start_deg": degrees(relative.get("relative_heading_start_rad")),
        "relative_middle_deg": degrees(relative.get("relative_heading_middle_rad")),
        "relative_end_deg": degrees(relative.get("relative_heading_end_rad")),
        "relative_first_change_deg": degrees(
            relative.get("relative_first_half_heading_change_rad")
        ),
        "relative_second_change_deg": degrees(
            relative.get("relative_second_half_heading_change_rad")
        ),
        "relative_total_change_deg": degrees(
            relative.get("relative_total_heading_change_rad")
        ),
        "maximum_absolute_relative_heading_deg": degrees(
            relative.get("maximum_absolute_relative_heading_rad")
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    def stats(key: str) -> dict[str, float | None]:
        items = sorted(values(key))
        return {
            "minimum": items[0] if items else None,
            "median": items[len(items) // 2] if items else None,
            "maximum": items[-1] if items else None,
        }

    return {
        "scene_count": len(rows),
        "junction_level_counts": dict(Counter(row["junction_level"] for row in rows)),
        "comparison_source_counts": dict(Counter(row["comparison_source"] for row in rows)),
        "start_projection_distance_m": stats("start_projection_distance_m"),
        "relative_start_deg": stats("relative_start_deg"),
        "relative_end_deg": stats("relative_end_deg"),
        "relative_total_change_deg": stats("relative_total_change_deg"),
        "maximum_absolute_relative_heading_deg": stats(
            "maximum_absolute_relative_heading_deg"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--review-input", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    features = {str(row["anchor_id"]): row for row in read_jsonl(args.feature_input)}
    reviews = read_jsonl(args.review_input)
    scene_rows = []
    for review in reviews:
        representative = str(review["representative_anchor_id"])
        if representative not in features:
            raise KeyError(f"review Anchor not found: {representative}")
        metrics = feature_metrics(features[representative])
        metrics.update(
            {
                "scene_id": review["scene_id"],
                "review_outcome": review["review_outcome"],
                "observed_action": review["observed_action"],
                "branch_candidate": review["branch_candidate"],
                "notes": review.get("notes", ""),
                "group_anchor_ids": review.get("group_anchor_ids", [representative]),
            }
        )
        scene_rows.append(metrics)

    true_rows = [row for row in scene_rows if row["review_outcome"] == "correct"]
    false_rows = [row for row in scene_rows if row["review_outcome"] == "false_positive"]
    result = {
        "scene_count": len(scene_rows),
        "outcome_counts": dict(Counter(row["review_outcome"] for row in scene_rows)),
        "correct_summary": summarize(true_rows),
        "false_positive_summary": summarize(false_rows),
        "scenes": scene_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
