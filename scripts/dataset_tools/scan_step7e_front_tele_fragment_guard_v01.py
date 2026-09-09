#!/usr/bin/env python3
"""Step 7E.10 shadow-scan front-tele fragment guards.

The scale rule remains:
    area >= 512 px2 OR height >= 16 px

A candidate is retained only when the scale rule passes and either enough of
the projected hull remains inside the image or the remaining in-image area is
large enough:
    inside_ratio >= ratio_floor OR inside_area >= fragment_area_floor

This is a read-only shadow scan. No Scene Fact is modified.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from project_paths import ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_JSON = ANNOTATION_ROOT / "step7e_front_tele_fragment_guard_scan_v01.json"
OUTPUT_CSV = ANNOTATION_ROOT / "step7e_front_tele_fragment_guard_scan_v01.csv"

CAMERA_NAME = "front_tele"
PRIMARY_AREA_PX2 = 512.0
PRIMARY_HEIGHT_PX = 16.0
RATIO_FLOORS = (0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.25)
FRAGMENT_AREA_FLOORS_PX2 = (96.0, 128.0, 160.0, 192.0, 256.0, 384.0)

CLASS_FAMILIES = {
    "automobile": "standard_vehicle",
    "other_vehicle": "standard_vehicle",
    "heavy_truck": "large_vehicle",
    "bus": "large_vehicle",
    "trailer": "large_vehicle",
    "train_or_tram_car": "large_vehicle",
    "person": "vulnerable_road_user",
    "rider": "vulnerable_road_user",
    "stroller": "vulnerable_road_user",
    "protruding_object": "special_object",
    "animal": "rare_fallback",
}

# Manually reviewed cases from round 2.
REVIEW_CASES = {
    ("test_clip_741_1414042502155000", "27"): "expected_reject",
    ("test_clip_440_1196517731585000", "24"): "expected_reject",
    ("test_clip_251_2354945826999000", "352"): "expected_keep",
    ("test_clip_556_2503816097000", "363"): "occlusion_or_correspondence",
    ("test_clip_533_3351209998000", "19"): "occlusion_ambiguous",
    ("test_clip_860_10022807928000", "101"): "expected_keep",
}


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def family(label_class: str) -> str:
    return CLASS_FAMILIES.get(label_class, "rare_fallback")


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("camera_name") == CAMERA_NAME:
                rows.append(row)
            elif row.get("camera_name") not in {
                "front_wide", "cross_left", "cross_right"
            }:
                raise ValueError(
                    f"Unexpected camera at line {line_number}: "
                    f"{row.get('camera_name')}"
                )
    if not rows:
        raise RuntimeError("No front_tele evidence rows found")
    return rows


def scale_pass(row: dict[str, Any]) -> bool:
    return (
        float(row["inside_image_hull_area_px"]) >= PRIMARY_AREA_PX2
        or float(row["projected_height_px"]) >= PRIMARY_HEIGHT_PX
    )


def decision(
    row: dict[str, Any],
    *,
    ratio_floor: float,
    fragment_area_floor: float,
) -> bool:
    fragment_guard = (
        float(row["inside_image_hull_ratio"]) >= ratio_floor
        or float(row["inside_image_hull_area_px"]) >= fragment_area_floor
    )
    return scale_pass(row) and fragment_guard


def summarize_counter(kept: Counter, total: Counter) -> dict[str, Any]:
    return {
        key: {
            "total": total[key],
            "retained": kept[key],
            "retention_ratio": ratio(kept[key], total[key]),
        }
        for key in sorted(total)
    }


def evaluate(
    rows: list[dict[str, Any]],
    *,
    ratio_floor: float,
    fragment_area_floor: float,
) -> dict[str, Any]:
    scale_eligible = [row for row in rows if scale_pass(row)]
    total_family: Counter[str] = Counter()
    kept_family: Counter[str] = Counter()
    total_truncation: Counter[str] = Counter()
    kept_truncation: Counter[str] = Counter()
    kept_count = 0
    rejected_by_guard = 0

    for row in scale_eligible:
        row_family = family(str(row["label_class"]))
        inside_ratio = float(row["inside_image_hull_ratio"])
        truncation = (
            "below_0.05" if inside_ratio < 0.05
            else "0.05_to_0.10" if inside_ratio < 0.10
            else "0.10_to_0.25" if inside_ratio < 0.25
            else "0.25_to_0.50" if inside_ratio < 0.50
            else "0.50_to_1.0" if inside_ratio < 1.0 - 1e-9
            else "not_truncated"
        )
        keep = decision(
            row,
            ratio_floor=ratio_floor,
            fragment_area_floor=fragment_area_floor,
        )
        total_family[row_family] += 1
        total_truncation[truncation] += 1
        if keep:
            kept_count += 1
            kept_family[row_family] += 1
            kept_truncation[truncation] += 1
        else:
            rejected_by_guard += 1

    reviewed: dict[str, Any] = {}
    row_index = {
        (str(row["anchor_id"]), str(row["track_id"])): row
        for row in rows
    }
    for reviewed_key, expectation in REVIEW_CASES.items():
        row = row_index.get(reviewed_key)
        if row is None:
            reviewed["|".join(reviewed_key)] = {
                "expectation": expectation,
                "found": False,
            }
            continue
        reviewed["|".join(reviewed_key)] = {
            "expectation": expectation,
            "found": True,
            "decision": decision(
                row,
                ratio_floor=ratio_floor,
                fragment_area_floor=fragment_area_floor,
            ),
            "area_px2": float(row["inside_image_hull_area_px"]),
            "height_px": float(row["projected_height_px"]),
            "inside_ratio": float(row["inside_image_hull_ratio"]),
        }

    return {
        "ratio_floor": ratio_floor,
        "fragment_area_floor_px2": fragment_area_floor,
        "valid_projection_rows": len(rows),
        "scale_eligible_rows": len(scale_eligible),
        "retained_rows": kept_count,
        "fragment_guard_rejected_rows": rejected_by_guard,
        "retention_vs_scale_eligible": ratio(kept_count, len(scale_eligible)),
        "by_family": summarize_counter(kept_family, total_family),
        "by_truncation_band": summarize_counter(
            kept_truncation, total_truncation
        ),
        "reviewed_cases": reviewed,
    }


def reviewed_case_matches(result: dict[str, Any]) -> dict[str, bool]:
    matches: dict[str, bool] = {}
    for key, item in result["reviewed_cases"].items():
        expectation = item["expectation"]
        if not item.get("found"):
            matches[key] = False
        elif expectation == "expected_keep":
            matches[key] = bool(item["decision"])
        elif expectation == "expected_reject":
            matches[key] = not bool(item["decision"])
        else:
            # Ambiguous/occlusion cases are reported but not constrained.
            matches[key] = True
    return matches


def main() -> int:
    rows = read_rows()
    results = [
        evaluate(
            rows,
            ratio_floor=ratio_floor,
            fragment_area_floor=fragment_area_floor,
        )
        for ratio_floor in RATIO_FLOORS
        for fragment_area_floor in FRAGMENT_AREA_FLOORS_PX2
    ]
    for result in results:
        result["reviewed_case_matches"] = reviewed_case_matches(result)
        result["all_non_ambiguous_reviewed_cases_match"] = all(
            result["reviewed_case_matches"].values()
        )

    report = {
        "schema_version": "step7e-front-tele-fragment-guard-scan-v01",
        "description": "Shadow scan only; no Scene Fact was modified.",
        "input_path": str(INPUT_PATH),
        "camera_name": CAMERA_NAME,
        "primary_scale_rule": {
            "area_px2": PRIMARY_AREA_PX2,
            "height_px": PRIMARY_HEIGHT_PX,
            "operator": "OR",
        },
        "fragment_guard_rule": (
            "inside_ratio >= ratio_floor OR "
            "inside_area_px2 >= fragment_area_floor_px2"
        ),
        "ratio_floors": RATIO_FLOORS,
        "fragment_area_floors_px2": FRAGMENT_AREA_FLOORS_PX2,
        "review_cases": {
            "|".join(key): expectation
            for key, expectation in REVIEW_CASES.items()
        },
        "result_count": len(results),
        "results": results,
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = (
        "ratio_floor",
        "fragment_area_floor_px2",
        "scale_eligible_rows",
        "retained_rows",
        "fragment_guard_rejected_rows",
        "retention_vs_scale_eligible",
        "standard_vehicle_retention_ratio",
        "large_vehicle_retention_ratio",
        "vulnerable_road_user_retention_ratio",
        "special_object_retention_ratio",
        "below_0_05_retention_ratio",
        "all_non_ambiguous_reviewed_cases_match",
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            by_family = result["by_family"]
            by_truncation = result["by_truncation_band"]
            writer.writerow(
                {
                    "ratio_floor": result["ratio_floor"],
                    "fragment_area_floor_px2": result[
                        "fragment_area_floor_px2"
                    ],
                    "scale_eligible_rows": result["scale_eligible_rows"],
                    "retained_rows": result["retained_rows"],
                    "fragment_guard_rejected_rows": result[
                        "fragment_guard_rejected_rows"
                    ],
                    "retention_vs_scale_eligible": result[
                        "retention_vs_scale_eligible"
                    ],
                    **{
                        f"{name}_retention_ratio": by_family.get(
                            name, {"retention_ratio": 0.0}
                        )["retention_ratio"]
                        for name in (
                            "standard_vehicle",
                            "large_vehicle",
                            "vulnerable_road_user",
                            "special_object",
                        )
                    },
                    "below_0_05_retention_ratio": by_truncation.get(
                        "below_0.05", {"retention_ratio": 0.0}
                    )["retention_ratio"],
                    "all_non_ambiguous_reviewed_cases_match": result[
                        "all_non_ambiguous_reviewed_cases_match"
                    ],
                }
            )

    matching = [
        result for result in results
        if result["all_non_ambiguous_reviewed_cases_match"]
    ]
    matching.sort(
        key=lambda result: (
            -result["fragment_guard_rejected_rows"],
            result["ratio_floor"],
            result["fragment_area_floor_px2"],
        )
    )

    print("front_tele valid evidence rows:", len(rows))
    print("Candidate results:", len(results))
    print("Candidates matching reviewed non-ambiguous cases:", len(matching))
    print("Top matching candidates:")
    for result in matching[:10]:
        family_stats = result["by_family"]
        print(
            f"  ratio_floor={result['ratio_floor']:.3f} "
            f"fragment_area_floor={result['fragment_area_floor_px2']:.0f} "
            f"guard_rejected={result['fragment_guard_rejected_rows']} "
            f"retention={result['retention_vs_scale_eligible']:.6f} "
            f"vulnerable={family_stats.get('vulnerable_road_user', {'retention_ratio': 0.0})['retention_ratio']:.6f}"
        )
    print("JSON output:", OUTPUT_JSON)
    print("CSV output:", OUTPUT_CSV)
    print("PASS: front_tele fragment-guard shadow scan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
