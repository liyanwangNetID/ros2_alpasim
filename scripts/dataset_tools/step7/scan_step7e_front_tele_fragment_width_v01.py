#!/usr/bin/env python3
"""Step 7E.12 shadow-scan front-tele fragment width guards.

Primary scale:
    area >= 512 px2 OR height >= 16 px

Fragment guard candidates:
    inside_ratio >= ratio_floor
    OR (inside_area >= fragment_area_floor AND clipped_width >= width_floor)

Read-only shadow scan. No Scene Fact is modified.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from project_paths import ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_JSON = ANNOTATION_ROOT / "step7e_front_tele_fragment_width_scan_v01.json"
OUTPUT_CSV = ANNOTATION_ROOT / "step7e_front_tele_fragment_width_scan_v01.csv"

CAMERA_NAME = "front_tele"
PRIMARY_AREA_PX2 = 512.0
PRIMARY_HEIGHT_PX = 16.0
RATIO_FLOORS = (0.10, 0.125, 0.15, 0.20, 0.25)
FRAGMENT_AREA_FLOORS_PX2 = (256.0, 384.0, 512.0, 768.0, 1024.0)
WIDTH_FLOORS_PX = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0)

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

# Clear outcomes from prior visual review. Ambiguous cases are excluded.
REVIEW_CASES = {
    ("test_clip_741_1414042502155000", "27"): False,
    ("test_clip_440_1196517731585000", "24"): False,
    ("test_clip_251_2354945826999000", "352"): True,
    ("test_clip_860_10022807928000", "101"): True,
    ("test_clip_169_10870718569000", "106"): False,
    ("test_clip_375_2225809508051000", "333"): False,
    ("test_clip_722_18203009247000", "26"): False,
    ("test_clip_419_1482573510399000", "48"): False,
    ("test_clip_262_183408928430000", "212"): False,
    ("test_clip_688_184226322239000", "197"): False,
    ("test_clip_554_1728913511118000", "375"): False,
    ("test_clip_748_2051502317244000", "274"): False,
    ("test_clip_826_764520925326000", "70"): False,
    ("test_clip_815_269423525625000", "252"): False,
    ("test_clip_101_4820268515510000", "241"): False,
    ("test_clip_006_4798878812152000", "193"): False,
    ("test_clip_629_291739108565000", "145"): False,
    ("test_clip_340_34498826682000", "71"): False,
    ("test_clip_119_1975560731004000", "145"): False,
}


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def family(label_class: str) -> str:
    return CLASS_FAMILIES.get(label_class, "rare_fallback")


def clipped_width(row: dict[str, Any]) -> float:
    bbox = row["clipped_bbox"]
    return float(bbox["max_u"]) - float(bbox["min_u"])


def clipped_height(row: dict[str, Any]) -> float:
    bbox = row["clipped_bbox"]
    return float(bbox["max_v"]) - float(bbox["min_v"])


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("camera_name") != CAMERA_NAME:
                continue
            width = clipped_width(row)
            height = clipped_height(row)
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"Invalid clipped bbox at line {line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError("No front_tele evidence rows found")
    return rows


def primary_pass(row: dict[str, Any]) -> bool:
    return (
        float(row["inside_image_hull_area_px"]) >= PRIMARY_AREA_PX2
        or float(row["projected_height_px"]) >= PRIMARY_HEIGHT_PX
    )


def decision(
    row: dict[str, Any],
    *,
    ratio_floor: float,
    fragment_area_floor: float,
    width_floor: float,
) -> bool:
    fragment_guard = (
        float(row["inside_image_hull_ratio"]) >= ratio_floor
        or (
            float(row["inside_image_hull_area_px"]) >= fragment_area_floor
            and clipped_width(row) >= width_floor
        )
    )
    return primary_pass(row) and fragment_guard


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
    width_floor: float,
) -> dict[str, Any]:
    eligible = [row for row in rows if primary_pass(row)]
    total_family: Counter[str] = Counter()
    kept_family: Counter[str] = Counter()
    kept_count = 0

    for row in eligible:
        row_family = family(str(row["label_class"]))
        total_family[row_family] += 1
        if decision(
            row,
            ratio_floor=ratio_floor,
            fragment_area_floor=fragment_area_floor,
            width_floor=width_floor,
        ):
            kept_count += 1
            kept_family[row_family] += 1

    row_index = {
        (str(row["anchor_id"]), str(row["track_id"])): row
        for row in rows
    }
    reviewed: dict[str, Any] = {}
    all_match = True
    for reviewed_key, expected in REVIEW_CASES.items():
        row = row_index.get(reviewed_key)
        item_key = "|".join(reviewed_key)
        if row is None:
            reviewed[item_key] = {"found": False, "expected": expected}
            all_match = False
            continue
        actual = decision(
            row,
            ratio_floor=ratio_floor,
            fragment_area_floor=fragment_area_floor,
            width_floor=width_floor,
        )
        match = actual == expected
        all_match &= match
        reviewed[item_key] = {
            "found": True,
            "expected": expected,
            "actual": actual,
            "match": match,
            "inside_ratio": float(row["inside_image_hull_ratio"]),
            "inside_area_px2": float(row["inside_image_hull_area_px"]),
            "clipped_width_px": clipped_width(row),
            "clipped_height_px": clipped_height(row),
        }

    return {
        "ratio_floor": ratio_floor,
        "fragment_area_floor_px2": fragment_area_floor,
        "fragment_width_floor_px": width_floor,
        "scale_eligible_rows": len(eligible),
        "retained_rows": kept_count,
        "guard_rejected_rows": len(eligible) - kept_count,
        "retention_vs_scale_eligible": ratio(kept_count, len(eligible)),
        "by_family": summarize_counter(kept_family, total_family),
        "reviewed_cases": reviewed,
        "all_reviewed_cases_match": all_match,
    }


def family_retention(result: dict[str, Any], name: str) -> float:
    item = result["by_family"].get(name)
    return 0.0 if item is None else float(item["retention_ratio"])


def main() -> int:
    rows = read_rows()
    results = [
        evaluate(
            rows,
            ratio_floor=ratio_floor,
            fragment_area_floor=area_floor,
            width_floor=width_floor,
        )
        for ratio_floor in RATIO_FLOORS
        for area_floor in FRAGMENT_AREA_FLOORS_PX2
        for width_floor in WIDTH_FLOORS_PX
    ]

    report = {
        "schema_version": "step7e-front-tele-fragment-width-scan-v01",
        "description": "Read-only shadow scan; no Scene Fact was modified.",
        "input_path": str(INPUT_PATH),
        "primary_rule": {
            "inside_area_px2": PRIMARY_AREA_PX2,
            "projected_height_px": PRIMARY_HEIGHT_PX,
            "operator": "OR",
        },
        "fragment_guard_rule": (
            "inside_ratio >= ratio_floor OR "
            "(inside_area >= fragment_area_floor AND clipped_width >= width_floor)"
        ),
        "ratio_floors": RATIO_FLOORS,
        "fragment_area_floors_px2": FRAGMENT_AREA_FLOORS_PX2,
        "fragment_width_floors_px": WIDTH_FLOORS_PX,
        "review_case_count": len(REVIEW_CASES),
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
        "fragment_width_floor_px",
        "scale_eligible_rows",
        "retained_rows",
        "guard_rejected_rows",
        "retention_vs_scale_eligible",
        "standard_vehicle_retention_ratio",
        "large_vehicle_retention_ratio",
        "vulnerable_road_user_retention_ratio",
        "special_object_retention_ratio",
        "all_reviewed_cases_match",
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "ratio_floor": result["ratio_floor"],
                    "fragment_area_floor_px2": result[
                        "fragment_area_floor_px2"
                    ],
                    "fragment_width_floor_px": result[
                        "fragment_width_floor_px"
                    ],
                    "scale_eligible_rows": result["scale_eligible_rows"],
                    "retained_rows": result["retained_rows"],
                    "guard_rejected_rows": result["guard_rejected_rows"],
                    "retention_vs_scale_eligible": result[
                        "retention_vs_scale_eligible"
                    ],
                    "standard_vehicle_retention_ratio": family_retention(
                        result, "standard_vehicle"
                    ),
                    "large_vehicle_retention_ratio": family_retention(
                        result, "large_vehicle"
                    ),
                    "vulnerable_road_user_retention_ratio": family_retention(
                        result, "vulnerable_road_user"
                    ),
                    "special_object_retention_ratio": family_retention(
                        result, "special_object"
                    ),
                    "all_reviewed_cases_match": result[
                        "all_reviewed_cases_match"
                    ],
                }
            )

    matching = [result for result in results if result["all_reviewed_cases_match"]]
    matching.sort(
        key=lambda result: (
            -result["guard_rejected_rows"],
            -family_retention(result, "vulnerable_road_user"),
            result["ratio_floor"],
            result["fragment_area_floor_px2"],
            result["fragment_width_floor_px"],
        )
    )

    print("front_tele valid evidence rows:", len(rows))
    print("Candidate results:", len(results))
    print("Reviewed cases:", len(REVIEW_CASES))
    print("Candidates matching all reviewed cases:", len(matching))
    print("Top matching candidates:")
    for result in matching[:12]:
        print(
            f"  ratio={result['ratio_floor']:.3f} "
            f"area={result['fragment_area_floor_px2']:.0f} "
            f"width={result['fragment_width_floor_px']:.0f} "
            f"rejected={result['guard_rejected_rows']} "
            f"retention={result['retention_vs_scale_eligible']:.6f} "
            f"vulnerable={family_retention(result, 'vulnerable_road_user'):.6f} "
            f"large={family_retention(result, 'large_vehicle'):.6f}"
        )
    print("JSON output:", OUTPUT_JSON)
    print("CSV output:", OUTPUT_CSV)
    print("PASS: front_tele Fragment Width shadow scan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
