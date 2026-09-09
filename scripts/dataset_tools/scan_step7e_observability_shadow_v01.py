#!/usr/bin/env python3
"""Step 7E.5 shadow-scan camera-specific area OR height rules.

Reads the valid Step 7E projection evidence table. No Scene Fact or annotation
record is modified. Each camera is scanned independently over a candidate grid:

    keep = inside_image_hull_area_px >= area_threshold
           OR projected_height_px >= height_threshold

The output reports row and unique Actor-anchor retention, class-family
retention, depth-band retention, truncation-band retention, and rejection
reason composition for every candidate pair.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from project_paths import ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_JSON = ANNOTATION_ROOT / "step7e_observability_shadow_scan_v01.json"
OUTPUT_CSV = ANNOTATION_ROOT / "step7e_observability_shadow_scan_v01.csv"

CAMERA_GRIDS = {
    "front_wide": {
        "area_px": (16.0, 24.0, 32.0, 48.0, 64.0),
        "height_px": (4.0, 5.0, 6.0, 7.0, 8.0),
    },
    "front_tele": {
        "area_px": (128.0, 192.0, 256.0, 384.0, 512.0),
        "height_px": (12.0, 14.0, 16.0, 18.0, 20.0),
    },
    "cross_left": {
        "area_px": (16.0, 24.0, 32.0, 48.0, 64.0),
        "height_px": (4.0, 5.0, 6.0, 7.0, 8.0),
    },
    "cross_right": {
        "area_px": (16.0, 24.0, 32.0, 48.0, 64.0),
        "height_px": (4.0, 5.0, 6.0, 7.0, 8.0),
    },
}

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

DEPTH_EDGES = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0, float("inf"))


def depth_band(value: float) -> str:
    for lower, upper in zip(DEPTH_EDGES, DEPTH_EDGES[1:]):
        if lower <= value < upper:
            high = "inf" if upper == float("inf") else f"{upper:g}"
            return f"[{lower:g},{high})"
    raise ValueError(f"Unexpected depth value: {value}")


def truncation_band(ratio: float) -> str:
    if ratio >= 1.0 - 1e-9:
        return "not_truncated"
    if ratio >= 0.75:
        return "inside_75_to_100_percent"
    if ratio >= 0.50:
        return "inside_50_to_75_percent"
    if ratio >= 0.25:
        return "inside_25_to_50_percent"
    return "inside_below_25_percent"


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def read_rows() -> dict[str, list[dict[str, Any]]]:
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "step7e-projection-evidence-v01":
                raise ValueError(f"Unexpected schema at line {line_number}")
            camera = str(row["camera_name"])
            if camera not in CAMERA_GRIDS:
                raise ValueError(f"Unexpected camera at line {line_number}: {camera}")
            by_camera[camera].append(row)
    return dict(by_camera)


def counter_ratio(retained: Counter, total: Counter) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "total": total[key],
            "retained": retained[key],
            "retention_ratio": ratio(retained[key], total[key]),
        }
        for key in sorted(total)
    }


def evaluate(
    rows: list[dict[str, Any]],
    *,
    camera_name: str,
    area_threshold: float,
    height_threshold: float,
) -> dict[str, Any]:
    total_family: Counter[str] = Counter()
    kept_family: Counter[str] = Counter()
    total_class: Counter[str] = Counter()
    kept_class: Counter[str] = Counter()
    total_depth: Counter[str] = Counter()
    kept_depth: Counter[str] = Counter()
    total_truncation: Counter[str] = Counter()
    kept_truncation: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    total_actor_keys: set[tuple[str, str]] = set()
    kept_actor_keys: set[tuple[str, str]] = set()
    kept_rows = 0

    for row in rows:
        actor_class = str(row["label_class"])
        family = CLASS_FAMILIES.get(actor_class, "rare_fallback")
        distance = depth_band(float(row["minimum_depth_m"]))
        truncation = truncation_band(float(row["inside_image_hull_ratio"]))
        actor_key = (str(row["anchor_id"]), str(row["track_id"]))
        area_pass = float(row["inside_image_hull_area_px"]) >= area_threshold
        height_pass = float(row["projected_height_px"]) >= height_threshold
        keep = area_pass or height_pass

        total_family[family] += 1
        total_class[actor_class] += 1
        total_depth[distance] += 1
        total_truncation[truncation] += 1
        total_actor_keys.add(actor_key)

        if keep:
            kept_rows += 1
            kept_family[family] += 1
            kept_class[actor_class] += 1
            kept_depth[distance] += 1
            kept_truncation[truncation] += 1
            kept_actor_keys.add(actor_key)
        else:
            rejection_reasons["below_area_and_height"] += 1

    return {
        "camera_name": camera_name,
        "area_threshold_px2": area_threshold,
        "height_threshold_px": height_threshold,
        "rule": "area_gte_threshold OR height_gte_threshold",
        "row_total": len(rows),
        "row_retained": kept_rows,
        "row_rejected": len(rows) - kept_rows,
        "row_retention_ratio": ratio(kept_rows, len(rows)),
        "unique_anchor_actor_total": len(total_actor_keys),
        "unique_anchor_actor_retained": len(kept_actor_keys),
        "unique_anchor_actor_retention_ratio": ratio(
            len(kept_actor_keys), len(total_actor_keys)
        ),
        "by_family": counter_ratio(kept_family, total_family),
        "by_actor_class": counter_ratio(kept_class, total_class),
        "by_minimum_depth_band_m": counter_ratio(kept_depth, total_depth),
        "by_image_truncation_band": counter_ratio(
            kept_truncation, total_truncation
        ),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def main() -> int:
    rows_by_camera = read_rows()
    expected_cameras = set(CAMERA_GRIDS)
    if set(rows_by_camera) != expected_cameras:
        raise RuntimeError(
            f"Camera mismatch: expected {expected_cameras}, got {set(rows_by_camera)}"
        )

    results: list[dict[str, Any]] = []
    for camera_name in sorted(CAMERA_GRIDS):
        grid = CAMERA_GRIDS[camera_name]
        rows = rows_by_camera[camera_name]
        for area_threshold in grid["area_px"]:
            for height_threshold in grid["height_px"]:
                results.append(
                    evaluate(
                        rows,
                        camera_name=camera_name,
                        area_threshold=area_threshold,
                        height_threshold=height_threshold,
                    )
                )

    report = {
        "schema_version": "step7e-observability-shadow-scan-v01",
        "description": (
            "Shadow scan only. No Scene Fact or annotation was modified. "
            "No distance, truncation, class, or occlusion hard gate was applied."
        ),
        "input_path": str(INPUT_PATH),
        "candidate_grids": CAMERA_GRIDS,
        "class_families": CLASS_FAMILIES,
        "result_count": len(results),
        "results": results,
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_fields = (
        "camera_name",
        "area_threshold_px2",
        "height_threshold_px",
        "row_total",
        "row_retained",
        "row_rejected",
        "row_retention_ratio",
        "unique_anchor_actor_total",
        "unique_anchor_actor_retained",
        "unique_anchor_actor_retention_ratio",
        "standard_vehicle_retention_ratio",
        "large_vehicle_retention_ratio",
        "vulnerable_road_user_retention_ratio",
        "special_object_retention_ratio",
        "rare_fallback_retention_ratio",
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            family = result["by_family"]
            writer.writerow(
                {
                    "camera_name": result["camera_name"],
                    "area_threshold_px2": result["area_threshold_px2"],
                    "height_threshold_px": result["height_threshold_px"],
                    "row_total": result["row_total"],
                    "row_retained": result["row_retained"],
                    "row_rejected": result["row_rejected"],
                    "row_retention_ratio": result["row_retention_ratio"],
                    "unique_anchor_actor_total": result[
                        "unique_anchor_actor_total"
                    ],
                    "unique_anchor_actor_retained": result[
                        "unique_anchor_actor_retained"
                    ],
                    "unique_anchor_actor_retention_ratio": result[
                        "unique_anchor_actor_retention_ratio"
                    ],
                    **{
                        f"{name}_retention_ratio": family.get(
                            name, {"retention_ratio": 0.0}
                        )["retention_ratio"]
                        for name in (
                            "standard_vehicle",
                            "large_vehicle",
                            "vulnerable_road_user",
                            "special_object",
                            "rare_fallback",
                        )
                    },
                }
            )

    print("Evidence rows by camera:")
    for camera_name in sorted(rows_by_camera):
        print(" ", camera_name, len(rows_by_camera[camera_name]))
    print("Candidate results:", len(results))
    print("JSON output:", OUTPUT_JSON)
    print("CSV output:", OUTPUT_CSV)
    print("PASS: Step 7E observability shadow grid scanned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
