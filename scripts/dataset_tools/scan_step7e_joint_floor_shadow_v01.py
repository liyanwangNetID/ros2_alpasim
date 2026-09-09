#!/usr/bin/env python3
"""Step 7E.8 shadow-scan joint area/height floor rules.

Reads Step 7E projection evidence and evaluates camera-specific rules of the form:

    keep = area >= area_floor
           and height >= height_floor
           and (area >= primary_area or height >= primary_height)

This is a read-only shadow scan. No Scene Fact is modified.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from project_paths import ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_JSON = ANNOTATION_ROOT / "step7e_observability_joint_floor_scan_v01.json"
OUTPUT_CSV = ANNOTATION_ROOT / "step7e_observability_joint_floor_scan_v01.csv"

# Primary thresholds are the current visual-review candidates.
PRIMARY = {
    "front_wide": {"area": 24.0, "height": 5.0},
    "front_tele": {"area": 512.0, "height": 16.0},
    "cross_left": {"area": 32.0, "height": 6.0},
    "cross_right": {"area": 48.0, "height": 6.0},
}

# Floor grids are deliberately small and local to the observed failure modes.
FLOORS = {
    "front_wide": {
        "area": (8.0, 12.0, 16.0, 20.0, 24.0),
        "height": (3.5, 4.0, 4.5, 5.0),
    },
    "front_tele": {
        "area": (128.0, 192.0, 256.0, 320.0),
        "height": (12.0, 14.0, 16.0),
    },
    "cross_left": {
        "area": (8.0, 12.0, 16.0, 24.0, 32.0),
        "height": (4.0, 4.5, 5.0, 5.5, 6.0),
    },
    "cross_right": {
        "area": (12.0, 16.0, 24.0, 32.0, 48.0),
        "height": (4.0, 4.5, 5.0, 5.5, 6.0),
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


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def depth_band(value: float) -> str:
    for lower, upper in zip(DEPTH_EDGES, DEPTH_EDGES[1:]):
        if lower <= value < upper:
            upper_text = "inf" if upper == float("inf") else f"{upper:g}"
            return f"[{lower:g},{upper_text})"
    raise ValueError(f"Unexpected depth: {value}")


def family(label_class: str) -> str:
    return CLASS_FAMILIES.get(label_class, "rare_fallback")


def read_rows() -> dict[str, list[dict[str, Any]]]:
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            camera = str(row["camera_name"])
            if camera not in PRIMARY:
                raise ValueError(f"Unexpected camera at line {line_number}: {camera}")
            by_camera[camera].append(row)
    return dict(by_camera)


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
    camera: str,
    area_floor: float,
    height_floor: float,
) -> dict[str, Any]:
    primary_area = PRIMARY[camera]["area"]
    primary_height = PRIMARY[camera]["height"]

    total_family: Counter[str] = Counter()
    kept_family: Counter[str] = Counter()
    total_depth: Counter[str] = Counter()
    kept_depth: Counter[str] = Counter()
    rejection_reason: Counter[str] = Counter()
    kept = 0

    for row in rows:
        area = float(row["inside_image_hull_area_px"])
        height = float(row["projected_height_px"])
        row_family = family(str(row["label_class"]))
        row_depth = depth_band(float(row["minimum_depth_m"]))

        area_floor_pass = area >= area_floor
        height_floor_pass = height >= height_floor
        primary_pass = area >= primary_area or height >= primary_height
        decision = area_floor_pass and height_floor_pass and primary_pass

        total_family[row_family] += 1
        total_depth[row_depth] += 1
        if decision:
            kept += 1
            kept_family[row_family] += 1
            kept_depth[row_depth] += 1
        else:
            if not area_floor_pass and not height_floor_pass:
                rejection_reason["below_both_floors"] += 1
            elif not area_floor_pass:
                rejection_reason["below_area_floor"] += 1
            elif not height_floor_pass:
                rejection_reason["below_height_floor"] += 1
            else:
                rejection_reason["below_primary_area_and_height"] += 1

    return {
        "camera_name": camera,
        "primary_area_px2": primary_area,
        "primary_height_px": primary_height,
        "area_floor_px2": area_floor,
        "height_floor_px": height_floor,
        "row_total": len(rows),
        "row_retained": kept,
        "row_rejected": len(rows) - kept,
        "row_retention_ratio": ratio(kept, len(rows)),
        "by_family": summarize_counter(kept_family, total_family),
        "by_minimum_depth_band_m": summarize_counter(kept_depth, total_depth),
        "rejection_reasons": dict(sorted(rejection_reason.items())),
    }


def family_retention(result: dict[str, Any], name: str) -> float:
    item = result["by_family"].get(name)
    return 0.0 if item is None else float(item["retention_ratio"])


def main() -> int:
    rows_by_camera = read_rows()
    if set(rows_by_camera) != set(PRIMARY):
        raise RuntimeError("Evidence does not contain exactly the four expected cameras")

    results: list[dict[str, Any]] = []
    for camera in sorted(PRIMARY):
        for area_floor in FLOORS[camera]["area"]:
            for height_floor in FLOORS[camera]["height"]:
                results.append(
                    evaluate(
                        rows_by_camera[camera],
                        camera=camera,
                        area_floor=area_floor,
                        height_floor=height_floor,
                    )
                )

    report = {
        "schema_version": "step7e-observability-joint-floor-scan-v01",
        "description": "Shadow scan only; no Scene Fact was modified.",
        "input_path": str(INPUT_PATH),
        "rule": (
            "area >= area_floor AND height >= height_floor AND "
            "(area >= primary_area OR height >= primary_height)"
        ),
        "primary_thresholds": PRIMARY,
        "floor_grids": FLOORS,
        "result_count": len(results),
        "results": results,
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = (
        "camera_name",
        "primary_area_px2",
        "primary_height_px",
        "area_floor_px2",
        "height_floor_px",
        "row_total",
        "row_retained",
        "row_rejected",
        "row_retention_ratio",
        "standard_vehicle_retention_ratio",
        "large_vehicle_retention_ratio",
        "vulnerable_road_user_retention_ratio",
        "special_object_retention_ratio",
        "depth_80_160_retention_ratio",
        "depth_160_inf_retention_ratio",
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            depth = result["by_minimum_depth_band_m"]
            writer.writerow(
                {
                    "camera_name": result["camera_name"],
                    "primary_area_px2": result["primary_area_px2"],
                    "primary_height_px": result["primary_height_px"],
                    "area_floor_px2": result["area_floor_px2"],
                    "height_floor_px": result["height_floor_px"],
                    "row_total": result["row_total"],
                    "row_retained": result["row_retained"],
                    "row_rejected": result["row_rejected"],
                    "row_retention_ratio": result["row_retention_ratio"],
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
                    "depth_80_160_retention_ratio": depth.get(
                        "[80,160)", {"retention_ratio": 0.0}
                    )["retention_ratio"],
                    "depth_160_inf_retention_ratio": depth.get(
                        "[160,inf)", {"retention_ratio": 0.0}
                    )["retention_ratio"],
                }
            )

    print("Evidence rows by camera:")
    for camera in sorted(rows_by_camera):
        print(" ", camera, len(rows_by_camera[camera]))
    print("Candidate results:", len(results))
    print("JSON output:", OUTPUT_JSON)
    print("CSV output:", OUTPUT_CSV)
    print("PASS: Step 7E joint-floor shadow scan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
