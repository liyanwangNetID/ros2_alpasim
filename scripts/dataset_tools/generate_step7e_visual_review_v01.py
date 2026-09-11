#!/usr/bin/env python3
"""Step 7E.7 generate deterministic visual review cases for shadow thresholds.

This diagnostic reads the valid projection evidence table, selects boundary and
risk cases for the provisional camera-specific area OR height rule, and creates
annotated review images plus a manifest. It does not modify Scene Facts.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_ROOT = ANNOTATION_ROOT / "step7e_observability_visual_review_v01"
MANIFEST_PATH = OUTPUT_ROOT / "review_manifest.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "review_summary.json"

THRESHOLDS = {
    "front_wide": {"area_px2": 24.0, "height_px": 5.0},
    "front_tele": {"area_px2": 512.0, "height_px": 16.0},
    "cross_left": {"area_px2": 32.0, "height_px": 6.0},
    "cross_right": {"area_px2": 48.0, "height_px": 6.0},
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

CASES_PER_CATEGORY = 3
COLORS = {True: "#00ff66", False: "#ff3344"}
HULL_COLOR = "#ffdd00"


def family(row: dict[str, Any]) -> str:
    return CLASS_FAMILIES.get(str(row["label_class"]), "rare_fallback")


def classify(row: dict[str, Any]) -> tuple[bool, bool, bool]:
    threshold = THRESHOLDS[str(row["camera_name"])]
    area_pass = float(row["inside_image_hull_area_px"]) >= threshold["area_px2"]
    height_pass = float(row["projected_height_px"]) >= threshold["height_px"]
    return area_pass or height_pass, area_pass, height_pass


def normalized_margin(row: dict[str, Any]) -> float:
    threshold = THRESHOLDS[str(row["camera_name"])]
    area_margin = float(row["inside_image_hull_area_px"]) / threshold["area_px2"] - 1.0
    height_margin = float(row["projected_height_px"]) / threshold["height_px"] - 1.0
    return max(area_margin, height_margin)


def read_rows() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            camera = str(row["camera_name"])
            if camera not in THRESHOLDS:
                raise ValueError(f"Unexpected camera at line {line_number}: {camera}")
            result[camera].append(row)
    return dict(result)


def select_nearest(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    count: int = CASES_PER_CATEGORY,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if predicate(row)]
    candidates.sort(
        key=lambda row: (
            abs(normalized_margin(row)),
            str(row["anchor_id"]),
            str(row["track_id"]),
        )
    )
    return candidates[:count]


def select_farthest_rejected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if not classify(row)[0]]
    candidates.sort(
        key=lambda row: (
            -float(row["minimum_depth_m"]),
            str(row["anchor_id"]),
            str(row["track_id"]),
        )
    )
    return candidates[:CASES_PER_CATEGORY]


def select_cases(rows_by_camera: dict[str, list[dict[str, Any]]]):
    selections: list[tuple[str, dict[str, Any]]] = []
    for camera_name in sorted(rows_by_camera):
        rows = rows_by_camera[camera_name]
        categories = {
            "just_rejected": lambda row: not classify(row)[0],
            "just_retained": lambda row: classify(row)[0],
            "area_only_retained": lambda row: (
                classify(row)[0] and classify(row)[1] and not classify(row)[2]
            ),
            "height_only_retained": lambda row: (
                classify(row)[0] and not classify(row)[1] and classify(row)[2]
            ),
            "vulnerable_rejected": lambda row: (
                not classify(row)[0] and family(row) == "vulnerable_road_user"
            ),
            "severely_truncated_retained": lambda row: (
                classify(row)[0]
                and float(row["inside_image_hull_ratio"]) < 0.25
            ),
        }
        for category, predicate in categories.items():
            for row in select_nearest(rows, predicate):
                selections.append((category, row))
        for row in select_farthest_rejected(rows):
            selections.append(("farthest_rejected", row))
    return selections


def draw_polygon(draw: ImageDraw.ImageDraw, points, *, fill: str, width: int) -> None:
    if len(points) < 2:
        return
    coordinates = [(float(point["u"]), float(point["v"])) for point in points]
    draw.line(coordinates + [coordinates[0]], fill=fill, width=width)


def make_review_image(
    row: dict[str, Any],
    *,
    reader_cache: dict[str, DrivingClipReader],
) -> tuple[Image.Image, str]:
    clip_id = str(row["clip_id"])
    reader = reader_cache.setdefault(
        clip_id,
        DrivingClipReader(ALPASIM_DATA_ROOT / clip_id),
    )
    camera_name = str(row["camera_name"])
    anchor_ns = int(row["anchor_ns"])
    exact = reader.camera_indexes[camera_name].exact(anchor_ns)
    if exact is None:
        raise RuntimeError(f"Exact image unavailable for {row['anchor_id']}/{camera_name}")
    image_path = exact.value.image_path
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    keep, area_pass, height_pass = classify(row)
    color = COLORS[keep]
    bbox = row["clipped_bbox"]
    draw.rectangle(
        (
            float(bbox["min_u"]),
            float(bbox["min_v"]),
            float(bbox["max_u"]),
            float(bbox["max_v"]),
        ),
        outline=color,
        width=3,
    )
    draw_polygon(draw, row["clipped_hull"], fill=HULL_COLOR, width=2)

    threshold = THRESHOLDS[camera_name]
    lines = [
        f"{camera_name} track={row['track_id']} class={row['label_class']}",
        f"decision={'KEEP' if keep else 'REJECT'} area_pass={area_pass} height_pass={height_pass}",
        f"area={float(row['inside_image_hull_area_px']):.1f}/{threshold['area_px2']:.0f}px2 height={float(row['projected_height_px']):.2f}/{threshold['height_px']:.0f}px",
        f"inside_ratio={float(row['inside_image_hull_ratio']):.3f} min_depth={float(row['minimum_depth_m']):.1f}m",
    ]
    panel_height = 16 * len(lines) + 8
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((8, 6 + 16 * index), line, fill=color if index == 1 else "white", font=font)
    return image, str(image_path)


def main() -> int:
    rows_by_camera = read_rows()
    selections = select_cases(rows_by_camera)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_manifest = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    reader_cache: dict[str, DrivingClipReader] = {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    manifest_rows: list[dict[str, Any]] = []

    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        for index, (category, row) in enumerate(selections, start=1):
            camera_name = str(row["camera_name"])
            keep, area_pass, height_pass = classify(row)
            image, source_image_path = make_review_image(
                row,
                reader_cache=reader_cache,
            )
            directory = OUTPUT_ROOT / camera_name / category
            directory.mkdir(parents=True, exist_ok=True)
            file_name = (
                f"{index:03d}_{row['anchor_id']}_track_{row['track_id']}.jpg"
            )
            output_path = directory / file_name
            image.save(output_path, quality=95)

            record = {
                "review_index": index,
                "category": category,
                "camera_name": camera_name,
                "anchor_id": str(row["anchor_id"]),
                "clip_id": str(row["clip_id"]),
                "anchor_ns": int(row["anchor_ns"]),
                "track_id": str(row["track_id"]),
                "label_class": str(row["label_class"]),
                "class_family": family(row),
                "candidate_visible": keep,
                "area_pass": area_pass,
                "height_pass": height_pass,
                "inside_image_hull_area_px": float(row["inside_image_hull_area_px"]),
                "projected_height_px": float(row["projected_height_px"]),
                "inside_image_hull_ratio": float(row["inside_image_hull_ratio"]),
                "minimum_depth_m": float(row["minimum_depth_m"]),
                "source_image_path": source_image_path,
                "review_image_path": str(output_path),
                "human_review": None,
                "human_review_notes": None,
            }
            manifest_rows.append(record)
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[camera_name][category] += 1

    os.replace(temporary_manifest, MANIFEST_PATH)
    summary = {
        "schema_version": "step7e-observability-visual-review-v01",
        "input_path": str(INPUT_PATH),
        "thresholds": THRESHOLDS,
        "cases_per_category": CASES_PER_CATEGORY,
        "review_case_count": len(manifest_rows),
        "counts_by_camera_and_category": {
            camera: dict(sorted(value.items()))
            for camera, value in sorted(counts.items())
        },
        "legend": {
            "green_bbox": "candidate_visible=True",
            "red_bbox": "candidate_visible=False",
            "yellow_polygon": "image-clipped projected AABB hull",
        },
        "manifest_path": str(MANIFEST_PATH),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Review cases:", len(manifest_rows))
    for camera_name in sorted(counts):
        print(camera_name, dict(sorted(counts[camera_name].items())))
    print("Review root:", OUTPUT_ROOT)
    print("Manifest:", MANIFEST_PATH)
    print("Summary:", SUMMARY_PATH)
    print("PASS: Step 7E visual review cases generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
