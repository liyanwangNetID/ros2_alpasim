#!/usr/bin/env python3
"""Step 7E.11 generate front-tele Fragment Guard visual-review cases.

Candidate policy:
    primary_scale_pass = area >= 512 px2 OR height >= 16 px
    fragment_guard_pass = inside_ratio >= 0.075 OR area >= 256 px2
    candidate_visible = primary_scale_pass AND fragment_guard_pass

The script reads projection evidence, selects deterministic boundary/risk cases,
deduplicates anchor/camera/track combinations, and creates annotated images.
No Scene Fact is modified.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_ROOT = (
    ANNOTATION_ROOT
    / "step7e_front_tele_fragment_guard_visual_review_v01"
)
MANIFEST_PATH = OUTPUT_ROOT / "review_manifest.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "review_summary.json"

CAMERA_NAME = "front_tele"
PRIMARY_AREA_PX2 = 512.0
PRIMARY_HEIGHT_PX = 16.0
RATIO_FLOOR = 0.075
FRAGMENT_AREA_FLOOR_PX2 = 256.0
CASES_PER_CATEGORY = 3

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

CATEGORY_ORDER = (
    "just_below_ratio_rejected",
    "just_above_ratio_retained",
    "low_ratio_large_area_retained",
    "vulnerable_guard_rejected",
    "large_vehicle_guard_rejected",
)


def class_family(row: dict[str, Any]) -> str:
    return CLASS_FAMILIES.get(str(row["label_class"]), "rare_fallback")


def primary_scale_pass(row: dict[str, Any]) -> bool:
    return (
        float(row["inside_image_hull_area_px"]) >= PRIMARY_AREA_PX2
        or float(row["projected_height_px"]) >= PRIMARY_HEIGHT_PX
    )


def guard_pass(row: dict[str, Any]) -> bool:
    return (
        float(row["inside_image_hull_ratio"]) >= RATIO_FLOOR
        or float(row["inside_image_hull_area_px"])
        >= FRAGMENT_AREA_FLOOR_PX2
    )


def final_decision(row: dict[str, Any]) -> bool:
    return primary_scale_pass(row) and guard_pass(row)


def unique_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["anchor_id"]),
        str(row["camera_name"]),
        str(row["track_id"]),
    )


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("camera_name") != CAMERA_NAME:
                continue
            if row.get("schema_version") != "step7e-projection-evidence-v01":
                raise ValueError(
                    f"Unexpected schema at line {line_number}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError("No front_tele evidence rows found")
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    predicate: Callable[[dict[str, Any]], bool],
    score: Callable[[dict[str, Any]], tuple],
    used: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if predicate(row) and unique_key(row) not in used
    ]
    candidates.sort(key=score)
    selected = candidates[:CASES_PER_CATEGORY]
    used.update(unique_key(row) for row in selected)
    return selected


def select_cases(rows: list[dict[str, Any]]):
    used: set[tuple[str, str, str]] = set()
    selections: list[tuple[str, dict[str, Any]]] = []

    specifications = {
        "just_below_ratio_rejected": (
            lambda row: (
                primary_scale_pass(row)
                and not final_decision(row)
                and float(row["inside_image_hull_ratio"]) < RATIO_FLOOR
                and float(row["inside_image_hull_area_px"])
                < FRAGMENT_AREA_FLOOR_PX2
            ),
            lambda row: (
                RATIO_FLOOR - float(row["inside_image_hull_ratio"]),
                FRAGMENT_AREA_FLOOR_PX2
                - float(row["inside_image_hull_area_px"]),
                unique_key(row),
            ),
        ),
        "just_above_ratio_retained": (
            lambda row: (
                final_decision(row)
                and float(row["inside_image_hull_ratio"]) >= RATIO_FLOOR
                and float(row["inside_image_hull_area_px"])
                < FRAGMENT_AREA_FLOOR_PX2
            ),
            lambda row: (
                float(row["inside_image_hull_ratio"]) - RATIO_FLOOR,
                unique_key(row),
            ),
        ),
        "low_ratio_large_area_retained": (
            lambda row: (
                final_decision(row)
                and float(row["inside_image_hull_ratio"]) < RATIO_FLOOR
                and float(row["inside_image_hull_area_px"])
                >= FRAGMENT_AREA_FLOOR_PX2
            ),
            lambda row: (
                float(row["inside_image_hull_area_px"])
                - FRAGMENT_AREA_FLOOR_PX2,
                float(row["inside_image_hull_ratio"]),
                unique_key(row),
            ),
        ),
        "vulnerable_guard_rejected": (
            lambda row: (
                primary_scale_pass(row)
                and not guard_pass(row)
                and class_family(row) == "vulnerable_road_user"
            ),
            lambda row: (
                abs(float(row["inside_image_hull_ratio"]) - RATIO_FLOOR),
                unique_key(row),
            ),
        ),
        "large_vehicle_guard_rejected": (
            lambda row: (
                primary_scale_pass(row)
                and not guard_pass(row)
                and class_family(row) == "large_vehicle"
            ),
            lambda row: (
                abs(float(row["inside_image_hull_ratio"]) - RATIO_FLOOR),
                unique_key(row),
            ),
        ),
    }

    for category in CATEGORY_ORDER:
        predicate, score = specifications[category]
        for row in select_rows(
            rows,
            predicate=predicate,
            score=score,
            used=used,
        ):
            selections.append((category, row))
    return selections


def bbox_tuple(bbox: dict[str, Any]):
    return (
        float(bbox["min_u"]),
        float(bbox["min_v"]),
        float(bbox["max_u"]),
        float(bbox["max_v"]),
    )


def draw_hull(
    draw: ImageDraw.ImageDraw,
    points: list[dict[str, Any]],
) -> None:
    coordinates = [
        (float(point["u"]), float(point["v"]))
        for point in points
    ]
    if len(coordinates) >= 2:
        draw.line(
            coordinates + [coordinates[0]],
            fill="#ffdd00",
            width=2,
        )


def render(
    row: dict[str, Any],
    reader_cache: dict[str, DrivingClipReader],
):
    clip_id = str(row["clip_id"])
    reader = reader_cache.setdefault(
        clip_id,
        DrivingClipReader(ALPASIM_DATA_ROOT / clip_id),
    )
    exact = reader.camera_indexes[CAMERA_NAME].exact(
        int(row["anchor_ns"])
    )
    if exact is None:
        raise RuntimeError(
            f"Exact image unavailable for {unique_key(row)}"
        )

    source_path = Path(exact.value.image_path)
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    keep = final_decision(row)
    color = "#00ff66" if keep else "#ff3344"

    draw.rectangle(
        bbox_tuple(row["clipped_bbox"]),
        outline=color,
        width=3,
    )
    draw_hull(draw, row["clipped_hull"])

    lines = [
        f"front_tele track={row['track_id']} class={row['label_class']}",
        f"SCALE={'PASS' if primary_scale_pass(row) else 'FAIL'} GUARD={'PASS' if guard_pass(row) else 'FAIL'} FINAL={'KEEP' if keep else 'REJECT'}",
        f"area={float(row['inside_image_hull_area_px']):.1f} primary={PRIMARY_AREA_PX2:.0f} fragment_floor={FRAGMENT_AREA_FLOOR_PX2:.0f}px2",
        f"height={float(row['projected_height_px']):.2f} primary={PRIMARY_HEIGHT_PX:.0f}px",
        f"inside_ratio={float(row['inside_image_hull_ratio']):.4f} ratio_floor={RATIO_FLOOR:.3f} min_depth={float(row['minimum_depth_m']):.1f}m",
    ]
    panel_height = 16 * len(lines) + 8
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text(
            (8, 6 + 16 * index),
            line,
            fill=color if index == 1 else "white",
            font=font,
        )
    return image, source_path


def main() -> int:
    rows = read_rows()
    selections = select_cases(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_manifest = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    reader_cache: dict[str, DrivingClipReader] = {}
    counts: Counter[str] = Counter()
    selected_keys: set[tuple[str, str, str]] = set()

    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        for index, (category, row) in enumerate(selections, start=1):
            row_key = unique_key(row)
            if row_key in selected_keys:
                raise RuntimeError(f"Duplicate review case: {row_key}")
            selected_keys.add(row_key)

            image, source_path = render(row, reader_cache)
            directory = OUTPUT_ROOT / category
            directory.mkdir(parents=True, exist_ok=True)
            output_path = directory / (
                f"{index:03d}_{row['anchor_id']}_track_{row['track_id']}.jpg"
            )
            image.save(output_path, quality=95)

            record = {
                "review_index": index,
                "category": category,
                "anchor_id": str(row["anchor_id"]),
                "clip_id": str(row["clip_id"]),
                "anchor_ns": int(row["anchor_ns"]),
                "camera_name": CAMERA_NAME,
                "track_id": str(row["track_id"]),
                "label_class": str(row["label_class"]),
                "class_family": class_family(row),
                "primary_scale_pass": primary_scale_pass(row),
                "fragment_guard_pass": guard_pass(row),
                "candidate_visible": final_decision(row),
                "inside_image_hull_area_px": float(
                    row["inside_image_hull_area_px"]
                ),
                "projected_height_px": float(row["projected_height_px"]),
                "inside_image_hull_ratio": float(
                    row["inside_image_hull_ratio"]
                ),
                "minimum_depth_m": float(row["minimum_depth_m"]),
                "source_image_path": str(source_path),
                "review_image_path": str(output_path),
                "human_review": None,
                "human_review_notes": None,
            }
            manifest.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            counts[category] += 1

    os.replace(temporary_manifest, MANIFEST_PATH)
    summary = {
        "schema_version": (
            "step7e-front-tele-fragment-guard-visual-review-v01"
        ),
        "input_path": str(INPUT_PATH),
        "camera_name": CAMERA_NAME,
        "primary_scale_rule": {
            "area_px2": PRIMARY_AREA_PX2,
            "height_px": PRIMARY_HEIGHT_PX,
            "operator": "OR",
        },
        "fragment_guard_rule": {
            "inside_image_hull_ratio_floor": RATIO_FLOOR,
            "inside_image_hull_area_floor_px2": (
                FRAGMENT_AREA_FLOOR_PX2
            ),
            "operator": "OR",
        },
        "category_order": CATEGORY_ORDER,
        "cases_per_category": CASES_PER_CATEGORY,
        "review_case_count": len(selections),
        "duplicate_case_count": 0,
        "counts_by_category": dict(sorted(counts.items())),
        "manifest_path": str(MANIFEST_PATH),
        "legend": {
            "green_bbox": "candidate_visible=True",
            "red_bbox": "candidate_visible=False",
            "yellow_polygon": "image-clipped projected AABB hull",
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    print("Review cases:", len(selections))
    print("Duplicate cases:", 0)
    print("Counts by category:", dict(sorted(counts.items())))
    print("Review root:", OUTPUT_ROOT)
    print("Manifest:", MANIFEST_PATH)
    print("Summary:", SUMMARY_PATH)
    print(
        "PASS: front_tele Fragment Guard visual review generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
