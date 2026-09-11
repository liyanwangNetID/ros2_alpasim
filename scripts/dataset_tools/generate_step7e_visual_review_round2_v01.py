#!/usr/bin/env python3
"""Step 7E.9 generate deduplicated second-round visual review cases.

The script compares the original camera-specific primary OR rule with a
provisional joint-floor rule, selects deterministic boundary and transition
cases, deduplicates each anchor/camera/track across categories, and writes
annotated images plus a review manifest. It does not modify Scene Facts.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_ROOT = ANNOTATION_ROOT / "step7e_observability_visual_review_round2_v01"
MANIFEST_PATH = OUTPUT_ROOT / "review_manifest.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "review_summary.json"

POLICIES = {
    "front_wide": {
        "primary_area_px2": 24.0,
        "primary_height_px": 5.0,
        "area_floor_px2": 12.0,
        "height_floor_px": 4.5,
    },
    "front_tele": {
        "primary_area_px2": 512.0,
        "primary_height_px": 16.0,
        "area_floor_px2": 128.0,
        "height_floor_px": 14.0,
    },
    "cross_left": {
        "primary_area_px2": 32.0,
        "primary_height_px": 6.0,
        "area_floor_px2": 12.0,
        "height_floor_px": 5.0,
    },
    "cross_right": {
        "primary_area_px2": 48.0,
        "primary_height_px": 6.0,
        "area_floor_px2": 16.0,
        "height_floor_px": 5.5,
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

CASES_PER_CATEGORY = 3
CATEGORY_ORDER = (
    "old_keep_new_reject",
    "new_just_rejected",
    "new_just_retained",
    "new_keep_near_area_floor",
    "new_keep_near_height_floor",
    "vulnerable_new_rejected",
)


def family(row: dict[str, Any]) -> str:
    return CLASS_FAMILIES.get(str(row["label_class"]), "rare_fallback")


def decisions(row: dict[str, Any]) -> tuple[bool, bool, dict[str, bool]]:
    policy = POLICIES[str(row["camera_name"])]
    area = float(row["inside_image_hull_area_px"])
    height = float(row["projected_height_px"])
    primary_area_pass = area >= policy["primary_area_px2"]
    primary_height_pass = height >= policy["primary_height_px"]
    area_floor_pass = area >= policy["area_floor_px2"]
    height_floor_pass = height >= policy["height_floor_px"]
    old_keep = primary_area_pass or primary_height_pass
    new_keep = area_floor_pass and height_floor_pass and old_keep
    return old_keep, new_keep, {
        "primary_area_pass": primary_area_pass,
        "primary_height_pass": primary_height_pass,
        "area_floor_pass": area_floor_pass,
        "height_floor_pass": height_floor_pass,
    }


def new_margin(row: dict[str, Any]) -> float:
    policy = POLICIES[str(row["camera_name"])]
    area = float(row["inside_image_hull_area_px"])
    height = float(row["projected_height_px"])
    primary_margin = max(
        area / policy["primary_area_px2"] - 1.0,
        height / policy["primary_height_px"] - 1.0,
    )
    area_floor_margin = area / policy["area_floor_px2"] - 1.0
    height_floor_margin = height / policy["height_floor_px"] - 1.0
    return min(primary_margin, area_floor_margin, height_floor_margin)


def area_floor_margin(row: dict[str, Any]) -> float:
    policy = POLICIES[str(row["camera_name"])]
    return float(row["inside_image_hull_area_px"]) / policy["area_floor_px2"] - 1.0


def height_floor_margin(row: dict[str, Any]) -> float:
    policy = POLICIES[str(row["camera_name"])]
    return float(row["projected_height_px"]) / policy["height_floor_px"] - 1.0


def read_rows() -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            camera = str(row["camera_name"])
            if camera not in POLICIES:
                raise ValueError(f"Unexpected camera at line {line_number}: {camera}")
            rows[camera].append(row)
    return dict(rows)


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["anchor_id"]),
        str(row["camera_name"]),
        str(row["track_id"]),
    )


def take(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    score: Callable[[dict[str, Any]], tuple],
    used: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if predicate(row) and key(row) not in used]
    candidates.sort(key=score)
    selected = candidates[:CASES_PER_CATEGORY]
    used.update(key(row) for row in selected)
    return selected


def select_cases(rows_by_camera: dict[str, list[dict[str, Any]]]):
    selected: list[tuple[str, dict[str, Any]]] = []
    for camera in sorted(rows_by_camera):
        rows = rows_by_camera[camera]
        used: set[tuple[str, str, str]] = set()
        specifications = {
            "old_keep_new_reject": (
                lambda row: decisions(row)[0] and not decisions(row)[1],
                lambda row: (abs(new_margin(row)), key(row)),
            ),
            "new_just_rejected": (
                lambda row: not decisions(row)[1],
                lambda row: (abs(new_margin(row)), key(row)),
            ),
            "new_just_retained": (
                lambda row: decisions(row)[1],
                lambda row: (abs(new_margin(row)), key(row)),
            ),
            "new_keep_near_area_floor": (
                lambda row: decisions(row)[1],
                lambda row: (abs(area_floor_margin(row)), key(row)),
            ),
            "new_keep_near_height_floor": (
                lambda row: decisions(row)[1],
                lambda row: (abs(height_floor_margin(row)), key(row)),
            ),
            "vulnerable_new_rejected": (
                lambda row: (
                    not decisions(row)[1]
                    and family(row) == "vulnerable_road_user"
                ),
                lambda row: (abs(new_margin(row)), key(row)),
            ),
        }
        for category in CATEGORY_ORDER:
            predicate, score = specifications[category]
            for row in take(rows, predicate, score, used):
                selected.append((category, row))
    return selected


def bbox_tuple(bbox: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(bbox["min_u"]),
        float(bbox["min_v"]),
        float(bbox["max_u"]),
        float(bbox["max_v"]),
    )


def draw_polygon(draw: ImageDraw.ImageDraw, points: list[dict[str, Any]]) -> None:
    coordinates = [(float(point["u"]), float(point["v"])) for point in points]
    if len(coordinates) >= 2:
        draw.line(coordinates + [coordinates[0]], fill="#ffdd00", width=2)


def render(row: dict[str, Any], cache: dict[str, DrivingClipReader]):
    clip_id = str(row["clip_id"])
    reader = cache.setdefault(
        clip_id,
        DrivingClipReader(ALPASIM_DATA_ROOT / clip_id),
    )
    exact = reader.camera_indexes[str(row["camera_name"])].exact(int(row["anchor_ns"]))
    if exact is None:
        raise RuntimeError(f"Exact image unavailable for {key(row)}")
    source_path = Path(exact.value.image_path)
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    old_keep, new_keep, flags = decisions(row)
    color = "#00ff66" if new_keep else "#ff3344"
    draw.rectangle(bbox_tuple(row["clipped_bbox"]), outline=color, width=3)
    draw_polygon(draw, row["clipped_hull"])

    policy = POLICIES[str(row["camera_name"])]
    lines = [
        f"{row['camera_name']} track={row['track_id']} class={row['label_class']}",
        f"OLD={'KEEP' if old_keep else 'REJECT'} NEW={'KEEP' if new_keep else 'REJECT'}",
        f"area={float(row['inside_image_hull_area_px']):.1f} floor={policy['area_floor_px2']:.0f} primary={policy['primary_area_px2']:.0f}px2",
        f"height={float(row['projected_height_px']):.2f} floor={policy['height_floor_px']:.1f} primary={policy['primary_height_px']:.0f}px",
        f"inside_ratio={float(row['inside_image_hull_ratio']):.3f} min_depth={float(row['minimum_depth_m']):.1f}m",
    ]
    panel_height = 16 * len(lines) + 8
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((8, 6 + 16 * index), line, fill=color if index == 1 else "white", font=font)
    return image, source_path, old_keep, new_keep, flags


def main() -> int:
    rows_by_camera = read_rows()
    selections = select_cases(rows_by_camera)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_manifest = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    cache: dict[str, DrivingClipReader] = {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    selected_keys: set[tuple[str, str, str]] = set()
    manifest_rows: list[dict[str, Any]] = []

    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        for index, (category, row) in enumerate(selections, start=1):
            row_key = key(row)
            if row_key in selected_keys:
                raise RuntimeError(f"Duplicate second-round review case: {row_key}")
            selected_keys.add(row_key)
            image, source_path, old_keep, new_keep, flags = render(row, cache)
            camera = str(row["camera_name"])
            directory = OUTPUT_ROOT / camera / category
            directory.mkdir(parents=True, exist_ok=True)
            output_path = directory / (
                f"{index:03d}_{row['anchor_id']}_track_{row['track_id']}.jpg"
            )
            image.save(output_path, quality=95)

            record = {
                "review_index": index,
                "category": category,
                "camera_name": camera,
                "anchor_id": str(row["anchor_id"]),
                "clip_id": str(row["clip_id"]),
                "anchor_ns": int(row["anchor_ns"]),
                "track_id": str(row["track_id"]),
                "label_class": str(row["label_class"]),
                "class_family": family(row),
                "old_candidate_visible": old_keep,
                "new_candidate_visible": new_keep,
                **flags,
                "inside_image_hull_area_px": float(row["inside_image_hull_area_px"]),
                "projected_height_px": float(row["projected_height_px"]),
                "inside_image_hull_ratio": float(row["inside_image_hull_ratio"]),
                "minimum_depth_m": float(row["minimum_depth_m"]),
                "source_image_path": str(source_path),
                "review_image_path": str(output_path),
                "human_review": None,
                "human_review_notes": None,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_rows.append(record)
            counts[camera][category] += 1

    os.replace(temporary_manifest, MANIFEST_PATH)
    summary = {
        "schema_version": "step7e-observability-visual-review-round2-v01",
        "input_path": str(INPUT_PATH),
        "policies": POLICIES,
        "cases_per_category": CASES_PER_CATEGORY,
        "category_order": CATEGORY_ORDER,
        "deduplication_key": ["anchor_id", "camera_name", "track_id"],
        "review_case_count": len(manifest_rows),
        "duplicate_case_count": 0,
        "counts_by_camera_and_category": {
            camera: dict(sorted(category_counts.items()))
            for camera, category_counts in sorted(counts.items())
        },
        "legend": {
            "green_bbox": "new_candidate_visible=True",
            "red_bbox": "new_candidate_visible=False",
            "yellow_polygon": "image-clipped projected AABB hull",
        },
        "manifest_path": str(MANIFEST_PATH),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Review cases:", len(manifest_rows))
    print("Duplicate cases:", 0)
    for camera in sorted(counts):
        print(camera, dict(sorted(counts[camera].items())))
    print("Review root:", OUTPUT_ROOT)
    print("Manifest:", MANIFEST_PATH)
    print("Summary:", SUMMARY_PATH)
    print("PASS: Step 7E round-2 visual review cases generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
