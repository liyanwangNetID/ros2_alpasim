#!/usr/bin/env python3
"""Generate focused visual review for the selected front-tele width guard.

Candidate:
  scale = area >= 512 OR height >= 16
  guard = ratio >= 0.10 OR (area >= 384 AND clipped_width >= 6)
  keep = scale AND guard

Read-only with respect to Scene Facts and projection evidence.
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
OUTPUT_ROOT = ANNOTATION_ROOT / "step7e_front_tele_width_guard_review_v01"
MANIFEST_PATH = OUTPUT_ROOT / "review_manifest.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "review_summary.json"

PRIMARY_AREA = 512.0
PRIMARY_HEIGHT = 16.0
RATIO_FLOOR = 0.10
FRAGMENT_AREA_FLOOR = 384.0
WIDTH_FLOOR = 6.0
CASES_PER_CATEGORY = 3

CATEGORIES = (
    "just_below_ratio_rejected",
    "just_above_ratio_retained",
    "width_escape_just_rejected",
    "width_escape_just_retained",
    "low_ratio_escape_retained",
    "vulnerable_rejected",
    "large_vehicle_rejected",
)

FAMILIES = {
    "automobile": "standard_vehicle", "other_vehicle": "standard_vehicle",
    "heavy_truck": "large_vehicle", "bus": "large_vehicle",
    "trailer": "large_vehicle", "train_or_tram_car": "large_vehicle",
    "person": "vulnerable_road_user", "rider": "vulnerable_road_user",
    "stroller": "vulnerable_road_user", "protruding_object": "special_object",
    "animal": "rare_fallback",
}


def family(row: dict[str, Any]) -> str:
    return FAMILIES.get(str(row["label_class"]), "rare_fallback")


def width(row: dict[str, Any]) -> float:
    box = row["clipped_bbox"]
    return float(box["max_u"]) - float(box["min_u"])


def scale(row: dict[str, Any]) -> bool:
    return float(row["inside_image_hull_area_px"]) >= PRIMARY_AREA or float(row["projected_height_px"]) >= PRIMARY_HEIGHT


def guard(row: dict[str, Any]) -> bool:
    return float(row["inside_image_hull_ratio"]) >= RATIO_FLOOR or (
        float(row["inside_image_hull_area_px"]) >= FRAGMENT_AREA_FLOOR and width(row) >= WIDTH_FLOOR
    )


def keep(row: dict[str, Any]) -> bool:
    return scale(row) and guard(row)


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["anchor_id"]), str(row["camera_name"]), str(row["track_id"])


def read_rows() -> list[dict[str, Any]]:
    rows = []
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row["camera_name"] == "front_tele" and scale(row):
                    rows.append(row)
    return rows


def take(rows, predicate: Callable, score: Callable, used: set, count=CASES_PER_CATEGORY):
    candidates = [row for row in rows if predicate(row) and key(row) not in used]
    candidates.sort(key=lambda row: (score(row), key(row)))
    chosen = candidates[:count]
    used.update(key(row) for row in chosen)
    return chosen


def select(rows):
    used = set()
    specs = {
        "just_below_ratio_rejected": (
            lambda r: not keep(r) and float(r["inside_image_hull_ratio"]) < RATIO_FLOOR,
            lambda r: RATIO_FLOOR - float(r["inside_image_hull_ratio"]),
        ),
        "just_above_ratio_retained": (
            lambda r: keep(r) and float(r["inside_image_hull_ratio"]) >= RATIO_FLOOR and float(r["inside_image_hull_area_px"]) < FRAGMENT_AREA_FLOOR,
            lambda r: float(r["inside_image_hull_ratio"]) - RATIO_FLOOR,
        ),
        "width_escape_just_rejected": (
            lambda r: not keep(r) and float(r["inside_image_hull_ratio"]) < RATIO_FLOOR and float(r["inside_image_hull_area_px"]) >= FRAGMENT_AREA_FLOOR and width(r) < WIDTH_FLOOR,
            lambda r: WIDTH_FLOOR - width(r),
        ),
        "width_escape_just_retained": (
            lambda r: keep(r) and float(r["inside_image_hull_ratio"]) < RATIO_FLOOR and float(r["inside_image_hull_area_px"]) >= FRAGMENT_AREA_FLOOR and width(r) >= WIDTH_FLOOR,
            lambda r: width(r) - WIDTH_FLOOR,
        ),
        "low_ratio_escape_retained": (
            lambda r: keep(r) and float(r["inside_image_hull_ratio"]) < RATIO_FLOOR,
            lambda r: float(r["inside_image_hull_ratio"]),
        ),
        "vulnerable_rejected": (
            lambda r: not keep(r) and family(r) == "vulnerable_road_user",
            lambda r: abs(float(r["inside_image_hull_ratio"]) - RATIO_FLOOR),
        ),
        "large_vehicle_rejected": (
            lambda r: not keep(r) and family(r) == "large_vehicle",
            lambda r: abs(float(r["inside_image_hull_ratio"]) - RATIO_FLOOR),
        ),
    }
    result = []
    for category in CATEGORIES:
        predicate, score = specs[category]
        result.extend((category, row) for row in take(rows, predicate, score, used))
    return result


def render(row, cache):
    clip = str(row["clip_id"])
    reader = cache.setdefault(clip, DrivingClipReader(ALPASIM_DATA_ROOT / clip))
    exact = reader.camera_indexes["front_tele"].exact(int(row["anchor_ns"]))
    if exact is None:
        raise RuntimeError(f"Missing exact image for {key(row)}")
    source = Path(exact.value.image_path)
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    color = "#00ff66" if keep(row) else "#ff3344"
    box = row["clipped_bbox"]
    draw.rectangle((float(box["min_u"]), float(box["min_v"]), float(box["max_u"]), float(box["max_v"])), outline=color, width=3)
    points = [(float(p["u"]), float(p["v"])) for p in row["clipped_hull"]]
    if len(points) >= 2:
        draw.line(points + [points[0]], fill="#ffdd00", width=2)
    lines = [
        f"front_tele track={row['track_id']} class={row['label_class']}",
        f"SCALE=PASS GUARD={'PASS' if guard(row) else 'FAIL'} FINAL={'KEEP' if keep(row) else 'REJECT'}",
        f"area={float(row['inside_image_hull_area_px']):.1f} primary=512 escape=384px2 width={width(row):.2f}/6px",
        f"height={float(row['projected_height_px']):.2f}/16px ratio={float(row['inside_image_hull_ratio']):.4f}/0.10",
        f"min_depth={float(row['minimum_depth_m']):.1f}m",
    ]
    draw.rectangle((0, 0, image.width, 88), fill=(0, 0, 0))
    font = ImageFont.load_default()
    for i, text in enumerate(lines):
        draw.text((8, 6 + 16*i), text, fill=color if i == 1 else "white", font=font)
    return image, source


def main() -> int:
    rows = read_rows()
    selections = select(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temp = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    cache = {}
    counts = Counter()
    with temp.open("w", encoding="utf-8") as manifest:
        for index, (category, row) in enumerate(selections, 1):
            image, source = render(row, cache)
            directory = OUTPUT_ROOT / category
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{index:03d}_{row['anchor_id']}_track_{row['track_id']}.jpg"
            image.save(target, quality=95)
            record = {
                "review_index": index, "category": category,
                "anchor_id": str(row["anchor_id"]), "clip_id": str(row["clip_id"]),
                "anchor_ns": int(row["anchor_ns"]), "camera_name": "front_tele",
                "track_id": str(row["track_id"]), "label_class": str(row["label_class"]),
                "class_family": family(row), "candidate_visible": keep(row),
                "inside_area_px2": float(row["inside_image_hull_area_px"]),
                "projected_height_px": float(row["projected_height_px"]),
                "inside_ratio": float(row["inside_image_hull_ratio"]),
                "clipped_width_px": width(row), "minimum_depth_m": float(row["minimum_depth_m"]),
                "source_image_path": str(source), "review_image_path": str(target),
                "human_review": None, "human_review_notes": None,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[category] += 1
    os.replace(temp, MANIFEST_PATH)
    summary = {
        "schema_version": "step7e-front-tele-width-guard-review-v01",
        "policy": {"primary_area": 512, "primary_height": 16, "ratio_floor": 0.10, "escape_area": 384, "width_floor": 6},
        "review_case_count": len(selections), "counts": dict(counts),
        "manifest_path": str(MANIFEST_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Review cases:", len(selections))
    print("Counts:", dict(counts))
    print("Review root:", OUTPUT_ROOT)
    print("PASS: focused width-guard review generated.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
