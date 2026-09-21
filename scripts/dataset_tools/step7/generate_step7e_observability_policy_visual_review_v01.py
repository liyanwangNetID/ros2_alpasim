#!/usr/bin/env python3
"""Generate annotated images for narrowed Step 7E observability policies."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from step2.clip_reader import DrivingClipReader
from step7.actor_observability_policy_visual_review_v01 import (
    prepare_policy_visual_review_cases,
)

SELECTION_PATH = Path("/tmp/step7e_observability_policy_review_selection_v01.json")
V02_PATH = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
PROJECTION_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_ROOT = ANNOTATION_ROOT / "step7e_observability_policy_visual_review_v01"
MANIFEST_PATH = OUTPUT_ROOT / "review_manifest.jsonl"
SUMMARY_PATH = OUTPUT_ROOT / "review_summary.json"


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return tuple(rows)


def bbox_tuple(box: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(box["min_u"]), float(box["min_v"]),
        float(box["max_u"]), float(box["max_v"]),
    )


def render(case: dict[str, Any], cache: dict[str, DrivingClipReader]):
    clip_id = str(case["clip_id"])
    reader = cache.setdefault(
        clip_id, DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    )
    camera = str(case["camera_name"])
    exact = reader.camera_indexes[camera].exact(int(case["anchor_ns"]))
    if exact is None:
        raise RuntimeError(
            f"Exact image unavailable for {case['anchor_id']}/{camera}/"
            f"track={case['track_id']}"
        )
    source = Path(exact.value.image_path)
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle(bbox_tuple(case["clipped_bbox"]), outline="#ff3344", width=3)
    points = [
        (float(point["u"]), float(point["v"]))
        for point in case["clipped_hull"]
    ]
    if len(points) >= 2:
        draw.line(points + [points[0]], fill="#ffdd00", width=2)
    occluders = ",".join(case["occluding_actor_ids"]) or "none"
    lines = [
        f"{camera} track={case['track_id']} class={case['label_class']}",
        f"category={case['review_category']}",
        f"winning={case['total_winning_cell_count']} occupied={case['total_occupied_cell_count']} visible_fraction={float(case['maximum_visible_fraction']):.6f}",
        f"area={float(case['inside_image_hull_area_px']):.1f}px2 height={float(case['projected_height_px']):.2f}px ratio={float(case['inside_image_hull_ratio']):.4f}",
        f"depth={float(case['minimum_depth_m']):.1f}m occluders={occluders}",
    ]
    panel_height = 16 * len(lines) + 8
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0))
    for index, text in enumerate(lines):
        draw.text(
            (8, 6 + 16 * index), text,
            fill="#ffcc33" if index == 1 else "white", font=font,
        )
    return image, source


def main() -> int:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected_ids = {
        (str(row["anchor_id"]), str(row["track_id"]))
        for row in selection["cases"]
    }
    v02_rows = tuple(
        row for row in read_jsonl(V02_PATH)
        if (str(row["anchor_id"]), str(row["track_id"])) in selected_ids
    )
    projection_rows = tuple(
        row for row in read_jsonl(PROJECTION_PATH)
        if (str(row["anchor_id"]), str(row["track_id"])) in selected_ids
    )
    prepared = prepare_policy_visual_review_cases(
        selection=selection,
        v02_rows=v02_rows,
        projection_rows=projection_rows,
    )
    if prepared["failure_count"]:
        raise RuntimeError(
            f"Review input preparation failed: {prepared['failure_counts']}"
        )
    if prepared["prepared_case_count"] != 36:
        raise RuntimeError(
            f"Expected 36 prepared cases, found {prepared['prepared_case_count']}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_manifest = MANIFEST_PATH.with_suffix(".jsonl.tmp")
    cache: dict[str, DrivingClipReader] = {}
    counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    records = []
    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        for index, case in enumerate(prepared["cases"], start=1):
            image, source = render(case, cache)
            category = str(case["review_category"])
            directory = OUTPUT_ROOT / category
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / (
                f"{index:03d}_{case['anchor_id']}_track_{case['track_id']}_"
                f"{case['camera_name']}.jpg"
            )
            image.save(target, quality=95)
            record = dict(case)
            record.update(
                {
                    "review_index": index,
                    "source_image_path": str(source),
                    "review_image_path": str(target),
                    "human_review": None,
                    "human_review_notes": None,
                }
            )
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            counts[category] += 1
            camera_counts[str(case["camera_name"])] += 1
    os.replace(temporary_manifest, MANIFEST_PATH)

    summary = {
        "schema_version": "step7e-observability-policy-visual-review-v01",
        "selection_path": str(SELECTION_PATH),
        "v02_evidence_path": str(V02_PATH),
        "projection_evidence_path": str(PROJECTION_PATH),
        "camera_selection_basis": "single_occlusion_winning_camera",
        "review_case_count": len(records),
        "failure_count": prepared["failure_count"],
        "failure_counts": prepared["failure_counts"],
        "counts_by_category": dict(sorted(counts.items())),
        "counts_by_camera": dict(sorted(camera_counts.items())),
        "manifest_path": str(MANIFEST_PATH),
        "legend": {
            "red_bbox": "selected Actor projection bounding box",
            "yellow_polygon": "image-clipped projected AABB hull",
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Step 7E observability policy visual review")
    print("review cases:", len(records))
    print("failures:", prepared["failure_count"], prepared["failure_counts"])
    print("counts by category:", dict(sorted(counts.items())))
    print("counts by camera:", dict(sorted(camera_counts.items())))
    print("review root:", OUTPUT_ROOT)
    print("manifest:", MANIFEST_PATH)
    print("summary:", SUMMARY_PATH)
    print("PASS: policy visual-review images generated without modifying evidence or labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
