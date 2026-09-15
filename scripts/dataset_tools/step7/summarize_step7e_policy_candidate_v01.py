#!/usr/bin/env python3
"""Step 7E.13 summarize the first complete observability policy candidate.

Candidate rules derived from the completed shadow scans and visual review:

- front_wide: projected_height_px >= 5
- cross_left: projected_height_px >= 6
- cross_right: projected_height_px >= 6
- front_tele:
    (inside_area_px2 >= 512 OR projected_height_px >= 16)
    AND inside_ratio >= 0.10

The script reads valid projection evidence only and reports row-level and
anchor/Actor-level retention. It is read-only with respect to Scene Facts.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from project_paths import ANNOTATION_ROOT

INPUT_PATH = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT_PATH = ANNOTATION_ROOT / "step7e_observability_policy_candidate_v01.json"

POLICY = {
    "front_wide": {
        "minimum_projected_height_px": 5.0,
    },
    "cross_left": {
        "minimum_projected_height_px": 6.0,
    },
    "cross_right": {
        "minimum_projected_height_px": 6.0,
    },
    "front_tele": {
        "minimum_inside_image_hull_area_px": 512.0,
        "minimum_projected_height_px": 16.0,
        "minimum_inside_image_hull_ratio": 0.10,
        "scale_operator": "OR",
        "fragment_guard_operator": "AND",
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


def family(label_class: str) -> str:
    return CLASS_FAMILIES.get(label_class, "rare_fallback")


def depth_band(value: float) -> str:
    for lower, upper in zip(DEPTH_EDGES, DEPTH_EDGES[1:]):
        if lower <= value < upper:
            upper_text = "inf" if upper == float("inf") else f"{upper:g}"
            return f"[{lower:g},{upper_text})"
    raise ValueError(f"Unexpected minimum depth: {value}")


def retained(row: dict[str, Any]) -> tuple[bool, str | None]:
    camera = str(row["camera_name"])
    height = float(row["projected_height_px"])

    if camera in ("front_wide", "cross_left", "cross_right"):
        threshold = float(POLICY[camera]["minimum_projected_height_px"])
        if height < threshold:
            return False, "below_minimum_projected_height"
        return True, None

    if camera == "front_tele":
        area = float(row["inside_image_hull_area_px"])
        inside_ratio = float(row["inside_image_hull_ratio"])
        scale_pass = (
            area >= float(POLICY[camera]["minimum_inside_image_hull_area_px"])
            or height >= float(POLICY[camera]["minimum_projected_height_px"])
        )
        if not scale_pass:
            return False, "below_primary_area_and_height"
        if inside_ratio < float(
            POLICY[camera]["minimum_inside_image_hull_ratio"]
        ):
            return False, "below_minimum_inside_image_hull_ratio"
        return True, None

    raise ValueError(f"Unexpected camera: {camera}")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def counter_summary(kept: Counter, total: Counter) -> dict[str, Any]:
    return {
        key: {
            "total": total[key],
            "retained": kept[key],
            "rejected": total[key] - kept[key],
            "retention_ratio": ratio(kept[key], total[key]),
        }
        for key in sorted(total)
    }


def main() -> int:
    row_total = 0
    row_retained = 0
    total_camera: Counter[str] = Counter()
    kept_camera: Counter[str] = Counter()
    total_family: Counter[str] = Counter()
    kept_family: Counter[str] = Counter()
    total_class: Counter[str] = Counter()
    kept_class: Counter[str] = Counter()
    total_depth: Counter[str] = Counter()
    kept_depth: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()

    actor_camera_total: dict[tuple[str, str], set[str]] = defaultdict(set)
    actor_camera_kept: dict[tuple[str, str], set[str]] = defaultdict(set)
    actor_classes: dict[tuple[str, str], str] = {}

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "step7e-projection-evidence-v01":
                raise ValueError(f"Unexpected schema at line {line_number}")

            camera = str(row["camera_name"])
            label_class = str(row["label_class"])
            row_family = family(label_class)
            row_depth = depth_band(float(row["minimum_depth_m"]))
            actor_key = (str(row["anchor_id"]), str(row["track_id"]))
            decision, reason = retained(row)

            row_total += 1
            total_camera[camera] += 1
            total_family[row_family] += 1
            total_class[label_class] += 1
            total_depth[row_depth] += 1
            actor_camera_total[actor_key].add(camera)
            actor_classes[actor_key] = label_class

            if decision:
                row_retained += 1
                kept_camera[camera] += 1
                kept_family[row_family] += 1
                kept_class[label_class] += 1
                kept_depth[row_depth] += 1
                actor_camera_kept[actor_key].add(camera)
            else:
                assert reason is not None
                rejection_reasons[f"{camera}|{reason}"] += 1

    actor_total = len(actor_camera_total)
    actor_retained = sum(bool(actor_camera_kept[key]) for key in actor_camera_total)
    retained_camera_count: Counter[int] = Counter(
        len(actor_camera_kept[key]) for key in actor_camera_total
    )
    actor_total_family: Counter[str] = Counter()
    actor_kept_family: Counter[str] = Counter()
    for actor_key, label_class in actor_classes.items():
        row_family = family(label_class)
        actor_total_family[row_family] += 1
        if actor_camera_kept[actor_key]:
            actor_kept_family[row_family] += 1

    report = {
        "schema_version": "step7e-observability-policy-candidate-v01",
        "description": (
            "First complete shadow policy candidate. Valid geometric "
            "projections only; Actor-to-Actor occlusion is not evaluated."
        ),
        "input_path": str(INPUT_PATH),
        "policy": POLICY,
        "row_level": {
            "valid_projection_rows": row_total,
            "retained_rows": row_retained,
            "rejected_rows": row_total - row_retained,
            "retention_ratio": ratio(row_retained, row_total),
            "by_camera": counter_summary(kept_camera, total_camera),
            "by_family": counter_summary(kept_family, total_family),
            "by_actor_class": counter_summary(kept_class, total_class),
            "by_minimum_depth_band_m": counter_summary(
                kept_depth, total_depth
            ),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "anchor_actor_level_among_geometrically_valid_projections": {
            "unique_anchor_actor_count": actor_total,
            "retained_by_at_least_one_camera": actor_retained,
            "rejected_by_all_geometrically_valid_cameras": (
                actor_total - actor_retained
            ),
            "retention_ratio": ratio(actor_retained, actor_total),
            "retained_camera_count_distribution": {
                str(count): retained_camera_count[count]
                for count in sorted(retained_camera_count)
            },
            "by_family": counter_summary(
                actor_kept_family, actor_total_family
            ),
        },
        "limitations": [
            "Input contains valid geometric projections only.",
            "Actor-to-Actor occlusion is not evaluated.",
            "Image appearance is not evaluated by the policy.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Valid projection rows:", row_total)
    print("Retained rows:", row_retained)
    print("Rejected rows:", row_total - row_retained)
    print("Row retention ratio:", ratio(row_retained, row_total))
    print("By camera:")
    for camera, item in report["row_level"]["by_camera"].items():
        print(
            f"  {camera}: retained={item['retained']} "
            f"rejected={item['rejected']} "
            f"ratio={item['retention_ratio']:.6f}"
        )
    print("By family:")
    for name, item in report["row_level"]["by_family"].items():
        print(
            f"  {name}: retained={item['retained']} "
            f"rejected={item['rejected']} "
            f"ratio={item['retention_ratio']:.6f}"
        )
    print("Unique anchor/Actor pairs with valid projection:", actor_total)
    print("Retained by at least one camera:", actor_retained)
    print(
        "Rejected by all geometrically valid cameras:",
        actor_total - actor_retained,
    )
    print("Output:", OUTPUT_PATH)

    assert row_total == 163717
    assert row_retained + (row_total - row_retained) == row_total
    assert sum(total_camera.values()) == row_total
    assert sum(retained_camera_count.values()) == actor_total

    print("PASS: Step 7E complete policy candidate summarized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
