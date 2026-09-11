#!/usr/bin/env python3
"""Profile projection geometry for candidate cameras with no sampled surface.

Joins the formal Step 7E combined-evidence JSONL with the existing valid
projection-evidence JSONL by anchor_id, track_id, and camera_name. It reports
projection size, truncation, depth, and image-boundary proximity for every
candidate_without_sampled_surface camera without recomputing geometry or
applying final visibility thresholds.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

from project_paths import ANNOTATION_ROOT


COMBINED_INPUT = (
    ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v01.jsonl"
)
PROJECTION_INPUT = ANNOTATION_ROOT / "step7e_projection_evidence_v01.jsonl"
OUTPUT = Path(
    "/tmp/step7e_candidate_without_sampled_surface_projection_profile.json"
)
TARGET_STATUS = "candidate_without_sampled_surface"
EXPECTED_ACTOR_COUNT = 56


def load_targets():
    targets = {}
    with COMBINED_INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evidence_status") != TARGET_STATUS:
                continue
            anchor_id = str(row["anchor_id"])
            track_id = str(row["track_id"])
            cameras = tuple(
                row["geometric_candidate_without_sampled_surface_camera_names"]
            )
            if not cameras:
                raise RuntimeError(
                    f"Target row has no missing-surface camera at line {line_number}"
                )
            for camera_name in cameras:
                key = (anchor_id, track_id, str(camera_name))
                if key in targets:
                    raise RuntimeError(f"Duplicate target key: {key}")
                targets[key] = row
    actor_keys = {(key[0], key[1]) for key in targets}
    if len(actor_keys) != EXPECTED_ACTOR_COUNT:
        raise RuntimeError(
            f"Target Actor baseline changed: actual={len(actor_keys)}, "
            f"expected={EXPECTED_ACTOR_COUNT}"
        )
    return targets


def load_matching_projections(targets):
    matches = {}
    with PROJECTION_INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["anchor_id"]),
                str(row["track_id"]),
                str(row["camera_name"]),
            )
            if key not in targets:
                continue
            if key in matches:
                raise RuntimeError(
                    f"Duplicate projection evidence at line {line_number}: {key}"
                )
            matches[key] = row
    return matches


def describe(values):
    values = tuple(float(value) for value in values)
    if not values:
        return None
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def main() -> int:
    targets = load_targets()
    projections = load_matching_projections(targets)
    missing_projection_keys = sorted(set(targets) - set(projections))

    rows = []
    camera_counts = Counter()
    class_counts = Counter()
    truncated_counts = Counter()
    boundary_touch_counts = Counter()

    for key in sorted(projections):
        projection = projections[key]
        target = targets[key]
        bbox = projection["clipped_bbox"]
        projected_bbox = projection["projected_bbox"]
        truncated = bool(projection["truncated"])
        touches_image_boundary = (
            float(bbox["min_u"]) <= 0.0
            or float(bbox["min_v"]) <= 0.0
            or float(bbox["max_u"]) >= float(projected_bbox["max_u"])
            and truncated
            or float(bbox["max_v"]) >= float(projected_bbox["max_v"])
            and truncated
        )
        camera_name = key[2]
        label_class = str(target["label_class"])
        camera_counts[camera_name] += 1
        class_counts[label_class] += 1
        truncated_counts[str(truncated).lower()] += 1
        boundary_touch_counts[str(touches_image_boundary).lower()] += 1

        rows.append(
            {
                "anchor_id": key[0],
                "track_id": key[1],
                "camera_name": camera_name,
                "label_class": label_class,
                "is_static": bool(target["is_static"]),
                "inside_image_hull_area_px": float(
                    projection["inside_image_hull_area_px"]
                ),
                "projected_height_px": float(
                    projection["projected_height_px"]
                ),
                "inside_image_hull_ratio": float(
                    projection["inside_image_hull_ratio"]
                ),
                "minimum_depth_m": float(projection["minimum_depth_m"]),
                "maximum_depth_m": float(projection["maximum_depth_m"]),
                "camera_sample_count": int(projection["camera_sample_count"]),
                "truncated": truncated,
                "touches_image_boundary": touches_image_boundary,
            }
        )

    report = {
        "schema_version": (
            "step7e-candidate-without-surface-projection-profile-v01"
        ),
        "target_actor_count": EXPECTED_ACTOR_COUNT,
        "target_candidate_camera_count": len(targets),
        "matched_projection_count": len(rows),
        "missing_projection_count": len(missing_projection_keys),
        "missing_projection_keys": [list(key) for key in missing_projection_keys],
        "rows_by_camera": dict(sorted(camera_counts.items())),
        "rows_by_actor_class": dict(sorted(class_counts.items())),
        "truncated_counts": dict(sorted(truncated_counts.items())),
        "touches_image_boundary_counts": dict(
            sorted(boundary_touch_counts.items())
        ),
        "inside_image_hull_area_px": describe(
            row["inside_image_hull_area_px"] for row in rows
        ),
        "projected_height_px": describe(
            row["projected_height_px"] for row in rows
        ),
        "inside_image_hull_ratio": describe(
            row["inside_image_hull_ratio"] for row in rows
        ),
        "minimum_depth_m": describe(row["minimum_depth_m"] for row in rows),
        "camera_sample_count": describe(
            row["camera_sample_count"] for row in rows
        ),
        "rows": rows,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Candidate-without-surface projection profile")
    for key in (
        "target_actor_count",
        "target_candidate_camera_count",
        "matched_projection_count",
        "missing_projection_count",
        "rows_by_camera",
        "rows_by_actor_class",
        "truncated_counts",
        "touches_image_boundary_counts",
        "inside_image_hull_area_px",
        "projected_height_px",
        "inside_image_hull_ratio",
        "minimum_depth_m",
        "camera_sample_count",
    ):
        print(f"{key}:", report[key])
    if missing_projection_keys:
        print("missing_projection_keys:")
        for key in missing_projection_keys:
            print(" ", key)
    print("smallest matched projections by hull area:")
    for row in sorted(rows, key=lambda value: value["inside_image_hull_area_px"])[:20]:
        print(
            f"  anchor={row['anchor_id']} track={row['track_id']} "
            f"camera={row['camera_name']} class={row['label_class']} "
            f"area={row['inside_image_hull_area_px']:.6f} "
            f"height={row['projected_height_px']:.6f} "
            f"ratio={row['inside_image_hull_ratio']:.6f} "
            f"min_depth={row['minimum_depth_m']:.6f} "
            f"truncated={row['truncated']}"
        )
    print("output:", OUTPUT)
    print(
        "PASS: candidate-camera projection evidence was joined and profiled "
        "without recomputing geometry."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
