#!/usr/bin/env python3
"""Inspect exported Actors whose geometric candidates have no sampled surface.

Reads the formal Step 7E JSONL product, validates the summary baseline, and
reports every candidate_without_sampled_surface row grouped by Actor class and
candidate camera. No geometry is recomputed and no visibility threshold or
final label is applied.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from project_paths import ANNOTATION_ROOT
from scene_fact_schema_v01 import CAMERA_NAMES


INPUT = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v01.jsonl"
SUMMARY = (
    ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v01.summary.json"
)
TARGET_STATUS = "candidate_without_sampled_surface"
EXPECTED_COUNT = 56


def main() -> int:
    rows = []
    class_counts = Counter()
    candidate_camera_counts = Counter()
    missing_surface_camera_counts = Counter()
    reason_counts = Counter()
    aggregate_surface_counts = Counter()
    identities = set()

    with INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evidence_status") != TARGET_STATUS:
                continue

            identity = (str(row["anchor_id"]), str(row["track_id"]))
            if identity in identities:
                raise RuntimeError(
                    f"Duplicate target identity at line {line_number}: {identity}"
                )
            identities.add(identity)
            rows.append(row)
            class_counts[str(row["label_class"])] += 1
            candidate_camera_counts.update(
                row["geometric_candidate_camera_names"]
            )
            missing_surface_camera_counts.update(
                row[
                    "geometric_candidate_without_sampled_surface_camera_names"
                ]
            )
            reason_counts.update(row["reasons"])
            aggregate_surface_counts[
                "rows_with_any_evaluated_surface"
                if int(row["total_occupied_cell_count"]) > 0
                else "rows_without_any_evaluated_surface"
            ] += 1

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary_count = int(
        summary["evidence_status_counts"].get(TARGET_STATUS, 0)
    )
    if len(rows) != summary_count:
        raise RuntimeError(
            f"JSONL and summary target counts differ: "
            f"rows={len(rows)}, summary={summary_count}"
        )
    if len(rows) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Target baseline changed: actual={len(rows)}, "
            f"expected={EXPECTED_COUNT}"
        )

    canonical = set(CAMERA_NAMES)
    for row in rows:
        candidates = tuple(row["geometric_candidate_camera_names"])
        with_surface = tuple(
            row[
                "geometric_candidate_with_sampled_surface_camera_names"
            ]
        )
        without_surface = tuple(
            row[
                "geometric_candidate_without_sampled_surface_camera_names"
            ]
        )
        if not candidates:
            raise RuntimeError("Target row has no geometric candidate camera")
        if with_surface:
            raise RuntimeError(
                "candidate_without_sampled_surface row unexpectedly has a "
                "candidate camera with sampled surface"
            )
        if tuple(candidates) != tuple(without_surface):
            raise RuntimeError(
                "Candidate and candidate-without-surface camera sets differ"
            )
        if any(name not in canonical for name in candidates):
            raise RuntimeError("Target row contains a noncanonical camera")
        # Aggregate occlusion values cover all evaluated cameras, not only
        # geometric-candidate cameras. Therefore these values may be nonzero
        # even when every geometric candidate lacks a sampled surface.
        if row["maximum_visible_fraction"] is not None:
            value = float(row["maximum_visible_fraction"])
            if not 0.0 <= value <= 1.0:
                raise RuntimeError("Target row has invalid visible fraction")
        occupied = int(row["total_occupied_cell_count"])
        winning = int(row["total_winning_cell_count"])
        occluded = int(row["total_occluded_cell_count"])
        if min(occupied, winning, occluded) < 0:
            raise RuntimeError("Target row has a negative sampled-cell count")
        if winning + occluded != occupied:
            raise RuntimeError("Target row sampled-cell counts do not close")

    print("Candidate-without-sampled-surface export inspection")
    print("input:", INPUT)
    print("summary:", SUMMARY)
    print("count:", len(rows))
    print("rows_by_actor_class:", dict(sorted(class_counts.items())))
    print(
        "geometric_candidate_camera_counts:",
        dict(sorted(candidate_camera_counts.items())),
    )
    print(
        "missing_surface_camera_counts:",
        dict(sorted(missing_surface_camera_counts.items())),
    )
    print("reason_counts:", dict(sorted(reason_counts.items())))
    print(
        "aggregate_surface_counts:",
        dict(sorted(aggregate_surface_counts.items())),
    )
    print("rows:")

    for row in sorted(
        rows,
        key=lambda value: (str(value["anchor_id"]), str(value["track_id"])),
    ):
        print(
            f"  anchor={row['anchor_id']} "
            f"track={row['track_id']} "
            f"class={row['label_class']} "
            f"is_static={row['is_static']} "
            f"candidates={row['geometric_candidate_camera_names']} "
            f"evaluated={row['occlusion_evaluated_camera_names']} "
            f"winning={row['occlusion_winning_camera_names']} "
            f"occupied={row['total_occupied_cell_count']} "
            f"maximum_visible_fraction={row['maximum_visible_fraction']} "
            f"reasons={row['reasons']}"
        )

    print(
        "PASS: all exported candidate-without-sampled-surface rows were "
        "validated with candidate-camera and aggregate-camera scopes separated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
