#!/usr/bin/env python3
"""Validate and explain Step 7E v02 missing-surface projection contexts.

The script reads the completed v02 JSONL and summary, separates Actor-level
status counts from camera-level projection-context counts, and validates that
every embedded context corresponds exactly to one missing-surface candidate
camera. No geometry is recomputed.
"""

from __future__ import annotations

import json
from collections import Counter

from project_paths import ANNOTATION_ROOT


INPUT = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
SUMMARY = (
    ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.summary.json"
)
EXPECTED_ROW_COUNT = 151908
EXPECTED_CONTEXT_COUNT = 209


def main() -> int:
    row_count = 0
    identities = set()
    actor_status_counts = Counter()
    context_parent_status_counts = Counter()
    context_camera_counts = Counter()
    context_status_counts = Counter()
    actors_with_context_by_status = Counter()
    context_count_histogram = Counter()
    context_count = 0

    with INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1
            identity = (str(row["anchor_id"]), str(row["track_id"]))
            if identity in identities:
                raise RuntimeError(
                    f"Duplicate identity at line {line_number}: {identity}"
                )
            identities.add(identity)

            actor_status = str(row["evidence_status"])
            actor_status_counts[actor_status] += 1
            contexts = tuple(
                row[
                    "candidate_without_sampled_surface_projection_evidence"
                ]
            )
            missing_cameras = tuple(
                row[
                    "geometric_candidate_without_sampled_surface_camera_names"
                ]
            )
            context_cameras = tuple(
                str(item["camera_name"]) for item in contexts
            )
            if context_cameras != missing_cameras:
                raise RuntimeError(
                    f"Context cameras differ from missing-surface cameras: "
                    f"{identity} context={context_cameras} "
                    f"missing={missing_cameras}"
                )

            per_actor_count = len(contexts)
            context_count_histogram[per_actor_count] += 1
            if contexts:
                actors_with_context_by_status[actor_status] += 1
            for item in contexts:
                if str(item["track_id"]) != identity[1]:
                    raise RuntimeError(
                        f"Embedded context track_id mismatch: {identity}"
                    )
                context_count += 1
                context_parent_status_counts[actor_status] += 1
                context_camera_counts[str(item["camera_name"])] += 1
                context_status_counts[str(item["evidence_status"])] += 1

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if row_count != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"Row baseline changed: actual={row_count}, "
            f"expected={EXPECTED_ROW_COUNT}"
        )
    if context_count != EXPECTED_CONTEXT_COUNT:
        raise RuntimeError(
            f"Context baseline changed: actual={context_count}, "
            f"expected={EXPECTED_CONTEXT_COUNT}"
        )
    if int(summary["row_count"]) != row_count:
        raise RuntimeError("Summary row_count differs from JSONL")
    if int(summary["missing_surface_projection_context_count"]) != context_count:
        raise RuntimeError("Summary context count differs from JSONL")
    if summary["evidence_status_counts"] != dict(actor_status_counts):
        raise RuntimeError("Summary Actor status counts differ from JSONL")
    if summary[
        "missing_surface_projection_context_status_counts"
    ] != dict(context_status_counts):
        raise RuntimeError("Summary context status counts differ from JSONL")

    print("Step 7E v02 projection-context validation")
    print("rows:", row_count)
    print("actor_status_counts:", dict(sorted(actor_status_counts.items())))
    print("projection_context_count:", context_count)
    print(
        "actors_with_context_by_status:",
        dict(sorted(actors_with_context_by_status.items())),
    )
    print(
        "projection_context_parent_status_counts:",
        dict(sorted(context_parent_status_counts.items())),
    )
    print(
        "projection_context_camera_counts:",
        dict(sorted(context_camera_counts.items())),
    )
    print(
        "projection_context_status_counts:",
        dict(sorted(context_status_counts.items())),
    )
    print(
        "projection_context_count_per_actor_histogram:",
        dict(sorted(context_count_histogram.items())),
    )
    print("input:", INPUT)
    print("summary:", SUMMARY)
    print(
        "PASS: v02 Actor-level statuses and camera-level projection contexts "
        "close independently and match the saved summary."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
