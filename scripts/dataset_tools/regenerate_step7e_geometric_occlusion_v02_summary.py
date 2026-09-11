#!/usr/bin/env python3
"""Regenerate the Step 7E v02 summary from the existing JSONL product.

The script reads the completed v02 JSONL, validates unique Anchor/Actor keys and
embedded projection-context consistency, then atomically replaces only the v02
summary JSON. It does not recompute geometry or rewrite the JSONL product.
"""

from __future__ import annotations

import json
import os
from collections import Counter

from actor_export_anchor_snapshot_coverage_v01 import (
    summarize_actor_export_anchor_snapshot_coverage,
)
from actor_geometric_occlusion_export_v02 import SCHEMA_VERSION
from clip_reader import DrivingClipReader
from export_step7e_geometric_occlusion_evidence_v01 import read_keyframes
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from step7e_v02_summary_anchor_coverage_v01 import (
    attach_anchor_snapshot_coverage_to_summary,
)


KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
INPUT = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
SUMMARY = (
    ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.summary.json"
)


def main() -> int:
    identities = set()
    anchors = set()
    status_counts = Counter()
    class_counts = Counter()
    context_status_counts = Counter()
    context_camera_counts = Counter()
    actors_with_context_by_status = Counter()
    context_parent_status_counts = Counter()
    context_count_histogram = Counter()
    actor_to_actor_count = 0
    static_occlusion_count = 0
    row_count = 0
    context_count = 0

    with INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unexpected schema at line {line_number}: "
                    f"{row.get('schema_version')}"
                )

            identity = (str(row["anchor_id"]), str(row["track_id"]))
            if identity in identities:
                raise RuntimeError(
                    f"Duplicate Anchor/Actor key at line {line_number}: "
                    f"{identity}"
                )
            identities.add(identity)
            anchors.add(identity[0])
            row_count += 1

            status = str(row["evidence_status"])
            status_counts[status] += 1
            class_counts[str(row["label_class"])] += 1
            actor_to_actor_count += int(
                bool(row["actor_to_actor_occlusion_evaluated"])
            )
            static_occlusion_count += int(
                bool(row["static_occlusion_evaluated"])
            )

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
                    f"{identity}"
                )

            context_count_histogram[len(contexts)] += 1
            if contexts:
                actors_with_context_by_status[status] += 1
            for context in contexts:
                if str(context["track_id"]) != identity[1]:
                    raise RuntimeError(
                        f"Embedded context track_id mismatch: {identity}"
                    )
                context_count += 1
                context_parent_status_counts[status] += 1
                context_camera_counts[str(context["camera_name"])] += 1
                context_status_counts[str(context["evidence_status"])] += 1

    keyframes = read_keyframes(KEYFRAMES)
    coverage = summarize_actor_export_anchor_snapshot_coverage(
        keyframes=keyframes,
        exported_actor_identities=tuple(sorted(identities)),
        reader_factory=lambda clip_id: DrivingClipReader(
            ALPASIM_DATA_ROOT / clip_id
        ),
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Threshold-free Step 7E geometric and Actor-to-Actor occlusion "
            "evidence with missing-surface projection context. No final "
            "visibility labels."
        ),
        "row_count": row_count,
        "anchor_count": len(anchors),
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "rows_by_actor_class": dict(sorted(class_counts.items())),
        "missing_surface_projection_context_count": context_count,
        "missing_surface_projection_context_status_counts": dict(
            sorted(context_status_counts.items())
        ),
        "missing_surface_projection_context_camera_counts": dict(
            sorted(context_camera_counts.items())
        ),
        "actors_with_missing_surface_projection_context_by_evidence_status": dict(
            sorted(actors_with_context_by_status.items())
        ),
        "missing_surface_projection_context_parent_evidence_status_counts": dict(
            sorted(context_parent_status_counts.items())
        ),
        "missing_surface_projection_context_count_per_actor_histogram": {
            str(count): actor_count
            for count, actor_count in sorted(context_count_histogram.items())
        },
        "actor_to_actor_occlusion_evaluated_count": actor_to_actor_count,
        "static_occlusion_evaluated_count": static_occlusion_count,
        "duplicate_key_count": 0,
    }

    summary = attach_anchor_snapshot_coverage_to_summary(
        summary=summary,
        coverage=coverage,
    )

    temporary = SUMMARY.with_suffix(SUMMARY.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, SUMMARY)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print("Rows:", row_count)
    print("Anchors with Actor rows:", len(anchors))
    print("Source Keyframes:", coverage.coverage.source_keyframe_count)
    print(
        "Anchors without Actor rows:",
        coverage.coverage.anchor_without_actor_rows_count,
    )
    print(
        "Rowless Anchor reason counts:",
        dict(coverage.rowless_anchor_reason_counts),
    )
    print("Evidence status counts:", dict(sorted(status_counts.items())))
    print("Missing-surface projection contexts:", context_count)
    print(
        "Actors with contexts by evidence status:",
        dict(sorted(actors_with_context_by_status.items())),
    )
    print(
        "Context parent evidence status counts:",
        dict(sorted(context_parent_status_counts.items())),
    )
    print(
        "Context camera counts:",
        dict(sorted(context_camera_counts.items())),
    )
    print(
        "Context count per Actor histogram:",
        dict(sorted(context_count_histogram.items())),
    )
    print("Summary:", SUMMARY)
    print(
        "PASS: Step 7E v02 summary regenerated atomically from the existing "
        "JSONL without recomputing geometry."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
