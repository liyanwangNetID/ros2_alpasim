#!/usr/bin/env python3
"""Validate source-Keyframe coverage of the Step 7E v02 Actor JSONL.

The script compares all source Keyframes with Anchor/Actor identities in the
completed v02 JSONL. It lists Anchors that produced no Actor rows, but does not
infer why they have no rows and does not recompute geometry.
"""

from __future__ import annotations

import json

from actor_export_anchor_coverage_summary_v01 import (
    summarize_actor_export_anchor_coverage,
)
from export_step7e_geometric_occlusion_evidence_v01 import read_keyframes
from project_paths import ANNOTATION_ROOT


KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
INPUT = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
EXPECTED_SOURCE_KEYFRAME_COUNT = 3500
EXPECTED_ANCHOR_WITH_ROWS_COUNT = 3474
EXPECTED_ANCHOR_WITHOUT_ROWS_COUNT = 26


def read_exported_identities() -> tuple[tuple[str, str], ...]:
    identities = []
    with INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("anchor_id", "track_id"):
                if field not in row:
                    raise ValueError(
                        f"{INPUT}:{line_number}: missing {field}"
                    )
            identities.append(
                (str(row["anchor_id"]), str(row["track_id"]))
            )
    return tuple(identities)


def main() -> int:
    keyframes = read_keyframes(KEYFRAMES)
    identities = read_exported_identities()
    coverage = summarize_actor_export_anchor_coverage(
        keyframes=keyframes,
        exported_actor_identities=identities,
    )

    expected = (
        EXPECTED_SOURCE_KEYFRAME_COUNT,
        EXPECTED_ANCHOR_WITH_ROWS_COUNT,
        EXPECTED_ANCHOR_WITHOUT_ROWS_COUNT,
    )
    actual = (
        coverage.source_keyframe_count,
        coverage.anchor_with_actor_rows_count,
        coverage.anchor_without_actor_rows_count,
    )
    if actual != expected:
        raise RuntimeError(
            f"Anchor coverage baseline changed: actual={actual}, "
            f"expected={expected}"
        )

    print("Step 7E v02 Anchor coverage")
    print("source_keyframe_count:", coverage.source_keyframe_count)
    print(
        "anchor_with_actor_rows_count:",
        coverage.anchor_with_actor_rows_count,
    )
    print(
        "anchor_without_actor_rows_count:",
        coverage.anchor_without_actor_rows_count,
    )
    print("anchor_without_actor_rows:")
    for anchor_id in coverage.anchor_without_actor_rows:
        print(" ", anchor_id)
    print("keyframes:", KEYFRAMES)
    print("input:", INPUT)
    print(
        "PASS: source-Keyframe and exported Actor-row Anchor coverage "
        "closes without inferring the reason for rowless Anchors."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
