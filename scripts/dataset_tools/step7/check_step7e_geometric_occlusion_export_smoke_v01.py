#!/usr/bin/env python3
"""Smoke-test the formal Step 7E exporter on four validated real Anchors.

Creates a temporary Keyframe JSONL containing the four established real
Anchors, invokes the production exporter with /tmp outputs, and validates the
exported row identities and aggregate evidence-status baseline. The annotation
root is not modified.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from step7.export_step7e_geometric_occlusion_evidence_v01 import export
from project_paths import ANNOTATION_ROOT


ANCHOR_IDS = (
    "test_clip_001_9306612661000",
    "test_clip_063_18787721418000",
    "test_clip_564_95233721166000",
    "test_clip_316_1404486225261000",
)
SOURCE_KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
SMOKE_KEYFRAMES = Path("/tmp/step7e_geometric_occlusion_smoke_keyframes.jsonl")
OUTPUT = Path("/tmp/step7e_geometric_occlusion_smoke_evidence_v01.jsonl")
SUMMARY = Path(
    "/tmp/step7e_geometric_occlusion_smoke_evidence_v01.summary.json"
)
EXPECTED_ROW_COUNT = 441
EXPECTED_STATUS_COUNTS = {
    "combined_evidence_available": 265,
    "no_geometric_candidate": 176,
}


def read_selected_keyframes() -> tuple[dict[str, object], ...]:
    selected = {}
    with SOURCE_KEYFRAMES.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            anchor_id = str(row.get("anchor_id", ""))
            if anchor_id not in ANCHOR_IDS:
                continue
            if anchor_id in selected:
                raise RuntimeError(
                    f"Duplicate selected Anchor at line {line_number}: "
                    f"{anchor_id}"
                )
            selected[anchor_id] = row

    missing = sorted(set(ANCHOR_IDS) - set(selected))
    if missing:
        raise RuntimeError(f"Smoke-test Anchors missing from Keyframes: {missing}")
    return tuple(selected[anchor_id] for anchor_id in sorted(selected))


def write_smoke_keyframes(rows: tuple[dict[str, object], ...]) -> None:
    SMOKE_KEYFRAMES.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def validate_outputs() -> None:
    rows = []
    identities = set()
    status_counts = Counter()

    with OUTPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (str(row["anchor_id"]), str(row["track_id"]))
            if identity in identities:
                raise RuntimeError(
                    f"Duplicate exported identity at line {line_number}: "
                    f"{identity}"
                )
            identities.add(identity)
            rows.append(row)
            status_counts[str(row["evidence_status"])] += 1

    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"Exported row baseline changed: actual={len(rows)}, "
            f"expected={EXPECTED_ROW_COUNT}"
        )
    if dict(status_counts) != EXPECTED_STATUS_COUNTS:
        raise RuntimeError(
            f"Evidence-status baseline changed: actual={dict(status_counts)}, "
            f"expected={EXPECTED_STATUS_COUNTS}"
        )
    if {identity[0] for identity in identities} != set(ANCHOR_IDS):
        raise RuntimeError("Exported Anchor set differs from smoke-test input")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if int(summary["row_count"]) != EXPECTED_ROW_COUNT:
        raise RuntimeError("Summary row_count differs from JSONL row count")
    if int(summary["anchor_count"]) != len(ANCHOR_IDS):
        raise RuntimeError("Summary anchor_count differs from smoke-test input")
    if summary["evidence_status_counts"] != EXPECTED_STATUS_COUNTS:
        raise RuntimeError("Summary evidence-status counts differ from JSONL")
    if int(summary["actor_to_actor_occlusion_evaluated_count"]) != len(rows):
        raise RuntimeError("Summary reports incomplete Actor-to-Actor occlusion")
    if int(summary["static_occlusion_evaluated_count"]) != 0:
        raise RuntimeError("Static occlusion was unexpectedly exported")
    if int(summary["duplicate_key_count"]) != 0:
        raise RuntimeError("Summary reports duplicate export keys")

    print("Smoke export validation")
    print("anchors:", len(ANCHOR_IDS))
    print("rows:", len(rows))
    print("evidence_status_counts:", dict(sorted(status_counts.items())))
    print("actor_to_actor_occlusion_evaluated_count:", len(rows))
    print("static_occlusion_evaluated_count: 0")
    print("duplicate_key_count: 0")
    print("output:", OUTPUT)
    print("summary:", SUMMARY)


def main() -> int:
    rows = read_selected_keyframes()
    write_smoke_keyframes(rows)
    export(
        keyframe_path=SMOKE_KEYFRAMES,
        output_path=OUTPUT,
        summary_path=SUMMARY,
    )
    validate_outputs()
    print(
        "PASS: formal Step 7E exporter reproduced the four-Anchor real "
        "evidence baseline without modifying the annotation root."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
