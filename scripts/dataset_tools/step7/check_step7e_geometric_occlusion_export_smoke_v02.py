#!/usr/bin/env python3
"""Smoke-test the formal Step 7E v02 exporter on four real Anchors.

The check creates a temporary Keyframe JSONL for the four established Anchors,
runs the production v02 exporter with /tmp outputs, compares all inherited v01
fields against the existing smoke baseline, and validates that no projection
context is emitted for these four Anchors. The annotation root is not modified.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from step7.check_step7e_geometric_occlusion_export_smoke_v01 import (
    ANCHOR_IDS,
    read_selected_keyframes,
)
from step7.export_step7e_geometric_occlusion_evidence_v02 import export


SMOKE_KEYFRAMES = Path(
    "/tmp/step7e_geometric_occlusion_v02_smoke_keyframes.jsonl"
)
OUTPUT_V01 = Path(
    "/tmp/step7e_geometric_occlusion_smoke_evidence_v01.jsonl"
)
OUTPUT_V02 = Path(
    "/tmp/step7e_geometric_occlusion_smoke_evidence_v02.jsonl"
)
SUMMARY_V02 = Path(
    "/tmp/step7e_geometric_occlusion_smoke_evidence_v02.summary.json"
)
EXPECTED_ROW_COUNT = 441
EXPECTED_STATUS_COUNTS = {
    "combined_evidence_available": 265,
    "no_geometric_candidate": 176,
}
V02_ONLY_FIELDS = {
    "candidate_without_sampled_surface_projection_evidence",
}


def write_keyframes(rows) -> None:
    SMOKE_KEYFRAMES.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def load_rows(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["anchor_id"]), str(row["track_id"]))
            if key in rows:
                raise RuntimeError(
                    f"Duplicate row at {path}:{line_number}: {key}"
                )
            rows[key] = row
    return rows


def normalized_v01_row(row):
    value = dict(row)
    value["schema_version"] = "step7e-geometric-occlusion-evidence-v01"
    for field in V02_ONLY_FIELDS:
        value.pop(field, None)
    return value


def validate() -> None:
    if not OUTPUT_V01.exists():
        raise RuntimeError(
            "v01 smoke output is unavailable; run "
            "check_step7e_geometric_occlusion_export_smoke_v01.py first"
        )

    v01 = load_rows(OUTPUT_V01)
    v02 = load_rows(OUTPUT_V02)
    if len(v02) != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"v02 row baseline changed: actual={len(v02)}, "
            f"expected={EXPECTED_ROW_COUNT}"
        )
    if set(v01) != set(v02):
        raise RuntimeError("v01 and v02 smoke Actor identities differ")

    status_counts = Counter()
    projection_context_count = 0
    inherited_mismatches = []
    for key in sorted(v02):
        row = v02[key]
        status_counts[str(row["evidence_status"])] += 1
        contexts = row[
            "candidate_without_sampled_surface_projection_evidence"
        ]
        projection_context_count += len(contexts)
        if normalized_v01_row(row) != v01[key]:
            inherited_mismatches.append(key)

    if dict(status_counts) != EXPECTED_STATUS_COUNTS:
        raise RuntimeError(
            f"v02 evidence-status baseline changed: {dict(status_counts)}"
        )
    if inherited_mismatches:
        raise RuntimeError(
            "v02 inherited fields differ from v01 for Actor keys: "
            f"{inherited_mismatches[:20]}"
        )
    if projection_context_count != 0:
        raise RuntimeError(
            "Four-Anchor baseline unexpectedly contains missing-surface "
            f"projection context: {projection_context_count}"
        )

    summary = json.loads(SUMMARY_V02.read_text(encoding="utf-8"))
    if int(summary["row_count"]) != EXPECTED_ROW_COUNT:
        raise RuntimeError("v02 summary row_count differs from JSONL")
    if int(summary["anchor_count"]) != len(ANCHOR_IDS):
        raise RuntimeError("v02 summary anchor_count differs from input")
    if summary["evidence_status_counts"] != EXPECTED_STATUS_COUNTS:
        raise RuntimeError("v02 summary evidence-status counts differ")
    if int(summary["missing_surface_projection_context_count"]) != 0:
        raise RuntimeError("v02 summary projection-context count is not zero")
    if summary["missing_surface_projection_context_status_counts"] != {}:
        raise RuntimeError("v02 summary has unexpected context statuses")
    if int(summary["actor_to_actor_occlusion_evaluated_count"]) != len(v02):
        raise RuntimeError("v02 summary reports incomplete Actor occlusion")
    if int(summary["static_occlusion_evaluated_count"]) != 0:
        raise RuntimeError("v02 summary unexpectedly reports static occlusion")
    if int(summary["duplicate_key_count"]) != 0:
        raise RuntimeError("v02 summary reports duplicate keys")

    print("Step 7E v02 smoke export validation")
    print("anchors:", len(ANCHOR_IDS))
    print("rows:", len(v02))
    print("evidence_status_counts:", dict(sorted(status_counts.items())))
    print("inherited_field_mismatch_count:", len(inherited_mismatches))
    print("missing_surface_projection_context_count:", projection_context_count)
    print("actor_to_actor_occlusion_evaluated_count:", len(v02))
    print("static_occlusion_evaluated_count: 0")
    print("duplicate_key_count: 0")
    print("output:", OUTPUT_V02)
    print("summary:", SUMMARY_V02)


def main() -> int:
    rows = read_selected_keyframes()
    write_keyframes(rows)
    export(
        keyframe_path=SMOKE_KEYFRAMES,
        output_path=OUTPUT_V02,
        summary_path=SUMMARY_V02,
    )
    validate()
    print(
        "PASS: Step 7E v02 reproduced the four-Anchor v01 evidence baseline "
        "and added an empty projection-context field without modifying the "
        "annotation root."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
