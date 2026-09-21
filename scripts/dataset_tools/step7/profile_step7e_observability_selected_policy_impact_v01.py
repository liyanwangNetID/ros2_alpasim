#!/usr/bin/env python3
"""Profile affected Actors for the narrowed Step 7E policy candidates."""
from __future__ import annotations

import json
from pathlib import Path

from project_paths import ANNOTATION_ROOT
from step7.actor_observability_selected_policy_impact_v01 import (
    analyze_selected_policy_changes,
)

INPUT = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
OUTPUT = Path("/tmp/step7e_observability_selected_policy_impact_v01.json")


def read_rows() -> tuple[dict[str, object], ...]:
    rows = []
    with INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "step7e-geometric-occlusion-evidence-v02":
                raise RuntimeError(f"Unexpected schema at line {line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No evidence rows found in {INPUT}")
    return tuple(rows)


def main() -> int:
    report = analyze_selected_policy_changes(
        evidence_rows=read_rows(),
        expected_winning_affected_count=361,
        expected_fraction_affected_count=515,
    )
    report["source_path"] = str(INPUT)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Step 7E selected-policy affected-row analysis")
    print("source:", INPUT)
    print("evidence actors:", report["evidence_actor_count"])
    print("affected actors in union:", report["affected_actor_count"])
    print("affected Clips:", report["affected_clip_count"])
    print("change categories:", report["change_category_counts"])
    print("affected by winning-cell candidate:", report["affected_by_winning_cell_candidate_count"])
    print("affected by visible-fraction candidate:", report["affected_by_visible_fraction_candidate_count"])
    print("output:", OUTPUT)
    print("PASS: narrowed policies analyzed without modifying evidence or labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
