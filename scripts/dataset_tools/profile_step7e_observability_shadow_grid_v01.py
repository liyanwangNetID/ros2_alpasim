#!/usr/bin/env python3
"""Profile Step 7E v02 evidence across a shadow observability policy grid.

Reads the completed v02 Actor evidence product and compares nine explicitly
named winning-cell and visible-fraction policies. The report is written to /tmp.
No production policy is selected, no final labels are written, and the formal
v02 evidence product is not modified.
"""

from __future__ import annotations

import json
from pathlib import Path

from actor_observability_shadow_grid_summary_v01 import (
    summarize_observability_shadow_policy_grid,
)
from actor_observability_shadow_policy_v01 import ObservabilityShadowPolicy
from project_paths import ANNOTATION_ROOT


INPUT = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
OUTPUT = Path("/tmp/step7e_observability_shadow_policy_grid_v01.json")
BASELINE_POLICY_NAME = "winning_1_fraction_0"
WINNING_CELL_VALUES = (1, 2, 4)
VISIBLE_FRACTION_VALUES = (0.0, 0.01, 0.05)


def policy_name(winning_cells: int, visible_fraction: float) -> str:
    fraction_text = format(visible_fraction, "g").replace(".", "p")
    return f"winning_{winning_cells}_fraction_{fraction_text}"


def build_policies() -> tuple[ObservabilityShadowPolicy, ...]:
    return tuple(
        ObservabilityShadowPolicy(
            policy_name=policy_name(winning_cells, visible_fraction),
            minimum_winning_cell_count=winning_cells,
            minimum_visible_fraction=visible_fraction,
        )
        for winning_cells in WINNING_CELL_VALUES
        for visible_fraction in VISIBLE_FRACTION_VALUES
    )


def read_evidence_rows() -> tuple[dict[str, object], ...]:
    rows = []
    with INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != (
                "step7e-geometric-occlusion-evidence-v02"
            ):
                raise RuntimeError(
                    f"Unexpected schema at line {line_number}: "
                    f"{row.get('schema_version')}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No evidence rows found in {INPUT}")
    return tuple(rows)


def main() -> int:
    rows = read_evidence_rows()
    policies = build_policies()
    summary = summarize_observability_shadow_policy_grid(
        evidence_rows=rows,
        policies=policies,
        baseline_policy_name=BASELINE_POLICY_NAME,
    )

    report = {
        "schema_version": "step7e-observability-shadow-policy-grid-v01",
        "description": (
            "Offline comparison of dynamic-occlusion shadow policies. "
            "Static-scene occlusion is unevaluated and no production policy "
            "or final visibility label is selected."
        ),
        "source_path": str(INPUT),
        "winning_cell_values": list(WINNING_CELL_VALUES),
        "visible_fraction_values": list(VISIBLE_FRACTION_VALUES),
        **summary.to_dict(),
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Step 7E observability shadow policy grid")
    print("source:", INPUT)
    print("actors:", summary.actor_count)
    print("policies:", summary.policy_count)
    print("baseline_policy:", summary.baseline_policy_name)
    for item in summary.policy_summaries:
        print(
            f"policy={item.policy_name} "
            f"winning>={item.minimum_winning_cell_count} "
            f"fraction>={item.minimum_visible_fraction:g} "
            f"statuses={dict(item.shadow_status_counts)} "
            f"changed={item.status_changed_from_baseline_count} "
            f"visible_to_not_visible="
            f"{item.visible_to_not_visible_from_baseline_count} "
            f"not_visible_to_visible="
            f"{item.not_visible_to_visible_from_baseline_count}"
        )
    print("output:", OUTPUT)
    print(
        "PASS: Step 7E v02 evidence was profiled across the shadow policy "
        "grid without selecting a production policy or writing final labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
