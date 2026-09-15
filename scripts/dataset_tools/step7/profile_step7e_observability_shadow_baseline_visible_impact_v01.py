#!/usr/bin/env python3
"""Profile baseline-visible retention across the Step 7E shadow policy grid.

Reads the completed v02 evidence and reports, for every policy and Actor class,
how many baseline-visible Actors remain visible or become not visible. Rates use
the baseline-visible class population as denominator. Output is written to /tmp;
no production policy is selected and no final labels are written.
"""

from __future__ import annotations

import json
from pathlib import Path

from step7.actor_observability_shadow_baseline_visible_impact_v01 import (
    summarize_shadow_baseline_visible_class_impact,
)
from step7.profile_step7e_observability_shadow_grid_v01 import (
    BASELINE_POLICY_NAME,
    INPUT,
    build_policies,
    read_evidence_rows,
)


OUTPUT = Path(
    "/tmp/step7e_observability_shadow_baseline_visible_impact_v01.json"
)


def main() -> int:
    rows = read_evidence_rows()
    policies = build_policies()
    impacts = summarize_shadow_baseline_visible_class_impact(
        evidence_rows=rows,
        policies=policies,
        baseline_policy_name=BASELINE_POLICY_NAME,
    )

    report = {
        "schema_version": (
            "step7e-observability-shadow-baseline-visible-impact-v01"
        ),
        "description": (
            "Class-level retention of baseline-visible Actors under dynamic-"
            "occlusion shadow policies. Static-scene occlusion is unevaluated "
            "and no final visibility labels are written."
        ),
        "source_path": str(INPUT),
        "baseline_policy_name": BASELINE_POLICY_NAME,
        "actor_count": len(rows),
        "policy_count": len(policies),
        "actor_class_count": len({str(row["label_class"]) for row in rows}),
        "policy_class_impacts": [item.to_dict() for item in impacts],
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Step 7E baseline-visible retention by Actor class")
    print("source:", INPUT)
    print("actors:", report["actor_count"])
    print("policies:", report["policy_count"])
    print("actor_classes:", report["actor_class_count"])
    print("baseline_policy:", BASELINE_POLICY_NAME)

    for policy in policies:
        values = tuple(
            item for item in impacts if item.policy_name == policy.policy_name
        )
        changed = tuple(
            item for item in values if item.visible_to_not_visible_count > 0
        )
        total_baseline_visible = sum(
            item.baseline_visible_count for item in values
        )
        total_lost = sum(
            item.visible_to_not_visible_count for item in values
        )
        total_loss_rate = (
            total_lost / total_baseline_visible
            if total_baseline_visible
            else None
        )
        print(
            f"policy={policy.policy_name} "
            f"winning>={policy.minimum_winning_cell_count} "
            f"fraction>={policy.minimum_visible_fraction:g} "
            f"baseline_visible={total_baseline_visible} "
            f"visible_to_not_visible={total_lost} "
            f"loss_rate={total_loss_rate if total_loss_rate is not None else 'null'} "
            f"changed_classes={len(changed)}"
        )
        for item in sorted(
            changed,
            key=lambda value: (
                -(
                    value.visible_to_not_visible_rate
                    if value.visible_to_not_visible_rate is not None
                    else -1.0
                ),
                -value.visible_to_not_visible_count,
                value.actor_class,
            ),
        ):
            print(
                f"  class={item.actor_class} "
                f"baseline_visible={item.baseline_visible_count} "
                f"retained={item.retained_visible_count} "
                f"lost={item.visible_to_not_visible_count} "
                f"retained_rate={item.retained_visible_rate:.6f} "
                f"loss_rate={item.visible_to_not_visible_rate:.6f}"
            )

    print("output:", OUTPUT)
    print(
        "PASS: baseline-visible retention was profiled by Actor class without "
        "selecting a production policy or writing final labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
