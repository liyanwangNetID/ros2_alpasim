#!/usr/bin/env python3
"""Profile Step 7E shadow-policy effects by Actor class.

Reads the completed v02 Actor evidence, applies the established nine-policy
shadow grid, and reports class-level visible-to-not-visible changes relative to
the permissive baseline. Output is written to /tmp. No production policy is
selected and no final visibility labels are written.
"""

from __future__ import annotations

import json
from pathlib import Path

from actor_observability_shadow_class_summary_v01 import (
    summarize_observability_shadow_grid_by_actor_class,
)
from profile_step7e_observability_shadow_grid_v01 import (
    BASELINE_POLICY_NAME,
    INPUT,
    build_policies,
    read_evidence_rows,
)


OUTPUT = Path(
    "/tmp/step7e_observability_shadow_policy_grid_by_actor_class_v01.json"
)


def main() -> int:
    rows = read_evidence_rows()
    policies = build_policies()
    summaries = summarize_observability_shadow_grid_by_actor_class(
        evidence_rows=rows,
        policies=policies,
        baseline_policy_name=BASELINE_POLICY_NAME,
    )

    report = {
        "schema_version": (
            "step7e-observability-shadow-policy-grid-by-actor-class-v01"
        ),
        "description": (
            "Class-level offline comparison of dynamic-occlusion shadow "
            "policies. Static-scene occlusion is unevaluated and no final "
            "visibility labels are written."
        ),
        "source_path": str(INPUT),
        "baseline_policy_name": BASELINE_POLICY_NAME,
        "actor_count": len(rows),
        "policy_count": len(policies),
        "actor_class_count": len(
            {str(row["label_class"]) for row in rows}
        ),
        "policy_class_summaries": [item.to_dict() for item in summaries],
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Step 7E observability shadow policy grid by Actor class")
    print("source:", INPUT)
    print("actors:", report["actor_count"])
    print("policies:", report["policy_count"])
    print("actor_classes:", report["actor_class_count"])
    print("baseline_policy:", BASELINE_POLICY_NAME)

    for policy in policies:
        values = tuple(
            item for item in summaries if item.policy_name == policy.policy_name
        )
        changed = tuple(
            item for item in values
            if item.status_changed_from_baseline_count > 0
        )
        print(
            f"policy={policy.policy_name} "
            f"winning>={policy.minimum_winning_cell_count} "
            f"fraction>={policy.minimum_visible_fraction:g} "
            f"changed_classes={len(changed)}"
        )
        for item in sorted(
            changed,
            key=lambda value: (
                -value.visible_to_not_visible_from_baseline_count,
                value.actor_class,
            ),
        ):
            rate = (
                item.visible_to_not_visible_from_baseline_count
                / item.actor_count
                if item.actor_count
                else 0.0
            )
            print(
                f"  class={item.actor_class} "
                f"actors={item.actor_count} "
                f"visible_to_not_visible="
                f"{item.visible_to_not_visible_from_baseline_count} "
                f"class_row_rate={rate:.6f} "
                f"statuses={dict(item.shadow_status_counts)}"
            )

    print("output:", OUTPUT)
    print(
        "PASS: class-level shadow-policy effects were profiled without "
        "selecting a production policy or writing final labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
