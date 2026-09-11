#!/usr/bin/env python3
"""Profile Step 7E shadow-policy failure attribution on v02 evidence.

Reads the completed v02 Actor evidence and applies the established nine-policy
shadow grid. For baseline-visible Actors lost by each policy, it separates
winning-cell-only, visible-fraction-only, and joint failures, both overall and
by Actor class. Output is written to /tmp. No production policy is selected and
no final visibility labels are written.
"""

from __future__ import annotations

import json
from pathlib import Path

from actor_observability_shadow_failure_attribution_v01 import (
    summarize_shadow_failure_attribution,
)
from profile_step7e_observability_shadow_grid_v01 import (
    BASELINE_POLICY_NAME,
    INPUT,
    build_policies,
    read_evidence_rows,
)


OUTPUT = Path(
    "/tmp/step7e_observability_shadow_failure_attribution_v01.json"
)


def main() -> int:
    rows = read_evidence_rows()
    policies = build_policies()
    attributions = summarize_shadow_failure_attribution(
        evidence_rows=rows,
        policies=policies,
        baseline_policy_name=BASELINE_POLICY_NAME,
    )

    report = {
        "schema_version": (
            "step7e-observability-shadow-failure-attribution-v01"
        ),
        "description": (
            "Offline attribution of baseline-visible Actor losses to "
            "winning-cell and visible-fraction requirements. Static-scene "
            "occlusion is unevaluated and no final labels are written."
        ),
        "source_path": str(INPUT),
        "baseline_policy_name": BASELINE_POLICY_NAME,
        "actor_count": len(rows),
        "policy_count": len(policies),
        "actor_class_count": len({str(row["label_class"]) for row in rows}),
        "policy_class_attributions": [
            item.to_dict() for item in attributions
        ],
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Step 7E observability shadow failure attribution")
    print("source:", INPUT)
    print("actors:", report["actor_count"])
    print("policies:", report["policy_count"])
    print("actor_classes:", report["actor_class_count"])
    print("baseline_policy:", BASELINE_POLICY_NAME)

    for policy in policies:
        values = tuple(
            item
            for item in attributions
            if item.policy_name == policy.policy_name
        )
        only_winning = sum(
            item.only_winning_cell_failed_count for item in values
        )
        only_fraction = sum(
            item.only_visible_fraction_failed_count for item in values
        )
        both = sum(item.both_requirements_failed_count for item in values)
        lost = sum(
            item.total_visible_to_not_visible_count for item in values
        )
        if only_winning + only_fraction + both != lost:
            raise RuntimeError("overall failure attribution does not close")

        print(
            f"policy={policy.policy_name} "
            f"winning>={policy.minimum_winning_cell_count} "
            f"fraction>={policy.minimum_visible_fraction:g} "
            f"only_winning={only_winning} "
            f"only_fraction={only_fraction} "
            f"both={both} "
            f"total_lost={lost}"
        )
        for item in sorted(
            (value for value in values if value.total_visible_to_not_visible_count),
            key=lambda value: (
                -value.total_visible_to_not_visible_count,
                value.actor_class,
            ),
        ):
            print(
                f"  class={item.actor_class} "
                f"baseline_visible={item.baseline_visible_count} "
                f"only_winning={item.only_winning_cell_failed_count} "
                f"only_fraction={item.only_visible_fraction_failed_count} "
                f"both={item.both_requirements_failed_count} "
                f"total_lost={item.total_visible_to_not_visible_count}"
            )

    print("output:", OUTPUT)
    print(
        "PASS: shadow-policy losses were attributed to policy requirements "
        "without selecting a production policy or writing final labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
