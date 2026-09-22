#!/usr/bin/env python3
"""Build the frozen Step 7E dynamic-occlusion policy report."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from project_paths import ANNOTATION_ROOT
from step7.observability import (
    MINIMUM_VISIBLE_FRACTION,
    MINIMUM_WINNING_CELL_COUNT,
    POLICY_NAME,
    evaluate_frozen_dynamic_occlusion_policy,
)

EVIDENCE_PATH = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
OUTPUT_PATH = ANNOTATION_ROOT / "step7e_dynamic_occlusion_policy_v01.json"

REVIEW_LABELS = {
    "KEEP": (13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 31, 34, 35, 36),
    "REJECT": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 25, 26, 27, 28, 29, 30, 32, 33),
    "UNCERTAIN": (),
}
REVIEW_CATEGORY_RESULTS = {
    "rejected_by_winning_cell_candidate_only": {
        "KEEP": 0, "REJECT": 12, "UNCERTAIN": 0,
    },
    "rejected_by_visible_fraction_candidate_only": {
        "KEEP": 11, "REJECT": 1, "UNCERTAIN": 0,
    },
    "rejected_by_both_candidates": {
        "KEEP": 4, "REJECT": 8, "UNCERTAIN": 0,
    },
}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return tuple(rows)


def main() -> int:
    evidence_rows = read_jsonl(EVIDENCE_PATH)
    decisions = evaluate_frozen_dynamic_occlusion_policy(
        evidence_rows=evidence_rows
    )
    if len(decisions) != len(evidence_rows):
        raise RuntimeError("frozen policy decision count does not close")
    status_counts = Counter(item.shadow_status for item in decisions)

    # The selected-policy impact count is part of the frozen reviewed contract.
    # Do not depend on a development-time /tmp profiling artifact.
    baseline_visible_actor_changes = 361
    if sum(len(values) for values in REVIEW_LABELS.values()) != 36:
        raise RuntimeError("manual review labels do not close to 36 cases")
    flattened = [value for values in REVIEW_LABELS.values() for value in values]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != list(range(1, 37)):
        raise RuntimeError("manual review indices must uniquely cover 1 through 36")

    report = {
        "schema_version": "step7e-dynamic-occlusion-policy-v01",
        "policy_name": POLICY_NAME,
        "status": "frozen",
        "scope": "dynamic_actor_to_actor_occlusion_only",
        "policy": {
            "minimum_winning_cell_count": MINIMUM_WINNING_CELL_COUNT,
            "minimum_visible_fraction": MINIMUM_VISIBLE_FRACTION,
        },
        "source_evidence_path": str(EVIDENCE_PATH),
        "source_evidence_schema": "step7e-geometric-occlusion-evidence-v02",
        "source_actor_count": len(evidence_rows),
        "decision_status_counts": dict(sorted(status_counts.items())),
        "baseline_visible_actor_changes": baseline_visible_actor_changes,
        "affected_clip_count_not_recomputed": True,
        "manual_review": {
            "case_count": 36,
            "labels_by_decision": {
                key: list(values) for key, values in REVIEW_LABELS.items()
            },
            "category_results": REVIEW_CATEGORY_RESULTS,
        },
        "decision_rationale": [
            "The winning-cell-only review sample contained 12 REJECT and 0 KEEP cases.",
            "The visible-fraction-only review sample contained 11 KEEP and 1 REJECT cases.",
            "The policy therefore uses a two-winning-cell floor and no visible-fraction floor.",
            "The policy is a reasonable v01 baseline rather than a claim of zero review disagreement.",
        ],
        "known_limitations": [
            "Four review cases rejected by both candidate thresholds were manually labeled KEEP.",
            "Static-scene occlusion is not evaluated by the v02 evidence and is not resolved here.",
            "This report freezes the dynamic policy only and does not write final Scene Facts.",
        ],
    }
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT_PATH)
    print("Step 7E dynamic-occlusion policy freeze")
    print("policy:", POLICY_NAME)
    print("minimum winning cells:", MINIMUM_WINNING_CELL_COUNT)
    print("minimum visible fraction:", MINIMUM_VISIBLE_FRACTION)
    print("source actors:", len(evidence_rows))
    print("decision status counts:", dict(sorted(status_counts.items())))
    print("baseline-visible changes:", baseline_visible_actor_changes)
    print("manual review cases:", 36)
    print("output:", OUTPUT_PATH)
    print("PASS: dynamic policy frozen without modifying evidence or final labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
