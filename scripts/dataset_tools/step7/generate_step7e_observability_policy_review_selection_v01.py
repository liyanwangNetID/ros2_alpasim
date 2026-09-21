#!/usr/bin/env python3
"""Create the compact Step 7E observability policy review manifest."""
from __future__ import annotations

import json
from pathlib import Path

from step7.actor_observability_policy_review_selection_v01 import (
    select_policy_review_cases,
)

INPUT = Path("/tmp/step7e_observability_selected_policy_impact_v01.json")
OUTPUT = Path("/tmp/step7e_observability_policy_review_selection_v01.json")


def main() -> int:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    selection = select_policy_review_cases(
        impact_report=report,
        quota_per_category=12,
    )
    selection["source_path"] = str(INPUT)
    OUTPUT.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Step 7E observability policy review selection")
    print("source affected actors:", selection["source_affected_actor_count"])
    print("selected cases:", selection["selected_case_count"])
    print("selected Clips:", selection["selected_clip_count"])
    print("selected categories:", selection["selected_category_counts"])
    print("output:", OUTPUT)
    print("PASS: review manifest selected without modifying evidence or labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
