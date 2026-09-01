#!/usr/bin/env python3
"""Diagnose first observed Navigation branches before freezing v0.1 rules."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from project_paths import INTERMEDIATE_ROOT

PATH = (
    INTERMEDIATE_ROOT
    / "navigation_branch_context_v0.1.jsonl"
)


def distance_bucket(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "missing"
    distance = float(value)
    if distance <= 20.0:
        return "<=20m"
    if distance <= 40.0:
        return "20_to_40m"
    if distance <= 60.0:
        return "40_to_60m"
    return ">60m"


def main() -> int:
    records = []
    with PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    relation_by_reliability = Counter()
    relation_by_distance = Counter()
    reliability_reasons = Counter()
    reliable_relation_by_distance = Counter()
    reliable_relation_by_intersection_distance = Counter()
    first_intersection_evidence = Counter()
    unresolved_reasons = Counter()
    branch_counts_by_anchor = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)

    for record in records:
        context = record.get("branch_context", {})
        branch_counts_by_anchor[int(context.get("observed_branch_count", 0))] += 1

        intersection = context.get("first_intersection_evidence")
        if isinstance(intersection, dict):
            for evidence in intersection.get("evidence", []):
                first_intersection_evidence[str(evidence)] += 1

        branch = context.get("first_observed_branch")
        if not isinstance(branch, dict):
            continue

        relation = str(branch.get("route_relation_to_natural", "missing"))
        reliability = str(branch.get("reliability_status", "missing"))
        branch_bucket = distance_bucket(branch.get("route_distance_m"))

        relation_by_reliability[(relation, reliability)] += 1
        relation_by_distance[(relation, branch_bucket)] += 1

        for reason in branch.get("reliability_reasons", []):
            reliability_reasons[str(reason)] += 1

        if reliability == "reliable":
            reliable_relation_by_distance[(relation, branch_bucket)] += 1
            intersection_bucket = distance_bucket(
                intersection.get("route_distance_m")
                if isinstance(intersection, dict)
                else None
            )
            reliable_relation_by_intersection_distance[
                (relation, intersection_bucket)
            ] += 1

        if relation in {"actual_successor_not_candidate", "not_observed"}:
            unresolved_reasons[relation] += 1

        key = (relation, reliability)
        if len(examples[key]) < 5:
            examples[key].append(str(record["anchor_id"]))

    print("=" * 78)
    print("STEP 6 FIRST-BRANCH DIAGNOSTICS")
    print("=" * 78)
    print("Records:", len(records))

    print()
    print("Relation by reliability:")
    for key, count in sorted(relation_by_reliability.items()):
        print(" ", key[0], "+", key[1], ":", count)

    print()
    print("Reliable relation by branch distance:")
    for key, count in sorted(reliable_relation_by_distance.items()):
        print(" ", key[0], "+", key[1], ":", count)

    print()
    print("Reliable relation by first-intersection distance:")
    for key, count in sorted(reliable_relation_by_intersection_distance.items()):
        print(" ", key[0], "+", key[1], ":", count)

    print()
    print("All relation by branch distance:")
    for key, count in sorted(relation_by_distance.items()):
        print(" ", key[0], "+", key[1], ":", count)

    print()
    print("Reliability reasons:", dict(reliability_reasons))
    print("First intersection evidence:", dict(first_intersection_evidence))
    print("Unresolved relations:", dict(unresolved_reasons))
    print("Observed branch counts per Anchor:", dict(sorted(branch_counts_by_anchor.items())))

    print()
    print("Example Anchor IDs:")
    for key in sorted(examples):
        print(" ", key[0], "+", key[1], ":", examples[key])

    print()
    print("PASS: first-branch diagnostics completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
