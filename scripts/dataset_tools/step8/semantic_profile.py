#!/usr/bin/env python3
"""Profile the current Step 8 Structured CoC rules on all frozen Keyframes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from step8.causality import build_structured_causality
from step8.contract import Step8Paths, load_and_validate_inputs


def sorted_numeric(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: int(item[0])))


def append_unique_example(examples: list[str], anchor_id: str, limit: int) -> None:
    if anchor_id not in examples and len(examples) < limit:
        examples.append(anchor_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = Step8Paths.from_data_root(args.data_root)
    output = args.output or paths.data_root / "reports" / "step8_semantic_profile_v02.json"

    print("[Step 8 profile] Validating and joining frozen inputs", flush=True)
    joined_rows, provenance = load_and_validate_inputs(paths)
    print(f"[Step 8 profile] Joined rows: {len(joined_rows)}", flush=True)

    relation_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    node_type_counts: Counter[str] = Counter()
    link_count_distribution: Counter[str] = Counter()
    actor_node_count_distribution: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    rule_examples: dict[str, list[str]] = {}
    conflict_examples: list[str] = []
    no_supported_examples: list[str] = []

    total = len(joined_rows)
    for index, joined_row in enumerate(joined_rows, 1):
        record = build_structured_causality(joined_row)
        nodes = record["structured_coc"]["nodes"]
        links = record["structured_coc"]["links"]
        quality_counts[record["quality"]["status"]] += 1
        link_count_distribution[str(len(links))] += 1
        actor_node_count_distribution[str(sum(
            node["node_type"] == "actor_state" for node in nodes
        ))] += 1
        for node in nodes:
            node_type_counts[node["node_type"]] += 1
        has_supported = False
        for link in links:
            relation_counts[link["relation"]] += 1
            confidence_counts[link["confidence"]] += 1
            rule_id = link["rule_id"]
            rule_counts[rule_id] += 1
            has_supported = has_supported or link["confidence"] == "supported"
            rule_examples.setdefault(rule_id, [])
            append_unique_example(
                rule_examples[rule_id], joined_row.anchor_id, 10
            )
            if link["relation"] == "conflicts_with":
                append_unique_example(
                    conflict_examples, joined_row.anchor_id, 50
                )
        for reason in record["quality"]["reasons"]:
            reason_counts[reason] += 1
        if not has_supported:
            append_unique_example(
                no_supported_examples, joined_row.anchor_id, 50
            )
        if index % 500 == 0 or index == total:
            print(
                f"[Step 8 profile] {index}/{total} | "
                f"{100.0 * index / total:.1f}%",
                flush=True,
            )

    report = {
        "report_format_version": "0.2-draft",
        "record_count": total,
        "input_provenance": provenance,
        "quality": dict(quality_counts.most_common()),
        "relations": dict(relation_counts.most_common()),
        "confidence": dict(confidence_counts.most_common()),
        "rules": dict(rule_counts.most_common()),
        "node_types": dict(node_type_counts.most_common()),
        "link_count_distribution": sorted_numeric(link_count_distribution),
        "actor_node_count_distribution": sorted_numeric(actor_node_count_distribution),
        "quality_reasons": dict(reason_counts.most_common()),
        "rule_anchor_examples": dict(sorted(rule_examples.items())),
        "conflict_anchor_examples": conflict_examples,
        "no_supported_link_anchor_examples": no_supported_examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"[Step 8 profile] Report: {output}", flush=True)
    print(f"[Step 8 profile] Quality: {dict(quality_counts)}", flush=True)
    print(f"[Step 8 profile] Relations: {dict(relation_counts)}", flush=True)
    print("[Step 8 profile] Completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
