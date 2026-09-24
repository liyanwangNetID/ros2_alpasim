#!/usr/bin/env python3
"""Read-only Step 8 input inventory for the AlpaSim dataset pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            yield line_number, value


def shape(value: Any, depth: int = 0, max_depth: int = 3) -> Any:
    if depth >= max_depth:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: shape(value[key], depth + 1, max_depth) for key in sorted(value)}
    if isinstance(value, list):
        if not value:
            return []
        return [shape(value[0], depth + 1, max_depth)]
    return type(value).__name__


def get_path(record: dict[str, Any], dotted_path: str) -> Any:
    value: Any = record
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def summarize_jsonl(name: str, path: Path) -> tuple[dict[str, Any], set[str]]:
    anchors: list[str] = []
    first_row: dict[str, Any] | None = None
    field_presence: Counter[str] = Counter()
    candidate_distributions: dict[str, Counter[str]] = {
        "quality.status": Counter(),
        "quality_status": Counter(),
        "navigation.command": Counter(),
        "navigation_direction": Counter(),
        "meta_action": Counter(),
        "action": Counter(),
        "decision": Counter(),
    }
    row_count = 0
    parse_errors: list[str] = []

    try:
        for line_number, row in iter_jsonl(path):
            row_count += 1
            if first_row is None:
                first_row = row
            for key in row:
                field_presence[key] += 1
            anchor = row.get("anchor_id")
            if anchor is not None:
                anchors.append(str(anchor))
            else:
                parse_errors.append(f"line {line_number}: missing anchor_id")
            for dotted_path, counter in candidate_distributions.items():
                value = get_path(row, dotted_path)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    if value is not None:
                        counter[str(value)] += 1
    except Exception as exc:
        parse_errors.append(str(exc))

    anchor_counts = Counter(anchors)
    duplicates = sorted(anchor for anchor, count in anchor_counts.items() if count > 1)
    distributions = {
        key: dict(counter.most_common())
        for key, counter in candidate_distributions.items()
        if counter
    }
    summary = {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": row_count,
        "anchor_count": len(anchors),
        "unique_anchor_count": len(anchor_counts),
        "missing_anchor_count": row_count - len(anchors),
        "duplicate_anchor_count": len(duplicates),
        "duplicate_anchor_examples": duplicates[:20],
        "top_level_fields": sorted(first_row) if first_row else [],
        "first_row_shape": shape(first_row) if first_row else None,
        "field_presence": dict(sorted(field_presence.items())),
        "candidate_distributions": distributions,
        "errors": parse_errors[:50],
    }
    return summary, set(anchor_counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("ALPASIM_DATA_ROOT", "/home/lab/data_from_alpasim")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path; defaults to reports/step8_input_inventory_v02.json under data root.",
    )
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output or root / "reports" / "step8_input_inventory_v02.json"

    inputs = {
        "keyframes": root / "annotations" / "v0.1-draft" / "keyframes.jsonl",
        "meta_actions": root / "annotations" / "v0.1-draft" / "meta_actions_v0.2.jsonl",
        "navigation": root / "annotations" / "v0.1-draft" / "navigation.jsonl",
        "scene_facts": root / "annotations" / "v0.1-draft" / "scene_facts.jsonl",
    }
    contract_path = root / "manifests" / "keyframe_contract_v0.1.json"

    print("[Step 8 inventory] Starting read-only input audit", flush=True)
    print(f"[Step 8 inventory] Data root: {root}", flush=True)

    missing = [str(path) for path in [*inputs.values(), contract_path] if not path.is_file()]
    if missing:
        print("[Step 8 inventory] Missing required files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    summaries: dict[str, Any] = {}
    anchor_sets: dict[str, set[str]] = {}
    for index, (name, path) in enumerate(inputs.items(), start=1):
        print(f"[Step 8 inventory] {index}/4 Reading {name}: {path}", flush=True)
        summary, anchors = summarize_jsonl(name, path)
        summaries[name] = summary
        anchor_sets[name] = anchors
        print(
            f"[Step 8 inventory] {name}: rows={summary['row_count']} "
            f"unique_anchors={summary['unique_anchor_count']} "
            f"duplicates={summary['duplicate_anchor_count']}",
            flush=True,
        )

    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    keyframe_anchors = anchor_sets["keyframes"]
    coverage: dict[str, Any] = {}
    for name, anchors in anchor_sets.items():
        missing_from_input = sorted(keyframe_anchors - anchors)
        extra_in_input = sorted(anchors - keyframe_anchors)
        coverage[name] = {
            "exact_match_with_keyframes": anchors == keyframe_anchors,
            "missing_keyframe_anchor_count": len(missing_from_input),
            "missing_keyframe_anchor_examples": missing_from_input[:20],
            "extra_anchor_count": len(extra_in_input),
            "extra_anchor_examples": extra_in_input[:20],
        }

    expected_hashes = {
        "keyframes": "bb3cc755d537c0b8fa0c68aff457106ee00d583bff448bf737c2d286e0ccabf7",
        "meta_actions": "a07aacf417829e11d2fe437f01318d509d2d5a007a196440f3d9be95110f5973",
        "navigation": "d025699fcfff677e7929c9df13eb72023d8acd6b815d0fa80c604044c6b7bf90",
        "scene_facts": "735203f9ddf3b9f49e892edbb185936caa9db1cd46cbfcdd1b9e0f685958e2b5",
    }
    hash_checks = {
        name: {
            "expected": expected_hashes[name],
            "actual": summaries[name]["sha256"],
            "matches": summaries[name]["sha256"] == expected_hashes[name],
        }
        for name in inputs
    }

    report = {
        "report_format_version": "0.2-draft",
        "purpose": "Read-only Step 8 input inventory with dependency-aware coverage rules",
        "data_root": str(root),
        "inputs": summaries,
        "keyframe_contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "top_level_fields": sorted(contract) if isinstance(contract, dict) else [],
            "content": contract,
        },
        "anchor_coverage": coverage,
        "frozen_hash_checks": hash_checks,
        "coverage_policy": {
            "keyframes": "exact Keyframe anchor set",
            "meta_actions": "must contain every Keyframe anchor; extra Candidate Anchors are expected",
            "navigation": "exact Keyframe anchor set",
            "scene_facts": "exact Keyframe anchor set"
        },
        "ready_for_step8_schema_design": (
            coverage["keyframes"]["exact_match_with_keyframes"]
            and coverage["meta_actions"]["missing_keyframe_anchor_count"] == 0
            and coverage["navigation"]["exact_match_with_keyframes"]
            and coverage["scene_facts"]["exact_match_with_keyframes"]
            and all(item["matches"] for item in hash_checks.values())
            and all(not summary["errors"] for summary in summaries.values())
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"[Step 8 inventory] Report: {output}", flush=True)
    print(
        "[Step 8 inventory] Ready for schema design: "
        f"{report['ready_for_step8_schema_design']}",
        flush=True,
    )
    print("[Step 8 inventory] Completed", flush=True)
    return 0 if report["ready_for_step8_schema_design"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
