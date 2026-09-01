#!/usr/bin/env python3
"""Finalize Meta-action v0.2 from frozen v0.1 labels and reviewed shadow output.

The script never overwrites v0.1. It updates only the lateral action and its
explanation when the reviewed shadow policy proposes a change. Longitudinal
labels are copied verbatim from v0.1.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from project_paths import ALPASIM_DATA_ROOT

VERSION = "0.2.1"
RULE_VERSION = "meta_action_rules_v0.2.1"
ROOT = ALPASIM_DATA_ROOT
DEFAULT_V01_INPUT = ROOT / "annotations/v0.1-draft/meta_actions_v0.1.jsonl"
DEFAULT_SHADOW_INPUT = ROOT / "reports/lateral_shadow_evaluation_v0.1.jsonl"
DEFAULT_OUTPUT = ROOT / "annotations/v0.1-draft/meta_actions_v0.2.jsonl"
DEFAULT_SUMMARY = ROOT / "reports/meta_action_generation_summary_v0.2.json"

VALID_LATERAL = {
    "keep_direction",
    "turn_left",
    "turn_right",
    "change_lane_left",
    "change_lane_right",
    "unknown",
}
VALID_LONGITUDINAL = {
    "accelerate",
    "maintain_speed",
    "decelerate",
    "stop",
    "unknown",
}


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in index:
                raise ValueError(f"duplicate anchor_id at {path}:{line_number}: {anchor_id}")
            index[anchor_id] = record
    return index


def identity_check(base: Mapping[str, Any], shadow: Mapping[str, Any]) -> None:
    for key in ("anchor_id", "clip_id", "anchor_ns"):
        if base.get(key) != shadow.get(key):
            raise ValueError(
                f"identity mismatch for {base.get('anchor_id')}: {key}: "
                f"{base.get(key)!r} != {shadow.get(key)!r}"
            )


def finalize_record(
    base_record: Mapping[str, Any], shadow_record: Mapping[str, Any]
) -> dict[str, Any]:
    identity_check(base_record, shadow_record)
    result = copy.deepcopy(dict(base_record))

    old_action = str(base_record["lateral"]["action"])
    shadow_old = str(shadow_record["old_lateral_action"])
    proposed_action = str(shadow_record["proposed_lateral_action"])

    if old_action != shadow_old:
        raise ValueError(
            f"shadow old action mismatch for {base_record['anchor_id']}: "
            f"{old_action} != {shadow_old}"
        )
    if proposed_action not in VALID_LATERAL:
        raise ValueError(f"invalid proposed lateral action: {proposed_action}")

    changed = old_action != proposed_action
    lateral = result["lateral"]
    lateral["action"] = proposed_action

    if changed:
        lateral["decision_stage"] = "reviewed_shadow_geometry_v0.2"
        lateral["reasons"] = list(shadow_record.get("reasons", []))
        lateral["quality_status"] = "usable" if proposed_action != "unknown" else "unknown"
        lateral["shadow_revision"] = {
            "old_action": old_action,
            "decision_source": shadow_record.get("decision_source"),
            "shadow_evaluator_version": shadow_record.get("shadow_evaluator_version"),
        }
    else:
        lateral["shadow_revision"] = {
            "old_action": old_action,
            "decision_source": shadow_record.get("decision_source"),
            "changed": False,
        }

    longitudinal_action = str(result["longitudinal"]["action"])
    if longitudinal_action not in VALID_LONGITUDINAL:
        raise ValueError(f"invalid longitudinal action: {longitudinal_action}")

    result["label_format_version"] = "0.2-draft"
    result["generator_version"] = VERSION
    result["rule_version"] = RULE_VERSION
    result["joint_action"] = {
        "lateral": proposed_action,
        "longitudinal": longitudinal_action,
    }
    result["overall_quality_status"] = (
        "usable"
        if (
            result["lateral"].get("quality_status") == "usable"
            and result["longitudinal"].get("quality_status") == "usable"
        )
        else "unknown"
    )
    return result


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}; use --force")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v01-input", type=Path, default=DEFAULT_V01_INPUT)
    parser.add_argument("--shadow-input", type=Path, default=DEFAULT_SHADOW_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_index = read_jsonl_index(args.v01_input)
    shadow_index = read_jsonl_index(args.shadow_input)

    if set(base_index) != set(shadow_index):
        raise ValueError(
            "v0.1 and shadow anchor sets differ; "
            f"missing shadow={sorted(set(base_index)-set(shadow_index))[:10]}, "
            f"missing v0.1={sorted(set(shadow_index)-set(base_index))[:10]}"
        )

    records: list[dict[str, Any]] = []
    transition_counts = Counter()
    changed_sources = Counter()

    for anchor_id, base in base_index.items():
        shadow = shadow_index[anchor_id]
        record = finalize_record(base, shadow)
        old_action = str(base["lateral"]["action"])
        new_action = str(record["lateral"]["action"])
        transition_counts[(old_action, new_action)] += 1
        if old_action != new_action:
            changed_sources[str(shadow.get("decision_source"))] += 1
        records.append(record)

    lateral_counts = Counter(record["lateral"]["action"] for record in records)
    longitudinal_counts = Counter(record["longitudinal"]["action"] for record in records)
    quality_counts = Counter(record["overall_quality_status"] for record in records)
    changed_count = sum(
        count for (old, new), count in transition_counts.items() if old != new
    )

    summary = {
        "label_format_version": "0.2-draft",
        "generator_version": VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_v01": str(args.v01_input),
        "source_shadow": str(args.shadow_input),
        "anchor_count": len(records),
        "changed_lateral_count": changed_count,
        "unchanged_lateral_count": len(records) - changed_count,
        "lateral_action_counts": dict(lateral_counts),
        "longitudinal_action_counts": dict(longitudinal_counts),
        "overall_quality_counts": dict(quality_counts),
        "changed_decision_source_counts": dict(changed_sources),
        "lateral_transition_matrix": [
            {"old_action": old, "new_action": new, "count": count}
            for (old, new), count in sorted(transition_counts.items())
        ],
        "frozen_policy": {
            "lane_change_to_keep_direction": "preserve_v0.1_lane_change",
            "lane_change_to_turn": {
                "apply_reviewed_shadow_geometry": True,
                "allowed_junction_levels": [
                    "A",
                    "B"
                ],
                "minimum_absolute_post_transition_ego_heading_change_deg": 8.0,
                "otherwise": "preserve_v0.1_lane_change"
            },
            "keep_direction_to_lane_change": {
                "apply_reviewed_shadow_geometry": True,
                "minimum_final_target_advantage_m": -2.0,
                "maximum_absolute_directional_heading_progress_deg": 10.0,
                "minimum_absolute_ego_to_map_heading_residual_deg": 2.0,
            },
            "longitudinal": "inherit_v0.1_unchanged",
        },
    }

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )

    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("Label SHA-256:", hashlib.sha256(output_text.encode()).hexdigest())
    print("Anchor count:", len(records))
    print("Changed lateral:", changed_count)
    print("Lateral counts:", dict(lateral_counts))
    print("Longitudinal counts:", dict(longitudinal_counts))
    print("Overall quality:", dict(quality_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
