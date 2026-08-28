#!/usr/bin/env python3
"""Generate Step 6 v0.1 Navigation candidates for manual review."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from navigation_rules_v01 import (
    GENERATOR_VERSION,
    OUTPUT_FORMAT_VERSION,
    RULE_VERSION,
    DIRECTION_GEOMETRY_MINIMUM_DEG,
    ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG,
    MAXIMUM_UPCOMING_DISTANCE_M,
    MINIMUM_UPCOMING_DISTANCE_M,
    UPCOMING_TIME_HORIZON_SEC,
    VALID_ACTIONS,
    classify_navigation,
)

ROOT = Path("/home/lab/data_from_alpasim")
ANN = ROOT / "annotations/v0.1-draft"
DEFAULT_KEYFRAMES = ANN / "keyframes.jsonl"
DEFAULT_ROUTE_FEATURES = ANN / "intermediate/navigation_route_features_v0.1.jsonl"
DEFAULT_BRANCH_FEATURES = ANN / "intermediate/navigation_branch_context_v0.1.jsonl"
DEFAULT_ROAD_LEVEL_FEATURES = ANN / "intermediate/road_level_navigation_features_v0.1.jsonl"
DEFAULT_OUTPUT = ANN / "intermediate/navigation_candidates_v0.1.jsonl"
DEFAULT_SUMMARY = ROOT / "reports/navigation_candidate_summary_v0.1.json"


def read_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in result:
                raise ValueError(f"duplicate Anchor at {path}:{line_number}: {anchor_id}")
            result[anchor_id] = record
    return result


def require_identity(anchor_id: str, reference: Mapping[str, Any], *others: Mapping[str, Any]) -> None:
    for record in others:
        for key in ("anchor_id", "clip_id", "anchor_ns"):
            if reference.get(key) != record.get(key):
                raise ValueError(f"{anchor_id}: identity mismatch for {key}")


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}; use --force")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-input", type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--route-feature-input", type=Path, default=DEFAULT_ROUTE_FEATURES)
    parser.add_argument("--branch-feature-input", type=Path, default=DEFAULT_BRANCH_FEATURES)
    parser.add_argument("--road-level-feature-input", type=Path, default=DEFAULT_ROAD_LEVEL_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyframes = read_index(args.keyframe_input)
    route_features = read_index(args.route_feature_input)
    branch_features = read_index(args.branch_feature_input)
    road_level_features = read_index(args.road_level_feature_input)
    expected = set(keyframes)
    if (
        set(route_features) != expected
        or set(branch_features) != expected
        or set(road_level_features) != expected
    ):
        raise ValueError("Navigation input Anchor sets differ")

    output = []
    action_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unknown_reason_counts: Counter[str] = Counter()

    for anchor_id in sorted(
        expected,
        key=lambda value: (
            str(keyframes[value]["clip_id"]),
            int(keyframes[value]["anchor_ns"]),
            value,
        ),
    ):
        keyframe = keyframes[anchor_id]
        route = route_features[anchor_id]
        branch = branch_features[anchor_id]
        road_level = road_level_features[anchor_id]
        require_identity(anchor_id, keyframe, route, branch, road_level)
        navigation = classify_navigation(route, branch, road_level)
        if navigation["action"] not in VALID_ACTIONS:
            raise ValueError(f"{anchor_id}: invalid Navigation action")
        if navigation["action"] == "unknown" and navigation["text"] is not None:
            raise ValueError(f"{anchor_id}: unknown Navigation must use null text")

        record = {
            "navigation_format_version": OUTPUT_FORMAT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "rule_version": RULE_VERSION,
            "anchor_id": anchor_id,
            "clip_id": str(keyframe["clip_id"]),
            "anchor_ns": int(keyframe["anchor_ns"]),
            "navigation": navigation,
            "source_versions": {
                "route_feature_format_version": route.get("feature_format_version"),
                "branch_context_format_version": branch.get("feature_format_version"),
                "road_level_feature_format_version": road_level.get("feature_format_version"),
            },
        }
        output.append(record)
        action_counts[navigation["action"]] += 1
        quality_counts[navigation["quality_status"]] += 1
        source_counts[navigation["decision_source"]] += 1
        text_counts[str(navigation["text"])] += 1
        if navigation["action"] == "unknown":
            unknown_reason_counts.update(navigation["reasons"])

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "navigation_format_version": OUTPUT_FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_mode": True,
        "input_keyframe_count": len(expected),
        "output_record_count": len(output),
        "upcoming_time_horizon_sec": UPCOMING_TIME_HORIZON_SEC,
        "minimum_upcoming_distance_m": MINIMUM_UPCOMING_DISTANCE_M,
        "maximum_upcoming_distance_m": MAXIMUM_UPCOMING_DISTANCE_M,
        "direction_geometry_minimum_deg": DIRECTION_GEOMETRY_MINIMUM_DEG,
        "road_level_intersection_direction_threshold_deg": ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG,
        "action_counts": dict(action_counts),
        "quality_status_counts": dict(quality_counts),
        "decision_source_counts": dict(source_counts),
        "text_counts": dict(text_counts),
        "unknown_reason_counts": dict(unknown_reason_counts),
        "output_sha256": sha256,
        "leakage_controls": {
            "future_ego_trajectory_used": False,
            "future_speed_used": False,
            "future_control_used": False,
            "meta_action_used_for_generation": False,
        },
    }
    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )

    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("SHA-256:", sha256)
    print("Records:", len(output))
    print("Actions:", dict(action_counts))
    print("Quality:", dict(quality_counts))
    print("Decision sources:", dict(source_counts))
    print("Texts:", dict(text_counts))
    print("Unknown reasons:", dict(unknown_reason_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
