#!/usr/bin/env python3
"""Profile road-level Route geometry for all Step 6 Keyframes.

Read-only with respect to existing annotations. Uses only Anchor-time Route
features already extracted without future information and current Branch
Context. It does not classify or overwrite Navigation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from navigation_route_features_v01 import valid_local_points
from road_level_navigation_features_v01 import (
    FEATURE_VERSION,
    extract_road_level_features,
)
from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

ROOT = ALPASIM_DATA_ROOT
ANN = ANNOTATION_ROOT
DEFAULT_KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
DEFAULT_BRANCH = (
    INTERMEDIATE_ROOT
    / "navigation_branch_context_v0.1.jsonl"
)
DEFAULT_OUTPUT = (
    INTERMEDIATE_ROOT
    / "road_level_navigation_features_v0.1.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT
    / "road_level_navigation_feature_summary_v0.1.json"
)


def read_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in result:
                raise ValueError(f"duplicate Anchor at {path}:{line_number}")
            result[anchor_id] = record
    return result


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
    parser.add_argument("--branch-input", type=Path, default=DEFAULT_BRANCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pre-window-m", type=float, default=15.0)
    parser.add_argument("--post-offset-m", type=float, default=10.0)
    parser.add_argument("--post-window-m", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyframes = read_index(args.keyframe_input)
    branches = read_index(args.branch_input)
    if set(keyframes) != set(branches):
        raise ValueError("Keyframe and Branch Context Anchor sets differ")

    # Route messages are read directly because valid local points are not stored
    # in the route-feature JSONL.
    from clip_reader import DrivingClipReader

    readers: dict[str, DrivingClipReader] = {}
    output = []
    status_counts = Counter()
    reason_counts = Counter()
    relation_counts = Counter()

    ordered = sorted(keyframes.values(), key=lambda item: (
        str(item["clip_id"]), int(item["anchor_ns"]), str(item["anchor_id"])
    ))
    for index, keyframe in enumerate(ordered, start=1):
        anchor_id = str(keyframe["anchor_id"])
        clip_id = str(keyframe["clip_id"])
        anchor_ns = int(keyframe["anchor_ns"])
        branch_record = branches[anchor_id]
        first_branch = branch_record.get("branch_context", {}).get(
            "first_observed_branch"
        )
        features: dict[str, Any]
        if not isinstance(first_branch, dict):
            features = {
                "status": "unavailable",
                "reasons": ["no_observed_route_branch"],
            }
        else:
            reader = readers.get(clip_id)
            if reader is None:
                reader = DrivingClipReader(ROOT / clip_id)
                readers[clip_id] = reader
            timed = reader.get_navigation_route_at(anchor_ns)
            if timed is None:
                features = {
                    "status": "unavailable",
                    "reasons": ["navigation_route_unavailable"],
                }
            else:
                try:
                    features = extract_road_level_features(
                        valid_local_points(timed.message),
                        branch_distance_m=float(first_branch["route_distance_m"]),
                        pre_window_m=args.pre_window_m,
                        post_offset_m=args.post_offset_m,
                        post_window_m=args.post_window_m,
                    )
                except (ValueError, KeyError) as error:
                    features = {
                        "status": "unavailable",
                        "reasons": ["road_level_geometry_error", str(error)],
                    }

        status_counts[features["status"]] += 1
        reason_counts.update(features.get("reasons", []))
        relation_counts[
            str(first_branch.get("route_relation_to_natural"))
            if isinstance(first_branch, dict) else "none"
        ] += 1
        output.append({
            "feature_format_version": FEATURE_VERSION,
            "anchor_id": anchor_id,
            "clip_id": clip_id,
            "anchor_ns": anchor_ns,
            "source_branch_relation": (
                first_branch.get("route_relation_to_natural")
                if isinstance(first_branch, dict) else None
            ),
            "source_branch_reliability": (
                first_branch.get("reliability_status")
                if isinstance(first_branch, dict) else None
            ),
            "road_level_route_geometry": features,
        })
        if index == 1 or index % 250 == 0 or index == len(ordered):
            print(f"Processed {index}/{len(ordered)} Keyframes: {clip_id}")

    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output
    )
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    summary = {
        "feature_format_version": FEATURE_VERSION,
        "input_keyframe_count": len(ordered),
        "output_record_count": len(output),
        "parameters": {
            "pre_window_m": args.pre_window_m,
            "post_offset_m": args.post_offset_m,
            "post_window_m": args.post_window_m,
        },
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "source_branch_relation_counts": dict(relation_counts),
        "output_sha256": sha256,
        "production_artifacts_modified": False,
        "leakage_controls": {
            "future_ego_trajectory_used": False,
            "meta_action_used": False,
        },
    }
    atomic_write(args.output, text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )
    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("SHA-256:", sha256)
    print("Records:", len(output))
    print("Status:", dict(status_counts))
    print("Reasons:", dict(reason_counts))
    print("Production artifacts modified: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
