#!/usr/bin/env python3
"""Randomly review one Meta-action Anchor and generate its video.

Each run samples one record uniformly from all non-empty records in the final
Meta-action JSONL using reservoir sampling and a cryptographic system RNG.
"""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
)

DATA_ROOT = ALPASIM_DATA_ROOT
TOOL_ROOT = Path(__file__).resolve().parent

DEFAULT_LABEL_INPUT = (
    ANNOTATION_ROOT / "meta_actions_v0.2.jsonl"
)
DEFAULT_REVIEW_SCRIPT = TOOL_ROOT / "review_lateral_action.py"
DEFAULT_LATERAL_FEATURE_INPUT = (
    INTERMEDIATE_ROOT / "lateral_action_features_v0.3.jsonl"
)


CATEGORY_ALIASES = {
    "straight": "straight",
    "直行": "straight",
    "turn": "turn",
    "转弯": "turn",
    "lane_change": "lane_change",
    "变道": "lane_change",
    "maintain_speed": "maintain_speed",
    "匀速": "maintain_speed",
    "accelerate": "accelerate",
    "加速": "accelerate",
    "decelerate": "decelerate",
    "减速": "decelerate",
    "stop": "stop",
    "停车": "stop",
}


def record_matches_category(record: dict[str, Any], category: str | None) -> bool:
    if category is None:
        return True

    lateral_action = str(record.get("lateral", {}).get("action", ""))
    longitudinal_action = str(record.get("longitudinal", {}).get("action", ""))

    if category == "straight":
        return lateral_action == "keep_direction"
    if category == "turn":
        return lateral_action in {"turn_left", "turn_right"}
    if category == "lane_change":
        return lateral_action in {"change_lane_left", "change_lane_right"}
    return longitudinal_action == category


def sample_uniform_record(
    path: Path,
    category: str | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Uniformly sample one matching JSONL record using reservoir sampling."""
    selected: dict[str, Any] | None = None
    total_count = 0
    eligible_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")

            for required_key in ("anchor_id", "clip_id", "anchor_ns"):
                if required_key not in record:
                    raise ValueError(
                        f"{path}:{line_number} is missing {required_key}"
                    )

            total_count += 1
            if not record_matches_category(record, category):
                continue

            eligible_count += 1
            if secrets.randbelow(eligible_count) == 0:
                selected = record

    if selected is None:
        display = category if category is not None else "all"
        raise ValueError(f"no records found for category: {display}")

    return selected, eligible_count, total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample one final Meta-action Anchor, print its predicted "
            "actions, and generate the front_wide review video."
        )
    )
    parser.add_argument(
        "--label-input",
        type=Path,
        default=DEFAULT_LABEL_INPUT,
    )
    parser.add_argument(
        "--review-script",
        type=Path,
        default=DEFAULT_REVIEW_SCRIPT,
    )
    parser.add_argument(
        "--category",
        choices=tuple(CATEGORY_ALIASES),
        help=(
            "Restrict random sampling to one category. Supported Chinese and "
            "English values: 直行/straight, 转弯/turn, 变道/lane_change, "
            "匀速/maintain_speed, 加速/accelerate, 减速/decelerate, 停车/stop. "
            "If omitted, sample globally from all Anchors."
        ),
    )
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument("--maximum-width", type=int, default=960)
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not pass --force to the video generator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.label_input.is_file():
        raise FileNotFoundError(
            "Meta-action label file not found: " + str(args.label_input)
        )
    if not args.review_script.is_file():
        raise FileNotFoundError(
            "Review script not found: " + str(args.review_script)
        )
    if args.maximum_width <= 0:
        raise ValueError("--maximum-width must be positive")

    normalized_category = (
        CATEGORY_ALIASES[args.category]
        if args.category is not None
        else None
    )
    record, eligible_count, total_count = sample_uniform_record(
        args.label_input,
        normalized_category,
    )

    lateral = record.get("lateral", {})
    longitudinal = record.get("longitudinal", {})
    joint_action = record.get("joint_action", {})

    anchor_id = str(record["anchor_id"])

    print("=" * 78)
    print("RANDOM META-ACTION REVIEW")
    print("=" * 78)
    print("Sampling method: uniform reservoir sampling with secrets.SystemRandom")
    print("Requested category:", args.category or "all")
    print("Normalized category:", normalized_category or "all")
    print("Total Anchors:", total_count)
    print("Eligible Anchors:", eligible_count)
    print("Anchor ID:", anchor_id)
    print("Clip ID:", record["clip_id"])
    print("Anchor ns:", record["anchor_ns"])
    print("LATERAL ACTION:", lateral.get("action"))
    print("LONGITUDINAL ACTION:", longitudinal.get("action"))
    print("JOINT ACTION:", joint_action)
    print("Overall quality:", record.get("overall_quality_status"))
    print("Lateral decision stage:", lateral.get("decision_stage"))
    print("Lateral reasons:", lateral.get("reasons"))
    print("Longitudinal decision stage:", longitudinal.get("decision_stage"))
    print("Longitudinal reasons:", longitudinal.get("reasons"))
    print("=" * 78)

    command = [
        sys.executable,
        str(args.review_script),
        "--anchor-id",
        anchor_id,
        "--feature-input",
        str(DEFAULT_LATERAL_FEATURE_INPUT),
        "--preset",
        args.preset,
        "--maximum-width",
        str(args.maximum_width),
    ]
    if not args.no_force:
        command.append("--force")

    completed = subprocess.run(
        command,
        cwd=args.review_script.parent,
        stdin=subprocess.DEVNULL,
        check=False,
    )

    print()
    print("Video command exit code:", completed.returncode)
    if completed.returncode != 0:
        print("Video generation failed for Anchor:", anchor_id)
        return completed.returncode

    print("PASS: random review video generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
