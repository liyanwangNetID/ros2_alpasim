#!/usr/bin/env python3
"""Direct production builder for frozen Step 4 Meta-action v0.2 labels.

This is the only production entry point for Step 4. It reads the three frozen
feature inputs, applies meta_action_rules_v02 directly, and writes the final
Meta-action labels. It does not generate v0.1 labels or Shadow artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from meta_action_rules_v02 import (
    GENERATOR_VERSION,
    LABEL_FORMAT_VERSION,
    RULE_VERSION,
    SHADOW_POLICY_VERSION,
    make_meta_action_record,
)
from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

ROOT = ALPASIM_DATA_ROOT
DEFAULT_LATERAL_INPUT = (
    INTERMEDIATE_ROOT / "lateral_action_features_v0.3.jsonl"
)
DEFAULT_LONGITUDINAL_INPUT = (
    INTERMEDIATE_ROOT / "meta_action_features_v0.2.jsonl"
)
DEFAULT_GEOMETRY_INPUT = (
    INTERMEDIATE_ROOT / "lane_change_geometry_features_v0.1.jsonl"
)
DEFAULT_OUTPUT = (
    ANNOTATION_ROOT / "meta_actions_v0.2.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "meta_action_generation_summary_v0.2.json"
)

EXPECTED_ANCHOR_COUNT = 10231
EXPECTED_LATERAL_COUNTS = {
    "keep_direction": 9472,
    "unknown": 387,
    "turn_left": 16,
    "turn_right": 57,
    "change_lane_left": 155,
    "change_lane_right": 144,
}
EXPECTED_LONGITUDINAL_COUNTS = {
    "maintain_speed": 5589,
    "unknown": 401,
    "accelerate": 1532,
    "decelerate": 1810,
    "stop": 899,
}
EXPECTED_QUALITY_COUNTS = {"usable": 9465, "unknown": 766}


def step4_feature_commands() -> tuple[tuple[str, ...], ...]:
    """Return the frozen Step 4 feature-generation command sequence."""
    return (
        (sys.executable, "profile_lane_matching_features.py"),
        (sys.executable, "refine_lane_matching_features.py"),
        (sys.executable, "profile_lateral_action_features.py"),
        (sys.executable, "profile_meta_action_features.py"),
        (
            sys.executable,
            "profile_lane_change_geometry_features.py",
            "--all",
            "--force",
        ),
    )


def run_step4_feature_pipeline() -> None:
    """Generate all Step 4 feature inputs from Candidate Anchors."""
    script_directory = Path(__file__).resolve().parent
    commands = step4_feature_commands()

    for index, command in enumerate(commands, start=1):
        print()
        print("=" * 78)
        print(f"STEP 4 FEATURE STAGE {index}/{len(commands)}")
        print("Command:", " ".join(command))
        print("=" * 78)

        completed = subprocess.run(
            command,
            cwd=script_directory,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Step 4 feature stage failed with exit code "
                f"{completed.returncode}: {' '.join(command)}"
            )


def require_feature_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required Step 4 feature files are missing: " + ", ".join(missing)
        )


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in index:
                raise ValueError(
                    f"duplicate anchor_id at {path}:{line_number}: {anchor_id}"
                )
            index[anchor_id] = record
    return index


def require_matching_identity(
    anchor_id: str,
    lateral: Mapping[str, Any],
    longitudinal: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> None:
    for name, record in (
        ("longitudinal", longitudinal),
        ("geometry", geometry),
    ):
        for key in ("anchor_id", "clip_id", "anchor_ns"):
            if lateral.get(key) != record.get(key):
                raise ValueError(
                    f"{anchor_id}: {key} mismatch between lateral and {name} inputs"
                )


def future_horizon_ns(*records: Mapping[str, Any]) -> int:
    values = {
        int(record["future_horizon_ns"])
        for record in records
        if "future_horizon_ns" in record
    }
    if len(values) != 1:
        raise ValueError(f"future_horizon_ns mismatch or missing: {sorted(values)}")
    return next(iter(values))


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


def validate_distribution(
    *,
    anchor_count: int,
    lateral_counts: Counter[str],
    longitudinal_counts: Counter[str],
    quality_counts: Counter[str],
    strict: bool,
) -> None:
    if not strict:
        return
    if anchor_count != EXPECTED_ANCHOR_COUNT:
        raise ValueError(
            f"frozen dataset anchor count mismatch: {anchor_count} "
            f"!= {EXPECTED_ANCHOR_COUNT}"
        )
    if dict(lateral_counts) != EXPECTED_LATERAL_COUNTS:
        raise ValueError(
            f"frozen lateral distribution mismatch: {dict(lateral_counts)}"
        )
    if dict(longitudinal_counts) != EXPECTED_LONGITUDINAL_COUNTS:
        raise ValueError(
            "frozen longitudinal distribution mismatch: "
            f"{dict(longitudinal_counts)}"
        )
    if dict(quality_counts) != EXPECTED_QUALITY_COUNTS:
        raise ValueError(
            f"frozen quality distribution mismatch: {dict(quality_counts)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lateral-input", type=Path, default=DEFAULT_LATERAL_INPUT)
    parser.add_argument(
        "--longitudinal-input", type=Path, default=DEFAULT_LONGITUDINAL_INPUT
    )
    parser.add_argument("--geometry-input", type=Path, default=DEFAULT_GEOMETRY_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-existing-features",
        action="store_true",
        help=(
            "Skip Step 4 feature generation and build labels from existing "
            "feature files. The default production mode regenerates all "
            "Step 4 features from Candidate Anchors."
        ),
    )
    parser.add_argument(
        "--no-strict-distribution-check",
        action="store_true",
        help="Allow the frozen rules to run on a different Anchor set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.reuse_existing_features:
        run_step4_feature_pipeline()

    require_feature_files(
        args.lateral_input,
        args.longitudinal_input,
        args.geometry_input,
    )

    lateral_index = read_jsonl_index(args.lateral_input)
    longitudinal_index = read_jsonl_index(args.longitudinal_input)
    geometry_index = read_jsonl_index(args.geometry_input)

    lateral_ids = set(lateral_index)
    longitudinal_ids = set(longitudinal_index)
    geometry_ids = set(geometry_index)
    if lateral_ids != longitudinal_ids or lateral_ids != geometry_ids:
        raise ValueError(
            "feature Anchor sets differ; "
            f"missing longitudinal={sorted(lateral_ids-longitudinal_ids)[:10]}, "
            f"extra longitudinal={sorted(longitudinal_ids-lateral_ids)[:10]}, "
            f"missing geometry={sorted(lateral_ids-geometry_ids)[:10]}, "
            f"extra geometry={sorted(geometry_ids-lateral_ids)[:10]}"
        )

    records: list[dict[str, Any]] = []
    lateral_counts: Counter[str] = Counter()
    longitudinal_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for anchor_id in sorted(lateral_ids):
        lateral = lateral_index[anchor_id]
        longitudinal = longitudinal_index[anchor_id]
        geometry = geometry_index[anchor_id]
        require_matching_identity(anchor_id, lateral, longitudinal, geometry)

        record = make_meta_action_record(
            anchor_id=anchor_id,
            clip_id=str(lateral["clip_id"]),
            anchor_ns=int(lateral["anchor_ns"]),
            future_horizon_ns=future_horizon_ns(
                lateral, longitudinal, geometry
            ),
            lateral_features=lateral,
            longitudinal_features=longitudinal,
            geometry_features=geometry,
        )
        record["source_versions"] = {
            "lateral_feature_format_version": str(
                lateral["feature_format_version"]
            ),
            "longitudinal_feature_format_version": str(
                longitudinal["feature_format_version"]
            ),
        }

        records.append(record)
        lateral_counts[str(record["lateral"]["action"])] += 1
        longitudinal_counts[str(record["longitudinal"]["action"])] += 1
        quality_counts[str(record["overall_quality_status"])] += 1

    validate_distribution(
        anchor_count=len(records),
        lateral_counts=lateral_counts,
        longitudinal_counts=longitudinal_counts,
        quality_counts=quality_counts,
        strict=not args.no_strict_distribution_check,
    )

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    output_sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()

    summary = {
        "label_format_version": LABEL_FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rule_version": RULE_VERSION,
        "shadow_policy_version": SHADOW_POLICY_VERSION,
        "production_mode": "direct_frozen_rules",
        "step4_feature_generation": (
            "reused_existing_features"
            if args.reuse_existing_features
            else "regenerated_from_candidate_anchors"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "lateral_features": str(args.lateral_input),
            "longitudinal_features": str(args.longitudinal_input),
            "geometry_features": str(args.geometry_input),
        },
        "output": str(args.output),
        "label_sha256": output_sha256,
        "anchor_count": len(records),
        "lateral_action_counts": dict(lateral_counts),
        "longitudinal_action_counts": dict(longitudinal_counts),
        "overall_quality_counts": dict(quality_counts),
        "strict_distribution_check": not args.no_strict_distribution_check,
        "intermediate_label_or_shadow_artifacts_created": False,
    }

    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )

    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("Label SHA-256:", output_sha256)
    print("Anchor count:", len(records))
    print("Lateral counts:", dict(lateral_counts))
    print("Longitudinal counts:", dict(longitudinal_counts))
    print("Overall quality:", dict(quality_counts))
    print("Production mode: direct_frozen_rules")
    print(
        "Step 4 feature generation:",
        "reused_existing_features"
        if args.reuse_existing_features
        else "regenerated_from_candidate_anchors",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
