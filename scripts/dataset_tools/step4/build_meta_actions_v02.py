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

from step4.meta_action_rules_v02 import (
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
    MANIFEST_ROOT,
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
DEFAULT_CANDIDATE_ANCHORS = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
DEFAULT_ANCHOR_CONTRACT = (
    MANIFEST_ROOT / "candidate_anchor_contract_v0.1.json"
)
EXPECTED_CONTRACT_VERSION = "0.1"
EXPECTED_CONTRACT_PRODUCER_STEP = 3


def step4_feature_commands() -> tuple[tuple[str, ...], ...]:
    """Return the frozen Step 4 feature-generation command sequence."""
    return (
        (sys.executable, "-m", "step4.profile_lane_matching_features"),
        (sys.executable, "-m", "step4.refine_lane_matching_features"),
        (sys.executable, "-m", "step4.profile_lateral_action_features"),
        (sys.executable, "-m", "step4.profile_meta_action_features"),
        (
            sys.executable,
            "-m",
            "step4.profile_lane_change_geometry_features",
            "--all",
            "--force",
        ),
    )


def run_step4_feature_pipeline() -> None:
    """Generate all Step 4 feature inputs from Candidate Anchors."""
    project_directory = Path(__file__).resolve().parent.parent
    commands = step4_feature_commands()

    for index, command in enumerate(commands, start=1):
        print()
        print("=" * 78)
        print(f"STEP 4 FEATURE STAGE {index}/{len(commands)}")
        print("Command:", " ".join(command))
        print("=" * 78)

        completed = subprocess.run(
            command,
            cwd=project_directory,
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



def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"required JSON file not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON root must be an object: {path}"
        )

    return value


def validate_candidate_anchor_contract(
    *,
    contract_path: Path,
    candidate_anchor_path: Path,
) -> tuple[dict[str, Any], set[str]]:
    contract = read_json_object(contract_path)

    if (
        str(contract.get("contract_version"))
        != EXPECTED_CONTRACT_VERSION
    ):
        raise ValueError(
            "unsupported Candidate Anchor contract version: "
            f"{contract.get('contract_version')}"
        )

    if (
        contract.get("producer_step")
        != EXPECTED_CONTRACT_PRODUCER_STEP
    ):
        raise ValueError(
            "Candidate Anchor contract producer_step must be "
            f"{EXPECTED_CONTRACT_PRODUCER_STEP}"
        )

    try:
        candidate_data = candidate_anchor_path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Candidate Anchor file not found: "
            f"{candidate_anchor_path}"
        ) from exc

    actual_digest = hashlib.sha256(
        candidate_data
    ).hexdigest()
    expected_digest = str(
        contract.get("candidate_anchor_sha256", "")
    )

    if actual_digest != expected_digest:
        raise ValueError(
            "Candidate Anchor SHA-256 mismatch: "
            f"{actual_digest} != {expected_digest}"
        )

    candidate_index = read_jsonl_index(
        candidate_anchor_path
    )
    candidate_ids = set(candidate_index)

    expected_count = int(
        contract["candidate_anchor_count"]
    )

    if len(candidate_ids) != expected_count:
        raise ValueError(
            "Candidate Anchor count mismatch: "
            f"{len(candidate_ids)} != {expected_count}"
        )

    return contract, candidate_ids


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lateral-input", type=Path, default=DEFAULT_LATERAL_INPUT)
    parser.add_argument(
        "--longitudinal-input", type=Path, default=DEFAULT_LONGITUDINAL_INPUT
    )
    parser.add_argument("--geometry-input", type=Path, default=DEFAULT_GEOMETRY_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--candidate-anchor-input",
        type=Path,
        default=DEFAULT_CANDIDATE_ANCHORS,
    )
    parser.add_argument(
        "--anchor-contract",
        type=Path,
        default=DEFAULT_ANCHOR_CONTRACT,
    )
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.reuse_existing_features:
        run_step4_feature_pipeline()

    require_feature_files(
        args.lateral_input,
        args.longitudinal_input,
        args.geometry_input,
        args.candidate_anchor_input,
        args.anchor_contract,
    )

    (
        anchor_contract,
        candidate_anchor_ids,
    ) = validate_candidate_anchor_contract(
        contract_path=args.anchor_contract,
        candidate_anchor_path=args.candidate_anchor_input,
    )

    lateral_index = read_jsonl_index(args.lateral_input)
    longitudinal_index = read_jsonl_index(args.longitudinal_input)
    geometry_index = read_jsonl_index(args.geometry_input)

    lateral_ids = set(lateral_index)
    longitudinal_ids = set(longitudinal_index)
    geometry_ids = set(geometry_index)
    if (
        lateral_ids != candidate_anchor_ids
        or longitudinal_ids != candidate_anchor_ids
        or geometry_ids != candidate_anchor_ids
    ):
        raise ValueError(
            "Step 4 feature Anchor sets do not match "
            "Candidate Anchors; "
            f"lateral_missing="
            f"{len(candidate_anchor_ids - lateral_ids)}, "
            f"lateral_extra="
            f"{len(lateral_ids - candidate_anchor_ids)}, "
            f"longitudinal_missing="
            f"{len(candidate_anchor_ids - longitudinal_ids)}, "
            f"longitudinal_extra="
            f"{len(longitudinal_ids - candidate_anchor_ids)}, "
            f"geometry_missing="
            f"{len(candidate_anchor_ids - geometry_ids)}, "
            f"geometry_extra="
            f"{len(geometry_ids - candidate_anchor_ids)}"
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
        "candidate_anchor_contract": str(
            args.anchor_contract
        ),
        "candidate_anchor_input": str(
            args.candidate_anchor_input
        ),
        "candidate_anchor_sha256": str(
            anchor_contract["candidate_anchor_sha256"]
        ),
        "candidate_anchor_count": int(
            anchor_contract["candidate_anchor_count"]
        ),
        "anchor_coverage_valid": True,
        "lateral_action_counts": dict(lateral_counts),
        "longitudinal_action_counts": dict(longitudinal_counts),
        "overall_quality_counts": dict(quality_counts),
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
