#!/usr/bin/env python3
"""Step 5C v0.1 balanced keyframe selector."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    MANIFEST_ROOT,
    REPORT_ROOT,
)

SELECTOR_VERSION = "0.1.0"


RULE_VERSION = "keyframe_selection_rules_v0.1"


OUTPUT_FORMAT_VERSION = "0.1-draft"


DETERMINISTIC_SEED = "alpasim-keyframe-selection-v0.1"


REFERENCE_ANCHOR_COUNT = 10_231

REFERENCE_STABLE_LATERAL_QUOTAS = {
    "turn_left": None,
    "turn_right": None,
    "change_lane_left": 50,
    "change_lane_right": 50,
}
REFERENCE_STABLE_LONGITUDINAL_QUOTAS = {
    "accelerate": 100,
    "decelerate": 100,
    "stop": 100,
}
REFERENCE_NORMAL_BASELINE_QUOTA = 500


def scaled_quota(
    anchor_count: int,
    reference_quota: int,
) -> int:
    if anchor_count <= 0:
        raise ValueError(
            "anchor_count must be positive"
        )

    if reference_quota <= 0:
        raise ValueError(
            "reference_quota must be positive"
        )

    scale = anchor_count / REFERENCE_ANCHOR_COUNT
    scaled = round(scale * reference_quota)

    return max(1, scaled)


def selection_quotas(
    anchor_count: int,
) -> dict[str, Any]:
    if anchor_count <= 0:
        raise ValueError(
            "anchor_count must be positive"
        )

    lateral = {
        action: (
            None
            if reference_quota is None
            else scaled_quota(
                anchor_count,
                reference_quota,
            )
        )
        for action, reference_quota
        in REFERENCE_STABLE_LATERAL_QUOTAS.items()
    }

    longitudinal = {
        action: scaled_quota(
            anchor_count,
            reference_quota,
        )
        for action, reference_quota
        in REFERENCE_STABLE_LONGITUDINAL_QUOTAS.items()
    }

    return {
        "reference_anchor_count": (
            REFERENCE_ANCHOR_COUNT
        ),
        "anchor_count": anchor_count,
        "stable_lateral_quotas": lateral,
        "stable_longitudinal_quotas": longitudinal,
        "normal_baseline_quota": scaled_quota(
            anchor_count,
            REFERENCE_NORMAL_BASELINE_QUOTA,
        ),
    }



def deterministic_rank(anchor_id: str, bucket: str) -> str:
    value = f"{DETERMINISTIC_SEED}|{bucket}|{anchor_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_with_clip_preference(
    records: Sequence[Mapping[str, Any]],
    *,
    quota: int | None,
    bucket: str,
    clip_id: Callable[[Mapping[str, Any]], str] = lambda record: str(record["clip_id"]),
    anchor_id: Callable[[Mapping[str, Any]], str] = lambda record: str(record["anchor_id"]),
) -> list[Mapping[str, Any]]:
    """Select deterministically, choosing at most one per Clip before filling."""
    ordered = sorted(records, key=lambda record: deterministic_rank(anchor_id(record), bucket))
    if quota is None or quota >= len(ordered):
        return ordered

    by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in ordered:
        by_clip[clip_id(record)].append(record)

    chosen: list[Mapping[str, Any]] = []
    chosen_ids: set[str] = set()
    clip_order = sorted(
        by_clip,
        key=lambda value: deterministic_rank(anchor_id(by_clip[value][0]), bucket + "|clip"),
    )
    for clip in clip_order:
        record = by_clip[clip][0]
        chosen.append(record)
        chosen_ids.add(anchor_id(record))
        if len(chosen) == quota:
            return chosen

    for record in ordered:
        if anchor_id(record) in chosen_ids:
            continue
        chosen.append(record)
        if len(chosen) == quota:
            return chosen
    return chosen


ROOT = ALPASIM_DATA_ROOT
ANN = ANNOTATION_ROOT
DEFAULT_CANDIDATES = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
DEFAULT_META = (
    ANNOTATION_ROOT / "meta_actions_v0.2.jsonl"
)
DEFAULT_EVENTS = (
    INTERMEDIATE_ROOT
    / "keyframe_event_candidates_deduplicated_v0.1.jsonl"
)
DEFAULT_OUTPUT = ANNOTATION_ROOT / "keyframes.jsonl"
DEFAULT_SUMMARY = (
    REPORT_ROOT / "keyframe_selection_summary_v0.1.json"
)
DEFAULT_META_ACTION_CONTRACT = (
    MANIFEST_ROOT / "meta_action_contract_v0.2.json"
)
EXPECTED_META_ACTION_CONTRACT_VERSION = "0.2"
EXPECTED_META_ACTION_PRODUCER_STEP = 4


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



def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
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


def validate_meta_action_contract(
    contract_path: Path,
    meta_path: Path,
    candidate_count: int,
) -> dict[str, Any]:
    contract = read_json_object(contract_path)

    if (
        str(contract.get("contract_version"))
        != EXPECTED_META_ACTION_CONTRACT_VERSION
    ):
        raise ValueError(
            "unsupported Meta-action contract version: "
            f"{contract.get('contract_version')}"
        )

    if (
        contract.get("producer_step")
        != EXPECTED_META_ACTION_PRODUCER_STEP
    ):
        raise ValueError(
            "Meta-action contract producer_step must be "
            f"{EXPECTED_META_ACTION_PRODUCER_STEP}"
        )

    try:
        meta_data = meta_path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Meta-action file not found: {meta_path}"
        ) from exc

    actual_digest = hashlib.sha256(
        meta_data
    ).hexdigest()
    expected_digest = str(
        contract.get("meta_action_sha256", "")
    )

    if actual_digest != expected_digest:
        raise ValueError(
            "Meta-action SHA-256 mismatch: "
            f"{actual_digest} != {expected_digest}"
        )

    contract_count = int(
        contract["meta_action_count"]
    )
    contract_candidate_count = int(
        contract["candidate_anchor_count"]
    )

    if contract_count != candidate_count:
        raise ValueError(
            "Meta-action contract count mismatch: "
            f"{contract_count} != {candidate_count}"
        )

    if contract_candidate_count != candidate_count:
        raise ValueError(
            "Candidate Anchor contract count mismatch: "
            f"{contract_candidate_count} != {candidate_count}"
        )

    return contract


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
    parser.add_argument("--candidate-input", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--meta-input", type=Path, default=DEFAULT_META)
    parser.add_argument("--event-input", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--meta-action-contract",
        type=Path,
        default=DEFAULT_META_ACTION_CONTRACT,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_index(args.candidate_input)
    meta = read_index(args.meta_input)
    events = read_index(args.event_input)
    candidate_ids = set(candidates)
    if set(meta) != candidate_ids:
        raise ValueError("Candidate and Meta-action Anchor sets differ")
    if not set(events) <= candidate_ids:
        raise ValueError("Event records contain unknown Anchors")

    meta_action_contract = validate_meta_action_contract(
        args.meta_action_contract,
        args.meta_input,
        len(candidate_ids),
    )
    quotas = selection_quotas(len(candidate_ids))
    stable_lateral_quotas = quotas[
        "stable_lateral_quotas"
    ]
    stable_longitudinal_quotas = quotas[
        "stable_longitudinal_quotas"
    ]
    normal_baseline_quota = quotas[
        "normal_baseline_quota"
    ]

    selected: dict[str, dict[str, Any]] = {}

    def add(anchor_id: str, source: str, reason: str) -> None:
        if anchor_id not in selected:
            selected[anchor_id] = {
                "selection_sources": [],
                "selection_reasons": [],
            }
        if source not in selected[anchor_id]["selection_sources"]:
            selected[anchor_id]["selection_sources"].append(source)
        if reason not in selected[anchor_id]["selection_reasons"]:
            selected[anchor_id]["selection_reasons"].append(reason)

    # A. Keep every event Anchor.
    for anchor_id in events:
        add(anchor_id, "event_candidate", "contains_detected_keyframe_event")

    baseline_ids = candidate_ids - set(events)

    # B. Stable rare lateral states from the non-event, usable pool.
    for action, quota in stable_lateral_quotas.items():
        pool = [
            candidates[anchor_id]
            for anchor_id in baseline_ids
            if anchor_id not in selected
            and meta[anchor_id]["overall_quality_status"] == "usable"
            and meta[anchor_id]["lateral"]["action"] == action
        ]
        chosen = select_with_clip_preference(
            pool, quota=quota, bucket=f"stable_lateral:{action}"
        )
        for record in chosen:
            add(
                str(record["anchor_id"]),
                "balanced_stable_lateral",
                f"stable_lateral_{action}",
            )

    # C. Stable longitudinal states while keeping direction.
    for action, quota in stable_longitudinal_quotas.items():
        pool = [
            candidates[anchor_id]
            for anchor_id in baseline_ids
            if anchor_id not in selected
            and meta[anchor_id]["overall_quality_status"] == "usable"
            and meta[anchor_id]["lateral"]["action"] == "keep_direction"
            and meta[anchor_id]["longitudinal"]["action"] == action
        ]
        chosen = select_with_clip_preference(
            pool, quota=quota, bucket=f"stable_longitudinal:{action}"
        )
        if len(chosen) != quota:
            raise ValueError(f"insufficient pool for stable longitudinal {action}: {len(chosen)}")
        for record in chosen:
            add(
                str(record["anchor_id"]),
                "balanced_stable_longitudinal",
                f"stable_longitudinal_{action}",
            )

    # D. Normal straight and steady-speed baseline.
    pool = [
        candidates[anchor_id]
        for anchor_id in baseline_ids
        if anchor_id not in selected
        and meta[anchor_id]["overall_quality_status"] == "usable"
        and meta[anchor_id]["lateral"]["action"] == "keep_direction"
        and meta[anchor_id]["longitudinal"]["action"] == "maintain_speed"
    ]
    chosen = select_with_clip_preference(
        pool, quota=normal_baseline_quota, bucket="normal_driving_baseline"
    )
    if len(chosen) != normal_baseline_quota:
        raise ValueError(f"insufficient normal baseline pool: {len(chosen)}")
    for record in chosen:
        add(
            str(record["anchor_id"]),
            "normal_driving_baseline",
            "stable_keep_direction_and_maintain_speed",
        )

    output_records = []
    source_counts: Counter[str] = Counter()
    lateral_counts: Counter[str] = Counter()
    longitudinal_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for anchor_id in sorted(
        selected,
        key=lambda value: (
            str(candidates[value]["clip_id"]),
            int(candidates[value]["anchor_ns"]),
            value,
        ),
    ):
        candidate = candidates[anchor_id]
        meta_record = meta[anchor_id]
        selection = selected[anchor_id]
        event_record = events.get(anchor_id)
        record = {
            "keyframe_format_version": OUTPUT_FORMAT_VERSION,
            "selector_version": SELECTOR_VERSION,
            "rule_version": RULE_VERSION,
            "anchor_id": anchor_id,
            "clip_id": str(candidate["clip_id"]),
            "anchor_ns": int(candidate["anchor_ns"]),
            "future_horizon_ns": int(candidate["future_horizon_ns"]),
            "selection_sources": selection["selection_sources"],
            "selection_reasons": selection["selection_reasons"],
            "events": event_record["events"] if event_record else [],
            "meta_action": {
                "lateral": meta_record["lateral"]["action"],
                "longitudinal": meta_record["longitudinal"]["action"],
                "overall_quality_status": meta_record["overall_quality_status"],
            },
        }
        output_records.append(record)
        for source in record["selection_sources"]:
            source_counts[source] += 1
        lateral_counts[record["meta_action"]["lateral"]] += 1
        longitudinal_counts[record["meta_action"]["longitudinal"]] += 1
        quality_counts[record["meta_action"]["overall_quality_status"]] += 1

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output_records
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "keyframe_format_version": OUTPUT_FORMAT_VERSION,
        "selector_version": SELECTOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "keep_all_event_anchors": True,
            "stable_lateral_quotas": stable_lateral_quotas,
            "stable_longitudinal_quotas": stable_longitudinal_quotas,
            "normal_baseline_quota": normal_baseline_quota,
            "deterministic_selection": True,
            "prefer_distinct_clips": True,
            "temporal_suppression": False,
            "abnormal_scene_filtering": False,
        },
        "candidate_anchor_count": len(candidate_ids),
        "meta_action_contract": str(
            args.meta_action_contract
        ),
        "meta_action_sha256": str(
            meta_action_contract["meta_action_sha256"]
        ),
        "quota_policy": {
            "mode": "proportional_to_anchor_count",
            "reference_anchor_count": (
                quotas["reference_anchor_count"]
            ),
            "current_anchor_count": quotas["anchor_count"],
        },
        "event_anchor_count": len(events),
        "selected_keyframe_count": len(output_records),
        "selection_source_counts": dict(source_counts),
        "lateral_action_counts": dict(lateral_counts),
        "longitudinal_action_counts": dict(longitudinal_counts),
        "quality_counts": dict(quality_counts),
        "output_sha256": sha256,
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
    print("Candidate Anchors:", len(candidate_ids))
    print("Event Anchors retained:", len(events))
    print("Selected Keyframes:", len(output_records))
    print("Selection sources:", dict(source_counts))
    print("Lateral actions:", dict(lateral_counts))
    print("Longitudinal actions:", dict(longitudinal_counts))
    print("Quality:", dict(quality_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
