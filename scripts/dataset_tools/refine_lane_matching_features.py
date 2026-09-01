#!/usr/bin/env python3
"""Refine lane-matching features with adjacent-transition persistence and gates.

This Step 4 tool reads lane_matching_features_v0.1.jsonl and adds:
- one evidence record for every left/right adjacent transition;
- exact target-segment and successor-corridor durations from future GT stamps;
- return-to-source and competing-direction diagnostics;
- a conservative lateral-label quality gate.

It does not rerun lane matching. Raw clips are opened only to recover the exact
future trajectory timestamps needed for duration calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from clip_reader import DrivingClipReader, stamp_mapping_to_ns  # noqa: E402
from project_paths import (  # noqa: E402
    ALPASIM_DATA_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

SCRIPT_VERSION = "0.2.0"
FEATURE_FORMAT_VERSION = "0.2-draft"
DEFAULT_ROOT = ALPASIM_DATA_ROOT
DEFAULT_INPUT = (
    INTERMEDIATE_ROOT / "lane_matching_features_v0.1.jsonl"
)
DEFAULT_OUTPUT = (
    INTERMEDIATE_ROOT / "lane_matching_features_v0.2.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "lane_matching_refinement_summary_v0.2.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object")
            records.append(value)
    return records


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


def returns_to_previous_lane(sequence: Sequence[str]) -> bool:
    previous_positions: dict[str, int] = {}
    for index, lane_id in enumerate(sequence):
        if lane_id in previous_positions and index - previous_positions[lane_id] >= 2:
            return True
        previous_positions[lane_id] = index
    return False


def conservative_quality_gate(record: Mapping[str, Any]) -> dict[str, Any]:
    relations = {
        str(item.get("relation", ""))
        for item in record.get("transitions", [])
        if isinstance(item, Mapping)
    }
    sequence = [str(value) for value in record.get("compressed_lane_sequence", [])]
    reasons: list[str] = []
    if float(record.get("matched_fraction", 0.0)) < 0.8:
        reasons.append("matched_fraction_below_0_8")
    if "unrelated" in relations:
        reasons.append("contains_unrelated_transition")
    if "predecessor" in relations:
        reasons.append("contains_predecessor_transition")
    if "left_adjacent" in relations and "right_adjacent" in relations:
        reasons.append("contains_both_adjacent_directions")
    if returns_to_previous_lane(sequence):
        reasons.append("returns_to_previous_lane")
    return {
        "status": "lateral_unknown" if reasons else "usable",
        "passed": not reasons,
        "reasons": reasons,
        "policy_version": "conservative_v0.1",
    }


def _transition_target_indexes(
    transitions: Sequence[Mapping[str, Any]],
) -> list[int]:
    return [int(item["target_point_index"]) for item in transitions]


def adjacent_transition_evidence(
    record: Mapping[str, Any],
    point_stamps_ns: Sequence[int],
) -> list[dict[str, Any]]:
    """Compute persistence for each adjacent transition.

    A target corridor begins at the adjacent target lane and continues through
    any immediately following successor transitions. It ends before the next
    non-successor transition or at the trajectory horizon.
    """
    transitions = [
        item for item in record.get("transitions", [])
        if isinstance(item, Mapping)
    ]
    sequence = [str(value) for value in record.get("compressed_lane_sequence", [])]
    if len(sequence) != len(transitions) + 1 and transitions:
        raise ValueError("lane sequence and transitions are inconsistent")
    total_points = len(point_stamps_ns)
    if total_points == 0:
        return []
    target_indexes = _transition_target_indexes(transitions)
    evidence: list[dict[str, Any]] = []

    for transition_index, transition in enumerate(transitions):
        relation = str(transition.get("relation", ""))
        if relation not in ("left_adjacent", "right_adjacent"):
            continue

        start_index = int(transition["target_point_index"])
        if not 0 <= start_index < total_points:
            raise ValueError("adjacent target index is outside trajectory")

        corridor_last_transition = transition_index
        while corridor_last_transition + 1 < len(transitions):
            next_relation = str(
                transitions[corridor_last_transition + 1].get("relation", "")
            )
            if next_relation != "successor":
                break
            corridor_last_transition += 1

        if corridor_last_transition + 1 < len(transitions):
            corridor_end_index = (
                int(transitions[corridor_last_transition + 1]["target_point_index"]) - 1
            )
            terminated_by_relation = str(
                transitions[corridor_last_transition + 1].get("relation", "")
            )
        else:
            corridor_end_index = total_points - 1
            terminated_by_relation = None

        direct_segment_end_index = (
            int(transitions[transition_index + 1]["target_point_index"]) - 1
            if transition_index + 1 < len(transitions)
            else total_points - 1
        )
        direct_segment_end_index = max(start_index, direct_segment_end_index)
        corridor_end_index = max(start_index, corridor_end_index)

        source_lane_id = str(transition["source_lane_id"])
        target_lane_id = str(transition["target_lane_id"])
        later_sequence = sequence[transition_index + 2 :]
        return_to_source = source_lane_id in later_sequence
        opposite_relation = (
            "right_adjacent" if relation == "left_adjacent" else "left_adjacent"
        )
        later_relations = [
            str(item.get("relation", ""))
            for item in transitions[transition_index + 1 :]
        ]

        evidence.append(
            {
                "transition_index": transition_index,
                "direction": "left" if relation == "left_adjacent" else "right",
                "relation": relation,
                "source_lane_id": source_lane_id,
                "target_lane_id": target_lane_id,
                "start_point_index": start_index,
                "direct_target_end_point_index": direct_segment_end_index,
                "direct_target_point_count": direct_segment_end_index - start_index + 1,
                "direct_target_duration_sec": (
                    point_stamps_ns[direct_segment_end_index]
                    - point_stamps_ns[start_index]
                ) / 1e9,
                "corridor_end_point_index": corridor_end_index,
                "corridor_point_count": corridor_end_index - start_index + 1,
                "corridor_duration_sec": (
                    point_stamps_ns[corridor_end_index]
                    - point_stamps_ns[start_index]
                ) / 1e9,
                "successor_transitions_after_adjacent": (
                    corridor_last_transition - transition_index
                ),
                "corridor_reaches_horizon": corridor_end_index == total_points - 1,
                "terminated_by_relation": terminated_by_relation,
                "returns_to_source_lane": return_to_source,
                "contains_opposite_adjacent_after": opposite_relation in later_relations,
            }
        )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.feature_input.expanduser().resolve())
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_clip[str(record["clip_id"])].append(record)

    refined: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    gate_counts: dict[str, int] = defaultdict(int)
    adjacent_evidence_count = 0

    for clip_index, (clip_id, clip_records) in enumerate(by_clip.items(), start=1):
        reader: DrivingClipReader | None = None
        for record in clip_records:
            updated = dict(record)
            updated["feature_format_version"] = FEATURE_FORMAT_VERSION
            updated["refiner_version"] = SCRIPT_VERSION
            gate = conservative_quality_gate(record)
            updated["lateral_quality_gate"] = gate
            gate_counts[gate["status"]] += 1

            transitions = record.get("transitions", [])
            has_adjacent = any(
                isinstance(item, Mapping)
                and item.get("relation") in ("left_adjacent", "right_adjacent")
                for item in transitions
            )
            updated["adjacent_transition_evidence"] = []
            if has_adjacent:
                try:
                    if reader is None:
                        reader = DrivingClipReader(args.dataset_root / clip_id)
                    horizon_ns = int(record.get("future_horizon_ns", 3_000_000_000))
                    future = reader.get_future_ego_trajectory(
                        int(record["anchor_ns"]), horizon_ns=horizon_ns
                    )
                    if future is None:
                        raise RuntimeError("future trajectory unavailable")
                    stamps = [
                        stamp_mapping_to_ns(point["stamp"])
                        for point in future.points
                    ]
                    evidence = adjacent_transition_evidence(record, stamps)
                    updated["adjacent_transition_evidence"] = evidence
                    adjacent_evidence_count += len(evidence)
                except Exception as exc:
                    errors.append(
                        {
                            "anchor_id": str(record["anchor_id"]),
                            "clip_id": clip_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            refined.append(updated)

        if clip_index == 1 or clip_index % 50 == 0 or clip_index == len(by_clip):
            print(f"Processed {clip_index}/{len(by_clip)} clips: {clip_id}")

    feature_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in refined
    )
    summary = {
        "feature_format_version": FEATURE_FORMAT_VERSION,
        "refiner_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_feature_file": str(args.feature_input),
        "record_count": len(refined),
        "adjacent_transition_evidence_count": adjacent_evidence_count,
        "lateral_quality_gate_counts": dict(gate_counts),
        "refinement_error_count": len(errors),
        "refinement_errors": errors,
    }
    atomic_write(args.feature_output, feature_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        args.force,
    )
    print("Feature output:", args.feature_output)
    print("Summary output:", args.summary_output)
    print("Feature SHA-256:", hashlib.sha256(feature_text.encode()).hexdigest())
    print("Records:", len(refined))
    print("Adjacent evidence records:", adjacent_evidence_count)
    print("Quality gate counts:", dict(gate_counts))
    print("Refinement errors:", len(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
