#!/usr/bin/env python3
"""Generate final rule-based Meta-action labels for selected Anchors.

Inputs:
- lateral_action_features_v0.3.jsonl, feature format 0.3.2-draft;
- meta_action_features_v0.2.jsonl, pose/time-derived longitudinal speed.

The generator applies the frozen v0.1 policy and never reads or modifies raw
clips. Lateral and longitudinal actions are generated independently, with
explicit quality status, reasons, metrics, and rule provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

GENERATOR_VERSION = "0.1.1"
LABEL_FORMAT_VERSION = "0.1-draft"
RULE_VERSION = "meta_action_rules_v0.1"
ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_LATERAL_INPUT = (
    ROOT / "annotations/v0.1-draft/intermediate/lateral_action_features_v0.3.jsonl"
)
DEFAULT_LONGITUDINAL_INPUT = (
    ROOT / "annotations/v0.1-draft/intermediate/meta_action_features_v0.2.jsonl"
)
DEFAULT_OUTPUT = ROOT / "annotations/v0.1-draft/meta_actions_v0.1.jsonl"
DEFAULT_SUMMARY = ROOT / "reports/meta_action_generation_summary_v0.1.json"

# Frozen lateral thresholds.
STRAIGHT_MAX_TOTAL_YAW_DEG = 3.0
STRAIGHT_MAX_YAW_EXCURSION_DEG = 3.0
STRAIGHT_MAX_TOTAL_ABSOLUTE_YAW_DEG = 5.0
TURN_MIN_DIRECTIONAL_PROGRESS_DEG = 10.0
TURN_MIN_ABSOLUTE_RELATIVE_DEVIATION_DEG = 10.0

# Frozen longitudinal thresholds.
STOP_MAX_FINAL_SPEED_MPS = 0.3
STOP_MIN_CONTINUOUS_LOW_SPEED_SEC = 1.0
MOTION_MIN_SPEED_DELTA_MPS = 1.0
MOTION_MIN_HALF_MEAN_DELTA_MPS = 0.5

LATERAL_LABELS = {
    "unknown",
    "keep_direction",
    "change_lane_left",
    "change_lane_right",
    "turn_left",
    "turn_right",
}
LONGITUDINAL_LABELS = {
    "unknown",
    "stop",
    "accelerate",
    "decelerate",
    "maintain_speed",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def index_unique(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        anchor_id = str(row["anchor_id"])
        if anchor_id in result:
            raise ValueError(f"duplicate anchor_id in {source}: {anchor_id}")
        result[anchor_id] = row
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


def straight_motion_override(lateral: Mapping[str, Any]) -> bool:
    return (
        abs(math.degrees(float(lateral["ego_total_yaw_change_rad"])))
        <= STRAIGHT_MAX_TOTAL_YAW_DEG
        and math.degrees(float(lateral["ego_maximum_yaw_excursion_rad"]))
        <= STRAIGHT_MAX_YAW_EXCURSION_DEG
        and math.degrees(float(lateral["ego_total_absolute_yaw_change_rad"]))
        <= STRAIGHT_MAX_TOTAL_ABSOLUTE_YAW_DEG
    )


def _truthy_ambiguity_flags(value: Any, path: str = "") -> list[str]:
    """Return only explicitly truthy adjacent-evidence ambiguity flags.

    Field names alone are not evidence. For example, return_to_source=False
    must not make a normal lane change ambiguous.
    """
    ambiguity_keys = {
        "return_to_source",
        "returns_to_source",
        "opposite_adjacent",
        "ambiguous",
        "oscillation",
        "oscillating",
    }
    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            normalized_key = str(key).lower()
            if normalized_key in ambiguity_keys and child is True:
                reasons.append(child_path)
            reasons.extend(_truthy_ambiguity_flags(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            reasons.extend(_truthy_ambiguity_flags(child, f"{path}[{index}]"))
    return reasons


def lane_change_action(lateral: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Return a conservative adjacent-topology lane-change action."""
    counts = lateral.get("transition_relation_counts", {})
    left_count = int(counts.get("left_adjacent", 0))
    right_count = int(counts.get("right_adjacent", 0))
    ambiguity_flags = _truthy_ambiguity_flags(
        lateral.get("adjacent_transition_evidence", [])
    )
    if ambiguity_flags:
        return "unknown", [
            "ambiguous_adjacent_transition_evidence",
            *[f"truthy_flag:{flag}" for flag in sorted(set(ambiguity_flags))],
        ]
    if left_count > 0 and right_count > 0:
        return "unknown", ["both_left_and_right_adjacent_transitions"]
    if left_count > 0:
        return "change_lane_left", ["left_adjacent_transition"]
    if right_count > 0:
        return "change_lane_right", ["right_adjacent_transition"]
    if bool(lateral.get("contains_adjacent_transition")):
        return "unknown", ["adjacent_transition_direction_unavailable"]
    return None, []


def unique_branch_direction(natural: Mapping[str, Any]) -> str | None:
    relations = set(
        str(value) for value in natural.get("reliable_directional_relations", [])
    )
    if relations == {"left_of_natural"}:
        return "left"
    if relations == {"right_of_natural"}:
        return "right"
    return None


def directed_degrees(value_rad: float, direction: str) -> float:
    value = math.degrees(float(value_rad))
    return value if direction == "left" else -value


def branch_relative_metrics(lateral: Mapping[str, Any]) -> dict[str, Any] | None:
    natural = lateral.get("natural_corridor", {})
    direction = unique_branch_direction(natural)
    relative = natural.get("relative_heading", {})
    required = (
        "relative_first_half_heading_change_rad",
        "relative_second_half_heading_change_rad",
        "relative_total_heading_change_rad",
        "relative_heading_start_rad",
        "relative_heading_middle_rad",
        "relative_heading_end_rad",
        "maximum_absolute_relative_heading_rad",
    )
    if direction is None or any(key not in relative for key in required):
        return None
    first = directed_degrees(relative[required[0]], direction)
    second = directed_degrees(relative[required[1]], direction)
    total = directed_degrees(relative[required[2]], direction)
    start = directed_degrees(relative[required[3]], direction)
    middle = directed_degrees(relative[required[4]], direction)
    end = directed_degrees(relative[required[5]], direction)
    progress = max(
        0.0,
        first,
        second,
        total,
        middle - start,
        end - start,
        end - middle,
    )
    return {
        "direction": direction,
        "maximum_directional_progress_deg": progress,
        "maximum_absolute_relative_heading_deg": math.degrees(
            float(relative[required[6]])
        ),
        "directed_first_half_change_deg": first,
        "directed_second_half_change_deg": second,
        "directed_total_change_deg": total,
        "directed_end_heading_deg": end,
    }


def classify_lateral(lateral: Mapping[str, Any]) -> dict[str, Any]:
    gate = lateral["lateral_quality_gate"]
    if not bool(gate["passed"]):
        return {
            "action": "unknown",
            "quality_status": "unknown",
            "reasons": ["lateral_quality_gate_failed"] + list(gate.get("reasons", [])),
            "decision_stage": "quality_gate",
            "metrics": {},
        }

    lane_action, lane_reasons = lane_change_action(lateral)
    if lane_action is not None:
        return {
            "action": lane_action,
            "quality_status": "usable" if lane_action != "unknown" else "unknown",
            "reasons": lane_reasons,
            "decision_stage": "lane_change_priority",
            "metrics": {
                "transition_relation_counts": lateral.get(
                    "transition_relation_counts", {}
                )
            },
        }

    straight_metrics = {
        "absolute_total_yaw_change_deg": abs(
            math.degrees(float(lateral["ego_total_yaw_change_rad"]))
        ),
        "maximum_yaw_excursion_deg": math.degrees(
            float(lateral["ego_maximum_yaw_excursion_rad"])
        ),
        "total_absolute_yaw_change_deg": math.degrees(
            float(lateral["ego_total_absolute_yaw_change_rad"])
        ),
    }
    if straight_motion_override(lateral):
        return {
            "action": "keep_direction",
            "quality_status": "usable",
            "reasons": ["straight_motion_override"],
            "decision_stage": "straight_motion_override",
            "metrics": straight_metrics,
        }

    natural = lateral.get("natural_corridor", {})
    relative_metrics = branch_relative_metrics(lateral)
    if (
        natural.get("turn_evidence_status") == "directional_branch_observed"
        and relative_metrics is not None
        and relative_metrics["maximum_directional_progress_deg"]
        >= TURN_MIN_DIRECTIONAL_PROGRESS_DEG
        and relative_metrics["maximum_absolute_relative_heading_deg"]
        >= TURN_MIN_ABSOLUTE_RELATIVE_DEVIATION_DEG
    ):
        direction = str(relative_metrics["direction"])
        return {
            "action": f"turn_{direction}",
            "quality_status": "usable",
            "reasons": [f"{direction}_of_natural_branch", "relative_turn_thresholds_passed"],
            "decision_stage": "branch_relative_turn",
            "metrics": relative_metrics,
        }

    fallback_reasons = list(natural.get("fallback_reasons", []))
    if natural.get("turn_evidence_status") == "fallback_keep_direction":
        fallback_reasons.insert(0, "natural_branch_uncertain_or_unavailable")
    elif relative_metrics is not None:
        fallback_reasons.append("relative_turn_thresholds_not_met")
    else:
        fallback_reasons.append("no_reliable_directional_branch")
    return {
        "action": "keep_direction",
        "quality_status": "usable",
        "reasons": sorted(set(fallback_reasons)),
        "decision_stage": "keep_direction_fallback",
        "metrics": relative_metrics or straight_metrics,
    }


def classify_longitudinal(longitudinal: Mapping[str, Any]) -> dict[str, Any]:
    final_speed = float(longitudinal["final_speed_mps"])
    low_duration = float(longitudinal["longest_duration_below_0_3_mps_sec"])
    total_delta = float(longitudinal["speed_delta_mps"])
    half_delta = float(
        longitudinal["second_half_minus_first_half_mean_speed_mps"]
    )
    metrics = {
        "speed_source_used": longitudinal.get("speed_source_used"),
        "initial_speed_mps": longitudinal["initial_speed_mps"],
        "final_speed_mps": final_speed,
        "speed_delta_mps": total_delta,
        "second_half_minus_first_half_mean_speed_mps": half_delta,
        "longest_duration_below_0_3_mps_sec": low_duration,
        "reported_speed_reliable": longitudinal.get("reported_speed_reliable"),
    }
    if (
        final_speed <= STOP_MAX_FINAL_SPEED_MPS
        and low_duration >= STOP_MIN_CONTINUOUS_LOW_SPEED_SEC
    ):
        return {
            "action": "stop",
            "quality_status": "usable",
            "reasons": ["terminal_low_speed", "sustained_low_speed"],
            "decision_stage": "stop_priority",
            "metrics": metrics,
        }
    if (
        total_delta >= MOTION_MIN_SPEED_DELTA_MPS
        and half_delta >= MOTION_MIN_HALF_MEAN_DELTA_MPS
    ):
        return {
            "action": "accelerate",
            "quality_status": "usable",
            "reasons": ["positive_total_speed_change", "positive_half_mean_change"],
            "decision_stage": "consistent_speed_change",
            "metrics": metrics,
        }
    if (
        total_delta <= -MOTION_MIN_SPEED_DELTA_MPS
        and half_delta <= -MOTION_MIN_HALF_MEAN_DELTA_MPS
    ):
        return {
            "action": "decelerate",
            "quality_status": "usable",
            "reasons": ["negative_total_speed_change", "negative_half_mean_change"],
            "decision_stage": "consistent_speed_change",
            "metrics": metrics,
        }
    if total_delta * half_delta < 0.0:
        return {
            "action": "unknown",
            "quality_status": "unknown",
            "reasons": ["mixed_or_conflicting_longitudinal_motion"],
            "decision_stage": "conflicting_speed_signs",
            "metrics": metrics,
        }
    return {
        "action": "maintain_speed",
        "quality_status": "usable",
        "reasons": ["speed_changes_below_action_thresholds"],
        "decision_stage": "maintain_speed_fallback",
        "metrics": metrics,
    }


def validate_output(record: Mapping[str, Any]) -> None:
    lateral_action = record["lateral"]["action"]
    longitudinal_action = record["longitudinal"]["action"]
    if lateral_action not in LATERAL_LABELS:
        raise ValueError(f"invalid lateral action: {lateral_action}")
    if longitudinal_action not in LONGITUDINAL_LABELS:
        raise ValueError(f"invalid longitudinal action: {longitudinal_action}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lateral-input", type=Path, default=DEFAULT_LATERAL_INPUT)
    parser.add_argument(
        "--longitudinal-input", type=Path, default=DEFAULT_LONGITUDINAL_INPUT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lateral_rows = read_jsonl(args.lateral_input.expanduser().resolve())
    longitudinal_rows = read_jsonl(args.longitudinal_input.expanduser().resolve())
    lateral_by_id = index_unique(lateral_rows, "lateral input")
    longitudinal_by_id = index_unique(longitudinal_rows, "longitudinal input")
    lateral_ids = set(lateral_by_id)
    longitudinal_ids = set(longitudinal_by_id)
    if lateral_ids != longitudinal_ids:
        missing_longitudinal = sorted(lateral_ids - longitudinal_ids)
        missing_lateral = sorted(longitudinal_ids - lateral_ids)
        raise ValueError(
            "input anchor sets differ; "
            f"missing longitudinal={missing_longitudinal[:10]}, "
            f"missing lateral={missing_lateral[:10]}"
        )

    output: list[dict[str, Any]] = []
    for lateral_source in lateral_rows:
        anchor_id = str(lateral_source["anchor_id"])
        longitudinal_source = longitudinal_by_id[anchor_id]
        if str(lateral_source["clip_id"]) != str(longitudinal_source["clip_id"]):
            raise ValueError(f"clip_id mismatch for {anchor_id}")
        if int(lateral_source["anchor_ns"]) != int(longitudinal_source["anchor_ns"]):
            raise ValueError(f"anchor_ns mismatch for {anchor_id}")
        record = {
            "label_format_version": LABEL_FORMAT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "rule_version": RULE_VERSION,
            "anchor_id": anchor_id,
            "clip_id": str(lateral_source["clip_id"]),
            "anchor_ns": int(lateral_source["anchor_ns"]),
            "future_horizon_ns": int(lateral_source["future_horizon_ns"]),
            "lateral": classify_lateral(lateral_source["lateral"]),
            "longitudinal": classify_longitudinal(
                longitudinal_source["longitudinal"]
            ),
            "source_versions": {
                "lateral_feature_format_version": lateral_source.get(
                    "feature_format_version"
                ),
                "longitudinal_feature_format_version": longitudinal_source.get(
                    "feature_format_version"
                ),
            },
        }
        record["joint_action"] = {
            "lateral": record["lateral"]["action"],
            "longitudinal": record["longitudinal"]["action"],
        }
        record["overall_quality_status"] = (
            "usable"
            if record["lateral"]["quality_status"] == "usable"
            and record["longitudinal"]["quality_status"] == "usable"
            else "unknown"
        )
        validate_output(record)
        output.append(record)

    lateral_counts = Counter(item["lateral"]["action"] for item in output)
    longitudinal_counts = Counter(
        item["longitudinal"]["action"] for item in output
    )
    joint_counts = Counter(
        (
            item["lateral"]["action"],
            item["longitudinal"]["action"],
        )
        for item in output
    )
    summary = {
        "label_format_version": LABEL_FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lateral_input": str(args.lateral_input),
        "longitudinal_input": str(args.longitudinal_input),
        "anchor_count": len(output),
        "lateral_action_counts": dict(lateral_counts),
        "longitudinal_action_counts": dict(longitudinal_counts),
        "overall_quality_counts": dict(
            Counter(item["overall_quality_status"] for item in output)
        ),
        "joint_action_counts": [
            {
                "lateral": lateral,
                "longitudinal": longitudinal,
                "count": count,
            }
            for (lateral, longitudinal), count in sorted(joint_counts.items())
        ],
        "frozen_rules": {
            "lateral_priority": [
                "quality_gate",
                "lane_change",
                "straight_motion_override",
                "branch_relative_turn",
                "keep_direction_fallback",
            ],
            "straight_motion_override": {
                "maximum_total_yaw_change_deg": STRAIGHT_MAX_TOTAL_YAW_DEG,
                "maximum_yaw_excursion_deg": STRAIGHT_MAX_YAW_EXCURSION_DEG,
                "maximum_total_absolute_yaw_change_deg": (
                    STRAIGHT_MAX_TOTAL_ABSOLUTE_YAW_DEG
                ),
            },
            "branch_relative_turn": {
                "minimum_directional_progress_deg": (
                    TURN_MIN_DIRECTIONAL_PROGRESS_DEG
                ),
                "minimum_absolute_relative_deviation_deg": (
                    TURN_MIN_ABSOLUTE_RELATIVE_DEVIATION_DEG
                ),
            },
            "longitudinal": {
                "speed_source": "pose_time_derived",
                "stop_maximum_final_speed_mps": STOP_MAX_FINAL_SPEED_MPS,
                "stop_minimum_continuous_low_speed_sec": (
                    STOP_MIN_CONTINUOUS_LOW_SPEED_SEC
                ),
                "motion_minimum_speed_delta_mps": MOTION_MIN_SPEED_DELTA_MPS,
                "motion_minimum_half_mean_delta_mps": (
                    MOTION_MIN_HALF_MEAN_DELTA_MPS
                ),
                "conflicting_signs_action": "unknown",
            },
        },
    }

    output_text = "".join(
        json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n"
        for item in output
    )
    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        args.force,
    )
    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("Label SHA-256:", hashlib.sha256(output_text.encode()).hexdigest())
    print("Anchor count:", len(output))
    print("Lateral counts:", dict(lateral_counts))
    print("Longitudinal counts:", dict(longitudinal_counts))
    print("Overall quality:", summary["overall_quality_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
