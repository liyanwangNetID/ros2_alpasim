#!/usr/bin/env python3
"""Step 5A v0.1 event-candidate detector."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    REPORT_ROOT,
)

EVENT_FORMAT_VERSION = "0.1-draft"


DETECTOR_VERSION = "0.1.0"


RULE_VERSION = "keyframe_event_rules_v0.1"


TURN_ACTIONS = {"turn_left", "turn_right"}


LANE_CHANGE_ACTIONS = {"change_lane_left", "change_lane_right"}


LATERAL_ACTIONS = {
    "keep_direction", "turn_left", "turn_right",
    "change_lane_left", "change_lane_right", "unknown",
}


LONGITUDINAL_ACTIONS = {
    "maintain_speed", "accelerate", "decelerate", "stop", "unknown",
}


def _event(
    event_type: str,
    *,
    source: str,
    confidence: str,
    reasons: list[str],
    metrics: Mapping[str, Any] | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": event_type,
        "confidence": confidence,
        "source": source,
        "reasons": reasons,
        "metrics": dict(metrics or {}),
    }
    if direction is not None:
        result["direction"] = direction
    return result


def _action(record: Mapping[str, Any], section: str) -> str:
    return str(record.get(section, {}).get("action", "unknown"))


def _section(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = record.get(name, record)
    return value if isinstance(value, Mapping) else {}


def _lateral(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return _section(record, "lateral")


def _topology(lateral_record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _lateral(lateral_record).get("topology", {})
    return value if isinstance(value, Mapping) else {}


def _junction_level(lateral_record: Mapping[str, Any]) -> str:
    return str(_topology(lateral_record).get("junction_evidence_level", "C"))


def _junction_entry_state(lateral_record: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    lateral = _lateral(lateral_record)
    topology = _topology(lateral_record)
    lane_sequence = [str(value) for value in lateral.get("lane_sequence", [])]
    markers = set()
    for key in (
        "wait_line_lane_ids",
        "branching_lane_ids",
        "boundary_predecessor_branch_lane_ids",
    ):
        markers.update(str(value) for value in topology.get(key, []))

    marker_indices = [index for index, lane_id in enumerate(lane_sequence) if lane_id in markers]
    entered = any(index < len(lane_sequence) - 1 for index in marker_indices)
    return entered, {
        "lane_sequence": lane_sequence,
        "junction_marker_lane_ids": sorted(markers),
    }


def _reviewed_in_progress_events(
    lateral_record: Mapping[str, Any],
    geometry_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lateral = _lateral(lateral_record)
    ego_change = lateral.get("ego_total_yaw_change_rad")
    map_change = lateral.get("map_corridor_heading_change_rad")
    if not isinstance(ego_change, (int, float)):
        return []
    if not isinstance(map_change, (int, float)):
        return []

    residual_deg = abs(float(ego_change) - float(map_change)) * 180.0 / math.pi
    if residual_deg < 2.0:
        return []

    events: list[dict[str, Any]] = []
    for item in geometry_record.get("in_progress_candidates", []):
        if not bool(item.get("candidate")):
            continue
        direction = str(item.get("direction", ""))
        final_advantage = item.get("final_target_advantage_m")
        heading_progress = item.get("directional_heading_progress_deg")
        if direction not in {"left", "right"}:
            continue
        if not isinstance(final_advantage, (int, float)):
            continue
        if not isinstance(heading_progress, (int, float)):
            continue
        if float(final_advantage) < -2.0:
            continue
        if abs(float(heading_progress)) > 10.0:
            continue

        events.append(_event(
            "lane_change_in_progress",
            source="reviewed_lane_change_geometry",
            confidence="high",
            direction=direction,
            reasons=[
                "in_progress_candidate_passed",
                "final_target_advantage_at_least_minus_2m",
                "directional_heading_progress_at_most_10deg",
                "absolute_ego_to_map_heading_residual_at_least_2deg",
            ],
            metrics={
                "source_lane_id": item.get("source_lane_id"),
                "target_lane_id": item.get("target_lane_id"),
                "final_target_advantage_m": float(final_advantage),
                "directional_heading_progress_deg": float(heading_progress),
                "ego_to_map_heading_residual_deg": residual_deg,
            },
        ))
    return events


def detect_anchor_events(
    *,
    previous_meta: Mapping[str, Any] | None,
    current_meta: Mapping[str, Any],
    previous_lateral_features: Mapping[str, Any] | None,
    current_lateral_features: Mapping[str, Any],
    previous_longitudinal_features: Mapping[str, Any] | None,
    current_longitudinal_features: Mapping[str, Any],
    current_geometry_features: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Detect all v0.1 events for one Anchor.

    Start/transition events require a previous Anchor. Feature-state events may
    be emitted for the first Anchor if their current-window evidence is valid.
    """
    events: list[dict[str, Any]] = []
    current_lat = _action(current_meta, "lateral")
    current_lon = _action(current_meta, "longitudinal")

    if current_lat not in LATERAL_ACTIONS:
        raise ValueError(f"unsupported lateral action: {current_lat}")
    if current_lon not in LONGITUDINAL_ACTIONS:
        raise ValueError(f"unsupported longitudinal action: {current_lon}")

    if previous_meta is not None:
        previous_lat = _action(previous_meta, "lateral")
        previous_lon = _action(previous_meta, "longitudinal")

        if previous_lat != current_lat:
            confidence = "low" if "unknown" in {previous_lat, current_lat} else "high"
            events.append(_event(
                "lateral_action_transition",
                source="meta_action_transition",
                confidence=confidence,
                reasons=[f"{previous_lat}_to_{current_lat}"],
                metrics={"previous_action": previous_lat, "current_action": current_lat},
            ))

        if previous_lon != current_lon:
            confidence = "low" if "unknown" in {previous_lon, current_lon} else "high"
            events.append(_event(
                "longitudinal_action_transition",
                source="meta_action_transition",
                confidence=confidence,
                reasons=[f"{previous_lon}_to_{current_lon}"],
                metrics={"previous_action": previous_lon, "current_action": current_lon},
            ))

        if previous_lat not in TURN_ACTIONS and current_lat in TURN_ACTIONS:
            events.append(_event(
                "turn_start",
                source="meta_action_transition",
                confidence="high",
                direction=current_lat.removeprefix("turn_"),
                reasons=[f"{previous_lat}_to_{current_lat}"],
            ))

        if previous_lat not in LANE_CHANGE_ACTIONS and current_lat in LANE_CHANGE_ACTIONS:
            events.append(_event(
                "lane_change_start",
                source="meta_action_transition",
                confidence="high",
                direction=current_lat.removeprefix("change_lane_"),
                reasons=[f"{previous_lat}_to_{current_lat}"],
            ))

        if previous_lon != "accelerate" and current_lon == "accelerate":
            confidence = "low" if previous_lon == "unknown" else "high"
            events.append(_event(
                "acceleration_start",
                source="meta_action_transition",
                confidence=confidence,
                reasons=[f"{previous_lon}_to_accelerate"],
            ))

        if previous_lon not in {"decelerate", "stop"} and current_lon == "decelerate":
            confidence = "low" if previous_lon == "unknown" else "high"
            events.append(_event(
                "deceleration_start",
                source="meta_action_transition",
                confidence=confidence,
                reasons=[f"{previous_lon}_to_decelerate"],
            ))

        if previous_lon != "stop" and current_lon == "stop":
            events.append(_event(
                "stop_start",
                source="meta_action_transition_and_pose_speed",
                confidence="high",
                reasons=[f"{previous_lon}_to_stop"],
            ))

        if previous_lon == "stop" and current_lon in {"accelerate", "maintain_speed"}:
            longitudinal = _section(current_longitudinal_features, "longitudinal")
            final_speed = longitudinal.get("final_speed_mps")
            speed_delta = longitudinal.get("speed_delta_mps")
            if (
                isinstance(final_speed, (int, float))
                and isinstance(speed_delta, (int, float))
                and float(final_speed) > 0.3
                and float(speed_delta) > 0.0
            ):
                events.append(_event(
                    "restart",
                    source="meta_action_transition_and_pose_speed",
                    confidence="high",
                    reasons=[f"stop_to_{current_lon}", "final_speed_above_stop_range", "positive_speed_recovery"],
                    metrics={"final_speed_mps": float(final_speed), "speed_delta_mps": float(speed_delta)},
                ))

        previous_level = _junction_level(previous_lateral_features or {})
        current_level = _junction_level(current_lateral_features)
        if previous_level == "C" and current_level in {"A", "B"}:
            events.append(_event(
                "junction_approach",
                source="lateral_topology_transition",
                confidence="high",
                reasons=[f"junction_level_{previous_level.lower()}_to_{current_level.lower()}"],
                metrics={"previous_junction_level": previous_level, "current_junction_level": current_level},
            ))

        previous_entered, _ = _junction_entry_state(previous_lateral_features or {})
        current_entered, entry_metrics = _junction_entry_state(current_lateral_features)
        if not previous_entered and current_entered:
            events.append(_event(
                "junction_entry",
                source="future_lane_sequence_topology",
                confidence="high",
                reasons=["junction_marker_lane_followed_by_downstream_lane"],
                metrics=entry_metrics,
            ))

    events.extend(_reviewed_in_progress_events(
        current_lateral_features,
        current_geometry_features,
    ))

    # Stable deterministic ordering supports reproducible JSONL output.
    events.sort(key=lambda item: (item["type"], item.get("direction", ""), item["source"]))
    return events


ROOT = ALPASIM_DATA_ROOT
DEFAULT_CANDIDATE_INPUT = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
DEFAULT_META_INPUT = (
    ANNOTATION_ROOT / "meta_actions_v0.2.jsonl"
)
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
    INTERMEDIATE_ROOT / "keyframe_event_candidates_v0.1.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "keyframe_event_candidate_summary_v0.1.json"
)


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
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
    parser.add_argument("--candidate-input", type=Path, default=DEFAULT_CANDIDATE_INPUT)
    parser.add_argument("--meta-input", type=Path, default=DEFAULT_META_INPUT)
    parser.add_argument("--lateral-input", type=Path, default=DEFAULT_LATERAL_INPUT)
    parser.add_argument("--longitudinal-input", type=Path, default=DEFAULT_LONGITUDINAL_INPUT)
    parser.add_argument("--geometry-input", type=Path, default=DEFAULT_GEOMETRY_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_jsonl_index(args.candidate_input)
    meta = read_jsonl_index(args.meta_input)
    lateral = read_jsonl_index(args.lateral_input)
    longitudinal = read_jsonl_index(args.longitudinal_input)
    geometry = read_jsonl_index(args.geometry_input)

    expected = set(candidates)
    for name, index in (
        ("meta", meta), ("lateral", lateral),
        ("longitudinal", longitudinal), ("geometry", geometry),
    ):
        if set(index) != expected:
            raise ValueError(
                f"{name} Anchor set differs; missing={sorted(expected-set(index))[:10]}, "
                f"extra={sorted(set(index)-expected)[:10]}"
            )

    clips: dict[str, list[str]] = defaultdict(list)
    for anchor_id, record in candidates.items():
        clips[str(record["clip_id"])].append(anchor_id)

    output_records: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    event_clips: dict[str, set[str]] = defaultdict(set)
    multi_event_anchor_count = 0

    for clip_id in sorted(clips):
        anchor_ids = sorted(clips[clip_id], key=lambda value: int(candidates[value]["anchor_ns"]))
        previous_id: str | None = None

        for anchor_id in anchor_ids:
            current_candidate = candidates[anchor_id]
            require_identity(
                anchor_id,
                current_candidate,
                meta[anchor_id], lateral[anchor_id], longitudinal[anchor_id], geometry[anchor_id],
            )

            events = detect_anchor_events(
                previous_meta=meta[previous_id] if previous_id else None,
                current_meta=meta[anchor_id],
                previous_lateral_features=lateral[previous_id] if previous_id else None,
                current_lateral_features=lateral[anchor_id],
                previous_longitudinal_features=longitudinal[previous_id] if previous_id else None,
                current_longitudinal_features=longitudinal[anchor_id],
                current_geometry_features=geometry[anchor_id],
            )

            if events:
                record = {
                    "event_format_version": EVENT_FORMAT_VERSION,
                    "detector_version": DETECTOR_VERSION,
                    "rule_version": RULE_VERSION,
                    "anchor_id": anchor_id,
                    "clip_id": clip_id,
                    "anchor_ns": int(current_candidate["anchor_ns"]),
                    "future_horizon_ns": int(current_candidate["future_horizon_ns"]),
                    "events": events,
                }
                output_records.append(record)
                if len(events) > 1:
                    multi_event_anchor_count += 1
                for event in events:
                    event_type = str(event["type"])
                    event_counts[event_type] += 1
                    confidence_counts[str(event["confidence"])] += 1
                    event_clips[event_type].add(clip_id)

            previous_id = anchor_id

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output_records
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "event_format_version": EVENT_FORMAT_VERSION,
        "detector_version": DETECTOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_anchor_count": len(expected),
        "event_anchor_count": len(output_records),
        "no_event_anchor_count": len(expected) - len(output_records),
        "multi_event_anchor_count": multi_event_anchor_count,
        "event_counts": dict(event_counts),
        "confidence_counts": dict(confidence_counts),
        "event_clip_counts": {key: len(value) for key, value in sorted(event_clips.items())},
        "output_sha256": sha256,
        "excluded_event_families": ["navigation_direction_change", "actor_interaction_change"],
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
    print("Total Anchors:", len(expected))
    print("Event Anchors:", len(output_records))
    print("No-event Anchors:", len(expected) - len(output_records))
    print("Multi-event Anchors:", multi_event_anchor_count)
    print("Event counts:", dict(event_counts))
    print("Confidence counts:", dict(confidence_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
