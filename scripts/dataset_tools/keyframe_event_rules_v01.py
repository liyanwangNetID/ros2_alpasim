#!/usr/bin/env python3
"""Frozen Step 5A v0.1 keyframe event-candidate rules.

The module performs no file I/O. It detects 11 event types from consecutive
Anchors in the same Clip using frozen Step 4 labels and features.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

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
