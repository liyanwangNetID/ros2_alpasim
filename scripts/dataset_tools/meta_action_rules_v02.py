#!/usr/bin/env python3
"""Frozen Meta-action rules for Step 4 production.

This module is self-contained and performs no file I/O. It directly applies
base Meta-action rules and the reviewed v0.2.2 lateral geometry arbitration.
"""
from __future__ import annotations

RULE_VERSION = "meta_action_rules_v0.2.1"
SHADOW_POLICY_VERSION = "0.2.2"
GENERATOR_VERSION = "0.2.1"
LABEL_FORMAT_VERSION = "0.2-draft"


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

STRAIGHT_MAX_TOTAL_YAW_DEG = 3.0

STRAIGHT_MAX_YAW_EXCURSION_DEG = 3.0

STRAIGHT_MAX_TOTAL_ABSOLUTE_YAW_DEG = 5.0

TURN_MIN_DIRECTIONAL_PROGRESS_DEG = 10.0

TURN_MIN_ABSOLUTE_RELATIVE_DEVIATION_DEG = 10.0

STOP_MAX_FINAL_SPEED_MPS = 0.3

STOP_MIN_CONTINUOUS_LOW_SPEED_SEC = 1.0

MOTION_MIN_SPEED_DELTA_MPS = 1.0

MOTION_MIN_HALF_MEAN_DELTA_MPS = 0.5

def straight_motion_override(lateral: Mapping[str, Any]) -> bool:
    return abs(math.degrees(float(lateral['ego_total_yaw_change_rad']))) <= STRAIGHT_MAX_TOTAL_YAW_DEG and math.degrees(float(lateral['ego_maximum_yaw_excursion_rad'])) <= STRAIGHT_MAX_YAW_EXCURSION_DEG and (math.degrees(float(lateral['ego_total_absolute_yaw_change_rad'])) <= STRAIGHT_MAX_TOTAL_ABSOLUTE_YAW_DEG)

def _truthy_ambiguity_flags(value: Any, path: str='') -> list[str]:
    """Return only explicitly truthy adjacent-evidence ambiguity flags.

    Field names alone are not evidence. For example, return_to_source=False
    must not make a normal lane change ambiguous.
    """
    ambiguity_keys = {'return_to_source', 'returns_to_source', 'opposite_adjacent', 'ambiguous', 'oscillation', 'oscillating'}
    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f'{path}.{key}' if path else str(key)
            normalized_key = str(key).lower()
            if normalized_key in ambiguity_keys and child is True:
                reasons.append(child_path)
            reasons.extend(_truthy_ambiguity_flags(child, child_path))
    elif isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        for index, child in enumerate(value):
            reasons.extend(_truthy_ambiguity_flags(child, f'{path}[{index}]'))
    return reasons

def lane_change_action(lateral: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Return a conservative adjacent-topology lane-change action."""
    counts = lateral.get('transition_relation_counts', {})
    left_count = int(counts.get('left_adjacent', 0))
    right_count = int(counts.get('right_adjacent', 0))
    ambiguity_flags = _truthy_ambiguity_flags(lateral.get('adjacent_transition_evidence', []))
    if ambiguity_flags:
        return ('unknown', ['ambiguous_adjacent_transition_evidence', *[f'truthy_flag:{flag}' for flag in sorted(set(ambiguity_flags))]])
    if left_count > 0 and right_count > 0:
        return ('unknown', ['both_left_and_right_adjacent_transitions'])
    if left_count > 0:
        return ('change_lane_left', ['left_adjacent_transition'])
    if right_count > 0:
        return ('change_lane_right', ['right_adjacent_transition'])
    if bool(lateral.get('contains_adjacent_transition')):
        return ('unknown', ['adjacent_transition_direction_unavailable'])
    return (None, [])

def unique_branch_direction(natural: Mapping[str, Any]) -> str | None:
    relations = set((str(value) for value in natural.get('reliable_directional_relations', [])))
    if relations == {'left_of_natural'}:
        return 'left'
    if relations == {'right_of_natural'}:
        return 'right'
    return None

def directed_degrees(value_rad: float, direction: str) -> float:
    value = math.degrees(float(value_rad))
    return value if direction == 'left' else -value

def branch_relative_metrics(lateral: Mapping[str, Any]) -> dict[str, Any] | None:
    natural = lateral.get('natural_corridor', {})
    direction = unique_branch_direction(natural)
    relative = natural.get('relative_heading', {})
    required = ('relative_first_half_heading_change_rad', 'relative_second_half_heading_change_rad', 'relative_total_heading_change_rad', 'relative_heading_start_rad', 'relative_heading_middle_rad', 'relative_heading_end_rad', 'maximum_absolute_relative_heading_rad')
    if direction is None or any((key not in relative for key in required)):
        return None
    first = directed_degrees(relative[required[0]], direction)
    second = directed_degrees(relative[required[1]], direction)
    total = directed_degrees(relative[required[2]], direction)
    start = directed_degrees(relative[required[3]], direction)
    middle = directed_degrees(relative[required[4]], direction)
    end = directed_degrees(relative[required[5]], direction)
    progress = max(0.0, first, second, total, middle - start, end - start, end - middle)
    return {'direction': direction, 'maximum_directional_progress_deg': progress, 'maximum_absolute_relative_heading_deg': math.degrees(float(relative[required[6]])), 'directed_first_half_change_deg': first, 'directed_second_half_change_deg': second, 'directed_total_change_deg': total, 'directed_end_heading_deg': end}

def classify_lateral(lateral: Mapping[str, Any]) -> dict[str, Any]:
    gate = lateral['lateral_quality_gate']
    if not bool(gate['passed']):
        return {'action': 'unknown', 'quality_status': 'unknown', 'reasons': ['lateral_quality_gate_failed'] + list(gate.get('reasons', [])), 'decision_stage': 'quality_gate', 'metrics': {}}
    lane_action, lane_reasons = lane_change_action(lateral)
    if lane_action is not None:
        return {'action': lane_action, 'quality_status': 'usable' if lane_action != 'unknown' else 'unknown', 'reasons': lane_reasons, 'decision_stage': 'lane_change_priority', 'metrics': {'transition_relation_counts': lateral.get('transition_relation_counts', {})}}
    straight_metrics = {'absolute_total_yaw_change_deg': abs(math.degrees(float(lateral['ego_total_yaw_change_rad']))), 'maximum_yaw_excursion_deg': math.degrees(float(lateral['ego_maximum_yaw_excursion_rad'])), 'total_absolute_yaw_change_deg': math.degrees(float(lateral['ego_total_absolute_yaw_change_rad']))}
    if straight_motion_override(lateral):
        return {'action': 'keep_direction', 'quality_status': 'usable', 'reasons': ['straight_motion_override'], 'decision_stage': 'straight_motion_override', 'metrics': straight_metrics}
    natural = lateral.get('natural_corridor', {})
    relative_metrics = branch_relative_metrics(lateral)
    if natural.get('turn_evidence_status') == 'directional_branch_observed' and relative_metrics is not None and (relative_metrics['maximum_directional_progress_deg'] >= TURN_MIN_DIRECTIONAL_PROGRESS_DEG) and (relative_metrics['maximum_absolute_relative_heading_deg'] >= TURN_MIN_ABSOLUTE_RELATIVE_DEVIATION_DEG):
        direction = str(relative_metrics['direction'])
        return {'action': f'turn_{direction}', 'quality_status': 'usable', 'reasons': [f'{direction}_of_natural_branch', 'relative_turn_thresholds_passed'], 'decision_stage': 'branch_relative_turn', 'metrics': relative_metrics}
    fallback_reasons = list(natural.get('fallback_reasons', []))
    if natural.get('turn_evidence_status') == 'fallback_keep_direction':
        fallback_reasons.insert(0, 'natural_branch_uncertain_or_unavailable')
    elif relative_metrics is not None:
        fallback_reasons.append('relative_turn_thresholds_not_met')
    else:
        fallback_reasons.append('no_reliable_directional_branch')
    return {'action': 'keep_direction', 'quality_status': 'usable', 'reasons': sorted(set(fallback_reasons)), 'decision_stage': 'keep_direction_fallback', 'metrics': relative_metrics or straight_metrics}

def classify_longitudinal(longitudinal: Mapping[str, Any]) -> dict[str, Any]:
    final_speed = float(longitudinal['final_speed_mps'])
    low_duration = float(longitudinal['longest_duration_below_0_3_mps_sec'])
    total_delta = float(longitudinal['speed_delta_mps'])
    half_delta = float(longitudinal['second_half_minus_first_half_mean_speed_mps'])
    metrics = {'speed_source_used': longitudinal.get('speed_source_used'), 'initial_speed_mps': longitudinal['initial_speed_mps'], 'final_speed_mps': final_speed, 'speed_delta_mps': total_delta, 'second_half_minus_first_half_mean_speed_mps': half_delta, 'longest_duration_below_0_3_mps_sec': low_duration, 'reported_speed_reliable': longitudinal.get('reported_speed_reliable')}
    if final_speed <= STOP_MAX_FINAL_SPEED_MPS and low_duration >= STOP_MIN_CONTINUOUS_LOW_SPEED_SEC:
        return {'action': 'stop', 'quality_status': 'usable', 'reasons': ['terminal_low_speed', 'sustained_low_speed'], 'decision_stage': 'stop_priority', 'metrics': metrics}
    if total_delta >= MOTION_MIN_SPEED_DELTA_MPS and half_delta >= MOTION_MIN_HALF_MEAN_DELTA_MPS:
        return {'action': 'accelerate', 'quality_status': 'usable', 'reasons': ['positive_total_speed_change', 'positive_half_mean_change'], 'decision_stage': 'consistent_speed_change', 'metrics': metrics}
    if total_delta <= -MOTION_MIN_SPEED_DELTA_MPS and half_delta <= -MOTION_MIN_HALF_MEAN_DELTA_MPS:
        return {'action': 'decelerate', 'quality_status': 'usable', 'reasons': ['negative_total_speed_change', 'negative_half_mean_change'], 'decision_stage': 'consistent_speed_change', 'metrics': metrics}
    if total_delta * half_delta < 0.0:
        return {'action': 'unknown', 'quality_status': 'unknown', 'reasons': ['mixed_or_conflicting_longitudinal_motion'], 'decision_stage': 'conflicting_speed_signs', 'metrics': metrics}
    return {'action': 'maintain_speed', 'quality_status': 'usable', 'reasons': ['speed_changes_below_action_thresholds'], 'decision_stage': 'maintain_speed_fallback', 'metrics': metrics}

from typing import Any, Mapping, Sequence

def normalize_interpretation(value: str) -> str | None:
    mapping = {'turn_left_candidate': 'turn_left', 'turn_right_candidate': 'turn_right', 'change_lane_left': 'change_lane_left', 'change_lane_right': 'change_lane_right', 'keep_direction': 'keep_direction'}
    return mapping.get(value)

def observed_proposal(items: Sequence[Mapping[str, Any]]) -> tuple[str | None, list[str]]:
    proposals: list[tuple[str, str]] = []
    for item in items:
        normalized = normalize_interpretation(str(item.get('interpretation', '')))
        if normalized is None:
            continue
        proposals.append((normalized, str(item.get('interpretation_reason', ''))))
    if not proposals:
        return (None, [])
    actions = {action for action, _ in proposals}
    if len(actions) != 1:
        return ('unknown', ['conflicting_observed_geometry_interpretations'])
    action = proposals[0][0]
    return (action, sorted({reason for _, reason in proposals if reason}))

def reviewed_in_progress_proposal(geometry: Mapping[str, Any], lateral_record: Mapping[str, Any] | None=None) -> tuple[str | None, list[str]]:
    actions: set[str] = set()
    if lateral_record is not None:
        lateral = lateral_record.get('lateral', lateral_record)
        ego_change = lateral.get('ego_total_yaw_change_rad')
        map_change = lateral.get('map_corridor_heading_change_rad')
        if not isinstance(ego_change, (int, float)):
            return (None, [])
        if not isinstance(map_change, (int, float)):
            return (None, [])
        residual_deg = abs(float(ego_change) - float(map_change)) * 180.0 / 3.141592653589793
        if residual_deg < 2.0:
            return (None, [])
    for item in geometry.get('in_progress_candidates', []):
        if not bool(item.get('candidate')):
            continue
        direction = str(item.get('direction', ''))
        if direction not in {'left', 'right'}:
            continue
        final_advantage = item.get('final_target_advantage_m')
        heading_progress = item.get('directional_heading_progress_deg')
        if not isinstance(final_advantage, (int, float)):
            continue
        if not isinstance(heading_progress, (int, float)):
            continue
        if float(final_advantage) < -2.0:
            continue
        if abs(float(heading_progress)) > 10.0:
            continue
        actions.add(f'change_lane_{direction}')
    if not actions:
        return (None, [])
    if len(actions) > 1:
        return ('unknown', ['conflicting_reviewed_in_progress_directions'])
    return (next(iter(actions)), ['reviewed_in_progress_lane_change_geometry', 'final_target_advantage_at_least_minus_2m', 'directional_heading_progress_at_most_10deg', 'absolute_ego_to_map_heading_residual_at_least_2deg'])

def reviewed_lane_change_to_turn_allowed(old_action: str, proposed_action: str, geometry: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if not old_action.startswith('change_lane_'):
        return (True, [])
    if not proposed_action.startswith('turn_'):
        return (True, [])
    expected_interpretation = proposed_action + '_candidate'
    for item in geometry.get('observed_adjacent_transitions', []):
        if str(item.get('interpretation', '')) != expected_interpretation:
            continue
        junction_level = str(item.get('junction_evidence_level', 'C'))
        residual = item.get('source_heading_residual', {})
        ego_heading_change_deg = residual.get('ego_heading_change_deg')
        if junction_level not in {'A', 'B'}:
            continue
        if not isinstance(ego_heading_change_deg, (int, float)):
            continue
        if abs(float(ego_heading_change_deg)) < 8.0:
            continue
        return (True, ['reviewed_lane_change_to_turn_geometry', 'junction_level_a_or_b', 'post_transition_ego_heading_change_at_least_8deg'])
    return (False, ['lane_change_to_turn_review_gate_not_met', 'requires_junction_level_a_or_b', 'requires_post_transition_ego_heading_change_at_least_8deg'])

def _section(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = record.get(name, record)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} section must be a mapping")
    return value


def classify_final_lateral(
    lateral_features: Mapping[str, Any],
    geometry_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen base and reviewed geometry rules in one direct pass."""
    lateral = _section(lateral_features, "lateral")
    base = classify_lateral(lateral)
    old_action = str(base["action"])

    if str(base.get("quality_status", "unknown")) != "usable":
        return base

    observed_action, observed_reasons = observed_proposal(
        geometry_features.get("observed_adjacent_transitions", [])
    )

    if observed_action is not None:
        if old_action.startswith("change_lane_") and observed_action == "keep_direction":
            return base

        allowed, gate_reasons = reviewed_lane_change_to_turn_allowed(
            old_action,
            observed_action,
            geometry_features,
        )
        if not allowed:
            return base

        if observed_action != old_action:
            result = dict(base)
            result["action"] = observed_action
            result["quality_status"] = (
                "usable" if observed_action != "unknown" else "unknown"
            )
            result["decision_stage"] = "reviewed_shadow_geometry_v0.2"
            result["reasons"] = observed_reasons + gate_reasons
            return result
        return base

    reviewed_action, reviewed_reasons = reviewed_in_progress_proposal(
        geometry_features,
        lateral_features,
    )
    if reviewed_action is not None and reviewed_action != old_action:
        result = dict(base)
        result["action"] = reviewed_action
        result["quality_status"] = (
            "usable" if reviewed_action != "unknown" else "unknown"
        )
        result["decision_stage"] = "reviewed_shadow_geometry_v0.2"
        result["reasons"] = reviewed_reasons
        return result

    return base


def classify_meta_action(
    lateral_features: Mapping[str, Any],
    longitudinal_features: Mapping[str, Any],
    geometry_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete frozen classification without reading or writing files."""
    lateral = classify_final_lateral(lateral_features, geometry_features)
    longitudinal = classify_longitudinal(
        _section(longitudinal_features, "longitudinal")
    )
    return {
        "lateral": lateral,
        "longitudinal": longitudinal,
        "joint_action": {
            "lateral": lateral["action"],
            "longitudinal": longitudinal["action"],
        },
        "overall_quality_status": (
            "usable"
            if lateral.get("quality_status") == "usable"
            and longitudinal.get("quality_status") == "usable"
            else "unknown"
        ),
    }


def make_meta_action_record(
    *,
    anchor_id: str,
    clip_id: str,
    anchor_ns: int,
    future_horizon_ns: int,
    lateral_features: Mapping[str, Any],
    longitudinal_features: Mapping[str, Any],
    geometry_features: Mapping[str, Any],
) -> dict[str, Any]:
    classification = classify_meta_action(
        lateral_features,
        longitudinal_features,
        geometry_features,
    )
    return {
        "label_format_version": LABEL_FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rule_version": RULE_VERSION,
        "anchor_id": anchor_id,
        "clip_id": clip_id,
        "anchor_ns": int(anchor_ns),
        "future_horizon_ns": int(future_horizon_ns),
        **classification,
    }

