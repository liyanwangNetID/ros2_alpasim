#!/usr/bin/env python3
"""Conservative Step 6 v0.1.1 Navigation candidate rules."""
from __future__ import annotations

import math
from typing import Any, Mapping

RULE_VERSION = "navigation_rules_v0.1.4-candidate"
GENERATOR_VERSION = "0.1.4"
OUTPUT_FORMAT_VERSION = "0.1-draft"
UPCOMING_TIME_HORIZON_SEC = 10.0
MINIMUM_UPCOMING_DISTANCE_M = 15.0
MAXIMUM_UPCOMING_DISTANCE_M = 80.0
DIRECTION_GEOMETRY_MINIMUM_DEG = 5.0
ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG = 20.0
VALID_ACTIONS = {"straight", "left", "right", "unknown"}


def _unknown(*reasons: str) -> dict[str, Any]:
    return {
        "action": "unknown",
        "text": None,
        "quality_status": "unknown",
        "decision_source": "insufficient_route_or_map_evidence",
        "reasons": list(dict.fromkeys(reasons)),
    }


def dynamic_upcoming_distance_m(
    *, current_speed_mps: float, route_lookahead_distance_m: float,
) -> float:
    speed = max(0.0, float(current_speed_mps))
    route_limit = max(0.0, float(route_lookahead_distance_m))
    configured = max(
        MINIMUM_UPCOMING_DISTANCE_M,
        speed * UPCOMING_TIME_HORIZON_SEC,
    )
    return min(route_limit, MAXIMUM_UPCOMING_DISTANCE_M, configured)


def _is_upcoming_intersection(
    context: Mapping[str, Any], *, upcoming_distance_m: float,
) -> bool:
    evidence = context.get("first_intersection_evidence")
    if not isinstance(evidence, Mapping):
        return False
    distance = evidence.get("route_distance_m")
    if not isinstance(distance, (int, float)):
        return False
    reasons = {str(value) for value in evidence.get("evidence", [])}
    return bool(reasons & {"wait_line", "branching", "merging"}) and float(distance) <= upcoming_distance_m


def classify_navigation(
    route_features: Mapping[str, Any],
    branch_features: Mapping[str, Any],
    road_level_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if str(route_features.get("quality_status")) != "usable":
        return _unknown("navigation_route_geometry_not_usable")
    if str(branch_features.get("quality_status")) != "usable":
        return _unknown("navigation_branch_context_not_usable")

    context = branch_features.get("branch_context", {})
    route_geometry = route_features.get("route", {})
    speed = branch_features.get("anchor_speed_mps")
    lookahead = route_features.get("route_lookahead_distance_m")
    if not isinstance(context, Mapping) or not isinstance(route_geometry, Mapping):
        return _unknown("navigation_context_missing")
    if not isinstance(speed, (int, float)) or not isinstance(lookahead, (int, float)):
        return _unknown("anchor_speed_or_route_lookahead_missing")

    upcoming_distance = dynamic_upcoming_distance_m(
        current_speed_mps=float(speed),
        route_lookahead_distance_m=float(lookahead),
    )
    upcoming_intersection = _is_upcoming_intersection(
        context, upcoming_distance_m=upcoming_distance,
    )
    branch = context.get("first_observed_branch")

    if branch is None:
        return {
            "action": "straight",
            "text": "Continue straight through the upcoming intersection." if upcoming_intersection else "Continue along the road.",
            "quality_status": "usable",
            "decision_source": "route_no_observed_branch_upcoming_intersection" if upcoming_intersection else "route_no_observed_branch",
            "reasons": ["route_follows_observed_lane_sequence_without_branch_choice"],
        }
    if not isinstance(branch, Mapping):
        return _unknown("first_observed_branch_invalid")

    relation = str(branch.get("route_relation_to_natural", ""))
    reliability = str(branch.get("reliability_status", ""))
    branch_distance = branch.get("route_distance_m")
    if not isinstance(branch_distance, (int, float)):
        return _unknown("first_observed_branch_distance_missing")

    if float(branch_distance) > upcoming_distance:
        return {
            "action": "straight",
            "text": "Continue along the road.",
            "quality_status": "usable",
            "decision_source": "first_observed_branch_beyond_dynamic_preview_horizon",
            "reasons": [
                "route_branch_not_yet_within_navigation_preview_horizon"
            ],
        }

    if relation == "actual_successor_not_candidate":
        return _unknown("route_successor_not_in_branch_candidates")
    if relation == "natural_continuation":
        natural_successor = branch.get(
            "natural_successor_lane_id"
        )
        route_successor = branch.get(
            "route_successor_lane_id"
        )

        if (
            natural_successor is not None
            and route_successor is not None
            and str(natural_successor) != str(route_successor)
        ):
            return _unknown(
                "inconsistent_natural_continuation_successor_identity"
            )

        if not upcoming_intersection:
            return {
                "action": "straight",
                "text": "Continue along the road.",
                "quality_status": "usable",
                "decision_source": "route_natural_continuation",
                "reasons": ["route_selects_natural_successor", "no_upcoming_intersection"],
            }
        road_geometry = (
            road_level_features.get("road_level_route_geometry", {})
            if isinstance(road_level_features, Mapping)
            else {}
        )
        road_change = (
            road_geometry.get("route_road_level_heading_change_deg")
            if isinstance(road_geometry, Mapping)
            and str(road_geometry.get("status")) == "available"
            else None
        )
        if isinstance(road_change, (int, float)):
            if float(road_change) > ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG:
                action = "left"
            elif float(road_change) < -ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG:
                action = "right"
            else:
                if reliability != "reliable":
                    return _unknown(
                        "first_observed_branch_not_reliable",
                        "road_level_direction_below_intersection_threshold",
                    )
                action = "straight"
            return {
                "action": action,
                "text": (
                    "Continue straight through the upcoming intersection."
                    if action == "straight"
                    else f"Turn {action} at the upcoming intersection."
                ),
                "quality_status": "usable",
                "decision_source": f"road_level_natural_continuation_intersection_{action}",
                "reasons": [
                    "route_selects_natural_successor",
                    "upcoming_intersection_detected",
                    "road_level_direction_available",
                ],
            }
        if reliability != "reliable":
            return _unknown(
                "first_observed_branch_not_reliable",
                "road_level_direction_geometry_not_available",
            )
        return {
            "action": "straight",
            "text": "Continue straight through the upcoming intersection.",
            "quality_status": "usable",
            "decision_source": "route_natural_continuation_upcoming_intersection_legacy_fallback",
            "reasons": [
                "route_selects_natural_successor",
                "road_level_direction_geometry_not_available",
            ],
        }
    if reliability != "reliable":
        return _unknown("first_observed_branch_not_reliable")

    signed_change = route_geometry.get("route_signed_heading_change_rad")
    if not isinstance(signed_change, (int, float)):
        return _unknown("route_direction_geometry_missing")
    signed_deg = math.degrees(float(signed_change))

    if relation == "left_of_natural":
        if signed_deg < DIRECTION_GEOMETRY_MINIMUM_DEG:
            return _unknown(
                "left_branch_route_geometry_not_consistently_left"
                if signed_deg < -DIRECTION_GEOMETRY_MINIMUM_DEG
                else "left_branch_route_geometry_too_weak"
            )
        action = "left"
    elif relation == "right_of_natural":
        if signed_deg > -DIRECTION_GEOMETRY_MINIMUM_DEG:
            return _unknown(
                "right_branch_route_geometry_not_consistently_right"
                if signed_deg > DIRECTION_GEOMETRY_MINIMUM_DEG
                else "right_branch_route_geometry_too_weak"
            )
        action = "right"
    else:
        return _unknown("first_observed_branch_relation_unresolved")

    if upcoming_intersection:
        return {
            "action": action,
            "text": f"Turn {action} at the upcoming intersection.",
            "quality_status": "usable",
            "decision_source": f"route_intersection_{action}_branch",
            "reasons": ["upcoming_intersection_detected", f"route_selects_{action}_branch", "route_geometry_direction_consistent"],
        }
    return {
        "action": action,
        "text": f"Follow the {action} branch ahead.",
        "quality_status": "usable",
        "decision_source": f"route_{action}_branch",
        "reasons": [f"route_selects_{action}_branch", "route_geometry_direction_consistent"],
    }
