"""Step 7I and 7J deterministic Scene-Fact feature rules v01."""
from __future__ import annotations

from typing import Any, Mapping

from step7.scene_fact_schema_v01 import (
    DISTANCE_TREND_CATEGORIES,
    RELATIVE_DISTANCE_CATEGORIES,
    RELATIVE_SPEED_CATEGORIES,
    ROAD_CONTEXT_TYPES,
)

NEAR_DISTANCE_MAX_M = 10.0
MEDIUM_DISTANCE_MAX_M = 30.0
STABLE_DISTANCE_RATE_MAX_MPS = 0.5
STATIONARY_SPEED_MAX_MPS = 0.5
SIMILAR_SPEED_DIFFERENCE_MAX_MPS = 1.0
INTERSECTION_WAIT_LINE_AT_MAX_M = 3.0
INTERSECTION_WAIT_LINE_APPROACH_MAX_M = 15.0
INTERSECTION_LANE_END_AT_MAX_M = 3.0


def classify_road_context(ego_road: Mapping[str, Any]) -> dict[str, Any]:
    """Classify current road context from current Ego map evidence only."""
    if ego_road["lane_match_status"] != "matched":
        result = {
            "type": "unknown",
            "proximity_status": "unknown",
            "lane_id": None,
            "nearest_wait_line_distance_m": None,
            "evidence": ["ego_lane_unmatched"],
        }
    else:
        wait_distance = ego_road["nearest_wait_line_distance_m"]
        evidence = list(ego_road["intersection_evidence"])
        remaining = None
        if (
            ego_road["lane_length_m"] is not None
            and ego_road["centerline_arc_length_m"] is not None
        ):
            remaining = max(
                0.0,
                float(ego_road["lane_length_m"])
                - float(ego_road["centerline_arc_length_m"]),
            )
        if wait_distance is not None and float(wait_distance) <= INTERSECTION_WAIT_LINE_AT_MAX_M:
            context_type = "intersection"
            proximity = "at"
        elif wait_distance is not None and float(wait_distance) <= INTERSECTION_WAIT_LINE_APPROACH_MAX_M:
            context_type = "intersection_approach"
            proximity = "approaching"
        elif bool(ego_road["has_intersection_evidence"]):
            if remaining is not None and remaining <= INTERSECTION_LANE_END_AT_MAX_M:
                context_type = "intersection"
                proximity = "at"
            else:
                context_type = "intersection_approach"
                proximity = "near"
        else:
            context_type = "lane_following"
            proximity = "none"
        result = {
            "type": context_type,
            "proximity_status": proximity,
            "lane_id": ego_road["lane_id"],
            "nearest_wait_line_distance_m": wait_distance,
            "lane_remaining_distance_m": remaining,
            "evidence": evidence,
        }
    if result["type"] not in ROAD_CONTEXT_TYPES:
        raise RuntimeError("unexpected road-context type")
    return result


def relative_distance_category(distance_m: float) -> str:
    distance = float(distance_m)
    if distance < 0.0:
        raise ValueError("distance_m must be non-negative")
    value = "near" if distance <= NEAR_DISTANCE_MAX_M else "medium" if distance <= MEDIUM_DISTANCE_MAX_M else "far"
    if value not in RELATIVE_DISTANCE_CATEGORIES:
        raise RuntimeError("unexpected relative-distance category")
    return value


def distance_trend(history_status: str, mean_distance_rate_mps: Any) -> str:
    if history_status != "usable" or mean_distance_rate_mps is None:
        return "uncertain"
    rate = float(mean_distance_rate_mps)
    value = (
        "stable_distance"
        if abs(rate) <= STABLE_DISTANCE_RATE_MAX_MPS
        else "approaching"
        if rate < 0.0
        else "receding"
    )
    if value not in DISTANCE_TREND_CATEGORIES:
        raise RuntimeError("unexpected distance-trend category")
    return value


def relative_speed_category(
    *, actor_speed_mps: float, ego_speed_mps: float
) -> str:
    actor_speed = float(actor_speed_mps)
    ego_speed = float(ego_speed_mps)
    if actor_speed <= STATIONARY_SPEED_MAX_MPS:
        value = "stationary"
    elif abs(actor_speed - ego_speed) <= SIMILAR_SPEED_DIFFERENCE_MAX_MPS:
        value = "similar_to_ego"
    elif actor_speed < ego_speed:
        value = "slower_than_ego"
    else:
        value = "faster_than_ego"
    if value not in RELATIVE_SPEED_CATEGORIES:
        raise RuntimeError("unexpected relative-speed category")
    return value


def assemble_selected_actor_feature(
    *, role: str, selected_role: Mapping[str, Any] | None,
    empty_reason: str | None, current_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if selected_role is None:
        if empty_reason is None:
            raise ValueError("empty Actor role requires an explicit reason")
        return {
            "presence_status": "not_present",
            "role": role,
            "absence_reason": empty_reason,
        }
    if current_geometry is None:
        raise ValueError("selected Actor requires current geometry")
    if str(current_geometry["track_id"]) != str(selected_role["track_id"]):
        raise ValueError("selected Actor and current geometry identities differ")
    history_status = str(selected_role["history_status"])
    return {
        "presence_status": "present",
        "role": role,
        "track_id": str(selected_role["track_id"]),
        "label_class": str(selected_role["label_class"]),
        "relative_position": str(selected_role["geometric_region"]),
        "relative_distance": relative_distance_category(
            float(selected_role["planar_distance_m"])
        ),
        "distance_m": float(selected_role["planar_distance_m"]),
        "relative_x_m": float(selected_role["relative_x_m"]),
        "relative_y_m": float(selected_role["relative_y_m"]),
        "distance_trend": distance_trend(
            history_status,
            selected_role["mean_distance_rate_mps"],
        ),
        "relative_speed": relative_speed_category(
            actor_speed_mps=float(current_geometry["actor_speed_mps"]),
            ego_speed_mps=float(current_geometry["ego_speed_mps"]),
        ),
        "actor_speed_mps": float(current_geometry["actor_speed_mps"]),
        "ego_speed_mps": float(current_geometry["ego_speed_mps"]),
        "observability_status": "candidate_visible",
        "visibility_policy_status": str(selected_role["visibility_policy_status"]),
        "history_status": history_status,
        "lane_match_status": str(selected_role["lane_match_status"]),
        "ego_lane_relation": str(selected_role["ego_lane_relation"]),
    }
