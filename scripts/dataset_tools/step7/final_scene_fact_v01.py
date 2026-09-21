"""Step 7L final Scene-Fact record mapping v01."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from step7.scene_fact_schema_v01 import (
    ACTOR_ROLE_KEYS,
    GENERATOR_VERSION,
    RULE_VERSION,
    SCENE_FACT_FORMAT_VERSION,
)


def _unique_reasons(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _line_proximities(road: Mapping[str, Any]) -> tuple[str, str]:
    """Map available wait-line evidence conservatively to stop/yield fields."""
    distance = road.get("nearest_wait_line_distance_m")
    if distance is None:
        return "none", "none"
    proximity = str(road["proximity_status"])
    # Step 7G wait-line type is not carried into 7K. Do not guess stop vs yield.
    return "unknown", "unknown"


def final_road_context(feature: Mapping[str, Any]) -> dict[str, Any]:
    road = feature["road_context"]
    context_type = str(road["type"])
    stop_proximity, yield_proximity = _line_proximities(road)
    reasons = list(road.get("evidence", ()))
    if context_type == "unknown":
        reasons.append("ego_road_context_unknown")
        quality = "unknown"
    else:
        quality = "usable"
    if road.get("nearest_wait_line_distance_m") is not None:
        reasons.append("wait_line_type_not_propagated_to_feature_layer")
    return {
        "type": context_type,
        "intersection_proximity": str(road["proximity_status"]),
        "stop_line_proximity": stop_proximity,
        "yield_line_proximity": yield_proximity,
        "quality_status": quality,
        "reasons": _unique_reasons(reasons),
    }


def final_actor_role(
    feature: Mapping[str, Any], *, visible_cameras: Sequence[str] | None
) -> dict[str, Any]:
    if feature["presence_status"] != "present":
        return {
            "presence_status": "not_present",
            "quality_status": "usable",
            "reasons": [str(feature["absence_reason"])],
        }
    reasons = []
    if feature["history_status"] != "usable":
        reasons.append(f"history_{feature['history_status']}")
    if feature["lane_match_status"] != "matched":
        reasons.append("lane_unmatched")
    cameras = sorted(set(str(value) for value in (visible_cameras or ())))
    if not cameras:
        raise ValueError("present Actor role requires at least one visible camera")
    quality = "usable" if not reasons else "unknown"
    return {
        "presence_status": "present",
        "track_id": str(feature["track_id"]),
        "actor_class": str(feature["label_class"]),
        "relative_position": str(feature["relative_position"]),
        "relative_distance": str(feature["relative_distance"]),
        "distance_trend": str(feature["distance_trend"]),
        "relative_speed_category": str(feature["relative_speed"]),
        "observability_status": str(feature["observability_status"]),
        "visible_in_cameras": cameras,
        "quality_status": quality,
        "reasons": reasons,
    }


def build_final_scene_fact_record(
    *, feature: Mapping[str, Any], visible_cameras_by_track_id: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    roles = {}
    for role in ACTOR_ROLE_KEYS:
        value = feature[role]
        track_id = None if value["presence_status"] != "present" else str(value["track_id"])
        roles[role] = final_actor_role(
            value,
            visible_cameras=None if track_id is None else visible_cameras_by_track_id.get(track_id),
        )
    return {
        "scene_fact_format_version": SCENE_FACT_FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rule_version": RULE_VERSION,
        "anchor_id": str(feature["anchor_id"]),
        "clip_id": str(feature["clip_id"]),
        "anchor_ns": int(feature["anchor_ns"]),
        "road_context": final_road_context(feature),
        **roles,
        "quality": {
            "status": str(feature["quality"]["status"]),
            "static_occlusion_evaluated": False,
            "reasons": _unique_reasons(feature["quality"]["reasons"]),
        },
    }
