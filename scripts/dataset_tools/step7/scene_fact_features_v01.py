"""Step 7K one-row-per-Keyframe Scene-Fact feature assembly v01."""
from __future__ import annotations
from typing import Any, Mapping
from step7.scene_fact_quality_v01 import assemble_feature_quality
from step7.scene_fact_rules_v01 import (
    assemble_selected_actor_feature,
    classify_road_context,
)
from step7.scene_fact_schema_v01 import ACTOR_ROLE_KEYS, FEATURE_FORMAT_VERSION


def assemble_scene_fact_feature_row(
    *, keyframe: Mapping[str, Any], ego_road: Mapping[str, Any],
    role_selection: Mapping[str, Any], current_geometry_by_track_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_id = str(keyframe["anchor_id"])
    for name, row in (("ego_road", ego_road), ("role_selection", role_selection)):
        if str(row["anchor_id"]) != anchor_id:
            raise ValueError(f"{name} Anchor does not match Keyframe")
    role_features = {}
    for role in ACTOR_ROLE_KEYS:
        selected = role_selection["roles"][role]
        geometry = (
            None if selected is None
            else current_geometry_by_track_id.get(str(selected["track_id"]))
        )
        role_features[role] = assemble_selected_actor_feature(
            role=role,
            selected_role=selected,
            empty_reason=role_selection["empty_role_reasons"][role],
            current_geometry=geometry,
        )
    road_context = classify_road_context(ego_road)
    quality = assemble_feature_quality(
        road_context=road_context,
        actor_roles=role_features,
    )
    return {
        "schema_version": "step7k-scene-fact-features-v01",
        "feature_format_version": FEATURE_FORMAT_VERSION,
        "anchor_id": anchor_id,
        "clip_id": str(keyframe["clip_id"]),
        "anchor_ns": int(keyframe["anchor_ns"]),
        "road_context": road_context,
        **role_features,
        "quality": quality,
        "future_actor_data_used": False,
        "future_ego_data_used": False,
        "planner_output_used": False,
        "meta_action_used": False,
    }
