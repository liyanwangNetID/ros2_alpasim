"""Step 7K conservative feature-layer quality propagation v01."""
from __future__ import annotations
from typing import Any, Mapping
from step7.scene_fact_schema_v01 import ACTOR_ROLE_KEYS, QUALITY_STATUSES


def assemble_feature_quality(
    *, road_context: Mapping[str, Any], actor_roles: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    reasons = []
    if road_context["type"] == "unknown":
        reasons.append("ego_road_context_unknown")
    for role in ACTOR_ROLE_KEYS:
        value = actor_roles[role]
        if value["presence_status"] != "present":
            continue
        if value["history_status"] != "usable":
            reasons.append(f"{role}_history_{value['history_status']}")
        if value["lane_match_status"] != "matched":
            reasons.append(f"{role}_lane_unmatched")
    status = "usable" if not reasons else "unknown"
    if status not in QUALITY_STATUSES:
        raise RuntimeError("unexpected quality status")
    return {
        "status": status,
        "reasons": reasons,
        "static_occlusion_evaluated": False,
        "current_and_past_inputs_only": True,
    }
