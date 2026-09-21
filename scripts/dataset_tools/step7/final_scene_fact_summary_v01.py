"""Step 7L final Scene-Fact summary."""
from collections import Counter
from step7.scene_fact_schema_v01 import ACTOR_ROLE_KEYS


def summarize_final_scene_facts(*, keyframes, rows):
    anchors = [str(row["anchor_id"]) for row in keyframes]
    row_anchors = [str(row["anchor_id"]) for row in rows]
    if len(anchors) != len(set(anchors)):
        raise ValueError("Keyframe Anchor ids must be unique")
    if len(rows) != len(anchors) or set(row_anchors) != set(anchors):
        raise ValueError("final Scene Facts must contain exactly one row per Keyframe")
    return {
        "schema_version": "step7l-final-scene-facts-summary-v01",
        "keyframe_count": len(anchors),
        "scene_fact_row_count": len(rows),
        "road_context_type_counts": dict(sorted(Counter(
            row["road_context"]["type"] for row in rows
        ).items())),
        "quality_status_counts": dict(sorted(Counter(
            row["quality"]["status"] for row in rows
        ).items())),
        "role_presence_status_counts": {
            role: dict(sorted(Counter(row[role]["presence_status"] for row in rows).items()))
            for role in ACTOR_ROLE_KEYS
        },
        "schema_validation_error_count": 0,
    }
