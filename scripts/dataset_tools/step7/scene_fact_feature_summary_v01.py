"""Step 7K Scene-Fact feature summary v01."""
from collections import Counter
from step7.scene_fact_schema_v01 import ACTOR_ROLE_KEYS


def summarize_scene_fact_feature_rows(*, keyframes, rows):
    anchors = [str(row["anchor_id"]) for row in keyframes]
    row_anchors = [str(row["anchor_id"]) for row in rows]
    if len(anchors) != len(set(anchors)):
        raise ValueError("Keyframe Anchor ids must be unique")
    if len(rows) != len(anchors) or set(row_anchors) != set(anchors):
        raise ValueError("feature rows must contain exactly one row per Keyframe")
    role_presence = {
        role: dict(sorted(Counter(row[role]["presence_status"] for row in rows).items()))
        for role in ACTOR_ROLE_KEYS
    }
    return {
        "schema_version": "step7k-scene-fact-features-summary-v01",
        "keyframe_count": len(anchors),
        "feature_row_count": len(rows),
        "road_context_type_counts": dict(sorted(Counter(
            row["road_context"]["type"] for row in rows
        ).items())),
        "quality_status_counts": dict(sorted(Counter(
            row["quality"]["status"] for row in rows
        ).items())),
        "quality_reason_counts": dict(sorted(Counter(
            reason for row in rows for reason in row["quality"]["reasons"]
        ).items())),
        "role_presence_status_counts": role_presence,
    }
