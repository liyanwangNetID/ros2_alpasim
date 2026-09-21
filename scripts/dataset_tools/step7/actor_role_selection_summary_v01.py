"""Step 7H Actor-role selection summary."""
from collections import Counter
from step7.scene_fact_schema_v01 import ACTOR_ROLE_KEYS


def summarize_actor_role_rows(*, keyframes, rows):
    anchors = [str(row["anchor_id"]) for row in keyframes]
    if len(anchors) != len(set(anchors)):
        raise ValueError("Keyframe Anchor ids must be unique")
    row_anchors = [str(row["anchor_id"]) for row in rows]
    if len(rows) != len(anchors) or set(row_anchors) != set(anchors):
        raise ValueError("role rows must contain exactly one row per Keyframe")
    role_counts = {}
    empty_reasons = {}
    selected_pairs = []
    for role in ACTOR_ROLE_KEYS:
        role_counts[role] = sum(row["roles"][role] is not None for row in rows)
        empty_reasons[role] = dict(sorted(Counter(
            row["empty_role_reasons"][role]
            for row in rows
            if row["empty_role_reasons"][role] is not None
        ).items()))
    for row in rows:
        selected = [
            value["track_id"] for value in row["roles"].values() if value is not None
        ]
        if len(selected) != len(set(selected)):
            raise ValueError("one Actor occupies multiple roles in one Keyframe")
        selected_pairs.extend((row["anchor_id"], track_id) for track_id in selected)
    return {
        "schema_version": "step7h-actor-role-selection-summary-v01",
        "keyframe_count": len(anchors),
        "role_row_count": len(rows),
        "selected_role_counts": role_counts,
        "empty_role_reason_counts": empty_reasons,
        "selected_actor_role_assignment_count": len(selected_pairs),
        "role_conflict_count": 0,
    }
