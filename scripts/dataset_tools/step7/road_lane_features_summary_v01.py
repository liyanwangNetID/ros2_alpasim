"""Step 7G road-feature summary."""
from collections import Counter

def summarize_road_feature_rows(*, keyframes, actor_rows, ego_rows):
    anchors=[str(x["anchor_id"]) for x in keyframes]
    if len(anchors)!=len(set(anchors)): raise ValueError("keyframe anchors must be unique")
    if len(ego_rows)!=len(anchors): raise ValueError("exactly one Ego road row per Keyframe is required")
    actor_ids=[(str(x["anchor_id"]),str(x["track_id"])) for x in actor_rows]
    if len(actor_ids)!=len(set(actor_ids)): raise ValueError("Actor road identities must be unique")
    return {
      "schema_version":"step7g-road-features-summary-v01",
      "keyframe_count":len(anchors),"ego_row_count":len(ego_rows),"actor_row_count":len(actor_rows),
      "ego_match_status_counts":dict(sorted(Counter(x["lane_match_status"] for x in ego_rows).items())),
      "actor_match_status_counts":dict(sorted(Counter(x["lane_match_status"] for x in actor_rows).items())),
      "actor_ego_lane_relation_counts":dict(sorted(Counter(x["ego_lane_relation"] for x in actor_rows).items())),
      "ego_intersection_evidence_count":sum(bool(x["has_intersection_evidence"]) for x in ego_rows),
      "actor_intersection_evidence_count":sum(bool(x["has_intersection_evidence"]) for x in actor_rows),
    }
