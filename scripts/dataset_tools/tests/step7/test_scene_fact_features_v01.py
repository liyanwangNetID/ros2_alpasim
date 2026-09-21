from step7.scene_fact_features_v01 import assemble_scene_fact_feature_row


def test_row_assembly_preserves_absence_and_quality():
    keyframe = {"anchor_id": "a", "clip_id": "c", "anchor_ns": 1}
    ego = {"anchor_id": "a", "lane_match_status": "unmatched", "lane_id": None,
           "nearest_wait_line_distance_m": None, "has_intersection_evidence": None,
           "intersection_evidence": [], "lane_length_m": None,
           "centerline_arc_length_m": None}
    roles = {"anchor_id": "a", "roles": {"lead_vehicle": None,
             "left_nearby_vehicle": None, "right_nearby_vehicle": None},
             "empty_role_reasons": {"lead_vehicle": "no_current_actors",
             "left_nearby_vehicle": "no_current_actors",
             "right_nearby_vehicle": "no_current_actors"}}
    row = assemble_scene_fact_feature_row(keyframe=keyframe, ego_road=ego,
        role_selection=roles, current_geometry_by_track_id={})
    assert row["road_context"]["type"] == "unknown"
    assert row["lead_vehicle"]["presence_status"] == "not_present"
    assert row["quality"]["status"] == "unknown"
