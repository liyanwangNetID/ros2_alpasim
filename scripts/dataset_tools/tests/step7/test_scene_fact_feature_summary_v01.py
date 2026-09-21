from step7.scene_fact_feature_summary_v01 import summarize_scene_fact_feature_rows

def test_summary_closes_one_row_per_keyframe():
    row = {"anchor_id":"a", "road_context":{"type":"lane_following"},
           "quality":{"status":"usable","reasons":[]},
           "lead_vehicle":{"presence_status":"not_present"},
           "left_nearby_vehicle":{"presence_status":"present"},
           "right_nearby_vehicle":{"presence_status":"not_present"}}
    result = summarize_scene_fact_feature_rows(keyframes=({"anchor_id":"a"},), rows=(row,))
    assert result["feature_row_count"] == 1
    assert result["role_presence_status_counts"]["left_nearby_vehicle"] == {"present":1}
