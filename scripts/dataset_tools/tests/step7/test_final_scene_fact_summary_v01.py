from step7.final_scene_fact_summary_v01 import summarize_final_scene_facts

def test_summary_requires_one_row_per_keyframe():
    row={"anchor_id":"a","road_context":{"type":"lane_following"},"quality":{"status":"usable"},
         "lead_vehicle":{"presence_status":"not_present"},"left_nearby_vehicle":{"presence_status":"not_present"},
         "right_nearby_vehicle":{"presence_status":"not_present"}}
    result=summarize_final_scene_facts(keyframes=({"anchor_id":"a"},),rows=(row,))
    assert result["scene_fact_row_count"] == 1
    assert result["schema_validation_error_count"] == 0
