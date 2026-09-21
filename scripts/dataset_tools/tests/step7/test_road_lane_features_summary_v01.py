from step7.road_lane_features_summary_v01 import summarize_road_feature_rows
def test_summary_counts_statuses_and_relations():
 r=summarize_road_feature_rows(keyframes=({"anchor_id":"a"},),ego_rows=({"anchor_id":"a","lane_match_status":"matched","has_intersection_evidence":True},),actor_rows=({"anchor_id":"a","track_id":"1","lane_match_status":"matched","ego_lane_relation":"same","has_intersection_evidence":False},))
 assert r["keyframe_count"]==1 and r["actor_ego_lane_relation_counts"]=={"same":1}
