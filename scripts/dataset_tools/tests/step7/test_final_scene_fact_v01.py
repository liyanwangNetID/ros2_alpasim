import pytest
from step7.final_scene_fact_v01 import build_final_scene_fact_record


def feature(present=True):
    absent = {"presence_status":"not_present","absence_reason":"none"}
    actor = {"presence_status":"present","track_id":"1","label_class":"automobile",
             "relative_position":"front","relative_distance":"near","distance_trend":"stable_distance",
             "relative_speed":"similar_to_ego","observability_status":"candidate_visible",
             "history_status":"usable","lane_match_status":"matched"}
    return {"anchor_id":"a","clip_id":"test_clip_001","anchor_ns":1,
            "road_context":{"type":"lane_following","proximity_status":"none","evidence":[],
                            "nearest_wait_line_distance_m":None},
            "lead_vehicle":actor if present else absent,
            "left_nearby_vehicle":absent,"right_nearby_vehicle":absent,
            "quality":{"status":"usable","reasons":[]}}


def test_final_mapping_is_schema_shaped():
    row=build_final_scene_fact_record(feature=feature(),visible_cameras_by_track_id={"1":["front_wide"]})
    assert row["lead_vehicle"]["visible_in_cameras"] == ["front_wide"]
    assert row["road_context"]["stop_line_proximity"] == "none"
    assert row["quality"]["static_occlusion_evaluated"] is False


def test_present_actor_requires_camera_evidence():
    with pytest.raises(ValueError, match="visible camera"):
        build_final_scene_fact_record(feature=feature(),visible_cameras_by_track_id={})



def test_visible_cameras_are_sorted_and_deduplicated():
    row = build_final_scene_fact_record(
        feature=feature(),
        visible_cameras_by_track_id={
            "1": ["front_wide", "cross_left", "front_wide"],
        },
    )
    assert row["lead_vehicle"]["visible_in_cameras"] == [
        "cross_left",
        "front_wide",
    ]
