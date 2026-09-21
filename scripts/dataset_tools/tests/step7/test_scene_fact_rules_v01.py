from step7.scene_fact_rules_v01 import (
    classify_road_context, distance_trend, relative_distance_category,
    relative_speed_category,
)


def ego(**updates):
    value = {
        "lane_match_status": "matched", "lane_id": "A",
        "nearest_wait_line_distance_m": None,
        "has_intersection_evidence": False, "intersection_evidence": [],
        "lane_length_m": 20.0, "centerline_arc_length_m": 5.0,
    }
    value.update(updates)
    return value


def test_road_context_rules():
    assert classify_road_context(ego())["type"] == "lane_following"
    assert classify_road_context(ego(nearest_wait_line_distance_m=10.0, has_intersection_evidence=True))["type"] == "intersection_approach"
    assert classify_road_context(ego(nearest_wait_line_distance_m=2.0, has_intersection_evidence=True))["type"] == "intersection"
    assert classify_road_context(ego(lane_match_status="unmatched"))["type"] == "unknown"


def test_actor_semantic_categories():
    assert relative_distance_category(10.0) == "near"
    assert relative_distance_category(20.0) == "medium"
    assert distance_trend("usable", -1.0) == "approaching"
    assert distance_trend("insufficient_span", None) == "uncertain"
    assert relative_speed_category(actor_speed_mps=0.1, ego_speed_mps=5.0) == "stationary"
    assert relative_speed_category(actor_speed_mps=7.0, ego_speed_mps=5.0) == "faster_than_ego"
