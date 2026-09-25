from step6.generate_navigation_v01 import classify_navigation


def route(start_deg: float, final_x: float, final_y: float):
    import math
    return {
        "quality_status": "usable",
        "route_lookahead_distance_m": 80.0,
        "route": {
            "route_start_heading_rad": math.radians(start_deg),
            "route_signed_heading_change_rad": math.radians(-3.976),
            "final_local_x_m": final_x,
            "final_local_y_m": final_y,
        },
    }


def branch():
    return {
        "quality_status": "usable",
        "anchor_speed_mps": 5.055514957685494,
        "branch_context": {
            "first_intersection_evidence": {
                "lane_id": "200",
                "route_distance_m": 4.198947420260386,
                "evidence": ["merging"],
            },
            "first_observed_branch": None,
        },
    }


def test_clip_750_shape_is_unknown_without_using_future_action():
    result = classify_navigation(
        route(-34.42489133773965, 63.5991325378418, -47.84734344482422),
        branch(),
    )
    assert result["action"] == "unknown"
    assert result["quality_status"] == "unknown"
    assert result["text"] is None
    assert "strong_right_route_geometry_without_resolved_branch" in result["reasons"]


def test_ordinary_no_branch_intersection_remains_straight():
    result = classify_navigation(route(0.0, 80.0, 0.0), branch())
    assert result["action"] == "straight"
    assert result["quality_status"] == "usable"


def test_right_final_bearing_without_right_start_heading_is_not_enough():
    result = classify_navigation(route(0.0, 63.0, -48.0), branch())
    assert result["action"] == "straight"
