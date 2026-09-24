"""Tests for frozen Step 7 Actor-list selection."""
from __future__ import annotations
import pytest
from step7.actor_roles import ActorRoleCandidate, actor_selection_horizons, select_actor_roles, summarize_actor_role_rows


def candidate(track, x, y, relation="unknown", visible="shadow_visible", label="automobile", actor_speed=5.0, ego_speed=5.0):
    return ActorRoleCandidate(
        track_id=track, label_class=label,
        geometry={"relative_x_m": x, "relative_y_m": y, "planar_distance_m": (x*x+y*y)**0.5, "geometric_region": "front", "actor_speed_mps": actor_speed, "ego_speed_mps": ego_speed, "relative_longitudinal_speed_mps": actor_speed-ego_speed},
        visibility={"shadow_status": visible, "winning_cell_count": 3},
        history={"history_status": "usable", "mean_distance_rate_mps": -1.0},
        road={"ego_lane_relation": relation, "lane_match_status": "matched", "lane_direction_relation": "same_direction"},
    )


def test_horizons_are_split():
    value = actor_selection_horizons(20.0)
    assert value == {"reference_ego_speed_mps": 20.0, "forward_horizon_m": 200.0, "side_forward_horizon_m": 100.0, "rear_horizon_m": 40.0}


def test_lists_are_bounded_mutually_exclusive_and_ranked():
    roles = select_actor_roles(candidates=(candidate("lead", 10, 0), candidate("left", 2, 3), candidate("right", -2, -3)))
    assert [actor["track_id"] for actor in roles["lead_actors"]] == ["lead"]
    assert [actor["track_id"] for actor in roles["left_nearby_actors"]] == ["left"]
    assert [actor["track_id"] for actor in roles["right_nearby_actors"]] == ["right"]


def test_person_and_rider_can_be_lead():
    roles = select_actor_roles(candidates=(candidate("person", 5, 0, label="person"), candidate("rider", 8, 0.2, relation="unrelated", label="rider")))
    assert [actor["track_id"] for actor in roles["lead_actors"]] == ["person", "rider"]


def test_side_person_is_capped_at_thirty_metres():
    roles = select_actor_roles(candidates=(candidate("near", 25, -4, label="person"), candidate("far", 31, -4, label="person"), candidate("vehicle", 50, -4)), reference_ego_speed_mps=12.0)
    assert [actor["track_id"] for actor in roles["right_nearby_actors"]] == ["near", "vehicle"]


def test_lead_order_is_longitudinal_before_lane_relation():
    roles = select_actor_roles(candidates=(candidate("far_same", 80, 0.1, "same"), candidate("near_unrelated", 30, 0.2, "unrelated")), reference_ego_speed_mps=10.0)
    assert [actor["track_id"] for actor in roles["lead_actors"]] == ["near_unrelated", "far_same"]


def test_invisible_actor_is_excluded():
    roles = select_actor_roles(candidates=(candidate("hidden", 5, 0, visible="shadow_not_visible"),))
    assert roles["lead_actors"] == []


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        select_actor_roles(candidates=(candidate("x", 5, 0), candidate("x", 6, 0)))


def test_summary_rejects_cross_list_duplicates():
    actor = {"track_id": "1"}
    roles = {"lead_actors": [actor], "left_nearby_actors": [actor], "right_nearby_actors": [], "selection_context": {"lists": {key: {"truncated": False} for key in ("lead_actors", "left_nearby_actors", "right_nearby_actors")}}}
    with pytest.raises(ValueError, match="multiple"):
        summarize_actor_role_rows(keyframes=({"anchor_id": "a"},), rows=({"anchor_id": "a", "roles": roles},))

def test_role_record_propagates_lane_direction_relation():
    value = candidate("left", 2, 3)
    value = ActorRoleCandidate(
        track_id=value.track_id,
        label_class=value.label_class,
        geometry=value.geometry,
        visibility=value.visibility,
        history=value.history,
        road={**value.road, "lane_direction_relation": "opposing"},
    )
    roles = select_actor_roles(candidates=(value,))
    assert roles["left_nearby_actors"][0]["lane_direction_relation"] == "opposing"
