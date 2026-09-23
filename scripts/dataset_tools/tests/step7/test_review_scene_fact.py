from __future__ import annotations
import pytest
from step7 import review_scene_fact as target


def row(anchor_id, present=False):
    return {"anchor_id": anchor_id, "clip_id": "test_clip_001", "anchor_ns": 10, "scene_fact_format_version": "0.1-draft", "rule_version": "rules", "actor_context": {"lead_actors": [{"track_id": "1", "role_rank": 1}] if present else [], "left_nearby_actors": [], "right_nearby_actors": []}, "road_context": {}, "quality": {}}


def test_select_by_anchor():
    assert target.select_scene_fact((row("a"), row("b")), anchor_id="b", seed=None, require_present_role=False)["anchor_id"] == "b"


def test_random_selection_requires_present_actor():
    assert target.select_scene_fact((row("empty"), row("present", True)), anchor_id=None, seed=7, require_present_role=True)["anchor_id"] == "present"


def test_unknown_anchor_is_rejected():
    with pytest.raises(KeyError, match="not found"):
        target.select_scene_fact((row("a"),), anchor_id="missing", seed=None, require_present_role=False)


def test_clip_specific_review_offset():
    assert target.review_principal_point_y_offset("test_clip_894", "front_wide") == -16.0
    assert target.review_principal_point_y_offset("test_clip_001", "front_wide") == 0.0
