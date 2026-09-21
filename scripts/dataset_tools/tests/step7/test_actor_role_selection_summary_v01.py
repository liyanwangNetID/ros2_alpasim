import pytest
from step7.actor_role_selection_summary_v01 import summarize_actor_role_rows


def row(anchor, duplicate=False):
    lead = {"track_id": "1"}
    return {
        "anchor_id": anchor,
        "roles": {
            "lead_vehicle": lead,
            "left_nearby_vehicle": lead if duplicate else None,
            "right_nearby_vehicle": None,
        },
        "empty_role_reasons": {
            "lead_vehicle": None,
            "left_nearby_vehicle": None if duplicate else "no_candidate_in_role_region",
            "right_nearby_vehicle": "no_candidate_in_role_region",
        },
    }


def test_summary_counts_roles_and_empty_reasons():
    result = summarize_actor_role_rows(
        keyframes=({"anchor_id": "a"},), rows=(row("a"),)
    )
    assert result["selected_role_counts"]["lead_vehicle"] == 1
    assert result["role_conflict_count"] == 0


def test_duplicate_actor_across_roles_is_rejected():
    with pytest.raises(ValueError, match="multiple roles"):
        summarize_actor_role_rows(
            keyframes=({"anchor_id": "a"},), rows=(row("a", duplicate=True),)
        )
