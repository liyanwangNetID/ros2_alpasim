from step7.actor_observability_dynamic_occlusion_policy_v01 import (
    FROZEN_DYNAMIC_OCCLUSION_POLICY,
    MINIMUM_VISIBLE_FRACTION,
    MINIMUM_WINNING_CELL_COUNT,
    POLICY_NAME,
    evaluate_frozen_dynamic_occlusion_policy,
)


def row(track_id, winning, visible_fraction):
    return {
        "anchor_id": "a",
        "clip_id": "c",
        "track_id": track_id,
        "label_class": "automobile",
        "evidence_status": "combined_evidence_available",
        "total_winning_cell_count": winning,
        "maximum_visible_fraction": visible_fraction,
        "static_occlusion_evaluated": False,
    }


def test_frozen_policy_constants_are_explicit():
    assert POLICY_NAME == "step7e_dynamic_occlusion_v01"
    assert MINIMUM_WINNING_CELL_COUNT == 2
    assert MINIMUM_VISIBLE_FRACTION == 0.0
    assert FROZEN_DYNAMIC_OCCLUSION_POLICY.minimum_winning_cell_count == 2
    assert FROZEN_DYNAMIC_OCCLUSION_POLICY.minimum_visible_fraction == 0.0


def test_two_winning_cells_are_visible_even_with_low_fraction():
    decisions = evaluate_frozen_dynamic_occlusion_policy(
        evidence_rows=(row("1", 2, 0.0001),)
    )
    assert len(decisions) == 1
    assert decisions[0].policy_name == POLICY_NAME
    assert decisions[0].shadow_status == "shadow_visible"


def test_one_winning_cell_is_not_visible_even_with_high_fraction():
    decisions = evaluate_frozen_dynamic_occlusion_policy(
        evidence_rows=(row("1", 1, 1.0),)
    )
    assert decisions[0].shadow_status == "shadow_not_visible"
