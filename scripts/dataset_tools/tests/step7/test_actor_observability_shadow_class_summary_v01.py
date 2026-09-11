import pytest

from actor_observability_shadow_class_summary_v01 import (
    summarize_observability_shadow_grid_by_actor_class,
)
from actor_observability_shadow_policy_v01 import ObservabilityShadowPolicy


def policy(name, winning, fraction):
    return ObservabilityShadowPolicy(name, winning, fraction)


def row(anchor, track, actor_class, winning, fraction):
    return {
        "anchor_id": anchor,
        "track_id": track,
        "label_class": actor_class,
        "evidence_status": "combined_evidence_available",
        "total_winning_cell_count": winning,
        "maximum_visible_fraction": fraction,
        "static_occlusion_evaluated": False,
    }


def test_summarizes_each_policy_and_class_deterministically():
    result = summarize_observability_shadow_grid_by_actor_class(
        evidence_rows=(
            row("a", "1", "person", 1, 0.2),
            row("a", "2", "automobile", 5, 0.8),
            row("a", "3", "person", 5, 0.8),
        ),
        policies=(
            policy("baseline", 1, 0.1),
            policy("strict", 2, 0.5),
        ),
        baseline_policy_name="baseline",
    )

    assert tuple((item.policy_name, item.actor_class) for item in result) == (
        ("baseline", "automobile"),
        ("baseline", "person"),
        ("strict", "automobile"),
        ("strict", "person"),
    )
    strict_person = result[3]
    assert strict_person.actor_count == 2
    assert dict(strict_person.shadow_status_counts) == {
        "shadow_not_visible": 1,
        "shadow_visible": 1,
    }
    assert strict_person.visible_to_not_visible_from_baseline_count == 1


def test_empty_rows_return_no_class_summaries():
    result = summarize_observability_shadow_grid_by_actor_class(
        evidence_rows=(),
        policies=(policy("baseline", 1, 0.0),),
        baseline_policy_name="baseline",
    )
    assert result == ()


def test_missing_label_class_is_rejected():
    value = row("a", "1", "person", 1, 0.2)
    del value["label_class"]
    with pytest.raises(ValueError, match="missing label_class"):
        summarize_observability_shadow_grid_by_actor_class(
            evidence_rows=(value,),
            policies=(policy("baseline", 1, 0.0),),
            baseline_policy_name="baseline",
        )


def test_missing_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline_policy_name"):
        summarize_observability_shadow_grid_by_actor_class(
            evidence_rows=(row("a", "1", "person", 1, 0.2),),
            policies=(policy("p", 1, 0.0),),
            baseline_policy_name="missing",
        )
