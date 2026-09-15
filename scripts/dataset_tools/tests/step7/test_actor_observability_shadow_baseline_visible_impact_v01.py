import pytest

from step7.actor_observability_shadow_baseline_visible_impact_v01 import (
    summarize_shadow_baseline_visible_class_impact,
)
from step7.actor_observability_shadow_policy_v01 import ObservabilityShadowPolicy


def policy(name, winning, fraction):
    return ObservabilityShadowPolicy(name, winning, fraction)


def row(track, actor_class, winning, fraction):
    return {
        "anchor_id": "a",
        "track_id": track,
        "label_class": actor_class,
        "evidence_status": "combined_evidence_available",
        "total_winning_cell_count": winning,
        "maximum_visible_fraction": fraction,
        "static_occlusion_evaluated": False,
    }


def test_rates_use_baseline_visible_class_count_as_denominator():
    result = summarize_shadow_baseline_visible_class_impact(
        evidence_rows=(
            row("1", "person", 1, 0.2),
            row("2", "person", 5, 0.8),
            row("3", "automobile", 5, 0.8),
        ),
        policies=(
            policy("baseline", 1, 0.0),
            policy("strict", 2, 0.5),
        ),
        baseline_policy_name="baseline",
    )

    strict_person = next(
        item for item in result
        if item.policy_name == "strict" and item.actor_class == "person"
    )
    assert strict_person.baseline_visible_count == 2
    assert strict_person.retained_visible_count == 1
    assert strict_person.visible_to_not_visible_count == 1
    assert strict_person.retained_visible_rate == 0.5
    assert strict_person.visible_to_not_visible_rate == 0.5


def test_baseline_policy_retains_all_baseline_visible_actors():
    result = summarize_shadow_baseline_visible_class_impact(
        evidence_rows=(row("1", "person", 1, 0.2),),
        policies=(policy("baseline", 1, 0.0),),
        baseline_policy_name="baseline",
    )

    assert result[0].retained_visible_rate == 1.0
    assert result[0].visible_to_not_visible_rate == 0.0


def test_class_with_no_baseline_visible_actor_has_null_rates():
    result = summarize_shadow_baseline_visible_class_impact(
        evidence_rows=(row("1", "person", 0, 0.0),),
        policies=(policy("baseline", 1, 0.0),),
        baseline_policy_name="baseline",
    )

    assert result[0].baseline_visible_count == 0
    assert result[0].retained_visible_rate is None
    assert result[0].visible_to_not_visible_rate is None


def test_missing_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline_policy_name"):
        summarize_shadow_baseline_visible_class_impact(
            evidence_rows=(row("1", "person", 1, 0.2),),
            policies=(policy("p", 1, 0.0),),
            baseline_policy_name="missing",
        )


def test_empty_rows_return_no_impacts():
    result = summarize_shadow_baseline_visible_class_impact(
        evidence_rows=(),
        policies=(policy("baseline", 1, 0.0),),
        baseline_policy_name="baseline",
    )
    assert result == ()
