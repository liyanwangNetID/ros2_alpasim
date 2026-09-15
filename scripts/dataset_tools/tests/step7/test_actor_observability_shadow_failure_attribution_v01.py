import pytest

from step7.actor_observability_shadow_failure_attribution_v01 import (
    summarize_shadow_failure_attribution,
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


def test_attributes_only_winning_only_fraction_and_both_failures():
    result = summarize_shadow_failure_attribution(
        evidence_rows=(
            row("1", "person", 1, 0.8),
            row("2", "person", 5, 0.2),
            row("3", "person", 1, 0.2),
            row("4", "person", 5, 0.8),
        ),
        policies=(
            policy("baseline", 1, 0.0),
            policy("strict", 2, 0.5),
        ),
        baseline_policy_name="baseline",
    )

    strict = next(item for item in result if item.policy_name == "strict")
    assert strict.baseline_visible_count == 4
    assert strict.retained_visible_count == 1
    assert strict.only_winning_cell_failed_count == 1
    assert strict.only_visible_fraction_failed_count == 1
    assert strict.both_requirements_failed_count == 1
    assert strict.total_visible_to_not_visible_count == 3


def test_baseline_policy_has_no_failures():
    result = summarize_shadow_failure_attribution(
        evidence_rows=(row("1", "person", 1, 0.2),),
        policies=(policy("baseline", 1, 0.0),),
        baseline_policy_name="baseline",
    )

    assert result[0].retained_visible_count == 1
    assert result[0].total_visible_to_not_visible_count == 0


def test_output_is_policy_then_class_order():
    result = summarize_shadow_failure_attribution(
        evidence_rows=(
            row("1", "person", 1, 0.2),
            row("2", "automobile", 1, 0.2),
        ),
        policies=(policy("base", 1, 0.0), policy("strict", 2, 0.5)),
        baseline_policy_name="base",
    )

    assert tuple((item.policy_name, item.actor_class) for item in result) == (
        ("base", "automobile"),
        ("base", "person"),
        ("strict", "automobile"),
        ("strict", "person"),
    )


def test_missing_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline_policy_name"):
        summarize_shadow_failure_attribution(
            evidence_rows=(row("1", "person", 1, 0.2),),
            policies=(policy("p", 1, 0.0),),
            baseline_policy_name="missing",
        )


def test_empty_rows_return_no_attributions():
    result = summarize_shadow_failure_attribution(
        evidence_rows=(),
        policies=(policy("baseline", 1, 0.0),),
        baseline_policy_name="baseline",
    )
    assert result == ()
