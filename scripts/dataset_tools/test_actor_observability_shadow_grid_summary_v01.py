import pytest

from actor_observability_shadow_grid_summary_v01 import (
    summarize_observability_shadow_policy_grid,
)
from actor_observability_shadow_policy_v01 import ObservabilityShadowPolicy


def policy(name, winning, fraction):
    return ObservabilityShadowPolicy(
        policy_name=name,
        minimum_winning_cell_count=winning,
        minimum_visible_fraction=fraction,
    )


def row(anchor, track, status, winning, fraction):
    return {
        "anchor_id": anchor,
        "track_id": track,
        "evidence_status": status,
        "total_winning_cell_count": winning,
        "maximum_visible_fraction": fraction,
        "static_occlusion_evaluated": False,
    }


def rows():
    return (
        row("a", "1", "combined_evidence_available", 1, 0.2),
        row("a", "2", "combined_evidence_available", 5, 0.8),
        row("a", "3", "no_geometric_candidate", 0, None),
        row("a", "4", "candidate_without_sampled_surface", 0, None),
    )


def test_summarizes_policy_statuses_and_reasons():
    result = summarize_observability_shadow_policy_grid(
        evidence_rows=rows(),
        policies=(policy("baseline", 1, 0.1),),
        baseline_policy_name="baseline",
    )

    summary = result.policy_summaries[0]
    assert result.actor_count == 4
    assert dict(summary.shadow_status_counts) == {
        "shadow_indeterminate": 1,
        "shadow_not_visible": 1,
        "shadow_visible": 2,
    }
    assert dict(summary.reason_counts) == {
        "dynamic_occlusion_policy_requirements_met": 2,
        "geometric_candidate_without_sampled_surface": 1,
        "no_geometric_candidate": 1,
    }
    assert summary.status_changed_from_baseline_count == 0


def test_compares_stricter_policy_with_baseline():
    result = summarize_observability_shadow_policy_grid(
        evidence_rows=rows(),
        policies=(
            policy("baseline", 1, 0.1),
            policy("strict", 2, 0.5),
        ),
        baseline_policy_name="baseline",
    )

    strict = result.policy_summaries[1]
    assert strict.status_changed_from_baseline_count == 1
    assert strict.visible_to_not_visible_from_baseline_count == 1
    assert strict.not_visible_to_visible_from_baseline_count == 0
    assert dict(strict.shadow_status_counts) == {
        "shadow_indeterminate": 1,
        "shadow_not_visible": 2,
        "shadow_visible": 1,
    }


def test_more_permissive_policy_can_add_visible_actor():
    result = summarize_observability_shadow_policy_grid(
        evidence_rows=rows(),
        policies=(
            policy("baseline", 2, 0.5),
            policy("permissive", 1, 0.1),
        ),
        baseline_policy_name="baseline",
    )

    permissive = result.policy_summaries[1]
    assert permissive.not_visible_to_visible_from_baseline_count == 1
    assert permissive.visible_to_not_visible_from_baseline_count == 0


def test_missing_baseline_policy_is_rejected():
    with pytest.raises(ValueError, match="baseline_policy_name"):
        summarize_observability_shadow_policy_grid(
            evidence_rows=rows(),
            policies=(policy("p", 1, 0.1),),
            baseline_policy_name="missing",
        )


def test_empty_evidence_rows_are_supported():
    result = summarize_observability_shadow_policy_grid(
        evidence_rows=(),
        policies=(policy("baseline", 1, 0.1),),
        baseline_policy_name="baseline",
    )

    assert result.actor_count == 0
    assert result.policy_summaries[0].shadow_status_counts == ()
    assert result.to_dict()["policy_summaries"][0]["actor_count"] == 0
