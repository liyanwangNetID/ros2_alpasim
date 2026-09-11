import pytest

from actor_observability_shadow_policy_v01 import (
    ObservabilityShadowPolicy,
    evaluate_actor_observability_shadow,
    evaluate_actor_observability_shadow_policies,
)


def policy(winning=1, fraction=0.1, name="p1"):
    return ObservabilityShadowPolicy(
        policy_name=name,
        minimum_winning_cell_count=winning,
        minimum_visible_fraction=fraction,
    )


def evidence(
    *,
    status="combined_evidence_available",
    winning=5,
    fraction=0.5,
    anchor="a",
    track="1",
):
    return {
        "anchor_id": anchor,
        "track_id": track,
        "evidence_status": status,
        "total_winning_cell_count": winning,
        "maximum_visible_fraction": fraction,
        "static_occlusion_evaluated": False,
    }


def test_combined_evidence_is_shadow_visible_when_both_requirements_pass():
    result = evaluate_actor_observability_shadow(
        evidence=evidence(),
        policy=policy(),
    )

    assert result.shadow_status == "shadow_visible"
    assert result.winning_cell_requirement_met
    assert result.visible_fraction_requirement_met
    assert result.reasons == ("dynamic_occlusion_policy_requirements_met",)


def test_combined_evidence_reports_each_failed_requirement():
    result = evaluate_actor_observability_shadow(
        evidence=evidence(winning=0, fraction=0.05),
        policy=policy(),
    )

    assert result.shadow_status == "shadow_not_visible"
    assert result.reasons == (
        "minimum_winning_cell_count_not_met",
        "minimum_visible_fraction_not_met",
    )


def test_no_geometric_candidate_is_shadow_not_visible():
    result = evaluate_actor_observability_shadow(
        evidence=evidence(
            status="no_geometric_candidate",
            winning=10,
            fraction=1.0,
        ),
        policy=policy(),
    )

    assert result.shadow_status == "shadow_not_visible"
    assert result.reasons == ("no_geometric_candidate",)


def test_candidate_without_surface_is_shadow_indeterminate():
    result = evaluate_actor_observability_shadow(
        evidence=evidence(
            status="candidate_without_sampled_surface",
            winning=0,
            fraction=None,
        ),
        policy=policy(),
    )

    assert result.shadow_status == "shadow_indeterminate"
    assert result.reasons == (
        "geometric_candidate_without_sampled_surface",
    )


def test_threshold_boundaries_are_inclusive():
    result = evaluate_actor_observability_shadow(
        evidence=evidence(winning=2, fraction=0.25),
        policy=policy(winning=2, fraction=0.25),
    )

    assert result.shadow_status == "shadow_visible"


def test_policy_grid_is_deterministic():
    result = evaluate_actor_observability_shadow_policies(
        evidence_rows=(
            evidence(anchor="b", track="2"),
            evidence(anchor="a", track="9"),
        ),
        policies=(policy(name="first"), policy(name="second")),
    )

    assert tuple(
        (item.policy_name, item.anchor_id, item.track_id)
        for item in result
    ) == (
        ("first", "a", "9"),
        ("first", "b", "2"),
        ("second", "a", "9"),
        ("second", "b", "2"),
    )


def test_duplicate_policy_name_is_rejected():
    with pytest.raises(ValueError, match="policy names must be unique"):
        evaluate_actor_observability_shadow_policies(
            evidence_rows=(evidence(),),
            policies=(policy(), policy()),
        )


def test_static_occlusion_result_is_rejected():
    row = evidence()
    row["static_occlusion_evaluated"] = True

    with pytest.raises(ValueError, match="static occlusion"):
        evaluate_actor_observability_shadow(
            evidence=row,
            policy=policy(),
        )


def test_invalid_policy_threshold_is_rejected():
    with pytest.raises(ValueError, match="within"):
        policy(fraction=1.1)
