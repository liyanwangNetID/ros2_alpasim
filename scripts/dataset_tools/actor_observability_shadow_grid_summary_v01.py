"""Summarize Step 7E observability shadow decisions across policy grids.

This module aggregates already evaluated shadow decisions by policy, status,
and reason. It also compares each policy with a designated baseline policy.
It does not choose a production policy, change evidence, or write final labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from actor_observability_shadow_policy_v01 import (
    ActorObservabilityShadowDecision,
    ObservabilityShadowPolicy,
    evaluate_actor_observability_shadow_policies,
)


@dataclass(frozen=True, slots=True)
class ObservabilityShadowPolicySummary:
    policy_name: str
    minimum_winning_cell_count: int
    minimum_visible_fraction: float
    actor_count: int
    shadow_status_counts: tuple[tuple[str, int], ...]
    reason_counts: tuple[tuple[str, int], ...]
    status_changed_from_baseline_count: int
    visible_to_not_visible_from_baseline_count: int
    not_visible_to_visible_from_baseline_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "minimum_winning_cell_count": self.minimum_winning_cell_count,
            "minimum_visible_fraction": self.minimum_visible_fraction,
            "actor_count": self.actor_count,
            "shadow_status_counts": dict(self.shadow_status_counts),
            "reason_counts": dict(self.reason_counts),
            "status_changed_from_baseline_count": (
                self.status_changed_from_baseline_count
            ),
            "visible_to_not_visible_from_baseline_count": (
                self.visible_to_not_visible_from_baseline_count
            ),
            "not_visible_to_visible_from_baseline_count": (
                self.not_visible_to_visible_from_baseline_count
            ),
        }


@dataclass(frozen=True, slots=True)
class ObservabilityShadowGridSummary:
    baseline_policy_name: str
    actor_count: int
    policy_count: int
    policy_summaries: tuple[ObservabilityShadowPolicySummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_policy_name": self.baseline_policy_name,
            "actor_count": self.actor_count,
            "policy_count": self.policy_count,
            "policy_summaries": [
                item.to_dict() for item in self.policy_summaries
            ],
        }


def summarize_observability_shadow_policy_grid(
    *,
    evidence_rows: Sequence[dict[str, Any]],
    policies: Sequence[ObservabilityShadowPolicy],
    baseline_policy_name: str,
) -> ObservabilityShadowGridSummary:
    """Evaluate and summarize a policy grid against one baseline policy."""

    policy_values = tuple(policies)
    policy_by_name = {item.policy_name: item for item in policy_values}
    if len(policy_by_name) != len(policy_values):
        raise ValueError("shadow policy names must be unique")
    if baseline_policy_name not in policy_by_name:
        raise ValueError("baseline_policy_name must identify one policy")

    rows = tuple(evidence_rows)
    decisions = evaluate_actor_observability_shadow_policies(
        evidence_rows=rows,
        policies=policy_values,
    )
    decisions_by_policy: dict[
        str, tuple[ActorObservabilityShadowDecision, ...]
    ] = {}
    for policy in policy_values:
        values = tuple(
            item for item in decisions if item.policy_name == policy.policy_name
        )
        if len(values) != len(rows):
            raise RuntimeError("shadow decision count does not close")
        decisions_by_policy[policy.policy_name] = values

    baseline = decisions_by_policy[baseline_policy_name]
    baseline_by_id = {
        (item.anchor_id, item.track_id): item.shadow_status
        for item in baseline
    }
    if len(baseline_by_id) != len(baseline):
        raise RuntimeError("baseline shadow identities are not unique")

    summaries = []
    for policy in policy_values:
        values = decisions_by_policy[policy.policy_name]
        status_counts = Counter(item.shadow_status for item in values)
        reason_counts = Counter(
            reason for item in values for reason in item.reasons
        )
        changed = 0
        visible_to_not_visible = 0
        not_visible_to_visible = 0
        for item in values:
            previous = baseline_by_id[(item.anchor_id, item.track_id)]
            current = item.shadow_status
            changed += int(current != previous)
            visible_to_not_visible += int(
                previous == "shadow_visible"
                and current == "shadow_not_visible"
            )
            not_visible_to_visible += int(
                previous == "shadow_not_visible"
                and current == "shadow_visible"
            )

        if sum(status_counts.values()) != len(rows):
            raise RuntimeError("shadow status counts do not close")
        summaries.append(
            ObservabilityShadowPolicySummary(
                policy_name=policy.policy_name,
                minimum_winning_cell_count=(
                    policy.minimum_winning_cell_count
                ),
                minimum_visible_fraction=policy.minimum_visible_fraction,
                actor_count=len(rows),
                shadow_status_counts=tuple(sorted(status_counts.items())),
                reason_counts=tuple(sorted(reason_counts.items())),
                status_changed_from_baseline_count=changed,
                visible_to_not_visible_from_baseline_count=(
                    visible_to_not_visible
                ),
                not_visible_to_visible_from_baseline_count=(
                    not_visible_to_visible
                ),
            )
        )

    return ObservabilityShadowGridSummary(
        baseline_policy_name=baseline_policy_name,
        actor_count=len(rows),
        policy_count=len(policy_values),
        policy_summaries=tuple(summaries),
    )
