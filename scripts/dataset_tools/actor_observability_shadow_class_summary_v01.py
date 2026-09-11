"""Break down Step 7E shadow policy results by Actor class.

This module evaluates an observability shadow policy grid and aggregates each
policy by Actor class and shadow status. It compares class-level status changes
with a named baseline policy. It does not recommend a production policy or
write final visibility labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from actor_observability_shadow_policy_v01 import (
    ObservabilityShadowPolicy,
    evaluate_actor_observability_shadow_policies,
)


@dataclass(frozen=True, slots=True)
class ObservabilityShadowClassPolicySummary:
    policy_name: str
    actor_class: str
    actor_count: int
    shadow_status_counts: tuple[tuple[str, int], ...]
    status_changed_from_baseline_count: int
    visible_to_not_visible_from_baseline_count: int
    not_visible_to_visible_from_baseline_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "actor_class": self.actor_class,
            "actor_count": self.actor_count,
            "shadow_status_counts": dict(self.shadow_status_counts),
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


def summarize_observability_shadow_grid_by_actor_class(
    *,
    evidence_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[ObservabilityShadowPolicy],
    baseline_policy_name: str,
) -> tuple[ObservabilityShadowClassPolicySummary, ...]:
    """Return deterministic policy-by-class shadow summaries."""

    rows = tuple(evidence_rows)
    policies = tuple(policies)
    policy_names = tuple(item.policy_name for item in policies)
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("shadow policy names must be unique")
    if baseline_policy_name not in policy_names:
        raise ValueError("baseline_policy_name must identify one policy")

    classes_by_identity = {}
    for index, row in enumerate(rows):
        for field in ("anchor_id", "track_id", "label_class"):
            if field not in row:
                raise ValueError(f"evidence row {index} is missing {field}")
        identity = (str(row["anchor_id"]), str(row["track_id"]))
        if identity in classes_by_identity:
            raise ValueError("evidence Anchor/Actor identities must be unique")
        classes_by_identity[identity] = str(row["label_class"])

    decisions = evaluate_actor_observability_shadow_policies(
        evidence_rows=rows,
        policies=policies,
    )
    by_policy = {
        policy_name: tuple(
            item for item in decisions if item.policy_name == policy_name
        )
        for policy_name in policy_names
    }
    baseline = {
        (item.anchor_id, item.track_id): item.shadow_status
        for item in by_policy[baseline_policy_name]
    }

    actor_classes = tuple(sorted(set(classes_by_identity.values())))
    summaries = []
    for policy_name in policy_names:
        decisions_for_policy = by_policy[policy_name]
        for actor_class in actor_classes:
            values = tuple(
                item
                for item in decisions_for_policy
                if classes_by_identity[(item.anchor_id, item.track_id)]
                == actor_class
            )
            status_counts = Counter(item.shadow_status for item in values)
            changed = 0
            visible_to_not_visible = 0
            not_visible_to_visible = 0
            for item in values:
                previous = baseline[(item.anchor_id, item.track_id)]
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
            if sum(status_counts.values()) != len(values):
                raise RuntimeError("class-level shadow status counts do not close")
            summaries.append(
                ObservabilityShadowClassPolicySummary(
                    policy_name=policy_name,
                    actor_class=actor_class,
                    actor_count=len(values),
                    shadow_status_counts=tuple(sorted(status_counts.items())),
                    status_changed_from_baseline_count=changed,
                    visible_to_not_visible_from_baseline_count=(
                        visible_to_not_visible
                    ),
                    not_visible_to_visible_from_baseline_count=(
                        not_visible_to_visible
                    ),
                )
            )

    return tuple(summaries)
