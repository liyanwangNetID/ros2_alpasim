"""Attribute Step 7E shadow visibility losses to policy requirements.

For every policy and Actor class, this module examines Actors that are visible
under a named baseline but not visible under the compared policy. Each loss is
classified as a winning-cell-only failure, a visible-fraction-only failure, or
a failure of both requirements. It does not select a production policy, modify
formal evidence, or write final visibility labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from step7.actor_observability_shadow_policy_v01 import (
    ObservabilityShadowPolicy,
    evaluate_actor_observability_shadow_policies,
)


@dataclass(frozen=True, slots=True)
class ObservabilityShadowFailureAttribution:
    policy_name: str
    actor_class: str
    baseline_visible_count: int
    retained_visible_count: int
    only_winning_cell_failed_count: int
    only_visible_fraction_failed_count: int
    both_requirements_failed_count: int
    total_visible_to_not_visible_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "actor_class": self.actor_class,
            "baseline_visible_count": self.baseline_visible_count,
            "retained_visible_count": self.retained_visible_count,
            "only_winning_cell_failed_count": (
                self.only_winning_cell_failed_count
            ),
            "only_visible_fraction_failed_count": (
                self.only_visible_fraction_failed_count
            ),
            "both_requirements_failed_count": (
                self.both_requirements_failed_count
            ),
            "total_visible_to_not_visible_count": (
                self.total_visible_to_not_visible_count
            ),
        }


def summarize_shadow_failure_attribution(
    *,
    evidence_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[ObservabilityShadowPolicy],
    baseline_policy_name: str,
) -> tuple[ObservabilityShadowFailureAttribution, ...]:
    """Classify policy losses among baseline-visible Actors by failed rule."""

    rows = tuple(evidence_rows)
    policy_values = tuple(policies)
    policy_names = tuple(item.policy_name for item in policy_values)
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("shadow policy names must be unique")
    if baseline_policy_name not in policy_names:
        raise ValueError("baseline_policy_name must identify one policy")

    class_by_identity = {}
    for index, row in enumerate(rows):
        for field in ("anchor_id", "track_id", "label_class"):
            if field not in row:
                raise ValueError(f"evidence row {index} is missing {field}")
        identity = (str(row["anchor_id"]), str(row["track_id"]))
        if identity in class_by_identity:
            raise ValueError("evidence Anchor/Actor identities must be unique")
        class_by_identity[identity] = str(row["label_class"])

    decisions = evaluate_actor_observability_shadow_policies(
        evidence_rows=rows,
        policies=policy_values,
    )
    by_policy = {
        name: {
            (item.anchor_id, item.track_id): item
            for item in decisions
            if item.policy_name == name
        }
        for name in policy_names
    }
    if any(len(items) != len(rows) for items in by_policy.values()):
        raise RuntimeError("shadow decision counts do not close")

    baseline = by_policy[baseline_policy_name]
    actor_classes = tuple(sorted(set(class_by_identity.values())))
    output = []

    for policy_name in policy_names:
        current = by_policy[policy_name]
        for actor_class in actor_classes:
            baseline_visible_ids = tuple(
                identity
                for identity, decision in baseline.items()
                if decision.shadow_status == "shadow_visible"
                and class_by_identity[identity] == actor_class
            )
            retained = 0
            only_winning = 0
            only_fraction = 0
            both = 0

            for identity in baseline_visible_ids:
                decision = current[identity]
                if decision.shadow_status == "shadow_visible":
                    retained += 1
                    continue
                if decision.shadow_status != "shadow_not_visible":
                    raise RuntimeError(
                        "baseline-visible Actor became an unsupported status"
                    )
                winning_failed = not decision.winning_cell_requirement_met
                fraction_failed = not decision.visible_fraction_requirement_met
                if winning_failed and fraction_failed:
                    both += 1
                elif winning_failed:
                    only_winning += 1
                elif fraction_failed:
                    only_fraction += 1
                else:
                    raise RuntimeError(
                        "not-visible decision has no failed policy requirement"
                    )

            lost = only_winning + only_fraction + both
            if retained + lost != len(baseline_visible_ids):
                raise RuntimeError("failure attribution counts do not close")
            output.append(
                ObservabilityShadowFailureAttribution(
                    policy_name=policy_name,
                    actor_class=actor_class,
                    baseline_visible_count=len(baseline_visible_ids),
                    retained_visible_count=retained,
                    only_winning_cell_failed_count=only_winning,
                    only_visible_fraction_failed_count=only_fraction,
                    both_requirements_failed_count=both,
                    total_visible_to_not_visible_count=lost,
                )
            )

    return tuple(output)
