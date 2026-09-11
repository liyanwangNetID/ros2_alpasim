"""Summarize Step 7E shadow-policy losses relative to baseline-visible Actors.

The module computes, for each policy and Actor class, how many Actors visible
under a designated baseline become not visible under the compared policy. Rates
use the baseline-visible class population as denominator, rather than all Actor
rows. Results remain offline shadow statistics and are not final labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from actor_observability_shadow_policy_v01 import (
    ObservabilityShadowPolicy,
    evaluate_actor_observability_shadow_policies,
)


@dataclass(frozen=True, slots=True)
class ObservabilityShadowBaselineVisibleClassImpact:
    policy_name: str
    actor_class: str
    baseline_visible_count: int
    retained_visible_count: int
    visible_to_not_visible_count: int
    retained_visible_rate: float | None
    visible_to_not_visible_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "actor_class": self.actor_class,
            "baseline_visible_count": self.baseline_visible_count,
            "retained_visible_count": self.retained_visible_count,
            "visible_to_not_visible_count": self.visible_to_not_visible_count,
            "retained_visible_rate": self.retained_visible_rate,
            "visible_to_not_visible_rate": self.visible_to_not_visible_rate,
        }


def summarize_shadow_baseline_visible_class_impact(
    *,
    evidence_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[ObservabilityShadowPolicy],
    baseline_policy_name: str,
) -> tuple[ObservabilityShadowBaselineVisibleClassImpact, ...]:
    """Return policy-by-class retention rates among baseline-visible Actors."""

    rows = tuple(evidence_rows)
    policy_values = tuple(policies)
    names = tuple(item.policy_name for item in policy_values)
    if len(set(names)) != len(names):
        raise ValueError("shadow policy names must be unique")
    if baseline_policy_name not in names:
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
            (item.anchor_id, item.track_id): item.shadow_status
            for item in decisions
            if item.policy_name == name
        }
        for name in names
    }
    if any(len(values) != len(rows) for values in by_policy.values()):
        raise RuntimeError("shadow decision counts do not close")

    baseline = by_policy[baseline_policy_name]
    actor_classes = tuple(sorted(set(class_by_identity.values())))
    output = []
    for policy_name in names:
        current = by_policy[policy_name]
        for actor_class in actor_classes:
            baseline_visible_ids = tuple(
                identity
                for identity, value in baseline.items()
                if value == "shadow_visible"
                and class_by_identity[identity] == actor_class
            )
            retained = sum(
                current[identity] == "shadow_visible"
                for identity in baseline_visible_ids
            )
            lost = sum(
                current[identity] == "shadow_not_visible"
                for identity in baseline_visible_ids
            )
            if retained + lost != len(baseline_visible_ids):
                raise RuntimeError(
                    "baseline-visible Actor became an unsupported shadow status"
                )
            denominator = len(baseline_visible_ids)
            retained_rate = retained / denominator if denominator else None
            lost_rate = lost / denominator if denominator else None
            output.append(
                ObservabilityShadowBaselineVisibleClassImpact(
                    policy_name=policy_name,
                    actor_class=actor_class,
                    baseline_visible_count=denominator,
                    retained_visible_count=retained,
                    visible_to_not_visible_count=lost,
                    retained_visible_rate=retained_rate,
                    visible_to_not_visible_rate=lost_rate,
                )
            )

    return tuple(output)
