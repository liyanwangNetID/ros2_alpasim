"""Shadow-evaluate final Actor observability policies from Step 7E v02 evidence.

This module compares configurable winning-cell and visible-fraction policies
without modifying evidence, selecting a production policy, or writing final
visibility labels. Results are explicitly shadow decisions for offline analysis.
Static-scene occlusion remains outside this evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ObservabilityShadowPolicy:
    policy_name: str
    minimum_winning_cell_count: int
    minimum_visible_fraction: float

    def __post_init__(self) -> None:
        if not self.policy_name:
            raise ValueError("policy_name must be non-empty")
        if (
            isinstance(self.minimum_winning_cell_count, bool)
            or not isinstance(self.minimum_winning_cell_count, int)
            or self.minimum_winning_cell_count < 0
        ):
            raise ValueError(
                "minimum_winning_cell_count must be a non-negative integer"
            )
        value = float(self.minimum_visible_fraction)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "minimum_visible_fraction must be finite and within [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class ActorObservabilityShadowDecision:
    policy_name: str
    anchor_id: str
    track_id: str
    shadow_status: str
    winning_cell_count: int
    maximum_visible_fraction: float | None
    winning_cell_requirement_met: bool
    visible_fraction_requirement_met: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "anchor_id": self.anchor_id,
            "track_id": self.track_id,
            "shadow_status": self.shadow_status,
            "winning_cell_count": self.winning_cell_count,
            "maximum_visible_fraction": self.maximum_visible_fraction,
            "winning_cell_requirement_met": self.winning_cell_requirement_met,
            "visible_fraction_requirement_met": (
                self.visible_fraction_requirement_met
            ),
            "reasons": list(self.reasons),
        }


def evaluate_actor_observability_shadow(
    *,
    evidence: Mapping[str, Any],
    policy: ObservabilityShadowPolicy,
) -> ActorObservabilityShadowDecision:
    """Evaluate one policy against one exported Actor evidence row."""

    for field in (
        "anchor_id",
        "track_id",
        "evidence_status",
        "total_winning_cell_count",
        "maximum_visible_fraction",
        "static_occlusion_evaluated",
    ):
        if field not in evidence:
            raise ValueError(f"evidence is missing {field}")

    if bool(evidence["static_occlusion_evaluated"]):
        raise ValueError(
            "shadow evaluation expects static occlusion to remain unevaluated"
        )

    winning_count = evidence["total_winning_cell_count"]
    if (
        isinstance(winning_count, bool)
        or not isinstance(winning_count, int)
        or winning_count < 0
    ):
        raise ValueError(
            "total_winning_cell_count must be a non-negative integer"
        )

    visible_fraction = evidence["maximum_visible_fraction"]
    if visible_fraction is not None:
        visible_fraction = float(visible_fraction)
        if not math.isfinite(visible_fraction) or not 0.0 <= visible_fraction <= 1.0:
            raise ValueError(
                "maximum_visible_fraction must be finite and within [0, 1]"
            )

    evidence_status = str(evidence["evidence_status"])
    winning_met = winning_count >= policy.minimum_winning_cell_count
    fraction_met = (
        visible_fraction is not None
        and visible_fraction >= policy.minimum_visible_fraction
    )

    if evidence_status == "no_geometric_candidate":
        shadow_status = "shadow_not_visible"
        reasons = ("no_geometric_candidate",)
        winning_met = False
        fraction_met = False
    elif evidence_status == "candidate_without_sampled_surface":
        shadow_status = "shadow_indeterminate"
        reasons = ("geometric_candidate_without_sampled_surface",)
        winning_met = False
        fraction_met = False
    elif evidence_status == "combined_evidence_available":
        failures = []
        if not winning_met:
            failures.append("minimum_winning_cell_count_not_met")
        if not fraction_met:
            failures.append("minimum_visible_fraction_not_met")
        if failures:
            shadow_status = "shadow_not_visible"
            reasons = tuple(failures)
        else:
            shadow_status = "shadow_visible"
            reasons = ("dynamic_occlusion_policy_requirements_met",)
    else:
        raise ValueError(f"unexpected evidence_status: {evidence_status}")

    return ActorObservabilityShadowDecision(
        policy_name=policy.policy_name,
        anchor_id=str(evidence["anchor_id"]),
        track_id=str(evidence["track_id"]),
        shadow_status=shadow_status,
        winning_cell_count=winning_count,
        maximum_visible_fraction=visible_fraction,
        winning_cell_requirement_met=winning_met,
        visible_fraction_requirement_met=fraction_met,
        reasons=reasons,
    )


def evaluate_actor_observability_shadow_policies(
    *,
    evidence_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[ObservabilityShadowPolicy],
) -> tuple[ActorObservabilityShadowDecision, ...]:
    """Evaluate all policies and rows in deterministic policy/Actor order."""

    policy_values = tuple(policies)
    names = tuple(policy.policy_name for policy in policy_values)
    if len(set(names)) != len(names):
        raise ValueError("shadow policy names must be unique")

    rows = tuple(evidence_rows)
    identities = tuple(
        (str(row.get("anchor_id", "")), str(row.get("track_id", "")))
        for row in rows
    )
    if any(not anchor_id or not track_id for anchor_id, track_id in identities):
        raise ValueError("evidence rows must have usable Anchor/Actor identities")
    if len(set(identities)) != len(identities):
        raise ValueError("evidence Anchor/Actor identities must be unique")

    ordered_rows = tuple(
        row
        for _, row in sorted(
            zip(identities, rows),
            key=lambda item: item[0],
        )
    )
    return tuple(
        evaluate_actor_observability_shadow(evidence=row, policy=policy)
        for policy in policy_values
        for row in ordered_rows
    )
