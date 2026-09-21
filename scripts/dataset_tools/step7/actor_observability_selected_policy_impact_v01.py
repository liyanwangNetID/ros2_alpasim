"""Analyze affected Actors for narrowed Step 7E observability policies.

This module compares a permissive baseline with two selected dynamic-occlusion
policy candidates. It operates only on exported v02 evidence and does not
modify evidence or write final observability labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from step7.actor_observability_shadow_policy_v01 import (
    ObservabilityShadowPolicy,
    evaluate_actor_observability_shadow_policies,
)


BASELINE_POLICY = ObservabilityShadowPolicy(
    policy_name="winning_1_fraction_0",
    minimum_winning_cell_count=1,
    minimum_visible_fraction=0.0,
)
WINNING_CELL_CANDIDATE = ObservabilityShadowPolicy(
    policy_name="winning_2_fraction_0",
    minimum_winning_cell_count=2,
    minimum_visible_fraction=0.0,
)
VISIBLE_FRACTION_CANDIDATE = ObservabilityShadowPolicy(
    policy_name="winning_1_fraction_0p01",
    minimum_winning_cell_count=1,
    minimum_visible_fraction=0.01,
)
SELECTED_POLICIES = (
    BASELINE_POLICY,
    WINNING_CELL_CANDIDATE,
    VISIBLE_FRACTION_CANDIDATE,
)


@dataclass(frozen=True, slots=True)
class AffectedActorRow:
    anchor_id: str
    clip_id: str
    track_id: str
    label_class: str
    total_winning_cell_count: int
    maximum_visible_fraction: float | None
    baseline_status: str
    winning_cell_candidate_status: str
    visible_fraction_candidate_status: str
    change_category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "clip_id": self.clip_id,
            "track_id": self.track_id,
            "label_class": self.label_class,
            "total_winning_cell_count": self.total_winning_cell_count,
            "maximum_visible_fraction": self.maximum_visible_fraction,
            "baseline_status": self.baseline_status,
            "winning_cell_candidate_status": self.winning_cell_candidate_status,
            "visible_fraction_candidate_status": self.visible_fraction_candidate_status,
            "change_category": self.change_category,
        }


def _required_text(row: Mapping[str, Any], field: str, index: int) -> str:
    if field not in row:
        raise ValueError(f"evidence row {index} is missing {field}")
    value = str(row[field])
    if not value:
        raise ValueError(f"evidence row {index} has empty {field}")
    return value


def analyze_selected_policy_changes(
    *,
    evidence_rows: Sequence[Mapping[str, Any]],
    expected_winning_affected_count: int | None = None,
    expected_fraction_affected_count: int | None = None,
) -> dict[str, Any]:
    """Return affected-row, Clip-level, and class-level comparisons."""
    rows = tuple(evidence_rows)
    identities: set[tuple[str, str]] = set()
    row_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        anchor_id = _required_text(row, "anchor_id", index)
        track_id = _required_text(row, "track_id", index)
        _required_text(row, "clip_id", index)
        _required_text(row, "label_class", index)
        identity = (anchor_id, track_id)
        if identity in identities:
            raise ValueError("evidence Anchor/Actor identities must be unique")
        identities.add(identity)
        row_by_identity[identity] = row

    decisions = evaluate_actor_observability_shadow_policies(
        evidence_rows=rows,
        policies=SELECTED_POLICIES,
    )
    status_by_policy = {
        policy.policy_name: {
            (item.anchor_id, item.track_id): item.shadow_status
            for item in decisions
            if item.policy_name == policy.policy_name
        }
        for policy in SELECTED_POLICIES
    }
    if any(len(values) != len(rows) for values in status_by_policy.values()):
        raise RuntimeError("selected policy decision counts do not close")

    baseline = status_by_policy[BASELINE_POLICY.policy_name]
    winning = status_by_policy[WINNING_CELL_CANDIDATE.policy_name]
    fraction = status_by_policy[VISIBLE_FRACTION_CANDIDATE.policy_name]
    affected: list[AffectedActorRow] = []
    for identity in sorted(identities):
        if baseline[identity] != "shadow_visible":
            continue
        winning_changed = winning[identity] == "shadow_not_visible"
        fraction_changed = fraction[identity] == "shadow_not_visible"
        if not winning_changed and not fraction_changed:
            continue
        if winning_changed and fraction_changed:
            category = "rejected_by_both_candidates"
        elif winning_changed:
            category = "rejected_by_winning_cell_candidate_only"
        else:
            category = "rejected_by_visible_fraction_candidate_only"
        row = row_by_identity[identity]
        affected.append(
            AffectedActorRow(
                anchor_id=identity[0],
                clip_id=str(row["clip_id"]),
                track_id=identity[1],
                label_class=str(row["label_class"]),
                total_winning_cell_count=int(row["total_winning_cell_count"]),
                maximum_visible_fraction=(
                    None
                    if row["maximum_visible_fraction"] is None
                    else float(row["maximum_visible_fraction"])
                ),
                baseline_status=baseline[identity],
                winning_cell_candidate_status=winning[identity],
                visible_fraction_candidate_status=fraction[identity],
                change_category=category,
            )
        )

    category_counts = Counter(item.change_category for item in affected)
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    clip_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in affected:
        class_counts[item.label_class][item.change_category] += 1
        clip_counts[item.clip_id][item.change_category] += 1

    affected_by_winning = sum(
        item.winning_cell_candidate_status == "shadow_not_visible"
        for item in affected
    )
    affected_by_fraction = sum(
        item.visible_fraction_candidate_status == "shadow_not_visible"
        for item in affected
    )
    if (
        expected_winning_affected_count is not None
        and affected_by_winning != expected_winning_affected_count
    ):
        raise RuntimeError(
            f"winning-cell candidate affected count changed: {affected_by_winning}"
        )
    if (
        expected_fraction_affected_count is not None
        and affected_by_fraction != expected_fraction_affected_count
    ):
        raise RuntimeError(
            f"visible-fraction candidate affected count changed: {affected_by_fraction}"
        )

    return {
        "schema_version": "step7e-observability-selected-policy-impact-v01",
        "description": (
            "Affected baseline-visible Actors for the two narrowed dynamic-"
            "occlusion policy candidates. Static-scene occlusion remains "
            "unevaluated and no final labels are written."
        ),
        "policies": [
            {
                "policy_name": policy.policy_name,
                "minimum_winning_cell_count": policy.minimum_winning_cell_count,
                "minimum_visible_fraction": policy.minimum_visible_fraction,
            }
            for policy in SELECTED_POLICIES
        ],
        "evidence_actor_count": len(rows),
        "affected_actor_count": len(affected),
        "affected_by_winning_cell_candidate_count": affected_by_winning,
        "affected_by_visible_fraction_candidate_count": affected_by_fraction,
        "affected_clip_count": len(clip_counts),
        "change_category_counts": dict(sorted(category_counts.items())),
        "by_actor_class": {
            key: dict(sorted(values.items()))
            for key, values in sorted(class_counts.items())
        },
        "by_clip": {
            key: {
                "affected_actor_count": sum(values.values()),
                "change_category_counts": dict(sorted(values.items())),
            }
            for key, values in sorted(clip_counts.items())
        },
        "affected_rows": [item.to_dict() for item in affected],
    }
