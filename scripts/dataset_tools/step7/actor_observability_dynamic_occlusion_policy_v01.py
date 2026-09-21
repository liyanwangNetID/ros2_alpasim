"""Frozen Step 7E dynamic actor-to-actor occlusion policy v01.

This policy is intentionally simple. An Actor is dynamically visible when the
existing shadow evaluator finds at least two winning raster cells. No minimum
visible-fraction floor is applied. Static-scene occlusion remains outside this
policy and final Scene Facts must preserve that limitation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from step7.actor_observability_shadow_policy_v01 import (
    ObservabilityShadowPolicy,
    evaluate_actor_observability_shadow_policies,
)

POLICY_NAME = "step7e_dynamic_occlusion_v01"
MINIMUM_WINNING_CELL_COUNT = 2
MINIMUM_VISIBLE_FRACTION = 0.0

FROZEN_DYNAMIC_OCCLUSION_POLICY = ObservabilityShadowPolicy(
    policy_name=POLICY_NAME,
    minimum_winning_cell_count=MINIMUM_WINNING_CELL_COUNT,
    minimum_visible_fraction=MINIMUM_VISIBLE_FRACTION,
)


def evaluate_frozen_dynamic_occlusion_policy(
    *, evidence_rows: Sequence[Mapping[str, Any]]
):
    """Evaluate the frozen policy without modifying source evidence."""
    return evaluate_actor_observability_shadow_policies(
        evidence_rows=tuple(evidence_rows),
        policies=(FROZEN_DYNAMIC_OCCLUSION_POLICY,),
    )
