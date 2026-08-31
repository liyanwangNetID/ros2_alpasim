#!/usr/bin/env python3
"""Direction-family guard for Step 6 branch relations.

The guard validates a raw left_of_natural/right_of_natural relation over
40 m, 60 m, and 80 m candidate-path horizons. It never rewrites a
successor identity. If the Route and Natural successor direction families
are not both stable and different, the relation becomes
family_guard_unresolved so Navigation can conservatively output unknown.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from natural_lane_corridor import (
    BranchCandidate,
    NaturalCorridorConfig,
    evaluate_branch_candidates,
)
from vector_map_reader import VectorMapReader

FAMILY_GUARD_VERSION = "natural_corridor_family_guard_v0.3"
DEFAULT_HORIZONS_M = (40.0, 60.0, 80.0)
DEFAULT_FAMILY_THRESHOLD_DEG = 15.0
DIRECTIONAL_RELATIONS = frozenset({"left_of_natural", "right_of_natural"})


@dataclass(frozen=True, slots=True)
class DirectionFamilyObservation:
    horizon_m: float
    natural_family: str | None
    route_family: str | None


@dataclass(frozen=True, slots=True)
class DirectionFamilyGuardResult:
    relation: str
    status: str
    reason: str
    family_relationship: str
    observations: tuple[DirectionFamilyObservation, ...]


def direction_family(signed_heading_change_rad: float, *, threshold_deg: float) -> str:
    signed_deg = math.degrees(float(signed_heading_change_rad))
    if signed_deg > threshold_deg:
        return "left"
    if signed_deg < -threshold_deg:
        return "right"
    return "straight"


def stable_family(values: Sequence[str | None]) -> str | None:
    if not values or any(value is None for value in values):
        return None
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


def guard_from_observations(
    *,
    raw_relation: str,
    natural_successor_lane_id: str | None,
    route_successor_lane_id: str | None,
    observations: Sequence[DirectionFamilyObservation],
) -> DirectionFamilyGuardResult:
    observation_tuple = tuple(observations)

    if raw_relation == "natural_continuation":
        if (
            natural_successor_lane_id is not None
            and route_successor_lane_id is not None
            and str(natural_successor_lane_id) == str(route_successor_lane_id)
        ):
            return DirectionFamilyGuardResult(
                relation=raw_relation,
                status="not_applicable",
                reason="successor_identity_matches_natural_continuation",
                family_relationship="same_successor",
                observations=observation_tuple,
            )
        return DirectionFamilyGuardResult(
            relation="family_guard_unresolved",
            status="unresolved",
            reason="inconsistent_natural_continuation_successor_identity",
            family_relationship="unresolved",
            observations=observation_tuple,
        )

    if raw_relation not in DIRECTIONAL_RELATIONS:
        return DirectionFamilyGuardResult(
            relation=raw_relation,
            status="not_applicable",
            reason="relation_not_directional",
            family_relationship="not_applicable",
            observations=observation_tuple,
        )

    natural = stable_family([item.natural_family for item in observation_tuple])
    route = stable_family([item.route_family for item in observation_tuple])

    if natural is None or route is None:
        return DirectionFamilyGuardResult(
            relation="family_guard_unresolved",
            status="unresolved",
            reason="direction_family_unstable_or_unavailable",
            family_relationship="family_unresolved",
            observations=observation_tuple,
        )

    if natural == route:
        return DirectionFamilyGuardResult(
            relation="family_guard_unresolved",
            status="unresolved",
            reason="route_same_direction_family_as_natural_successor",
            family_relationship="same_family_as_natural",
            observations=observation_tuple,
        )

    return DirectionFamilyGuardResult(
        relation=raw_relation,
        status="accepted_directional",
        reason="route_different_stable_direction_family_from_natural_successor",
        family_relationship="different_family_from_natural",
        observations=observation_tuple,
    )


def evaluate_direction_family_guard(
    vector_map: VectorMapReader,
    *,
    branch_lane_id: str,
    natural_successor_lane_id: str | None,
    route_successor_lane_id: str | None,
    raw_relation: str,
    incoming_heading_rad: float,
    config: NaturalCorridorConfig | None = None,
    horizons_m: Sequence[float] = DEFAULT_HORIZONS_M,
    family_threshold_deg: float = DEFAULT_FAMILY_THRESHOLD_DEG,
) -> DirectionFamilyGuardResult:
    base_config = config or NaturalCorridorConfig()
    observations: list[DirectionFamilyObservation] = []

    for horizon_m in horizons_m:
        candidates = evaluate_branch_candidates(
            vector_map,
            branch_lane_id,
            incoming_heading_rad=incoming_heading_rad,
            config=replace(
                base_config,
                branch_evaluation_distance_m=float(horizon_m),
                maximum_lookahead_m=max(
                    base_config.maximum_lookahead_m,
                    float(horizon_m),
                ),
            ),
        )
        indexed: Mapping[str, BranchCandidate] = {
            str(candidate.successor_lane_id): candidate
            for candidate in candidates
        }
        natural_candidate = indexed.get(str(natural_successor_lane_id))
        route_candidate = indexed.get(str(route_successor_lane_id))
        observations.append(
            DirectionFamilyObservation(
                horizon_m=float(horizon_m),
                natural_family=(
                    direction_family(
                        natural_candidate.signed_heading_change_rad,
                        threshold_deg=family_threshold_deg,
                    )
                    if natural_candidate is not None
                    else None
                ),
                route_family=(
                    direction_family(
                        route_candidate.signed_heading_change_rad,
                        threshold_deg=family_threshold_deg,
                    )
                    if route_candidate is not None
                    else None
                ),
            )
        )

    return guard_from_observations(
        raw_relation=raw_relation,
        natural_successor_lane_id=natural_successor_lane_id,
        route_successor_lane_id=route_successor_lane_id,
        observations=observations,
    )
