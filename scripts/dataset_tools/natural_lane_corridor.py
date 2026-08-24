#!/usr/bin/env python3
"""Construct a map-only natural continuation corridor through lane successors.

The natural continuation is independent of future ground-truth branch choice.
At each branch, candidate successor paths are compared over a configurable
lookahead distance. The path with the smallest absolute heading change from
the incoming lane direction is selected, with deterministic tie-breaking.

This module does not classify Meta-actions and does not use camera data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from coordinate_utils import Point2D, normalize_angle
from vector_map_reader import Lane, VectorMapReader, polyline_length

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class NaturalCorridorConfig:
    maximum_lookahead_m: float = 80.0
    branch_evaluation_distance_m: float = 30.0
    maximum_lane_count: int = 20
    heading_change_weight: float = 1.0
    absolute_curvature_weight: float = 0.25
    maximum_reliable_absolute_heading_change_rad: float = math.pi
    minimum_reliable_score_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.maximum_lookahead_m <= 0.0:
            raise ValueError("maximum_lookahead_m must be positive")
        if self.branch_evaluation_distance_m <= 0.0:
            raise ValueError("branch_evaluation_distance_m must be positive")
        if self.maximum_lane_count <= 0:
            raise ValueError("maximum_lane_count must be positive")
        if self.heading_change_weight < 0.0:
            raise ValueError("heading_change_weight must be non-negative")
        if self.absolute_curvature_weight < 0.0:
            raise ValueError("absolute_curvature_weight must be non-negative")
        if self.maximum_reliable_absolute_heading_change_rad <= 0.0:
            raise ValueError(
                "maximum_reliable_absolute_heading_change_rad must be positive"
            )
        if self.minimum_reliable_score_margin < 0.0:
            raise ValueError("minimum_reliable_score_margin must be non-negative")


@dataclass(frozen=True, slots=True)
class CorridorPoint:
    x: float
    y: float
    heading_rad: float
    distance_m: float
    lane_id: str


@dataclass(frozen=True, slots=True)
class BranchCandidate:
    successor_lane_id: str
    lane_ids: tuple[str, ...]
    evaluated_distance_m: float
    signed_heading_change_rad: float
    absolute_heading_change_rad: float
    score: float


@dataclass(frozen=True, slots=True)
class BranchDecision:
    branch_lane_id: str
    incoming_heading_rad: float
    chosen_successor_lane_id: str
    candidates: tuple[BranchCandidate, ...]
    reliability_status: str
    reliability_reasons: tuple[str, ...]
    score_margin: float | None


@dataclass(frozen=True, slots=True)
class NaturalLaneCorridor:
    start_lane_id: str
    lane_ids: tuple[str, ...]
    points: tuple[CorridorPoint, ...]
    total_distance_m: float
    branch_decisions: tuple[BranchDecision, ...]
    terminated_reason: str

    @property
    def end_lane_id(self) -> str:
        return self.lane_ids[-1]


@dataclass(frozen=True, slots=True)
class ActualBranchComparison:
    branch_lane_id: str
    natural_successor_lane_id: str
    actual_successor_lane_id: str | None
    actual_matches_natural: bool | None
    actual_relation_to_natural: str


def _segment_heading(first: Point2D, second: Point2D) -> float:
    return math.atan2(second.y - first.y, second.x - first.x)


def lane_entry_heading(lane: Lane) -> float:
    if lane.centerline_headings:
        return float(lane.centerline_headings[0])
    return _segment_heading(lane.centerline[0], lane.centerline[1])


def lane_exit_heading(lane: Lane) -> float:
    if lane.centerline_headings:
        return float(lane.centerline_headings[-1])
    return _segment_heading(lane.centerline[-2], lane.centerline[-1])


def _polyline_headings(points: Sequence[Point2D]) -> tuple[float, ...]:
    return tuple(
        _segment_heading(first, second)
        for first, second in zip(points, points[1:])
        if math.hypot(second.x - first.x, second.y - first.y) > _EPSILON
    )


def _heading_metrics(
    incoming_heading_rad: float,
    points: Sequence[Point2D],
) -> tuple[float, float]:
    headings = _polyline_headings(points)
    if not headings:
        return 0.0, 0.0
    previous = incoming_heading_rad
    signed = 0.0
    absolute = 0.0
    for heading in headings:
        change = normalize_angle(heading - previous)
        signed += change
        absolute += abs(change)
        previous = heading
    return signed, absolute


def _truncate_polyline(
    points: Sequence[Point2D],
    maximum_distance_m: float,
) -> tuple[tuple[Point2D, ...], float]:
    if not points:
        return tuple(), 0.0
    result = [points[0]]
    accumulated = 0.0
    for first, second in zip(points, points[1:]):
        segment = math.hypot(second.x - first.x, second.y - first.y)
        if segment <= _EPSILON:
            continue
        remaining = maximum_distance_m - accumulated
        if remaining <= _EPSILON:
            break
        if segment <= remaining + _EPSILON:
            result.append(second)
            accumulated += segment
        else:
            ratio = remaining / segment
            result.append(
                Point2D(
                    first.x + ratio * (second.x - first.x),
                    first.y + ratio * (second.y - first.y),
                )
            )
            accumulated = maximum_distance_m
            break
    return tuple(result), accumulated


def _enumerate_successor_paths(
    vector_map: VectorMapReader,
    start_lane_id: str,
    *,
    maximum_distance_m: float,
    maximum_lane_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate acyclic successor paths until distance or lane limits."""
    vector_map.require_lane(start_lane_id)
    completed: list[tuple[str, ...]] = []
    stack: list[tuple[tuple[str, ...], float]] = [((start_lane_id,), 0.0)]
    while stack:
        path, distance_before_last = stack.pop()
        lane = vector_map.require_lane(path[-1])
        distance = distance_before_last + lane.length_m
        successors = vector_map.valid_related_lane_ids(path[-1], "successor")
        successors = tuple(
            successor for successor in successors if successor not in path
        )
        if (
            distance >= maximum_distance_m
            or len(path) >= maximum_lane_count
            or not successors
        ):
            completed.append(path)
            continue
        for successor in reversed(sorted(successors)):
            stack.append((path + (successor,), distance))
    return tuple(completed)


def _path_points(
    vector_map: VectorMapReader,
    lane_ids: Sequence[str],
    maximum_distance_m: float,
) -> tuple[tuple[Point2D, ...], float]:
    merged: list[Point2D] = []
    for lane_id in lane_ids:
        points = list(vector_map.require_lane(lane_id).centerline)
        if merged and points and merged[-1] == points[0]:
            points = points[1:]
        merged.extend(points)
    return _truncate_polyline(merged, maximum_distance_m)


def evaluate_branch_candidates(
    vector_map: VectorMapReader,
    branch_lane_id: str,
    *,
    incoming_heading_rad: float | None = None,
    config: NaturalCorridorConfig | None = None,
) -> tuple[BranchCandidate, ...]:
    config = config or NaturalCorridorConfig()
    branch_lane = vector_map.require_lane(branch_lane_id)
    incoming = (
        lane_exit_heading(branch_lane)
        if incoming_heading_rad is None
        else float(incoming_heading_rad)
    )
    successors = vector_map.valid_related_lane_ids(branch_lane_id, "successor")
    candidates: list[BranchCandidate] = []
    for successor_id in successors:
        paths = _enumerate_successor_paths(
            vector_map,
            successor_id,
            maximum_distance_m=config.branch_evaluation_distance_m,
            maximum_lane_count=config.maximum_lane_count,
        )
        path_candidates: list[BranchCandidate] = []
        for path in paths:
            points, evaluated_distance = _path_points(
                vector_map,
                path,
                config.branch_evaluation_distance_m,
            )
            signed, absolute = _heading_metrics(incoming, points)
            score = (
                config.heading_change_weight * abs(signed)
                + config.absolute_curvature_weight * absolute
            )
            path_candidates.append(
                BranchCandidate(
                    successor_lane_id=successor_id,
                    lane_ids=tuple(path),
                    evaluated_distance_m=evaluated_distance,
                    signed_heading_change_rad=signed,
                    absolute_heading_change_rad=absolute,
                    score=score,
                )
            )
        if path_candidates:
            candidates.append(
                min(
                    path_candidates,
                    key=lambda item: (
                        item.score,
                        -item.evaluated_distance_m,
                        item.lane_ids,
                    ),
                )
            )
    candidates.sort(
        key=lambda item: (
            item.score,
            -item.evaluated_distance_m,
            item.successor_lane_id,
        )
    )
    return tuple(candidates)



def assess_branch_candidate_reliability(
    candidates: Sequence[BranchCandidate],
    *,
    config: NaturalCorridorConfig | None = None,
) -> tuple[str, tuple[str, ...], float | None]:
    """Assess whether a natural-successor choice is safe to use for labels.

    Reliability is deliberately conservative. Abnormal accumulated curvature,
    insufficient candidate separation, or too few candidates prevents a hard
    left/right turn decision and must fall back to keep_direction downstream.
    """
    config = config or NaturalCorridorConfig()
    reasons: list[str] = []
    if len(candidates) < 2:
        reasons.append("fewer_than_two_branch_candidates")
        return "unavailable", tuple(reasons), None

    chosen = candidates[0]
    score_margin = candidates[1].score - chosen.score
    if any(
        candidate.absolute_heading_change_rad
        > config.maximum_reliable_absolute_heading_change_rad
        for candidate in candidates
    ):
        reasons.append("abnormal_candidate_absolute_heading_change")
    if score_margin < config.minimum_reliable_score_margin:
        reasons.append("insufficient_candidate_score_margin")

    return (
        ("reliable" if not reasons else "unreliable"),
        tuple(reasons),
        score_margin,
    )

def choose_natural_successor(
    vector_map: VectorMapReader,
    branch_lane_id: str,
    *,
    incoming_heading_rad: float | None = None,
    config: NaturalCorridorConfig | None = None,
) -> tuple[str | None, tuple[BranchCandidate, ...]]:
    candidates = evaluate_branch_candidates(
        vector_map,
        branch_lane_id,
        incoming_heading_rad=incoming_heading_rad,
        config=config,
    )
    return (candidates[0].successor_lane_id if candidates else None, candidates)


def _append_lane_points(
    vector_map: VectorMapReader,
    lane_id: str,
    current_distance_m: float,
    maximum_distance_m: float,
    output: list[CorridorPoint],
    *,
    start_arc_length_m: float = 0.0,
) -> float:
    """Append a lane beginning at an arc-length offset on its centerline."""
    lane = vector_map.require_lane(lane_id)
    if start_arc_length_m < 0.0:
        raise ValueError("start_arc_length_m must be non-negative")
    if start_arc_length_m > lane.length_m + _EPSILON:
        raise ValueError(
            f"start_arc_length_m exceeds lane length for {lane_id}: "
            f"{start_arc_length_m} > {lane.length_m}"
        )

    traversed_on_lane = 0.0
    for first, second in zip(lane.centerline, lane.centerline[1:]):
        segment = math.hypot(second.x - first.x, second.y - first.y)
        if segment <= _EPSILON:
            continue
        segment_start = traversed_on_lane
        segment_end = traversed_on_lane + segment
        traversed_on_lane = segment_end
        if segment_end < start_arc_length_m - _EPSILON:
            continue

        local_start = first
        usable_segment = segment
        if start_arc_length_m > segment_start + _EPSILON:
            ratio = (start_arc_length_m - segment_start) / segment
            local_start = Point2D(
                first.x + ratio * (second.x - first.x),
                first.y + ratio * (second.y - first.y),
            )
            usable_segment = math.hypot(
                second.x - local_start.x,
                second.y - local_start.y,
            )
        if usable_segment <= _EPSILON:
            continue

        remaining = maximum_distance_m - current_distance_m
        if remaining <= _EPSILON:
            return current_distance_m
        heading = _segment_heading(local_start, second)
        if not output:
            output.append(
                CorridorPoint(
                    local_start.x, local_start.y, heading, current_distance_m, lane_id
                )
            )
        if usable_segment <= remaining + _EPSILON:
            current_distance_m += usable_segment
            output.append(
                CorridorPoint(
                    second.x, second.y, heading, current_distance_m, lane_id
                )
            )
        else:
            ratio = remaining / usable_segment
            current_distance_m = maximum_distance_m
            output.append(
                CorridorPoint(
                    local_start.x + ratio * (second.x - local_start.x),
                    local_start.y + ratio * (second.y - local_start.y),
                    heading,
                    current_distance_m,
                    lane_id,
                )
            )
            return current_distance_m
        start_arc_length_m = segment_end
    return current_distance_m


def build_natural_lane_corridor(
    vector_map: VectorMapReader,
    start_lane_id: str,
    *,
    lookahead_distance_m: float,
    start_arc_length_m: float = 0.0,
    config: NaturalCorridorConfig | None = None,
) -> NaturalLaneCorridor:
    config = config or NaturalCorridorConfig()
    if lookahead_distance_m <= 0.0:
        raise ValueError("lookahead_distance_m must be positive")
    maximum_distance = min(lookahead_distance_m, config.maximum_lookahead_m)
    start_lane = vector_map.require_lane(start_lane_id)
    if not 0.0 <= start_arc_length_m <= start_lane.length_m + _EPSILON:
        raise ValueError(
            "start_arc_length_m must lie on the start lane centerline"
        )
    current_lane_id = start_lane.lane_id
    lane_ids: list[str] = []
    points: list[CorridorPoint] = []
    decisions: list[BranchDecision] = []
    visited: set[str] = set()
    distance = 0.0
    terminated_reason = "lookahead_reached"

    while distance < maximum_distance - _EPSILON:
        if current_lane_id in visited:
            terminated_reason = "cycle_detected"
            break
        if len(lane_ids) >= config.maximum_lane_count:
            terminated_reason = "maximum_lane_count_reached"
            break
        visited.add(current_lane_id)
        lane_ids.append(current_lane_id)
        distance = _append_lane_points(
            vector_map,
            current_lane_id,
            distance,
            maximum_distance,
            points,
            start_arc_length_m=(
                start_arc_length_m if current_lane_id == start_lane.lane_id else 0.0
            ),
        )
        if distance >= maximum_distance - _EPSILON:
            terminated_reason = "lookahead_reached"
            break

        successors = vector_map.valid_related_lane_ids(current_lane_id, "successor")
        if not successors:
            terminated_reason = "no_successor"
            break
        if len(successors) == 1:
            current_lane_id = successors[0]
            continue

        incoming_heading = lane_exit_heading(
            vector_map.require_lane(current_lane_id)
        )
        chosen, candidates = choose_natural_successor(
            vector_map,
            current_lane_id,
            incoming_heading_rad=incoming_heading,
            config=config,
        )
        if chosen is None:
            terminated_reason = "branch_without_valid_candidate"
            break
        reliability_status, reliability_reasons, score_margin = (
            assess_branch_candidate_reliability(candidates, config=config)
        )
        decisions.append(
            BranchDecision(
                branch_lane_id=current_lane_id,
                incoming_heading_rad=incoming_heading,
                chosen_successor_lane_id=chosen,
                candidates=candidates,
                reliability_status=reliability_status,
                reliability_reasons=reliability_reasons,
                score_margin=score_margin,
            )
        )
        current_lane_id = chosen

    return NaturalLaneCorridor(
        start_lane_id=str(start_lane_id),
        lane_ids=tuple(lane_ids),
        points=tuple(points),
        total_distance_m=distance,
        branch_decisions=tuple(decisions),
        terminated_reason=terminated_reason,
    )


def recover_boundary_branch_comparisons(
    vector_map: VectorMapReader,
    start_lane_id: str,
    *,
    config: NaturalCorridorConfig | None = None,
) -> tuple[ActualBranchComparison, ...]:
    """Recover a branch choice immediately before the observed start lane.

    This supports windows that begin after a divergence. For each valid
    predecessor with multiple successors, the function compares the observed
    start lane with the predecessor's map-only natural successor.
    """
    config = config or NaturalCorridorConfig()
    start_lane_id = vector_map.require_lane(start_lane_id).lane_id
    comparisons: list[ActualBranchComparison] = []
    for predecessor_id in vector_map.valid_related_lane_ids(
        start_lane_id, "predecessor"
    ):
        successors = vector_map.valid_related_lane_ids(
            predecessor_id, "successor"
        )
        if len(successors) <= 1 or start_lane_id not in successors:
            continue
        natural_successor, candidates = choose_natural_successor(
            vector_map,
            predecessor_id,
            incoming_heading_rad=lane_exit_heading(
                vector_map.require_lane(predecessor_id)
            ),
            config=config,
        )
        if natural_successor is None:
            continue
        reliability_status, _, _ = assess_branch_candidate_reliability(
            candidates, config=config
        )
        if reliability_status != "reliable":
            relation = "natural_continuation_uncertain"
            matches = None
        elif start_lane_id == natural_successor:
            relation = "natural_continuation"
            matches = True
        else:
            matches = False
            natural_candidate = next(
                candidate
                for candidate in candidates
                if candidate.successor_lane_id == natural_successor
            )
            actual_candidate = next(
                candidate
                for candidate in candidates
                if candidate.successor_lane_id == start_lane_id
            )
            difference = normalize_angle(
                actual_candidate.signed_heading_change_rad
                - natural_candidate.signed_heading_change_rad
            )
            if abs(difference) <= _EPSILON:
                relation = "ambiguous_relative_direction"
            else:
                relation = (
                    "left_of_natural" if difference > 0.0
                    else "right_of_natural"
                )
        comparisons.append(
            ActualBranchComparison(
                branch_lane_id=predecessor_id,
                natural_successor_lane_id=natural_successor,
                actual_successor_lane_id=start_lane_id,
                actual_matches_natural=matches,
                actual_relation_to_natural=relation,
            )
        )
    comparisons.sort(
        key=lambda item: (
            item.branch_lane_id,
            item.natural_successor_lane_id,
        )
    )
    return tuple(comparisons)


def compare_actual_lane_sequence(
    actual_lane_sequence: Sequence[str],
    corridor: NaturalLaneCorridor,
) -> tuple[ActualBranchComparison, ...]:
    """Compare actual map-matched branch choices with natural decisions."""
    comparisons: list[ActualBranchComparison] = []
    for decision in corridor.branch_decisions:
        actual_successor: str | None = None
        try:
            index = list(actual_lane_sequence).index(decision.branch_lane_id)
        except ValueError:
            index = -1
        if index >= 0 and index + 1 < len(actual_lane_sequence):
            actual_successor = str(actual_lane_sequence[index + 1])

        if actual_successor is None:
            matches: bool | None = None
            relation = "not_observed"
        elif actual_successor == decision.chosen_successor_lane_id:
            matches = True
            relation = "natural_continuation"
        else:
            matches = False
            natural_candidate = next(
                candidate
                for candidate in decision.candidates
                if candidate.successor_lane_id == decision.chosen_successor_lane_id
            )
            actual_candidate = next(
                (
                    candidate
                    for candidate in decision.candidates
                    if candidate.successor_lane_id == actual_successor
                ),
                None,
            )
            if actual_candidate is None:
                relation = "actual_successor_not_candidate"
            else:
                difference = normalize_angle(
                    actual_candidate.signed_heading_change_rad
                    - natural_candidate.signed_heading_change_rad
                )
                relation = "left_of_natural" if difference > 0.0 else "right_of_natural"

        comparisons.append(
            ActualBranchComparison(
                branch_lane_id=decision.branch_lane_id,
                natural_successor_lane_id=decision.chosen_successor_lane_id,
                actual_successor_lane_id=actual_successor,
                actual_matches_natural=matches,
                actual_relation_to_natural=relation,
            )
        )
    return tuple(comparisons)
