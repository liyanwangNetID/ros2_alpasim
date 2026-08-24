#!/usr/bin/env python3
"""Topology-aware lane matching for AlpaSim ego trajectories.

This module matches timestamped map-frame trajectory poses to VectorMap lanes.
It combines lane-polygon containment, centerline distance, heading agreement,
and temporal topology consistency. It does not classify meta-actions itself.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from coordinate_utils import Point2D, normalize_angle, pose2d_from_pose_mapping
from vector_map_reader import NearbyLane, VectorMapReader


@dataclass(frozen=True, slots=True)
class LaneMatcherConfig:
    search_radius_m: float = 6.0
    maximum_heading_error_rad: float = math.radians(70.0)
    maximum_candidates_per_point: int = 6
    polygon_outside_penalty: float = 1.0
    distance_weight: float = 1.0
    heading_weight: float = 2.0
    same_lane_transition_cost: float = 0.0
    successor_transition_cost: float = 0.15
    adjacent_transition_cost: float = 0.45
    predecessor_transition_cost: float = 2.0
    unrelated_transition_cost: float = 6.0
    unmatched_emission_cost: float = 5.0
    unmatched_transition_cost: float = 1.0
    minimum_match_confidence: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "search_radius_m": self.search_radius_m,
            "maximum_heading_error_rad": self.maximum_heading_error_rad,
            "maximum_candidates_per_point": self.maximum_candidates_per_point,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        nonnegative = {
            "polygon_outside_penalty": self.polygon_outside_penalty,
            "distance_weight": self.distance_weight,
            "heading_weight": self.heading_weight,
            "same_lane_transition_cost": self.same_lane_transition_cost,
            "successor_transition_cost": self.successor_transition_cost,
            "adjacent_transition_cost": self.adjacent_transition_cost,
            "predecessor_transition_cost": self.predecessor_transition_cost,
            "unrelated_transition_cost": self.unrelated_transition_cost,
            "unmatched_emission_cost": self.unmatched_emission_cost,
            "unmatched_transition_cost": self.unmatched_transition_cost,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.minimum_match_confidence <= 1.0:
            raise ValueError("minimum_match_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrajectoryPose:
    stamp_ns: int
    x: float
    y: float
    yaw: float


@dataclass(frozen=True, slots=True)
class LaneCandidate:
    lane_id: str | None
    emission_cost: float
    distance_m: float | None
    heading_error_rad: float | None
    inside_polygon: bool


@dataclass(frozen=True, slots=True)
class MatchedTrajectoryPoint:
    stamp_ns: int
    x: float
    y: float
    yaw: float
    lane_id: str | None
    distance_m: float | None
    heading_error_rad: float | None
    inside_polygon: bool
    emission_cost: float
    point_confidence: float


@dataclass(frozen=True, slots=True)
class LaneTransition:
    source_lane_id: str
    target_lane_id: str
    relation: str
    source_point_index: int
    target_point_index: int


@dataclass(frozen=True, slots=True)
class LaneMatchResult:
    points: tuple[MatchedTrajectoryPoint, ...]
    compressed_lane_sequence: tuple[str, ...]
    transitions: tuple[LaneTransition, ...]
    matched_point_count: int
    unmatched_point_count: int
    matched_fraction: float
    mean_distance_m: float | None
    maximum_distance_m: float | None
    mean_heading_error_rad: float | None
    total_cost: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stamp_mapping_to_ns(stamp: Mapping[str, Any]) -> int:
    sec = stamp.get("sec")
    nanosec = stamp.get("nanosec")
    if isinstance(sec, bool) or not isinstance(sec, int):
        raise ValueError("stamp.sec must be an integer")
    if isinstance(nanosec, bool) or not isinstance(nanosec, int):
        raise ValueError("stamp.nanosec must be an integer")
    if not 0 <= nanosec < 1_000_000_000:
        raise ValueError("stamp.nanosec must be in [0, 1e9)")
    return sec * 1_000_000_000 + nanosec


def trajectory_pose_from_mapping(point: Mapping[str, Any]) -> TrajectoryPose:
    pose = pose2d_from_pose_mapping(point["pose"])
    return TrajectoryPose(
        stamp_ns=stamp_mapping_to_ns(point["stamp"]),
        x=pose.x,
        y=pose.y,
        yaw=pose.yaw,
    )


def trajectory_poses_from_gt_points(
    points: Sequence[Mapping[str, Any]],
) -> tuple[TrajectoryPose, ...]:
    poses = tuple(trajectory_pose_from_mapping(point) for point in points)
    if not poses:
        return tuple()
    for previous, current in zip(poses, poses[1:]):
        if current.stamp_ns <= previous.stamp_ns:
            raise ValueError("trajectory timestamps must be strictly increasing")
    return poses


class LaneMatcher:
    """Viterbi-style topology-aware lane matcher."""

    def __init__(
        self,
        vector_map: VectorMapReader,
        config: LaneMatcherConfig | None = None,
    ) -> None:
        self.vector_map = vector_map
        self.config = config or LaneMatcherConfig()

    def _emission_cost(self, candidate: NearbyLane) -> float:
        heading_error = candidate.heading_error_rad or 0.0
        return (
            self.config.distance_weight * candidate.distance_m
            + self.config.heading_weight * heading_error
            + (
                0.0
                if candidate.inside_polygon
                else self.config.polygon_outside_penalty
            )
        )

    def candidates_for_pose(
        self,
        pose: TrajectoryPose,
    ) -> tuple[LaneCandidate, ...]:
        nearby = self.vector_map.find_nearby_lanes(
            pose.x,
            pose.y,
            radius_m=self.config.search_radius_m,
            yaw_rad=pose.yaw,
            maximum_heading_error_rad=(
                self.config.maximum_heading_error_rad
            ),
            limit=self.config.maximum_candidates_per_point,
        )
        candidates = [
            LaneCandidate(
                lane_id=item.lane_id,
                emission_cost=self._emission_cost(item),
                distance_m=item.distance_m,
                heading_error_rad=item.heading_error_rad,
                inside_polygon=item.inside_polygon,
            )
            for item in nearby
        ]
        candidates.append(
            LaneCandidate(
                lane_id=None,
                emission_cost=self.config.unmatched_emission_cost,
                distance_m=None,
                heading_error_rad=None,
                inside_polygon=False,
            )
        )
        return tuple(candidates)

    def transition_relation(
        self,
        source_lane_id: str | None,
        target_lane_id: str | None,
    ) -> str:
        if source_lane_id is None or target_lane_id is None:
            return "unmatched"
        return self.vector_map.relation(source_lane_id, target_lane_id)

    def transition_cost(
        self,
        source_lane_id: str | None,
        target_lane_id: str | None,
    ) -> float:
        relation = self.transition_relation(source_lane_id, target_lane_id)
        if relation == "same":
            return self.config.same_lane_transition_cost
        if relation == "successor":
            return self.config.successor_transition_cost
        if relation in ("left_adjacent", "right_adjacent"):
            return self.config.adjacent_transition_cost
        if relation == "predecessor":
            return self.config.predecessor_transition_cost
        if relation == "unmatched":
            return self.config.unmatched_transition_cost
        return self.config.unrelated_transition_cost

    @staticmethod
    def _point_confidence(candidate: LaneCandidate) -> float:
        if candidate.lane_id is None:
            return 0.0
        return 1.0 / (1.0 + candidate.emission_cost)

    def match(
        self,
        trajectory: Sequence[TrajectoryPose],
    ) -> LaneMatchResult:
        if not trajectory:
            return LaneMatchResult(
                points=tuple(),
                compressed_lane_sequence=tuple(),
                transitions=tuple(),
                matched_point_count=0,
                unmatched_point_count=0,
                matched_fraction=0.0,
                mean_distance_m=None,
                maximum_distance_m=None,
                mean_heading_error_rad=None,
                total_cost=0.0,
                confidence=0.0,
            )

        candidate_rows = [self.candidates_for_pose(pose) for pose in trajectory]
        costs: list[list[float]] = []
        backpointers: list[list[int | None]] = []

        first_costs = [candidate.emission_cost for candidate in candidate_rows[0]]
        costs.append(first_costs)
        backpointers.append([None] * len(first_costs))

        for row_index in range(1, len(candidate_rows)):
            previous_candidates = candidate_rows[row_index - 1]
            current_candidates = candidate_rows[row_index]
            current_costs: list[float] = []
            current_backpointers: list[int | None] = []
            for current in current_candidates:
                choices = [
                    (
                        costs[row_index - 1][previous_index]
                        + self.transition_cost(
                            previous.lane_id,
                            current.lane_id,
                        ),
                        previous_index,
                    )
                    for previous_index, previous in enumerate(previous_candidates)
                ]
                best_previous_cost, best_previous_index = min(
                    choices,
                    key=lambda value: (value[0], value[1]),
                )
                current_costs.append(
                    best_previous_cost + current.emission_cost
                )
                current_backpointers.append(best_previous_index)
            costs.append(current_costs)
            backpointers.append(current_backpointers)

        last_index = min(
            range(len(costs[-1])),
            key=lambda index: (costs[-1][index], index),
        )
        total_cost = costs[-1][last_index]
        selected_indexes = [last_index]
        for row_index in range(len(candidate_rows) - 1, 0, -1):
            previous_index = backpointers[row_index][selected_indexes[-1]]
            if previous_index is None:
                raise RuntimeError("lane matcher backpointer is unexpectedly missing")
            selected_indexes.append(previous_index)
        selected_indexes.reverse()

        matched_points: list[MatchedTrajectoryPoint] = []
        for pose, candidates, selected_index in zip(
            trajectory,
            candidate_rows,
            selected_indexes,
        ):
            candidate = candidates[selected_index]
            matched_points.append(
                MatchedTrajectoryPoint(
                    stamp_ns=pose.stamp_ns,
                    x=pose.x,
                    y=pose.y,
                    yaw=pose.yaw,
                    lane_id=candidate.lane_id,
                    distance_m=candidate.distance_m,
                    heading_error_rad=candidate.heading_error_rad,
                    inside_polygon=candidate.inside_polygon,
                    emission_cost=candidate.emission_cost,
                    point_confidence=self._point_confidence(candidate),
                )
            )

        compressed: list[str] = []
        lane_start_indexes: list[int] = []
        for point_index, point in enumerate(matched_points):
            if point.lane_id is None:
                continue
            if not compressed or compressed[-1] != point.lane_id:
                compressed.append(point.lane_id)
                lane_start_indexes.append(point_index)

        transitions: list[LaneTransition] = []
        for index in range(1, len(compressed)):
            source = compressed[index - 1]
            target = compressed[index]
            transitions.append(
                LaneTransition(
                    source_lane_id=source,
                    target_lane_id=target,
                    relation=self.vector_map.relation(source, target),
                    source_point_index=lane_start_indexes[index] - 1,
                    target_point_index=lane_start_indexes[index],
                )
            )

        valid_distances = [
            point.distance_m
            for point in matched_points
            if point.distance_m is not None
        ]
        valid_heading_errors = [
            point.heading_error_rad
            for point in matched_points
            if point.heading_error_rad is not None
        ]
        matched_count = sum(point.lane_id is not None for point in matched_points)
        unmatched_count = len(matched_points) - matched_count
        matched_fraction = matched_count / len(matched_points)
        mean_point_confidence = (
            sum(point.point_confidence for point in matched_points)
            / len(matched_points)
        )
        confidence = matched_fraction * mean_point_confidence
        if confidence < self.config.minimum_match_confidence:
            confidence = 0.0

        return LaneMatchResult(
            points=tuple(matched_points),
            compressed_lane_sequence=tuple(compressed),
            transitions=tuple(transitions),
            matched_point_count=matched_count,
            unmatched_point_count=unmatched_count,
            matched_fraction=matched_fraction,
            mean_distance_m=(
                sum(valid_distances) / len(valid_distances)
                if valid_distances
                else None
            ),
            maximum_distance_m=(
                max(valid_distances) if valid_distances else None
            ),
            mean_heading_error_rad=(
                sum(valid_heading_errors) / len(valid_heading_errors)
                if valid_heading_errors
                else None
            ),
            total_cost=total_cost,
            confidence=confidence,
        )
