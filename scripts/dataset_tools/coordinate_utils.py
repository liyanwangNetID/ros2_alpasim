#!/usr/bin/env python3
"""Planar coordinate and trajectory geometry utilities for AlpaSim data.

Conventions follow ROS REP-103 for body frames: x forward, y left, z up.
Planar yaw is positive counter-clockwise about +z. Raw map-frame poses are
converted to an anchor ego-local frame using the anchor pose as origin.

This module is independent of ROS 2 and accepts primitive numeric values or
JSON-like dictionaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True, slots=True)
class RelativePose2D:
    relative_x: float
    relative_y: float
    relative_yaw: float


@dataclass(frozen=True, slots=True)
class TrajectoryGeometry:
    point_count: int
    path_length: float
    displacement: float
    heading_change: float
    lateral_displacement: float
    mean_absolute_curvature: float | None
    maximum_absolute_curvature: float | None


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def normalize_angle(angle: float) -> float:
    """Normalize a radian angle to [-pi, pi)."""
    value = _finite_float(angle, "angle")
    normalized = (value + math.pi) % (2.0 * math.pi) - math.pi
    if normalized >= math.pi:
        normalized -= 2.0 * math.pi
    return normalized


def shortest_angular_distance(from_angle: float, to_angle: float) -> float:
    """Return the signed shortest rotation from from_angle to to_angle."""
    return normalize_angle(
        _finite_float(to_angle, "to_angle")
        - _finite_float(from_angle, "from_angle")
    )


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract planar yaw from a quaternion after normalization."""
    qx = _finite_float(x, "quaternion.x")
    qy = _finite_float(y, "quaternion.y")
    qz = _finite_float(z, "quaternion.z")
    qw = _finite_float(w, "quaternion.w")
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= _EPSILON:
        raise ValueError("quaternion norm must be positive")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return normalize_angle(math.atan2(sin_yaw, cos_yaw))


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Create a planar quaternion (x, y, z, w) from yaw."""
    angle = _finite_float(yaw, "yaw")
    half = 0.5 * angle
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_orientation(orientation: Mapping[str, Any]) -> float:
    """Read x/y/z/w from a JSON-like orientation mapping."""
    return quaternion_to_yaw(
        orientation["x"],
        orientation["y"],
        orientation["z"],
        orientation["w"],
    )


def pose2d_from_pose_mapping(pose: Mapping[str, Any]) -> Pose2D:
    """Convert a geometry_msgs/Pose-like mapping to Pose2D."""
    position = pose["position"]
    orientation = pose["orientation"]
    return Pose2D(
        x=_finite_float(position["x"], "pose.position.x"),
        y=_finite_float(position["y"], "pose.position.y"),
        yaw=yaw_from_orientation(orientation),
    )


def rotate_vector_2d(x: float, y: float, angle: float) -> Point2D:
    """Rotate a planar vector counter-clockwise by angle."""
    px = _finite_float(x, "x")
    py = _finite_float(y, "y")
    theta = _finite_float(angle, "angle")
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return Point2D(
        x=cosine * px - sine * py,
        y=sine * px + cosine * py,
    )


def map_point_to_anchor_ego(
    point_x: float,
    point_y: float,
    anchor_x: float,
    anchor_y: float,
    anchor_yaw: float,
) -> Point2D:
    """Transform a map-frame point into the anchor ego-local frame."""
    dx = _finite_float(point_x, "point_x") - _finite_float(anchor_x, "anchor_x")
    dy = _finite_float(point_y, "point_y") - _finite_float(anchor_y, "anchor_y")
    return rotate_vector_2d(dx, dy, -_finite_float(anchor_yaw, "anchor_yaw"))


def anchor_ego_point_to_map(
    relative_x: float,
    relative_y: float,
    anchor_x: float,
    anchor_y: float,
    anchor_yaw: float,
) -> Point2D:
    """Transform an anchor ego-local point into the map frame."""
    rotated = rotate_vector_2d(relative_x, relative_y, anchor_yaw)
    return Point2D(
        x=rotated.x + _finite_float(anchor_x, "anchor_x"),
        y=rotated.y + _finite_float(anchor_y, "anchor_y"),
    )


def map_pose_to_anchor_ego(pose: Pose2D, anchor: Pose2D) -> RelativePose2D:
    """Transform a map-frame planar pose into anchor ego-local coordinates."""
    point = map_point_to_anchor_ego(
        pose.x,
        pose.y,
        anchor.x,
        anchor.y,
        anchor.yaw,
    )
    return RelativePose2D(
        relative_x=point.x,
        relative_y=point.y,
        relative_yaw=shortest_angular_distance(anchor.yaw, pose.yaw),
    )


def anchor_ego_pose_to_map(
    relative_pose: RelativePose2D,
    anchor: Pose2D,
) -> Pose2D:
    """Inverse of map_pose_to_anchor_ego."""
    point = anchor_ego_point_to_map(
        relative_pose.relative_x,
        relative_pose.relative_y,
        anchor.x,
        anchor.y,
        anchor.yaw,
    )
    return Pose2D(
        x=point.x,
        y=point.y,
        yaw=normalize_angle(anchor.yaw + relative_pose.relative_yaw),
    )


def map_vector_to_anchor_ego(
    vector_x: float,
    vector_y: float,
    anchor_yaw: float,
) -> Point2D:
    """Rotate a map-frame vector into anchor ego-local coordinates."""
    return rotate_vector_2d(vector_x, vector_y, -anchor_yaw)


def euclidean_distance_2d(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
) -> float:
    return math.hypot(
        _finite_float(second_x, "second_x") - _finite_float(first_x, "first_x"),
        _finite_float(second_y, "second_y") - _finite_float(first_y, "first_y"),
    )


def planar_heading(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
) -> float:
    """Heading from the first point to the second point."""
    dx = _finite_float(second_x, "second_x") - _finite_float(first_x, "first_x")
    dy = _finite_float(second_y, "second_y") - _finite_float(first_y, "first_y")
    if math.hypot(dx, dy) <= _EPSILON:
        raise ValueError("heading is undefined for coincident points")
    return math.atan2(dy, dx)


def cumulative_distances(points: Sequence[Point2D]) -> tuple[float, ...]:
    """Return cumulative planar path distance beginning at zero."""
    if not points:
        return tuple()
    distances = [0.0]
    for previous, current in zip(points, points[1:]):
        distances.append(
            distances[-1]
            + euclidean_distance_2d(previous.x, previous.y, current.x, current.y)
        )
    return tuple(distances)


def unwrap_angles(angles: Iterable[float]) -> tuple[float, ...]:
    """Unwrap a radian sequence using shortest angular increments."""
    values = tuple(_finite_float(value, "angle") for value in angles)
    if not values:
        return tuple()
    unwrapped = [values[0]]
    for current in values[1:]:
        increment = shortest_angular_distance(unwrapped[-1], current)
        unwrapped.append(unwrapped[-1] + increment)
    return tuple(unwrapped)


def interpolate_scalar(
    first_value: float,
    second_value: float,
    ratio: float,
) -> float:
    """Linearly interpolate without clamping ratio."""
    first = _finite_float(first_value, "first_value")
    second = _finite_float(second_value, "second_value")
    alpha = _finite_float(ratio, "ratio")
    return first + alpha * (second - first)


def interpolate_angle(
    first_angle: float,
    second_angle: float,
    ratio: float,
) -> float:
    """Interpolate along the shortest angular path."""
    first = _finite_float(first_angle, "first_angle")
    alpha = _finite_float(ratio, "ratio")
    delta = shortest_angular_distance(first, second_angle)
    return normalize_angle(first + alpha * delta)


def interpolate_pose2d(first: Pose2D, second: Pose2D, ratio: float) -> Pose2D:
    """Interpolate planar position and shortest-path yaw."""
    return Pose2D(
        x=interpolate_scalar(first.x, second.x, ratio),
        y=interpolate_scalar(first.y, second.y, ratio),
        yaw=interpolate_angle(first.yaw, second.yaw, ratio),
    )


def relative_region(
    relative_x: float,
    relative_y: float,
    *,
    longitudinal_deadband: float = 0.5,
    lateral_deadband: float = 0.5,
) -> str:
    """Return a coarse ego-relative region using x-forward/y-left axes.

    Possible values are: ahead, behind, left, right, ahead_left,
    ahead_right, behind_left, behind_right, overlap.
    """
    x = _finite_float(relative_x, "relative_x")
    y = _finite_float(relative_y, "relative_y")
    x_deadband = _finite_float(longitudinal_deadband, "longitudinal_deadband")
    y_deadband = _finite_float(lateral_deadband, "lateral_deadband")
    if x_deadband < 0.0 or y_deadband < 0.0:
        raise ValueError("deadbands must be non-negative")

    longitudinal = ""
    lateral = ""
    if x > x_deadband:
        longitudinal = "ahead"
    elif x < -x_deadband:
        longitudinal = "behind"
    if y > y_deadband:
        lateral = "left"
    elif y < -y_deadband:
        lateral = "right"

    if longitudinal and lateral:
        return f"{longitudinal}_{lateral}"
    if longitudinal:
        return longitudinal
    if lateral:
        return lateral
    return "overlap"


def signed_lateral_offset_to_segment(
    point: Point2D,
    segment_start: Point2D,
    segment_end: Point2D,
) -> float:
    """Signed perpendicular distance; positive is left of segment direction."""
    dx = segment_end.x - segment_start.x
    dy = segment_end.y - segment_start.y
    length = math.hypot(dx, dy)
    if length <= _EPSILON:
        raise ValueError("segment must have positive length")
    px = point.x - segment_start.x
    py = point.y - segment_start.y
    cross = dx * py - dy * px
    return cross / length


def trajectory_geometry(points: Sequence[Point2D]) -> TrajectoryGeometry:
    """Compute coarse planar route or trajectory geometry."""
    count = len(points)
    if count == 0:
        return TrajectoryGeometry(0, 0.0, 0.0, 0.0, 0.0, None, None)
    if count == 1:
        return TrajectoryGeometry(1, 0.0, 0.0, 0.0, 0.0, None, None)

    distances = cumulative_distances(points)
    headings: list[float] = []
    segment_lengths: list[float] = []
    for previous, current in zip(points, points[1:]):
        length = euclidean_distance_2d(previous.x, previous.y, current.x, current.y)
        if length <= _EPSILON:
            continue
        segment_lengths.append(length)
        headings.append(planar_heading(previous.x, previous.y, current.x, current.y))

    displacement = euclidean_distance_2d(
        points[0].x,
        points[0].y,
        points[-1].x,
        points[-1].y,
    )
    lateral_displacement = points[-1].y - points[0].y

    if not headings:
        return TrajectoryGeometry(
            count,
            distances[-1],
            displacement,
            0.0,
            lateral_displacement,
            None,
            None,
        )

    unwrapped = unwrap_angles(headings)
    heading_change = unwrapped[-1] - unwrapped[0]
    curvatures: list[float] = []
    for position in range(1, len(unwrapped)):
        distance = 0.5 * (
            segment_lengths[position - 1] + segment_lengths[position]
        )
        if distance > _EPSILON:
            curvatures.append((unwrapped[position] - unwrapped[position - 1]) / distance)

    absolute_curvatures = [abs(value) for value in curvatures]
    mean_curvature = (
        sum(absolute_curvatures) / len(absolute_curvatures)
        if absolute_curvatures
        else None
    )
    maximum_curvature = max(absolute_curvatures) if absolute_curvatures else None
    return TrajectoryGeometry(
        point_count=count,
        path_length=distances[-1],
        displacement=displacement,
        heading_change=heading_change,
        lateral_displacement=lateral_displacement,
        mean_absolute_curvature=mean_curvature,
        maximum_absolute_curvature=maximum_curvature,
    )
