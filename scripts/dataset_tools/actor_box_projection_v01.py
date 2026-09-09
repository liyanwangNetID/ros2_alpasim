#!/usr/bin/env python3
"""Step 7D Actor oriented-box construction and map-to-rig transformation.

ActorState explicitly defines pose as the AABB center and dimensions as full
length, width, and height along the Actor box x/y/z axes. This module therefore
constructs local corners with symmetric half extents and applies the complete
Actor quaternion. It does not yet create a final image bounding box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from camera_projection_v01 import (
    Quaternion,
    Vector3,
    normalize_quaternion,
    rotate_vector_by_quaternion,
    rotate_vector_by_quaternion_inverse,
)


@dataclass(frozen=True, slots=True)
class Pose3D:
    position: Vector3
    orientation: Quaternion


@dataclass(frozen=True, slots=True)
class ActorBox3D:
    track_id: str
    actor_class: str
    pose_map: Pose3D
    dimensions: Vector3
    corners_map: tuple[Vector3, ...]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def parse_pose3d(pose: Mapping[str, Any], *, name: str) -> Pose3D:
    position = _mapping(pose.get("position"), f"{name}.position")
    orientation = _mapping(pose.get("orientation"), f"{name}.orientation")
    return Pose3D(
        position=Vector3(
            _finite(position.get("x"), f"{name}.position.x"),
            _finite(position.get("y"), f"{name}.position.y"),
            _finite(position.get("z"), f"{name}.position.z"),
        ),
        orientation=normalize_quaternion(
            Quaternion(
                _finite(orientation.get("x"), f"{name}.orientation.x"),
                _finite(orientation.get("y"), f"{name}.orientation.y"),
                _finite(orientation.get("z"), f"{name}.orientation.z"),
                _finite(orientation.get("w"), f"{name}.orientation.w"),
            )
        ),
    )


def parse_recorded_ego_pose3d(message: Mapping[str, Any]) -> Pose3D:
    if message.get("pose_frame_id") != "map":
        raise ValueError("recorded Ego pose_frame_id must be 'map'")
    return parse_pose3d(
        {
            "position": _mapping(message.get("position"), "recorded_ego.position"),
            "orientation": _mapping(
                message.get("orientation"),
                "recorded_ego.orientation",
            ),
        },
        name="recorded_ego.pose",
    )


def local_box_corners(dimensions: Vector3) -> tuple[Vector3, ...]:
    """Return eight centered corners for full x/y/z Actor dimensions."""
    length = _finite(dimensions.x, "dimensions.x")
    width = _finite(dimensions.y, "dimensions.y")
    height = _finite(dimensions.z, "dimensions.z")
    if length <= 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError("Actor dimensions must all be positive")

    half_x = 0.5 * length
    half_y = 0.5 * width
    half_z = 0.5 * height
    return tuple(
        Vector3(x_sign * half_x, y_sign * half_y, z_sign * half_z)
        for z_sign in (-1.0, 1.0)
        for y_sign in (-1.0, 1.0)
        for x_sign in (-1.0, 1.0)
    )


def transform_local_point(pose: Pose3D, point: Vector3) -> Vector3:
    rotated = rotate_vector_by_quaternion(point, pose.orientation)
    return Vector3(
        rotated.x + pose.position.x,
        rotated.y + pose.position.y,
        rotated.z + pose.position.z,
    )


def map_point_to_rig(point_map: Vector3, ego_pose_map: Pose3D) -> Vector3:
    shifted = Vector3(
        point_map.x - ego_pose_map.position.x,
        point_map.y - ego_pose_map.position.y,
        point_map.z - ego_pose_map.position.z,
    )
    return rotate_vector_by_quaternion_inverse(
        shifted,
        ego_pose_map.orientation,
    )


def rig_point_to_map(point_rig: Vector3, ego_pose_map: Pose3D) -> Vector3:
    rotated = rotate_vector_by_quaternion(point_rig, ego_pose_map.orientation)
    return Vector3(
        rotated.x + ego_pose_map.position.x,
        rotated.y + ego_pose_map.position.y,
        rotated.z + ego_pose_map.position.z,
    )


def build_actor_box3d(actor: Mapping[str, Any]) -> ActorBox3D:
    """Build one oriented Actor box in the map frame."""
    pose = parse_pose3d(
        _mapping(actor.get("pose"), "actor.pose"),
        name="actor.pose",
    )
    dimensions_mapping = _mapping(actor.get("dimensions"), "actor.dimensions")
    dimensions = Vector3(
        _finite(dimensions_mapping.get("x"), "actor.dimensions.x"),
        _finite(dimensions_mapping.get("y"), "actor.dimensions.y"),
        _finite(dimensions_mapping.get("z"), "actor.dimensions.z"),
    )
    corners = tuple(
        transform_local_point(pose, corner)
        for corner in local_box_corners(dimensions)
    )
    return ActorBox3D(
        track_id=_required_string(actor.get("track_id"), "actor.track_id"),
        actor_class=_required_string(actor.get("label_class"), "actor.label_class"),
        pose_map=pose,
        dimensions=dimensions,
        corners_map=corners,
    )


def actor_box_corners_in_rig(
    actor: Mapping[str, Any],
    *,
    recorded_ego_message: Mapping[str, Any],
) -> tuple[Vector3, ...]:
    """Return the Actor box corners in the Anchor Ego rig/base frame."""
    box = build_actor_box3d(actor)
    ego_pose = parse_recorded_ego_pose3d(recorded_ego_message)
    return tuple(
        map_point_to_rig(corner, ego_pose)
        for corner in box.corners_map
    )
