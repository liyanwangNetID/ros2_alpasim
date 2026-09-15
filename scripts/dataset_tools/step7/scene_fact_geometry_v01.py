#!/usr/bin/env python3
"""Step 7C Ego-relative planar Actor geometry.

This module converts one current Actor state from map coordinates into the
Anchor-time Ego-local frame. It contains no Scene-Fact classification
thresholds and does not select Actor roles.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from step2.coordinate_utils import (
    Pose2D,
    map_pose_to_anchor_ego,
    map_vector_to_anchor_ego,
    pose2d_from_pose_mapping,
    relative_region,
)


_REGION_NAMES = {
    "ahead": "front",
    "ahead_left": "front_left",
    "ahead_right": "front_right",
    "left": "left",
    "right": "right",
    "behind_left": "rear_left",
    "behind": "rear",
    "behind_right": "rear_right",
    "overlap": "overlapping",
}


@dataclass(frozen=True, slots=True)
class ActorGeometry:
    """Continuous current-Actor geometry in the Anchor Ego-local frame."""

    track_id: str
    actor_class: str
    is_static: bool
    relative_x_m: float
    relative_y_m: float
    planar_distance_m: float
    longitudinal_distance_m: float
    lateral_distance_m: float
    relative_yaw_rad: float
    ego_speed_mps: float
    actor_speed_mps: float
    actor_velocity_x_mps: float
    actor_velocity_y_mps: float
    relative_velocity_x_mps: float
    relative_velocity_y_mps: float
    relative_longitudinal_speed_mps: float
    relative_lateral_speed_mps: float
    geometric_region: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def classify_geometric_region(
    relative_x_m: float,
    relative_y_m: float,
    *,
    longitudinal_deadband_m: float = 0.5,
    lateral_deadband_m: float = 0.5,
) -> str:
    """Map the shared coordinate region vocabulary to Step 7 names.

    Deadbands are geometric parameters only. They do not define Lead, Left,
    or Right Actor-role selection thresholds.
    """
    base_region = relative_region(
        relative_x_m,
        relative_y_m,
        longitudinal_deadband=longitudinal_deadband_m,
        lateral_deadband=lateral_deadband_m,
    )
    try:
        return _REGION_NAMES[base_region]
    except KeyError as exc:
        raise ValueError(f"unsupported relative region: {base_region}") from exc


def recorded_ego_pose_and_speed(
    message: Mapping[str, Any],
) -> tuple[Pose2D, float]:
    """Parse one recorded ego/ego_state.jsonl message.

    Step 7 requires map-frame pose and base_link-frame dynamics.
    """
    if message.get("pose_frame_id") != "map":
        raise ValueError("recorded Ego pose_frame_id must be 'map'")
    if message.get("dynamics_frame_id") != "base_link":
        raise ValueError(
            "recorded Ego dynamics_frame_id must be 'base_link'"
        )
    pose = pose2d_from_pose_mapping(
        {
            "position": _required_mapping(
                message.get("position"),
                "recorded_ego.position",
            ),
            "orientation": _required_mapping(
                message.get("orientation"),
                "recorded_ego.orientation",
            ),
        }
    )
    speed = _finite_number(message.get("speed"), "recorded_ego.speed")
    return pose, speed


def validate_actor_snapshot_frames(message: Mapping[str, Any]) -> None:
    """Validate the coordinate frames used by actors/current."""
    if message.get("pose_frame_id") != "map":
        raise ValueError("Actor pose_frame_id must be 'map'")
    if message.get("dynamics_frame_id") != "map":
        raise ValueError("Actor dynamics_frame_id must be 'map'")

def compute_actor_geometry(
    actor: Mapping[str, Any],
    *,
    anchor_ego_pose: Pose2D,
    ego_speed_mps: float,
    longitudinal_deadband_m: float = 0.5,
    lateral_deadband_m: float = 0.5,
) -> ActorGeometry:
    """Compute current planar Actor geometry relative to the Anchor Ego.

    The Actor pose and planar velocity are expected in the map frame. Ego
    forward speed is represented as +x in the Anchor Ego-local frame.
    """
    track_id = _required_string(actor.get("track_id"), "actor.track_id")
    actor_class = _required_string(
        actor.get("label_class"),
        "actor.label_class",
    )
    is_static = actor.get("is_static")
    if not isinstance(is_static, bool):
        raise TypeError("actor.is_static must be a boolean")

    pose_mapping = _required_mapping(actor.get("pose"), "actor.pose")
    actor_pose = pose2d_from_pose_mapping(pose_mapping)
    relative_pose = map_pose_to_anchor_ego(actor_pose, anchor_ego_pose)

    velocity = _required_mapping(
        actor.get("linear_velocity"),
        "actor.linear_velocity",
    )
    actor_velocity = map_vector_to_anchor_ego(
        _finite_number(velocity.get("x"), "actor.linear_velocity.x"),
        _finite_number(velocity.get("y"), "actor.linear_velocity.y"),
        anchor_ego_pose.yaw,
    )

    ego_speed = _finite_number(ego_speed_mps, "ego_speed_mps")
    actor_speed = _finite_number(actor.get("speed"), "actor.speed")
    relative_velocity_x = actor_velocity.x - ego_speed
    relative_velocity_y = actor_velocity.y
    planar_distance = math.hypot(
        relative_pose.relative_x,
        relative_pose.relative_y,
    )

    return ActorGeometry(
        track_id=track_id,
        actor_class=actor_class,
        is_static=is_static,
        relative_x_m=relative_pose.relative_x,
        relative_y_m=relative_pose.relative_y,
        planar_distance_m=planar_distance,
        longitudinal_distance_m=relative_pose.relative_x,
        lateral_distance_m=relative_pose.relative_y,
        relative_yaw_rad=relative_pose.relative_yaw,
        ego_speed_mps=ego_speed,
        actor_speed_mps=actor_speed,
        actor_velocity_x_mps=actor_velocity.x,
        actor_velocity_y_mps=actor_velocity.y,
        relative_velocity_x_mps=relative_velocity_x,
        relative_velocity_y_mps=relative_velocity_y,
        relative_longitudinal_speed_mps=relative_velocity_x,
        relative_lateral_speed_mps=relative_velocity_y,
        geometric_region=classify_geometric_region(
            relative_pose.relative_x,
            relative_pose.relative_y,
            longitudinal_deadband_m=longitudinal_deadband_m,
            lateral_deadband_m=lateral_deadband_m,
        ),
    )


def compute_snapshot_actor_geometries(
    actor_snapshot_message: Mapping[str, Any],
    *,
    recorded_ego_message: Mapping[str, Any],
    longitudinal_deadband_m: float = 0.5,
    lateral_deadband_m: float = 0.5,
) -> tuple[ActorGeometry, ...]:
    """Compute geometry for every Actor in one current snapshot."""
    validate_actor_snapshot_frames(actor_snapshot_message)
    anchor_pose, ego_speed = recorded_ego_pose_and_speed(
        recorded_ego_message
    )
    actors = actor_snapshot_message.get("actors")
    if not isinstance(actors, list):
        raise TypeError("Actor snapshot actors must be a list")

    geometries = tuple(
        compute_actor_geometry(
            _required_mapping(actor, "actor"),
            anchor_ego_pose=anchor_pose,
            ego_speed_mps=ego_speed,
            longitudinal_deadband_m=longitudinal_deadband_m,
            lateral_deadband_m=lateral_deadband_m,
        )
        for actor in actors
    )
    track_ids = [item.track_id for item in geometries]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Actor snapshot contains duplicate track_id values")
    return geometries
