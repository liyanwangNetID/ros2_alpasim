"""Step 7 Scene-Fact domain.

Contains vocabulary, current geometry, semantic rules, quality propagation, feature assembly, final schema mapping, validation, and summaries."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping
from step2.coordinate_utils import Pose2D, map_pose_to_anchor_ego, map_vector_to_anchor_ego, pose2d_from_pose_mapping, relative_region
from collections import Counter
from typing import Any, Mapping, Sequence
import json
from pathlib import Path
from jsonschema import Draft202012Validator
SCENE_FACT_FORMAT_VERSION = '0.1-draft'
OBSERVABILITY_FORMAT_VERSION = '0.1-draft'
FEATURE_FORMAT_VERSION = '0.1-draft'
GENERATOR_VERSION = '0.1.0'
RULE_VERSION = 'scene_fact_rules_v0.1-draft'
CAMERA_NAMES = ('front_wide', 'front_tele', 'cross_left', 'cross_right')
ROAD_CONTEXT_TYPES = frozenset({'lane_following', 'intersection_approach', 'intersection', 'unknown'})
PROXIMITY_STATUSES = frozenset({'at', 'near', 'approaching', 'far', 'none', 'unknown'})
FINAL_PRESENCE_STATUSES = frozenset({'present', 'not_present', 'unknown'})
OBSERVABILITY_STATUSES = frozenset({'candidate_visible', 'partially_occluded', 'heavily_occluded', 'not_visible', 'unknown'})
INTERNAL_VISIBILITY_DECISIONS = frozenset({'included', 'not_observed', 'rejected', 'unknown'})
RELATIVE_POSITION_REGIONS = frozenset({'front', 'front_left', 'front_right', 'left', 'right', 'rear_left', 'rear', 'rear_right', 'overlapping', 'unknown'})
RELATIVE_DISTANCE_CATEGORIES = frozenset({'near', 'medium', 'far', 'unknown'})
DISTANCE_TREND_CATEGORIES = frozenset({'approaching', 'receding', 'stable_distance', 'uncertain'})
RELATIVE_SPEED_CATEGORIES = frozenset({'slower_than_ego', 'similar_to_ego', 'faster_than_ego', 'stationary', 'uncertain'})
LANE_DIRECTION_RELATIONS = frozenset({'same_direction', 'opposing', 'unknown'})
QUALITY_STATUSES = frozenset({'usable', 'unknown'})
FORBIDDEN_FUTURE_INPUTS = ('actors/future.jsonl', 'ego/ground_truth_future.jsonl', 'ego/complete_recording_ground_truth.json', 'ego/planner_output.jsonl')
ALLOWED_CURRENT_OR_PAST_INPUTS = ('keyframes.jsonl', 'ego/ego_state.jsonl', 'actors/current.jsonl', 'calibration/*.json', 'cameras/*/timestamps.jsonl', 'map/vector_map.json')
_REGION_NAMES = {'ahead': 'front', 'ahead_left': 'front_left', 'ahead_right': 'front_right', 'left': 'left', 'right': 'right', 'behind_left': 'rear_left', 'behind': 'rear', 'behind_right': 'rear_right', 'overlap': 'overlapping'}

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
        raise TypeError(f'{name} must be a real number')
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted

def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()

def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    return value

def classify_geometric_region(relative_x_m: float, relative_y_m: float, *, longitudinal_deadband_m: float=0.5, lateral_deadband_m: float=0.5) -> str:
    """Map the shared coordinate region vocabulary to Step 7 names.

    Deadbands are geometric parameters only. They do not define Lead, Left,
    or Right Actor-role selection thresholds.
    """
    base_region = relative_region(relative_x_m, relative_y_m, longitudinal_deadband=longitudinal_deadband_m, lateral_deadband=lateral_deadband_m)
    try:
        return _REGION_NAMES[base_region]
    except KeyError as exc:
        raise ValueError(f'unsupported relative region: {base_region}') from exc

def recorded_ego_pose_and_speed(message: Mapping[str, Any]) -> tuple[Pose2D, float]:
    """Parse one recorded ego/ego_state.jsonl message.

    Step 7 requires map-frame pose and base_link-frame dynamics.
    """
    if message.get('pose_frame_id') != 'map':
        raise ValueError("recorded Ego pose_frame_id must be 'map'")
    if message.get('dynamics_frame_id') != 'base_link':
        raise ValueError("recorded Ego dynamics_frame_id must be 'base_link'")
    pose = pose2d_from_pose_mapping({'position': _required_mapping(message.get('position'), 'recorded_ego.position'), 'orientation': _required_mapping(message.get('orientation'), 'recorded_ego.orientation')})
    speed = _finite_number(message.get('speed'), 'recorded_ego.speed')
    return (pose, speed)

def validate_actor_snapshot_frames(message: Mapping[str, Any]) -> None:
    """Validate the coordinate frames used by actors/current."""
    if message.get('pose_frame_id') != 'map':
        raise ValueError("Actor pose_frame_id must be 'map'")
    if message.get('dynamics_frame_id') != 'map':
        raise ValueError("Actor dynamics_frame_id must be 'map'")

def compute_actor_geometry(actor: Mapping[str, Any], *, anchor_ego_pose: Pose2D, ego_speed_mps: float, longitudinal_deadband_m: float=0.5, lateral_deadband_m: float=0.5) -> ActorGeometry:
    """Compute current planar Actor geometry relative to the Anchor Ego.

    The Actor pose and planar velocity are expected in the map frame. Ego
    forward speed is represented as +x in the Anchor Ego-local frame.
    """
    track_id = _required_string(actor.get('track_id'), 'actor.track_id')
    actor_class = _required_string(actor.get('label_class'), 'actor.label_class')
    is_static = actor.get('is_static')
    if not isinstance(is_static, bool):
        raise TypeError('actor.is_static must be a boolean')
    pose_mapping = _required_mapping(actor.get('pose'), 'actor.pose')
    actor_pose = pose2d_from_pose_mapping(pose_mapping)
    relative_pose = map_pose_to_anchor_ego(actor_pose, anchor_ego_pose)
    velocity = _required_mapping(actor.get('linear_velocity'), 'actor.linear_velocity')
    actor_velocity = map_vector_to_anchor_ego(_finite_number(velocity.get('x'), 'actor.linear_velocity.x'), _finite_number(velocity.get('y'), 'actor.linear_velocity.y'), anchor_ego_pose.yaw)
    ego_speed = _finite_number(ego_speed_mps, 'ego_speed_mps')
    actor_speed = _finite_number(actor.get('speed'), 'actor.speed')
    relative_velocity_x = actor_velocity.x - ego_speed
    relative_velocity_y = actor_velocity.y
    planar_distance = math.hypot(relative_pose.relative_x, relative_pose.relative_y)
    return ActorGeometry(track_id=track_id, actor_class=actor_class, is_static=is_static, relative_x_m=relative_pose.relative_x, relative_y_m=relative_pose.relative_y, planar_distance_m=planar_distance, longitudinal_distance_m=relative_pose.relative_x, lateral_distance_m=relative_pose.relative_y, relative_yaw_rad=relative_pose.relative_yaw, ego_speed_mps=ego_speed, actor_speed_mps=actor_speed, actor_velocity_x_mps=actor_velocity.x, actor_velocity_y_mps=actor_velocity.y, relative_velocity_x_mps=relative_velocity_x, relative_velocity_y_mps=relative_velocity_y, relative_longitudinal_speed_mps=relative_velocity_x, relative_lateral_speed_mps=relative_velocity_y, geometric_region=classify_geometric_region(relative_pose.relative_x, relative_pose.relative_y, longitudinal_deadband_m=longitudinal_deadband_m, lateral_deadband_m=lateral_deadband_m))

def compute_snapshot_actor_geometries(actor_snapshot_message: Mapping[str, Any], *, recorded_ego_message: Mapping[str, Any], longitudinal_deadband_m: float=0.5, lateral_deadband_m: float=0.5) -> tuple[ActorGeometry, ...]:
    """Compute geometry for every Actor in one current snapshot."""
    validate_actor_snapshot_frames(actor_snapshot_message)
    anchor_pose, ego_speed = recorded_ego_pose_and_speed(recorded_ego_message)
    actors = actor_snapshot_message.get('actors')
    if not isinstance(actors, list):
        raise TypeError('Actor snapshot actors must be a list')
    geometries = tuple((compute_actor_geometry(_required_mapping(actor, 'actor'), anchor_ego_pose=anchor_pose, ego_speed_mps=ego_speed, longitudinal_deadband_m=longitudinal_deadband_m, lateral_deadband_m=lateral_deadband_m) for actor in actors))
    track_ids = [item.track_id for item in geometries]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError('Actor snapshot contains duplicate track_id values')
    return geometries
NEAR_DISTANCE_MAX_M = 10.0
MEDIUM_DISTANCE_MAX_M = 30.0
STABLE_DISTANCE_RATE_MAX_MPS = 0.5
STATIONARY_SPEED_MAX_MPS = 0.5
SIMILAR_SPEED_DIFFERENCE_MAX_MPS = 1.0
INTERSECTION_WAIT_LINE_AT_MAX_M = 3.0
INTERSECTION_WAIT_LINE_APPROACH_MAX_M = 15.0
INTERSECTION_LANE_END_AT_MAX_M = 3.0

def classify_road_context(ego_road: Mapping[str, Any]) -> dict[str, Any]:
    """Classify current road context from current Ego map evidence only."""
    if ego_road['lane_match_status'] != 'matched':
        result = {'type': 'unknown', 'proximity_status': 'unknown', 'lane_id': None, 'nearest_wait_line_distance_m': None, 'evidence': ['ego_lane_unmatched']}
    else:
        wait_distance = ego_road['nearest_wait_line_distance_m']
        evidence = list(ego_road['intersection_evidence'])
        remaining = None
        if ego_road['lane_length_m'] is not None and ego_road['centerline_arc_length_m'] is not None:
            remaining = max(0.0, float(ego_road['lane_length_m']) - float(ego_road['centerline_arc_length_m']))
        if wait_distance is not None and float(wait_distance) <= INTERSECTION_WAIT_LINE_AT_MAX_M:
            context_type = 'intersection'
            proximity = 'at'
        elif wait_distance is not None and float(wait_distance) <= INTERSECTION_WAIT_LINE_APPROACH_MAX_M:
            context_type = 'intersection_approach'
            proximity = 'approaching'
        elif bool(ego_road['has_intersection_evidence']):
            if remaining is not None and remaining <= INTERSECTION_LANE_END_AT_MAX_M:
                context_type = 'intersection'
                proximity = 'at'
            else:
                context_type = 'intersection_approach'
                proximity = 'near'
        else:
            context_type = 'lane_following'
            proximity = 'none'
        result = {'type': context_type, 'proximity_status': proximity, 'lane_id': ego_road['lane_id'], 'nearest_wait_line_distance_m': wait_distance, 'lane_remaining_distance_m': remaining, 'evidence': evidence}
    if result['type'] not in ROAD_CONTEXT_TYPES:
        raise RuntimeError('unexpected road-context type')
    return result

def relative_distance_category(distance_m: float) -> str:
    distance = float(distance_m)
    if distance < 0.0:
        raise ValueError('distance_m must be non-negative')
    value = 'near' if distance <= NEAR_DISTANCE_MAX_M else 'medium' if distance <= MEDIUM_DISTANCE_MAX_M else 'far'
    if value not in RELATIVE_DISTANCE_CATEGORIES:
        raise RuntimeError('unexpected relative-distance category')
    return value

def distance_trend(history_status: str, mean_distance_rate_mps: Any) -> str:
    if history_status != 'usable' or mean_distance_rate_mps is None:
        return 'uncertain'
    rate = float(mean_distance_rate_mps)
    value = 'stable_distance' if abs(rate) <= STABLE_DISTANCE_RATE_MAX_MPS else 'approaching' if rate < 0.0 else 'receding'
    if value not in DISTANCE_TREND_CATEGORIES:
        raise RuntimeError('unexpected distance-trend category')
    return value

def relative_speed_category(*, actor_speed_mps: float, ego_speed_mps: float) -> str:
    actor_speed = float(actor_speed_mps)
    ego_speed = float(ego_speed_mps)
    if actor_speed <= STATIONARY_SPEED_MAX_MPS:
        value = 'stationary'
    elif abs(actor_speed - ego_speed) <= SIMILAR_SPEED_DIFFERENCE_MAX_MPS:
        value = 'similar_to_ego'
    elif actor_speed < ego_speed:
        value = 'slower_than_ego'
    else:
        value = 'faster_than_ego'
    if value not in RELATIVE_SPEED_CATEGORIES:
        raise RuntimeError('unexpected relative-speed category')
    return value

def assemble_selected_actor_feature(*, role, selected_role, current_geometry):
    if current_geometry is None:
        raise ValueError("selected Actor requires current geometry")
    if str(current_geometry["track_id"]) != str(selected_role["track_id"]):
        raise ValueError("selected Actor and current geometry identities differ")
    history_status = str(selected_role["history_status"])
    return {
        "role": role, "role_rank": int(selected_role["role_rank"]),
        "track_id": str(selected_role["track_id"]), "label_class": str(selected_role["label_class"]),
        "relative_position": str(selected_role["geometric_region"]),
        "relative_distance": relative_distance_category(float(selected_role["planar_distance_m"])),
        "distance_m": float(selected_role["planar_distance_m"]),
        "relative_x_m": float(selected_role["relative_x_m"]), "relative_y_m": float(selected_role["relative_y_m"]),
        "distance_trend": distance_trend(history_status, selected_role["mean_distance_rate_mps"]),
        "relative_speed": relative_speed_category(actor_speed_mps=float(current_geometry["actor_speed_mps"]), ego_speed_mps=float(current_geometry["ego_speed_mps"])),
        "actor_speed_mps": float(current_geometry["actor_speed_mps"]), "ego_speed_mps": float(current_geometry["ego_speed_mps"]),
        "observability_status": "candidate_visible", "visibility_policy_status": str(selected_role["visibility_policy_status"]),
        "history_status": history_status, "lane_match_status": str(selected_role["lane_match_status"]),
        "ego_lane_relation": str(selected_role["ego_lane_relation"]),
        "lane_direction_relation": str(selected_role.get("lane_direction_relation", "unknown")),
    }


def _unique_reasons(values):
    return list(dict.fromkeys(str(value) for value in values))


def _line_proximities(road):
    return ("none", "none") if road.get("nearest_wait_line_distance_m") is None else ("unknown", "unknown")


def final_road_context(feature):
    road = feature["road_context"]
    context_type = str(road["type"])
    stop, yield_ = _line_proximities(road)
    reasons = list(road.get("evidence", ()))
    quality = "unknown" if context_type == "unknown" else "usable"
    if context_type == "unknown":
        reasons.append("ego_road_context_unknown")
    if road.get("nearest_wait_line_distance_m") is not None:
        reasons.append("wait_line_type_not_propagated_to_feature_layer")
    return {"type": context_type, "intersection_proximity": str(road["proximity_status"]), "stop_line_proximity": stop, "yield_line_proximity": yield_, "quality_status": quality, "reasons": _unique_reasons(reasons)}


ACTOR_LIST_KEYS = ("lead_actors", "left_nearby_actors", "right_nearby_actors")
FINAL_RECORD_REQUIRED_KEYS = frozenset({"scene_fact_format_version", "generator_version", "rule_version", "anchor_id", "clip_id", "anchor_ns", "road_context", "actor_context", "quality"})


def assemble_scene_fact_feature_row(*, keyframe, ego_road, role_selection, current_geometry_by_track_id):
    anchor = str(keyframe["anchor_id"])
    if str(ego_road["anchor_id"]) != anchor or str(role_selection["anchor_id"]) != anchor:
        raise ValueError("Anchor input does not match Keyframe")
    names = {"lead_actors": "lead_actor", "left_nearby_actors": "left_nearby_actor", "right_nearby_actors": "right_nearby_actor"}
    lists = {}
    for key in ACTOR_LIST_KEYS:
        lists[key] = []
        for selected in role_selection["roles"][key]:
            track = str(selected["track_id"])
            lists[key].append(assemble_selected_actor_feature(role=names[key], selected_role=selected, current_geometry=current_geometry_by_track_id.get(track)))
    road = classify_road_context(ego_road)
    reasons = []
    if road["type"] == "unknown":
        reasons.append("ego_road_context_unknown")
    for key in ACTOR_LIST_KEYS:
        for actor in lists[key]:
            if actor["history_status"] != "usable":
                reasons.append(f"{key}_history_{actor['history_status']}")
            if actor["lane_match_status"] != "matched":
                reasons.append(f"{key}_lane_unmatched")
    return {
        "schema_version": "step7k-scene-fact-features-v02", "feature_format_version": FEATURE_FORMAT_VERSION,
        "anchor_id": anchor, "clip_id": str(keyframe["clip_id"]), "anchor_ns": int(keyframe["anchor_ns"]),
        "road_context": road, "actor_context": {**role_selection["roles"]["selection_context"], **lists},
        "quality": {"status": "usable" if not reasons else "unknown", "reasons": _unique_reasons(reasons), "static_occlusion_evaluated": False, "current_and_past_inputs_only": True},
        "future_actor_data_used": False, "future_ego_data_used": False, "planner_output_used": False, "meta_action_used": False,
    }


def _final_actor(actor, cameras):
    reasons = []
    if actor["history_status"] != "usable":
        reasons.append(f"history_{actor['history_status']}")
    if actor["lane_match_status"] != "matched":
        reasons.append("lane_unmatched")
    visible = sorted(set(str(value) for value in cameras or ()))
    if not visible:
        raise ValueError("selected Actor requires at least one visible camera")
    return {
        "role_rank": int(actor["role_rank"]), "track_id": str(actor["track_id"]), "actor_class": str(actor["label_class"]),
        "relative_position": str(actor["relative_position"]), "relative_distance": str(actor["relative_distance"]),
        "lane_direction_relation": str(actor["lane_direction_relation"]),
        "relative_x_m": float(actor["relative_x_m"]), "relative_y_m": float(actor["relative_y_m"]), "distance_m": float(actor["distance_m"]),
        "distance_trend": str(actor["distance_trend"]), "relative_speed_category": str(actor["relative_speed"]),
        "actor_speed_mps": float(actor["actor_speed_mps"]), "ego_speed_mps": float(actor["ego_speed_mps"]),
        "observability_status": str(actor["observability_status"]), "visible_in_cameras": visible,
        "quality_status": "usable" if not reasons else "unknown", "reasons": reasons,
    }


def build_final_scene_fact_record(*, feature, visible_cameras_by_track_id):
    context = feature["actor_context"]
    lists = {key: [_final_actor(actor, visible_cameras_by_track_id.get(str(actor["track_id"]))) for actor in context[key]] for key in ACTOR_LIST_KEYS}
    return {
        "scene_fact_format_version": SCENE_FACT_FORMAT_VERSION, "generator_version": GENERATOR_VERSION, "rule_version": RULE_VERSION,
        "anchor_id": str(feature["anchor_id"]), "clip_id": str(feature["clip_id"]), "anchor_ns": int(feature["anchor_ns"]),
        "road_context": final_road_context(feature),
        "actor_context": {"reference_ego_speed_mps": float(context["reference_ego_speed_mps"]), "forward_horizon_m": float(context["forward_horizon_m"]), "side_forward_horizon_m": float(context["side_forward_horizon_m"]), "rear_horizon_m": float(context["rear_horizon_m"]), "lists": context["lists"], **lists},
        "quality": {"status": str(feature["quality"]["status"]), "static_occlusion_evaluated": False, "reasons": _unique_reasons(feature["quality"]["reasons"])},
    }


def _validate_rows(keyframes, rows, name):
    anchors = [str(row["anchor_id"]) for row in keyframes]
    if len(anchors) != len(set(anchors)) or {str(row["anchor_id"]) for row in rows} != set(anchors):
        raise ValueError(f"{name} must contain exactly one row per Keyframe")
    return anchors


def summarize_scene_fact_feature_rows(*, keyframes, rows):
    anchors = _validate_rows(keyframes, rows, "feature rows")
    return {"schema_version": "step7k-scene-fact-features-summary-v02", "keyframe_count": len(anchors), "feature_row_count": len(rows), "road_context_type_counts": dict(sorted(Counter(row["road_context"]["type"] for row in rows).items())), "quality_status_counts": dict(sorted(Counter(row["quality"]["status"] for row in rows).items())), "quality_reason_counts": dict(sorted(Counter(reason for row in rows for reason in row["quality"]["reasons"]).items())), "selected_actor_counts": {key: sum(len(row["actor_context"][key]) for row in rows) for key in ACTOR_LIST_KEYS}}


def summarize_final_scene_facts(*, keyframes, rows):
    anchors = _validate_rows(keyframes, rows, "final Scene Facts")
    return {"schema_version": "step7l-final-scene-facts-summary-v02", "keyframe_count": len(anchors), "scene_fact_row_count": len(rows), "road_context_type_counts": dict(sorted(Counter(row["road_context"]["type"] for row in rows).items())), "quality_status_counts": dict(sorted(Counter(row["quality"]["status"] for row in rows).items())), "selected_actor_counts": {key: sum(len(row["actor_context"][key]) for row in rows) for key in ACTOR_LIST_KEYS}, "schema_validation_error_count": 0}


class SceneFactValidationError(ValueError):
    pass


def load_scene_fact_validator(schema_path):
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_scene_fact_record(record, *, validator):
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(value) for value in error.absolute_path) or "$"
        raise SceneFactValidationError(f"{path}: {error.message}")
