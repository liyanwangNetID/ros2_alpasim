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
QUALITY_STATUSES = frozenset({'usable', 'unknown'})
ACTOR_ROLE_KEYS = ('lead_vehicle', 'left_nearby_vehicle', 'right_nearby_vehicle')
FINAL_RECORD_REQUIRED_KEYS = frozenset({'scene_fact_format_version', 'generator_version', 'rule_version', 'anchor_id', 'clip_id', 'anchor_ns', 'road_context', 'lead_vehicle', 'left_nearby_vehicle', 'right_nearby_vehicle', 'quality'})
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

def assemble_selected_actor_feature(*, role: str, selected_role: Mapping[str, Any] | None, empty_reason: str | None, current_geometry: Mapping[str, Any] | None) -> dict[str, Any]:
    if selected_role is None:
        if empty_reason is None:
            raise ValueError('empty Actor role requires an explicit reason')
        return {'presence_status': 'not_present', 'role': role, 'absence_reason': empty_reason}
    if current_geometry is None:
        raise ValueError('selected Actor requires current geometry')
    if str(current_geometry['track_id']) != str(selected_role['track_id']):
        raise ValueError('selected Actor and current geometry identities differ')
    history_status = str(selected_role['history_status'])
    return {'presence_status': 'present', 'role': role, 'track_id': str(selected_role['track_id']), 'label_class': str(selected_role['label_class']), 'relative_position': str(selected_role['geometric_region']), 'relative_distance': relative_distance_category(float(selected_role['planar_distance_m'])), 'distance_m': float(selected_role['planar_distance_m']), 'relative_x_m': float(selected_role['relative_x_m']), 'relative_y_m': float(selected_role['relative_y_m']), 'distance_trend': distance_trend(history_status, selected_role['mean_distance_rate_mps']), 'relative_speed': relative_speed_category(actor_speed_mps=float(current_geometry['actor_speed_mps']), ego_speed_mps=float(current_geometry['ego_speed_mps'])), 'actor_speed_mps': float(current_geometry['actor_speed_mps']), 'ego_speed_mps': float(current_geometry['ego_speed_mps']), 'observability_status': 'candidate_visible', 'visibility_policy_status': str(selected_role['visibility_policy_status']), 'history_status': history_status, 'lane_match_status': str(selected_role['lane_match_status']), 'ego_lane_relation': str(selected_role['ego_lane_relation'])}

def assemble_feature_quality(*, road_context: Mapping[str, Any], actor_roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reasons = []
    if road_context['type'] == 'unknown':
        reasons.append('ego_road_context_unknown')
    for role in ACTOR_ROLE_KEYS:
        value = actor_roles[role]
        if value['presence_status'] != 'present':
            continue
        if value['history_status'] != 'usable':
            reasons.append(f"{role}_history_{value['history_status']}")
        if value['lane_match_status'] != 'matched':
            reasons.append(f'{role}_lane_unmatched')
    status = 'usable' if not reasons else 'unknown'
    if status not in QUALITY_STATUSES:
        raise RuntimeError('unexpected quality status')
    return {'status': status, 'reasons': reasons, 'static_occlusion_evaluated': False, 'current_and_past_inputs_only': True}

def assemble_scene_fact_feature_row(*, keyframe: Mapping[str, Any], ego_road: Mapping[str, Any], role_selection: Mapping[str, Any], current_geometry_by_track_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    anchor_id = str(keyframe['anchor_id'])
    for name, row in (('ego_road', ego_road), ('role_selection', role_selection)):
        if str(row['anchor_id']) != anchor_id:
            raise ValueError(f'{name} Anchor does not match Keyframe')
    role_features = {}
    for role in ACTOR_ROLE_KEYS:
        selected = role_selection['roles'][role]
        geometry = None if selected is None else current_geometry_by_track_id.get(str(selected['track_id']))
        role_features[role] = assemble_selected_actor_feature(role=role, selected_role=selected, empty_reason=role_selection['empty_role_reasons'][role], current_geometry=geometry)
    road_context = classify_road_context(ego_road)
    quality = assemble_feature_quality(road_context=road_context, actor_roles=role_features)
    return {'schema_version': 'step7k-scene-fact-features-v01', 'feature_format_version': FEATURE_FORMAT_VERSION, 'anchor_id': anchor_id, 'clip_id': str(keyframe['clip_id']), 'anchor_ns': int(keyframe['anchor_ns']), 'road_context': road_context, **role_features, 'quality': quality, 'future_actor_data_used': False, 'future_ego_data_used': False, 'planner_output_used': False, 'meta_action_used': False}

def summarize_scene_fact_feature_rows(*, keyframes, rows):
    anchors = [str(row['anchor_id']) for row in keyframes]
    row_anchors = [str(row['anchor_id']) for row in rows]
    if len(anchors) != len(set(anchors)):
        raise ValueError('Keyframe Anchor ids must be unique')
    if len(rows) != len(anchors) or set(row_anchors) != set(anchors):
        raise ValueError('feature rows must contain exactly one row per Keyframe')
    role_presence = {role: dict(sorted(Counter((row[role]['presence_status'] for row in rows)).items())) for role in ACTOR_ROLE_KEYS}
    return {'schema_version': 'step7k-scene-fact-features-summary-v01', 'keyframe_count': len(anchors), 'feature_row_count': len(rows), 'road_context_type_counts': dict(sorted(Counter((row['road_context']['type'] for row in rows)).items())), 'quality_status_counts': dict(sorted(Counter((row['quality']['status'] for row in rows)).items())), 'quality_reason_counts': dict(sorted(Counter((reason for row in rows for reason in row['quality']['reasons'])).items())), 'role_presence_status_counts': role_presence}

def _unique_reasons(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys((str(value) for value in values)))

def _line_proximities(road: Mapping[str, Any]) -> tuple[str, str]:
    """Map available wait-line evidence conservatively to stop/yield fields."""
    distance = road.get('nearest_wait_line_distance_m')
    if distance is None:
        return ('none', 'none')
    proximity = str(road['proximity_status'])
    return ('unknown', 'unknown')

def final_road_context(feature: Mapping[str, Any]) -> dict[str, Any]:
    road = feature['road_context']
    context_type = str(road['type'])
    stop_proximity, yield_proximity = _line_proximities(road)
    reasons = list(road.get('evidence', ()))
    if context_type == 'unknown':
        reasons.append('ego_road_context_unknown')
        quality = 'unknown'
    else:
        quality = 'usable'
    if road.get('nearest_wait_line_distance_m') is not None:
        reasons.append('wait_line_type_not_propagated_to_feature_layer')
    return {'type': context_type, 'intersection_proximity': str(road['proximity_status']), 'stop_line_proximity': stop_proximity, 'yield_line_proximity': yield_proximity, 'quality_status': quality, 'reasons': _unique_reasons(reasons)}

def final_actor_role(feature: Mapping[str, Any], *, visible_cameras: Sequence[str] | None) -> dict[str, Any]:
    if feature['presence_status'] != 'present':
        return {'presence_status': 'not_present', 'quality_status': 'usable', 'reasons': [str(feature['absence_reason'])]}
    reasons = []
    if feature['history_status'] != 'usable':
        reasons.append(f"history_{feature['history_status']}")
    if feature['lane_match_status'] != 'matched':
        reasons.append('lane_unmatched')
    cameras = sorted(set((str(value) for value in visible_cameras or ())))
    if not cameras:
        raise ValueError('present Actor role requires at least one visible camera')
    quality = 'usable' if not reasons else 'unknown'
    return {'presence_status': 'present', 'track_id': str(feature['track_id']), 'actor_class': str(feature['label_class']), 'relative_position': str(feature['relative_position']), 'relative_distance': str(feature['relative_distance']), 'distance_trend': str(feature['distance_trend']), 'relative_speed_category': str(feature['relative_speed']), 'observability_status': str(feature['observability_status']), 'visible_in_cameras': cameras, 'quality_status': quality, 'reasons': reasons}

def build_final_scene_fact_record(*, feature: Mapping[str, Any], visible_cameras_by_track_id: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    roles = {}
    for role in ACTOR_ROLE_KEYS:
        value = feature[role]
        track_id = None if value['presence_status'] != 'present' else str(value['track_id'])
        roles[role] = final_actor_role(value, visible_cameras=None if track_id is None else visible_cameras_by_track_id.get(track_id))
    return {'scene_fact_format_version': SCENE_FACT_FORMAT_VERSION, 'generator_version': GENERATOR_VERSION, 'rule_version': RULE_VERSION, 'anchor_id': str(feature['anchor_id']), 'clip_id': str(feature['clip_id']), 'anchor_ns': int(feature['anchor_ns']), 'road_context': final_road_context(feature), **roles, 'quality': {'status': str(feature['quality']['status']), 'static_occlusion_evaluated': False, 'reasons': _unique_reasons(feature['quality']['reasons'])}}

def summarize_final_scene_facts(*, keyframes, rows):
    anchors = [str(row['anchor_id']) for row in keyframes]
    row_anchors = [str(row['anchor_id']) for row in rows]
    if len(anchors) != len(set(anchors)):
        raise ValueError('Keyframe Anchor ids must be unique')
    if len(rows) != len(anchors) or set(row_anchors) != set(anchors):
        raise ValueError('final Scene Facts must contain exactly one row per Keyframe')
    return {'schema_version': 'step7l-final-scene-facts-summary-v01', 'keyframe_count': len(anchors), 'scene_fact_row_count': len(rows), 'road_context_type_counts': dict(sorted(Counter((row['road_context']['type'] for row in rows)).items())), 'quality_status_counts': dict(sorted(Counter((row['quality']['status'] for row in rows)).items())), 'role_presence_status_counts': {role: dict(sorted(Counter((row[role]['presence_status'] for row in rows)).items())) for role in ACTOR_ROLE_KEYS}, 'schema_validation_error_count': 0}

class SceneFactValidationError(ValueError):
    pass

def load_scene_fact_validator(schema_path: Path) -> Draft202012Validator:
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)

def validate_scene_fact_record(record: Mapping[str, Any], *, validator: Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        path = '.'.join((str(value) for value in error.absolute_path)) or '$'
        raise SceneFactValidationError(f'{path}: {error.message}')
