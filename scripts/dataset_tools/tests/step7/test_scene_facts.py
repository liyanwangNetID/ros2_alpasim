"""Tests for the consolidated Step 7 Scene-Fact domain."""
from __future__ import annotations
from step7.scene_facts import summarize_final_scene_facts
import pytest
from step7.scene_facts import build_final_scene_fact_record
from step7.scene_facts import summarize_scene_fact_feature_rows
from step7.scene_facts import assemble_scene_fact_feature_row
import math
import sys
import unittest
from pathlib import Path
from step2.coordinate_utils import Pose2D, yaw_to_quaternion
from step7.scene_facts import classify_geometric_region, compute_snapshot_actor_geometries, recorded_ego_pose_and_speed, compute_actor_geometry
from step7.scene_facts import classify_road_context, distance_trend, relative_distance_category, relative_speed_category
import json
from project_paths import SCHEMA_ROOT
from step7.scene_facts import ACTOR_ROLE_KEYS, CAMERA_NAMES, DISTANCE_TREND_CATEGORIES, FINAL_PRESENCE_STATUSES, FINAL_RECORD_REQUIRED_KEYS, FORBIDDEN_FUTURE_INPUTS, OBSERVABILITY_STATUSES, PROXIMITY_STATUSES, QUALITY_STATUSES, RELATIVE_DISTANCE_CATEGORIES, RELATIVE_POSITION_REGIONS, RELATIVE_SPEED_CATEGORIES, ROAD_CONTEXT_TYPES, SCENE_FACT_FORMAT_VERSION
from step7.scene_facts import SceneFactValidationError, load_scene_fact_validator, validate_scene_fact_record

def test_summary_requires_one_row_per_keyframe():
    row = {'anchor_id': 'a', 'road_context': {'type': 'lane_following'}, 'quality': {'status': 'usable'}, 'lead_vehicle': {'presence_status': 'not_present'}, 'left_nearby_vehicle': {'presence_status': 'not_present'}, 'right_nearby_vehicle': {'presence_status': 'not_present'}}
    result = summarize_final_scene_facts(keyframes=({'anchor_id': 'a'},), rows=(row,))
    assert result['scene_fact_row_count'] == 1
    assert result['schema_validation_error_count'] == 0

def feature(present=True):
    absent = {'presence_status': 'not_present', 'absence_reason': 'none'}
    actor = {'presence_status': 'present', 'track_id': '1', 'label_class': 'automobile', 'relative_position': 'front', 'relative_distance': 'near', 'distance_trend': 'stable_distance', 'relative_speed': 'similar_to_ego', 'observability_status': 'candidate_visible', 'history_status': 'usable', 'lane_match_status': 'matched'}
    return {'anchor_id': 'a', 'clip_id': 'test_clip_001', 'anchor_ns': 1, 'road_context': {'type': 'lane_following', 'proximity_status': 'none', 'evidence': [], 'nearest_wait_line_distance_m': None}, 'lead_vehicle': actor if present else absent, 'left_nearby_vehicle': absent, 'right_nearby_vehicle': absent, 'quality': {'status': 'usable', 'reasons': []}}

def test_final_mapping_is_schema_shaped():
    row = build_final_scene_fact_record(feature=feature(), visible_cameras_by_track_id={'1': ['front_wide']})
    assert row['lead_vehicle']['visible_in_cameras'] == ['front_wide']
    assert row['road_context']['stop_line_proximity'] == 'none'
    assert row['quality']['static_occlusion_evaluated'] is False

def test_present_actor_requires_camera_evidence():
    with pytest.raises(ValueError, match='visible camera'):
        build_final_scene_fact_record(feature=feature(), visible_cameras_by_track_id={})

def test_visible_cameras_are_sorted_and_deduplicated():
    row = build_final_scene_fact_record(feature=feature(), visible_cameras_by_track_id={'1': ['front_wide', 'cross_left', 'front_wide']})
    assert row['lead_vehicle']['visible_in_cameras'] == ['cross_left', 'front_wide']

def test_summary_closes_one_row_per_keyframe():
    row = {'anchor_id': 'a', 'road_context': {'type': 'lane_following'}, 'quality': {'status': 'usable', 'reasons': []}, 'lead_vehicle': {'presence_status': 'not_present'}, 'left_nearby_vehicle': {'presence_status': 'present'}, 'right_nearby_vehicle': {'presence_status': 'not_present'}}
    result = summarize_scene_fact_feature_rows(keyframes=({'anchor_id': 'a'},), rows=(row,))
    assert result['feature_row_count'] == 1
    assert result['role_presence_status_counts']['left_nearby_vehicle'] == {'present': 1}

def test_row_assembly_preserves_absence_and_quality():
    keyframe = {'anchor_id': 'a', 'clip_id': 'c', 'anchor_ns': 1}
    ego = {'anchor_id': 'a', 'lane_match_status': 'unmatched', 'lane_id': None, 'nearest_wait_line_distance_m': None, 'has_intersection_evidence': None, 'intersection_evidence': [], 'lane_length_m': None, 'centerline_arc_length_m': None}
    roles = {'anchor_id': 'a', 'roles': {'lead_vehicle': None, 'left_nearby_vehicle': None, 'right_nearby_vehicle': None}, 'empty_role_reasons': {'lead_vehicle': 'no_current_actors', 'left_nearby_vehicle': 'no_current_actors', 'right_nearby_vehicle': 'no_current_actors'}}
    row = assemble_scene_fact_feature_row(keyframe=keyframe, ego_road=ego, role_selection=roles, current_geometry_by_track_id={})
    assert row['road_context']['type'] == 'unknown'
    assert row['lead_vehicle']['presence_status'] == 'not_present'
    assert row['quality']['status'] == 'unknown'

def actor(*, x: float, y: float, yaw: float=0.0, velocity_x: float=0.0, velocity_y: float=0.0, speed: float=0.0, track_id: str='actor-1', label_class: str='automobile', is_static: bool=False) -> dict:
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    return {'track_id': track_id, 'label_class': label_class, 'is_static': is_static, 'pose': {'position': {'x': x, 'y': y, 'z': 0.0}, 'orientation': {'x': qx, 'y': qy, 'z': qz, 'w': qw}}, 'linear_velocity': {'x': velocity_x, 'y': velocity_y, 'z': 0.0}, 'speed': speed}

class GeometricRegionTests(unittest.TestCase):

    def test_step7_region_names(self) -> None:
        self.assertEqual(classify_geometric_region(5.0, 0.0), 'front')
        self.assertEqual(classify_geometric_region(5.0, 5.0), 'front_left')
        self.assertEqual(classify_geometric_region(5.0, -5.0), 'front_right')
        self.assertEqual(classify_geometric_region(-5.0, 0.0), 'rear')
        self.assertEqual(classify_geometric_region(0.0, 5.0), 'left')
        self.assertEqual(classify_geometric_region(0.0, -5.0), 'right')
        self.assertEqual(classify_geometric_region(0.1, 0.1), 'overlapping')

    def test_negative_deadband_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_geometric_region(1.0, 1.0, longitudinal_deadband_m=-1.0)

class ActorGeometryTests(unittest.TestCase):

    def test_identity_anchor_geometry(self) -> None:
        result = compute_actor_geometry(actor(x=3.0, y=4.0, velocity_x=8.0, speed=8.0), anchor_ego_pose=Pose2D(0.0, 0.0, 0.0), ego_speed_mps=5.0)
        self.assertAlmostEqual(result.relative_x_m, 3.0)
        self.assertAlmostEqual(result.relative_y_m, 4.0)
        self.assertAlmostEqual(result.planar_distance_m, 5.0)
        self.assertAlmostEqual(result.relative_longitudinal_speed_mps, 3.0)
        self.assertAlmostEqual(result.relative_lateral_speed_mps, 0.0)
        self.assertEqual(result.geometric_region, 'front_left')

    def test_ninety_degree_anchor_rotates_position_and_velocity(self) -> None:
        result = compute_actor_geometry(actor(x=10.0, y=3.0, yaw=math.pi / 2.0, velocity_x=0.0, velocity_y=4.0, speed=4.0), anchor_ego_pose=Pose2D(10.0, 1.0, math.pi / 2.0), ego_speed_mps=1.0)
        self.assertAlmostEqual(result.relative_x_m, 2.0, places=7)
        self.assertAlmostEqual(result.relative_y_m, 0.0, places=7)
        self.assertAlmostEqual(result.actor_velocity_x_mps, 4.0, places=7)
        self.assertAlmostEqual(result.actor_velocity_y_mps, 0.0, places=7)
        self.assertAlmostEqual(result.relative_velocity_x_mps, 3.0, places=7)
        self.assertAlmostEqual(result.relative_yaw_rad, 0.0, places=7)
        self.assertEqual(result.geometric_region, 'front')

    def test_relative_yaw_wraps(self) -> None:
        result = compute_actor_geometry(actor(x=1.0, y=0.0, yaw=math.radians(-179.0)), anchor_ego_pose=Pose2D(0.0, 0.0, math.radians(179.0)), ego_speed_mps=0.0)
        self.assertAlmostEqual(result.relative_yaw_rad, math.radians(2.0))

    def test_static_actor_is_preserved(self) -> None:
        result = compute_actor_geometry(actor(x=2.0, y=0.0, is_static=True), anchor_ego_pose=Pose2D(0.0, 0.0, 0.0), ego_speed_mps=0.0)
        self.assertTrue(result.is_static)
        self.assertEqual(result.track_id, 'actor-1')
        self.assertEqual(result.actor_class, 'automobile')

    def test_to_dict_is_json_compatible_shape(self) -> None:
        result = compute_actor_geometry(actor(x=2.0, y=-1.0), anchor_ego_pose=Pose2D(0.0, 0.0, 0.0), ego_speed_mps=0.0).to_dict()
        self.assertEqual(result['track_id'], 'actor-1')
        self.assertEqual(result['geometric_region'], 'front_right')
        self.assertIsInstance(result['planar_distance_m'], float)

    def test_missing_track_id_is_rejected(self) -> None:
        value = actor(x=1.0, y=0.0)
        del value['track_id']
        with self.assertRaises(ValueError):
            compute_actor_geometry(value, anchor_ego_pose=Pose2D(0.0, 0.0, 0.0), ego_speed_mps=0.0)

    def test_boolean_speed_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            compute_actor_geometry(actor(x=1.0, y=0.0), anchor_ego_pose=Pose2D(0.0, 0.0, 0.0), ego_speed_mps=True)

class SnapshotIntegrationTests(unittest.TestCase):

    @staticmethod
    def ego_message() -> dict:
        qx, qy, qz, qw = yaw_to_quaternion(0.0)
        return {'pose_frame_id': 'map', 'dynamics_frame_id': 'base_link', 'position': {'x': 10.0, 'y': 2.0, 'z': 0.0}, 'orientation': {'x': qx, 'y': qy, 'z': qz, 'w': qw}, 'speed': 3.0}

    def test_recorded_ego_message_parsing(self) -> None:
        pose, speed = recorded_ego_pose_and_speed(self.ego_message())
        self.assertEqual(pose, Pose2D(10.0, 2.0, 0.0))
        self.assertEqual(speed, 3.0)

    def test_snapshot_geometry_integration(self) -> None:
        snapshot = {'pose_frame_id': 'map', 'dynamics_frame_id': 'map', 'actors': [actor(x=15.0, y=2.0, velocity_x=4.0, speed=4.0), actor(x=8.0, y=4.0, track_id='actor-2')]}
        result = compute_snapshot_actor_geometries(snapshot, recorded_ego_message=self.ego_message())
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].relative_x_m, 5.0)
        self.assertAlmostEqual(result[0].relative_velocity_x_mps, 1.0)
        self.assertEqual(result[0].geometric_region, 'front')
        self.assertEqual(result[1].geometric_region, 'rear_left')

    def test_invalid_ego_frames_are_rejected(self) -> None:
        message = self.ego_message()
        message['pose_frame_id'] = 'base_link'
        with self.assertRaises(ValueError):
            recorded_ego_pose_and_speed(message)

    def test_invalid_actor_frames_are_rejected(self) -> None:
        snapshot = {'pose_frame_id': 'map', 'dynamics_frame_id': 'base_link', 'actors': []}
        with self.assertRaises(ValueError):
            compute_snapshot_actor_geometries(snapshot, recorded_ego_message=self.ego_message())

    def test_duplicate_track_ids_are_rejected(self) -> None:
        snapshot = {'pose_frame_id': 'map', 'dynamics_frame_id': 'map', 'actors': [actor(x=1.0, y=0.0), actor(x=2.0, y=0.0)]}
        with self.assertRaises(ValueError):
            compute_snapshot_actor_geometries(snapshot, recorded_ego_message=self.ego_message())
if __name__ == '__main__':
    unittest.main(verbosity=2)

def ego(**updates):
    value = {'lane_match_status': 'matched', 'lane_id': 'A', 'nearest_wait_line_distance_m': None, 'has_intersection_evidence': False, 'intersection_evidence': [], 'lane_length_m': 20.0, 'centerline_arc_length_m': 5.0}
    value.update(updates)
    return value

def test_road_context_rules():
    assert classify_road_context(ego())['type'] == 'lane_following'
    assert classify_road_context(ego(nearest_wait_line_distance_m=10.0, has_intersection_evidence=True))['type'] == 'intersection_approach'
    assert classify_road_context(ego(nearest_wait_line_distance_m=2.0, has_intersection_evidence=True))['type'] == 'intersection'
    assert classify_road_context(ego(lane_match_status='unmatched'))['type'] == 'unknown'

def test_actor_semantic_categories():
    assert relative_distance_category(10.0) == 'near'
    assert relative_distance_category(20.0) == 'medium'
    assert distance_trend('usable', -1.0) == 'approaching'
    assert distance_trend('insufficient_span', None) == 'uncertain'
    assert relative_speed_category(actor_speed_mps=0.1, ego_speed_mps=5.0) == 'stationary'
    assert relative_speed_category(actor_speed_mps=7.0, ego_speed_mps=5.0) == 'faster_than_ego'

class SceneFactSchemaTests(unittest.TestCase):

    def test_identity_and_role_fields_are_required(self):
        for key in ('anchor_id', 'clip_id', 'anchor_ns', *ACTOR_ROLE_KEYS):
            self.assertIn(key, FINAL_RECORD_REQUIRED_KEYS)

    def test_final_presence_does_not_reveal_hidden_actor(self):
        self.assertNotIn('not_observed', FINAL_PRESENCE_STATUSES)
        self.assertEqual(FINAL_PRESENCE_STATUSES, {'present', 'not_present', 'unknown'})

    def test_first_version_has_four_selected_cameras(self):
        self.assertEqual(CAMERA_NAMES, ('front_wide', 'front_tele', 'cross_left', 'cross_right'))

    def test_conservative_unknown_states_exist(self):
        self.assertIn('unknown', ROAD_CONTEXT_TYPES)
        self.assertIn('unknown', OBSERVABILITY_STATUSES)
        self.assertIn('unknown', QUALITY_STATUSES)
        self.assertIn('uncertain', DISTANCE_TREND_CATEGORIES)
        self.assertIn('uncertain', RELATIVE_SPEED_CATEGORIES)

    def test_future_sources_are_explicitly_forbidden(self):
        self.assertIn('actors/future.jsonl', FORBIDDEN_FUTURE_INPUTS)
        self.assertIn('ego/ground_truth_future.jsonl', FORBIDDEN_FUTURE_INPUTS)
        self.assertIn('ego/planner_output.jsonl', FORBIDDEN_FUTURE_INPUTS)

    def test_schema_version_is_draft(self):
        self.assertEqual(SCENE_FACT_FORMAT_VERSION, '0.1-draft')

    def test_json_schema_uses_configured_schema_root(self):
        schema_path = SCHEMA_ROOT / 'scene_fact_schema_v0.1-draft.json'
        self.assertTrue(schema_path.is_file())

    def test_python_vocabulary_matches_json_schema(self):
        schema_path = SCHEMA_ROOT / 'scene_fact_schema_v0.1-draft.json'
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        definitions = schema['$defs']
        comparisons = {'cameraName': set(CAMERA_NAMES), 'proximityStatus': set(PROXIMITY_STATUSES), 'relativePosition': set(RELATIVE_POSITION_REGIONS), 'relativeDistance': set(RELATIVE_DISTANCE_CATEGORIES), 'distanceTrend': set(DISTANCE_TREND_CATEGORIES), 'relativeSpeed': set(RELATIVE_SPEED_CATEGORIES), 'qualityStatus': set(QUALITY_STATUSES)}
        for definition_name, expected in comparisons.items():
            with self.subTest(definition_name=definition_name):
                self.assertEqual(set(definitions[definition_name]['enum']), expected)
        self.assertEqual(set(definitions['roadContext']['properties']['type']['enum']), set(ROAD_CONTEXT_TYPES))
        self.assertEqual(set(definitions['absentActorRole']['properties']['presence_status']['enum']), FINAL_PRESENCE_STATUSES - {'present'})
        self.assertEqual(set(definitions['presentActorRole']['properties']['observability_status']['enum']), {'candidate_visible', 'partially_occluded'})
if __name__ == '__main__':
    unittest.main(verbosity=2)

def test_validator_accepts_minimal_valid_record(tmp_path: Path):
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'additionalProperties': False, 'required': ['x'], 'properties': {'x': {'type': 'integer'}}}
    path = tmp_path / 'schema.json'
    path.write_text(json.dumps(schema))
    validator = load_scene_fact_validator(path)
    validate_scene_fact_record({'x': 1}, validator=validator)
    with pytest.raises(SceneFactValidationError):
        validate_scene_fact_record({'x': 'bad'}, validator=validator)
