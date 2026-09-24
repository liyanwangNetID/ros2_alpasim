"""Tests for the consolidated Step 7 road-context feature domain."""
from __future__ import annotations
import math
from step2.coordinate_utils import Point2D, Pose2D, yaw_to_quaternion
from step2.vector_map_reader import VectorMapReader
from step7.road_context import RoadFeatureMapContext, classify_lane_direction_relation, compute_ego_and_actor_road_features, match_point_to_road
from step7.road_context import summarize_road_feature_rows

def poly(points):
    return {'points': [{'x': x, 'y': y, 'z': 0} for x, y in points], 'headings': []}

def lane(lane_id, y, wait=()):
    return {'id': lane_id, 'centerline': poly([(0, y), (10, y)]), 'left_boundary': poly([(0, y + 1), (10, y + 1)]), 'right_boundary': poly([(0, y - 1), (10, y - 1)]), 'predecessor_ids': [], 'successor_ids': [], 'left_adjacent_ids': [], 'right_adjacent_ids': [], 'road_area_ids': [], 'traffic_sign_ids': [], 'wait_line_ids': list(wait)}

def raw_map():
    return {'frame_id': 'map', 'map_id': 'm', 'revision': 1, 'lanes': [lane('A', 0, ('W',)), lane('B', 4)], 'road_edges': [], 'traffic_signs': [], 'wait_lines': [{'id': 'W', 'is_implicit': False, 'wait_line_type': 'STOP', 'polyline': poly([(8, -1), (8, 1)])}]}

def actor():
    qx, qy, qz, qw = yaw_to_quaternion(0)
    return {'track_id': '1', 'label_class': 'automobile', 'pose': {'position': {'x': 6, 'y': 0, 'z': 0}, 'orientation': {'x': qx, 'y': qy, 'z': qz, 'w': qw}}}

def test_spatial_index_matches_brute_force_queries():
    raw = raw_map()
    context = RoadFeatureMapContext(raw_map=raw)
    brute = VectorMapReader.from_dict(raw)
    for point in (Point2D(1, 0), Point2D(5, 2), Point2D(100, 100)):
        indexed = context.find_nearby_lanes(point, radius_m=6.0, limit=5)
        expected = brute.find_nearby_lanes(point.x, point.y, radius_m=6.0, limit=5)
        assert [(x.lane_id, x.distance_m) for x in indexed] == [(x.lane_id, x.distance_m) for x in expected]

def test_match_reports_wait_line_evidence():
    context = RoadFeatureMapContext(raw_map=raw_map())
    result = match_point_to_road(context=context, point=Point2D(0, 0))
    assert result.lane_id == 'A'
    assert result.has_wait_line is True
    assert result.nearest_wait_line_type == 'STOP'
    assert result.has_intersection_evidence is True

def test_ego_actor_relation_and_order():
    context = RoadFeatureMapContext(raw_map=raw_map())
    ego, rows = compute_ego_and_actor_road_features(context=context, ego_pose=Pose2D(2, 0, 0), actors=(actor(),))
    assert ego.lane_id == 'A'
    assert rows[0][3] == 'same'
    assert rows[0][4] == 'same_direction'

def test_unmatched_is_explicit():
    context = RoadFeatureMapContext(raw_map=raw_map())
    result = match_point_to_road(context=context, point=Point2D(100, 100))
    assert result.status == 'unmatched'
    assert result.lane_id is None

def test_summary_counts_statuses_and_relations():
    r = summarize_road_feature_rows(keyframes=({'anchor_id': 'a'},), ego_rows=({'anchor_id': 'a', 'lane_match_status': 'matched', 'has_intersection_evidence': True},), actor_rows=({'anchor_id': 'a', 'track_id': '1', 'lane_match_status': 'matched', 'ego_lane_relation': 'same', 'lane_direction_relation': 'same_direction', 'has_intersection_evidence': False},))
    assert r['keyframe_count'] == 1 and r['actor_ego_lane_relation_counts'] == {'same': 1}
    assert r['actor_lane_direction_relation_counts'] == {'same_direction': 1}

def test_lane_direction_relation_classifies_parallel_opposing_and_crossing():
    context = RoadFeatureMapContext(raw_map=raw_map())
    ego = match_point_to_road(context=context, point=Point2D(2, 0))
    same = match_point_to_road(context=context, point=Point2D(6, 0))
    assert classify_lane_direction_relation(ego, same) == 'same_direction'
    opposing = type(same)(
        same.status, same.lane_id, same.centerline_distance_m,
        same.centerline_arc_length_m, same.lane_length_m,
        same.inside_lane_polygon, same.heading_error_rad,
        math.pi, same.has_wait_line, same.wait_line_ids,
        same.nearest_wait_line_id, same.nearest_wait_line_type,
        same.nearest_wait_line_distance_m, same.nearest_wait_line_is_implicit,
        same.has_intersection_evidence, same.intersection_evidence,
    )
    crossing = type(same)(
        same.status, same.lane_id, same.centerline_distance_m,
        same.centerline_arc_length_m, same.lane_length_m,
        same.inside_lane_polygon, same.heading_error_rad,
        math.pi / 2.0, same.has_wait_line, same.wait_line_ids,
        same.nearest_wait_line_id, same.nearest_wait_line_type,
        same.nearest_wait_line_distance_m, same.nearest_wait_line_is_implicit,
        same.has_intersection_evidence, same.intersection_evidence,
    )
    assert classify_lane_direction_relation(ego, opposing) == 'opposing'
    assert classify_lane_direction_relation(ego, crossing) == 'unknown'
