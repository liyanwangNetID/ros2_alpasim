import math

from step2.coordinate_utils import Point2D, Pose2D, yaw_to_quaternion
from step2.vector_map_reader import VectorMapReader
from step7.road_lane_features_v01 import (
    RoadFeatureMapContext,
    compute_ego_and_actor_road_features,
    match_point_to_road,
)


def poly(points):
    return {"points": [{"x": x, "y": y, "z": 0} for x, y in points], "headings": []}


def lane(lane_id, y, wait=()):
    return {
        "id": lane_id,
        "centerline": poly([(0, y), (10, y)]),
        "left_boundary": poly([(0, y + 1), (10, y + 1)]),
        "right_boundary": poly([(0, y - 1), (10, y - 1)]),
        "predecessor_ids": [],
        "successor_ids": [],
        "left_adjacent_ids": [],
        "right_adjacent_ids": [],
        "road_area_ids": [],
        "traffic_sign_ids": [],
        "wait_line_ids": list(wait),
    }


def raw_map():
    return {
        "frame_id": "map",
        "map_id": "m",
        "revision": 1,
        "lanes": [lane("A", 0, ("W",)), lane("B", 4)],
        "road_edges": [],
        "traffic_signs": [],
        "wait_lines": [{
            "id": "W",
            "is_implicit": False,
            "wait_line_type": "STOP",
            "polyline": poly([(8, -1), (8, 1)]),
        }],
    }


def actor():
    qx, qy, qz, qw = yaw_to_quaternion(0)
    return {
        "track_id": "1",
        "label_class": "automobile",
        "pose": {
            "position": {"x": 6, "y": 0, "z": 0},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
    }


def test_spatial_index_matches_brute_force_queries():
    raw = raw_map()
    context = RoadFeatureMapContext(raw_map=raw)
    brute = VectorMapReader.from_dict(raw)
    for point in (Point2D(1, 0), Point2D(5, 2), Point2D(100, 100)):
        indexed = context.find_nearby_lanes(point, radius_m=6.0, limit=5)
        expected = brute.find_nearby_lanes(point.x, point.y, radius_m=6.0, limit=5)
        assert [(x.lane_id, x.distance_m) for x in indexed] == [
            (x.lane_id, x.distance_m) for x in expected
        ]


def test_match_reports_wait_line_evidence():
    context = RoadFeatureMapContext(raw_map=raw_map())
    result = match_point_to_road(context=context, point=Point2D(0, 0))
    assert result.lane_id == "A"
    assert result.has_wait_line is True
    assert result.nearest_wait_line_type == "STOP"
    assert result.has_intersection_evidence is True


def test_ego_actor_relation_and_order():
    context = RoadFeatureMapContext(raw_map=raw_map())
    ego, rows = compute_ego_and_actor_road_features(
        context=context,
        ego_pose=Pose2D(2, 0, 0),
        actors=(actor(),),
    )
    assert ego.lane_id == "A"
    assert rows[0][3] == "same"


def test_unmatched_is_explicit():
    context = RoadFeatureMapContext(raw_map=raw_map())
    result = match_point_to_road(context=context, point=Point2D(100, 100))
    assert result.status == "unmatched"
    assert result.lane_id is None
