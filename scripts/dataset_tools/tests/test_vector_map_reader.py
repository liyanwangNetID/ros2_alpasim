#!/usr/bin/env python3
"""Synthetic and real-map tests for vector_map_reader.py."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from coordinate_utils import Point2D  # noqa: E402
from project_paths import ALPASIM_DATA_ROOT  # noqa: E402
from vector_map_reader import (  # noqa: E402
    VectorMapError,
    VectorMapReader,
    build_lane_polygon,
    point_in_polygon,
    project_point_to_polyline,
)


def polyline(points):
    return {
        "points": [
            {"x": x, "y": y, "z": 0.0}
            for x, y in points
        ],
        "headings": [],
    }


def lane(
    lane_id,
    center_y,
    *,
    predecessors=(),
    successors=(),
    left=(),
    right=(),
):
    return {
        "id": lane_id,
        "centerline": polyline([(0.0, center_y), (10.0, center_y)]),
        "left_boundary": polyline(
            [(0.0, center_y + 1.0), (10.0, center_y + 1.0)]
        ),
        "right_boundary": polyline(
            [(0.0, center_y - 1.0), (10.0, center_y - 1.0)]
        ),
        "predecessor_ids": list(predecessors),
        "successor_ids": list(successors),
        "left_adjacent_ids": list(left),
        "right_adjacent_ids": list(right),
        "road_area_ids": [],
        "traffic_sign_ids": [],
        "wait_line_ids": [],
    }


def synthetic_map():
    return {
        "frame_id": "map",
        "map_id": "synthetic",
        "revision": 1,
        "lanes": [
            lane("A", 0.0, successors=("B",), left=("L",)),
            lane("B", 0.0, predecessors=("A",)),
            lane("L", 3.0, right=("A",)),
            lane("M", -3.0, left=("A",), successors=("MISSING",)),
        ],
        "road_edges": [],
        "traffic_signs": [],
        "wait_lines": [],
    }


class GeometryTests(unittest.TestCase):
    def test_polygon_contains_boundary_and_inside(self):
        polygon = build_lane_polygon(
            (Point2D(0.0, 1.0), Point2D(10.0, 1.0)),
            (Point2D(0.0, -1.0), Point2D(10.0, -1.0)),
        )
        self.assertTrue(point_in_polygon(Point2D(5.0, 0.0), polygon))
        self.assertTrue(point_in_polygon(Point2D(5.0, 1.0), polygon))
        self.assertFalse(point_in_polygon(Point2D(5.0, 2.0), polygon))

    def test_projection(self):
        projection = project_point_to_polyline(
            Point2D(4.0, 3.0),
            (Point2D(0.0, 0.0), Point2D(10.0, 0.0)),
        )
        self.assertAlmostEqual(projection.point.x, 4.0)
        self.assertAlmostEqual(projection.point.y, 0.0)
        self.assertAlmostEqual(projection.distance_m, 3.0)
        self.assertAlmostEqual(projection.heading_rad, 0.0)
        self.assertAlmostEqual(projection.arc_length_m, 4.0)


class ParsingAndTopologyTests(unittest.TestCase):
    def setUp(self):
        self.reader = VectorMapReader.from_dict(synthetic_map())

    def test_parse(self):
        self.assertEqual(len(self.reader), 4)
        self.assertEqual(self.reader.frame_id, "map")
        self.assertAlmostEqual(self.reader.require_lane("A").length_m, 10.0)

    def test_relations(self):
        self.assertEqual(self.reader.relation("A", "A"), "same")
        self.assertEqual(self.reader.relation("A", "B"), "successor")
        self.assertEqual(self.reader.relation("A", "L"), "left_adjacent")
        self.assertEqual(self.reader.relation("L", "A"), "right_adjacent")
        self.assertEqual(self.reader.relation("A", "M"), "unrelated")

    def test_missing_reference_is_warning(self):
        self.assertEqual(len(self.reader.topology_warnings), 1)
        warning = self.reader.topology_warnings[0]
        self.assertEqual(warning.source_lane_id, "M")
        self.assertEqual(warning.missing_lane_id, "MISSING")
        self.assertEqual(
            self.reader.valid_related_lane_ids("M", "successor"),
            tuple(),
        )

    def test_follow_unique_successor(self):
        self.assertEqual(
            self.reader.follow_successors("A", maximum_depth=5),
            ("A", "B"),
        )

    def test_lane_contains_point(self):
        self.assertTrue(
            self.reader.lane_contains_point("A", Point2D(5.0, 0.5))
        )
        self.assertFalse(
            self.reader.lane_contains_point("A", Point2D(5.0, 2.0))
        )

    def test_nearby_lane_sorting_prefers_polygon_containment(self):
        candidates = self.reader.find_nearby_lanes(
            5.0,
            0.8,
            radius_m=5.0,
            yaw_rad=0.0,
            maximum_heading_error_rad=0.2,
        )
        self.assertEqual(candidates[0].lane_id, "A")
        self.assertTrue(candidates[0].inside_polygon)

    def test_heading_filter(self):
        candidates = self.reader.find_nearby_lanes(
            5.0,
            0.0,
            radius_m=2.0,
            yaw_rad=math.pi,
            maximum_heading_error_rad=0.2,
        )
        self.assertEqual(candidates, tuple())

    def test_duplicate_lane_id_rejected(self):
        data = synthetic_map()
        data["lanes"].append(data["lanes"][0])
        with self.assertRaises(VectorMapError):
            VectorMapReader.from_dict(data)


class RealMapSmokeTest(unittest.TestCase):
    def test_real_map(self):
        path = (
            ALPASIM_DATA_ROOT
            / "test_clip_001"
            / "map"
            / "vector_map.json"
        )
        if not path.is_file():
            self.skipTest("real VectorMap is unavailable")
        data = json.loads(path.read_text(encoding="utf-8"))
        reader = VectorMapReader.from_dict(data)
        self.assertEqual(reader.frame_id, "map")
        self.assertEqual(len(reader), 442)
        self.assertEqual(len(reader.topology_warnings), 1)
        lane_a = reader.require_lane("17448943")
        self.assertIn("17448944", lane_a.right_adjacent_ids)
        self.assertEqual(
            reader.relation("17448943", "17448944"),
            "right_adjacent",
        )
        projection = reader.project_to_lane(
            lane_a.lane_id,
            lane_a.centerline[0],
        )
        self.assertAlmostEqual(projection.distance_m, 0.0)
        nearby = reader.find_nearby_lanes(
            lane_a.centerline[0].x,
            lane_a.centerline[0].y,
            radius_m=5.0,
            yaw_rad=projection.heading_rad,
            maximum_heading_error_rad=0.5,
            limit=5,
        )
        self.assertTrue(any(item.lane_id == lane_a.lane_id for item in nearby))


if __name__ == "__main__":
    unittest.main(verbosity=2)
