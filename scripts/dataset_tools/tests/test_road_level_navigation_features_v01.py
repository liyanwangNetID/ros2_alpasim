#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from road_level_navigation_features_v01 import extract_road_level_features


class RoadLevelNavigationFeatureTests(unittest.TestCase):
    def test_straight_route(self):
        points = [(float(x), 0.0) for x in range(0, 81, 5)]
        result = extract_road_level_features(points, branch_distance_m=30.0)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["route_road_level_heading_change_deg"], 0.0)

    def test_left_post_branch_route(self):
        points = [
            (0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0),
            (40.0, 0.0), (47.0, 3.0), (52.0, 8.0), (55.0, 15.0),
            (56.0, 25.0), (56.0, 35.0),
        ]
        result = extract_road_level_features(points, branch_distance_m=30.0)
        self.assertEqual(result["status"], "available")
        self.assertGreater(result["route_road_level_heading_change_deg"], 20.0)

    def test_insufficient_post_branch_length(self):
        points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        result = extract_road_level_features(points, branch_distance_m=15.0)
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
