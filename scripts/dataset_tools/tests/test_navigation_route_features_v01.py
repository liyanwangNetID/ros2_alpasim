#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation_route_features_v01 import route_geometry_features, valid_local_points


class NavigationRouteFeatureTests(unittest.TestCase):
    def test_straight_geometry(self):
        result = route_geometry_features([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
        self.assertAlmostEqual(result["route_signed_heading_change_rad"], 0.0)
        self.assertAlmostEqual(result["final_local_y_m"], 0.0)
        self.assertEqual(result["forward_point_fraction"], 1.0)

    def test_left_curve_has_positive_heading_change(self):
        result = route_geometry_features([(0.0, 0.0), (5.0, 0.0), (9.0, 3.0), (11.0, 7.0)])
        self.assertGreater(result["route_signed_heading_change_rad"], 0.0)
        self.assertGreater(result["final_local_y_m"], 0.0)

    def test_valid_points_filter(self):
        message = {"points": [
            {"valid": True, "position": {"x": 0.0, "y": 1.0}},
            {"valid": False, "position": {"x": 2.0, "y": 3.0}},
            {"valid": True, "position": {"x": 4.0, "y": 5.0}},
        ]}
        self.assertEqual(valid_local_points(message), [(0.0, 1.0), (4.0, 5.0)])

    def test_short_route_rejected(self):
        with self.assertRaises(ValueError):
            route_geometry_features([(0.0, 0.0)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
