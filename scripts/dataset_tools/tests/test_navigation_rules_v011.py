#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation_rules_v01 import classify_navigation, dynamic_upcoming_distance_m


def route(signed_deg=0.0, lookahead=80.0, status='usable'):
    return {
        'quality_status': status,
        'route_lookahead_distance_m': lookahead,
        'route': {'route_signed_heading_change_rad': math.radians(signed_deg)},
    }


def branch(first=None, intersection_distance=None, speed=5.0, status='usable'):
    evidence = None if intersection_distance is None else {
        'route_distance_m': intersection_distance,
        'evidence': ['wait_line'],
    }
    return {
        'quality_status': status,
        'anchor_speed_mps': speed,
        'branch_context': {
            'first_observed_branch': first,
            'first_intersection_evidence': evidence,
        },
    }


class NavigationRuleV012Tests(unittest.TestCase):
    def test_dynamic_preview_distance(self):
        self.assertEqual(dynamic_upcoming_distance_m(current_speed_mps=0.0, route_lookahead_distance_m=80.0), 15.0)
        self.assertEqual(dynamic_upcoming_distance_m(current_speed_mps=5.0, route_lookahead_distance_m=80.0), 50.0)
        self.assertEqual(dynamic_upcoming_distance_m(current_speed_mps=10.0, route_lookahead_distance_m=80.0), 80.0)

    def test_left_requires_consistent_geometry(self):
        first = {'route_relation_to_natural': 'left_of_natural', 'reliability_status': 'reliable', 'route_distance_m': 10.0}
        self.assertEqual(classify_navigation(route(30.0), branch(first))['action'], 'left')
        self.assertEqual(
            classify_navigation(
                route(1.0),
                branch(first),
            )["action"],
            "unknown",
        )
        self.assertEqual(
            classify_navigation(
                route(3.735),
                branch(first),
            )["action"],
            "unknown",
        )
        self.assertEqual(
            classify_navigation(
                route(-8.0),
                branch(first),
            )["action"],
            "unknown",
        )

    def test_right_requires_consistent_geometry(self):
        first = {'route_relation_to_natural': 'right_of_natural', 'reliability_status': 'reliable', 'route_distance_m': 10.0}
        self.assertEqual(classify_navigation(route(-30.0), branch(first))['action'], 'right')
        self.assertEqual(classify_navigation(route(-1.0), branch(first))['action'], 'unknown')
        self.assertEqual(classify_navigation(route(8.0), branch(first))['action'], 'unknown')

    def test_time_based_intersection_template(self):
        self.assertIn('intersection', classify_navigation(route(), branch(intersection_distance=45.0, speed=5.0))['text'])
        self.assertEqual(classify_navigation(route(), branch(intersection_distance=45.0, speed=2.0))['text'], 'Continue along the road.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
