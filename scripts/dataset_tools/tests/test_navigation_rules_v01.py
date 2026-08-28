#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation_rules_v01 import classify_navigation


def route(
    status="usable",
    signed_heading_deg=0.0,
    lookahead_distance_m=80.0,
):
    return {
        "quality_status": status,
        "route_lookahead_distance_m":
            lookahead_distance_m,
        "route": {
            "route_signed_heading_change_rad":
                math.radians(
                    signed_heading_deg
                ),
        },
    }


def branch(
    first=None,
    intersection_distance=None,
    status="usable",
    speed_mps=5.0,
):
    evidence = None
    if intersection_distance is not None:
        evidence = {
            "route_distance_m": intersection_distance,
            "evidence": ["wait_line"],
        }
    return {
        "quality_status": status,
        "anchor_speed_mps": speed_mps,
        "branch_context": {
            "first_observed_branch": first,
            "first_intersection_evidence": evidence,
        },
    }


class NavigationRuleTests(unittest.TestCase):
    def test_no_branch_straight_road(self):
        result = classify_navigation(route(), branch())
        self.assertEqual(result["action"], "straight")
        self.assertEqual(result["text"], "Continue along the road.")

    def test_no_branch_straight_intersection(self):
        result = classify_navigation(route(), branch(intersection_distance=20.0))
        self.assertEqual(result["action"], "straight")
        self.assertIn("intersection", result["text"])

    def test_reliable_left_intersection(self):
        first = {
            "route_relation_to_natural": "left_of_natural",
            "reliability_status": "reliable",
            "route_distance_m": 10.0,
        }
        result = classify_navigation(
            route(signed_heading_deg=30.0),
            branch(first, 30.0),
        )
        self.assertEqual(result["action"], "left")
        self.assertEqual(result["text"], "Turn left at the upcoming intersection.")

    def test_reliable_right_branch(self):
        first = {
            "route_relation_to_natural": "right_of_natural",
            "reliability_status": "reliable",
            "route_distance_m": 10.0,
        }
        result = classify_navigation(
            route(signed_heading_deg=-30.0),
            branch(first, 60.0),
        )
        self.assertEqual(result["action"], "right")
        self.assertEqual(result["text"], "Follow the right branch ahead.")

    def test_unreliable_is_unknown(self):
        first = {
            "route_relation_to_natural": "left_of_natural",
            "reliability_status": "unreliable",
            "route_distance_m": 10.0,
        }
        result = classify_navigation(route(), branch(first, 10.0))
        self.assertEqual(result["action"], "unknown")
        self.assertIsNone(result["text"])

    def test_unresolved_successor_is_unknown(self):
        first = {
            "route_relation_to_natural": "actual_successor_not_candidate",
            "reliability_status": "reliable",
            "route_distance_m": 10.0,
        }
        result = classify_navigation(route(), branch(first, 10.0))
        self.assertEqual(result["action"], "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
