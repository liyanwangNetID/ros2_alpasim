#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation_rules_v01 import classify_navigation


def route(lookahead=80.0):
    return {
        "quality_status": "usable",
        "route_lookahead_distance_m": lookahead,
        "route": {"route_signed_heading_change_rad": math.radians(30.0)},
    }


def branch(*, relation="natural_continuation", reliability="reliable", intersection=True):
    return {
        "quality_status": "usable",
        "anchor_speed_mps": 2.0,
        "branch_context": {
            "first_intersection_evidence": (
                {"route_distance_m": 8.0, "evidence": ["wait_line", "branching"]}
                if intersection else None
            ),
            "first_observed_branch": {
                "route_distance_m": 8.0,
                "route_relation_to_natural": relation,
                "reliability_status": reliability,
            },
        },
    }


def road_level(change_deg=None, status="available"):
    geometry = {"status": status}
    if change_deg is not None:
        geometry["route_road_level_heading_change_deg"] = change_deg
    return {
        "feature_format_version": "0.1-draft",
        "road_level_route_geometry": geometry,
    }


class NavigationRuleV014Tests(unittest.TestCase):
    def test_natural_curve_without_intersection_continues_along_road(self):
        result = classify_navigation(
            route(), branch(intersection=False), road_level(52.0)
        )
        self.assertEqual(result["action"], "straight")
        self.assertEqual(result["text"], "Continue along the road.")

    def test_natural_continuation_at_intersection_with_small_change_is_straight(self):
        result = classify_navigation(route(), branch(), road_level(9.147))
        self.assertEqual(result["action"], "straight")
        self.assertEqual(
            result["text"], "Continue straight through the upcoming intersection."
        )

    def test_natural_continuation_at_intersection_with_large_left_change_is_left(self):
        result = classify_navigation(route(), branch(), road_level(52.776))
        self.assertEqual(result["action"], "left")
        self.assertEqual(result["text"], "Turn left at the upcoming intersection.")

    def test_natural_continuation_at_intersection_with_large_right_change_is_right(self):
        result = classify_navigation(route(), branch(), road_level(-91.74))
        self.assertEqual(result["action"], "right")
        self.assertEqual(result["text"], "Turn right at the upcoming intersection.")

    def test_unreliable_natural_relation_can_use_available_road_level_evidence(self):
        result = classify_navigation(
            route(), branch(reliability="unreliable"), road_level(28.447)
        )
        self.assertEqual(result["action"], "left")

    def test_unreliable_natural_relation_without_road_level_evidence_is_unknown(self):
        result = classify_navigation(
            route(), branch(reliability="unreliable"), road_level(None, "unavailable")
        )
        self.assertEqual(result["action"], "unknown")

    def test_reliable_natural_relation_without_geometry_preserves_legacy_straight(self):
        result = classify_navigation(
            route(), branch(reliability="reliable"), road_level(None, "unavailable")
        )
        self.assertEqual(result["action"], "straight")


    def test_unreliable_below_threshold_remains_unknown(self):
        result = classify_navigation(
            route(),
            branch(reliability="unreliable"),
            road_level(-19.399),
        )
        self.assertEqual(result["action"], "unknown")
        self.assertIn(
            "road_level_direction_below_intersection_threshold",
            result["reasons"],
        )


    def test_inconsistent_natural_successor_identity_is_unknown(self):
        branch_record = branch(
            relation="natural_continuation",
            reliability="reliable",
        )
        first_branch = branch_record[
            "branch_context"
        ]["first_observed_branch"]

        first_branch["natural_successor_lane_id"] = "lane_a"
        first_branch["route_successor_lane_id"] = "lane_b"

        result = classify_navigation(
            route(),
            branch_record,
            road_level(52.0),
        )

        self.assertEqual(result["action"], "unknown")
        self.assertIn(
            "inconsistent_natural_continuation_successor_identity",
            result["reasons"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
