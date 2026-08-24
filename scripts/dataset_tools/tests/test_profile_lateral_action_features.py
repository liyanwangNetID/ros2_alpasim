#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from coordinate_utils import Point2D
from profile_lateral_action_features import (
    filtered_path_heading_features,
    signed_and_absolute_change,
    topology_evidence,
)
from vector_map_reader import VectorMapReader


def polyline(points):
    return {
        "points": [
            {"x": x, "y": y, "z": 0.0} for x, y in points
        ],
        "headings": [],
    }


def lane(
    lane_id,
    *,
    predecessors=(),
    successors=(),
    wait=(),
    signs=(),
):
    return {
        "id": lane_id,
        "centerline": polyline([(0.0, 0.0), (10.0, 0.0)]),
        "left_boundary": polyline([(0.0, 1.0), (10.0, 1.0)]),
        "right_boundary": polyline([(0.0, -1.0), (10.0, -1.0)]),
        "predecessor_ids": list(predecessors),
        "successor_ids": list(successors),
        "left_adjacent_ids": [],
        "right_adjacent_ids": [],
        "road_area_ids": [],
        "traffic_sign_ids": list(signs),
        "wait_line_ids": list(wait),
    }


def vector_map(lanes):
    return VectorMapReader.from_dict(
        {
            "frame_id": "map",
            "map_id": "synthetic",
            "revision": 1,
            "lanes": lanes,
            "road_edges": [],
            "traffic_signs": [],
            "wait_lines": [],
        }
    )


class HeadingFeatureTests(unittest.TestCase):
    def test_unwrapped_signed_change(self):
        signed, absolute = signed_and_absolute_change(
            [math.radians(170), math.radians(179), math.radians(-170)]
        )
        self.assertAlmostEqual(signed, math.radians(20))
        self.assertAlmostEqual(absolute, math.radians(20))

    def test_short_segments_are_rejected(self):
        result = filtered_path_heading_features(
            (
                Point2D(0.0, 0.0),
                Point2D(0.01, 0.01),
                Point2D(1.0, 0.0),
                Point2D(2.0, 1.0),
            ),
            minimum_segment_length_m=0.1,
        )
        self.assertEqual(result["valid_path_segment_count"], 2)
        self.assertEqual(result["rejected_path_segment_count"], 1)
        self.assertTrue(result["path_heading_reliable"])

    def test_all_short_segments_are_unreliable(self):
        result = filtered_path_heading_features(
            (Point2D(0.0, 0.0), Point2D(0.01, 0.0)),
            minimum_segment_length_m=0.1,
        )
        self.assertFalse(result["path_heading_reliable"])
        self.assertEqual(
            result["filtered_path_absolute_heading_change_rad"], 0.0
        )


class JunctionLevelTests(unittest.TestCase):
    def test_level_a_wait_line(self):
        result = topology_evidence(
            vector_map([lane("A", wait=("W",))]),
            ["A"],
        )
        self.assertEqual(result["junction_evidence_level"], "A")
        self.assertIn(
            "sequence_contains_wait_line_lane",
            result["junction_evidence_reasons"],
        )

    def test_level_b_boundary_topology(self):
        result = topology_evidence(
            vector_map(
                [
                    lane("P", successors=("A", "X")),
                    lane("X", predecessors=("P",)),
                    lane("A", predecessors=("P",), successors=("B",)),
                    lane("B", predecessors=("A", "M")),
                    lane("M", successors=("B",)),
                ]
            ),
            ["A"],
        )
        self.assertEqual(result["junction_evidence_level"], "B")
        self.assertIn(
            "start_lane_follows_branching_predecessor",
            result["junction_evidence_reasons"],
        )
        self.assertIn(
            "end_lane_precedes_merging_successor",
            result["junction_evidence_reasons"],
        )

    def test_level_c_traffic_sign_only(self):
        result = topology_evidence(
            vector_map([lane("A", signs=("S",))]),
            ["A"],
        )
        self.assertEqual(result["junction_evidence_level"], "C")
        self.assertTrue(result["traffic_sign_only"])

    def test_level_a_takes_priority_over_level_b(self):
        result = topology_evidence(
            vector_map(
                [
                    lane("P", successors=("A", "X")),
                    lane("X", predecessors=("P",)),
                    lane("A", predecessors=("P",), wait=("W",)),
                ]
            ),
            ["A"],
        )
        self.assertEqual(result["junction_evidence_level"], "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
