#!/usr/bin/env python3
from __future__ import annotations
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from refine_lane_matching_features import (
    adjacent_transition_evidence,
    conservative_quality_gate,
    returns_to_previous_lane,
)


class QualityGateTests(unittest.TestCase):
    def test_usable_successor_only(self):
        record = {
            "matched_fraction": 1.0,
            "compressed_lane_sequence": ["A", "B"],
            "transitions": [{"relation": "successor"}],
        }
        self.assertTrue(conservative_quality_gate(record)["passed"])

    def test_all_conservative_reasons(self):
        record = {
            "matched_fraction": 0.7,
            "compressed_lane_sequence": ["A", "B", "A"],
            "transitions": [
                {"relation": "left_adjacent"},
                {"relation": "right_adjacent"},
                {"relation": "unrelated"},
                {"relation": "predecessor"},
            ],
        }
        gate = conservative_quality_gate(record)
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["reasons"]), 5)

    def test_return_detection(self):
        self.assertTrue(returns_to_previous_lane(["A", "B", "A"]))
        self.assertFalse(returns_to_previous_lane(["A", "B", "C"]))


class AdjacentPersistenceTests(unittest.TestCase):
    def test_direct_adjacent_to_horizon(self):
        record = {
            "compressed_lane_sequence": ["A", "L"],
            "transitions": [
                {
                    "source_lane_id": "A",
                    "target_lane_id": "L",
                    "relation": "left_adjacent",
                    "target_point_index": 3,
                }
            ],
        }
        stamps = [i * 100_000_000 for i in range(10)]
        evidence = adjacent_transition_evidence(record, stamps)[0]
        self.assertEqual(evidence["direct_target_point_count"], 7)
        self.assertEqual(evidence["corridor_point_count"], 7)
        self.assertAlmostEqual(evidence["corridor_duration_sec"], 0.6)
        self.assertTrue(evidence["corridor_reaches_horizon"])

    def test_successor_extends_target_corridor(self):
        record = {
            "compressed_lane_sequence": ["A", "L", "L2", "X"],
            "transitions": [
                {"source_lane_id": "A", "target_lane_id": "L", "relation": "left_adjacent", "target_point_index": 2},
                {"source_lane_id": "L", "target_lane_id": "L2", "relation": "successor", "target_point_index": 5},
                {"source_lane_id": "L2", "target_lane_id": "X", "relation": "unrelated", "target_point_index": 8},
            ],
        }
        stamps = [i * 100_000_000 for i in range(10)]
        evidence = adjacent_transition_evidence(record, stamps)[0]
        self.assertEqual(evidence["direct_target_point_count"], 3)
        self.assertEqual(evidence["corridor_point_count"], 6)
        self.assertEqual(evidence["successor_transitions_after_adjacent"], 1)
        self.assertEqual(evidence["terminated_by_relation"], "unrelated")

    def test_return_and_opposite_direction(self):
        record = {
            "compressed_lane_sequence": ["A", "R", "A"],
            "transitions": [
                {"source_lane_id": "A", "target_lane_id": "R", "relation": "right_adjacent", "target_point_index": 2},
                {"source_lane_id": "R", "target_lane_id": "A", "relation": "left_adjacent", "target_point_index": 6},
            ],
        }
        stamps = [i * 100_000_000 for i in range(10)]
        evidence = adjacent_transition_evidence(record, stamps)[0]
        self.assertTrue(evidence["returns_to_source_lane"])
        self.assertTrue(evidence["contains_opposite_adjacent_after"])
        self.assertEqual(evidence["corridor_point_count"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
