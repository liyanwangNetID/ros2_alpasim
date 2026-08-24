#!/usr/bin/env python3
from __future__ import annotations
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from scan_lateral_action_thresholds import (
    eligible_turn_base,
    evaluate_threshold_pair,
    parse_thresholds,
    signs_agree,
)


def record(anchor_id, level, ego_deg, map_deg, *, gate=True, adjacent=False):
    import math
    return {
        "anchor_id": anchor_id,
        "lateral": {
            "topology": {"junction_evidence_level": level},
            "lateral_quality_gate": {"passed": gate},
            "contains_adjacent_transition": adjacent,
            "trajectory_yaw_signed_change_rad": math.radians(ego_deg),
            "map_corridor_heading_change_rad": math.radians(map_deg),
        },
    }


class ScanTests(unittest.TestCase):
    def test_parse_thresholds(self):
        self.assertEqual(parse_thresholds("20,10,20"), (10.0, 20.0))
        with self.assertRaises(ValueError):
            parse_thresholds("0")

    def test_sign_agreement(self):
        self.assertTrue(signs_agree(1.0, 2.0))
        self.assertTrue(signs_agree(-1.0, -2.0))
        self.assertFalse(signs_agree(1.0, -1.0))
        self.assertFalse(signs_agree(0.0, 1.0))

    def test_base_exclusions(self):
        self.assertEqual(
            eligible_turn_base(record("a", "A", 20, 20, gate=False))[1],
            "quality_gate_failed",
        )
        self.assertEqual(
            eligible_turn_base(record("a", "A", 20, 20, adjacent=True))[1],
            "contains_adjacent_transition",
        )

    def test_threshold_evaluation(self):
        records = [
            record("positive", "A", 25, 22),
            record("negative", "A", -30, -21),
            record("disagree", "A", 30, -30),
            record("small", "A", 5, 5),
            record("wrong_level", "B", 40, 40),
        ]
        result = evaluate_threshold_pair(
            records,
            evidence_level="A",
            ego_threshold_deg=20,
            map_threshold_deg=20,
        )
        self.assertEqual(result["counts"]["candidate_total"], 2)
        self.assertEqual(result["counts"]["positive_candidate"], 1)
        self.assertEqual(result["counts"]["negative_candidate"], 1)
        self.assertEqual(result["counts"]["direction_disagreement"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
