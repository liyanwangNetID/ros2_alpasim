#!/usr/bin/env python3
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from scan_branch_relative_lateral_thresholds import (
    base_eligibility,
    evidence_metrics,
    evaluate,
    parse_thresholds,
)


def record(anchor_id, relation, *, first=0, second=0, total=0, start=0, middle=0, end=0, maximum=0):
    direction = "directional_branch_observed" if relation else "natural_or_unobserved_keep_direction"
    return {
        "anchor_id": anchor_id,
        "lateral": {
            "lateral_quality_gate": {"passed": True},
            "contains_adjacent_transition": False,
            "ego_total_yaw_change_rad": math.radians(10.0),
            "ego_maximum_yaw_excursion_rad": math.radians(10.0),
            "ego_total_absolute_yaw_change_rad": math.radians(10.0),
            "natural_corridor": {
                "turn_evidence_status": direction,
                "reliable_directional_relations": [relation] if relation else [],
                "relative_heading": {
                    "relative_first_half_heading_change_rad": math.radians(first),
                    "relative_second_half_heading_change_rad": math.radians(second),
                    "relative_total_heading_change_rad": math.radians(total),
                    "relative_heading_start_rad": math.radians(start),
                    "relative_heading_middle_rad": math.radians(middle),
                    "relative_heading_end_rad": math.radians(end),
                    "maximum_absolute_relative_heading_rad": math.radians(maximum),
                },
            },
        },
    }


class BranchRelativeScanTests(unittest.TestCase):
    def test_parse_thresholds(self):
        self.assertEqual(parse_thresholds("10,0,10", allow_zero=True), (0.0, 10.0))
        with self.assertRaises(ValueError):
            parse_thresholds("0", allow_zero=False)

    def test_right_direction_normalization(self):
        value = record(
            "r", "right_of_natural", first=-5, second=-8, total=-13,
            start=0, middle=-5, end=-13, maximum=13,
        )
        metrics = evidence_metrics(value)
        self.assertEqual(metrics["direction"], "right")
        self.assertAlmostEqual(metrics["maximum_directional_progress_deg"], 13.0)

    def test_left_and_right_candidates(self):
        values = [
            record("l", "left_of_natural", first=6, second=6, total=12, start=0, middle=6, end=12, maximum=12),
            record("r", "right_of_natural", first=-6, second=-6, total=-12, start=0, middle=-6, end=-12, maximum=12),
            record("small", "left_of_natural", first=2, second=2, total=4, start=0, middle=2, end=4, maximum=4),
        ]
        result = evaluate(values, progress_threshold_deg=10, deviation_threshold_deg=10)
        self.assertEqual(result["counts"]["candidate_total"], 2)
        self.assertEqual(result["counts"]["turn_left_candidate"], 1)
        self.assertEqual(result["counts"]["turn_right_candidate"], 1)

    def test_non_directional_is_ineligible(self):
        eligible, reason = base_eligibility(record("x", None))
        self.assertFalse(eligible)
        self.assertEqual(reason, "not_directional_branch_observed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
