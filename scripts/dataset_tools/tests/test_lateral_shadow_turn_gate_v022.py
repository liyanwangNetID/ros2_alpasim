#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_lateral_shadow_rules import propose_lateral


def frozen(action):
    return {"lateral": {"action": action, "quality_status": "usable"}}


def geometry(interpretation, level, ego_deg):
    return {
        "observed_adjacent_transitions": [{
            "interpretation": interpretation,
            "interpretation_reason": "adjacent_target_is_not_parallel_downstream_corridor",
            "junction_evidence_level": level,
            "source_heading_residual": {"ego_heading_change_deg": ego_deg},
        }],
        "in_progress_candidates": [],
    }


class TurnGateTests(unittest.TestCase):
    def test_reviewed_023_strong_junction_turn_allowed(self):
        result = propose_lateral(
            frozen("change_lane_right"),
            geometry("turn_left_candidate", "A", 16.39),
        )
        self.assertEqual(result["action"], "turn_left")

    def test_reviewed_512_strong_junction_turn_allowed(self):
        result = propose_lateral(
            frozen("change_lane_right"),
            geometry("turn_right_candidate", "A", -9.21),
        )
        self.assertEqual(result["action"], "turn_right")

    def test_391_level_c_preserves_lane_change(self):
        result = propose_lateral(
            frozen("change_lane_left"),
            geometry("turn_right_candidate", "C", -3.65),
        )
        self.assertEqual(result["action"], "change_lane_left")

    def test_703_weak_junction_yaw_preserves_lane_change(self):
        result = propose_lateral(
            frozen("change_lane_right"),
            geometry("turn_left_candidate", "A", 4.13),
        )
        self.assertEqual(result["action"], "change_lane_right")

    def test_non_lane_change_turn_behavior_not_restricted(self):
        result = propose_lateral(
            frozen("keep_direction"),
            geometry("turn_right_candidate", "C", -3.0),
        )
        self.assertEqual(result["action"], "turn_right")


if __name__ == "__main__":
    unittest.main(verbosity=2)
