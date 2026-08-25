#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_lateral_shadow_rules import propose_lateral


def frozen(action="keep_direction", quality="usable"):
    return {"lateral": {"action": action, "quality_status": quality}}


def geometry(*, observed=None, candidates=None):
    return {
        "observed_adjacent_transitions": observed or [],
        "in_progress_candidates": candidates or [],
        "inferred_in_progress_action": None,
    }


class ShadowPolicyV02Tests(unittest.TestCase):
    def test_lane_change_to_keep_is_preserved(self):
        proposal = propose_lateral(
            frozen("change_lane_left"),
            geometry(observed=[{"interpretation": "keep_direction"}]),
        )
        self.assertEqual(proposal["action"], "change_lane_left")

    def test_lane_change_to_turn_is_allowed(self):
        proposal = propose_lateral(
            frozen("change_lane_right"),
            geometry(
                observed=[
                    {
                        "interpretation": "turn_left_candidate",
                        "interpretation_reason": (
                            "adjacent_target_is_not_parallel_"
                            "downstream_corridor"
                        ),
                        "junction_evidence_level": "A",
                        "source_heading_residual": {
                            "ego_heading_change_deg": 16.0,
                        },
                    }
                ]
            ),
        )
        self.assertEqual(proposal["action"], "turn_left")

    def test_reviewed_true_in_progress_is_allowed(self):
        proposal = propose_lateral(
            frozen(),
            geometry(candidates=[{
                "candidate": True,
                "direction": "left",
                "final_target_advantage_m": -1.79,
                "directional_heading_progress_deg": 4.49,
            }]),
        )
        self.assertEqual(proposal["action"], "change_lane_left")

    def test_curved_road_large_heading_is_rejected(self):
        proposal = propose_lateral(
            frozen(),
            geometry(candidates=[{
                "candidate": True,
                "direction": "right",
                "final_target_advantage_m": -0.45,
                "directional_heading_progress_deg": 16.65,
            }]),
        )
        self.assertEqual(proposal["action"], "keep_direction")

    def test_target_too_far_is_rejected(self):
        proposal = propose_lateral(
            frozen(),
            geometry(candidates=[{
                "candidate": True,
                "direction": "left",
                "final_target_advantage_m": -2.14,
                "directional_heading_progress_deg": 4.24,
            }]),
        )
        self.assertEqual(proposal["action"], "keep_direction")


if __name__ == "__main__":
    unittest.main(verbosity=2)
