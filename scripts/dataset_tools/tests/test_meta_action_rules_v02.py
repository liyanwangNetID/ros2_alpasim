#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_action_rules_v02 import classify_final_lateral


def base_lateral(action_relation=None):
    lateral = {
        "lateral_quality_gate": {"passed": True, "reasons": []},
        "transition_relation_counts": {},
        "ego_total_yaw_change_rad": 0.0,
        "ego_maximum_yaw_excursion_rad": 0.0,
        "ego_total_absolute_yaw_change_rad": 0.0,
        "natural_corridor": {},
    }
    if action_relation:
        lateral["adjacent_transition_evidence"] = [
            {
                "direction": action_relation,
                "returns_to_source_lane": False,
                "contains_opposite_adjacent_after": False,
            }
        ]
        lateral["transition_relation_counts"] = {
            f"{action_relation}_adjacent": 1
        }
    else:
        lateral["adjacent_transition_evidence"] = []
    return {"lateral": lateral}


class FrozenRulesTests(unittest.TestCase):
    def test_keep_to_reviewed_in_progress_lane_change(self):
        lateral = base_lateral()
        lateral["lateral"]["ego_total_yaw_change_rad"] = math.radians(5)
        lateral["lateral"]["map_corridor_heading_change_rad"] = math.radians(1)
        geometry = {
            "observed_adjacent_transitions": [],
            "in_progress_candidates": [
                {
                    "candidate": True,
                    "direction": "left",
                    "final_target_advantage_m": -1.0,
                    "directional_heading_progress_deg": 5.0,
                }
            ],
        }
        result = classify_final_lateral(lateral, geometry)
        self.assertEqual(result["action"], "change_lane_left")

    def test_lane_change_to_keep_downgrade_is_blocked(self):
        lateral = base_lateral("left")
        geometry = {
            "observed_adjacent_transitions": [
                {"interpretation": "keep_direction"}
            ],
            "in_progress_candidates": [],
        }
        result = classify_final_lateral(lateral, geometry)
        self.assertEqual(result["action"], "change_lane_left")


if __name__ == "__main__":
    unittest.main(verbosity=2)
