#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_action_rules_v02 import classify_final_lateral, classify_lateral


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


    def test_branch_relative_opposite_consensus_is_unknown(self):
        lateral = {
            "lateral_quality_gate": {
                "passed": True,
                "reasons": [],
            },
            "transition_relation_counts": {},
            "contains_adjacent_transition": False,
            "ego_total_yaw_change_rad": 0.08,
            "ego_maximum_yaw_excursion_rad": 0.25,
            "ego_total_absolute_yaw_change_rad": 0.40,
            "filtered_path_signed_heading_change_rad": 0.10,
            "final_relative_y_m": 4.0,
            "natural_corridor": {
                "turn_evidence_status": "directional_branch_observed",
                "reliable_directional_relations": [
                    "right_of_natural"
                ],
                "relative_heading": {
                    "relative_first_half_heading_change_rad": -0.32,
                    "relative_second_half_heading_change_rad": 0.25,
                    "relative_total_heading_change_rad": -0.07,
                    "relative_heading_start_rad": 0.08,
                    "relative_heading_middle_rad": 0.40,
                    "relative_heading_end_rad": 0.14,
                    "maximum_absolute_relative_heading_rad": 0.40,
                },
                "fallback_reasons": [],
            },
        }

        result = classify_lateral(lateral)

        self.assertEqual(result["action"], "unknown")
        self.assertEqual(
            result["decision_stage"],
            "branch_relative_direction_consistency_guard",
        )
        self.assertIn(
            "branch_relative_direction_conflicts_with_ego_yaw",
            result["reasons"],
        )

    def test_direction_consistent_branch_relative_turn_is_preserved(self):
        lateral = {
            "lateral_quality_gate": {
                "passed": True,
                "reasons": [],
            },
            "transition_relation_counts": {},
            "contains_adjacent_transition": False,
            "ego_total_yaw_change_rad": -0.20,
            "ego_maximum_yaw_excursion_rad": 0.25,
            "ego_total_absolute_yaw_change_rad": 0.30,
            "filtered_path_signed_heading_change_rad": -0.18,
            "final_relative_y_m": -3.0,
            "natural_corridor": {
                "turn_evidence_status": "directional_branch_observed",
                "reliable_directional_relations": [
                    "right_of_natural"
                ],
                "relative_heading": {
                    "relative_first_half_heading_change_rad": -0.25,
                    "relative_second_half_heading_change_rad": -0.10,
                    "relative_total_heading_change_rad": -0.35,
                    "relative_heading_start_rad": 0.0,
                    "relative_heading_middle_rad": -0.25,
                    "relative_heading_end_rad": -0.35,
                    "maximum_absolute_relative_heading_rad": 0.35,
                },
                "fallback_reasons": [],
            },
        }

        result = classify_lateral(lateral)

        self.assertEqual(result["action"], "turn_right")
        self.assertEqual(
            result["decision_stage"],
            "branch_relative_turn",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
