#!/usr/bin/env python3
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from generate_meta_actions import classify_lateral, classify_longitudinal


def lateral_base():
    return {
        "lateral_quality_gate": {"passed": True, "reasons": []},
        "transition_relation_counts": {},
        "contains_adjacent_transition": False,
        "adjacent_transition_evidence": [],
        "ego_total_yaw_change_rad": 0.0,
        "ego_maximum_yaw_excursion_rad": 0.0,
        "ego_total_absolute_yaw_change_rad": 0.0,
        "natural_corridor": {
            "turn_evidence_status": "natural_or_unobserved_keep_direction",
            "reliable_directional_relations": [],
            "relative_heading": {},
            "fallback_reasons": [],
        },
    }


def longitudinal_base(**updates):
    value = {
        "speed_source_used": "pose_time_derived",
        "initial_speed_mps": 5.0,
        "final_speed_mps": 5.0,
        "speed_delta_mps": 0.0,
        "second_half_minus_first_half_mean_speed_mps": 0.0,
        "longest_duration_below_0_3_mps_sec": 0.0,
        "reported_speed_reliable": True,
    }
    value.update(updates)
    return value


class LateralRuleTests(unittest.TestCase):
    def test_lane_change_precedes_straight(self):
        value = lateral_base()
        value["transition_relation_counts"] = {"left_adjacent": 1}
        value["contains_adjacent_transition"] = True
        self.assertEqual(classify_lateral(value)["action"], "change_lane_left")

    def test_false_ambiguity_flags_do_not_block_lane_change(self):
        value = lateral_base()
        value["transition_relation_counts"] = {"right_adjacent": 1}
        value["contains_adjacent_transition"] = True
        value["adjacent_transition_evidence"] = [
            {
                "return_to_source": False,
                "ambiguous": False,
                "nested": {"opposite_adjacent": False},
            }
        ]
        self.assertEqual(
            classify_lateral(value)["action"],
            "change_lane_right",
        )

    def test_true_ambiguity_flag_blocks_lane_change(self):
        value = lateral_base()
        value["transition_relation_counts"] = {"left_adjacent": 1}
        value["contains_adjacent_transition"] = True
        value["adjacent_transition_evidence"] = [
            {"return_to_source": True}
        ]
        result = classify_lateral(value)
        self.assertEqual(result["action"], "unknown")
        self.assertIn(
            "ambiguous_adjacent_transition_evidence",
            result["reasons"],
        )

    def test_straight_override(self):
        self.assertEqual(classify_lateral(lateral_base())["action"], "keep_direction")

    def test_branch_relative_turn(self):
        value = lateral_base()
        value["ego_total_yaw_change_rad"] = math.radians(20)
        value["ego_maximum_yaw_excursion_rad"] = math.radians(20)
        value["ego_total_absolute_yaw_change_rad"] = math.radians(20)
        value["natural_corridor"] = {
            "turn_evidence_status": "directional_branch_observed",
            "reliable_directional_relations": ["right_of_natural"],
            "fallback_reasons": [],
            "relative_heading": {
                "relative_first_half_heading_change_rad": math.radians(-5),
                "relative_second_half_heading_change_rad": math.radians(-10),
                "relative_total_heading_change_rad": math.radians(-15),
                "relative_heading_start_rad": 0.0,
                "relative_heading_middle_rad": math.radians(-5),
                "relative_heading_end_rad": math.radians(-15),
                "maximum_absolute_relative_heading_rad": math.radians(15),
            },
        }
        self.assertEqual(classify_lateral(value)["action"], "turn_right")

    def test_failed_gate_is_unknown(self):
        value = lateral_base()
        value["lateral_quality_gate"] = {
            "passed": False,
            "reasons": ["matched_fraction_below_0_8"],
        }
        self.assertEqual(classify_lateral(value)["action"], "unknown")


class LongitudinalRuleTests(unittest.TestCase):
    def test_stop_priority(self):
        value = longitudinal_base(
            final_speed_mps=0.1,
            speed_delta_mps=-5.0,
            second_half_minus_first_half_mean_speed_mps=-2.0,
            longest_duration_below_0_3_mps_sec=1.2,
        )
        self.assertEqual(classify_longitudinal(value)["action"], "stop")

    def test_accelerate_and_decelerate(self):
        accelerate = longitudinal_base(
            speed_delta_mps=2.0,
            second_half_minus_first_half_mean_speed_mps=1.0,
        )
        decelerate = longitudinal_base(
            speed_delta_mps=-2.0,
            second_half_minus_first_half_mean_speed_mps=-1.0,
        )
        self.assertEqual(classify_longitudinal(accelerate)["action"], "accelerate")
        self.assertEqual(classify_longitudinal(decelerate)["action"], "decelerate")

    def test_conflict_is_unknown(self):
        value = longitudinal_base(
            speed_delta_mps=2.0,
            second_half_minus_first_half_mean_speed_mps=-1.0,
        )
        result = classify_longitudinal(value)
        self.assertEqual(result["action"], "unknown")
        self.assertEqual(result["quality_status"], "unknown")

    def test_small_changes_maintain_speed(self):
        self.assertEqual(
            classify_longitudinal(longitudinal_base())["action"],
            "maintain_speed",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
