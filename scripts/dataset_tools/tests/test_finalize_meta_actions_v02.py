#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finalize_meta_actions_v02 import finalize_record


def base(action="keep_direction"):
    return {
        "anchor_id": "a",
        "clip_id": "c",
        "anchor_ns": 1,
        "label_format_version": "0.1-draft",
        "generator_version": "0.1.1",
        "rule_version": "meta_action_rules_v0.1",
        "lateral": {
            "action": action,
            "quality_status": "usable",
            "reasons": ["old"],
            "decision_stage": "old_stage",
        },
        "longitudinal": {
            "action": "maintain_speed",
            "quality_status": "usable",
            "reasons": ["unchanged"],
            "decision_stage": "maintain_speed_fallback",
        },
        "joint_action": {"lateral": action, "longitudinal": "maintain_speed"},
        "overall_quality_status": "usable",
    }


def shadow(old="keep_direction", proposed="change_lane_left"):
    return {
        "anchor_id": "a",
        "clip_id": "c",
        "anchor_ns": 1,
        "old_lateral_action": old,
        "proposed_lateral_action": proposed,
        "decision_source": "reviewed_in_progress_lane_change_geometry",
        "reasons": ["reviewed"],
        "shadow_evaluator_version": "0.2.0",
    }


class FinalizeTests(unittest.TestCase):
    def test_changed_lateral_and_joint_action(self):
        result = finalize_record(base(), shadow())
        self.assertEqual(result["lateral"]["action"], "change_lane_left")
        self.assertEqual(result["joint_action"]["lateral"], "change_lane_left")
        self.assertEqual(result["lateral"]["decision_stage"], "reviewed_shadow_geometry_v0.2")

    def test_longitudinal_copied_unchanged(self):
        result = finalize_record(base(), shadow())
        self.assertEqual(result["longitudinal"]["action"], "maintain_speed")
        self.assertEqual(result["longitudinal"]["reasons"], ["unchanged"])

    def test_unchanged_lateral_preserves_stage(self):
        result = finalize_record(base(), shadow(proposed="keep_direction"))
        self.assertEqual(result["lateral"]["decision_stage"], "old_stage")

    def test_shadow_old_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            finalize_record(base("turn_left"), shadow(old="keep_direction"))

    def test_identity_mismatch_rejected(self):
        value = shadow()
        value["anchor_ns"] = 2
        with self.assertRaises(ValueError):
            finalize_record(base(), value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
