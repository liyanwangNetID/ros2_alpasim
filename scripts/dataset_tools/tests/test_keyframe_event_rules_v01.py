#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyframe_event_rules_v01 import detect_anchor_events


def meta(lateral, longitudinal):
    return {"lateral": {"action": lateral}, "longitudinal": {"action": longitudinal}}


def lateral(level="C", lane_sequence=None, markers=None, ego=0.0, road=0.0):
    return {"lateral": {
        "ego_total_yaw_change_rad": math.radians(ego),
        "map_corridor_heading_change_rad": math.radians(road),
        "lane_sequence": lane_sequence or [],
        "topology": {
            "junction_evidence_level": level,
            "wait_line_lane_ids": markers or [],
            "branching_lane_ids": [],
            "boundary_predecessor_branch_lane_ids": [],
        },
    }}


def longitudinal(final_speed=1.0, speed_delta=0.0):
    return {"longitudinal": {"final_speed_mps": final_speed, "speed_delta_mps": speed_delta}}


class EventRuleTests(unittest.TestCase):
    def types(self, events):
        return {event["type"] for event in events}

    def test_turn_and_deceleration_start(self):
        events = detect_anchor_events(
            previous_meta=meta("keep_direction", "maintain_speed"),
            current_meta=meta("turn_left", "decelerate"),
            previous_lateral_features=lateral("C"),
            current_lateral_features=lateral("A"),
            previous_longitudinal_features=longitudinal(),
            current_longitudinal_features=longitudinal(),
            current_geometry_features={"in_progress_candidates": []},
        )
        self.assertTrue({
            "lateral_action_transition", "longitudinal_action_transition",
            "turn_start", "deceleration_start", "junction_approach",
        }.issubset(self.types(events)))

    def test_restart_requires_positive_motion(self):
        events = detect_anchor_events(
            previous_meta=meta("keep_direction", "stop"),
            current_meta=meta("keep_direction", "accelerate"),
            previous_lateral_features=lateral(),
            current_lateral_features=lateral(),
            previous_longitudinal_features=longitudinal(0.0, 0.0),
            current_longitudinal_features=longitudinal(2.0, 1.5),
            current_geometry_features={"in_progress_candidates": []},
        )
        self.assertIn("restart", self.types(events))

    def test_in_progress_geometry_gate(self):
        events = detect_anchor_events(
            previous_meta=None,
            current_meta=meta("change_lane_left", "maintain_speed"),
            previous_lateral_features=None,
            current_lateral_features=lateral(ego=5.0, road=1.0),
            previous_longitudinal_features=None,
            current_longitudinal_features=longitudinal(),
            current_geometry_features={"in_progress_candidates": [{
                "candidate": True,
                "direction": "left",
                "source_lane_id": "s",
                "target_lane_id": "t",
                "final_target_advantage_m": -1.0,
                "directional_heading_progress_deg": 5.0,
            }]},
        )
        self.assertIn("lane_change_in_progress", self.types(events))

    def test_junction_entry_transition(self):
        events = detect_anchor_events(
            previous_meta=meta("keep_direction", "maintain_speed"),
            current_meta=meta("keep_direction", "maintain_speed"),
            previous_lateral_features=lateral("A", ["w"], ["w"]),
            current_lateral_features=lateral("A", ["w", "next"], ["w"]),
            previous_longitudinal_features=longitudinal(),
            current_longitudinal_features=longitudinal(),
            current_geometry_features={"in_progress_candidates": []},
        )
        self.assertIn("junction_entry", self.types(events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
