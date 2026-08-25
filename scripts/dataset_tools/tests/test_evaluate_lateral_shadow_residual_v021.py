#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_lateral_shadow_rules import propose_lateral


def frozen():
    return {
        "lateral": {
            "action": "keep_direction",
            "quality_status": "usable",
        }
    }


def geometry():
    return {
        "observed_adjacent_transitions": [],
        "in_progress_candidates": [
            {
                "candidate": True,
                "direction": "left",
                "final_target_advantage_m": 1.0,
                "directional_heading_progress_deg": 5.0,
            }
        ],
    }


def lateral(ego_deg, map_deg):
    return {
        "lateral": {
            "ego_total_yaw_change_rad": math.radians(ego_deg),
            "map_corridor_heading_change_rad": math.radians(map_deg),
        }
    }


class ResidualGateTests(unittest.TestCase):
    def test_small_residual_preserves_keep(self):
        result = propose_lateral(frozen(), geometry(), lateral(8.4, 8.0))
        self.assertEqual(result["action"], "keep_direction")

    def test_two_degree_residual_allows_lane_change(self):
        result = propose_lateral(frozen(), geometry(), lateral(4.1, 2.0))
        self.assertEqual(result["action"], "change_lane_left")

    def test_missing_map_heading_preserves_keep(self):
        result = propose_lateral(
            frozen(),
            geometry(),
            {"lateral": {"ego_total_yaw_change_rad": math.radians(4.0)}},
        )
        self.assertEqual(result["action"], "keep_direction")


if __name__ == "__main__":
    unittest.main(verbosity=2)
