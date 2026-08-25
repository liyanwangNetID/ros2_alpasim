#!/usr/bin/env python3
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane_change_geometry import InProgressLaneChangeConfig
from profile_lane_change_geometry_features import (
    corridor_heading_residual,
    spatial_candidate_direction,
)


def straight(y: float):
    return [{"x": float(x), "y": y} for x in range(41)]


class V03Tests(unittest.TestCase):
    def test_early_in_progress_threshold(self):
        self.assertEqual(
            InProgressLaneChangeConfig().minimum_final_target_advantage_m,
            -2.25,
        )

    def test_heading_residual_detects_turn_away(self):
        xy = [{"x": float(x), "y": 0.0} for x in range(21)]
        yaw = [0.0] * 20 + [math.radians(-19.0)]
        result = corridor_heading_residual(xy, yaw, straight(0.0))
        self.assertLess(result["heading_change_residual_deg"], -18.0)

    def test_spatial_side(self):
        self.assertEqual(
            spatial_candidate_direction(straight(0.0), straight(3.5)),
            "left",
        )
        self.assertEqual(
            spatial_candidate_direction(straight(0.0), straight(-3.5)),
            "right",
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
