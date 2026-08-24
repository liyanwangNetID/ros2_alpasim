#!/usr/bin/env python3
from __future__ import annotations
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from profile_meta_action_features import extract_longitudinal_features


def point(stamp_ns, x, speed):
    return {
        "stamp": {
            "sec": stamp_ns // 1_000_000_000,
            "nanosec": stamp_ns % 1_000_000_000,
        },
        "pose": {
            "position": {"x": x, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "speed": speed,
        "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


class PoseDerivedSpeedTests(unittest.TestCase):
    def test_reported_zero_spikes_do_not_change_pose_speed(self):
        points = [
            point(0, 0.0, 0.0),
            point(1_000_000_000, 10.0, 10.0),
            point(2_000_000_000, 20.0, 0.0),
            point(3_000_000_000, 30.0, 10.0),
        ]
        result = extract_longitudinal_features(points)
        self.assertEqual(result["speed_source_used"], "pose_time_derived")
        self.assertAlmostEqual(result["initial_speed_mps"], 10.0)
        self.assertAlmostEqual(result["final_speed_mps"], 10.0)
        self.assertAlmostEqual(result["speed_delta_mps"], 0.0)
        self.assertFalse(result["reported_speed_reliable"])

    def test_pose_acceleration(self):
        points = [
            point(0, 0.0, 1.0),
            point(1_000_000_000, 1.0, 1.0),
            point(2_000_000_000, 3.0, 2.0),
            point(3_000_000_000, 6.0, 3.0),
        ]
        result = extract_longitudinal_features(points)
        self.assertGreater(result["speed_delta_mps"], 1.0)
        self.assertGreater(
            result["second_half_minus_first_half_mean_speed_mps"], 0.0
        )

    def test_stationary_pose_is_stop(self):
        points = [
            point(0, 5.0, 20.0),
            point(1_000_000_000, 5.0, 0.0),
            point(2_000_000_000, 5.0, 20.0),
        ]
        result = extract_longitudinal_features(points)
        self.assertEqual(result["maximum_speed_mps"], 0.0)
        self.assertTrue(result["final_speed_below_0_3_mps"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
