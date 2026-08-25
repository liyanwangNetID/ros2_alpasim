#!/usr/bin/env python3
from __future__ import annotations
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from profile_meta_action_features import (
    extract_longitudinal_features,
    longest_contiguous_duration_below,
    trapezoid_duration_below,
)


def point(stamp_ns, x, speed, acceleration=0.0):
    return {
        "stamp": {
            "sec": stamp_ns // 1_000_000_000,
            "nanosec": stamp_ns % 1_000_000_000,
        },
        "pose": {
            "position": {
                "x": x,
                "y": 0.0,
                "z": 0.0,
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "w": 1.0,
            },
        },
        "speed": speed,
        "linear_acceleration": {
            "x": acceleration,
            "y": 0.0,
            "z": 0.0,
        },
    }


class LongitudinalFeatureTests(unittest.TestCase):
    def test_acceleration_features(self):
        points = [
            point(0, 0.0, 1.0),
            point(1_000_000_000, 1.0, 2.0),
            point(2_000_000_000, 4.0, 3.0),
        ]
        result = extract_longitudinal_features(points)
        self.assertEqual(result["speed_delta_mps"], 2.0)
        self.assertEqual(result["derived_mean_acceleration_mps2"], 1.0)
        self.assertEqual(result["minimum_speed_mps"], 1.0)
        self.assertEqual(result["maximum_speed_mps"], 3.0)

    def test_stop_duration(self):
        stamps = [0, 1_000_000_000, 2_000_000_000, 3_000_000_000]
        speeds = [1.0, 0.0, 0.0, 0.0]
        self.assertEqual(trapezoid_duration_below(stamps, speeds, 0.3), 2.0)
        self.assertEqual(longest_contiguous_duration_below(stamps, speeds, 0.3), 2.0)

    def test_invalid_short_trajectory(self):
        with self.assertRaises(ValueError):
            extract_longitudinal_features(
                [point(0, 0.0, 1.0)]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
