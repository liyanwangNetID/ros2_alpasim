#!/usr/bin/env python3
from __future__ import annotations
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from scan_longitudinal_action_thresholds import classify, evaluate, low_speed_duration_key


def record(anchor_id, *, final=5.0, duration03=0.0, duration05=0.0, delta=0.0, half=0.0):
    return {
        "anchor_id": anchor_id,
        "longitudinal": {
            "final_speed_mps": final,
            "longest_duration_below_0_3_mps_sec": duration03,
            "longest_duration_below_0_5_mps_sec": duration05,
            "speed_delta_mps": delta,
            "second_half_minus_first_half_mean_speed_mps": half,
        },
    }


class LongitudinalScanTests(unittest.TestCase):
    def test_stop_has_priority(self):
        label, _ = classify(
            record("s", final=0.1, duration03=1.2, delta=-5, half=-3),
            stop_speed_mps=0.3,
            stop_duration_sec=1.0,
            speed_delta_mps=1.0,
            half_mean_delta_mps=0.5,
        )
        self.assertEqual(label, "stop")

    def test_accelerate_and_decelerate_require_agreement(self):
        common = dict(stop_speed_mps=0.3, stop_duration_sec=1.0, speed_delta_mps=1.0, half_mean_delta_mps=0.5)
        self.assertEqual(classify(record("a", delta=2, half=1), **common)[0], "accelerate")
        self.assertEqual(classify(record("d", delta=-2, half=-1), **common)[0], "decelerate")
        label, reasons = classify(record("x", delta=2, half=-1), **common)
        self.assertEqual(label, "maintain_speed")
        self.assertIn("conflicting_speed_change_signs", reasons)

    def test_evaluation_is_exhaustive(self):
        values = [
            record("s", final=0.1, duration03=1.2),
            record("a", delta=2, half=1),
            record("d", delta=-2, half=-1),
            record("m"),
        ]
        result = evaluate(
            values,
            stop_speed_mps=0.3,
            stop_duration_sec=1.0,
            speed_delta_mps=1.0,
            half_mean_delta_mps=0.5,
        )
        self.assertEqual(sum(result["counts"].get(k, 0) for k in ("stop", "accelerate", "decelerate", "maintain_speed")), 4)

    def test_supported_duration_keys(self):
        self.assertEqual(low_speed_duration_key(0.3), "longest_duration_below_0_3_mps_sec")
        self.assertEqual(low_speed_duration_key(0.5), "longest_duration_below_0_5_mps_sec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
