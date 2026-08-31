#!/usr/bin/env python3
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_meta_actions_v02 import validate_distribution


class ProductionBuilderTests(unittest.TestCase):
    def test_frozen_distribution_is_accepted(self):
        validate_distribution(
            anchor_count=10231,
            lateral_counts=Counter({
                "keep_direction": 9472,
                "unknown": 387,
                "turn_left": 16,
                "turn_right": 57,
                "change_lane_left": 155,
                "change_lane_right": 144,
            }),
            longitudinal_counts=Counter({
                "maintain_speed": 5589,
                "unknown": 401,
                "accelerate": 1532,
                "decelerate": 1810,
                "stop": 899,
            }),
            quality_counts=Counter({"usable": 9465, "unknown": 766}),
            strict=True,
        )

    def test_wrong_distribution_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_distribution(
                anchor_count=1,
                lateral_counts=Counter(),
                longitudinal_counts=Counter(),
                quality_counts=Counter(),
                strict=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
