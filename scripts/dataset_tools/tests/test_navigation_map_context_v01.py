#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordinate_utils import Pose2D
from navigation_map_context_v01 import local_route_to_map_trajectory


class NavigationMapContextTests(unittest.TestCase):
    def test_local_route_transforms_to_map(self):
        trajectory = local_route_to_map_trajectory(
            [(0.0, 0.0), (5.0, 0.0)],
            Pose2D(x=10.0, y=20.0, yaw=math.pi / 2.0),
            stamp_ns=100,
        )
        self.assertAlmostEqual(trajectory[0].x, 10.0)
        self.assertAlmostEqual(trajectory[0].y, 20.0)
        self.assertAlmostEqual(trajectory[1].x, 10.0)
        self.assertAlmostEqual(trajectory[1].y, 25.0)
        self.assertAlmostEqual(trajectory[0].yaw, math.pi / 2.0)

    def test_short_route_rejected(self):
        with self.assertRaises(ValueError):
            local_route_to_map_trajectory(
                [(0.0, 0.0)], Pose2D(0.0, 0.0, 0.0), stamp_ns=0
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
