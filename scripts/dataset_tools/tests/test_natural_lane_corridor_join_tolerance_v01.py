#!/usr/bin/env python3
import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordinate_utils import Point2D
from natural_lane_corridor import _heading_metrics, _path_points


@dataclass(frozen=True)
class FakeLane:
    centerline: tuple[Point2D, ...]


class FakeMap:
    def __init__(self, lanes):
        self._lanes = lanes

    def require_lane(self, lane_id):
        return self._lanes[lane_id]


class NaturalCorridorJoinToleranceTests(unittest.TestCase):
    def test_millimetre_boundary_gap_does_not_create_heading_spike(self):
        source = FakeLane((
            Point2D(0.0, 0.0),
            Point2D(10.0, 0.0),
        ))
        target = FakeLane((
            Point2D(10.0, 0.004),
            Point2D(20.0, 0.0),
        ))
        points, distance = _path_points(
            FakeMap({'source': source, 'target': target}),
            ('source', 'target'),
            30.0,
        )
        self.assertEqual(points, (
            Point2D(0.0, 0.0),
            Point2D(10.0, 0.0),
            Point2D(20.0, 0.0),
        ))
        signed, absolute = _heading_metrics(0.0, points)
        self.assertLess(abs(math.degrees(signed)), 0.1)
        self.assertLess(math.degrees(absolute), 0.1)
        self.assertAlmostEqual(distance, 20.0)

    def test_meaningful_boundary_gap_is_preserved(self):
        source = FakeLane((
            Point2D(0.0, 0.0),
            Point2D(10.0, 0.0),
        ))
        target = FakeLane((
            Point2D(10.0, 0.10),
            Point2D(20.0, 0.10),
        ))
        points, _ = _path_points(
            FakeMap({'source': source, 'target': target}),
            ('source', 'target'),
            30.0,
        )
        self.assertIn(Point2D(10.0, 0.10), points)


if __name__ == '__main__':
    unittest.main(verbosity=2)
