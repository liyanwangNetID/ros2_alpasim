#!/usr/bin/env python3
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from inspect_natural_corridor_case import (
    corridor_heading_at_distance,
    cumulative_trajectory_distances,
    relative_heading_features,
    trajectory_path_length,
)
from natural_lane_corridor import CorridorPoint, NaturalLaneCorridor


def point(x, y, yaw):
    return {
        "pose": {
            "position": {"x": x, "y": y, "z": 0.0},
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(yaw / 2.0),
                "w": math.cos(yaw / 2.0),
            },
        }
    }


def corridor(headings):
    points = tuple(
        CorridorPoint(float(index), 0.0, heading, float(index), "A")
        for index, heading in enumerate(headings)
    )
    return NaturalLaneCorridor(
        start_lane_id="A",
        lane_ids=("A",),
        points=points,
        total_distance_m=float(len(points) - 1),
        branch_decisions=tuple(),
        terminated_reason="lookahead_reached",
    )


class DiagnosticMathTests(unittest.TestCase):
    def test_path_length(self):
        self.assertAlmostEqual(
            trajectory_path_length(
                [point(0.0, 0.0, 0.0), point(3.0, 4.0, 0.0)]
            ),
            5.0,
        )

    def test_cumulative_distances(self):
        distances = cumulative_trajectory_distances(
            [point(0.0, 0.0, 0.0), point(3.0, 4.0, 0.0), point(6.0, 8.0, 0.0)]
        )
        self.assertEqual(distances, (0.0, 5.0, 10.0))

    def test_corridor_heading_lookup(self):
        value = corridor_heading_at_distance(
            corridor((0.0, 0.2, 0.4)), 1.1
        )
        self.assertAlmostEqual(value, 0.4)

    def test_relative_heading_cancels_road_curvature(self):
        future = [
            point(0.0, 0.0, 0.0),
            point(1.0, 0.0, 0.2),
            point(2.0, 0.0, 0.4),
        ]
        result = relative_heading_features(
            future, corridor((0.0, 0.2, 0.4))
        )
        self.assertAlmostEqual(
            result["relative_total_heading_change_rad"], 0.0
        )

    def test_relative_right_deviation_on_left_curving_road(self):
        future = [
            point(0.0, 0.0, 0.0),
            point(1.0, 0.0, 0.1),
            point(2.0, 0.0, 0.2),
        ]
        result = relative_heading_features(
            future, corridor((0.0, 0.2, 0.4))
        )
        self.assertLess(result["relative_total_heading_change_rad"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
