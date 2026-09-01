#!/usr/bin/env python3
"""Synthetic and real-data smoke tests for lane_matcher.py."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from clip_reader import DrivingClipReader  # noqa: E402
from lane_matcher import (  # noqa: E402
    LaneMatcher,
    LaneMatcherConfig,
    TrajectoryPose,
    trajectory_poses_from_gt_points,
)
from vector_map_reader import VectorMapReader  # noqa: E402
from project_paths import ALPASIM_DATA_ROOT  # noqa: E402


def polyline(points):
    return {
        "points": [
            {"x": x, "y": y, "z": 0.0}
            for x, y in points
        ],
        "headings": [],
    }


def lane(
    lane_id,
    points,
    *,
    width=2.0,
    predecessors=(),
    successors=(),
    left=(),
    right=(),
):
    start = points[0]
    end = points[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length
    half = width / 2.0
    left_points = [(x + nx * half, y + ny * half) for x, y in points]
    right_points = [(x - nx * half, y - ny * half) for x, y in points]
    return {
        "id": lane_id,
        "centerline": polyline(points),
        "left_boundary": polyline(left_points),
        "right_boundary": polyline(right_points),
        "predecessor_ids": list(predecessors),
        "successor_ids": list(successors),
        "left_adjacent_ids": list(left),
        "right_adjacent_ids": list(right),
        "road_area_ids": [],
        "traffic_sign_ids": [],
        "wait_line_ids": [],
    }


def synthetic_map():
    return VectorMapReader.from_dict(
        {
            "frame_id": "map",
            "map_id": "synthetic",
            "revision": 1,
            "lanes": [
                lane("A", [(0.0, 0.0), (10.0, 0.0)], successors=("B",), left=("L",)),
                lane("B", [(10.0, 0.0), (20.0, 0.0)], predecessors=("A",), left=("L2",)),
                lane("L", [(0.0, 3.0), (10.0, 3.0)], successors=("L2",), right=("A",)),
                lane("L2", [(10.0, 3.0), (20.0, 3.0)], predecessors=("L",), right=("B",)),
            ],
            "road_edges": [],
            "traffic_signs": [],
            "wait_lines": [],
        }
    )


def poses(points, yaw=0.0):
    return tuple(
        TrajectoryPose(index * 100_000_000, x, y, yaw)
        for index, (x, y) in enumerate(points)
    )


class ConfigTests(unittest.TestCase):
    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            LaneMatcherConfig(search_radius_m=0.0)
        with self.assertRaises(ValueError):
            LaneMatcherConfig(minimum_match_confidence=1.5)


class LaneMatcherTests(unittest.TestCase):
    def setUp(self):
        self.vector_map = synthetic_map()
        self.matcher = LaneMatcher(
            self.vector_map,
            LaneMatcherConfig(
                search_radius_m=4.0,
                maximum_heading_error_rad=0.5,
                maximum_candidates_per_point=4,
            ),
        )

    def test_same_lane(self):
        result = self.matcher.match(
            poses([(1.0, 0.1), (4.0, -0.1), (8.0, 0.0)])
        )
        self.assertEqual(result.compressed_lane_sequence, ("A",))
        self.assertEqual(result.transitions, tuple())
        self.assertEqual(result.matched_fraction, 1.0)

    def test_successor_transition(self):
        result = self.matcher.match(
            poses([(7.0, 0.0), (10.5, 0.0), (15.0, 0.0)])
        )
        self.assertEqual(result.compressed_lane_sequence, ("A", "B"))
        self.assertEqual(len(result.transitions), 1)
        self.assertEqual(result.transitions[0].relation, "successor")

    def test_left_adjacent_transition(self):
        result = self.matcher.match(
            poses(
                [
                    (2.0, 0.0),
                    (4.0, 0.7),
                    (6.0, 1.6),
                    (8.0, 2.7),
                    (9.0, 3.0),
                ]
            )
        )
        self.assertEqual(result.compressed_lane_sequence, ("A", "L"))
        self.assertEqual(result.transitions[0].relation, "left_adjacent")

    def test_topology_prevents_parallel_lane_jitter(self):
        result = self.matcher.match(
            poses(
                [
                    (2.0, 1.45),
                    (4.0, 1.55),
                    (6.0, 1.45),
                    (8.0, 1.55),
                ]
            )
        )
        self.assertLessEqual(len(result.compressed_lane_sequence), 2)

    def test_heading_filter_rejects_opposite_direction(self):
        result = self.matcher.match(
            poses([(2.0, 0.0), (5.0, 0.0)], yaw=math.pi)
        )
        self.assertEqual(result.matched_point_count, 0)
        self.assertEqual(result.unmatched_point_count, 2)

    def test_far_points_are_unmatched(self):
        result = self.matcher.match(
            poses([(2.0, 20.0), (5.0, 20.0)])
        )
        self.assertEqual(result.compressed_lane_sequence, tuple())
        self.assertEqual(result.matched_fraction, 0.0)

    def test_empty_trajectory(self):
        result = self.matcher.match(tuple())
        self.assertEqual(result.points, tuple())
        self.assertEqual(result.confidence, 0.0)

    def test_serialization(self):
        result = self.matcher.match(poses([(2.0, 0.0), (5.0, 0.0)]))
        value = result.to_dict()
        self.assertIn("compressed_lane_sequence", value)
        self.assertIn("confidence", value)


class RealDataSmokeTest(unittest.TestCase):
    def test_selected_anchor_from_clip_001(self):
        clip_path = ALPASIM_DATA_ROOT / "test_clip_001"
        if not clip_path.is_dir():
            self.skipTest("real test_clip_001 is unavailable")
        reader = DrivingClipReader(clip_path)
        anchor_ns = 9_306_612_661_000
        future = reader.get_future_ego_trajectory(anchor_ns)
        self.assertIsNotNone(future)
        assert future is not None
        vector_map = VectorMapReader.from_dict(reader.get_vector_map())
        matcher = LaneMatcher(vector_map)
        trajectory = trajectory_poses_from_gt_points(future.points)
        result = matcher.match(trajectory)
        self.assertGreater(len(result.points), 0)
        self.assertGreater(result.matched_fraction, 0.5)
        self.assertGreater(len(result.compressed_lane_sequence), 0)
        for transition in result.transitions:
            self.assertIn(
                transition.relation,
                {
                    "same",
                    "successor",
                    "predecessor",
                    "left_adjacent",
                    "right_adjacent",
                    "unrelated",
                },
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
