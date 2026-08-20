#!/usr/bin/env python3
"""Unit tests for coordinate_utils.py."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from coordinate_utils import (  # noqa: E402
    Point2D,
    Pose2D,
    RelativePose2D,
    anchor_ego_point_to_map,
    anchor_ego_pose_to_map,
    cumulative_distances,
    interpolate_angle,
    interpolate_pose2d,
    map_point_to_anchor_ego,
    map_pose_to_anchor_ego,
    map_vector_to_anchor_ego,
    normalize_angle,
    planar_heading,
    pose2d_from_pose_mapping,
    quaternion_to_yaw,
    relative_region,
    rotate_vector_2d,
    shortest_angular_distance,
    signed_lateral_offset_to_segment,
    trajectory_geometry,
    unwrap_angles,
    yaw_to_quaternion,
)


class AngleTests(unittest.TestCase):
    def test_normalize_angle(self) -> None:
        self.assertAlmostEqual(normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(normalize_angle(2.0 * math.pi), 0.0)
        self.assertAlmostEqual(normalize_angle(math.pi), -math.pi)
        self.assertAlmostEqual(normalize_angle(-math.pi), -math.pi)

    def test_shortest_angular_distance_wrap(self) -> None:
        first = math.radians(179.0)
        second = math.radians(-179.0)
        self.assertAlmostEqual(
            shortest_angular_distance(first, second),
            math.radians(2.0),
        )

    def test_quaternion_to_yaw(self) -> None:
        quaternion = yaw_to_quaternion(math.pi / 2.0)
        self.assertAlmostEqual(
            quaternion_to_yaw(*quaternion),
            math.pi / 2.0,
        )

    def test_quaternion_normalized_automatically(self) -> None:
        quaternion = yaw_to_quaternion(-math.pi / 3.0)
        scaled = tuple(value * 5.0 for value in quaternion)
        self.assertAlmostEqual(
            quaternion_to_yaw(*scaled),
            -math.pi / 3.0,
        )

    def test_zero_quaternion_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quaternion_to_yaw(0.0, 0.0, 0.0, 0.0)

    def test_unwrap_angles(self) -> None:
        values = unwrap_angles(
            [
                math.radians(170.0),
                math.radians(179.0),
                math.radians(-175.0),
            ]
        )
        self.assertGreater(values[2], values[1])
        self.assertAlmostEqual(
            values[2] - values[1],
            math.radians(6.0),
        )

    def test_interpolate_angle_across_wrap(self) -> None:
        result = interpolate_angle(
            math.radians(170.0),
            math.radians(-170.0),
            0.5,
        )
        self.assertAlmostEqual(abs(result), math.pi)


class TransformTests(unittest.TestCase):
    def test_rotate_vector_counter_clockwise(self) -> None:
        result = rotate_vector_2d(1.0, 0.0, math.pi / 2.0)
        self.assertAlmostEqual(result.x, 0.0, places=7)
        self.assertAlmostEqual(result.y, 1.0, places=7)

    def test_map_point_identity_anchor(self) -> None:
        result = map_point_to_anchor_ego(12.0, 4.0, 10.0, 1.0, 0.0)
        self.assertAlmostEqual(result.x, 2.0)
        self.assertAlmostEqual(result.y, 3.0)

    def test_map_point_with_ninety_degree_anchor(self) -> None:
        result = map_point_to_anchor_ego(
            10.0,
            3.0,
            10.0,
            1.0,
            math.pi / 2.0,
        )
        self.assertAlmostEqual(result.x, 2.0, places=7)
        self.assertAlmostEqual(result.y, 0.0, places=7)

    def test_map_point_to_local_and_back(self) -> None:
        local = map_point_to_anchor_ego(7.0, -2.0, 3.0, 1.0, 0.7)
        recovered = anchor_ego_point_to_map(
            local.x,
            local.y,
            3.0,
            1.0,
            0.7,
        )
        self.assertAlmostEqual(recovered.x, 7.0)
        self.assertAlmostEqual(recovered.y, -2.0)

    def test_pose_to_local_and_back(self) -> None:
        anchor = Pose2D(3.0, 1.0, math.radians(170.0))
        pose = Pose2D(7.0, -2.0, math.radians(-175.0))
        relative = map_pose_to_anchor_ego(pose, anchor)
        recovered = anchor_ego_pose_to_map(relative, anchor)
        self.assertAlmostEqual(recovered.x, pose.x)
        self.assertAlmostEqual(recovered.y, pose.y)
        self.assertAlmostEqual(
            shortest_angular_distance(recovered.yaw, pose.yaw),
            0.0,
        )

    def test_map_vector_rotation_has_no_translation(self) -> None:
        local = map_vector_to_anchor_ego(0.0, 2.0, math.pi / 2.0)
        self.assertAlmostEqual(local.x, 2.0, places=7)
        self.assertAlmostEqual(local.y, 0.0, places=7)

    def test_pose_mapping(self) -> None:
        qx, qy, qz, qw = yaw_to_quaternion(0.25)
        mapping = {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        }
        pose = pose2d_from_pose_mapping(mapping)
        self.assertEqual(pose.x, 1.0)
        self.assertEqual(pose.y, 2.0)
        self.assertAlmostEqual(pose.yaw, 0.25)

    def test_interpolate_pose(self) -> None:
        first = Pose2D(0.0, 0.0, 0.0)
        second = Pose2D(10.0, 4.0, math.pi / 2.0)
        middle = interpolate_pose2d(first, second, 0.5)
        self.assertAlmostEqual(middle.x, 5.0)
        self.assertAlmostEqual(middle.y, 2.0)
        self.assertAlmostEqual(middle.yaw, math.pi / 4.0)


class RelativeRegionTests(unittest.TestCase):
    def test_regions_follow_x_forward_y_left(self) -> None:
        self.assertEqual(relative_region(5.0, 0.0), "ahead")
        self.assertEqual(relative_region(-5.0, 0.0), "behind")
        self.assertEqual(relative_region(0.0, 5.0), "left")
        self.assertEqual(relative_region(0.0, -5.0), "right")
        self.assertEqual(relative_region(5.0, 5.0), "ahead_left")
        self.assertEqual(relative_region(5.0, -5.0), "ahead_right")
        self.assertEqual(relative_region(0.1, 0.1), "overlap")

    def test_negative_deadband_rejected(self) -> None:
        with self.assertRaises(ValueError):
            relative_region(1.0, 1.0, longitudinal_deadband=-1.0)


class GeometryTests(unittest.TestCase):
    def test_cumulative_distances(self) -> None:
        points = [Point2D(0.0, 0.0), Point2D(3.0, 4.0), Point2D(6.0, 8.0)]
        self.assertEqual(cumulative_distances(points), (0.0, 5.0, 10.0))

    def test_planar_heading(self) -> None:
        self.assertAlmostEqual(planar_heading(0.0, 0.0, 0.0, 1.0), math.pi / 2.0)

    def test_heading_rejects_coincident_points(self) -> None:
        with self.assertRaises(ValueError):
            planar_heading(1.0, 1.0, 1.0, 1.0)

    def test_signed_lateral_offset(self) -> None:
        start = Point2D(0.0, 0.0)
        end = Point2D(10.0, 0.0)
        self.assertAlmostEqual(
            signed_lateral_offset_to_segment(Point2D(5.0, 2.0), start, end),
            2.0,
        )
        self.assertAlmostEqual(
            signed_lateral_offset_to_segment(Point2D(5.0, -2.0), start, end),
            -2.0,
        )

    def test_straight_trajectory_geometry(self) -> None:
        points = [Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(2.0, 0.0)]
        geometry = trajectory_geometry(points)
        self.assertEqual(geometry.point_count, 3)
        self.assertAlmostEqual(geometry.path_length, 2.0)
        self.assertAlmostEqual(geometry.displacement, 2.0)
        self.assertAlmostEqual(geometry.heading_change, 0.0)
        self.assertAlmostEqual(geometry.lateral_displacement, 0.0)
        self.assertAlmostEqual(geometry.mean_absolute_curvature or 0.0, 0.0)

    def test_left_turn_geometry(self) -> None:
        points = [
            Point2D(0.0, 0.0),
            Point2D(1.0, 0.0),
            Point2D(2.0, 1.0),
            Point2D(2.0, 2.0),
        ]
        geometry = trajectory_geometry(points)
        self.assertGreater(geometry.heading_change, 0.0)
        self.assertGreater(geometry.lateral_displacement, 0.0)
        self.assertIsNotNone(geometry.maximum_absolute_curvature)

    def test_empty_and_single_point_geometry(self) -> None:
        empty = trajectory_geometry([])
        single = trajectory_geometry([Point2D(1.0, 2.0)])
        self.assertEqual(empty.point_count, 0)
        self.assertEqual(single.point_count, 1)
        self.assertEqual(single.path_length, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
