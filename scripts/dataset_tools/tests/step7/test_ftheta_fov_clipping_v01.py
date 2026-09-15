#!/usr/bin/env python3
"""Tests for analytic F-theta angular-FOV segment clipping."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


from step7.camera_projection_v01 import Vector3  # noqa: E402
from step7.ftheta_fov_clipping_v01 import (  # noqa: E402
    clip_segment_to_angular_fov,
    quadratic_nonpositive_interval,
)


def theta(point: Vector3) -> float:
    return math.atan2(math.hypot(point.x, point.y), point.z)


class QuadraticIntervalTests(unittest.TestCase):
    def test_inside_between_two_roots(self) -> None:
        result = quadratic_nonpositive_interval(1.0, -1.0, 0.16)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.start_ratio, 0.2)
        self.assertAlmostEqual(result.end_ratio, 0.8)

    def test_always_inside(self) -> None:
        self.assertEqual(
            quadratic_nonpositive_interval(0.0, 0.0, -1.0),
            type(quadratic_nonpositive_interval(0.0, 0.0, -1.0))(0.0, 1.0),
        )

    def test_always_outside(self) -> None:
        self.assertIsNone(quadratic_nonpositive_interval(0.0, 0.0, 1.0))


class AngularFovClipTests(unittest.TestCase):
    def test_fully_inside_is_unchanged(self) -> None:
        first = Vector3(-0.2, 0.0, 2.0)
        second = Vector3(0.2, 0.0, 2.0)
        self.assertEqual(
            clip_segment_to_angular_fov(
                first, second, max_angle_rad=0.5
            ),
            (first, second),
        )

    def test_fully_outside_is_removed(self) -> None:
        result = clip_segment_to_angular_fov(
            Vector3(10.0, 0.0, 1.0),
            Vector3(12.0, 0.0, 1.0),
            max_angle_rad=0.5,
        )
        self.assertIsNone(result)

    def test_one_endpoint_outside_is_clipped_to_boundary(self) -> None:
        maximum = 0.5
        result = clip_segment_to_angular_fov(
            Vector3(0.0, 0.0, 2.0),
            Vector3(4.0, 0.0, 2.0),
            max_angle_rad=maximum,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(theta(result[0]), 0.0)
        self.assertAlmostEqual(theta(result[1]), maximum, places=10)

    def test_two_outside_endpoints_can_cross_fov(self) -> None:
        maximum = 0.5
        result = clip_segment_to_angular_fov(
            Vector3(-4.0, 0.0, 2.0),
            Vector3(4.0, 0.0, 2.0),
            max_angle_rad=maximum,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(theta(result[0]), maximum, places=10)
        self.assertAlmostEqual(theta(result[1]), maximum, places=10)
        self.assertLess(result[0].x, 0.0)
        self.assertGreater(result[1].x, 0.0)

    def test_none_max_angle_returns_original_segment(self) -> None:
        first = Vector3(10.0, 0.0, 1.0)
        second = Vector3(12.0, 0.0, 1.0)
        self.assertEqual(
            clip_segment_to_angular_fov(
                first, second, max_angle_rad=None
            ),
            (first, second),
        )

    def test_nonconvex_angle_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clip_segment_to_angular_fov(
                Vector3(0.0, 0.0, 1.0),
                Vector3(1.0, 0.0, 1.0),
                max_angle_rad=math.pi / 2.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
