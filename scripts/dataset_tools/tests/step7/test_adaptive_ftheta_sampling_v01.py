#!/usr/bin/env python3
"""Tests for adaptive F-theta edge sampling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


from step7.adaptive_ftheta_sampling_v01 import (  # noqa: E402
    point_to_segment_distance_px,
    sample_projected_edge_adaptive,
)
from step7.camera_projection_v01 import (  # noqa: E402
    FthetaCameraCalibration,
    Quaternion,
    Vector3,
)


def calibration() -> FthetaCameraCalibration:
    return FthetaCameraCalibration(
        camera_name="test",
        logical_id="camera_test",
        width=2000,
        height=1200,
        principal_point_x=1000.0,
        principal_point_y=600.0,
        angle_to_pixeldist_poly=(0.0, 600.0, 0.0, 120.0),
        pixeldist_to_angle_poly=(0.0, 1.0 / 600.0),
        reference_poly="ANGLE_TO_PIXELDIST",
        max_angle_rad=1.5,
        linear_c=1.0,
        linear_d=0.0,
        linear_e=0.0,
        rig_to_camera_translation=Vector3(0.0, 0.0, 0.0),
        rig_to_camera_rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )


class DistanceTests(unittest.TestCase):
    def test_distance_to_horizontal_segment(self) -> None:
        self.assertAlmostEqual(
            point_to_segment_distance_px(5.0, 3.0, 0.0, 0.0, 10.0, 0.0),
            3.0,
        )

    def test_distance_to_degenerate_segment(self) -> None:
        self.assertAlmostEqual(
            point_to_segment_distance_px(3.0, 4.0, 0.0, 0.0, 0.0, 0.0),
            5.0,
        )


class AdaptiveSamplingTests(unittest.TestCase):
    def test_nearly_straight_projection_stops_immediately(self) -> None:
        result = sample_projected_edge_adaptive(
            Vector3(-0.1, 0.0, 10.0),
            Vector3(0.1, 0.0, 10.0),
            calibration(),
            maximum_chord_error_px=1.0,
            maximum_depth=8,
        )
        self.assertEqual(len(result.projections), 3)
        self.assertEqual(result.maximum_depth_reached, 0)
        self.assertFalse(result.stopped_by_depth_limit)

    def test_tighter_error_produces_at_least_as_many_samples(self) -> None:
        first = Vector3(2.0, -3.0, 2.0)
        second = Vector3(8.0, 4.0, 1.0)
        loose = sample_projected_edge_adaptive(
            first,
            second,
            calibration(),
            maximum_chord_error_px=2.0,
            maximum_depth=10,
        )
        tight = sample_projected_edge_adaptive(
            first,
            second,
            calibration(),
            maximum_chord_error_px=0.25,
            maximum_depth=10,
        )
        self.assertGreaterEqual(len(tight.projections), len(loose.projections))
        self.assertGreater(len(tight.projections), 3)

    def test_endpoints_are_preserved(self) -> None:
        first = Vector3(2.0, -3.0, 2.0)
        second = Vector3(8.0, 4.0, 1.0)
        result = sample_projected_edge_adaptive(
            first,
            second,
            calibration(),
            maximum_chord_error_px=0.5,
            maximum_depth=10,
        )
        self.assertEqual(result.camera_points[0], first)
        self.assertEqual(result.camera_points[-1], second)

    def test_depth_limit_is_reported(self) -> None:
        result = sample_projected_edge_adaptive(
            Vector3(2.0, -3.0, 2.0),
            Vector3(8.0, 4.0, 1.0),
            calibration(),
            maximum_chord_error_px=1e-12,
            maximum_depth=1,
        )
        self.assertTrue(result.stopped_by_depth_limit)
        self.assertEqual(result.maximum_depth_reached, 1)

    def test_invalid_error_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_projected_edge_adaptive(
                Vector3(0.0, 0.0, 1.0),
                Vector3(0.1, 0.0, 1.0),
                calibration(),
                maximum_chord_error_px=0.0,
                maximum_depth=4,
            )

    def test_unprojectable_endpoint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_projected_edge_adaptive(
                Vector3(0.0, 0.0, -1.0),
                Vector3(0.1, 0.0, 1.0),
                calibration(),
                maximum_chord_error_px=1.0,
                maximum_depth=4,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
