#!/usr/bin/env python3
"""Tests for Step 7D F-theta calibration and point projection."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


from camera_projection_v01 import (  # noqa: E402
    FthetaCameraCalibration,
    Quaternion,
    Vector3,
    evaluate_polynomial_low_to_high,
    load_camera_calibration,
    normalize_quaternion,
    parse_camera_calibration,
    project_ftheta_point,
    rotate_vector_by_quaternion,
)
from project_paths import ALPASIM_DATA_ROOT  # noqa: E402


def calibration_dict(
    *,
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    translation: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> dict:
    qx, qy, qz, qw = quaternion
    tx, ty, tz = translation
    return {
        "available_camera": {
            "logical_id": "camera_test",
            "intrinsics": {
                "logical_id": "camera_test",
                "resolution_w": 200,
                "resolution_h": 100,
                "shutter_type": "ROLLING_TOP_TO_BOTTOM",
                "ftheta_param": {
                    "angle_to_pixeldist_poly": [0.0, 50.0],
                    "pixeldist_to_angle_poly": [0.0, 0.02],
                    "principal_point_x": 100.0,
                    "principal_point_y": 50.0,
                    "max_angle": 1.0,
                    "reference_poly": "PIXELDIST_TO_ANGLE",
                    "linear_cde": {
                        "linear_c": 1.0,
                        "linear_d": 0.0,
                        "linear_e": 0.0,
                    },
                },
            },
            "rig_to_camera": {
                "vec": {"x": tx, "y": ty, "z": tz},
                "quat": {"x": qx, "y": qy, "z": qz, "w": qw},
            },
        }
    }


class PolynomialTests(unittest.TestCase):
    def test_low_to_high_order(self) -> None:
        self.assertEqual(
            evaluate_polynomial_low_to_high(2.0, [1.0, 3.0, 4.0]),
            23.0,
        )


class QuaternionTests(unittest.TestCase):
    def test_quaternion_is_normalized(self) -> None:
        result = normalize_quaternion(Quaternion(0.0, 0.0, 0.0, 5.0))
        self.assertEqual(result, Quaternion(0.0, 0.0, 0.0, 1.0))

    def test_rotation_about_z(self) -> None:
        half = math.pi / 4.0
        result = rotate_vector_by_quaternion(
            Vector3(1.0, 0.0, 0.0),
            Quaternion(0.0, 0.0, math.sin(half), math.cos(half)),
        )
        self.assertAlmostEqual(result.x, 0.0, places=7)
        self.assertAlmostEqual(result.y, 1.0, places=7)
        self.assertAlmostEqual(result.z, 0.0, places=7)


class RigidTransformTests(unittest.TestCase):
    def test_rig_camera_round_trip(self) -> None:
        half = 0.5 * math.radians(30.0)
        calibration = parse_camera_calibration(
            calibration_dict(
                quaternion=(0.0, 0.0, math.sin(half), math.cos(half)),
            ),
            camera_name="test",
        )
        original = Vector3(7.0, -2.0, 5.0)
        camera = calibration.rig_point_to_camera(original)
        recovered = calibration.camera_point_to_rig(camera)
        self.assertAlmostEqual(recovered.x, original.x, places=7)
        self.assertAlmostEqual(recovered.y, original.y, places=7)
        self.assertAlmostEqual(recovered.z, original.z, places=7)

    def test_identity_rotation_subtracts_camera_translation(self) -> None:
        calibration = parse_camera_calibration(
            calibration_dict(),
            camera_name="test",
        )
        result = calibration.rig_point_to_camera(Vector3(1.0, 2.0, 8.0))
        self.assertEqual(result, Vector3(0.0, 0.0, 5.0))


class ProjectionTests(unittest.TestCase):
    def test_optical_axis_projects_to_principal_point(self) -> None:
        result = project_ftheta_point(
            Vector3(0.0, 0.0, 10.0),
            width=200,
            height=100,
            principal_point_x=100.0,
            principal_point_y=50.0,
            angle_to_pixeldist_poly=[0.0, 50.0],
            max_angle_rad=1.0,
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.u, 100.0)
        self.assertAlmostEqual(result.v, 50.0)

    def test_positive_x_moves_pixel_right(self) -> None:
        result = project_ftheta_point(
            Vector3(1.0, 0.0, 10.0),
            width=200,
            height=100,
            principal_point_x=100.0,
            principal_point_y=50.0,
            angle_to_pixeldist_poly=[0.0, 50.0],
            max_angle_rad=1.0,
        )
        self.assertTrue(result.valid)
        self.assertGreater(result.u, 100.0)
        self.assertAlmostEqual(result.v, 50.0)

    def test_positive_y_moves_pixel_down(self) -> None:
        result = project_ftheta_point(
            Vector3(0.0, 1.0, 10.0),
            width=200,
            height=100,
            principal_point_x=100.0,
            principal_point_y=50.0,
            angle_to_pixeldist_poly=[0.0, 50.0],
            max_angle_rad=1.0,
        )
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.u, 100.0)
        self.assertGreater(result.v, 50.0)

    def test_behind_camera_is_invalid(self) -> None:
        result = project_ftheta_point(
            Vector3(0.0, 0.0, -1.0),
            width=200,
            height=100,
            principal_point_x=100.0,
            principal_point_y=50.0,
            angle_to_pixeldist_poly=[0.0, 50.0],
            max_angle_rad=1.0,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, "behind_or_on_camera_plane")

    def test_outside_max_angle_is_invalid(self) -> None:
        result = project_ftheta_point(
            Vector3(10.0, 0.0, 1.0),
            width=1000,
            height=1000,
            principal_point_x=500.0,
            principal_point_y=500.0,
            angle_to_pixeldist_poly=[0.0, 50.0],
            max_angle_rad=0.5,
        )
        self.assertFalse(result.valid)
        self.assertFalse(result.within_fov)
        self.assertEqual(result.failure_reason, "outside_max_angle")

    def test_linear_cde_is_applied(self) -> None:
        result = project_ftheta_point(
            Vector3(1.0, 0.0, 10.0),
            width=300,
            height=200,
            principal_point_x=100.0,
            principal_point_y=50.0,
            angle_to_pixeldist_poly=[0.0, 50.0],
            max_angle_rad=1.0,
            linear_c=2.0,
            linear_e=0.5,
        )
        self.assertGreater(result.u, 100.0)
        self.assertGreater(result.v, 50.0)


class CalibrationParserTests(unittest.TestCase):
    def test_parser_preserves_reference_poly_but_uses_forward_poly(self) -> None:
        result = parse_camera_calibration(
            calibration_dict(),
            camera_name="test",
        )
        self.assertEqual(result.reference_poly, "PIXELDIST_TO_ANGLE")
        self.assertEqual(result.angle_to_pixeldist_poly, (0.0, 50.0))
        projection = result.project_camera_point(Vector3(0.0, 0.0, 2.0))
        self.assertTrue(projection.valid)

    def test_resolution_scaling_matches_driver_policy(self) -> None:
        result = parse_camera_calibration(
            calibration_dict(),
            camera_name="test",
            source_width=100,
            source_height=50,
        )
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 50)
        self.assertAlmostEqual(result.principal_point_x, 50.0)
        self.assertAlmostEqual(result.principal_point_y, 25.0)
        self.assertEqual(result.angle_to_pixeldist_poly, (0.0, 25.0))

    def test_real_front_wide_calibration_parses_when_available(self) -> None:
        path = (
            ALPASIM_DATA_ROOT
            / "test_clip_001"
            / "calibration"
            / "front_wide.json"
        )
        if not path.is_file():
            self.skipTest("real test_clip_001 calibration is unavailable")
        result = load_camera_calibration(path, camera_name="front_wide")
        self.assertIsInstance(result, FthetaCameraCalibration)
        self.assertEqual(result.width, 1920)
        self.assertEqual(result.height, 1080)
        self.assertEqual(result.logical_id, "camera_front_wide_120fov")
        self.assertGreater(len(result.angle_to_pixeldist_poly), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
